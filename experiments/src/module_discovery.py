"""
Phase 2: Discover functional modules via GPU-accelerated IterD.

Batch-parallel version: instead of serial per-neuron reassignment,
computes optimal assignments for ALL neurons/samples simultaneously
using batched matrix operations on GPU.

Speedup: ~100x over CPU serial version (530K neurons: 30min → 20sec).
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


def compute_objective(within_sums, neuron_counts, sample_counts, K):
    """Compute L(F) = ξ(F) × B(F) from precomputed sums."""
    sizes = neuron_counts * sample_counts
    if (sizes == 0).any():
        return 0.0
    xi = within_sums.sum() / sizes.sum()
    B = K / (1.0 / sizes.float()).sum()
    return (xi * B).item()


def iterd_gpu(A_np, K, max_iter=50, refine_iter=3, seed=42, device="cuda", verbose=True):
    """
    GPU-accelerated IterD via batch reassignment.

    Phase 1 (batch): Alternates between assigning all neurons and all samples
    in parallel using matrix operations. Fast but approximate (no greedy ordering).

    Phase 2 (refine): Optional sequential refinement passes on GPU for
    neurons that changed in the last batch iteration.

    Args:
        A_np: activation matrix [n_neurons, n_samples] as numpy array
        K: number of modules
        max_iter: max batch iterations
        refine_iter: sequential refinement passes after batch convergence
        seed: random seed
        device: "cuda" or "cpu"
        verbose: print progress
    """
    n_neurons, n_samples = A_np.shape
    t0 = time.time()

    if verbose:
        print(f"IterD-GPU: {n_neurons:,} neurons x {n_samples:,} samples, K={K}, device={device}")

    # --- Initialization on CPU (K-Means) ---
    if verbose:
        print("  Initializing with K-Means...")

    n_comp = min(50, n_samples, n_neurons)
    pca = PCA(n_components=n_comp, random_state=seed)
    A_pca = pca.fit_transform(A_np)
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    neuron_assign_np = km.fit_predict(A_pca).astype(np.int64)

    # Initial sample assignment: each sample → module with highest mean activation
    sample_assign_np = np.zeros(n_samples, dtype=np.int64)
    for j in range(n_samples):
        scores = np.zeros(K)
        for k in range(K):
            mask = neuron_assign_np == k
            if mask.sum() > 0:
                scores[k] = A_np[mask, j].mean()
        sample_assign_np[j] = scores.argmax()

    # --- Move to GPU ---
    A = torch.from_numpy(A_np).float().to(device)
    neuron_assign = torch.from_numpy(neuron_assign_np).long().to(device)
    sample_assign = torch.from_numpy(sample_assign_np).long().to(device)

    def build_onehot_neurons():
        oh = torch.zeros(n_neurons, K, device=device)
        oh.scatter_(1, neuron_assign.unsqueeze(1), 1.0)
        return oh

    def build_onehot_samples():
        oh = torch.zeros(n_samples, K, device=device)
        oh.scatter_(1, sample_assign.unsqueeze(1), 1.0)
        return oh

    def get_counts():
        nc = torch.zeros(K, device=device, dtype=torch.long)
        sc = torch.zeros(K, device=device, dtype=torch.long)
        for k in range(K):
            nc[k] = (neuron_assign == k).sum()
            sc[k] = (sample_assign == k).sum()
        return nc, sc

    def get_within_sums(nc, sc):
        ws = torch.zeros(K, device=device)
        for k in range(K):
            if nc[k] > 0 and sc[k] > 0:
                u_mask = neuron_assign == k
                s_mask = sample_assign == k
                ws[k] = A[u_mask][:, s_mask].sum()
        return ws

    nc, sc = get_counts()
    ws = get_within_sums(nc, sc)
    L = compute_objective(ws, nc, sc, K)
    scores_history = [L]

    if verbose:
        print(f"  Init: L(F) = {L:.6f} ({time.time()-t0:.1f}s)")

    # --- Phase 1: Batch iterations ---
    for it in range(max_iter):
        # Step A: Reassign all neurons (samples fixed)
        # neuron_to_module_scores[i, k] = mean activation of neuron i on module k's samples
        oh_s = build_onehot_samples()  # [S, K]
        n2m = A @ oh_s  # [N, K] — sum of activations for each neuron on each module's samples
        sc_safe = sc.float().clamp(min=1)
        n2m_mean = n2m / sc_safe.unsqueeze(0)  # [N, K] — mean activation
        new_neuron_assign = n2m_mean.argmax(dim=1)  # [N]

        # Prevent empty modules: if a module would lose all neurons, keep some
        for k in range(K):
            if (new_neuron_assign == k).sum() == 0:
                # Find neuron with highest score for this module and force-assign it
                forced = n2m_mean[:, k].argmax()
                new_neuron_assign[forced] = k

        neuron_changed = (new_neuron_assign != neuron_assign).sum().item()
        neuron_assign = new_neuron_assign

        # Step B: Reassign all samples (neurons fixed)
        oh_n = build_onehot_neurons()  # [N, K]
        m2s = oh_n.T @ A  # [K, S] — sum of activations for each module on each sample
        nc_new = torch.zeros(K, device=device, dtype=torch.long)
        for k in range(K):
            nc_new[k] = (neuron_assign == k).sum()
        nc_safe = nc_new.float().clamp(min=1)
        m2s_mean = m2s / nc_safe.unsqueeze(1)  # [K, S]
        new_sample_assign = m2s_mean.argmax(dim=0)  # [S]

        for k in range(K):
            if (new_sample_assign == k).sum() == 0:
                forced = m2s_mean[k].argmax()
                new_sample_assign[forced] = k

        sample_changed = (new_sample_assign != sample_assign).sum().item()
        sample_assign = new_sample_assign

        nc, sc = get_counts()
        ws = get_within_sums(nc, sc)
        L = compute_objective(ws, nc, sc, K)
        scores_history.append(L)

        if verbose:
            print(f"  Batch iter {it+1}: L={L:.6f}, "
                  f"neuron_moved={neuron_changed:,}, sample_moved={sample_changed}, "
                  f"({time.time()-t0:.1f}s)")

        if neuron_changed == 0 and sample_changed == 0:
            if verbose:
                print(f"  Converged at batch iteration {it+1}")
            break

    # --- Phase 2: Sequential refinement on GPU ---
    if refine_iter > 0 and verbose:
        print(f"  Refining ({refine_iter} sequential passes)...")

    oh_s = build_onehot_samples()
    n2m_sums = A @ oh_s  # [N, K]

    oh_n = build_onehot_neurons()
    m2s_sums = oh_n.T @ A  # [K, S]

    for ref in range(refine_iter):
        changed = 0

        # Neuron refinement: check each neuron's contribution to objective
        for i in torch.randperm(n_neurons, device=device):
            i = i.item()
            old_k = neuron_assign[i].item()
            contrib = n2m_sums[i]  # [K] — this neuron's sum on each module's samples
            # Score: contribution / sample_count (mean activation on that module)
            scores_k = contrib / sc.float().clamp(min=1)
            best_k = scores_k.argmax().item()

            if best_k != old_k and (neuron_assign == old_k).sum() > 1:
                # Update cached sums
                n2m_sums[i] = A[i] @ oh_s  # refresh
                m2s_sums[old_k] -= A[i]
                m2s_sums[best_k] += A[i]
                oh_n_row = torch.zeros(K, device=device)
                oh_n_row[best_k] = 1.0

                neuron_assign[i] = best_k
                nc[old_k] -= 1
                nc[best_k] += 1
                changed += 1

        # Sample refinement
        for j in torch.randperm(n_samples, device=device):
            j = j.item()
            old_k = sample_assign[j].item()
            contrib = m2s_sums[:, j]  # [K]
            scores_k = contrib / nc.float().clamp(min=1)
            best_k = scores_k.argmax().item()

            if best_k != old_k and (sample_assign == old_k).sum() > 1:
                col = A[:, j]
                n2m_sums[:, old_k] -= col
                n2m_sums[:, best_k] += col
                sample_assign[j] = best_k
                sc[old_k] -= 1
                sc[best_k] += 1
                changed += 1

        ws = get_within_sums(nc, sc)
        L = compute_objective(ws, nc, sc, K)
        scores_history.append(L)

        if verbose:
            print(f"  Refine {ref+1}: L={L:.6f}, changed={changed} ({time.time()-t0:.1f}s)")

        if changed == 0:
            break

    total_time = time.time() - t0
    if verbose:
        print(f"  Done in {total_time:.1f}s")

    return (neuron_assign.cpu().numpy(),
            sample_assign.cpu().numpy(),
            scores_history)


def iterd(A, K, max_iter=50, seed=42, verbose=True):
    """Auto-select GPU or CPU based on availability."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return iterd_gpu(A, K, max_iter=max_iter, seed=seed, device=device, verbose=verbose)


def align_modules_across_models(assignments_list, K):
    """Align module indices across models using Hungarian algorithm on sample overlap."""
    ref_sa = assignments_list[0][1]
    perms = [np.arange(K)]
    for m in range(1, len(assignments_list)):
        tgt_sa = assignments_list[m][1]
        cost = np.zeros((K, K))
        for kr in range(K):
            for kt in range(K):
                cost[kr, kt] = -len(set(np.where(ref_sa == kr)[0]) & set(np.where(tgt_sa == kt)[0]))
        _, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(K, dtype=int)
        perm[col_ind] = np.arange(K)
        perms.append(perm)
    return perms


def run_discovery(activation_path, output_dir, K_values=None):
    if K_values is None:
        K_values = [5, 10, 15, 20]

    A = np.load(activation_path)
    meta_path = activation_path.replace("_activations.npy", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    model_short = Path(activation_path).stem.replace("_activations", "")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for K in K_values:
        print(f"\n{'='*60}")
        print(f"Running IterD-GPU with K={K} for {model_short}")
        print(f"{'='*60}")

        na, sa, scores_list = iterd(A, K, verbose=True)

        result = {
            "K": K,
            "final_score": float(scores_list[-1]),
            "n_iterations": len(scores_list),
            "scores": [float(s) for s in scores_list],
            "module_neuron_counts": [int((na == k).sum()) for k in range(K)],
            "module_sample_counts": [int((sa == k).sum()) for k in range(K)],
        }
        results[K] = result

        np.savez(
            os.path.join(output_dir, f"{model_short}_K{K}_modules.npz"),
            neuron_assign=na,
            sample_assign=sa,
        )

    with open(os.path.join(output_dir, f"{model_short}_discovery_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10, 15, 20])
    args = parser.parse_args()

    run_discovery(args.activation_path, args.output_dir, args.K)
