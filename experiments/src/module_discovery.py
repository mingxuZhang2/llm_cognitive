"""
Phase 2: Discover functional modules via IterD dual partitioning.

Reimplements ULCMOD's IterD algorithm: jointly partition neurons and samples
into K mutually exclusive modules maximizing L(F) = ξ(F) × B(F).
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


def activation_modularity(A, neuron_assign, sample_assign, K):
    """Compute ξ(F) = mean within-module activation."""
    total = 0.0
    count = 0
    for k in range(K):
        u_mask = neuron_assign == k
        s_mask = sample_assign == k
        if u_mask.sum() == 0 or s_mask.sum() == 0:
            continue
        block = A[np.ix_(u_mask, s_mask)]
        total += block.sum()
        count += u_mask.sum() * s_mask.sum()
    return total / max(count, 1)


def balance_score(neuron_assign, sample_assign, K):
    """Compute B(F) = harmonic mean of module sizes."""
    inv_sum = 0.0
    for k in range(K):
        nu = (neuron_assign == k).sum()
        ns = (sample_assign == k).sum()
        size = nu * ns
        if size == 0:
            return 0.0
        inv_sum += 1.0 / size
    return K / inv_sum


def objective(A, neuron_assign, sample_assign, K):
    """L(F) = ξ(F) × B(F)"""
    xi = activation_modularity(A, neuron_assign, sample_assign, K)
    b = balance_score(neuron_assign, sample_assign, K)
    return xi * b


def iterd(A, K, max_iter=100, seed=42, verbose=True):
    """
    IterD algorithm for dual partitioning.

    Args:
        A: activation matrix [n_neurons, n_samples], z-score normalized
        K: number of modules
        max_iter: maximum iterations
        seed: random seed
        verbose: print progress

    Returns:
        neuron_assign: [n_neurons] array of module indices
        sample_assign: [n_samples] array of module indices
        scores: list of L(F) per iteration
    """
    n_neurons, n_samples = A.shape
    rng = np.random.RandomState(seed)

    # Initialization: K-Means on PCA-reduced neuron activations
    n_components = min(50, n_samples, n_neurons)
    pca = PCA(n_components=n_components, random_state=seed)
    A_pca = pca.fit_transform(A)  # [n_neurons, n_components]

    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    neuron_assign = km.fit_predict(A_pca)

    # Initialize sample assignments from neuron assignments
    sample_assign = np.zeros(n_samples, dtype=int)
    for j in range(n_samples):
        scores_k = np.zeros(K)
        for k in range(K):
            u_mask = neuron_assign == k
            if u_mask.sum() > 0:
                scores_k[k] = A[u_mask, j].mean()
        sample_assign[j] = scores_k.argmax()

    scores = []
    L = objective(A, neuron_assign, sample_assign, K)
    scores.append(L)
    if verbose:
        print(f"Init: L(F) = {L:.4f}")

    for it in range(max_iter):
        changed = 0

        # Neuron reassignment step
        for i in rng.permutation(n_neurons):
            old_k = neuron_assign[i]
            best_k = old_k
            best_L = L

            for k in range(K):
                if k == old_k:
                    continue
                neuron_assign[i] = k
                new_L = objective(A, neuron_assign, sample_assign, K)
                if new_L > best_L:
                    best_L = new_L
                    best_k = k

            if best_k != old_k:
                neuron_assign[i] = best_k
                L = best_L
                changed += 1
            else:
                neuron_assign[i] = old_k

        # Sample reassignment step
        for j in rng.permutation(n_samples):
            old_k = sample_assign[j]
            best_k = old_k
            best_L = L

            for k in range(K):
                if k == old_k:
                    continue
                sample_assign[j] = k
                new_L = objective(A, neuron_assign, sample_assign, K)
                if new_L > best_L:
                    best_L = new_L
                    best_k = k

            if best_k != old_k:
                sample_assign[j] = best_k
                L = best_L
                changed += 1
            else:
                sample_assign[j] = old_k

        scores.append(L)
        if verbose:
            print(f"Iter {it+1}: L(F) = {L:.4f}, changed = {changed}")

        if changed == 0:
            if verbose:
                print(f"Converged at iteration {it+1}")
            break

    return neuron_assign, sample_assign, scores


def align_modules_across_models(assignments_list, A_list, K):
    """
    Align module indices across models using Hungarian algorithm.

    Args:
        assignments_list: list of (neuron_assign, sample_assign) per model
        A_list: list of activation matrices per model
        K: number of modules

    Returns:
        permutations: list of index permutations to align each model to the first
    """
    ref_sample_assign = assignments_list[0][1]
    permutations = [np.arange(K)]

    for m in range(1, len(assignments_list)):
        target_sample_assign = assignments_list[m][1]

        cost = np.zeros((K, K))
        for k_ref in range(K):
            for k_tgt in range(K):
                ref_set = set(np.where(ref_sample_assign == k_ref)[0])
                tgt_set = set(np.where(target_sample_assign == k_tgt)[0])
                overlap = len(ref_set & tgt_set)
                cost[k_ref, k_tgt] = -overlap  # negative because we minimize

        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(K, dtype=int)
        perm[col_ind] = row_ind
        permutations.append(perm)

    return permutations


def run_discovery(activation_path, output_dir, K_values=[5, 10, 15, 20]):
    """Run module discovery for multiple K values."""
    A = np.load(activation_path)
    meta_path = activation_path.replace("_activations.npy", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    model_short = Path(activation_path).stem.replace("_activations", "")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for K in K_values:
        print(f"\n{'='*60}")
        print(f"Running IterD with K={K} for {model_short}")
        print(f"{'='*60}")

        neuron_assign, sample_assign, scores = iterd(A, K, verbose=True)

        result = {
            "K": K,
            "final_score": scores[-1],
            "n_iterations": len(scores),
            "scores": [float(s) for s in scores],
            "module_neuron_counts": [int((neuron_assign == k).sum()) for k in range(K)],
            "module_sample_counts": [int((sample_assign == k).sum()) for k in range(K)],
        }
        results[K] = result

        np.savez(
            os.path.join(output_dir, f"{model_short}_K{K}_modules.npz"),
            neuron_assign=neuron_assign,
            sample_assign=sample_assign,
        )

    with open(os.path.join(output_dir, f"{model_short}_discovery_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10, 15, 20])
    args = parser.parse_args()

    run_discovery(args.activation_path, args.output_dir, args.K)
