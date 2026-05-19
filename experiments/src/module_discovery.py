"""
Phase 2: Discover functional modules via IterD dual partitioning.

Reimplements ULCMOD's IterD algorithm: jointly partition neurons and samples
into K mutually exclusive modules maximizing L(F) = ξ(F) × B(F).

Performance-critical: uses precomputed module sums for O(K) incremental updates
instead of O(N*S) full recomputation per reassignment.
"""

import json
import os
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


class ModuleState:
    """Maintains precomputed module statistics for fast incremental updates."""

    def __init__(self, A, neuron_assign, sample_assign, K):
        self.A = A
        self.K = K
        self.n_neurons, self.n_samples = A.shape
        self.neuron_assign = neuron_assign.copy()
        self.sample_assign = sample_assign.copy()

        self.neuron_counts = np.zeros(K, dtype=np.int64)
        self.sample_counts = np.zeros(K, dtype=np.int64)
        self.within_sums = np.zeros(K, dtype=np.float64)

        # neuron_to_module_sums[i, k] = sum of A[i, j] for all j in module k
        self.neuron_to_module_sums = np.zeros((self.n_neurons, K), dtype=np.float64)
        # module_neuron_sums[k, j] = sum of A[i, j] for all i in module k
        self.module_neuron_sums = np.zeros((K, self.n_samples), dtype=np.float64)

        for k in range(K):
            u_mask = neuron_assign == k
            s_mask = sample_assign == k
            self.neuron_counts[k] = u_mask.sum()
            self.sample_counts[k] = s_mask.sum()
            if u_mask.any():
                self.module_neuron_sums[k] = A[u_mask].sum(axis=0)
            if u_mask.any() and s_mask.any():
                self.within_sums[k] = A[np.ix_(u_mask, s_mask)].sum()

        for i in range(self.n_neurons):
            for k in range(K):
                s_mask = sample_assign == k
                if s_mask.any():
                    self.neuron_to_module_sums[i, k] = A[i, s_mask].sum()

    def compute_objective(self):
        total_within = 0.0
        total_count = 0
        inv_sum = 0.0
        for k in range(self.K):
            nu = self.neuron_counts[k]
            ns = self.sample_counts[k]
            size = nu * ns
            if size == 0:
                return 0.0
            total_within += self.within_sums[k]
            total_count += size
            inv_sum += 1.0 / size
        xi = total_within / max(total_count, 1)
        B = self.K / inv_sum if inv_sum > 0 else 0
        return xi * B

    def try_move_neuron(self, i, new_k):
        """Compute objective change if neuron i moves to new_k. O(K) operation."""
        old_k = self.neuron_assign[i]
        if old_k == new_k:
            return self.compute_objective()

        old_within_old = self.within_sums[old_k]
        old_within_new = self.within_sums[new_k]

        contribution_to_old = self.neuron_to_module_sums[i, old_k]
        contribution_to_new = self.neuron_to_module_sums[i, new_k]

        new_within_old = old_within_old - contribution_to_old
        new_within_new = old_within_new + contribution_to_new

        new_nc_old = self.neuron_counts[old_k] - 1
        new_nc_new = self.neuron_counts[new_k] + 1

        if new_nc_old == 0:
            return -1e10

        total_within = 0.0
        total_count = 0
        inv_sum = 0.0
        for k in range(self.K):
            if k == old_k:
                nu, ws = new_nc_old, new_within_old
            elif k == new_k:
                nu, ws = new_nc_new, new_within_new
            else:
                nu, ws = self.neuron_counts[k], self.within_sums[k]
            ns = self.sample_counts[k]
            size = nu * ns
            if size == 0:
                return -1e10
            total_within += ws
            total_count += size
            inv_sum += 1.0 / size

        xi = total_within / max(total_count, 1)
        B = self.K / inv_sum if inv_sum > 0 else 0
        return xi * B

    def apply_move_neuron(self, i, new_k):
        """Actually move neuron i to new_k and update all cached sums."""
        old_k = self.neuron_assign[i]

        self.within_sums[old_k] -= self.neuron_to_module_sums[i, old_k]
        self.within_sums[new_k] += self.neuron_to_module_sums[i, new_k]

        self.module_neuron_sums[old_k] -= self.A[i]
        self.module_neuron_sums[new_k] += self.A[i]

        self.neuron_counts[old_k] -= 1
        self.neuron_counts[new_k] += 1

        self.neuron_assign[i] = new_k

    def try_move_sample(self, j, new_k):
        """Compute objective change if sample j moves to new_k. O(K) operation."""
        old_k = self.sample_assign[j]
        if old_k == new_k:
            return self.compute_objective()

        contribution_to_old = self.module_neuron_sums[old_k, j]
        contribution_to_new = self.module_neuron_sums[new_k, j]

        new_within_old = self.within_sums[old_k] - contribution_to_old
        new_within_new = self.within_sums[new_k] + contribution_to_new

        new_sc_old = self.sample_counts[old_k] - 1
        new_sc_new = self.sample_counts[new_k] + 1

        if new_sc_old == 0:
            return -1e10

        total_within = 0.0
        total_count = 0
        inv_sum = 0.0
        for k in range(self.K):
            if k == old_k:
                ns, ws = new_sc_old, new_within_old
            elif k == new_k:
                ns, ws = new_sc_new, new_within_new
            else:
                ns, ws = self.sample_counts[k], self.within_sums[k]
            nu = self.neuron_counts[k]
            size = nu * ns
            if size == 0:
                return -1e10
            total_within += ws
            total_count += size
            inv_sum += 1.0 / size

        xi = total_within / max(total_count, 1)
        B = self.K / inv_sum if inv_sum > 0 else 0
        return xi * B

    def apply_move_sample(self, j, new_k):
        """Actually move sample j to new_k and update all cached sums."""
        old_k = self.sample_assign[j]

        self.within_sums[old_k] -= self.module_neuron_sums[old_k, j]
        self.within_sums[new_k] += self.module_neuron_sums[new_k, j]

        # Update neuron_to_module_sums for all neurons
        col = self.A[:, j]
        self.neuron_to_module_sums[:, old_k] -= col
        self.neuron_to_module_sums[:, new_k] += col

        self.sample_counts[old_k] -= 1
        self.sample_counts[new_k] += 1

        self.sample_assign[j] = new_k


def iterd(A, K, max_iter=100, seed=42, verbose=True):
    """
    IterD algorithm for dual partitioning with incremental updates.

    Complexity per iteration: O((N + S) * K) instead of O(N * S * K).
    """
    n_neurons, n_samples = A.shape
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"IterD: {n_neurons:,} neurons × {n_samples:,} samples, K={K}")

    n_components = min(50, n_samples, n_neurons)
    pca = PCA(n_components=n_components, random_state=seed)
    A_pca = pca.fit_transform(A)

    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    neuron_assign = km.fit_predict(A_pca)

    sample_assign = np.zeros(n_samples, dtype=int)
    for j in range(n_samples):
        scores_k = np.zeros(K)
        for k in range(K):
            u_mask = neuron_assign == k
            if u_mask.sum() > 0:
                scores_k[k] = A[u_mask, j].mean()
        sample_assign[j] = scores_k.argmax()

    if verbose:
        print("Building module state cache...")
    state = ModuleState(A, neuron_assign, sample_assign, K)
    L = state.compute_objective()

    scores = [L]
    if verbose:
        print(f"Init: L(F) = {L:.4f}")

    for it in range(max_iter):
        changed = 0

        # Neuron reassignment step
        for i in rng.permutation(n_neurons):
            old_k = state.neuron_assign[i]
            best_k = old_k
            best_L = L

            for k in range(K):
                if k == old_k:
                    continue
                new_L = state.try_move_neuron(i, k)
                if new_L > best_L:
                    best_L = new_L
                    best_k = k

            if best_k != old_k:
                state.apply_move_neuron(i, best_k)
                L = best_L
                changed += 1

        # Sample reassignment step
        for j in rng.permutation(n_samples):
            old_k = state.sample_assign[j]
            best_k = old_k
            best_L = L

            for k in range(K):
                if k == old_k:
                    continue
                new_L = state.try_move_sample(j, k)
                if new_L > best_L:
                    best_L = new_L
                    best_k = k

            if best_k != old_k:
                state.apply_move_sample(j, best_k)
                L = best_L
                changed += 1

        scores.append(L)
        if verbose:
            print(f"Iter {it+1}: L(F) = {L:.4f}, changed = {changed}")

        if changed == 0:
            if verbose:
                print(f"Converged at iteration {it+1}")
            break

    return state.neuron_assign, state.sample_assign, scores


def align_modules_across_models(assignments_list, K):
    """Align module indices across models using Hungarian algorithm on sample overlap."""
    ref_sample_assign = assignments_list[0][1]
    permutations = [np.arange(K)]

    for m in range(1, len(assignments_list)):
        target_sample_assign = assignments_list[m][1]
        cost = np.zeros((K, K))
        for k_ref in range(K):
            for k_tgt in range(K):
                ref_set = set(np.where(ref_sample_assign == k_ref)[0])
                tgt_set = set(np.where(target_sample_assign == k_tgt)[0])
                cost[k_ref, k_tgt] = -len(ref_set & tgt_set)
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = np.zeros(K, dtype=int)
        perm[col_ind] = row_ind
        permutations.append(perm)

    return permutations


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
        print(f"Running IterD with K={K} for {model_short}")
        print(f"{'='*60}")

        neuron_assign, sample_assign, scores_list = iterd(A, K, verbose=True)

        result = {
            "K": K,
            "final_score": scores_list[-1],
            "n_iterations": len(scores_list),
            "scores": [float(s) for s in scores_list],
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, nargs="+", default=[5, 10, 15, 20])
    args = parser.parse_args()

    run_discovery(args.activation_path, args.output_dir, args.K)
