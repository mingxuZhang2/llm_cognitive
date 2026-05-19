"""
Phase 6: Brain Network Graph-Theoretic Comparison.

Compute graph metrics (modularity Q, small-worldness, clustering coefficient,
participation coefficient) for LLM module networks and compare with
published human brain network values.
"""

import json
import os
import numpy as np
import networkx as nx
from pathlib import Path
from itertools import combinations


# Published human brain network metrics (from Sporns, Bullmore, Meunier et al.)
BRAIN_REFERENCE = {
    "source": "Bullmore & Sporns (2009), Meunier et al. (2010), HCP parcellation studies",
    "modularity_Q": {"mean": 0.45, "std": 0.08, "range": [0.3, 0.6]},
    "clustering_coefficient": {"mean": 0.53, "std": 0.10, "range": [0.35, 0.70]},
    "small_worldness_sigma": {"mean": 2.5, "std": 0.8, "range": [1.5, 4.0]},
    "characteristic_path_length_ratio": {"mean": 1.05, "std": 0.05, "range": [1.0, 1.2]},
    "n_modules_typical": {"range": [4, 8], "note": "varies with parcellation resolution"},
}


def build_module_coactivation_graph(A, neuron_assign, K):
    """
    Build a weighted graph where nodes are modules and edges
    represent cross-module coactivation strength.

    Edge weight(i,j) = mean activation of module-i neurons on module-j samples.
    """
    G = nx.Graph()
    for k in range(K):
        n_neurons = (neuron_assign == k).sum()
        G.add_node(k, size=int(n_neurons))

    for ki, kj in combinations(range(K), 2):
        ui_mask = neuron_assign == ki
        uj_mask = neuron_assign == kj

        # Cross-activation: module-i neurons responding to module-j samples
        # (we need sample assignments too, but can use activation patterns)
        cross_ij = A[ui_mask, :][:, uj_mask].mean() if ui_mask.any() and uj_mask.any() else 0
        cross_ji = A[uj_mask, :][:, ui_mask].mean() if uj_mask.any() and ui_mask.any() else 0
        weight = (cross_ij + cross_ji) / 2

        if weight > 0:
            G.add_edge(ki, kj, weight=float(weight))

    return G


def build_module_network_from_activations(A, neuron_assign, sample_assign, K):
    """
    Build module-level network from activation patterns.

    Nodes = modules, edges = correlation of module activation profiles across samples.
    """
    n_samples = A.shape[1]
    module_profiles = np.zeros((K, n_samples))

    for k in range(K):
        u_mask = neuron_assign == k
        if u_mask.sum() > 0:
            module_profiles[k] = A[u_mask, :].mean(axis=0)

    corr = np.corrcoef(module_profiles)
    np.fill_diagonal(corr, 0)
    corr = np.abs(corr)

    G = nx.Graph()
    for k in range(K):
        G.add_node(k, size=int((neuron_assign == k).sum()))

    threshold = np.percentile(corr[corr > 0], 50) if (corr > 0).any() else 0
    for ki in range(K):
        for kj in range(ki + 1, K):
            if corr[ki, kj] > threshold:
                G.add_edge(ki, kj, weight=float(corr[ki, kj]))

    return G, corr


def compute_graph_metrics(G):
    """Compute standard graph-theoretic metrics."""
    metrics = {}

    n = G.number_of_nodes()
    m = G.number_of_edges()
    metrics["n_nodes"] = n
    metrics["n_edges"] = m
    metrics["density"] = nx.density(G)

    if nx.is_connected(G):
        metrics["avg_path_length"] = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        metrics["avg_path_length"] = nx.average_shortest_path_length(subG)
        metrics["n_components"] = nx.number_connected_components(G)

    metrics["clustering_coefficient"] = nx.average_clustering(G, weight="weight")
    metrics["transitivity"] = nx.transitivity(G)

    if n > 0 and m > 0:
        try:
            communities = nx.community.greedy_modularity_communities(G, weight="weight")
            metrics["modularity_Q"] = nx.community.modularity(G, communities, weight="weight")
            metrics["n_communities"] = len(communities)
        except Exception:
            metrics["modularity_Q"] = None

    metrics["degree_sequence"] = sorted([d for _, d in G.degree()], reverse=True)
    metrics["avg_degree"] = np.mean([d for _, d in G.degree()])

    return metrics


def compute_small_worldness(G, n_random=100, seed=42):
    """
    Compute small-worldness σ = (C/C_rand) / (L/L_rand).

    σ > 1 indicates small-world properties.
    """
    rng = np.random.RandomState(seed)

    C_real = nx.average_clustering(G)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
    else:
        G_connected = G

    L_real = nx.average_shortest_path_length(G_connected)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    C_rands = []
    L_rands = []
    for _ in range(n_random):
        G_rand = nx.gnm_random_graph(n, m, seed=rng.randint(0, 100000))
        if nx.is_connected(G_rand) and G_rand.number_of_edges() > 0:
            C_rands.append(nx.average_clustering(G_rand))
            L_rands.append(nx.average_shortest_path_length(G_rand))

    if not C_rands or not L_rands:
        return {"sigma": None, "note": "could not generate connected random graphs"}

    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)

    gamma = C_real / C_rand if C_rand > 0 else float('inf')
    lam = L_real / L_rand if L_rand > 0 else float('inf')
    sigma = gamma / lam if lam > 0 else float('inf')

    return {
        "sigma": float(sigma),
        "gamma": float(gamma),
        "lambda": float(lam),
        "C_real": float(C_real),
        "C_rand_mean": float(C_rand),
        "L_real": float(L_real),
        "L_rand_mean": float(L_rand),
    }


def participation_coefficient(G, module_assign_nodes):
    """
    Compute participation coefficient for each node.

    P_i = 1 - sum_k (k_is / k_i)^2
    where k_is is the number of edges from node i to module s.
    High P → hub connecting multiple modules.
    """
    P = {}
    modules = set(module_assign_nodes.values())

    for node in G.nodes():
        k_i = G.degree(node)
        if k_i == 0:
            P[node] = 0.0
            continue

        sum_sq = 0
        for m in modules:
            k_is = sum(1 for neighbor in G.neighbors(node)
                       if module_assign_nodes.get(neighbor) == m)
            sum_sq += (k_is / k_i) ** 2

        P[node] = 1 - sum_sq

    return P


def run_brain_comparison(
    activation_paths: dict,
    module_paths: dict,
    K: int,
    output_path: str,
):
    """
    Run full brain comparison analysis for all models.

    Args:
        activation_paths: {model_name: path_to_activations.npy}
        module_paths: {model_name: path_to_modules.npz}
        K: number of modules
        output_path: where to save results
    """
    all_results = {}

    for model_name in activation_paths:
        print(f"\n{'='*60}")
        print(f"Brain comparison: {model_name}")
        print(f"{'='*60}")

        A = np.load(activation_paths[model_name])
        modules = np.load(module_paths[model_name])
        neuron_assign = modules["neuron_assign"]
        sample_assign = modules["sample_assign"]

        G, corr_matrix = build_module_network_from_activations(
            A, neuron_assign, sample_assign, K
        )

        metrics = compute_graph_metrics(G)
        sw = compute_small_worldness(G)
        P = participation_coefficient(G, {k: k for k in range(K)})

        model_results = {
            "graph_metrics": metrics,
            "small_worldness": sw,
            "participation_coefficients": {int(k): float(v) for k, v in P.items()},
            "correlation_matrix": corr_matrix.tolist(),
        }

        # Compare with brain reference
        comparisons = {}
        brain_Q = BRAIN_REFERENCE["modularity_Q"]
        if metrics.get("modularity_Q") is not None:
            comparisons["modularity_Q"] = {
                "llm": metrics["modularity_Q"],
                "brain_mean": brain_Q["mean"],
                "brain_range": brain_Q["range"],
                "within_brain_range": brain_Q["range"][0] <= metrics["modularity_Q"] <= brain_Q["range"][1],
            }

        brain_C = BRAIN_REFERENCE["clustering_coefficient"]
        comparisons["clustering_coefficient"] = {
            "llm": metrics["clustering_coefficient"],
            "brain_mean": brain_C["mean"],
            "brain_range": brain_C["range"],
            "within_brain_range": brain_C["range"][0] <= metrics["clustering_coefficient"] <= brain_C["range"][1],
        }

        brain_sw = BRAIN_REFERENCE["small_worldness_sigma"]
        if sw.get("sigma") is not None:
            comparisons["small_worldness"] = {
                "llm": sw["sigma"],
                "brain_mean": brain_sw["mean"],
                "brain_range": brain_sw["range"],
                "within_brain_range": brain_sw["range"][0] <= sw["sigma"] <= brain_sw["range"][1],
            }

        model_results["brain_comparison"] = comparisons
        all_results[model_name] = model_results

        print(f"  Modularity Q: {metrics.get('modularity_Q', 'N/A'):.4f} (brain: {brain_Q['mean']:.2f}±{brain_Q['std']:.2f})")
        print(f"  Clustering:   {metrics['clustering_coefficient']:.4f} (brain: {brain_C['mean']:.2f}±{brain_C['std']:.2f})")
        if sw.get("sigma"):
            print(f"  Small-world σ: {sw['sigma']:.4f} (brain: {brain_sw['mean']:.1f}±{brain_sw['std']:.1f})")
        print(f"  Hub modules (high P): {[k for k, v in P.items() if v > 0.5]}")

    output = {
        "models": all_results,
        "brain_reference": BRAIN_REFERENCE,
        "K": K,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--activation_dir", type=str, required=True)
    parser.add_argument("--module_dir", type=str, required=True)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    act_paths = {m: os.path.join(args.activation_dir, f"{m}_activations.npy") for m in args.models}
    mod_paths = {m: os.path.join(args.module_dir, f"{m}_K{args.K}_modules.npz") for m in args.models}

    run_brain_comparison(act_paths, mod_paths, args.K, args.output)
