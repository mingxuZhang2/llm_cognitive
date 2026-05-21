"""
Cross-layer distribution + neuron overlap analysis.
Runs locally from pre-computed attribution .npz files (no GPU needed).

Analysis 1: Cross-Layer Distribution
  - For each category, compute layer-wise density of top-5000 neurons
  - Test hypothesis: hierarchy bottom (language/code) → early layers,
    hierarchy top (reasoning/ethics) → late layers

Analysis 2: Neuron Overlap (Jaccard Index)
  - 8x8 Jaccard index matrix of top-5000 neuron sets
  - Compare overlap structure with dependency DAG
  - Categories with causal dependency should share more neurons

Analysis 3: Layer Concentration (Entropy)
  - How concentrated vs distributed are each category's neurons across layers
  - Formal (math) should be concentrated, distributed (reasoning) should be spread
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


MODELS = {
    "Qwen2.5-7B-Instruct": {"n_layers": 28, "ffn_dim": 18944},
    "Meta-Llama-3.1-8B-Instruct": {"n_layers": 32, "ffn_dim": 14336},
    "Mistral-7B-Instruct-v0.3": {"n_layers": 32, "ffn_dim": 14336},
    "gemma-2-9b-it": {"n_layers": 42, "ffn_dim": 14336},
}

CATEGORIES = ["math", "code", "reasoning", "language", "science", "ethics", "factual_qa", "humanities"]

N_TOP = 5000


def get_top_neurons(attr_data, category, n_top=N_TOP):
    selectivity = attr_data[f"{category}_selectivity"]
    return np.argsort(selectivity)[-n_top:]


def neuron_to_layer(neuron_idx, ffn_dim):
    return neuron_idx // ffn_dim


def layer_distribution(top_neurons, n_layers, ffn_dim):
    layers = np.array([neuron_to_layer(n, ffn_dim) for n in top_neurons])
    dist = np.zeros(n_layers)
    for l in layers:
        if l < n_layers:
            dist[l] += 1
    return dist / dist.sum()


def layer_entropy(dist):
    dist = dist[dist > 0]
    return -np.sum(dist * np.log2(dist))


def weighted_mean_layer(dist):
    layers = np.arange(len(dist))
    return np.sum(layers * dist)


def jaccard_index(set_a, set_b):
    a = set(set_a)
    b = set(set_b)
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0


def run_analysis(attr_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, meta in MODELS.items():
        npz_path = os.path.join(attr_dir, f"{model_name}_multi_attribution.npz")
        if not os.path.exists(npz_path):
            print(f"  Skipping {model_name}: no attribution file")
            continue

        print(f"\n{'='*70}")
        print(f"{model_name} (layers={meta['n_layers']}, ffn_dim={meta['ffn_dim']})")
        print(f"{'='*70}")

        attr_data = np.load(npz_path)
        n_layers = meta["n_layers"]
        ffn_dim = meta["ffn_dim"]

        top_neurons = {}
        layer_dists = {}
        mean_layers = {}
        entropies = {}

        for cat in CATEGORIES:
            top_n = get_top_neurons(attr_data, cat)
            top_neurons[cat] = top_n
            dist = layer_distribution(top_n, n_layers, ffn_dim)
            layer_dists[cat] = dist
            mean_layers[cat] = weighted_mean_layer(dist)
            entropies[cat] = layer_entropy(dist)

        # Analysis 1: Cross-Layer Distribution
        print(f"\n  [Analysis 1: Layer Distribution]")
        print(f"  {'Category':>12s}  {'Mean Layer':>10s}  {'Normalized':>10s}  {'Entropy':>8s}  {'Concentration':>13s}")
        for cat in sorted(CATEGORIES, key=lambda c: mean_layers[c]):
            norm = mean_layers[cat] / (n_layers - 1)
            conc = "concentrated" if entropies[cat] < np.log2(n_layers) * 0.7 else "distributed"
            print(f"  {cat:>12s}  {mean_layers[cat]:>10.2f}  {norm:>10.3f}  {entropies[cat]:>8.2f}  {conc:>13s}")

        # Analysis 2: Neuron Overlap (Jaccard)
        print(f"\n  [Analysis 2: Neuron Overlap (Jaccard Index)]")
        jaccard_matrix = {}
        header = f"  {'':>12s}"
        for c in CATEGORIES:
            header += f"  {c[:6]:>6s}"
        print(header)

        for c1 in CATEGORIES:
            jaccard_matrix[c1] = {}
            row = f"  {c1:>12s}"
            for c2 in CATEGORIES:
                j = jaccard_index(top_neurons[c1], top_neurons[c2])
                jaccard_matrix[c1][c2] = j
                row += f"  {j:>6.3f}"
            print(row)

        # Average overlap by dependency relationship
        dependency_pairs = [
            ("math", "reasoning"), ("science", "reasoning"),
            ("language", "reasoning"), ("science", "humanities"),
            ("humanities", "science"),
        ]
        non_dependency_pairs = [
            ("code", "reasoning"), ("code", "ethics"),
            ("math", "humanities"), ("factual_qa", "code"),
        ]

        dep_overlaps = [jaccard_matrix[a][b] for a, b in dependency_pairs if a in jaccard_matrix and b in jaccard_matrix.get(a, {})]
        nondep_overlaps = [jaccard_matrix[a][b] for a, b in non_dependency_pairs if a in jaccard_matrix and b in jaccard_matrix.get(a, {})]

        print(f"\n  Dependency pairs avg Jaccard: {np.mean(dep_overlaps):.4f}")
        print(f"  Non-dependency pairs avg Jaccard: {np.mean(nondep_overlaps):.4f}")
        print(f"  Ratio: {np.mean(dep_overlaps)/np.mean(nondep_overlaps):.2f}x")

        # Analysis 3: Layer profile similarity across models
        print(f"\n  [Analysis 3: Layer Concentration]")
        max_entropy = np.log2(n_layers)
        for cat in sorted(CATEGORIES, key=lambda c: entropies[c]):
            rel_ent = entropies[cat] / max_entropy
            bar = "#" * int(rel_ent * 40)
            print(f"  {cat:>12s}: entropy={entropies[cat]:.2f}/{max_entropy:.2f} ({rel_ent:.2%}) {bar}")

        all_results[model_name] = {
            "n_layers": n_layers,
            "ffn_dim": ffn_dim,
            "mean_layers": {c: float(mean_layers[c]) for c in CATEGORIES},
            "normalized_mean_layers": {c: float(mean_layers[c] / (n_layers - 1)) for c in CATEGORIES},
            "entropies": {c: float(entropies[c]) for c in CATEGORIES},
            "max_entropy": float(max_entropy),
            "jaccard_matrix": {c1: {c2: float(jaccard_matrix[c1][c2]) for c2 in CATEGORIES} for c1 in CATEGORIES},
            "layer_distributions": {c: layer_dists[c].tolist() for c in CATEGORIES},
            "dependency_overlap_mean": float(np.mean(dep_overlaps)),
            "non_dependency_overlap_mean": float(np.mean(nondep_overlaps)),
        }

    # Cross-model summary
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")

    print(f"\n  [Mean Layer Position (normalized 0=early, 1=late)]")
    header = f"  {'Category':>12s}"
    for m in MODELS:
        header += f"  {m[:8]:>8s}"
    header += "  {'Mean':>8s}"
    print(header)

    for cat in CATEGORIES:
        row = f"  {cat:>12s}"
        vals = []
        for m in MODELS:
            if m in all_results:
                v = all_results[m]["normalized_mean_layers"][cat]
                vals.append(v)
                row += f"  {v:>8.3f}"
            else:
                row += f"  {'N/A':>8s}"
        if vals:
            row += f"  {np.mean(vals):>8.3f}"
        print(row)

    print(f"\n  [Jaccard Overlap: Dependency vs Non-dependency]")
    for m in MODELS:
        if m in all_results:
            dep = all_results[m]["dependency_overlap_mean"]
            nondep = all_results[m]["non_dependency_overlap_mean"]
            ratio = dep / nondep if nondep > 0 else float("inf")
            print(f"  {m:40s}: dep={dep:.4f}, nondep={nondep:.4f}, ratio={ratio:.2f}x")

    # Layer ordering consistency
    print(f"\n  [Layer Ordering Consistency]")
    print(f"  Do hierarchy-bottom categories (language, code) sit in earlier layers")
    print(f"  than hierarchy-top categories (reasoning, ethics)?")
    bottom = ["language", "code"]
    top_cats = ["reasoning", "ethics"]
    for m in MODELS:
        if m not in all_results:
            continue
        nml = all_results[m]["normalized_mean_layers"]
        bottom_mean = np.mean([nml[c] for c in bottom])
        top_mean = np.mean([nml[c] for c in top_cats])
        consistent = bottom_mean < top_mean
        print(f"  {m:40s}: bottom={bottom_mean:.3f}, top={top_mean:.3f} {'CONSISTENT' if consistent else 'INCONSISTENT'}")

    out_path = os.path.join(output_dir, "layer_overlap_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr_dir", type=str, default="experiments/results/scaled_multi")
    parser.add_argument("--output_dir", type=str, default="experiments/results/layer_overlap")
    args = parser.parse_args()
    run_analysis(args.attr_dir, args.output_dir)
