"""
Deep analysis of multi-function dissociation data to surface actual findings.

Not "we can identify task neurons" (tautological), but:
1. Cross-function dependency structure (which functions share neurons?)
2. Functional hierarchy (is reasoning built on math?)
3. Cross-model universality (same depth ordering across architectures?)
4. Functional core size (how much of the network is truly needed?)
5. Asymmetric spillover patterns (does A→B but not B→A?)
"""

import json
import os
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr


MODELS = [
    ("Qwen2.5-7B-Instruct", "Qwen-7B"),
    ("Meta-Llama-3.1-8B-Instruct", "LLaMA-8B"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("gemma-2-9b-it", "Gemma-9B"),
]

CAT_ORDER = ["math", "code", "reasoning", "language", "science", "ethics", "factual_qa", "humanities"]


def load_all_data(dissoc_dir, layer_stats_dir):
    dissoc = {}
    stats = {}
    for model_full, model_short in MODELS:
        dp = os.path.join(dissoc_dir, f"{model_full}_multi_dissociation.json")
        sp = os.path.join(layer_stats_dir, f"{model_full}_layer_stats.json")
        if os.path.exists(dp):
            with open(dp) as f:
                dissoc[model_short] = json.load(f)
        if os.path.exists(sp):
            with open(sp) as f:
                stats[model_short] = json.load(f)
    return dissoc, stats


def analysis_1_dependency_structure(dissoc, output_dir):
    """Which functions depend on each other? Look at off-diagonal spillover."""
    print("\n" + "="*70)
    print("ANALYSIS 1: Cross-Function Dependency Structure")
    print("="*70)
    print("  Ablating X-selective neurons → how much does it hurt Y?")
    print("  Asymmetric spillover reveals functional hierarchy.\n")

    for model_short, data in dissoc.items():
        cats = data["categories"]
        cat_idx = {c: cats.index(c) for c in CAT_ORDER if c in cats}
        ordered_cats = [c for c in CAT_ORDER if c in cat_idx]
        order = [cat_idx[c] for c in ordered_cats]

        matrix = np.array(data["dissociation_matrix_logppl_delta"])
        matrix = matrix[np.ix_(order, order)]
        n = len(ordered_cats)

        print(f"  {model_short}:")
        print(f"    {'Ablate →':>12s}  Top 3 collateral damage (off-diagonal):")

        for i in range(n):
            row = matrix[i].copy()
            target_damage = row[i]
            row[i] = -999  # exclude diagonal
            top3_idx = np.argsort(-row)[:3]
            items = []
            for j in top3_idx:
                if row[j] > 0.01:
                    items.append(f"{ordered_cats[j]}({row[j]:+.3f})")
            if items:
                print(f"    {ordered_cats[i]:>12s}: {', '.join(items)}")

        # Asymmetry analysis: A→B vs B→A
        print(f"\n    Asymmetric dependencies (A→B > B→A by >2x):")
        for i in range(n):
            for j in range(i+1, n):
                a_to_b = matrix[i, j]
                b_to_a = matrix[j, i]
                if a_to_b > 0.02 and b_to_a > 0.02:
                    ratio = a_to_b / b_to_a if b_to_a > 0 else float("inf")
                    if ratio > 2:
                        print(f"      {ordered_cats[i]:>12s} → {ordered_cats[j]:<12s}: {a_to_b:+.3f} vs reverse {b_to_a:+.3f} (ratio {ratio:.1f}x)")
                    elif 1/ratio > 2:
                        print(f"      {ordered_cats[j]:>12s} → {ordered_cats[i]:<12s}: {b_to_a:+.3f} vs reverse {a_to_b:+.3f} (ratio {1/ratio:.1f}x)")

        # Reasoning dependency: which functions does reasoning borrow from?
        if "reasoning" in cat_idx:
            r_idx = ordered_cats.index("reasoning")
            print(f"\n    Reasoning dependency profile (which ablations hurt reasoning):")
            deps = []
            for i in range(n):
                if i != r_idx:
                    deps.append((ordered_cats[i], matrix[i, r_idx]))
            deps.sort(key=lambda x: -x[1])
            for cat, val in deps:
                bar = "█" * int(max(0, val) * 30)
                print(f"      Ablate {cat:>12s} → reasoning: {val:+.4f} {bar}")

        print()


def analysis_2_functional_hierarchy(dissoc, output_dir):
    """Cluster functions by their damage profiles to find hierarchy."""
    print("\n" + "="*70)
    print("ANALYSIS 2: Functional Hierarchy via Damage Profile Clustering")
    print("="*70)
    print("  Functions with similar damage profiles share neural substrates.\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (model_short, data) in enumerate(dissoc.items()):
        cats = data["categories"]
        cat_idx = {c: cats.index(c) for c in CAT_ORDER if c in cats}
        ordered_cats = [c for c in CAT_ORDER if c in cat_idx]
        order = [cat_idx[c] for c in ordered_cats]

        matrix = np.array(data["dissociation_matrix_logppl_delta"])
        matrix = matrix[np.ix_(order, order)]

        # Each function's "damage profile" = which other functions it affects
        # Cluster functions by similarity of their damage columns
        # Column j = "how much does ablating each function hurt function j"
        profiles = matrix.T  # [n_cats, n_cats] — row j = damage profile of function j

        dist = pdist(profiles, metric="correlation")
        Z = linkage(dist, method="ward")

        ax = axes[idx]
        dendrogram(Z, labels=ordered_cats, ax=ax, leaf_rotation=45)
        ax.set_title(f"{model_short}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Distance")

        print(f"  {model_short}: Functional clusters (Ward linkage on damage profiles)")
        # Print cluster structure
        from scipy.cluster.hierarchy import fcluster
        for n_clusters in [2, 3, 4]:
            clusters = fcluster(Z, n_clusters, criterion="maxclust")
            cluster_groups = {}
            for i, c in enumerate(clusters):
                if c not in cluster_groups:
                    cluster_groups[c] = []
                cluster_groups[c].append(ordered_cats[i])
            groups_str = " | ".join([", ".join(g) for g in cluster_groups.values()])
            print(f"    K={n_clusters}: [{groups_str}]")
        print()

    fig.suptitle("Functional Hierarchy: Clustering by Shared Neural Substrates",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "functional_hierarchy.png"), dpi=200, bbox_inches="tight")
    print(f"  Saved: functional_hierarchy.png")


def analysis_3_cross_model_universality(dissoc, stats, output_dir):
    """Do all models organize functions in the same depth order?"""
    print("\n" + "="*70)
    print("ANALYSIS 3: Cross-Model Universality of Functional Organization")
    print("="*70)

    # For each model, compute "center of mass" depth for each function
    depth_profiles = {}  # {model: {cat: center_of_mass}}

    for model_short, stat in stats.items():
        n_layers = stat["n_layers"]
        depth_profiles[model_short] = {}
        for cat in CAT_ORDER:
            if cat not in stat.get("top5000_layer_dist", {}):
                continue
            counts = np.array(stat["top5000_layer_dist"][cat], dtype=float)
            if counts.sum() > 0:
                # Normalized depth (0-1)
                layers_norm = np.arange(n_layers) / (n_layers - 1)
                center = np.average(layers_norm, weights=counts)
                spread = np.sqrt(np.average((layers_norm - center)**2, weights=counts))
                depth_profiles[model_short][cat] = {"center": center, "spread": spread}

    # Print depth ordering per model
    print("\n  Functional depth ordering (center of mass, 0=first layer, 1=last):")
    for model_short in depth_profiles:
        cats_sorted = sorted(depth_profiles[model_short].items(), key=lambda x: x[1]["center"])
        ordering = " → ".join([f"{c}({v['center']:.2f})" for c, v in cats_sorted])
        print(f"    {model_short:>10s}: {ordering}")

    # Cross-model correlation of depth orderings
    print("\n  Cross-model Spearman correlation of depth orderings:")
    model_names = list(depth_profiles.keys())
    common_cats = sorted(set.intersection(*[set(depth_profiles[m].keys()) for m in model_names]))

    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            depths1 = [depth_profiles[m1][c]["center"] for c in common_cats]
            depths2 = [depth_profiles[m2][c]["center"] for c in common_cats]
            rho, p = spearmanr(depths1, depths2)
            print(f"    {m1:>10s} vs {m2:<10s}: ρ={rho:.3f}, p={p:.4f}")

    # Average depth per function across models
    print("\n  Average functional depth across all models:")
    avg_depths = {}
    for cat in common_cats:
        depths = [depth_profiles[m][cat]["center"] for m in model_names]
        avg_depths[cat] = np.mean(depths)
    for cat, d in sorted(avg_depths.items(), key=lambda x: x[1]):
        bar = "█" * int(d * 40)
        print(f"    {cat:>12s}: {d:.3f} {bar}")

    # Spread analysis: which functions are concentrated vs distributed?
    print("\n  Functional spread (lower = more concentrated, higher = more distributed):")
    for model_short in depth_profiles:
        print(f"    {model_short}:")
        for cat in common_cats:
            spread = depth_profiles[model_short][cat]["spread"]
            bar = "█" * int(spread * 60)
            print(f"      {cat:>12s}: σ={spread:.3f} {bar}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(common_cats))
    width = 0.18
    for mi, model_short in enumerate(model_names):
        centers = [depth_profiles[model_short][c]["center"] for c in common_cats]
        spreads = [depth_profiles[model_short][c]["spread"] for c in common_cats]
        ax.bar(x_pos + mi * width, centers, width, yerr=spreads, label=model_short, alpha=0.8)
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(common_cats, rotation=45, ha="right")
    ax.set_ylabel("Normalized depth (0=early, 1=late)")
    ax.set_title("Functional Depth Across Models")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "depth_universality.png"), dpi=200, bbox_inches="tight")
    print(f"\n  Saved: depth_universality.png")


def analysis_4_functional_core(dissoc, output_dir):
    """How much of the network is truly functional vs redundant?"""
    print("\n" + "="*70)
    print("ANALYSIS 4: Functional Core Size")
    print("="*70)
    print("  From dissociation data: each function uses ~5000 causal neurons.")
    print("  Total unique functional neurons = union across all categories.\n")

    for model_short, data in dissoc.items():
        n_neurons = data["n_neurons"]
        n_ablate = data["n_ablate"]
        n_cats = len(data["categories"])
        # Upper bound: n_cats * n_ablate (if no overlap)
        max_functional = n_cats * n_ablate
        # The actual union might be smaller due to overlap
        pct_max = max_functional / n_neurons * 100
        print(f"  {model_short}: {n_neurons:,} total neurons")
        print(f"    {n_cats} functions × {n_ablate:,} causal neurons = {max_functional:,} ({pct_max:.1f}%) upper bound")
        print(f"    Even if all unique: {100 - pct_max:.1f}% of network is potentially redundant")
        print()


def analysis_5_spillover_matrix(dissoc, output_dir):
    """Create a clean asymmetric spillover matrix — the real finding."""
    print("\n" + "="*70)
    print("ANALYSIS 5: Asymmetric Spillover Matrix (averaged across models)")
    print("="*70)
    print("  Averaged off-diagonal values reveal stable cross-function dependencies.\n")

    all_matrices = []
    for model_short, data in dissoc.items():
        cats = data["categories"]
        cat_idx = {c: cats.index(c) for c in CAT_ORDER if c in cats}
        ordered_cats = [c for c in CAT_ORDER if c in cat_idx]
        order = [cat_idx[c] for c in ordered_cats]
        matrix = np.array(data["dissociation_matrix_logppl_delta"])
        matrix = matrix[np.ix_(order, order)]
        all_matrices.append(matrix)

    avg_matrix = np.mean(all_matrices, axis=0)
    std_matrix = np.std(all_matrices, axis=0)
    n = avg_matrix.shape[0]

    # Print averaged matrix
    header = f"  {'Ablated':>12s} |" + "".join(f" {c:>8s}" for c in ordered_cats)
    print(header)
    print(f"  {'-'*len(header)}")
    for i in range(n):
        row = f"  {ordered_cats[i]:>12s} |"
        for j in range(n):
            v = avg_matrix[i, j]
            s = std_matrix[i, j]
            if i == j:
                row += f" {v:>7.3f}*"
            elif v > 0.05:
                row += f" {v:>7.3f}!"
            else:
                row += f" {v:>8.3f}"
        print(row)

    # Identify consistent spillovers (>0.05 in avg AND low std)
    print(f"\n  Consistent cross-function spillovers (avg > 0.05, across ≥3 models):")
    spillovers = []
    for i in range(n):
        for j in range(n):
            if i != j and avg_matrix[i, j] > 0.05:
                # Count in how many models this spillover is > 0.03
                n_consistent = sum(1 for m in all_matrices if m[i, j] > 0.03)
                if n_consistent >= 3:
                    spillovers.append((ordered_cats[i], ordered_cats[j], avg_matrix[i, j], std_matrix[i, j], n_consistent))

    spillovers.sort(key=lambda x: -x[2])
    for src, tgt, avg, std, nc in spillovers:
        print(f"    Ablate {src:>12s} → hurts {tgt:<12s}: avg={avg:+.3f} ± {std:.3f} ({nc}/4 models)")

    # Save figure
    fig, ax = plt.subplots(figsize=(10, 8))
    # Only show off-diagonal
    display_matrix = avg_matrix.copy()
    np.fill_diagonal(display_matrix, 0)
    im = ax.imshow(display_matrix, cmap="YlOrRd", interpolation="nearest", vmin=0, vmax=0.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered_cats, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered_cats)
    ax.set_xlabel("Affected function")
    ax.set_ylabel("Ablated function")
    ax.set_title("Cross-Function Spillover (off-diagonal, averaged across 4 models)")
    for i in range(n):
        for j in range(n):
            if i != j and display_matrix[i, j] > 0.03:
                ax.text(j, i, f"{display_matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="Log-PPL increase")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spillover_matrix.png"), dpi=200, bbox_inches="tight")
    print(f"\n  Saved: spillover_matrix.png")


def main(dissoc_dir, layer_stats_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dissoc, stats = load_all_data(dissoc_dir, layer_stats_dir)

    analysis_1_dependency_structure(dissoc, output_dir)
    analysis_2_functional_hierarchy(dissoc, output_dir)
    analysis_3_cross_model_universality(dissoc, stats, output_dir)
    analysis_4_functional_core(dissoc, output_dir)
    analysis_5_spillover_matrix(dissoc, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dissoc_dir", type=str, required=True)
    parser.add_argument("--layer_stats_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.dissoc_dir, args.layer_stats_dir, args.output_dir)
