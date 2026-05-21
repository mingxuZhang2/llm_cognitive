"""
Visualize functional atlas: layer-wise distribution of causally selective neurons.

Generates:
1. Heatmap: categories x layers (selective neuron density)
2. Cross-model comparison panel
3. Dissociation matrix heatmaps
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


MODELS = [
    ("Qwen2.5-7B-Instruct", "Qwen-7B"),
    ("Meta-Llama-3.1-8B-Instruct", "LLaMA-8B"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("gemma-2-9b-it", "Gemma-9B"),
]

CAT_COLORS = {
    "math": "#e41a1c",
    "code": "#377eb8",
    "reasoning": "#4daf4a",
    "language": "#984ea3",
    "science": "#ff7f00",
    "ethics": "#a65628",
    "factual_qa": "#f781bf",
    "humanities": "#999999",
}

CAT_ORDER = ["math", "code", "reasoning", "language", "science", "ethics", "factual_qa", "humanities"]


def load_layer_stats(stats_dir):
    all_stats = {}
    for model_full, model_short in MODELS:
        path = os.path.join(stats_dir, f"{model_full}_layer_stats.json")
        if os.path.exists(path):
            with open(path) as f:
                all_stats[model_short] = json.load(f)
    return all_stats


def plot_single_atlas(stats, model_name, ax, threshold="n_selective_05"):
    """Plot heatmap for one model: rows=categories, cols=layers."""
    n_layers = stats["n_layers"]
    cats = [c for c in CAT_ORDER if c in stats["categories"]]
    n_cats = len(cats)

    matrix = np.zeros((n_cats, n_layers))
    for j in range(n_layers):
        layer_data = stats["layers"][str(j)]
        for i, cat in enumerate(cats):
            if cat in layer_data:
                matrix[i, j] = layer_data[cat][threshold]

    # Normalize each row to [0, 1] for visualization
    row_maxes = matrix.max(axis=1, keepdims=True)
    row_maxes[row_maxes == 0] = 1
    matrix_norm = matrix / row_maxes

    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_title(model_name, fontsize=11, fontweight="bold")

    # Add layer ticks
    tick_interval = 4 if n_layers > 20 else 2
    ax.set_xticks(range(0, n_layers, tick_interval))
    ax.set_xticklabels(range(0, n_layers, tick_interval), fontsize=8)

    return im


def plot_top5000_distribution(stats, model_name, ax):
    """Stacked bar: per-layer distribution of top-5000 selective neurons."""
    n_layers = stats["n_layers"]
    cats = [c for c in CAT_ORDER if c in stats.get("top5000_layer_dist", {})]
    if not cats:
        return

    x = np.arange(n_layers)
    bottoms = np.zeros(n_layers)

    for cat in cats:
        counts = np.array(stats["top5000_layer_dist"][cat])
        ax.bar(x, counts, bottom=bottoms, color=CAT_COLORS.get(cat, "#888"),
               label=cat, width=0.8, alpha=0.85)
        bottoms += counts

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("# selective neurons", fontsize=10)
    ax.set_title(f"{model_name}: Top-5000 per category", fontsize=11, fontweight="bold")
    tick_interval = 4 if n_layers > 20 else 2
    ax.set_xticks(range(0, n_layers, tick_interval))
    ax.legend(fontsize=7, ncol=4, loc="upper right")


def plot_dissociation_matrix(dissoc_path, model_name, ax):
    """Plot 8x8 dissociation matrix as heatmap."""
    with open(dissoc_path) as f:
        data = json.load(f)

    cats = data["categories"]
    cat_order = [c for c in CAT_ORDER if c in cats]
    idx_map = {c: cats.index(c) for c in cat_order}

    matrix = np.array(data["dissociation_matrix_logppl_delta"])
    order = [idx_map[c] for c in cat_order]
    matrix = matrix[np.ix_(order, order)]

    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=-0.05, vmax=1.3, interpolation="nearest")
    ax.set_xticks(range(len(cat_order)))
    ax.set_xticklabels(cat_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels(cat_order, fontsize=9)
    ax.set_xlabel("Measured category", fontsize=10)
    ax.set_ylabel("Ablated category", fontsize=10)
    ax.set_title(model_name, fontsize=12, fontweight="bold")

    for i in range(len(cat_order)):
        for j in range(len(cat_order)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            if i == j:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)
            elif abs(val) > 0.04:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    return im


def main(stats_dir, dissoc_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_stats = load_layer_stats(stats_dir)

    if not all_stats:
        print("No layer stats found!")
        return

    # === Figure 1: 4-panel atlas heatmap ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (model_full, model_short) in enumerate(MODELS):
        if model_short in all_stats:
            im = plot_single_atlas(all_stats[model_short], model_short, axes[idx])
    fig.suptitle("Functional Atlas: Layer-wise Distribution of Causally Selective Neurons",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Normalized selective neuron density")
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(os.path.join(output_dir, "atlas_heatmap.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: atlas_heatmap.png")
    plt.close()

    # === Figure 2: Stacked bar distribution ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (model_full, model_short) in enumerate(MODELS):
        if model_short in all_stats:
            plot_top5000_distribution(all_stats[model_short], model_short, axes[idx])
    fig.suptitle("Top-5000 Causally Selective Neurons per Function (by Layer)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "atlas_stacked_bar.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: atlas_stacked_bar.png")
    plt.close()

    # === Figure 3: 4-panel dissociation matrices ===
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    axes_flat = axes.flatten()
    im = None
    for idx, (model_full, model_short) in enumerate(MODELS):
        path = os.path.join(dissoc_dir, f"{model_full}_multi_dissociation.json")
        if os.path.exists(path):
            im = plot_dissociation_matrix(path, model_short, axes_flat[idx])
    fig.suptitle("8-way Functional Dissociation Matrices (Log-PPL delta)",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    if im is not None:
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_label("Log-PPL increase (ablated − baseline)", fontsize=11)
    fig.savefig(os.path.join(output_dir, "dissociation_matrices.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: dissociation_matrices.png")
    plt.close()

    # === Figure 4: Cross-model layer similarity ===
    # For each category, compare layer profiles across models
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for cat_idx, cat in enumerate(CAT_ORDER):
        ax = axes[cat_idx]
        for model_full, model_short in MODELS:
            if model_short not in all_stats:
                continue
            stats = all_stats[model_short]
            if cat not in stats.get("top5000_layer_dist", {}):
                continue
            counts = np.array(stats["top5000_layer_dist"][cat])
            n_layers = stats["n_layers"]
            # Normalize x-axis to [0, 1] for cross-model comparison
            x_norm = np.arange(n_layers) / (n_layers - 1)
            counts_norm = counts / counts.sum() if counts.sum() > 0 else counts
            ax.plot(x_norm, counts_norm, label=model_short, alpha=0.8, linewidth=1.5)

        ax.set_title(cat, fontsize=11, fontweight="bold", color=CAT_COLORS.get(cat, "black"))
        ax.set_xlabel("Relative depth", fontsize=9)
        ax.set_ylabel("Neuron density", fontsize=9)
        if cat_idx == 0:
            ax.legend(fontsize=7)
        ax.set_xlim(0, 1)

    fig.suptitle("Cross-Model Functional Neuron Distribution (normalized depth)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cross_model_comparison.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: cross_model_comparison.png")
    plt.close()

    print("\nAll figures generated!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_dir", type=str, required=True)
    parser.add_argument("--dissoc_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.stats_dir, args.dissoc_dir, args.output_dir)
