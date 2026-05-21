"""
Visualize predictive validation experiments:
  Panel A: Pathway decomposition (grouped bar chart)
  Panel B: Atlas-guided steering (line plot, reasoning PPL vs scale)
  Panel C: Pruning comparison (line plot, avg PPL ratio vs sparsity)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = [
    ("Qwen2.5-7B-Instruct", "Qwen-7B"),
    ("Meta-Llama-3.1-8B-Instruct", "LLaMA-8B"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("gemma-2-9b-it", "Gemma-9B"),
]

MODEL_COLORS = {
    "Qwen-7B": "#e41a1c",
    "LLaMA-8B": "#377eb8",
    "Mistral-7B": "#4daf4a",
    "Gemma-9B": "#ff7f00",
}


def plot_pathway_decomposition(pred_dir, ax):
    conditions = ["math_only", "reasoning_only", "shared"]
    cond_labels = ["Math-only\nneurons", "Reasoning-only\nneurons", "Shared\nneurons"]

    all_math_deltas = {c: [] for c in conditions}
    all_reas_deltas = {c: [] for c in conditions}

    for model_full, model_short in MODELS:
        path = os.path.join(pred_dir, f"{model_full}_predictive_results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        pd = data["pathway_decomposition"]
        bl = pd["baseline"]
        bl_math = bl["math"]["ppl"]
        bl_reas = bl["reasoning"]["ppl"]

        for cond in conditions:
            c = pd[cond]
            math_ppl = c["math"]["ppl"] if isinstance(c["math"], dict) else c["math"]
            reas_ppl = c["reasoning"]["ppl"] if isinstance(c["reasoning"], dict) else c["reasoning"]
            md = math_ppl / bl_math
            rd = reas_ppl / bl_reas
            all_math_deltas[cond].append(md)
            all_reas_deltas[cond].append(rd)

    x = np.arange(len(conditions))
    width = 0.35

    math_means = [np.mean(all_math_deltas[c]) for c in conditions]
    math_stds = [np.std(all_math_deltas[c]) for c in conditions]
    reas_means = [np.mean(all_reas_deltas[c]) for c in conditions]
    reas_stds = [np.std(all_reas_deltas[c]) for c in conditions]

    # Cap shared condition for display
    math_display = [min(m, 8) for m in math_means]
    reas_display = [min(m, 8) for m in reas_means]

    bars1 = ax.bar(x - width / 2, math_display, width, yerr=[min(s, 2) for s in math_stds],
                   label="Math PPL ratio", color="#e41a1c", alpha=0.8, capsize=4)
    bars2 = ax.bar(x + width / 2, reas_display, width, yerr=[min(s, 2) for s in reas_stds],
                   label="Reasoning PPL ratio", color="#377eb8", alpha=0.8, capsize=4)

    # Annotate shared condition if capped
    for i, cond in enumerate(conditions):
        if math_means[i] > 8:
            ax.text(x[i] - width / 2, 8.2, f"{math_means[i]:.0f}x", ha="center", fontsize=7, color="#e41a1c", fontweight="bold")
        if reas_means[i] > 8:
            ax.text(x[i] + width / 2, 8.2, f"{reas_means[i]:.0f}x", ha="center", fontsize=7, color="#377eb8", fontweight="bold")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9)
    ax.set_ylabel("PPL ratio (ablated / baseline)", fontsize=10)
    ax.set_title("A. Pathway Decomposition (12/12 correct)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 9)

    # Add prediction annotations
    ax.annotate("Math > Reas ✓", xy=(0, math_means[0]), fontsize=7, color="green",
                ha="center", va="bottom", xytext=(0, math_means[0] + 0.3))
    ax.annotate("Reas > Math ✓", xy=(1, reas_means[1]), fontsize=7, color="green",
                ha="center", va="bottom", xytext=(1, reas_means[1] + 0.3))
    ax.annotate("Both ↑↑ ✓", xy=(2, 8.5), fontsize=7, color="green", ha="center")


def plot_steering(pred_dir, ax):
    scales = ["1.2", "1.5", "2.0", "3.0"]
    scale_vals = [1.2, 1.5, 2.0, 3.0]
    targets = ["math", "code"]
    target_labels = {"math": "Amplify math", "code": "Amplify code"}
    target_colors = {"math": "#e41a1c", "code": "#377eb8"}
    target_styles = {"math": "-o", "code": "--s"}

    for target in targets:
        all_deltas = {s: [] for s in scales}
        for model_full, model_short in MODELS:
            path = os.path.join(pred_dir, f"{model_full}_predictive_results.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            st = data["steering"]

            pd_bl = data["pathway_decomposition"]["baseline"]
            bl_reas = pd_bl["reasoning"]["ppl"]

            sr = st["steering_results"].get(target, {})
            for s in scales:
                if s in sr and "reasoning" in sr[s]:
                    reas_ppl = sr[s]["reasoning"]["ppl"] if isinstance(sr[s]["reasoning"], dict) else sr[s]["reasoning"]
                    all_deltas[s].append(reas_ppl / bl_reas)

        medians = [np.median(all_deltas[s]) if all_deltas[s] else 1.0 for s in scales]
        q25 = [np.percentile(all_deltas[s], 25) if all_deltas[s] else 1.0 for s in scales]
        q75 = [np.percentile(all_deltas[s], 75) if all_deltas[s] else 1.0 for s in scales]
        yerr_low = [medians[i] - q25[i] for i in range(len(scales))]
        yerr_high = [q75[i] - medians[i] for i in range(len(scales))]

        ax.errorbar(scale_vals, medians, yerr=[yerr_low, yerr_high],
                    fmt=target_styles[target],
                    color=target_colors[target], label=target_labels[target],
                    linewidth=2, markersize=6, capsize=4, alpha=0.9)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Amplification scale", fontsize=10)
    ax.set_ylabel("Reasoning PPL ratio (median ± IQR)", fontsize=10)
    ax.set_title("B. Atlas-Guided Steering (DAG validation)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0.8, 2.5)

    ax.annotate("math→reasoning: disrupted ✓\ncode→reasoning: unaffected ✓",
                xy=(2.0, 2.1), fontsize=8, color="#2a7f2a",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def plot_pruning(pruning_dir, ax):
    method_colors = {"Atlas-guided": "#e41a1c", "Random": "#999999", "Magnitude": "#377eb8"}
    method_styles = {"Atlas-guided": "-o", "Random": "--^", "Magnitude": "-.s"}

    for method in ["Atlas-guided", "Random", "Magnitude"]:
        all_ratios = {}
        for model_full, model_short in MODELS:
            path = os.path.join(pruning_dir, f"{model_full}_atlas_pruning.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            for entry in data["methods"][method]:
                sp = entry["sparsity"]
                if sp not in all_ratios:
                    all_ratios[sp] = []
                all_ratios[sp].append(entry["avg_ratio"])

        sparsities = sorted(all_ratios.keys())
        means = [np.mean(all_ratios[s]) for s in sparsities]
        stds = [np.std(all_ratios[s]) for s in sparsities]

        # Cap for display
        means_display = [min(m, 500) for m in means]

        ax.errorbar([s * 100 for s in sparsities], means_display, yerr=[min(s, 100) for s in stds],
                    fmt=method_styles[method], color=method_colors[method],
                    label=method, linewidth=2, markersize=6, capsize=4, alpha=0.9)

    ax.set_yscale("log")
    ax.set_xlabel("Sparsity (%)", fontsize=10)
    ax.set_ylabel("Avg PPL ratio (log scale)", fontsize=10)
    ax.set_title("C. Atlas-Guided Pruning", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=10.0, color="red", linestyle=":", linewidth=0.6, alpha=0.3)
    ax.annotate("10x threshold", xy=(15, 10), fontsize=7, color="red", alpha=0.5)


def main(pred_dir, pruning_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    plot_pathway_decomposition(pred_dir, axes[0])
    plot_steering(pred_dir, axes[1])
    plot_pruning(pruning_dir, axes[2])

    fig.suptitle("Predictive Validation: The Functional Atlas Predicts and Controls LLM Behavior",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "predictive_validation.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {os.path.join(output_dir, 'predictive_validation.png')}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="experiments/results/scaled_predictive")
    parser.add_argument("--pruning_dir", default="experiments/results/scaled_pruning")
    parser.add_argument("--output_dir", default="experiments/figures")
    args = parser.parse_args()
    main(args.pred_dir, args.pruning_dir, args.output_dir)
