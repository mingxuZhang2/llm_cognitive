#!/usr/bin/env python3
"""
Per-condition double dissociation from the 14x14 causal coupling matrices.

Reanalyzes existing coupling data: for each condition, compares the causal effect
of ablating its top neurons on SAME-block vs CROSS-block conditions. If same-block
effects are systematically larger than cross-block effects, this is the per-condition
double dissociation (emotion neurons hurt emotion more, social neurons hurt social more).

Reads: experiments/results/brain_causal_coupling/{model}_brain_causal_coupling.json
Writes:
  - experiments/results/clinical_dissociation/coupling_reanalysis.json
  - experiments/figures/coupling_dissociation_bar.png
  - experiments/figures/coupling_matrix_heatmap.png
  - experiments/figures/coupling_2x2_summary.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------- constants ----------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUPLING_DIR = os.path.join(BASE, "results", "brain_causal_coupling")
OUT_DIR = os.path.join(BASE, "results", "clinical_dissociation")
FIG_DIR = os.path.join(BASE, "figures")

MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]

MODEL_SHORT = {
    "Qwen2.5-7B-Instruct": "Qwen",
    "Meta-Llama-3.1-8B-Instruct": "Llama",
    "Mistral-7B-Instruct-v0.3": "Mistral",
    "gemma-2-9b-it": "Gemma",
}

AFFECTIVE = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
SOCIAL = {"belief", "mentalizing", "intention", "theory_of_mind",
          "empathy", "self_referential", "judgment", "moral"}

# Display order: affective first, then social
AFF_ORDER = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
SOC_ORDER = ["belief", "mentalizing", "intention", "theory_of_mind",
             "empathy", "self_referential", "judgment", "moral"]
DISPLAY_ORDER = AFF_ORDER + SOC_ORDER


# ---------- helpers ----------

def load_coupling(model_name: str):
    """Load coupling matrix and condition list from JSON."""
    path = os.path.join(COUPLING_DIR, f"{model_name}_brain_causal_coupling.json")
    with open(path) as f:
        data = json.load(f)
    conditions = data["conditions"]  # alphabetical order
    matrix = np.array(data["coupling_matrix_logppl"])  # [ablated, measured]
    return conditions, matrix


def reorder_matrix(conditions, matrix, order):
    """Reorder rows and columns from alphabetical to display order."""
    idx = [conditions.index(c) for c in order]
    return matrix[np.ix_(idx, idx)]


def block_membership(order):
    """Return boolean arrays for affective and social membership."""
    aff_mask = np.array([c in AFFECTIVE for c in order])
    soc_mask = np.array([c in SOCIAL for c in order])
    return aff_mask, soc_mask


def compute_per_condition_selectivity(matrix, conditions):
    """
    For each condition i, compute:
      same_block_effect = mean coupling[i, j] for j in same block, j != i
      cross_block_effect = mean coupling[i, j] for j in other block
      selectivity_ratio = same_block / cross_block  (>1 = selective)
    """
    n = len(conditions)
    aff_mask, soc_mask = block_membership(conditions)
    results = {}

    for i, cond in enumerate(conditions):
        is_aff = cond in AFFECTIVE
        same_mask = aff_mask if is_aff else soc_mask
        cross_mask = soc_mask if is_aff else aff_mask

        # Exclude self from same-block
        same_idx = [j for j in range(n) if same_mask[j] and j != i]
        cross_idx = [j for j in range(n) if cross_mask[j]]

        same_effect = np.mean(matrix[i, same_idx]) if same_idx else 0.0
        cross_effect = np.mean(matrix[i, cross_idx]) if cross_idx else 0.0

        # Ratio (handle edge cases)
        if cross_effect > 0:
            ratio = same_effect / cross_effect
        elif same_effect > 0 and cross_effect <= 0:
            ratio = float("inf")  # same positive, cross zero/negative
        else:
            ratio = float("nan")

        results[cond] = {
            "same_block_effect": float(same_effect),
            "cross_block_effect": float(cross_effect),
            "selectivity_ratio": float(ratio),
            "block": "affective" if is_aff else "social",
        }

    return results


def compute_2x2_summary(matrix, conditions):
    """
    Build the 2x2 block summary:
      A_aa = mean coupling[aff_i, aff_j] for i != j
      A_as = mean coupling[aff_i, soc_j]
      A_sa = mean coupling[soc_i, aff_j]
      A_ss = mean coupling[soc_i, soc_j] for i != j
    """
    n = len(conditions)
    aff_mask, soc_mask = block_membership(conditions)

    vals = {"A_aa": [], "A_as": [], "A_sa": [], "A_ss": []}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if aff_mask[i] and aff_mask[j]:
                vals["A_aa"].append(matrix[i, j])
            elif aff_mask[i] and soc_mask[j]:
                vals["A_as"].append(matrix[i, j])
            elif soc_mask[i] and aff_mask[j]:
                vals["A_sa"].append(matrix[i, j])
            elif soc_mask[i] and soc_mask[j]:
                vals["A_ss"].append(matrix[i, j])

    summary = {k: float(np.mean(v)) for k, v in vals.items()}
    # Double dissociation check
    summary["dd_affective"] = summary["A_aa"] > summary["A_as"]
    summary["dd_social"] = summary["A_ss"] > summary["A_sa"]
    summary["double_dissociation"] = summary["dd_affective"] and summary["dd_social"]
    return summary


def run_wilcoxon_test(selectivity_results):
    """
    Paired Wilcoxon signed-rank test: are same-block effects > cross-block effects
    across all 14 conditions?
    """
    same = []
    cross = []
    for cond in DISPLAY_ORDER:
        r = selectivity_results[cond]
        same.append(r["same_block_effect"])
        cross.append(r["cross_block_effect"])
    same = np.array(same)
    cross = np.array(cross)
    diff = same - cross

    # Wilcoxon signed-rank (alternative: same > cross)
    try:
        stat, p = wilcoxon(diff, alternative="greater")
    except ValueError:
        # All differences might be zero
        stat, p = float("nan"), 1.0

    return {
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "n_same_greater": int(np.sum(diff > 0)),
        "n_total": len(diff),
        "mean_diff": float(np.mean(diff)),
    }


# ---------- figures ----------

def fig_dissociation_bar(all_selectivity, save_path):
    """
    Grouped bar chart: 14 conditions, 2 bars each (same-block, cross-block).
    """
    n_models = len(all_selectivity)
    fig, axes = plt.subplots(n_models, 1, figsize=(16, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(len(DISPLAY_ORDER))
    width = 0.35

    for ax_idx, (model, sel) in enumerate(all_selectivity.items()):
        ax = axes[ax_idx]
        same_vals = [sel[c]["same_block_effect"] for c in DISPLAY_ORDER]
        cross_vals = [sel[c]["cross_block_effect"] for c in DISPLAY_ORDER]

        # Color by block
        same_colors = ["#c0392b" if c in AFFECTIVE else "#2471a3" for c in DISPLAY_ORDER]
        cross_colors = ["#e6b0aa" if c in AFFECTIVE else "#aed6f1" for c in DISPLAY_ORDER]

        bars1 = ax.bar(x - width/2, same_vals, width, color=same_colors, edgecolor="white",
                       linewidth=0.5, label="Same-block effect")
        bars2 = ax.bar(x + width/2, cross_vals, width, color=cross_colors, edgecolor="white",
                       linewidth=0.5, label="Cross-block effect")

        ax.set_ylabel("log(PPL_abl / PPL_base)", fontsize=11)
        short = MODEL_SHORT.get(model, model)
        ax.set_title(f"{short}", fontsize=13, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        # Vertical line between blocks
        ax.axvline(len(AFF_ORDER) - 0.5, color="black", linewidth=1.5, linestyle=":")

        # Block labels
        ax.text(len(AFF_ORDER)/2 - 0.5, ax.get_ylim()[1]*0.92, "AFFECTIVE",
                ha="center", fontsize=10, color="#c0392b", fontweight="bold")
        ax.text(len(AFF_ORDER) + len(SOC_ORDER)/2 - 0.5, ax.get_ylim()[1]*0.92, "SOCIAL",
                ha="center", fontsize=10, color="#2471a3", fontweight="bold")

        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # x labels on bottom
    axes[-1].set_xticks(x)
    labels = [c.replace("_", "\n") for c in DISPLAY_ORDER]
    axes[-1].set_xticklabels(labels, fontsize=9, rotation=0, ha="center")

    fig.suptitle("Per-condition Double Dissociation: Same-block vs Cross-block Causal Effect",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig_coupling_heatmap(all_matrices, all_conditions, save_path):
    """
    14x14 coupling matrix heatmap, averaged across models.
    Conditions reordered: affective first, then social. Block boundaries drawn.
    """
    # Average across models (all reordered to DISPLAY_ORDER)
    reordered = []
    for model in MODELS:
        m = reorder_matrix(all_conditions[model], all_matrices[model], DISPLAY_ORDER)
        reordered.append(m)
    avg_matrix = np.mean(reordered, axis=0)

    fig, ax = plt.subplots(figsize=(10, 8.5))

    # Mask diagonal for cleaner visual
    display = avg_matrix.copy()
    np.fill_diagonal(display, np.nan)

    vmax = np.nanpercentile(np.abs(display), 97)
    im = ax.imshow(display, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")

    # Labels
    labels = [c.replace("_", "\n") for c in DISPLAY_ORDER]
    ax.set_xticks(range(len(DISPLAY_ORDER)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(DISPLAY_ORDER)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Effect measured ON", fontsize=11)
    ax.set_ylabel("Neurons ablated FROM", fontsize=11)

    # Block boundary lines
    b = len(AFF_ORDER) - 0.5
    ax.axhline(b, color="black", linewidth=2)
    ax.axvline(b, color="black", linewidth=2)

    # Quadrant labels
    n_aff, n_soc = len(AFF_ORDER), len(SOC_ORDER)
    ax.text(n_aff/2 - 0.5, n_aff/2 - 0.5, "Aff→Aff", ha="center", va="center",
            fontsize=12, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.text(n_aff + n_soc/2 - 0.5, n_aff/2 - 0.5, "Aff→Soc", ha="center", va="center",
            fontsize=12, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.text(n_aff/2 - 0.5, n_aff + n_soc/2 - 0.5, "Soc→Aff", ha="center", va="center",
            fontsize=12, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.text(n_aff + n_soc/2 - 0.5, n_aff + n_soc/2 - 0.5, "Soc→Soc", ha="center",
            va="center", fontsize=12, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="log(PPL_abl / PPL_base), 4-model mean")
    ax.set_title("14x14 Causal Coupling Matrix (4-model average)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig_2x2_summary(all_summaries, save_path):
    """
    2x2 block summary for each model + average.
    """
    n_panels = len(all_summaries) + 1  # +1 for average
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5))

    # Compute average
    avg_2x2 = {}
    for key in ["A_aa", "A_as", "A_sa", "A_ss"]:
        avg_2x2[key] = np.mean([s[key] for s in all_summaries.values()])

    panels = list(all_summaries.items()) + [("Average (4 models)", avg_2x2)]

    for idx, (label, s) in enumerate(panels):
        ax = axes[idx]
        mat_2x2 = np.array([[s["A_aa"], s["A_as"]],
                            [s["A_sa"], s["A_ss"]]])

        vmax = np.max(np.abs(mat_2x2)) * 1.1
        im = ax.imshow(mat_2x2, cmap="YlOrRd", vmin=0, vmax=vmax,
                       aspect="equal", interpolation="nearest")

        # Annotate cells
        for r in range(2):
            for c in range(2):
                val = mat_2x2[r, c]
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(c, r, f"{val:.4f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ON aff.", "ON soc."], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Ablate aff.", "Ablate soc."], fontsize=10)

        short = MODEL_SHORT.get(label, label)
        ax.set_title(short, fontsize=12, fontweight="bold")

        # Double dissociation markers
        dd_aff = s["A_aa"] > s["A_as"] if isinstance(s.get("dd_affective"), bool) else s["A_aa"] > s["A_as"]
        dd_soc = s["A_ss"] > s["A_sa"] if isinstance(s.get("dd_social"), bool) else s["A_ss"] > s["A_sa"]
        if dd_aff and dd_soc:
            ax.text(0.5, -0.25, "DD", ha="center", va="top", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", color="green")

    fig.suptitle("2x2 Block Summary: Double Dissociation Pattern",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ---------- main ----------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    all_selectivity = {}
    all_summaries = {}
    all_wilcoxon = {}
    all_matrices = {}
    all_conditions = {}

    print("=" * 80)
    print("COUPLING DISSOCIATION REANALYSIS")
    print("=" * 80)

    for model in MODELS:
        short = MODEL_SHORT[model]
        print(f"\n--- {short} ({model}) ---")

        conditions, matrix = load_coupling(model)
        all_matrices[model] = matrix
        all_conditions[model] = conditions

        # 1. Per-condition selectivity
        sel = compute_per_condition_selectivity(matrix, conditions)
        all_selectivity[model] = sel

        # 2. 2x2 summary
        summary = compute_2x2_summary(matrix, conditions)
        all_summaries[model] = summary

        # 3. Wilcoxon test
        wilc = run_wilcoxon_test(sel)
        all_wilcoxon[model] = wilc

        # Print 2x2 table
        print(f"\n  2x2 Block Summary:")
        print(f"  {'':>15s}  ON affective   ON social")
        print(f"  {'Ablate aff.':>15s}  {summary['A_aa']:>12.5f}   {summary['A_as']:>9.5f}")
        print(f"  {'Ablate soc.':>15s}  {summary['A_sa']:>12.5f}   {summary['A_ss']:>9.5f}")
        dd_str = "YES" if summary["double_dissociation"] else "NO"
        print(f"  Double dissociation (A_aa > A_as AND A_ss > A_sa): {dd_str}")

        # Count selectivity > 1
        aff_sel = [sel[c]["selectivity_ratio"] for c in AFF_ORDER
                   if not np.isinf(sel[c]["selectivity_ratio"])]
        soc_sel = [sel[c]["selectivity_ratio"] for c in SOC_ORDER
                   if not np.isinf(sel[c]["selectivity_ratio"])]

        n_aff_selective = sum(1 for c in AFF_ORDER if sel[c]["selectivity_ratio"] > 1)
        n_soc_selective = sum(1 for c in SOC_ORDER if sel[c]["selectivity_ratio"] > 1)

        print(f"\n  Selectivity ratios (same/cross, >1 = block-selective):")
        print(f"  {'Condition':>20s}  {'Block':>10s}  {'Same':>8s}  {'Cross':>8s}  {'Ratio':>8s}")
        for c in DISPLAY_ORDER:
            r = sel[c]
            ratio_str = f"{r['selectivity_ratio']:.3f}" if not np.isinf(r["selectivity_ratio"]) else "inf"
            print(f"  {c:>20s}  {r['block']:>10s}  {r['same_block_effect']:>8.5f}  "
                  f"{r['cross_block_effect']:>8.5f}  {ratio_str:>8s}")

        print(f"\n  Affective conditions with ratio > 1: {n_aff_selective}/{len(AFF_ORDER)}")
        print(f"  Social conditions with ratio > 1:    {n_soc_selective}/{len(SOC_ORDER)}")

        print(f"\n  Wilcoxon signed-rank (same > cross):")
        print(f"    W = {wilc['wilcoxon_stat']:.1f}, p = {wilc['wilcoxon_p']:.6f}")
        print(f"    Conditions with same > cross: {wilc['n_same_greater']}/{wilc['n_total']}")
        print(f"    Mean difference: {wilc['mean_diff']:.5f}")

    # ---- Cross-model average ----
    print("\n" + "=" * 80)
    print("CROSS-MODEL SUMMARY")
    print("=" * 80)

    # Average 2x2
    avg_2x2 = {}
    for key in ["A_aa", "A_as", "A_sa", "A_ss"]:
        avg_2x2[key] = float(np.mean([all_summaries[m][key] for m in MODELS]))
    avg_2x2["dd_affective"] = avg_2x2["A_aa"] > avg_2x2["A_as"]
    avg_2x2["dd_social"] = avg_2x2["A_ss"] > avg_2x2["A_sa"]
    avg_2x2["double_dissociation"] = avg_2x2["dd_affective"] and avg_2x2["dd_social"]

    print(f"\n  Average 2x2 Block Summary (4 models):")
    print(f"  {'':>15s}  ON affective   ON social")
    print(f"  {'Ablate aff.':>15s}  {avg_2x2['A_aa']:>12.5f}   {avg_2x2['A_as']:>9.5f}")
    print(f"  {'Ablate soc.':>15s}  {avg_2x2['A_sa']:>12.5f}   {avg_2x2['A_ss']:>9.5f}")
    dd_str = "YES" if avg_2x2["double_dissociation"] else "NO"
    print(f"  Double dissociation: {dd_str}")

    # Average selectivity
    avg_sel = {}
    for c in DISPLAY_ORDER:
        same_vals = [all_selectivity[m][c]["same_block_effect"] for m in MODELS]
        cross_vals = [all_selectivity[m][c]["cross_block_effect"] for m in MODELS]
        avg_same = float(np.mean(same_vals))
        avg_cross = float(np.mean(cross_vals))
        ratio = avg_same / avg_cross if avg_cross > 0 else float("inf")
        avg_sel[c] = {
            "same_block_effect": avg_same,
            "cross_block_effect": avg_cross,
            "selectivity_ratio": ratio,
            "block": "affective" if c in AFFECTIVE else "social",
        }

    n_aff_avg = sum(1 for c in AFF_ORDER if avg_sel[c]["selectivity_ratio"] > 1)
    n_soc_avg = sum(1 for c in SOC_ORDER if avg_sel[c]["selectivity_ratio"] > 1)
    print(f"\n  Average selectivity: {n_aff_avg}/{len(AFF_ORDER)} affective, "
          f"{n_soc_avg}/{len(SOC_ORDER)} social have ratio > 1")
    print(f"  Total: {n_aff_avg + n_soc_avg}/14 conditions block-selective")

    # Average Wilcoxon p
    ps = [all_wilcoxon[m]["wilcoxon_p"] for m in MODELS]
    print(f"\n  Wilcoxon p-values: {', '.join(f'{p:.6f}' for p in ps)}")
    print(f"  All significant (p < 0.05): {'YES' if all(p < 0.05 for p in ps) else 'NO'}")

    # ---- Figures ----
    print("\n--- Generating figures ---")
    fig_dissociation_bar(
        all_selectivity,
        os.path.join(FIG_DIR, "coupling_dissociation_bar.png")
    )
    fig_coupling_heatmap(
        all_matrices, all_conditions,
        os.path.join(FIG_DIR, "coupling_matrix_heatmap.png")
    )
    fig_2x2_summary(
        all_summaries,
        os.path.join(FIG_DIR, "coupling_2x2_summary.png")
    )

    # ---- Save JSON ----
    output = {
        "per_model": {},
        "cross_model_average": {
            "summary_2x2": avg_2x2,
            "selectivity": avg_sel,
            "wilcoxon_ps": {MODEL_SHORT[m]: all_wilcoxon[m]["wilcoxon_p"] for m in MODELS},
            "n_selective_affective": n_aff_avg,
            "n_selective_social": n_soc_avg,
        },
    }
    for model in MODELS:
        short = MODEL_SHORT[model]
        output["per_model"][short] = {
            "selectivity": {c: all_selectivity[model][c] for c in DISPLAY_ORDER},
            "summary_2x2": all_summaries[model],
            "wilcoxon": all_wilcoxon[model],
        }

    out_path = os.path.join(OUT_DIR, "coupling_reanalysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
