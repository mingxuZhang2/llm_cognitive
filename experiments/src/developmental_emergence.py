#!/usr/bin/env python3
"""
Direction C: Does LLM cognitive capability emergence follow the same
order as human developmental psychology?

For each Qwen model size (0.5B, 1.5B, 3B, 7B), compute per-condition
alignment with the brain RDM. If basic emotions reach brain-like
organization at smaller sizes while moral/ToM requires larger models,
this mirrors the human developmental trajectory.

Uses existing per-stimulus activation NPZ files + brain RDM.
No GPU needed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "developmental_emergence"
FIG = BASE / "figures"

SIZES = [
    ("0.5B", "Qwen2.5-0.5B-Instruct", 494),
    ("1.5B", "Qwen2.5-1.5B-Instruct", 1540),
    ("3B",   "Qwen2.5-3B-Instruct",   3090),
    ("7B",   "Qwen2.5-7B-Instruct",   7610),
]

# Developmental psychology ordering (approximate age of emergence)
# Earlier = lower number = should appear in smaller models
DEV_ORDER = {
    "fear":             1.0,  # basic emotion, ~6-12 months
    "anger":            1.0,
    "disgust":          1.0,
    "sadness":          1.0,
    "happiness":        1.0,
    "valence":          1.5,  # affective valence, ~1-2 years
    "empathy":          2.5,  # emotional empathy, ~2-3 years
    "intention":        3.0,  # understanding goals, ~2-3 years
    "belief":           4.0,  # understanding beliefs, ~3-4 years
    "theory_of_mind":   4.5,  # false belief task, ~4-5 years
    "mentalizing":      4.5,  # mental state reasoning, ~4-5 years
    "self_referential": 5.0,  # self-awareness develops gradually
    "moral":            7.0,  # moral reasoning, ~6-8 years (Kohlberg)
    "judgment":         7.0,  # evaluative judgment, ~6-8 years
}

AFFECTIVE = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
MENTALISTIC = {"belief", "mentalizing", "intention", "theory_of_mind",
               "empathy", "self_referential", "judgment", "moral"}


def rdm_cosine_centered(act):
    act = act - act.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - np.clip(a @ a.T, -1, 1)


def per_condition_alignment(brain_rdm, llm_rdm, conditions):
    """Per-condition row-wise Spearman with brain RDM."""
    n = len(conditions)
    results = {}
    for i, cond in enumerate(conditions):
        # Row i of both RDMs, excluding diagonal
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        row_brain = brain_rdm[i, mask]
        row_llm = llm_rdm[i, mask]
        rho, p = spearmanr(row_brain, row_llm)
        results[cond] = {"rho": float(rho), "p": float(p)}
    return results


def overall_rsa(brain_rdm, llm_rdm):
    triu = np.triu_indices(brain_rdm.shape[0], k=1)
    rho, p = spearmanr(brain_rdm[triu], llm_rdm[triu])
    return float(rho), float(p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    brain_data = np.load(RES / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain_data["rdm"].astype(np.float64)
    brain_conds = list(brain_data["conditions"])
    n_conds = len(brain_conds)
    print(f"Brain RDM: {n_conds} conditions: {brain_conds}")

    all_results = {}

    for size_label, model_short, n_params in SIZES:
        npz_path = RES / f"{model_short}_rsa_v2_per_stim.npz"
        if not npz_path.exists():
            print(f"[skip] {npz_path}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        per_stim = data["per_stim_activations"]  # (3, n_stim, n_layers, hidden)
        conditions = list(data["conditions"])
        pooling_names = list(data["pooling_names"])
        pool_idx = pooling_names.index("mean_all")
        n_layers = per_stim.shape[2]

        # Build condition centroids
        unique_conds = sorted(set(conditions))
        stim_cond = np.array([unique_conds.index(c) for c in conditions])
        cond_means = np.zeros((len(unique_conds), n_layers, per_stim.shape[-1]),
                              dtype=np.float32)
        for c in range(len(unique_conds)):
            idx = np.where(stim_cond == c)[0]
            cond_means[c] = per_stim[pool_idx, idx].mean(axis=0)

        # Reorder to match brain conditions
        order = [unique_conds.index(c) for c in brain_conds]
        cond_means = cond_means[order]

        # Per-layer analysis
        best_overall_rho = -1
        best_layer = 0
        per_layer_overall = []
        per_layer_per_cond = []

        for L in range(n_layers):
            act = cond_means[:, L, :].astype(np.float64)
            llm_rdm = rdm_cosine_centered(act)
            rho, p = overall_rsa(brain_rdm, llm_rdm)
            per_cond = per_condition_alignment(brain_rdm, llm_rdm, brain_conds)
            per_layer_overall.append({"layer": L, "rho": rho, "p": p})
            per_layer_per_cond.append(per_cond)
            if rho > best_overall_rho:
                best_overall_rho = rho
                best_layer = L

        # Extract per-condition alignment at peak layer
        peak_cond = per_layer_per_cond[best_layer]

        # Also compute per-condition alignment averaged over top-5 layers
        top5_layers = sorted(range(n_layers),
                             key=lambda l: per_layer_overall[l]["rho"],
                             reverse=True)[:5]
        avg_cond = {}
        for cond in brain_conds:
            rhos = [per_layer_per_cond[l][cond]["rho"] for l in top5_layers]
            avg_cond[cond] = {
                "rho_mean": float(np.mean(rhos)),
                "rho_std": float(np.std(rhos)),
                "rho_peak": peak_cond[cond]["rho"],
            }

        # Affective vs mentalistic cluster alignment
        aff_rhos = [avg_cond[c]["rho_mean"] for c in brain_conds if c in AFFECTIVE]
        ment_rhos = [avg_cond[c]["rho_mean"] for c in brain_conds if c in MENTALISTIC]

        all_results[size_label] = {
            "model": model_short,
            "n_params_M": n_params,
            "n_layers": n_layers,
            "peak_layer": best_layer,
            "peak_overall_rho": best_overall_rho,
            "per_condition_alignment": avg_cond,
            "affective_mean_rho": float(np.mean(aff_rhos)),
            "mentalistic_mean_rho": float(np.mean(ment_rhos)),
        }

        print(f"\n{size_label} ({model_short}): peak L{best_layer}, "
              f"overall ρ={best_overall_rho:.4f}")
        print(f"  Affective mean: {np.mean(aff_rhos):.4f}")
        print(f"  Mentalistic mean: {np.mean(ment_rhos):.4f}")
        for cond in brain_conds:
            dev = DEV_ORDER.get(cond, 5.0)
            print(f"  {cond:>20s} (dev={dev:.1f}): "
                  f"ρ={avg_cond[cond]['rho_mean']:+.4f} ± {avg_cond[cond]['rho_std']:.3f}")

    # ================================================================
    # Key test: does emergence order correlate with developmental order?
    # ================================================================
    print("\n" + "=" * 70)
    print("KEY TEST: Emergence order vs developmental order")
    print("=" * 70)

    # For each condition, compute "emergence score" = at what model size does
    # alignment first exceed a threshold (or the slope of alignment vs size)
    cond_emergence = {}
    for cond in brain_conds:
        sizes_M = []
        rhos = []
        for size_label, _, n_params in SIZES:
            if size_label in all_results:
                r = all_results[size_label]["per_condition_alignment"]
                if cond in r:
                    sizes_M.append(n_params)
                    rhos.append(r[cond]["rho_mean"])
        if len(rhos) >= 2:
            # "Maturity" = alignment at smallest size / alignment at largest size
            # High ratio = already mature at small size (early emergence)
            # Low ratio = needs bigger model (late emergence)
            maturity = rhos[0] / (rhos[-1] + 1e-6) if rhos[-1] > 0 else 0
            # Slope of improvement
            slope = (rhos[-1] - rhos[0]) / (sizes_M[-1] - sizes_M[0]) * 1000
            cond_emergence[cond] = {
                "rho_smallest": rhos[0],
                "rho_largest": rhos[-1],
                "maturity_ratio": maturity,
                "slope_per_B": slope,
                "rhos": rhos,
                "dev_order": DEV_ORDER.get(cond, 5.0),
            }

    # Correlation: developmental order vs maturity ratio
    dev_orders = [cond_emergence[c]["dev_order"] for c in cond_emergence]
    maturities = [cond_emergence[c]["maturity_ratio"] for c in cond_emergence]
    slopes = [cond_emergence[c]["slope_per_B"] for c in cond_emergence]
    rho_smallest = [cond_emergence[c]["rho_smallest"] for c in cond_emergence]

    # Early-developing functions should have HIGH maturity (already good at small size)
    # So developmental order should NEGATIVELY correlate with maturity
    rho_mat, p_mat = spearmanr(dev_orders, maturities)
    # Late-developing functions should have POSITIVE slope (still improving with size)
    rho_slope, p_slope = spearmanr(dev_orders, slopes)
    # Early-developing should have HIGH alignment at smallest model
    rho_small, p_small = spearmanr(dev_orders, rho_smallest)

    print(f"\n  Developmental order vs maturity ratio:  ρ={rho_mat:+.4f}  p={p_mat:.4f}")
    print(f"    (negative = early dev → already mature at 0.5B → SUPPORTS hypothesis)")
    print(f"  Developmental order vs slope:           ρ={rho_slope:+.4f}  p={p_slope:.4f}")
    print(f"    (positive = late dev → still improving with scale → SUPPORTS hypothesis)")
    print(f"  Developmental order vs ρ at 0.5B:       ρ={rho_small:+.4f}  p={p_small:.4f}")
    print(f"    (negative = early dev → high alignment at 0.5B → SUPPORTS hypothesis)")

    print(f"\nPer-condition detail:")
    for cond in sorted(cond_emergence, key=lambda c: DEV_ORDER.get(c, 5)):
        e = cond_emergence[cond]
        print(f"  {cond:>20s} (dev={e['dev_order']:.1f}): "
              f"ρ@0.5B={e['rho_smallest']:+.4f}  ρ@7B={e['rho_largest']:+.4f}  "
              f"maturity={e['maturity_ratio']:.3f}  slope={e['slope_per_B']:+.4f}")

    # ================================================================
    # Figures
    # ================================================================

    # Figure 1: Per-condition alignment across model sizes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    size_labels = [s[0] for s in SIZES if s[0] in all_results]
    size_params = [s[2] for s in SIZES if s[0] in all_results]

    # Left: affective conditions
    ax = axes[0]
    for cond in sorted(AFFECTIVE):
        rhos = [all_results[s]["per_condition_alignment"][cond]["rho_mean"]
                for s in size_labels]
        ax.plot(size_params, rhos, "o-", label=cond, linewidth=2, markersize=6)
    ax.set_xlabel("Model parameters (M)")
    ax.set_xscale("log")
    ax.set_ylabel("Per-condition brain alignment (ρ)")
    ax.set_title("Affective conditions (dev age: 1-2.5 yr)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: mentalistic conditions
    ax = axes[1]
    for cond in sorted(MENTALISTIC):
        rhos = [all_results[s]["per_condition_alignment"][cond]["rho_mean"]
                for s in size_labels]
        ax.plot(size_params, rhos, "o-", label=cond, linewidth=2, markersize=6)
    ax.set_xlabel("Model parameters (M)")
    ax.set_xscale("log")
    ax.set_ylabel("Per-condition brain alignment (ρ)")
    ax.set_title("Mentalistic conditions (dev age: 3-7 yr)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Direction C: Cognitive capability emergence across model sizes",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG / "developmental_emergence_curves.png", dpi=150)
    plt.close()
    print(f"\nSaved figure: {FIG / 'developmental_emergence_curves.png'}")

    # Figure 2: Developmental order vs emergence metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    conds_sorted = sorted(cond_emergence.keys())

    for ax, (metric, metric_name, expect) in zip(axes, [
        (maturities, "Maturity ratio (ρ@0.5B / ρ@7B)", "neg"),
        (slopes, "Slope (Δρ per billion params)", "pos"),
        (rho_smallest, "Alignment ρ at 0.5B", "neg"),
    ]):
        devs = [DEV_ORDER[c] for c in conds_sorted]
        vals = [cond_emergence[c][{
            "Maturity ratio (ρ@0.5B / ρ@7B)": "maturity_ratio",
            "Slope (Δρ per billion params)": "slope_per_B",
            "Alignment ρ at 0.5B": "rho_smallest",
        }[metric_name]] for c in conds_sorted]
        colors = ["tab:red" if c in AFFECTIVE else "tab:blue" for c in conds_sorted]
        ax.scatter(devs, vals, c=colors, s=80, zorder=3)
        for c, d, v in zip(conds_sorted, devs, vals):
            ax.annotate(c, (d, v), fontsize=7, ha="center", va="bottom",
                        rotation=30)
        r, p = spearmanr(devs, vals)
        ax.set_xlabel("Developmental order (years)")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Spearman ρ={r:+.3f}, p={p:.3f}\n(expect {expect})")
        ax.grid(alpha=0.3)

    plt.suptitle("Does LLM emergence order match human developmental order?",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG / "developmental_order_correlation.png", dpi=150)
    plt.close()
    print(f"Saved figure: {FIG / 'developmental_order_correlation.png'}")

    # Save results
    final = {
        "per_size": all_results,
        "emergence_analysis": cond_emergence,
        "developmental_order": DEV_ORDER,
        "correlation_maturity_vs_dev": {"rho": rho_mat, "p": p_mat},
        "correlation_slope_vs_dev": {"rho": rho_slope, "p": p_slope},
        "correlation_rho_smallest_vs_dev": {"rho": rho_small, "p": p_small},
    }
    with open(OUT / "developmental_emergence.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"Saved results: {OUT / 'developmental_emergence.json'}")


if __name__ == "__main__":
    main()
