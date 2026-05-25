"""
Layer-by-layer profile of contrast-attributed neurons across 4 cognitive conditions.

For each (model, condition):
  - Take top-k positive contrast neurons (atlas of "a-side"-specific units)
  - Compute the fraction in each layer (count / ffn_dim)
  - Compare across 4 models

The output answers: do norm_type contrast neurons cluster in similar layers
across architectures? Late-layer concentration would parallel the brain's
vmPFC / lateral PFC localization of moral judgment. Distributed profile would
indicate the function is computed broadly.

Local analysis — no GPU, fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "contrast_pilot"
META_DIR = ROOT / "activations"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
SHORT = {"Qwen2.5-7B-Instruct": "Qwen 7B",
         "Meta-Llama-3.1-8B-Instruct": "Llama 8B",
         "Mistral-7B-Instruct-v0.3": "Mistral 7B",
         "gemma-2-9b-it": "Gemma 9B"}
CONDITIONS = ["intent", "outcome", "norm_type", "morality"]
COND_COLORS = {"intent": "#1f77b4", "outcome": "#ff7f0e",
               "norm_type": "#d62728", "morality": "#2ca02c"}
N_ABLATE = 5000


def load_meta(model_short):
    with open(META_DIR / f"{model_short}_cognitive_pilot_meta.json") as f:
        m = json.load(f)
    return m["n_layers"], m["ffn_dim"]


def layer_profile(neuron_indices, n_layers, ffn_dim, normalize="fraction"):
    """Return per-layer count (or fraction) of selected neurons."""
    out = np.zeros(n_layers)
    for n in neuron_indices:
        out[n // ffn_dim] += 1
    if normalize == "fraction":
        out /= ffn_dim  # fraction of layer's FFN width that's in the selected set
    elif normalize == "share":
        out /= max(len(neuron_indices), 1)  # fraction of the atlas in each layer
    return out


def main():
    profiles = {}  # (model, cond) -> 1D array length n_layers
    layer_counts = {}

    for model in MODELS:
        n_layers, ffn_dim = load_meta(model)
        layer_counts[model] = (n_layers, ffn_dim)
        npz = np.load(RESULTS / f"{model}_contrast_attribution.npz")
        for cond in CONDITIONS:
            sel = npz[f"{cond}_selectivity"]
            top = np.argsort(-sel)[:N_ABLATE]
            profiles[(model, cond)] = layer_profile(
                top, n_layers, ffn_dim, normalize="share")

    # ---------- Plot 1: 4x4 panel grid (4 conditions x 4 models) ----------
    fig, axes = plt.subplots(len(CONDITIONS), len(MODELS),
                              figsize=(14, 9), sharex=False, sharey=False)
    for i, cond in enumerate(CONDITIONS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            prof = profiles[(model, cond)]
            n_layers = layer_counts[model][0]
            ax.bar(np.arange(n_layers) / (n_layers - 1), prof,
                   color=COND_COLORS[cond], width=1.0/n_layers,
                   alpha=0.85, edgecolor="black", linewidth=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(0.20, prof.max() * 1.15))
            if i == 0:
                ax.set_title(SHORT[model], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{cond}\nshare of atlas", fontsize=10)
            if i == len(CONDITIONS) - 1:
                ax.set_xlabel("layer depth (0=input, 1=output)", fontsize=9)
            ax.tick_params(axis="both", labelsize=8)
    plt.suptitle(
        "Layer-by-layer share of top-5000 contrast-selective neurons\n"
        "(positive contrast = selectively engaged for a-side of paired decomposition)",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_DIR / "contrast_layer_profile_grid.png", dpi=140)
    plt.close()

    # ---------- Plot 2: norm_type cross-model overlay ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in MODELS:
        prof = profiles[(model, "norm_type")]
        n_layers = layer_counts[model][0]
        depth = np.arange(n_layers) / (n_layers - 1)
        ax.plot(depth, prof, marker="o", markersize=5, linewidth=1.5,
                label=f"{SHORT[model]} ({n_layers} layers)")
    ax.set_xlabel("Normalized layer depth (0=input, 1=output)")
    ax.set_ylabel("Share of top-5000 norm_type contrast neurons")
    ax.set_title("Cross-architecture layer profile: moral-vs-conventional contrast neurons")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "norm_type_cross_model.png", dpi=140)
    plt.close()

    # ---------- Quantitative summary ----------
    print("=" * 80)
    print("LAYER PROFILE SUMMARY (top-5000 contrast neurons per condition)")
    print("=" * 80)
    for cond in CONDITIONS:
        print(f"\n  {cond}:")
        print(f"    {'Model':<12s} | {'layers':>7s} | {'peak L':>7s} | {'peak frac':>10s} | "
              f"{'early (0-1/3)':>14s} {'mid (1/3-2/3)':>14s} {'late (2/3-1)':>14s}")
        for model in MODELS:
            prof = profiles[(model, cond)]
            n_layers = layer_counts[model][0]
            peak = int(np.argmax(prof))
            third = n_layers // 3
            early = prof[:third].sum()
            mid = prof[third:2 * third].sum()
            late = prof[2 * third:].sum()
            print(f"    {SHORT[model]:<12s} | {n_layers:>7d} | {peak:>7d} | "
                  f"{prof[peak]:>10.3f} | {early:>14.2%} {mid:>14.2%} {late:>14.2%}")

    # Cross-architecture correlation of layer profiles for norm_type
    # (interpolate to common 100-bin depth axis for comparison)
    common_depth = np.linspace(0, 1, 100)
    interp_profiles = {}
    for model in MODELS:
        prof = profiles[(model, "norm_type")]
        n_layers = layer_counts[model][0]
        depth = np.arange(n_layers) / (n_layers - 1)
        interp_profiles[model] = np.interp(common_depth, depth, prof)
    print("\n  Cross-model Pearson correlation of norm_type layer profile:")
    print(f"    {'':>14s}" + "".join(f"{SHORT[m]:>12s}" for m in MODELS))
    for m1 in MODELS:
        row = f"    {SHORT[m1]:>14s}"
        for m2 in MODELS:
            r = float(np.corrcoef(interp_profiles[m1], interp_profiles[m2])[0, 1])
            row += f"{r:>12.3f}"
        print(row)

    # Mean correlation off-diagonal
    rs = []
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i < j:
                rs.append(np.corrcoef(interp_profiles[m1], interp_profiles[m2])[0, 1])
    print(f"\n  Mean off-diagonal correlation: {np.mean(rs):.3f}")

    # Sanity check: is the moral_anti (bottom-k) layer profile DIFFERENT from top?
    print(f"\n  Bottom-5000 vs top-5000 layer profile correlation (norm_type):")
    for model in MODELS:
        n_layers, ffn_dim = layer_counts[model]
        npz = np.load(RESULTS / f"{model}_contrast_attribution.npz")
        sel = npz["norm_type_selectivity"]
        top = np.argsort(-sel)[:N_ABLATE]
        bot = np.argsort(sel)[:N_ABLATE]
        top_prof = layer_profile(top, n_layers, ffn_dim, "share")
        bot_prof = layer_profile(bot, n_layers, ffn_dim, "share")
        r = float(np.corrcoef(top_prof, bot_prof)[0, 1])
        ov = len(set(top) & set(bot))
        print(f"    {SHORT[model]:>12s}: corr(top, bot)={r:>+.3f}, |top∩bot|={ov}")

    print(f"\n  Saved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
