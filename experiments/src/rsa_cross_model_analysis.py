"""
Cross-model synthesis of cognitive RSA results.

Reads per-model {model_short}_rsa.json files, builds three figures:

  Figure A: per-layer RSA alignment curves (4 models overlaid), with
            permutation null shading
  Figure B: peak-layer LLM RDM side-by-side with brain RDM, per model
  Figure C: cross-model consensus of preserved pairs:
            which (cond_i, cond_j) pairs land at low RDM in BOTH brain and
            ALL LLMs (the universal coupling structure)

Output:
  figures/cognitive_rsa_alignment.png
  figures/cognitive_rsa_rdm_grid.png
  figures/cognitive_rsa_preserved_pairs.png
  results/cognitive_rsa/cross_model_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


RES = Path("/hpc2hdd/home/mzhang630/data/nature/experiments/results/cognitive_rsa")
FIG = Path("/hpc2hdd/home/mzhang630/data/nature/experiments/figures")
FIG.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]
COLORS = {"Qwen2.5-7B-Instruct": "#1f77b4",
          "Meta-Llama-3.1-8B-Instruct": "#ff7f0e",
          "Mistral-7B-Instruct-v0.3": "#2ca02c",
          "gemma-2-9b-it": "#d62728"}


def load_brain():
    d = np.load(RES / "brain_rdm.npz", allow_pickle=True)
    return d["rdm"], list(d["conditions"])


def load_model(name):
    p = RES / f"{name}_rsa.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_llm_rdms(name):
    p = RES / f"{name}_rsa_llm_rdms.npz"
    if not p.exists():
        return None, None
    d = np.load(p, allow_pickle=True)
    return d["llm_rdms"], list(d["conditions"])


def figure_alignment(model_results):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    for name in MODELS:
        r = model_results.get(name)
        if r is None:
            continue
        layers = [pl["layer_index"] for pl in r["per_layer"]]
        # x-axis: relative layer (0..1) for cross-model comparison
        rel_x = np.linspace(0, 1, len(layers))
        rhos = [pl["rho"] for pl in r["per_layer"]]
        null_p95 = [pl["null_p95"] for pl in r["per_layer"]]
        null_mean = [pl["null_mean"] for pl in r["per_layer"]]
        ax.plot(rel_x, rhos, "-", color=COLORS[name], label=name, linewidth=2)
        ax.fill_between(rel_x, null_mean, null_p95, color=COLORS[name],
                        alpha=0.12, linewidth=0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Relative layer depth (0 = embedding, 1 = final)")
    ax.set_ylabel("Spearman ρ (brain RDM vs LLM RDM, off-diagonal)")
    ax.set_title("Cognitive-domain RSA alignment: LLM vs human brain\n"
                 "(shaded band = 5–95% permutation null)")
    ax.legend(fontsize=8, loc="best")
    out = FIG / "cognitive_rsa_alignment.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


def figure_rdm_grid(brain_rdm, conditions, model_results):
    n_models = sum(1 for n in MODELS if model_results.get(n) is not None)
    n_cols = 1 + n_models
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5),
                             constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    def show(ax, mat, title):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1.4)
        ax.set_xticks(range(len(conditions)))
        ax.set_yticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=80, fontsize=6)
        ax.set_yticklabels(conditions, fontsize=6)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)

    show(axes[0], brain_rdm, "Brain (Neurosynth + HCP)")
    col = 1
    for name in MODELS:
        r = model_results.get(name)
        if r is None:
            continue
        rdms, llm_conds = load_llm_rdms(name)
        if rdms is None:
            continue
        order = [llm_conds.index(c) for c in conditions]
        peak_idx = r["peak_layer"]["layer_index"]
        rdm = rdms[peak_idx][np.ix_(order, order)]
        show(axes[col],
             rdm,
             f"{name.split('-')[0]} peak L{peak_idx}\nρ={r['peak_layer']['rho']:.3f}")
        col += 1

    out = FIG / "cognitive_rsa_rdm_grid.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


def cross_model_consensus(brain_rdm, conditions, model_results):
    """For each pair, average LLM RDM at peak layer across models. Identify pairs
    that are SIMILAR (low) in both brain and consensus LLM, and pairs that are
    DISSIMILAR (high) in both. These define the preserved coupling structure.
    """
    n = len(conditions)
    consensus = np.zeros((n, n), dtype=np.float64)
    n_used = 0
    for name in MODELS:
        r = model_results.get(name)
        if r is None:
            continue
        rdms, llm_conds = load_llm_rdms(name)
        if rdms is None:
            continue
        order = [llm_conds.index(c) for c in conditions]
        peak_idx = r["peak_layer"]["layer_index"]
        rdm = rdms[peak_idx][np.ix_(order, order)]
        consensus += rdm
        n_used += 1
    if n_used == 0:
        return None
    consensus /= n_used

    # Spearman of brain vs consensus
    triu = np.triu_indices(n, k=1)
    rho_consensus, _ = spearmanr(brain_rdm[triu], consensus[triu])
    print(f"\nConsensus LLM RDM vs brain RDM: Spearman ρ = {rho_consensus:.4f} "
          f"(averaged across {n_used} models)")

    # Z-score both halves and find universal pairs
    pairs = []
    for i, j in zip(triu[0], triu[1]):
        pairs.append({
            "pair": [conditions[i], conditions[j]],
            "brain_rdm": float(brain_rdm[i, j]),
            "consensus_llm_rdm": float(consensus[i, j]),
        })

    # Preserved similar = low in both (after z-score within each side)
    brain_z = (brain_rdm - brain_rdm[triu].mean()) / brain_rdm[triu].std()
    cons_z = (consensus - consensus[triu].mean()) / consensus[triu].std()
    combo = brain_z + cons_z
    sorted_pairs = sorted(zip(triu[0], triu[1], combo[triu]), key=lambda x: x[2])
    preserved_similar = sorted_pairs[:8]
    preserved_dissimilar = sorted_pairs[-8:][::-1]

    print("\n[PRESERVED-SIMILAR universal pairs] (low z-RDM in brain AND consensus LLM):")
    for i, j, z in preserved_similar:
        print(f"  {conditions[i]:>16s} <-> {conditions[j]:<16s}  "
              f"combined z-score = {z:+.2f}")

    print("\n[PRESERVED-DISSIMILAR universal pairs] (high in both):")
    for i, j, z in preserved_dissimilar:
        print(f"  {conditions[i]:>16s} <-> {conditions[j]:<16s}  "
              f"combined z-score = {z:+.2f}")

    return {
        "rho_consensus_vs_brain": float(rho_consensus),
        "n_models_used": n_used,
        "preserved_similar_pairs": [
            {"pair": [conditions[i], conditions[j]], "combined_z": float(z)}
            for i, j, z in preserved_similar
        ],
        "preserved_dissimilar_pairs": [
            {"pair": [conditions[i], conditions[j]], "combined_z": float(z)}
            for i, j, z in preserved_dissimilar
        ],
        "consensus_rdm": consensus,
    }


def figure_preserved_pairs(brain_rdm, consensus_rdm, conditions, consensus_summary):
    """Scatter plot of each off-diagonal pair: x = brain RDM, y = consensus LLM RDM.
    Annotate the most extreme universal pairs.
    """
    triu = np.triu_indices(len(conditions), k=1)
    bx = brain_rdm[triu]
    cy = consensus_rdm[triu]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))
    ax.scatter(bx, cy, s=30, color="gray", alpha=0.5)

    # Annotate preserved pairs (similar + dissimilar)
    annot = set()
    for p in consensus_summary["preserved_similar_pairs"][:6]:
        annot.add(tuple(sorted(p["pair"])))
    for p in consensus_summary["preserved_dissimilar_pairs"][:6]:
        annot.add(tuple(sorted(p["pair"])))
    for i, j in zip(triu[0], triu[1]):
        pair = tuple(sorted([conditions[i], conditions[j]]))
        if pair in annot:
            ax.scatter([brain_rdm[i,j]], [consensus_rdm[i,j]], s=60,
                       color="crimson", zorder=5)
            label = f"{conditions[i][:6]}–{conditions[j][:6]}"
            ax.annotate(label, (brain_rdm[i,j], consensus_rdm[i,j]),
                        fontsize=7, alpha=0.85)

    rho = consensus_summary["rho_consensus_vs_brain"]
    ax.set_xlabel("Brain RDM (1 − Pearson over voxels)")
    ax.set_ylabel("Consensus LLM RDM (peak layer, 4-model avg)")
    ax.set_title(f"Pairwise cognitive-domain dissimilarity: brain vs LLM\n"
                 f"Spearman ρ = {rho:.3f}  ({consensus_summary['n_models_used']} models, "
                 f"~712 stimuli, 14 conditions)")
    # diagonal reference
    lo = min(bx.min(), cy.min())
    hi = max(bx.max(), cy.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    out = FIG / "cognitive_rsa_preserved_pairs.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


def main():
    brain_rdm, conditions = load_brain()
    model_results = {name: load_model(name) for name in MODELS}
    missing = [n for n, r in model_results.items() if r is None]
    if missing:
        print(f"WARN: missing model results: {missing}")
    have = [n for n in MODELS if model_results[n] is not None]
    print(f"Loaded {len(have)} model results: {have}")

    figure_alignment(model_results)
    figure_rdm_grid(brain_rdm, conditions, model_results)
    consensus = cross_model_consensus(brain_rdm, conditions, model_results)

    summary = {
        "models_present": have,
        "models_missing": missing,
        "per_model_peak": {
            n: r["peak_layer"] for n, r in model_results.items() if r is not None
        },
        "cross_model_consensus": {
            k: v for k, v in (consensus or {}).items() if k != "consensus_rdm"
        },
    }
    if consensus is not None:
        figure_preserved_pairs(brain_rdm, consensus["consensus_rdm"],
                                conditions, consensus)

    out = RES / "cross_model_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved cross-model summary to {out}")


if __name__ == "__main__":
    main()
