#!/usr/bin/env python3
"""
Deep geometry analysis: what do the LLM's internal representations of
social cognition and emotion actually look like, and why?

Key questions:
  1. What shape is the LLM geometry? (compression, separation ratios)
  2. What organizing dimensions explain each block?
  3. Where do brain and LLM agree/disagree, and why?
  4. What does the compression phenomenon mean mechanistically?

CPU-only, uses existing RDM files.

Output:
  results/mechanistic/geometry_deep_analysis.json
  figures/geometry_rdm_comparison.png
  figures/geometry_mds_and_scatter.png
  figures/geometry_compression.png
  figures/geometry_within_block.png
  figures/geometry_empathy_bridge.png
  figures/geometry_dendrograms.png
  figures/geometry_decompression.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
V2_DIR = BASE / "results" / "template_matched_rsa_v2"
FIG_DIR = BASE / "figures"
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CONDITIONS = [
    "anger", "belief", "disgust", "empathy", "fear", "happiness",
    "intention", "judgment", "mentalizing", "moral", "sadness",
    "self_referential", "theory_of_mind", "valence",
]
SHORT = {
    "anger": "ANG", "belief": "BEL", "disgust": "DIS", "empathy": "EMP",
    "fear": "FEA", "happiness": "HAP", "intention": "INT", "judgment": "JDG",
    "mentalizing": "MNT", "moral": "MOR", "sadness": "SAD",
    "self_referential": "SLF", "theory_of_mind": "ToM", "valence": "VAL",
}

AFF_SET = {"anger", "disgust", "fear", "happiness", "sadness", "valence"}
SOC_SET = {"belief", "empathy", "intention", "judgment", "mentalizing",
           "moral", "self_referential", "theory_of_mind"}
AFF5 = ["anger", "disgust", "fear", "happiness", "sadness"]
SOC7 = ["belief", "empathy", "intention", "judgment", "mentalizing",
        "self_referential", "theory_of_mind"]

C_AFF = "#E74C3C"
C_SOC = "#3498DB"
C_EMP = "#9B59B6"
C_BRAIN = "#2C3E50"

MODEL_NAMES = {
    "Qwen2.5-7B-Instruct": "Qwen-7B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-8B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "gemma-2-9b-it": "Gemma-9B",
}

tri_idx = np.triu_indices(14, k=1)

# VAD norms (Warriner 2013)
VAD = {
    "anger": (2.34, 6.18, 4.11), "fear": (2.64, 6.33, 2.77),
    "disgust": (2.42, 5.08, 4.03), "sadness": (2.10, 3.49, 2.82),
    "happiness": (8.21, 6.49, 6.63),
}

RELATIONAL = {
    "belief": (2, 1, 0.5, 0, 0), "intention": (1.5, 0, 0, 0, 0.3),
    "mentalizing": (2, 2, 0.3, 0, 0), "theory_of_mind": (2, 2, 0.7, 0, 0),
    "judgment": (1, 0, 0, 1, 0.3), "moral": (2, 0, 0, 1, 0),
    "empathy": (2, 1, 0, 0, 0.5), "self_referential": (1, 0, 0, 0, 1),
}


def block_color(c):
    if c == "empathy":
        return C_EMP
    return C_AFF if c in AFF_SET else C_SOC


def load_all():
    brain = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)["rdm"]
    models = {}
    for full, short in MODEL_NAMES.items():
        f = RSA_DIR / f"{full}_rdm14_headline.npz"
        if f.exists():
            models[short] = np.load(f)["rdm"]
    v2 = {}
    for full, short in MODEL_NAMES.items():
        f = V2_DIR / f"{full}_tm_v2_rdm14.npz"
        if f.exists():
            v2[short] = np.load(f)["rdm"]
    return brain, models, v2


def pair_types():
    types = []
    for i, j in zip(tri_idx[0], tri_idx[1]):
        c1, c2 = CONDITIONS[i], CONDITIONS[j]
        if c1 in AFF_SET and c2 in AFF_SET:
            types.append("within-aff")
        elif c1 in SOC_SET and c2 in SOC_SET:
            types.append("within-soc")
        else:
            types.append("cross")
    return np.array(types)


def do_mds(rdm, n=2):
    mds = MDS(n_components=n, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto", n_init=10)
    return mds.fit_transform(rdm)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: RDM heatmaps — brain vs mean-LLM vs difference
# ═══════════════════════════════════════════════════════════════════════

def fig1(brain, models, results):
    mean_llm = np.mean(list(models.values()), axis=0)
    labels = [SHORT[c] for c in CONDITIONS]
    colors = [block_color(c) for c in CONDITIONS]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax_i, (rdm, title) in enumerate([
        (brain, "Brain (Neurosynth)"),
        (mean_llm, "LLM (mean of 4 models)"),
    ]):
        ax = axes[ax_i]
        im = ax.imshow(rdm, cmap="YlOrRd", vmin=0, vmax=max(brain.max(), mean_llm.max()))
        ax.set_xticks(range(14)); ax.set_yticks(range(14))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        for idx in range(14):
            ax.get_xticklabels()[idx].set_color(colors[idx])
            ax.get_yticklabels()[idx].set_color(colors[idx])
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8, label="1−cosine")
        for pos in [5.5]:
            ax.axhline(pos, color="white", lw=1.5, ls="--", alpha=0.8)
            ax.axvline(pos, color="white", lw=1.5, ls="--", alpha=0.8)

    # Rank difference
    ax = axes[2]
    bt = brain[tri_idx]; lt = mean_llm[tri_idx]
    br = np.argsort(np.argsort(bt)).astype(float) / (len(bt)-1)
    lr = np.argsort(np.argsort(lt)).astype(float) / (len(lt)-1)
    diff = np.zeros((14,14))
    diff[tri_idx] = lr - br; diff += diff.T
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(14)); ax.set_yticks(range(14))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for idx in range(14):
        ax.get_xticklabels()[idx].set_color(colors[idx])
        ax.get_yticklabels()[idx].set_color(colors[idx])
    ax.set_title("Rank difference (LLM − Brain)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Rank Δ")
    for pos in [5.5]:
        ax.axhline(pos, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.axvline(pos, color="black", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_rdm_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ geometry_rdm_comparison.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: MDS + pair scatter
# ═══════════════════════════════════════════════════════════════════════

def fig2(brain, models, results):
    mean_llm = np.mean(list(models.values()), axis=0)
    brain_xy = do_mds(brain)
    llm_xy = do_mds(mean_llm)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    def plot_mds(ax, coords, title):
        for i, c in enumerate(CONDITIONS):
            col = block_color(c)
            m = "o" if c in AFF_SET else ("D" if c == "empathy" else "s")
            ax.scatter(coords[i, 0], coords[i, 1], c=col, s=100, marker=m,
                       edgecolors="white", linewidths=0.7, zorder=3)
            ax.annotate(SHORT[c], (coords[i, 0], coords[i, 1]),
                        fontsize=7.5, fontweight="bold", color=col,
                        ha="center", va="bottom", xytext=(0, 7),
                        textcoords="offset points")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("MDS dim 1"); ax.set_ylabel("MDS dim 2")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.axhline(0, color="#ddd", lw=0.5); ax.axvline(0, color="#ddd", lw=0.5)

    plot_mds(axes[0], brain_xy, "Brain")
    plot_mds(axes[1], llm_xy, "LLM (mean of 4)")

    # Scatter
    ax = axes[2]
    pt = pair_types()
    bv = brain[tri_idx]; lv = mean_llm[tri_idx]
    for t, c, lab in [("within-aff", C_AFF, "Within-Affective"),
                       ("within-soc", C_SOC, "Within-Social"),
                       ("cross", "#95A5A6", "Cross-Block")]:
        mask = pt == t
        ax.scatter(bv[mask], lv[mask], c=c, s=40, alpha=0.7, label=lab,
                   edgecolors="white", linewidths=0.4)
    rho, _ = spearmanr(bv, lv)
    ax.set_xlabel("Brain distance"); ax.set_ylabel("LLM distance")
    ax.set_title("91 pairs: brain vs LLM distance", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.97, 0.03, f"ρ = {rho:.3f}", transform=ax.transAxes,
            fontsize=13, ha="right", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    results["scatter"] = {
        "full_rho": float(rho),
        "within_aff_n": int((pt == "within-aff").sum()),
        "within_soc_n": int((pt == "within-soc").sum()),
        "cross_n": int((pt == "cross").sum()),
    }

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_mds_and_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ geometry_mds_and_scatter.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Compression — distance distributions + separation ratios
# ═══════════════════════════════════════════════════════════════════════

def fig3(brain, models, results):
    pt = pair_types()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax_i, (rdm, title) in enumerate([
        (brain, "Brain"), (np.mean(list(models.values()), axis=0), "LLM (mean)")
    ]):
        ax = axes[ax_i]
        vals = rdm[tri_idx]
        for t, c, lab in [("within-aff", C_AFF, "Within-Aff"),
                           ("within-soc", C_SOC, "Within-Soc"),
                           ("cross", "#7F8C8D", "Cross")]:
            mask = pt == t
            ax.hist(vals[mask], bins=12, alpha=0.55, color=c, label=lab, density=True)
        ax.set_xlabel("Distance"); ax.set_ylabel("Density")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Separation ratios
    ax = axes[2]
    systems = ["Brain"]
    aff_r, soc_r = [], []
    bv = brain[tri_idx]
    aff_r.append(bv[pt=="cross"].mean() / max(bv[pt=="within-aff"].mean(), 1e-6))
    soc_r.append(bv[pt=="cross"].mean() / max(bv[pt=="within-soc"].mean(), 1e-6))
    for short, rdm in models.items():
        systems.append(short)
        v = rdm[tri_idx]
        aff_r.append(v[pt=="cross"].mean() / max(v[pt=="within-aff"].mean(), 1e-6))
        soc_r.append(v[pt=="cross"].mean() / max(v[pt=="within-soc"].mean(), 1e-6))

    x = np.arange(len(systems))
    w = 0.32
    b1 = ax.bar(x - w/2, aff_r, w, color=C_AFF, alpha=0.8, label="Cross / Within-Aff")
    b2 = ax.bar(x + w/2, soc_r, w, color=C_SOC, alpha=0.8, label="Cross / Within-Soc")
    for bar_set in [b1, b2]:
        for bar in bar_set:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}×" if h < 10 else f"{h:.0f}×",
                    ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("Separation ratio"); ax.set_title("Block separation", fontweight="bold")
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    results["separation_ratios"] = {
        s: {"aff": float(a), "soc": float(sc)} for s, a, sc in zip(systems, aff_r, soc_r)
    }

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_compression.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ geometry_compression.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Within-block fine structure (pair-by-pair rank comparison)
# ═══════════════════════════════════════════════════════════════════════

def fig4(brain, models, results):
    mean_llm = np.mean(list(models.values()), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    def plot_block(ax, conds, title, color):
        idx = [CONDITIONS.index(c) for c in conds]
        pairs, bd, ld = [], [], []
        for i in range(len(conds)):
            for j in range(i+1, len(conds)):
                pairs.append(f"{SHORT[conds[i]]}–{SHORT[conds[j]]}")
                bd.append(brain[idx[i], idx[j]])
                ld.append(mean_llm[idx[i], idx[j]])
        order = np.argsort(bd)
        pairs = [pairs[k] for k in order]
        bd = np.array([bd[k] for k in order])
        ld = np.array([ld[k] for k in order])
        # Normalize to [0,1]
        bn = (bd - bd.min()) / (bd.max() - bd.min() + 1e-10)
        ln = (ld - ld.min()) / (ld.max() - ld.min() + 1e-10)
        x = np.arange(len(pairs))
        ax.plot(x, bn, "o-", color=C_BRAIN, lw=2, ms=5, label="Brain", zorder=3)
        ax.plot(x, ln, "s--", color=color, lw=2, ms=5, label="LLM", alpha=0.85, zorder=3)
        ax.fill_between(x, bn, ln, alpha=0.1, color=color)
        ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("Normalized distance")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        rho, p = spearmanr(bd, ld)
        ax.text(0.97, 0.05, f"ρ = {rho:+.3f}\np = {p:.3f}", transform=ax.transAxes,
                fontsize=11, ha="right", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        return {"rho": float(rho), "p": float(p), "n_pairs": len(pairs)}

    r_aff = plot_block(axes[0], AFF5, f"Within-Affective (5 emotions, 10 pairs)", C_AFF)
    r_soc = plot_block(axes[1], SOC7, f"Within-Social (7 conditions, 21 pairs)", C_SOC)
    results["within_aff_headline"] = r_aff
    results["within_soc_headline"] = r_soc

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_within_block.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ geometry_within_block.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Empathy as bridge
# ═══════════════════════════════════════════════════════════════════════

def fig5(brain, models, results):
    mean_llm = np.mean(list(models.values()), axis=0)
    emp = CONDITIONS.index("empathy")
    others = [c for c in CONDITIONS if c != "empathy"]
    oi = [CONDITIONS.index(c) for c in others]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, (rdm, title) in enumerate([(brain, "Brain"), (mean_llm, "LLM (mean)")]):
        ax = axes[ax_i]
        dists = [rdm[emp, i] for i in oi]
        order = np.argsort(dists)
        sc = [others[k] for k in order]
        sd = [dists[k] for k in order]
        cols = [block_color(c) for c in sc]
        bars = ax.barh(range(len(sc)), sd, color=cols, alpha=0.8, edgecolor="white", lw=0.5)
        ax.set_yticks(range(len(sc)))
        ax.set_yticklabels([SHORT[c] for c in sc], fontsize=8)
        for tick, col in zip(ax.get_yticklabels(), cols):
            tick.set_color(col); tick.set_fontweight("bold")
        ax.set_xlabel("Distance to Empathy")
        ax.set_title(f"{title}: Empathy's neighborhood", fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    # empathy position summary
    brain_to_aff = np.mean([brain[emp, CONDITIONS.index(c)] for c in AFF5])
    brain_to_soc = np.mean([brain[emp, CONDITIONS.index(c)] for c in SOC7 if c != "empathy"])
    llm_to_aff = np.mean([mean_llm[emp, CONDITIONS.index(c)] for c in AFF5])
    llm_to_soc = np.mean([mean_llm[emp, CONDITIONS.index(c)] for c in SOC7 if c != "empathy"])
    results["empathy_position"] = {
        "brain_to_aff": float(brain_to_aff), "brain_to_soc": float(brain_to_soc),
        "brain_closer_to": "social" if brain_to_soc < brain_to_aff else "affective",
        "llm_to_aff": float(llm_to_aff), "llm_to_soc": float(llm_to_soc),
        "llm_closer_to": "social" if llm_to_soc < llm_to_aff else "affective",
    }

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_empathy_bridge.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ geometry_empathy_bridge.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6: Decompression (headline vs V2)
# ═══════════════════════════════════════════════════════════════════════

def fig6(brain, models, v2, results):
    if not v2:
        print("  ⏭ Skipping decompression (no V2)")
        return

    aff_idx = [CONDITIONS.index(c) for c in AFF5]
    soc_idx = [CONDITIONS.index(c) for c in SOC7]

    def block_dists(rdm, idx_list):
        d = []
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                d.append(rdm[idx_list[i], idx_list[j]])
        return np.array(d)

    brain_aff = block_dists(brain, aff_idx)
    brain_soc = block_dists(brain, soc_idx)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ms = list(models.keys())
    x = np.arange(len(ms))
    w = 0.3

    # Panel 1: within-aff mean distance
    ax = axes[0]
    h_means = [block_dists(models[m], aff_idx).mean() for m in ms]
    v2_means = [block_dists(v2[m], aff_idx).mean() for m in ms]
    ax.bar(x - w/2, h_means, w, color=C_AFF, alpha=0.4, label="Headline (short)")
    ax.bar(x + w/2, v2_means, w, color=C_AFF, alpha=0.9, label="V2 (rich)")
    ax.axhline(brain_aff.mean(), color=C_BRAIN, ls="--", lw=2, label=f"Brain ({brain_aff.mean():.3f})")
    ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=8)
    ax.set_ylabel("Mean within-aff distance"); ax.set_title("Affective distances\nexpand under rich input", fontweight="bold")
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: within-aff rho
    ax = axes[1]
    h_rho = [spearmanr(brain_aff, block_dists(models[m], aff_idx))[0] for m in ms]
    v2_rho = [spearmanr(brain_aff, block_dists(v2[m], aff_idx))[0] for m in ms]
    ax.bar(x - w/2, h_rho, w, color=C_AFF, alpha=0.4, label="Headline")
    ax.bar(x + w/2, v2_rho, w, color=C_AFF, alpha=0.9, label="V2 rich")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=8)
    ax.set_ylabel("ρ vs brain"); ax.set_title("Affective fine structure\naligns when decompressed", fontweight="bold")
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 3: within-soc rho
    ax = axes[2]
    h_rho_s = [spearmanr(brain_soc, block_dists(models[m], soc_idx))[0] for m in ms]
    v2_rho_s = [spearmanr(brain_soc, block_dists(v2[m], soc_idx))[0] for m in ms]
    ax.bar(x - w/2, h_rho_s, w, color=C_SOC, alpha=0.4, label="Headline")
    ax.bar(x + w/2, v2_rho_s, w, color=C_SOC, alpha=0.9, label="V2 rich")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=8)
    ax.set_ylabel("ρ vs brain"); ax.set_title("Social fine structure:\nstable across conditions", fontweight="bold")
    ax.legend(fontsize=7); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    results["decompression"] = {
        m: {
            "headline_aff_mean_dist": float(block_dists(models[m], aff_idx).mean()),
            "v2_aff_mean_dist": float(block_dists(v2[m], aff_idx).mean()),
            "headline_aff_rho": float(spearmanr(brain_aff, block_dists(models[m], aff_idx))[0]),
            "v2_aff_rho": float(spearmanr(brain_aff, block_dists(v2[m], aff_idx))[0]),
            "headline_soc_rho": float(spearmanr(brain_soc, block_dists(models[m], soc_idx))[0]),
            "v2_soc_rho": float(spearmanr(brain_soc, block_dists(v2[m], soc_idx))[0]),
        } for m in ms
    }

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_decompression.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ geometry_decompression.png")


# ═══════════════════════════════════════════════════════════════════════
# Figure 7: Dendrograms
# ═══════════════════════════════════════════════════════════════════════

def fig7(brain, models, results):
    mean_llm = np.mean(list(models.values()), axis=0)
    labels = [SHORT[c] for c in CONDITIONS]
    cmap = {SHORT[c]: block_color(c) for c in CONDITIONS}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_i, (rdm, title) in enumerate([(brain, "Brain"), (mean_llm, "LLM (mean of 4)")]):
        ax = axes[ax_i]
        rdm_clean = rdm.copy(); np.fill_diagonal(rdm_clean, 0)
        Z = linkage(squareform(rdm_clean), method="average")
        dn = dendrogram(Z, labels=labels, ax=ax, leaf_rotation=45,
                        leaf_font_size=9, above_threshold_color="#95A5A6")
        ax.set_title(title, fontweight="bold"); ax.set_ylabel("Distance")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for lbl in ax.get_xticklabels():
            if lbl.get_text() in cmap:
                lbl.set_color(cmap[lbl.get_text()]); lbl.set_fontweight("bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_dendrograms.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ geometry_dendrograms.png")


# ═══════════════════════════════════════════════════════════════════════
# Deep analysis: what dimensions organize each block?
# ═══════════════════════════════════════════════════════════════════════

def deep_analysis(brain, models, results):
    mean_llm = np.mean(list(models.values()), axis=0)
    print("\n" + "="*70)
    print("DEEP ANALYSIS: organizing dimensions")
    print("="*70)

    # Emotion block: VAD correlation
    aff_idx = [CONDITIONS.index(c) for c in AFF5]
    brain_aff_dists = []
    llm_aff_dists = []
    for i in range(5):
        for j in range(i+1, 5):
            brain_aff_dists.append(brain[aff_idx[i], aff_idx[j]])
            llm_aff_dists.append(mean_llm[aff_idx[i], aff_idx[j]])

    # Build VAD distance matrix for 5 emotions
    vad_mat = np.array([VAD[c] for c in AFF5])
    vad_dists = []
    for i in range(5):
        for j in range(i+1, 5):
            vad_dists.append(np.linalg.norm(vad_mat[i] - vad_mat[j]))

    rho_vad_brain, p_vad_brain = spearmanr(vad_dists, brain_aff_dists)
    rho_vad_llm, p_vad_llm = spearmanr(vad_dists, llm_aff_dists)
    print(f"\n  VAD distance vs brain within-aff: ρ={rho_vad_brain:+.3f} (p={p_vad_brain:.3f})")
    print(f"  VAD distance vs LLM within-aff:   ρ={rho_vad_llm:+.3f} (p={p_vad_llm:.3f})")

    # Valence-only
    val_dists = [abs(VAD[AFF5[i]][0] - VAD[AFF5[j]][0]) for i in range(5) for j in range(i+1,5)]
    rho_val_brain, _ = spearmanr(val_dists, brain_aff_dists)
    rho_val_llm, _ = spearmanr(val_dists, llm_aff_dists)
    # Arousal-only
    aro_dists = [abs(VAD[AFF5[i]][1] - VAD[AFF5[j]][1]) for i in range(5) for j in range(i+1,5)]
    rho_aro_brain, _ = spearmanr(aro_dists, brain_aff_dists)
    rho_aro_llm, _ = spearmanr(aro_dists, llm_aff_dists)
    print(f"  Valence-only: brain ρ={rho_val_brain:+.3f}, LLM ρ={rho_val_llm:+.3f}")
    print(f"  Arousal-only: brain ρ={rho_aro_brain:+.3f}, LLM ρ={rho_aro_llm:+.3f}")

    results["emotion_organizing_dimensions"] = {
        "vad_full": {"brain_rho": float(rho_vad_brain), "llm_rho": float(rho_vad_llm)},
        "valence_only": {"brain_rho": float(rho_val_brain), "llm_rho": float(rho_val_llm)},
        "arousal_only": {"brain_rho": float(rho_aro_brain), "llm_rho": float(rho_aro_llm)},
    }

    # Social block: relational features
    soc_idx = [CONDITIONS.index(c) for c in SOC7]
    brain_soc_dists = []
    llm_soc_dists = []
    for i in range(7):
        for j in range(i+1, 7):
            brain_soc_dists.append(brain[soc_idx[i], soc_idx[j]])
            llm_soc_dists.append(mean_llm[soc_idx[i], soc_idx[j]])

    rel_mat = np.array([RELATIONAL[c] for c in SOC7])
    rel_mat_s = StandardScaler().fit_transform(rel_mat)
    rel_dists = []
    for i in range(7):
        for j in range(i+1, 7):
            rel_dists.append(np.linalg.norm(rel_mat_s[i] - rel_mat_s[j]))

    rho_rel_brain, p_rel_brain = spearmanr(rel_dists, brain_soc_dists)
    rho_rel_llm, p_rel_llm = spearmanr(rel_dists, llm_soc_dists)
    print(f"\n  Relational features vs brain within-soc: ρ={rho_rel_brain:+.3f} (p={p_rel_brain:.3f})")
    print(f"  Relational features vs LLM within-soc:   ρ={rho_rel_llm:+.3f} (p={p_rel_llm:.3f})")

    # Per-feature
    feat_names = ["n_agents", "recursion_depth", "false_belief", "normative", "self_directed"]
    print(f"\n  Per-feature correlation with within-social distances:")
    feat_results = {}
    for fi, fn in enumerate(feat_names):
        feat_dists = [abs(rel_mat[i, fi] - rel_mat[j, fi]) for i in range(7) for j in range(i+1, 7)]
        rb, _ = spearmanr(feat_dists, brain_soc_dists)
        rl, _ = spearmanr(feat_dists, llm_soc_dists)
        print(f"    {fn:>20s}: brain ρ={rb:+.3f}  LLM ρ={rl:+.3f}")
        feat_results[fn] = {"brain_rho": float(rb), "llm_rho": float(rl)}

    results["social_organizing_dimensions"] = {
        "relational_full": {"brain_rho": float(rho_rel_brain), "llm_rho": float(rho_rel_llm)},
        "per_feature": feat_results,
    }

    # Key insight: what separates empathy?
    print(f"\n  Empathy analysis:")
    emp_idx = CONDITIONS.index("empathy")
    for c in ["mentalizing", "theory_of_mind", "moral", "anger", "sadness"]:
        ci = CONDITIONS.index(c)
        print(f"    empathy-{c}: brain={brain[emp_idx,ci]:.4f}  LLM={mean_llm[emp_idx,ci]:.4f}")


# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading RDMs...")
    brain, models, v2 = load_all()
    print(f"  Brain: {brain.shape}")
    print(f"  Models: {list(models.keys())}")
    print(f"  V2: {list(v2.keys())}")

    results = {}

    print("\nGenerating figures...")
    fig1(brain, models, results)
    fig2(brain, models, results)
    fig3(brain, models, results)
    fig4(brain, models, results)
    fig5(brain, models, results)
    fig6(brain, models, v2, results)
    fig7(brain, models, results)

    deep_analysis(brain, models, results)

    out_path = OUT_DIR / "geometry_deep_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis: {out_path}")
    print(f"All figures: {FIG_DIR}/geometry_*.png")


if __name__ == "__main__":
    main()
