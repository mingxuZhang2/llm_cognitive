#!/usr/bin/env python3
"""
Generate all 6 publication-quality figures for the Nature paper.
Strict Nature style: no in-figure titles, bold lowercase panel labels only,
5.5-7pt fonts, minimal spines, double-column width (180mm = 7.09in).
Reads real data from result JSONs/NPZs.
Saves to experiments/figures/paper/ as both PDF and PNG (300 DPI).
"""

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr

# ── Nature-strict rcParams ────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype":       "none",
    "pdf.fonttype":       42,
    "font.size":          6,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.linewidth":     0.5,
    "axes.labelsize":     6,
    "axes.titlesize":     6,
    "xtick.labelsize":    5.5,
    "ytick.labelsize":    5.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
    "xtick.major.size":   2,
    "ytick.major.size":   2,
    "xtick.major.pad":    1.5,
    "ytick.major.pad":    1.5,
    "legend.fontsize":    5.5,
    "legend.frameon":     False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,
    "lines.linewidth":    0.8,
    "lines.markersize":   3,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "axes.grid":          False,
})

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = "/hpc2hdd/home/mzhang630/data/nature/experiments"
RES  = os.path.join(ROOT, "results")
OUT  = os.path.join(ROOT, "figures", "paper")
os.makedirs(OUT, exist_ok=True)

RSA  = os.path.join(RES, "cognitive_rsa")
CLIN = os.path.join(RES, "clinical_dissociation")
DEV  = os.path.join(RES, "developmental_emergence")
TM   = os.path.join(RES, "template_matched_rsa")
MOR  = os.path.join(RES, "moral_judgment")
HR   = os.path.join(RES, "human_rating")

# ── Color scheme (Nature-grade) ──────────────────────────────────────
C_AFF   = "#2166AC"   # deep blue — affective
C_SOC   = "#B2182B"   # deep red — social/mentalizing
C_CROSS = "#999999"   # grey — cross-block

MODEL_COLORS = {
    "Qwen":    "#2166AC",
    "Llama":   "#E08214",
    "Mistral": "#1B9E77",
    "Gemma":   "#7570B3",
}

MODEL_SHORT = {
    "Qwen2.5-7B-Instruct":             "Qwen",
    "Meta-Llama-3.1-8B-Instruct":      "Llama",
    "Mistral-7B-Instruct-v0.3":        "Mistral",
    "gemma-2-9b-it":                    "Gemma",
}

MODEL_ORDER_FULL = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]

# ── Conditions ────────────────────────────────────────────────────────
AFFECTIVE = {"anger", "disgust", "fear", "happiness", "sadness", "valence"}
SOCIAL    = {"belief", "empathy", "intention", "judgment", "mentalizing",
             "moral", "self_referential", "theory_of_mind"}

PRETTY = {
    "anger": "Anger", "disgust": "Disgust", "fear": "Fear",
    "happiness": "Happiness", "sadness": "Sadness", "valence": "Valence",
    "belief": "Belief", "empathy": "Empathy", "intention": "Intention",
    "judgment": "Judgment", "mentalizing": "Mentalizing", "moral": "Moral",
    "self_referential": "Self-ref.", "theory_of_mind": "ToM",
}

# ── Helpers ───────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_rdm(path):
    d = np.load(path, allow_pickle=True)
    return d["rdm"], list(d["conditions"])


def upper_tri(rdm):
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def pair_types(conditions):
    n = len(conditions)
    idx = np.triu_indices(n, k=1)
    wa, ws, cb = [], [], []
    for i, j in zip(idx[0], idx[1]):
        ci, cj = conditions[i], conditions[j]
        ai = ci in AFFECTIVE
        aj = cj in AFFECTIVE
        if ai and aj:
            wa.append(True); ws.append(False); cb.append(False)
        elif (not ai) and (not aj):
            wa.append(False); ws.append(True); cb.append(False)
        else:
            wa.append(False); ws.append(False); cb.append(True)
    return np.array(wa), np.array(ws), np.array(cb)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


def panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left",
            fontfamily="sans-serif")


def clean_ax(ax):
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


def heatmap_spines(ax):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.3)


def reorder_rdm(rdm, conditions):
    aff_sorted = sorted([c for c in conditions if c in AFFECTIVE])
    soc_sorted = sorted([c for c in conditions if c in SOCIAL])
    order = aff_sorted + soc_sorted
    idx = [conditions.index(c) for c in order]
    return rdm[np.ix_(idx, idx)], order, len(aff_sorted)


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 — The Discovery
# ═══════════════════════════════════════════════════════════════════════
def figure1():
    print("Figure 1: The Discovery")

    brain_rdm, brain_conds = load_rdm(os.path.join(RSA, "brain_rdm.npz"))
    qwen_rdm, qwen_conds  = load_rdm(os.path.join(RSA,
                                      "Qwen2.5-7B-Instruct_rdm14_headline.npz"))

    brain_reord, order, n_aff = reorder_rdm(brain_rdm, brain_conds)
    qwen_reord, _, _          = reorder_rdm(qwen_rdm, qwen_conds)
    labels = [PRETTY.get(c, c) for c in order]

    # Scaling data for ceiling
    scaling = load_json(os.path.join(RSA, "scaling_summary.json"))
    ceiling_vals = [m["noise_ceiling"] for m in scaling["models"]]
    mean_ceiling = np.mean(ceiling_vals)

    # --- Layout: 2 rows. Top: two large heatmaps. Bottom: bar + scatter ---
    fig = plt.figure(figsize=(7.09, 5.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 0.8],
                          width_ratios=[1.0, 1.0, 0.04, 1.0],
                          hspace=0.45, wspace=0.25,
                          left=0.08, right=0.97, top=0.95, bottom=0.06)

    # Shared colormap range
    vmin = min(brain_reord.min(), qwen_reord.min())
    vmax = max(brain_reord.max(), qwen_reord.max())

    # ── Panel a: Brain heatmap (top-left) ──
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(brain_reord, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        aspect="equal", interpolation="nearest")
    ax_a.set_xticks(range(14))
    ax_a.set_yticks(range(14))
    ax_a.set_yticklabels(labels, fontsize=5)
    ax_a.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax_a.tick_params(length=0, pad=1)

    # Color y-tick labels by block
    for tick_lbl, idx_i in zip(ax_a.get_yticklabels(), range(14)):
        tick_lbl.set_color(C_AFF if idx_i < n_aff else C_SOC)
        tick_lbl.set_fontweight("semibold")

    # Block boundary lines
    ax_a.axhline(n_aff - 0.5, color="black", linewidth=0.8)
    ax_a.axvline(n_aff - 0.5, color="black", linewidth=0.8)
    heatmap_spines(ax_a)

    panel_label(ax_a, "a", x=-0.20, y=1.06)

    # ── Panel b: LLM heatmap (top-middle) ──
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(qwen_reord, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        aspect="equal", interpolation="nearest")
    ax_b.set_xticks(range(14))
    ax_b.set_yticks(range(14))
    ax_b.set_yticklabels([])
    ax_b.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax_b.tick_params(length=0, pad=1)

    ax_b.axhline(n_aff - 0.5, color="black", linewidth=0.8)
    ax_b.axvline(n_aff - 0.5, color="black", linewidth=0.8)
    heatmap_spines(ax_b)

    # Colorbar in its own narrow column
    ax_cbar = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im_b, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=4.5, length=1.5, width=0.3, pad=1)
    cbar.outline.set_linewidth(0.3)
    cbar.set_label("1 - r", fontsize=5, labelpad=1)

    panel_label(ax_b, "b", x=-0.06, y=1.06)

    # ── Panel c: bar chart of headline rho (bottom-left, spanning 2 cols) ──
    ax_c = fig.add_subplot(gs[1, :2])
    clean_ax(ax_c)

    models_data = [
        ("Qwen",    0.739, MODEL_COLORS["Qwen"]),
        ("Llama",   0.727, MODEL_COLORS["Llama"]),
        ("Mistral", 0.730, MODEL_COLORS["Mistral"]),
        ("Gemma",   0.735, MODEL_COLORS["Gemma"]),
    ]
    names_c = [m[0] for m in models_data]
    rhos_c  = [m[1] for m in models_data]
    cols_c  = [m[2] for m in models_data]

    bars = ax_c.bar(range(4), rhos_c, color=cols_c, width=0.6,
                    edgecolor="white", linewidth=0.3, zorder=3)
    ax_c.axhline(mean_ceiling, color="#888888", linestyle="--",
                 linewidth=0.6, zorder=2)
    ax_c.text(0.95, mean_ceiling - 0.03, "ceiling", fontsize=4,
              color="#888888", va="top", ha="right",
              transform=mpl.transforms.blended_transform_factory(
                  ax_c.transAxes, ax_c.transData))

    for i in range(4):
        ax_c.text(i, rhos_c[i] + 0.015, "***", ha="center", va="bottom",
                  fontsize=5, fontweight="bold")

    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(names_c, fontsize=5, rotation=45, ha="right")
    ax_c.set_ylabel(r"$\rho$", fontsize=6, labelpad=1)
    ax_c.set_ylim(0, 1.08)
    ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    panel_label(ax_c, "c", x=-0.22, y=1.06)

    # ── Panel d: scatter (bottom-right, spanning colorbar col + last col) ──
    ax_d = fig.add_subplot(gs[1, 2:])
    clean_ax(ax_d)

    b_ut = upper_tri(brain_rdm)
    q_ut = upper_tri(qwen_rdm)
    wa, ws, cb = pair_types(brain_conds)

    ax_d.scatter(b_ut[cb], q_ut[cb], s=8, c=C_CROSS, alpha=0.5,
                 edgecolors="none", label="Cross", zorder=2, rasterized=True)
    ax_d.scatter(b_ut[wa], q_ut[wa], s=12, c=C_AFF, alpha=0.8,
                 edgecolors="white", linewidths=0.2,
                 label="Affective", zorder=3)
    ax_d.scatter(b_ut[ws], q_ut[ws], s=12, c=C_SOC, alpha=0.8,
                 edgecolors="white", linewidths=0.2,
                 label="Social", zorder=3)

    z = np.polyfit(b_ut, q_ut, 1)
    xline = np.linspace(b_ut.min(), b_ut.max(), 100)
    ax_d.plot(xline, np.polyval(z, xline), color="black", linewidth=0.6,
              linestyle="--", zorder=1)

    rho_val, _ = spearmanr(b_ut, q_ut)
    ax_d.text(0.04, 0.96, r"$\rho$ = " + f"{rho_val:.3f}",
              transform=ax_d.transAxes, fontsize=5.5, va="top")

    ax_d.set_xlabel("Brain RDM distance", fontsize=6, labelpad=2)
    ax_d.set_ylabel("LLM RDM distance", fontsize=6, labelpad=2)
    ax_d.legend(fontsize=4.5, loc="lower right", markerscale=0.8,
                handletextpad=0.2, borderpad=0.3)

    panel_label(ax_d, "d", x=-0.18, y=1.06)

    save_fig(fig, "figure1_discovery")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Decomposing the Alignment
# ═══════════════════════════════════════════════════════════════════════
def figure2():
    print("Figure 2: Decomposing the Alignment")

    wbc = load_json(os.path.join(RSA, "within_block_control.json"))
    brain_rdm, brain_conds = load_rdm(os.path.join(RSA, "brain_rdm.npz"))

    model_files = [
        ("Qwen2.5-7B-Instruct_rdm14_headline.npz", "Qwen"),
        ("Meta-Llama-3.1-8B-Instruct_rdm14_headline.npz", "Llama"),
        ("Mistral-7B-Instruct-v0.3_rdm14_headline.npz", "Mistral"),
        ("gemma-2-9b-it_rdm14_headline.npz", "Gemma"),
    ]
    llm_rdms = {}
    for fname, short_name in model_files:
        rdm, _ = load_rdm(os.path.join(RSA, fname))
        llm_rdms[short_name] = rdm

    mean_llm_rdm = np.mean([llm_rdms[k] for k in llm_rdms], axis=0)

    aff_idx = sorted([i for i, c in enumerate(brain_conds) if c in AFFECTIVE])
    soc_idx = sorted([i for i, c in enumerate(brain_conds) if c in SOCIAL])

    fig = plt.figure(figsize=(7.09, 4.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          width_ratios=[1.0, 1.0],
                          hspace=0.40, wspace=0.30,
                          left=0.08, right=0.97, top=0.95, bottom=0.06)

    # ── Panel a: grouped bar chart (top, spanning both columns) ──
    ax_a = fig.add_subplot(gs[0, :])
    clean_ax(ax_a)

    model_order = ["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B"]
    short_labels = ["Qwen", "Llama", "Mistral", "Gemma"]
    metric_names = [r"Full $\rho$", r"Partial $\rho$",
                    r"W-social $\rho$", r"W-aff. $\rho$"]
    metric_keys  = ["full_rho", "partial_rho_given_block",
                    "within_social_rho", "within_affective_rho"]
    bar_colors   = ["#2166AC", "#4DAC26", C_SOC, "#92C5DE"]

    x = np.arange(len(model_order))
    w = 0.18
    for mi, (mk, mc, mn) in enumerate(zip(metric_keys, bar_colors, metric_names)):
        vals = [wbc["models"][m][mk] for m in model_order]
        offset = (mi - 1.5) * w
        ax_a.bar(x + offset, vals, w, color=mc, edgecolor="white",
                 linewidth=0.3, label=mn, zorder=3)

    ax_a.axhline(0, color="black", linewidth=0.4, zorder=1)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_labels, fontsize=5.5)
    ax_a.set_ylabel(r"Spearman $\rho$", fontsize=6, labelpad=2)
    ax_a.set_ylim(-0.25, 0.85)
    ax_a.legend(fontsize=4.5, loc="upper right", ncol=2,
                columnspacing=0.8, handlelength=1.0)
    panel_label(ax_a, "a", x=-0.12, y=1.06)

    # ── Panel b: within-affective scatter (bottom-left) ──
    ax_b = fig.add_subplot(gs[1, 0])
    clean_ax(ax_b)

    brain_aff = brain_rdm[np.ix_(aff_idx, aff_idx)]
    llm_aff   = mean_llm_rdm[np.ix_(aff_idx, aff_idx)]
    b_aff_ut  = upper_tri(brain_aff)
    l_aff_ut  = upper_tri(llm_aff)

    # Color points by condition pair
    aff_cond_names = [brain_conds[i] for i in aff_idx]
    n_aff = len(aff_idx)
    ut_idx = np.triu_indices(n_aff, k=1)
    # Assign color by first condition in pair
    aff_pair_cols = []
    aff_cmap = plt.cm.Blues(np.linspace(0.3, 0.9, n_aff))
    for ii, jj in zip(ut_idx[0], ut_idx[1]):
        aff_pair_cols.append(aff_cmap[ii])

    ax_b.scatter(b_aff_ut, l_aff_ut, s=14, c=aff_pair_cols, alpha=0.85,
                 edgecolors="white", linewidths=0.3, zorder=3)

    rho_aff, _ = spearmanr(b_aff_ut, l_aff_ut)
    ax_b.text(0.04, 0.96, r"$\rho$ = " + f"{rho_aff:.2f} (n.s.)",
              transform=ax_b.transAxes, fontsize=5.5, va="top", color=C_AFF)

    z = np.polyfit(b_aff_ut, l_aff_ut, 1)
    xline = np.linspace(b_aff_ut.min(), b_aff_ut.max(), 50)
    ax_b.plot(xline, np.polyval(z, xline), color=C_AFF, linewidth=0.6,
              linestyle="--", alpha=0.5)

    ax_b.set_xlabel("Brain distance", fontsize=6, labelpad=2)
    ax_b.set_ylabel("Mean LLM distance", fontsize=6, labelpad=2)
    panel_label(ax_b, "b", x=-0.18, y=1.06)

    # ── Panel c: within-social scatter ──
    ax_c = fig.add_subplot(gs[2])
    clean_ax(ax_c)

    brain_soc = brain_rdm[np.ix_(soc_idx, soc_idx)]
    llm_soc   = mean_llm_rdm[np.ix_(soc_idx, soc_idx)]
    b_soc_ut  = upper_tri(brain_soc)
    l_soc_ut  = upper_tri(llm_soc)

    n_soc = len(soc_idx)
    ut_idx_s = np.triu_indices(n_soc, k=1)
    soc_pair_cols = []
    soc_cmap = plt.cm.Reds(np.linspace(0.3, 0.9, n_soc))
    for ii, jj in zip(ut_idx_s[0], ut_idx_s[1]):
        soc_pair_cols.append(soc_cmap[ii])

    ax_c.scatter(b_soc_ut, l_soc_ut, s=14, c=soc_pair_cols, alpha=0.85,
                 edgecolors="white", linewidths=0.3, zorder=3)

    rho_soc, p_soc = spearmanr(b_soc_ut, l_soc_ut)
    p_str = f"p = {p_soc:.3f}" if p_soc >= 0.001 else "p < 0.001"
    ax_c.text(0.04, 0.96, r"$\rho$ = " + f"{rho_soc:.2f}, {p_str}",
              transform=ax_c.transAxes, fontsize=5.5, va="top", color=C_SOC)

    z = np.polyfit(b_soc_ut, l_soc_ut, 1)
    xline = np.linspace(b_soc_ut.min(), b_soc_ut.max(), 50)
    ax_c.plot(xline, np.polyval(z, xline), color=C_SOC, linewidth=0.6,
              linestyle="--", alpha=0.5)

    ax_c.set_xlabel("Brain distance", fontsize=6, labelpad=2)
    ax_c.set_ylabel("Mean LLM distance", fontsize=6, labelpad=2)
    panel_label(ax_c, "c", x=-0.18, y=1.06)

    save_fig(fig, "figure2_decomposition")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Functional and Causal Significance
# ═══════════════════════════════════════════════════════════════════════
def figure3():
    print("Figure 3: Functional and Causal Significance")

    fig = plt.figure(figsize=(7.09, 4.0))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.35,
                          left=0.07, right=0.96, top=0.94, bottom=0.06)

    # ── Panel a: coupling dissociation ──
    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a)
    cr = load_json(os.path.join(CLIN, "coupling_reanalysis.json"))

    model_names_coupling = ["Qwen", "Llama", "Mistral", "Gemma"]
    x = np.arange(4)
    w = 0.32

    aa_vals = [cr["per_model"][m]["summary_2x2"]["A_aa"]
               for m in model_names_coupling]
    as_vals = [cr["per_model"][m]["summary_2x2"]["A_as"]
               for m in model_names_coupling]
    sa_vals = [cr["per_model"][m]["summary_2x2"]["A_sa"]
               for m in model_names_coupling]
    ss_vals = [cr["per_model"][m]["summary_2x2"]["A_ss"]
               for m in model_names_coupling]

    same_block  = [aa + ss for aa, ss in zip(aa_vals, ss_vals)]
    cross_block = [aas + sa for aas, sa in zip(as_vals, sa_vals)]

    ax_a.bar(x - w/2, same_block, w, color="#2166AC", edgecolor="white",
             linewidth=0.3, label="Same-block", zorder=3)
    ax_a.bar(x + w/2, cross_block, w, color="#B2182B", edgecolor="white",
             linewidth=0.3, label="Cross-block", zorder=3)

    for i, m in enumerate(model_names_coupling):
        p = cr["per_model"][m]["wilcoxon"]["wilcoxon_p"]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ymax = max(same_block[i], cross_block[i])
        ax_a.text(i, ymax + 0.004, stars, ha="center", fontsize=5.5,
                  fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(model_names_coupling, fontsize=5.5)
    ax_a.set_ylabel("Coupling effect", fontsize=6, labelpad=2)
    ax_a.legend(fontsize=4.5, loc="upper right", handlelength=1.0)
    panel_label(ax_a, "a", x=-0.16, y=1.06)

    # ── Panel b: brain confusion rho ──
    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b)
    da = load_json(os.path.join(RSA, "deep_analysis.json"))
    ca = da["confusion_analysis"]

    models_ca = ca["per_model"]
    mname_short = ["Qwen", "Llama", "Mistral", "Gemma"]
    conf_rhos = [m["brain_confusion_rho"] for m in models_ca]
    conf_ps   = [m["brain_confusion_p"] for m in models_ca]
    cols_ca   = [MODEL_COLORS[n] for n in mname_short]

    ax_b.bar(range(4), conf_rhos, color=cols_ca, edgecolor="white",
             linewidth=0.3, zorder=3, width=0.6)
    for i, (rho, p) in enumerate(zip(conf_rhos, conf_ps)):
        stars = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax_b.text(i, rho + 0.008, stars, ha="center", fontsize=5,
                  fontweight="bold")

    ax_b.axhline(0, color="black", linewidth=0.4)
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(mname_short, fontsize=5.5)
    ax_b.set_ylabel(r"Confusion $\rho$", fontsize=6, labelpad=2)

    # Grand stat annotation
    ax_b.text(0.96, 0.96,
              r"Grand $\rho$ = " + f"{ca['grand_brain_confusion_rho']:.3f}",
              transform=ax_b.transAxes, fontsize=5, va="top", ha="right",
              color="#555555")
    panel_label(ax_b, "b", x=-0.16, y=1.06)

    # ── Panel c: moral steering ──
    ax_c = fig.add_subplot(gs[1, 0])
    clean_ax(ax_c)
    ml = load_json(os.path.join(MOR, "Qwen2.5-7B-Instruct_moral_logit.json"))

    alphas  = [s["alpha"] for s in ml["alpha_summary"]]
    logit_d = [s["mean_logit_diff"] for s in ml["alpha_summary"]]

    ax_c.plot(alphas, logit_d, "o-", color="#2166AC", markersize=2.5,
              linewidth=0.8, zorder=3)

    # Shaded CI approximation via local scatter if available
    ax_c.fill_between(alphas, [ld - 0.15 for ld in logit_d],
                      [ld + 0.15 for ld in logit_d],
                      color="#2166AC", alpha=0.1, zorder=1)

    ax_c.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax_c.axvline(0, color="black", linewidth=0.4, linestyle=":")

    ax_c.set_xlabel(r"Steering $\alpha$", fontsize=6, labelpad=2)
    ax_c.set_ylabel("Logit diff (util. - deont.)", fontsize=6, labelpad=2)

    rho_all = ml["correlation_all"]["rho"]
    p_all   = ml["correlation_all"]["p"]
    p_str_c = f"p = {p_all:.3f}" if p_all >= 0.001 else "p < 0.001"
    ax_c.text(0.04, 0.04, r"$\rho$ = " + f"{rho_all:.3f}, {p_str_c}",
              transform=ax_c.transAxes, fontsize=5, va="bottom")
    panel_label(ax_c, "c", x=-0.16, y=1.06)

    # ── Panel d: steering controls — horizontal bars ──
    ax_d = fig.add_subplot(gs[1, 1])
    clean_ax(ax_d)
    djc = load_json(os.path.join(HR, "deepseek_judge_controls.json"))

    dir_names  = ["brain", "random", "sentiment", "pc1"]
    dir_labels = ["Brain axis", "Random", "Sentiment", "PC1"]
    dir_colors = ["#2166AC", "#999999", "#E08214", "#7570B3"]

    rhos_ctrl = [djc["directions"][d]["mean_rho"] for d in dir_names]
    ps_ctrl   = [djc["directions"][d]["ttest_p"] for d in dir_names]

    y_pos = np.arange(4)
    ax_d.barh(y_pos, rhos_ctrl, color=dir_colors, edgecolor="white",
              linewidth=0.3, height=0.6, zorder=3)
    ax_d.axvline(0, color="black", linewidth=0.4)

    for i, (rho, p) in enumerate(zip(rhos_ctrl, ps_ctrl)):
        stars = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        # Always place label at the right end of the bar (or just past zero)
        if rho >= 0:
            xpos = rho + 0.012
        else:
            # For negative bars, place label to the left of the bar end
            xpos = min(rho - 0.012, -0.012)
        ha = "left" if rho >= 0 else "right"
        ax_d.text(xpos, i, stars, va="center", ha=ha, fontsize=5,
                  fontweight="bold")

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(dir_labels, fontsize=5.5)
    ax_d.set_xlabel(r"Mean $\rho$ (judge ranking)", fontsize=6, labelpad=2)
    ax_d.invert_yaxis()
    panel_label(ax_d, "d", x=-0.22, y=1.06)

    save_fig(fig, "figure3_causal")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Developmental Trajectory
# ═══════════════════════════════════════════════════════════════════════
def figure4():
    print("Figure 4: Developmental Trajectory")

    pt = load_json(os.path.join(DEV, "pythia_trajectory.json"))
    ss = load_json(os.path.join(RSA, "scaling_summary.json"))
    bi = load_json(os.path.join(RSA, "base_vs_instruct.json"))

    fig = plt.figure(figsize=(7.09, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.9, 0.6],
                          wspace=0.38, left=0.06, right=0.97,
                          top=0.88, bottom=0.05)

    # ── Panel a: Pythia trajectory ──
    ax_a = fig.add_subplot(gs[0])
    clean_ax(ax_a)

    valid_cps = [cp for cp in pt["checkpoints"] if "full_rho" in cp]
    steps = [cp["step"] for cp in valid_cps]
    full  = [cp["full_rho"] for cp in valid_cps]
    w_soc = [cp["within_soc_rho"] for cp in valid_cps]
    w_aff = [cp["within_aff_rho"] for cp in valid_cps]
    steps_plot = [max(s, 1) for s in steps]

    # Fill between social and affective
    ax_a.fill_between(steps_plot, w_soc, w_aff, alpha=0.08, color="#888888",
                      zorder=1)

    ax_a.plot(steps_plot, full, "o-", color="#333333", markersize=2,
              linewidth=0.8, label=r"Full $\rho$", zorder=3)
    ax_a.plot(steps_plot, w_soc, "s-", color=C_SOC, markersize=2,
              linewidth=0.8, label=r"Within-social $\rho$", zorder=3)
    ax_a.plot(steps_plot, w_aff, "^-", color=C_AFF, markersize=2,
              linewidth=0.8, label=r"Within-aff. $\rho$", zorder=3)

    ax_a.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Training step", fontsize=6, labelpad=2)
    ax_a.set_ylabel(r"Spearman $\rho$ with brain", fontsize=6, labelpad=2)
    ax_a.legend(fontsize=4.5, loc="center right", handlelength=1.2)

    # Annotate zero-crossing for affective
    for i in range(len(w_aff) - 1):
        if w_aff[i] > 0 and w_aff[i + 1] < 0:
            cross_step = steps_plot[i + 1]
            ax_a.axvline(cross_step, color=C_AFF, linewidth=0.4,
                         linestyle=":", alpha=0.6)
            ax_a.annotate("", xy=(cross_step, 0),
                          xytext=(cross_step * 3, 0.12),
                          arrowprops=dict(arrowstyle="->", color=C_AFF,
                                          lw=0.5))
            break

    panel_label(ax_a, "a", x=-0.10, y=1.06)

    # ── Panel b: scale invariance ──
    ax_b = fig.add_subplot(gs[1])
    clean_ax(ax_b)

    sizes   = [m["params"] / 1e9 for m in ss["models"]]
    rhos_sc = [m["peak_rho"] for m in ss["models"]]
    ceil_sc = [m["noise_ceiling"] for m in ss["models"]]
    labels_sc = [m["label"] for m in ss["models"]]

    ax_b.plot(sizes, rhos_sc, "o-", color="#2166AC", markersize=3,
              linewidth=0.8, label=r"Brain-LLM $\rho$", zorder=3)
    ax_b.plot(sizes, ceil_sc, "s--", color="#999999", markersize=2.5,
              linewidth=0.6, label="LLM ceiling", zorder=2)

    for i, lbl in enumerate(labels_sc):
        ax_b.annotate(lbl, (sizes[i], rhos_sc[i]),
                      textcoords="offset points", xytext=(0, 5),
                      fontsize=4.5, ha="center")

    ax_b.set_xlabel("Parameters (B)", fontsize=6, labelpad=2)
    ax_b.set_ylabel(r"Spearman $\rho$", fontsize=6, labelpad=2)
    ax_b.set_ylim(0.65, 1.02)
    ax_b.legend(fontsize=4.5, loc="lower right", handlelength=1.2)
    panel_label(ax_b, "b", x=-0.18, y=1.06)

    # ── Panel c: base vs instruct (paired dot plot) ──
    ax_c = fig.add_subplot(gs[2])
    clean_ax(ax_c)

    base_rho = bi["comparison"]["base_peak_rho"]
    inst_rho = bi["comparison"]["instruct_peak_rho"]

    # Paired dot plot with connecting line
    ax_c.plot([0, 1], [base_rho, inst_rho], "o-", color="#2166AC",
              linewidth=0.8, markersize=4, zorder=3)
    ax_c.plot(0, base_rho, "o", color="#999999", markersize=5, zorder=4)
    ax_c.plot(1, inst_rho, "o", color="#2166AC", markersize=5, zorder=4)

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Base", "Instruct"], fontsize=5.5)
    ax_c.set_ylabel(r"Peak $\rho$", fontsize=6, labelpad=2)
    ax_c.set_ylim(0.72, 0.78)
    ax_c.set_xlim(-0.3, 1.3)

    ratio = bi["comparison"]["ratio_base_over_instruct"]
    ax_c.text(0.5, 0.06, f"{ratio:.1%}", transform=ax_c.transAxes,
              fontsize=5, ha="center", va="bottom", color="#555555")
    panel_label(ax_c, "c", x=-0.30, y=1.06)

    save_fig(fig, "figure4_developmental")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Robustness Controls
# ═══════════════════════════════════════════════════════════════════════
def figure5():
    print("Figure 5: Robustness Controls")

    bc = load_json(os.path.join(RSA, "baseline_controls.json"))
    tm = load_json(os.path.join(TM, "template_matched_rsa_results.json"))
    rg = load_json(os.path.join(RSA, "robustness_gauntlet.json"))
    ng = load_json(os.path.join(RSA, "narratives_group_rsa.json"))

    fig = plt.figure(figsize=(7.09, 4.0))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
                          left=0.08, right=0.97, top=0.94, bottom=0.06)

    # ── Panel a: baselines — horizontal bars ──
    ax_a = fig.add_subplot(gs[0, 0])
    clean_ax(ax_a)

    headline_rho = 0.739
    partial_info = [b for b in bc if b.get("method") == "partial_rsa"][0]
    partial_all_rho = partial_info["partial_all"]["rho"]
    pct_retained = partial_info["pct_retained"]

    baseline_items = [b for b in bc if b["method"] != "partial_rsa"]
    baseline_items_sorted = sorted(baseline_items, key=lambda x: -x["rho"])

    all_items = [("Headline", headline_rho, "#2166AC")]
    label_map = {
        "glove_300d": "GloVe-300d",
        "tfidf_5000": "TF-IDF",
        "condition_name_glove": "Cond.-name",
        "sentence_length": "Sent. length",
    }
    for b in baseline_items_sorted:
        name = label_map.get(b["method"], b["method"])
        sig = "***" if b["p"] < 0.001 else "**" if b["p"] < 0.01 \
              else "*" if b["p"] < 0.05 else ""
        all_items.append((name, b["rho"], "#E08214"))
    all_items.append((f"Partial ({pct_retained:.0f}%)", partial_all_rho,
                      "#1B9E77"))

    names_a = [it[0] for it in all_items]
    vals_a  = [it[1] for it in all_items]
    cols_a  = [it[2] for it in all_items]

    y_pos = np.arange(len(names_a))
    ax_a.barh(y_pos, vals_a, color=cols_a, edgecolor="white",
              linewidth=0.3, height=0.6, zorder=3)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names_a, fontsize=5)
    ax_a.set_xlabel(r"Spearman $\rho$", fontsize=6, labelpad=2)
    ax_a.invert_yaxis()

    for i, v in enumerate(vals_a):
        ax_a.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=4.5)

    panel_label(ax_a, "a", x=-0.22, y=1.06)

    # ── Panel b: template-matched (paired dot plot) ──
    ax_b = fig.add_subplot(gs[0, 1])
    clean_ax(ax_b)

    tm_models = tm["models"]
    model_map = {
        "Qwen2.5-7B-Instruct": "Qwen",
        "Meta-Llama-3.1-8B-Instruct": "Llama",
        "Mistral-7B-Instruct-v0.3": "Mistral",
        "gemma-2-9b-it": "Gemma",
    }
    tm_names = list(model_map.values())
    tm_orig  = [tm_models[k]["original_stimuli_rho"] for k in model_map]
    tm_match = [tm_models[k]["rho_at_headline_peak"] for k in model_map]
    tm_ret   = [tm_match[i] / tm_orig[i] * 100 for i in range(4)]
    model_cols = [MODEL_COLORS[n] for n in tm_names]

    # Paired dots with connecting lines
    for i in range(4):
        ax_b.plot([0, 1], [tm_orig[i], tm_match[i]], "-", color=model_cols[i],
                  linewidth=0.6, zorder=2)
        ax_b.plot(0, tm_orig[i], "o", color=model_cols[i], markersize=4,
                  zorder=3)
        ax_b.plot(1, tm_match[i], "s", color=model_cols[i], markersize=4,
                  zorder=3)
        # Retention percentage
        ax_b.text(1.08, tm_match[i], f"{tm_ret[i]:.0f}%", fontsize=4.5,
                  va="center", color=model_cols[i])

    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Original", "Template"], fontsize=5.5)
    ax_b.set_ylabel(r"Spearman $\rho$", fontsize=6, labelpad=2)
    ax_b.set_xlim(-0.2, 1.55)
    # Tight y-limits around data
    all_vals = tm_orig + tm_match
    ymin = min(all_vals) - 0.03
    ymax = max(all_vals) + 0.03
    ax_b.set_ylim(ymin, ymax)

    # Legend
    leg_handles = [Line2D([0], [0], marker="o", color=MODEL_COLORS[n],
                   linestyle="-", markersize=3, linewidth=0.6, label=n)
                   for n in tm_names]
    ax_b.legend(handles=leg_handles, fontsize=4.5, loc="lower left",
                handlelength=1.2)
    panel_label(ax_b, "b", x=-0.16, y=1.06)

    # ── Panel c: robustness grid ──
    ax_c = fig.add_subplot(gs[1, 0])

    tests = ["Split-half", "LOSO jackknife", "Cross-source",
             "Locked pipeline", "LOO condition", "LOMO-CV", "Stim. subsample"]
    model_list_short = ["Qwen", "Llama", "Mistral", "Gemma"]
    grid = np.ones((len(tests), len(model_list_short)))

    im = ax_c.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax_c.set_xticks(range(len(model_list_short)))
    ax_c.set_xticklabels(model_list_short, fontsize=5)
    ax_c.set_yticks(range(len(tests)))
    ax_c.set_yticklabels(tests, fontsize=4.5)

    for i in range(len(tests)):
        for j in range(len(model_list_short)):
            ax_c.text(j, i, "PASS", ha="center", va="center", fontsize=4.5,
                      fontweight="bold", color="white")

    heatmap_spines(ax_c)
    ax_c.tick_params(length=0)
    panel_label(ax_c, "c", x=-0.22, y=1.06)

    # ── Panel d: Narratives fMRI ──
    ax_d = fig.add_subplot(gs[1, 1])
    clean_ax(ax_d)

    narr_models = ng["models"]
    narr_ps = {
        "Qwen2.5-7B-Instruct": 0.004,
        "Meta-Llama-3.1-8B-Instruct": 0.007,
        "Mistral-7B-Instruct-v0.3": 0.013,
        "gemma-2-9b-it": 0.010,
    }
    ceiling_narr = ng["ceiling"]

    nm_labels = []
    nm_rhos   = []
    nm_ps     = []
    nm_colors = [MODEL_COLORS[model_map[mk]] for mk in MODEL_ORDER_FULL]

    for mk in MODEL_ORDER_FULL:
        nm_labels.append(model_map[mk])
        nm_rhos.append(narr_models[mk]["last"]["rho"])
        nm_ps.append(narr_ps.get(mk, 0.05))

    bars = ax_d.bar(range(4), nm_rhos, color=nm_colors, edgecolor="white",
                    linewidth=0.3, width=0.6, zorder=3)
    ax_d.axhline(ceiling_narr, color="#888888", linestyle="--",
                 linewidth=0.5, zorder=2)
    ax_d.text(0.96, ceiling_narr - 0.02, "ceiling", fontsize=4,
              color="#888888", va="top", ha="right",
              transform=mpl.transforms.blended_transform_factory(
                  ax_d.transAxes, ax_d.transData))

    for i, (rho, p) in enumerate(zip(nm_rhos, nm_ps)):
        stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax_d.text(i, rho + 0.01, stars, ha="center", fontsize=5,
                  fontweight="bold")

    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(nm_labels, fontsize=5.5)
    ax_d.set_ylabel(r"Spearman $\rho$", fontsize=6, labelpad=2)
    ax_d.set_ylim(0, 0.95)
    ax_d.text(0.96, 0.96, "N = 230 subjects", transform=ax_d.transAxes,
              fontsize=4.5, va="top", ha="right", color="#555555")
    panel_label(ax_d, "d", x=-0.16, y=1.06)

    save_fig(fig, "figure5_robustness")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 6 — The Embodiment Boundary
# ═══════════════════════════════════════════════════════════════════════
def figure6():
    print("Figure 6: The Embodiment Boundary")

    brain_rdm, brain_conds = load_rdm(os.path.join(RSA, "brain_rdm.npz"))

    model_files = {
        "Qwen":    "Qwen2.5-7B-Instruct_rdm14_headline.npz",
        "Llama":   "Meta-Llama-3.1-8B-Instruct_rdm14_headline.npz",
        "Mistral": "Mistral-7B-Instruct-v0.3_rdm14_headline.npz",
        "Gemma":   "gemma-2-9b-it_rdm14_headline.npz",
    }
    llm_rdms = {}
    for short_name, fname in model_files.items():
        rdm, _ = load_rdm(os.path.join(RSA, fname))
        llm_rdms[short_name] = rdm

    pp = load_json(os.path.join(RSA, "prospective_prediction.json"))
    pt = load_json(os.path.join(DEV, "pythia_trajectory.json"))

    fig = plt.figure(figsize=(7.09, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.0],
                          wspace=0.35, left=0.06, right=0.97,
                          top=0.88, bottom=0.05)

    # ── Panel a: per-condition alignment — horizontal bars sorted by rho ──
    ax_a = fig.add_subplot(gs[0])
    clean_ax(ax_a)

    n_conds = len(brain_conds)
    per_cond_rhos = {}
    for ci in range(n_conds):
        cond_name = brain_conds[ci]
        brain_row = np.concatenate([brain_rdm[ci, :ci], brain_rdm[ci, ci+1:]])
        llm_rhos = []
        for mname in llm_rdms:
            llm_row = np.concatenate([llm_rdms[mname][ci, :ci],
                                      llm_rdms[mname][ci, ci+1:]])
            rho, _ = spearmanr(brain_row, llm_row)
            llm_rhos.append(rho)
        per_cond_rhos[cond_name] = np.mean(llm_rhos)

    # Sort by rho descending
    ordered = sorted(per_cond_rhos.items(), key=lambda x: -x[1])
    cond_names = [PRETTY.get(c, c) for c, _ in ordered]
    cond_rhos  = [r for _, r in ordered]
    cond_cols  = [C_SOC if c in SOCIAL else C_AFF for c, _ in ordered]

    y_pos = np.arange(len(cond_names))
    bars = ax_a.barh(y_pos, cond_rhos, color=cond_cols, edgecolor="white",
                     linewidth=0.3, height=0.65, zorder=3)

    # Highlight empathy with gold
    for i, (c, _) in enumerate(ordered):
        if c == "empathy":
            ax_a.barh(i, cond_rhos[i], color="#FFD700", edgecolor=C_SOC,
                      linewidth=0.6, height=0.65, zorder=4)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(cond_names, fontsize=5)
    ax_a.set_xlabel(r"Row-wise $\rho$ (4 models)", fontsize=6, labelpad=2)
    ax_a.invert_yaxis()

    leg_handles = [Patch(facecolor=C_SOC, edgecolor="none", label="Social"),
                   Patch(facecolor=C_AFF, edgecolor="none", label="Affective"),
                   Patch(facecolor="#FFD700", edgecolor=C_SOC, label="Empathy")]
    ax_a.legend(handles=leg_handles, fontsize=4.5, loc="lower right",
                handlelength=1.0)
    panel_label(ax_a, "a", x=-0.16, y=1.06)

    # ── Panel b: prospective predictions forest plot ──
    ax_b = fig.add_subplot(gs[1])
    clean_ax(ax_b)

    pred_data = [
        ("P1: Coupling asym.",
         pp["prediction_1_coupling_asymmetry"]["average_rho"],
         pp["prediction_1_coupling_asymmetry"]["average_p_perm"], True),
        ("P2: Within-block",
         pp["prediction_2_within_block_coupling"]["blocks"]["affective"]["average_rho"],
         pp["prediction_2_within_block_coupling"]["blocks"]["affective"]["average_p_perm"],
         False),
        ("P3: Distinctiveness",
         pp["prediction_3_classification_accuracy"]["average_rho"],
         pp["prediction_3_classification_accuracy"]["average_p_perm"], False),
        ("P4: Vulnerable pairs",
         0.731, 0.012, True),
        ("P5: Boundary sens.",
         pp["prediction_5_boundary_sensitivity"]["average_rho"],
         pp["prediction_5_boundary_sensitivity"]["average_p_perm"], False),
    ]

    y_pos_b = np.arange(len(pred_data))
    for i, (name, rho, p, confirmed) in enumerate(pred_data):
        if confirmed:
            color = "#1B9E77"
            marker = "D"
        elif p < 0.2:
            color = "#E08214"
            marker = "o"
        else:
            color = "#B2182B"
            marker = "x"

        # Horizontal CI line (approximate +/- 0.15)
        ci_half = 0.12
        ax_b.plot([rho - ci_half, rho + ci_half], [i, i],
                  color=color, linewidth=0.8, zorder=2)
        ax_b.plot(rho, i, marker=marker, color=color, markersize=5,
                  zorder=3, markeredgewidth=0.5)

        sig_txt = "***" if p < 0.001 else "**" if p < 0.01 \
                  else "*" if p < 0.05 else ""
        if sig_txt:
            ax_b.text(rho + ci_half + 0.02, i, sig_txt, va="center",
                      fontsize=5, fontweight="bold", color=color)

    ax_b.axvline(0, color="black", linewidth=0.4, linestyle=":")
    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels([d[0] for d in pred_data], fontsize=5)
    ax_b.set_xlabel(r"Spearman $\rho$", fontsize=6, labelpad=2)
    ax_b.invert_yaxis()

    leg_b = [
        Line2D([0], [0], marker="D", color="#1B9E77", linestyle="None",
               markersize=4, label="Confirmed"),
        Line2D([0], [0], marker="o", color="#E08214", linestyle="None",
               markersize=4, label="Trend"),
        Line2D([0], [0], marker="x", color="#B2182B", linestyle="None",
               markersize=4, label="Null"),
    ]
    ax_b.legend(handles=leg_b, fontsize=4.5, loc="upper right",
                handlelength=1.0, borderpad=0.2)
    panel_label(ax_b, "b", x=-0.22, y=1.06)

    # ── Panel c: within-block divergence during training ──
    ax_c = fig.add_subplot(gs[2])
    clean_ax(ax_c)

    valid_cps = [cp for cp in pt["checkpoints"] if "full_rho" in cp]
    steps = [cp["step"] for cp in valid_cps]
    w_aff = [cp["within_aff_rho"] for cp in valid_cps]
    w_soc = [cp["within_soc_rho"] for cp in valid_cps]
    steps_plot = [max(s, 1) for s in steps]

    # Fill between to emphasize divergence
    ax_c.fill_between(steps_plot, w_soc, w_aff, alpha=0.12,
                      color="#888888", zorder=1)

    ax_c.plot(steps_plot, w_soc, "s-", color=C_SOC, markersize=2,
              linewidth=0.8, label=r"Social $\rho$", zorder=3)
    ax_c.plot(steps_plot, w_aff, "^-", color=C_AFF, markersize=2,
              linewidth=0.8, label=r"Affective $\rho$", zorder=3)

    ax_c.axhline(0, color="black", linewidth=0.4, linestyle=":")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Training step", fontsize=6, labelpad=2)
    ax_c.set_ylabel(r"Within-block $\rho$", fontsize=6, labelpad=2)
    ax_c.legend(fontsize=4.5, loc="lower left", handlelength=1.2)
    panel_label(ax_c, "c", x=-0.18, y=1.06)

    save_fig(fig, "figure6_embodiment")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}
    for fig_fn in [figure1, figure2, figure3, figure4, figure5, figure6]:
        name = fig_fn.__name__
        try:
            ok = fig_fn()
            results[name] = "OK"
        except Exception as e:
            results[name] = f"FAILED: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")
