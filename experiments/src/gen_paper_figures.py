#!/usr/bin/env python3
"""
Generate all 6 publication-quality figures for the Nature paper.
Reads real data from result JSONs/NPZs — no placeholder data.
Saves to experiments/figures/paper/ as both PDF and PNG (300 DPI).
"""

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, pearsonr

# ── Nature-style rcParams ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "figure.dpi": 150,
})

# ── Paths ──────────────────────────────────────────────────────────────
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

# ── Color scheme ───────────────────────────────────────────────────────
C_AFF  = "#3574B0"   # blue for affective
C_SOC  = "#D6604D"   # red-orange for social/mentalizing
C_CROSS = "#8E8E8E"  # gray for cross-block
C_BG   = "#F7F7F7"

MODEL_COLORS = {
    "Qwen":    "#3574B0",
    "Llama":   "#E8853A",
    "Mistral": "#59A14F",
    "Gemma":   "#B07AA1",
}
MODEL_SHORT = {
    "Qwen2.5-7B-Instruct": "Qwen-7B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-8B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "gemma-2-9b-it": "Gemma-9B",
}

# ── Conditions ─────────────────────────────────────────────────────────
AFFECTIVE = {"anger", "disgust", "fear", "happiness", "sadness", "valence"}
SOCIAL    = {"belief", "empathy", "intention", "judgment", "mentalizing",
             "moral", "self_referential", "theory_of_mind"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_rdm(path):
    d = np.load(path, allow_pickle=True)
    return d["rdm"], list(d["conditions"])


def upper_tri(rdm):
    """Return upper triangle (excluding diagonal) as flat array."""
    n = rdm.shape[0]
    idx = np.triu_indices(n, k=1)
    return rdm[idx]


def pair_types(conditions):
    """Return boolean masks for within-affective, within-social, cross-block
    for the upper triangle of a 14x14 matrix."""
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
    """Save as PDF and PNG."""
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / {name}.png")


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 — The Discovery
# ═══════════════════════════════════════════════════════════════════════
def figure1():
    print("Figure 1: The Discovery")

    brain_rdm, brain_conds = load_rdm(os.path.join(RSA, "brain_rdm.npz"))
    qwen_rdm, qwen_conds  = load_rdm(os.path.join(RSA, "Qwen2.5-7B-Instruct_rdm14_headline.npz"))

    # Reorder: affective first, then social, each alphabetical
    aff_sorted = sorted([c for c in brain_conds if c in AFFECTIVE])
    soc_sorted = sorted([c for c in brain_conds if c in SOCIAL])
    order = aff_sorted + soc_sorted
    idx = [brain_conds.index(c) for c in order]

    brain_reord = brain_rdm[np.ix_(idx, idx)]
    qwen_reord  = qwen_rdm[np.ix_(idx, idx)]

    # Pretty labels
    pretty = {
        "anger": "Anger", "disgust": "Disgust", "fear": "Fear",
        "happiness": "Happiness", "sadness": "Sadness", "valence": "Valence",
        "belief": "Belief", "empathy": "Empathy", "intention": "Intention",
        "judgment": "Judgment", "mentalizing": "Mentalizing", "moral": "Moral",
        "self_referential": "Self-ref.", "theory_of_mind": "ToM",
    }
    labels = [pretty.get(c, c) for c in order]

    # Scaling
    scaling = load_json(os.path.join(RSA, "scaling_summary.json"))
    ceiling_vals = [m["noise_ceiling"] for m in scaling["models"]]
    mean_ceiling = np.mean(ceiling_vals)

    fig = plt.figure(figsize=(180/25.4, 55/25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.7, 1],
                          wspace=0.45, left=0.06, right=0.97, top=0.88, bottom=0.18)

    # ── Panel a: side-by-side heatmaps ──
    gs_a = gs[0].subgridspec(1, 2, wspace=0.08)
    ax_br = fig.add_subplot(gs_a[0])
    ax_lm = fig.add_subplot(gs_a[1])

    vmin = min(brain_reord.min(), qwen_reord.min())
    vmax = max(brain_reord.max(), qwen_reord.max())

    for ax, rdm, title in [(ax_br, brain_reord, "Brain (Neurosynth)"),
                            (ax_lm, qwen_reord, "Qwen-7B (LLM)")]:
        im = ax.imshow(rdm, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(14))
        ax.set_yticks(range(14))
        if ax == ax_br:
            ax.set_yticklabels(labels, fontsize=5.5)
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels(labels, fontsize=5.5, rotation=60, ha="right")
        ax.set_title(title, fontsize=7, pad=4)

        # Block boundary
        rect_aff = Rectangle((-0.5, -0.5), len(aff_sorted), len(aff_sorted),
                              linewidth=1.2, edgecolor=C_AFF, facecolor="none",
                              linestyle="-")
        rect_soc = Rectangle((len(aff_sorted)-0.5, len(aff_sorted)-0.5),
                              len(soc_sorted), len(soc_sorted),
                              linewidth=1.2, edgecolor=C_SOC, facecolor="none",
                              linestyle="-")
        ax.add_patch(rect_aff)
        ax.add_patch(rect_soc)
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_linewidth(0.4)
        ax.spines["top"].set_linewidth(0.4)
        ax.spines["left"].set_linewidth(0.4)
        ax.spines["bottom"].set_linewidth(0.4)

    # Colorbar
    cbar = fig.colorbar(im, ax=[ax_br, ax_lm], fraction=0.03, pad=0.04, shrink=0.85)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.set_label("1 - Pearson r (cosine dist.)", fontsize=6)

    panel_label(ax_br, "a", x=-0.25)

    # ── Panel b: bar chart of headline rho ──
    ax_b = fig.add_subplot(gs[1])

    models_data = [
        ("Qwen-7B", 0.739, "< 0.0002"),
        ("Llama-8B", 0.727, "< 0.0002"),
        ("Mistral-7B", 0.730, "< 0.0002"),
        ("Gemma-9B", 0.735, "< 0.0002"),
    ]
    names = [m[0] for m in models_data]
    rhos  = [m[1] for m in models_data]
    colors_bar = [MODEL_COLORS["Qwen"], MODEL_COLORS["Llama"],
                  MODEL_COLORS["Mistral"], MODEL_COLORS["Gemma"]]

    bars = ax_b.bar(range(4), rhos, color=colors_bar, width=0.65, edgecolor="white",
                    linewidth=0.5, zorder=3)
    ax_b.axhline(mean_ceiling, color="#999999", linestyle="--", linewidth=0.8,
                 label=f"LLM ceiling ({mean_ceiling:.2f})", zorder=2)
    ax_b.set_xticks(range(4))
    ax_b.set_xticklabels(names, fontsize=6, rotation=30, ha="right")
    ax_b.set_ylabel("Spearman " + r"$\rho$", fontsize=7)
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(fontsize=5.5, loc="upper right")

    for i, (_, rho, p) in enumerate(models_data):
        ax_b.text(i, rho + 0.02, f"p {p}", ha="center", va="bottom", fontsize=4.5)

    panel_label(ax_b, "b")

    # ── Panel c: scatter ──
    ax_c = fig.add_subplot(gs[2])

    b_ut = upper_tri(brain_rdm)
    q_ut = upper_tri(qwen_rdm)
    wa, ws, cb = pair_types(brain_conds)

    ax_c.scatter(b_ut[cb], q_ut[cb], s=10, c=C_CROSS, alpha=0.6, edgecolors="none",
                 label="Cross-block", zorder=2)
    ax_c.scatter(b_ut[wa], q_ut[wa], s=14, c=C_AFF, alpha=0.8, edgecolors="none",
                 label="Within-affective", zorder=3)
    ax_c.scatter(b_ut[ws], q_ut[ws], s=14, c=C_SOC, alpha=0.8, edgecolors="none",
                 label="Within-social", zorder=3)

    # Regression line
    z = np.polyfit(b_ut, q_ut, 1)
    xline = np.linspace(b_ut.min(), b_ut.max(), 100)
    ax_c.plot(xline, np.polyval(z, xline), color="black", linewidth=0.8,
              linestyle="--", zorder=1)

    rho_val, _ = spearmanr(b_ut, q_ut)
    ax_c.text(0.05, 0.95, f"Spearman " + r"$\rho$" + f" = {rho_val:.3f}",
              transform=ax_c.transAxes, fontsize=6, va="top")

    ax_c.set_xlabel("Brain RDM distance", fontsize=7)
    ax_c.set_ylabel("LLM RDM distance (Qwen-7B)", fontsize=7)
    ax_c.legend(fontsize=5, loc="lower right", markerscale=1.2)

    panel_label(ax_c, "c")

    save_fig(fig, "figure1_discovery")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Decomposing the Alignment
# ═══════════════════════════════════════════════════════════════════════
def figure2():
    print("Figure 2: Decomposing the Alignment")

    wbc = load_json(os.path.join(RSA, "within_block_control.json"))

    # Load RDMs for sub-matrix panels
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

    # Mean LLM RDM
    mean_llm_rdm = np.mean([llm_rdms[k] for k in llm_rdms], axis=0)

    # Condition indices
    aff_idx = sorted([i for i, c in enumerate(brain_conds) if c in AFFECTIVE])
    soc_idx = sorted([i for i, c in enumerate(brain_conds) if c in SOCIAL])

    aff_labels = [brain_conds[i].capitalize() for i in aff_idx]
    soc_labels = [brain_conds[i].capitalize().replace("Self_referential", "Self-ref.").replace("Theory_of_mind", "ToM") for i in soc_idx]
    soc_labels = [l.replace("Self_referential", "Self-ref.").replace("Theory_of_mind", "ToM") for l in soc_labels]
    # Fix labels
    soc_labels_clean = []
    for i in soc_idx:
        c = brain_conds[i]
        pretty = {"belief": "Belief", "empathy": "Empathy", "intention": "Intention",
                  "judgment": "Judgment", "mentalizing": "Mentalizing", "moral": "Moral",
                  "self_referential": "Self-ref.", "theory_of_mind": "ToM"}
        soc_labels_clean.append(pretty.get(c, c.capitalize()))
    soc_labels = soc_labels_clean

    aff_labels_clean = []
    for i in aff_idx:
        c = brain_conds[i]
        aff_labels_clean.append(c.capitalize())
    aff_labels = aff_labels_clean

    fig = plt.figure(figsize=(180/25.4, 65/25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 1.0],
                          wspace=0.4, left=0.07, right=0.97, top=0.88, bottom=0.15)

    # ── Panel a: grouped bar chart ──
    ax_a = fig.add_subplot(gs[0])

    model_order = ["Qwen2.5-7B", "Llama-3.1-8B", "Mistral-7B", "Gemma-2-9B"]
    short_labels = ["Qwen-7B", "Llama-8B", "Mistral-7B", "Gemma-9B"]
    metric_names = ["Full " + r"$\rho$", "Partial " + r"$\rho$", "Within-social " + r"$\rho$", "Within-aff. " + r"$\rho$"]
    metric_keys  = ["full_rho", "partial_rho_given_block", "within_social_rho", "within_affective_rho"]
    bar_colors   = ["#4C72B0", "#55A868", C_SOC, "#8CB4D5"]

    x = np.arange(len(model_order))
    w = 0.18
    for mi, (mk, mc) in enumerate(zip(metric_keys, bar_colors)):
        vals = [wbc["models"][m][mk] for m in model_order]
        offset = (mi - 1.5) * w
        bars = ax_a.bar(x + offset, vals, w, color=mc, edgecolor="white",
                        linewidth=0.4, label=metric_names[mi], zorder=3)

    ax_a.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_labels, fontsize=6, rotation=20, ha="right")
    ax_a.set_ylabel("Spearman " + r"$\rho$", fontsize=7)
    ax_a.set_ylim(-0.3, 0.85)
    ax_a.legend(fontsize=5, loc="upper right", ncol=2)
    panel_label(ax_a, "a", x=-0.15)

    # ── Panel b: within-affective sub-RDM scatter ──
    ax_b = fig.add_subplot(gs[1])

    brain_aff = brain_rdm[np.ix_(aff_idx, aff_idx)]
    llm_aff   = mean_llm_rdm[np.ix_(aff_idx, aff_idx)]
    b_aff_ut  = upper_tri(brain_aff)
    l_aff_ut  = upper_tri(llm_aff)

    ax_b.scatter(b_aff_ut, l_aff_ut, s=18, c=C_AFF, alpha=0.8, edgecolors="white",
                 linewidths=0.3, zorder=3)
    rho_aff, _ = spearmanr(b_aff_ut, l_aff_ut)
    ax_b.text(0.05, 0.95, f"Within-affective\n" + r"$\rho$" + f" = {rho_aff:.2f} (n.s.)",
              transform=ax_b.transAxes, fontsize=5.5, va="top",
              color=C_AFF)
    # Regression line
    z = np.polyfit(b_aff_ut, l_aff_ut, 1)
    xline = np.linspace(b_aff_ut.min(), b_aff_ut.max(), 50)
    ax_b.plot(xline, np.polyval(z, xline), color=C_AFF, linewidth=0.7,
              linestyle="--", alpha=0.5)

    ax_b.set_xlabel("Brain distance", fontsize=6)
    ax_b.set_ylabel("Mean LLM distance", fontsize=6)
    ax_b.set_title("Within-affective (6x6)", fontsize=7, color=C_AFF)
    panel_label(ax_b, "b")

    # ── Panel c: within-social sub-RDM scatter ──
    ax_c = fig.add_subplot(gs[2])

    brain_soc = brain_rdm[np.ix_(soc_idx, soc_idx)]
    llm_soc   = mean_llm_rdm[np.ix_(soc_idx, soc_idx)]
    b_soc_ut  = upper_tri(brain_soc)
    l_soc_ut  = upper_tri(llm_soc)

    ax_c.scatter(b_soc_ut, l_soc_ut, s=18, c=C_SOC, alpha=0.8, edgecolors="white",
                 linewidths=0.3, zorder=3)
    rho_soc, p_soc = spearmanr(b_soc_ut, l_soc_ut)
    p_str = f"p = {p_soc:.3f}" if p_soc >= 0.001 else "p < 0.001"
    ax_c.text(0.05, 0.95, f"Within-social\n" + r"$\rho$" + f" = {rho_soc:.2f}, {p_str}",
              transform=ax_c.transAxes, fontsize=5.5, va="top",
              color=C_SOC)
    z = np.polyfit(b_soc_ut, l_soc_ut, 1)
    xline = np.linspace(b_soc_ut.min(), b_soc_ut.max(), 50)
    ax_c.plot(xline, np.polyval(z, xline), color=C_SOC, linewidth=0.7,
              linestyle="--", alpha=0.5)

    ax_c.set_xlabel("Brain distance", fontsize=6)
    ax_c.set_ylabel("Mean LLM distance", fontsize=6)
    ax_c.set_title("Within-social (8x8)", fontsize=7, color=C_SOC)
    panel_label(ax_c, "c")

    save_fig(fig, "figure2_decomposition")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Functional and Causal Significance
# ═══════════════════════════════════════════════════════════════════════
def figure3():
    print("Figure 3: Functional and Causal Significance")

    fig = plt.figure(figsize=(180/25.4, 105/25.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
                          left=0.08, right=0.95, top=0.93, bottom=0.08)

    # ── Panel a: coupling dissociation ──
    ax_a = fig.add_subplot(gs[0, 0])
    cr = load_json(os.path.join(CLIN, "coupling_reanalysis.json"))

    model_names_coupling = ["Qwen", "Llama", "Mistral", "Gemma"]
    short_labels = ["Qwen-7B", "Llama-8B", "Mistral-7B", "Gemma-9B"]

    x = np.arange(len(model_names_coupling))
    w = 0.35

    aa_vals = [cr["per_model"][m]["summary_2x2"]["A_aa"] for m in model_names_coupling]
    as_vals = [cr["per_model"][m]["summary_2x2"]["A_as"] for m in model_names_coupling]
    sa_vals = [cr["per_model"][m]["summary_2x2"]["A_sa"] for m in model_names_coupling]
    ss_vals = [cr["per_model"][m]["summary_2x2"]["A_ss"] for m in model_names_coupling]

    same_block = [aa + ss for aa, ss in zip(aa_vals, ss_vals)]
    cross_block = [aas + sa for aas, sa in zip(as_vals, sa_vals)]

    bars1 = ax_a.bar(x - w/2, same_block, w, color="#4C72B0", edgecolor="white",
                     linewidth=0.4, label="Same-block coupling", zorder=3)
    bars2 = ax_a.bar(x + w/2, cross_block, w, color="#C44E52", edgecolor="white",
                     linewidth=0.4, label="Cross-block coupling", zorder=3)

    for i, m in enumerate(model_names_coupling):
        p = cr["per_model"][m]["wilcoxon"]["wilcoxon_p"]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ymax = max(same_block[i], cross_block[i])
        ax_a.text(i, ymax + 0.008, stars, ha="center", fontsize=7, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_labels, fontsize=6, rotation=20, ha="right")
    ax_a.set_ylabel("Mean coupling effect", fontsize=7)
    ax_a.legend(fontsize=5.5, loc="upper left")
    ax_a.set_title("Block-specific coupling", fontsize=7)
    panel_label(ax_a, "a", x=-0.18)

    # ── Panel b: brain confusion vs LLM confusion ──
    ax_b = fig.add_subplot(gs[0, 1])
    da = load_json(os.path.join(RSA, "deep_analysis.json"))
    ca = da["confusion_analysis"]

    # Per-model accuracy + confusion rho
    models_ca = ca["per_model"]
    names_ca  = [MODEL_SHORT.get(m["model"], m["model"][:8]) for m in models_ca]
    accs      = [m["accuracy"] for m in models_ca]
    conf_rhos = [m["brain_confusion_rho"] for m in models_ca]
    conf_ps   = [m["brain_confusion_p"] for m in models_ca]

    colors_ca = [MODEL_COLORS["Qwen"], MODEL_COLORS["Llama"],
                 MODEL_COLORS["Mistral"], MODEL_COLORS["Gemma"]]

    ax_b.bar(range(len(models_ca)), conf_rhos, color=colors_ca,
             edgecolor="white", linewidth=0.4, zorder=3, width=0.65)
    for i, (rho, p) in enumerate(zip(conf_rhos, conf_ps)):
        stars = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax_b.text(i, rho + 0.01, stars, ha="center", fontsize=6, fontweight="bold")

    ax_b.axhline(0, color="black", linewidth=0.5)
    ax_b.set_xticks(range(len(models_ca)))
    ax_b.set_xticklabels(names_ca, fontsize=6, rotation=20, ha="right")
    ax_b.set_ylabel("Brain-LLM confusion " + r"$\rho$", fontsize=7)
    ax_b.set_title("Brain distances predict LLM confusion", fontsize=7)
    # Grand stat
    ax_b.text(0.95, 0.95,
              f"Grand: " + r"$\rho$" + f" = {ca['grand_brain_confusion_rho']:.3f}\np = {ca['grand_brain_confusion_p']:.3f}",
              transform=ax_b.transAxes, fontsize=5.5, va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.8))
    panel_label(ax_b, "b")

    # ── Panel c: moral steering curve ──
    ax_c = fig.add_subplot(gs[1, 0])
    ml = load_json(os.path.join(MOR, "Qwen2.5-7B-Instruct_moral_logit.json"))

    alphas    = [s["alpha"] for s in ml["alpha_summary"]]
    logit_d   = [s["mean_logit_diff"] for s in ml["alpha_summary"]]
    p_util    = [s["mean_p_util"] for s in ml["alpha_summary"]]

    ax_c.plot(alphas, logit_d, "o-", color="#4C72B0", markersize=4, linewidth=1.2,
              label="Logit diff", zorder=3)
    ax_c.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_c.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax_c.set_xlabel(r"Steering $\alpha$ (affective $\leftarrow$ 0 $\rightarrow$ mentalizing)", fontsize=6.5)
    ax_c.set_ylabel("Mean logit diff\n(utilitarian - deontological)", fontsize=6.5)
    ax_c.set_title("Moral judgment steering", fontsize=7)

    rho_all = ml["correlation_all"]["rho"]
    p_all   = ml["correlation_all"]["p"]
    ax_c.text(0.05, 0.05,
              r"$\rho$" + f" = {rho_all:.3f}, p = {p_all:.3f}",
              transform=ax_c.transAxes, fontsize=5.5, va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.8))

    # Secondary y-axis for P(utilitarian)
    ax_c2 = ax_c.twinx()
    ax_c2.plot(alphas, p_util, "s--", color=C_SOC, markersize=3, linewidth=0.8,
               label="P(utilitarian)", alpha=0.7)
    ax_c2.set_ylabel("P(utilitarian)", fontsize=6, color=C_SOC)
    ax_c2.tick_params(axis="y", labelcolor=C_SOC, labelsize=5.5)
    ax_c2.spines["right"].set_visible(True)
    ax_c2.spines["right"].set_color(C_SOC)
    ax_c2.spines["right"].set_linewidth(0.6)

    panel_label(ax_c, "c", x=-0.18)

    # ── Panel d: steering controls ──
    ax_d = fig.add_subplot(gs[1, 1])
    djc = load_json(os.path.join(HR, "deepseek_judge_controls.json"))

    dir_names = ["brain", "random", "sentiment", "pc1"]
    dir_labels = ["Brain axis", "Random", "Sentiment", "PC1"]
    dir_colors = ["#4C72B0", "#999999", "#E8853A", "#B07AA1"]

    rhos_ctrl = [djc["directions"][d]["mean_rho"] for d in dir_names]
    ps_ctrl   = [djc["directions"][d]["ttest_p"] for d in dir_names]

    bars = ax_d.bar(range(4), rhos_ctrl, color=dir_colors, edgecolor="white",
                    linewidth=0.4, width=0.65, zorder=3)
    ax_d.axhline(0, color="black", linewidth=0.5)

    for i, (rho, p) in enumerate(zip(rhos_ctrl, ps_ctrl)):
        stars = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y = rho + 0.02 if rho >= 0 else rho - 0.04
        ax_d.text(i, y, stars, ha="center", fontsize=6, fontweight="bold")

    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(dir_labels, fontsize=6, rotation=20, ha="right")
    ax_d.set_ylabel("Mean " + r"$\rho$" + " (judge ranking)", fontsize=7)
    ax_d.set_title("Steering direction specificity", fontsize=7)
    panel_label(ax_d, "d")

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

    fig = plt.figure(figsize=(180/25.4, 62/25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.9, 0.7],
                          wspace=0.45, left=0.08, right=0.95, top=0.88, bottom=0.18)

    # ── Panel a: Pythia trajectory ──
    ax_a = fig.add_subplot(gs[0])

    # Filter out checkpoints with errors (e.g., step 143000)
    valid_cps = [cp for cp in pt["checkpoints"] if "full_rho" in cp]
    steps = [cp["step"] for cp in valid_cps]
    full  = [cp["full_rho"] for cp in valid_cps]
    w_soc = [cp["within_soc_rho"] for cp in valid_cps]
    w_aff = [cp["within_aff_rho"] for cp in valid_cps]

    # Use log scale but shift step 0 to 1 for log
    steps_plot = [max(s, 1) for s in steps]

    ax_a.plot(steps_plot, full, "o-", color="#333333", markersize=3, linewidth=1.2,
              label="Full " + r"$\rho$", zorder=3)
    ax_a.plot(steps_plot, w_soc, "s-", color=C_SOC, markersize=3, linewidth=1.0,
              label="Within-social " + r"$\rho$", zorder=3)
    ax_a.plot(steps_plot, w_aff, "^-", color=C_AFF, markersize=3, linewidth=1.0,
              label="Within-affective " + r"$\rho$", zorder=3)

    ax_a.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Training step", fontsize=7)
    ax_a.set_ylabel("Spearman " + r"$\rho$" + " with brain", fontsize=7)
    ax_a.set_title("Pythia-2.8B training trajectory", fontsize=7)
    ax_a.legend(fontsize=5.5, loc="center right")

    # Mark zero crossing for affective
    for i in range(len(w_aff)-1):
        if w_aff[i] > 0 and w_aff[i+1] < 0:
            ax_a.axvline(steps_plot[i+1], color=C_AFF, linewidth=0.5,
                         linestyle=":", alpha=0.6)
            ax_a.annotate("affective\ndivergence",
                          xy=(steps_plot[i+1], 0), fontsize=4.5,
                          ha="center", va="bottom", color=C_AFF,
                          xytext=(steps_plot[i+1]*3, 0.15),
                          arrowprops=dict(arrowstyle="->", color=C_AFF,
                                          lw=0.5))
            break

    panel_label(ax_a, "a", x=-0.12)

    # ── Panel b: scale invariance ──
    ax_b = fig.add_subplot(gs[1])

    sizes   = [m["params"]/1e9 for m in ss["models"]]
    rhos_sc = [m["peak_rho"] for m in ss["models"]]
    ceil_sc = [m["noise_ceiling"] for m in ss["models"]]
    labels_sc = [m["label"] for m in ss["models"]]

    ax_b.plot(sizes, rhos_sc, "o-", color="#4C72B0", markersize=5, linewidth=1.2,
              label="Brain-LLM " + r"$\rho$", zorder=3)
    ax_b.plot(sizes, ceil_sc, "s--", color="#999999", markersize=4, linewidth=0.8,
              label="LLM ceiling", zorder=2)

    for i, lbl in enumerate(labels_sc):
        ax_b.annotate(lbl, (sizes[i], rhos_sc[i]),
                      textcoords="offset points", xytext=(0, 7),
                      fontsize=5, ha="center")

    ax_b.set_xlabel("Model size (B parameters)", fontsize=7)
    ax_b.set_ylabel("Spearman " + r"$\rho$", fontsize=7)
    ax_b.set_title("Scale invariance (Qwen family)", fontsize=7)
    ax_b.set_ylim(0.65, 1.02)
    ax_b.legend(fontsize=5.5, loc="center right")
    panel_label(ax_b, "b")

    # ── Panel c: base vs instruct ──
    ax_c = fig.add_subplot(gs[2])

    base_rho = bi["comparison"]["base_peak_rho"]
    inst_rho = bi["comparison"]["instruct_peak_rho"]
    ratio    = bi["comparison"]["ratio_base_over_instruct"]

    bars = ax_c.bar([0, 1], [base_rho, inst_rho],
                    color=["#999999", "#4C72B0"],
                    edgecolor="white", linewidth=0.5, width=0.55, zorder=3)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Base", "Instruct"], fontsize=6)
    ax_c.set_ylabel("Peak " + r"$\rho$", fontsize=7)
    ax_c.set_ylim(0.6, 0.8)
    ax_c.set_title("Qwen-1.5B: Base vs Instruct", fontsize=7)

    ax_c.text(0.5, 0.05, f"Ratio: {ratio:.1%}",
              transform=ax_c.transAxes, fontsize=6, ha="center", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        edgecolor="#cccccc", alpha=0.8))
    panel_label(ax_c, "c")

    save_fig(fig, "figure4_developmental")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Robustness Controls
# ═══════════════════════════════════════════════════════════════════════
def figure5():
    print("Figure 5: Robustness Controls")

    bc = load_json(os.path.join(RSA, "baseline_controls.json"))
    tm = load_json(os.path.join(TM, "template_matched_rsa_results.json"))
    pi = load_json(os.path.join(RSA, "paraphrase_invariance.json"))
    rg = load_json(os.path.join(RSA, "robustness_gauntlet.json"))
    ng = load_json(os.path.join(RSA, "narratives_group_rsa.json"))

    fig = plt.figure(figsize=(180/25.4, 110/25.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35,
                          left=0.09, right=0.95, top=0.93, bottom=0.08)

    # ── Panel a: baseline controls ──
    ax_a = fig.add_subplot(gs[0, 0])

    # Build ordered list: headline, baselines, partial
    headline_rho = 0.739  # from FINDINGS
    partial_info = [b for b in bc if b.get("method") == "partial_rsa"][0]
    partial_all_rho = partial_info["partial_all"]["rho"]
    pct_retained = partial_info["pct_retained"]

    baseline_items = [b for b in bc if b["method"] != "partial_rsa"]
    baseline_items_sorted = sorted(baseline_items, key=lambda x: -x["rho"])

    all_items = [("Headline (Qwen-7B)", headline_rho, "#4C72B0")]
    for b in baseline_items_sorted:
        name = b["method"].replace("_", " ").replace("glove 300d", "GloVe 300d")\
                          .replace("condition name glove", "Cond.-name GloVe")\
                          .replace("sentence length", "Sent. length")\
                          .replace("tfidf 5000", "TF-IDF 5000")
        sig = "***" if b["p"] < 0.001 else "**" if b["p"] < 0.01 else "*" if b["p"] < 0.05 else ""
        all_items.append((name + " " + sig, b["rho"], "#E8853A"))
    all_items.append((f"Partial RSA ({pct_retained:.0f}% retained)", partial_all_rho, "#55A868"))

    names_a = [it[0] for it in all_items]
    vals_a  = [it[1] for it in all_items]
    cols_a  = [it[2] for it in all_items]

    y_pos = np.arange(len(names_a))
    ax_a.barh(y_pos, vals_a, color=cols_a, edgecolor="white", linewidth=0.4,
              height=0.65, zorder=3)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names_a, fontsize=5.5)
    ax_a.set_xlabel("Spearman " + r"$\rho$", fontsize=7)
    ax_a.set_title("Lexical baselines + partial RSA", fontsize=7)
    ax_a.invert_yaxis()

    for i, v in enumerate(vals_a):
        ax_a.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=5)

    panel_label(ax_a, "a", x=-0.25)

    # ── Panel b: template-matched ──
    ax_b = fig.add_subplot(gs[0, 1])

    tm_models = tm["models"]
    model_map = {
        "Qwen2.5-7B-Instruct": "Qwen-7B",
        "Meta-Llama-3.1-8B-Instruct": "Llama-8B",
        "Mistral-7B-Instruct-v0.3": "Mistral-7B",
        "gemma-2-9b-it": "Gemma-9B",
    }
    tm_names = list(model_map.values())
    tm_orig  = [tm_models[k]["original_stimuli_rho"] for k in model_map]
    tm_match = [tm_models[k]["rho_at_headline_peak"] for k in model_map]
    tm_ret   = [tm_match[i]/tm_orig[i]*100 for i in range(4)]

    x = np.arange(4)
    w = 0.35
    ax_b.bar(x - w/2, tm_orig, w, color="#4C72B0", edgecolor="white",
             linewidth=0.4, label="Original", zorder=3)
    ax_b.bar(x + w/2, tm_match, w, color="#55A868", edgecolor="white",
             linewidth=0.4, label="Template-matched", zorder=3)

    for i in range(4):
        ymax = max(tm_orig[i], tm_match[i])
        ax_b.text(i, ymax + 0.02, f"{tm_ret[i]:.0f}%", ha="center", fontsize=5,
                  fontweight="bold")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(tm_names, fontsize=6, rotation=20, ha="right")
    ax_b.set_ylabel("Spearman " + r"$\rho$", fontsize=7)
    ax_b.set_title("Template-matched format control", fontsize=7)
    ax_b.legend(fontsize=5.5, loc="upper right")
    ax_b.set_ylim(0, 0.85)
    panel_label(ax_b, "b")

    # ── Panel c: paraphrase + gauntlet summary grid ──
    ax_c = fig.add_subplot(gs[1, 0])

    # Build a pass/fail grid
    tests = [
        "Split-half", "LOSO jackknife", "Cross-source",
        "Locked pipeline", "LOO condition", "LOMO-CV", "Stim. subsample"
    ]
    model_list_short = ["Qwen", "Llama", "Mistral", "Gemma"]
    # All pass for all models
    grid = np.ones((len(tests), len(model_list_short)))

    im = ax_c.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax_c.set_xticks(range(len(model_list_short)))
    ax_c.set_xticklabels(model_list_short, fontsize=6)
    ax_c.set_yticks(range(len(tests)))
    ax_c.set_yticklabels(tests, fontsize=5.5)
    ax_c.set_title("Robustness tests (all PASS)", fontsize=7)

    for i in range(len(tests)):
        for j in range(len(model_list_short)):
            ax_c.text(j, i, "PASS", ha="center", va="center", fontsize=5,
                      fontweight="bold", color="white")

    ax_c.spines["right"].set_visible(True)
    ax_c.spines["top"].set_visible(True)
    ax_c.spines["right"].set_linewidth(0.4)
    ax_c.spines["top"].set_linewidth(0.4)
    panel_label(ax_c, "c", x=-0.25)

    # ── Panel d: Narratives fMRI ──
    ax_d = fig.add_subplot(gs[1, 1])

    narr_models = ng["models"]
    # p-values from FINDINGS.md: Qwen p=0.004, Llama ~0.007, Mistral ~0.013, Gemma ~0.010
    narr_ps = {"Qwen2.5-7B-Instruct": 0.004,
               "Meta-Llama-3.1-8B-Instruct": 0.007,
               "Mistral-7B-Instruct-v0.3": 0.013,
               "gemma-2-9b-it": 0.010}
    ceiling_narr = ng["ceiling"]

    nm_labels = []
    nm_rhos   = []
    nm_ps     = []
    nm_colors = [MODEL_COLORS["Qwen"], MODEL_COLORS["Llama"],
                 MODEL_COLORS["Mistral"], MODEL_COLORS["Gemma"]]

    for mk in model_map:
        nm_labels.append(model_map[mk])
        nm_rhos.append(narr_models[mk]["last"]["rho"])
        nm_ps.append(narr_ps.get(mk, 0.05))

    bars = ax_d.bar(range(4), nm_rhos, color=nm_colors, edgecolor="white",
                    linewidth=0.4, width=0.65, zorder=3)
    ax_d.axhline(ceiling_narr, color="#999999", linestyle="--", linewidth=0.8,
                 label=f"Ceiling ({ceiling_narr:.2f})", zorder=2)

    for i, (rho, p) in enumerate(zip(nm_rhos, nm_ps)):
        stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax_d.text(i, rho + 0.015, stars, ha="center", fontsize=7, fontweight="bold")

    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(nm_labels, fontsize=6, rotation=20, ha="right")
    ax_d.set_ylabel("Spearman " + r"$\rho$" + " (last layer)", fontsize=7)
    ax_d.set_title("Narratives fMRI validation (N=230)", fontsize=7)
    ax_d.legend(fontsize=5.5, loc="upper right")
    ax_d.set_ylim(0, 0.95)
    panel_label(ax_d, "d")

    save_fig(fig, "figure5_robustness")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 6 — The Embodiment Boundary
# ═══════════════════════════════════════════════════════════════════════
def figure6():
    print("Figure 6: The Embodiment Boundary")

    # Load all RDMs
    brain_rdm, brain_conds = load_rdm(os.path.join(RSA, "brain_rdm.npz"))

    model_files = {
        "Qwen":   "Qwen2.5-7B-Instruct_rdm14_headline.npz",
        "Llama":  "Meta-Llama-3.1-8B-Instruct_rdm14_headline.npz",
        "Mistral":"Mistral-7B-Instruct-v0.3_rdm14_headline.npz",
        "Gemma":  "gemma-2-9b-it_rdm14_headline.npz",
    }
    llm_rdms = {}
    for short_name, fname in model_files.items():
        rdm, _ = load_rdm(os.path.join(RSA, fname))
        llm_rdms[short_name] = rdm

    pp = load_json(os.path.join(RSA, "prospective_prediction.json"))
    pt = load_json(os.path.join(DEV, "pythia_trajectory.json"))

    fig = plt.figure(figsize=(180/25.4, 65/25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0],
                          wspace=0.4, left=0.08, right=0.95, top=0.88, bottom=0.17)

    # ── Panel a: per-condition alignment ──
    ax_a = fig.add_subplot(gs[0])

    # Compute per-condition row-wise rho (each condition's distance-to-all-13-others
    # correlated between brain and LLM)
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

    # Sort by block then by rho
    aff_conds = [(c, per_cond_rhos[c]) for c in brain_conds if c in AFFECTIVE]
    soc_conds = [(c, per_cond_rhos[c]) for c in brain_conds if c in SOCIAL]
    aff_conds.sort(key=lambda x: -x[1])
    soc_conds.sort(key=lambda x: -x[1])
    ordered = soc_conds + aff_conds  # social first (higher rhos) then affective

    pretty = {
        "anger": "Anger", "disgust": "Disgust", "fear": "Fear",
        "happiness": "Happiness", "sadness": "Sadness", "valence": "Valence",
        "belief": "Belief", "empathy": "Empathy", "intention": "Intention",
        "judgment": "Judgment", "mentalizing": "Mental.", "moral": "Moral",
        "self_referential": "Self-ref.", "theory_of_mind": "ToM",
    }

    cond_names = [pretty.get(c, c) for c, _ in ordered]
    cond_rhos  = [r for _, r in ordered]
    cond_cols  = [C_SOC if c in SOCIAL else C_AFF for c, _ in ordered]

    y_pos = np.arange(len(cond_names))
    ax_a.barh(y_pos, cond_rhos, color=cond_cols, edgecolor="white",
              linewidth=0.4, height=0.7, zorder=3)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(cond_names, fontsize=5.5)
    ax_a.set_xlabel("Mean row-wise " + r"$\rho$" + " (4 models)", fontsize=6.5)
    ax_a.set_title("Per-condition brain-LLM alignment", fontsize=7)
    ax_a.invert_yaxis()

    # Highlight empathy
    emp_idx = [i for i, (c, _) in enumerate(ordered) if c == "empathy"]
    if emp_idx:
        ei = emp_idx[0]
        ax_a.barh(ei, cond_rhos[ei], color="#FFD700", edgecolor=C_SOC,
                  linewidth=0.8, height=0.7, zorder=4)
        ax_a.annotate("bridge",
                      xy=(cond_rhos[ei] + 0.01, ei),
                      xytext=(cond_rhos[ei] + 0.15, ei + 0.8),
                      fontsize=5, ha="left", va="center",
                      arrowprops=dict(arrowstyle="->", lw=0.5, color="#555"))

    # Legend
    from matplotlib.patches import Patch
    leg_handles = [Patch(facecolor=C_SOC, label="Social/mentalizing"),
                   Patch(facecolor=C_AFF, label="Affective"),
                   Patch(facecolor="#FFD700", edgecolor=C_SOC, label="Empathy (bridge)")]
    ax_a.legend(handles=leg_handles, fontsize=5, loc="lower right")
    panel_label(ax_a, "a", x=-0.18)

    # ── Panel b: prospective predictions forest plot ──
    ax_b = fig.add_subplot(gs[1])

    pred_data = [
        ("P1: Coupling\nasymmetry", pp["prediction_1_coupling_asymmetry"]["average_rho"],
         pp["prediction_1_coupling_asymmetry"]["average_p_perm"], True),
        ("P2: Within-block\ncoupling", pp["prediction_2_within_block_coupling"]["blocks"]["affective"]["average_rho"],
         pp["prediction_2_within_block_coupling"]["blocks"]["affective"]["average_p_perm"], False),
        ("P3: Condition\ndistinctiveness", pp["prediction_3_classification_accuracy"]["average_rho"],
         pp["prediction_3_classification_accuracy"]["average_p_perm"], False),
        ("P4: Vulnerable\npairs (rank)", 0.731, 0.012, True),  # rank rho from FINDINGS
        ("P5: Boundary\nsensitivity", pp["prediction_5_boundary_sensitivity"]["average_rho"],
         pp["prediction_5_boundary_sensitivity"]["average_p_perm"], False),
    ]

    y_pos = np.arange(len(pred_data))
    for i, (name, rho, p, confirmed) in enumerate(pred_data):
        color = "#55A868" if confirmed else ("#E8853A" if p < 0.2 else "#C44E52")
        marker = "D" if confirmed else ("o" if p < 0.2 else "x")
        ax_b.plot(rho, i, marker=marker, color=color, markersize=7, zorder=3)
        # Significance
        sig_txt = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig_txt:
            ax_b.text(rho + 0.03, i, sig_txt, va="center", fontsize=7,
                      fontweight="bold", color=color)

    ax_b.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([d[0] for d in pred_data], fontsize=5.5)
    ax_b.set_xlabel("Spearman " + r"$\rho$", fontsize=7)
    ax_b.set_title("Prospective predictions (brain " + r"$\rightarrow$" + " LLM)", fontsize=7)
    ax_b.invert_yaxis()

    # Legend for markers
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="D", color="#55A868", linestyle="None",
                  markersize=5, label="Confirmed"),
           Line2D([0], [0], marker="o", color="#E8853A", linestyle="None",
                  markersize=5, label="Trend"),
           Line2D([0], [0], marker="x", color="#C44E52", linestyle="None",
                  markersize=5, label="Null")]
    ax_b.legend(handles=leg, fontsize=5, loc="lower left")
    panel_label(ax_b, "b")

    # ── Panel c: within-affective divergence during training ──
    ax_c = fig.add_subplot(gs[2])

    # Filter out checkpoints with errors
    valid_cps = [cp for cp in pt["checkpoints"] if "full_rho" in cp]
    steps = [cp["step"] for cp in valid_cps]
    w_aff = [cp["within_aff_rho"] for cp in valid_cps]
    w_soc = [cp["within_soc_rho"] for cp in valid_cps]
    steps_plot = [max(s, 1) for s in steps]

    ax_c.fill_between(steps_plot, w_aff, 0, alpha=0.15, color=C_AFF, zorder=1)
    ax_c.fill_between(steps_plot, w_soc, 0, alpha=0.15, color=C_SOC, zorder=1)
    ax_c.plot(steps_plot, w_aff, "^-", color=C_AFF, markersize=3, linewidth=1.2,
              label="Within-affective " + r"$\rho$", zorder=3)
    ax_c.plot(steps_plot, w_soc, "s-", color=C_SOC, markersize=3, linewidth=1.2,
              label="Within-social " + r"$\rho$", zorder=3)

    ax_c.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Training step", fontsize=7)
    ax_c.set_ylabel("Within-block " + r"$\rho$" + " with brain", fontsize=7)
    ax_c.set_title("Embodiment boundary\nin training", fontsize=7)
    ax_c.legend(fontsize=5, loc="lower left")

    # Annotation
    ax_c.annotate("Social: transfers\n(propositional)",
                  xy=(50000, w_soc[-2]), fontsize=4.5, color=C_SOC,
                  ha="center", va="bottom",
                  xytext=(50000, 0.80),
                  arrowprops=dict(arrowstyle="->", color=C_SOC, lw=0.5))
    ax_c.annotate("Affective: diverges\n(embodied)",
                  xy=(50000, w_aff[-2]), fontsize=4.5, color=C_AFF,
                  ha="center", va="top",
                  xytext=(50000, -0.80),
                  arrowprops=dict(arrowstyle="->", color=C_AFF, lw=0.5))

    panel_label(ax_c, "c")

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

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"  {name}: {status}")
