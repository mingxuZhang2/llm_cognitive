#!/usr/bin/env python
"""
Per-pair consistency/inconsistency analysis for the brain-LLM RSA headline.

For each of the 91 upper-triangle pairs in the 14x14 RDM, compare brain and
LLM distances (rank-based). Identify which pairs are most/least consistent,
and break down by pair type (within-affective, within-social, cross-block).

Outputs:
  results/cognitive_rsa/pairwise_consistency.json   — full numeric results
  figures/pairwise_scatter.png                      — brain vs LLM distance scatter
  figures/residual_heatmap.png                      — signed residual heatmap

Run on the login node (tiny 14x14 matrices).
"""

import json, os
import numpy as np
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
RSA_DIR  = os.path.join(BASE, "..", "results", "cognitive_rsa")
FIG_DIR  = os.path.join(BASE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── model map ──────────────────────────────────────────────────────────────
MODELS = {
    "Qwen2.5-7B":   "Qwen2.5-7B-Instruct_rdm14_headline.npz",
    "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct_rdm14_headline.npz",
    "Mistral-7B":   "Mistral-7B-Instruct-v0.3_rdm14_headline.npz",
    "Gemma-2-9B":   "gemma-2-9b-it_rdm14_headline.npz",
}

# ── condition grouping ─────────────────────────────────────────────────────
AFFECTIVE = {"anger", "disgust", "fear", "happiness", "sadness", "valence"}
# everything else is the social/mentalistic block

# display order: affective first (alphabetical), then social (alphabetical)
DISPLAY_ORDER = [
    "anger", "disgust", "fear", "happiness", "sadness", "valence",
    "belief", "empathy", "intention", "judgment", "mentalizing",
    "moral", "self_referential", "theory_of_mind",
]


def load_rdm(path):
    """Load RDM, return (matrix, list-of-condition-strings)."""
    d = np.load(path, allow_pickle=True)
    return d["rdm"].astype(float), list(d["conditions"])


def reorder_rdm(rdm, conds, target_order):
    """Reorder RDM rows/cols from `conds` order to `target_order`."""
    idx = [conds.index(c) for c in target_order]
    return rdm[np.ix_(idx, idx)]


def utri_indices(n):
    return np.triu_indices(n, k=1)


def pair_type(c1, c2):
    a1 = c1 in AFFECTIVE
    a2 = c2 in AFFECTIVE
    if a1 and a2:
        return "within-affective"
    if (not a1) and (not a2):
        return "within-social"
    return "cross-block"


# ── load data ──────────────────────────────────────────────────────────────
brain_rdm_raw, brain_conds = load_rdm(os.path.join(RSA_DIR, "brain_rdm.npz"))
brain_rdm = reorder_rdm(brain_rdm_raw, brain_conds, DISPLAY_ORDER)

llm_rdms = {}
for short, fname in MODELS.items():
    raw, conds = load_rdm(os.path.join(RSA_DIR, fname))
    llm_rdms[short] = reorder_rdm(raw, conds, DISPLAY_ORDER)

n = len(DISPLAY_ORDER)
iu = utri_indices(n)
n_pairs = len(iu[0])  # 91

# ── mean LLM RDM ──────────────────────────────────────────────────────────
mean_llm_rdm = np.mean(list(llm_rdms.values()), axis=0)

# ── extract upper-triangle vectors ────────────────────────────────────────
brain_vec   = brain_rdm[iu]
mean_llm_vec = mean_llm_rdm[iu]

# per-model vectors
model_vecs = {m: llm_rdms[m][iu] for m in MODELS}

# ── rank-based signed residuals ───────────────────────────────────────────
brain_ranks    = rankdata(brain_vec)
mean_llm_ranks = rankdata(mean_llm_vec)
signed_residuals = brain_ranks - mean_llm_ranks  # positive = brain thinks more distant

# ── pair labels & types ───────────────────────────────────────────────────
pair_labels = []
pair_types  = []
for k in range(n_pairs):
    i, j = iu[0][k], iu[1][k]
    c1, c2 = DISPLAY_ORDER[i], DISPLAY_ORDER[j]
    pair_labels.append(f"{c1} - {c2}")
    pair_types.append(pair_type(c1, c2))

pair_types_arr = np.array(pair_types)

# ── per-type statistics ───────────────────────────────────────────────────
type_stats = {}
for t in ["within-affective", "within-social", "cross-block"]:
    mask = pair_types_arr == t
    cnt = int(mask.sum())
    abs_res = np.abs(signed_residuals[mask])
    if cnt >= 3:
        rho, pval = spearmanr(brain_vec[mask], mean_llm_vec[mask])
    else:
        rho, pval = float("nan"), float("nan")
    type_stats[t] = {
        "count": cnt,
        "mean_abs_residual": round(float(abs_res.mean()), 3),
        "median_abs_residual": round(float(np.median(abs_res)), 3),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(pval), 6),
    }

# ── overall stats ─────────────────────────────────────────────────────────
overall_rho, overall_p = spearmanr(brain_vec, mean_llm_vec)
overall_stats = {
    "n_pairs": n_pairs,
    "overall_spearman_rho": round(float(overall_rho), 4),
    "overall_spearman_p": float(f"{overall_p:.2e}"),
    "mean_abs_residual": round(float(np.abs(signed_residuals).mean()), 3),
    "median_abs_residual": round(float(np.median(np.abs(signed_residuals))), 3),
}

# ── top-5 consistent / inconsistent ──────────────────────────────────────
abs_res = np.abs(signed_residuals)
idx_sorted = np.argsort(abs_res)

top5_consistent = []
for k in idx_sorted[:5]:
    top5_consistent.append({
        "pair": pair_labels[k],
        "type": pair_types[k],
        "brain_dist": round(float(brain_vec[k]), 4),
        "mean_llm_dist": round(float(mean_llm_vec[k]), 4),
        "signed_residual": round(float(signed_residuals[k]), 2),
        "abs_residual": round(float(abs_res[k]), 2),
    })

top5_inconsistent = []
for k in idx_sorted[-5:][::-1]:  # largest first
    top5_inconsistent.append({
        "pair": pair_labels[k],
        "type": pair_types[k],
        "brain_dist": round(float(brain_vec[k]), 4),
        "mean_llm_dist": round(float(mean_llm_vec[k]), 4),
        "signed_residual": round(float(signed_residuals[k]), 2),
        "abs_residual": round(float(abs_res[k]), 2),
    })

# ── per-model rho for reference ───────────────────────────────────────────
per_model_rho = {}
for m in MODELS:
    rho_m, _ = spearmanr(brain_vec, model_vecs[m])
    per_model_rho[m] = round(float(rho_m), 4)

# ── full pair table (for JSON) ────────────────────────────────────────────
pair_table = []
for k in range(n_pairs):
    entry = {
        "pair": pair_labels[k],
        "type": pair_types[k],
        "brain_dist": round(float(brain_vec[k]), 4),
        "mean_llm_dist": round(float(mean_llm_vec[k]), 4),
        "signed_residual": round(float(signed_residuals[k]), 2),
        "abs_residual": round(float(abs_res[k]), 2),
    }
    for m in MODELS:
        entry[f"{m}_dist"] = round(float(model_vecs[m][k]), 4)
    pair_table.append(entry)

# ── save JSON ─────────────────────────────────────────────────────────────
results = {
    "overall": overall_stats,
    "per_type": type_stats,
    "per_model_rho": per_model_rho,
    "top5_consistent": top5_consistent,
    "top5_inconsistent": top5_inconsistent,
    "all_pairs": pair_table,
}

out_json = os.path.join(RSA_DIR, "pairwise_consistency.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_json}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: pairwise scatter (brain dist vs mean-LLM dist)
# ══════════════════════════════════════════════════════════════════════════
COLOR_MAP = {
    "within-affective": "#d62728",  # red
    "within-social":    "#1f77b4",  # blue
    "cross-block":      "#999999",  # gray
}

fig, ax = plt.subplots(figsize=(7, 6))
for t in ["cross-block", "within-social", "within-affective"]:
    mask = pair_types_arr == t
    ax.scatter(brain_vec[mask], mean_llm_vec[mask],
               c=COLOR_MAP[t], label=t, alpha=0.65, s=32, edgecolors="white",
               linewidth=0.4)

# regression line
m_slope, b_int = np.polyfit(brain_vec, mean_llm_vec, 1)
xs = np.linspace(brain_vec.min(), brain_vec.max(), 100)
ax.plot(xs, m_slope * xs + b_int, "k--", lw=1, alpha=0.5)
ax.annotate(f"Spearman $\\rho$ = {overall_rho:.3f}",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=11, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# label top-5 inconsistent
top5_idx = idx_sorted[-5:]
for k in top5_idx:
    short_label = pair_labels[k].replace("_", " ").replace(" - ", "\n")
    ax.annotate(short_label,
                xy=(brain_vec[k], mean_llm_vec[k]),
                fontsize=6.5, ha="center", va="bottom",
                textcoords="offset points", xytext=(0, 6),
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"),
                bbox=dict(boxstyle="round,pad=0.15", fc="lightyellow",
                          ec="0.7", lw=0.4))

ax.set_xlabel("Brain distance (1 - Pearson)", fontsize=11)
ax.set_ylabel("Mean LLM distance (1 - cosine)", fontsize=11)
ax.set_title("Brain vs LLM pairwise distances (91 pairs)", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
scatter_path = os.path.join(FIG_DIR, "pairwise_scatter.png")
fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {scatter_path}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: signed-residual heatmap (14x14)
# ══════════════════════════════════════════════════════════════════════════
res_mat = np.full((n, n), np.nan)
for k in range(n_pairs):
    i, j = iu[0][k], iu[1][k]
    res_mat[i, j] = signed_residuals[k]
    res_mat[j, i] = signed_residuals[k]  # symmetric for display

# short display labels
short_labels = [c.replace("_", " ").replace("self referential", "self-ref")
                  .replace("theory of mind", "ToM") for c in DISPLAY_ORDER]

fig, ax = plt.subplots(figsize=(9, 8))
vmax = np.nanmax(np.abs(signed_residuals))
im = ax.imshow(res_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

# axis labels
ax.set_xticks(range(n))
ax.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=8.5)
ax.set_yticks(range(n))
ax.set_yticklabels(short_labels, fontsize=8.5)

# block separator lines
ax.axhline(5.5, color="k", lw=1.2)
ax.axvline(5.5, color="k", lw=1.2)

# annotate top-5 outlier cells with value
top5_idx_set = set(idx_sorted[-5:].tolist())
for k in range(n_pairs):
    if k in top5_idx_set:
        i, j = iu[0][k], iu[1][k]
        val = signed_residuals[k]
        ax.text(j, i, f"{val:+.0f}", ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white" if abs(val) > vmax * 0.6 else "black")

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Signed residual  (rank_brain - rank_LLM)\n"
               "Red = brain more distant  |  Blue = LLM more distant",
               fontsize=9)

ax.set_title("Rank residuals: brain vs mean-LLM distance", fontsize=12)
fig.tight_layout()
heatmap_path = os.path.join(FIG_DIR, "residual_heatmap.png")
fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {heatmap_path}")

# ══════════════════════════════════════════════════════════════════════════
# STDOUT SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PAIRWISE CONSISTENCY ANALYSIS — BRAIN vs MEAN-LLM (91 pairs)")
print("=" * 70)

print(f"\nOverall Spearman rho:  {overall_rho:.4f}  (p = {overall_p:.2e})")
print(f"Mean |residual|:       {overall_stats['mean_abs_residual']:.3f}")
print(f"Median |residual|:     {overall_stats['median_abs_residual']:.3f}")

print(f"\nPer-model rho (each vs brain):")
for m, rho in per_model_rho.items():
    print(f"  {m:18s}  rho = {rho:.4f}")

print(f"\n{'Type':<20s} {'N':>4s} {'Mean|res|':>10s} {'Med|res|':>10s} "
      f"{'rho':>8s} {'p':>10s}")
print("-" * 66)
for t in ["within-affective", "within-social", "cross-block"]:
    s = type_stats[t]
    print(f"{t:<20s} {s['count']:>4d} {s['mean_abs_residual']:>10.3f} "
          f"{s['median_abs_residual']:>10.3f} {s['spearman_rho']:>8.4f} "
          f"{s['spearman_p']:>10.6f}")

print(f"\nTop-5 MOST CONSISTENT pairs (smallest |residual|):")
for entry in top5_consistent:
    print(f"  {entry['pair']:<40s}  type={entry['type']:<18s}  "
          f"resid={entry['signed_residual']:+6.1f}  "
          f"brain={entry['brain_dist']:.4f}  llm={entry['mean_llm_dist']:.4f}")

print(f"\nTop-5 MOST INCONSISTENT pairs (largest |residual|):")
for entry in top5_inconsistent:
    print(f"  {entry['pair']:<40s}  type={entry['type']:<18s}  "
          f"resid={entry['signed_residual']:+6.1f}  "
          f"brain={entry['brain_dist']:.4f}  llm={entry['mean_llm_dist']:.4f}")

print("\nDone.")
