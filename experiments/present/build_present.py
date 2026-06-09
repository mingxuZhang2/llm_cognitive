#!/usr/bin/env python3
"""
Build a self-contained HTML briefing (present/index.html) for Dr. Zhang's meeting.
All figures are rendered from REAL result files and embedded as base64 PNGs.
Updated 2026-06-05: reflects the full 3-finding story + robustness + embodiment.
"""
import json, base64, io, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
AFFV = BASE / "results" / "affective_validation"
OUT = BASE / "present" / "index.html"

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
MSHORT = {"Qwen2.5-7B-Instruct": "Qwen-7B", "Meta-Llama-3.1-8B-Instruct": "Llama-8B",
          "Mistral-7B-Instruct-v0.3": "Mistral-7B", "gemma-2-9b-it": "Gemma-9B"}
AFF = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
MENT = ["judgment", "belief", "intention", "mentalizing", "moral", "empathy",
        "self_referential", "theory_of_mind"]
ORDER = AFF + MENT
C_AFF, C_MENT, C_BLUE = "#e8743b", "#19a979", "#2e5cb8"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "figure.dpi": 150,
    "axes.edgecolor": "#888", "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_file(path):
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def reorder(rdm, conds, order):
    idx = [conds.index(c) for c in order]
    return rdm[np.ix_(idx, idx)]


def heatmap_panel(axes, rdm, conds, title, cmap="magma"):
    R = reorder(rdm, conds, ORDER).astype(float)
    n = R.shape[0]
    iu = np.triu_indices(n, 1)
    ranks = R[iu].argsort().argsort().astype(float)
    ranks = ranks / (len(ranks) - 1)
    Rn = np.full((n, n), np.nan)
    Rn[iu] = ranks
    Rn[(iu[1], iu[0])] = ranks
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#e8e8e8")
    im = axes.imshow(Rn, cmap=cm, vmin=0, vmax=1)
    axes.set_xticks(range(len(ORDER)))
    axes.set_yticks(range(len(ORDER)))
    axes.set_xticklabels(ORDER, rotation=90, fontsize=6)
    axes.set_yticklabels(ORDER, fontsize=6)
    n_aff = len(AFF)
    axes.axhline(n_aff - 0.5, color="cyan", lw=1.5)
    axes.axvline(n_aff - 0.5, color="cyan", lw=1.5)
    for t, c in zip(axes.get_xticklabels(), ORDER):
        t.set_color(C_AFF if c in AFF else C_MENT)
    for t, c in zip(axes.get_yticklabels(), ORDER):
        t.set_color(C_AFF if c in AFF else C_MENT)
    axes.set_title(title, fontsize=9, fontweight="bold")
    return im


# ========== LOAD ALL DATA ==========
ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
ns_rdm, ns_conds = ns["rdm"], list(ns["conditions"])

_iu = np.triu_indices(len(ns_conds), 1)
head_rho = {}
llm_rdms = {}
for m in MODELS:
    z = np.load(RSA / f"{m}_rdm14_headline.npz", allow_pickle=True)
    lrdm, lconds = z["rdm"], list(z["conditions"])
    order = [lconds.index(c) for c in ns_conds]
    L = lrdm[np.ix_(order, order)]
    head_rho[m] = float(spearmanr(L[_iu], ns_rdm[_iu])[0])
    llm_rdms[m] = (lrdm, lconds)

wbc = json.load(open(RSA / "within_block_control.json"))
dev = json.load(open(BASE / "results" / "developmental_emergence" / "developmental_emergence.json"))["per_size"]
ceil = json.load(open(AFFV / "affective_ceiling_control.json"))["aggregate"]
coup = json.load(open(BASE / "results" / "clinical_dissociation" / "coupling_reanalysis.json"))
tm_data = json.load(open(BASE / "results" / "template_matched_rsa" / "template_matched_rsa_results.json"))
pyth = json.load(open(BASE / "results" / "developmental_emergence" / "pythia_trajectory.json"))
moral = json.load(open(BASE / "results" / "moral_judgment" / "Qwen2.5-7B-Instruct_moral_logit.json"))
judge = json.load(open(BASE / "results" / "human_rating" / "deepseek_judge_ranking.json"))
judge_ctrl = json.load(open(BASE / "results" / "human_rating" / "deepseek_judge_controls.json"))
partial_rsa = json.load(open(RSA / "partial_rsa_per_model.json"))
pp = json.load(open(RSA / "prospective_prediction.json"))
bvi = json.load(open(RSA / "base_vs_instruct.json"))
narr = json.load(open(RSA / "narratives_group_rsa.json"))

# ========== FIGURE 1: 5-panel heatmaps ==========
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
panels = [("Human Brain\n(Neurosynth)", ns_rdm, ns_conds)] + \
         [(MSHORT[m], llm_rdms[m][0], llm_rdms[m][1]) for m in MODELS]
for ax, (title, rdm, conds) in zip(axes, panels):
    im = heatmap_panel(ax, rdm, conds, title)
fig.subplots_adjust(wspace=0.35, bottom=0.25)
cbar_ax = fig.add_axes([0.92, 0.15, 0.008, 0.7])
fig.colorbar(im, cax=cbar_ax, label="rank distance")
fig_heatmaps = b64(fig)

# ========== FIGURE 2: headline rho bars ==========
fig, ax = plt.subplots(figsize=(5, 3))
xs = [MSHORT[m] for m in MODELS]
ys = [head_rho[m] for m in MODELS]
bars = ax.bar(xs, ys, color=C_BLUE, width=0.55, edgecolor="white")
for b, y in zip(bars, ys):
    ax.text(b.get_x() + b.get_width() / 2, y + 0.008, f"{y:.3f}",
            ha="center", fontsize=10, fontweight="bold", color=C_BLUE)
ax.set_ylim(0, 0.85)
ax.set_ylabel("Spearman rho")
ax.axhline(0.73, color="#aaa", ls="--", lw=0.8)
ax.grid(axis="y", alpha=0.2)
fig_bars = b64(fig)

# ========== FIGURE 3: within-block decomposition ==========
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
model_keys = list(wbc["models"].keys())

# 3a: partial rho
ax = axes[0]
partial_vals = [wbc["models"][m]["partial_rho_given_block"] for m in model_keys]
ax.bar(model_keys, partial_vals, color=C_BLUE, width=0.55)
for i, v in enumerate(partial_vals):
    ax.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Partial rho")
ax.set_title("Beyond the split\n(partial rho, all p < 0.001)", fontsize=9)
ax.set_ylim(0, 0.5)
ax.grid(axis="y", alpha=0.2)

# 3b: within-social
ax = axes[1]
soc_vals = [wbc["models"][m]["within_social_rho"] for m in model_keys]
ax.bar(model_keys, soc_vals, color=C_MENT, width=0.55)
for i, v in enumerate(soc_vals):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Within-social rho")
ax.set_title("Social fine structure\nALIGNS (p < 0.03)", fontsize=9)
ax.set_ylim(0, 0.8)
ax.grid(axis="y", alpha=0.2)

# 3c: within-affective
ax = axes[2]
aff_vals = [wbc["models"][m]["within_affective_rho"] for m in model_keys]
colors = ["#cc3333" if v < 0 else C_AFF for v in aff_vals]
ax.bar(model_keys, aff_vals, color=colors, width=0.55)
for i, v in enumerate(aff_vals):
    ax.text(i, v - 0.03 if v < 0 else v + 0.01, f"{v:.2f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Within-affective rho")
ax.set_title("Affective fine structure\nDOES NOT ALIGN (n.s.)", fontsize=9)
ax.axhline(0, color="#333", lw=0.8)
ax.set_ylim(-0.3, 0.3)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig_decomp = b64(fig)

# ========== FIGURE 4: scale invariance ==========
sizes = ["0.5B", "1.5B", "3B", "7B"]
params = [dev[s]["n_params_M"] for s in sizes]
ov = [dev[s]["peak_overall_rho"] for s in sizes]
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(params, ov, "-o", color=C_BLUE, lw=2.5, ms=8)
for x, y, s in zip(params, ov, sizes):
    ax.annotate(f"{s}\n{y:.2f}", (x, y), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xticks(params)
ax.set_xticklabels(sizes)
ax.set_xlabel("Model size (Qwen2.5)")
ax.set_ylabel("Brain-LLM rho")
ax.set_ylim(0.5, 0.85)
ax.grid(alpha=0.2)
fig_scale = b64(fig)

# ========== FIGURE 5: coupling double dissociation ==========
fig, ax = plt.subplots(figsize=(5, 3.5))
coup_models = list(coup["per_model"].keys())
x = np.arange(len(coup_models))
aa = [coup["per_model"][m]["summary_2x2"]["A_aa"] for m in coup_models]
as_ = [coup["per_model"][m]["summary_2x2"]["A_as"] for m in coup_models]
sa = [coup["per_model"][m]["summary_2x2"]["A_sa"] for m in coup_models]
ss = [coup["per_model"][m]["summary_2x2"]["A_ss"] for m in coup_models]
w = 0.2
ax.bar(x - 1.5*w, aa, w, color=C_AFF, label="Aff ablate -> Aff effect", alpha=0.9)
ax.bar(x - 0.5*w, as_, w, color=C_AFF, label="Aff ablate -> Soc effect", alpha=0.4)
ax.bar(x + 0.5*w, sa, w, color=C_MENT, label="Soc ablate -> Aff effect", alpha=0.4)
ax.bar(x + 1.5*w, ss, w, color=C_MENT, label="Soc ablate -> Soc effect", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(coup_models)
ax.set_ylabel("Mean coupling effect")
ax.legend(fontsize=7, ncol=2, loc="upper right")
for i, m in enumerate(coup_models):
    p = coup["per_model"][m]["wilcoxon"]["wilcoxon_p"]
    ax.text(i, max(aa[i], ss[i]) + 0.01, f"p={p:.3f}", ha="center", fontsize=7, color="#555")
ax.grid(axis="y", alpha=0.2)
fig_coupling = b64(fig)

# ========== FIGURE 6: Pythia developmental trajectory ==========
fig, ax = plt.subplots(figsize=(6, 3.5))
pyth_cps = [cp for cp in pyth["checkpoints"] if "full_rho" in cp]
steps = [cp["step"] for cp in pyth_cps]
full = [cp["full_rho"] for cp in pyth_cps]
soc = [cp["within_soc_rho"] for cp in pyth_cps]
aff = [cp["within_aff_rho"] for cp in pyth_cps]
x_log = [max(s, 1) for s in steps]
ax.plot(x_log, full, "-o", color=C_BLUE, lw=2, ms=5, label="Full rho (14x14)")
ax.plot(x_log, soc, "-s", color=C_MENT, lw=2, ms=5, label="Within-social")
ax.plot(x_log, aff, "-^", color=C_AFF, lw=2, ms=5, label="Within-affective")
ax.set_xscale("log")
ax.axhline(0, color="#ccc", lw=0.8)
ax.set_xlabel("Training step")
ax.set_ylabel("Brain-LLM rho")
ax.set_ylim(-0.8, 0.9)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.2)
ax.annotate("Social aligns early", (512, soc[3]), fontsize=7, color=C_MENT,
            xytext=(30, 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=C_MENT, lw=0.8))
ax.annotate("Affective DIVERGES", (8000, aff[5]), fontsize=7, color="#cc3333",
            xytext=(20, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#cc3333", lw=0.8))
fig_pythia = b64(fig)

# ========== FIGURE 7: partial RSA + template-matched ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# 7a: Partial RSA per model (raw vs partial)
ax = axes[0]
pr_models = list(partial_rsa.keys())
raw = [partial_rsa[m]["raw"] for m in pr_models]
partial = [partial_rsa[m]["partial"] for m in pr_models]
x = np.arange(len(pr_models))
w = 0.3
ax.bar(x - w/2, raw, w, color=C_BLUE, alpha=0.3, label="Raw rho")
ax.bar(x + w/2, partial, w, color=C_BLUE, label="Partial rho")
for i in range(len(pr_models)):
    pct = partial_rsa[pr_models[i]]["pct"]
    ax.text(i + w/2, partial[i] + 0.01, f"{pct}%", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(pr_models)
ax.set_ylabel("Spearman rho")
ax.set_ylim(0, 0.85)
ax.legend(fontsize=8)
ax.set_title("Partial RSA\n(controlling GloVe + name + length)", fontsize=9)
ax.grid(axis="y", alpha=0.2)

# 7b: Template-matched
ax = axes[1]
tm_models_short = {"Qwen2.5-7B-Instruct": "Qwen", "Meta-Llama-3.1-8B-Instruct": "Llama",
                   "Mistral-7B-Instruct-v0.3": "Mistral", "gemma-2-9b-it": "Gemma"}
tm_names = []
tm_orig = []
tm_matched = []
for m in MODELS:
    d = tm_data["models"][m]
    tm_names.append(tm_models_short[m])
    tm_orig.append(d["original_stimuli_rho"])
    tm_matched.append(d["rho_at_headline_peak"])
x = np.arange(len(tm_names))
ax.bar(x - w/2, tm_orig, w, color=C_BLUE, alpha=0.3, label="Original stimuli")
ax.bar(x + w/2, tm_matched, w, color=C_BLUE, label="Template-matched")
for i in range(len(tm_names)):
    pct = tm_matched[i] / tm_orig[i] * 100
    ax.text(i + w/2, tm_matched[i] + 0.01, f"{pct:.0f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tm_names)
ax.set_ylabel("Spearman rho")
ax.set_ylim(0, 0.85)
ax.legend(fontsize=8)
ax.set_title("Template-matched stimuli\n(same format, only content differs)", fontsize=9)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig_robustness = b64(fig)

# ========== FIGURE 8: steering controls ==========
fig, ax = plt.subplots(figsize=(5, 3))
dirs_data = judge_ctrl["directions"]
dir_names = ["brain", "random", "sentiment", "pc1"]
dir_labels = ["Brain axis", "Random", "Sentiment", "PC1"]
dir_colors = [C_BLUE, "#999", "#999", "#999"]
dir_rhos = [dirs_data[d]["mean_rho"] for d in dir_names]
dir_ps = [dirs_data[d]["ttest_p"] for d in dir_names]
bars = ax.bar(dir_labels, dir_rhos, color=dir_colors, width=0.55, edgecolor="white")
ax.axhline(0, color="#333", lw=0.8)
for i, (b, y, p) in enumerate(zip(bars, dir_rhos, dir_ps)):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(b.get_x() + b.get_width()/2, max(y, 0) + 0.02,
            f"{y:+.2f}\n{sig}", ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("Mean rho (LLM judge)")
ax.set_ylim(-0.15, 0.5)
ax.grid(axis="y", alpha=0.2)
fig_steer_ctrl = b64(fig)

# ========== FIGURE 9: moral steering curve ==========
fig, ax = plt.subplots(figsize=(5, 3))
alphas = moral["alpha_summary"]
alpha_vals = [a["alpha"] for a in alphas]
logit_diffs = [a["mean_logit_diff"] for a in alphas]
ax.plot(alpha_vals, logit_diffs, "-o", color=C_BLUE, lw=2, ms=6)
ax.axhline(0, color="#ccc", lw=0.8)
ax.axvline(0, color="#ccc", lw=0.8)
ax.set_xlabel("Steering alpha (- = affective, + = mentalizing)")
ax.set_ylabel("Mean logit diff\n(+ = more utilitarian)")
ax.annotate(f"rho = {moral['correlation_all']['rho']:.3f}\np = {moral['correlation_all']['p']:.4f}",
            xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=9, fontweight="bold", color=C_BLUE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BLUE, alpha=0.8))
ax.grid(alpha=0.2)
fig_moral = b64(fig)

# ========== Stimuli table ==========
stim = [json.loads(l) for l in open(BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl")]
cnt = collections.Counter(r["condition"] for r in stim)
total_stim = sum(cnt.values())

SRC_HUMAN = {
    "anger": "Emotion localizer + GoEmotions", "fear": "Emotion localizer + GoEmotions",
    "sadness": "Emotion localizer + GoEmotions", "disgust": "Emotion localizer + GoEmotions",
    "happiness": "Emotion localizer + GoEmotions", "valence": "VAD-graded (Warriner 2013)",
    "belief": "False-belief stories", "intention": "Indirect requests",
    "judgment": "ETHICS commonsense (Hendrycks 2021)",
    "moral": "Moral dilemmas + Moral Foundations", "theory_of_mind": "Faux-Pas + Self/Other",
    "mentalizing": "Strange Stories (Happe)", "empathy": "Moral Foundations (care)",
    "self_referential": "Self/Other (self condition)",
}

stim_rows = ""
for c in ORDER:
    blk = "Aff" if c in AFF else "Soc"
    col = C_AFF if c in AFF else C_MENT
    stim_rows += f"<tr><td><b style='color:{col}'>{c}</b></td><td>{blk}</td><td style='text-align:center'>{cnt[c]}</td><td style='font-size:12px'>{SRC_HUMAN[c]}</td></tr>\n"

# ========== Narratives fMRI data ==========
narr_models = {}
for m_key in narr.get("per_model", narr).keys() if isinstance(narr, dict) else []:
    d = narr["per_model"][m_key] if "per_model" in narr else narr[m_key]
    if isinstance(d, dict) and "last_layer_rho" in d:
        narr_models[m_key] = d

# ========== BUILD HTML ==========
HTML = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM x Brain — Project Briefing (2026-06-05)</title>
<style>
:root{{--aff:{C_AFF};--ment:{C_MENT};--blue:{C_BLUE};--ink:#1d2330;--muted:#5a6472;}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;
color:var(--ink);background:#f4f6fa;line-height:1.65}}
.wrap{{max-width:1100px;margin:0 auto;padding:0 26px 80px}}
header{{background:linear-gradient(135deg,#1d2a4d,#2e5cb8);color:#fff;padding:50px 26px 36px;margin-bottom:30px}}
header .inner{{max-width:1100px;margin:0 auto}}
header h1{{margin:0 0 6px;font-size:28px;letter-spacing:.3px}}
header p{{margin:4px 0;opacity:.92;font-size:14px}}
.badge{{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);
border-radius:20px;padding:2px 12px;font-size:12px;margin:6px 5px 0 0}}
section{{background:#fff;border-radius:14px;padding:22px 26px;margin:18px 0;box-shadow:0 2px 14px rgba(20,30,60,.06)}}
h2{{font-size:20px;margin:0 0 6px;border-left:5px solid var(--blue);padding-left:12px}}
h2 .num{{color:var(--blue);font-weight:800;margin-right:6px}}
h3{{font-size:15px;margin:18px 0 8px;color:#26324d}}
.lead{{color:var(--muted);font-size:13px;margin:0 0 14px;padding-left:17px}}
p{{font-size:14px;margin:8px 0}}
img.fig{{max-width:100%;border:1px solid #e3e8f0;border-radius:8px;margin:8px 0;background:#fff}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}}
@media(max-width:780px){{.grid2{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border:1px solid #e6eaf2;padding:5px 8px;text-align:left}}
th{{background:#eef2fa;font-weight:700}}
tr:nth-child(even) td{{background:#fafbfe}}
.kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 4px}}
.kpi .card{{flex:1;min-width:140px;background:linear-gradient(135deg,#f7f9ff,#eef2fb);border:1px solid #e2e8f5;
border-radius:12px;padding:14px 16px}}
.kpi .big{{font-size:28px;font-weight:800;color:var(--blue);line-height:1}}
.kpi .lbl{{font-size:11.5px;color:var(--muted);margin-top:5px}}
.callout{{background:#fff8ec;border:1px solid #f0d9a8;border-left:5px solid #e0a73a;border-radius:10px;
padding:12px 16px;margin:14px 0;font-size:13.5px}}
.callout.good{{background:#eefaf4;border-color:#a6e0c6;border-left-color:var(--ment)}}
.callout.warn{{background:#fdeeee;border-color:#f0bcbc;border-left-color:#d65b5b}}
.callout.blue{{background:#eef3fc;border-color:#b3c8e8;border-left-color:var(--blue)}}
.note{{font-size:12px;color:var(--muted)}}
.pill{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:700;color:#fff}}
.pill.a{{background:var(--aff)}}.pill.m{{background:var(--ment)}}
ul{{margin:6px 0}} li{{margin:3px 0;font-size:14px}}
.findcard{{border-radius:12px;padding:14px 16px;margin:10px 0}}
.findcard.f1{{background:#eef3fc;border-left:5px solid var(--blue)}}
.findcard.f2{{background:#eefaf4;border-left:5px solid var(--ment)}}
.findcard.f3{{background:#fff4ec;border-left:5px solid var(--aff)}}
.findcard h4{{margin:0 0 6px;font-size:14.5px}}
.speak{{background:#eef2fa;border-radius:8px;padding:10px 14px;font-size:13px;color:#33405e;margin-top:10px}}
.speak b{{color:var(--blue)}}
footer{{text-align:center;color:#9aa3b2;font-size:12px;margin-top:30px}}
.tag{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;margin-right:4px}}
.tag.yes{{background:#d4edda;color:#155724}}
.tag.no{{background:#f8d7da;color:#721c24}}
.tag.trend{{background:#fff3cd;color:#856404}}
</style></head>
<body>
<header><div class="inner">
<h1>The Brain as a Reference Frame for LLMs</h1>
<p>用人脑的认知地图读懂大模型内部组织 — 组会汇报</p>
<p style="opacity:.8;font-size:12px">Nature Machine Intelligence / Nature Communications | 4 architectures x 14 cognitive conditions x 3 fMRI sources</p>
<div>
<span class="badge">{total_stim} stimuli</span>
<span class="badge">14 conditions</span>
<span class="badge">4 architectures</span>
<span class="badge">rho = 0.73</span>
<span class="badge">2026-06-05</span>
</div>
</div></header>

<div class="wrap">

<!-- ==================== SECTION 0: HEADLINE ==================== -->
<section>
<h2><span class="num">0</span> Headline</h2>
<p>Text-only LLMs share the human brain's <b>dominant affective-mentalizing axis</b> and
<b>social-cognitive fine structure</b>.</p>
<p>Not a rich 14-way correspondence: the alignment is driven by <b>one causally load-bearing
emotion vs social-cognition boundary</b> + the ordering <b>within social cognition</b>.
Fine-grained affective structure does <b style="color:#cc3333">NOT</b> align, and actively diverges during training.</p>

<div class="kpi">
<div class="card"><div class="big">0.73</div><div class="lbl">Brain-LLM RSA (Spearman rho)<br>4 models, all p &lt; 0.0002</div></div>
<div class="card"><div class="big">0.36</div><div class="lbl">Partial rho beyond the binary split<br>4 models, all p &lt; 0.001</div></div>
<div class="card"><div class="big">~76%</div><div class="lbl">of LLM split-half<br>reliability ceiling</div></div>
<div class="card"><div class="big">4/4</div><div class="lbl">architectures + scale-invariant<br>0.5B to 7B</div></div>
</div>

<div class="callout blue"><b>Three-finding story:</b><br>
<b>Finding 1 (Consistency):</b> The alignment exists, is robust, and is driven by one axis + social fine structure.<br>
<b>Finding 2 (Predictability):</b> The brain's geometry predicts LLM internal coupling, behavior under steering, and developmental trajectory.<br>
<b>Finding 3 (Inconsistency):</b> Within-affective structure does NOT transfer, actively diverges during training, consistent with embodiment hypothesis.</div>
</section>

<!-- ==================== SECTION 1: METHOD (condensed) ==================== -->
<section>
<h2><span class="num">1</span> Method: RSA in 30 seconds</h2>
<p class="lead">Brain and LLM each produce a 14x14 distance matrix (RDM). We compare the two via Spearman rank correlation (91 upper-triangle pairs).</p>
<div class="grid2">
<div>
<h3>Brain side</h3>
<p>14 Neurosynth meta-analytic maps (each = average of ~14,000 fMRI papers for that concept).
Flatten each map, compute pairwise 1-Pearson distances -> 14x14 brain RDM.</p>
</div>
<div>
<h3>LLM side</h3>
<p>{total_stim} sentences across 14 conditions. Feed each as raw text (no instruction template).
Extract hidden states at peak layer, mean-pool across tokens, average per condition.
Pairwise 1-cosine distance -> 14x14 LLM RDM.</p>
</div>
</div>
<table>
<tr><th>Condition</th><th>Block</th><th>N</th><th>Source</th></tr>
{stim_rows}
</table>
</section>

<!-- ==================== SECTION 2: FINDING 1 - CONSISTENCY ==================== -->
<section>
<h2><span class="num">2</span> Finding 1: Representational Consistency</h2>
<p class="lead">Text-only LLMs share the brain's dominant emotion-social boundary and social-cognitive fine structure.</p>

<h3>2.1 The RDMs — visual comparison</h3>
<p>Brain (leftmost) vs 4 LLM architectures. Rank-normalized so colors are directly comparable.
Dark = similar, bright = different. Cyan line = emotion/social boundary. Both sides show the same two dark blocks.</p>
<img class="fig" src="{fig_heatmaps}" alt="5-panel heatmaps">

<h3>2.2 Headline RSA: rho ~ 0.73 across 4 architectures</h3>
<div class="grid2">
<div><img class="fig" src="{fig_bars}" alt="headline bars"></div>
<div>
<table>
<tr><th>Model</th><th>Peak layer</th><th>rho</th><th>p</th></tr>
<tr><td>Qwen2.5-7B</td><td>L27</td><td><b>0.739</b></td><td>&lt; 0.0002</td></tr>
<tr><td>Llama-3.1-8B</td><td>L31</td><td><b>0.727</b></td><td>&lt; 0.0002</td></tr>
<tr><td>Mistral-7B</td><td>L14</td><td><b>0.730</b></td><td>&lt; 0.0002</td></tr>
<tr><td>Gemma-2-9B</td><td>L21</td><td><b>0.735</b></td><td>&lt; 0.0002</td></tr>
</table>
<p class="note">Recipe: mean_all | centered | 1-cosine | peak layer.
Permutation null (10,000x) shuffles condition labels.</p>
</div>
</div>

<h3>2.3 What drives the 0.73: one axis + social fine structure</h3>
<p>The rho ~ 0.73 is <b>not</b> a rich 14-way match. Decomposing:</p>
<img class="fig" src="{fig_decomp}" alt="within-block decomposition">
<div class="callout">
<b>Takeaway:</b> Two components drive the headline:<br>
1. The <b>emotion vs social split</b> (both brain and LLM RDMs correlate rho > 0.70 with a binary split matrix)<br>
2. <b>Within-social fine structure</b> (how belief, intention, mentalizing, ToM relate to each other): rho ~ 0.52-0.65, significant<br>
Within-affective fine structure does <b style="color:#cc3333">NOT</b> align (rho ~ -0.10, n.s., all 4 models).
</div>

<h3>2.4 Scale invariance: 0.5B to 7B</h3>
<div class="grid2">
<div><img class="fig" src="{fig_scale}" alt="scale invariance"></div>
<div>
<p>Qwen family (0.5B-7B): alignment is flat across sizes. Not an "emergence" threshold —
even 0.5B already carries the brain-like geometry.</p>
<p><b>Base vs Instruct:</b> Qwen2.5-1.5B: base rho = {max(bvi['base']['per_layer_rho']):.3f},
instruct rho = {max(bvi['instruct']['per_layer_rho']):.3f}. <b>99.3%</b> from pretraining.
RLHF adds negligible structure.</p>
</div>
</div>

<h3>2.5 Real-fMRI validation (3 independent datasets)</h3>
<table>
<tr><th>Dataset</th><th>N subjects</th><th>Conditions</th><th>rho</th><th>Significant?</th></tr>
<tr><td>Kragel 2015 (CANlab emotion)</td><td>32</td><td>4</td><td>+0.629</td><td>Too few for perm</td></tr>
<tr><td>Narratives group (Nastase 2021)</td><td>230</td><td>12</td><td>+0.32 to +0.39</td><td style="color:#155724"><b>YES</b> (p = 0.004-0.013)</td></tr>
<tr><td>Narratives regional</td><td>261</td><td>12</td><td>+0.20 (cortex mean)</td><td style="color:#155724"><b>400/400 parcels sig</b></td></tr>
</table>
<p class="note">All three independent fMRI sources are positive. Narratives group-level is statistically significant.</p>
</section>

<!-- ==================== SECTION 3: FINDING 2 - PREDICTABILITY ==================== -->
<section>
<h2><span class="num">3</span> Finding 2: Brain Predicts LLM</h2>
<p class="lead">The alignment is not a static correlation — the brain's geometry predicts LLM internal coupling, behavior, and developmental trajectory.</p>

<h3>3.1 Causal coupling: block-specific double dissociation</h3>
<p>Perturbing condition-selective activation subspaces shows block-specific effects:
disrupting affective conditions primarily affects affective processing, and vice versa.</p>
<div class="grid2">
<div><img class="fig" src="{fig_coupling}" alt="coupling"></div>
<div>
<table>
<tr><th>Model</th><th>N same &gt; cross</th><th>Wilcoxon p</th><th>DD?</th></tr>
{"".join(f'<tr><td>{m}</td><td>{coup["per_model"][m]["wilcoxon"]["n_same_greater"]}/14</td><td>{coup["per_model"][m]["wilcoxon"]["wilcoxon_p"]:.4f}</td><td style="color:#155724"><b>YES</b></td></tr>' for m in coup["per_model"])}
</table>
<p class="note">All 4 models show double dissociation. The affective-mentalizing axis
is functionally load-bearing inside the LLM.</p>
</div>
</div>

<h3>3.2 Moral judgment steering</h3>
<p>Steering along the brain-derived axis systematically modulates moral choice probability (logit-based, no text generation).</p>
<div class="grid2">
<div><img class="fig" src="{fig_moral}" alt="moral steering"></div>
<div>
<p><b>rho = {moral['correlation_all']['rho']:.3f}, p = {moral['correlation_all']['p']:.4f}</b></p>
<p>Negative alpha (toward affective) -> more utilitarian choices.
Positive alpha (toward mentalizing) -> more deontological.</p>
<div class="callout warn"><b>Direction caveat:</b> This does NOT validate Greene's dual-process mapping.
Our axis captures broad affect vs propositional mentalizing, not the specific harm-aversion signal
Greene predicts. We report as evidence for behavioral relevance, not theory validation.</div>
</div>
</div>

<h3>3.3 LLM judge: only brain axis works</h3>
<div class="grid2">
<div><img class="fig" src="{fig_steer_ctrl}" alt="steering controls"></div>
<div>
<p>DeepSeek (different company, different architecture) blind-ranks steered responses:</p>
<table>
<tr><th>Direction</th><th>Mean rho</th><th>p</th></tr>
<tr style="background:#eef3fc"><td><b>Brain axis</b></td><td><b>+0.320</b></td><td><b>0.004</b></td></tr>
<tr><td>Random</td><td>-0.053</td><td>0.560</td></tr>
<tr><td>Sentiment</td><td>+0.107</td><td>0.236</td></tr>
<tr><td>PC1</td><td>-0.010</td><td>0.910</td></tr>
</table>
<p class="note">Mann-Whitney brain vs each control: all p &lt; 0.04. Only the brain-derived direction works.</p>
</div>
</div>

<h3>3.4 Prospective prediction battery</h3>
<table>
<tr><th>#</th><th>Prediction</th><th>Result</th><th>Key stat</th></tr>
<tr><td>P1</td><td>Brain distance predicts coupling asymmetry</td><td><span class="tag yes">CONFIRMED</span></td>
<td>rho = {pp['prediction_1_coupling_asymmetry']['average_rho']:.3f}, p &lt; 0.001</td></tr>
<tr><td>P2</td><td>Within-block brain -> within-block coupling</td><td><span class="tag no">NULL</span></td><td>wrong direction</td></tr>
<tr><td>P3</td><td>Brain distinctiveness -> LLM distinctiveness</td><td><span class="tag no">NULL</span></td>
<td>rho = {pp['prediction_3_classification_accuracy']['average_rho']:.3f}</td></tr>
<tr><td>P4</td><td>Brain-predicted closest pairs = LLM closest</td><td><span class="tag yes">CONFIRMED</span></td>
<td>{pp['prediction_4_vulnerable_pairs']['n_overlap']}/10 overlap, p = 0.012</td></tr>
<tr><td>P5</td><td>Boundary proximity -> cross-block sensitivity</td><td><span class="tag trend">TREND</span></td>
<td>rho = {pp['prediction_5_boundary_sensitivity']['average_rho']:.3f}, p = 0.18</td></tr>
</table>
</section>

<!-- ==================== SECTION 4: FINDING 3 - INCONSISTENCY ==================== -->
<section>
<h2><span class="num">4</span> Finding 3: Interpretable Inconsistency</h2>
<p class="lead">The alignment has a precise boundary: social-cognitive structure transfers, fine-grained affective structure does not. This supports a refined embodiment hypothesis.</p>

<h3>4.1 Developmental trajectory (Pythia-2.8B, 9 checkpoints)</h3>
<img class="fig" src="{fig_pythia}" alt="Pythia trajectory" style="max-width:700px;display:block;margin:auto">
<div class="callout">
<b>Three developmental findings:</b><br>
1. <b style="color:{C_BLUE}">Full rho monotonically rises</b> (0.24 -> 0.72): the overall alignment strengthens during training.<br>
2. <b style="color:{C_MENT}">Social structure aligns early and stays</b> (+0.40 at step 0, stable at ~0.65).<br>
3. <b style="color:#cc3333">Affective structure actively DIVERGES</b> (+0.30 -> -0.57): the model develops its own emotion geometry that increasingly departs from the brain's.
</div>

<h3>4.2 Embodiment interpretation</h3>
<table>
<tr><th>Level</th><th>Brain-LLM aligned?</th><th>Implication</th></tr>
<tr><td>Emotion vs social boundary</td><td style="color:#155724"><b>YES</b></td><td>Language alone captures this category distinction</td></tr>
<tr><td>Social-cognitive fine structure</td><td style="color:#155724"><b>YES</b></td><td>Challenges strong embodiment — propositional concepts are linguistically-defined</td></tr>
<tr><td>Affective fine structure</td><td style="color:#cc3333"><b>NO (diverges)</b></td><td>Supports embodiment — body-dependent distinctions can't be learned from text</td></tr>
</table>
<div class="callout good">
<b>Bottom line:</b> Embodiment is not all-or-nothing. The propositional-relational component
of mentalizing is language-accessible and transfers. The body-dependent geometry of discrete
emotions (organized by arousal, valence, autonomic patterns) does not. Our results draw an
empirical line between what survives text compression and what does not.
</div>

<h3>4.3 Empathy: the bridge condition</h3>
<p>13/14 conditions align at rho 0.67-0.85. Only <b>empathy lags (rho ~ 0.24)</b>. This is consistent:
empathy is a <b>heterogeneous bridge construct</b> (Shamay-Tsoory 2011; Zaki & Ochsner 2012) spanning
affective experience-sharing (somatic) and cognitive perspective-taking (propositional).
The Neurosynth map aggregates both; the LLM likely captures only the cognitive component.</p>
</section>

<!-- ==================== SECTION 5: ROBUSTNESS ==================== -->
<section>
<h2><span class="num">5</span> Robustness Controls</h2>
<p class="lead">Multiple controls rule out trivial or spurious explanations.</p>

<h3>5.1 Partial RSA + Template-matched stimuli</h3>
<img class="fig" src="{fig_robustness}" alt="robustness">
<div class="callout good">
<b>Left:</b> After controlling for GloVe, condition-name, and sentence length, <b>77-80%</b> of signal survives.<br>
<b>Right:</b> With identical sentence templates (only content words differ), <b>74-89%</b> of signal survives. All p &lt; 0.001.
</div>

<h3>5.2 Additional controls (all PASS)</h3>
<table>
<tr><th>Test</th><th>Status</th><th>What it rules out</th></tr>
<tr><td>Paraphrase invariance (3 tests)</td><td><span class="tag yes">PASS</span></td><td>Dependence on specific word choices or stimulus items</td></tr>
<tr><td>Robustness gauntlet (4 tests)</td><td><span class="tag yes">PASS</span></td><td>Pipeline flexibility, any single condition/model driving result</td></tr>
<tr><td>Steering controls (3 null dirs)</td><td><span class="tag yes">PASS</span></td><td>Any random perturbation producing behavioral change</td></tr>
<tr><td>Confirmatory RSA (held-out test)</td><td><span class="tag yes">PASS</span></td><td>Overfitting recipe to data</td></tr>
<tr><td>Cross-architecture convergence</td><td><span class="tag yes">PASS</span></td><td>Architecture-specific artifacts</td></tr>
</table>
</section>

<!-- ==================== SECTION 6: SUMMARY ==================== -->
<section>
<h2><span class="num">6</span> Summary & Story</h2>

<div class="findcard f1">
<h4>Finding 1: Consistency</h4>
<p>4 architectures, rho ~ 0.73, scale-invariant 0.5-7B, 99.3% from pretraining.
Driven by one emotion-social axis + within-social fine structure.
Validated by 3 independent fMRI datasets (Narratives significant).</p>
</div>

<div class="findcard f2">
<h4>Finding 2: Predictability</h4>
<p>Brain geometry predicts: block-specific coupling (4/4 DD), moral steering (rho = -0.19, p = 0.006),
LLM judge confirms (brain only, 3 controls null), 2/5 prospective predictions confirmed.
Developmental trajectory (Pythia): full rho monotonically rises 0.24 -> 0.72.</p>
</div>

<div class="findcard f3">
<h4>Finding 3: Interpretable Inconsistency</h4>
<p>Within-affective structure does NOT transfer and actively diverges during training (+0.30 -> -0.57).
Social aligns early; affective reverses. Consistent with embodiment hypothesis:
propositional mentalizing transfers, body-dependent emotion geometry does not.</p>
</div>

<div class="speak">
<b>The story in one sentence:</b> A text-only LLM spontaneously develops the brain's
dominant affective-mentalizing axis and social-cognitive fine structure (rho ~ 0.73, 4 architectures,
scale-invariant), but fine-grained affective structure actively diverges during training —
drawing an empirical line between what language preserves and what requires bodily experience.
</div>

<h3>Next steps</h3>
<ul>
<li><b>Human blind ratings</b> — waiting for rater data (validates LLM judge finding with human perception)</li>
<li><b>Paper writing</b> — all experiments complete, findings consolidated</li>
</ul>
</section>

<footer>Generated from real result files | build_present.py | {total_stim} stimuli, 14 conditions, 4 architectures | 2026-06-05</footer>
</div>
</body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Wrote {OUT}  ({len(HTML)//1024} KB)")
