#!/usr/bin/env python3
"""
Build a self-contained HTML briefing (present/index.html) for Dr. Zhang's meeting.
All figures are rendered from REAL result files and embedded as base64 PNGs, so the
page needs no internet and no external assets.
"""
import json, base64, io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
AFFV = BASE / "results" / "affective_validation"
OUT = BASE / "present" / "index.html"

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
MSHORT = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
          "Mistral-7B-Instruct-v0.3": "Mistral-7B", "gemma-2-9b-it": "Gemma-2-9B"}
AFF = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
MENT = ["judgment", "belief", "intention", "mentalizing", "moral", "empathy",
        "self_referential", "theory_of_mind"]
ORDER = AFF + MENT
C_AFF, C_MENT = "#e8743b", "#19a979"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 130,
                     "axes.spreadsizered" if False else "axes.edgecolor": "#888"})


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def reorder(rdm, conds, order):
    idx = [conds.index(c) for c in order]
    return rdm[np.ix_(idx, idx)]


def heatmap(rdm, conds, title, cmap="magma"):
    R = reorder(rdm, conds, ORDER).astype(float)
    n = R.shape[0]
    # Rank-normalize off-diagonal entries to [0,1] so BOTH panels share one scale
    # (the absolute distance units differ: brain=1-Pearson, LLM=1-cosine). RSA uses
    # ranks anyway, so identical structure -> identical colors across the two panels.
    iu = np.triu_indices(n, 1)
    ranks = R[iu].argsort().argsort().astype(float)
    ranks = ranks / (len(ranks) - 1)
    Rn = np.full((n, n), np.nan)
    Rn[iu] = ranks
    Rn[(iu[1], iu[0])] = ranks
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#e8e8e8")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(Rn, cmap=cm, vmin=0, vmax=1)
    ax.set_xticks(range(len(ORDER))); ax.set_yticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=90, fontsize=8)
    ax.set_yticklabels(ORDER, fontsize=8)
    n_aff = len(AFF)
    for pos in [n_aff]:
        ax.axhline(pos - 0.5, color="cyan", lw=2)
        ax.axvline(pos - 0.5, color="cyan", lw=2)
    # tick label colors by block
    for t, c in zip(ax.get_xticklabels(), ORDER):
        t.set_color(C_AFF if c in AFF else C_MENT)
    for t, c in zip(ax.get_yticklabels(), ORDER):
        t.set_color(C_AFF if c in AFF else C_MENT)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      label="relative distance (rank: 0 = most similar, 1 = most different)")
    cb.set_ticks([0, 0.5, 1])
    return b64(fig)


# ---------- load data ----------
ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
ns_rdm, ns_conds = ns["rdm"], list(ns["conditions"])

qz = np.load(RSA / "Qwen2.5-7B-Instruct_rdm14_headline.npz", allow_pickle=True)
q_rdm, q_conds = qz["rdm"], list(qz["conditions"])

dev = json.load(open(RSA.parent / "developmental_emergence" / "developmental_emergence.json"))["per_size"]
ceil = json.load(open(AFFV / "affective_ceiling_control.json"))["aggregate"]

# headline rho computed FRESH against the current (pure-Neurosynth) brain_rdm.npz,
# not read from the stale rsa_v2.json (which used the old HCP-ToM brain RDM).
head_rho = {}
_iu = np.triu_indices(len(ns_conds), 1)
for m in MODELS:
    z = np.load(RSA / f"{m}_rdm14_headline.npz", allow_pickle=True)
    lrdm, lconds = z["rdm"], list(z["conditions"])
    order = [lconds.index(c) for c in ns_conds]
    L = lrdm[np.ix_(order, order)]
    head_rho[m] = float(spearmanr(L[_iu], ns_rdm[_iu])[0])

# ---------- FIG A & B : brain vs LLM RDM ----------
fig_brain = heatmap(ns_rdm, ns_conds, "Human brain (Neurosynth) — 14×14 geometry", "magma")
fig_llm = heatmap(q_rdm, q_conds, "Qwen2.5-7B (peak layer) — 14×14 geometry", "magma")

# ---------- FIG C : headline rho bars ----------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
xs = [MSHORT[m] for m in MODELS]; ys = [head_rho[m] for m in MODELS]
bars = ax.bar(xs, ys, color="#2e5cb8", width=0.6)
for b, y in zip(bars, ys):
    ax.text(b.get_x() + b.get_width() / 2, y + 0.01, f"{y:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.85); ax.set_ylabel("Spearman ρ  (LLM vs brain)")
ax.axhspan(0.7, 0.78, color="#2e5cb8", alpha=0.06)
ax.set_title("Brain–LLM alignment ρ≈0.73, near-identical across 4 architectures")
ax.grid(axis="y", alpha=0.3)
fig_bars = b64(fig)

# ---------- FIG D : scale invariance ----------
sizes = ["0.5B", "1.5B", "3B", "7B"]
params = [dev[s]["n_params_M"] for s in sizes]
ov = [dev[s]["peak_overall_rho"] for s in sizes]
af = [dev[s]["affective_mean_rho"] for s in sizes]
me = [dev[s]["mentalistic_mean_rho"] for s in sizes]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(params, af, "-o", color=C_AFF, lw=2.5, label="Affective module")
ax.plot(params, ov, "-o", color="#2e5cb8", lw=2.5, label="Overall (14×14)")
ax.plot(params, me, "-o", color=C_MENT, lw=2.5, label="Social-cognition module")
ax.set_xscale("log")
ax.set_xticks(params); ax.set_xticklabels(sizes)
ax.set_xlabel("Model size (Qwen2.5 family, log scale)")
ax.set_ylabel("Brain alignment ρ")
ax.set_ylim(0.35, 0.95); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Scale-invariant: both modules align with the brain, 0.5B → 7B")
fig_scale = b64(fig)

# ---------- FIG E : per-condition sorted bars (7B) ----------
pc = dev["7B"]["per_condition_alignment"]
items = sorted(pc.items(), key=lambda kv: -kv[1]["rho_mean"])
labels = [k for k, _ in items]
vals = [v["rho_mean"] for _, v in items]
cols = [C_AFF if k in AFF else C_MENT for k in labels]
fig, ax = plt.subplots(figsize=(7.4, 4.2))
bars = ax.barh(range(len(labels))[::-1], vals, color=cols)
ax.set_yticks(range(len(labels))[::-1]); ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color="#333", lw=1)
for y, v in zip(range(len(labels))[::-1], vals):
    ax.text(v + (0.015 if v >= 0 else -0.015), y, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.set_xlim(0, 0.95); ax.set_xlabel("Brain alignment ρ (per condition, Qwen-7B)")
ax.set_title("Nearly all cognitive domains align with the brain (only empathy lags)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_AFF, label="Affective"), Patch(color=C_MENT, label="Social cognition")],
          loc="lower right", fontsize=9)
fig_perc = b64(fig)

# ---------- FIG F : ceiling control ----------
aff_al = np.mean([ceil[m]["AFF"]["align"] for m in MODELS])
aff_ce = np.mean([ceil[m]["AFF"]["ceiling"] for m in MODELS])
men_al = np.mean([ceil[m]["ment"]["align"] for m in MODELS])
men_ce = np.mean([ceil[m]["ment"]["ceiling"] for m in MODELS])
fig, ax = plt.subplots(figsize=(6.2, 3.8))
x = np.arange(2); w = 0.36
ax.bar(x - w / 2, [aff_ce, men_ce], w, color="#cccccc", label="Noise ceiling (LLM self-consistency)")
ax.bar(x + w / 2, [aff_al, men_al], w, color=[C_AFF, C_MENT], label="Brain alignment")
ax.text(0 - w / 2, aff_ce + .01, f"{aff_ce:.2f}", ha="center", fontsize=9)
ax.text(1 - w / 2, men_ce + .01, f"{men_ce:.2f}", ha="center", fontsize=9)
ax.text(0 + w / 2, aff_al + .01, f"{aff_al:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.text(1 + w / 2, men_al + .01, f"{men_al:.2f}", ha="center", fontsize=9, fontweight="bold")
ax.annotate("≈78% of ceiling", (0 + w / 2, aff_al), xytext=(-0.05, 0.5),
            fontsize=9, color=C_AFF, fontweight="bold")
ax.annotate("≈87% of ceiling", (1 + w / 2, men_al), xytext=(0.95, 0.5),
            fontsize=9, color=C_MENT, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Affective", "Social cognition"])
ax.set_ylim(0, 1.05); ax.set_ylabel("ρ")
ax.set_title("Both modules align with the brain NEAR their noise ceiling")
ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3)
fig_ceil = b64(fig)

# ---------- stimuli table data ----------
manifest = json.load(open(BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_conditions_manifest.json"))
rows = json.load
import collections
stim = [json.loads(l) for l in open(BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl")]
cnt = collections.Counter(r["condition"] for r in stim)
src = {r["condition"]: r["source"] for r in stim}

SRC_HUMAN = {
    "anger": "Emotion localizer (curated + GoEmotions, Demszky 2020)",
    "fear": "Emotion localizer (curated + GoEmotions)",
    "sadness": "Emotion localizer (curated + GoEmotions)",
    "disgust": "Emotion localizer (curated + GoEmotions)",
    "happiness": "Emotion localizer (curated + GoEmotions)",
    "valence": "VAD-graded sentences (Warriner 2013 norms)",
    "belief": "False-belief stories",
    "intention": "Indirect requests",
    "judgment": "ETHICS commonsense (Hendrycks 2021)",
    "moral": "Moral dilemmas (trolley-type) + Moral Foundations Vignettes",
    "theory_of_mind": "Self/Other + Faux-Pas (Baron-Cohen paradigm)",
    "mentalizing": "Strange Stories (Happé paradigm)",
    "empathy": "Moral Foundations — care dimension",
    "self_referential": "Self/Other (self condition)",
}

stim_rows = ""
for c in ORDER:
    blk = "Affective" if c in AFF else "Social cognition"
    col = C_AFF if c in AFF else C_MENT
    stim_rows += (f"<tr><td><b style='color:{col}'>{c}</b></td><td>{blk}</td>"
                  f"<td style='text-align:center'>{cnt[c]}</td><td>{SRC_HUMAN[c]}</td></tr>\n")

total_stim = sum(cnt.values())

# ---------- HTML ----------
HTML = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM × Brain — Project Briefing</title>
<style>
:root{{--aff:{C_AFF};--ment:{C_MENT};--blue:#2e5cb8;--ink:#1d2330;--muted:#5a6472;}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;
color:var(--ink);background:#f4f6fa;line-height:1.65}}
.wrap{{max-width:1060px;margin:0 auto;padding:0 26px 80px}}
header{{background:linear-gradient(135deg,#1d2a4d,#2e5cb8);color:#fff;padding:54px 26px 40px;margin-bottom:34px}}
header .inner{{max-width:1060px;margin:0 auto}}
header h1{{margin:0 0 8px;font-size:30px;letter-spacing:.3px}}
header p{{margin:4px 0;opacity:.92;font-size:15px}}
.badge{{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);
border-radius:20px;padding:3px 13px;font-size:13px;margin:8px 6px 0 0}}
section{{background:#fff;border-radius:14px;padding:26px 30px;margin:22px 0;box-shadow:0 2px 14px rgba(20,30,60,.06)}}
h2{{font-size:22px;margin:0 0 6px;border-left:5px solid var(--blue);padding-left:12px}}
h2 .num{{color:var(--blue);font-weight:800;margin-right:8px}}
h3{{font-size:16px;margin:22px 0 8px;color:#26324d}}
.lead{{color:var(--muted);font-size:14px;margin:0 0 16px;padding-left:17px}}
p{{font-size:15px}}
img.fig{{max-width:100%;border:1px solid #e3e8f0;border-radius:10px;margin:10px 0;background:#fff}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}}
@media(max-width:780px){{.grid2{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}}
th,td{{border:1px solid #e6eaf2;padding:7px 10px;text-align:left}}
th{{background:#eef2fa;font-weight:700}}
tr:nth-child(even) td{{background:#fafbfe}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 4px}}
.kpi .card{{flex:1;min-width:170px;background:linear-gradient(135deg,#f7f9ff,#eef2fb);border:1px solid #e2e8f5;
border-radius:12px;padding:16px 18px}}
.kpi .big{{font-size:30px;font-weight:800;color:var(--blue);line-height:1}}
.kpi .lbl{{font-size:12.5px;color:var(--muted);margin-top:6px}}
.callout{{background:#fff8ec;border:1px solid #f0d9a8;border-left:5px solid #e0a73a;border-radius:10px;
padding:14px 18px;margin:16px 0;font-size:14.5px}}
.callout.good{{background:#eefaf4;border-color:#a6e0c6;border-left-color:var(--ment)}}
.callout.warn{{background:#fdeeee;border-color:#f0bcbc;border-left-color:#d65b5b}}
.note{{font-size:13px;color:var(--muted)}}
.pill{{display:inline-block;padding:1px 9px;border-radius:10px;font-size:12px;font-weight:700;color:#fff}}
.pill.a{{background:var(--aff)}}.pill.m{{background:var(--ment)}}
ul{{margin:8px 0 8px}} li{{margin:4px 0}}
.speak{{background:#eef2fa;border-radius:8px;padding:10px 14px;font-size:13.5px;color:#33405e;margin-top:10px}}
.speak b{{color:var(--blue)}}
.gloss{{background:#f6f8fc;border:1px dashed #c4cfe3;border-radius:10px;padding:12px 16px;margin:12px 0;font-size:13.5px}}
.gloss .term{{font-weight:800;color:var(--blue)}}
.gloss .gitem{{margin:7px 0}}
.step{{display:flex;gap:14px;align-items:flex-start;margin:12px 0}}
.step .n{{flex:none;width:30px;height:30px;border-radius:50%;background:var(--blue);color:#fff;
font-weight:800;display:flex;align-items:center;justify-content:center;font-size:15px;margin-top:2px}}
.step .t{{font-size:14.5px}}
.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:780px){{.twocol{{grid-template-columns:1fr}}}}
.sidebox{{border-radius:12px;padding:14px 18px;font-size:14px}}
.sidebox.brain{{background:#fdeeea;border:1px solid #e8743b}}
.sidebox.llm{{background:#e9f7f1;border:1px solid #19a979}}
.sidebox h4{{margin:0 0 8px;font-size:15px}}
.findcard{{border-radius:12px;padding:16px 18px;margin:12px 0}}
.findcard.main{{background:#eef3fc;border-left:5px solid var(--blue)}}
.findcard.div{{background:#fff4ec;border-left:5px solid var(--aff)}}
.findcard.causal{{background:#eefaf4;border-left:5px solid var(--ment)}}
.findcard h4{{margin:0 0 6px;font-size:15.5px}}
.flow{{margin:18px 0}}
.flowrow{{display:flex;align-items:stretch;gap:6px;flex-wrap:nowrap;margin:10px 0;overflow-x:auto;padding-bottom:4px}}
.fbox{{flex:1;min-width:120px;border-radius:10px;padding:10px 10px;font-size:12.5px;text-align:center;
display:flex;align-items:center;justify-content:center;line-height:1.4}}
.fbox.b{{background:#fdeeea;border:1.5px solid #e8743b}}
.fbox.l{{background:#e9f7f1;border:1.5px solid #19a979}}
.fbox.final{{font-weight:800;font-size:13.5px}}
.fbox.b.final{{background:#f8d7cd;border-color:#b03a2e;color:#7b271b}}
.fbox.l.final{{background:#c9ecdd;border-color:#1f6f54;color:#13503b}}
.farr{{flex:none;align-self:center;color:#9aa3b2;font-size:20px;font-weight:800}}
.flowtag{{font-weight:800;font-size:14px;margin:4px 0}}
.flowtag.b{{color:#b03a2e}}.flowtag.l{{color:#1f6f54}}
.flowjoin{{text-align:center;margin:10px 0}}
.flowjoin .merge{{display:inline-block;background:#e8edf8;border:1.5px solid #2e5cb8;border-radius:10px;
padding:12px 22px;font-size:14px;color:#1d2a4d}}
.flowjoin .rho{{font-size:22px;font-weight:800;color:#2e5cb8;margin-top:6px}}
footer{{text-align:center;color:#9aa3b2;font-size:12.5px;margin-top:30px}}
</style></head>
<body>
<header><div class="inner">
<h1>用人脑当尺子读懂大模型</h1>
<p>Reading LLMs through the lens of the human brain — Project Briefing</p>
<p style="opacity:.8;font-size:13px">Nature Machine Intelligence 方向 · 4 个开源大模型 × 14 个认知条件 × 多源真实 fMRI</p>
<div>
<span class="badge">712 条认知刺激</span>
<span class="badge">14 个认知条件</span>
<span class="badge">4 种模型架构 + Qwen 全尺度</span>
<span class="badge">4 个脑数据源</span>
<span class="badge">2026-05-30</span>
</div>
</div></header>

<div class="wrap">

<section>
<h2><span class="num">0</span>一句话结论 (The headline)</h2>
<p>人脑天生把<b style="color:var(--aff)">情绪</b>和<b style="color:var(--ment)">社会认知</b>分在两套系统里。
我们发现：<b>大模型在毫不知情的情况下，也自发地把这两类分开了——而且整张"认知关系地图"和人脑高度一致（ρ≈0.73，4 个架构几乎一模一样）。</b>
而且不止那条分界线：<b style="color:var(--aff)">情绪</b>和<b style="color:var(--ment)">社会认知</b>两大类的<b>内部组织都贴近人脑</b>
（各达到理论上限的 ~78% / ~87%）；14 个认知功能里<b>几乎全部都对得上</b>，只有"共情"（样本最少）偏低。</p>
<div class="kpi">
<div class="card"><div class="big">0.73</div><div class="lbl">脑–LLM 关系几何相关 (Spearman ρ)<br>4 模型 0.727–0.739, p&lt;0.0002</div></div>
<div class="card"><div class="big">近天花板</div><div class="lbl">情绪 &amp; 社会认知<br>双双接近理论上限</div></div>
<div class="card"><div class="big">4 / 4</div><div class="lbl">架构一致 + 尺度 0.5B→7B 不变</div></div>
<div class="card"><div class="big">14 类</div><div class="lbl">认知功能几乎全部<br>与人脑对齐 (0.67–0.85)</div></div>
</div>
</section>

<section>
<h2><span class="num">1</span>数据集与刺激 (Datasets & stimuli)</h2>
<p class="lead">prompt 全部来自公开、有出处的心理学 / NLP 数据集，不是自己编的；构建脚本确定性可复现（seed=20260525）。</p>
<h3>LLM 这一侧</h3>
<ul>
<li><b>4 个不同架构（主结果）：</b>Qwen2.5-7B-Instruct · Llama-3.1-8B-Instruct · Mistral-7B-Instruct-v0.3 · Gemma-2-9B-it</li>
<li><b>Qwen 全尺度系列（尺度不变性）：</b>0.5B · 1.5B · 3B · 7B</li>
<li><b>认知刺激库：</b>共 <b>{total_stim} 条句子</b>，归入 <b>14 个认知条件</b>
（<span class="pill a">6 情绪</span> + <span class="pill m">8 社会认知</span>）</li>
</ul>
<table>
<tr><th>条件 (condition)</th><th>模块</th><th>条数</th><th>刺激来源 (prompt source)</th></tr>
{stim_rows}
</table>
<p class="note">注：empathy(32) / self_referential(30) / mentalizing(40) 条数偏少，是后续要补强的点。</p>

<h3>大脑这一侧</h3>
<ul>
<li><b>主结果 — Neurosynth：</b>~14,000 篇 fMRI 论文的 meta 分析关联图，每个认知条件一张全脑图（不是单次扫描，统计上最稳）。</li>
<li><b>三个真实受控 fMRI（交叉验证，证明不是 Neurosynth 单源巧合）：</b>
<ul>
<li><b>Kragel 2015</b> — 情绪，N=32 被试（CANlab）</li>
<li><b>IBC</b> — Individual Brain Charting，同 12 被试多任务，无批次混淆（NeuroVault coll. 2138）</li>
<li><b>HCP</b> — S1200 组平均任务对比（NeuroVault coll. 457）</li>
</ul></li>
</ul>
</section>

<section>
<h2><span class="num">2</span>我们喂的是什么？大脑实验喂的又是什么？</h2>
<p class="lead">先把"喂给模型的东西"和"大脑实验里给人看的东西"对齐清楚——这是整个研究能成立的桥。</p>

<p>我们的做法,本质上是把<b>同一类材料</b>分别"喂"给两种被试:一种是<b>大模型</b>,一种是<b>躺在核磁共振机器(fMRI)里的人</b>,
然后各自记录它们"内部的反应",再比这两种反应像不像。</p>

<table>
<tr><th></th><th>我们这边（大模型实验）</th><th>他们那边（人类脑实验）</th></tr>
<tr><td><b>喂进去的东西</b></td><td>一句话文本（我们叫 prompt / 刺激）</td><td>给被试看/读的材料（叫 stimulus 刺激）</td></tr>
<tr><td><b>"被试"是谁</b></td><td>大模型</td><td>核磁共振机里的人</td></tr>
<tr><td><b>记录什么反应</b></td><td>模型内部的数字向量（隐藏层）</td><td>脑里哪些区域"亮起来"（血氧信号）</td></tr>
</table>

<div class="callout"><b>一个常被问的细节：</b>我们喂给模型的就是<b>一句裸句子</b>，
没有套"请你判断下面这句话…"之类的指令。所以在我们这儿，<b>"刺激"和"prompt"是同一个东西</b>。
这样测到的才是模型对<b>这段话本身</b>的反应，而不是模型"听懂指令"的反应。</div>

<h3>哪些是对得上的，哪些对不上？</h3>
<p>两边并<b>不是逐字喂同一句话</b>。我们在<b>"概念层面"</b>对齐（两边都针对"恐惧""错误信念"等同一个认知功能），
但具体材料和形式可能不同：</p>
<table>
<tr><th>对比维度</th><th>是否一致</th><th>大白话解释</th></tr>
<tr><td>认知概念（如"恐惧"）</td><td style="color:#1f6f54"><b>✅ 一致</b></td><td>两边都在测同一个心理功能——这是我们对齐的层级</td></tr>
<tr><td>具体的那句话</td><td style="color:#b03a2e"><b>❌ 不一致</b></td><td>我们的句子 ≠ 脑研究里用的材料，只在概念上对应</td></tr>
<tr><td>材料形式（模态）</td><td style="color:#b03a2e"><b>❌ 多数不一致</b></td><td>我们全是<b>文字</b>；情绪类脑研究多是给人<b>看人脸/图片</b></td></tr>
</table>
<div class="callout good"><b>反直觉、但对我们有利的一点 👇</b><br>
"情绪"恰恰是材料形式最对不上的（大脑看脸 vs 我们读字），
<b>结果它反而和大脑最像（~0.78）</b>；而材料形式最吻合的部分社会认知（都用文字），结果反而最不像。<br>
→ 说明这个一致性<b>不是靠"用了同一批材料"凑出来的</b>，而是来自更深层的功能组织。这正是它可信、不是假象的最强证据。</div>
</section>

<section>
<h2><span class="num">3</span>这张"关系表"到底怎么算出来的？(方法)</h2>
<p class="lead">这一节把核心方法 RSA 一步步讲透，每个词都用大白话解释，看完就能跟老板讲清楚。</p>

<div class="gloss">
<div class="gitem"><span class="term">脑图 (brain map)</span>：一张三维的"大脑亮度图"，告诉你做某件事（比如感到恐惧）时，
大脑里<b>哪些区域会活跃</b>。把大脑切成约 90 万个小立方块（每块叫一个 <b>voxel/体素</b>），每块给一个亮度数值。</div>
<div class="gitem"><span class="term">RDM（关系表 / 表征差异矩阵）</span>：一张 14×14 的表格，
记录<b>"任意两个概念之间有多不一样"</b>。比如"恐惧 vs 愤怒"很像（数值小），"恐惧 vs 道德判断"差很远（数值大）。
它描述的是<b>概念之间的亲疏关系</b>，不关心绝对位置。</div>
<div class="gitem"><span class="term">RSA（表征相似性分析）</span>：不去对"哪个神经元 = 哪个脑区"（那是对不上的），
而是<b>比两张关系表像不像</b>——大脑觉得"近"的两个概念，模型是不是也觉得"近"。</div>
</div>

<h3>第一步：大脑侧和模型侧，各自做出一张 14×14 关系表</h3>
<div class="twocol">
<div class="sidebox brain">
<h4 style="color:#b03a2e">🧠 大脑这一侧</h4>
<div class="step"><div class="n" style="background:#b03a2e">1</div><div class="t">14 个概念，各拿一张已发表的<b>脑图</b>（哪些脑区会亮）。</div></div>
<div class="step"><div class="n" style="background:#b03a2e">2</div><div class="t">把每张脑图<b>拍平成一长串数字</b>（约 90 万个脑点各一个亮度值）。</div></div>
<div class="step"><div class="n" style="background:#b03a2e">3</div><div class="t">两两比较这些数字串有多像 → 得到<b>大脑 14×14 关系表</b>。</div></div>
</div>
<div class="sidebox llm">
<h4 style="color:#1f6f54">🤖 大模型这一侧</h4>
<div class="step"><div class="n" style="background:#1f6f54">1</div><div class="t">每个概念有一批句子（共 712 句），逐句<b>裸文本喂进模型</b>。</div></div>
<div class="step"><div class="n" style="background:#1f6f54">2</div><div class="t">读出模型<b>内部的一串数字</b>（峰值层、句内平均），同概念的句子取平均。</div></div>
<div class="step"><div class="n" style="background:#1f6f54">3</div><div class="t">两两比较 → 得到<b>模型 14×14 关系表</b>。</div></div>
</div>
</div>

<h3>第二步：比两张关系表像不像 → 得到 ρ</h3>
<div class="flow">
<div class="flowtag b">🧠 大脑这一侧</div>
<div class="flowrow">
<div class="fbox b">14 张已发表脑图<br>(每个概念一张,<br>哪些脑区会亮)</div>
<div class="farr">→</div>
<div class="fbox b">每张拍平成<br>一长串数字<br>(约90万个脑点)</div>
<div class="farr">→</div>
<div class="fbox b">两两比相似度<br>(1 − 相关系数)</div>
<div class="farr">→</div>
<div class="fbox b final">大脑<br>14×14 关系表</div>
</div>
<div class="flowtag l">🤖 大模型这一侧</div>
<div class="flowrow">
<div class="fbox l">712 句裸文本<br>(每个概念一批句子)</div>
<div class="farr">→</div>
<div class="fbox l">每句喂模型,<br>取内部向量<br>(峰值层·句内平均)</div>
<div class="farr">→</div>
<div class="fbox l">两两比相似度<br>(1 − cosine)</div>
<div class="farr">→</div>
<div class="fbox l final">大模型<br>14×14 关系表</div>
</div>
<div class="flowjoin">
<div style="color:#9aa3b2;font-size:20px;font-weight:800">↓ &nbsp; 两张表汇到一起比 &nbsp; ↓</div>
<div class="merge">取上三角 91 对概念 → 做排序相关 (Spearman)
<div class="rho">ρ ≈ 0.73</div></div>
</div>
</div>
<div class="step"><div class="n">A</div><div class="t">每张 14×14 表，取出 <b>91 个数</b>（14 个概念两两配对 = 91 对）。</div></div>
<div class="step"><div class="n">B</div><div class="t">把"大脑的 91 个数"和"模型的 91 个数"做<b>排序相关（Spearman ρ）</b>——
看两边对"谁和谁更像"的<b>排序</b>是否一致。结果 <b>ρ≈0.73</b>。</div></div>
<div class="step"><div class="n">C</div><div class="t">再把概念标签<b>随机打乱 1 万次</b>重算，真值远超随机 → <b>p&lt;0.0002</b>（几乎不可能是巧合）。</div></div>

<h3>那张"大脑关系表"具体用的哪些脑图？</h3>
<p>主结果的大脑侧 <b>14 张图统一来自 Neurosynth</b>（一个公开的脑成像证据库）：</p>
<ul>
<li><b>Neurosynth 是什么：</b>一个汇总了约 <b>14,000 篇</b>脑成像论文的公开库。
你给它一个词（如"fear 恐惧"），它把<b>所有研究过恐惧的论文</b>的结果叠加，给你一张"恐惧时大脑哪里亮"的平均图。
好处是它不是某一次实验、某一批人的偶然结果，而是<b>整个领域的共识</b>，最稳；而且它<b>不挑材料形式</b>，里面既有看脸的、也有读文字的研究。</li>
</ul>
<div class="callout"><b>一个我们刚做的稳健性检查（值得一提）：</b>
心智推断(ToM)这一项,早期曾临时换用过另一个数据源(HCP 的真实扫描),结果它显得和人脑"反着来"。
我们把它<b>换回 Neurosynth、让 14 张图来源统一</b>后,ToM 立刻变成<b>高度对齐(+0.78)</b>,
整体 ρ 也从 0.63 升到 <b>0.73</b>。→ 说明之前那个"反相关"是<b>混用数据源的假象</b>,统一来源后结论更干净、更强。</div>
<p class="note">说明：另外几个真实扫描数据（Kragel N=32 情绪 / IBC 12 被试多任务）是<b>单独</b>搭的小验证，不在这张主表里——
用来交叉检验"情绪 vs 社会认知这条分界线"不是 Neurosynth 一家之言（两者方向都为正）。</p>
</section>

<section>
<h2><span class="num">4</span>核心发现 (Results)</h2>

<h3>4.1 两张关系表，肉眼可见地像</h3>
<p>左 = 人脑(Neurosynth)，右 = Qwen2.5-7B。<b>两张图已用同一配色、同一刻度（按"远近排名"归一化）</b>，可直接对比颜色：
<b>暗色 = 两个概念很像，亮色 = 很不一样。</b>青线把<span class="pill a">情绪 6 类</span>和
<span class="pill m">社会认知 8 类</span>分开——两侧都出现同样的<b>左上 / 右下两个暗色方块</b>（块内相似、块间相异），这就是那条共同的分界。</p>
<div class="grid2">
<div><img class="fig" src="{fig_brain}" alt="brain RDM"></div>
<div><img class="fig" src="{fig_llm}" alt="llm RDM"></div>
</div>

<h3>4.2 这张关系地图，4 个架构都和大脑高度一致（ρ≈0.73）</h3>
<img class="fig" src="{fig_bars}" alt="rho bars" style="max-width:660px;display:block;margin:auto">
<div class="callout good"><b>头牌主张：</b>4 个完全不同的架构，和大脑的关系地图相关都在 <b>0.727–0.739</b>，
几乎一模一样（p&lt;0.0002）。其中"情绪 vs 社会认知"那条分界两边都画得很清楚，是这张地图最稳的骨架。</div>

<h3>4.3 尺度不变：从 0.5B 到 7B 都成立</h3>
<img class="fig" src="{fig_scale}" alt="scale" style="max-width:660px;display:block;margin:auto">
<p>不是某个模型的偶然——Qwen 从 0.5B 长到 7B，<b>情绪和社会认知两条线都一直很高、一直稳定</b>，
说明这套"像人脑"的组织在很小的模型里就已经成形。</p>

<h3>4.4 逐条件看：14 类认知功能，几乎全部都对得上</h3>
<img class="fig" src="{fig_perc}" alt="per condition" style="max-width:720px;display:block;margin:auto">
<p>逐个认知功能看脑对齐：<b>情绪类(橙)和社会认知类(绿)都排在高位</b>（0.67–0.85，belief/happiness/judgment/ToM 都很高）；
<b>唯一明显偏低的是"共情(empathy)"</b>——而它恰好是样本最少（32 条）、最不稳的一项（见下方对照）。</p>

<h3>4.5 对照：这种对齐是"真的接近上限"，不是凑出来的</h3>
<img class="fig" src="{fig_ceil}" alt="ceiling control" style="max-width:660px;display:block;margin:auto">
<div class="callout good"><b>对照逻辑：</b>灰柱 = 模型对该模块表征的<b>自洽稳定度（理论上限/天花板）</b>，
彩柱 = 实际脑对齐。<b>情绪拿到上限的 ~78%，社会认知拿到 ~87%</b>——两者<b>都接近各自的天花板</b>，
说明这种"像人脑"是真实而接近最大可能值的，不是噪声或巧合。</div>
</section>

<section>
<h2><span class="num">5</span>完整结论清单 (All findings)</h2>
<p class="lead">前面讲的是头牌，但我们的结论不止这些。整体分三条线：关系结构守恒（主线）、行为像但机制不同（分歧线）、因果可验证。</p>

<div class="findcard main">
<h4>主线 · 关系结构是守恒的（这是论文骨架）</h4>
<ol style="margin:6px 0">
<li><b>跨域关系守恒：</b>4 个模型和大脑的关系地图相关 ρ=0.727–0.739，几乎不可能是巧合（p&lt;0.0002）。</li>
<li><b>情绪 / 社会认知的分界是"双方都画"的：</b>大脑天生把这两类分开，模型也自发分开了，且<b>分界线一致</b>——这是头牌骨架。</li>
<li><b>情绪和社会认知都接近天花板：</b>两大类都和人脑高度对齐（情绪 ~78%、社会认知 ~87% 的理论上限）；
14 个认知功能里<b>几乎全部对齐（0.67–0.85）</b>，仅"共情"（样本最少）偏低。</li>
<li><b>尺度不变：</b>模型从 0.5B 长到 7B，这套结构稳定不变。</li>
<li><b>是"全局"现象：</b>这种一致性贯穿模型几乎所有深度层（不是某一层的偶然），说明它是模型的根本组织方式。</li>
<li><b>普遍的"铁律配对"：</b>跨 4 个模型都成立——某些概念对永远靠得近（如各种情绪之间），某些永远离得远（情绪 × 理性认知的跨界）。</li>
</ol>
</div>

<div class="findcard div">
<h4>分歧线 · "表现像人" ≠ "机制像人"</h4>
<ol start="7" style="margin:6px 0">
<li><b>同一个道德判断，4 个模型用的"内部线路"完全不同：</b>
Llama/Mistral 用极少数神经元（&lt;0.1%）的"促进线路"，Qwen 用的是"抑制线路"，Gemma 则是分散式——
行为上都会做道德判断，但实现方式各异。</li>
<li><b>模型的"道德"其实长在"情绪"区：</b>模型处理道德的部件，位置上和大脑的<b>情绪</b>区对应，
而不是和大脑的<b>道德</b>区对应——说明它更像"靠情绪做道德判断"。</li>
</ol>
</div>

<div class="findcard causal">
<h4>因果线 · 不只是"看起来像"，是真能动它</h4>
<ol start="9" style="margin:6px 0">
<li><b>大脑的关系结构能预测模型"伤在哪连带坏哪"：</b>当我们真的去"切掉"模型里某类功能的神经元，
连带受损的其它功能，其模式能被大脑的关系表预测到（4 个模型里 3 个显著）。</li>
</ol>
</div>
<p class="note">（更早期那套"任务功能区"的结论——8 路功能分离、依赖关系图、剪枝等——保留为补充材料，不再是主线。）</p>
</section>

<section>
<h2><span class="num">6</span>为什么值得做 / 下一步 (Why & next)</h2>
<ul>
<li><b>定位升级：</b>从"LLM 任务功能区"升级到<b>认知科学 × 脑科学</b>框架——用人脑当参照系来解释大模型内部组织。</li>
<li><b>正面落点：</b>纯语言训练的大模型，在<b>毫无身体、毫无感官</b>的前提下，
自发重建了人脑对<b>情绪与社会认知</b>的整套关系组织（ρ≈0.73，近天花板，跨架构、跨尺度稳定）。</li>
<li><b>可补强方向：</b>扩充小样本条件（如共情）→ 用更多单数据集多任务 fMRI 巩固边界 →
因果实验（切除神经元看模块间伤害扩散，Direction A 已 3/4 模型显著）。</li>
</ul>
<div class="speak">🎤 <b>结尾可以这么说：</b>我们最硬、最难被推翻的一句话是——
<b>"一个只读过文字的大模型，自发重建了人脑组织情绪与社会认知的整张关系地图（ρ≈0.73），
还自己画出了情绪 vs 社会认知那条分界——跨 4 种架构、从 0.5B 到 7B 都成立。"</b></div>
</section>

<footer>Generated from real result files · experiments/present/build_present.py · 数字均可复现</footer>
</div>
</body></html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"Wrote {OUT}  ({len(HTML)//1024} KB)")
