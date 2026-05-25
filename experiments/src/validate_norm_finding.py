"""
Statistical validation of the moral-conventional (norm_type) causal finding.

Two complementary analyses:

1. Per-pair paired tests (using v2 moral_decomposition per-pair data):
   - For each model, take the 50 norm_type pairs.
   - Compute per-pair (a-b) log-odds delta under baseline and under each
     ablation condition (moral / moral_anti / emotion / tom / self /
     neutral_control / random).
   - Paired t-test: is (baseline_pair_delta - ablation_pair_delta) different
     from zero across the 50 pairs?
   - Bootstrap 95% CI on the per-model mean shrink.
   - BH-FDR across the 4 conditions (intent, outcome, norm_type, morality).

2. Cross-model aggregate test (using contrast_decomposition diagonal deltas):
   - Treat each model as one observation point.
   - One-sample t-test against zero on the 4 diagonal deltas (top vs base).
   - Compare against the 4 anti-diagonal deltas (bot vs base) — should be
     much closer to zero if atlas is localized.

Local analysis, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "results" / "cognitive_pilot_v2"
CONTRAST_DIR = ROOT / "results" / "contrast_pilot"

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
SHORT = {"Qwen2.5-7B-Instruct": "Qwen 7B",
         "Meta-Llama-3.1-8B-Instruct": "Llama 8B",
         "Mistral-7B-Instruct-v0.3": "Mistral 7B",
         "gemma-2-9b-it": "Gemma 9B"}
CONDITIONS = ["intent", "outcome", "norm_type", "morality"]


def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    adjusted = pvals * n / ranks
    # enforce monotonicity
    sorted_p = adjusted[order]
    sorted_p = np.minimum.accumulate(sorted_p[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(sorted_p, 0, 1)
    return out


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    arr = np.asarray(values)
    boots = arr[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def per_pair_deltas(record_set):
    """Return dict {pair_id: delta} from a list of pair-records."""
    return {r["pair_id"]: r["delta"] for r in record_set}


# ------------------------------------------------------------
# ANALYSIS 1: per-pair paired tests using v2 moral_decomposition data
# ------------------------------------------------------------
print("=" * 90)
print("ANALYSIS 1: Per-pair paired tests (v2 atlas, moral-vs-other-domains)")
print("=" * 90)

# Aggregate (baseline_delta - ablation_delta) per pair, per model, per condition
abl_targets = ["moral", "moral_anti", "emotion", "tom", "self", "neutral_control"]

all_pvals = {}      # (cond, target) -> [p per model]
all_effects = {}    # (cond, target) -> [Cohen's d per model]
all_shrinks = {}    # (cond, target) -> [mean shrink per model]
all_cis = {}        # (cond, target) -> [(lo, hi) per model]

for cond in CONDITIONS:
    print(f"\n--- {cond} ---")
    print(f"  {'Model':<12s} | {'target':>14s} | {'mean Δ':>8s} | {'95% CI':>17s} | "
          f"{'Cohen d':>8s} | {'paired t':>10s} | {'p-val':>10s}")
    print(f"  {'-'*12} + {'-'*14} + {'-'*8} + {'-'*17} + {'-'*8} + {'-'*10} + {'-'*10}")
    for m in MODELS:
        d = json.load(open(V2_DIR / f"{m}_moral_decomposition.json"))
        base_pairs = [r for r in d["baseline"]["pairs"] if r["condition"] == cond]
        base_deltas = {r["pair_id"]: r["delta"] for r in base_pairs}

        for tgt in abl_targets:
            if tgt not in d["ablation_results"]:
                continue
            abl_pairs = [r for r in d["ablation_results"][tgt]["pairs"]
                          if r["condition"] == cond]
            abl_deltas = {r["pair_id"]: r["delta"] for r in abl_pairs}
            # paired per-pair difference
            diffs = []
            for pid, bd in base_deltas.items():
                if pid in abl_deltas:
                    diffs.append(bd - abl_deltas[pid])
            if not diffs:
                continue
            arr = np.asarray(diffs)
            mean_d = arr.mean()
            t, p = stats.ttest_1samp(arr, 0.0)
            cohen = mean_d / (arr.std(ddof=1) + 1e-12)
            lo, hi = bootstrap_ci(arr)
            all_pvals.setdefault((cond, tgt), []).append(p)
            all_effects.setdefault((cond, tgt), []).append(cohen)
            all_shrinks.setdefault((cond, tgt), []).append(mean_d)
            all_cis.setdefault((cond, tgt), []).append((lo, hi))
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"  {SHORT[m]:<12s} | {tgt:>14s} | {mean_d:>+8.3f} | "
                  f"[{lo:>+6.2f},{hi:>+6.2f}] | {cohen:>+8.2f} | "
                  f"{t:>+10.2f} | {p:>10.3g} {sig}")

# Cross-model summary per (cond, target)
print("\n" + "=" * 90)
print("CROSS-MODEL META-ANALYSIS (per-pair test combined, BH-FDR across 4 conditions)")
print("=" * 90)
print(f"\n  {'target':>14s} | {'cond':>10s} | {'mean Δ across models':>22s} | "
      f"{'mean d':>7s} | {'min p':>9s} | {'p_adj (BH-FDR)':>15s} | replicates")
print("  " + "-" * 14 + " + " + "-" * 10 + " + " + "-" * 22 + " + " +
      "-" * 7 + " + " + "-" * 9 + " + " + "-" * 15 + " + " + "-" * 11)

# Apply BH-FDR to (cond) within each target across the 4 conds
flat_p = []
flat_keys = []
for tgt in abl_targets:
    for cond in CONDITIONS:
        key = (cond, tgt)
        if key in all_pvals:
            flat_p.append(min(all_pvals[key]))  # min across models
            flat_keys.append((tgt, cond))

flat_adj = bh_fdr(flat_p)
adj_map = {k: a for k, a in zip(flat_keys, flat_adj)}

for tgt in abl_targets:
    for cond in CONDITIONS:
        key = (cond, tgt)
        if key not in all_pvals:
            continue
        sh = all_shrinks[key]
        d = all_effects[key]
        ps = all_pvals[key]
        adj = adj_map[(tgt, cond)]
        # replicates: how many models show direction matching most-common sign
        sign = np.sign(np.mean(sh))
        rep = sum(1 for v in sh if np.sign(v) == sign)
        sig_adj = "***" if adj < 0.001 else ("**" if adj < 0.01 else ("*" if adj < 0.05 else ""))
        print(f"  {tgt:>14s} | {cond:>10s} | {np.mean(sh):>+10.3f} (±{np.std(sh):.2f}) | "
              f"{np.mean(d):>+7.2f} | {min(ps):>9.3g} | {adj:>10.3g} {sig_adj:>4s} | {rep}/4")

# ------------------------------------------------------------
# ANALYSIS 2: cross-model aggregate using contrast_pilot data
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("ANALYSIS 2: Contrast atlas diagonal effect (cross-model, one-sample test)")
print("=" * 90)

# Pull the diagonal deltas (top_cond ablation -> cond discrimination) and
# antidiagonal (bot_cond -> cond) across 4 models.
top_diag = {c: [] for c in CONDITIONS}
bot_diag = {c: [] for c in CONDITIONS}
for m in MODELS:
    d = json.load(open(CONTRAST_DIR / f"{m}_contrast_decomposition.json"))
    base = d["baseline"]
    abl = d["ablation_summaries"]
    for c in CONDITIONS:
        b = base[c]["mean"]
        top_diag[c].append(abl[f"top_{c}"][c]["mean"] - b)
        bot_diag[c].append(abl[f"bot_{c}"][c]["mean"] - b)

print(f"\n  {'Condition':>12s} | {'top diag (n=4 models)':>30s} | {'bot diag':>30s}")
print("  " + "-" * 12 + " + " + "-" * 30 + " + " + "-" * 30)
for c in CONDITIONS:
    td = top_diag[c]
    bd = bot_diag[c]
    print(f"  {c:>12s} | mean={np.mean(td):>+6.2f} std={np.std(td):.2f} values={[f'{x:+.2f}' for x in td]} | "
          f"mean={np.mean(bd):>+6.2f} std={np.std(bd):.2f}")

print(f"\n  Wilcoxon paired test (top vs bot diagonal across 4 models):")
for c in CONDITIONS:
    td, bd = np.array(top_diag[c]), np.array(bot_diag[c])
    # paired test
    if len(set(td - bd)) > 1:
        w_stat, w_p = stats.wilcoxon(td, bd, alternative="two-sided")
        # for sample of 4, also report sign
        diff = td - bd
        sign_neg = sum(1 for x in diff if x < 0)
        print(f"    {c:>12s}: top-bot = {diff.tolist()}, W={w_stat:.1f}, p={w_p:.3f}, "
              f"|top|<|bot| in {sign_neg}/4 models")

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("HEADLINE STATISTICAL VERDICT")
print("=" * 90)

key = ("norm_type", "moral")
if key in all_shrinks:
    sh = all_shrinks[key]
    d = all_effects[key]
    ps = all_pvals[key]
    print(f"\n  norm_type discrimination under MORAL ablation (v2 atlas):")
    print(f"    Mean shrink across 4 models: {np.mean(sh):+.3f} log-odds (per-pair Δ)")
    print(f"    Per-model Cohen's d: {[f'{x:+.2f}' for x in d]}, mean {np.mean(d):+.2f}")
    print(f"    Per-model p-values: {[f'{x:.2g}' for x in ps]}")
    print(f"    BH-FDR adjusted: {adj_map[('moral', 'norm_type')]:.3g}")

# Compare to neutral_control (should be ~null)
key = ("norm_type", "neutral_control")
if key in all_shrinks:
    sh = all_shrinks[key]
    d = all_effects[key]
    print(f"\n  CONTROL (neutral_control ablation) on norm_type:")
    print(f"    Mean shrink: {np.mean(sh):+.3f}")
    print(f"    Cohen's d: {[f'{x:+.2f}' for x in d]}")
