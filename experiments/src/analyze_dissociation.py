"""Diagnostic analysis of why double dissociation failed."""
import json
import numpy as np
import sys

result_path = sys.argv[1]

with open(result_path) as f:
    d = json.load(f)

baseline = d['baseline']
ablated_math = d['ablated_math']
ablated_code = d['ablated_code']
ablated_random = d['ablated_random']

cats = sorted(baseline.keys())

print("=== DIAGNOSTIC 1: Absolute PPL increase (delta) ===")
header = f"{'Category':>12s}  {'Baseline':>8s}  {'d(math)':>8s}  {'d(code)':>8s}  {'d(rand)':>8s}"
print(header)
for c in cats:
    b = baseline[c]['perplexity']
    dm = ablated_math[c]['perplexity'] - b
    dc = ablated_code[c]['perplexity'] - b
    dr = ablated_random[c]['perplexity'] - b
    print(f"{c:>12s}  {b:>8.2f}  {dm:>+8.2f}  {dc:>+8.2f}  {dr:>+8.2f}")

print()
print("=== DIAGNOSTIC 2: Normalized damage (delta / baseline) ===")
header2 = f"{'Category':>12s}  {'Norm(math)':>10s}  {'Norm(code)':>10s}  {'Norm(rand)':>10s}"
print(header2)
for c in cats:
    b = baseline[c]['perplexity']
    nm = (ablated_math[c]['perplexity'] - b) / b
    nc = (ablated_code[c]['perplexity'] - b) / b
    nr = (ablated_random[c]['perplexity'] - b) / b
    print(f"{c:>12s}  {nm:>10.2f}  {nc:>10.2f}  {nr:>10.2f}")

print()
print("=== DIAGNOSTIC 3: Selectivity Index ===")
print("  = (target_norm_damage - avg_other_norm_damage) / (target + avg_other)")
for ablation_name, ablated, target_cat in [
    ("Math ablation", ablated_math, "math"),
    ("Code ablation", ablated_code, "code"),
]:
    b_target = baseline[target_cat]['perplexity']
    target_delta = (ablated[target_cat]['perplexity'] - b_target) / b_target
    other_deltas = []
    for c in cats:
        if c != target_cat:
            b_c = baseline[c]['perplexity']
            other_deltas.append((ablated[c]['perplexity'] - b_c) / b_c)
    avg_other = np.mean(other_deltas)
    denom = target_delta + avg_other
    selectivity = (target_delta - avg_other) / denom if denom > 0 else 0
    print(f"  {ablation_name}: target_damage={target_delta:.2f}, avg_collateral={avg_other:.2f}, selectivity={selectivity:+.3f}")

print()
print("=== DIAGNOSTIC 4: Math ablation vs Random (same neuron count) ===")
print("  If module is functional, targeted ablation should do MORE damage to target than random")
for c in cats:
    dm = ablated_math[c]['perplexity'] / baseline[c]['perplexity']
    dr = ablated_random[c]['perplexity'] / baseline[c]['perplexity']
    diff = dm - dr
    label = "LESS damage than random" if diff < 0 else "MORE damage than random"
    marker = " <-- TARGET" if c == "math" else ""
    print(f"  {c:>12s}: math_abl={dm:.2f}x, rand={dr:.2f}x, diff={diff:+.2f}  ({label}){marker}")

print()
print("=== DIAGNOSTIC 5: Damage ranking from MATH module ablation ===")
print("  If module is math-specific, 'math' should rank #1")
damages = []
for c in cats:
    b = baseline[c]['perplexity']
    norm_d = (ablated_math[c]['perplexity'] - b) / b
    damages.append((c, norm_d))
damages.sort(key=lambda x: -x[1])
for rank, (c, d_val) in enumerate(damages, 1):
    marker = " <-- TARGET" if c == "math" else ""
    print(f"  #{rank} {c:>12s}: +{d_val:.0%}{marker}")

print()
print("=== DIAGNOSTIC 6: Damage ranking from CODE module ablation ===")
damages_c = []
for c in cats:
    b = baseline[c]['perplexity']
    norm_d = (ablated_code[c]['perplexity'] - b) / b
    damages_c.append((c, norm_d))
damages_c.sort(key=lambda x: -x[1])
for rank, (c, d_val) in enumerate(damages_c, 1):
    marker = " <-- TARGET" if c == "code" else ""
    print(f"  #{rank} {c:>12s}: +{d_val:.0%}{marker}")

print()
print("=== DIAGNOSTIC 7: Log-PPL analysis (more meaningful for low-baseline categories) ===")
print("  Using avg_loss (log-PPL) which is additive and baseline-independent")
header3 = f"{'Category':>12s}  {'base_logPPL':>11s}  {'d(math)':>8s}  {'d(code)':>8s}  {'d(rand)':>8s}"
print(header3)
for c in cats:
    bl = baseline[c].get('avg_loss', 0) or 0
    ml = ablated_math[c].get('avg_loss', 0) or 0
    cl = ablated_code[c].get('avg_loss', 0) or 0
    rl = ablated_random[c].get('avg_loss', 0) or 0
    print(f"  {c:>12s}  {bl:>11.3f}  {ml-bl:>+8.3f}  {cl-bl:>+8.3f}  {rl-bl:>+8.3f}")

print()
print("=== SUMMARY ===")
# Check the key conditions
math_on_math = d['dissociation_ratios']['math_ablation_on_math']
math_on_code = d['dissociation_ratios']['math_ablation_on_code']
code_on_code = d['dissociation_ratios']['code_ablation_on_code']
code_on_math = d['dissociation_ratios']['code_ablation_on_math']

print(f"  Condition 1 (math_abl→math > math_abl→code * 1.5): {math_on_math:.2f} > {math_on_code*1.5:.2f} → {'PASS' if math_on_math > math_on_code * 1.5 else 'FAIL'}")
print(f"  Condition 2 (code_abl→code > code_abl→math * 1.5): {code_on_code:.2f} > {code_on_math*1.5:.2f} → {'PASS' if code_on_code > code_on_math * 1.5 else 'FAIL'}")
print()

n_math = d['module_map']['math']['n_neurons']
n_code = d['module_map']['code']['n_neurons']
total = n_math + n_code  # approximate from pct
pct_math = d['module_map']['math']['pct_neurons']
pct_code = d['module_map']['code']['pct_neurons']
print(f"  Module sizes: math={n_math:,} ({pct_math:.1%}), code={n_code:,} ({pct_code:.1%})")
print(f"  Combined: {pct_math+pct_code:.1%} of all neurons")
print()
print("  Key failure modes identified:")
print("  1. Ablating code module damages math MORE than ablating math module (5.29x vs 4.75x)")
print("  2. Random ablation damages math MORE than either targeted ablation (6.70x)")
print("  3. Math ablation damages ethics(+303%) and factual_qa(+173%) more than math(+375% but from tiny baseline)")
print("  4. Modules are ~15% of network each — ablation is too coarse for specificity")
