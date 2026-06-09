#!/usr/bin/env python3
"""
Template-matched compression analysis: length-controlled replication.

The original compression evidence used natural stimuli where affective
conditions average 8-13 tokens vs social 20-63 tokens. This script reruns
the identical analysis on TEMPLATE-MATCHED stimuli (4 templates × 14
conditions × 15 = 840 stimuli, identical format, ±20% length) to rule out
stimulus length as a confound.

If compression persists with matched lengths → it's representational, not
a length artifact.

CPU-only. Processes one model at a time.
"""
from __future__ import annotations
import gc, json
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
from itertools import combinations

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

AFF = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
SOC = {"belief", "intention", "judgment", "mentalizing", "moral",
       "empathy", "self_referential", "theory_of_mind"}

PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}

MODELS = list(PEAK.keys())


def pair_type(c1, c2):
    if (c1 in AFF) != (c2 in AFF):
        return "cross"
    return "within_aff" if c1 in AFF else "within_soc"


def cohens_d_along_axis(X1, X2):
    mu1, mu2 = X1.mean(0), X2.mean(0)
    diff = mu1 - mu2
    norm = np.linalg.norm(diff)
    if norm < 1e-10:
        return 0.0
    axis = diff / norm
    p1, p2 = X1 @ axis, X2 @ axis
    pooled = np.sqrt((np.var(p1, ddof=1) + np.var(p2, ddof=1)) / 2)
    return float(abs(p1.mean() - p2.mean()) / pooled) if pooled > 1e-10 else 0.0


def centroid_cos_dist(X1, X2):
    mu1, mu2 = X1.mean(0), X2.mean(0)
    cos = np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2) + 1e-10)
    return float(1 - cos)


def eta_squared(cond_acts):
    all_a = np.concatenate(list(cond_acts.values()))
    grand = all_a.mean(0)
    ss_b = sum(len(a) * np.sum((a.mean(0) - grand)**2) for a in cond_acts.values())
    ss_t = np.sum((all_a - grand)**2)
    return float(ss_b / ss_t) if ss_t > 0 else 0.0


def analyze_model(model_name):
    f = RSA / f"{model_name}_tm_rsa_v2_per_stim.npz"
    if not f.exists():
        print(f"  SKIP {model_name}: {f.name} not found")
        return None

    print(f"\n  Loading {model_name}...", flush=True)
    npz = np.load(f, allow_pickle=True)
    conditions = list(npz["conditions"])
    pool_idx = list(npz["pooling_names"]).index("mean_all")
    acts_all = npz["per_stim_activations"][pool_idx]
    n_stim, n_layers, hd = acts_all.shape
    del npz; gc.collect()

    peak = PEAK[model_name]
    print(f"  Peak: L{peak}, n_stim={n_stim}, {n_layers} layers, dim={hd}", flush=True)

    uniq = sorted(set(conditions))
    acts_peak = acts_all[:, peak, :].astype(np.float32)
    cond_stims = {c: acts_peak[[i for i, cc in enumerate(conditions) if cc == c]]
                  for c in uniq}

    # Pairwise Cohen's d + centroid distance
    pairs = list(combinations(uniq, 2))
    pair_data = []
    for c1, c2 in pairs:
        d = cohens_d_along_axis(cond_stims[c1], cond_stims[c2])
        cd = centroid_cos_dist(cond_stims[c1], cond_stims[c2])
        pair_data.append({"c1": c1, "c2": c2, "type": pair_type(c1, c2),
                          "d": d, "cos_dist": cd})

    cross = [p for p in pair_data if p["type"] == "cross"]
    w_soc = [p for p in pair_data if p["type"] == "within_soc"]
    w_aff = [p for p in pair_data if p["type"] == "within_aff"]

    d_c = [p["d"] for p in cross]
    d_s = [p["d"] for p in w_soc]
    d_a = [p["d"] for p in w_aff]
    cd_c = [p["cos_dist"] for p in cross]
    cd_s = [p["cos_dist"] for p in w_soc]
    cd_a = [p["cos_dist"] for p in w_aff]

    u_d, p_d = mannwhitneyu(d_a, d_s, alternative="less") if d_a and d_s else (0, 1)
    u_cd, p_cd = mannwhitneyu(cd_a, cd_s, alternative="less") if cd_a and cd_s else (0, 1)

    aff_acts = {c: cond_stims[c] for c in uniq if c in AFF}
    soc_acts = {c: cond_stims[c] for c in uniq if c in SOC}
    e2_a = eta_squared(aff_acts)
    e2_s = eta_squared(soc_acts)

    comp = np.mean(cd_a) / np.mean(cd_c) if np.mean(cd_c) > 0 else 0

    rng = np.random.default_rng(42)
    boot_ratios = []
    for _ in range(5000):
        ba = rng.choice(cd_a, len(cd_a), replace=True)
        bc = rng.choice(cd_c, len(cd_c), replace=True)
        if np.mean(bc) > 0:
            boot_ratios.append(np.mean(ba) / np.mean(bc))
    ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])

    # Layer-wise compression
    layer_idx = np.linspace(0, n_layers - 1, min(10, n_layers), dtype=int)
    layer_comp = []
    for li in layer_idx:
        la = acts_all[:, li, :].astype(np.float32)
        lc = {c: la[[i for i, cc in enumerate(conditions) if cc == c]] for c in uniq}
        wa = [centroid_cos_dist(lc[c1], lc[c2])
              for c1, c2 in combinations([c for c in uniq if c in AFF], 2)]
        cx = [centroid_cos_dist(lc[c1], lc[c2])
              for c1 in uniq if c1 in AFF for c2 in uniq if c2 in SOC]
        r = np.mean(wa) / np.mean(cx) if np.mean(cx) > 0 else 0
        layer_comp.append({"layer": int(li), "ratio_pct": float(r * 100)})

    del acts_all, acts_peak
    gc.collect()

    # Load original compression for comparison
    orig_file = OUT / "emotion_compression_evidence.json"
    orig = {}
    if orig_file.exists():
        with open(orig_file) as f:
            orig_data = json.load(f)
        if model_name in orig_data:
            o = orig_data[model_name]
            orig = {
                "d_aff": o["cohens_d"]["within_aff"]["mean"],
                "d_soc": o["cohens_d"]["within_soc"]["mean"],
                "eta2_aff": o["eta2"]["affective"],
                "eta2_soc": o["eta2"]["social"],
                "compression_pct": o["compression_pct"],
            }

    print(f"\n  {model_name} @ L{peak} (TEMPLATE-MATCHED):")
    print(f"    Cohen's d:     cross={np.mean(d_c):.2f}  within-soc={np.mean(d_s):.2f}  within-aff={np.mean(d_a):.2f}")
    print(f"    Centroid dist:  cross={np.mean(cd_c):.4f}  within-soc={np.mean(cd_s):.4f}  within-aff={np.mean(cd_a):.4f}")
    print(f"    MW d(aff<soc):  U={u_d:.0f}, p={p_d:.2e}")
    print(f"    η²: aff={e2_a:.5f}, soc={e2_s:.5f}, ratio={e2_s/e2_a:.1f}×")
    print(f"    Compression:    {comp*100:.2f}% [95% CI: {ci_lo*100:.2f}–{ci_hi*100:.2f}%]")

    if orig:
        print(f"\n    vs ORIGINAL (natural stimuli):")
        print(f"      d(aff):       {orig['d_aff']:.2f} → {np.mean(d_a):.2f}")
        print(f"      d(soc):       {orig['d_soc']:.2f} → {np.mean(d_s):.2f}")
        print(f"      η²(aff):      {orig['eta2_aff']:.5f} → {e2_a:.5f}")
        print(f"      η²(soc):      {orig['eta2_soc']:.5f} → {e2_s:.5f}")
        print(f"      Compression:  {orig['compression_pct']:.2f}% → {comp*100:.2f}%")

    w_aff_sorted = sorted(w_aff, key=lambda x: x["d"])
    print(f"\n    Within-affective pairs (sorted by d):")
    for p in w_aff_sorted:
        print(f"      {p['c1']:>10s}—{p['c2']:<10s}: d={p['d']:.2f}, cos_dist={p['cos_dist']:.5f}")

    return {
        "model": model_name,
        "peak_layer": peak,
        "n_stim": n_stim,
        "n_per_cond": {c: int(len(cond_stims[c])) for c in uniq},
        "cohens_d": {
            "cross": {"mean": float(np.mean(d_c)), "std": float(np.std(d_c))},
            "within_soc": {"mean": float(np.mean(d_s)), "std": float(np.std(d_s))},
            "within_aff": {"mean": float(np.mean(d_a)), "std": float(np.std(d_a))},
        },
        "centroid_dist": {
            "cross": {"mean": float(np.mean(cd_c)), "std": float(np.std(cd_c))},
            "within_soc": {"mean": float(np.mean(cd_s)), "std": float(np.std(cd_s))},
            "within_aff": {"mean": float(np.mean(cd_a)), "std": float(np.std(cd_a))},
        },
        "mann_whitney_d": {"U": float(u_d), "p": float(p_d)},
        "mann_whitney_dist": {"U": float(u_cd), "p": float(p_cd)},
        "eta2": {"affective": e2_a, "social": e2_s, "ratio": float(e2_s / e2_a) if e2_a > 0 else 0},
        "compression_pct": float(comp * 100),
        "compression_ci95": [float(ci_lo * 100), float(ci_hi * 100)],
        "layer_compression": layer_comp,
        "original_comparison": orig,
        "within_aff_pairs": [{"pair": f"{p['c1']}—{p['c2']}", "d": p["d"], "cos_dist": p["cos_dist"]}
                              for p in w_aff_sorted],
    }


def main():
    print("=" * 75)
    print("TEMPLATE-MATCHED COMPRESSION (length-controlled replication)")
    print("=" * 75)
    print("Each condition has 60 stimuli using identical sentence templates.")
    print("Length variation ±20%. This controls for stimulus length confound.\n")

    all_results = {}

    for m in MODELS:
        r = analyze_model(m)
        if r:
            all_results[m] = r

    # Summary comparison table
    print(f"\n{'='*75}")
    print("SUMMARY: Template-matched vs Original")
    print(f"{'='*75}\n")
    print(f"  {'Model':>35s} | {'d(aff)':>12s} | {'d(soc)':>12s} | {'η² ratio':>12s} | {'Compress':>12s} | {'MW p':>10s}")
    print(f"  {'':>35s} | {'TM → Orig':>12s} | {'TM → Orig':>12s} | {'TM → Orig':>12s} | {'TM → Orig':>12s} |")
    print(f"  {'-'*110}")

    for m in MODELS:
        if m not in all_results:
            continue
        r = all_results[m]
        da = r["cohens_d"]["within_aff"]["mean"]
        ds = r["cohens_d"]["within_soc"]["mean"]
        ratio = r["eta2"]["ratio"]
        comp = r["compression_pct"]
        p = r["mann_whitney_d"]["p"]
        o = r.get("original_comparison", {})

        da_o = o.get("d_aff", 0)
        ds_o = o.get("d_soc", 0)
        e_o_a = o.get("eta2_aff", 0)
        e_o_s = o.get("eta2_soc", 0)
        ratio_o = e_o_s / e_o_a if e_o_a > 0 else 0
        comp_o = o.get("compression_pct", 0)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {m:>35s} | {da:.2f} → {da_o:.2f} | {ds:.2f} → {ds_o:.2f} | {ratio:.1f}× → {ratio_o:.1f}× | {comp:.1f}% → {comp_o:.1f}% | {p:.2e} {sig}")

    # Pooled test
    all_d_aff = []
    all_d_soc = []
    for m in MODELS:
        if m not in all_results:
            continue
        r = all_results[m]
        aff_pairs = [p for p in combinations(sorted(AFF), 2)]
        soc_pairs = [p for p in combinations(sorted(SOC), 2)]
        for p_data in [{"c1": c1, "c2": c2} for c1, c2 in aff_pairs]:
            pass
    # Use stored d values
    for m in MODELS:
        if m in all_results:
            for p in all_results[m]["within_aff_pairs"]:
                all_d_aff.append(p["d"])
            # Reconstruct soc d values from the mean/count
            # Actually let's just pool from the model results directly
    # Simpler: just pool cohens_d means
    tm_aff = [all_results[m]["cohens_d"]["within_aff"]["mean"] for m in MODELS if m in all_results]
    tm_soc = [all_results[m]["cohens_d"]["within_soc"]["mean"] for m in MODELS if m in all_results]
    print(f"\n  Across 4 models (template-matched):")
    print(f"    Mean d(aff) = {np.mean(tm_aff):.2f} ± {np.std(tm_aff):.2f}")
    print(f"    Mean d(soc) = {np.mean(tm_soc):.2f} ± {np.std(tm_soc):.2f}")

    print(f"\n  CONCLUSION: Compression {'PERSISTS' if all(all_results[m]['mann_whitney_d']['p'] < 0.05 for m in MODELS if m in all_results) else 'does NOT persist'} after controlling for stimulus length.")

    json.dump(all_results, open(OUT / "template_matched_compression.json", "w"),
              indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved: {OUT / 'template_matched_compression.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
