#!/usr/bin/env python3
"""
Bulletproof evidence for LLM within-affective compression.

Uses per-STIMULUS activations (712 stimuli, ~51 per condition) to show:
1. Cohen's d: within-aff << within-social (stimulus-level, not centroid-level)
2. η²: affective between-condition variance << social between-condition variance
3. Cross-architecture consistency (4 architectures, all p < ?)
4. Cross-scale consistency (Qwen 0.5B → 7B)
5. Layer-wise: compression at EVERY layer, not a layer artifact

Processes one model at a time to manage memory (~600MB per model).
CPU-only.
"""
from __future__ import annotations
import gc, json, sys
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

MAIN_MODELS = list(PEAK.keys())
SCALE_MODELS = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]


def pair_type(c1, c2):
    if (c1 in AFF) != (c2 in AFF):
        return "cross"
    return "within_aff" if c1 in AFF else "within_soc"


def cohens_d_along_axis(X1, X2):
    """1D Cohen's d projected onto the axis connecting group centroids."""
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
    """Between-condition η² (MANOVA-style, trace-based)."""
    all_a = np.concatenate(list(cond_acts.values()))
    grand = all_a.mean(0)
    ss_b = sum(len(a) * np.sum((a.mean(0) - grand)**2) for a in cond_acts.values())
    ss_t = np.sum((all_a - grand)**2)
    return float(ss_b / ss_t) if ss_t > 0 else 0.0


def find_peak(acts_3d, conditions):
    """Find layer maximizing brain RSA."""
    brain = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_rdm, brain_conds = brain["rdm"], list(brain["conditions"])
    brain_ut = brain_rdm[np.triu_indices(14, k=1)]
    uniq = sorted(set(conditions))

    best_rho, best_l = -1, 0
    for li in range(acts_3d.shape[1]):
        a = acts_3d[:, li, :].astype(np.float32)
        centroids = {c: a[[i for i, cc in enumerate(conditions) if cc == c]].mean(0) for c in uniq}
        gm = np.mean(list(centroids.values()), 0)
        for c in centroids:
            centroids[c] = centroids[c] - gm
        rdm = np.zeros((14, 14))
        for i, ci in enumerate(brain_conds):
            for j, cj in enumerate(brain_conds):
                if i < j:
                    n1, n2 = np.linalg.norm(centroids[ci]), np.linalg.norm(centroids[cj])
                    d = 1 - np.dot(centroids[ci], centroids[cj]) / (n1 * n2 + 1e-10)
                    rdm[i, j] = rdm[j, i] = d
        rho, _ = spearmanr(brain_ut, rdm[np.triu_indices(14, k=1)])
        if rho > best_rho:
            best_rho, best_l = rho, li
    return best_l, float(best_rho)


def analyze_model(model_name):
    """Full analysis for one model. Returns metrics dict."""
    f = RSA / f"{model_name}_rsa_v2_per_stim.npz"
    if not f.exists():
        print(f"  SKIP {model_name}: file not found")
        return None

    print(f"\n  Loading {model_name}...", flush=True)
    npz = np.load(f, allow_pickle=True)
    conditions = list(npz["conditions"])
    pool_idx = list(npz["pooling_names"]).index("mean_all")
    acts_all = npz["per_stim_activations"][pool_idx]  # (n_stim, n_layers, hidden)
    n_stim, n_layers, hd = acts_all.shape
    del npz; gc.collect()

    # Peak layer
    peak = PEAK.get(model_name)
    if peak is None:
        peak, peak_rho = find_peak(acts_all, conditions)
        print(f"  Peak: L{peak} (ρ={peak_rho:.3f})", flush=True)
    else:
        print(f"  Peak: L{peak} (preset)", flush=True)

    uniq = sorted(set(conditions))
    acts_peak = acts_all[:, peak, :].astype(np.float32)
    cond_stims = {c: acts_peak[[i for i, cc in enumerate(conditions) if cc == c]]
                  for c in uniq}

    # ── 1. Pairwise Cohen's d + centroid distance ──
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

    # Mann-Whitney: d(within-aff) < d(within-soc)?
    u_d, p_d = mannwhitneyu(d_a, d_s, alternative="less") if d_a and d_s else (0, 1)
    u_cd, p_cd = mannwhitneyu(cd_a, cd_s, alternative="less") if cd_a and cd_s else (0, 1)

    # ── 2. η² per block ──
    aff_acts = {c: cond_stims[c] for c in uniq if c in AFF}
    soc_acts = {c: cond_stims[c] for c in uniq if c in SOC}
    e2_a = eta_squared(aff_acts)
    e2_s = eta_squared(soc_acts)

    # ── 3. Compression ratio ──
    comp = np.mean(cd_a) / np.mean(cd_c) if np.mean(cd_c) > 0 else 0

    # Bootstrap CI on compression ratio
    rng = np.random.default_rng(42)
    boot_ratios = []
    for _ in range(5000):
        ba = rng.choice(cd_a, len(cd_a), replace=True)
        bc = rng.choice(cd_c, len(cd_c), replace=True)
        if np.mean(bc) > 0:
            boot_ratios.append(np.mean(ba) / np.mean(bc))
    ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])

    # ── 4. Layer-wise compression (10 sampled layers) ──
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

    # ── Print ──
    print(f"\n  {model_name} @ L{peak}:")
    print(f"    Cohen's d:    cross={np.mean(d_c):.1f}  within-soc={np.mean(d_s):.1f}  within-aff={np.mean(d_a):.1f}")
    print(f"    Centroid dist: cross={np.mean(cd_c):.4f}  within-soc={np.mean(cd_s):.4f}  within-aff={np.mean(cd_a):.4f}")
    print(f"    MW d(aff<soc): U={u_d:.0f}, p={p_d:.6f}")
    print(f"    MW cos(aff<soc): U={u_cd:.0f}, p={p_cd:.6f}")
    print(f"    η²: aff={e2_a:.5f}, soc={e2_s:.5f}, ratio={e2_s/e2_a:.1f}×")
    print(f"    Compression: {comp*100:.2f}% [95% CI: {ci_lo*100:.2f}–{ci_hi*100:.2f}%]")
    print(f"    Layer-wise: {', '.join(f'L{lc['layer']}={lc['ratio_pct']:.1f}%' for lc in layer_comp)}", flush=True)

    # Within-aff pair details
    print(f"\n    Within-affective pairs (sorted by d):")
    w_aff_sorted = sorted(w_aff, key=lambda x: x["d"])
    for p in w_aff_sorted:
        print(f"      {p['c1']:>10s}—{p['c2']:<10s}: d={p['d']:.2f}, cos_dist={p['cos_dist']:.5f}")

    return {
        "model": model_name,
        "peak_layer": peak,
        "n_stim": n_stim,
        "n_per_cond": {c: int(len(cond_stims[c])) for c in uniq},
        "cohens_d": {
            "cross": {"mean": float(np.mean(d_c)), "std": float(np.std(d_c)), "all": d_c},
            "within_soc": {"mean": float(np.mean(d_s)), "std": float(np.std(d_s)), "all": d_s},
            "within_aff": {"mean": float(np.mean(d_a)), "std": float(np.std(d_a)), "all": d_a},
        },
        "centroid_dist": {
            "cross": {"mean": float(np.mean(cd_c)), "std": float(np.std(cd_c))},
            "within_soc": {"mean": float(np.mean(cd_s)), "std": float(np.std(cd_s))},
            "within_aff": {"mean": float(np.mean(cd_a)), "std": float(np.std(cd_a))},
        },
        "mann_whitney_d": {"U": float(u_d), "p": float(p_d)},
        "mann_whitney_dist": {"U": float(u_cd), "p": float(p_cd)},
        "eta2": {"affective": e2_a, "social": e2_s},
        "compression_pct": float(comp * 100),
        "compression_ci95": [float(ci_lo * 100), float(ci_hi * 100)],
        "layer_compression": layer_comp,
        "within_aff_pairs": [{"pair": f"{p['c1']}—{p['c2']}", "d": p["d"], "cos_dist": p["cos_dist"]}
                              for p in w_aff_sorted],
    }


def main():
    print("=" * 75)
    print("BULLETPROOF EVIDENCE: LLM Within-Affective Compression")
    print("=" * 75)

    all_results = {}

    # ── Main 4 architectures ──
    print(f"\n{'='*75}")
    print("SECTION 1: Cross-architecture (4 models, 7-9B)")
    print(f"{'='*75}")

    for m in MAIN_MODELS:
        r = analyze_model(m)
        if r:
            all_results[m] = r

    # ── Qwen scale series ──
    print(f"\n{'='*75}")
    print("SECTION 2: Cross-scale (Qwen 0.5B → 7B)")
    print(f"{'='*75}")

    for m in SCALE_MODELS:
        if m not in all_results:
            r = analyze_model(m)
            if r:
                all_results[m] = r

    # ═══ SUMMARY TABLE ═══
    print(f"\n{'='*75}")
    print("SUMMARY TABLE")
    print(f"{'='*75}\n")

    print(f"  {'Model':>35s} | {'d(cross)':>8s} {'d(soc)':>7s} {'d(aff)':>7s} | {'p(a<s)':>10s} | {'η²(aff)':>8s} {'η²(soc)':>8s} | {'Comp%':>6s}")
    print(f"  {'-'*105}")
    for m in list(PEAK.keys()) + [m for m in SCALE_MODELS if m not in PEAK]:
        if m not in all_results:
            continue
        r = all_results[m]
        dc = r["cohens_d"]["cross"]["mean"]
        ds = r["cohens_d"]["within_soc"]["mean"]
        da = r["cohens_d"]["within_aff"]["mean"]
        p = r["mann_whitney_d"]["p"]
        ea = r["eta2"]["affective"]
        es = r["eta2"]["social"]
        cp = r["compression_pct"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {m:>35s} | {dc:8.1f} {ds:7.1f} {da:7.1f} | {p:8.6f}{sig:>2s} | {ea:8.5f} {es:8.5f} | {cp:6.2f}")

    # Cross-architecture consistency test
    print(f"\n  Cross-architecture consistency:")
    all_d_aff = []
    all_d_soc = []
    for m in MAIN_MODELS:
        if m in all_results:
            all_d_aff.extend(all_results[m]["cohens_d"]["within_aff"]["all"])
            all_d_soc.extend(all_results[m]["cohens_d"]["within_soc"]["all"])
    if all_d_aff and all_d_soc:
        u, p = mannwhitneyu(all_d_aff, all_d_soc, alternative="less")
        print(f"    Pooled across 4 architectures: MW U={u:.0f}, p={p:.2e}")
        print(f"    d(aff) mean={np.mean(all_d_aff):.2f}, d(soc) mean={np.mean(all_d_soc):.2f}")

    # Save
    json.dump(all_results, open(OUT / "emotion_compression_evidence.json", "w"),
              indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved: {OUT / 'emotion_compression_evidence.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
