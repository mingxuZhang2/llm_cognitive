"""
v2 RSA computation — exhaustive sweep over:
  - pooling strategy   : last_tok | mean_all | last_8_mean
  - LLM normalization  : raw | centered | feature_zscore | per_cond_zscore
  - distance           : 1_pearson | 1_cosine | euclidean
  - layer              : every layer
  - brain RDM          : v1 (Neurosynth-only) | v2 (HCP-augmented)

Also computes the LLM split-half noise ceiling per (pooling, distance, layer):
  Randomly split stimuli within each condition into halves -> two per-condition
  means -> two LLM RDMs -> Pearson between their upper-triangles. Average over
  K=50 random splits.

For each (pooling, distance, brain_rdm) combo, prints the best layer's:
  - observed Spearman ρ vs brain
  - 10K permutation p-value
  - noise ceiling estimate
  - ρ / ceiling ratio (how much of the achievable upper-bound we capture)

Input:
  --per_stim_npz       {model_short}_rsa_v2_per_stim.npz
  --brain_rdm_npz_v1   brain_rdm.npz
  --brain_rdm_npz_v2   brain_rdm_v2.npz
  --output_path        {model_short}_rsa_v2.json
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def normalize(act, mode):
    """act: [n_cond, hidden_dim]"""
    if mode == "raw":
        return act
    if mode == "centered":
        return act - act.mean(axis=0, keepdims=True)
    if mode == "feature_zscore":
        m = act.mean(axis=0, keepdims=True)
        s = act.std(axis=0, keepdims=True) + 1e-8
        return (act - m) / s
    if mode == "per_cond_zscore":
        m = act.mean(axis=1, keepdims=True)
        s = act.std(axis=1, keepdims=True) + 1e-8
        return (act - m) / s
    raise ValueError(mode)


def rdm_from(act, distance):
    """act: [n_cond, hidden_dim] -> [n_cond, n_cond] RDM."""
    if distance == "1_pearson":
        X = act - act.mean(axis=1, keepdims=True)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        Xn = X / n
        return 1.0 - Xn @ Xn.T
    if distance == "1_cosine":
        n = np.linalg.norm(act, axis=1, keepdims=True)
        n[n == 0] = 1.0
        Xn = act / n
        return 1.0 - Xn @ Xn.T
    if distance == "euclidean":
        return squareform(pdist(act, metric="euclidean"))
    raise ValueError(distance)


def triu_vals(rdm):
    n = rdm.shape[0]
    return rdm[np.triu_indices(n, k=1)]


def split_half_ceiling(per_cond_stims, layer_act, distance, normalization,
                       n_splits=50, rng=None):
    """LLM split-half noise ceiling.
    per_cond_stims: dict {cond_idx: list of stim indices}
    layer_act: [n_stim_total, hidden_dim] activations at the chosen layer/pooling
    """
    if rng is None:
        rng = np.random.default_rng(123)
    n_cond = len(per_cond_stims)
    rhos = []
    for s in range(n_splits):
        half_a = np.zeros((n_cond, layer_act.shape[1]), dtype=np.float64)
        half_b = np.zeros_like(half_a)
        for c, stim_idxs in per_cond_stims.items():
            idx = np.array(stim_idxs, dtype=np.int64)
            rng.shuffle(idx)
            cut = len(idx) // 2
            if cut < 1 or len(idx) - cut < 1:
                return float("nan")
            half_a[c] = layer_act[idx[:cut]].mean(axis=0).astype(np.float64)
            half_b[c] = layer_act[idx[cut:]].mean(axis=0).astype(np.float64)
        a_n = normalize(half_a, normalization)
        b_n = normalize(half_b, normalization)
        rdm_a = rdm_from(a_n, distance)
        rdm_b = rdm_from(b_n, distance)
        # Spearman of upper triangles
        rho, _ = spearmanr(triu_vals(rdm_a), triu_vals(rdm_b))
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def permutation_null(brain_rdm, llm_rdm, n_perm=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = brain_rdm.shape[0]
    btr = triu_vals(brain_rdm)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        permuted = llm_rdm[np.ix_(perm, perm)]
        rho, _ = spearmanr(btr, triu_vals(permuted))
        null[i] = rho
    return null


def run(per_stim_npz, brain_rdm_v1_npz, brain_rdm_v2_npz, output_path,
        n_permutations=5000, n_ceiling_splits=50):
    t0 = time.time()
    data = np.load(per_stim_npz, allow_pickle=True)
    per_stim = data["per_stim_activations"]  # [3, n_stim, n_layers+1, hidden_dim]
    conditions = list(data["conditions"])
    stim_ids = list(data["stim_ids"])
    pooling_names = list(data["pooling_names"])
    layer_names = list(data["layer_names"])
    model_short = str(data["model_short"])
    n_pool, n_stim, n_layers, hidden_dim = per_stim.shape
    print(f"v2 RSA for {model_short}: pools={n_pool}, stim={n_stim}, "
          f"layers={n_layers}, hidden={hidden_dim}")

    unique_conditions = sorted(set(conditions))
    cond_to_idx = {c: i for i, c in enumerate(unique_conditions)}
    stim_cond_idx = np.array([cond_to_idx[c] for c in conditions], dtype=np.int64)
    per_cond_stims = {c: np.where(stim_cond_idx == c)[0].tolist()
                       for c in range(len(unique_conditions))}

    # Compute condition-mean activations (per pool, per layer): [n_cond, hidden]
    cond_means = np.zeros(
        (n_pool, len(unique_conditions), n_layers, hidden_dim), dtype=np.float32)
    for ci, c_idx in enumerate(range(len(unique_conditions))):
        idxs = per_cond_stims[c_idx]
        cond_means[:, ci, :, :] = per_stim[:, idxs, :, :].mean(axis=1)

    # Load both brain RDMs
    def load_brain(p):
        d = np.load(p, allow_pickle=True)
        rdm = d["rdm"]; conds = list(d["conditions"])
        order = [conds.index(c) for c in unique_conditions]
        rdm_re = rdm[np.ix_(order, order)]
        return rdm_re

    brain_rdms = {"v1_NS_only": load_brain(brain_rdm_v1_npz),
                  "v2_HCP_aug": load_brain(brain_rdm_v2_npz)}

    NORMS = ["raw", "centered", "feature_zscore"]
    DISTS = ["1_pearson", "1_cosine"]

    results = {}
    summary_rows = []
    rng = np.random.default_rng(20260525)

    for pool_idx, pool_name in enumerate(pooling_names):
        for norm in NORMS:
            for dist in DISTS:
                for brain_name, brain_rdm in brain_rdms.items():
                    btr = triu_vals(brain_rdm)
                    per_layer = []
                    for L in range(n_layers):
                        act = cond_means[pool_idx, :, L, :].astype(np.float64)
                        a_n = normalize(act, norm)
                        llm_rdm = rdm_from(a_n, dist)
                        rho, _ = spearmanr(btr, triu_vals(llm_rdm))
                        per_layer.append({"layer": L, "rho": float(rho)})
                    # find peak
                    peak = max(per_layer, key=lambda r: r["rho"])
                    L_peak = peak["layer"]
                    act_peak = cond_means[pool_idx, :, L_peak, :].astype(np.float64)
                    a_n = normalize(act_peak, norm)
                    llm_rdm_peak = rdm_from(a_n, dist)
                    # null
                    null = permutation_null(brain_rdm, llm_rdm_peak,
                                             n_perm=n_permutations, rng=rng)
                    obs_rho = peak["rho"]
                    p_val = float(((null >= obs_rho).sum() + 1) / (n_permutations + 1))
                    # ceiling
                    ceiling = split_half_ceiling(
                        per_cond_stims,
                        per_stim[pool_idx, :, L_peak, :].astype(np.float32),
                        dist, norm,
                        n_splits=n_ceiling_splits,
                        rng=np.random.default_rng(100 + pool_idx * 13))
                    ratio = obs_rho / ceiling if ceiling and np.isfinite(ceiling) and ceiling > 0 else float("nan")
                    key = f"{pool_name}|{norm}|{dist}|{brain_name}"
                    results[key] = {
                        "pool": pool_name, "norm": norm, "dist": dist,
                        "brain_rdm": brain_name,
                        "peak_layer": L_peak, "peak_layer_name": layer_names[L_peak],
                        "rho": obs_rho, "p_value": p_val,
                        "noise_ceiling": ceiling,
                        "rho_over_ceiling": ratio,
                        "per_layer_rho": [r["rho"] for r in per_layer],
                    }
                    summary_rows.append((obs_rho, ceiling, ratio, p_val, key))

    # Rank by ρ/ceiling, then by ρ
    summary_rows.sort(key=lambda x: (-(x[2] if np.isfinite(x[2]) else x[0]), -x[0]))
    print(f"\n[Top 10 configurations by ρ/ceiling then ρ]")
    print(f"  {'pool':>10s} {'norm':>14s} {'dist':>10s} {'brain':>12s}  "
          f"L{'':>2s}  {'ρ':>7s}  {'ceil':>7s}  {'ρ/ceil':>7s}  {'p':>7s}")
    for obs_rho, ceil, ratio, p, key in summary_rows[:10]:
        r = results[key]
        print(f"  {r['pool']:>10s} {r['norm']:>14s} {r['dist']:>10s} {r['brain_rdm']:>12s}  "
              f"L{r['peak_layer']:<2d}  {r['rho']:>+7.4f}  {ceil:>7.4f}  "
              f"{ratio:>+7.3f}  {p:>7.4g}")

    out = {
        "model_short": model_short,
        "conditions": unique_conditions,
        "n_layers": n_layers,
        "n_permutations": n_permutations,
        "n_ceiling_splits": n_ceiling_splits,
        "results": results,
        "best_by_ratio_then_rho": [k for *_, k in summary_rows[:5]],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved to {output_path}  ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_stim_npz", required=True)
    ap.add_argument("--brain_rdm_v1", required=True)
    ap.add_argument("--brain_rdm_v2", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--n_permutations", type=int, default=5000)
    ap.add_argument("--n_ceiling_splits", type=int, default=50)
    args = ap.parse_args()
    run(args.per_stim_npz, args.brain_rdm_v1, args.brain_rdm_v2,
        args.output_path,
        n_permutations=args.n_permutations,
        n_ceiling_splits=args.n_ceiling_splits)


if __name__ == "__main__":
    main()
