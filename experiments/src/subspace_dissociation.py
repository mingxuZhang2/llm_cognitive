#!/usr/bin/env python3
"""
Subspace double-dissociation: project out emotion / social SUBSPACES from
per-stimulus hidden-state activations and measure selective collapse of
brain-LLM alignment.

Motivation: the neuron-level ablation (clinical_dissociation.py) was too coarse
-- 45-63% neuron-set overlap, 1/4 models confirmed, non-significant p-values.
This script operates in REPRESENTATION SPACE, consistent with our headline
mechanistic finding (removing PC1 = the emotion-social boundary inverts rho
from +0.73 to -0.36).

Method:
  1. Load per-stim activations from {model}_rsa_v2_per_stim.npz (mean_all pooled,
     peak layer).
  2. Compute condition centroids (14 x hidden_dim).
  3. Define emotion subspace = top-k PCs of the 6 affective centroids (centered
     among themselves).
  4. Define social subspace = top-k PCs of the 8 social centroids (centered
     among themselves).
  5. For each ablation type (project out emotion / social / random):
     - Apply projection removal to ALL per-stim activations
     - Re-compute condition centroids from projected activations
     - Re-center, cosine distance -> new RDM
     - Spearman rho vs brain RDM (full, within-affective, within-social)
  6. The double dissociation:
     - Project out emotion subspace -> within-aff rho drops, within-soc survives
     - Project out social subspace -> within-soc rho drops, within-aff survives

Runs on CPU (loads one model's per-stim NPZ at a time, ~5-6GB float64 peak).
Output: results/clinical_dissociation/subspace_dissociation.json
        figures/subspace_dissociation.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "clinical_dissociation"
FIG_DIR = BASE / "figures"

# ── Model configs ──────────────────────────────────────────────────────
MODELS = {
    "Qwen2.5-7B-Instruct":          {"peak_layer": 27},
    "Meta-Llama-3.1-8B-Instruct":   {"peak_layer": 31},
    "Mistral-7B-Instruct-v0.3":     {"peak_layer": 14},
    "gemma-2-9b-it":                 {"peak_layer": 21},
}

AFFECTIVE = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
SOCIAL = ["belief", "mentalizing", "intention", "theory_of_mind",
          "empathy", "self_referential", "judgment", "moral"]
ALL_CONDS = sorted(set(AFFECTIVE + SOCIAL))  # 14

N_RANDOM = 100      # random subspace null repeats
K_VALUES = [1, 2, 3, 4, 5]   # subspace dims to sweep
K_DEFAULT = 3       # the main reported value
SEED = 2026


# ── Core utilities (reused from rsa_deep_analysis.py) ──────────────────
def normalize_centered(act):
    """Center across conditions (axis 0)."""
    return act - act.mean(axis=0, keepdims=True)


def rdm_cosine(act):
    """1 - cosine similarity matrix."""
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    xn = act / norms
    return 1.0 - xn @ xn.T


def utri(mat):
    """Upper-triangle vector."""
    return mat[np.triu_indices(mat.shape[0], k=1)]


def spearman_rho(a, b):
    """Spearman rho between two vectors."""
    if len(a) < 3:
        return np.nan
    r, _ = spearmanr(a, b)
    return r


def pca_subspace(points, k):
    """Top-k PCs of a (n, d) matrix (centered internally).

    Returns (k, d) orthonormal basis.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:k]  # (k, d) — each row is a PC direction


def project_out(X, basis):
    """Remove projection of X onto subspace spanned by rows of basis.

    X: (n, d), basis: (k, d) orthonormal rows.
    Returns (n, d) with the subspace removed.
    """
    # P = basis.T @ basis  (d, d) projection matrix
    # X_proj = X - X @ P = X - (X @ basis.T) @ basis
    coeff = X @ basis.T          # (n, k)
    return X - coeff @ basis     # (n, d)


def random_orthonormal_basis(d, k, rng):
    """Random k-dimensional subspace in R^d (orthonormal)."""
    Z = rng.standard_normal((k, d))
    Q, _ = np.linalg.qr(Z.T)    # (d, k) orthonormal columns
    return Q.T[:k]               # (k, d) rows


# ── Load model data ────────────────────────────────────────────────────
def load_model_data(model_name, brain_conds):
    """Load per-stim activations at peak layer, return stim acts + condition indices.

    Returns dict with:
      stim_acts: (n_stim, hidden_dim) float64
      stim_cond_idx: (n_stim,) int — index into brain_conds ordering
      model: str
      peak_layer: int
      hidden_dim: int
    """
    peak_L = MODELS[model_name]["peak_layer"]
    npz_path = RSA_DIR / f"{model_name}_rsa_v2_per_stim.npz"
    if not npz_path.exists():
        print(f"  [SKIP] {model_name}: {npz_path} not found")
        return None

    print(f"  Loading {model_name} ({npz_path.stat().st_size / 1e6:.0f} MB)...", flush=True)
    t0 = time.time()
    data = np.load(npz_path, allow_pickle=True)
    per_stim = data["per_stim_activations"]      # (pool, stim, layer, hidden), float16
    conditions = list(data["conditions"])
    pools = list(data["pooling_names"])
    pidx = pools.index("mean_all")

    # Extract peak layer, mean_all pooling, cast to float64
    stim_acts = per_stim[pidx, :, peak_L, :].astype(np.float64)  # (n_stim, hidden)
    del per_stim, data  # free memory

    # Map stim conditions to brain_conds ordering
    unique_conds = sorted(set(conditions))
    cond_name_to_idx = {c: brain_conds.index(c) for c in unique_conds}
    stim_cond_idx = np.array([cond_name_to_idx[c] for c in conditions])

    elapsed = time.time() - t0
    print(f"    n_stim={stim_acts.shape[0]}, hidden={stim_acts.shape[1]}, "
          f"loaded in {elapsed:.1f}s", flush=True)

    return {
        "model": model_name,
        "peak_layer": peak_L,
        "hidden_dim": stim_acts.shape[1],
        "stim_acts": stim_acts,
        "stim_cond_idx": stim_cond_idx,
    }


# ── Compute RDM from (projected) per-stim acts ────────────────────────
def compute_rdm_from_stim(stim_acts, stim_cond_idx, n_conds):
    """Condition centroids -> center -> cosine RDM."""
    centroids = np.zeros((n_conds, stim_acts.shape[1]), dtype=np.float64)
    for c in range(n_conds):
        mask = stim_cond_idx == c
        if mask.any():
            centroids[c] = stim_acts[mask].mean(axis=0)
    return rdm_cosine(normalize_centered(centroids))


# ── Evaluate one ablation ──────────────────────────────────────────────
def evaluate_ablation(stim_acts_proj, stim_cond_idx, brain_rdm,
                      aff_idx, soc_idx, n_conds):
    """Compute full/within-aff/within-soc rho after projection."""
    rdm = compute_rdm_from_stim(stim_acts_proj, stim_cond_idx, n_conds)
    brain_vec = utri(brain_rdm)
    llm_vec = utri(rdm)

    full_rho = spearman_rho(brain_vec, llm_vec)

    # Within-affective sub-RDM
    brain_aff = brain_rdm[np.ix_(aff_idx, aff_idx)]
    llm_aff = rdm[np.ix_(aff_idx, aff_idx)]
    within_aff_rho = spearman_rho(utri(brain_aff), utri(llm_aff))

    # Within-social sub-RDM
    brain_soc = brain_rdm[np.ix_(soc_idx, soc_idx)]
    llm_soc = rdm[np.ix_(soc_idx, soc_idx)]
    within_soc_rho = spearman_rho(utri(brain_soc), utri(llm_soc))

    # Cross-block pairs
    cross_pairs_brain = []
    cross_pairs_llm = []
    for i in aff_idx:
        for j in soc_idx:
            cross_pairs_brain.append(brain_rdm[i, j])
            cross_pairs_llm.append(rdm[i, j])
    cross_rho = spearman_rho(np.array(cross_pairs_brain), np.array(cross_pairs_llm))

    return {
        "full_rho": float(full_rho),
        "within_aff_rho": float(within_aff_rho),
        "within_soc_rho": float(within_soc_rho),
        "cross_rho": float(cross_rho),
        "rdm": rdm,
    }


# ── Main per-model analysis ───────────────────────────────────────────
def analyze_model(md, brain_rdm, brain_conds):
    """Run the full subspace dissociation for one model."""
    model_name = md["model"]
    stim_acts = md["stim_acts"]
    stim_cond_idx = md["stim_cond_idx"]
    n_conds = len(brain_conds)
    hidden_dim = stim_acts.shape[1]

    # Condition indices in brain_conds ordering
    aff_idx = np.array([brain_conds.index(c) for c in AFFECTIVE])
    soc_idx = np.array([brain_conds.index(c) for c in SOCIAL])

    # Compute condition centroids for defining subspaces
    centroids = np.zeros((n_conds, hidden_dim), dtype=np.float64)
    for c in range(n_conds):
        mask = stim_cond_idx == c
        if mask.any():
            centroids[c] = stim_acts[mask].mean(axis=0)

    aff_centroids = centroids[aff_idx]   # (6, hidden)
    soc_centroids = centroids[soc_idx]   # (8, hidden)

    rng = np.random.default_rng(SEED)
    result = {"model": model_name, "peak_layer": md["peak_layer"],
              "hidden_dim": hidden_dim, "n_stim": stim_acts.shape[0]}

    # ── Baseline (no projection) ──
    print(f"\n  {model_name}: baseline...", flush=True)
    baseline = evaluate_ablation(stim_acts, stim_cond_idx, brain_rdm,
                                 aff_idx, soc_idx, n_conds)
    result["baseline"] = {k: v for k, v in baseline.items() if k != "rdm"}
    print(f"    baseline: full={baseline['full_rho']:.3f}, "
          f"within-aff={baseline['within_aff_rho']:.3f}, "
          f"within-soc={baseline['within_soc_rho']:.3f}")

    # ── Sweep over k values ──
    k_sweep = {}
    for k in K_VALUES:
        print(f"  k={k}:", flush=True)
        emo_basis = pca_subspace(aff_centroids, min(k, len(aff_idx) - 1))
        soc_basis = pca_subspace(soc_centroids, min(k, len(soc_idx) - 1))

        # Subspace angle between emotion and social subspaces
        # (principal angles via SVD of cross-product)
        S_cross = np.linalg.svd(emo_basis @ soc_basis.T, compute_uv=False)
        principal_angles = np.arccos(np.clip(S_cross, -1, 1))
        min_angle_deg = float(np.degrees(principal_angles.min())) if len(principal_angles) > 0 else 90.0

        # Project out emotion subspace
        stim_proj_emo = project_out(stim_acts, emo_basis)
        eval_emo = evaluate_ablation(stim_proj_emo, stim_cond_idx, brain_rdm,
                                     aff_idx, soc_idx, n_conds)

        # Project out social subspace
        stim_proj_soc = project_out(stim_acts, soc_basis)
        eval_soc = evaluate_ablation(stim_proj_soc, stim_cond_idx, brain_rdm,
                                     aff_idx, soc_idx, n_conds)

        # Random subspace null (N_RANDOM repeats)
        rand_fulls = np.zeros(N_RANDOM)
        rand_affs = np.zeros(N_RANDOM)
        rand_socs = np.zeros(N_RANDOM)
        actual_k = emo_basis.shape[0]  # may be < k if fewer conditions
        for ri in range(N_RANDOM):
            rand_basis = random_orthonormal_basis(hidden_dim, actual_k, rng)
            stim_proj_rand = project_out(stim_acts, rand_basis)
            eval_rand = evaluate_ablation(stim_proj_rand, stim_cond_idx, brain_rdm,
                                          aff_idx, soc_idx, n_conds)
            rand_fulls[ri] = eval_rand["full_rho"]
            rand_affs[ri] = eval_rand["within_aff_rho"]
            rand_socs[ri] = eval_rand["within_soc_rho"]

        k_sweep[k] = {
            "k": k, "actual_k": int(actual_k),
            "min_angle_deg": min_angle_deg,
            "project_out_emotion": {kk: v for kk, v in eval_emo.items() if kk != "rdm"},
            "project_out_social": {kk: v for kk, v in eval_soc.items() if kk != "rdm"},
            "random_null": {
                "full_rho_mean": float(rand_fulls.mean()),
                "full_rho_std": float(rand_fulls.std()),
                "full_rho_p95": float(np.percentile(rand_fulls, 95)),
                "within_aff_rho_mean": float(rand_affs.mean()),
                "within_aff_rho_std": float(rand_affs.std()),
                "within_soc_rho_mean": float(rand_socs.mean()),
                "within_soc_rho_std": float(rand_socs.std()),
            },
        }

        emo_r = eval_emo
        soc_r = eval_soc
        print(f"    proj-out-emo: full={emo_r['full_rho']:.3f}, "
              f"w-aff={emo_r['within_aff_rho']:.3f}, w-soc={emo_r['within_soc_rho']:.3f}")
        print(f"    proj-out-soc: full={soc_r['full_rho']:.3f}, "
              f"w-aff={soc_r['within_aff_rho']:.3f}, w-soc={soc_r['within_soc_rho']:.3f}")
        print(f"    random null:  full={rand_fulls.mean():.3f}+/-{rand_fulls.std():.3f}, "
              f"w-aff={rand_affs.mean():.3f}, w-soc={rand_socs.mean():.3f}")
        print(f"    subspace angle: {min_angle_deg:.1f} deg")

    result["k_sweep"] = k_sweep

    # ── Headline result at K_DEFAULT ──
    headline = k_sweep[K_DEFAULT]
    emo_h = headline["project_out_emotion"]
    soc_h = headline["project_out_social"]
    rand_h = headline["random_null"]

    # Double dissociation test at k=K_DEFAULT
    # Criterion 1: projecting out emotion hurts within-aff more than within-soc
    emo_aff_drop = baseline["within_aff_rho"] - emo_h["within_aff_rho"]
    emo_soc_drop = baseline["within_soc_rho"] - emo_h["within_soc_rho"]
    # Criterion 2: projecting out social hurts within-soc more than within-aff
    soc_soc_drop = baseline["within_soc_rho"] - soc_h["within_soc_rho"]
    soc_aff_drop = baseline["within_aff_rho"] - soc_h["within_aff_rho"]

    dd_confirmed = (emo_aff_drop > emo_soc_drop) and (soc_soc_drop > soc_aff_drop)

    # Permutation p-values: is the selective drop more extreme than random?
    # For proj-out-emotion: compare within-aff rho to random null of within-aff
    k_entry = k_sweep[K_DEFAULT]
    # Recompute using the stored random data
    # p(within_aff after emo projection <= random): how often random within-aff is
    # as bad or worse than the emotion-projection within-aff
    # Actually, we need the per-trial random values, let me recompute the p-value
    # from the stored stats using a z-score approximation
    # Better: just rerun the random loop for k=K_DEFAULT and store individual values

    # Store the detailed headline
    result["headline_k"] = K_DEFAULT
    result["double_dissociation"] = {
        "confirmed": dd_confirmed,
        "proj_out_emotion": {
            "within_aff_drop": float(emo_aff_drop),
            "within_soc_drop": float(emo_soc_drop),
            "selective": bool(emo_aff_drop > emo_soc_drop),
        },
        "proj_out_social": {
            "within_soc_drop": float(soc_soc_drop),
            "within_aff_drop": float(soc_aff_drop),
            "selective": bool(soc_soc_drop > soc_aff_drop),
        },
    }

    # Free stim_acts memory
    del stim_acts
    return result


# ── Permutation test at k=K_DEFAULT (detailed, stored per-trial) ──────
def permutation_test(md, brain_rdm, brain_conds, k=K_DEFAULT, n_perm=10000):
    """Condition-label permutation test for the subspace dissociation.

    Shuffles condition labels -> re-defines emotion/social subspaces ->
    re-projects -> measures whether the observed selectivity (asymmetry
    in drop) exceeds chance.
    """
    stim_acts = md["stim_acts"]
    stim_cond_idx = md["stim_cond_idx"]
    n_conds = len(brain_conds)
    hidden_dim = stim_acts.shape[1]
    aff_idx = np.array([brain_conds.index(c) for c in AFFECTIVE])
    soc_idx = np.array([brain_conds.index(c) for c in SOCIAL])

    # Observed values
    centroids = np.zeros((n_conds, hidden_dim), dtype=np.float64)
    for c in range(n_conds):
        mask = stim_cond_idx == c
        if mask.any():
            centroids[c] = stim_acts[mask].mean(axis=0)

    # Define true subspaces
    emo_basis = pca_subspace(centroids[aff_idx], min(k, len(aff_idx) - 1))
    soc_basis = pca_subspace(centroids[soc_idx], min(k, len(soc_idx) - 1))

    # Observed: project out emotion
    stim_proj_emo = project_out(stim_acts, emo_basis)
    eval_emo_obs = evaluate_ablation(stim_proj_emo, stim_cond_idx, brain_rdm,
                                     aff_idx, soc_idx, n_conds)
    # Observed: project out social
    stim_proj_soc = project_out(stim_acts, soc_basis)
    eval_soc_obs = evaluate_ablation(stim_proj_soc, stim_cond_idx, brain_rdm,
                                     aff_idx, soc_idx, n_conds)

    # Observed selectivity indices
    baseline = evaluate_ablation(stim_acts, stim_cond_idx, brain_rdm,
                                 aff_idx, soc_idx, n_conds)
    obs_emo_selectivity = ((baseline["within_aff_rho"] - eval_emo_obs["within_aff_rho"])
                           - (baseline["within_soc_rho"] - eval_emo_obs["within_soc_rho"]))
    obs_soc_selectivity = ((baseline["within_soc_rho"] - eval_soc_obs["within_soc_rho"])
                           - (baseline["within_aff_rho"] - eval_soc_obs["within_aff_rho"]))

    # Permutation null: shuffle which conditions are "affective" vs "social"
    rng = np.random.default_rng(SEED + 1)
    perm_emo_sel = np.zeros(n_perm)
    perm_soc_sel = np.zeros(n_perm)

    all_idx = np.arange(n_conds)
    n_aff = len(aff_idx)

    for pi in range(n_perm):
        shuffled = rng.permutation(all_idx)
        perm_aff = shuffled[:n_aff]
        perm_soc = shuffled[n_aff:]

        perm_aff_centroids = centroids[perm_aff]
        perm_soc_centroids = centroids[perm_soc]

        perm_emo_basis = pca_subspace(perm_aff_centroids, min(k, n_aff - 1))
        perm_soc_basis = pca_subspace(perm_soc_centroids, min(k, len(perm_soc) - 1))

        # Project out "emotion" subspace (permuted labels)
        sp_emo = project_out(stim_acts, perm_emo_basis)
        e_emo = evaluate_ablation(sp_emo, stim_cond_idx, brain_rdm,
                                  aff_idx, soc_idx, n_conds)
        # Project out "social" subspace (permuted labels)
        sp_soc = project_out(stim_acts, perm_soc_basis)
        e_soc = evaluate_ablation(sp_soc, stim_cond_idx, brain_rdm,
                                  aff_idx, soc_idx, n_conds)

        perm_emo_sel[pi] = ((baseline["within_aff_rho"] - e_emo["within_aff_rho"])
                            - (baseline["within_soc_rho"] - e_emo["within_soc_rho"]))
        perm_soc_sel[pi] = ((baseline["within_soc_rho"] - e_soc["within_soc_rho"])
                            - (baseline["within_aff_rho"] - e_soc["within_aff_rho"]))

    p_emo = float((np.sum(perm_emo_sel >= obs_emo_selectivity) + 1) / (n_perm + 1))
    p_soc = float((np.sum(perm_soc_sel >= obs_soc_selectivity) + 1) / (n_perm + 1))

    return {
        "obs_emo_selectivity": float(obs_emo_selectivity),
        "obs_soc_selectivity": float(obs_soc_selectivity),
        "p_emo": p_emo,
        "p_soc": p_soc,
        "perm_emo_sel_mean": float(perm_emo_sel.mean()),
        "perm_emo_sel_std": float(perm_emo_sel.std()),
        "perm_soc_sel_mean": float(perm_soc_sel.mean()),
        "perm_soc_sel_std": float(perm_soc_sel.std()),
    }


# ── Visualization ──────────────────────────────────────────────────────
def make_figure(all_results, brain_conds):
    """Multi-panel figure summarizing the subspace dissociation."""
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    model_short = {
        "Qwen2.5-7B-Instruct": "Qwen-7B",
        "Meta-Llama-3.1-8B-Instruct": "Llama-8B",
        "Mistral-7B-Instruct-v0.3": "Mistral-7B",
        "gemma-2-9b-it": "Gemma-9B",
    }
    model_colors = {
        "Qwen2.5-7B-Instruct": "#e74c3c",
        "Meta-Llama-3.1-8B-Instruct": "#3498db",
        "Mistral-7B-Instruct-v0.3": "#27ae60",
        "gemma-2-9b-it": "#9b59b6",
    }

    k = K_DEFAULT

    # ── Panel A: dissociation bar chart (4 models x 3 conditions) ──
    ax = axes[0, 0]
    x = np.arange(n_models)
    w = 0.2
    metrics = ["full_rho", "within_aff_rho", "within_soc_rho"]
    metric_labels = ["Full", "Within-aff", "Within-soc"]
    conditions_plot = ["baseline", "project_out_emotion", "project_out_social"]
    cond_labels = ["Baseline", "Proj-out emotion", "Proj-out social"]
    cond_colors = ["#95a5a6", "#e74c3c", "#3498db"]

    bar_data = {}  # {cond: {metric: [val per model]}}
    for cond_key in conditions_plot:
        bar_data[cond_key] = {m: [] for m in metrics}
        for res in all_results:
            if cond_key == "baseline":
                src = res["baseline"]
            else:
                src = res["k_sweep"][k][cond_key]
            for m in metrics:
                bar_data[cond_key][m].append(src[m])

    # Plot within-aff rho across models: baseline, proj-emo, proj-soc
    for ci, (cond_key, clabel, cc) in enumerate(zip(conditions_plot, cond_labels, cond_colors)):
        vals = bar_data[cond_key]["within_aff_rho"]
        ax.bar(x + (ci - 1) * w, vals, w * 0.9, color=cc, label=clabel,
               edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([model_short.get(r["model"], r["model"][:10]) for r in all_results])
    ax.set_ylabel("Within-affective rho")
    ax.set_title(f"Within-AFFECTIVE alignment (k={k})")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Panel B: within-social rho ──
    ax = axes[0, 1]
    for ci, (cond_key, clabel, cc) in enumerate(zip(conditions_plot, cond_labels, cond_colors)):
        vals = bar_data[cond_key]["within_soc_rho"]
        ax.bar(x + (ci - 1) * w, vals, w * 0.9, color=cc, label=clabel,
               edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([model_short.get(r["model"], r["model"][:10]) for r in all_results])
    ax.set_ylabel("Within-social rho")
    ax.set_title(f"Within-SOCIAL alignment (k={k})")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    # ── Panel C: k sweep (first model as exemplar) ──
    ax = axes[1, 0]
    res0 = all_results[0]
    ks = sorted(res0["k_sweep"].keys())
    for metric, color, label in [("within_aff_rho", "#e74c3c", "Within-aff"),
                                  ("within_soc_rho", "#3498db", "Within-soc")]:
        # Project out emotion
        vals_emo = [res0["k_sweep"][kk]["project_out_emotion"][metric] for kk in ks]
        ax.plot(ks, vals_emo, "o-", color=color, linewidth=2, markersize=7,
                label=f"Proj-out emo: {label}")
        # Project out social
        vals_soc = [res0["k_sweep"][kk]["project_out_social"][metric] for kk in ks]
        ax.plot(ks, vals_soc, "s--", color=color, linewidth=2, markersize=7,
                alpha=0.5, label=f"Proj-out soc: {label}")
    base_aff = res0["baseline"]["within_aff_rho"]
    base_soc = res0["baseline"]["within_soc_rho"]
    ax.axhline(base_aff, color="#e74c3c", linestyle=":", alpha=0.4, label=f"Baseline w-aff={base_aff:.2f}")
    ax.axhline(base_soc, color="#3498db", linestyle=":", alpha=0.4, label=f"Baseline w-soc={base_soc:.2f}")
    ax.set_xlabel("Subspace dimensionality (k)")
    ax.set_ylabel("rho")
    ax.set_title(f"k sweep ({model_short.get(res0['model'], res0['model'][:10])})")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)

    # ── Panel D: full rho comparison (baseline, proj-emo, proj-soc, random) ──
    ax = axes[1, 1]
    for ci, (cond_key, clabel, cc) in enumerate(zip(conditions_plot, cond_labels, cond_colors)):
        vals = bar_data[cond_key]["full_rho"]
        ax.bar(x + (ci - 1) * w, vals, w * 0.9, color=cc, label=clabel,
               edgecolor="k", linewidth=0.5, alpha=0.85)
    # Random null (mean + error bar)
    rand_means = [r["k_sweep"][k]["random_null"]["full_rho_mean"] for r in all_results]
    rand_stds = [r["k_sweep"][k]["random_null"]["full_rho_std"] for r in all_results]
    ax.bar(x + 2 * w, rand_means, w * 0.9, color="#bdc3c7", label="Random null",
           edgecolor="k", linewidth=0.5, yerr=rand_stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([model_short.get(r["model"], r["model"][:10]) for r in all_results])
    ax.set_ylabel("Full rho (91 pairs)")
    ax.set_title(f"Full brain-LLM alignment (k={k})")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out_path = FIG_DIR / "subspace_dissociation.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out_path}")


def print_summary_table(all_results):
    """Print the key result table."""
    k = K_DEFAULT

    print("\n" + "=" * 95)
    print(f"SUBSPACE DOUBLE-DISSOCIATION SUMMARY  (k={k})")
    print("=" * 95)

    header = (f"{'Model':<28s} {'Ablation':<18s} {'Full rho':>9s} "
              f"{'W-aff rho':>10s} {'W-soc rho':>10s} {'Cross rho':>10s}")
    print(header)
    print("-" * 95)

    for res in all_results:
        mname = res["model"]
        b = res["baseline"]
        emo = res["k_sweep"][k]["project_out_emotion"]
        soc = res["k_sweep"][k]["project_out_social"]
        rnd = res["k_sweep"][k]["random_null"]

        print(f"{mname:<28s} {'Baseline':<18s} {b['full_rho']:>+9.3f} "
              f"{b['within_aff_rho']:>+10.3f} {b['within_soc_rho']:>+10.3f} "
              f"{b['cross_rho']:>+10.3f}")
        print(f"{'':28s} {'Proj-out emotion':<18s} {emo['full_rho']:>+9.3f} "
              f"{emo['within_aff_rho']:>+10.3f} {emo['within_soc_rho']:>+10.3f} "
              f"{emo['cross_rho']:>+10.3f}")
        print(f"{'':28s} {'Proj-out social':<18s} {soc['full_rho']:>+9.3f} "
              f"{soc['within_aff_rho']:>+10.3f} {soc['within_soc_rho']:>+10.3f} "
              f"{soc['cross_rho']:>+10.3f}")
        print(f"{'':28s} {'Random (mean)':<18s} {rnd['full_rho_mean']:>+9.3f} "
              f"{rnd['within_aff_rho_mean']:>+10.3f} {rnd['within_soc_rho_mean']:>+10.3f} "
              f"{'':>10s}")

        dd = res["double_dissociation"]
        tag = "YES" if dd["confirmed"] else "NO"
        print(f"  -> Double dissociation: {tag}  "
              f"(emo->w-aff drop={dd['proj_out_emotion']['within_aff_drop']:+.3f} vs "
              f"w-soc drop={dd['proj_out_emotion']['within_soc_drop']:+.3f}; "
              f"soc->w-soc drop={dd['proj_out_social']['within_soc_drop']:+.3f} vs "
              f"w-aff drop={dd['proj_out_social']['within_aff_drop']:+.3f})")

        if "permutation_test" in res:
            pt = res["permutation_test"]
            print(f"  -> Permutation: p_emo_selective={pt['p_emo']:.4f}, "
                  f"p_soc_selective={pt['p_soc']:.4f}")
        print()


# ── Main ───────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load brain RDM
    brain_data = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain_data["rdm"].astype(np.float64)
    brain_conds = list(brain_data["conditions"])
    print(f"Brain RDM: {brain_rdm.shape[0]} conditions: {brain_conds}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    all_results = []

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        md = load_model_data(model_name, brain_conds)
        if md is None:
            continue

        result = analyze_model(md, brain_rdm, brain_conds)

        # Permutation test (condition-label shuffle) — runs on centroids only, fast
        # Skip if stim_acts was freed inside analyze_model; re-load is expensive.
        # Actually, stim_acts was NOT freed from md dict — analyze_model only deletes
        # its local reference. But we passed md by ref, so md["stim_acts"] may be gone.
        # Re-load just the peak-layer slice for the permutation test.
        md2 = load_model_data(model_name, brain_conds)
        if md2 is not None:
            print(f"\n  Running permutation test (n=10000)...", flush=True)
            t_perm = time.time()
            pt = permutation_test(md2, brain_rdm, brain_conds, k=K_DEFAULT, n_perm=10000)
            result["permutation_test"] = pt
            print(f"    p_emo_selective={pt['p_emo']:.4f}, p_soc_selective={pt['p_soc']:.4f} "
                  f"({time.time()-t_perm:.0f}s)")
            del md2

        all_results.append(result)

    # Print summary
    print_summary_table(all_results)

    # Make figure
    make_figure(all_results, brain_conds)

    # Save JSON
    out_path = OUT_DIR / "subspace_dissociation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    elapsed = time.time() - t0
    print(f"\nResults saved: {out_path}")
    print(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
