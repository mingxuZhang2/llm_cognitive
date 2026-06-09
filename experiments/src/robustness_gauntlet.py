#!/usr/bin/env python3
"""
Robustness gauntlet: four anti-spurious-alignment tests on the headline RSA
(brain RDM vs LLM RDM, 14 conditions, 91 pairs, Spearman rho ~ 0.73).

Pre-empts Hadidi et al. 2026 (Nature Communications) critique that LLM-brain
alignment can arise from cherry-picked layers, fragile condition sets, or
stimulus idiosyncrasies.

Tests:
  1. Locked-pipeline (no peak selection)
  2. Leave-one-condition-out (LOO)
  3. Leave-one-model-out cross-validation
  4. Stimulus sub-sampling stability

All tests use EXISTING per-stim NPZs — no GPU needed.

Output: results/cognitive_rsa/robustness_gauntlet.json + printed summary.

Usage:
    python src/robustness_gauntlet.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
STIM_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"

MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]

# Peak layers from rsa_v2.json (config: mean_all|centered|1_cosine|v1_NS_only)
PEAK_LAYERS = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}

# Total layer counts (from NPZ shapes — 0-indexed, n_layers is the count)
N_LAYERS = {
    "Qwen2.5-7B-Instruct": 29,
    "Meta-Llama-3.1-8B-Instruct": 33,
    "Mistral-7B-Instruct-v0.3": 33,
    "gemma-2-9b-it": 43,
}

AFFECTIVE = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
SOCIAL = ["belief", "mentalizing", "intention", "theory_of_mind",
          "empathy", "self_referential", "judgment", "moral"]

N_BOOTSTRAP = 100  # for stimulus sub-sampling
SUBSAMPLE_FRAC = 0.5
RNG_SEED = 20260603


# ---------------------------------------------------------------------------
# Core RDM helpers (same recipe as reconstruct_headline_rdms.py)
# ---------------------------------------------------------------------------
def center(act: np.ndarray) -> np.ndarray:
    """Subtract condition mean (axis 0)."""
    return act - act.mean(axis=0, keepdims=True)


def rdm_cosine(act: np.ndarray) -> np.ndarray:
    """1 - cosine similarity on centered activations."""
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = act / norms
    return 1.0 - np.clip(Xn @ Xn.T, -1, 1)


def build_llm_rdm(cond_means: np.ndarray) -> np.ndarray:
    """cond_means: (n_conds, hidden_dim) float64 -> (n_conds, n_conds) RDM."""
    return rdm_cosine(center(cond_means))


def rsa_rho(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman rho on upper triangle."""
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, k=1)
    rho, _ = spearmanr(rdm_a[iu], rdm_b[iu])
    return float(rho)


def rsa_rho_pval(rdm_a: np.ndarray, rdm_b: np.ndarray) -> tuple[float, float]:
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, k=1)
    rho, p = spearmanr(rdm_a[iu], rdm_b[iu])
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_brain_rdm():
    """Load corrected Neurosynth brain RDM."""
    d = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    return d["rdm"].astype(np.float64), list(d["conditions"])


def load_per_stim(model: str):
    """
    Load per-stim activations for one model (memory-mapped).
    Returns: per_stim_act (mmap), conditions list, pooling index for mean_all.
    """
    path = RSA_DIR / f"{model}_rsa_v2_per_stim.npz"
    z = np.load(path, allow_pickle=True, mmap_mode="r")
    pools = list(z["pooling_names"])
    pidx = pools.index("mean_all")
    conds = list(z["conditions"])
    return z["per_stim_activations"], conds, pidx


def compute_cond_means(per_stim, conds, pidx, layer, cond_order,
                       stim_mask=None):
    """
    Compute condition-mean activations at a specific layer.

    per_stim: (n_pools, n_stim, n_layers, hidden_dim), mmap'd
    conds: list of condition strings per stimulus
    pidx: pooling index
    layer: layer index
    cond_order: list of condition names in desired output order
    stim_mask: optional boolean array (n_stim,) to sub-select stimuli

    Returns: (n_conds, hidden_dim) float64
    """
    n_stim = len(conds)
    indices = {c: [] for c in cond_order}
    for i in range(n_stim):
        if stim_mask is not None and not stim_mask[i]:
            continue
        c = conds[i]
        if c in indices:
            indices[c].append(i)

    # Extract only the needed slice: per_stim[pidx, :, layer, :]
    # With mmap, this reads only the needed layer from disk.
    layer_act = per_stim[pidx, :, layer, :]  # (n_stim, hidden_dim)

    cmeans = []
    for c in cond_order:
        idx = indices[c]
        if len(idx) == 0:
            raise ValueError(f"No stimuli for condition '{c}' (after masking)")
        cmeans.append(layer_act[idx].astype(np.float64).mean(axis=0))

    return np.stack(cmeans)


def compute_cond_means_all_layers(per_stim, conds, pidx, cond_order,
                                  stim_mask=None):
    """
    Compute condition-mean activations at ALL layers.
    Returns: (n_conds, n_layers, hidden_dim) float64
    """
    n_layers = per_stim.shape[2]
    results = []
    for L in range(n_layers):
        results.append(compute_cond_means(per_stim, conds, pidx, L,
                                          cond_order, stim_mask))
    return np.stack(results, axis=1)  # (n_conds, n_layers, hidden_dim)


# ---------------------------------------------------------------------------
# TEST 1: Locked-pipeline (no peak selection)
# ---------------------------------------------------------------------------
def test_locked_pipeline(brain_rdm, brain_conds):
    """
    Instead of using per-model peak layers, test with:
    (a) Fixed layers: 15, plus the median layer of each model
    (b) Mean across all layers
    (c) Random 5 layers (averaged)
    """
    print("=" * 72)
    print("TEST 1: Locked-pipeline (no peak selection)")
    print("=" * 72)

    results = {}
    rng = np.random.RandomState(RNG_SEED)

    for model in MODELS:
        print(f"\n  {model}:")
        per_stim, conds, pidx = load_per_stim(model)
        n_layers = per_stim.shape[2]
        peak = PEAK_LAYERS[model]
        model_results = {"peak_layer": peak, "n_layers": n_layers}

        # (a) Peak-layer baseline (for comparison)
        cm = compute_cond_means(per_stim, conds, pidx, peak, brain_conds)
        llm_rdm = build_llm_rdm(cm)
        rho_peak = rsa_rho(brain_rdm, llm_rdm)
        model_results["peak_rho"] = rho_peak
        print(f"    Peak (L{peak:2d}):       rho = {rho_peak:+.4f}")

        # (b) Fixed layer 15 for all models
        fixed_L = 15
        cm = compute_cond_means(per_stim, conds, pidx, fixed_L, brain_conds)
        llm_rdm = build_llm_rdm(cm)
        rho_fixed = rsa_rho(brain_rdm, llm_rdm)
        model_results["fixed_L15_rho"] = rho_fixed
        print(f"    Fixed (L15):       rho = {rho_fixed:+.4f}")

        # (c) Median layer
        median_L = n_layers // 2
        cm = compute_cond_means(per_stim, conds, pidx, median_L, brain_conds)
        llm_rdm = build_llm_rdm(cm)
        rho_median = rsa_rho(brain_rdm, llm_rdm)
        model_results["median_layer"] = median_L
        model_results["median_rho"] = rho_median
        print(f"    Median (L{median_L:2d}):     rho = {rho_median:+.4f}")

        # (d) Mean across ALL layers
        cm_all = compute_cond_means_all_layers(per_stim, conds, pidx,
                                               brain_conds)
        cm_mean_all = cm_all.mean(axis=1)  # (n_conds, hidden_dim)
        llm_rdm = build_llm_rdm(cm_mean_all)
        rho_mean_all = rsa_rho(brain_rdm, llm_rdm)
        model_results["mean_all_layers_rho"] = rho_mean_all
        print(f"    Mean-all-layers:   rho = {rho_mean_all:+.4f}")

        # (e) Full layer profile (rho at every layer)
        layer_rhos = []
        for L in range(n_layers):
            cm = compute_cond_means(per_stim, conds, pidx, L, brain_conds)
            rdm = build_llm_rdm(cm)
            layer_rhos.append(rsa_rho(brain_rdm, rdm))
        layer_rhos = np.array(layer_rhos)

        # Count how many layers exceed various thresholds
        n_above_05 = int(np.sum(layer_rhos > 0.5))
        n_above_06 = int(np.sum(layer_rhos > 0.6))
        n_above_07 = int(np.sum(layer_rhos > 0.7))
        model_results["layer_rhos"] = [float(r) for r in layer_rhos]
        model_results["n_layers_above_0.5"] = n_above_05
        model_results["n_layers_above_0.6"] = n_above_06
        model_results["n_layers_above_0.7"] = n_above_07
        model_results["min_rho"] = float(layer_rhos.min())
        model_results["max_rho"] = float(layer_rhos.max())
        model_results["mean_rho"] = float(layer_rhos.mean())
        print(f"    Layer profile:     min={layer_rhos.min():+.4f}  "
              f"mean={layer_rhos.mean():+.4f}  max={layer_rhos.max():+.4f}")
        print(f"    Layers > 0.5: {n_above_05}/{n_layers}  "
              f"> 0.6: {n_above_06}/{n_layers}  "
              f"> 0.7: {n_above_07}/{n_layers}")

        # (f) 5 random layers averaged (repeat 20 times for stability)
        random_rhos = []
        for _ in range(20):
            rand_layers = rng.choice(n_layers, size=5, replace=False)
            cm_rand = cm_all[:, rand_layers, :].mean(axis=1)
            rdm_rand = build_llm_rdm(cm_rand)
            random_rhos.append(rsa_rho(brain_rdm, rdm_rand))
        model_results["random_5layer_mean_rho"] = float(np.mean(random_rhos))
        model_results["random_5layer_std_rho"] = float(np.std(random_rhos))
        print(f"    Random 5-layer avg: rho = {np.mean(random_rhos):+.4f} "
              f"+/- {np.std(random_rhos):.4f}")

        results[model] = model_results

        # Free memory
        del per_stim, cm_all
        gc.collect()

    # Cross-model summary
    print("\n  SUMMARY (locked-pipeline):")
    for label in ["peak_rho", "fixed_L15_rho", "median_rho",
                   "mean_all_layers_rho"]:
        vals = [results[m][label] for m in MODELS]
        nice = label.replace("_rho", "").replace("_", " ")
        print(f"    {nice:20s}: {np.mean(vals):+.4f} "
              f"(range {np.min(vals):+.4f} to {np.max(vals):+.4f})")

    return results


# ---------------------------------------------------------------------------
# TEST 2: Leave-one-condition-out (LOO)
# ---------------------------------------------------------------------------
def test_loo(brain_rdm, brain_conds):
    """
    For each condition, drop it, rebuild 13x13 RDMs, compute rho on 78 pairs.
    """
    print("\n" + "=" * 72)
    print("TEST 2: Leave-one-condition-out (LOO)")
    print("=" * 72)

    results = {}

    for model in MODELS:
        print(f"\n  {model}:")
        per_stim, conds, pidx = load_per_stim(model)
        peak = PEAK_LAYERS[model]

        # Full RDM (for reference)
        cm_full = compute_cond_means(per_stim, conds, pidx, peak, brain_conds)
        llm_rdm_full = build_llm_rdm(cm_full)
        rho_full = rsa_rho(brain_rdm, llm_rdm_full)

        loo_rhos = {}
        for drop_idx, drop_cond in enumerate(brain_conds):
            # Build sub-condition list
            sub_conds = [c for i, c in enumerate(brain_conds) if i != drop_idx]
            mask = np.ones(len(brain_conds), dtype=bool)
            mask[drop_idx] = False

            # Sub brain RDM
            sub_brain = brain_rdm[np.ix_(mask, mask)]

            # Sub LLM RDM: recompute condition means for the 13 conditions
            cm_sub = compute_cond_means(per_stim, conds, pidx, peak, sub_conds)
            sub_llm = build_llm_rdm(cm_sub)

            rho_loo, p_loo = rsa_rho_pval(sub_brain, sub_llm)
            loo_rhos[drop_cond] = {"rho": rho_loo, "p": p_loo}

            block = "AFF" if drop_cond in AFFECTIVE else "SOC"
            sig = "***" if p_loo < 0.001 else ("**" if p_loo < 0.01 else
                  ("*" if p_loo < 0.05 else "n.s."))
            print(f"    drop {drop_cond:>20s} [{block}]: "
                  f"rho={rho_loo:+.4f}  p={p_loo:.4f} {sig}")

        rho_vals = np.array([v["rho"] for v in loo_rhos.values()])
        n_sig = sum(1 for v in loo_rhos.values() if v["p"] < 0.05)
        n_killed = sum(1 for v in loo_rhos.values() if v["rho"] < 0.3)

        print(f"    ---")
        print(f"    Full rho: {rho_full:+.4f}  |  "
              f"LOO mean: {rho_vals.mean():+.4f}  "
              f"range: [{rho_vals.min():+.4f}, {rho_vals.max():+.4f}]")
        print(f"    Significant (p<0.05): {n_sig}/14  |  "
              f"Any condition kills signal (rho<0.3): {n_killed}")

        results[model] = {
            "full_rho": rho_full,
            "loo": loo_rhos,
            "loo_mean": float(rho_vals.mean()),
            "loo_std": float(rho_vals.std()),
            "loo_min": float(rho_vals.min()),
            "loo_max": float(rho_vals.max()),
            "n_significant": n_sig,
            "n_killed": n_killed,
        }

        del per_stim
        gc.collect()

    # Cross-model summary
    print("\n  SUMMARY (LOO):")
    for m in MODELS:
        r = results[m]
        short = m.split("-")[0][:8]
        print(f"    {short:10s}: full={r['full_rho']:+.4f}  "
              f"LOO mean={r['loo_mean']:+.4f}  "
              f"min={r['loo_min']:+.4f}  sig={r['n_significant']}/14")

    return results


# ---------------------------------------------------------------------------
# TEST 3: Leave-one-model-out cross-validation
# ---------------------------------------------------------------------------
def test_lomo_cv(brain_rdm, brain_conds):
    """
    For each model: compute peak layer on the OTHER 3 models, then test on
    the held-out model at that consensus peak layer.
    """
    print("\n" + "=" * 72)
    print("TEST 3: Leave-one-model-out cross-validation")
    print("=" * 72)

    # Step 1: For each model, compute rho at every layer (reuse from test 1
    # if available, but for clean separation we recompute here).
    all_layer_rhos = {}
    for model in MODELS:
        per_stim, conds, pidx = load_per_stim(model)
        n_layers = per_stim.shape[2]
        layer_rhos = []
        for L in range(n_layers):
            cm = compute_cond_means(per_stim, conds, pidx, L, brain_conds)
            rdm = build_llm_rdm(cm)
            layer_rhos.append(rsa_rho(brain_rdm, rdm))
        all_layer_rhos[model] = np.array(layer_rhos)
        del per_stim
        gc.collect()

    # Step 2: For each held-out model, find consensus peak from other 3.
    # Since models have different numbers of layers, normalize to fractional
    # depth (0-1) and find the fractional peak from the 3 training models,
    # then map back to the held-out model's layer space.
    results = {}
    print()
    for held_out in MODELS:
        train_models = [m for m in MODELS if m != held_out]

        # Compute fractional peak for each training model
        train_frac_peaks = []
        for m in train_models:
            rhos = all_layer_rhos[m]
            peak_idx = int(np.argmax(rhos))
            frac = peak_idx / (len(rhos) - 1)  # normalize to [0, 1]
            train_frac_peaks.append(frac)

        # Consensus: mean of fractional peaks
        consensus_frac = np.mean(train_frac_peaks)

        # Map to held-out model's layer space
        n_held = len(all_layer_rhos[held_out])
        consensus_layer = int(round(consensus_frac * (n_held - 1)))
        consensus_layer = min(consensus_layer, n_held - 1)

        # Get rho at consensus layer vs at actual peak
        rho_consensus = float(all_layer_rhos[held_out][consensus_layer])
        actual_peak = PEAK_LAYERS[held_out]
        rho_actual_peak = float(all_layer_rhos[held_out][actual_peak])
        rho_best = float(np.max(all_layer_rhos[held_out]))

        short = held_out.split("-")[0][:8]
        print(f"  Held-out: {held_out}")
        print(f"    Train fractional peaks: "
              f"{[f'{f:.3f}' for f in train_frac_peaks]}")
        print(f"    Consensus frac: {consensus_frac:.3f} -> "
              f"L{consensus_layer} (actual peak: L{actual_peak})")
        print(f"    rho@consensus: {rho_consensus:+.4f}  "
              f"rho@actual_peak: {rho_actual_peak:+.4f}  "
              f"rho@best: {rho_best:+.4f}")
        print(f"    Retention: {rho_consensus / rho_best * 100:.1f}% of best")

        results[held_out] = {
            "train_models": train_models,
            "train_frac_peaks": [float(f) for f in train_frac_peaks],
            "consensus_frac": float(consensus_frac),
            "consensus_layer": consensus_layer,
            "actual_peak": actual_peak,
            "rho_consensus": rho_consensus,
            "rho_actual_peak": rho_actual_peak,
            "rho_best": rho_best,
            "retention_pct": float(rho_consensus / rho_best * 100)
                             if rho_best > 0 else 0.0,
        }

    # Summary
    consensus_rhos = [results[m]["rho_consensus"] for m in MODELS]
    peak_rhos = [results[m]["rho_actual_peak"] for m in MODELS]
    print(f"\n  SUMMARY (LOMO-CV):")
    print(f"    Consensus-layer rho: {np.mean(consensus_rhos):+.4f} "
          f"(range {np.min(consensus_rhos):+.4f} "
          f"to {np.max(consensus_rhos):+.4f})")
    print(f"    Actual-peak rho:     {np.mean(peak_rhos):+.4f} "
          f"(range {np.min(peak_rhos):+.4f} "
          f"to {np.max(peak_rhos):+.4f})")
    retentions = [results[m]["retention_pct"] for m in MODELS]
    print(f"    Retention: {np.mean(retentions):.1f}% "
          f"(range {np.min(retentions):.1f}% to {np.max(retentions):.1f}%)")

    return results


# ---------------------------------------------------------------------------
# TEST 4: Stimulus sub-sampling stability
# ---------------------------------------------------------------------------
def test_stim_subsample(brain_rdm, brain_conds):
    """
    Randomly sample 50% of stimuli per condition, rebuild RDMs, compute rho.
    Repeat N_BOOTSTRAP times.
    """
    print("\n" + "=" * 72)
    print(f"TEST 4: Stimulus sub-sampling stability "
          f"({int(SUBSAMPLE_FRAC*100)}% x {N_BOOTSTRAP} bootstraps)")
    print("=" * 72)

    rng = np.random.RandomState(RNG_SEED)
    results = {}

    for model in MODELS:
        print(f"\n  {model}:")
        per_stim, conds, pidx = load_per_stim(model)
        peak = PEAK_LAYERS[model]
        n_stim = len(conds)

        # Build per-condition stimulus index lists
        cond_indices = {c: [] for c in brain_conds}
        for i, c in enumerate(conds):
            if c in cond_indices:
                cond_indices[c].append(i)

        # Full-data rho for reference
        cm_full = compute_cond_means(per_stim, conds, pidx, peak, brain_conds)
        llm_rdm_full = build_llm_rdm(cm_full)
        rho_full = rsa_rho(brain_rdm, llm_rdm_full)

        # Pre-extract the peak layer for speed (avoids repeated mmap seeks)
        layer_act = np.array(per_stim[pidx, :, peak, :])  # (n_stim, hidden)

        bootstrap_rhos = []
        for b in range(N_BOOTSTRAP):
            # Sub-sample: for each condition, pick SUBSAMPLE_FRAC of stimuli
            mask = np.zeros(n_stim, dtype=bool)
            for c in brain_conds:
                idx = np.array(cond_indices[c])
                n_keep = max(1, int(len(idx) * SUBSAMPLE_FRAC))
                chosen = rng.choice(idx, size=n_keep, replace=False)
                mask[chosen] = True

            # Compute condition means from sub-sampled stimuli
            cmeans = []
            for c in brain_conds:
                idx = np.array(cond_indices[c])
                sub_idx = idx[mask[idx]]
                cmeans.append(layer_act[sub_idx].astype(np.float64).mean(0))
            cm = np.stack(cmeans)

            rdm = build_llm_rdm(cm)
            bootstrap_rhos.append(rsa_rho(brain_rdm, rdm))

        bootstrap_rhos = np.array(bootstrap_rhos)
        ci_lo, ci_hi = np.percentile(bootstrap_rhos, [2.5, 97.5])
        n_above_05 = int(np.sum(bootstrap_rhos > 0.5))
        n_above_06 = int(np.sum(bootstrap_rhos > 0.6))

        print(f"    Full rho:    {rho_full:+.4f}")
        print(f"    Bootstrap:   mean={bootstrap_rhos.mean():+.4f}  "
              f"std={bootstrap_rhos.std():.4f}")
        print(f"    95% CI:      [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    Range:       [{bootstrap_rhos.min():+.4f}, "
              f"{bootstrap_rhos.max():+.4f}]")
        print(f"    > 0.5: {n_above_05}/{N_BOOTSTRAP}  "
              f"> 0.6: {n_above_06}/{N_BOOTSTRAP}")

        results[model] = {
            "full_rho": rho_full,
            "bootstrap_mean": float(bootstrap_rhos.mean()),
            "bootstrap_std": float(bootstrap_rhos.std()),
            "ci_95": [float(ci_lo), float(ci_hi)],
            "bootstrap_min": float(bootstrap_rhos.min()),
            "bootstrap_max": float(bootstrap_rhos.max()),
            "n_above_0.5": n_above_05,
            "n_above_0.6": n_above_06,
            "n_bootstrap": N_BOOTSTRAP,
            "subsample_frac": SUBSAMPLE_FRAC,
        }

        del per_stim, layer_act
        gc.collect()

    # Cross-model summary
    print(f"\n  SUMMARY (sub-sampling, {N_BOOTSTRAP} bootstraps, "
          f"{int(SUBSAMPLE_FRAC*100)}% stimuli):")
    for m in MODELS:
        r = results[m]
        short = m.split("-")[0][:8]
        print(f"    {short:10s}: {r['bootstrap_mean']:+.4f} +/- "
              f"{r['bootstrap_std']:.4f}  "
              f"CI=[{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("ROBUSTNESS GAUNTLET")
    print("=" * 72)
    print(f"Models: {MODELS}")
    print(f"Brain RDM: {RSA_DIR / 'brain_rdm.npz'}")
    print(f"N_bootstrap: {N_BOOTSTRAP}, subsample_frac: {SUBSAMPLE_FRAC}")
    print()

    brain_rdm, brain_conds = load_brain_rdm()
    assert len(brain_conds) == 14, f"Expected 14 conditions, got {len(brain_conds)}"

    # Run the four tests
    results = {}

    results["test1_locked_pipeline"] = test_locked_pipeline(brain_rdm,
                                                            brain_conds)

    results["test2_loo"] = test_loo(brain_rdm, brain_conds)

    results["test3_lomo_cv"] = test_lomo_cv(brain_rdm, brain_conds)

    results["test4_stim_subsample"] = test_stim_subsample(brain_rdm,
                                                          brain_conds)

    # -----------------------------------------------------------------------
    # Overall verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("OVERALL VERDICT")
    print("=" * 72)

    verdicts = []

    # Test 1: signal survives without peak selection?
    mean_all_rhos = [results["test1_locked_pipeline"][m]["mean_all_layers_rho"]
                     for m in MODELS]
    fixed_rhos = [results["test1_locked_pipeline"][m]["fixed_L15_rho"]
                  for m in MODELS]
    v1 = (np.mean(mean_all_rhos) > 0.3 and np.mean(fixed_rhos) > 0.3)
    verdicts.append(v1)
    print(f"  Test 1 (locked-pipeline):  "
          f"mean-all-layers rho={np.mean(mean_all_rhos):+.4f}, "
          f"fixed-L15 rho={np.mean(fixed_rhos):+.4f}  "
          f"{'PASS' if v1 else 'FAIL'}")

    # Test 2: no single condition kills signal?
    loo_mins = [results["test2_loo"][m]["loo_min"] for m in MODELS]
    loo_sigs = [results["test2_loo"][m]["n_significant"] for m in MODELS]
    v2 = (min(loo_mins) > 0.3 and min(loo_sigs) >= 12)
    verdicts.append(v2)
    print(f"  Test 2 (LOO):              "
          f"min LOO rho={min(loo_mins):+.4f}, "
          f"min sig count={min(loo_sigs)}/14  "
          f"{'PASS' if v2 else 'FAIL'}")

    # Test 3: held-out model retains most of signal?
    retentions = [results["test3_lomo_cv"][m]["retention_pct"] for m in MODELS]
    consensus_rhos = [results["test3_lomo_cv"][m]["rho_consensus"]
                      for m in MODELS]
    v3 = (min(consensus_rhos) > 0.5 and np.mean(retentions) > 80)
    verdicts.append(v3)
    print(f"  Test 3 (LOMO-CV):          "
          f"min consensus rho={min(consensus_rhos):+.4f}, "
          f"mean retention={np.mean(retentions):.1f}%  "
          f"{'PASS' if v3 else 'FAIL'}")

    # Test 4: signal stable under 50% stimulus sub-sampling?
    boot_stds = [results["test4_stim_subsample"][m]["bootstrap_std"]
                 for m in MODELS]
    boot_means = [results["test4_stim_subsample"][m]["bootstrap_mean"]
                  for m in MODELS]
    boot_lows = [results["test4_stim_subsample"][m]["ci_95"][0]
                 for m in MODELS]
    v4 = (min(boot_lows) > 0.4 and max(boot_stds) < 0.1)
    verdicts.append(v4)
    print(f"  Test 4 (sub-sampling):     "
          f"min CI lower={min(boot_lows):+.4f}, "
          f"max std={max(boot_stds):.4f}  "
          f"{'PASS' if v4 else 'FAIL'}")

    overall = all(verdicts)
    print(f"\n  OVERALL: {sum(verdicts)}/4 passed  "
          f"{'-> ROBUST' if overall else '-> CONCERNS'}")

    # Add metadata
    results["metadata"] = {
        "brain_rdm": "brain_rdm.npz (14-condition Neurosynth, corrected)",
        "n_conditions": 14,
        "n_pairs": 91,
        "models": MODELS,
        "peak_layers": PEAK_LAYERS,
        "n_bootstrap": N_BOOTSTRAP,
        "subsample_frac": SUBSAMPLE_FRAC,
        "rng_seed": RNG_SEED,
        "verdict": {
            "test1_pass": verdicts[0],
            "test2_pass": verdicts[1],
            "test3_pass": verdicts[2],
            "test4_pass": verdicts[3],
            "overall": overall,
        },
        "runtime_s": time.time() - t0,
    }

    # Save
    out_path = RSA_DIR / "robustness_gauntlet.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
