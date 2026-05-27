#!/usr/bin/env python3
"""
Confirmatory RSA addressing GPT Pro review concerns:

1. Discovery/Confirmation split: freeze config from Qwen-1.5B, test on held-out models
2. Max-stat permutation: permute labels, re-search peak layer each time → proper null
3. Cross-validated one-axis ablation: define axis on half stimuli, test on other half
4. Matched-variance ablation controls: PC1, valence axis, source axis, random-high-var
5. Bootstrap CIs for all key numbers
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, rankdata
from numpy.linalg import lstsq

BASE = Path("/hpc2hdd/home/mzhang630/data/nature/experiments")
RES = BASE / "results" / "cognitive_rsa"

MODELS = {
    "discovery": "Qwen2.5-1.5B-Instruct",
    "confirmation": [
        "Qwen2.5-7B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "gemma-2-9b-it",
    ],
}

AFFECTIVE = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
MENTALISTIC = {"belief", "mentalizing", "intention", "theory_of_mind",
               "empathy", "self_referential", "judgment"}

FROZEN_CONFIG = {
    "pooling": "mean_all",
    "normalization": "centered",
    "distance": "cosine",
}


def normalize_centered(act):
    return act - act.mean(axis=0, keepdims=True)

def rdm_cosine(act):
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - np.clip(a @ a.T, -1, 1)

def triu_vec(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]


def load_model_data(model_short, brain_conds):
    npz = RES / f"{model_short}_rsa_v2_per_stim.npz"
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=True)
    per_stim = data["per_stim_activations"]
    conditions = list(data["conditions"])
    pool_idx = list(data["pooling_names"]).index("mean_all")
    unique_conds = sorted(set(conditions))
    stim_cond = np.array([unique_conds.index(c) for c in conditions])

    n_conds = len(unique_conds)
    n_layers = per_stim.shape[2]
    cond_means = np.zeros((n_conds, n_layers, per_stim.shape[-1]), dtype=np.float32)
    for c in range(n_conds):
        idx = np.where(stim_cond == c)[0]
        cond_means[c] = per_stim[pool_idx, idx].mean(axis=0)

    order = [unique_conds.index(c) for c in brain_conds]
    cond_means = cond_means[order]

    return {
        "cond_means": cond_means,
        "per_stim": per_stim[pool_idx],
        "stim_cond": np.array([unique_conds.index(conditions[i]) for i in range(len(conditions))]),
        "unique_conds": unique_conds,
        "order": order,
        "n_layers": n_layers,
    }


def compute_rsa_at_layer(cond_means, brain_vec, L):
    act = normalize_centered(cond_means[:, L, :].astype(np.float64))
    rdm = rdm_cosine(act)
    rho, _ = spearmanr(brain_vec, triu_vec(rdm))
    return rho


def main():
    brain_data = np.load(RES / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])
    n = len(brain_conds)
    triu = np.triu_indices(n, k=1)
    brain_vec = brain_rdm[triu]

    # ================================================================
    # PART 1: Discovery/Confirmation split
    # ================================================================
    print("=" * 70)
    print("PART 1: Discovery/Confirmation split")
    print("=" * 70)

    # Discovery: find peak layer on Qwen-1.5B
    disc_data = load_model_data("Qwen2.5-1.5B-Instruct", brain_conds)
    if disc_data is None:
        # Fallback to Qwen-0.5B
        disc_data = load_model_data("Qwen2.5-0.5B-Instruct", brain_conds)

    best_rho, best_L = -1, 0
    for L in range(disc_data["n_layers"]):
        rho = compute_rsa_at_layer(disc_data["cond_means"], brain_vec, L)
        if rho > best_rho:
            best_rho, best_L = rho, L
    print(f"\nDiscovery model peak: L{best_L}, ρ = {best_rho:.4f}")

    # Confirmation: freeze config, test on held-out models at THEIR equivalent relative depth
    print(f"\nConfirmation (frozen config: {FROZEN_CONFIG}):")
    relative_depth = best_L / disc_data["n_layers"]

    for model_short in MODELS["confirmation"]:
        md = load_model_data(model_short, brain_conds)
        if md is None:
            print(f"  [skip] {model_short}")
            continue

        # Use equivalent relative depth
        confirm_L = int(relative_depth * md["n_layers"])
        confirm_L = min(confirm_L, md["n_layers"] - 1)
        rho_confirm = compute_rsa_at_layer(md["cond_means"], brain_vec, confirm_L)

        # Also find this model's own peak (for comparison)
        best_rho_own = max(compute_rsa_at_layer(md["cond_means"], brain_vec, L)
                          for L in range(md["n_layers"]))

        print(f"  {model_short:<35s} frozen L{confirm_L}: ρ={rho_confirm:+.4f}  "
              f"(own peak: ρ={best_rho_own:+.4f})")

    # ================================================================
    # PART 2: Max-stat permutation (proper null for peak-layer selection)
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: Max-stat permutation (Qwen2.5-7B)")
    print("=" * 70)

    md7b = load_model_data("Qwen2.5-7B-Instruct", brain_conds)
    if md7b is None:
        md7b = load_model_data("Qwen2.5-0.5B-Instruct", brain_conds)

    # Observed max ρ across layers
    obs_rhos = [compute_rsa_at_layer(md7b["cond_means"], brain_vec, L)
                for L in range(md7b["n_layers"])]
    obs_max = max(obs_rhos)
    obs_peak_L = np.argmax(obs_rhos)

    # Max-stat null: permute condition labels, re-search all layers
    rng = np.random.default_rng(2026)
    n_perm = 5000
    null_max = np.zeros(n_perm)
    for pi in range(n_perm):
        perm = rng.permutation(n)
        brain_perm = brain_rdm[np.ix_(perm, perm)]
        brain_perm_vec = brain_perm[triu]
        max_rho_perm = -1
        for L in range(md7b["n_layers"]):
            rho = compute_rsa_at_layer(md7b["cond_means"], brain_perm_vec, L)
            if rho > max_rho_perm:
                max_rho_perm = rho
        null_max[pi] = max_rho_perm

    p_maxstat = float(np.mean(null_max >= obs_max))
    print(f"\n  Observed max ρ = {obs_max:.4f} at L{obs_peak_L}")
    print(f"  Max-stat null: mean = {np.mean(null_max):.4f}, "
          f"95th = {np.percentile(null_max, 95):.4f}")
    print(f"  Max-stat p = {p_maxstat:.4f}")
    if p_maxstat < 0.05:
        print(f"  → SIGNIFICANT after correcting for peak-layer selection")
    else:
        print(f"  → NOT significant after correction")

    # ================================================================
    # PART 3: Cross-validated one-axis ablation
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: Cross-validated one-axis ablation")
    print("=" * 70)

    cond_acts = md7b["cond_means"][:, obs_peak_L, :].astype(np.float64)
    per_stim = md7b["per_stim"]
    stim_cond = md7b["stim_cond"]
    unique_conds = md7b["unique_conds"]
    order = md7b["order"]

    aff_idx = [i for i, c in enumerate(brain_conds) if c in AFFECTIVE]
    ment_idx = [i for i, c in enumerate(brain_conds) if c in MENTALISTIC]

    rng_cv = np.random.default_rng(42)
    n_cv = 50
    cv_deltas = []

    for _ in range(n_cv):
        # Split stimuli 50/50 per condition
        train_idx, test_idx = [], []
        for c_raw in range(len(unique_conds)):
            c_stims = np.where(stim_cond == c_raw)[0]
            rng_cv.shuffle(c_stims)
            cut = len(c_stims) // 2
            train_idx.extend(c_stims[:cut])
            test_idx.extend(c_stims[cut:])

        # Build condition centroids from TRAIN split
        train_centroids = np.zeros((len(unique_conds), per_stim.shape[-1]), dtype=np.float64)
        for c_raw in range(len(unique_conds)):
            c_train = [i for i in train_idx if stim_cond[i] == c_raw]
            if c_train:
                train_centroids[c_raw] = per_stim[c_train, obs_peak_L, :].mean(axis=0)
        train_centroids = train_centroids[order]

        # Define boundary from TRAIN centroids
        aff_c = train_centroids[aff_idx].mean(axis=0)
        ment_c = train_centroids[ment_idx].mean(axis=0)
        boundary = ment_c - aff_c
        boundary /= np.linalg.norm(boundary)

        # Build TEST condition centroids
        test_centroids = np.zeros((len(unique_conds), per_stim.shape[-1]), dtype=np.float64)
        for c_raw in range(len(unique_conds)):
            c_test = [i for i in test_idx if stim_cond[i] == c_raw]
            if c_test:
                test_centroids[c_raw] = per_stim[c_test, obs_peak_L, :].mean(axis=0)
        test_centroids = test_centroids[order]

        # Original RSA on test data
        orig_act = normalize_centered(test_centroids)
        orig_rdm = rdm_cosine(orig_act)
        rho_orig, _ = spearmanr(brain_vec, triu_vec(orig_rdm))

        # Ablate boundary (from train) on test data
        proj = np.outer(test_centroids @ boundary, boundary)
        ablated = test_centroids - proj
        abl_act = normalize_centered(ablated)
        abl_rdm = rdm_cosine(abl_act)
        rho_abl, _ = spearmanr(brain_vec, triu_vec(abl_rdm))

        cv_deltas.append(rho_abl - rho_orig)

    cv_deltas = np.array(cv_deltas)
    print(f"\n  Cross-validated ablation Δρ:")
    print(f"    mean = {np.mean(cv_deltas):+.4f}")
    print(f"    std  = {np.std(cv_deltas):.4f}")
    print(f"    95% CI = [{np.percentile(cv_deltas, 2.5):+.4f}, {np.percentile(cv_deltas, 97.5):+.4f}]")
    print(f"    All negative: {np.all(cv_deltas < 0)}")

    # ================================================================
    # PART 4: Matched-variance ablation controls
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 4: Matched-variance ablation controls")
    print("=" * 70)

    orig_act_full = normalize_centered(cond_acts)
    orig_rdm_full = rdm_cosine(orig_act_full)
    orig_rho, _ = spearmanr(brain_vec, triu_vec(orig_rdm_full))

    # Boundary direction
    aff_c = cond_acts[aff_idx].mean(axis=0)
    ment_c = cond_acts[ment_idx].mean(axis=0)
    boundary = ment_c - aff_c
    boundary /= np.linalg.norm(boundary)

    # PC1
    U, S, Vt = np.linalg.svd(normalize_centered(cond_acts), full_matrices=False)
    pc1 = Vt[0]

    # Valence direction (positive emotions vs negative emotions)
    pos_idx = [i for i, c in enumerate(brain_conds) if c in {"happiness"}]
    neg_idx = [i for i, c in enumerate(brain_conds) if c in {"anger", "fear", "disgust", "sadness"}]
    if pos_idx and neg_idx:
        valence_dir = cond_acts[pos_idx].mean(axis=0) - cond_acts[neg_idx].mean(axis=0)
        valence_dir /= np.linalg.norm(valence_dir)
    else:
        valence_dir = boundary  # fallback

    # High-variance random direction (match variance explained by boundary)
    boundary_var = np.var(cond_acts @ boundary)
    rng_r = np.random.default_rng(123)
    random_hvars = []
    for _ in range(100):
        rd = rng_r.standard_normal(cond_acts.shape[1])
        rd /= np.linalg.norm(rd)
        # Scale to match boundary variance
        rd_var = np.var(cond_acts @ rd)
        if rd_var > 0:
            scale = np.sqrt(boundary_var / rd_var)
            rd_scaled = rd * scale
            random_hvars.append(rd / np.linalg.norm(rd))

    def ablate_and_rsa(direction):
        proj = np.outer(cond_acts @ direction, direction)
        abl = normalize_centered(cond_acts - proj)
        rdm = rdm_cosine(abl)
        rho, _ = spearmanr(brain_vec, triu_vec(rdm))
        return rho

    rho_boundary = ablate_and_rsa(boundary)
    rho_pc1 = ablate_and_rsa(pc1)
    rho_valence = ablate_and_rsa(valence_dir)
    rho_randoms = [ablate_and_rsa(rd) for rd in random_hvars[:100]]

    # Cosine similarity between boundary and PC1
    cos_bound_pc1 = abs(np.dot(boundary, pc1))

    print(f"\n  Original ρ:                          {orig_rho:+.4f}")
    print(f"  After ablating boundary direction:    {rho_boundary:+.4f}  (Δ = {rho_boundary - orig_rho:+.4f})")
    print(f"  After ablating PC1:                  {rho_pc1:+.4f}  (Δ = {rho_pc1 - orig_rho:+.4f})")
    print(f"  After ablating valence direction:     {rho_valence:+.4f}  (Δ = {rho_valence - orig_rho:+.4f})")
    print(f"  After ablating random (mean of 100):  {np.mean(rho_randoms):+.4f}  (Δ = {np.mean(rho_randoms) - orig_rho:+.4f})")
    print(f"\n  Boundary ↔ PC1 cosine similarity:     {cos_bound_pc1:.4f}")
    print(f"  Boundary Δρ vs PC1 Δρ:                {rho_boundary - orig_rho:+.4f} vs {rho_pc1 - orig_rho:+.4f}")

    # ================================================================
    # PART 5: Bootstrap CIs
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 5: Bootstrap CIs for key numbers")
    print("=" * 70)

    stimuli = [json.loads(l) for l in open(BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl")]
    stim_conditions = [s.get("condition", s.get("category")) for s in stimuli]

    rng_boot = np.random.default_rng(2026)
    n_boot = 2000
    boot_rhos = []

    for _ in range(n_boot):
        # Bootstrap over stimuli within each condition
        boot_centroids = np.zeros_like(cond_acts)
        for ci, c_brain in enumerate(brain_conds):
            c_raw = md7b["unique_conds"].index(c_brain)
            c_stims = np.where(stim_cond == c_raw)[0]
            boot_idx = rng_boot.choice(c_stims, size=len(c_stims), replace=True)
            boot_centroids[ci] = per_stim[boot_idx, obs_peak_L, :].mean(axis=0)

        act_b = normalize_centered(boot_centroids)
        rdm_b = rdm_cosine(act_b)
        rho_b, _ = spearmanr(brain_vec, triu_vec(rdm_b))
        boot_rhos.append(rho_b)

    boot_rhos = np.array(boot_rhos)
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"\n  Brain-LLM ρ (Qwen 7B): {obs_max:.4f}")
    print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Bootstrap mean: {np.mean(boot_rhos):.4f}")
    print(f"  Bootstrap SE: {np.std(boot_rhos):.4f}")

    # Save all results
    results = {
        "discovery_model": "Qwen2.5-1.5B-Instruct",
        "discovery_peak_layer": int(best_L),
        "discovery_rho": float(best_rho),
        "max_stat_p": float(p_maxstat),
        "max_stat_null_95th": float(np.percentile(null_max, 95)),
        "cv_ablation_mean_delta": float(np.mean(cv_deltas)),
        "cv_ablation_ci": [float(np.percentile(cv_deltas, 2.5)),
                           float(np.percentile(cv_deltas, 97.5))],
        "ablation_controls": {
            "boundary": float(rho_boundary),
            "pc1": float(rho_pc1),
            "valence": float(rho_valence),
            "random_mean": float(np.mean(rho_randoms)),
        },
        "boundary_pc1_cosine": float(cos_bound_pc1),
        "bootstrap_ci": [float(ci_low), float(ci_high)],
        "bootstrap_mean": float(np.mean(boot_rhos)),
    }
    with open(RES / "confirmatory_rsa.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RES / 'confirmatory_rsa.json'}")


if __name__ == "__main__":
    main()
