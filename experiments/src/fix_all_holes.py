#!/usr/bin/env python3
"""
Fix all holes in the three-direction analysis:
  #4: Stronger partial RSA (control for sentence embeddings)
  #6: Leave-out cross-validation for Direction A
  + Leave-one-condition-out stability test for Direction C
  + Robustness checks

No GPU needed. Uses existing data.
"""
from __future__ import annotations
import json
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances

BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
COUPLING_DIR = BASE / "results" / "brain_causal_coupling"
DEV_DIR = BASE / "results" / "developmental_emergence"
OUT = BASE / "results" / "robustness_checks"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]

SIZES = [
    ("0.5B", "Qwen2.5-0.5B-Instruct"),
    ("1.5B", "Qwen2.5-1.5B-Instruct"),
    ("3B",   "Qwen2.5-3B-Instruct"),
    ("7B",   "Qwen2.5-7B-Instruct"),
]

AFFECTIVE = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
MENTALISTIC = {"belief", "mentalizing", "intention", "theory_of_mind",
               "empathy", "self_referential", "judgment", "moral"}


def rdm_cosine_centered(act):
    act = act - act.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - np.clip(a @ a.T, -1, 1)


def main():
    results = {}

    # Load brain RDM and conditions
    brain_data = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain_data["rdm"].astype(np.float64)
    brain_conds = list(brain_data["conditions"])
    n = len(brain_conds)
    triu = np.triu_indices(n, k=1)
    brain_vec = brain_rdm[triu]

    # Load RSA stimuli for sentence embeddings
    stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    stimuli = [json.loads(l) for l in open(stim_path)]

    # ================================================================
    # FIX #4: Stronger partial RSA
    # ================================================================
    print("=" * 70)
    print("FIX #4: Partial RSA controlling for sentence-level features")
    print("=" * 70)

    # Build condition-level text features
    # (a) Mean sentence length per condition
    # (b) TF-IDF / bag-of-words similarity between conditions
    # (c) Lexical overlap
    from sklearn.feature_extraction.text import TfidfVectorizer

    cond_texts = {c: [] for c in brain_conds}
    for s in stimuli:
        c = s["condition"]
        if c in cond_texts:
            cond_texts[c].append(s["text"])

    # Concatenate all texts per condition for TF-IDF
    cond_docs = [" ".join(cond_texts[c]) for c in brain_conds]

    # TF-IDF RDM
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(cond_docs).toarray()
    tfidf_rdm = cosine_distances(tfidf_matrix)

    # Sentence length RDM
    cond_lengths = np.array([np.mean([len(t.split()) for t in cond_texts[c]])
                             for c in brain_conds])
    length_rdm = np.abs(cond_lengths[:, None] - cond_lengths[None, :])

    # Load LLM RDMs at peak layer for partial RSA
    print("\n  Partial RSA for each model:")
    partial_results = {}

    for model in MODELS:
        npz = RSA_DIR / f"{model}_rsa_v2_per_stim.npz"
        if not npz.exists():
            continue
        data = np.load(npz, allow_pickle=True)
        per_stim = data["per_stim_activations"]
        conditions = list(data["conditions"])
        pooling_names = list(data["pooling_names"])
        pool_idx = pooling_names.index("mean_all")

        unique_conds = sorted(set(conditions))
        stim_cond = np.array([unique_conds.index(c) for c in conditions])
        n_layers = per_stim.shape[2]

        cond_means = np.zeros((len(unique_conds), n_layers, per_stim.shape[-1]),
                              dtype=np.float32)
        for c in range(len(unique_conds)):
            idx = np.where(stim_cond == c)[0]
            cond_means[c] = per_stim[pool_idx, idx].mean(axis=0)

        order = [unique_conds.index(c) for c in brain_conds]
        cond_means = cond_means[order]

        # Find peak layer
        best_rho, best_L = -1, 0
        for L in range(n_layers):
            act = cond_means[:, L, :].astype(np.float64)
            llm_rdm = rdm_cosine_centered(act)
            rho, _ = spearmanr(brain_vec, llm_rdm[triu])
            if rho > best_rho:
                best_rho, best_L = rho, L

        llm_rdm = rdm_cosine_centered(cond_means[:, best_L, :].astype(np.float64))
        llm_vec = llm_rdm[triu]

        # Raw RSA
        rho_raw, p_raw = spearmanr(brain_vec, llm_vec)

        # Partial RSA: regress out confounds
        from scipy.stats import rankdata
        def partial_spearman(x, y, confounds):
            """Partial Spearman: rank-transform, then partial correlation."""
            rx = rankdata(x)
            ry = rankdata(y)
            # Regress out confounds from both
            C = np.column_stack(confounds)
            C = np.column_stack([C, np.ones(len(x))])
            # Residualize
            proj = C @ np.linalg.lstsq(C, np.column_stack([rx, ry]), rcond=None)[0]
            rx_res = rx - proj[:, 0]
            ry_res = ry - proj[:, 1]
            r, p = spearmanr(rx_res, ry_res)
            return r, p

        tfidf_vec = tfidf_rdm[triu]
        length_vec = length_rdm[triu]

        rho_partial_tfidf, p_partial_tfidf = partial_spearman(
            brain_vec, llm_vec, [tfidf_vec])
        rho_partial_length, p_partial_length = partial_spearman(
            brain_vec, llm_vec, [length_vec])
        rho_partial_all, p_partial_all = partial_spearman(
            brain_vec, llm_vec, [tfidf_vec, length_vec])

        pct_retained = (rho_partial_all / rho_raw * 100) if rho_raw > 0 else 0

        partial_results[model] = {
            "raw_rho": float(rho_raw), "raw_p": float(p_raw),
            "partial_tfidf": {"rho": float(rho_partial_tfidf), "p": float(p_partial_tfidf)},
            "partial_length": {"rho": float(rho_partial_length), "p": float(p_partial_length)},
            "partial_all": {"rho": float(rho_partial_all), "p": float(p_partial_all)},
            "pct_retained": float(pct_retained),
        }

        short = model.split("-")[0][:8]
        print(f"\n  {model}:")
        print(f"    Raw:               ρ={rho_raw:+.4f}")
        print(f"    Partial (TF-IDF):  ρ={rho_partial_tfidf:+.4f}  p={p_partial_tfidf:.4f}")
        print(f"    Partial (length):  ρ={rho_partial_length:+.4f}  p={p_partial_length:.4f}")
        print(f"    Partial (all):     ρ={rho_partial_all:+.4f}  p={p_partial_all:.4f}")
        print(f"    Retained: {pct_retained:.1f}%")

    results["partial_rsa"] = partial_results

    # ================================================================
    # FIX #6: Leave-out cross-validation for Direction A
    # ================================================================
    print("\n" + "=" * 70)
    print("FIX #6: Leave-out cross-validation for Direction A")
    print("=" * 70)

    # Load all coupling matrices
    all_coupling = []
    for model in MODELS:
        with open(COUPLING_DIR / f"{model}_brain_causal_coupling.json") as f:
            d = json.load(f)
        all_coupling.append(np.array(d["coupling_matrix_logppl"]))
        conditions = d["conditions"]

    consensus = np.mean(all_coupling, axis=0)
    consensus_sym = (consensus + consensus.T) / 2
    coupling_vec = consensus_sym[triu]

    # Full correlation
    rho_full, p_full = spearmanr(brain_vec, coupling_vec)
    print(f"\n  Full (14 conditions): ρ={rho_full:+.4f}, p={p_full:.4f}")

    # Leave-one-condition-out
    print(f"\n  Leave-one-condition-out stability:")
    loo_rhos = []
    for leave_out in range(n):
        mask = np.ones(n, dtype=bool)
        mask[leave_out] = False
        sub_brain = brain_rdm[np.ix_(mask, mask)]
        sub_coupling = consensus_sym[np.ix_(mask, mask)]
        sub_triu = np.triu_indices(n - 1, k=1)
        rho_loo, _ = spearmanr(sub_brain[sub_triu], sub_coupling[sub_triu])
        loo_rhos.append(rho_loo)
        print(f"    drop {brain_conds[leave_out]:>20s}: ρ={rho_loo:+.4f}")

    loo_rhos = np.array(loo_rhos)
    print(f"\n  LOO range: [{loo_rhos.min():+.4f}, {loo_rhos.max():+.4f}]")
    print(f"  LOO mean: {loo_rhos.mean():+.4f} (full: {rho_full:+.4f})")
    print(f"  All negative: {np.all(loo_rhos < 0)}")
    results["direction_a_loo"] = {
        "full_rho": float(rho_full),
        "loo_rhos": {brain_conds[i]: float(loo_rhos[i]) for i in range(n)},
        "loo_mean": float(loo_rhos.mean()),
        "loo_min": float(loo_rhos.min()),
        "loo_max": float(loo_rhos.max()),
        "all_negative": bool(np.all(loo_rhos < 0)),
    }

    # Leave-TWO-out (stronger test)
    print(f"\n  Leave-two-out (91 combinations):")
    l2o_rhos = []
    for i, j in combinations(range(n), 2):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        sub_brain = brain_rdm[np.ix_(mask, mask)]
        sub_coupling = consensus_sym[np.ix_(mask, mask)]
        sub_triu = np.triu_indices(n - 2, k=1)
        rho_l2o, _ = spearmanr(sub_brain[sub_triu], sub_coupling[sub_triu])
        l2o_rhos.append(rho_l2o)

    l2o_rhos = np.array(l2o_rhos)
    print(f"    Range: [{l2o_rhos.min():+.4f}, {l2o_rhos.max():+.4f}]")
    print(f"    Mean: {l2o_rhos.mean():+.4f}")
    print(f"    % negative: {100 * np.mean(l2o_rhos < 0):.1f}%")
    print(f"    % significant (p<0.05): ", end="")
    n_sig = sum(1 for i, j in zip(
        combinations(range(n), 2), l2o_rhos)
        if True)  # need to compute p
    # Actually compute p for each
    sig_count = 0
    for idx, (i, j) in enumerate(combinations(range(n), 2)):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        sub_brain = brain_rdm[np.ix_(mask, mask)]
        sub_coupling = consensus_sym[np.ix_(mask, mask)]
        sub_triu = np.triu_indices(n - 2, k=1)
        _, p = spearmanr(sub_brain[sub_triu], sub_coupling[sub_triu])
        if p < 0.05:
            sig_count += 1
    print(f"{sig_count}/{len(l2o_rhos)} ({100*sig_count/len(l2o_rhos):.1f}%)")

    results["direction_a_l2o"] = {
        "mean": float(l2o_rhos.mean()),
        "min": float(l2o_rhos.min()),
        "max": float(l2o_rhos.max()),
        "pct_negative": float(100 * np.mean(l2o_rhos < 0)),
        "pct_significant": float(100 * sig_count / len(l2o_rhos)),
    }

    # ================================================================
    # FIX for Direction C: Leave-one-condition-out stability
    # ================================================================
    print("\n" + "=" * 70)
    print("Direction C stability: Leave-one-condition-out")
    print("=" * 70)

    # Load developmental emergence data
    with open(DEV_DIR / "developmental_emergence.json") as f:
        dev_data = json.load(f)

    dev_order = dev_data["developmental_order"]

    # Test: if we drop each condition one at a time, does p stay significant?
    # Use the "ρ at 0.5B vs developmental order" test
    size_key = "0.5B"
    per_cond = dev_data["per_size"][size_key]["per_condition_alignment"]

    # Full test
    conds_with_data = [c for c in brain_conds if c in per_cond and c in dev_order]
    full_devs = [dev_order[c] for c in conds_with_data]
    full_rhos_05 = [per_cond[c]["rho_mean"] for c in conds_with_data]
    rho_full_c, p_full_c = spearmanr(full_devs, full_rhos_05)
    print(f"\n  Full (14 conds): ρ={rho_full_c:+.4f}, p={p_full_c:.4f}")

    loo_results_c = {}
    for drop_cond in conds_with_data:
        subset = [c for c in conds_with_data if c != drop_cond]
        devs = [dev_order[c] for c in subset]
        rhos = [per_cond[c]["rho_mean"] for c in subset]
        rho_loo, p_loo = spearmanr(devs, rhos)
        loo_results_c[drop_cond] = {"rho": float(rho_loo), "p": float(p_loo)}
        status = "✓ p<0.05" if p_loo < 0.05 else "✗ p≥0.05"
        print(f"    drop {drop_cond:>20s}: ρ={rho_loo:+.4f}  p={p_loo:.4f}  {status}")

    n_stable = sum(1 for v in loo_results_c.values() if v["p"] < 0.05)
    print(f"\n  Stable: {n_stable}/{len(loo_results_c)} remain p<0.05 after dropping")

    results["direction_c_loo"] = {
        "full_rho": float(rho_full_c),
        "full_p": float(p_full_c),
        "loo": loo_results_c,
        "n_stable": n_stable,
    }

    # ================================================================
    # Ablation size sensitivity (Direction A)
    # ================================================================
    print("\n" + "=" * 70)
    print("Direction A: sensitivity to condition count")
    print("=" * 70)

    # Test with only affective vs only mentalistic
    aff_idx = [i for i, c in enumerate(brain_conds) if c in AFFECTIVE]
    ment_idx = [i for i, c in enumerate(brain_conds) if c in MENTALISTIC]

    for subset_name, subset_idx in [("affective only", aff_idx),
                                      ("mentalistic only", ment_idx)]:
        sub_brain = brain_rdm[np.ix_(subset_idx, subset_idx)]
        sub_coupling = consensus_sym[np.ix_(subset_idx, subset_idx)]
        sub_triu = np.triu_indices(len(subset_idx), k=1)
        rho_sub, p_sub = spearmanr(sub_brain[sub_triu], sub_coupling[sub_triu])
        print(f"  {subset_name} ({len(subset_idx)} conds): ρ={rho_sub:+.4f}, p={p_sub:.4f}")

    # Save all results
    with open(OUT / "robustness_checks.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUT / 'robustness_checks.json'}")


if __name__ == "__main__":
    main()
