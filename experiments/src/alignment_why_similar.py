#!/usr/bin/env python3
"""
Why does brain-LLM representational alignment work?

Part A: What creates the emotion↔social boundary in both systems?
Part B: What dimensions drive within-social alignment (ρ≈0.52-0.65)?

Tests theoretical dimension models (from neuroscience literature) against
both brain and LLM RDMs to find shared organizing principles.

CPU-only. Operates on existing RDMs.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from itertools import combinations

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]

AFF_SET = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
SOC_SET = {"belief", "intention", "judgment", "mentalizing", "moral",
           "empathy", "self_referential", "theory_of_mind"}

# ── Theoretical dimensions for the FULL 14 conditions (boundary analysis) ──
# Each condition scored on dimensions that might create the emotion↔social split.
FULL_DIMS = {
    "propositional_structure": {
        # Emotions = felt states; social cognition = propositions about mental states
        # (Fodor 1975; Perner 1991 metarepresentation; Barrett 2017 conceptual act)
        "anger": 1, "fear": 1, "disgust": 1, "sadness": 1, "happiness": 1, "valence": 1,
        "belief": 4, "intention": 4, "judgment": 4, "mentalizing": 4,
        "moral": 4, "empathy": 2, "self_referential": 3, "theory_of_mind": 5,
    },
    "other_modeling": {
        # Degree of modeling another agent's internal states
        # (Premack & Woodruff 1978; Frith & Frith 2006)
        "anger": 1, "fear": 1, "disgust": 1, "sadness": 1, "happiness": 1, "valence": 1,
        "theory_of_mind": 5, "mentalizing": 5, "belief": 4, "intention": 4,
        "empathy": 4, "judgment": 3, "moral": 3, "self_referential": 1,
    },
    "bodily_grounding": {
        # How much the concept requires interoceptive/somatic experience
        # (Damasio 1994 somatic marker; Craig 2009 interoception; Barrett 2017)
        "anger": 5, "fear": 5, "disgust": 5, "sadness": 4, "happiness": 4, "valence": 4,
        "empathy": 3, "moral": 2, "self_referential": 2, "judgment": 1,
        "belief": 1, "intention": 1, "mentalizing": 1, "theory_of_mind": 1,
    },
    "linguistic_definability": {
        # Can the concept be fully defined through language/text alone?
        # Social concepts are relationally defined; emotions require experience
        "anger": 2, "fear": 2, "disgust": 2, "sadness": 2, "happiness": 2, "valence": 2,
        "belief": 5, "intention": 5, "theory_of_mind": 5, "mentalizing": 4,
        "judgment": 4, "moral": 4, "empathy": 3, "self_referential": 3,
    },
}

# ── Theoretical dimensions for SOCIAL conditions (within-social analysis) ──
SOCIAL_DIMS = {
    "cognitive_vs_affective": {
        # Shamay-Tsoory (2011 Brain): cognitive ToM (TPJ/mPFC) vs affective ToM (vmPFC/IFG)
        "belief": 5, "intention": 4, "theory_of_mind": 5, "mentalizing": 4,
        "judgment": 3, "self_referential": 3, "moral": 2, "empathy": 1,
    },
    "self_vs_other": {
        # Self-referential (mPFC/PCC) vs other-modeling (TPJ/STS)
        # (Northoff 2006; Buckner 2008 DMN)
        "self_referential": 5, "empathy": 3, "judgment": 2, "moral": 2,
        "mentalizing": 1, "belief": 1, "intention": 1, "theory_of_mind": 1,
    },
    "normative_evaluative": {
        # Evaluative/prescriptive (moral, judgment) vs descriptive (belief, intention)
        # (Greene 2001 Science; Moll 2005)
        "judgment": 5, "moral": 5, "self_referential": 2, "empathy": 2,
        "mentalizing": 1, "belief": 1, "intention": 1, "theory_of_mind": 1,
    },
    "recursion_depth": {
        # Depth of mental state embedding
        # ToM = "X believes Y thinks P" (3+); belief = "X believes P" (2)
        # (Perner & Wimmer 1985; Kinderman 1998)
        "theory_of_mind": 5, "mentalizing": 4, "moral": 4, "judgment": 3,
        "belief": 3, "intention": 3, "empathy": 2, "self_referential": 2,
    },
    "affective_loading": {
        # How much emotional content is involved in the social process
        # (Shamay-Tsoory 2009; Zaki & Ochsner 2012)
        "empathy": 5, "moral": 4, "self_referential": 3, "judgment": 3,
        "mentalizing": 2, "belief": 1, "intention": 1, "theory_of_mind": 1,
    },
}


def upper_tri(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def model_rdm(scores, conditions):
    n = len(conditions)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(scores[conditions[i]] - scores[conditions[j]])
    return rdm


def perm_rsa(model_r, target_r, n_perm=10000, seed=42):
    """Permutation test: shuffle rows+cols of target RDM, two-tailed."""
    rng = np.random.default_rng(seed)
    m_ut = upper_tri(model_r)
    t_ut = upper_tri(target_r)
    rho_obs, _ = spearmanr(m_ut, t_ut)
    n = target_r.shape[0]
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        rho_p, _ = spearmanr(m_ut, upper_tri(target_r[np.ix_(p, p)]))
        if abs(rho_p) >= abs(rho_obs):
            count += 1
    return float(rho_obs), count / n_perm


def extract_block(rdm, conds, subset):
    idx = [conds.index(c) for c in subset if c in conds]
    return rdm[np.ix_(idx, idx)], [conds[i] for i in idx]


def print_rdm(rdm, conds, label=""):
    if label:
        print(f"\n  {label}:")
    w = max(len(c) for c in conds)
    print(f"  {'':>{w}}", end="")
    for c in conds:
        print(f" {c[:7]:>7}", end="")
    print()
    for i, ci in enumerate(conds):
        print(f"  {ci:>{w}}", end="")
        for j in range(len(conds)):
            print(f" {rdm[i,j]:7.3f}", end="")
        print()


def main():
    # ── Load data ──
    brain_data = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_rdm_full = brain_data["rdm"]
    conds = list(brain_data["conditions"])

    llm_rdms = {}
    for m in MODELS:
        f = RSA / f"{m}_rdm14_headline.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            llm_rdms[m] = d["rdm"]

    mean_llm = np.mean(list(llm_rdms.values()), axis=0)

    print("=" * 75)
    print("WHY DOES BRAIN-LLM ALIGNMENT WORK?")
    print("=" * 75)
    print(f"  Conditions: {conds}")
    print(f"  Models: {list(llm_rdms.keys())}")

    results = {}

    # ═══════════════════════════════════════════
    # PART A: BOUNDARY
    # ═══════════════════════════════════════════
    print(f"\n{'='*75}")
    print("PART A: What creates the emotion↔social boundary?")
    print(f"{'='*75}")

    # A1: Binary boundary model
    n = len(conds)
    binary = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            same = (conds[i] in AFF_SET) == (conds[j] in AFF_SET)
            binary[i, j] = 0 if same else 1

    rho_b_bin, p_b_bin = perm_rsa(binary, brain_rdm_full)
    rho_l_bin, p_l_bin = perm_rsa(binary, mean_llm)
    print(f"\n  Binary boundary model (0=same-block, 1=cross-block):")
    print(f"    vs Brain:    ρ = {rho_b_bin:+.3f}, p = {p_b_bin:.4f}")
    print(f"    vs LLM mean: ρ = {rho_l_bin:+.3f}, p = {p_l_bin:.4f}")

    results["binary_boundary"] = {
        "brain": {"rho": rho_b_bin, "p": p_b_bin},
        "llm_mean": {"rho": rho_l_bin, "p": p_l_bin},
    }

    # A2: Cross/within distance ratios
    aff_idx = [i for i, c in enumerate(conds) if c in AFF_SET]
    soc_idx = [i for i, c in enumerate(conds) if c in SOC_SET]

    print(f"\n  Distance ratios:")
    for label, rdm in [("Brain", brain_rdm_full), ("LLM-mean", mean_llm)]:
        wa = np.mean([rdm[i,j] for i in aff_idx for j in aff_idx if i < j])
        ws = np.mean([rdm[i,j] for i in soc_idx for j in soc_idx if i < j])
        cx = np.mean([rdm[i,j] for i in aff_idx for j in soc_idx])
        print(f"    {label}: within-aff={wa:.4f}, within-soc={ws:.4f}, cross={cx:.4f}, "
              f"cross/mean-within={cx/((wa+ws)/2):.2f}")

    # A3: Theoretical dimensions predicting 14×14 structure
    print(f"\n  Theoretical dimensions (full 14×14, 91 pairs):")
    print(f"  {'Dimension':>25s} | {'Brain ρ':>8s} {'p':>7s} | {'LLM ρ':>8s} {'p':>7s} | Shared?")
    print(f"  {'-'*75}")

    results["full_dimensions"] = {}
    for dim_name, scores in FULL_DIMS.items():
        m_rdm = model_rdm(scores, conds)
        rb, pb = perm_rsa(m_rdm, brain_rdm_full)
        rl, pl = perm_rsa(m_rdm, mean_llm)
        shared = "YES" if (pb < 0.05 and pl < 0.05 and rb * rl > 0) else "no"
        print(f"  {dim_name:>25s} | {rb:+8.3f} {pb:7.4f} | {rl:+8.3f} {pl:7.4f} | {shared}")
        results["full_dimensions"][dim_name] = {
            "brain_rho": rb, "brain_p": pb, "llm_rho": rl, "llm_p": pl, "shared": shared == "YES"}

    # A4: Per-model boundary RSA
    print(f"\n  Per-model binary boundary RSA:")
    results["per_model_boundary"] = {}
    for m, rdm in llm_rdms.items():
        r, p = perm_rsa(binary, rdm)
        print(f"    {m}: ρ = {r:+.3f}, p = {p:.4f}")
        results["per_model_boundary"][m] = {"rho": r, "p": p}

    # ═══════════════════════════════════════════
    # PART B: WITHIN-SOCIAL ALIGNMENT
    # ═══════════════════════════════════════════
    print(f"\n{'='*75}")
    print("PART B: What drives within-social alignment?")
    print(f"{'='*75}")

    soc_list = [c for c in conds if c in SOC_SET]
    brain_soc, soc_order = extract_block(brain_rdm_full, conds, soc_list)
    llm_soc, _ = extract_block(mean_llm, conds, soc_list)

    print_rdm(brain_soc, soc_order, "Brain within-social RDM")
    print_rdm(llm_soc, soc_order, "LLM mean within-social RDM")

    # Within-social alignment
    rho_ws, _ = spearmanr(upper_tri(brain_soc), upper_tri(llm_soc))
    print(f"\n  Within-social brain↔LLM mean: ρ = {rho_ws:+.3f}")

    per_model_ws = {}
    for m, rdm in llm_rdms.items():
        s, _ = extract_block(rdm, conds, soc_list)
        r, _ = spearmanr(upper_tri(brain_soc), upper_tri(s))
        print(f"    {m}: ρ = {r:+.3f}")
        per_model_ws[m] = float(r)

    # Dimension analysis
    print(f"\n  Social dimension analysis (8 conditions, 28 pairs):")
    print(f"  {'Dimension':>25s} | {'Brain ρ':>8s} {'p':>7s} | {'LLM ρ':>8s} {'p':>7s} | Shared?")
    print(f"  {'-'*75}")

    results["within_social"] = {
        "brain_llm_mean_rho": float(rho_ws),
        "per_model_rho": per_model_ws,
        "dimensions": {},
    }
    for dim_name, scores in SOCIAL_DIMS.items():
        m_rdm = model_rdm(scores, soc_order)
        rb, pb = perm_rsa(m_rdm, brain_soc)
        rl, pl = perm_rsa(m_rdm, llm_soc)
        shared = "YES" if (pb < 0.05 and pl < 0.05 and rb * rl > 0) else "no"
        print(f"  {dim_name:>25s} | {rb:+8.3f} {pb:7.4f} | {rl:+8.3f} {pl:7.4f} | {shared}")
        results["within_social"]["dimensions"][dim_name] = {
            "brain_rho": rb, "brain_p": pb, "llm_rho": rl, "llm_p": pl, "shared": shared == "YES"}

    # B2: Which condition pairs are most/least similar in both?
    print(f"\n  Condition pairs ranked by brain distance:")
    pairs = list(combinations(range(len(soc_order)), 2))
    brain_dists = [(soc_order[i], soc_order[j], brain_soc[i,j], llm_soc[i,j])
                   for i, j in pairs]
    brain_dists.sort(key=lambda x: x[2])

    print(f"  {'Pair':>35s} | {'Brain d':>8s} | {'LLM d':>8s} | Match?")
    print(f"  {'-'*65}")
    for ci, cj, bd, ld in brain_dists:
        median_b = np.median([x[2] for x in brain_dists])
        median_l = np.median([x[3] for x in brain_dists])
        match = "✓" if (bd < median_b) == (ld < median_l) else "✗"
        print(f"  {ci+' — '+cj:>35s} | {bd:8.4f} | {ld:8.4f} | {match}")

    # Save
    json.dump(results, open(OUT / "alignment_why_similar.json", "w"), indent=2)
    print(f"\nSaved: {OUT / 'alignment_why_similar.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
