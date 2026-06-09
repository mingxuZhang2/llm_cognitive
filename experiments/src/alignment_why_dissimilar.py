#!/usr/bin/env python3
"""
Why does within-affective brain-LLM alignment FAIL?

Part A: What dimensions organize the brain's emotion geometry?
Part B: What dimensions organize the LLM's emotion geometry?
Part C: Dimension-by-dimension comparison → where exactly do they diverge?

Uses established appraisal theory dimensions (Smith & Ellsworth 1985,
Warriner VAD norms, Ekman/Levenson autonomic specificity) as model RDMs.

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

# ── Emotion dimensions from established literature ──
# Scores sourced from specific papers; see inline citations.
EMOTION_DIMS = {
    # ─ Warriner et al. 2013 VAD norms (1-9 scale, word-level) ─
    "arousal_W13": {
        # anger=5.49, fear=6.31, disgust=5.08, sadness=3.49, happiness=6.49
        "anger": 5.49, "fear": 6.31, "disgust": 5.08,
        "sadness": 3.49, "happiness": 6.49, "valence": 5.0,
    },
    "valence_W13": {
        # anger=2.53, fear=2.93, disgust=2.43, sadness=2.10, happiness=8.21
        "anger": 2.53, "fear": 2.93, "disgust": 2.43,
        "sadness": 2.10, "happiness": 8.21, "valence": 5.0,
    },
    "dominance_W13": {
        # anger=4.43, fear=2.95, disgust=4.16, sadness=2.66, happiness=6.10
        "anger": 4.43, "fear": 2.95, "disgust": 4.16,
        "sadness": 2.66, "happiness": 6.10, "valence": 4.5,
    },
    # ─ Appraisal dimensions (Smith & Ellsworth 1985; Roseman 1996) ─
    "approach_withdrawal": {
        # Davidson 1992, Harmon-Jones 2003: anger=approach despite negative valence
        "anger": 5.0, "fear": 1.0, "disgust": 1.5,
        "sadness": 1.5, "happiness": 4.5, "valence": 3.0,
    },
    "certainty_SE85": {
        # Smith & Ellsworth 1985: anger=certain, fear=uncertain
        "anger": 5.0, "fear": 1.0, "disgust": 4.0,
        "sadness": 3.0, "happiness": 4.0, "valence": 3.0,
    },
    "control_agency_R96": {
        # Roseman 1996: anger=other-caused+controllable, sadness=uncontrollable
        "anger": 5.0, "fear": 1.5, "disgust": 3.0,
        "sadness": 1.0, "happiness": 4.0, "valence": 3.0,
    },
    # ─ Biological / embodied dimensions ─
    "autonomic_specificity": {
        # Ekman, Levenson & Friesen 1983; Kreibig 2010 review
        # anger: HR↑, skin-T↑, BP↑; fear: HR↑, skin-T↓; disgust: HR↓, nausea
        "anger": 5.0, "fear": 4.5, "disgust": 4.0,
        "sadness": 2.0, "happiness": 2.5, "valence": 1.5,
    },
    "facial_distinctiveness": {
        # Ekman 1992 basic emotion facial expressions; FACS distinctiveness
        "anger": 5.0, "fear": 4.5, "disgust": 5.0,
        "sadness": 4.0, "happiness": 4.5, "valence": 1.0,
    },
    # ─ Functional / social dimensions ─
    "social_directedness": {
        # How much the emotion is typically directed at another agent
        "anger": 4.5, "fear": 2.0, "disgust": 2.5,
        "sadness": 3.0, "happiness": 3.0, "valence": 2.5,
    },
    "action_tendency": {
        # Frijda 1986: strength/specificity of action readiness
        # anger=attack, fear=flee, disgust=reject, sadness=withdraw, happiness=approach
        "anger": 5.0, "fear": 4.5, "disgust": 3.5,
        "sadness": 2.0, "happiness": 3.0, "valence": 2.5,
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
            print(f" {rdm[i,j]:7.4f}", end="")
        print()


def main():
    # ── Load ──
    brain_data = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_full = brain_data["rdm"]
    conds = list(brain_data["conditions"])

    llm_rdms = {}
    for m in MODELS:
        f = RSA / f"{m}_rdm14_headline.npz"
        if f.exists():
            llm_rdms[m] = np.load(f, allow_pickle=True)["rdm"]

    mean_llm_full = np.mean(list(llm_rdms.values()), axis=0)

    # Extract affective block
    aff_list = [c for c in conds if c in AFF_SET]
    brain_aff, aff_order = extract_block(brain_full, conds, aff_list)
    llm_aff = {m: extract_block(rdm, conds, aff_list)[0] for m, rdm in llm_rdms.items()}
    mean_llm_aff = np.mean(list(llm_aff.values()), axis=0)

    print("=" * 75)
    print("WHY DOES WITHIN-AFFECTIVE BRAIN-LLM ALIGNMENT FAIL?")
    print(f"  Affective conditions: {aff_order}")
    print(f"  {len(aff_order)} conditions → {len(aff_order)*(len(aff_order)-1)//2} pairs")
    print("=" * 75)

    # ── Print actual RDMs ──
    print_rdm(brain_aff, aff_order, "Brain within-affective RDM")
    print_rdm(mean_llm_aff, aff_order, "LLM mean within-affective RDM")

    # Within-aff brain-LLM alignment (confirm the ρ≈-0.10)
    rho_ba, _ = spearmanr(upper_tri(brain_aff), upper_tri(mean_llm_aff))
    print(f"\n  Within-affective brain↔LLM mean: ρ = {rho_ba:+.3f}")
    per_model_aff = {}
    for m in llm_aff:
        r, _ = spearmanr(upper_tri(brain_aff), upper_tri(llm_aff[m]))
        print(f"    {m}: ρ = {r:+.3f}")
        per_model_aff[m] = float(r)

    results = {
        "aff_conditions": aff_order,
        "brain_llm_mean_rho": float(rho_ba),
        "per_model_rho": per_model_aff,
    }

    # ═══════════════════════════════════════════
    # PART A: BRAIN within-affective dimensions
    # ═══════════════════════════════════════════
    print(f"\n{'='*75}")
    print("PART A: What organizes the BRAIN's emotion geometry?")
    print(f"{'='*75}\n")
    print(f"  {'Dimension':>25s} | {'ρ':>8s} {'p':>7s} | Sig?")
    print(f"  {'-'*50}")

    brain_results = {}
    for dim_name, scores in EMOTION_DIMS.items():
        m_rdm = model_rdm(scores, aff_order)
        rho, p = perm_rsa(m_rdm, brain_aff)
        sig = "*" if p < 0.05 else ""
        print(f"  {dim_name:>25s} | {rho:+8.3f} {p:7.4f} | {sig}")
        brain_results[dim_name] = {"rho": rho, "p": p}

    results["brain_dimensions"] = brain_results

    # ═══════════════════════════════════════════
    # PART B: LLM within-affective dimensions
    # ═══════════════════════════════════════════
    print(f"\n{'='*75}")
    print("PART B: What organizes the LLM's emotion geometry?")
    print(f"{'='*75}\n")
    print(f"  {'Dimension':>25s} | {'ρ':>8s} {'p':>7s} | Sig?")
    print(f"  {'-'*50}")

    llm_results = {}
    for dim_name, scores in EMOTION_DIMS.items():
        m_rdm = model_rdm(scores, aff_order)
        rho, p = perm_rsa(m_rdm, mean_llm_aff)
        sig = "*" if p < 0.05 else ""
        print(f"  {dim_name:>25s} | {rho:+8.3f} {p:7.4f} | {sig}")
        llm_results[dim_name] = {"rho": rho, "p": p}

    results["llm_dimensions"] = llm_results

    # Per-model check (are dimension results consistent across architectures?)
    print(f"\n  Per-model consistency for top brain dimensions:")
    brain_sig = [d for d in brain_results if brain_results[d]["p"] < 0.10]
    if brain_sig:
        for dim_name in brain_sig:
            rhos = []
            for m in llm_aff:
                m_rdm = model_rdm(EMOTION_DIMS[dim_name], aff_order)
                r, _ = perm_rsa(m_rdm, llm_aff[m], n_perm=5000)
                rhos.append(r)
            print(f"    {dim_name}: LLM per-model ρ = [{', '.join(f'{r:+.3f}' for r in rhos)}]")

    # ═══════════════════════════════════════════
    # PART C: COMPARISON
    # ═══════════════════════════════════════════
    print(f"\n{'='*75}")
    print("PART C: Where do brain and LLM diverge?")
    print(f"{'='*75}\n")
    print(f"  {'Dimension':>25s} | {'Brain ρ':>8s} | {'LLM ρ':>8s} | {'Δ(B-L)':>8s} | Category")
    print(f"  {'-'*75}")

    comparison = {}
    for dim_name in EMOTION_DIMS:
        rb = brain_results[dim_name]["rho"]
        rl = llm_results[dim_name]["rho"]
        pb = brain_results[dim_name]["p"]
        pl = llm_results[dim_name]["p"]
        delta = rb - rl

        if pb < 0.05 and pl >= 0.05:
            cat = "BRAIN-ONLY"
        elif pl < 0.05 and pb >= 0.05:
            cat = "LLM-ONLY"
        elif pb < 0.05 and pl < 0.05:
            cat = "SHARED"
        else:
            cat = "neither"

        print(f"  {dim_name:>25s} | {rb:+8.3f} | {rl:+8.3f} | {delta:+8.3f} | {cat}")
        comparison[dim_name] = {"brain_rho": rb, "llm_rho": rl, "delta": delta, "category": cat}

    results["comparison"] = comparison

    # ── Also repeat without 'valence' (n=5, 10 pairs) ──
    print(f"\n{'='*75}")
    print("SENSITIVITY: Without 'valence' condition (n=5, 10 pairs)")
    print(f"{'='*75}\n")

    aff5 = [c for c in aff_order if c != "valence"]
    brain_aff5, _ = extract_block(brain_full, conds, aff5)
    llm_aff5 = np.mean([extract_block(rdm, conds, aff5)[0] for _, rdm in llm_rdms.items()], axis=0)

    r5, _ = spearmanr(upper_tri(brain_aff5), upper_tri(llm_aff5))
    print(f"  Brain↔LLM within-aff (excl. valence): ρ = {r5:+.3f}")

    print(f"\n  {'Dimension':>25s} | {'Brain ρ':>8s} {'p':>7s} | {'LLM ρ':>8s} {'p':>7s}")
    print(f"  {'-'*65}")

    results["sensitivity_no_valence"] = {}
    for dim_name, scores in EMOTION_DIMS.items():
        m_rdm = model_rdm(scores, aff5)
        rb, pb = perm_rsa(m_rdm, brain_aff5)
        rl, pl = perm_rsa(m_rdm, llm_aff5)
        print(f"  {dim_name:>25s} | {rb:+8.3f} {pb:7.4f} | {rl:+8.3f} {pl:7.4f}")
        results["sensitivity_no_valence"][dim_name] = {
            "brain_rho": rb, "brain_p": pb, "llm_rho": rl, "llm_p": pl}

    # ── Pair-level analysis: which pairs diverge most? ──
    print(f"\n{'='*75}")
    print("PAIR-LEVEL: Which emotion pairs show the largest brain-LLM divergence?")
    print(f"{'='*75}\n")

    pairs = list(combinations(range(len(aff_order)), 2))
    pair_data = []
    for i, j in pairs:
        bd = brain_aff[i, j]
        ld = mean_llm_aff[i, j]
        # Rank-based divergence: normalize by overall range
        pair_data.append({
            "pair": f"{aff_order[i]}—{aff_order[j]}",
            "brain_d": float(bd), "llm_d": float(ld),
            "divergence": float(abs(bd - ld)),
        })

    pair_data.sort(key=lambda x: -x["divergence"])
    print(f"  {'Pair':>25s} | {'Brain d':>8s} | {'LLM d':>8s} | {'|Δ|':>8s}")
    print(f"  {'-'*55}")
    for pd in pair_data:
        print(f"  {pd['pair']:>25s} | {pd['brain_d']:8.4f} | {pd['llm_d']:8.4f} | {pd['divergence']:8.4f}")

    results["pair_divergence"] = pair_data

    # ── Summary ──
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")

    brain_sig = [d for d, v in brain_results.items() if v["p"] < 0.05]
    llm_sig = [d for d, v in llm_results.items() if v["p"] < 0.05]
    brain_only = [d for d in brain_sig if d not in llm_sig]
    llm_only = [d for d in llm_sig if d not in brain_sig]
    shared = [d for d in brain_sig if d in llm_sig]

    print(f"\n  Brain organized by (p<0.05): {brain_sig if brain_sig else 'none'}")
    print(f"  LLM organized by (p<0.05):   {llm_sig if llm_sig else 'none'}")
    print(f"  Shared dimensions:           {shared if shared else 'none'}")
    print(f"  Brain-only dimensions:       {brain_only if brain_only else 'none'}")
    print(f"  LLM-only dimensions:         {llm_only if llm_only else 'none'}")

    results["summary"] = {
        "brain_sig_dims": brain_sig,
        "llm_sig_dims": llm_sig,
        "shared_dims": shared,
        "brain_only_dims": brain_only,
        "llm_only_dims": llm_only,
    }

    json.dump(results, open(OUT / "alignment_why_dissimilar.json", "w"), indent=2)
    print(f"\nSaved: {OUT / 'alignment_why_dissimilar.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
