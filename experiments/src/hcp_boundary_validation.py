#!/usr/bin/env python3
"""
Within-dataset (single-ruler) test of the affective<->mentalistic BOUNDARY using
HCP S1200 group-average task contrasts (NeuroVault coll. 457, same subjects/
pipeline -> no cross-dataset batch confound).

The rho=0.63 brain<->LLM alignment is driven entirely by the between-block
separation (affective conditions vs mentalistic conditions), NOT by within-block
fine geometry (which is ~0 even for Neurosynth-vs-LLM). So the right thing to
validate with controlled fMRI is the BOUNDARY, and it must be measured within a
single dataset that contains BOTH kinds of task -- HCP does.

Clean difference-contrasts that map onto our two blocks:
  AFFECTIVE   : emotion_faces_vs_shapes (fearful/angry faces), gambling_reward_vs_punish (valence)
  MENTALISTIC : social_tom_vs_random (theory of mind), relational_rel_vs_match (judgment)

Tests:
  1. HCP boundary statistic: mean(between-block dist) - mean(within-block dist).
     Positive => affective and mentalistic separate within real controlled fMRI.
     Exact permutation over block-label assignments (small n; reported honestly).
  2. HCP 4x4 RDM vs each LLM 4x4 RDM (peak layer) over the corresponding conditions.
  3. HCP 4x4 RDM vs Neurosynth 4x4 RDM (brain-brain convergence).
Also prints a wider 6-contrast HCP RDM (adds language_story_vs_math, wm_2bk_vs_0bk)
for descriptive context.
"""
import json
from pathlib import Path
from itertools import permutations, combinations

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
HCP_DIR = BASE / "data" / "brain_maps" / "hcp"
NS_PATH = BASE / "results" / "cognitive_rsa" / "brain_rdm.npz"
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "affective_validation"

# HCP contrast slug -> (our LLM condition analog, block)
HCP_MAP = {
    "emotion_faces_vs_shapes":   ("fear",           "affective"),
    "gambling_reward_vs_punish": ("valence",        "affective"),
    "social_tom_vs_random":      ("theory_of_mind", "mentalistic"),
    "relational_rel_vs_match":   ("judgment",       "mentalistic"),
}
EXTRA = ["language_story_vs_math", "wm_2bk_vs_0bk"]  # descriptive only

PEAK = {
    "Qwen2.5-7B-Instruct": "layer_26",
    "Meta-Llama-3.1-8B-Instruct": "layer_30",
    "Mistral-7B-Instruct-v0.3": "layer_13",
    "gemma-2-9b-it": "layer_20",
}


def load_vec(slug):
    img = nib.load(str(HCP_DIR / f"{slug}.nii.gz"))
    return img.get_fdata(dtype=np.float64).ravel()


def rdm_from_vecs(vecs):
    X = np.stack(vecs).astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = X / n
    return 1.0 - (Xn @ Xn.T)


def upper(m):
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def boundary_stat(rdm, blocks):
    """mean(between-block) - mean(within-block) on an RDM, given block labels."""
    n = len(blocks)
    within, between = [], []
    for i, j in combinations(range(n), 2):
        (between if blocks[i] != blocks[j] else within).append(rdm[i, j])
    return float(np.mean(between) - np.mean(within)), float(np.mean(within)), float(np.mean(between))


def exact_boundary_perm(rdm, blocks):
    """Exact permutation of block labels; p = P(stat >= observed)."""
    obs, _, _ = boundary_stat(rdm, blocks)
    uniq = list(set(blocks))
    # enumerate all assignments of the same composition (multiset of labels)
    from itertools import permutations as perm
    seen = set()
    stats = []
    for p in perm(blocks):
        if p in seen:
            continue
        seen.add(p)
        s, _, _ = boundary_stat(rdm, list(p))
        stats.append(s)
    stats = np.array(stats)
    p_val = float((stats >= obs - 1e-12).mean())
    return obs, p_val, len(stats)


def exact_rdm_perm(rdm_a, rdm_b):
    n = rdm_a.shape[0]
    obs, _ = spearmanr(upper(rdm_a), upper(rdm_b))
    rhos = []
    for p in permutations(range(n)):
        pm = rdm_b[np.ix_(p, p)]
        r, _ = spearmanr(upper(rdm_a), upper(pm))
        rhos.append(r)
    rhos = np.array(rhos)
    return float(obs), float((np.abs(rhos) >= abs(obs) - 1e-12).mean())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slugs = list(HCP_MAP.keys())
    conds = [HCP_MAP[s][0] for s in slugs]
    blocks = [HCP_MAP[s][1] for s in slugs]

    # sanity: all HCP maps share a grid
    vecs = [load_vec(s) for s in slugs]
    lens = {v.shape[0] for v in vecs}
    assert len(lens) == 1, f"HCP grid mismatch: {lens}"

    hcp_rdm = rdm_from_vecs(vecs)
    print("HCP 4x4 RDM (affective: fear,valence | mentalistic: ToM,judgment)")
    print("  order:", conds)
    print(np.round(hcp_rdm, 3))

    # ---- Test 1: boundary within HCP ----
    bstat, w, b = boundary_stat(hcp_rdm, blocks)
    bobs, bp, nperm = exact_boundary_perm(hcp_rdm, blocks)
    print(f"\n[HCP boundary] within={w:.3f} between={b:.3f} "
          f"between-within={bstat:+.3f}  exact_p={bp:.3f} ({nperm} label-perms)")

    results = {
        "source": "HCP_S1200_groupavg_coll457",
        "conditions": conds, "blocks": blocks,
        "hcp_rdm4": hcp_rdm.tolist(),
        "boundary": {"within": w, "between": b, "between_minus_within": bstat,
                     "exact_p": bp, "n_perm": nperm},
        "comparisons": {},
    }

    # ---- Test 2: HCP vs each LLM over the 4 corresponding conditions ----
    for model, peak in PEAK.items():
        npz = np.load(RSA_DIR / f"{model}_rsa_llm_rdms.npz", allow_pickle=True)
        li = list(npz["layer_names"]).index(peak)
        lr = npz["llm_rdms"][li]
        lc = list(npz["conditions"])
        idx = [lc.index(c) for c in conds]
        llm_rdm = lr[np.ix_(idx, idx)]
        rho, p = exact_rdm_perm(hcp_rdm, llm_rdm)
        lb_stat, _, _ = boundary_stat(llm_rdm, blocks)
        print(f"[HCP vs {model:28s}] rho={rho:+.3f} exact_p={p:.3f} | LLM boundary={lb_stat:+.3f}")
        results["comparisons"][model] = {"rho": rho, "exact_p": p,
                                         "llm_boundary": lb_stat,
                                         "llm_rdm4": llm_rdm.tolist()}

    # ---- Test 3: HCP vs Neurosynth over the 4 conditions ----
    ns = np.load(NS_PATH, allow_pickle=True)
    ns_rdm = ns["rdm"]; ns_conds = list(ns["conditions"])
    nidx = [ns_conds.index(c) for c in conds]
    ns_rdm4 = ns_rdm[np.ix_(nidx, nidx)]
    rho, p = exact_rdm_perm(hcp_rdm, ns_rdm4)
    ns_bstat, _, _ = boundary_stat(ns_rdm4, blocks)
    print(f"[HCP vs Neurosynth (brain-brain)] rho={rho:+.3f} exact_p={p:.3f} | NS boundary={ns_bstat:+.3f}")
    results["comparisons"]["neurosynth"] = {"rho": rho, "exact_p": p,
                                            "ns_boundary": ns_bstat,
                                            "ns_rdm4": ns_rdm4.tolist()}

    # ---- Descriptive: wider 6-contrast HCP RDM ----
    wide_slugs = slugs + [e for e in EXTRA if (HCP_DIR / f"{e}.nii.gz").exists()]
    wide_vecs = [load_vec(s) for s in wide_slugs]
    wide_rdm = rdm_from_vecs(wide_vecs)
    print("\n[Descriptive] HCP 6-contrast RDM:")
    print("  order:", wide_slugs)
    print(np.round(wide_rdm, 3))
    results["wide_slugs"] = wide_slugs
    results["wide_rdm"] = wide_rdm.tolist()

    # ---- Summary ----
    llm_rhos = [results["comparisons"][m]["rho"] for m in PEAK]
    llm_bnds = [results["comparisons"][m]["llm_boundary"] for m in PEAK]
    print("\n" + "=" * 64)
    print("SUMMARY — boundary validation in HCP (single controlled dataset)")
    print("=" * 64)
    print(f"  HCP brain boundary (between-within): {bstat:+.3f} (p={bp:.3f})")
    print(f"  Neurosynth boundary on same 4:       {ns_bstat:+.3f}")
    print(f"  LLM boundaries (4 models):           {[round(x,3) for x in llm_bnds]}")
    print(f"  HCP-vs-LLM rho:                      {[round(x,3) for x in llm_rhos]} (mean {np.mean(llm_rhos):+.3f})")
    print(f"  HCP-vs-Neurosynth rho:               {results['comparisons']['neurosynth']['rho']:+.3f}")
    results["summary"] = {
        "hcp_boundary": bstat, "hcp_boundary_p": bp,
        "ns_boundary_same4": ns_bstat,
        "llm_boundaries": llm_bnds, "llm_mean_rho": float(np.mean(llm_rhos)),
        "hcp_vs_ns_rho": results["comparisons"]["neurosynth"]["rho"],
    }

    out = OUT_DIR / "hcp_boundary_validation.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
