#!/usr/bin/env python3
"""
Independent affective validation using Kragel et al. 2015 controlled emotion fMRI.

Goal: test whether the affective representational geometry we found (Neurosynth
brain RDM <-> LLM RDM) REPLICATES against a *fully independent, controlled*
brain source. The Kragel maps are group-level emotion-category signature maps
(PLS bootstrap-z) derived from controlled film/music emotion induction (N=32,
MNI). They are NOT meta-analytic and NOT naturalistic story-listening.

We do NOT pool maps into one RDM (avoids batch effects). Instead we build a
Kragel-only emotion RDM and ask: does it agree with (a) the Neurosynth affective
sub-RDM and (b) each LLM's affective sub-RDM?

Caveat (logged): Kragel maps are predictive/classifier weight patterns, so their
geometry reflects discriminative signatures rather than raw evoked activation.
This is a first-pass independent check; raw trial estimates are a future upgrade.

Overlap with our 14-condition set: anger, fear, sadness, happiness (= amused).
disgust and valence are not available as discrete Kragel categories.
Small n (4 conditions -> 6 pairs) => exact 4! permutation test.
"""
import json
from pathlib import Path
from itertools import permutations

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
KRAGEL_DIR = BASE / "data" / "brain_maps" / "kragel_emotion"
NS_PATH = BASE / "results" / "cognitive_rsa" / "brain_rdm.npz"
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "affective_validation"

# Kragel category file stems
KRAGEL = {
    "amused":    "mean_3comp_amused_group_emotion_PLS_beta_BSz_10000it.img",
    "angry":     "mean_3comp_angry_group_emotion_PLS_beta_BSz_10000it.img",
    "content":   "mean_3comp_content_group_emotion_PLS_beta_BSz_10000it.img",
    "fearful":   "mean_3comp_fearful_group_emotion_PLS_beta_BSz_10000it.img",
    "neutral":   "mean_3comp_neutral_group_emotion_PLS_beta_BSz_10000it.img",
    "sad":       "mean_3comp_sad_group_emotion_PLS_beta_BSz_10000it.img",
    "surprised": "mean_3comp_surprised_group_emotion_PLS_beta_BSz_10000it.img",
}
# Map Kragel category -> our condition name
K2OURS = {"angry": "anger", "fearful": "fear", "sad": "sadness", "amused": "happiness"}

# Peak layer name per model (from cross_model_summary_v2)
PEAK = {
    "Qwen2.5-7B-Instruct": "layer_26",
    "Meta-Llama-3.1-8B-Instruct": "layer_30",
    "Mistral-7B-Instruct-v0.3": "layer_13",
    "gemma-2-9b-it": "layer_20",
}


def rdm_from_maps(vecs):
    """1 - Pearson RDM from a list of flattened, co-registered maps."""
    X = np.stack(vecs).astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = X / n
    return 1.0 - (Xn @ Xn.T)


def upper(m):
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def exact_perm_test(rdm_a, rdm_b):
    """Exact permutation test over condition-label permutations of rdm_b.
    Returns observed Spearman rho and exact two-sided p over all n! perms."""
    n = rdm_a.shape[0]
    obs, _ = spearmanr(upper(rdm_a), upper(rdm_b))
    rhos = []
    for p in permutations(range(n)):
        pm = rdm_b[np.ix_(p, p)]
        r, _ = spearmanr(upper(rdm_a), upper(pm))
        rhos.append(r)
    rhos = np.array(rhos)
    # two-sided: fraction of |perm rho| >= |obs|
    p_two = float((np.abs(rhos) >= abs(obs) - 1e-12).mean())
    return float(obs), p_two, int(n), len(rhos)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load all 7 Kragel maps in their native (shared) voxel grid ----
    cat_vecs = {}
    ref_shape = None
    for cat, fname in KRAGEL.items():
        img = nib.load(str(KRAGEL_DIR / fname))
        d = img.get_fdata(dtype=np.float64)
        if ref_shape is None:
            ref_shape = d.shape
        assert d.shape == ref_shape, f"{cat} shape mismatch"
        cat_vecs[cat] = d.ravel()
    # restrict to voxels finite AND nonzero in ALL maps (the analysis mask).
    # NOTE: .img background is NaN, and NaN != 0 is True, so we must test finiteness.
    stacked = np.stack([cat_vecs[c] for c in KRAGEL])
    mask = np.all(np.isfinite(stacked) & (stacked != 0), axis=0)
    print(f"Kragel grid {ref_shape}, shared-nonzero voxels: {int(mask.sum())}")
    for c in cat_vecs:
        cat_vecs[c] = cat_vecs[c][mask]

    # ---- Full 7x7 Kragel emotion RDM (record/visualize) ----
    cats7 = list(KRAGEL.keys())
    rdm7 = rdm_from_maps([cat_vecs[c] for c in cats7])
    print("\nFull Kragel 7x7 emotion RDM:")
    print("  cats:", cats7)
    print(np.round(rdm7, 3))

    # ---- 4-condition overlap with our set ----
    kr_overlap = ["angry", "fearful", "sad", "amused"]
    ours_overlap = [K2OURS[c] for c in kr_overlap]   # anger, fear, sadness, happiness
    kragel_rdm4 = rdm_from_maps([cat_vecs[c] for c in kr_overlap])
    print(f"\nOverlap conditions (ours): {ours_overlap}")
    print("Kragel 4x4 RDM:\n", np.round(kragel_rdm4, 3))

    results = {
        "source": "Kragel2015_emotion_PLS_BSz",
        "note": "controlled film/music emotion induction, N=32, MNI; predictive signature maps",
        "kragel_grid": list(ref_shape),
        "shared_voxels": int(mask.sum()),
        "overlap_conditions": ours_overlap,
        "kragel_rdm4": kragel_rdm4.tolist(),
        "kragel_cats7": cats7,
        "kragel_rdm7": rdm7.tolist(),
        "comparisons": {},
    }

    # ---- Neurosynth affective sub-RDM (same 4 conditions) ----
    ns = np.load(NS_PATH, allow_pickle=True)
    ns_rdm = ns["rdm"]
    ns_conds = list(ns["conditions"])
    idx = [ns_conds.index(c) for c in ours_overlap]
    ns_rdm4 = ns_rdm[np.ix_(idx, idx)]
    print("\nNeurosynth 4x4 (same conds):\n", np.round(ns_rdm4, 3))

    rho, p, n, nperm = exact_perm_test(kragel_rdm4, ns_rdm4)
    print(f"\n[Kragel vs Neurosynth]  rho={rho:+.3f}  exact_p={p:.4f}  (n={n}, {nperm} perms)")
    results["comparisons"]["neurosynth"] = {
        "rho": rho, "exact_p": p, "n_cond": n, "n_perm": nperm,
        "rdm4": ns_rdm4.tolist(),
    }

    # ---- Each LLM's affective sub-RDM at peak layer ----
    for model, peak_name in PEAK.items():
        npz = np.load(RSA_DIR / f"{model}_rsa_llm_rdms.npz", allow_pickle=True)
        llm_rdms = npz["llm_rdms"]            # (layers,14,14)
        conds = list(npz["conditions"])
        lnames = list(npz["layer_names"])
        li = lnames.index(peak_name)
        llm_rdm = llm_rdms[li]
        lidx = [conds.index(c) for c in ours_overlap]
        llm_rdm4 = llm_rdm[np.ix_(lidx, lidx)]

        rho, p, n, nperm = exact_perm_test(kragel_rdm4, llm_rdm4)
        print(f"[Kragel vs {model:30s} @{peak_name}]  rho={rho:+.3f}  exact_p={p:.4f}")
        results["comparisons"][model] = {
            "peak_layer": peak_name, "rho": rho, "exact_p": p,
            "n_cond": n, "n_perm": nperm, "rdm4": llm_rdm4.tolist(),
        }

    # ---- Summary ----
    llm_rhos = [results["comparisons"][m]["rho"] for m in PEAK]
    print("\n" + "=" * 60)
    print("SUMMARY (affective geometry replication, Kragel controlled fMRI)")
    print("=" * 60)
    print(f"  Kragel vs Neurosynth (brain-brain): rho={results['comparisons']['neurosynth']['rho']:+.3f}")
    print(f"  Kragel vs 4 LLMs: rhos={[round(r,3) for r in llm_rhos]}, mean={np.mean(llm_rhos):+.3f}")
    results["llm_mean_rho"] = float(np.mean(llm_rhos))

    out = OUT_DIR / "kragel_affective_validation.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
