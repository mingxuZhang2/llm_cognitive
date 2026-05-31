#!/usr/bin/env python3
"""
Within-dataset (single-ruler) boundary test using IBC (Individual Brain Charting,
NeuroVault coll. 2138). Same 12 subjects, same pipeline -> no batch confound.

IBC gives a RICHER mentalistic side than HCP. Affective side is still face-based
(controlled fMRI evokes emotion via faces), so we keep that caveat explicit.

Conditions (group mean over per-subject contrast maps):
  AFFECTIVE   : fear   <- face-shape (emotion faces)        [face-based caveat]
                valence<- reward-punishment (gambling)
  MENTALISTIC : belief <- belief-mechanistic_video (false belief)
                theory_of_mind <- mental-random (triangle animacy)
                intention <- intention-control (intention attribution)
                judgment  <- trusty-control (trustworthiness judgment)

Tests:
  1. IBC boundary: mean(between-block) - mean(within-block); exact label-perm p.
  2. IBC 6x6 RDM vs each LLM 6x6 RDM (peak layer) over corresponding conditions.
  3. IBC vs Neurosynth (brain-brain) over the same 6.
  4. Mentalistic-cluster cohesion: is the 4-cond mentalistic block internally
     tighter than its distance to the 2 affective conds? (the cleaner sub-claim)
"""
import json, os, re
from pathlib import Path
from itertools import permutations, combinations

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
IBC_DIR = BASE / "data" / "brain_maps" / "ibc"
NS_PATH = BASE / "results" / "cognitive_rsa" / "brain_rdm.npz"
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "affective_validation"

# base-contrast dir -> (our condition, block)
CHOICE = {
    "face-shape":               ("fear",           "affective"),
    "reward-punishment":        ("valence",        "affective"),
    "belief-mechanistic_video": ("belief",         "mentalistic"),
    "mental-random":            ("theory_of_mind", "mentalistic"),
    "intention-control":        ("intention",      "mentalistic"),
    "trusty-control":           ("judgment",       "mentalistic"),
}
PEAK = {
    "Qwen2.5-7B-Instruct": "layer_26",
    "Meta-Llama-3.1-8B-Instruct": "layer_30",
    "Mistral-7B-Instruct-v0.3": "layer_13",
    "gemma-2-9b-it": "layer_20",
}


def group_mean_vec(contrast_dir, ref_affine=None, ref_shape=None):
    """Average all per-subject maps in a contrast dir. Resample to ref grid if set."""
    from nibabel.processing import resample_from_to
    files = sorted(f for f in os.listdir(contrast_dir) if f.endswith(".nii.gz"))
    acc, n, used_ref = None, 0, (ref_affine, ref_shape)
    for f in files:
        img = nib.load(str(Path(contrast_dir) / f))
        if used_ref[1] is None:
            used_ref = (img.affine, img.shape[:3])
        if img.shape[:3] != used_ref[1] or not np.allclose(img.affine, used_ref[0]):
            img = resample_from_to(img, (used_ref[1], used_ref[0]), order=1)
        d = img.get_fdata(dtype=np.float64)
        d = np.nan_to_num(d)
        acc = d if acc is None else acc + d
        n += 1
    return (acc / n).ravel(), n, used_ref[0], used_ref[1]


def rdm_from_vecs(vecs, mask):
    X = np.stack([v[mask] for v in vecs]).astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    nn = np.linalg.norm(X, axis=1, keepdims=True); nn[nn == 0] = 1.0
    Xn = X / nn
    return 1.0 - (Xn @ Xn.T)


def upper(m):
    iu = np.triu_indices(m.shape[0], k=1); return m[iu]


def boundary_stat(rdm, blocks):
    n = len(blocks); within, between = [], []
    for i, j in combinations(range(n), 2):
        (between if blocks[i] != blocks[j] else within).append(rdm[i, j])
    return float(np.mean(between) - np.mean(within)), float(np.mean(within)), float(np.mean(between))


def exact_boundary_perm(rdm, blocks):
    obs, _, _ = boundary_stat(rdm, blocks)
    seen, stats = set(), []
    for p in permutations(blocks):
        if p in seen: continue
        seen.add(p); s, _, _ = boundary_stat(rdm, list(p)); stats.append(s)
    stats = np.array(stats)
    return obs, float((stats >= obs - 1e-12).mean()), len(stats)


def exact_rdm_perm(a, b):
    n = a.shape[0]; obs, _ = spearmanr(upper(a), upper(b)); rhos = []
    for p in permutations(range(n)):
        rhos.append(spearmanr(upper(a), upper(b[np.ix_(p, p)]))[0])
    rhos = np.array(rhos)
    return float(obs), float((np.abs(rhos) >= abs(obs) - 1e-12).mean())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conds, blocks, vecs, ns_used = [], [], [], []
    ref_aff, ref_shp = None, None
    for cdir, (cond, block) in CHOICE.items():
        v, n, ref_aff, ref_shp = group_mean_vec(IBC_DIR / cdir, ref_aff, ref_shp)
        conds.append(cond); blocks.append(block); vecs.append(v); ns_used.append(n)
        print(f"  {cdir:28s} -> {cond:14s} [{block}]  ({n} maps averaged)")
    print(f"Reference grid: {ref_shp}")

    stacked = np.stack(vecs)
    mask = np.all(np.isfinite(stacked) & (stacked != 0), axis=0)
    print(f"Shared-nonzero voxels: {int(mask.sum())}")

    ibc_rdm = rdm_from_vecs(vecs, mask)
    print("\nIBC 6x6 RDM order:", conds)
    print(np.round(ibc_rdm, 3))

    bstat, w, b = boundary_stat(ibc_rdm, blocks)
    bobs, bp, nperm = exact_boundary_perm(ibc_rdm, blocks)
    print(f"\n[IBC boundary] within={w:.3f} between={b:.3f} between-within={bstat:+.3f} "
          f"exact_p={bp:.3f} ({nperm} perms)")

    # mentalistic-cluster cohesion: within-mentalistic vs mentalistic-to-affective
    ment_idx = [i for i, bl in enumerate(blocks) if bl == "mentalistic"]
    aff_idx = [i for i, bl in enumerate(blocks) if bl == "affective"]
    wm = np.mean([ibc_rdm[i, j] for i, j in combinations(ment_idx, 2)])
    cross = np.mean([ibc_rdm[i, j] for i in ment_idx for j in aff_idx])
    print(f"[IBC mentalistic cohesion] within-ment={wm:.3f}  ment-to-aff={cross:.3f}  gap={cross-wm:+.3f}")

    results = {"source": "IBC_coll2138_groupmean", "conditions": conds, "blocks": blocks,
               "n_maps": ns_used, "ref_shape": list(ref_shp), "shared_voxels": int(mask.sum()),
               "ibc_rdm": ibc_rdm.tolist(),
               "boundary": {"within": w, "between": b, "between_minus_within": bstat,
                            "exact_p": bp, "n_perm": nperm},
               "ment_cohesion": {"within_ment": float(wm), "ment_to_aff": float(cross),
                                 "gap": float(cross - wm)},
               "comparisons": {}}

    for model, peak in PEAK.items():
        npz = np.load(RSA_DIR / f"{model}_rsa_llm_rdms.npz", allow_pickle=True)
        li = list(npz["layer_names"]).index(peak)
        lr = npz["llm_rdms"][li]; lc = list(npz["conditions"])
        idx = [lc.index(c) for c in conds]
        llm = lr[np.ix_(idx, idx)]
        rho, p = exact_rdm_perm(ibc_rdm, llm)
        lb, _, _ = boundary_stat(llm, blocks)
        print(f"[IBC vs {model:28s}] rho={rho:+.3f} exact_p={p:.3f} | LLM boundary={lb:+.3f}")
        results["comparisons"][model] = {"rho": rho, "exact_p": p, "llm_boundary": lb,
                                         "llm_rdm": llm.tolist()}

    ns = np.load(NS_PATH, allow_pickle=True)
    nr = ns["rdm"]; nc = list(ns["conditions"])
    nidx = [nc.index(c) for c in conds]
    ns6 = nr[np.ix_(nidx, nidx)]
    rho, p = exact_rdm_perm(ibc_rdm, ns6)
    nb, _, _ = boundary_stat(ns6, blocks)
    print(f"[IBC vs Neurosynth] rho={rho:+.3f} exact_p={p:.3f} | NS boundary={nb:+.3f}")
    results["comparisons"]["neurosynth"] = {"rho": rho, "exact_p": p, "ns_boundary": nb,
                                            "ns_rdm": ns6.tolist()}

    llm_rhos = [results["comparisons"][m]["rho"] for m in PEAK]
    llm_bnds = [results["comparisons"][m]["llm_boundary"] for m in PEAK]
    print("\n" + "=" * 64)
    print("SUMMARY — IBC single-dataset boundary (2 affective + 4 mentalistic)")
    print("=" * 64)
    print(f"  IBC brain boundary:        {bstat:+.3f} (p={bp:.3f})")
    print(f"  IBC mentalistic cohesion gap (cross-within): {cross-wm:+.3f}")
    print(f"  Neurosynth boundary same6: {nb:+.3f}")
    print(f"  LLM boundaries:            {[round(x,3) for x in llm_bnds]}")
    print(f"  IBC-vs-LLM rho:            {[round(x,3) for x in llm_rhos]} (mean {np.mean(llm_rhos):+.3f})")
    print(f"  IBC-vs-Neurosynth rho:     {results['comparisons']['neurosynth']['rho']:+.3f}")
    results["summary"] = {"ibc_boundary": bstat, "ibc_boundary_p": bp,
                          "ment_cohesion_gap": float(cross - wm),
                          "ns_boundary_same6": nb, "llm_boundaries": llm_bnds,
                          "llm_mean_rho": float(np.mean(llm_rhos)),
                          "ibc_vs_ns_rho": results["comparisons"]["neurosynth"]["rho"]}
    out = OUT_DIR / "ibc_boundary_validation.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
