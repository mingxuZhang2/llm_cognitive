#!/usr/bin/env python3
"""
Robustness check: the main 14x14 brain RDM uses HCP for theory_of_mind and
Neurosynth for the other 13. ToM is simultaneously (a) the only non-Neurosynth
map and (b) the only negatively-aligned condition. Could the negative ToM result
be an artifact of the HCP source rather than ToM itself?

This script rebuilds the brain RDM with theory_of_mind <- Neurosynth (so all 14
maps are Neurosynth), then recomputes, for each of the 4 models:
  - headline 14x14 RSA rho vs the new pure-Neurosynth brain RDM
  - per-condition row-wise brain alignment (esp. ToM)
and compares against the current HCP-version numbers.

No GPU. Uses the already-saved headline LLM RDMs (*_rdm14_headline.npz).
"""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn import image as nl_image
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
BRAIN = BASE / "data" / "brain_maps"
RSA = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "affective_validation" / "tom_source_check.json"

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
AFF = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}

# 14 conditions -> Neurosynth map ONLY (ToM swapped from HCP to Neurosynth)
NS_MAP = {
    "anger": "anger", "fear": "fear", "disgust": "disgust", "sadness": "sadness",
    "happiness": "happiness", "valence": "valence", "belief": "beliefs",
    "intention": "intention", "judgment": "judgment", "mentalizing": "mentalizing",
    "moral": "moral", "empathy": "empathy", "self_referential": "self_referential",
    "theory_of_mind": "theory_of_mind",   # <-- the swap (was hcp/social_tom_vs_random)
}


def pearson_rdm(X):
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    Xn = X / n
    return 1.0 - Xn @ Xn.T


def build_ns_brain_rdm(conditions):
    anchor = nib.load(str(BRAIN / "neurosynth" / f"{NS_MAP[conditions[0]]}.nii.gz"))
    arrays = []
    for c in conditions:
        img = nib.load(str(BRAIN / "neurosynth" / f"{NS_MAP[c]}.nii.gz"))
        if img.shape != anchor.shape or not np.allclose(img.affine, anchor.affine):
            img = nl_image.resample_to_img(img, anchor, interpolation="linear",
                                           force_resample=True, copy_header=True)
        arrays.append(np.asarray(img.get_fdata(dtype=np.float32)))
    stack = np.stack(arrays, 0)
    finite = np.isfinite(stack).all(0)
    nonzero = (stack != 0).sum(0)
    valid = finite & (nonzero >= len(conditions) // 2)
    vox = stack.reshape(len(conditions), -1)[:, valid.ravel()]
    vox = np.nan_to_num(vox).astype(np.float64)
    return pearson_rdm(vox), int(valid.sum())


def upper(m):
    iu = np.triu_indices(m.shape[0], 1); return m[iu]


def rowvals(R, i): return np.delete(R[i], i)


def main():
    # reference order from current (HCP) brain rdm
    cur = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    conds = list(cur["conditions"])
    cur_rdm = cur["rdm"]

    ns_rdm, nvox = build_ns_brain_rdm(conds)
    print(f"Built pure-Neurosynth brain RDM (ToM<-Neurosynth). valid voxels={nvox}")

    # brain-brain: how different are the two ToM rows?
    i = conds.index("theory_of_mind")
    rho_tom_rows = spearmanr(rowvals(cur_rdm, i), rowvals(ns_rdm, i))[0]
    print(f"ToM row (HCP) vs ToM row (Neurosynth): rho={rho_tom_rows:+.3f}  "
          f"(how much the ToM geometry changed)")

    results = {"conditions": conds, "valid_voxels": nvox,
               "tom_row_hcp_vs_ns_rho": float(rho_tom_rows), "models": {}}
    print(f"\n{'model':28s} {'rho_HCP':>8s} {'rho_NS':>8s} {'ToM_HCP':>8s} {'ToM_NS':>8s}")
    head_hcp_all, head_ns_all, tom_hcp_all, tom_ns_all = [], [], [], []
    for m in MODELS:
        z = np.load(RSA / f"{m}_rdm14_headline.npz", allow_pickle=True)
        lrdm = z["rdm"]; lconds = list(z["conditions"])
        order = [lconds.index(c) for c in conds]
        L = lrdm[np.ix_(order, order)]

        rho_hcp = spearmanr(upper(L), upper(cur_rdm))[0]
        rho_ns = spearmanr(upper(L), upper(ns_rdm))[0]
        tom_hcp = spearmanr(rowvals(L, i), rowvals(cur_rdm, i))[0]
        tom_ns = spearmanr(rowvals(L, i), rowvals(ns_rdm, i))[0]
        head_hcp_all.append(rho_hcp); head_ns_all.append(rho_ns)
        tom_hcp_all.append(tom_hcp); tom_ns_all.append(tom_ns)

        # also affective & mentalistic block means under NS version
        def blockmean(R, names):
            return float(np.mean([spearmanr(rowvals(L, conds.index(c)),
                                            rowvals(R, conds.index(c)))[0] for c in names]))
        aff_ns = blockmean(ns_rdm, [c for c in conds if c in AFF])
        men_ns = blockmean(ns_rdm, [c for c in conds if c not in AFF])

        results["models"][m] = {
            "headline_rho_HCP": float(rho_hcp), "headline_rho_NS": float(rho_ns),
            "tom_align_HCP": float(tom_hcp), "tom_align_NS": float(tom_ns),
            "aff_mean_NS": aff_ns, "ment_mean_NS": men_ns}
        print(f"{m:28s} {rho_hcp:>+8.3f} {rho_ns:>+8.3f} {tom_hcp:>+8.3f} {tom_ns:>+8.3f}")

    print(f"\n{'MEAN':28s} {np.mean(head_hcp_all):>+8.3f} {np.mean(head_ns_all):>+8.3f} "
          f"{np.mean(tom_hcp_all):>+8.3f} {np.mean(tom_ns_all):>+8.3f}")
    print("\nInterpretation:")
    print(f"  Headline rho  : HCP {np.mean(head_hcp_all):+.3f} -> NS {np.mean(head_ns_all):+.3f}")
    print(f"  ToM alignment : HCP {np.mean(tom_hcp_all):+.3f} -> NS {np.mean(tom_ns_all):+.3f}")
    results["summary"] = {
        "headline_rho_HCP_mean": float(np.mean(head_hcp_all)),
        "headline_rho_NS_mean": float(np.mean(head_ns_all)),
        "tom_align_HCP_mean": float(np.mean(tom_hcp_all)),
        "tom_align_NS_mean": float(np.mean(tom_ns_all))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
