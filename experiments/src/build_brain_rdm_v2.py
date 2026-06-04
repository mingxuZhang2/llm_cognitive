"""
v2 brain RDM: selectively override Neurosynth concept maps with HCP task
contrasts where HCP provides a cleaner, well-matched signal *without*
collapsing distinct conditions onto the same map.

Conservative HCP overrides (3 conditions):
  theory_of_mind  -> hcp/social_tom_vs_random.nii.gz       (already in v1)
  judgment        -> hcp/relational_rel_vs_match.nii.gz    (deliberative reasoning)
  valence         -> hcp/gambling_reward_vs_punish.nii.gz  (signed valence axis)

Everything else stays on Neurosynth term-association maps so that within-emotion
Ekman granularity (anger / fear / disgust / sadness / happiness) is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import image as nl_image


_EXP = Path(__file__).resolve().parents[1]
BASE = _EXP / "data" / "cognitive_stimuli" / "rsa"
BRAIN_DIR = _EXP / "data" / "brain_maps"
OUT_DIR = _EXP / "results" / "cognitive_rsa"
OUT_PATH = OUT_DIR / "brain_rdm_v2.npz"

HCP_OVERRIDES = {
    "theory_of_mind": "hcp/social_tom_vs_random.nii.gz",
    "judgment":       "hcp/relational_rel_vs_match.nii.gz",
    "valence":        "hcp/gambling_reward_vs_punish.nii.gz",
}


def correlate_pearson(X):
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASE / "rsa_conditions_manifest.json") as f:
        manifest = json.load(f)
    cond_brain = {c: info["brain_map"] for c, info in manifest["conditions"].items()}
    for c, hcp_path in HCP_OVERRIDES.items():
        if c in cond_brain:
            print(f"  override {c}: {cond_brain[c]} -> {hcp_path}")
            cond_brain[c] = hcp_path
    conditions = sorted(cond_brain.keys())

    anchor_path = BRAIN_DIR / cond_brain[conditions[0]]
    anchor_img = nib.load(str(anchor_path))
    anchor_shape = anchor_img.shape
    print(f"Anchor: {anchor_path.name}  shape={anchor_shape}")

    print(f"Resampling {len(conditions)} maps onto anchor grid...")
    arrays, sources = [], []
    for cond in conditions:
        path = BRAIN_DIR / cond_brain[cond]
        img = nib.load(str(path))
        if img.shape != anchor_shape or not np.allclose(img.affine, anchor_img.affine):
            img = nl_image.resample_to_img(img, anchor_img, interpolation="linear",
                                           force_resample=True, copy_header=True)
        arr = np.asarray(img.get_fdata(dtype=np.float32))
        arrays.append(arr)
        sources.append(cond_brain[cond])
        kind = "HCP" if cond_brain[cond].startswith("hcp/") else "NS"
        print(f"  [{kind}] {cond:>16s}: nonzero={int((arr!=0).sum()):>7d}  "
              f"max={arr.max():.2f}  min={arr.min():.2f}  src={cond_brain[cond]}")

    stack = np.stack(arrays, axis=0)
    finite_mask = np.isfinite(stack).all(axis=0)
    nonzero_count = (stack != 0).sum(axis=0)
    valid = finite_mask & (nonzero_count >= len(conditions) // 2)
    n_voxels = int(valid.sum())
    print(f"\nValid voxels: {n_voxels}")

    voxel_matrix = stack.reshape(len(conditions), -1)
    voxel_matrix = voxel_matrix[:, valid.ravel()]
    voxel_matrix = np.nan_to_num(voxel_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    corr = correlate_pearson(voxel_matrix.astype(np.float64))
    rdm = 1.0 - corr
    triu = np.triu_indices(len(conditions), k=1)
    print(f"\nBrain RDM v2:")
    print(f"  off-diag mean: {rdm[triu].mean():.4f}")
    print(f"  off-diag range: [{rdm[triu].min():.4f}, {rdm[triu].max():.4f}]")
    pair_vals = list(zip(triu[0], triu[1], rdm[triu]))
    pair_vals.sort(key=lambda x: x[2])
    print("\nMost similar pairs (top 6):")
    for i, j, v in pair_vals[:6]:
        print(f"    {conditions[i]:>16s} <-> {conditions[j]:<16s}  rdm={v:.4f}")
    print("Most dissimilar (top 6):")
    for i, j, v in pair_vals[-6:][::-1]:
        print(f"    {conditions[i]:>16s} <-> {conditions[j]:<16s}  rdm={v:.4f}")

    np.savez_compressed(
        OUT_PATH,
        rdm=rdm,
        conditions=np.array(conditions),
        sources=np.array(sources),
        n_voxels=n_voxels,
        grid_shape=np.array(anchor_shape),
        distance="1_minus_pearson",
        anchor_map=str(cond_brain[conditions[0]]),
        hcp_overrides=np.array(list(HCP_OVERRIDES.keys())),
    )
    print(f"\nSaved brain RDM v2 to {OUT_PATH}")


if __name__ == "__main__":
    main()
