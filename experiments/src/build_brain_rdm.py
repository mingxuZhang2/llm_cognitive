"""
Build the brain-side 14x14 representational dissimilarity matrix (RDM) for the
RSA pipeline.

Steps:
  1. Read rsa_conditions_manifest.json to learn which NIfTI corresponds to each
     condition.
  2. Resample every map onto the first (anchor) map's grid using nilearn so all
     maps share voxel space.
  3. Build a "valid voxel" mask: voxels that are finite and non-zero in at
     least 50% of the 14 maps.
  4. Flatten each map to a voxel vector restricted to that mask.
  5. Pairwise distance = 1 - Pearson correlation; save 14x14 brain RDM.

Output:
  results/cognitive_rsa/brain_rdm.npz
    rdm                 : float64 [n_cond, n_cond]
    conditions          : list[str]
    voxel_mask_nonzero  : int counts
    n_voxels            : total kept
    grid_shape          : anchor map shape
    distance            : "1_minus_pearson"
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
OUT_PATH = OUT_DIR / "brain_rdm.npz"


def correlate_pearson(X):
    """Row-wise Pearson correlation matrix. X is [n, d]."""
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
    conditions = sorted(cond_brain.keys())

    # Anchor on the first Neurosynth map's grid (all neurosynth maps are MNI152 2mm)
    anchor_path = BRAIN_DIR / cond_brain[conditions[0]]
    anchor_img = nib.load(str(anchor_path))
    anchor_shape = anchor_img.shape
    print(f"Anchor: {anchor_path.name}  shape={anchor_shape}  "
          f"affine_det={np.linalg.det(anchor_img.affine[:3,:3]):.2f}")

    # Resample every map onto anchor grid
    print(f"Resampling {len(conditions)} maps onto anchor grid...")
    arrays = []
    for cond in conditions:
        path = BRAIN_DIR / cond_brain[cond]
        img = nib.load(str(path))
        if img.shape != anchor_shape or not np.allclose(img.affine, anchor_img.affine):
            print(f"  resampling {cond}: {img.shape} -> {anchor_shape}")
            img = nl_image.resample_to_img(img, anchor_img, interpolation="linear",
                                           force_resample=True, copy_header=True)
        arr = np.asarray(img.get_fdata(dtype=np.float32))
        arrays.append(arr)
        print(f"  {cond:>16s}: nonzero={int((arr != 0).sum())}  "
              f"max={arr.max():.2f}  min={arr.min():.2f}")

    stack = np.stack(arrays, axis=0)  # [n_cond, X, Y, Z]
    finite_mask = np.isfinite(stack).all(axis=0)
    nonzero_count = (stack != 0).sum(axis=0)
    # Voxels active (non-zero) in at least half the conditions and finite everywhere
    valid = finite_mask & (nonzero_count >= len(conditions) // 2)
    n_voxels = int(valid.sum())
    print(f"\nValid voxels (finite in all + nonzero in >= {len(conditions)//2} maps): {n_voxels}")

    # Flatten and build RDM
    voxel_matrix = stack.reshape(len(conditions), -1)
    voxel_matrix = voxel_matrix[:, valid.ravel()]
    # NaNs already excluded by mask, but be defensive
    voxel_matrix = np.nan_to_num(voxel_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    corr = correlate_pearson(voxel_matrix.astype(np.float64))
    rdm = 1.0 - corr

    print(f"\nBrain RDM 14x14:")
    print(f"  diag (should be ~0): {np.diag(rdm).max():.4f}")
    print(f"  off-diag mean: {rdm[np.triu_indices(len(conditions), k=1)].mean():.4f}")
    print(f"  off-diag range: [{rdm[np.triu_indices(len(conditions), k=1)].min():.4f}, "
          f"{rdm[np.triu_indices(len(conditions), k=1)].max():.4f}]")

    print(f"\nMost similar pairs (top 5 lowest off-diag):")
    triu = np.triu_indices(len(conditions), k=1)
    pair_vals = list(zip(triu[0], triu[1], rdm[triu]))
    pair_vals.sort(key=lambda x: x[2])
    for i, j, v in pair_vals[:5]:
        print(f"    {conditions[i]:>16s} <-> {conditions[j]:<16s}  rdm={v:.4f}")
    print(f"Most dissimilar pairs (top 5 highest):")
    for i, j, v in pair_vals[-5:][::-1]:
        print(f"    {conditions[i]:>16s} <-> {conditions[j]:<16s}  rdm={v:.4f}")

    np.savez_compressed(
        OUT_PATH,
        rdm=rdm,
        conditions=np.array(conditions),
        n_voxels=n_voxels,
        grid_shape=np.array(anchor_shape),
        distance="1_minus_pearson",
        anchor_map=str(cond_brain[conditions[0]]),
    )
    print(f"\nSaved brain RDM to {OUT_PATH}")


if __name__ == "__main__":
    main()
