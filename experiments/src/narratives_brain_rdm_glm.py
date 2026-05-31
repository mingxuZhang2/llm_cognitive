#!/usr/bin/env python3
"""
Build stimulus-locked brain RDM using GLM approach (fixes multi-label contamination).

Instead of averaging brain volumes by condition (which mixes co-occurring conditions),
we fit a GLM per subject per story:

  brain_activity(t) = sum_c beta_c * X_c(t) + noise

where X_c(t) is the design matrix column for condition c, convolved with the HRF.
The beta maps are the "pure" condition-specific brain patterns.

Then build RDM from condition-average beta maps.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr, gamma as gamma_dist
from numpy.linalg import lstsq

BASE = Path(__file__).resolve().parents[1]
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
SENT_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
OUT_DIR = BASE / "results" / "narratives_brain_rdm"

MIN_SUBJECTS = 15
MAX_SUBJECTS = 30  # group average saturates; cap to limit IO
TR = 1.5
VARIANCE_THRESHOLD = 10.0  # keep voxels with temporal variance above this

TARGET_STORIES = [
    "pieman", "tunnel", "notthefallintact", "black", "prettymouth",
    "forgot", "sherlock", "merlin", "bronx", "21styear",
    "slumlordreach", "lucy",
]


def canonical_hrf(tr, n_trs, peak=6.0, undershoot=16.0, ratio=6.0):
    """Canonical double-gamma HRF sampled at TR intervals."""
    t = np.arange(1, n_trs) * tr  # skip t=0 to avoid gamma(0)=0
    t = np.concatenate([[0.001], t])
    hrf = (gamma_dist.pdf(t, peak, scale=1.0) -
           gamma_dist.pdf(t, undershoot, scale=1.0) / ratio)
    mx = hrf.max()
    if mx > 0:
        hrf = hrf / mx
    return hrf


def build_design_matrix(sentences, conditions, n_trs, tr):
    """Build design matrix: one column per condition, convolved with HRF."""
    n_conds = len(conditions)
    cond_idx = {c: i for i, c in enumerate(conditions)}

    # Raw onset matrix (before HRF convolution)
    raw = np.zeros((n_trs, n_conds), dtype=np.float64)

    for sent in sentences:
        onset_tr = int(sent["onset"] / tr)
        offset_tr = min(int(sent["offset"] / tr) + 1, n_trs)
        for c in sent.get("conditions", {}):
            if c in cond_idx:
                raw[onset_tr:offset_tr, cond_idx[c]] = 1.0

    # Convolve each column with HRF
    hrf = canonical_hrf(tr, min(30, n_trs))  # 30 TRs = 45s HRF window
    design = np.zeros_like(raw)
    for ci in range(n_conds):
        design[:, ci] = np.convolve(raw[:, ci], hrf)[:n_trs]

    # Add intercept + linear drift
    intercept = np.ones((n_trs, 1))
    drift = np.linspace(-1, 1, n_trs).reshape(-1, 1)
    design = np.column_stack([design, intercept, drift])

    return design, n_conds


def get_story_subjects(story):
    subjects = []
    for sub in sorted(os.listdir(FMRI_DIR)):
        func_dir = FMRI_DIR / sub / "func"
        if not func_dir.is_dir():
            continue
        candidates = []
        for f in sorted(os.listdir(func_dir)):
            if f"task-{story}" in f and f.endswith(".nii.gz"):
                candidates.append(f)
        if candidates:
            pick = candidates[0]
            for c in candidates:
                if "run" not in c:
                    pick = c
                    break
            subjects.append((sub, func_dir / pick))
    return subjects


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_sents = [json.loads(l) for l in open(SENT_PATH)]
    story_sents = defaultdict(list)
    for s in all_sents:
        story_sents[s["story"]].append(s)

    # Determine valid conditions
    global_cond_counts = defaultdict(int)
    for s in all_sents:
        for c in s.get("conditions", {}):
            global_cond_counts[c] += 1
    valid_conditions = sorted(c for c, n in global_cond_counts.items() if n >= 5)
    print(f"Conditions: {valid_conditions}")

    # Per-story RDMs (condition×condition), averaged across stories.
    # This sidesteps cross-story voxel-dimension mismatch.
    cond_pair_sum = defaultdict(float)   # (c_i, c_j) -> summed RDM value
    cond_pair_count = defaultdict(int)   # (c_i, c_j) -> number of stories contributing
    RIDGE = 1.0  # regularization to handle collinear design columns

    for story in TARGET_STORIES:
        sents = story_sents.get(story, [])
        if not sents:
            continue

        subjects = get_story_subjects(story)
        if len(subjects) < MIN_SUBJECTS:
            print(f"[skip] {story}: {len(subjects)} subjects", flush=True)
            continue
        subjects = subjects[:MAX_SUBJECTS]

        # Check which conditions appear in this story
        story_conds = set()
        for s in sents:
            story_conds.update(c for c in s.get("conditions", {}) if c in valid_conditions)
        story_conds = sorted(story_conds)
        if len(story_conds) < 3:
            print(f"[skip] {story}: only {len(story_conds)} conditions")
            continue

        print(f"\n=== {story}: {len(sents)} sents, {len(subjects)} subs, "
              f"{len(story_conds)} conds ===")

        n_subs_ok = 0
        brain_mask = None
        # Accumulate beta maps for THIS story's voxel space
        story_beta_sum = {c: None for c in story_conds}
        story_beta_count = {c: 0 for c in story_conds}

        for sub_name, nii_path in subjects:
            try:
                img = nib.load(str(nii_path))
                data = img.get_fdata(dtype=np.float32)
                n_trs = data.shape[-1]
                tr_actual = img.header.get_zooms()[-1]
                flat = data.reshape(-1, n_trs)

                if brain_mask is None:
                    voxel_var = flat.var(axis=1)
                    brain_mask = voxel_var > VARIANCE_THRESHOLD
                    nvox = int(brain_mask.sum())
                    print(f"  Brain mask: {nvox} voxels", flush=True)
                    for c in story_conds:
                        story_beta_sum[c] = np.zeros(nvox, dtype=np.float64)

                Y = flat[brain_mask].T

                design, n_conds = build_design_matrix(
                    sents, story_conds, n_trs, tr_actual)

                # Ridge-regularized GLM to handle collinear conditions
                XtX = design.T @ design
                XtX += RIDGE * np.eye(XtX.shape[0])
                XtY = design.T @ Y
                betas = np.linalg.solve(XtX, XtY)

                for ci, cond in enumerate(story_conds):
                    story_beta_sum[cond] += betas[ci].astype(np.float64)
                    story_beta_count[cond] += 1

                n_subs_ok += 1
            except Exception as e:
                print(f"    [err] {sub_name}: {e}", flush=True)
                continue

        # Build this story's condition centroids and partial RDM
        story_final = [c for c in story_conds if story_beta_count[c] > 0]
        if len(story_final) < 3:
            print(f"  {story}: too few conditions with data, skipping RDM")
            continue
        cents = np.stack([story_beta_sum[c] / story_beta_count[c] for c in story_final])
        Xc = cents - cents.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(Xc, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = Xc / norms
        story_rdm = 1.0 - (Xn @ Xn.T)

        # Accumulate into global pair sums (weighted by n_subjects)
        for i, ci in enumerate(story_final):
            for j, cj in enumerate(story_final):
                if i < j:
                    key = (ci, cj)
                    cond_pair_sum[key] += story_rdm[i, j] * n_subs_ok
                    cond_pair_count[key] += n_subs_ok

        print(f"  {story}: {n_subs_ok} subjects, {len(story_final)} conds, "
              f"RDM mean={story_rdm[np.triu_indices(len(story_final),k=1)].mean():.3f}",
              flush=True)

    # Build global RDM from cross-story pair averages
    print(f"\n{'='*60}")
    print("Building GLM-based stimulus-locked brain RDM (cross-story pair average)")
    print(f"{'='*60}")

    # Conditions that appear in enough pairs
    cond_in_pairs = defaultdict(int)
    for (ci, cj), cnt in cond_pair_count.items():
        cond_in_pairs[ci] += 1
        cond_in_pairs[cj] += 1
    final_conditions = sorted(c for c in valid_conditions if cond_in_pairs.get(c, 0) >= 3)
    n_cond = len(final_conditions)
    print(f"Final conditions ({n_cond}): {final_conditions}")

    # Assemble RDM
    rdm = np.zeros((n_cond, n_cond), dtype=np.float64)
    missing_pairs = 0
    for i, ci in enumerate(final_conditions):
        for j, cj in enumerate(final_conditions):
            if i >= j:
                continue
            key = (ci, cj) if (ci, cj) in cond_pair_sum else (cj, ci)
            if cond_pair_count.get(key, 0) > 0:
                val = cond_pair_sum[key] / cond_pair_count[key]
                rdm[i, j] = rdm[j, i] = val
            else:
                missing_pairs += 1
                rdm[i, j] = rdm[j, i] = np.nan
    print(f"Missing pairs: {missing_pairs}/{n_cond*(n_cond-1)//2}")

    triu = np.triu_indices(n_cond, k=1)
    valid_vals = rdm[triu][~np.isnan(rdm[triu])]
    print(f"\nGLM brain RDM: {n_cond}x{n_cond}")
    print(f"  Off-diag: mean={valid_vals.mean():.4f}, "
          f"range=[{valid_vals.min():.4f}, {valid_vals.max():.4f}]")

    # Compare with Neurosynth (only non-NaN pairs)
    ns_path = BASE / "results" / "cognitive_rsa" / "brain_rdm.npz"
    if ns_path.exists():
        ns_data = np.load(ns_path, allow_pickle=True)
        ns_rdm = ns_data["rdm"]
        ns_conds = list(ns_data["conditions"])
        overlap = [c for c in final_conditions if c in ns_conds]
        if len(overlap) >= 5:
            ov_s = [final_conditions.index(c) for c in overlap]
            ov_n = [ns_conds.index(c) for c in overlap]
            sub_s = rdm[np.ix_(ov_s, ov_s)]
            sub_n = ns_rdm[np.ix_(ov_n, ov_n)]
            ov_triu = np.triu_indices(len(overlap), k=1)
            sv = sub_s[ov_triu]
            nv = sub_n[ov_triu]
            mask = ~np.isnan(sv)
            rho, p = spearmanr(sv[mask], nv[mask])
            print(f"\n  GLM stimulus-locked vs Neurosynth ({len(overlap)} overlap, "
                  f"{mask.sum()} valid pairs):")
            print(f"    Spearman rho = {rho:+.4f}, p = {p:.4f}")

    # Save
    np.savez_compressed(
        OUT_DIR / "narratives_brain_rdm_glm.npz",
        rdm=rdm,
        conditions=np.array(final_conditions),
        distance="1_minus_pearson",
        method="GLM_HRF_per_story_RDM_average_ridge",
    )
    print(f"\nSaved: {OUT_DIR / 'narratives_brain_rdm_glm.npz'}")


if __name__ == "__main__":
    main()
