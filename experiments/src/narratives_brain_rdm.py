#!/usr/bin/env python3
"""
Build brain RDM from Narratives fMRI data using cognitive condition annotations.

Pipeline:
1. Load denoised BOLD (afni-nosmooth, MNI space)
2. Parcellate with Schaefer 400-parcel atlas
3. Align with sentence timestamps (hemodynamic lag = 5s)
4. Average parcellated timeseries by cognitive condition
5. Build brain RDM (1-cosine distance between condition activation patterns)
6. Compute noise ceiling via split-half subjects

Output:
  results/cognitive_rsa/narratives_brain_rdm.npz
  results/cognitive_rsa/narratives_rsa.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib

BASE = Path("/hpc2hdd/home/mzhang630/data/nature/experiments")
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
ANN_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
RES = BASE / "results" / "cognitive_rsa"

CONDITIONS = [
    "anger", "fear", "sadness", "happiness",
    "belief", "mentalizing", "intention", "theory_of_mind",
    "empathy", "self_referential", "judgment", "moral",
]

HRF_LAG = 5.0  # hemodynamic delay in seconds


def get_schaefer_atlas():
    """Download Schaefer 400-parcel atlas in MNI space."""
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    return atlas["maps"], atlas["labels"]


def parcellate_bold(bold_path: str, atlas_path: str) -> np.ndarray:
    """Extract parcel-averaged timeseries from BOLD."""
    from nilearn.maskers import NiftiLabelsMasker
    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize="zscore_sample",
        resampling_target="data",
    )
    ts = masker.fit_transform(bold_path)  # (n_TRs, n_parcels)
    return ts


def load_annotations(story: str) -> list[dict]:
    """Load annotated sentences for a story."""
    sents = []
    with open(ANN_PATH) as f:
        for line in f:
            s = json.loads(line)
            if s["story"] == story:
                sents.append(s)
    return sents


def assign_trs_to_conditions(sents: list[dict], n_trs: int, tr: float) -> dict[str, list[int]]:
    """Map TRs to conditions using sentence timing + HRF lag."""
    cond_trs = defaultdict(set)

    for s in sents:
        labels = s.get("llm_labels", [])
        if not labels:
            continue
        onset_hrf = s["onset"] + HRF_LAG
        offset_hrf = s["offset"] + HRF_LAG
        tr_start = int(onset_hrf / tr)
        tr_end = int(offset_hrf / tr) + 1
        tr_start = max(0, min(tr_start, n_trs - 1))
        tr_end = max(0, min(tr_end, n_trs))

        for label in labels:
            if label in CONDITIONS:
                for t in range(tr_start, tr_end):
                    cond_trs[label].add(t)

    return {c: sorted(trs) for c, trs in cond_trs.items()}


def build_condition_patterns(ts: np.ndarray, cond_trs: dict[str, list[int]]) -> tuple[np.ndarray, list[str]]:
    """Average parcellated timeseries by condition."""
    used_conds = []
    patterns = []
    for cond in CONDITIONS:
        trs = cond_trs.get(cond, [])
        if len(trs) < 3:
            continue
        pattern = ts[trs].mean(axis=0)
        patterns.append(pattern)
        used_conds.append(cond)
    return np.array(patterns), used_conds


def rdm_cosine(patterns: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(patterns, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = patterns / norms
    return 1.0 - normed @ normed.T


def main():
    from scipy.stats import spearmanr

    print("Loading Schaefer 400 atlas...")
    atlas_path, atlas_labels = get_schaefer_atlas()
    print(f"  Atlas: {atlas_path}")

    story = "pieman"
    sents = load_annotations(story)
    print(f"\nStory: {story}, {len(sents)} annotated sentences")

    bold_files = sorted(FMRI_DIR.glob(f"*/func/*task-{story}*MNI152*desc-clean_bold.nii.gz"))
    print(f"Found {len(bold_files)} subjects")

    if not bold_files:
        print("No BOLD files found!")
        return

    img0 = nib.load(str(bold_files[0]))
    n_trs = img0.shape[3]
    tr = img0.header.get_zooms()[3]
    print(f"  TRs: {n_trs}, TR: {tr:.2f}s")

    cond_trs = assign_trs_to_conditions(sents, n_trs, tr)
    print(f"\n  TR assignment (HRF lag={HRF_LAG}s):")
    for cond in CONDITIONS:
        trs = cond_trs.get(cond, [])
        print(f"    {cond:<20s} {len(trs):>4d} TRs")

    # Parcellate each subject
    all_subject_patterns = []
    subject_ids = []
    for bi, bf in enumerate(bold_files):
        sub = bf.parts[-3]
        print(f"\n  [{bi+1}/{len(bold_files)}] Parcellating {sub}...", end="", flush=True)
        try:
            ts = parcellate_bold(str(bf), atlas_path)
            patterns, used_conds = build_condition_patterns(ts, cond_trs)
            if len(used_conds) >= 5:
                all_subject_patterns.append(patterns)
                subject_ids.append(sub)
                print(f" {ts.shape} → {patterns.shape[0]} conditions OK")
            else:
                print(f" only {len(used_conds)} conditions, skip")
        except Exception as e:
            print(f" ERROR: {e}")

    if len(all_subject_patterns) < 2:
        print("Too few subjects!")
        return

    n_subs = len(all_subject_patterns)
    print(f"\n=== {n_subs} subjects, {len(used_conds)} conditions ===")

    # Group-average patterns → brain RDM
    group_patterns = np.mean(all_subject_patterns, axis=0)
    brain_rdm = rdm_cosine(group_patterns)
    print(f"\nBrain RDM: {brain_rdm.shape}")

    # Noise ceiling via split-half subjects
    rng = np.random.default_rng(42)
    ceil_rhos = []
    for _ in range(50):
        perm = rng.permutation(n_subs)
        half = n_subs // 2
        pat_a = np.mean([all_subject_patterns[i] for i in perm[:half]], axis=0)
        pat_b = np.mean([all_subject_patterns[i] for i in perm[half:2*half]], axis=0)
        rdm_a = rdm_cosine(pat_a)
        rdm_b = rdm_cosine(pat_b)
        triu = np.triu_indices(rdm_a.shape[0], k=1)
        rho, _ = spearmanr(rdm_a[triu], rdm_b[triu])
        if np.isfinite(rho):
            ceil_rhos.append(rho)
    ceiling = float(np.mean(ceil_rhos)) if ceil_rhos else float("nan")
    print(f"Brain noise ceiling (split-half): {ceiling:.4f}")

    # Compare with LLM RDM (if exists)
    llm_rdm_path = RES / "brain_rdm.npz"
    if llm_rdm_path.exists():
        llm_data = np.load(llm_rdm_path, allow_pickle=True)
        llm_brain_conds = list(llm_data["conditions"])

        shared = [c for c in used_conds if c in llm_brain_conds]
        print(f"\nShared conditions with Neurosynth brain RDM: {shared}")

    # Save
    np.savez(
        RES / "narratives_brain_rdm.npz",
        rdm=brain_rdm,
        conditions=np.array(used_conds),
        group_patterns=group_patterns,
        n_subjects=n_subs,
        subject_ids=np.array(subject_ids),
        ceiling=ceiling,
        story=story,
    )
    print(f"\nSaved {RES / 'narratives_brain_rdm.npz'}")

    # Print RDM
    n = len(used_conds)
    print(f"\nBrain RDM ({n}×{n}):")
    print("         " + " ".join(f"{c[:7]:>8s}" for c in used_conds))
    for i in range(n):
        row = " ".join(f"{brain_rdm[i,j]:>8.3f}" for j in range(n))
        print(f"  {used_conds[i]:<8s} {row}")


if __name__ == "__main__":
    main()
