#!/usr/bin/env python3
"""
Build stimulus-locked brain RDM from Narratives fMRI data.

For each story:
  1. Load fMRI for all subjects who listened to it
  2. Map each annotated sentence to its TR (accounting for HRF delay)
  3. Extract brain activation pattern per sentence per subject
  4. Group by cognitive condition, average
  5. Accumulate across stories and subjects

Output: a stimulus-locked brain RDM from real fMRI, replacing the
Neurosynth meta-analytic version.
"""
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
SENT_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
OUT_DIR = BASE / "results" / "narratives_brain_rdm"

HRF_DELAY = 5.0
MIN_SENTENCES_PER_COND = 5
MIN_SUBJECTS = 15

TARGET_STORIES = [
    "pieman", "tunnel", "notthefallintact", "black", "prettymouth",
    "forgot", "sherlock", "merlin", "bronx", "21styear",
    "slumlordreach", "lucy",
]


def get_story_subjects(story):
    subjects = []
    for sub in sorted(os.listdir(FMRI_DIR)):
        func_dir = FMRI_DIR / sub / "func"
        if not func_dir.is_dir():
            continue
        # prefer non-run files, else run-1
        candidates = []
        for f in sorted(os.listdir(func_dir)):
            if f"task-{story}" in f and f.endswith(".nii.gz"):
                candidates.append(f)
        if candidates:
            # pick the one without "run" or the first one
            pick = candidates[0]
            for c in candidates:
                if "run" not in c:
                    pick = c
                    break
            subjects.append((sub, func_dir / pick))
    return subjects


def load_and_extract(nii_path, sentences, tr, hrf_delay=5.0):
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    n_trs = data.shape[-1]

    patterns = []
    for sent in sentences:
        midpoint = (sent["onset"] + sent["offset"]) / 2.0
        target_time = midpoint + hrf_delay
        tr_idx = int(target_time / tr)

        if tr_idx < 0 or tr_idx >= n_trs:
            patterns.append(None)
            continue

        vol = data[:, :, :, tr_idx].flatten()
        patterns.append(vol)

    return patterns


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_sents = [json.loads(l) for l in open(SENT_PATH)]
    story_sents = defaultdict(list)
    for s in all_sents:
        story_sents[s["story"]].append(s)

    global_cond_counts = defaultdict(int)
    for s in all_sents:
        for c in s.get("conditions", {}):
            global_cond_counts[c] += 1
    valid_conditions = sorted(c for c, n in global_cond_counts.items()
                              if n >= MIN_SENTENCES_PER_COND)
    print(f"Valid conditions (>={MIN_SENTENCES_PER_COND} sents): {len(valid_conditions)}")
    for c in valid_conditions:
        print(f"  {c:>20s}: {global_cond_counts[c]} sentences")

    cond_sum = {c: None for c in valid_conditions}
    cond_count = {c: 0 for c in valid_conditions}
    n_voxels = None
    total_subjects_used = 0

    for story in TARGET_STORIES:
        sents = story_sents.get(story, [])
        if not sents:
            continue

        subjects = get_story_subjects(story)
        if len(subjects) < MIN_SUBJECTS:
            print(f"\n[skip] {story}: only {len(subjects)} subjects (need {MIN_SUBJECTS})")
            continue

        first_img = nib.load(str(subjects[0][1]))
        tr = first_img.header.get_zooms()[-1]
        shape = first_img.shape

        print(f"\n=== {story}: {len(sents)} sents, {len(subjects)} subs, "
              f"TR={tr:.2f}s ===")

        valid_sents = []
        for s in sents:
            conds = [c for c in s.get("conditions", {}) if c in valid_conditions]
            if conds:
                valid_sents.append((s, conds))

        if not valid_sents:
            continue

        n_subs_ok = 0
        for sub_name, nii_path in subjects:
            try:
                patterns = load_and_extract(
                    nii_path, [vs[0] for vs in valid_sents], tr, HRF_DELAY)
            except Exception as e:
                continue

            if n_voxels is None and patterns and patterns[0] is not None:
                n_voxels = len(patterns[0])
                for c in valid_conditions:
                    cond_sum[c] = np.zeros(n_voxels, dtype=np.float64)

            for (sent, conds), pat in zip(valid_sents, patterns):
                if pat is None or n_voxels is None or len(pat) != n_voxels:
                    continue
                pat64 = pat.astype(np.float64)
                for c in conds:
                    cond_sum[c] += pat64
                    cond_count[c] += 1

            n_subs_ok += 1

        total_subjects_used += n_subs_ok
        print(f"  {n_subs_ok} subjects OK. Cond counts: "
              + ", ".join(f"{c}={cond_count[c]}" for c in valid_conditions if cond_count[c] > 0))

    # Build RDM
    print(f"\n{'='*60}")
    print(f"Building stimulus-locked brain RDM")
    print(f"{'='*60}")
    print(f"Total subject-stories: {total_subjects_used}")

    final_conditions = []
    centroids = []
    for c in valid_conditions:
        if cond_count[c] >= MIN_SENTENCES_PER_COND and cond_sum[c] is not None:
            centroid = cond_sum[c] / cond_count[c]
            centroids.append(centroid)
            final_conditions.append(c)
            print(f"  {c:>20s}: {cond_count[c]} obs")

    n_cond = len(final_conditions)
    centroids = np.stack(centroids)

    X = centroids - centroids.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    corr = Xn @ Xn.T
    rdm = 1.0 - corr

    triu = np.triu_indices(n_cond, k=1)
    print(f"\nRDM: {n_cond}x{n_cond}")
    print(f"  Off-diag: mean={rdm[triu].mean():.4f}, "
          f"range=[{rdm[triu].min():.4f}, {rdm[triu].max():.4f}]")

    # Compare with Neurosynth
    ns_path = BASE / "results" / "cognitive_rsa" / "brain_rdm.npz"
    if ns_path.exists():
        ns_data = np.load(ns_path, allow_pickle=True)
        ns_rdm = ns_data["rdm"]
        ns_conds = list(ns_data["conditions"])
        overlap = [c for c in final_conditions if c in ns_conds]
        if len(overlap) >= 5:
            ov_stim = [final_conditions.index(c) for c in overlap]
            ov_ns = [ns_conds.index(c) for c in overlap]
            sub_stim = rdm[np.ix_(ov_stim, ov_stim)]
            sub_ns = ns_rdm[np.ix_(ov_ns, ov_ns)]
            ov_triu = np.triu_indices(len(overlap), k=1)
            rho, p = spearmanr(sub_stim[ov_triu], sub_ns[ov_triu])
            print(f"\n  Stimulus-locked vs Neurosynth ({len(overlap)} overlap):")
            print(f"    Spearman rho = {rho:+.4f}, p = {p:.4f}")

    np.savez_compressed(
        OUT_DIR / "narratives_brain_rdm.npz",
        rdm=rdm,
        conditions=np.array(final_conditions),
        n_voxels=n_voxels,
        cond_counts=np.array([cond_count[c] for c in final_conditions]),
        total_subjects=total_subjects_used,
        hrf_delay=HRF_DELAY,
        distance="1_minus_pearson",
    )
    print(f"\nSaved: {OUT_DIR / 'narratives_brain_rdm.npz'}")


if __name__ == "__main__":
    main()
