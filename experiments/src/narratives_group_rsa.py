#!/usr/bin/env python
"""
GROUP-LEVEL Narratives brain-LLM RSA (denoised real-fMRI defense).

The biggest cheap lever against single-subject fMRI noise: average each condition's brain
pattern across ALL subjects/stories BEFORE building the RDM (sqrt-N noise reduction), in a
common Schaefer-400 parcel space (parcellation also denoises + fixes the native-resolution
shape mismatch). This is the analog of what makes Neurosynth clean (averaging many sources).

Per (subject, story): parcellate to Schaefer-400, z-score each parcel time-series, assign TRs to
conditions (HRF lag 5 s, llm_labels), take each condition's mean 400-dim parcel pattern. Average
those per-condition vectors across all subjects -> a clean GROUP condition x 400 matrix -> RDM
(1 - Pearson). Compare to each architecture's Narratives LLM RDM (frozen Neurosynth peak layer).
Noise ceiling = split-half across subjects (Spearman-Brown corrected), so we report rho AND
rho/ceiling honestly.

CPU compute-node job (loads real fMRI + nilearn) -> scripts/slurm/narratives_group.sh.
Writes results/cognitive_rsa/narratives_group_rsa.json (+ subj x cond x parcel cache).
"""
from __future__ import annotations
import json, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.stats import rankdata
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img

BASE = Path("/hpc2hdd/home/mzhang630/data/nature/experiments")
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
ANN_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
LLM_DIR = BASE / "results" / "narratives_llm"
RES = BASE / "results" / "cognitive_rsa"
CACHE = RES / "narratives_group_patterns.npz"

STORIES = ["pieman", "tunnel", "notthefallintact", "black", "prettymouth",
           "forgot", "sherlock", "merlin", "bronx", "21styear", "slumlordreach", "lucy"]
CONDITIONS = ["anger", "fear", "happiness", "sadness", "valence",
              "belief", "mentalizing", "intention", "theory_of_mind",
              "empathy", "self_referential", "judgment", "moral"]
HRF_LAG = 5.0
MIN_VOXELS = 10
MIN_LABELED = 10
SEED = 20260525
FROZEN_PEAK = {"Qwen2.5-7B-Instruct": 27, "Meta-Llama-3.1-8B-Instruct": 31,
               "Mistral-7B-Instruct-v0.3": 14, "gemma-2-9b-it": 21}
MODELS = list(FROZEN_PEAK.keys())


def utri(M):
    iu = np.triu_indices_from(M, k=1); return M[iu]

def sp(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

def corr_rdm(M):
    """1 - Pearson correlation between rows (conditions)."""
    Mc = M - M.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Mc, axis=1, keepdims=True); n[n == 0] = 1.0
    Z = Mc / n
    return 1.0 - np.clip(Z @ Z.T, -1, 1)

def cosine_rdm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1.0
    Z = M / n
    return 1.0 - np.clip(Z @ Z.T, -1, 1)


def load_ann():
    by = defaultdict(list)
    for line in open(ANN_PATH):
        s = json.loads(line)
        if s["story"] in STORIES:
            by[s["story"]].append(s)
    return by

def valid_conditions(by):
    cnt = defaultdict(int)
    for sents in by.values():
        for s in sents:
            for lab in s.get("llm_labels", []):
                if lab in CONDITIONS:
                    cnt[lab] += 1
    return [c for c in CONDITIONS if cnt[c] >= MIN_LABELED], dict(cnt)

def assign_trs(sents, n_trs, tr, conds):
    ct = defaultdict(set)
    for s in sents:
        for lab in s.get("llm_labels", []):
            if lab in conds:
                on, off = s["onset"] + HRF_LAG, s["offset"] + HRF_LAG
                for t in range(max(0, int(on / tr)), min(n_trs, int(off / tr) + 1)):
                    ct[lab].add(t)
    return {c: sorted(v) for c, v in ct.items()}


def build_subject_patterns(by, conds):
    """subj x cond x 400 parcel patterns (NaN where a subject lacks a condition)."""
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas["maps"])
    n_parcels = 400

    story_bold = {}
    subs = set()
    for st in STORIES:
        fs = sorted(FMRI_DIR.glob(f"*/func/*task-{st}*MNI152*desc-clean_bold.nii.gz"))
        story_bold[st] = {f.parts[-3]: f for f in fs}
        subs.update(story_bold[st].keys())
    subs = sorted(subs)
    print(f"subjects: {len(subs)}; stories: { {s: len(story_bold[s]) for s in STORIES} }", flush=True)

    acache, vcache = {}, {}
    n_cond = len(conds)
    subj_patterns = np.full((len(subs), n_cond, n_parcels), np.nan)

    for si, sub in enumerate(subs):
        if si % 20 == 0:
            print(f"  [{si+1}/{len(subs)}] {sub}", flush=True)
        csum = np.zeros((n_cond, n_parcels)); ccnt = np.zeros(n_cond)
        for st in STORIES:
            if sub not in story_bold[st]:
                continue
            img = nib.load(str(story_bold[st][sub]))
            data = img.get_fdata()
            n_trs = data.shape[3]; tr = float(img.header.get_zooms()[3])
            ct = assign_trs(by.get(st, []), n_trs, tr, conds)
            key = data.shape[:3]
            if key not in acache:
                ref = nib.Nifti1Image(data[:, :, :, 0], img.affine)
                ar = resample_to_img(atlas_img, ref, interpolation="nearest").get_fdata().astype(int)
                acache[key] = ar
                vcache[key] = {p: np.where(ar == p) for p in range(1, n_parcels + 1)
                               if np.sum(ar == p) >= MIN_VOXELS}
            # parcel timeseries (n_parcels x n_trs), z-scored over time
            pts = np.full((n_parcels, n_trs), np.nan)
            for p_id, vox in vcache[key].items():
                ts = data[vox[0], vox[1], vox[2], :].mean(0)
                sd = ts.std(); pts[p_id - 1] = (ts - ts.mean()) / sd if sd > 0 else 0.0
            for ci, c in enumerate(conds):
                idx = ct.get(c, [])
                if len(idx) >= 2:
                    vec = np.nanmean(pts[:, idx], axis=1)
                    csum[ci] += np.nan_to_num(vec); ccnt[ci] += 1
            del data
        for ci in range(n_cond):
            if ccnt[ci] > 0:
                subj_patterns[si, ci] = csum[ci] / ccnt[ci]
    np.savez_compressed(CACHE, subj_patterns=subj_patterns, conds=np.array(conds),
                        subs=np.array(subs))
    print(f"cached {subj_patterns.shape} -> {CACHE}", flush=True)
    return subj_patterns, conds


def build_llm_rdm(model, layer, conds):
    cond_acts = defaultdict(list)
    for st in STORIES:
        fn = LLM_DIR / f"{model}_{st}_narratives.npz"
        if not fn.exists():
            continue
        d = np.load(fn, allow_pickle=True)
        acts, labels = d["activations"], d["labels"]
        L = min(layer, acts.shape[1] - 1)
        for i, labs in enumerate(labels):
            for lab in labs:
                if lab in conds:
                    cond_acts[lab].append(acts[i, L, :])
    cm = np.stack([np.mean(cond_acts[c], 0).astype(np.float64) for c in conds])
    cm -= cm.mean(0, keepdims=True)
    return cosine_rdm(cm)


def group_rdm(subj_patterns, idx):
    g = np.nanmean(subj_patterns[idx], axis=0)        # cond x parcel
    return corr_rdm(g)


def main():
    by = load_ann()
    conds, counts = valid_conditions(by)
    print(f"valid conditions ({len(conds)}): {conds}")

    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if [str(c) for c in z["conds"]] == conds:
            print(f"[cache hit] {CACHE.name}")
            subj_patterns = z["subj_patterns"]
        else:
            subj_patterns, conds = build_subject_patterns(by, conds)
    else:
        subj_patterns, conds = build_subject_patterns(by, conds)

    # keep subjects that have >= 6 conditions populated
    have = np.sum(np.all(np.isfinite(subj_patterns), axis=2), axis=1)
    keep = np.where(have >= 6)[0]
    print(f"subjects with >=6 conditions: {len(keep)} / {subj_patterns.shape[0]}")
    SP = subj_patterns[keep]
    n_sub = len(SP)

    brain_rdm = group_rdm(SP, np.arange(n_sub))
    triu = np.triu_indices(len(conds), 1)
    bz = brain_rdm[triu]

    # split-half noise ceiling across subjects (Spearman-Brown), averaged over K splits
    rng = np.random.default_rng(SEED)
    sh = []
    for _ in range(200):
        perm = rng.permutation(n_sub); h1, h2 = perm[:n_sub // 2], perm[n_sub // 2:]
        r = sp(group_rdm(SP, h1)[triu], group_rdm(SP, h2)[triu])
        sh.append(2 * r / (1 + r) if r > -1 else r)     # Spearman-Brown
    ceiling = float(np.mean(sh))
    print(f"\ngroup brain RDM built from {n_sub} subjects; split-half ceiling = {ceiling:.3f}\n")

    out = {"conditions": conds, "n_subjects": int(n_sub), "ceiling": ceiling, "models": {}}
    print(f"{'model':28s} {'layer':>5s} {'rho':>8s} {'rho/ceiling':>12s}")
    print("-" * 60)
    for model in MODELS:
        if not list(LLM_DIR.glob(f"{model}_*_narratives.npz")):
            print(f"{model}: [no extractions]"); continue
        rec = {}
        for tag, layer in [("frozen_peak", FROZEN_PEAK[model]), ("last", 10_000)]:
            lr = build_llm_rdm(model, layer, conds)
            rho = sp(bz, lr[triu])
            rec[tag] = {"rho": rho, "rho_over_ceiling": rho / ceiling if ceiling > 0 else None}
        out["models"][model] = rec
        fp = rec["frozen_peak"]
        print(f"{model:28s} {FROZEN_PEAK[model]:>5d} {fp['rho']:>+8.3f} {fp['rho_over_ceiling']:>11.0%}")

    vals = [m["frozen_peak"]["rho"] for m in out["models"].values()]
    if vals:
        out["summary"] = {"mean_rho": round(float(np.mean(vals)), 3),
                          "ceiling": round(ceiling, 3),
                          "mean_frac_ceiling": round(float(np.mean(vals)) / ceiling, 3) if ceiling > 0 else None}
        print(f"\nGROUP-LEVEL: mean rho = {out['summary']['mean_rho']:+.3f} "
              f"(= {out['summary']['mean_frac_ceiling']:.0%} of ceiling {ceiling:.3f})")
        print("Compare: per-subject method gave ~0.20 (45% of a 0.44 ceiling).")
    with open(RES / "narratives_group_rsa.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {RES / 'narratives_group_rsa.json'}")


if __name__ == "__main__":
    main()
