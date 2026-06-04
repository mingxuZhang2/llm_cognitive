#!/usr/bin/env python
"""
Cross-architecture regional Narratives brain-LLM RSA (the PROPER "same story to real
brain & model" defense).

Why this exists: the saved whole-brain narratives_brain_rdm.npz is degenerate (raw, unmasked,
single-TR, un-z-scored -> condition centroids ~mutually uncorrelated; it does not even
correlate with our own Neurosynth RDM, rho=-0.001). Comparing any LLM to it gives ~0. The real
fMRI signal is recovered only with proper preprocessing at the parcel level (Schaefer-400,
z-scored, multi-voxel patterns), exactly as regional_rsa_multistory.py established for Qwen-1.5B
(all networks significant, Limbic/DMN highest). This script generalizes that proven pipeline to
all 4 architectures, addressing the 'Neurosynth is text-derived' confound with real BOLD on the
SAME story text.

Efficiency: the brain side (per-subject per-parcel condition RDMs) is identical across models,
so it is computed ONCE and cached to disk; then each model's LLM RDM is correlated against it.

LLM layer: frozen Neurosynth-headline peak layer per model (no double-dipping on Narratives);
last layer also reported for comparison with the prior pipeline.

CPU compute-node job (loads real fMRI + nilearn) -> scripts/slurm/regional_xarch.sh.
Writes results/cognitive_rsa/regional_rsa_xarch.json (+ brain cache npz).
"""
from __future__ import annotations
import json, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr, ttest_1samp
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parents[1]
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
ANN_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
LLM_DIR = BASE / "results" / "narratives_llm"
RES = BASE / "results" / "cognitive_rsa"
BRAIN_CACHE = RES / "regional_xarch_brain_cache.npz"

# stories with both LLM extractions (cross-arch) and fMRI
STORIES = ["pieman", "tunnel", "notthefallintact", "black", "prettymouth",
           "forgot", "sherlock", "merlin", "bronx", "21styear", "slumlordreach", "lucy"]
CONDITIONS = ["anger", "fear", "happiness", "sadness", "valence",
              "belief", "mentalizing", "intention", "theory_of_mind",
              "empathy", "self_referential", "judgment", "moral"]
HRF_LAG = 5.0
MIN_VOXELS = 10
MIN_LABELED = 10           # min labeled sentences for a condition to be used
FROZEN_PEAK = {"Qwen2.5-7B-Instruct": 27, "Meta-Llama-3.1-8B-Instruct": 31,
               "Mistral-7B-Instruct-v0.3": 14, "gemma-2-9b-it": 21}
MODELS = list(FROZEN_PEAK.keys())

YEO = {"Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal_Attention",
       "SalVentAttn": "Ventral_Attention", "Limbic": "Limbic",
       "Cont": "Frontoparietal", "Default": "Default_Mode"}


def label_to_network(lab):
    for k, n in YEO.items():
        if k in lab:
            return n
    return "Unknown"


def rdm_cosine(act):
    n = np.linalg.norm(act, axis=1, keepdims=True); n[n == 0] = 1.0
    a = act / n
    return 1.0 - np.clip(a @ a.T, -1, 1)


def load_annotations():
    by_story = defaultdict(list)
    with open(ANN_PATH) as f:
        for line in f:
            s = json.loads(line)
            if s["story"] in STORIES:
                by_story[s["story"]].append(s)
    return by_story


def assign_trs(sents, n_trs, tr, valid_conds):
    cond_trs = defaultdict(set)
    for s in sents:
        for lab in s.get("llm_labels", []):
            if lab not in valid_conds:
                continue
            on, off = s["onset"] + HRF_LAG, s["offset"] + HRF_LAG
            for t in range(max(0, int(on / tr)), min(n_trs, int(off / tr) + 1)):
                cond_trs[lab].add(t)
    return {c: sorted(v) for c, v in cond_trs.items()}


def determine_valid_conds(by_story):
    cnt = defaultdict(int)
    for sents in by_story.values():
        for s in sents:
            for lab in s.get("llm_labels", []):
                if lab in CONDITIONS:
                    cnt[lab] += 1
    return [c for c in CONDITIONS if cnt.get(c, 0) >= MIN_LABELED], dict(cnt)


def build_brain_cache(by_story, valid_conds):
    """Per-subject per-parcel condition RDM (avg across that subject's stories). Cached."""
    n_conds = len(valid_conds)
    triu = np.triu_indices(n_conds, k=1)

    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas["maps"])
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    labels = [l for l in labels if "Background" not in l]
    n_parcels = len(labels)
    networks = [label_to_network(l) for l in labels]

    story_bold = {}
    all_subs = set()
    for story in STORIES:
        files = sorted(FMRI_DIR.glob(f"*/func/*task-{story}*MNI152*desc-clean_bold.nii.gz"))
        story_bold[story] = {f.parts[-3]: f for f in files}
        all_subs.update(story_bold[story].keys())
    all_subs = sorted(all_subs)
    print(f"stories with fMRI: { {s: len(story_bold[s]) for s in STORIES} }", flush=True)
    print(f"total unique subjects: {len(all_subs)}", flush=True)

    atlas_cache, voxel_cache = {}, {}
    subj_parcel_rdm = []   # [n_subj, n_parcels, n_pairs]
    subj_ids = []

    for si, sub in enumerate(all_subs):
        if si % 20 == 0:
            print(f"  [{si+1}/{len(all_subs)}] {sub}", flush=True)
        accum = defaultdict(list)
        for story in STORIES:
            if sub not in story_bold[story]:
                continue
            img = nib.load(str(story_bold[story][sub]))
            data = img.get_fdata()
            n_trs = data.shape[3]; tr = float(img.header.get_zooms()[3])
            cond_trs = assign_trs(by_story.get(story, []), n_trs, tr, valid_conds)
            if sum(len(cond_trs.get(c, [])) >= 2 for c in valid_conds) < 5:
                del data; continue
            key = data.shape[:3]
            if key not in atlas_cache:
                ref = nib.Nifti1Image(data[:, :, :, 0], img.affine)
                ar = resample_to_img(atlas_img, ref, interpolation="nearest").get_fdata().astype(int)
                atlas_cache[key] = ar
                voxel_cache[key] = {p: np.where(ar == p) for p in range(1, n_parcels + 1)
                                    if np.sum(ar == p) >= MIN_VOXELS}
            pv = voxel_cache[key]
            for p_id, vox in pv.items():
                ts = data[vox[0], vox[1], vox[2], :]
                m = ts.mean(1, keepdims=True); sd = ts.std(1, keepdims=True); sd[sd == 0] = 1
                ts = (ts - m) / sd
                pats = []
                for c in valid_conds:
                    idx = cond_trs.get(c, [])
                    pats.append(ts[:, idx].mean(1) if len(idx) >= 2 else np.zeros(ts.shape[0]))
                pats = np.array(pats, np.float64)
                pats -= pats.mean(0, keepdims=True)
                accum[p_id].append(rdm_cosine(pats)[triu])
            del data
        row = np.full((n_parcels, len(triu[0])), np.nan)
        for p_id, lst in accum.items():
            row[p_id - 1] = np.mean(lst, axis=0)
        subj_parcel_rdm.append(row)
        subj_ids.append(sub)

    subj_parcel_rdm = np.array(subj_parcel_rdm)
    np.savez_compressed(BRAIN_CACHE, subj_parcel_rdm=subj_parcel_rdm,
                        parcel_labels=np.array(labels), parcel_networks=np.array(networks),
                        valid_conds=np.array(valid_conds), subj_ids=np.array(subj_ids))
    print(f"cached brain: {subj_parcel_rdm.shape} -> {BRAIN_CACHE}", flush=True)
    return subj_parcel_rdm, np.array(networks), valid_conds


def build_llm_rdm(model, layer, valid_conds, triu):
    cond_acts = defaultdict(list)
    for story in STORIES:
        fn = LLM_DIR / f"{model}_{story}_narratives.npz"
        if not fn.exists():
            continue
        d = np.load(fn, allow_pickle=True)
        acts, labels = d["activations"], d["labels"]
        L = min(layer, acts.shape[1] - 1)
        for i, labs in enumerate(labels):
            for lab in labs:
                if lab in valid_conds:
                    cond_acts[lab].append(acts[i, L, :])
    cm = np.stack([np.mean(cond_acts[c], axis=0).astype(np.float64) for c in valid_conds])
    cm -= cm.mean(0, keepdims=True)
    return rdm_cosine(cm)[triu]


def correlate(subj_parcel_rdm, networks, llm_vec):
    n_subj, n_parcels, _ = subj_parcel_rdm.shape
    parcel_rhos = np.full((n_subj, n_parcels), np.nan)
    for s in range(n_subj):
        for p in range(n_parcels):
            v = subj_parcel_rdm[s, p]
            if np.all(np.isfinite(v)) and np.std(v) > 0:
                parcel_rhos[s, p] = spearmanr(v, llm_vec)[0]
    mean_rho = np.nanmean(parcel_rhos, axis=0)
    p_vals = np.ones(n_parcels)
    for p in range(n_parcels):
        col = parcel_rhos[:, p][np.isfinite(parcel_rhos[:, p])]
        if len(col) > 10:
            p_vals[p] = ttest_1samp(col, 0)[1]
    reject, _, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    out = {}
    for net in ["Limbic", "Default_Mode", "Ventral_Attention", "Frontoparietal",
                "Visual", "Dorsal_Attention", "Somatomotor"]:
        idx = [i for i, n in enumerate(networks) if n == net]
        if not idx:
            continue
        nsig = int(sum(1 for i in idx if reject[i] and mean_rho[i] > 0))
        out[net] = {"n": len(idx), "mean_rho": float(np.nanmean(mean_rho[idx])), "n_sig": nsig}
    out["_whole_cortex_mean_rho"] = float(np.nanmean(mean_rho))
    out["_n_parcels_sig_pos"] = int(np.sum(reject & (mean_rho > 0)))
    return out


def main():
    by_story = load_annotations()
    valid_conds, counts = determine_valid_conds(by_story)
    n_conds = len(valid_conds)
    triu = np.triu_indices(n_conds, k=1)
    print(f"valid conditions ({n_conds}): {valid_conds}")
    for c in valid_conds:
        print(f"  {c:<18s} {counts[c]} labeled sentences")

    if BRAIN_CACHE.exists():
        z = np.load(BRAIN_CACHE, allow_pickle=True)
        cached_conds = [str(c) for c in z["valid_conds"]]
        if cached_conds == valid_conds:
            print(f"\n[brain cache hit] {BRAIN_CACHE.name}")
            subj_parcel_rdm = z["subj_parcel_rdm"]; networks = z["parcel_networks"]
        else:
            print("\n[brain cache stale -> rebuild]")
            subj_parcel_rdm, networks, valid_conds = build_brain_cache(by_story, valid_conds)
    else:
        print("\n[building brain cache (loads fMRI, ~1-3h)]")
        subj_parcel_rdm, networks, valid_conds = build_brain_cache(by_story, valid_conds)
    networks = [str(n) for n in networks]
    print(f"brain: {subj_parcel_rdm.shape[0]} subjects x {subj_parcel_rdm.shape[1]} parcels\n")

    results = {"valid_conditions": valid_conds, "n_subjects": int(subj_parcel_rdm.shape[0]),
               "stories": STORIES, "models": {}}
    print(f"{'model':28s} {'layer':>6s} {'cortex_mean':>11s} {'Limbic':>8s} {'DMN':>8s} {'#parcels_sig':>13s}")
    print("-" * 80)
    for model in MODELS:
        if not list(LLM_DIR.glob(f"{model}_*_narratives.npz")):
            print(f"{model:28s}  [no extractions]"); continue
        rec = {}
        for tag, layer in [("frozen_peak", FROZEN_PEAK[model]), ("last", 10_000)]:
            llm_vec = build_llm_rdm(model, layer, valid_conds, triu)
            rec[tag] = correlate(subj_parcel_rdm, networks, llm_vec)
        results["models"][model] = rec
        fp = rec["frozen_peak"]
        print(f"{model:28s} {FROZEN_PEAK[model]:>6d} {fp['_whole_cortex_mean_rho']:>+11.4f} "
              f"{fp['Limbic']['mean_rho']:>+8.4f} {fp['Default_Mode']['mean_rho']:>+8.4f} "
              f"{fp['_n_parcels_sig_pos']:>13d}")

    with open(RES / "regional_rsa_xarch.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {RES / 'regional_rsa_xarch.json'}")
    # plain-language summary
    print("\n=== SUMMARY (frozen-peak layer) ===")
    for m, r in results["models"].items():
        fp = r["frozen_peak"]
        print(f"{m:28s} cortex mean rho={fp['_whole_cortex_mean_rho']:+.3f}, "
              f"{fp['_n_parcels_sig_pos']}/400 parcels sig+, Limbic={fp['Limbic']['mean_rho']:+.3f}")


if __name__ == "__main__":
    main()
