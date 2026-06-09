#!/usr/bin/env python3
"""
Multi-story regional brain-LLM RSA.

Pool data across 6 stories (pieman + 5 others) to build condition-level
brain and LLM RDMs with more data per condition. Then compute per-parcel
brain-LLM alignment using multi-voxel patterns.

Key improvement over single-story: more TRs per condition, more diverse
cognitive content, better condition separation.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr, ttest_1samp
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img

BASE = Path(__file__).resolve().parents[1]
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
ANN_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
LLM_DIR = BASE / "results" / "narratives_llm"
RES = BASE / "results" / "cognitive_rsa"
FIG = BASE / "figures"

STORIES = ["pieman", "notthefallintact", "black", "prettymouth",
           "shapessocial", "shapesphysical"]

CONDITIONS = [
    "anger", "fear", "happiness", "belief", "mentalizing",
    "intention", "theory_of_mind", "empathy", "self_referential",
    "judgment", "moral",
]
HRF_LAG = 5.0
MIN_VOXELS = 10

YEO_NETWORKS = {
    "Vis": "Visual", "SomMot": "Somatomotor", "DorsAttn": "Dorsal_Attention",
    "SalVentAttn": "Ventral_Attention", "Limbic": "Limbic",
    "Cont": "Frontoparietal", "Default": "Default_Mode",
}


def label_to_network(label):
    for key, name in YEO_NETWORKS.items():
        if key in label:
            return name
    return "Unknown"


def load_all_annotations():
    by_story = defaultdict(list)
    with open(ANN_PATH) as f:
        for line in f:
            s = json.loads(line)
            if s["story"] in STORIES:
                by_story[s["story"]].append(s)
    return by_story


def assign_trs(sents, n_trs, tr):
    cond_trs = defaultdict(set)
    for s in sents:
        for label in s.get("llm_labels", []):
            if label not in CONDITIONS:
                continue
            onset = s["onset"] + HRF_LAG
            offset = s["offset"] + HRF_LAG
            for t in range(max(0, int(onset / tr)), min(n_trs, int(offset / tr) + 1)):
                cond_trs[label].add(t)
    return {c: sorted(trs) for c, trs in cond_trs.items()}


def rdm_cosine(act):
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    sim = np.clip(a @ a.T, -1, 1)
    return 1.0 - sim


def build_multistory_llm_rdm(valid_conds):
    """Build LLM RDM pooled across stories."""
    cond_acts = defaultdict(list)

    for story in STORIES:
        npz_path = LLM_DIR / f"Qwen2.5-1.5B-Instruct_{story}_narratives.npz"
        if not npz_path.exists():
            print(f"  [skip LLM] {story}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        acts = data["activations"]
        labels_arr = data["labels"]
        n_layers = acts.shape[1]
        peak_L = n_layers - 1  # use last layer

        for i, labs in enumerate(labels_arr):
            for lab in labs:
                if lab in valid_conds:
                    cond_acts[lab].append(acts[i, peak_L, :])

    print(f"  LLM sentences per condition:")
    for c in valid_conds:
        print(f"    {c:<20s} {len(cond_acts.get(c, []))} sentences")

    cm = np.array([np.mean(cond_acts[c], axis=0) for c in valid_conds], dtype=np.float64)
    cm -= cm.mean(axis=0, keepdims=True)
    return rdm_cosine(cm)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.stats.multitest import multipletests

    print("Loading atlas...")
    atlas_info = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas_info["maps"])
    atlas_labels = [l.decode() if isinstance(l, bytes) else l for l in atlas_info["labels"]]
    atlas_labels = [l for l in atlas_labels if "Background" not in l]
    n_parcels = len(atlas_labels)
    parcel_networks = [label_to_network(l) for l in atlas_labels]

    all_ann = load_all_annotations()
    print(f"Stories: {list(all_ann.keys())}")

    # Determine valid conditions across all stories
    global_cond_trs = defaultdict(int)
    for story, sents in all_ann.items():
        for s in sents:
            for lab in s.get("llm_labels", []):
                if lab in CONDITIONS:
                    global_cond_trs[lab] += 1

    valid_conds = [c for c in CONDITIONS if global_cond_trs.get(c, 0) >= 10]
    n_conds = len(valid_conds)
    triu = np.triu_indices(n_conds, k=1)
    print(f"\nValid conditions ({n_conds}): {valid_conds}")
    for c in valid_conds:
        print(f"  {c:<20s} {global_cond_trs[c]} labeled sentences")

    # Build multi-story LLM RDM
    print("\nBuilding multi-story LLM RDM...")
    llm_rdm = build_multistory_llm_rdm(valid_conds)
    llm_vec = llm_rdm[triu]
    print(f"LLM RDM: {llm_rdm.shape}")

    # Find all subjects that appear in ANY story
    all_subjects = set()
    story_bold = {}
    for story in STORIES:
        files = sorted(FMRI_DIR.glob(f"*/func/*task-{story}*MNI152*desc-clean_bold.nii.gz"))
        story_bold[story] = {f.parts[-3]: f for f in files}
        all_subjects.update(story_bold[story].keys())
    all_subjects = sorted(all_subjects)
    print(f"\nTotal unique subjects across stories: {len(all_subjects)}")

    # For each subject: compute per-story per-parcel RDM, then average RDMs
    atlas_resampled_cache = {}
    parcel_voxels_cache = {}
    all_parcel_rhos = []
    subject_ids = []

    for si, sub in enumerate(all_subjects):
        if si % 20 == 0:
            print(f"  [{si+1}/{len(all_subjects)}] {sub}...")

        parcel_rdm_accum = defaultdict(list)

        for story in STORIES:
            if sub not in story_bold[story]:
                continue

            bf = story_bold[story][sub]
            bold_img = nib.load(str(bf))
            bold_data = bold_img.get_fdata()
            n_trs_local = bold_data.shape[3]
            tr_local = float(bold_img.header.get_zooms()[3])

            sents = all_ann.get(story, [])
            cond_trs = assign_trs(sents, n_trs_local, tr_local)

            shape_key = bold_data.shape[:3]
            if shape_key not in atlas_resampled_cache:
                ref_img = nib.Nifti1Image(bold_data[:,:,:,0], bold_img.affine)
                ar = resample_to_img(atlas_img, ref_img, interpolation="nearest")
                atlas_resampled_cache[shape_key] = ar.get_fdata().astype(int)
                pv = {}
                ad = atlas_resampled_cache[shape_key]
                for p in range(1, n_parcels + 1):
                    vox = np.where(ad == p)
                    if len(vox[0]) >= MIN_VOXELS:
                        pv[p] = vox
                parcel_voxels_cache[shape_key] = pv

            pv = parcel_voxels_cache[shape_key]

            # Check which conditions have enough TRs in this story
            story_conds = [c for c in valid_conds if len(cond_trs.get(c, [])) >= 2]
            if len(story_conds) < 5:
                del bold_data
                continue

            for p_id, vox in pv.items():
                ts = bold_data[vox[0], vox[1], vox[2], :]
                ts_mean = ts.mean(axis=1, keepdims=True)
                ts_std = ts.std(axis=1, keepdims=True)
                ts_std[ts_std == 0] = 1
                ts = (ts - ts_mean) / ts_std

                patterns = []
                ok = True
                for c in valid_conds:
                    t_idx = cond_trs.get(c, [])
                    if len(t_idx) < 2:
                        patterns.append(np.zeros(ts.shape[0]))
                    else:
                        patterns.append(ts[:, t_idx].mean(axis=1))

                patterns = np.array(patterns, dtype=np.float64)
                patterns -= patterns.mean(axis=0, keepdims=True)
                if patterns.shape[1] >= 5:
                    rdm_local = rdm_cosine(patterns)
                    parcel_rdm_accum[p_id].append(rdm_local)

            del bold_data

        # Average RDMs across stories for this subject
        parcel_rhos = np.full(n_parcels, np.nan)
        for p_id, rdm_list in parcel_rdm_accum.items():
            p_idx = p_id - 1
            avg_rdm = np.mean(rdm_list, axis=0)
            rho, _ = spearmanr(avg_rdm[triu], llm_vec)
            if np.isfinite(rho):
                parcel_rhos[p_idx] = rho

        all_parcel_rhos.append(parcel_rhos)
        subject_ids.append(sub)

    all_parcel_rhos = np.array(all_parcel_rhos)
    mean_rhos = np.nanmean(all_parcel_rhos, axis=0)
    n_subs = len(subject_ids)
    print(f"\n=== {n_subs} subjects processed ===")

    # Stats
    t_vals = np.zeros(n_parcels)
    p_vals = np.ones(n_parcels)
    for p in range(n_parcels):
        valid = all_parcel_rhos[:, p]
        valid = valid[np.isfinite(valid)]
        if len(valid) > 10:
            t, pv = ttest_1samp(valid, 0)
            t_vals[p] = t
            p_vals[p] = pv

    reject, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    n_sig = np.sum(reject & (mean_rhos > 0))
    n_ns = np.sum(~reject)
    print(f"Significant (FDR<0.05): {n_sig} positive, Non-sig: {n_ns}")

    # Network summary
    print(f"\n{'Network':<25s} {'n':>4s} {'mean_rho':>9s} {'std':>7s} {'sig':>5s} {'ns':>5s}")
    print("-" * 60)
    network_results = {}
    for net_name in ["Limbic", "Default_Mode", "Ventral_Attention", "Frontoparietal",
                     "Visual", "Dorsal_Attention", "Somatomotor"]:
        net_idx = [i for i, n in enumerate(parcel_networks) if n == net_name]
        if not net_idx:
            continue
        nr = mean_rhos[net_idx]
        n_sp = sum(1 for i in net_idx if reject[i] and mean_rhos[i] > 0)
        n_ns_net = len(net_idx) - n_sp
        print(f"  {net_name:<25s} {len(net_idx):>4d} {np.nanmean(nr):>+9.4f} {np.nanstd(nr):>7.4f} {n_sp:>5d} {n_ns_net:>5d}")
        network_results[net_name] = {
            "n": len(net_idx), "mean_rho": float(np.nanmean(nr)),
            "std": float(np.nanstd(nr)), "n_sig": n_sp,
            "parcel_rhos": [float(mean_rhos[i]) for i in net_idx],
        }

    # Save
    np.savez(RES / "regional_rsa_multistory.npz",
             mean_rhos=mean_rhos, t_vals=t_vals, p_vals=p_vals, p_adj=p_adj,
             reject=reject, all_parcel_rhos=all_parcel_rhos,
             parcel_labels=np.array(atlas_labels),
             parcel_networks=np.array(parcel_networks),
             conditions=np.array(valid_conds),
             stories=np.array(STORIES),
             n_subjects=n_subs)
    with open(RES / "regional_rsa_multistory.json", "w") as f:
        json.dump(network_results, f, indent=2)
    print(f"\nSaved results")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    nets = list(network_results.keys())
    net_means = [network_results[n]["mean_rho"] for n in nets]
    net_stds = [network_results[n]["std"] for n in nets]
    colors = {"Visual": "#9467bd", "Somatomotor": "#8c564b",
              "Dorsal_Attention": "#e377c2", "Ventral_Attention": "#d62728",
              "Limbic": "#ff7f0e", "Frontoparietal": "#2ca02c",
              "Default_Mode": "#1f77b4"}
    ax.barh(range(len(nets)), net_means, xerr=net_stds,
            color=[colors.get(n, "#999") for n in nets], edgecolor="k", linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(nets)))
    ax.set_yticklabels([n.replace("_", " ") for n in nets], fontsize=10)
    ax.set_xlabel("Brain-LLM RSA ρ (multi-voxel, 6 stories pooled)")
    ax.set_title(f"Brain-LLM alignment by network (N={n_subs})")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    valid_mask = np.isfinite(mean_rhos)
    sorted_idx = np.argsort(np.where(valid_mask, mean_rhos, -999))
    top10 = sorted_idx[-10:][::-1]
    bot10 = sorted_idx[valid_mask][:10] if valid_mask.sum() > 10 else sorted_idx[:10]
    items = []
    for i in top10:
        short = "_".join(atlas_labels[i].split("_")[2:5])
        items.append((short, mean_rhos[i], parcel_networks[i]))
    items.append(("---", 0, ""))
    for i in bot10[::-1]:
        short = "_".join(atlas_labels[i].split("_")[2:5])
        items.append((short, mean_rhos[i], parcel_networks[i]))
    ax.barh(range(len(items)), [it[1] for it in items],
            color=[colors.get(it[2], "white") for it in items], edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{it[0]} ({it[2][:3]})" if it[2] else "---" for it in items], fontsize=7)
    ax.set_xlabel("Brain-LLM RSA ρ")
    ax.set_title("Top 10 & Bottom 10 parcels (6 stories)")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG / "regional_rsa_multistory.png", dpi=160)
    plt.close()
    print(f"Wrote {FIG / 'regional_rsa_multistory.png'}")


if __name__ == "__main__":
    main()
