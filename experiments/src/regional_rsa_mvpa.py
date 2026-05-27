#!/usr/bin/env python3
"""
Region-specific brain-LLM alignment using multi-voxel patterns.

For each Schaefer-400 parcel: extract VOXEL-LEVEL timeseries within that
parcel → average by condition → compute multi-voxel pattern RDM →
correlate with LLM RDM. This captures spatial pattern information, not
just mean activation.

Output:
  results/cognitive_rsa/regional_rsa_mvpa.npz
  figures/regional_rsa_mvpa.png
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

BASE = Path("/hpc2hdd/home/mzhang630/data/nature/experiments")
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
ANN_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
RES = BASE / "results" / "cognitive_rsa"
FIG = BASE / "figures"

CONDITIONS = [
    "anger", "fear", "happiness", "belief", "mentalizing",
    "intention", "theory_of_mind", "self_referential", "judgment", "moral",
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


def load_annotations(story):
    sents = []
    with open(ANN_PATH) as f:
        for line in f:
            s = json.loads(line)
            if s["story"] == story:
                sents.append(s)
    return sents


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
    sim = a @ a.T
    np.clip(sim, -1, 1, out=sim)
    return 1.0 - sim


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.image import resample_to_img

    print("Loading atlas...")
    atlas_info = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_img = nib.load(atlas_info["maps"])
    atlas_labels = [l.decode() if isinstance(l, bytes) else l for l in atlas_info["labels"]]
    atlas_labels = [l for l in atlas_labels if "Background" not in l]
    n_parcels = len(atlas_labels)
    parcel_networks = [label_to_network(l) for l in atlas_labels]
    print(f"  {n_parcels} parcels")

    story = "pieman"
    sents = load_annotations(story)
    bold_files = sorted(FMRI_DIR.glob(f"*/func/*task-{story}*MNI152*desc-clean_bold.nii.gz"))
    print(f"Found {len(bold_files)} BOLD files")

    img0 = nib.load(str(bold_files[0]))
    n_trs = img0.shape[3]
    tr = float(img0.header.get_zooms()[3])
    cond_trs = assign_trs(sents, n_trs, tr)
    valid_conds = [c for c in CONDITIONS if len(cond_trs.get(c, [])) >= 3]
    n_conds = len(valid_conds)
    triu = np.triu_indices(n_conds, k=1)
    print(f"Conditions: {n_conds}, TRs: {n_trs}")

    # Load LLM RDM
    import glob
    llm_path = sorted(glob.glob(str(RES.parent / "narratives_llm" / "Qwen2.5-1.5B*pieman*.npz")))
    llm_data = np.load(llm_path[0], allow_pickle=True)
    llm_acts = llm_data["activations"]
    llm_labels = llm_data["labels"]

    llm_cond_sents = defaultdict(list)
    for i, labs in enumerate(llm_labels):
        for lab in labs:
            if lab in valid_conds:
                llm_cond_sents[lab].append(i)

    # Use peak layer from global analysis
    brain_global = np.load(RES / "narratives_brain_rdm.npz", allow_pickle=True)
    brain_global_conds = list(brain_global["conditions"])
    g_idx = [brain_global_conds.index(c) for c in valid_conds]
    brain_g_sub = brain_global["rdm"][np.ix_(g_idx, g_idx)]

    n_layers = llm_acts.shape[1]
    best_rho, best_L = -1, 0
    for L in range(n_layers):
        cm = np.array([llm_acts[llm_cond_sents[c], L, :].mean(axis=0)
                       for c in valid_conds], dtype=np.float64)
        cm -= cm.mean(axis=0)
        rho, _ = spearmanr(brain_g_sub[triu], rdm_cosine(cm)[triu])
        if rho > best_rho:
            best_rho, best_L = rho, L

    llm_cm = np.array([llm_acts[llm_cond_sents[c], best_L, :].mean(axis=0)
                       for c in valid_conds], dtype=np.float64)
    llm_cm -= llm_cm.mean(axis=0)
    llm_rdm = rdm_cosine(llm_cm)
    llm_vec = llm_rdm[triu]
    print(f"LLM RDM ready (L{best_L})")

    # Process subjects: extract voxel-level data per parcel
    all_parcel_rhos = np.full((len(bold_files), n_parcels), np.nan)

    for bi, bf in enumerate(bold_files):
        sub = bf.parts[-3]
        if bi % 10 == 0:
            print(f"  [{bi+1}/{len(bold_files)}] {sub}...")

        bold_img = nib.load(str(bf))

        # Resample atlas to BOLD resolution
        if bi == 0:
            atlas_resampled = resample_to_img(atlas_img, nib.Nifti1Image(
                bold_img.get_fdata()[:,:,:,0], bold_img.affine),
                interpolation="nearest")
            atlas_data = atlas_resampled.get_fdata().astype(int)
            # Pre-compute voxel indices per parcel
            parcel_voxels = {}
            for p in range(1, n_parcels + 1):
                vox = np.where(atlas_data == p)
                if len(vox[0]) >= MIN_VOXELS:
                    parcel_voxels[p] = vox

        bold_data = bold_img.get_fdata()  # (x, y, z, t)

        for p_id, vox in parcel_voxels.items():
            p_idx = p_id - 1
            ts = bold_data[vox[0], vox[1], vox[2], :]  # (n_voxels, n_trs)
            ts = ts.T  # (n_trs, n_voxels)

            # Z-score per voxel
            ts_mean = ts.mean(axis=0, keepdims=True)
            ts_std = ts.std(axis=0, keepdims=True)
            ts_std[ts_std == 0] = 1
            ts = (ts - ts_mean) / ts_std

            # Build condition patterns (multi-voxel)
            patterns = []
            ok = True
            for c in valid_conds:
                t_idx = cond_trs[c]
                if len(t_idx) < 2:
                    ok = False
                    break
                patterns.append(ts[t_idx].mean(axis=0))
            if not ok:
                continue

            patterns = np.array(patterns, dtype=np.float64)  # (n_conds, n_voxels)
            patterns -= patterns.mean(axis=0, keepdims=True)

            if patterns.shape[1] < 5:
                continue

            brain_rdm_local = rdm_cosine(patterns)
            rho, _ = spearmanr(brain_rdm_local[triu], llm_vec)
            if np.isfinite(rho):
                all_parcel_rhos[bi, p_idx] = rho

        del bold_data

    mean_rhos = np.nanmean(all_parcel_rhos, axis=0)
    n_valid = np.sum(np.isfinite(all_parcel_rhos), axis=0)

    # Stats
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests

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
    n_sig_pos = np.sum(reject & (mean_rhos > 0))
    n_sig_neg = np.sum(reject & (mean_rhos < 0))
    print(f"\nSignificant: {n_sig_pos} positive, {n_sig_neg} negative (FDR<0.05)")
    print(f"Non-significant: {np.sum(~reject)}")

    # Network summary
    print(f"\n{'Network':<25s} {'n':>4s} {'mean_rho':>9s} {'std':>7s} {'sig+':>5s} {'sig-':>5s} {'ns':>5s}")
    print("-" * 65)
    network_results = {}
    for net_name in ["Default_Mode", "Frontoparietal", "Limbic", "Ventral_Attention",
                     "Dorsal_Attention", "Somatomotor", "Visual"]:
        net_idx = [i for i, n in enumerate(parcel_networks) if n == net_name]
        if not net_idx:
            continue
        nr = mean_rhos[net_idx]
        n_sp = sum(1 for i in net_idx if reject[i] and mean_rhos[i] > 0)
        n_sn = sum(1 for i in net_idx if reject[i] and mean_rhos[i] < 0)
        n_ns = len(net_idx) - n_sp - n_sn
        print(f"  {net_name:<25s} {len(net_idx):>4d} {np.nanmean(nr):>+9.4f} {np.nanstd(nr):>7.4f} {n_sp:>5d} {n_sn:>5d} {n_ns:>5d}")
        network_results[net_name] = {
            "n": len(net_idx), "mean_rho": float(np.nanmean(nr)),
            "std": float(np.nanstd(nr)), "n_sig_pos": n_sp, "n_sig_neg": n_sn,
        }

    # Save
    np.savez(RES / "regional_rsa_mvpa.npz",
             mean_rhos=mean_rhos, t_vals=t_vals, p_vals=p_vals, p_adj=p_adj,
             reject=reject, all_parcel_rhos=all_parcel_rhos,
             parcel_labels=np.array(atlas_labels),
             parcel_networks=np.array(parcel_networks),
             conditions=np.array(valid_conds))

    with open(RES / "regional_rsa_mvpa.json", "w") as f:
        json.dump(network_results, f, indent=2)
    print(f"\nSaved {RES / 'regional_rsa_mvpa.npz'}")

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
    bar_colors = [colors.get(n, "#999") for n in nets]
    y = range(len(nets))
    ax.barh(y, net_means, xerr=net_stds, color=bar_colors, edgecolor="k",
            linewidth=0.5, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in nets], fontsize=10)
    ax.set_xlabel("Brain-LLM RSA ρ (multi-voxel pattern)")
    ax.set_title("Which brain networks align with LLM?")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # Top/bottom parcels
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

    labels_plot = [f"{it[0]} ({it[2][:3]})" if it[2] else "---" for it in items]
    vals = [it[1] for it in items]
    cs = [colors.get(it[2], "white") for it in items]
    ax.barh(range(len(items)), vals, color=cs, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels_plot, fontsize=7)
    ax.set_xlabel("Brain-LLM RSA ρ")
    ax.set_title("Top 10 & Bottom 10 brain parcels")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG / "regional_rsa_mvpa.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
