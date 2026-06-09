#!/usr/bin/env python3
"""
Region-specific brain-LLM alignment: which brain areas "think like LLMs"?

For each Schaefer-400 parcel, compute a local brain RDM using only that
parcel's timeseries, then correlate with the LLM RDM. Produces a whole-brain
map of brain-LLM alignment strength.

Compare with known functional networks (Yeo 7-network parcellation):
  - Visual, Somatomotor, Dorsal Attention, Ventral Attention,
    Limbic, Frontoparietal (Control), Default Mode Network

Hypothesis: Language network and Limbic regions will show HIGH alignment.
            Default Mode Network (ToM/mentalizing) will show LOW alignment.

Output:
  results/cognitive_rsa/regional_rsa.npz
  figures/regional_rsa_map.png
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
FMRI_DIR = BASE / "data" / "narratives" / "fmri" / "afni-nosmooth"
ANN_PATH = BASE / "data" / "narratives" / "annotated_sentences.jsonl"
RES = BASE / "results" / "cognitive_rsa"
FIG = BASE / "figures"

CONDITIONS = [
    "anger", "fear", "happiness", "belief", "mentalizing",
    "intention", "theory_of_mind", "self_referential", "judgment", "moral",
]
HRF_LAG = 5.0

YEO_NETWORKS = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "Dorsal_Attention",
    "SalVentAttn": "Ventral_Attention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal",
    "Default": "Default_Mode",
}


def get_schaefer_atlas():
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    return atlas["maps"], labels


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
    return 1.0 - a @ a.T


def label_to_network(label):
    for key, name in YEO_NETWORKS.items():
        if key in label:
            return name
    return "Unknown"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Loading atlas...")
    atlas_path, atlas_labels = get_schaefer_atlas()
    atlas_labels = [l for l in atlas_labels if "Background" not in l]
    n_parcels = len(atlas_labels)
    print(f"  {n_parcels} parcels")

    parcel_networks = [label_to_network(l) for l in atlas_labels]

    story = "pieman"
    sents = load_annotations(story)
    print(f"Story: {story}, {len(sents)} sentences")

    bold_files = sorted(FMRI_DIR.glob(f"*/func/*task-{story}*MNI152*desc-clean_bold.nii.gz"))
    print(f"Found {len(bold_files)} BOLD files")

    img0 = nib.load(str(bold_files[0]))
    n_trs = img0.shape[3]
    tr = float(img0.header.get_zooms()[3])
    cond_trs = assign_trs(sents, n_trs, tr)

    valid_conds = [c for c in CONDITIONS if len(cond_trs.get(c, [])) >= 3]
    n_conds = len(valid_conds)
    print(f"Valid conditions: {n_conds}")
    triu = np.triu_indices(n_conds, k=1)

    # Load LLM RDM (Qwen 1.5B on same text)
    import glob
    llm_path = sorted(glob.glob(str(RES.parent / "narratives_llm" / "Qwen2.5-1.5B*pieman*.npz")))
    if not llm_path:
        print("No LLM NPZ found!")
        return

    llm_data = np.load(llm_path[0], allow_pickle=True)
    llm_acts = llm_data["activations"]
    llm_labels = llm_data["labels"]
    n_layers = llm_acts.shape[1]

    llm_cond_sents = defaultdict(list)
    for i, labs in enumerate(llm_labels):
        for lab in labs:
            if lab in valid_conds:
                llm_cond_sents[lab].append(i)

    # Find LLM peak layer using global brain RDM
    brain_global = np.load(RES / "narratives_brain_rdm.npz", allow_pickle=True)
    brain_global_rdm = brain_global["rdm"]
    brain_global_conds = list(brain_global["conditions"])
    g_idx = [brain_global_conds.index(c) for c in valid_conds]
    brain_g_sub = brain_global_rdm[np.ix_(g_idx, g_idx)]

    best_rho, best_L = -1, 0
    for L in range(n_layers):
        cm = np.array([llm_acts[llm_cond_sents[c], L, :].mean(axis=0)
                       for c in valid_conds], dtype=np.float64)
        cm = cm - cm.mean(axis=0, keepdims=True)
        lr = rdm_cosine(cm)
        rho, _ = spearmanr(brain_g_sub[triu], lr[triu])
        if rho > best_rho:
            best_rho, best_L = rho, L

    # Build LLM RDM at peak layer
    llm_cm = np.array([llm_acts[llm_cond_sents[c], best_L, :].mean(axis=0)
                       for c in valid_conds], dtype=np.float64)
    llm_cm = llm_cm - llm_cm.mean(axis=0, keepdims=True)
    llm_rdm = rdm_cosine(llm_cm)
    llm_vec = llm_rdm[triu]
    print(f"\nLLM RDM: {llm_rdm.shape}, peak layer L{best_L}")

    # Parcellate and compute per-region RSA
    from nilearn.maskers import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=atlas_path, standardize="zscore_sample",
                               resampling_target="data")

    all_parcel_rhos = []

    for bi, bf in enumerate(bold_files):
        sub = bf.parts[-3]
        if bi % 10 == 0:
            print(f"  [{bi+1}/{len(bold_files)}] {sub}...")

        ts = masker.fit_transform(str(bf))  # (n_trs, 400)

        parcel_rhos = np.full(n_parcels, np.nan)
        for p in range(n_parcels):
            parcel_ts = ts[:, p]  # (n_trs,)
            cond_means = []
            for c in valid_conds:
                t_idx = cond_trs[c]
                if len(t_idx) < 2:
                    cond_means.append(0.0)
                else:
                    cond_means.append(parcel_ts[t_idx].mean())
            cond_means = np.array(cond_means).reshape(-1, 1)  # (n_conds, 1)

            diffs = cond_means - cond_means.T  # (n_conds, n_conds)
            brain_local = np.abs(diffs)
            rho, _ = spearmanr(brain_local[triu], llm_vec)
            if np.isfinite(rho):
                parcel_rhos[p] = rho

        all_parcel_rhos.append(parcel_rhos)

    all_parcel_rhos = np.array(all_parcel_rhos)  # (n_subs, 400)
    mean_rhos = np.nanmean(all_parcel_rhos, axis=0)
    std_rhos = np.nanstd(all_parcel_rhos, axis=0)

    # T-test: which parcels have significant alignment?
    from scipy.stats import ttest_1samp
    t_vals = np.zeros(n_parcels)
    p_vals = np.ones(n_parcels)
    for p in range(n_parcels):
        valid = all_parcel_rhos[:, p]
        valid = valid[np.isfinite(valid)]
        if len(valid) > 5:
            t, pv = ttest_1samp(valid, 0)
            t_vals[p] = t
            p_vals[p] = pv

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    reject, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    n_sig = np.sum(reject & (mean_rhos > 0))
    print(f"\nSignificant parcels (FDR<0.05, rho>0): {n_sig}/{n_parcels}")

    # Network-level summary
    print(f"\n{'Network':<25s} {'n_parcels':>10s} {'mean_rho':>10s} {'n_sig':>8s} {'%_sig':>8s}")
    print("-" * 65)
    network_results = {}
    for net_name in sorted(set(parcel_networks)):
        net_idx = [i for i, n in enumerate(parcel_networks) if n == net_name]
        net_rhos = mean_rhos[net_idx]
        net_sig = sum(1 for i in net_idx if reject[i] and mean_rhos[i] > 0)
        mr = np.nanmean(net_rhos)
        print(f"  {net_name:<25s} {len(net_idx):>10d} {mr:>+10.4f} {net_sig:>8d} {net_sig/len(net_idx)*100:>7.1f}%")
        network_results[net_name] = {
            "n_parcels": len(net_idx),
            "mean_rho": float(mr),
            "n_significant": net_sig,
            "parcel_rhos": [float(mean_rhos[i]) for i in net_idx],
        }

    # Save
    np.savez(
        RES / "regional_rsa.npz",
        mean_rhos=mean_rhos,
        std_rhos=std_rhos,
        t_vals=t_vals,
        p_vals=p_vals,
        p_adj=p_adj,
        reject=reject,
        all_parcel_rhos=all_parcel_rhos,
        parcel_labels=np.array(atlas_labels),
        parcel_networks=np.array(parcel_networks),
        conditions=np.array(valid_conds),
        llm_rdm=llm_rdm,
    )

    with open(RES / "regional_rsa.json", "w") as f:
        json.dump(network_results, f, indent=2)

    print(f"\nSaved {RES / 'regional_rsa.npz'}")

    # === FIGURES ===
    # 1. Network-level bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    nets = sorted(network_results.keys())
    net_means = [network_results[n]["mean_rho"] for n in nets]
    colors = {"Visual": "#9467bd", "Somatomotor": "#8c564b",
              "Dorsal_Attention": "#e377c2", "Ventral_Attention": "#d62728",
              "Limbic": "#ff7f0e", "Frontoparietal": "#2ca02c",
              "Default_Mode": "#1f77b4", "Unknown": "#7f7f7f"}
    bar_colors = [colors.get(n, "#999999") for n in nets]
    ax.barh(range(len(nets)), net_means, color=bar_colors, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(nets)))
    ax.set_yticklabels([n.replace("_", " ") for n in nets], fontsize=10)
    ax.set_xlabel("Mean brain-LLM RSA (Spearman ρ)")
    ax.set_title("Brain-LLM alignment by functional network")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # 2. Top and bottom parcels
    ax = axes[1]
    sorted_idx = np.argsort(mean_rhos)
    top10 = sorted_idx[-10:][::-1]
    bot10 = sorted_idx[:10]

    y_labels = []
    y_vals = []
    y_colors = []
    for i in top10:
        short = atlas_labels[i].split("_")
        name = "_".join(short[2:5]) if len(short) > 4 else atlas_labels[i]
        y_labels.append(f"{name} ({parcel_networks[i][:3]})")
        y_vals.append(mean_rhos[i])
        y_colors.append(colors.get(parcel_networks[i], "#999999"))
    y_labels.append("---")
    y_vals.append(0)
    y_colors.append("white")
    for i in bot10[::-1]:
        short = atlas_labels[i].split("_")
        name = "_".join(short[2:5]) if len(short) > 4 else atlas_labels[i]
        y_labels.append(f"{name} ({parcel_networks[i][:3]})")
        y_vals.append(mean_rhos[i])
        y_colors.append(colors.get(parcel_networks[i], "#999999"))

    ax.barh(range(len(y_labels)), y_vals, color=y_colors, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Mean brain-LLM RSA (Spearman ρ)")
    ax.set_title("Top 10 & Bottom 10 parcels")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG / "regional_rsa_networks.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
