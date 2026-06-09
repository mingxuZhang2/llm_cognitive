#!/usr/bin/env python3
"""
Direct activation-space analysis: what do LLM hidden states encode about
social-emotional concepts?

Works on the actual high-dimensional activations (not RDM-derived coordinates):
1. PCA on all 712 stimuli — what are the principal components? Does PC1 = boundary?
2. PCA on 14 condition centroids — interpret the organizing dimensions
3. Linear probing — which psychological dimensions are decodable from activations?
4. Block separability — how are emotion vs social separated in activation space?
5. Within-block structure — what dimensions organize each block?

Requires per-stim NPZ files (~400MB each). Run on SLURM if login node OOMs.

Output:
  results/mechanistic/activation_geometry.json
  figures/activation_geometry.png
"""
from __future__ import annotations
import json, csv, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, LeaveOneOut
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

AFF = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
MENT = ["judgment", "belief", "intention", "mentalizing", "moral", "empathy",
        "self_referential", "theory_of_mind"]
ORDER = AFF + MENT

MODELS = ["Qwen2.5-7B-Instruct"]  # Start with one model
MSHORT = {"Qwen2.5-7B-Instruct": "Qwen"}
PEAK_LAYERS = {"Qwen2.5-7B-Instruct": 27}

C_AFF, C_MENT, C_BLUE = "#e8743b", "#19a979", "#2e5cb8"


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def main():
    print("=" * 70)
    print("ACTIVATION-SPACE GEOMETRY ANALYSIS")
    print("=" * 70)

    # Load stimuli + conditions
    stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    stimuli = [json.loads(l) for l in open(stim_path)]
    print(f"Loaded {len(stimuli)} stimuli")

    # Load VAD norms
    vad_norms = {}
    vad_path = BASE / "data" / "cognitive_stimuli" / "emotion" / "warriner_vad.csv"
    with open(vad_path) as f:
        for row in csv.DictReader(f):
            vad_norms[row["word"].lower()] = {
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "dominance": float(row["dominance"]),
            }

    # Compute per-stimulus VAD
    stim_vad = []
    for s in stimuli:
        tokens = tokenize(s["text"])
        vals = [vad_norms[t] for t in tokens if t in vad_norms]
        if vals:
            stim_vad.append({
                "valence": np.mean([v["valence"] for v in vals]),
                "arousal": np.mean([v["arousal"] for v in vals]),
                "dominance": np.mean([v["dominance"] for v in vals]),
            })
        else:
            stim_vad.append({"valence": 5.0, "arousal": 5.0, "dominance": 5.0})

    results = {}

    for model in MODELS:
        short = MSHORT[model]
        peak = PEAK_LAYERS[model]
        print(f"\n{'='*60}")
        print(f"  MODEL: {short} (peak layer {peak})")
        print(f"{'='*60}")

        # Load per-stim activations
        npz_path = RSA / f"{model}_rsa_v2_per_stim.npz"
        print(f"  Loading {npz_path.name}...")
        data = np.load(npz_path, allow_pickle=True)
        conditions = list(data["conditions"])
        pooling_names = list(data["pooling_names"])
        pool_idx = pooling_names.index("mean_all")

        # Extract peak layer, mean_all pooling: (712, hidden_dim)
        all_acts = data["per_stim_activations"]  # (3, 712, n_layers, hidden_dim)
        acts = all_acts[pool_idx, :, peak, :].astype(np.float32)  # (712, 3584)
        print(f"  Activations shape: {acts.shape}")

        n_stim, hidden_dim = acts.shape
        cond_labels = conditions  # per-stimulus condition labels

        # Block labels
        is_aff = np.array([c in AFF for c in cond_labels])
        is_soc = np.array([c in MENT for c in cond_labels])

        # ═══════════════════════════════════════════════════════
        # 1. PCA on all 712 stimuli
        # ═══════════════════════════════════════════════════════
        print("\n--- 1. PCA on all 712 stimuli ---")
        acts_centered = acts - acts.mean(axis=0, keepdims=True)
        pca_all = PCA(n_components=20)
        pcs_all = pca_all.fit_transform(acts_centered)

        print(f"  Variance explained (top 10 PCs):")
        for i in range(10):
            print(f"    PC{i+1}: {pca_all.explained_variance_ratio_[i]:.3%} "
                  f"(cum: {sum(pca_all.explained_variance_ratio_[:i+1]):.3%})")

        # Does PC1 separate blocks?
        pc1_aff = pcs_all[is_aff, 0].mean()
        pc1_soc = pcs_all[is_soc, 0].mean()
        pc1_separation = abs(pc1_aff - pc1_soc) / pcs_all[:, 0].std()
        print(f"\n  PC1: aff mean={pc1_aff:.3f}, soc mean={pc1_soc:.3f}, "
              f"separation={pc1_separation:.2f} std devs")

        # Which PCs separate blocks most?
        for i in range(10):
            aff_m = pcs_all[is_aff, i].mean()
            soc_m = pcs_all[is_soc, i].mean()
            sep = abs(aff_m - soc_m) / pcs_all[:, i].std()
            if sep > 0.5:
                print(f"  PC{i+1}: block separation = {sep:.2f} std (aff={aff_m:.2f}, soc={soc_m:.2f})")

        # ═══════════════════════════════════════════════════════
        # 2. PCA on 14 condition centroids
        # ═══════════════════════════════════════════════════════
        print("\n--- 2. PCA on 14 condition centroids ---")
        centroids = {}
        for c in ORDER:
            mask = [i for i, cc in enumerate(cond_labels) if cc == c]
            centroids[c] = acts_centered[mask].mean(axis=0)

        centroid_mat = np.stack([centroids[c] for c in ORDER])  # (14, 3584)
        pca_cond = PCA(n_components=5)
        pcs_cond = pca_cond.fit_transform(centroid_mat)

        print(f"  Variance explained:")
        for i in range(5):
            print(f"    PC{i+1}: {pca_cond.explained_variance_ratio_[i]:.1%} "
                  f"(cum: {sum(pca_cond.explained_variance_ratio_[:i+1]):.1%})")

        # Interpret condition PCs
        print(f"\n  Condition positions on PC1 and PC2:")
        for i, c in enumerate(ORDER):
            blk = "AFF" if c in AFF else "SOC"
            print(f"    {c:>20s} [{blk}]: PC1={pcs_cond[i,0]:+.3f}  PC2={pcs_cond[i,1]:+.3f}  PC3={pcs_cond[i,2]:+.3f}")

        # Block separation on each PC
        aff_pcs = pcs_cond[:len(AFF)]
        soc_pcs = pcs_cond[len(AFF):]
        print(f"\n  Block separation per PC:")
        for i in range(5):
            sep = abs(aff_pcs[:, i].mean() - soc_pcs[:, i].mean()) / pcs_cond[:, i].std()
            print(f"    PC{i+1}: {sep:.2f} std")

        # ═══════════════════════════════════════════════════════
        # 3. Linear probing: what can be decoded from activations?
        # ═══════════════════════════════════════════════════════
        print("\n--- 3. Linear probing (per-stimulus) ---")

        # Targets to probe
        probe_targets = {
            "block": np.array([1.0 if c in AFF else 0.0 for c in cond_labels]),
            "valence": np.array([stim_vad[i]["valence"] for i in range(n_stim)]),
            "arousal": np.array([stim_vad[i]["arousal"] for i in range(n_stim)]),
            "dominance": np.array([stim_vad[i]["dominance"] for i in range(n_stim)]),
        }

        # Condition-based targets (coarser, per-condition coding)
        approach_scores = {"anger": 0.8, "fear": 0.1, "disgust": 0.1,
                          "sadness": 0.2, "happiness": 0.9, "valence": 0.5,
                          "judgment": 0.5, "belief": 0.5, "intention": 0.6,
                          "mentalizing": 0.5, "moral": 0.5, "empathy": 0.5,
                          "self_referential": 0.5, "theory_of_mind": 0.5}
        probe_targets["approach"] = np.array([approach_scores[c] for c in cond_labels])

        n_agents_scores = {"anger": 1.5, "fear": 1.5, "disgust": 1.5,
                          "sadness": 1.5, "happiness": 1.5, "valence": 1,
                          "judgment": 1.5, "belief": 2, "intention": 1.5,
                          "mentalizing": 2.5, "moral": 2.5, "empathy": 2,
                          "self_referential": 1, "theory_of_mind": 2.5}
        probe_targets["n_agents"] = np.array([n_agents_scores[c] for c in cond_labels])

        # Use PCA-reduced activations for faster cross-val
        acts_pca = pcs_all[:, :50]  # top 50 PCs

        print(f"\n  {'Target':>15s} | {'R² (CV)':>8s} | {'Decodable?':>10s}")
        print(f"  {'-'*45}")

        probe_results = {}
        for target_name, target_vals in probe_targets.items():
            # Standardize target
            if target_vals.std() < 1e-8:
                continue
            y = (target_vals - target_vals.mean()) / target_vals.std()

            # Ridge regression with 5-fold CV
            ridge = Ridge(alpha=1.0)
            scores = cross_val_score(ridge, acts_pca, y, cv=5, scoring="r2")
            mean_r2 = scores.mean()
            decodable = "YES" if mean_r2 > 0.05 else "no"

            print(f"  {target_name:>15s} | {mean_r2:>8.3f} | {decodable:>10s}")
            probe_results[target_name] = {"r2_cv": float(mean_r2), "std": float(scores.std())}

        # ═══════════════════════════════════════════════════════
        # 4. Within-block PCA: what organizes each block?
        # ═══════════════════════════════════════════════════════
        print("\n--- 4. Within-block activation PCA ---")

        for block_name, block_conds, block_mask in [
            ("EMOTION", AFF, is_aff), ("SOCIAL", MENT, is_soc)
        ]:
            print(f"\n  === {block_name} block ===")
            block_acts = acts_centered[block_mask]
            block_cond_labels = [c for c, m in zip(cond_labels, block_mask) if m]

            # Condition centroids within block
            block_centroids = []
            for c in block_conds:
                mask_c = [i for i, cc in enumerate(block_cond_labels) if cc == c]
                block_centroids.append(block_acts[mask_c].mean(axis=0))
            block_centroid_mat = np.stack(block_centroids)

            pca_block = PCA(n_components=min(len(block_conds)-1, 5))
            block_pcs = pca_block.fit_transform(block_centroid_mat)

            print(f"  Variance: " + ", ".join(
                f"PC{i+1}={v:.1%}" for i, v in enumerate(pca_block.explained_variance_ratio_)))

            print(f"  Positions:")
            for i, c in enumerate(block_conds):
                pcs_str = "  ".join(f"PC{j+1}={block_pcs[i,j]:+.3f}"
                                    for j in range(min(3, block_pcs.shape[1])))
                print(f"    {c:>20s}: {pcs_str}")

            # Correlate with targets
            if block_name == "EMOTION":
                # Correlate PCs with VAD, approach, etc.
                for dim_name in ["valence", "arousal", "dominance"]:
                    vals = [np.mean([stim_vad[i][dim_name]
                                    for i, c in enumerate(cond_labels)
                                    if c == cond])
                            for cond in block_conds]
                    for pc_i in range(min(3, block_pcs.shape[1])):
                        r, p = spearmanr(block_pcs[:, pc_i], vals)
                        if abs(r) > 0.5:
                            sig = "*" if p < 0.05 else ""
                            print(f"    {dim_name} vs PC{pc_i+1}: rho={r:+.3f}{sig}")

                # Approach
                approach_vals = [approach_scores[c] for c in block_conds]
                for pc_i in range(min(3, block_pcs.shape[1])):
                    r, p = spearmanr(block_pcs[:, pc_i], approach_vals)
                    if abs(r) > 0.5:
                        sig = "*" if p < 0.05 else ""
                        print(f"    approach vs PC{pc_i+1}: rho={r:+.3f}{sig}")

            elif block_name == "SOCIAL":
                SOCIAL_DIMS_LOCAL = {
                    "n_agents": {"belief": 2, "intention": 1.5, "mentalizing": 2.5,
                                 "theory_of_mind": 2.5, "judgment": 1.5, "moral": 2.5,
                                 "empathy": 2, "self_referential": 1},
                    "recursion_depth": {"belief": 1, "intention": 0.5, "mentalizing": 2,
                                        "theory_of_mind": 2, "judgment": 0.5, "moral": 0.5,
                                        "empathy": 1, "self_referential": 0},
                    "propositional_content": {"belief": 1, "intention": 0.8, "mentalizing": 1,
                                              "theory_of_mind": 1, "judgment": 0.5, "moral": 0.5,
                                              "empathy": 0.3, "self_referential": 0.3},
                    "normative": {"belief": 0, "intention": 0, "mentalizing": 0,
                                  "theory_of_mind": 0, "judgment": 0.8, "moral": 1,
                                  "empathy": 0.2, "self_referential": 0},
                    "affective_component": {"belief": 0.1, "intention": 0.1, "mentalizing": 0.2,
                                            "theory_of_mind": 0.2, "judgment": 0.3, "moral": 0.5,
                                            "empathy": 0.9, "self_referential": 0.4},
                }
                for dim_name in SOCIAL_DIMS_LOCAL:
                    vals = [SOCIAL_DIMS_LOCAL[dim_name][c] for c in block_conds]
                    for pc_i in range(min(3, block_pcs.shape[1])):
                        r, p = spearmanr(block_pcs[:, pc_i], vals)
                        if abs(r) > 0.5:
                            sig = "*" if p < 0.05 else ""
                            print(f"    {dim_name} vs PC{pc_i+1}: rho={r:+.3f}{sig}")

        # ═══════════════════════════════════════════════════════
        # 5. Visualization
        # ═══════════════════════════════════════════════════════
        print("\n--- 5. Generating figures ---")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 5a: All stimuli PC1 vs PC2 colored by block
        ax = axes[0, 0]
        ax.scatter(pcs_all[is_aff, 0], pcs_all[is_aff, 1],
                   c=C_AFF, alpha=0.4, s=15, label="Affective")
        ax.scatter(pcs_all[is_soc, 0], pcs_all[is_soc, 1],
                   c=C_MENT, alpha=0.4, s=15, label="Social")
        ax.set_xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"{short}: 712 stimuli in activation space", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        # 5b: Condition centroids PC1 vs PC2
        ax = axes[0, 1]
        for i, c in enumerate(ORDER):
            color = C_AFF if c in AFF else C_MENT
            ax.scatter(pcs_cond[i, 0], pcs_cond[i, 1], c=color, s=100, zorder=3)
            ax.annotate(c, (pcs_cond[i, 0], pcs_cond[i, 1]),
                       fontsize=8, fontweight="bold",
                       xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel(f"PC1 ({pca_cond.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_cond.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"{short}: 14 condition centroids", fontsize=11)
        ax.grid(alpha=0.2)
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.axvline(0, color="#ccc", lw=0.5)

        # 5c: Probing results
        ax = axes[1, 0]
        targets = list(probe_results.keys())
        r2s = [probe_results[t]["r2_cv"] for t in targets]
        colors = [C_BLUE if r > 0.05 else "#ccc" for r in r2s]
        ax.barh(range(len(targets)), r2s, color=colors)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets)
        ax.set_xlabel("Cross-validated R²")
        ax.set_title("Linear probing: what's decodable?", fontsize=11)
        ax.axvline(0.05, color="#999", ls="--", lw=0.8, label="threshold")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=8)

        # 5d: PC1 distribution by condition
        ax = axes[1, 1]
        cond_pc1_means = [pcs_all[[i for i, c in enumerate(cond_labels) if c == cond], 0].mean()
                          for cond in ORDER]
        colors = [C_AFF if c in AFF else C_MENT for c in ORDER]
        ax.barh(range(len(ORDER)), cond_pc1_means, color=colors)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER, fontsize=8)
        ax.set_xlabel("Mean PC1 score")
        ax.set_title("PC1 by condition (= block boundary?)", fontsize=11)
        ax.axvline(0, color="#333", lw=0.5)
        ax.grid(axis="x", alpha=0.2)

        fig.tight_layout()
        fig.savefig(FIG_DIR / "activation_geometry.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIG_DIR / 'activation_geometry.png'}")
        plt.close(fig)

        results[short] = {
            "pca_all_variance": pca_all.explained_variance_ratio_[:10].tolist(),
            "pca_cond_variance": pca_cond.explained_variance_ratio_.tolist(),
            "pc1_block_separation_std": float(pc1_separation),
            "probe_results": probe_results,
            "condition_pc_positions": {c: {"PC1": float(pcs_cond[i, 0]),
                                           "PC2": float(pcs_cond[i, 1]),
                                           "PC3": float(pcs_cond[i, 2])}
                                       for i, c in enumerate(ORDER)},
        }

    # Save
    json.dump(results, open(OUT_DIR / "activation_geometry.json", "w"), indent=2)
    print(f"\nSaved: {OUT_DIR / 'activation_geometry.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
