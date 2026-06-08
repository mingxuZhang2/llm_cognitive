#!/usr/bin/env python3
"""
Deep analysis of the two geometries: what dimensions organize the social
cognition subspace vs the emotion subspace in LLMs vs brain?

1. MDS from RDMs to recover coordinates
2. PCA on each block to find organizing dimensions
3. Compare LLM emotion dimensions with VAD (valence-arousal-dominance)
4. Compare LLM social dimensions with relational features
5. Cross-compare brain vs LLM principal axes

Output:
  results/mechanistic/geometry_deep_analysis.json
  figures/geometry_deep_analysis.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
MSHORT = {"Qwen2.5-7B-Instruct": "Qwen", "Meta-Llama-3.1-8B-Instruct": "Llama",
          "Mistral-7B-Instruct-v0.3": "Mistral", "gemma-2-9b-it": "Gemma"}

C_AFF, C_MENT, C_BLUE = "#e8743b", "#19a979", "#2e5cb8"

# ═══════════════════════════════════════════════════════════════
# Theoretical dimension codings
# ═══════════════════════════════════════════════════════════════

# VAD norms for emotion conditions (from Warriner 2013 canonical values)
# Scale: 1-9, centered at 5
VAD_NORMS = {
    "anger":     {"valence": 2.34, "arousal": 6.18, "dominance": 4.11},
    "fear":      {"valence": 2.64, "arousal": 6.33, "dominance": 2.77},
    "disgust":   {"valence": 2.42, "arousal": 5.08, "dominance": 4.03},
    "sadness":   {"valence": 2.10, "arousal": 3.49, "dominance": 2.82},
    "happiness": {"valence": 8.21, "arousal": 6.49, "dominance": 6.63},
    "valence":   {"valence": 5.85, "arousal": 4.27, "dominance": 5.12},
}

# Approach/withdrawal coding (theoretical, from appraisal theory)
APPROACH_WITHDRAWAL = {
    "anger":     {"approach": 0.8, "withdrawal": 0.2},  # anger → approach/confront
    "fear":      {"approach": 0.1, "withdrawal": 0.9},  # fear → flee/avoid
    "disgust":   {"approach": 0.1, "withdrawal": 0.9},  # disgust → reject/avoid
    "sadness":   {"approach": 0.2, "withdrawal": 0.8},  # sadness → withdraw/reflect
    "happiness": {"approach": 0.9, "withdrawal": 0.1},  # happiness → approach/engage
    "valence":   {"approach": 0.5, "withdrawal": 0.5},  # neutral
}

# Social/self-directed coding
SOCIAL_DIRECTION = {
    "anger":     {"other_directed": 0.8, "self_directed": 0.2},
    "fear":      {"other_directed": 0.3, "self_directed": 0.7},
    "disgust":   {"other_directed": 0.7, "self_directed": 0.3},
    "sadness":   {"other_directed": 0.2, "self_directed": 0.8},
    "happiness": {"other_directed": 0.5, "self_directed": 0.5},
    "valence":   {"other_directed": 0.5, "self_directed": 0.5},
}

# Relational features for social conditions
RELATIONAL_FEATURES = {
    "belief":          {"n_agents": 2, "recursion_depth": 1, "false_belief": 0.5,
                        "normative": 0, "self_directed": 0, "propositional": 1},
    "intention":       {"n_agents": 1.5, "recursion_depth": 0, "false_belief": 0,
                        "normative": 0, "self_directed": 0.3, "propositional": 0.8},
    "mentalizing":     {"n_agents": 2, "recursion_depth": 2, "false_belief": 0.3,
                        "normative": 0, "self_directed": 0, "propositional": 1},
    "theory_of_mind":  {"n_agents": 2, "recursion_depth": 2, "false_belief": 0.7,
                        "normative": 0, "self_directed": 0, "propositional": 1},
    "judgment":        {"n_agents": 1, "recursion_depth": 0, "false_belief": 0,
                        "normative": 1, "self_directed": 0.3, "propositional": 0.5},
    "moral":           {"n_agents": 2, "recursion_depth": 0, "false_belief": 0,
                        "normative": 1, "self_directed": 0, "propositional": 0.5},
    "empathy":         {"n_agents": 2, "recursion_depth": 1, "false_belief": 0,
                        "normative": 0, "self_directed": 0.5, "propositional": 0.5},
    "self_referential": {"n_agents": 1, "recursion_depth": 0, "false_belief": 0,
                         "normative": 0, "self_directed": 1, "propositional": 0.3},
}


def reorder_rdm(rdm, conds, order):
    idx = [conds.index(c) for c in order]
    return rdm[np.ix_(idx, idx)]


def rdm_to_coords(rdm, n_dims=3):
    mds = MDS(n_components=n_dims, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto")
    return mds.fit_transform(rdm)


def triu_vec(mat):
    return mat[np.triu_indices(mat.shape[0], 1)]


def analyze_block(coords, labels, block_name, theoretical_dims, dim_names):
    """PCA on a block's coordinates + correlation with theoretical dimensions."""
    print(f"\n{'='*60}")
    print(f"  {block_name} GEOMETRY")
    print(f"{'='*60}")

    n = coords.shape[0]
    pca = PCA()
    pca_coords = pca.fit_transform(StandardScaler().fit_transform(coords))

    print(f"\n  PCA variance explained:")
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_,
                                       np.cumsum(pca.explained_variance_ratio_))):
        print(f"    PC{i+1}: {var:.1%} (cumulative: {cum:.1%})")

    # Position of each condition on PC1, PC2
    print(f"\n  Condition positions on PC1, PC2:")
    for i, lab in enumerate(labels):
        print(f"    {lab:>20s}: PC1={pca_coords[i,0]:+.3f}  PC2={pca_coords[i,1]:+.3f}")

    # Correlate PCs with theoretical dimensions
    if theoretical_dims is not None:
        print(f"\n  Correlation of PCs with theoretical dimensions:")
        print(f"  {'Dimension':>25s} | {'PC1':>8s} | {'PC2':>8s} | {'PC3':>8s}")
        print(f"  {'-'*60}")

        corr_results = {}
        for dim_name in dim_names:
            vals = [theoretical_dims[lab][dim_name] for lab in labels]
            corrs = []
            for pc_i in range(min(3, pca_coords.shape[1])):
                r, p = spearmanr(pca_coords[:, pc_i], vals)
                corrs.append((r, p))
            sig = lambda r, p: f"{r:+.3f}{'*' if p < 0.05 else ' '}"
            line = f"  {dim_name:>25s} |"
            for r, p in corrs:
                line += f" {sig(r,p):>8s} |"
            print(line)
            corr_results[dim_name] = {f"PC{i+1}": {"rho": float(corrs[i][0]),
                                                     "p": float(corrs[i][1])}
                                       for i in range(len(corrs))}

        return {
            "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
            "condition_pc_coords": {lab: {"PC1": float(pca_coords[i, 0]),
                                          "PC2": float(pca_coords[i, 1])}
                                    for i, lab in enumerate(labels)},
            "theoretical_correlations": corr_results,
        }

    return {
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "condition_pc_coords": {lab: {"PC1": float(pca_coords[i, 0]),
                                      "PC2": float(pca_coords[i, 1])}
                                for i, lab in enumerate(labels)},
    }


def compare_axes(coords_a, coords_b, labels, name_a, name_b):
    """Compare PCA axes between two systems (e.g., brain vs LLM)."""
    pca_a = PCA(n_components=min(3, len(labels)-1))
    pca_b = PCA(n_components=min(3, len(labels)-1))

    sc_a = StandardScaler().fit_transform(coords_a)
    sc_b = StandardScaler().fit_transform(coords_b)

    ca = pca_a.fit_transform(sc_a)
    cb = pca_b.fit_transform(sc_b)

    print(f"\n  Axis alignment ({name_a} vs {name_b}):")
    results = {}
    for i in range(min(3, ca.shape[1], cb.shape[1])):
        for j in range(min(3, ca.shape[1], cb.shape[1])):
            r, p = spearmanr(ca[:, i], cb[:, j])
            if abs(r) > 0.4 or (i == j):
                sig = "*" if p < 0.05 else ""
                print(f"    {name_a}_PC{i+1} vs {name_b}_PC{j+1}: rho={r:+.3f}{sig}")
                results[f"{name_a}_PC{i+1}_vs_{name_b}_PC{j+1}"] = {
                    "rho": float(r), "p": float(p)}

    return results


def main():
    print("=" * 70)
    print("DEEP GEOMETRY ANALYSIS: Two subspaces, two organizing principles")
    print("=" * 70)

    # Load brain RDM
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = reorder_rdm(ns["rdm"], list(ns["conditions"]), ORDER)

    # Load LLM RDMs
    llm_rdms = {}
    for m in MODELS:
        z = np.load(RSA / f"{m}_rdm14_headline.npz", allow_pickle=True)
        llm_rdms[MSHORT[m]] = reorder_rdm(z["rdm"], list(z["conditions"]), ORDER)

    avg_llm_rdm = np.mean(list(llm_rdms.values()), axis=0)

    # Block indices
    aff_idx = [ORDER.index(c) for c in AFF]
    soc_idx = [ORDER.index(c) for c in MENT]

    # ═══════════════════════════════════════════════════════════
    # MDS to recover coordinates (full 14-condition space)
    # ═══════════════════════════════════════════════════════════
    print("\n--- MDS embedding (14 conditions → 3D) ---")
    brain_coords = rdm_to_coords(brain_rdm)
    llm_coords = rdm_to_coords(avg_llm_rdm)

    per_model_coords = {m: rdm_to_coords(rdm) for m, rdm in llm_rdms.items()}

    results = {}

    # ═══════════════════════════════════════════════════════════
    # EMOTION BLOCK ANALYSIS
    # ═══════════════════════════════════════════════════════════

    # --- Brain emotion geometry ---
    brain_aff_coords = brain_coords[aff_idx]
    print("\n" + "=" * 60)
    print("  BRAIN — EMOTION GEOMETRY")
    vad_dims = list(VAD_NORMS["anger"].keys())
    brain_aff_result = analyze_block(
        brain_aff_coords, AFF, "Brain Emotion",
        VAD_NORMS, vad_dims)
    results["brain_emotion"] = brain_aff_result

    # Also test approach/withdrawal
    aw_dims = list(APPROACH_WITHDRAWAL["anger"].keys())
    print("\n  + Approach/withdrawal:")
    for dim_name in aw_dims:
        vals = [APPROACH_WITHDRAWAL[c][dim_name] for c in AFF]
        pca = PCA()
        pc = pca.fit_transform(StandardScaler().fit_transform(brain_aff_coords))
        for i in range(min(2, pc.shape[1])):
            r, p = spearmanr(pc[:, i], vals)
            if abs(r) > 0.3:
                print(f"    {dim_name} vs PC{i+1}: rho={r:+.3f}{'*' if p<0.05 else ''}")

    # --- LLM emotion geometry (average across 4 models) ---
    llm_aff_coords = llm_coords[aff_idx]
    llm_aff_result = analyze_block(
        llm_aff_coords, AFF, "LLM(avg) Emotion",
        VAD_NORMS, vad_dims)
    results["llm_avg_emotion"] = llm_aff_result

    # Also approach/withdrawal for LLM
    print("\n  + Approach/withdrawal:")
    for dim_name in aw_dims:
        vals = [APPROACH_WITHDRAWAL[c][dim_name] for c in AFF]
        pca = PCA()
        pc = pca.fit_transform(StandardScaler().fit_transform(llm_aff_coords))
        for i in range(min(2, pc.shape[1])):
            r, p = spearmanr(pc[:, i], vals)
            if abs(r) > 0.3:
                print(f"    {dim_name} vs PC{i+1}: rho={r:+.3f}{'*' if p<0.05 else ''}")

    # --- Per-model emotion geometry ---
    for m, coords in per_model_coords.items():
        aff_c = coords[aff_idx]
        r = analyze_block(aff_c, AFF, f"{m} Emotion", VAD_NORMS, vad_dims)
        results[f"{m}_emotion"] = r

    # --- Brain vs LLM emotion axis comparison ---
    print("\n--- Brain vs LLM: emotion axis alignment ---")
    emo_axis = compare_axes(brain_aff_coords, llm_aff_coords, AFF, "Brain", "LLM")
    results["emotion_axis_alignment"] = emo_axis

    # ═══════════════════════════════════════════════════════════
    # SOCIAL COGNITION BLOCK ANALYSIS
    # ═══════════════════════════════════════════════════════════

    rel_dims = list(RELATIONAL_FEATURES["belief"].keys())

    # --- Brain social geometry ---
    brain_soc_coords = brain_coords[soc_idx]
    brain_soc_result = analyze_block(
        brain_soc_coords, MENT, "Brain Social",
        RELATIONAL_FEATURES, rel_dims)
    results["brain_social"] = brain_soc_result

    # --- LLM social geometry ---
    llm_soc_coords = llm_coords[soc_idx]
    llm_soc_result = analyze_block(
        llm_soc_coords, MENT, "LLM(avg) Social",
        RELATIONAL_FEATURES, rel_dims)
    results["llm_avg_social"] = llm_soc_result

    # --- Per-model social geometry ---
    for m, coords in per_model_coords.items():
        soc_c = coords[soc_idx]
        r = analyze_block(soc_c, MENT, f"{m} Social", RELATIONAL_FEATURES, rel_dims)
        results[f"{m}_social"] = r

    # --- Brain vs LLM social axis comparison ---
    print("\n--- Brain vs LLM: social axis alignment ---")
    soc_axis = compare_axes(brain_soc_coords, llm_soc_coords, MENT, "Brain", "LLM")
    results["social_axis_alignment"] = soc_axis

    # ═══════════════════════════════════════════════════════════
    # Build theoretical-dimension RDMs and compare
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  THEORETICAL RDMs vs BRAIN / LLM")
    print("=" * 60)

    from scipy.spatial.distance import pdist

    # VAD RDM (emotion block only)
    vad_mat = np.array([[VAD_NORMS[c][d] for d in vad_dims] for c in AFF])
    vad_rdm = squareform(pdist(StandardScaler().fit_transform(vad_mat), "euclidean"))
    vad_vec = triu_vec(vad_rdm)

    brain_aff_rdm = brain_rdm[np.ix_(aff_idx, aff_idx)]
    brain_aff_vec = triu_vec(brain_aff_rdm)
    llm_aff_rdm = avg_llm_rdm[np.ix_(aff_idx, aff_idx)]
    llm_aff_vec = triu_vec(llm_aff_rdm)

    r_vad_brain, p_vad_brain = spearmanr(vad_vec, brain_aff_vec)
    r_vad_llm, p_vad_llm = spearmanr(vad_vec, llm_aff_vec)
    print(f"\n  VAD RDM vs brain within-aff: rho={r_vad_brain:+.3f} (p={p_vad_brain:.3f})")
    print(f"  VAD RDM vs LLM  within-aff: rho={r_vad_llm:+.3f} (p={p_vad_llm:.3f})")

    results["vad_rdm_comparison"] = {
        "vs_brain": {"rho": float(r_vad_brain), "p": float(p_vad_brain)},
        "vs_llm": {"rho": float(r_vad_llm), "p": float(p_vad_llm)},
    }

    # Approach/withdrawal RDM
    aw_mat = np.array([[APPROACH_WITHDRAWAL[c][d] for d in aw_dims] for c in AFF])
    aw_rdm = squareform(pdist(aw_mat, "euclidean"))
    aw_vec = triu_vec(aw_rdm)

    r_aw_brain, p_aw_brain = spearmanr(aw_vec, brain_aff_vec)
    r_aw_llm, p_aw_llm = spearmanr(aw_vec, llm_aff_vec)
    print(f"\n  Approach/withdrawal RDM vs brain: rho={r_aw_brain:+.3f} (p={p_aw_brain:.3f})")
    print(f"  Approach/withdrawal RDM vs LLM:   rho={r_aw_llm:+.3f} (p={p_aw_llm:.3f})")

    results["approach_withdrawal_rdm"] = {
        "vs_brain": {"rho": float(r_aw_brain), "p": float(p_aw_brain)},
        "vs_llm": {"rho": float(r_aw_llm), "p": float(p_aw_llm)},
    }

    # Relational features RDM (social block only)
    rel_mat = np.array([[RELATIONAL_FEATURES[c][d] for d in rel_dims] for c in MENT])
    rel_rdm = squareform(pdist(StandardScaler().fit_transform(rel_mat), "euclidean"))
    rel_vec = triu_vec(rel_rdm)

    brain_soc_rdm = brain_rdm[np.ix_(soc_idx, soc_idx)]
    brain_soc_vec = triu_vec(brain_soc_rdm)
    llm_soc_rdm = avg_llm_rdm[np.ix_(soc_idx, soc_idx)]
    llm_soc_vec = triu_vec(llm_soc_rdm)

    r_rel_brain, p_rel_brain = spearmanr(rel_vec, brain_soc_vec)
    r_rel_llm, p_rel_llm = spearmanr(rel_vec, llm_soc_vec)
    print(f"\n  Relational RDM vs brain within-soc: rho={r_rel_brain:+.3f} (p={p_rel_brain:.3f})")
    print(f"  Relational RDM vs LLM  within-soc: rho={r_rel_llm:+.3f} (p={p_rel_llm:.3f})")

    results["relational_rdm_comparison"] = {
        "vs_brain": {"rho": float(r_rel_brain), "p": float(p_rel_brain)},
        "vs_llm": {"rho": float(r_rel_llm), "p": float(p_rel_llm)},
    }

    # ═══════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════
    print("\n--- Generating figures ---")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Emotion geometry
    # 1a: Brain emotion MDS
    ax = axes[0, 0]
    pca = PCA(n_components=2)
    bc = pca.fit_transform(StandardScaler().fit_transform(brain_aff_coords))
    for i, lab in enumerate(AFF):
        ax.scatter(bc[i, 0], bc[i, 1], c=C_AFF, s=80, zorder=3)
        ax.annotate(lab, (bc[i, 0], bc[i, 1]), fontsize=9, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("Brain — Emotion geometry", fontsize=11)
    ax.grid(alpha=0.2)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)

    # 1b: LLM emotion MDS
    ax = axes[0, 1]
    pca2 = PCA(n_components=2)
    lc = pca2.fit_transform(StandardScaler().fit_transform(llm_aff_coords))
    for i, lab in enumerate(AFF):
        ax.scatter(lc[i, 0], lc[i, 1], c=C_AFF, s=80, zorder=3)
        ax.annotate(lab, (lc[i, 0], lc[i, 1]), fontsize=9, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.0%})")
    ax.set_title("LLM(avg) — Emotion geometry", fontsize=11)
    ax.grid(alpha=0.2)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)

    # 1c: VAD vs approach/withdrawal comparison bars
    ax = axes[0, 2]
    comparisons = ["VAD", "Approach/\nWithdrawal"]
    brain_vals = [r_vad_brain, r_aw_brain]
    llm_vals = [r_vad_llm, r_aw_llm]
    x = np.arange(len(comparisons))
    w = 0.3
    ax.bar(x - w/2, brain_vals, w, color=C_AFF, alpha=0.6, label="vs Brain")
    ax.bar(x + w/2, llm_vals, w, color=C_BLUE, label="vs LLM")
    for i in range(len(comparisons)):
        ax.text(i - w/2, brain_vals[i] + 0.02, f"{brain_vals[i]:+.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, llm_vals[i] + 0.02, f"{llm_vals[i]:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(comparisons)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Emotion: which dimensions organize it?", fontsize=11)
    ax.legend(fontsize=8)
    ax.axhline(0, color="#333", lw=0.5)
    ax.set_ylim(-0.6, 0.8)
    ax.grid(axis="y", alpha=0.2)

    # Row 2: Social geometry
    # 2a: Brain social MDS
    ax = axes[1, 0]
    pca3 = PCA(n_components=2)
    bsc = pca3.fit_transform(StandardScaler().fit_transform(brain_soc_coords))
    for i, lab in enumerate(MENT):
        ax.scatter(bsc[i, 0], bsc[i, 1], c=C_MENT, s=80, zorder=3)
        ax.annotate(lab, (bsc[i, 0], bsc[i, 1]), fontsize=8, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]:.0%})")
    ax.set_title("Brain — Social geometry", fontsize=11)
    ax.grid(alpha=0.2)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)

    # 2b: LLM social MDS
    ax = axes[1, 1]
    pca4 = PCA(n_components=2)
    lsc = pca4.fit_transform(StandardScaler().fit_transform(llm_soc_coords))
    for i, lab in enumerate(MENT):
        ax.scatter(lsc[i, 0], lsc[i, 1], c=C_MENT, s=80, zorder=3)
        ax.annotate(lab, (lsc[i, 0], lsc[i, 1]), fontsize=8, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca4.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca4.explained_variance_ratio_[1]:.0%})")
    ax.set_title("LLM(avg) — Social geometry", fontsize=11)
    ax.grid(alpha=0.2)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axvline(0, color="#ccc", lw=0.5)

    # 2c: Relational features comparison
    ax = axes[1, 2]
    ax.bar([0], [r_rel_brain], 0.5, color=C_MENT, alpha=0.6, label="vs Brain")
    ax.bar([1], [r_rel_llm], 0.5, color=C_BLUE, label="vs LLM")
    ax.text(0, r_rel_brain + 0.02, f"{r_rel_brain:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.text(1, r_rel_llm + 0.02, f"{r_rel_llm:+.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["vs Brain", "vs LLM"])
    ax.set_ylabel("Spearman rho")
    ax.set_title("Social: relational features RDM", fontsize=11)
    ax.axhline(0, color="#333", lw=0.5)
    ax.set_ylim(-0.6, 0.8)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_deep_analysis.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIG_DIR / 'geometry_deep_analysis.png'}")
    plt.close(fig)

    # Save results
    json.dump(results, open(OUT_DIR / "geometry_deep_analysis.json", "w"), indent=2)
    print(f"\nSaved: {OUT_DIR / 'geometry_deep_analysis.json'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
