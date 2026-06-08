#!/usr/bin/env python3
"""
Precise geometry characterization: what dimensions organize emotion vs social
cognition in LLMs vs brain?

Uses stimulus-derived features (not hand-coded category labels):
  - VAD from Warriner 2013 norms (actual word-level lookup per stimulus)
  - Approach/withdrawal from action tendency literature
  - Agency/control dimension
  - Social direction (self vs other)
  - Propositionality features for social conditions

Method: pair-wise distance correlation (more powerful than PCA with small n).
For each theoretical dimension, compute inter-condition distance, then correlate
with brain/LLM within-block distances.

Output:
  results/mechanistic/geometry_precise.json
  figures/geometry_precise.png
"""
from __future__ import annotations
import json, re, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR = BASE / "figures"

AFF = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
MENT = ["judgment", "belief", "intention", "mentalizing", "moral", "empathy",
        "self_referential", "theory_of_mind"]
ORDER = AFF + MENT

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
MSHORT = {"Qwen2.5-7B-Instruct": "Qwen", "Meta-Llama-3.1-8B-Instruct": "Llama",
          "Mistral-7B-Instruct-v0.3": "Mistral", "gemma-2-9b-it": "Gemma"}


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def triu_vec(mat):
    return mat[np.triu_indices(mat.shape[0], 1)]


def reorder_rdm(rdm, conds, order):
    idx = [conds.index(c) for c in order]
    return rdm[np.ix_(idx, idx)]


# Appraisal theory dimensions (Smith & Ellsworth 1985; Frijda 1986)
# These are per-CONDITION theoretical codings
APPRAISAL_DIMS = {
    # certainty: how certain/predictable is the eliciting situation
    # (anger=certain about cause, fear=uncertain, sadness=certain about loss)
    "certainty": {
        "anger": 0.8, "fear": 0.2, "disgust": 0.7,
        "sadness": 0.7, "happiness": 0.6, "valence": 0.5,
    },
    # agency: how much personal control/agency (anger=high, fear=low, sadness=low)
    "agency": {
        "anger": 0.8, "fear": 0.2, "disgust": 0.5,
        "sadness": 0.2, "happiness": 0.7, "valence": 0.5,
    },
    # other_accountability: is someone else responsible (anger=yes, sadness=no/fate)
    "other_accountability": {
        "anger": 0.9, "fear": 0.4, "disgust": 0.6,
        "sadness": 0.3, "happiness": 0.3, "valence": 0.5,
    },
    # approach_tendency: action readiness direction (Frijda)
    "approach_tendency": {
        "anger": 0.8, "fear": 0.1, "disgust": 0.1,
        "sadness": 0.2, "happiness": 0.9, "valence": 0.5,
    },
    # social_sharing: tendency to share/communicate this emotion
    "social_sharing": {
        "anger": 0.7, "fear": 0.5, "disgust": 0.6,
        "sadness": 0.8, "happiness": 0.9, "valence": 0.5,
    },
}

# Social cognition: theory-driven relational features
SOCIAL_DIMS = {
    "n_agents": {
        "belief": 2, "intention": 1.5, "mentalizing": 2.5,
        "theory_of_mind": 2.5, "judgment": 1.5, "moral": 2.5,
        "empathy": 2, "self_referential": 1,
    },
    "recursion_depth": {
        "belief": 1, "intention": 0.5, "mentalizing": 2,
        "theory_of_mind": 2, "judgment": 0.5, "moral": 0.5,
        "empathy": 1, "self_referential": 0,
    },
    "propositional_content": {
        "belief": 1, "intention": 0.8, "mentalizing": 1,
        "theory_of_mind": 1, "judgment": 0.5, "moral": 0.5,
        "empathy": 0.3, "self_referential": 0.3,
    },
    "normative": {
        "belief": 0, "intention": 0, "mentalizing": 0,
        "theory_of_mind": 0, "judgment": 0.8, "moral": 1,
        "empathy": 0.2, "self_referential": 0,
    },
    "self_other": {  # 0=about others, 1=about self
        "belief": 0.2, "intention": 0.3, "mentalizing": 0.1,
        "theory_of_mind": 0.1, "judgment": 0.4, "moral": 0.2,
        "empathy": 0.5, "self_referential": 1.0,
    },
    "affective_component": {  # how much affect is involved
        "belief": 0.1, "intention": 0.1, "mentalizing": 0.2,
        "theory_of_mind": 0.2, "judgment": 0.3, "moral": 0.5,
        "empathy": 0.9, "self_referential": 0.4,
    },
}


def main():
    print("=" * 70)
    print("PRECISE GEOMETRY CHARACTERIZATION")
    print("=" * 70)

    # Load stimuli
    stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    stimuli = [json.loads(l) for l in open(stim_path)]

    # Load Warriner VAD norms
    vad_path = BASE / "data" / "cognitive_stimuli" / "emotion" / "warriner_vad.csv"
    vad_norms = {}
    with open(vad_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vad_norms[row["word"].lower()] = {
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "dominance": float(row["dominance"]),
            }
    print(f"Loaded {len(vad_norms)} VAD norms")

    # ═══════════════════════════════════════════════════════════
    # Step 1: Compute stimulus-derived VAD per condition
    # ═══════════════════════════════════════════════════════════
    print("\n--- Step 1: Stimulus-derived VAD per condition ---")

    cond_vad = {}
    for cond in ORDER:
        texts = [s["text"] for s in stimuli if s["condition"] == cond]
        all_v, all_a, all_d = [], [], []
        for text in texts:
            for tok in tokenize(text):
                if tok in vad_norms:
                    all_v.append(vad_norms[tok]["valence"])
                    all_a.append(vad_norms[tok]["arousal"])
                    all_d.append(vad_norms[tok]["dominance"])
        cond_vad[cond] = {
            "valence": np.mean(all_v) if all_v else 5.0,
            "arousal": np.mean(all_a) if all_a else 5.0,
            "dominance": np.mean(all_d) if all_d else 5.0,
            "n_words_matched": len(all_v),
        }

    print(f"{'Condition':>20s} | {'Valence':>8s} | {'Arousal':>8s} | {'Dominance':>8s} | {'N words':>7s}")
    print("-" * 65)
    for cond in ORDER:
        v = cond_vad[cond]
        blk = "AFF" if cond in AFF else "SOC"
        print(f"{cond:>20s} | {v['valence']:8.3f} | {v['arousal']:8.3f} | "
              f"{v['dominance']:8.3f} | {v['n_words_matched']:7d}  [{blk}]")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Load brain and LLM RDMs (within-block)
    # ═══════════════════════════════════════════════════════════
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = reorder_rdm(ns["rdm"], list(ns["conditions"]), ORDER)

    llm_rdms = {}
    for m in MODELS:
        z = np.load(RSA / f"{m}_rdm14_headline.npz", allow_pickle=True)
        llm_rdms[MSHORT[m]] = reorder_rdm(z["rdm"], list(z["conditions"]), ORDER)
    avg_llm_rdm = np.mean(list(llm_rdms.values()), axis=0)

    aff_idx = [ORDER.index(c) for c in AFF]
    soc_idx = [ORDER.index(c) for c in MENT]

    brain_aff = brain_rdm[np.ix_(aff_idx, aff_idx)]
    brain_soc = brain_rdm[np.ix_(soc_idx, soc_idx)]
    llm_aff = avg_llm_rdm[np.ix_(aff_idx, aff_idx)]
    llm_soc = avg_llm_rdm[np.ix_(soc_idx, soc_idx)]

    brain_aff_vec = triu_vec(brain_aff)
    brain_soc_vec = triu_vec(brain_soc)
    llm_aff_vec = triu_vec(llm_aff)
    llm_soc_vec = triu_vec(llm_soc)

    # ═══════════════════════════════════════════════════════════
    # Step 3: EMOTION — which dimensions predict brain vs LLM?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EMOTION GEOMETRY: what dimensions predict brain vs LLM distances?")
    print("=" * 70)

    # Build dimension-based distance vectors (15 pairs for 6 emotions)
    emotion_dims = {}

    # VAD from stimuli
    for vad_dim in ["valence", "arousal", "dominance"]:
        vec = np.array([cond_vad[c][vad_dim] for c in AFF]).reshape(-1, 1)
        rdm = squareform(pdist(vec, "euclidean"))
        emotion_dims[f"stim_{vad_dim}"] = triu_vec(rdm)

    # Appraisal dimensions
    for dim_name, dim_vals in APPRAISAL_DIMS.items():
        vec = np.array([dim_vals[c] for c in AFF]).reshape(-1, 1)
        rdm = squareform(pdist(vec, "euclidean"))
        emotion_dims[dim_name] = triu_vec(rdm)

    # Combined VAD distance
    vad_mat = np.array([[cond_vad[c]["valence"], cond_vad[c]["arousal"],
                         cond_vad[c]["dominance"]] for c in AFF])
    from sklearn.preprocessing import StandardScaler
    vad_rdm = squareform(pdist(StandardScaler().fit_transform(vad_mat), "euclidean"))
    emotion_dims["stim_VAD_combined"] = triu_vec(vad_rdm)

    # Report correlations
    print(f"\n{'Dimension':>25s} | {'vs Brain':>10s} | {'vs LLM':>10s} | {'Stronger for':>12s}")
    print("-" * 70)

    emotion_results = {}
    for dim_name, dim_vec in emotion_dims.items():
        r_brain, p_brain = spearmanr(dim_vec, brain_aff_vec)
        r_llm, p_llm = spearmanr(dim_vec, llm_aff_vec)
        stronger = "BRAIN" if abs(r_brain) > abs(r_llm) else "LLM" if abs(r_llm) > abs(r_brain) else "="
        sig_b = "*" if p_brain < 0.05 else ""
        sig_l = "*" if p_llm < 0.05 else ""
        print(f"{dim_name:>25s} | {r_brain:+.3f}{sig_b:>2s}  | {r_llm:+.3f}{sig_l:>2s}  | {stronger:>12s}")
        emotion_results[dim_name] = {
            "rho_brain": float(r_brain), "p_brain": float(p_brain),
            "rho_llm": float(r_llm), "p_llm": float(p_llm),
        }

    # Per-model emotion results
    print(f"\n--- Per-model emotion (approach_tendency) ---")
    approach_vec = emotion_dims["approach_tendency"]
    for m, short in MSHORT.items():
        m_rdm = llm_rdms[short][np.ix_(aff_idx, aff_idx)]
        m_vec = triu_vec(m_rdm)
        r, p = spearmanr(approach_vec, m_vec)
        print(f"  {short:>8s}: rho={r:+.3f} (p={p:.3f})")

    # ═══════════════════════════════════════════════════════════
    # Step 4: SOCIAL — which dimensions predict brain vs LLM?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SOCIAL GEOMETRY: what dimensions predict brain vs LLM distances?")
    print("=" * 70)

    social_dims = {}
    for dim_name, dim_vals in SOCIAL_DIMS.items():
        vec = np.array([dim_vals[c] for c in MENT]).reshape(-1, 1)
        rdm = squareform(pdist(vec, "euclidean"))
        social_dims[dim_name] = triu_vec(rdm)

    # Combined relational distance
    rel_mat = np.array([[SOCIAL_DIMS[d][c] for d in SOCIAL_DIMS] for c in MENT])
    rel_rdm = squareform(pdist(StandardScaler().fit_transform(rel_mat), "euclidean"))
    social_dims["combined_relational"] = triu_vec(rel_rdm)

    # Also add stimuli-derived VAD for social conditions
    for vad_dim in ["valence", "arousal", "dominance"]:
        vec = np.array([cond_vad[c][vad_dim] for c in MENT]).reshape(-1, 1)
        rdm = squareform(pdist(vec, "euclidean"))
        social_dims[f"stim_{vad_dim}"] = triu_vec(rdm)

    print(f"\n{'Dimension':>25s} | {'vs Brain':>10s} | {'vs LLM':>10s} | {'Stronger for':>12s}")
    print("-" * 70)

    social_results = {}
    for dim_name, dim_vec in social_dims.items():
        r_brain, p_brain = spearmanr(dim_vec, brain_soc_vec)
        r_llm, p_llm = spearmanr(dim_vec, llm_soc_vec)
        stronger = "BRAIN" if abs(r_brain) > abs(r_llm) else "LLM" if abs(r_llm) > abs(r_brain) else "="
        sig_b = "*" if p_brain < 0.05 else ""
        sig_l = "*" if p_llm < 0.05 else ""
        print(f"{dim_name:>25s} | {r_brain:+.3f}{sig_b:>2s}  | {r_llm:+.3f}{sig_l:>2s}  | {stronger:>12s}")
        social_results[dim_name] = {
            "rho_brain": float(r_brain), "p_brain": float(p_brain),
            "rho_llm": float(r_llm), "p_llm": float(p_llm),
        }

    # ═══════════════════════════════════════════════════════════
    # Step 5: KEY CONTRAST — which dimensions are shared vs divergent?
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONTRAST: Shared vs divergent organizing dimensions")
    print("=" * 70)

    print("\n  EMOTION — Dimensions that predict LLM but NOT brain (LLM-specific):")
    for dim, res in emotion_results.items():
        if abs(res["rho_llm"]) > 0.3 and abs(res["rho_brain"]) < 0.3:
            print(f"    {dim}: LLM rho={res['rho_llm']:+.3f}, brain rho={res['rho_brain']:+.3f}")

    print("\n  EMOTION — Dimensions that predict brain but NOT LLM (brain-specific):")
    for dim, res in emotion_results.items():
        if abs(res["rho_brain"]) > 0.3 and abs(res["rho_llm"]) < 0.3:
            print(f"    {dim}: brain rho={res['rho_brain']:+.3f}, LLM rho={res['rho_llm']:+.3f}")

    print("\n  EMOTION — Dimensions shared (both > 0.3):")
    for dim, res in emotion_results.items():
        if abs(res["rho_brain"]) > 0.3 and abs(res["rho_llm"]) > 0.3:
            print(f"    {dim}: brain rho={res['rho_brain']:+.3f}, LLM rho={res['rho_llm']:+.3f}")

    print("\n  SOCIAL — Dimensions shared (both > 0.3):")
    for dim, res in social_results.items():
        if abs(res["rho_brain"]) > 0.3 and abs(res["rho_llm"]) > 0.3:
            print(f"    {dim}: brain rho={res['rho_brain']:+.3f}, LLM rho={res['rho_llm']:+.3f}")

    print("\n  SOCIAL — Dimensions that predict LLM but NOT brain:")
    for dim, res in social_results.items():
        if abs(res["rho_llm"]) > 0.3 and abs(res["rho_brain"]) < 0.3:
            print(f"    {dim}: LLM rho={res['rho_llm']:+.3f}, brain rho={res['rho_brain']:+.3f}")

    print("\n  SOCIAL — Dimensions that predict brain but NOT LLM:")
    for dim, res in social_results.items():
        if abs(res["rho_brain"]) > 0.3 and abs(res["rho_llm"]) < 0.3:
            print(f"    {dim}: brain rho={res['rho_brain']:+.3f}, LLM rho={res['rho_llm']:+.3f}")

    # ═══════════════════════════════════════════════════════════
    # Step 6: Visualization
    # ═══════════════════════════════════════════════════════════
    print("\n--- Generating figures ---")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Emotion dimensions
    ax = axes[0]
    dims_e = list(emotion_results.keys())
    brain_r = [emotion_results[d]["rho_brain"] for d in dims_e]
    llm_r = [emotion_results[d]["rho_llm"] for d in dims_e]
    x = np.arange(len(dims_e))
    w = 0.35
    ax.barh(x - w/2, brain_r, w, color="#e8743b", alpha=0.7, label="vs Brain")
    ax.barh(x + w/2, llm_r, w, color="#2e5cb8", alpha=0.7, label="vs LLM")
    ax.set_yticks(x)
    ax.set_yticklabels([d.replace("stim_", "").replace("_", " ") for d in dims_e], fontsize=8)
    ax.axvline(0, color="#333", lw=0.5)
    ax.axvline(0.3, color="#999", lw=0.5, ls="--")
    ax.axvline(-0.3, color="#999", lw=0.5, ls="--")
    ax.set_xlabel("Spearman rho with within-block distances")
    ax.set_title("EMOTION: which dimensions organize it?", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.8, 0.8)
    ax.grid(axis="x", alpha=0.2)

    # Right: Social dimensions
    ax = axes[1]
    dims_s = list(social_results.keys())
    brain_r_s = [social_results[d]["rho_brain"] for d in dims_s]
    llm_r_s = [social_results[d]["rho_llm"] for d in dims_s]
    x = np.arange(len(dims_s))
    ax.barh(x - w/2, brain_r_s, w, color="#19a979", alpha=0.7, label="vs Brain")
    ax.barh(x + w/2, llm_r_s, w, color="#2e5cb8", alpha=0.7, label="vs LLM")
    ax.set_yticks(x)
    ax.set_yticklabels([d.replace("stim_", "").replace("_", " ") for d in dims_s], fontsize=8)
    ax.axvline(0, color="#333", lw=0.5)
    ax.axvline(0.3, color="#999", lw=0.5, ls="--")
    ax.axvline(-0.3, color="#999", lw=0.5, ls="--")
    ax.set_xlabel("Spearman rho with within-block distances")
    ax.set_title("SOCIAL: which dimensions organize it?", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.8, 0.8)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_precise.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIG_DIR / 'geometry_precise.png'}")
    plt.close(fig)

    # Save
    results = {
        "stimulus_vad_per_condition": {c: {k: v for k, v in cond_vad[c].items()}
                                       for c in ORDER},
        "emotion_dimension_correlations": emotion_results,
        "social_dimension_correlations": social_results,
        "n_emotion_pairs": 15,
        "n_social_pairs": 28,
    }
    json.dump(results, open(OUT_DIR / "geometry_precise.json", "w"), indent=2)
    print(f"Saved: {OUT_DIR / 'geometry_precise.json'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
