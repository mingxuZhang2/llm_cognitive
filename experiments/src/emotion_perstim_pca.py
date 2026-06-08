#!/usr/bin/env python3
"""
Per-stimulus PCA of the emotion subspace: what dimensions organize LLM's
representation of ~300 individual emotion stimuli?

Instead of 6 condition centroids (too few for stable PCA), uses all individual
emotion stimuli to identify the internal organizing dimensions with full
statistical power.

For each PC, correlates with:
  - Condition label (which of 6 emotions)
  - Per-stimulus VAD (Warriner norms)
  - Text properties (body words, action words, social words, etc.)
  - Narrative type features

This reveals what the LLM's emotion dimensions ACTUALLY ARE.
"""
from __future__ import annotations
import json, csv, re
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, f_oneway, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR = BASE / "figures"

AFF = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
ORDER_ALL = AFF + ["judgment", "belief", "intention", "mentalizing", "moral",
                   "empathy", "self_referential", "theory_of_mind"]

MODELS = ["Qwen2.5-7B-Instruct"]
PEAK = {"Qwen2.5-7B-Instruct": 27}

C_MAP = {
    "anger": "#e74c3c", "fear": "#8e44ad", "disgust": "#27ae60",
    "sadness": "#2980b9", "happiness": "#f39c12", "valence": "#95a5a6",
}


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


# Word lists for feature extraction
BODY_WORDS = {
    "body", "heart", "stomach", "hand", "hands", "face", "eye", "eyes",
    "skin", "blood", "breath", "chest", "throat", "head", "tremble",
    "shake", "cry", "scream", "sweat", "shiver", "gasp", "sob",
    "pain", "hurt", "touch", "feel", "felt", "warm", "cold", "tense",
    "dizzy", "faint", "tears", "pulse", "muscle", "fist", "fists",
}
ACTION_WORDS = {
    "run", "ran", "fight", "hit", "push", "pull", "grab", "throw",
    "scream", "yell", "shout", "cry", "laugh", "smile", "frown",
    "attack", "escape", "flee", "hide", "avoid", "confront", "approach",
    "retreat", "freeze", "stare", "slam", "kick", "punch", "break",
}
SOCIAL_WORDS = {
    "friend", "family", "mother", "father", "child", "children", "partner",
    "husband", "wife", "colleague", "boss", "neighbor", "stranger",
    "people", "person", "someone", "everyone", "they", "them", "her", "him",
    "relationship", "trust", "betray", "help", "support", "blame",
}
THREAT_WORDS = {
    "danger", "threat", "risk", "harm", "hurt", "kill", "die", "death",
    "attack", "violent", "unsafe", "scared", "terrified", "panic",
    "emergency", "warning", "escape", "trapped",
}
LOSS_WORDS = {
    "lost", "lose", "gone", "miss", "missing", "died", "death", "dead",
    "never", "goodbye", "farewell", "end", "over", "left", "alone",
    "empty", "nothing", "memory", "memories", "past", "used",
}
MORAL_WORDS = {
    "wrong", "right", "fair", "unfair", "just", "unjust", "moral",
    "immoral", "evil", "good", "bad", "shame", "guilty", "innocent",
    "deserve", "justice", "cruel", "kind", "honest", "dishonest",
    "betray", "cheat", "steal", "lie", "lied", "truth",
}
INTENSITY_WORDS = {
    "very", "extremely", "incredibly", "deeply", "intensely", "absolutely",
    "completely", "totally", "utterly", "overwhelming", "unbearable",
    "terrible", "horrible", "awful", "devastating", "excruciating",
}


def prop(tokens, word_set):
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in word_set) / len(tokens)


def main():
    print("=" * 70)
    print("PER-STIMULUS EMOTION PCA")
    print("=" * 70)

    # Load stimuli
    stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    all_stimuli = [json.loads(l) for l in open(stim_path)]

    # Filter to emotion stimuli only
    emo_stimuli = [s for s in all_stimuli if s["condition"] in AFF]
    emo_indices = [i for i, s in enumerate(all_stimuli) if s["condition"] in AFF]
    print(f"Emotion stimuli: {len(emo_stimuli)} out of {len(all_stimuli)} total")

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

    # Extract features per stimulus
    print("\n--- Extracting per-stimulus features ---")
    features = []
    for s in emo_stimuli:
        tokens = tokenize(s["text"])
        vad_vals = [vad_norms[t] for t in tokens if t in vad_norms]

        f = {
            "condition": s["condition"],
            "n_tokens": len(tokens),
            "valence": np.mean([v["valence"] for v in vad_vals]) if vad_vals else 5.0,
            "arousal": np.mean([v["arousal"] for v in vad_vals]) if vad_vals else 5.0,
            "dominance": np.mean([v["dominance"] for v in vad_vals]) if vad_vals else 5.0,
            "body_prop": prop(tokens, BODY_WORDS),
            "action_prop": prop(tokens, ACTION_WORDS),
            "social_prop": prop(tokens, SOCIAL_WORDS),
            "threat_prop": prop(tokens, THREAT_WORDS),
            "loss_prop": prop(tokens, LOSS_WORDS),
            "moral_prop": prop(tokens, MORAL_WORDS),
            "intensity_prop": prop(tokens, INTENSITY_WORDS),
        }
        features.append(f)

    feature_names = ["valence", "arousal", "dominance", "n_tokens",
                     "body_prop", "action_prop", "social_prop",
                     "threat_prop", "loss_prop", "moral_prop", "intensity_prop"]

    for model in MODELS:
        short = model.split("-")[0] if "Qwen" in model else model[:6]
        peak = PEAK[model]

        print(f"\n{'='*60}")
        print(f"  {model} — peak layer {peak}")
        print(f"{'='*60}")

        # Load activations
        npz = np.load(RSA / f"{model}_rsa_v2_per_stim.npz", allow_pickle=True)
        all_conds = list(npz["conditions"])
        pool_idx = list(npz["pooling_names"]).index("mean_all")
        all_acts = npz["per_stim_activations"][pool_idx, :, peak, :].astype(np.float32)

        # Extract emotion stimuli activations
        emo_acts = all_acts[emo_indices]
        print(f"  Emotion activations: {emo_acts.shape}")

        # Center
        emo_centered = emo_acts - emo_acts.mean(axis=0, keepdims=True)

        # PCA
        pca = PCA(n_components=20)
        pcs = pca.fit_transform(emo_centered)

        print(f"\n  Variance explained:")
        for i in range(10):
            print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]:.2%} "
                  f"(cum: {sum(pca.explained_variance_ratio_[:i+1]):.2%})")

        # Correlate each PC with features
        print(f"\n  PC correlations with stimulus features:")
        print(f"  {'Feature':>15s} | {'PC1':>10s} | {'PC2':>10s} | {'PC3':>10s} | {'PC4':>10s} | {'PC5':>10s}")
        print(f"  {'-'*75}")

        pc_corr_results = {}
        for fn in feature_names:
            vals = np.array([f[fn] for f in features])
            row = f"  {fn:>15s} |"
            corrs = {}
            for pc_i in range(5):
                r, p = spearmanr(pcs[:, pc_i], vals)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                row += f" {r:+.3f}{sig:>3s}  |"
                corrs[f"PC{pc_i+1}"] = {"rho": float(r), "p": float(p)}
            print(row)
            pc_corr_results[fn] = corrs

        # Condition means on each PC (ANOVA: does condition predict PC?)
        print(f"\n  Condition means on top PCs:")
        print(f"  {'Condition':>12s} | {'PC1':>8s} | {'PC2':>8s} | {'PC3':>8s} | {'PC4':>8s} | {'PC5':>8s}")
        print(f"  {'-'*60}")

        cond_groups = {c: [] for c in AFF}
        for i, f in enumerate(features):
            cond_groups[f["condition"]].append(i)

        for c in AFF:
            idxs = cond_groups[c]
            row = f"  {c:>12s} |"
            for pc_i in range(5):
                mean_pc = pcs[idxs, pc_i].mean()
                row += f" {mean_pc:+8.2f} |"
            print(row)

        # ANOVA per PC
        print(f"\n  ANOVA (does condition predict PC?):")
        for pc_i in range(5):
            groups = [pcs[cond_groups[c], pc_i] for c in AFF]
            F, p = f_oneway(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    PC{pc_i+1}: F={F:.2f}, p={p:.4f} {sig}")

        # Key finding: what does PC1 encode?
        print(f"\n  === PC1 INTERPRETATION ===")
        # Find strongest correlations
        pc1_corrs = [(fn, pc_corr_results[fn]["PC1"]["rho"],
                      pc_corr_results[fn]["PC1"]["p"]) for fn in feature_names]
        pc1_corrs.sort(key=lambda x: -abs(x[1]))
        print(f"  Strongest correlates of PC1:")
        for fn, r, p in pc1_corrs[:5]:
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {fn}: rho={r:+.3f} {sig}")

        # Which conditions are high vs low on PC1?
        cond_pc1 = [(c, pcs[cond_groups[c], 0].mean()) for c in AFF]
        cond_pc1.sort(key=lambda x: x[1])
        print(f"  Conditions on PC1 (low→high): {', '.join(f'{c}({m:+.1f})' for c, m in cond_pc1)}")

        # Same for PC2
        print(f"\n  === PC2 INTERPRETATION ===")
        pc2_corrs = [(fn, pc_corr_results[fn]["PC2"]["rho"],
                      pc_corr_results[fn]["PC2"]["p"]) for fn in feature_names]
        pc2_corrs.sort(key=lambda x: -abs(x[1]))
        print(f"  Strongest correlates of PC2:")
        for fn, r, p in pc2_corrs[:5]:
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {fn}: rho={r:+.3f} {sig}")

        cond_pc2 = [(c, pcs[cond_groups[c], 1].mean()) for c in AFF]
        cond_pc2.sort(key=lambda x: x[1])
        print(f"  Conditions on PC2 (low→high): {', '.join(f'{c}({m:+.1f})' for c, m in cond_pc2)}")

        # PC3
        print(f"\n  === PC3 INTERPRETATION ===")
        pc3_corrs = [(fn, pc_corr_results[fn]["PC3"]["rho"],
                      pc_corr_results[fn]["PC3"]["p"]) for fn in feature_names]
        pc3_corrs.sort(key=lambda x: -abs(x[1]))
        print(f"  Strongest correlates of PC3:")
        for fn, r, p in pc3_corrs[:5]:
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {fn}: rho={r:+.3f} {sig}")

        cond_pc3 = [(c, pcs[cond_groups[c], 2].mean()) for c in AFF]
        cond_pc3.sort(key=lambda x: x[1])
        print(f"  Conditions on PC3 (low→high): {', '.join(f'{c}({m:+.1f})' for c, m in cond_pc3)}")

        # ─── Visualization ───
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # PC1 vs PC2 colored by condition
        ax = axes[0]
        for c in AFF:
            idxs = cond_groups[c]
            ax.scatter(pcs[idxs, 0], pcs[idxs, 1], c=C_MAP[c], alpha=0.5, s=20, label=c)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Emotion stimuli: PC1 vs PC2")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.2)

        # PC1 vs PC3
        ax = axes[1]
        for c in AFF:
            idxs = cond_groups[c]
            ax.scatter(pcs[idxs, 0], pcs[idxs, 2], c=C_MAP[c], alpha=0.5, s=20, label=c)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.set_title("Emotion stimuli: PC1 vs PC3")
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.2)

        # Feature correlations heatmap
        ax = axes[2]
        corr_mat = np.zeros((len(feature_names), 5))
        for i, fn in enumerate(feature_names):
            for j in range(5):
                corr_mat[i, j] = pc_corr_results[fn][f"PC{j+1}"]["rho"]
        im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"PC{i+1}" for i in range(5)])
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_title("Feature-PC correlations")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman rho")

        # Mark significant cells
        for i, fn in enumerate(feature_names):
            for j in range(5):
                p = pc_corr_results[fn][f"PC{j+1}"]["p"]
                if p < 0.01:
                    ax.text(j, i, "**", ha="center", va="center", fontsize=10, fontweight="bold")
                elif p < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=10)

        fig.tight_layout()
        fig.savefig(FIG_DIR / "emotion_perstim_pca.png", dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {FIG_DIR / 'emotion_perstim_pca.png'}")

        # Save results
        results = {
            "model": model,
            "n_stimuli": len(emo_stimuli),
            "pca_variance": pca.explained_variance_ratio_[:10].tolist(),
            "pc_feature_correlations": pc_corr_results,
            "condition_pc_means": {
                c: {f"PC{i+1}": float(pcs[cond_groups[c], i].mean()) for i in range(5)}
                for c in AFF
            },
        }
        json.dump(results, open(OUT_DIR / f"emotion_perstim_pca_{model}.json", "w"), indent=2)
        print(f"  Saved: {OUT_DIR / f'emotion_perstim_pca_{model}.json'}")

    print("\nDONE")


if __name__ == "__main__":
    main()
