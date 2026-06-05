#!/usr/bin/env python3
"""
Distributional analysis: WHY does the emotion-social boundary exist in LLMs?

Hypothesis: Language itself encodes the emotion-social distinction through
different distributional patterns. Emotion text uses somatic/experiential
vocabulary; social cognition text uses propositional/inferential vocabulary.
The LLM learns this distributional structure, which mirrors the brain's
organization because the brain's organization shaped how humans use language.

Analyses:
  1. Word category profiling: what linguistic features distinguish the two blocks?
  2. Feature-based RDMs: which features predict brain vs LLM geometry?
  3. Variance decomposition: how much does each feature explain?
  4. Critical comparison: features that explain brain != features that explain LLM
     (embodiment shows up here — brain's within-emotion uses somatic features
      that LLM doesn't have access to)

Output:
  results/mechanistic/distributional_analysis.json
  figures/distributional_decomposition.png
"""
from __future__ import annotations

import json, re, collections
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

AFF = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
MENT = {"judgment", "belief", "intention", "mentalizing", "moral", "empathy",
        "self_referential", "theory_of_mind"}
ORDER = ["anger", "fear", "disgust", "sadness", "happiness", "valence",
         "judgment", "belief", "intention", "mentalizing", "moral", "empathy",
         "self_referential", "theory_of_mind"]

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
MSHORT = {"Qwen2.5-7B-Instruct": "Qwen", "Meta-Llama-3.1-8B-Instruct": "Llama",
          "Mistral-7B-Instruct-v0.3": "Mistral", "gemma-2-9b-it": "Gemma"}

# ═══════════════════════════════════════════════════════════════════════
# Linguistically-grounded word categories
# ═══════════════════════════════════════════════════════════════════════

# Mental state verbs — from mind perception / ToM literature
MENTAL_STATE_VERBS = {
    "think", "thinks", "thinking", "thought",
    "believe", "believes", "believed", "believing",
    "know", "knows", "knew", "knowing", "known",
    "understand", "understands", "understood", "understanding",
    "realize", "realizes", "realized", "realizing",
    "suppose", "supposes", "supposed",
    "imagine", "imagines", "imagined", "imagining",
    "expect", "expects", "expected", "expecting",
    "intend", "intends", "intended", "intending",
    "want", "wants", "wanted", "wanting",
    "desire", "desires", "desired",
    "wish", "wishes", "wished", "wishing",
    "hope", "hopes", "hoped", "hoping",
    "assume", "assumes", "assumed",
    "predict", "predicts", "predicted",
    "suspect", "suspects", "suspected",
    "doubt", "doubts", "doubted",
    "wonder", "wonders", "wondered", "wondering",
    "consider", "considers", "considered",
    "decide", "decides", "decided", "deciding",
    "judge", "judges", "judged", "judging",
    "reason", "reasons", "reasoned",
    "infer", "infers", "inferred",
    "conclude", "concludes", "concluded",
    "recall", "recalls", "recalled",
    "remember", "remembers", "remembered",
    "forget", "forgets", "forgot", "forgotten",
    "notice", "notices", "noticed",
    "recognize", "recognizes", "recognized",
    "perceive", "perceives", "perceived",
    "sense", "senses", "sensed",
    "plan", "plans", "planned", "planning",
    "pretend", "pretends", "pretended",
    "convince", "convinces", "convinced",
    "persuade", "persuades", "persuaded",
    "mislead", "misleads", "misled",
    "lie", "lied", "lying",
    "trick", "tricks", "tricked",
    "deceive", "deceives", "deceived",
}

# Body / sensory / physiological words
BODY_SENSORY = {
    "body", "heart", "stomach", "hand", "hands", "face", "faces",
    "eye", "eyes", "skin", "blood", "breath", "breathing",
    "chest", "throat", "muscle", "muscles", "nerve", "nerves",
    "head", "brain", "spine", "gut", "pulse", "vein", "veins",
    "tremble", "trembles", "trembling", "trembled",
    "shake", "shakes", "shaking", "shook", "shaken",
    "cry", "cries", "crying", "cried",
    "scream", "screams", "screaming", "screamed",
    "sweat", "sweats", "sweating", "sweated",
    "shiver", "shivers", "shivering", "shivered",
    "flinch", "flinches", "flinching", "flinched",
    "gasp", "gasps", "gasping", "gasped",
    "sob", "sobs", "sobbing", "sobbed",
    "wince", "winces", "wincing", "winced",
    "blush", "blushes", "blushing", "blushed",
    "pale", "pallid", "flush", "flushed",
    "nausea", "nauseous", "vomit", "vomiting", "retch",
    "pain", "painful", "ache", "aches", "aching",
    "hurt", "hurts", "hurting",
    "touch", "touches", "touching", "touched",
    "feel", "feels", "feeling", "felt",
    "smell", "smells", "smelling", "smelled",
    "taste", "tastes", "tasting", "tasted",
    "hear", "hears", "hearing", "heard",
    "see", "sees", "seeing", "saw", "seen",
    "warm", "warmth", "cold", "chill", "chills",
    "tense", "tension", "tight", "tightness",
    "dizzy", "dizziness", "faint", "fainting",
    "exhausted", "exhaustion", "fatigue", "tired",
    "hungry", "thirsty", "sleepy",
    "physical", "physically", "bodily", "somatic",
}

# Emotion words (adjectives, nouns, adverbs — the affective lexicon)
EMOTION_WORDS = {
    "angry", "anger", "furious", "fury", "rage", "enraged", "irate",
    "mad", "outraged", "irritated", "annoyed", "hostile", "resentful",
    "afraid", "fear", "fearful", "scared", "terrified", "terror",
    "frightened", "anxious", "anxiety", "nervous", "panicked", "panic",
    "dread", "dreading", "alarmed", "horrified", "horror",
    "sad", "sadness", "sorrow", "sorrowful", "grief", "grieving",
    "depressed", "depression", "melancholy", "mournful", "mourning",
    "unhappy", "miserable", "gloomy", "heartbroken",
    "happy", "happiness", "joy", "joyful", "joyous", "delight",
    "delighted", "pleased", "cheerful", "ecstatic", "elated",
    "glad", "grateful", "content", "satisfied", "bliss", "blissful",
    "disgusted", "disgust", "revolted", "repulsed", "repulsion",
    "sickened", "appalled", "loathing", "contempt", "contemptuous",
    "emotional", "emotions", "emotion", "emotionally",
    "love", "loved", "loving", "hate", "hated", "hatred", "hating",
    "jealous", "jealousy", "envious", "envy",
    "ashamed", "shame", "guilty", "guilt",
    "proud", "pride", "embarrassed", "embarrassment",
    "surprised", "surprise", "shocked", "shock", "astonished",
    "frustrated", "frustration", "disappointed", "disappointment",
    "hopeful", "hopeless", "desperate", "despair",
    "lonely", "loneliness", "isolated",
    "calm", "peaceful", "serene", "tranquil", "relaxed",
    "excited", "excitement", "thrilled", "enthusiastic",
}

# Proposition / inference markers
PROPOSITION_MARKERS = {
    "that", "whether", "because", "therefore", "however", "although",
    "moreover", "furthermore", "nevertheless", "thus", "hence",
    "since", "unless", "whereas", "while", "if", "then",
    "consequently", "accordingly", "alternatively",
    "true", "false", "truth", "right", "wrong", "correct", "incorrect",
    "fact", "facts", "evidence", "proof", "reason", "reasons",
    "argument", "claim", "claims", "hypothesis", "conclusion",
    "imply", "implies", "implication", "implication",
    "suggest", "suggests", "indicate", "indicates",
    "demonstrate", "demonstrates", "prove", "proves",
    "contradict", "contradicts", "contradiction",
    "consistent", "inconsistent", "compatible", "incompatible",
    "logical", "illogical", "rational", "irrational",
    "valid", "invalid", "reasonable", "unreasonable",
    "actually", "really", "indeed", "certainly", "probably",
    "perhaps", "possibly", "likely", "unlikely",
    "obviously", "clearly", "apparently", "seemingly",
    "supposedly", "allegedly", "presumably",
}

# Social / interpersonal words
SOCIAL_WORDS = {
    "person", "people", "someone", "somebody", "anyone", "everyone",
    "friend", "friends", "family", "neighbor", "neighbors",
    "colleague", "colleagues", "stranger", "strangers",
    "community", "society", "social", "socially",
    "relationship", "relationships", "interaction", "interactions",
    "conversation", "conversations", "communication",
    "trust", "trusts", "trusted", "trusting",
    "betray", "betrays", "betrayed", "betrayal",
    "cooperate", "cooperates", "cooperation",
    "help", "helps", "helped", "helping",
    "share", "shares", "shared", "sharing",
    "care", "cares", "caring", "cared",
    "support", "supports", "supporting", "supported",
    "respect", "respects", "respected", "disrespect",
    "fairness", "fair", "unfair", "justice", "injustice", "just", "unjust",
    "norm", "norms", "rule", "rules", "obligation", "obligations",
    "duty", "duties", "responsibility", "responsible",
    "moral", "morals", "morality", "immoral", "ethical", "unethical",
    "virtue", "virtuous", "vice", "sin",
    "permission", "forbidden", "allowed", "prohibited",
    "punish", "punishment", "reward", "forgive", "forgiveness",
}


def tokenize_simple(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def category_proportion(tokens, category_set):
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in category_set) / len(tokens)


def triu_vec(mat):
    iu = np.triu_indices(mat.shape[0], 1)
    return mat[iu]


def perm_p(x, y, observed_rho, n_perm=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(x))
        r, _ = spearmanr(x[perm], y)
        if abs(r) >= abs(observed_rho):
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    print("=" * 70)
    print("DISTRIBUTIONAL ANALYSIS: Why does the emotion-social boundary exist?")
    print("=" * 70)

    # Load stimuli
    stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    stimuli = [json.loads(l) for l in open(stim_path)]
    print(f"\nLoaded {len(stimuli)} stimuli")

    # ─── Step 1: Compute linguistic features per stimulus ───
    print("\n--- Step 1: Linguistic feature profiling ---")

    categories = {
        "mental_state_verbs": MENTAL_STATE_VERBS,
        "body_sensory": BODY_SENSORY,
        "emotion_words": EMOTION_WORDS,
        "proposition_markers": PROPOSITION_MARKERS,
        "social_words": SOCIAL_WORDS,
    }

    per_stim_features = []
    for s in stimuli:
        tokens = tokenize_simple(s["text"])
        feats = {
            "condition": s["condition"],
            "n_tokens": len(tokens),
        }
        for cat_name, cat_set in categories.items():
            feats[cat_name] = category_proportion(tokens, cat_set)
        per_stim_features.append(feats)

    # Aggregate per condition
    cond_features = {}
    for cond in ORDER:
        stims = [f for f in per_stim_features if f["condition"] == cond]
        n = len(stims)
        feats = {"n_stimuli": n, "n_tokens_mean": np.mean([s["n_tokens"] for s in stims])}
        for cat_name in categories:
            vals = [s[cat_name] for s in stims]
            feats[cat_name] = float(np.mean(vals))
            feats[cat_name + "_std"] = float(np.std(vals))
        cond_features[cond] = feats

    # Print feature profiles
    print(f"\n{'Condition':>20s} | {'Mental':>6s} | {'Body':>6s} | {'Emotion':>7s} | {'Propos':>6s} | {'Social':>6s} | {'Block':>5s}")
    print("-" * 80)
    for cond in ORDER:
        f = cond_features[cond]
        blk = "AFF" if cond in AFF else "SOC"
        print(f"{cond:>20s} | {f['mental_state_verbs']:6.3f} | {f['body_sensory']:6.3f} | "
              f"{f['emotion_words']:7.3f} | {f['proposition_markers']:6.3f} | "
              f"{f['social_words']:6.3f} | {blk:>5s}")

    # Block-level comparison
    print("\n--- Block-level comparison ---")
    for cat_name in categories:
        aff_vals = [cond_features[c][cat_name] for c in ORDER if c in AFF]
        soc_vals = [cond_features[c][cat_name] for c in ORDER if c in MENT]
        aff_mean = np.mean(aff_vals)
        soc_mean = np.mean(soc_vals)
        ratio = aff_mean / soc_mean if soc_mean > 0 else float("inf")
        dominant = "AFF" if aff_mean > soc_mean else "SOC"
        print(f"  {cat_name:>25s}: AFF={aff_mean:.4f}, SOC={soc_mean:.4f}, "
              f"ratio={ratio:.2f}x, dominant={dominant}")

    # ─── Step 2: Build feature-based RDMs ───
    print("\n--- Step 2: Feature-based RDMs ---")

    # Build feature matrix: conditions x features
    feat_names = list(categories.keys())
    feat_matrix = np.zeros((len(ORDER), len(feat_names)))
    for i, cond in enumerate(ORDER):
        for j, fn in enumerate(feat_names):
            feat_matrix[i, j] = cond_features[cond][fn]

    # Individual feature RDMs
    feature_rdms = {}
    for j, fn in enumerate(feat_names):
        vec = feat_matrix[:, j].reshape(-1, 1)
        rdm = squareform(pdist(vec, metric="euclidean"))
        feature_rdms[fn] = rdm

    # Combined feature RDM (all features together, standardized)
    from sklearn.preprocessing import StandardScaler
    feat_std = StandardScaler().fit_transform(feat_matrix)
    combined_rdm = squareform(pdist(feat_std, metric="euclidean"))
    feature_rdms["combined"] = combined_rdm

    # Binary block RDM (for reference)
    block_vec = np.array([0 if c in AFF else 1 for c in ORDER]).reshape(-1, 1)
    block_rdm = squareform(pdist(block_vec, metric="euclidean"))
    feature_rdms["binary_block"] = block_rdm

    # ─── Step 3: Compare with brain and LLM RDMs ───
    print("\n--- Step 3: Feature RDMs vs brain & LLM ---")

    # Load brain RDM
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_rdm_raw, brain_conds = ns["rdm"], list(ns["conditions"])
    brain_idx = [brain_conds.index(c) for c in ORDER]
    brain_rdm = brain_rdm_raw[np.ix_(brain_idx, brain_idx)]
    brain_vec = triu_vec(brain_rdm)

    # Load LLM RDMs
    llm_vecs = {}
    for m in MODELS:
        z = np.load(RSA / f"{m}_rdm14_headline.npz", allow_pickle=True)
        lrdm, lconds = z["rdm"], list(z["conditions"])
        lidx = [lconds.index(c) for c in ORDER]
        L = lrdm[np.ix_(lidx, lidx)]
        llm_vecs[MSHORT[m]] = triu_vec(L)

    llm_avg_vec = np.mean(list(llm_vecs.values()), axis=0)

    rng = np.random.default_rng(42)

    print(f"\n{'Feature RDM':>25s} | {'vs Brain':>10s} | {'vs LLM(avg)':>12s} | {'vs Block':>10s}")
    print("-" * 70)

    results_comparison = {}
    for fn, rdm in feature_rdms.items():
        fvec = triu_vec(rdm)
        rho_brain, p_brain = spearmanr(fvec, brain_vec)
        rho_llm, p_llm = spearmanr(fvec, llm_avg_vec)
        rho_block, _ = spearmanr(fvec, triu_vec(block_rdm))

        sig_brain = "***" if p_brain < 0.001 else "**" if p_brain < 0.01 else "*" if p_brain < 0.05 else ""
        sig_llm = "***" if p_llm < 0.001 else "**" if p_llm < 0.01 else "*" if p_llm < 0.05 else ""

        print(f"{fn:>25s} | {rho_brain:+.3f}{sig_brain:>4s} | {rho_llm:+.3f}{sig_llm:>4s} | {rho_block:+.3f}")

        results_comparison[fn] = {
            "rho_brain": float(rho_brain), "p_brain": float(p_brain),
            "rho_llm": float(rho_llm), "p_llm": float(p_llm),
            "rho_block": float(rho_block),
        }

    # ─── Step 4: Partial RSA — what survives after controlling features? ───
    print("\n--- Step 4: Partial RSA (control linguistic features) ---")

    # Regress out each feature from brain and LLM RDMs
    from numpy.linalg import lstsq

    def partial_rsa(target_vec, control_vecs):
        """Residualize target and brain_vec against controls, then correlate."""
        X = np.column_stack(control_vecs)
        X = np.column_stack([X, np.ones(len(X))])

        beta_t, _, _, _ = lstsq(X, target_vec, rcond=None)
        resid_t = target_vec - X @ beta_t

        beta_b, _, _, _ = lstsq(X, brain_vec, rcond=None)
        resid_b = brain_vec - X @ beta_b

        rho, p = spearmanr(resid_t, resid_b)
        return float(rho), float(p)

    control_vecs = [triu_vec(feature_rdms[fn]) for fn in feat_names]

    print(f"\n{'Model':>10s} | {'Raw rho':>8s} | {'Partial rho':>11s} | {'Retained':>8s}")
    print("-" * 50)

    partial_results = {}
    for mname, mvec in list(llm_vecs.items()) + [("LLM_avg", llm_avg_vec)]:
        raw_rho = float(spearmanr(mvec, brain_vec)[0])
        prho, pp = partial_rsa(mvec, control_vecs)
        pct = prho / raw_rho * 100 if raw_rho > 0 else 0
        print(f"{mname:>10s} | {raw_rho:+.3f}   | {prho:+.3f} (p={pp:.4f}) | {pct:.0f}%")
        partial_results[mname] = {
            "raw_rho": raw_rho, "partial_rho": prho, "partial_p": pp, "pct_retained": pct,
        }

    # ─── Step 5: Within-block feature profiles ───
    print("\n--- Step 5: Within-block analysis ---")
    print("Do different features predict WITHIN-affective vs WITHIN-social structure?")

    aff_idx = [ORDER.index(c) for c in ORDER if c in AFF]
    soc_idx = [ORDER.index(c) for c in ORDER if c in MENT]

    def within_block_rsa(rdm, ref_rdm, idx):
        sub = rdm[np.ix_(idx, idx)]
        ref_sub = ref_rdm[np.ix_(idx, idx)]
        sv = triu_vec(sub)
        rv = triu_vec(ref_sub)
        if np.std(sv) == 0 or np.std(rv) == 0:
            return 0.0, 1.0
        return spearmanr(sv, rv)

    print(f"\n{'Feature':>25s} | {'W-Aff vs Brain':>14s} | {'W-Soc vs Brain':>14s} | {'W-Aff vs LLM':>13s} | {'W-Soc vs LLM':>13s}")
    print("-" * 95)

    llm_avg_rdm = squareform(pdist(np.column_stack(list(llm_vecs.values())).mean(axis=1).reshape(-1, 1)))
    # Actually, need to reconstruct LLM avg RDM properly
    llm_rdms_ordered = {}
    for m in MODELS:
        z = np.load(RSA / f"{m}_rdm14_headline.npz", allow_pickle=True)
        lrdm, lconds = z["rdm"], list(z["conditions"])
        lidx = [lconds.index(c) for c in ORDER]
        llm_rdms_ordered[MSHORT[m]] = lrdm[np.ix_(lidx, lidx)]

    avg_llm_rdm = np.mean(list(llm_rdms_ordered.values()), axis=0)

    within_results = {}
    for fn, rdm in feature_rdms.items():
        if fn == "binary_block":
            continue
        r_aff_brain, p_aff_brain = within_block_rsa(rdm, brain_rdm, aff_idx)
        r_soc_brain, p_soc_brain = within_block_rsa(rdm, brain_rdm, soc_idx)
        r_aff_llm, p_aff_llm = within_block_rsa(rdm, avg_llm_rdm, aff_idx)
        r_soc_llm, p_soc_llm = within_block_rsa(rdm, avg_llm_rdm, soc_idx)

        sig = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{fn:>25s} | {r_aff_brain:+.3f}{sig(p_aff_brain):>4s}     | {r_soc_brain:+.3f}{sig(p_soc_brain):>4s}     | "
              f"{r_aff_llm:+.3f}{sig(p_aff_llm):>4s}    | {r_soc_llm:+.3f}{sig(p_soc_llm):>4s}")

        within_results[fn] = {
            "within_aff_brain": float(r_aff_brain), "p_aff_brain": float(p_aff_brain),
            "within_soc_brain": float(r_soc_brain), "p_soc_brain": float(p_soc_brain),
            "within_aff_llm": float(r_aff_llm), "p_aff_llm": float(p_aff_llm),
            "within_soc_llm": float(r_soc_llm), "p_soc_llm": float(p_soc_llm),
        }

    # ─── Step 6: GloVe distributional RDM ───
    print("\n--- Step 6: GloVe distributional RDM ---")

    glove_path = Path("/hpc2hdd/home/mzhang630/data/glove/glove.6B.300d.txt")
    if glove_path.exists():
        print("Loading GloVe 300d...")
        glove = {}
        with open(glove_path) as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) == 301:
                    glove[parts[0]] = np.array([float(x) for x in parts[1:]])
        print(f"  Loaded {len(glove)} vectors")

        # Per-condition GloVe centroid (average GloVe of all content words in stimuli)
        cond_glove = {}
        for cond in ORDER:
            stim_texts = [s["text"] for s in stimuli if s["condition"] == cond]
            vecs = []
            for text in stim_texts:
                for tok in tokenize_simple(text):
                    if tok in glove:
                        vecs.append(glove[tok])
            if vecs:
                cond_glove[cond] = np.mean(vecs, axis=0)

        if len(cond_glove) == 14:
            glove_mat = np.stack([cond_glove[c] for c in ORDER])
            glove_rdm = squareform(pdist(glove_mat, metric="cosine"))
            glove_vec = triu_vec(glove_rdm)

            rho_brain, _ = spearmanr(glove_vec, brain_vec)
            rho_llm, _ = spearmanr(glove_vec, llm_avg_vec)
            print(f"  GloVe centroid RDM vs brain: rho = {rho_brain:+.3f}")
            print(f"  GloVe centroid RDM vs LLM:   rho = {rho_llm:+.3f}")

            # GloVe within-block
            r_aff, _ = within_block_rsa(glove_rdm, brain_rdm, aff_idx)
            r_soc, _ = within_block_rsa(glove_rdm, brain_rdm, soc_idx)
            print(f"  GloVe within-aff vs brain: {r_aff:+.3f}")
            print(f"  GloVe within-soc vs brain: {r_soc:+.3f}")

            r_aff_llm, _ = within_block_rsa(glove_rdm, avg_llm_rdm, aff_idx)
            r_soc_llm, _ = within_block_rsa(glove_rdm, avg_llm_rdm, soc_idx)
            print(f"  GloVe within-aff vs LLM:   {r_aff_llm:+.3f}")
            print(f"  GloVe within-soc vs LLM:   {r_soc_llm:+.3f}")

            results_comparison["glove_centroid"] = {
                "rho_brain": float(rho_brain), "rho_llm": float(rho_llm),
                "within_aff_brain": float(r_aff), "within_soc_brain": float(r_soc),
                "within_aff_llm": float(r_aff_llm), "within_soc_llm": float(r_soc_llm),
            }
    else:
        print(f"  GloVe not found at {glove_path}, skipping")

    # ─── Step 7: Visualization ───
    print("\n--- Step 7: Generating figures ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 7a: Feature profiles by condition
    ax = axes[0, 0]
    x = np.arange(len(ORDER))
    width = 0.15
    colors = {"mental_state_verbs": "#19a979", "body_sensory": "#e8743b",
              "emotion_words": "#cc3333", "proposition_markers": "#2e5cb8",
              "social_words": "#8855aa"}
    for j, fn in enumerate(feat_names):
        vals = [cond_features[c][fn] for c in ORDER]
        ax.bar(x + j * width - 2 * width, vals, width, label=fn.replace("_", " "),
               color=colors.get(fn, "#888"))
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=90, fontsize=7)
    ax.axvline(5.5, color="cyan", lw=2, ls="--")
    ax.set_ylabel("Proportion")
    ax.set_title("Linguistic feature profiles by condition")
    ax.legend(fontsize=7, loc="upper right")

    # 7b: Feature RDMs vs brain/LLM
    ax = axes[0, 1]
    fn_labels = list(results_comparison.keys())
    fn_labels = [fn for fn in fn_labels if fn not in ("binary_block", "glove_centroid")]
    rho_b = [results_comparison[fn]["rho_brain"] for fn in fn_labels]
    rho_l = [results_comparison[fn]["rho_llm"] for fn in fn_labels]
    x = np.arange(len(fn_labels))
    ax.bar(x - 0.18, rho_b, 0.35, label="vs Brain", color="#e8743b", alpha=0.8)
    ax.bar(x + 0.18, rho_l, 0.35, label="vs LLM", color="#19a979", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([fn.replace("_", "\n") for fn in fn_labels], fontsize=7)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Which features predict brain vs LLM geometry?")
    ax.legend(fontsize=8)
    ax.axhline(0, color="#333", lw=0.5)
    ax.grid(axis="y", alpha=0.2)

    # 7c: Block-level feature comparison
    ax = axes[1, 0]
    aff_means = [np.mean([cond_features[c][fn] for c in ORDER if c in AFF]) for fn in feat_names]
    soc_means = [np.mean([cond_features[c][fn] for c in ORDER if c in MENT]) for fn in feat_names]
    x = np.arange(len(feat_names))
    ax.bar(x - 0.18, aff_means, 0.35, label="Affective", color="#e8743b")
    ax.bar(x + 0.18, soc_means, 0.35, label="Social cog", color="#19a979")
    ax.set_xticks(x)
    ax.set_xticklabels([fn.replace("_", "\n") for fn in feat_names], fontsize=7)
    ax.set_ylabel("Mean proportion")
    ax.set_title("Block-level linguistic feature comparison")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    # 7d: Partial RSA results
    ax = axes[1, 1]
    model_names = list(partial_results.keys())
    raw_vals = [partial_results[m]["raw_rho"] for m in model_names]
    part_vals = [partial_results[m]["partial_rho"] for m in model_names]
    x = np.arange(len(model_names))
    ax.bar(x - 0.18, raw_vals, 0.35, label="Raw rho", color="#2e5cb8", alpha=0.3)
    ax.bar(x + 0.18, part_vals, 0.35, label="After controlling\nall ling. features", color="#2e5cb8")
    for i in range(len(model_names)):
        pct = partial_results[model_names[i]]["pct_retained"]
        ax.text(i + 0.18, part_vals[i] + 0.01, f"{pct:.0f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Partial RSA: rho after controlling linguistic features")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "distributional_decomposition.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {FIG_DIR / 'distributional_decomposition.png'}")
    plt.close(fig)

    # ─── Save results ───
    results = {
        "condition_features": {c: {k: v for k, v in cond_features[c].items()
                                   if not k.endswith("_std")} for c in ORDER},
        "feature_vs_rdm": results_comparison,
        "partial_rsa_controlling_features": partial_results,
        "within_block_feature_analysis": within_results,
    }
    json.dump(results, open(OUT_DIR / "distributional_analysis.json", "w"), indent=2)
    print(f"\nSaved: {OUT_DIR / 'distributional_analysis.json'}")
    print("\nDONE")


if __name__ == "__main__":
    main()
