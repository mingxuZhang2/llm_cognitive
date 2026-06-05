#!/usr/bin/env python3
"""
Prediction mode analysis: WHAT does the emotion-social boundary mean for LLMs?

For each condition's stimuli, analyze the LLM's next-token predictions:
  1. Top-k predicted tokens — what TYPES of continuations does the model predict?
  2. Vocabulary category breakdown — emotion stimuli predict X, social stimuli predict Y
  3. Entropy/uncertainty patterns — is one block "harder" to predict than the other?
  4. Prediction overlap — do conditions within a block predict similar next tokens?

This answers: the boundary exists because the model uses two different PREDICTION
STRATEGIES for the two types of text.

Usage:
  # GPU needed (SLURM: i64m1tga800u)
  python src/prediction_mode_analysis.py --model_path /path/to/Qwen2.5-7B-Instruct

Output:
  results/mechanistic/prediction_mode_{model}.json
"""
from __future__ import annotations

import argparse, json, collections
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "mechanistic"
OUT_DIR.mkdir(exist_ok=True)

AFF = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
MENT = {"judgment", "belief", "intention", "mentalizing", "moral", "empathy",
        "self_referential", "theory_of_mind"}

# Vocabulary categories for classifying predicted tokens
VOCAB_CATEGORIES = {
    "emotion": {"angry", "sad", "happy", "afraid", "fear", "love", "hate", "joy",
                "pain", "grief", "rage", "fury", "delight", "horror", "anxiety",
                "worried", "excited", "disappointed", "frustrated", "grateful",
                "lonely", "jealous", "proud", "ashamed", "guilty", "disgusted",
                "terrified", "depressed", "cheerful", "nervous", "calm"},
    "mental_verb": {"think", "believe", "know", "understand", "realize", "imagine",
                    "expect", "want", "hope", "wish", "decide", "remember", "forget",
                    "notice", "wonder", "suppose", "doubt", "consider", "assume",
                    "suspect", "predict", "recognize", "sense", "plan", "pretend",
                    "thought", "believed", "knew", "understood", "realized"},
    "body_sensory": {"body", "heart", "hand", "face", "eye", "eyes", "head",
                     "breath", "blood", "skin", "stomach", "chest", "throat",
                     "tears", "cry", "scream", "tremble", "shake", "sweat",
                     "pain", "hurt", "touch", "feel", "warm", "cold"},
    "social": {"person", "people", "friend", "family", "child", "children",
               "woman", "man", "someone", "everyone", "together", "community",
               "relationship", "trust", "help", "care", "support"},
    "causal_logical": {"because", "therefore", "however", "although", "since",
                       "thus", "hence", "consequently", "moreover", "furthermore",
                       "nevertheless", "regardless", "instead", "otherwise",
                       "actually", "indeed", "certainly", "probably", "perhaps"},
    "moral_norm": {"right", "wrong", "should", "must", "fair", "unfair",
                   "moral", "ethical", "justice", "duty", "responsibility",
                   "punishment", "forgive", "blame", "guilty", "innocent",
                   "allowed", "forbidden", "deserve"},
    "action": {"did", "does", "made", "took", "went", "came", "got", "put",
               "gave", "told", "asked", "said", "looked", "turned", "walked",
               "started", "tried", "decided", "stopped", "continued"},
    "pronoun": {"he", "she", "they", "it", "his", "her", "their", "him",
                "them", "himself", "herself", "themselves"},
    "punctuation": {".", ",", "!", "?", ";", ":", "-", "\"", "'"},
}


def classify_token(token_str):
    t = token_str.strip().lower()
    for cat, words in VOCAB_CATEGORIES.items():
        if t in words:
            return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Prediction Mode Analysis ({model_short}) ===")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load stimuli
    stim_path = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
    stimuli = [json.loads(l) for l in open(stim_path)]
    print(f"Loaded {len(stimuli)} stimuli")

    # Per-stimulus analysis
    top_k = args.top_k
    per_condition = collections.defaultdict(list)

    for si, s in enumerate(stimuli):
        cond = s["condition"]
        text = s["text"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # last token logits

        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()

        topk_probs, topk_ids = torch.topk(probs, top_k)
        topk_tokens = [tokenizer.decode(tid.item()).strip() for tid in topk_ids]
        topk_probs = topk_probs.cpu().numpy().tolist()

        # Classify each top-k token
        categories = [classify_token(t) for t in topk_tokens]
        cat_probs = collections.defaultdict(float)
        for cat, prob in zip(categories, topk_probs):
            cat_probs[cat] += prob

        per_condition[cond].append({
            "entropy": entropy,
            "top5_tokens": topk_tokens[:5],
            "top5_probs": topk_probs[:5],
            "category_probs": dict(cat_probs),
        })

        if (si + 1) % 50 == 0:
            print(f"  [{si+1}/{len(stimuli)}]")

    # Aggregate per condition
    print("\n--- Per-condition summary ---")
    print(f"{'Condition':>20s} | {'Entropy':>7s} | {'Emotion':>7s} | {'Mental':>7s} | "
          f"{'Body':>7s} | {'Causal':>7s} | {'Moral':>7s} | {'Block':>5s}")
    print("-" * 90)

    condition_summary = {}
    for cond in sorted(per_condition.keys()):
        stims = per_condition[cond]
        avg_entropy = np.mean([s["entropy"] for s in stims])

        avg_cats = collections.defaultdict(float)
        for s in stims:
            for cat, prob in s["category_probs"].items():
                avg_cats[cat] += prob / len(stims)

        blk = "AFF" if cond in AFF else "SOC"
        print(f"{cond:>20s} | {avg_entropy:7.3f} | {avg_cats['emotion']:7.4f} | "
              f"{avg_cats['mental_verb']:7.4f} | {avg_cats['body_sensory']:7.4f} | "
              f"{avg_cats['causal_logical']:7.4f} | {avg_cats['moral_norm']:7.4f} | {blk:>5s}")

        condition_summary[cond] = {
            "n_stimuli": len(stims),
            "mean_entropy": float(avg_entropy),
            "mean_category_probs": {k: float(v) for k, v in avg_cats.items()},
            "top_tokens": collections.Counter(
                t for s in stims for t in s["top5_tokens"]
            ).most_common(20),
        }

    # Block-level comparison
    print("\n--- Block-level prediction mode comparison ---")
    for cat_name in VOCAB_CATEGORIES:
        aff_vals = [condition_summary[c]["mean_category_probs"].get(cat_name, 0)
                    for c in condition_summary if c in AFF]
        soc_vals = [condition_summary[c]["mean_category_probs"].get(cat_name, 0)
                    for c in condition_summary if c in MENT]
        aff_mean = np.mean(aff_vals) if aff_vals else 0
        soc_mean = np.mean(soc_vals) if soc_vals else 0
        dominant = "AFF" if aff_mean > soc_mean else "SOC" if soc_mean > aff_mean else "="
        ratio = aff_mean / soc_mean if soc_mean > 1e-8 else float("inf")
        print(f"  {cat_name:>15s}: AFF={aff_mean:.5f} SOC={soc_mean:.5f} "
              f"ratio={ratio:.2f}x dominant={dominant}")

    # Entropy comparison
    aff_ent = np.mean([condition_summary[c]["mean_entropy"]
                       for c in condition_summary if c in AFF])
    soc_ent = np.mean([condition_summary[c]["mean_entropy"]
                       for c in condition_summary if c in MENT])
    print(f"\n  Mean entropy: AFF={aff_ent:.3f} SOC={soc_ent:.3f} "
          f"({'AFF higher' if aff_ent > soc_ent else 'SOC higher'})")
    print(f"  Interpretation: {'Emotion text is harder to predict' if aff_ent > soc_ent else 'Social text is harder to predict'}")

    # Top tokens per block
    print("\n--- Most predicted tokens per block ---")
    for block_name, block_set in [("AFFECTIVE", AFF), ("SOCIAL", MENT)]:
        all_tops = collections.Counter()
        for c in condition_summary:
            if c in block_set:
                for tok, cnt in condition_summary[c]["top_tokens"]:
                    all_tops[tok] += cnt
        top20 = all_tops.most_common(20)
        print(f"  {block_name}: {', '.join(f'{t}({n})' for t, n in top20)}")

    # Save
    out = {
        "model": model_short,
        "top_k": top_k,
        "n_stimuli": len(stimuli),
        "per_condition": condition_summary,
        "block_comparison": {
            "aff_mean_entropy": float(aff_ent),
            "soc_mean_entropy": float(soc_ent),
        },
    }
    out_path = OUT_DIR / f"prediction_mode_{model_short}.json"
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
