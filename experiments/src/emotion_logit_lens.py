#!/usr/bin/env python3
"""
Emotion logit lens: what does the model INTERNALLY REPRESENT for each emotion?

Project each emotion condition's centroid hidden state through the LM head
to vocabulary space → see what words/concepts are most associated with each
emotion's internal representation.

Then: for each emotion PAIR, compute the difference vector and project it
→ see what DISTINGUISHES two emotions in the model's representation.

This is genuine mechanistic interpretability: not what the model OUTPUTS,
but what it REPRESENTS internally.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
OUT_DIR.mkdir(exist_ok=True)

AFF = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
ORDER = AFF + ["judgment", "belief", "intention", "mentalizing", "moral",
               "empathy", "self_referential", "theory_of_mind"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Emotion Logit Lens ({model_short}) ===")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    # Get the LM head (unembedding matrix)
    lm_head = model.lm_head.weight.float().cpu()  # (vocab_size, hidden_dim)
    print(f"  LM head shape: {lm_head.shape}")

    # Load per-stim activations
    peak_layer = 27  # Qwen peak
    npz = np.load(RSA / f"{model_short}_rsa_v2_per_stim.npz", allow_pickle=True)
    conditions = list(npz["conditions"])
    pool_idx = list(npz["pooling_names"]).index("mean_all")
    acts = npz["per_stim_activations"][pool_idx, :, peak_layer, :].astype(np.float32)
    print(f"  Activations: {acts.shape}")

    # Compute condition centroids
    centroids = {}
    for c in AFF:
        mask = [i for i, cc in enumerate(conditions) if cc == c]
        centroids[c] = acts[mask].mean(axis=0)

    # Also compute global emotion mean for centering
    all_emo_mask = [i for i, cc in enumerate(conditions) if cc in AFF]
    emo_mean = acts[all_emo_mask].mean(axis=0)

    # Project each centroid through LM head → top vocabulary items
    print(f"\n{'='*70}")
    print("WHAT DOES THE MODEL REPRESENT FOR EACH EMOTION?")
    print(f"{'='*70}")

    results = {"per_emotion": {}, "pair_differences": {}}

    for c in AFF:
        vec = torch.tensor(centroids[c])
        # Project to vocab space
        logits = lm_head @ vec  # (vocab_size,)
        probs = torch.softmax(logits, dim=-1)

        # Top tokens
        topk = 30
        top_probs, top_ids = torch.topk(probs, topk)
        top_tokens = [tokenizer.decode(tid.item()).strip() for tid in top_ids]
        top_probs = top_probs.tolist()

        print(f"\n  === {c.upper()} ===")
        print(f"  Top 20 tokens: {', '.join(f'{t}({p:.3f})' for t, p in zip(top_tokens[:20], top_probs[:20]))}")

        results["per_emotion"][c] = {
            "top_tokens": [(t, float(p)) for t, p in zip(top_tokens, top_probs)],
        }

    # Project CENTERED centroids (subtract emotion mean)
    print(f"\n{'='*70}")
    print("CENTERED: What makes each emotion UNIQUE (relative to emotion average)?")
    print(f"{'='*70}")

    for c in AFF:
        vec = torch.tensor(centroids[c] - emo_mean)
        logits = lm_head @ vec
        # Don't softmax — just look at highest/lowest logits (what's promoted/suppressed)
        top_vals, top_ids = torch.topk(logits, 20)
        bot_vals, bot_ids = torch.topk(-logits, 20)

        top_tokens = [tokenizer.decode(tid.item()).strip() for tid in top_ids]
        bot_tokens = [tokenizer.decode(tid.item()).strip() for tid in bot_ids]

        print(f"\n  === {c.upper()} (centered) ===")
        print(f"  PROMOTED (unique to {c}): {', '.join(top_tokens[:15])}")
        print(f"  SUPPRESSED (unlike {c}): {', '.join(bot_tokens[:15])}")

        results["per_emotion"][c]["centered_promoted"] = top_tokens[:20]
        results["per_emotion"][c]["centered_suppressed"] = bot_tokens[:20]

    # Pair differences: what distinguishes emotion A from emotion B?
    print(f"\n{'='*70}")
    print("PAIR DIFFERENCES: What distinguishes one emotion from another?")
    print(f"{'='*70}")

    key_pairs = [
        ("anger", "sadness", "Brain: FAR (different arousal). LLM: CLOSE"),
        ("anger", "disgust", "Brain: FAR (different autonomic). LLM: CLOSE"),
        ("anger", "fear", "Brain: CLOSE (both high arousal). LLM: MEDIUM"),
        ("fear", "disgust", "Brain: CLOSE (both aversive). LLM: MEDIUM-FAR"),
        ("fear", "sadness", "Brain: MEDIUM. LLM: CLOSE"),
    ]

    for e1, e2, note in key_pairs:
        diff = torch.tensor(centroids[e1] - centroids[e2])
        diff_norm = diff.norm().item()

        logits = lm_head @ diff
        top_vals, top_ids = torch.topk(logits, 15)
        bot_vals, bot_ids = torch.topk(-logits, 15)

        top_tokens = [tokenizer.decode(tid.item()).strip() for tid in top_ids]
        bot_tokens = [tokenizer.decode(tid.item()).strip() for tid in bot_ids]

        print(f"\n  === {e1.upper()} → {e2.upper()} ({note}) ===")
        print(f"  Distance: {diff_norm:.4f}")
        print(f"  {e1.upper()} side: {', '.join(top_tokens[:12])}")
        print(f"  {e2.upper()} side: {', '.join(bot_tokens[:12])}")

        results["pair_differences"][f"{e1}-{e2}"] = {
            "distance": float(diff_norm),
            "note": note,
            f"{e1}_side_tokens": top_tokens,
            f"{e2}_side_tokens": bot_tokens,
        }

    # Social vs emotion comparison
    print(f"\n{'='*70}")
    print("EMOTION vs SOCIAL: What separates the two blocks?")
    print(f"{'='*70}")

    soc_conds = ["belief", "intention", "mentalizing", "theory_of_mind",
                 "judgment", "moral", "empathy", "self_referential"]
    soc_mask = [i for i, cc in enumerate(conditions) if cc in soc_conds]
    soc_mean = acts[soc_mask].mean(axis=0)

    boundary_vec = torch.tensor(soc_mean - emo_mean)
    logits = lm_head @ boundary_vec
    top_vals, top_ids = torch.topk(logits, 20)
    bot_vals, bot_ids = torch.topk(-logits, 20)

    soc_tokens = [tokenizer.decode(tid.item()).strip() for tid in top_ids]
    emo_tokens = [tokenizer.decode(tid.item()).strip() for tid in bot_ids]

    print(f"  SOCIAL side: {', '.join(soc_tokens[:15])}")
    print(f"  EMOTION side: {', '.join(emo_tokens[:15])}")

    results["boundary"] = {
        "social_side_tokens": soc_tokens,
        "emotion_side_tokens": emo_tokens,
    }

    json.dump(results, open(OUT_DIR / f"emotion_logit_lens_{model_short}.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved results")
    print("DONE")


if __name__ == "__main__":
    main()
