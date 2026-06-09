#!/usr/bin/env python3
"""
Emotion confusion test: does within-affective compression cause behavioral confusion?

For each model, generate counseling advice for the same scenarios under
different emotion vs social-cognition framings. Measure response differentiation
via TF-IDF cosine distance.

Prediction: emotion framings → less differentiated advice (same steps, different
preamble) than social framings → genuinely different action strategies.

GPU required (text generation).

Usage:
  python src/emotion_confusion_test.py --model_path /path/to/model [--model_short name]
"""
from __future__ import annotations
import argparse, json, gc
import numpy as np
import torch
from pathlib import Path
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon, mannwhitneyu

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

SITUATIONS = [
    "Your teenage child was caught shoplifting from a store",
    "You discovered your business partner has been secretly diverting funds",
    "Your elderly parent's caregiver has been neglecting their duties",
    "A close friend publicly mocked your career at a dinner party",
    "You found out your medical records were shared without your consent",
    "Your neighbor has been deliberately damaging your property",
    "A teacher singled out and humiliated your child in front of the class",
    "You discovered that your promotion was given to a less qualified person due to favoritism",
    "Your sibling revealed a family secret that you had trusted them to keep",
    "A contractor took your payment but never completed the work on your home",
]

EMOTION_FRAMINGS = {
    "anger": "You feel very angry about this.",
    "sadness": "You feel deeply sad and heartbroken about this.",
    "fear": "You feel very afraid and anxious about this.",
    "disgust": "You feel deeply disgusted and revolted by this.",
    "happiness": "You feel surprisingly relieved and hopeful about how this might turn out.",
}

SOCIAL_FRAMINGS = {
    "belief": "You believe this was completely deliberate and calculated.",
    "empathy": "You are mainly worried about how this affects the other people involved.",
    "judgment": "You see this as a serious moral violation that crosses a clear line.",
    "intention": "You have decided to confront the person responsible directly.",
}

PROMPT_TEMPLATE = """A person is going through this situation: "{situation}"

{framing}

As their counselor, give them specific, practical advice. What concrete steps should they take? 3-4 sentences."""


def generate(model, tokenizer, prompt, device, max_new=200):
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"User: {prompt}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def pairwise_tfidf_dist(texts):
    if any(t == "" or len(t) < 10 for t in texts):
        return {}
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sim = cosine_similarity(tfidf)
    n = len(texts)
    return {(i, j): float(1 - sim[i, j]) for i, j in combinations(range(n), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*70}")
    print(f"EMOTION CONFUSION TEST: {model_short}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    emo_keys = list(EMOTION_FRAMINGS.keys())
    soc_keys = list(SOCIAL_FRAMINGS.keys())
    responses = {}

    n_total = len(SITUATIONS) * (len(emo_keys) + len(soc_keys))
    count = 0

    for si, sit in enumerate(SITUATIONS):
        for emo, framing in EMOTION_FRAMINGS.items():
            prompt = PROMPT_TEMPLATE.format(situation=sit, framing=framing)
            resp = generate(model, tokenizer, prompt, device)
            responses[("emo", emo, si)] = resp
            count += 1
            if count % 10 == 0:
                print(f"  [{count}/{n_total}]", flush=True)

        for soc, framing in SOCIAL_FRAMINGS.items():
            prompt = PROMPT_TEMPLATE.format(situation=sit, framing=framing)
            resp = generate(model, tokenizer, prompt, device)
            responses[("soc", soc, si)] = resp
            count += 1
            if count % 10 == 0:
                print(f"  [{count}/{n_total}]", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()

    # Compute per-situation pairwise distances
    emo_pair_dists = {pair: [] for pair in combinations(emo_keys, 2)}
    soc_pair_dists = {pair: [] for pair in combinations(soc_keys, 2)}

    for si in range(len(SITUATIONS)):
        emo_texts = [responses[("emo", k, si)] for k in emo_keys]
        soc_texts = [responses[("soc", k, si)] for k in soc_keys]

        ed = pairwise_tfidf_dist(emo_texts)
        sd = pairwise_tfidf_dist(soc_texts)

        for (i, j), d in ed.items():
            emo_pair_dists[(emo_keys[i], emo_keys[j])].append(d)
        for (i, j), d in sd.items():
            soc_pair_dists[(soc_keys[i], soc_keys[j])].append(d)

    all_emo_d = [d for dists in emo_pair_dists.values() for d in dists]
    all_soc_d = [d for dists in soc_pair_dists.values() for d in dists]
    emo_mean = np.mean(all_emo_d)
    soc_mean = np.mean(all_soc_d)

    # Per-situation means for paired test
    emo_per_sit = []
    soc_per_sit = []
    for si in range(len(SITUATIONS)):
        ed = [emo_pair_dists[pair][si] for pair in combinations(emo_keys, 2)
              if si < len(emo_pair_dists[pair])]
        sd = [soc_pair_dists[pair][si] for pair in combinations(soc_keys, 2)
              if si < len(soc_pair_dists[pair])]
        if ed and sd:
            emo_per_sit.append(np.mean(ed))
            soc_per_sit.append(np.mean(sd))

    stat, p_wil = wilcoxon(soc_per_sit, emo_per_sit, alternative="greater") if len(emo_per_sit) >= 6 else (0, 1)
    u, p_mw = mannwhitneyu(all_soc_d, all_emo_d, alternative="greater")

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS: {model_short}")
    print(f"{'='*70}")

    print(f"\n  EMOTION pairs:")
    for pair in sorted(emo_pair_dists.keys()):
        dists = emo_pair_dists[pair]
        print(f"    {pair[0]:>10s}—{pair[1]:<10s}: {np.mean(dists):.4f} ± {np.std(dists):.4f}")

    print(f"\n  SOCIAL pairs:")
    for pair in sorted(soc_pair_dists.keys()):
        dists = soc_pair_dists[pair]
        print(f"    {pair[0]:>10s}—{pair[1]:<10s}: {np.mean(dists):.4f} ± {np.std(dists):.4f}")

    print(f"\n  Emotion mean dist:  {emo_mean:.4f}")
    print(f"  Social mean dist:   {soc_mean:.4f}")
    print(f"  Ratio (emo/soc):    {emo_mean/soc_mean:.3f}")
    print(f"  Wilcoxon (soc>emo): stat={stat:.1f}, p={p_wil:.4f}")
    print(f"  Mann-Whitney:       U={u:.0f}, p={p_mw:.4f}")

    n_soc_wins = sum(s > e for s, e in zip(soc_per_sit, emo_per_sit))
    print(f"  Sign: {n_soc_wins}/{len(emo_per_sit)} situations social > emotion")

    # Example
    print(f"\n{'='*70}")
    print(f"EXAMPLE: \"{SITUATIONS[0]}\"")
    print(f"{'='*70}")
    for k in emo_keys:
        print(f"\n  [{k.upper()}]: {responses[('emo', k, 0)][:300]}")
    for k in soc_keys:
        print(f"\n  [{k.upper()}]: {responses[('soc', k, 0)][:300]}")

    # Save
    results = {
        "model": model_short,
        "n_situations": len(SITUATIONS),
        "emotion_keys": emo_keys,
        "social_keys": soc_keys,
        "emotion_mean_dist": float(emo_mean),
        "social_mean_dist": float(soc_mean),
        "ratio": float(emo_mean / soc_mean) if soc_mean > 0 else 0,
        "wilcoxon": {"stat": float(stat), "p": float(p_wil)},
        "mann_whitney": {"U": float(u), "p": float(p_mw)},
        "n_social_wins": int(n_soc_wins),
        "emotion_pair_dists": {f"{k[0]}-{k[1]}": {"mean": float(np.mean(v)), "per_sit": v}
                               for k, v in emo_pair_dists.items()},
        "social_pair_dists": {f"{k[0]}-{k[1]}": {"mean": float(np.mean(v)), "per_sit": v}
                              for k, v in soc_pair_dists.items()},
        "responses": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in responses.items()},
    }
    out_path = OUT / f"emotion_confusion_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
