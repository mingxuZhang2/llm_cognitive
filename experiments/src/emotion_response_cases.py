#!/usr/bin/env python3
"""
Generate actual model responses for matched emotion scenarios.
Show side-by-side: same scenario, different emotion → what does the model say?

This produces concrete cases showing the model's emotion confusion in practice.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "results" / "mechanistic"
OUT_DIR.mkdir(exist_ok=True)

SCENARIOS = [
    {
        "situation": "My best friend lied to me about something important.",
        "expected_diff": "Anger → confront/set boundaries; Sadness → grieve the trust; Fear → worry about future; Disgust → distance/reject",
    },
    {
        "situation": "I found out my coworker has been taking credit for my work.",
        "expected_diff": "Anger → assert rights; Sadness → feel undervalued; Disgust → moral contempt",
    },
    {
        "situation": "I watched a documentary about animal cruelty in factory farms.",
        "expected_diff": "Anger → take action/protest; Sadness → mourn; Disgust → physical revulsion + moral outrage",
    },
    {
        "situation": "My partner forgot our anniversary even though I reminded them twice.",
        "expected_diff": "Anger → communicate frustration; Sadness → feel unimportant; Fear → worry about relationship",
    },
    {
        "situation": "A stranger on the internet attacked me personally for sharing my opinion.",
        "expected_diff": "Anger → defend/push back; Sadness → hurt/withdraw; Fear → feel unsafe; Disgust → dismiss/block",
    },
]

EMOTIONS = {
    "anger":   "I feel really angry about this.",
    "sadness": "I feel really sad about this.",
    "fear":    "I feel really afraid because of this.",
    "disgust": "I feel really disgusted by this.",
}


def generate(model, tokenizer, prompt, device, max_tokens=150):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Emotion Response Cases ({model_short}) ===")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []

    for si, scenario in enumerate(SCENARIOS):
        print(f"\n{'='*80}")
        print(f"SCENARIO {si+1}: {scenario['situation']}")
        print(f"Expected differences: {scenario['expected_diff']}")
        print(f"{'='*80}")

        case = {"scenario": scenario["situation"], "expected": scenario["expected_diff"], "responses": {}}

        for emo, emo_text in EMOTIONS.items():
            prompt = f"{scenario['situation']} {emo_text} What should I do?"
            response = generate(model, tokenizer, prompt, device)
            case["responses"][emo] = response

            print(f"\n  [{emo.upper()}] {emo_text}")
            print(f"  Response: {response[:200]}...")

        # Compare: highlight similarities between responses
        print(f"\n  --- COMPARISON ---")
        resps = case["responses"]
        # Simple overlap: what fraction of sentences appear in multiple responses?
        from collections import Counter
        all_sentences = {}
        for emo, resp in resps.items():
            sents = [s.strip() for s in resp.replace(".", ".\n").split("\n") if len(s.strip()) > 20]
            all_sentences[emo] = sents

        # Check if anger and sadness responses share similar structure
        for e1, e2 in [("anger", "sadness"), ("anger", "disgust"), ("fear", "sadness")]:
            r1_words = set(resps[e1].lower().split())
            r2_words = set(resps[e2].lower().split())
            overlap = len(r1_words & r2_words) / max(len(r1_words | r2_words), 1)
            print(f"  Word overlap {e1}-{e2}: {overlap:.0%}")

        results.append(case)

    out_path = OUT_DIR / f"emotion_response_cases_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
