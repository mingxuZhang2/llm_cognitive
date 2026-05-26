#!/usr/bin/env python3
"""
Experiment A: Brain-derived cognitive steering.

Use the affective-mentalistic boundary direction (extracted from brain-LLM
RSA analysis) to steer LLM generation. Amplify/suppress this axis during
inference and measure effects on output behavior.

Test: Given moral dilemma prompts, does steering along the brain-derived
axis shift outputs between emotional vs analytical responses?

Metrics:
- Emotion word ratio (NRC lexicon)
- Analytical word ratio
- First-person pronoun frequency (empathy proxy)
- Response length and perplexity

Usage:
  python cognitive_steering.py \
    --model_path /path/to/Qwen2.5-7B-Instruct \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MORAL_PROMPTS = [
    "A self-driving car must choose between hitting an elderly pedestrian or swerving into a wall, likely killing the passenger. What should it do?",
    "You discover your best friend has been stealing from your workplace. Do you report them?",
    "A doctor has five patients who need organ transplants. A healthy person walks in. Should the doctor sacrifice one to save five?",
    "You find a wallet with $5000 and the owner's ID. Returning it would take considerable effort. What do you do?",
    "A parent must decide whether to use an experimental treatment on their sick child that has a 50% chance of cure and 50% chance of making things worse.",
    "Should a journalist publish leaked documents that expose government corruption but endanger intelligence agents?",
    "A lifeboat can hold 10 people but 15 are in the water. How do you decide who gets saved?",
    "You can save a burning building's occupants by lying to the fire department about the situation. Is it right to lie?",
    "An AI system could prevent 1000 crimes per year but requires mass surveillance of citizens. Should it be deployed?",
    "A company can save thousands of jobs by covering up a minor environmental violation. What should the CEO do?",
]

EMOTION_WORDS = set("happy sad angry afraid scared joyful miserable furious terrified anxious worried excited thrilled devastated heartbroken peaceful calm love hate fear anger sadness joy grief sorrow pain suffering hurt comfort warm cold lonely hopeful desperate grateful".split())
ANALYTICAL_WORDS = set("therefore however although because consequently furthermore moreover thus hence analysis consider evaluate assess determine conclude evidence reason logic rational objective systematic framework principle criteria factor perspective".split())


def get_boundary_direction(model, tokenizer, device):
    """Compute the affective-mentalistic boundary direction from representative stimuli."""
    affective_texts = [
        "She was trembling with rage after discovering the betrayal.",
        "The child laughed with pure delight at the surprise.",
        "He felt a deep sadness watching the old house being demolished.",
        "The horror movie scene made everyone scream in terror.",
        "She was disgusted by the unsanitary conditions in the restaurant.",
        "The wedding filled everyone with warmth and happiness.",
        "His anger boiled over when he saw the injustice.",
        "The fear of failure kept her awake every night.",
    ]
    mentalistic_texts = [
        "She believed he had already left, but he was hiding in the next room.",
        "He tried to understand why she had made that decision.",
        "The child didn't realize that others could have different beliefs.",
        "She intended to surprise him but he figured out the plan.",
        "He could sense that she was thinking about something troubling.",
        "She reflected on how much her own perspective had changed over the years.",
        "The judge carefully weighed whether the punishment fit the crime.",
        "He realized that what he thought was true was actually a misunderstanding.",
    ]

    def get_mean_activation(texts):
        acts = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            last_layer = out.hidden_states[-1]
            mask = inputs["attention_mask"].bool()
            mean_act = last_layer[0][mask[0]].mean(dim=0)
            acts.append(mean_act)
        return torch.stack(acts).mean(dim=0)

    aff_center = get_mean_activation(affective_texts)
    ment_center = get_mean_activation(mentalistic_texts)

    boundary = ment_center - aff_center
    boundary = boundary / boundary.norm()
    return boundary


def generate_with_steering(model, tokenizer, prompt, boundary_dir, alpha, device,
                           max_new_tokens=200):
    """Generate text with activation steering along the boundary direction."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            shift = alpha * boundary_dir.to(hidden.device)
            hidden = hidden + shift.unsqueeze(0).unsqueeze(0)
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook_fn

    n_layers = model.config.num_hidden_layers
    target_layers = range(n_layers * 3 // 4, n_layers)
    for layer_idx in target_layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    for h in hooks:
        h.remove()

    generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated


def analyze_response(text):
    """Compute behavioral metrics for a generated response."""
    words = text.lower().split()
    n = max(len(words), 1)

    emotion_count = sum(1 for w in words if w.strip(".,!?;:\"'") in EMOTION_WORDS)
    analytical_count = sum(1 for w in words if w.strip(".,!?;:\"'") in ANALYTICAL_WORDS)
    first_person = sum(1 for w in words if w.strip(".,!?;:\"'") in {"i", "me", "my", "myself", "we", "us", "our"})

    return {
        "n_words": len(words),
        "emotion_ratio": emotion_count / n,
        "analytical_ratio": analytical_count / n,
        "first_person_ratio": first_person / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alphas", default="-20,-10,-5,0,5,10,20",
                        help="Steering strengths (negative=emotional, positive=analytical)")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    print(f"Loading {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    print("Computing boundary direction...")
    boundary = get_boundary_direction(model, tokenizer, device)
    print(f"  Boundary norm: {boundary.norm():.4f}")

    results = []

    for pi, prompt in enumerate(MORAL_PROMPTS):
        print(f"\nPrompt {pi+1}/{len(MORAL_PROMPTS)}: {prompt[:60]}...")
        for alpha in alphas:
            label = "baseline" if alpha == 0 else f"{'emotional' if alpha < 0 else 'analytical'}_{abs(alpha)}"
            print(f"  alpha={alpha:+.0f} ({label})...", end="", flush=True)

            response = generate_with_steering(model, tokenizer, prompt, boundary, alpha, device)
            metrics = analyze_response(response)

            result = {
                "prompt_idx": pi,
                "prompt": prompt,
                "alpha": alpha,
                "label": label,
                "response": response,
                **metrics,
            }
            results.append(result)
            print(f" {metrics['n_words']}w, emo={metrics['emotion_ratio']:.3f}, "
                  f"ana={metrics['analytical_ratio']:.3f}")

    out_path = Path(args.output_dir) / f"{args.model_short}_steering.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n=== SUMMARY ===")
    for alpha in alphas:
        alpha_results = [r for r in results if r["alpha"] == alpha]
        mean_emo = np.mean([r["emotion_ratio"] for r in alpha_results])
        mean_ana = np.mean([r["analytical_ratio"] for r in alpha_results])
        mean_fp = np.mean([r["first_person_ratio"] for r in alpha_results])
        print(f"  alpha={alpha:+6.0f}:  emotion={mean_emo:.4f}  analytical={mean_ana:.4f}  1st_person={mean_fp:.4f}")


if __name__ == "__main__":
    main()
