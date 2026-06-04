#!/usr/bin/env python3
"""
Logit-based moral judgment steering (replaces forced-choice text generation).

Instead of generating text and parsing A/B, directly compute:
    logit_diff = log P(utilitarian_token) - log P(deontological_token)

at each steering alpha. No text generation → no degeneration artifacts.

Moderate alpha range [-5..+5] to stay in the coherent regime.

Brain boundary direction = mentalistic_center - affective_center.
  - Negative alpha: suppress mentalistic → more emotional → more deontological
  - Positive alpha: amplify mentalistic → more analytical → more utilitarian

Usage (GPU):
  python src/moral_judgment_logit.py \
      --model_path /path/to/model \
      --model_short Qwen2.5-7B-Instruct \
      --peak_layer 27
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path(__file__).resolve().parents[1]

AFFECTIVE_TEXTS = [
    "She was trembling with rage after discovering the betrayal.",
    "The child laughed with pure delight at the surprise.",
    "He felt a deep sadness watching the old house being demolished.",
    "The horror movie scene made everyone scream in terror.",
    "She was disgusted by the unsanitary conditions in the restaurant.",
    "The wedding filled everyone with warmth and happiness.",
    "His anger boiled over when he saw the injustice.",
    "The fear of failure kept her awake every night.",
]

MENTALISTIC_TEXTS = [
    "She believed he had already left, but he was hiding in the next room.",
    "He tried to understand why she had made that decision.",
    "The child didn't realize that others could have different beliefs.",
    "She intended to surprise him but he figured out the plan.",
    "He could sense that she was thinking about something troubling.",
    "She reflected on how much her own perspective had changed over the years.",
    "The judge carefully weighed whether the punishment fit the crime.",
    "He realized that what he thought was true was actually a misunderstanding.",
]

ALPHAS = [-5, -3, -1, 0, 1, 3, 5]


def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Cannot find model layers")


def get_boundary_direction(model, tokenizer, device):
    def get_mean_activation(texts):
        acts = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            last_layer = out.hidden_states[-1]
            mask = inputs["attention_mask"].bool()
            mean_act = last_layer[0][mask[0]].mean(dim=0)
            acts.append(mean_act)
        return torch.stack(acts).mean(dim=0)

    aff_center = get_mean_activation(AFFECTIVE_TEXTS)
    ment_center = get_mean_activation(MENTALISTIC_TEXTS)
    boundary = ment_center - aff_center
    boundary = boundary / boundary.norm()
    return boundary


def get_logit_diff(model, tokenizer, prompt, boundary_dir, alpha, device,
                   util_token_id, deont_token_id):
    """Forward pass with steering, return logit(util) - logit(deont) at next-token position."""
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

    layers = get_model_layers(model)
    n_layers = len(layers)
    target_layers = range(n_layers * 3 // 4, n_layers)
    for layer_idx in target_layers:
        h = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last token position

    for h in hooks:
        h.remove()

    log_probs = torch.log_softmax(logits, dim=-1)
    lp_util = log_probs[util_token_id].item()
    lp_deont = log_probs[deont_token_id].item()

    p_util = torch.softmax(logits, dim=-1)[util_token_id].item()
    p_deont = torch.softmax(logits, dim=-1)[deont_token_id].item()
    entropy = -(torch.softmax(logits, dim=-1) * log_probs).sum().item()

    return {
        "logit_diff": lp_util - lp_deont,
        "p_util": p_util,
        "p_deont": p_deont,
        "p_util_normalized": p_util / (p_util + p_deont) if (p_util + p_deont) > 0 else 0.5,
        "entropy": entropy,
    }


def build_prompt(dilemma):
    """Build a prompt ending with 'Answer: ' so next token is A or B."""
    scenario = dilemma["scenario"]
    opt_a = dilemma["options"][0]
    opt_b = dilemma["options"][1]
    return (
        f"Consider the following moral dilemma:\n\n"
        f"{scenario}\n\n"
        f"Option A: {opt_a}\n"
        f"Option B: {opt_b}\n\n"
        f"Which option do you choose? Answer with a single letter (A or B): "
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_short", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--peak_layer", type=int, default=27)
    parser.add_argument("--dilemmas_path", type=str,
                        default=str(BASE / "data/cognitive_stimuli/moral/moral_dilemmas.jsonl"))
    parser.add_argument("--output_dir", type=str,
                        default=str(BASE / "results/moral_judgment"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.model_short}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get token IDs for A and B
    # Try multiple encodings since tokenizers vary
    for candidate in ["A", " A"]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            token_a_id = ids[0]
            break
    else:
        token_a_id = tokenizer.encode("A", add_special_tokens=False)[-1]

    for candidate in ["B", " B"]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            token_b_id = ids[0]
            break
    else:
        token_b_id = tokenizer.encode("B", add_special_tokens=False)[-1]

    print(f"Token A id={token_a_id} ('{tokenizer.decode([token_a_id])}')")
    print(f"Token B id={token_b_id} ('{tokenizer.decode([token_b_id])}')")

    print("Computing boundary direction...")
    boundary_dir = get_boundary_direction(model, tokenizer, device)
    print(f"Boundary direction norm: {boundary_dir.norm().item():.4f}")

    # Load dilemmas
    dilemmas = [json.loads(l) for l in open(args.dilemmas_path)]
    print(f"Loaded {len(dilemmas)} dilemmas\n")

    results = []
    for di, dilemma in enumerate(dilemmas):
        util_idx = dilemma["utilitarian_option_index"]
        # util option = option[util_idx], which is "A" if util_idx==0, "B" if util_idx==1
        util_token = token_a_id if util_idx == 0 else token_b_id
        deont_token = token_b_id if util_idx == 0 else token_a_id

        prompt = build_prompt(dilemma)
        personal = "personal" in dilemma.get("subcategory", "").lower()

        for alpha in ALPHAS:
            out = get_logit_diff(model, tokenizer, prompt, boundary_dir,
                                alpha, device, util_token, deont_token)
            results.append({
                "dilemma_id": dilemma["id"],
                "dilemma_type": dilemma["dilemma_type"],
                "subcategory": dilemma.get("subcategory", ""),
                "personal": personal,
                "alpha": alpha,
                **out,
            })

        # Print progress
        last = results[-len(ALPHAS):]
        diffs = [r["logit_diff"] for r in last]
        p_utils = [r["p_util_normalized"] for r in last]
        print(f"  [{di+1:2d}/{len(dilemmas)}] {dilemma['id']:15s} "
              f"{'personal' if personal else 'impersnl':8s}  "
              f"logit_diff: {' '.join(f'{d:+.2f}' for d in diffs)}  "
              f"P(util): {' '.join(f'{p:.2f}' for p in p_utils)}")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"SUMMARY: Logit-based moral steering ({args.model_short})")
    print(f"{'='*70}")

    by_alpha = {}
    by_alpha_personal = {}
    by_alpha_impersonal = {}
    for r in results:
        a = r["alpha"]
        if a not in by_alpha:
            by_alpha[a] = []
            by_alpha_personal[a] = []
            by_alpha_impersonal[a] = []
        by_alpha[a].append(r["logit_diff"])
        if r["personal"]:
            by_alpha_personal[a].append(r["logit_diff"])
        else:
            by_alpha_impersonal[a].append(r["logit_diff"])

    print(f"\n{'alpha':>6s} {'mean_logit_diff':>16s} {'P(util)':>10s} {'entropy':>10s}  "
          f"{'personal':>10s} {'impersonal':>12s}")
    print("-" * 70)

    alpha_means = []
    for alpha in sorted(by_alpha.keys()):
        diffs = by_alpha[alpha]
        mean_diff = np.mean(diffs)
        mean_putil = np.mean([r["p_util_normalized"] for r in results if r["alpha"] == alpha])
        mean_entropy = np.mean([r["entropy"] for r in results if r["alpha"] == alpha])
        mean_personal = np.mean(by_alpha_personal[alpha]) if by_alpha_personal[alpha] else float('nan')
        mean_impersonal = np.mean(by_alpha_impersonal[alpha]) if by_alpha_impersonal[alpha] else float('nan')

        print(f"{alpha:+6d} {mean_diff:+16.3f} {mean_putil:10.3f} {mean_entropy:10.1f}  "
              f"{mean_personal:+10.3f} {mean_impersonal:+12.3f}")

        alpha_means.append({"alpha": alpha, "mean_logit_diff": mean_diff,
                           "mean_p_util": mean_putil, "mean_entropy": mean_entropy})

    # Spearman: does alpha predict logit_diff?
    from scipy.stats import spearmanr, pearsonr
    all_alphas = [r["alpha"] for r in results]
    all_diffs = [r["logit_diff"] for r in results]
    rho_all, p_all = spearmanr(all_alphas, all_diffs)
    r_all, pr_all = pearsonr(all_alphas, all_diffs)

    personal_alphas = [r["alpha"] for r in results if r["personal"]]
    personal_diffs = [r["logit_diff"] for r in results if r["personal"]]
    rho_pers, p_pers = spearmanr(personal_alphas, personal_diffs) if personal_alphas else (0, 1)

    impersonal_alphas = [r["alpha"] for r in results if not r["personal"]]
    impersonal_diffs = [r["logit_diff"] for r in results if not r["personal"]]
    rho_imp, p_imp = spearmanr(impersonal_alphas, impersonal_diffs) if impersonal_alphas else (0, 1)

    print(f"\nCorrelation alpha vs logit_diff(utilitarian):")
    print(f"  All:        Spearman ρ={rho_all:+.3f} p={p_all:.4f}  Pearson r={r_all:+.3f} p={pr_all:.4f}")
    print(f"  Personal:   Spearman ρ={rho_pers:+.3f} p={p_pers:.4f}")
    print(f"  Impersonal: Spearman ρ={rho_imp:+.3f} p={p_imp:.4f}")

    # Prediction: negative alpha (suppress mentalistic) → more utilitarian → positive logit_diff
    # So we expect NEGATIVE correlation (higher alpha = more mentalistic = more deontological = lower logit_diff)
    print(f"\nPREDICTION (Greene/Koenigs):")
    print(f"  Negative alpha (suppress mentalistic/amplify emotion) → more utilitarian")
    print(f"  → Expect NEGATIVE correlation: alpha ↑ → logit_diff(util) ↓")
    if rho_all < -0.05 and p_all < 0.05:
        print(f"  → CONFIRMED: ρ={rho_all:+.3f}, p={p_all:.4f}")
    elif rho_all < 0:
        print(f"  → TREND: ρ={rho_all:+.3f}, p={p_all:.4f}")
    else:
        print(f"  → NOT CONFIRMED: ρ={rho_all:+.3f}")

    # Quality check: entropy should not explode
    entropies = [r["entropy"] for r in results]
    baseline_entropy = np.mean([r["entropy"] for r in results if r["alpha"] == 0])
    max_entropy = max(entropies)
    print(f"\nQuality: baseline entropy={baseline_entropy:.1f}, max={max_entropy:.1f}, "
          f"ratio={max_entropy/baseline_entropy:.2f}")
    if max_entropy / baseline_entropy > 2.0:
        print(f"  WARNING: entropy ratio > 2x at extreme alphas — model may be degrading")

    # Save
    out_path = Path(args.output_dir) / f"{args.model_short}_moral_logit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model_short,
        "alphas": ALPHAS,
        "n_dilemmas": len(dilemmas),
        "token_a_id": token_a_id,
        "token_b_id": token_b_id,
        "correlation_all": {"rho": float(rho_all), "p": float(p_all),
                           "pearson_r": float(r_all), "pearson_p": float(pr_all)},
        "correlation_personal": {"rho": float(rho_pers), "p": float(p_pers)},
        "correlation_impersonal": {"rho": float(rho_imp), "p": float(p_imp)},
        "alpha_summary": alpha_means,
        "quality": {"baseline_entropy": float(baseline_entropy),
                    "max_entropy": float(max_entropy),
                    "entropy_ratio": float(max_entropy / baseline_entropy)},
        "per_dilemma": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
