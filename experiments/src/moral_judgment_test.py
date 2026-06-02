#!/usr/bin/env python3
"""
Moral judgment steering test  (Greene 2001 / Koenigs 2007 analog).

Neuroscience prediction:
  - Suppressing the emotion direction (alpha < 0) should increase utilitarian
    choices on PERSONAL dilemmas (where emotion normally blocks utilitarian
    reasoning).
  - The effect should be WEAKER on impersonal dilemmas (already cognitive).
  - Amplifying emotion (alpha > 0) should increase deontological choices on
    personal dilemmas.

Method:
  1. Compute the affective-mentalistic boundary direction (reused from
     cognitive_steering.py).
  2. For each moral dilemma (30 from moral_dilemmas.jsonl), at each steering
     alpha, generate a forced-choice A/B response (3 samples, majority vote).
  3. Map to utilitarian (1) or deontological (0).
  4. Logistic regression: utilitarian ~ alpha * dilemma_type (personal flag).
  5. Plot steering curves: utilitarian rate vs alpha, split by personal/impersonal.

Usage:
  python src/moral_judgment_test.py \
      --model_path /path/to/model \
      --model_short Qwen2.5-7B-Instruct \
      --dilemmas_path data/cognitive_stimuli/moral/moral_dilemmas.jsonl \
      --output_dir results/moral_judgment
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Boundary direction (reused from cognitive_steering.py)
# ---------------------------------------------------------------------------

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

ALPHAS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
N_SAMPLES = 3  # majority-vote samples per (dilemma, alpha)


def get_model_layers(model):
    """Return the list of transformer layers, handling architecture differences."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Cannot find model layers")


def get_boundary_direction(model, tokenizer, device):
    """Compute the affective-mentalistic boundary direction from representative stimuli."""

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

    boundary = ment_center - aff_center  # positive = mentalistic direction
    boundary = boundary / boundary.norm()
    return boundary


# ---------------------------------------------------------------------------
# Steering hooks (reused pattern from cognitive_steering.py, last 25% layers)
# ---------------------------------------------------------------------------

def generate_with_steering(model, tokenizer, prompt, boundary_dir, alpha, device,
                           max_new_tokens=10):
    """Generate text with activation steering along the boundary direction.

    Steering is applied to the last 25% of transformer layers as:
        hidden += alpha * direction
    Negative alpha suppresses the mentalistic (amplifies emotional) component.
    """
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

    generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True)
    return generated


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_choice(response_text: str) -> str | None:
    """Extract the first 'A' or 'B' from the model's response."""
    # Look for standalone A or B (possibly preceded by punctuation/whitespace)
    match = re.search(r'\b([AB])\b', response_text.strip())
    if match:
        return match.group(1)
    # Fallback: first character that is A or B
    for ch in response_text.strip():
        if ch in ("A", "B"):
            return ch
    return None


def choice_to_utilitarian(choice: str | None, util_idx: int) -> int | None:
    """Map choice letter to 1 (utilitarian) or 0 (deontological), or None if unparsable."""
    if choice is None:
        return None
    choice_idx = 0 if choice == "A" else 1
    return 1 if choice_idx == util_idx else 0


def classify_dilemma_type(subcategory: str) -> str:
    """Classify a dilemma as PERSONAL or IMPERSONAL based on subcategory."""
    if "personal" in subcategory.lower():
        # subcategories like personal_sacrificial, personal_sacrificial_loop,
        # personal_sacrificial_remote, personal_obligation,
        # personal_impersonal_obligation — if "personal" appears, classify as personal
        # UNLESS it is purely "impersonal" (impersonal_sacrificial, impersonal_triage, etc.)
        if subcategory.lower().startswith("impersonal"):
            return "impersonal"
        return "personal"
    return "impersonal"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_prompt(dilemma: dict) -> str:
    """Build the forced-choice prompt for a moral dilemma."""
    return (
        f"Consider the following moral dilemma:\n"
        f"{dilemma['scenario']}\n\n"
        f"Option A: {dilemma['options'][0]}\n"
        f"Option B: {dilemma['options'][1]}\n\n"
        f"Which option would you choose? Reply with ONLY \"A\" or \"B\"."
    )


def run_experiment(model, tokenizer, dilemmas, boundary_dir, alphas, device):
    """Run the full moral judgment steering experiment for one model."""
    all_results = []

    for di, dilemma in enumerate(dilemmas):
        prompt = build_prompt(dilemma)
        dtype = classify_dilemma_type(dilemma["subcategory"])
        util_idx = dilemma["utilitarian_option_index"]

        print(f"\n  Dilemma {di+1}/{len(dilemmas)} [{dtype}]: "
              f"{dilemma['dilemma_type']} ({dilemma['id']})")

        for alpha in alphas:
            votes = []
            raw_responses = []

            for sample_i in range(N_SAMPLES):
                response = generate_with_steering(
                    model, tokenizer, prompt, boundary_dir, alpha, device,
                    max_new_tokens=10,
                )
                choice = parse_choice(response)
                util = choice_to_utilitarian(choice, util_idx)
                votes.append(util)
                raw_responses.append(response.strip())

            # Majority vote (ignoring None)
            valid_votes = [v for v in votes if v is not None]
            if valid_votes:
                majority = 1 if sum(valid_votes) > len(valid_votes) / 2 else 0
            else:
                majority = None

            result = {
                "dilemma_id": dilemma["id"],
                "dilemma_type": dilemma["dilemma_type"],
                "subcategory": dilemma["subcategory"],
                "personal": dtype == "personal",
                "alpha": alpha,
                "votes": votes,
                "majority_vote": majority,  # 1=utilitarian, 0=deontological
                "raw_responses": raw_responses,
                "utilitarian_option_index": util_idx,
            }
            all_results.append(result)

            tag = "UTIL" if majority == 1 else ("DEON" if majority == 0 else "???")
            print(f"    alpha={alpha:+3d}  votes={votes}  -> {tag}  "
                  f"[{raw_responses[0][:30]}...]")

    return all_results


def compute_utilitarian_rates(results):
    """Compute utilitarian rate per (alpha, personal/impersonal)."""
    groups = defaultdict(list)
    for r in results:
        if r["majority_vote"] is not None:
            key = (r["alpha"], "personal" if r["personal"] else "impersonal")
            groups[key].append(r["majority_vote"])

    rates = {}
    for (alpha, dtype), votes in sorted(groups.items()):
        rates[(alpha, dtype)] = {
            "utilitarian_rate": np.mean(votes),
            "n": len(votes),
            "n_utilitarian": sum(votes),
        }
    return rates


def run_logistic_regression(results):
    """Logistic regression: utilitarian ~ alpha * personal.

    Returns coefficients and p-values for the interaction term.
    Uses statsmodels if available, otherwise a simple summary.
    """
    # Build arrays
    alphas_arr, personal_arr, y_arr = [], [], []
    for r in results:
        if r["majority_vote"] is not None:
            alphas_arr.append(r["alpha"])
            personal_arr.append(1 if r["personal"] else 0)
            y_arr.append(r["majority_vote"])

    alphas_arr = np.array(alphas_arr, dtype=float)
    personal_arr = np.array(personal_arr, dtype=float)
    y_arr = np.array(y_arr, dtype=float)

    if len(y_arr) < 10 or len(np.unique(y_arr)) < 2:
        return {"error": "insufficient data or no variance in outcome"}

    try:
        import statsmodels.api as sm

        # Design matrix: intercept, alpha, personal, alpha*personal
        interaction = alphas_arr * personal_arr
        X = np.column_stack([
            np.ones(len(alphas_arr)),
            alphas_arr,
            personal_arr,
            interaction,
        ])

        logit = sm.Logit(y_arr, X)
        fit = logit.fit(disp=0, maxiter=100)

        return {
            "coef_intercept": float(fit.params[0]),
            "coef_alpha": float(fit.params[1]),
            "coef_personal": float(fit.params[2]),
            "coef_interaction": float(fit.params[3]),
            "pvalue_intercept": float(fit.pvalues[0]),
            "pvalue_alpha": float(fit.pvalues[1]),
            "pvalue_personal": float(fit.pvalues[2]),
            "pvalue_interaction": float(fit.pvalues[3]),
            "n": int(len(y_arr)),
            "converged": bool(fit.mle_retvals["converged"]),
        }
    except ImportError:
        # Fallback: just report correlation between alpha and utilitarian
        # for personal vs impersonal separately
        personal_mask = personal_arr == 1
        impersonal_mask = personal_arr == 0

        from scipy.stats import pearsonr
        stats = {}
        if personal_mask.sum() > 3:
            r_p, p_p = pearsonr(alphas_arr[personal_mask], y_arr[personal_mask])
            stats["personal_corr"] = float(r_p)
            stats["personal_pvalue"] = float(p_p)
        if impersonal_mask.sum() > 3:
            r_i, p_i = pearsonr(alphas_arr[impersonal_mask], y_arr[impersonal_mask])
            stats["impersonal_corr"] = float(r_i)
            stats["impersonal_pvalue"] = float(p_i)
        stats["n"] = int(len(y_arr))
        stats["fallback"] = True
        return stats
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_steering_curves(all_model_results, output_path):
    """2x2 subplot: per model, utilitarian rate vs alpha for personal/impersonal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = sorted(all_model_results.keys())
    n_models = len(model_names)
    nrows = 2 if n_models > 2 else 1
    ncols = 2 if n_models > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), squeeze=False)

    for idx, model_name in enumerate(model_names):
        ax = axes[idx // ncols][idx % ncols]
        results = all_model_results[model_name]["results"]
        rates = compute_utilitarian_rates(results)

        alphas_sorted = sorted(set(r["alpha"] for r in results))

        # Personal
        personal_rates = [rates.get((a, "personal"), {}).get("utilitarian_rate", np.nan)
                          for a in alphas_sorted]
        # Impersonal
        impersonal_rates = [rates.get((a, "impersonal"), {}).get("utilitarian_rate", np.nan)
                            for a in alphas_sorted]

        ax.plot(alphas_sorted, personal_rates, "o-", color="red",
                label="Personal", linewidth=2, markersize=6)
        ax.plot(alphas_sorted, impersonal_rates, "s-", color="blue",
                label="Impersonal", linewidth=2, markersize=6)

        # Shade suppress-emotion region (alpha < 0)
        ax.axvspan(min(alphas_sorted) - 1, 0, alpha=0.08, color="blue",
                   label="Suppress emotion")
        ax.axvspan(0, max(alphas_sorted) + 1, alpha=0.08, color="red",
                   label="Amplify emotion")

        # Reference line at alpha=0
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_xlim(min(alphas_sorted) - 1, max(alphas_sorted) + 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Steering alpha (- = suppress emotion, + = amplify)")
        ax.set_ylabel("Utilitarian rate")
        ax.set_title(model_name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Moral Judgment Steering: Greene/Koenigs Prediction\n"
        "(Suppress emotion -> more utilitarian on personal dilemmas)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Moral judgment steering test (Greene/Koenigs analog)")
    parser.add_argument("--model_path", required=True,
                        help="Path to HuggingFace model")
    parser.add_argument("--model_short", required=True,
                        help="Short model name (e.g. Qwen2.5-7B-Instruct)")
    parser.add_argument("--dilemmas_path", required=True,
                        help="Path to moral_dilemmas.jsonl")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for results JSON")
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS),
                        help="Comma-separated steering alphas")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES,
                        help="Number of samples per (dilemma, alpha) for majority vote")
    parser.add_argument("--aggregate_json", default=None,
                        help="Path to aggregated multi-model JSON (for plotting). "
                             "If provided, skip experiment and just plot.")
    args = parser.parse_args()

    alphas = [int(a) for a in args.alphas.split(",")]
    n_samples_override = args.n_samples
    global N_SAMPLES
    N_SAMPLES = n_samples_override

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir.parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # If aggregate_json provided, just plot
    if args.aggregate_json:
        with open(args.aggregate_json) as f:
            all_model_results = json.load(f)
        plot_steering_curves(all_model_results,
                             str(figures_dir / "moral_steering_curve.png"))
        return

    # -----------------------------------------------------------------------
    # Load dilemmas
    # -----------------------------------------------------------------------
    dilemmas = []
    with open(args.dilemmas_path) as f:
        for line in f:
            line = line.strip()
            if line:
                dilemmas.append(json.loads(line))
    print(f"Loaded {len(dilemmas)} dilemmas from {args.dilemmas_path}")

    n_personal = sum(1 for d in dilemmas
                     if classify_dilemma_type(d["subcategory"]) == "personal")
    n_impersonal = len(dilemmas) - n_personal
    print(f"  Personal: {n_personal}, Impersonal: {n_impersonal}")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {args.model_short} ({args.model_path})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    print(f"  Layers: {n_layers}, device: {device}")

    # -----------------------------------------------------------------------
    # Compute boundary direction
    # -----------------------------------------------------------------------
    print("\nComputing affective-mentalistic boundary direction...")
    t0 = time.time()
    boundary_dir = get_boundary_direction(model, tokenizer, device)
    print(f"  Boundary norm: {boundary_dir.norm():.4f}  ({time.time()-t0:.1f}s)")
    print(f"  Steering target: last 25% of layers "
          f"({n_layers * 3 // 4}-{n_layers - 1})")

    # -----------------------------------------------------------------------
    # Run experiment
    # -----------------------------------------------------------------------
    print(f"\nRunning moral judgment test: {len(dilemmas)} dilemmas x "
          f"{len(alphas)} alphas x {N_SAMPLES} samples")
    t0 = time.time()
    results = run_experiment(model, tokenizer, dilemmas, boundary_dir, alphas, device)
    elapsed = time.time() - t0
    print(f"\nExperiment done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # -----------------------------------------------------------------------
    # Compute utilitarian rates
    # -----------------------------------------------------------------------
    rates = compute_utilitarian_rates(results)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {args.model_short}")
    print("=" * 70)
    print(f"{'Alpha':>6s}  {'Personal':>10s} (n)  {'Impersonal':>10s} (n)")
    print("-" * 50)
    for alpha in alphas:
        p = rates.get((alpha, "personal"), {})
        i = rates.get((alpha, "impersonal"), {})
        p_rate = f"{p['utilitarian_rate']:.2f}" if p else "  --"
        i_rate = f"{i['utilitarian_rate']:.2f}" if i else "  --"
        p_n = p.get("n", 0)
        i_n = i.get("n", 0)
        print(f"{alpha:+6d}  {p_rate:>10s} ({p_n:2d})  {i_rate:>10s} ({i_n:2d})")

    # -----------------------------------------------------------------------
    # Logistic regression
    # -----------------------------------------------------------------------
    print("\nLogistic regression: utilitarian ~ alpha * personal")
    logreg = run_logistic_regression(results)
    for k, v in logreg.items():
        print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "model": args.model_short,
        "n_dilemmas": len(dilemmas),
        "n_personal": n_personal,
        "n_impersonal": n_impersonal,
        "alphas": alphas,
        "n_samples": N_SAMPLES,
        "elapsed_s": elapsed,
        "results": results,
        "utilitarian_rates": {f"{k[0]}_{k[1]}": v for k, v in rates.items()},
        "logistic_regression": logreg,
    }

    per_model_path = output_dir / f"{args.model_short}_moral_judgment.json"
    with open(per_model_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nPer-model results saved: {per_model_path}")

    # -----------------------------------------------------------------------
    # Aggregate multi-model JSON (append/merge if others exist)
    # -----------------------------------------------------------------------
    agg_path = output_dir / "moral_judgment.json"
    if agg_path.exists():
        with open(agg_path) as f:
            agg = json.load(f)
    else:
        agg = {}

    agg[args.model_short] = output

    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"Aggregated results saved: {agg_path}")

    # -----------------------------------------------------------------------
    # Plot (single model; full 2x2 when all 4 are done)
    # -----------------------------------------------------------------------
    try:
        plot_steering_curves(agg, str(figures_dir / "moral_steering_curve.png"))
    except Exception as e:
        print(f"Warning: plotting failed ({e}); run again with --aggregate_json "
              f"after all models finish.")

    print("\nDone.")


if __name__ == "__main__":
    main()
