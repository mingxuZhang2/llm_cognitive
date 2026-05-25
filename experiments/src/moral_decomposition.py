"""
The linchpin experiment: test whether 'moral neurons' identified by the multi-domain
dissociation pipeline encode genuine moral cognition or are just negative-affect neurons.

Inputs
------
1. The pre-computed attribution file from multi_function_dissociation.py:
   `{model_short}_multi_attribution.npz` — contains per-category selectivity vectors
   for emotion, moral, tom, self, neutral_control.
2. The paired decomposition stimuli (400 items, 200 pairs, 4 conditions x 50 pairs).

For each pair (a, b), we elicit a graded moral wrongness judgment from the model.
We then compute pair discrimination = E[wrongness(a)] - E[wrongness(b)].

The prediction
--------------
- intent and norm_type pairs:    a (intentional / moral-violation) should be rated
                                 substantially higher than b. discrimination > 0.
- outcome pairs:                 a (completed) only slightly higher than b (attempted).
- morality pairs:                a (moral-negative) much higher than b (nonmoral-negative).

When we ablate moral-selective neurons vs emotion-selective neurons vs random:

  If moral neurons are genuine moral-cognition components:
    - moral ablation collapses INTENT and NORM_TYPE discriminations
      (these require recognizing agency, blameworthiness, normative wrongness)
    - moral ablation only partially collapses MORALITY discrimination
      (since some of that signal is affective and still carried by emotion neurons)
    - moral ablation barely affects OUTCOME discrimination
      (which depends on perceived harm severity, an affective property)
    - emotion ablation reduces MORALITY and OUTCOME discriminations
      (these depend on negative-affect representation)
    - emotion ablation preserves INTENT and NORM_TYPE discriminations
      (since these depend on agency/norm reasoning, not affect)

  If 'moral neurons' are actually just emotion neurons:
    - moral ablation and emotion ablation will produce nearly identical patterns
    - all four discriminations will collapse proportionally to baseline negative-affect signal

This script outputs the full crosstab (ablation x condition x discrimination) so the
compositional claim can be evaluated directly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Wrongness rating prompt — fixed template, deterministic
PROMPT_TEMPLATE = (
    "Read the scenario and rate how morally wrong the described action is, "
    "on a scale from 1 (not wrong at all) to 7 (extremely wrong).\n\n"
    "Scenario: {text}\n\nRating (1-7):"
)
RATING_TOKENS = [str(i) for i in range(1, 8)]  # "1" ... "7"


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    return None


def install_ablation_hooks(model, neuron_indices, n_layers, ffn_dim, device):
    """Zero out the specified gate_proj output dimensions via forward hooks.

    Returns the list of hook handles so the caller can remove them.
    """
    layers = get_layers(model)
    neuron_set = set(int(n) for n in neuron_indices)
    hooks = []
    for layer_idx in range(n_layers):
        start = layer_idx * ffn_dim
        end = start + ffn_dim
        layer_neurons = [n - start for n in neuron_set if start <= n < end]
        if not layer_neurons:
            continue
        mlp = layers[layer_idx].mlp
        if hasattr(mlp, "gate_proj"):
            target = mlp.gate_proj
        elif hasattr(mlp, "dense_h_to_4h"):
            target = mlp.dense_h_to_4h
        else:
            continue
        mask = torch.zeros(ffn_dim, dtype=torch.bool, device=device)
        mask[layer_neurons] = True

        def make_hook(m):
            def hook_fn(module, _input, output):
                output[:, :, m] = 0.0
                return output
            return hook_fn

        hooks.append(target.register_forward_hook(make_hook(mask)))
    return hooks


def compute_wrongness_ratings(model, tokenizer, stimuli, rating_token_ids,
                              device="cuda", max_length=384):
    """Return an N-length array of expected wrongness ratings (in [1, 7])."""
    model.eval()
    expected = np.zeros(len(stimuli), dtype=np.float64)
    pdists = np.zeros((len(stimuli), 7), dtype=np.float32)

    for i, sample in enumerate(stimuli):
        prompt = PROMPT_TEMPLATE.format(text=sample["text"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]  # next-token logits
        # Restrict to the 7 rating tokens and renormalize
        rating_logits = logits[rating_token_ids]
        probs = torch.softmax(rating_logits, dim=-1).cpu().float().numpy()
        pdists[i] = probs
        expected[i] = float((probs * np.arange(1, 8)).sum())

    return expected, pdists


def pair_discrimination(stimuli, ratings):
    """Return per-condition mean discrimination = mean over pairs of (rating_a - rating_b).

    Also returns the full per-pair table for downstream stats.
    """
    pair_table = defaultdict(dict)  # pair_id -> {subcondition: rating, condition: ..., a_or_b: ...}
    for s, r in zip(stimuli, ratings):
        pid = s["pair_id"]
        sub = s["subcondition"]
        a_or_b = s["id"][-1]  # last char "a" or "b"
        pair_table[pid][a_or_b] = {"sub": sub, "rating": float(r), "condition": s["condition"]}

    per_condition = defaultdict(list)
    pairs = []
    for pid, members in pair_table.items():
        if "a" not in members or "b" not in members:
            continue
        cond = members["a"]["condition"]
        delta = members["a"]["rating"] - members["b"]["rating"]
        per_condition[cond].append(delta)
        pairs.append({"pair_id": pid, "condition": cond, "delta": delta,
                      "a_rating": members["a"]["rating"],
                      "b_rating": members["b"]["rating"],
                      "a_sub": members["a"]["sub"], "b_sub": members["b"]["sub"]})
    summary = {c: {"mean": float(np.mean(v)),
                   "std": float(np.std(v)),
                   "n": len(v),
                   "median": float(np.median(v))}
               for c, v in per_condition.items()}
    return summary, pairs


def select_top_neurons(npz_path, category_key, n_ablate):
    """Load attribution.npz and return top-N selectivity indices for a category."""
    data = np.load(npz_path)
    key = f"{category_key}_selectivity"
    if key not in data.files:
        raise KeyError(f"{key} not found in {npz_path}. Available: {data.files}")
    sel = data[key]
    return np.argsort(-sel)[:n_ablate].astype(np.int64), sel


def run(model_path, model_short, attribution_npz, decomposition_jsonl,
        meta_path, output_dir, n_ablate=5000, device="cuda"):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Moral Decomposition Experiment: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim
    print(f"  Model dims: {n_layers} layers x {ffn_dim} FFN = {n_neurons:,} neurons")
    print(f"  Ablation budget: {n_ablate} ({n_ablate/n_neurons*100:.2f}% of network)")

    with open(decomposition_jsonl) as f:
        stimuli = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(stimuli)} decomposition items "
          f"({len(set(s['condition'] for s in stimuli))} conditions)")

    print("\n[Step 1] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve rating token IDs once. Models tokenize " 1" differently than "1", so
    # we use the prompt's leading space context to pick the right tokenization.
    rating_token_ids = []
    for tok in RATING_TOKENS:
        # encode " 1" — the leading space mirrors what follows ":" in the prompt
        ids = tokenizer.encode(" " + tok, add_special_tokens=False)
        if len(ids) == 1:
            rating_token_ids.append(ids[0])
        else:
            ids2 = tokenizer.encode(tok, add_special_tokens=False)
            rating_token_ids.append(ids2[-1])
    print(f"  Rating token ids: {rating_token_ids}")

    # Baseline
    print("\n[Step 2] Baseline wrongness ratings (no ablation)...")
    base_ratings, _ = compute_wrongness_ratings(
        model, tokenizer, stimuli, rating_token_ids, device,
    )
    base_summary, base_pairs = pair_discrimination(stimuli, base_ratings)
    print("  Baseline discrimination by condition:")
    for cond in ["intent", "outcome", "norm_type", "morality"]:
        s = base_summary.get(cond, {})
        print(f"    {cond:>12s}: delta(a-b) = {s.get('mean', float('nan')):+.3f} "
              f"(std {s.get('std', float('nan')):.2f}, n={s.get('n', 0)})")

    # Ablation conditions
    ablation_targets = ["moral", "emotion", "tom", "self", "neutral_control"]
    ablation_results = {}

    for target in ablation_targets:
        print(f"\n[Step 3.{target}] Ablating top-{n_ablate} {target}-selective neurons...")
        try:
            top_neurons, _sel = select_top_neurons(
                attribution_npz, target, n_ablate)
        except KeyError as e:
            print(f"  SKIP: {e}")
            continue
        hooks = install_ablation_hooks(
            model, top_neurons, n_layers, ffn_dim, device)
        ratings, _ = compute_wrongness_ratings(
            model, tokenizer, stimuli, rating_token_ids, device)
        for h in hooks:
            h.remove()
        summary, pairs = pair_discrimination(stimuli, ratings)
        ablation_results[target] = {"summary": summary, "pairs": pairs}
        for cond in ["intent", "outcome", "norm_type", "morality"]:
            base_m = base_summary[cond]["mean"]
            new_m = summary[cond]["mean"]
            delta = new_m - base_m
            shrink_pct = (1 - new_m / base_m) * 100 if abs(base_m) > 1e-6 else float("nan")
            print(f"    {cond:>12s}: {base_m:+.3f} -> {new_m:+.3f}  "
                  f"(delta {delta:+.3f}, shrink {shrink_pct:+.1f}%)")

    # Random control
    print(f"\n[Step 4] Random ablation control ({n_ablate} neurons, seed=42)...")
    rng = np.random.RandomState(42)
    rand_neurons = rng.choice(n_neurons, n_ablate, replace=False)
    hooks = install_ablation_hooks(
        model, rand_neurons, n_layers, ffn_dim, device)
    rand_ratings, _ = compute_wrongness_ratings(
        model, tokenizer, stimuli, rating_token_ids, device)
    for h in hooks:
        h.remove()
    rand_summary, rand_pairs = pair_discrimination(stimuli, rand_ratings)
    for cond in ["intent", "outcome", "norm_type", "morality"]:
        base_m = base_summary[cond]["mean"]
        new_m = rand_summary[cond]["mean"]
        print(f"    {cond:>12s}: {base_m:+.3f} -> {new_m:+.3f}  "
              f"(delta {new_m - base_m:+.3f})")

    # Compositional verdict
    print(f"\n{'='*70}")
    print("COMPOSITIONAL FINDING CHECK")
    print(f"{'='*70}")

    def shrink_pct(cond, target):
        b = base_summary[cond]["mean"]
        a = ablation_results[target]["summary"][cond]["mean"]
        return (1 - a / b) * 100 if abs(b) > 1e-6 else float("nan")

    if "moral" in ablation_results and "emotion" in ablation_results:
        print(f"  {'Condition':>12s} | {'moral abl':>10s} | {'emotion abl':>11s} | "
              f"{'random abl':>10s} | verdict")
        print(f"  {'-'*12} + {'-'*10} + {'-'*11} + {'-'*10} + {'-'*20}")
        for cond in ["intent", "outcome", "norm_type", "morality"]:
            moral_s = shrink_pct(cond, "moral")
            emo_s = shrink_pct(cond, "emotion")
            base_m = base_summary[cond]["mean"]
            rand_m = rand_summary[cond]["mean"]
            rand_s = (1 - rand_m / base_m) * 100 if abs(base_m) > 1e-6 else float("nan")
            if cond in ("intent", "norm_type"):
                verdict = "moral > emotion?" if moral_s > emo_s else "FAIL"
            elif cond == "outcome":
                verdict = "emotion >= moral?" if emo_s >= moral_s else "ambiguous"
            else:  # morality
                verdict = "both reduce"
            print(f"  {cond:>12s} | {moral_s:>9.1f}% | {emo_s:>10.1f}% | "
                  f"{rand_s:>9.1f}% | {verdict}")

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - t0
    payload = {
        "model": model_short,
        "n_ablate": n_ablate,
        "n_neurons": n_neurons,
        "baseline": {"summary": base_summary, "pairs": base_pairs},
        "ablation_results": ablation_results,
        "random_control": {"summary": rand_summary, "pairs": rand_pairs},
        "elapsed_seconds": elapsed,
    }
    out_path = Path(output_dir) / f"{model_short}_moral_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved to {out_path} (elapsed {elapsed:.0f}s)")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--attribution_npz", required=True,
                    help="Output from multi_function_dissociation.py "
                         "(e.g., {model}_multi_attribution.npz)")
    ap.add_argument("--decomposition_jsonl", required=True,
                    help="Path to decomposition_stimuli.jsonl (400 paired items)")
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_ablate", type=int, default=5000)
    args = ap.parse_args()

    run(args.model_path, args.model_short, args.attribution_npz,
        args.decomposition_jsonl, args.meta_path, args.output_dir, args.n_ablate)


if __name__ == "__main__":
    main()
