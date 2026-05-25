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

# Wrongness rating prompt — applied through chat template, then prefilled with
# a deterministic prefix so the next token is naturally a rating digit.
# Chat-tuned models are heavily RLHF-trained to expect chat formatting; feeding
# raw text drops them into out-of-distribution territory where their next-token
# distribution does not reflect aligned policy behavior.
PROMPT_USER = (
    "Rate how morally wrong the following scenario is, on a scale from 1 to 7 "
    "(1 = not wrong at all, 7 = extremely wrong). Reply with only a single digit.\n\n"
    "Scenario: {text}"
)
# Prefix injected at the start of the assistant turn so the next-token logits land
# on a digit position rather than on a natural-language preamble like "I" or "The".
ASSISTANT_PREFIX = "Rating: "
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


def build_chat_prompt(tokenizer, user_text: str, assistant_prefix: str) -> str:
    """Wrap a user message in the model's chat template and prefill the assistant
    turn with a deterministic prefix so the next-token logits land on a rating digit.

    Falls back to plain concatenation if the tokenizer has no chat template (rare
    for the production models, but defensive).
    """
    messages = [{"role": "user", "content": user_text}]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return user_text + "\n\n" + assistant_prefix
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return rendered + assistant_prefix


def resolve_rating_token_ids(tokenizer, sample_text: str):
    """Return the seven token IDs that the model would emit at the wrongness slot.

    Builds an actual chat-template + assistant-prefix prompt for a representative
    sample, then for each digit "1".."7" tokenizes ``prompt + digit`` and finds the
    token at the diverging position. This handles SentencePiece/SPIECE_UNDERLINE
    quirks across Qwen / Llama / Mistral / Gemma uniformly.
    """
    prompt = build_chat_prompt(
        tokenizer, PROMPT_USER.format(text=sample_text), ASSISTANT_PREFIX,
    )
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    rating_token_ids = []
    for digit in RATING_TOKENS:
        full = tokenizer.encode(prompt + digit, add_special_tokens=False)
        # Find first divergence
        idx = None
        for i in range(min(len(base_ids), len(full))):
            if base_ids[i] != full[i]:
                idx = i
                break
        if idx is None and len(full) > len(base_ids):
            idx = len(base_ids)
        if idx is None:
            raise RuntimeError(
                f"Could not resolve rating token for digit {digit!r} on this model")
        rating_token_ids.append(full[idx])
    return rating_token_ids


def compute_wrongness_scores(model, tokenizer, stimuli, rating_token_ids,
                              device="cuda", max_length=512):
    """Return per-stimulus wrongness scores.

    We use the log-odds of "wrong" vs "not wrong":
        score = logsumexp(logits[5,6,7]) - logsumexp(logits[1,2,3])
    where indices refer to the 7 rating-digit tokens. Compared to expected value
    (sum p_i * i), log-odds:
      - is unbounded (no ceiling/floor at extreme baseline)
      - is approximately additive in the same-direction shifts that ablation produces
      - directly mirrors the binary "wrong / not wrong" discrimination that the
        compositional hypothesis predicts

    Also returned: the full 7-class softmax distribution per stimulus, plus the
    expected-value rating (for diagnostic comparison with the v1 metric).
    """
    model.eval()
    scores = np.zeros(len(stimuli), dtype=np.float64)
    expected = np.zeros(len(stimuli), dtype=np.float64)
    pdists = np.zeros((len(stimuli), 7), dtype=np.float32)

    rating_id_t = torch.tensor(rating_token_ids, device=device)
    # Indices into the 7-rating logits for "wrong"=5,6,7 (positions 4,5,6) and
    # "not wrong"=1,2,3 (positions 0,1,2). The middle rating (4) is excluded.
    wrong_idx = torch.tensor([4, 5, 6], device=device)
    not_wrong_idx = torch.tensor([0, 1, 2], device=device)

    for i, sample in enumerate(stimuli):
        prompt = build_chat_prompt(
            tokenizer, PROMPT_USER.format(text=sample["text"]), ASSISTANT_PREFIX,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]  # next-token logits

        rating_logits = logits.index_select(0, rating_id_t)  # shape [7]
        # Log-odds without first restricting+renormalizing — these logits live in
        # the model's full vocabulary, so logsumexp is well-defined and the
        # vocabulary-wide normalizer cancels in the difference.
        score = torch.logsumexp(rating_logits.index_select(0, wrong_idx), dim=0) \
                - torch.logsumexp(rating_logits.index_select(0, not_wrong_idx), dim=0)
        scores[i] = float(score.item())

        probs = torch.softmax(rating_logits, dim=-1).cpu().float().numpy()
        pdists[i] = probs
        expected[i] = float((probs * np.arange(1, 8)).sum())

    return scores, expected, pdists


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


def select_neurons(npz_path, category_key, n_ablate, side="top"):
    """Load attribution.npz and return N selectivity indices for a category.

    side="top":    n_ablate most-positive-selectivity neurons (target-specific)
    side="bottom": n_ablate most-negative-selectivity neurons (anti-target;
                   should NOT carry the cognitive function if the atlas is real)
    """
    data = np.load(npz_path)
    key = f"{category_key}_selectivity"
    if key not in data.files:
        raise KeyError(f"{key} not found in {npz_path}. Available: {data.files}")
    sel = data[key]
    if side == "top":
        return np.argsort(-sel)[:n_ablate].astype(np.int64), sel
    elif side == "bottom":
        return np.argsort(sel)[:n_ablate].astype(np.int64), sel
    else:
        raise ValueError(f"side must be 'top' or 'bottom', got {side}")


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

    # Resolve rating token IDs by building the actual chat-template prompt with
    # the assistant prefix, then encoding `prompt + digit` for each digit and
    # picking the divergence position. This is robust across BPE / SentencePiece /
    # special-prefix tokenizations on Qwen, Llama-3, Mistral, Gemma.
    rating_token_ids = resolve_rating_token_ids(tokenizer, stimuli[0]["text"])
    surface = tokenizer.convert_ids_to_tokens(rating_token_ids)
    print(f"  Rating token ids: {rating_token_ids}")
    print(f"  Surface forms   : {surface}")
    # Sanity check: every surface form must contain a single digit 1-7
    for d, t in zip(RATING_TOKENS, surface):
        assert d in t, f"Rating token for {d!r} resolved to {t!r} — not a digit token!"

    # Baseline
    print("\n[Step 2] Baseline wrongness scores (no ablation)...")
    base_scores, base_expected, _ = compute_wrongness_scores(
        model, tokenizer, stimuli, rating_token_ids, device,
    )
    base_summary, base_pairs = pair_discrimination(stimuli, base_scores)
    base_exp_summary, _ = pair_discrimination(stimuli, base_expected)
    print("  Baseline log-odds discrimination by condition:")
    for cond in ["intent", "outcome", "norm_type", "morality"]:
        s = base_summary.get(cond, {})
        e = base_exp_summary.get(cond, {})
        print(f"    {cond:>12s}: log-odds delta = {s.get('mean', float('nan')):+.3f} "
              f"(std {s.get('std', float('nan')):.2f}, n={s.get('n', 0)})  "
              f"[diagnostic E[r] delta = {e.get('mean', float('nan')):+.3f}]")

    # Ablation conditions. The "_anti" suffix denotes the bottom-k (most-negative
    # selectivity) sanity check — those neurons should NOT carry moral cognition,
    # so ablating them should leave the discrimination basically unchanged.
    ablation_targets = [
        ("moral",            "top"),
        ("moral_anti",       "bottom_moral"),  # diagnostic
        ("emotion",          "top"),
        ("tom",              "top"),
        ("self",             "top"),
        ("neutral_control",  "top"),
    ]
    ablation_results = {}

    for target, side in ablation_targets:
        base_category = target.replace("_anti", "")
        select_side = "bottom" if side == "bottom_moral" else "top"
        label = f"bottom-{n_ablate}" if select_side == "bottom" else f"top-{n_ablate}"
        print(f"\n[Step 3.{target}] Ablating {label} {base_category}-selective neurons...")
        try:
            neurons, _sel = select_neurons(
                attribution_npz, base_category, n_ablate, side=select_side)
        except KeyError as e:
            print(f"  SKIP: {e}")
            continue
        hooks = install_ablation_hooks(
            model, neurons, n_layers, ffn_dim, device)
        scores, expected_vals, _ = compute_wrongness_scores(
            model, tokenizer, stimuli, rating_token_ids, device)
        for h in hooks:
            h.remove()
        summary, pairs = pair_discrimination(stimuli, scores)
        exp_summary, _ = pair_discrimination(stimuli, expected_vals)
        ablation_results[target] = {
            "summary": summary, "pairs": pairs,
            "expected_summary": exp_summary,
        }
        for cond in ["intent", "outcome", "norm_type", "morality"]:
            base_m = base_summary[cond]["mean"]
            new_m = summary[cond]["mean"]
            delta = new_m - base_m
            shrink_pct = (1 - new_m / base_m) * 100 if abs(base_m) > 1e-6 else float("nan")
            print(f"    {cond:>12s}: {base_m:+.3f} -> {new_m:+.3f}  "
                  f"(delta {delta:+.3f}, shrink {shrink_pct:+.1f}%)")

    # Random control — use 3 seeds to estimate noise floor properly
    print(f"\n[Step 4] Random ablation controls ({n_ablate} neurons, seeds 42/43/44)...")
    random_runs = []
    for seed in [42, 43, 44]:
        rng = np.random.RandomState(seed)
        rand_neurons = rng.choice(n_neurons, n_ablate, replace=False)
        hooks = install_ablation_hooks(
            model, rand_neurons, n_layers, ffn_dim, device)
        rand_scores, rand_expected, _ = compute_wrongness_scores(
            model, tokenizer, stimuli, rating_token_ids, device)
        for h in hooks:
            h.remove()
        rs, _ = pair_discrimination(stimuli, rand_scores)
        random_runs.append({"seed": seed, "summary": rs})
        print(f"  seed={seed}: " + ", ".join(
            f"{c}={rs[c]['mean']:+.3f}" for c in ["intent", "outcome", "norm_type", "morality"]))
    # Mean across seeds
    rand_summary = {}
    for cond in ["intent", "outcome", "norm_type", "morality"]:
        vals = [r["summary"][cond]["mean"] for r in random_runs]
        rand_summary[cond] = {"mean": float(np.mean(vals)),
                              "std": float(np.std(vals)),
                              "n_seeds": len(vals)}
    print(f"  Random ablation mean across 3 seeds (noise floor):")
    for cond in ["intent", "outcome", "norm_type", "morality"]:
        base_m = base_summary[cond]["mean"]
        rm = rand_summary[cond]["mean"]
        print(f"    {cond:>12s}: {base_m:+.3f} -> {rm:+.3f}  "
              f"(delta {rm - base_m:+.3f} ± {rand_summary[cond]['std']:.3f})")

    # Compositional verdict
    print(f"\n{'='*70}")
    print("COMPOSITIONAL FINDING CHECK (log-odds metric, signed deltas)")
    print(f"{'='*70}")

    def delta(cond, target):
        """Return (ablated - baseline) for the log-odds metric — negative means
        the ablation REDUCED the model's wrongness discrimination (the predicted
        direction for genuine cognitive ablation)."""
        b = base_summary[cond]["mean"]
        a = ablation_results[target]["summary"][cond]["mean"]
        return a - b

    have_all = all(t in ablation_results for t in
                   ["moral", "moral_anti", "emotion", "tom", "self", "neutral_control"])
    if have_all:
        targets_to_show = ["moral", "moral_anti", "emotion", "tom",
                           "self", "neutral_control"]
        header = f"  {'Condition':>12s} | {'baseline':>9s} |" + \
                 "".join(f" {t[:9]:>9s}" for t in targets_to_show) + \
                 f" | {'random (μ±σ)':>14s}"
        print(header)
        print(f"  {'-'*12} + {'-'*9} + " + " ".join(["-"*9]*6) + f" + {'-'*14}")
        for cond in ["intent", "outcome", "norm_type", "morality"]:
            b = base_summary[cond]["mean"]
            row = f"  {cond:>12s} | {b:>+9.3f} |"
            for t in targets_to_show:
                row += f" {delta(cond, t):>+9.3f}"
            rm = rand_summary[cond]["mean"] - b
            rs = rand_summary[cond]["std"]
            row += f" | {rm:>+7.3f}±{rs:.2f}"
            print(row)

        # Statistical check: is moral ablation farther from baseline than random?
        # Z-score = (|moral_delta| - 0) / random_std,  large means real signal
        print(f"\n  Z-scores vs random noise floor (|moral_delta| / random_std):")
        for cond in ["intent", "outcome", "norm_type", "morality"]:
            m_d = abs(delta(cond, "moral"))
            anti_d = abs(delta(cond, "moral_anti"))
            rs = rand_summary[cond]["std"] + 1e-6
            print(f"    {cond:>12s}: moral z = {m_d / rs:>5.2f}, "
                  f"anti-moral z = {anti_d / rs:>5.2f}  "
                  f"({'GOOD: moral > anti-moral' if m_d > anti_d + 0.1 else 'WEAK or equal'})")

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - t0
    payload = {
        "model": model_short,
        "n_ablate": n_ablate,
        "n_neurons": n_neurons,
        "metric": "log-odds wrongness (logsumexp(P[5,6,7]) - logsumexp(P[1,2,3]))",
        "prompt": {"user": PROMPT_USER, "assistant_prefix": ASSISTANT_PREFIX,
                    "rating_token_ids": rating_token_ids,
                    "rating_token_surface": surface},
        "baseline": {"summary": base_summary,
                      "expected_summary": base_exp_summary,
                      "pairs": base_pairs},
        "ablation_results": ablation_results,
        "random_control": {"summary": rand_summary, "runs": random_runs},
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
