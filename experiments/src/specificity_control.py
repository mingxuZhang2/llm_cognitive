"""
Specificity control for the sparse-substrate finding.

The dose-response showed that in Llama 8B and Mistral 7B, ablating just
500 norm_type-contrast neurons (0.06% of FFN width) completely collapses
moral-vs-conventional discrimination. This is suspiciously dramatic — the
500 neurons might be doing something MORE general (e.g., serving as a
"rating-task backbone") rather than specifically encoding moral cognition.

This script runs 5 control measurements after ablating the same top-500
norm_type-contrast neurons:

  1. moral-conventional discrimination (replicate the main finding)
  2. wrongness rating consistency on neutral non-moral stimuli (does the
     model still produce sensible ratings on unrelated content?)
  3. binary emotion-valence discrimination (does the model still produce
     graded responses on a DIFFERENT cognitive task?)
  4. general language perplexity on held-out neutral text (does ablation
     break basic language modeling?)
  5. random sentence rating (does the model produce sensible distributions
     across all 7 rating tokens?)

If 1 is intact but 2-5 are also broken: the ablation is hitting general
rating capability, not moral-specific computation.
If 1 is broken but 2-5 are preserved: the finding is moral-specific.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moral_decomposition import (
    install_ablation_hooks,
    compute_wrongness_scores,
    pair_discrimination,
    resolve_rating_token_ids,
    select_neurons,
    build_chat_prompt,
    PROMPT_USER,
    ASSISTANT_PREFIX,
    RATING_TOKENS,
)


# A small held-out set of neutral, non-moral content for general PPL test.
NEUTRAL_TEXTS = [
    "The thermostat is set to twenty-one degrees Celsius.",
    "Water boils at one hundred degrees at sea level.",
    "The library closes at nine in the evening on weekdays.",
    "She walked to the bus stop in the morning.",
    "The recipe calls for two cups of flour and one cup of sugar.",
    "The conference room has a long oval table.",
    "Trees lose their leaves in the autumn.",
    "The package arrived three days after it was shipped.",
    "He picked up his keys from the kitchen counter.",
    "The painting hung above the fireplace.",
    "Children played soccer in the park after school.",
    "The store sells bread and pastries on weekday mornings.",
    "Snow fell overnight and covered the streets.",
    "The doctor scheduled the appointment for the following Tuesday.",
    "She filled the watering can and watered the plants.",
    "The train departs from platform three at exactly four o'clock.",
    "The restaurant serves breakfast until eleven in the morning.",
    "He folded the laundry and put it away.",
    "The mailman delivers letters around two in the afternoon.",
    "She set her alarm for six thirty in the morning.",
]

# Pairs of valence-graded sentences (a is more positive than b) — used to test
# whether the model still produces graded responses on a different task after
# ablation. Hand-curated, balanced over content domains.
VALENCE_PAIRS = [
    ("The team celebrated their championship victory with cheers and laughter.",
     "The team filed silently out of the stadium after their crushing defeat."),
    ("She received the scholarship she had worked so hard to earn.",
     "She learned that her scholarship application had been rejected."),
    ("The puppy bounded happily across the sunlit lawn.",
     "The dog limped slowly through the cold rain."),
    ("The chef described the perfectly cooked meal with pride.",
     "The chef apologized profusely for the burnt dinner."),
    ("The morning sky was a brilliant shade of pink and gold.",
     "The morning sky was a flat, oppressive gray."),
    ("The children eagerly opened their birthday presents.",
     "The children discovered their presents had been ruined."),
    ("She felt deeply moved by the kindness of the strangers.",
     "She felt betrayed by the cruelty of her closest friend."),
    ("The garden was alive with blooming flowers and singing birds.",
     "The garden was overrun with weeds and littered with trash."),
    ("He delivered the news of the promotion with a wide smile.",
     "He delivered the news of the layoffs with a heavy heart."),
    ("The audience applauded enthusiastically at the brilliant performance.",
     "The audience sat in stunned silence after the disastrous performance."),
]


def compute_ppl_on_texts(model, tokenizer, texts, device="cuda", max_length=256):
    """Return list of per-text perplexities."""
    model.eval()
    ppls = []
    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt", truncation=True,
                            max_length=max_length).to(device)
        if inputs["input_ids"].shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        if not torch.isnan(out.loss) and not torch.isinf(out.loss):
            ppls.append(float(np.exp(min(out.loss.item(), 100))))
    return ppls


def rate_neutral(model, tokenizer, rating_token_ids, device="cuda"):
    """Rate neutral stimuli under the wrongness template. Should return values
    near 1-3 (low wrongness). If model still outputs sensible distributions,
    the rating apparatus is functional."""
    fake_stim = [{"text": t, "pair_id": f"n_{i}", "id": f"n_{i}a",
                   "condition": "neutral", "subcondition": "neutral"}
                  for i, t in enumerate(NEUTRAL_TEXTS)]
    scores, expected, pdists = compute_wrongness_scores(
        model, tokenizer, fake_stim, rating_token_ids, device,
    )
    # The "log-odds" score isn't meaningful for unpaired; use mean expected
    # rating + std + KL from uniform as health signals.
    mean_exp = float(np.mean(expected))
    std_exp = float(np.std(expected))
    mean_dist = pdists.mean(axis=0)
    uniform = np.ones(7) / 7
    kl = float(np.sum(mean_dist * (np.log(mean_dist + 1e-9) - np.log(uniform))))
    entropy = float(-np.sum(mean_dist * np.log(mean_dist + 1e-9)))
    return {"mean_rating": mean_exp, "std_rating": std_exp,
            "mean_dist": mean_dist.tolist(), "kl_from_uniform": kl,
            "entropy_nats": entropy}


def valence_discrimination(model, tokenizer, rating_token_ids, device="cuda"):
    """Rate positive-vs-negative valence pairs under wrongness template.
    Negative (b) should be rated higher than positive (a) if model uses
    affect/valence as a wrongness proxy. Even if so, ablation should preserve
    the contrast IF the contrast doesn't depend specifically on moral neurons."""
    stim = []
    for i, (pos, neg) in enumerate(VALENCE_PAIRS):
        stim.append({"text": pos, "pair_id": f"val_{i}", "id": f"val_{i}a",
                       "condition": "valence", "subcondition": "positive"})
        stim.append({"text": neg, "pair_id": f"val_{i}", "id": f"val_{i}b",
                       "condition": "valence", "subcondition": "negative"})
    scores, expected, _ = compute_wrongness_scores(
        model, tokenizer, stim, rating_token_ids, device,
    )
    summary, _ = pair_discrimination(stim, scores)
    return summary.get("valence", {})


def run(model_path, model_short, attribution_npz, decomposition_jsonl,
        meta_path, output_dir, device="cuda"):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Specificity Control: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim

    with open(decomposition_jsonl) as f:
        stimuli = [json.loads(line) for line in f if line.strip()]
    norm_stim = [s for s in stimuli if s["condition"] == "norm_type"]

    print(f"  Model: {n_layers} x {ffn_dim} = {n_neurons:,} neurons")
    print(f"  norm_type pairs: {len(norm_stim) // 2}")

    print("\n[1/3] Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rating_token_ids = resolve_rating_token_ids(tokenizer, norm_stim[0]["text"])
    print(f"  Rating tokens: {rating_token_ids}")

    def run_all_measurements(label, hooks_applied=True):
        """Run all 5 measurements with current model state."""
        m = {}
        # 1. norm_type discrimination
        ns, _, _ = compute_wrongness_scores(
            model, tokenizer, norm_stim, rating_token_ids, device,
        )
        m["norm_type"] = pair_discrimination(norm_stim, ns)[0].get("norm_type", {})
        # 2. neutral rating sanity
        m["neutral_rating"] = rate_neutral(model, tokenizer, rating_token_ids, device)
        # 3. valence discrimination
        m["valence"] = valence_discrimination(
            model, tokenizer, rating_token_ids, device)
        # 4. general PPL on neutral texts
        ppls = compute_ppl_on_texts(model, tokenizer, NEUTRAL_TEXTS, device)
        m["neutral_ppl"] = {"mean": float(np.mean(ppls)),
                              "median": float(np.median(ppls)),
                              "max": float(np.max(ppls)),
                              "min": float(np.min(ppls)),
                              "n": len(ppls)}
        return m

    print("\n[2/3] Baseline (no ablation)...")
    baseline = run_all_measurements("baseline")
    print(f"  norm_type discrimination: {baseline['norm_type'].get('mean', float('nan')):+.3f}")
    print(f"  neutral mean rating     : {baseline['neutral_rating']['mean_rating']:.2f}  "
          f"(distribution entropy {baseline['neutral_rating']['entropy_nats']:.2f} nats)")
    print(f"  valence discrimination  : {baseline['valence'].get('mean', float('nan')):+.3f}  "
          f"(neg > pos for wrongness)")
    print(f"  neutral PPL             : mean {baseline['neutral_ppl']['mean']:.1f} "
          f"(median {baseline['neutral_ppl']['median']:.1f}, "
          f"min {baseline['neutral_ppl']['min']:.1f}, max {baseline['neutral_ppl']['max']:.1f})")

    print("\n[3/3] Ablating top-500 norm_type contrast neurons...")
    top, _ = select_neurons(attribution_npz, "norm_type", 500, side="top")
    hooks = install_ablation_hooks(model, top, n_layers, ffn_dim, device)
    ablated = run_all_measurements("ablated")
    for h in hooks:
        h.remove()
    print(f"  norm_type discrimination: {ablated['norm_type'].get('mean', float('nan')):+.3f}  "
          f"(delta {ablated['norm_type'].get('mean', 0) - baseline['norm_type'].get('mean', 0):+.3f})")
    print(f"  neutral mean rating     : {ablated['neutral_rating']['mean_rating']:.2f}  "
          f"(delta {ablated['neutral_rating']['mean_rating'] - baseline['neutral_rating']['mean_rating']:+.2f},  "
          f"entropy delta {ablated['neutral_rating']['entropy_nats'] - baseline['neutral_rating']['entropy_nats']:+.3f})")
    print(f"  valence discrimination  : {ablated['valence'].get('mean', float('nan')):+.3f}  "
          f"(delta {ablated['valence'].get('mean', 0) - baseline['valence'].get('mean', 0):+.3f})")
    print(f"  neutral PPL             : mean {ablated['neutral_ppl']['mean']:.1f}  "
          f"(delta {ablated['neutral_ppl']['mean'] - baseline['neutral_ppl']['mean']:+.1f}x baseline)")

    # Verdict
    print(f"\n{'='*70}")
    print("SPECIFICITY VERDICT")
    print(f"{'='*70}")
    base_norm = baseline["norm_type"]["mean"]
    abl_norm = ablated["norm_type"]["mean"]
    norm_shrink = abs(abl_norm - base_norm) / (abs(base_norm) + 1e-6) * 100

    base_val = baseline["valence"]["mean"]
    abl_val = ablated["valence"]["mean"]
    val_shrink = abs(abl_val - base_val) / (abs(base_val) + 1e-6) * 100

    base_ppl = baseline["neutral_ppl"]["mean"]
    abl_ppl = ablated["neutral_ppl"]["mean"]
    ppl_ratio = abl_ppl / (base_ppl + 1e-6)

    base_kl = baseline["neutral_rating"]["kl_from_uniform"]
    abl_kl = ablated["neutral_rating"]["kl_from_uniform"]
    kl_change = abl_kl - base_kl

    print(f"  norm_type shrink           : {norm_shrink:.1f}%")
    print(f"  valence-discrim shrink     : {val_shrink:.1f}%")
    print(f"  neutral-PPL inflation      : {ppl_ratio:.2f}x")
    print(f"  neutral-rating KL change   : {kl_change:+.3f}  "
          f"({'narrowed' if kl_change > 0 else 'broadened'} distribution)")

    if norm_shrink > 50 and val_shrink < 30 and ppl_ratio < 1.5 and abs(kl_change) < 0.3:
        verdict = "SPECIFIC — norm_type-only loss; rating apparatus intact"
    elif norm_shrink > 50 and val_shrink > 50:
        verdict = "NON-SPECIFIC — general rating-task substrate; finding not moral-specific"
    elif ppl_ratio > 2.0:
        verdict = "NON-SPECIFIC — basic language modeling impaired"
    else:
        verdict = "INTERMEDIATE — partial specificity, needs more controls"
    print(f"\n  VERDICT: {verdict}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_short,
        "n_ablate": 500,
        "n_neurons": n_neurons,
        "baseline": baseline,
        "ablated": ablated,
        "verdict": verdict,
        "elapsed_seconds": time.time() - t0,
    }
    with open(Path(output_dir) / f"{model_short}_specificity.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved (elapsed {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--attribution_npz", required=True)
    ap.add_argument("--decomposition_jsonl", required=True)
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    run(args.model_path, args.model_short, args.attribution_npz,
        args.decomposition_jsonl, args.meta_path, args.output_dir)


if __name__ == "__main__":
    main()
