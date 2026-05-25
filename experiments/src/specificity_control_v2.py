"""
Specificity control for the v2 moral atlas (one-vs-rest moral_selectivity, n=5000).

The v3 contrast atlas at n=500 was just demonstrated to be a confound: it kills
general language modeling (PPL up 117x in Llama, 1.73x in Mistral). We need to
check whether the v2 atlas (top-5000 of moral_selectivity from the 5-way
one-vs-rest pilot) is also non-specific or genuinely moral-targeted.

The same 5 measurements as specificity_control.py, but reads from the v1
multi_attribution.npz and ablates top-5000 moral_selectivity neurons.

Importantly, we ALSO test ablating top-5000 neutral_control_selectivity neurons
as a "neutral atlas" comparison. If both produce the same impairment pattern,
v2 isn't moral-specific. If moral ablation produces stronger norm_type effects
than neutral ablation does on neutral content (and vice versa), we have a real
double dissociation.
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
)

from specificity_control import (
    NEUTRAL_TEXTS,
    VALENCE_PAIRS,
    compute_ppl_on_texts,
    rate_neutral,
    valence_discrimination,
)


def run(model_path, model_short, attribution_npz_v2, decomposition_jsonl,
        meta_path, output_dir, device="cuda"):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"v2 Specificity Control: {model_short}")
    print(f"  Atlas source: v1 multi_attribution (5-way one-vs-rest)")
    print(f"  Ablation: top-5000 of MORAL_selectivity vs NEUTRAL_CONTROL_selectivity")
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

    print("\n[1/4] Loading model + tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rating_token_ids = resolve_rating_token_ids(tokenizer, norm_stim[0]["text"])

    def run_all_measurements():
        m = {}
        ns, _, _ = compute_wrongness_scores(
            model, tokenizer, norm_stim, rating_token_ids, device,
        )
        m["norm_type"] = pair_discrimination(norm_stim, ns)[0].get("norm_type", {})
        m["neutral_rating"] = rate_neutral(model, tokenizer, rating_token_ids, device)
        m["valence"] = valence_discrimination(
            model, tokenizer, rating_token_ids, device)
        ppls = compute_ppl_on_texts(model, tokenizer, NEUTRAL_TEXTS, device)
        m["neutral_ppl"] = {"mean": float(np.mean(ppls)),
                             "median": float(np.median(ppls)),
                             "max": float(np.max(ppls)),
                             "min": float(np.min(ppls))}
        return m

    print("\n[2/4] Baseline...")
    baseline = run_all_measurements()
    print(f"  norm_type discrim : {baseline['norm_type'].get('mean', 0):+.3f}")
    print(f"  neutral rating    : {baseline['neutral_rating']['mean_rating']:.2f}")
    print(f"  valence discrim   : {baseline['valence'].get('mean', 0):+.3f}")
    print(f"  neutral PPL       : {baseline['neutral_ppl']['mean']:.1f}")

    print("\n[3/4] Ablating top-5000 MORAL_selectivity (v1 atlas)...")
    moral_neurons, _ = select_neurons(attribution_npz_v2, "moral", 5000, side="top")
    hooks = install_ablation_hooks(model, moral_neurons, n_layers, ffn_dim, device)
    after_moral = run_all_measurements()
    for h in hooks:
        h.remove()
    print(f"  norm_type discrim : {after_moral['norm_type'].get('mean', 0):+.3f}  "
          f"(delta {after_moral['norm_type'].get('mean', 0) - baseline['norm_type'].get('mean', 0):+.3f})")
    print(f"  neutral rating    : {after_moral['neutral_rating']['mean_rating']:.2f}  "
          f"(delta {after_moral['neutral_rating']['mean_rating'] - baseline['neutral_rating']['mean_rating']:+.2f})")
    print(f"  valence discrim   : {after_moral['valence'].get('mean', 0):+.3f}  "
          f"(delta {after_moral['valence'].get('mean', 0) - baseline['valence'].get('mean', 0):+.3f})")
    print(f"  neutral PPL       : {after_moral['neutral_ppl']['mean']:.1f}  "
          f"(ratio {after_moral['neutral_ppl']['mean']/baseline['neutral_ppl']['mean']:.2f}x)")

    print("\n[4/4] Ablating top-5000 NEUTRAL_CONTROL_selectivity (v1 atlas)...")
    neutral_neurons, _ = select_neurons(
        attribution_npz_v2, "neutral_control", 5000, side="top")
    hooks = install_ablation_hooks(model, neutral_neurons, n_layers, ffn_dim, device)
    after_neutral = run_all_measurements()
    for h in hooks:
        h.remove()
    print(f"  norm_type discrim : {after_neutral['norm_type'].get('mean', 0):+.3f}  "
          f"(delta {after_neutral['norm_type'].get('mean', 0) - baseline['norm_type'].get('mean', 0):+.3f})")
    print(f"  neutral rating    : {after_neutral['neutral_rating']['mean_rating']:.2f}  "
          f"(delta {after_neutral['neutral_rating']['mean_rating'] - baseline['neutral_rating']['mean_rating']:+.2f})")
    print(f"  valence discrim   : {after_neutral['valence'].get('mean', 0):+.3f}")
    print(f"  neutral PPL       : {after_neutral['neutral_ppl']['mean']:.1f}  "
          f"(ratio {after_neutral['neutral_ppl']['mean']/baseline['neutral_ppl']['mean']:.2f}x)")

    print(f"\n{'='*70}")
    print("DOUBLE DISSOCIATION CHECK")
    print(f"{'='*70}")
    print(f"  predicted: moral atlas hurts norm_type more than neutral atlas does,")
    print(f"             AND neutral atlas hurts neutral PPL more than moral atlas does")
    b_norm = baseline["norm_type"].get("mean", 0)
    moral_on_norm = after_moral["norm_type"].get("mean", 0) - b_norm
    neutral_on_norm = after_neutral["norm_type"].get("mean", 0) - b_norm
    b_ppl = baseline["neutral_ppl"]["mean"]
    moral_on_ppl = after_moral["neutral_ppl"]["mean"] / b_ppl
    neutral_on_ppl = after_neutral["neutral_ppl"]["mean"] / b_ppl
    print()
    print(f"  norm_type delta — moral atlas: {moral_on_norm:+.3f}  |  neutral atlas: {neutral_on_norm:+.3f}")
    print(f"  neutral PPL ratio — moral atlas: {moral_on_ppl:.2f}x  |  neutral atlas: {neutral_on_ppl:.2f}x")
    if abs(moral_on_norm) > abs(neutral_on_norm) + 0.5 and moral_on_ppl < neutral_on_ppl - 0.1:
        verdict = "DOUBLE DISSOCIATION ✓ — atlas effects are domain-specific"
    elif abs(moral_on_norm) > abs(neutral_on_norm) + 0.5:
        verdict = "PARTIAL — moral atlas more harmful to norm_type, but PPL impact comparable"
    elif moral_on_ppl > 2 and neutral_on_ppl > 2:
        verdict = "BOTH ATLASES BREAK LANGUAGE — finding is non-specific"
    else:
        verdict = "NO DISSOCIATION — atlases produce similar effects"
    print(f"\n  VERDICT: {verdict}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_short,
        "n_ablate": 5000,
        "baseline": baseline,
        "after_moral_ablation": after_moral,
        "after_neutral_ablation": after_neutral,
        "verdict": verdict,
        "elapsed_seconds": time.time() - t0,
    }
    with open(Path(output_dir) / f"{model_short}_specificity_v2.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved (elapsed {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--attribution_npz_v2", required=True,
                    help="Output of multi_function_dissociation (v1 5-way pilot)")
    ap.add_argument("--decomposition_jsonl", required=True)
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    run(args.model_path, args.model_short, args.attribution_npz_v2,
        args.decomposition_jsonl, args.meta_path, args.output_dir)


if __name__ == "__main__":
    main()
