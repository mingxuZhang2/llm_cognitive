"""
Dose-response for the norm_type contrast atlas.

For each n_ablate in [500, 1000, 2000, 5000, 10000, 20000]:
  - Ablate top-n_ablate norm_type contrast neurons → measure norm_type discrim
  - Ablate bottom-n_ablate (anti) → measure norm_type discrim
  - Ablate n_ablate random neurons (mean of 3 seeds) → measure norm_type discrim

If the top-k effect grows monotonically with n_ablate while bot-k stays near zero
and random scales as expected from noise, the moral-vs-conventional substrate is
real and dose-dependent — not an artifact of fixed k=5000.

Reuses contrast_attribution.npz from job 313722.
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
    RATING_TOKENS,
)

DOSES = [500, 1000, 2000, 5000, 10000, 20000]
COND = "norm_type"  # focus condition


def run(model_path, model_short, attribution_npz, decomposition_jsonl,
        meta_path, output_dir, device="cuda"):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Norm-type Dose-Response: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim
    print(f"  Model: {n_layers} x {ffn_dim} = {n_neurons:,} neurons")
    print(f"  Doses: {DOSES}")

    with open(decomposition_jsonl) as f:
        stimuli = [json.loads(line) for line in f if line.strip()]
    norm_stimuli = [s for s in stimuli if s["condition"] == COND]
    print(f"  norm_type stimuli: {len(norm_stimuli)} ({len(norm_stimuli)//2} pairs)")

    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rating_token_ids = resolve_rating_token_ids(tokenizer, norm_stimuli[0]["text"])
    print(f"  Rating tokens: {rating_token_ids}")

    print("\n[2/3] Baseline...")
    base_scores, _, _ = compute_wrongness_scores(
        model, tokenizer, norm_stimuli, rating_token_ids, device,
    )
    base_summary, _ = pair_discrimination(norm_stimuli, base_scores)
    base_d = base_summary[COND]["mean"]
    print(f"  Baseline norm_type discrimination: {base_d:+.3f}")

    print(f"\n[3/3] Sweeping {len(DOSES)} doses x (top, bot, 3xrandom)...")
    results = {"baseline": base_d, "doses": {}}
    rng_seeds = [42, 43, 44]

    for n_ab in DOSES:
        print(f"\n  n_ablate = {n_ab:,} ({n_ab/n_neurons*100:.2f}% of network)")
        entry = {}

        # top-k
        top, _ = select_neurons(attribution_npz, COND, n_ab, side="top")
        hooks = install_ablation_hooks(model, top, n_layers, ffn_dim, device)
        scores, _, _ = compute_wrongness_scores(
            model, tokenizer, norm_stimuli, rating_token_ids, device,
        )
        for h in hooks:
            h.remove()
        s, _ = pair_discrimination(norm_stimuli, scores)
        entry["top"] = s[COND]["mean"]
        print(f"    top : {entry['top']:+.3f}  (delta {entry['top']-base_d:+.3f})")

        # bot-k
        bot, _ = select_neurons(attribution_npz, COND, n_ab, side="bottom")
        hooks = install_ablation_hooks(model, bot, n_layers, ffn_dim, device)
        scores, _, _ = compute_wrongness_scores(
            model, tokenizer, norm_stimuli, rating_token_ids, device,
        )
        for h in hooks:
            h.remove()
        s, _ = pair_discrimination(norm_stimuli, scores)
        entry["bot"] = s[COND]["mean"]
        print(f"    bot : {entry['bot']:+.3f}  (delta {entry['bot']-base_d:+.3f})")

        # random across 3 seeds
        rand_vals = []
        for seed in rng_seeds:
            rng = np.random.RandomState(seed)
            rand = rng.choice(n_neurons, n_ab, replace=False)
            hooks = install_ablation_hooks(model, rand, n_layers, ffn_dim, device)
            scores, _, _ = compute_wrongness_scores(
                model, tokenizer, norm_stimuli, rating_token_ids, device,
            )
            for h in hooks:
                h.remove()
            s, _ = pair_discrimination(norm_stimuli, scores)
            rand_vals.append(s[COND]["mean"])
        entry["random_mean"] = float(np.mean(rand_vals))
        entry["random_std"] = float(np.std(rand_vals))
        entry["random_runs"] = rand_vals
        print(f"    rand: {entry['random_mean']:+.3f} ± {entry['random_std']:.3f}  "
              f"(delta {entry['random_mean']-base_d:+.3f})")

        results["doses"][n_ab] = entry

    # Dose-response analysis
    print(f"\n{'='*70}")
    print("DOSE-RESPONSE CURVE")
    print(f"{'='*70}")
    print(f"  {'n_ablate':>9s} | {'top - base':>10s} | {'bot - base':>10s} | "
          f"{'rand - base':>12s} | {'top - bot':>10s} | {'top / rand_σ':>13s}")
    print(f"  {'-'*9} + {'-'*10} + {'-'*10} + {'-'*12} + {'-'*10} + {'-'*13}")
    for n_ab in DOSES:
        e = results["doses"][n_ab]
        td = e["top"] - base_d
        bd = e["bot"] - base_d
        rd = e["random_mean"] - base_d
        rs = e["random_std"] + 1e-6
        z = abs(td) / rs
        print(f"  {n_ab:>9d} | {td:>+10.3f} | {bd:>+10.3f} | {rd:>+12.3f} | "
              f"{td-bd:>+10.3f} | {z:>13.2f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - t0
    results["model"] = model_short
    results["elapsed_seconds"] = elapsed
    results["n_neurons"] = n_neurons
    out_path = Path(output_dir) / f"{model_short}_norm_dose_response.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path} (elapsed {elapsed:.0f}s)")


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
