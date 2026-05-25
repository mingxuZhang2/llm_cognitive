"""
4x4 contrast-decomposition dissociation matrix.

Given a contrast attribution NPZ (output of contrast_attribution.py with keys
``{intent, outcome, norm_type, morality}_selectivity``), this script:

1. For each of the 4 paired conditions, selects top-k contrast neurons.
2. Ablates each set (zero gate_proj output) and measures all 4 paired
   discriminations on the same 400-item decomposition stimuli used to
   build the contrast attribution.
3. Outputs a 4×4 matrix: rows = ablation target, cols = measured discrimination.
   Diagonal should drop sharply; off-diagonal should be relatively spared if the
   compositional finding holds.

Also runs:
  - 3-seed random ablation (noise floor)
  - anti-contrast (bottom-k) ablation per condition (sanity check — should
    NOT collapse the matching diagonal entry if the atlas is real and not
    a symmetric artefact)

This script reuses helpers from moral_decomposition.py (chat-template prompt,
log-odds metric, hook installation).
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


CONDITIONS = ["intent", "outcome", "norm_type", "morality"]


def run(model_path, model_short, attribution_npz, decomposition_jsonl,
        meta_path, output_dir, n_ablate=5000, device="cuda"):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Contrast Decomposition Dissociation: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim

    with open(decomposition_jsonl) as f:
        stimuli = [json.loads(line) for line in f if line.strip()]
    print(f"  {n_neurons:,} neurons, {len(stimuli)} stimuli, n_ablate={n_ablate}")

    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rating_token_ids = resolve_rating_token_ids(tokenizer, stimuli[0]["text"])
    surface = tokenizer.convert_ids_to_tokens(rating_token_ids)
    print(f"  Rating token ids: {rating_token_ids}  surface={surface}")
    for d, t in zip(RATING_TOKENS, surface):
        assert d in t, f"Rating token for {d!r} resolved to {t!r}"

    # Baseline log-odds discrimination
    print("\n[2/4] Baseline (no ablation)...")
    base_scores, base_expected, _ = compute_wrongness_scores(
        model, tokenizer, stimuli, rating_token_ids, device,
    )
    base_summary, _ = pair_discrimination(stimuli, base_scores)
    print("  Baseline log-odds discrimination:")
    for c in CONDITIONS:
        s = base_summary[c]
        print(f"    {c:>12s}: {s['mean']:+7.3f} (std {s['std']:.2f}, n={s['n']})")

    # 4x4 ablation matrix + anti-ablation per condition + random control
    print(f"\n[3/4] Ablating top-{n_ablate} contrast neurons for each condition...")
    ablation_summaries = {}  # ablation_target -> {cond: summary_dict}
    for target_cond in CONDITIONS:
        # top-k contrast (a > b)
        try:
            neurons, _ = select_neurons(
                attribution_npz, target_cond, n_ablate, side="top")
        except KeyError as e:
            print(f"  SKIP top {target_cond}: {e}")
            continue
        hooks = install_ablation_hooks(model, neurons, n_layers, ffn_dim, device)
        scores, _, _ = compute_wrongness_scores(
            model, tokenizer, stimuli, rating_token_ids, device,
        )
        for h in hooks:
            h.remove()
        summary, _ = pair_discrimination(stimuli, scores)
        ablation_summaries[f"top_{target_cond}"] = summary

        # anti (bottom-k)
        neurons_anti, _ = select_neurons(
            attribution_npz, target_cond, n_ablate, side="bottom")
        hooks = install_ablation_hooks(
            model, neurons_anti, n_layers, ffn_dim, device)
        scores_anti, _, _ = compute_wrongness_scores(
            model, tokenizer, stimuli, rating_token_ids, device,
        )
        for h in hooks:
            h.remove()
        anti_summary, _ = pair_discrimination(stimuli, scores_anti)
        ablation_summaries[f"bot_{target_cond}"] = anti_summary

    # Random control (3 seeds)
    print(f"\n  Random ablation (3 seeds)...")
    random_summaries = []
    for seed in [42, 43, 44]:
        rng = np.random.RandomState(seed)
        rand_neurons = rng.choice(n_neurons, n_ablate, replace=False)
        hooks = install_ablation_hooks(model, rand_neurons, n_layers, ffn_dim, device)
        rscores, _, _ = compute_wrongness_scores(
            model, tokenizer, stimuli, rating_token_ids, device,
        )
        for h in hooks:
            h.remove()
        rs, _ = pair_discrimination(stimuli, rscores)
        random_summaries.append(rs)
    rand_summary = {}
    for c in CONDITIONS:
        vals = [r[c]["mean"] for r in random_summaries]
        rand_summary[c] = {"mean": float(np.mean(vals)),
                            "std": float(np.std(vals))}

    # Print 4x4 matrix (delta from baseline)
    print(f"\n[4/4] 4x4 contrast-decomposition matrix (deltas from baseline)")
    print(f"{'='*70}")
    print(f"  Rows = top-k of {{cond}}-contrast ablated; Cols = which discrimination measured")
    print(f"  Diagonal expected NEGATIVE (target ablation collapses target discrim)")
    print(f"  Off-diagonal expected near zero (no collateral damage)")
    print()
    print(f"  {'Ablated':>12s} |" + "".join(f" {c[:9]:>9s}" for c in CONDITIONS) + " | random μ±σ (noise floor)")
    print(f"  {'-'*12} + " + " ".join(["-"*9]*4) + " + " + "-"*24)
    diag_deltas, offdiag_deltas = [], []
    for tgt in CONDITIONS:
        row = f"  top_{tgt[:8]:>8s} |"
        s = ablation_summaries.get(f"top_{tgt}")
        if s is None:
            print(row + " (missing)")
            continue
        for c in CONDITIONS:
            d = s[c]["mean"] - base_summary[c]["mean"]
            row += f" {d:>+9.3f}"
            if tgt == c:
                diag_deltas.append(d)
            else:
                offdiag_deltas.append(d)
        row += " |"
        print(row)
    # also print anti row group
    print()
    print(f"  Anti-ablation (bottom-k contrast) — sanity check:")
    print(f"  {'Ablated':>12s} |" + "".join(f" {c[:9]:>9s}" for c in CONDITIONS))
    print(f"  {'-'*12} + " + " ".join(["-"*9]*4))
    anti_diag = []
    for tgt in CONDITIONS:
        row = f"  bot_{tgt[:8]:>8s} |"
        s = ablation_summaries.get(f"bot_{tgt}")
        if s is None:
            print(row + " (missing)")
            continue
        for c in CONDITIONS:
            d = s[c]["mean"] - base_summary[c]["mean"]
            row += f" {d:>+9.3f}"
            if tgt == c:
                anti_diag.append(d)
        print(row)
    # Random noise floor
    print()
    print(f"  Random ablation (noise floor, 3 seeds):")
    for c in CONDITIONS:
        rm = rand_summary[c]["mean"] - base_summary[c]["mean"]
        rs = rand_summary[c]["std"]
        print(f"    {c:>12s}: delta = {rm:+7.3f} ± {rs:.3f}")

    # Compositional verdict
    print(f"\n{'='*70}")
    print("COMPOSITIONAL VERDICT")
    print(f"{'='*70}")
    print(f"\n  Per-condition: is |top_cond| > |bot_cond| > |random|?")
    print(f"  (If yes → contrast atlas IS the specific substrate, anti is not)")
    print(f"\n  {'Condition':>12s} | {'|top diag|':>10s} | {'|bot diag|':>10s} | "
          f"{'random σ':>9s} | {'top z':>6s} | {'bot z':>6s} | verdict")
    print(f"  {'-'*12} + {'-'*10} + {'-'*10} + {'-'*9} + {'-'*6} + {'-'*6} + {'-'*16}")
    for i, c in enumerate(CONDITIONS):
        td = abs(diag_deltas[i])
        bd = abs(anti_diag[i])
        rs = rand_summary[c]["std"] + 1e-6
        tz = td / rs
        bz = bd / rs
        ok = (tz > 2 and td > bd + 0.3)
        verdict = "STRONG" if (tz > 3 and td > bd + 0.5) else ("OK" if ok else "weak/null")
        print(f"  {c:>12s} | {td:>10.3f} | {bd:>10.3f} | {rs:>9.3f} | "
              f"{tz:>6.2f} | {bz:>6.2f} | {verdict}")

    # Off-diagonal selectivity
    diag = np.abs(diag_deltas)
    offdiag = np.abs(offdiag_deltas).reshape(4, 3) if len(offdiag_deltas) == 12 else None
    if offdiag is not None:
        print(f"\n  Selectivity (per-row diagonal / mean off-diagonal):")
        for i, c in enumerate(CONDITIONS):
            sel = diag[i] / (offdiag[i].mean() + 1e-6)
            print(f"    {c:>12s}: diag={diag[i]:.2f}, mean_off={offdiag[i].mean():.2f}, ratio={sel:.2f}x")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_short,
        "n_ablate": n_ablate,
        "n_neurons": n_neurons,
        "metric": "log-odds wrongness, chat-template",
        "baseline": base_summary,
        "ablation_summaries": ablation_summaries,
        "random_control": rand_summary,
        "diagonal_deltas": diag_deltas,
        "antidiagonal_deltas": anti_diag,
        "elapsed_seconds": time.time() - t0,
    }
    out_path = Path(output_dir) / f"{model_short}_contrast_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved to {out_path} (elapsed {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--attribution_npz", required=True,
                    help="Output from contrast_attribution.py")
    ap.add_argument("--decomposition_jsonl", required=True)
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_ablate", type=int, default=5000)
    args = ap.parse_args()
    run(args.model_path, args.model_short, args.attribution_npz,
        args.decomposition_jsonl, args.meta_path, args.output_dir,
        n_ablate=args.n_ablate)


if __name__ == "__main__":
    main()
