"""
Contrast-based attribution for the decomposition design.

Instead of "moral vs other 4 domains" selectivity (which is dominated by topic
features of moral-text), we compute attribution as the PAIR CONTRAST within each
decomposition condition:

    contrast_attribution[cond] = mean_{pair (a,b) in cond} [ |grad×act|(a) - |grad×act|(b) ]

For each of the 4 paired conditions:
  intent      a=intentional,       b=accidental
  outcome     a=completed,         b=attempted
  norm_type   a=moral_violation,   b=conventional_violation
  morality    a=moral_negative,    b=nonmoral_negative

A neuron with high positive contrast for ``norm_type`` is selectively engaged when
processing moral violations more than conventional violations — that is, it
encodes the moral-violation feature controlled for the negative-valence base.
Top-k positives are the "moral-violation atlas"; ablating them should specifically
collapse moral-vs-conventional discrimination without damaging the other three.

Output: ``{model_short}_contrast_attribution.npz`` with keys
    {intent, outcome, norm_type, morality}_contrast — raw float32 vectors of length
    n_neurons = n_layers * ffn_dim. The sign convention is mean(imp_a) - mean(imp_b).
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


def get_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        return m.gpt_neox.layers
    return None


def install_attribution_hooks(model, n_layers):
    """Install forward hooks on FFN gate_proj that retain activations + grads."""
    activations = {}
    hooks = []
    layers = get_layers(model)
    for layer_idx in range(n_layers):
        mlp = layers[layer_idx].mlp
        if hasattr(mlp, "gate_proj"):
            target = mlp.gate_proj
        elif hasattr(mlp, "dense_h_to_4h"):
            target = mlp.dense_h_to_4h
        else:
            continue

        def make_hook(lidx):
            def fwd_hook(_module, _input, output):
                activations[lidx] = output
                output.retain_grad()
            return fwd_hook
        hooks.append(target.register_forward_hook(make_hook(layer_idx)))
    return activations, hooks


def importance_single(model, tokenizer, text, activations, n_layers, ffn_dim,
                       device="cuda", max_length=384):
    """Return the per-neuron |grad × activation| importance vector for one stimulus."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length).to(device)
    if inputs["input_ids"].shape[1] < 2:
        return None
    model.zero_grad()
    activations.clear()
    out = model(**inputs, labels=inputs["input_ids"])
    out.loss.backward()

    imp = np.zeros(n_layers * ffn_dim, dtype=np.float32)
    for layer_idx in range(n_layers):
        a = activations.get(layer_idx, None)
        if a is None or a.grad is None:
            continue
        v = (a.detach() * a.grad.detach()).abs().mean(dim=(0, 1)).cpu().float().numpy()
        start = layer_idx * ffn_dim
        imp[start:start + ffn_dim] = v
    return imp


def run(model_path, model_short, decomposition_jsonl, meta_path, output_dir,
        device="cuda"):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Contrast Attribution: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim
    print(f"  Model dims: {n_layers} layers x {ffn_dim} FFN = {n_neurons:,} neurons")

    with open(decomposition_jsonl) as f:
        stimuli = [json.loads(line) for line in f if line.strip()]

    # Group stimuli by pair_id, pick {a, b} members
    by_pair = defaultdict(dict)
    for s in stimuli:
        by_pair[s["pair_id"]][s["id"][-1]] = s
    pairs_by_cond = defaultdict(list)
    for pid, members in by_pair.items():
        if "a" in members and "b" in members:
            cond = members["a"]["condition"]
            pairs_by_cond[cond].append((members["a"], members["b"]))
    conditions = sorted(pairs_by_cond.keys())
    print(f"  Conditions: {conditions}")
    print(f"  Pairs per condition: " + ", ".join(
        f"{c}={len(v)}" for c, v in pairs_by_cond.items()))

    print("\n[Step 1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    activations, hooks = install_attribution_hooks(model, n_layers)

    print("\n[Step 2] Computing contrast attribution per condition...")
    contrast_imp = {c: np.zeros(n_neurons, dtype=np.float64) for c in conditions}
    a_imp_mean = {c: np.zeros(n_neurons, dtype=np.float64) for c in conditions}
    b_imp_mean = {c: np.zeros(n_neurons, dtype=np.float64) for c in conditions}
    pair_counts = {c: 0 for c in conditions}

    for cond, pair_list in pairs_by_cond.items():
        print(f"  {cond}: {len(pair_list)} pairs...")
        for sa, sb in pair_list:
            imp_a = importance_single(model, tokenizer, sa["text"], activations,
                                       n_layers, ffn_dim, device)
            imp_b = importance_single(model, tokenizer, sb["text"], activations,
                                       n_layers, ffn_dim, device)
            if imp_a is None or imp_b is None:
                continue
            contrast_imp[cond] += (imp_a - imp_b).astype(np.float64)
            a_imp_mean[cond] += imp_a.astype(np.float64)
            b_imp_mean[cond] += imp_b.astype(np.float64)
            pair_counts[cond] += 1
        if pair_counts[cond] > 0:
            contrast_imp[cond] /= pair_counts[cond]
            a_imp_mean[cond] /= pair_counts[cond]
            b_imp_mean[cond] /= pair_counts[cond]

    for h in hooks:
        h.remove()

    # Summary stats
    print("\n[Step 3] Summary of contrast vectors:")
    print(f"  {'Condition':>12s} | {'n_pairs':>7s} | {'max(+)':>8s} {'max(-)':>8s} | {'top-5000 sum':>13s}")
    save_dict = {}
    for cond in conditions:
        v = contrast_imp[cond].astype(np.float32)
        save_dict[f"{cond}_contrast"] = v
        save_dict[f"{cond}_a_importance"] = a_imp_mean[cond].astype(np.float32)
        save_dict[f"{cond}_b_importance"] = b_imp_mean[cond].astype(np.float32)
        top5000 = np.sort(v)[-5000:]
        print(f"  {cond:>12s} | {pair_counts[cond]:>7d} | {v.max():>+8.3e} {v.min():>+8.3e} | {top5000.sum():>+13.3e}")

    # Selectivity (one-vs-rest among the 4 conditions) for compatibility with
    # the existing moral_decomposition.py loader, which expects keys ending in
    # "_selectivity".
    print("\n[Step 4] Computing one-vs-rest contrast-selectivity (sign-preserving)...")
    eps = 1e-12
    for cond in conditions:
        target = contrast_imp[cond]
        others = np.mean([contrast_imp[c] for c in conditions if c != cond], axis=0)
        # Sign-preserving normalized contrast: positive when target > others.
        sel = (target - others) / (np.abs(target) + np.abs(others) + eps)
        save_dict[f"{cond}_selectivity"] = sel.astype(np.float32)
        n_high = (sel > 0.3).sum()
        print(f"  {cond:>12s}: selective(>0.3)={n_high:>6,}, max={sel.max():.4f}, min={sel.min():.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{model_short}_contrast_attribution.npz"
    np.savez(out_path, **save_dict)
    elapsed = time.time() - t0
    print(f"\n  Saved {len(save_dict)} arrays to {out_path}")
    print(f"  Elapsed: {elapsed:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--decomposition_jsonl", required=True)
    ap.add_argument("--meta_path", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    run(args.model_path, args.model_short, args.decomposition_jsonl,
        args.meta_path, args.output_dir)


if __name__ == "__main__":
    main()
