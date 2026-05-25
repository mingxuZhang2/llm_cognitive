"""
Extract per-condition last-token hidden states across every layer of an LLM,
for the RSA pipeline. Each cognitive condition gets a mean activation vector
at each layer; these are the LLM-side representations used to build the
14x14 representational dissimilarity matrix.

Standard NeuroAI extraction protocol (Schrimpf 2021, Caucheteux 2022):
  - feed each stimulus as raw text (no instruction wrapper)
  - take the last-token hidden state at each transformer layer
  - mean-pool across stimuli within each condition

Input:
  --stimuli_jsonl   rsa_stimuli.jsonl with fields {id, condition, text}
  --model_path      HF model snapshot path
  --output_dir      where to write {model_short}_rsa_activations.npz

Output NPZ keys:
  mean_activations  : float32 [n_conditions, n_layers+1, hidden_dim]
                      layer index 0 = embedding, 1..n_layers = transformer blocks
  conditions        : list[str], length n_conditions
  n_per_condition   : int[n_conditions]
  layer_names       : list[str] of length n_layers+1
  hidden_dim        : int
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_stimuli(jsonl_path):
    """Group stimuli by condition, preserving file order."""
    by_cond = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_cond[rec["condition"]].append(rec)
    return dict(by_cond)


@torch.no_grad()
def get_last_token_hiddens(model, tokenizer, text, device, max_length=512):
    """Return tensor [n_layers+1, hidden_dim] of last-token hidden states."""
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    # hidden_states is a tuple of length n_layers+1, each [1, seq, hidden]
    last_idx = enc["input_ids"].shape[1] - 1
    hs = torch.stack([h[0, last_idx].float() for h in out.hidden_states], dim=0)
    return hs.cpu().numpy()


def run(model_path, model_short, stimuli_jsonl, output_dir, device="cuda",
        max_length=512):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"RSA activation extraction: {model_short}")
    print(f"{'='*70}")

    by_cond = load_stimuli(stimuli_jsonl)
    conditions = sorted(by_cond.keys())
    print(f"  Conditions: {len(conditions)}")
    for c in conditions:
        print(f"    {c:>16s}: n={len(by_cond[c])}")

    print(f"\n  Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Probe layer count + hidden dim
    probe_text = by_cond[conditions[0]][0]["text"]
    probe = get_last_token_hiddens(model, tokenizer, probe_text, device, max_length)
    n_layers_plus_emb, hidden_dim = probe.shape
    print(f"  n_layers (incl. embedding) = {n_layers_plus_emb}, hidden_dim = {hidden_dim}")

    # Accumulate
    mean_activations = np.zeros(
        (len(conditions), n_layers_plus_emb, hidden_dim), dtype=np.float32)
    n_per_cond = np.zeros(len(conditions), dtype=np.int32)

    for ci, cond in enumerate(conditions):
        items = by_cond[cond]
        accum = np.zeros((n_layers_plus_emb, hidden_dim), dtype=np.float64)
        for i, item in enumerate(items):
            hs = get_last_token_hiddens(
                model, tokenizer, item["text"], device, max_length)
            accum += hs.astype(np.float64)
        mean_activations[ci] = (accum / len(items)).astype(np.float32)
        n_per_cond[ci] = len(items)
        print(f"  [{ci+1:2d}/{len(conditions)}] {cond:>16s}  "
              f"n={len(items):3d}  "
              f"||mean||={np.linalg.norm(mean_activations[ci].mean(0)):.2f}  "
              f"elapsed {time.time()-t0:.0f}s")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{model_short}_rsa_activations.npz"
    layer_names = ["embedding"] + [f"layer_{i}" for i in range(n_layers_plus_emb - 1)]
    np.savez_compressed(
        out_path,
        mean_activations=mean_activations,
        conditions=np.array(conditions),
        n_per_condition=n_per_cond,
        layer_names=np.array(layer_names),
        hidden_dim=hidden_dim,
        model_short=model_short,
    )
    print(f"\n  Saved {out_path}  ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--stimuli_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()
    run(args.model_path, args.model_short, args.stimuli_jsonl,
        args.output_dir, max_length=args.max_length)


if __name__ == "__main__":
    main()
