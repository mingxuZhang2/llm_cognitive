"""
v2 RSA activation extraction: save PER-STIMULUS activations across 3 pooling
strategies × every layer. Enables (a) split-half noise ceiling, (b) pooling
sweep at analysis time, (c) stimulus-level RDMs.

Pooling strategies (computed simultaneously per stimulus):
  - last_tok    : last non-pad token's hidden state (matches v1)
  - mean_all    : mean over all non-pad tokens
  - last_8_mean : mean over last 8 non-pad tokens (smoothed last-token)

Output NPZ:
  per_stim_activations : float16 [3, n_stim, n_layers+1, hidden_dim]
                          dim 0 = pooling strategy (last_tok / mean_all / last_8_mean)
  conditions           : str[n_stim]   (per-stim condition label)
  stim_ids             : str[n_stim]
  pooling_names        : ["last_tok", "mean_all", "last_8_mean"]
  layer_names          : str[n_layers+1]
  hidden_dim           : int

Storage estimate: 3 × 712 × ~30 × 4096 × 2 bytes ~ 500 MB per model uncompressed.
Save as compressed npz; expect ~150-200 MB on disk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_stimuli(jsonl_path):
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


@torch.no_grad()
def extract_three_poolings(model, tokenizer, text, device, max_length=512,
                            last_k=8):
    """Return tensor [3, n_layers+1, hidden_dim]:
       pooling 0: last-token, 1: mean-all-tokens, 2: mean-of-last-K-tokens.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    seq_len = enc["input_ids"].shape[1]
    # hidden_states: tuple of [1, seq_len, hidden] per layer (incl. embedding)
    layers = torch.stack([h[0].float() for h in out.hidden_states], dim=0)
    # layers: [n_layers+1, seq, hidden]
    last_tok = layers[:, seq_len - 1, :]
    mean_all = layers.mean(dim=1)
    k = min(last_k, seq_len)
    last_k_mean = layers[:, seq_len - k:, :].mean(dim=1)
    out_arr = torch.stack([last_tok, mean_all, last_k_mean], dim=0).cpu().numpy()
    return out_arr  # [3, n_layers+1, hidden_dim]


def run(model_path, model_short, stimuli_jsonl, output_dir, device="cuda",
        max_length=512):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"RSA activation extraction v2: {model_short}")
    print(f"{'='*70}")

    stimuli = load_stimuli(stimuli_jsonl)
    print(f"  {len(stimuli)} stimuli total")

    print(f"  Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    probe = extract_three_poolings(
        model, tokenizer, stimuli[0]["text"], device, max_length)
    n_pool, n_layers, hidden_dim = probe.shape
    print(f"  pools={n_pool}, layers={n_layers}, hidden={hidden_dim}")

    per_stim = np.empty(
        (n_pool, len(stimuli), n_layers, hidden_dim), dtype=np.float16)
    conditions = []
    stim_ids = []

    for i, item in enumerate(stimuli):
        per_stim[:, i] = extract_three_poolings(
            model, tokenizer, item["text"], device, max_length).astype(np.float16)
        conditions.append(item["condition"])
        stim_ids.append(item["id"])
        if (i + 1) % 100 == 0 or i + 1 == len(stimuli):
            print(f"  [{i+1:4d}/{len(stimuli)}]  elapsed {time.time()-t0:.0f}s")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{model_short}_rsa_v2_per_stim.npz"
    layer_names = ["embedding"] + [f"layer_{i}" for i in range(n_layers - 1)]
    np.savez_compressed(
        out_path,
        per_stim_activations=per_stim,
        conditions=np.array(conditions),
        stim_ids=np.array(stim_ids),
        pooling_names=np.array(["last_tok", "mean_all", "last_8_mean"]),
        layer_names=np.array(layer_names),
        hidden_dim=hidden_dim,
        model_short=model_short,
    )
    print(f"\n  Saved {out_path}  ({time.time()-t0:.0f}s, "
          f"size {out_path.stat().st_size/1e6:.0f} MB)")


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
