#!/usr/bin/env python3
"""
Extract LLM activations for Narratives story sentences.

For each sentence in a story, run forward pass and save per-layer
mean-pooled activations. These are then used to build LLM RDM on the
SAME text as the fMRI brain data.

Usage:
  python extract_narratives_llm.py \
    --model_path /data/user/mzhang630/data/models_dl/Qwen2.5-7B-Instruct \
    --model_short Qwen2.5-7B-Instruct \
    --sentences_jsonl /path/to/annotated_sentences.jsonl \
    --story pieman \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract(model, tokenizer, texts: list[str], device: str, batch_size: int = 8):
    """Extract per-layer mean-pooled activations for each text."""
    all_acts = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
        attention_mask = inputs["attention_mask"]  # (batch, seq)

        for b in range(len(batch_texts)):
            mask = attention_mask[b].bool()
            layers = []
            for layer_hs in hidden_states:
                tokens = layer_hs[b][mask]  # (valid_tokens, hidden)
                mean_pooled = tokens.mean(dim=0).cpu().numpy()
                layers.append(mean_pooled)
            all_acts.append(np.stack(layers))  # (n_layers, hidden)

        if (i // batch_size) % 5 == 0:
            print(f"  batch {i//batch_size}: {i+len(batch_texts)}/{len(texts)}")

    return np.stack(all_acts)  # (n_sentences, n_layers, hidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--sentences_jsonl", required=True)
    parser.add_argument("--story", default="pieman")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    sents = []
    with open(args.sentences_jsonl) as f:
        for line in f:
            s = json.loads(line)
            if s["story"] == args.story:
                sents.append(s)
    print(f"Story: {args.story}, {len(sents)} sentences")

    texts = [s["text"] for s in sents]
    labels = [s.get("llm_labels", []) for s in sents]
    onsets = [s["onset"] for s in sents]
    offsets = [s["offset"] for s in sents]

    print(f"Loading {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on {device}")

    print("Extracting activations...")
    acts = extract(model, tokenizer, texts, str(device), args.batch_size)
    print(f"Activations: {acts.shape}")  # (n_sents, n_layers, hidden)

    out_path = Path(args.output_dir) / f"{args.model_short}_{args.story}_narratives.npz"
    np.savez_compressed(
        out_path,
        activations=acts,
        texts=np.array(texts, dtype=object),
        labels=np.array(labels, dtype=object),
        onsets=np.array(onsets),
        offsets=np.array(offsets),
        story=args.story,
        model_short=args.model_short,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
