#!/usr/bin/env python3
"""
Emotion geometry analysis: does the LLM's emotion space match psychological
models (Russell's circumplex: valence × arousal)?

1. Extract LLM activations for GoEmotions 28-category stimuli
2. Build 28×28 emotion RDM from LLM representations
3. PCA on condition centroids → check if PC1=valence, PC2=arousal
4. Compare with Warriner VAD norms
5. Test base vs instruct models

Usage:
  python emotion_geometry.py \
    --model_path /path/to/model \
    --model_short Qwen2.5-1.5B-Instruct \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

# Russell's valence-arousal assignments for GoEmotions categories
# valence: -1 = negative, 0 = neutral, +1 = positive
# arousal: -1 = calm/low, 0 = medium, +1 = high/excited
EMOTION_VAD = {
    "admiration":     {"valence":  0.8, "arousal":  0.3},
    "amusement":      {"valence":  0.8, "arousal":  0.5},
    "anger":          {"valence": -0.8, "arousal":  0.8},
    "annoyance":      {"valence": -0.5, "arousal":  0.3},
    "approval":       {"valence":  0.5, "arousal":  0.1},
    "caring":         {"valence":  0.6, "arousal":  0.2},
    "confusion":      {"valence": -0.2, "arousal":  0.3},
    "curiosity":      {"valence":  0.3, "arousal":  0.5},
    "desire":         {"valence":  0.4, "arousal":  0.6},
    "disappointment": {"valence": -0.6, "arousal": -0.2},
    "disapproval":    {"valence": -0.5, "arousal":  0.2},
    "disgust":        {"valence": -0.8, "arousal":  0.4},
    "embarrassment":  {"valence": -0.5, "arousal":  0.5},
    "excitement":     {"valence":  0.7, "arousal":  0.9},
    "fear":           {"valence": -0.7, "arousal":  0.8},
    "gratitude":      {"valence":  0.8, "arousal":  0.2},
    "grief":          {"valence": -0.9, "arousal": -0.3},
    "joy":            {"valence":  0.9, "arousal":  0.7},
    "love":           {"valence":  0.9, "arousal":  0.5},
    "nervousness":    {"valence": -0.4, "arousal":  0.7},
    "neutral":        {"valence":  0.0, "arousal":  0.0},
    "optimism":       {"valence":  0.6, "arousal":  0.3},
    "pride":          {"valence":  0.7, "arousal":  0.5},
    "realization":    {"valence":  0.1, "arousal":  0.4},
    "relief":         {"valence":  0.5, "arousal": -0.3},
    "remorse":        {"valence": -0.7, "arousal": -0.1},
    "sadness":        {"valence": -0.8, "arousal": -0.4},
    "surprise":       {"valence":  0.1, "arousal":  0.8},
}


def extract_activations(model, tokenizer, texts, device, batch_size=16):
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for b in range(len(batch)):
            mask = inputs["attention_mask"][b].bool()
            last_hs = out.hidden_states[-1][b][mask].mean(dim=0)
            all_acts.append(last_hs.cpu().numpy())
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{len(texts)}")
    return np.array(all_acts)


def rdm_cosine(act):
    act = act - act.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - np.clip(a @ a.T, -1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--stimuli", default="/data/user/mzhang630/data/nature_exp/cognitive_stimuli/emotion/goemotions_sample.jsonl")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Load stimuli
    data = []
    with open(args.stimuli) as f:
        for line in f:
            d = json.loads(line)
            if d.get("emotion_labels") and len(d["emotion_labels"]) == 1:
                data.append(d)
    print(f"Loaded {len(data)} single-label stimuli")

    emotions = sorted(set(d["emotion_labels"][0] for d in data))
    print(f"Emotions: {len(emotions)}")

    # Load model
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

    # Extract activations
    texts = [d["text"] for d in data]
    labels = [d["emotion_labels"][0] for d in data]
    print(f"Extracting {len(texts)} activations...")
    acts = extract_activations(model, tokenizer, texts, device)
    print(f"  Shape: {acts.shape}")

    # Build emotion centroids
    centroids = {}
    for emo in emotions:
        idx = [i for i, l in enumerate(labels) if l == emo]
        if len(idx) >= 10:
            centroids[emo] = acts[idx].mean(axis=0)
    emo_list = sorted(centroids.keys())
    n = len(emo_list)
    print(f"\nEmotions with ≥10 samples: {n}")

    centroid_mat = np.array([centroids[e] for e in emo_list], dtype=np.float64)

    # RDM
    rdm = rdm_cosine(centroid_mat)

    # PCA
    centered = centroid_mat - centroid_mat.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U * S  # (n_emo, n_components)
    var_explained = S**2 / (S**2).sum()

    print(f"\nPCA variance explained:")
    for i in range(min(5, len(S))):
        print(f"  PC{i+1}: {var_explained[i]:.1%}")

    # Correlate PCs with valence and arousal
    val_scores = np.array([EMOTION_VAD.get(e, {}).get("valence", 0) for e in emo_list])
    aro_scores = np.array([EMOTION_VAD.get(e, {}).get("arousal", 0) for e in emo_list])

    print(f"\nPC-valence/arousal correlations:")
    for i in range(min(5, coords.shape[1])):
        rv, pv = spearmanr(coords[:, i], val_scores)
        ra, pa = spearmanr(coords[:, i], aro_scores)
        marker_v = " ***" if pv < 0.001 else " **" if pv < 0.01 else " *" if pv < 0.05 else ""
        marker_a = " ***" if pa < 0.001 else " **" if pa < 0.01 else " *" if pa < 0.05 else ""
        print(f"  PC{i+1} ({var_explained[i]:.1%}): "
              f"valence r={rv:+.3f}{marker_v}  arousal r={ra:+.3f}{marker_a}")

    # Save results
    results = {
        "model": args.model_short,
        "n_emotions": n,
        "emotions": emo_list,
        "variance_explained": var_explained[:10].tolist(),
        "pc_valence_corr": [],
        "pc_arousal_corr": [],
    }
    for i in range(min(10, coords.shape[1])):
        rv, pv = spearmanr(coords[:, i], val_scores)
        ra, pa = spearmanr(coords[:, i], aro_scores)
        results["pc_valence_corr"].append({"pc": i+1, "rho": float(rv), "p": float(pv)})
        results["pc_arousal_corr"].append({"pc": i+1, "rho": float(ra), "p": float(pa)})

    out_path = Path(args.output_dir) / f"{args.model_short}_emotion_geometry.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    np.savez(Path(args.output_dir) / f"{args.model_short}_emotion_geometry.npz",
             rdm=rdm, centroids=centroid_mat, coords=coords,
             emotions=np.array(emo_list), var_explained=var_explained,
             val_scores=val_scores, aro_scores=aro_scores)

    print(f"\nSaved {out_path}")

    # Print emotion coordinates (PC1 vs PC2)
    print(f"\nEmotion map (PC1 vs PC2):")
    for i, e in enumerate(emo_list):
        v = EMOTION_VAD.get(e, {}).get("valence", 0)
        a = EMOTION_VAD.get(e, {}).get("arousal", 0)
        print(f"  {e:<20s} PC1={coords[i,0]:+.3f} PC2={coords[i,1]:+.3f}  "
              f"(val={v:+.1f} aro={a:+.1f})")


if __name__ == "__main__":
    main()
