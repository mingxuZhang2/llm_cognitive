#!/usr/bin/env python3
"""
Extract activations for embodied factorial stimuli and compute within-aff RSA
per factorial condition.

GPU required for extraction.

Usage:
  python src/embodied_factorial_extract.py --model_path /path/to/model [--model_short name]
"""
from __future__ import annotations
import argparse, json, gc
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr
from itertools import combinations

BASE = Path(__file__).resolve().parents[1]
STIM_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "embodied_factorial_stimuli.jsonl"
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness"]
FACTORIALS = ["minimal_neutral", "minimal_embodied", "rich_neutral", "rich_embodied"]

PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}


def extract_hidden(model, tokenizer, texts, device, batch_size=8):
    """Extract mean-pooled hidden states at all layers."""
    all_hidden = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        for bi in range(len(batch)):
            mask = inputs["attention_mask"][bi].bool()
            layers = []
            for layer_h in out.hidden_states:
                h = layer_h[bi].float()
                layers.append(h[mask].mean(dim=0).cpu().numpy())
            all_hidden.append(np.stack(layers))  # (n_layers, hidden)

    return np.stack(all_hidden)  # (n_stim, n_layers, hidden)


def centroid_cos_dist(X1, X2):
    mu1, mu2 = X1.mean(0), X2.mean(0)
    cos = np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2) + 1e-10)
    return float(1 - cos)


def within_aff_rsa(acts, conditions, layer, brain_aff_dists):
    a = acts[:, layer, :].astype(np.float32)
    cond_stims = {c: a[[i for i, cc in enumerate(conditions) if cc == c]]
                  for c in EMOTIONS}

    centroids = [cond_stims[c].mean(0) for c in EMOTIONS]
    gm = np.mean(centroids, axis=0)
    for c in EMOTIONS:
        cond_stims[c] = cond_stims[c] - gm

    pairs = list(combinations(EMOTIONS, 2))
    llm_dists = [centroid_cos_dist(cond_stims[c1], cond_stims[c2]) for c1, c2 in pairs]

    rho, p = spearmanr(brain_aff_dists, llm_dists)
    return float(rho), float(p), llm_dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", default=None)
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peak = PEAK.get(model_short)
    if peak is None:
        print(f"WARNING: no peak layer for {model_short}, using L15")
        peak = 15

    print(f"{'='*70}")
    print(f"EMBODIED FACTORIAL: {model_short} (peak L{peak})")
    print(f"{'='*70}")

    # Load stimuli
    with open(STIM_PATH) as f:
        stimuli = [json.loads(l) for l in f]
    print(f"Loaded {len(stimuli)} stimuli")

    # Load brain within-aff distances
    brain = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    brain_conds = list(brain["conditions"])
    brain_rdm = brain["rdm"]
    brain_aff = [brain_rdm[brain_conds.index(c1), brain_conds.index(c2)]
                 for c1, c2 in combinations(EMOTIONS, 2)]

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # Extract per factorial condition
    results = {"model": model_short, "peak_layer": peak}

    for fac in FACTORIALS:
        fac_stims = [s for s in stimuli if s["factorial"] == fac]
        texts = [s["text"] for s in fac_stims]
        conditions = [s["condition"] for s in fac_stims]

        print(f"\n  Extracting {fac} ({len(texts)} stimuli)...", flush=True)
        acts = extract_hidden(model, tokenizer, texts, device)
        print(f"    Shape: {acts.shape}")

        # RSA at peak layer
        rho, p, llm_dists = within_aff_rsa(acts, conditions, peak, brain_aff)
        print(f"    Within-aff RSA @L{peak}: ρ={rho:+.3f} (p={p:.4f})")

        # Layer sweep
        layer_rhos = []
        for li in range(acts.shape[1]):
            r, _, _ = within_aff_rsa(acts, conditions, li, brain_aff)
            layer_rhos.append(r)

        best_layer = int(np.argmax(layer_rhos))
        best_rho = layer_rhos[best_layer]
        print(f"    Best layer: L{best_layer} (ρ={best_rho:+.3f})")

        results[fac] = {
            "n_stim": len(texts),
            "rho_at_peak": rho,
            "p_at_peak": p,
            "best_layer": best_layer,
            "best_rho": best_rho,
            "layer_rhos": layer_rhos,
            "llm_dists_at_peak": llm_dists,
        }

        del acts; gc.collect(); torch.cuda.empty_cache()

    del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ SUMMARY ═══
    print(f"\n{'='*70}")
    print(f"2×2 FACTORIAL RESULTS: {model_short}")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<25s} | {'ρ (peak)':>10s} | {'p':>8s} | {'ρ (best)':>10s} | {'Best L':>6s}")
    print(f"  {'-'*70}")
    for fac in FACTORIALS:
        r = results[fac]
        sig = "***" if r["p_at_peak"] < 0.001 else "**" if r["p_at_peak"] < 0.01 else "*" if r["p_at_peak"] < 0.05 else ""
        print(f"  {fac:<25s} | {r['rho_at_peak']:>+9.3f} | {r['p_at_peak']:>7.4f}{sig} | {r['best_rho']:>+9.3f} | L{r['best_layer']:>3d}")

    # 2×2 decomposition
    mn = results["minimal_neutral"]["rho_at_peak"]
    me = results["minimal_embodied"]["rho_at_peak"]
    rn = results["rich_neutral"]["rho_at_peak"]
    re = results["rich_embodied"]["rho_at_peak"]

    embodiment_effect = ((me + re) / 2) - ((mn + rn) / 2)
    situation_effect = ((rn + re) / 2) - ((mn + me) / 2)
    interaction = (re - rn) - (me - mn)

    print(f"\n  Main effect of EMBODIMENT:  {embodiment_effect:+.3f}")
    print(f"  Main effect of SITUATION:   {situation_effect:+.3f}")
    print(f"  Interaction:                {interaction:+.3f}")
    print(f"\n  If embodiment > situation → embodied information drives recovery")
    print(f"  If situation > embodiment → situational context drives recovery")

    results["factorial_effects"] = {
        "embodiment": float(embodiment_effect),
        "situation": float(situation_effect),
        "interaction": float(interaction),
    }

    out_path = OUT / f"embodied_factorial_{model_short}.json"
    json.dump(results, open(out_path, "w"), indent=2,
              default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
