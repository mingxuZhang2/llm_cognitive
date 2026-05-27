#!/usr/bin/env python3
"""
Test the core mechanistic hypothesis: next-token prediction similarity
between cognitive conditions determines which distinctions LLMs preserve
vs collapse.

For each condition pair (A, B):
  - Compute mean next-token probability distribution for A and B stimuli
  - Compute Jensen-Shannon divergence between distributions
  - Correlate with brain-LLM RDM residual

Prediction: pairs with similar next-token distributions (low JSD) will have
larger brain-LLM mismatch (LLM collapses them despite brain distinguishing them).

Usage:
  python next_token_similarity.py \
    --model_path /path/to/model \
    --model_short Qwen2.5-3B-Instruct \
    --stimuli_jsonl /path/to/rsa_stimuli.jsonl \
    --brain_rdm /path/to/brain_rdm.npz \
    --llm_rdm /path/to/model_rsa_v2_per_stim.npz \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import jensenshannon


def compute_next_token_distributions(model, tokenizer, texts, device, top_k=1000):
    """For each text, get the next-token probability distribution (top-k)."""
    all_dists = []
    vocab_size = model.config.vocab_size

    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]  # last position
            probs = torch.softmax(logits, dim=0)

        # Keep full distribution as numpy (sparse: only store top-k)
        top_vals, top_idx = probs.topk(top_k)
        sparse_dist = np.zeros(vocab_size, dtype=np.float32)
        sparse_dist[top_idx.cpu().numpy()] = top_vals.cpu().numpy()
        sparse_dist /= sparse_dist.sum()  # renormalize
        all_dists.append(sparse_dist)

        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(texts)}")

    return all_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--stimuli_jsonl", required=True)
    parser.add_argument("--brain_rdm", required=True)
    parser.add_argument("--llm_rdm", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Load stimuli
    stimuli = [json.loads(l) for l in open(args.stimuli_jsonl)]
    texts = [s.get("text", s.get("prompt", "")) for s in stimuli]
    conditions = [s.get("condition", s.get("category")) for s in stimuli]
    print(f"{len(stimuli)} stimuli")

    # Load brain RDM
    brain_data = np.load(args.brain_rdm, allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])
    n = len(brain_conds)
    triu = np.triu_indices(n, k=1)
    brain_vec = brain_rdm[triu]

    # Load LLM RDM
    llm_data = np.load(args.llm_rdm, allow_pickle=True)
    per_stim = llm_data["per_stim_activations"]
    llm_conditions = list(llm_data["conditions"])
    pool_idx = list(llm_data["pooling_names"]).index("mean_all")
    unique_conds = sorted(set(llm_conditions))
    stim_cond = np.array([unique_conds.index(c) for c in llm_conditions])
    n_layers = per_stim.shape[2]
    cond_means = np.zeros((len(unique_conds), n_layers, per_stim.shape[-1]), dtype=np.float32)
    for ci in range(len(unique_conds)):
        idx = np.where(stim_cond == ci)[0]
        cond_means[ci] = per_stim[pool_idx, idx].mean(axis=0)
    order = [unique_conds.index(c) for c in brain_conds]
    cond_means = cond_means[order]

    best_L = 0
    best_rho = -1
    for L in range(n_layers):
        act = cond_means[:, L, :].astype(np.float64)
        act = act - act.mean(axis=0)
        nn = np.linalg.norm(act, axis=1, keepdims=True); nn[nn==0]=1
        rdm = 1.0 - np.clip((act/nn) @ (act/nn).T, -1, 1)
        rho, _ = spearmanr(brain_vec, rdm[triu])
        if rho > best_rho:
            best_rho, best_L = rho, L

    act = cond_means[:, best_L, :].astype(np.float64)
    act = act - act.mean(axis=0)
    nn = np.linalg.norm(act, axis=1, keepdims=True); nn[nn==0]=1
    llm_rdm_mat = 1.0 - np.clip((act/nn) @ (act/nn).T, -1, 1)
    llm_vec = llm_rdm_mat[triu]
    del per_stim, llm_data

    # Compute brain-LLM residual
    brain_ranks = rankdata(brain_vec)
    llm_ranks = rankdata(llm_vec)
    residual = np.abs(brain_ranks - llm_ranks)

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

    # Compute next-token distributions
    print("Computing next-token distributions...")
    all_dists = compute_next_token_distributions(model, tokenizer, texts, device)
    print(f"  Got {len(all_dists)} distributions")

    # Build per-condition mean next-token distribution
    by_cond = defaultdict(list)
    for i, c in enumerate(conditions):
        by_cond[c].append(i)

    cond_dists = {}
    for c in brain_conds:
        if c in by_cond:
            mean_dist = np.mean([all_dists[i] for i in by_cond[c]], axis=0)
            mean_dist /= mean_dist.sum()
            cond_dists[c] = mean_dist

    # Compute pairwise JSD (next-token similarity)
    pairs = [(brain_conds[i], brain_conds[j]) for i, j in zip(*triu)]
    jsd_vec = np.zeros(len(pairs))
    for k, (ca, cb) in enumerate(pairs):
        if ca in cond_dists and cb in cond_dists:
            jsd_vec[k] = jensenshannon(cond_dists[ca], cond_dists[cb])

    # CORE TEST: does next-token similarity predict brain-LLM mismatch?
    print("\n=== CORE MECHANISTIC TEST ===")

    # JSD vs residual (higher JSD = more different predictions = LLM should preserve distinction)
    rho_jsd_res, p_jsd_res = spearmanr(jsd_vec, residual)
    print(f"  JSD vs brain-LLM residual:  ρ = {rho_jsd_res:+.4f}  p = {p_jsd_res:.4f}")
    print(f"  (Prediction: NEGATIVE — similar predictions → larger mismatch)")

    # JSD vs LLM distance (higher JSD = model should represent them as more different)
    rho_jsd_llm, p_jsd_llm = spearmanr(jsd_vec, llm_vec)
    print(f"  JSD vs LLM distance:        ρ = {rho_jsd_llm:+.4f}  p = {p_jsd_llm:.4f}")
    print(f"  (Prediction: POSITIVE — different predictions → larger LLM distance)")

    # JSD vs brain distance
    rho_jsd_brain, p_jsd_brain = spearmanr(jsd_vec, brain_vec)
    print(f"  JSD vs brain distance:      ρ = {rho_jsd_brain:+.4f}  p = {p_jsd_brain:.4f}")

    # Category-level JSD
    AFFECTIVE = {"anger","fear","disgust","sadness","happiness","valence"}
    MENTALISTIC = {"belief","mentalizing","intention","theory_of_mind","empathy","self_referential","judgment"}
    aff_pairs = [k for k, (a,b) in enumerate(pairs) if a in AFFECTIVE and b in AFFECTIVE]
    ment_pairs = [k for k, (a,b) in enumerate(pairs) if a in MENTALISTIC and b in MENTALISTIC]
    cross_pairs = [k for k, (a,b) in enumerate(pairs) if k not in aff_pairs and k not in ment_pairs]

    print(f"\n  Mean JSD by category:")
    print(f"    Within-affective:    {np.mean(jsd_vec[aff_pairs]):.4f}")
    print(f"    Within-mentalistic:  {np.mean(jsd_vec[ment_pairs]):.4f}")
    print(f"    Cross-domain:        {np.mean(jsd_vec[cross_pairs]):.4f}")

    # Unique variance: does JSD explain residual BEYOND lexical distance?
    # (would need GloVe RDM here, skip for now)

    # Top pairs by JSD (most similar predictions → most vulnerable to collapse)
    sorted_jsd = np.argsort(jsd_vec)
    print(f"\n  Top 5 MOST SIMILAR next-token predictions (collapse-prone):")
    for k in sorted_jsd[:5]:
        a, b = pairs[k]
        print(f"    {a:>18s} - {b:<18s}  JSD={jsd_vec[k]:.4f}  "
              f"residual={residual[k]:.0f}  brain_d={brain_vec[k]:.3f}")

    print(f"\n  Top 5 MOST DIFFERENT next-token predictions (preserved):")
    for k in sorted_jsd[-5:]:
        a, b = pairs[k]
        print(f"    {a:>18s} - {b:<18s}  JSD={jsd_vec[k]:.4f}  "
              f"residual={residual[k]:.0f}  brain_d={brain_vec[k]:.3f}")

    # Save
    results = {
        "model": args.model_short,
        "core_test": {
            "jsd_vs_residual": {"rho": float(rho_jsd_res), "p": float(p_jsd_res)},
            "jsd_vs_llm_dist": {"rho": float(rho_jsd_llm), "p": float(p_jsd_llm)},
            "jsd_vs_brain_dist": {"rho": float(rho_jsd_brain), "p": float(p_jsd_brain)},
        },
        "category_jsd": {
            "within_affective": float(np.mean(jsd_vec[aff_pairs])),
            "within_mentalistic": float(np.mean(jsd_vec[ment_pairs])),
            "cross_domain": float(np.mean(jsd_vec[cross_pairs])),
        },
        "pairs": [
            {"pair": f"{a}-{b}", "jsd": float(jsd_vec[k]),
             "residual": float(residual[k]),
             "brain_dist": float(brain_vec[k]),
             "llm_dist": float(llm_vec[k])}
            for k, (a, b) in enumerate(pairs)
        ],
    }
    out = Path(args.output_dir) / f"{args.model_short}_next_token_similarity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
