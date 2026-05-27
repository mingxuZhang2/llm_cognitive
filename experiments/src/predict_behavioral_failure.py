#!/usr/bin/env python3
"""
Predict LLM behavioral failures from brain-LLM RDM mismatch.

For each pair of cognitive conditions (A, B):
  - Sample stimuli from A and B
  - Ask LLM to classify: "Is this about A or B?"
  - Compute per-pair accuracy
  - Correlate with brain-LLM RDM residual

If mismatch predicts failure: brain geometry reveals LLM's cognitive blind spots.

Usage:
  python predict_behavioral_failure.py \
    --model_path /path/to/model \
    --model_short Qwen2.5-3B-Instruct \
    --stimuli_jsonl /path/to/rsa_stimuli.jsonl \
    --brain_rdm /path/to/brain_rdm.npz \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

CONDITION_DESCRIPTIONS = {
    "anger": "anger or hostility",
    "fear": "fear or anxiety",
    "disgust": "disgust or revulsion",
    "sadness": "sadness or grief",
    "happiness": "happiness or joy",
    "valence": "emotional valence or feeling",
    "belief": "someone's beliefs or assumptions about the world",
    "mentalizing": "reasoning about someone's mental state or knowledge",
    "intention": "someone's intentions, plans, or goals",
    "theory_of_mind": "understanding that someone's knowledge differs from reality (false belief, deception)",
    "empathy": "empathic concern or compassion for someone",
    "self_referential": "self-reflection or personal introspection",
    "judgment": "evaluative judgment (right/wrong, fair/unfair)",
    "moral": "moral or ethical reasoning",
}


def classify_pair(model, tokenizer, text, cond_a, cond_b, device):
    """Ask model to classify a text as belonging to condition A or B."""
    desc_a = CONDITION_DESCRIPTIONS[cond_a]
    desc_b = CONDITION_DESCRIPTIONS[cond_b]

    prompt = f"""Read the following text and decide which cognitive process it primarily engages.

Text: "{text}"

Option A: {desc_a}
Option B: {desc_b}

Answer with just A or B:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                               skip_special_tokens=True).strip().upper()

    if "A" in response[:3] and "B" not in response[:3]:
        return "A"
    elif "B" in response[:3] and "A" not in response[:3]:
        return "B"
    return "unclear"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--stimuli_jsonl", required=True)
    parser.add_argument("--brain_rdm", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    # Load stimuli
    stimuli = [json.loads(l) for l in open(args.stimuli_jsonl)]
    by_cond = defaultdict(list)
    for s in stimuli:
        c = s.get("condition", s.get("category"))
        t = s.get("text", s.get("prompt", ""))
        by_cond[c].append(t)

    # Load brain RDM
    brain_data = np.load(args.brain_rdm, allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])

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

    # Test all condition pairs
    conditions = [c for c in brain_conds if c in by_cond and len(by_cond[c]) >= 5]
    n_conds = len(conditions)
    print(f"Testing {n_conds} conditions, {n_conds*(n_conds-1)//2} pairs")

    rng = random.Random(42)
    pair_results = []

    for i, cond_a in enumerate(conditions):
        for j, cond_b in enumerate(conditions):
            if j <= i:
                continue

            correct = 0
            total = 0
            texts_a = rng.sample(by_cond[cond_a], min(args.n_trials, len(by_cond[cond_a])))
            texts_b = rng.sample(by_cond[cond_b], min(args.n_trials, len(by_cond[cond_b])))

            for t in texts_a:
                ans = classify_pair(model, tokenizer, t, cond_a, cond_b, device)
                if ans == "A":
                    correct += 1
                total += 1

            for t in texts_b:
                ans = classify_pair(model, tokenizer, t, cond_a, cond_b, device)
                if ans == "B":
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else 0.5

            bi = brain_conds.index(cond_a)
            bj = brain_conds.index(cond_b)
            brain_dist = brain_rdm[bi, bj]

            pair_results.append({
                "cond_a": cond_a, "cond_b": cond_b,
                "accuracy": acc, "n_trials": total,
                "brain_distance": float(brain_dist),
            })

            if len(pair_results) % 10 == 0:
                print(f"  {len(pair_results)} pairs done...")

    # Compute brain-LLM RDM residual per pair
    # Load LLM RDM (from existing NPZ or compute)
    llm_npz = Path(args.output_dir).parent / "cognitive_rsa" / f"{args.model_short}_rsa_v2_per_stim.npz"
    if llm_npz.exists():
        llm_data = np.load(llm_npz, allow_pickle=True)
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

        triu = np.triu_indices(len(brain_conds), k=1)
        brain_vec = brain_rdm[triu]
        from scipy.stats import rankdata
        brain_ranks = rankdata(brain_vec)

        best_L = 0
        best_rho = -1
        for L in range(n_layers):
            act = cond_means[:, L, :].astype(np.float64)
            act = act - act.mean(axis=0)
            norms = np.linalg.norm(act, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            rdm = 1.0 - np.clip((act/norms) @ (act/norms).T, -1, 1)
            rho, _ = spearmanr(brain_vec, rdm[triu])
            if rho > best_rho:
                best_rho, best_L = rho, L

        act = cond_means[:, best_L, :].astype(np.float64)
        act = act - act.mean(axis=0)
        norms = np.linalg.norm(act, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        llm_rdm = 1.0 - np.clip((act/norms) @ (act/norms).T, -1, 1)
        llm_ranks = rankdata(llm_rdm[triu])
        residuals = np.abs(brain_ranks - llm_ranks)

        # Map residuals to pair results
        for pr in pair_results:
            bi = brain_conds.index(pr["cond_a"])
            bj = brain_conds.index(pr["cond_b"])
            pair_idx = None
            for k, (ti, tj) in enumerate(zip(*triu)):
                if (ti == bi and tj == bj) or (ti == bj and tj == bi):
                    pair_idx = k
                    break
            if pair_idx is not None:
                pr["rdm_residual"] = float(residuals[pair_idx])
                pr["llm_distance"] = float(llm_rdm[bi, bj])

        del per_stim, llm_data

    # Core test: does RDM residual predict behavioral accuracy?
    accs = np.array([pr["accuracy"] for pr in pair_results])
    residuals_arr = np.array([pr.get("rdm_residual", 0) for pr in pair_results])
    brain_dists = np.array([pr["brain_distance"] for pr in pair_results])
    llm_dists = np.array([pr.get("llm_distance", 0) for pr in pair_results])

    rho_res_acc, p_res_acc = spearmanr(residuals_arr, accs)
    rho_bd_acc, p_bd_acc = spearmanr(brain_dists, accs)
    rho_ld_acc, p_ld_acc = spearmanr(llm_dists, accs)

    print(f"\n=== PREDICTION TEST ===")
    print(f"  RDM residual vs accuracy:  ρ = {rho_res_acc:+.4f}  p = {p_res_acc:.4f}")
    print(f"  Brain distance vs accuracy: ρ = {rho_bd_acc:+.4f}  p = {p_bd_acc:.4f}")
    print(f"  LLM distance vs accuracy:   ρ = {rho_ld_acc:+.4f}  p = {p_ld_acc:.4f}")
    print(f"\n  Mean accuracy: {np.mean(accs):.1%}")
    print(f"  Accuracy range: [{np.min(accs):.1%}, {np.max(accs):.1%}]")

    # Top easiest and hardest pairs
    sorted_by_acc = sorted(pair_results, key=lambda x: x["accuracy"])
    print(f"\n  5 HARDEST pairs (lowest accuracy):")
    for pr in sorted_by_acc[:5]:
        print(f"    {pr['cond_a']:>18s} vs {pr['cond_b']:<18s}  "
              f"acc={pr['accuracy']:.0%}  residual={pr.get('rdm_residual',0):.0f}")
    print(f"\n  5 EASIEST pairs (highest accuracy):")
    for pr in sorted_by_acc[-5:]:
        print(f"    {pr['cond_a']:>18s} vs {pr['cond_b']:<18s}  "
              f"acc={pr['accuracy']:.0%}  residual={pr.get('rdm_residual',0):.0f}")

    # Save
    output = {
        "model": args.model_short,
        "n_conditions": n_conds,
        "n_pairs": len(pair_results),
        "prediction": {
            "residual_vs_accuracy": {"rho": float(rho_res_acc), "p": float(p_res_acc)},
            "brain_dist_vs_accuracy": {"rho": float(rho_bd_acc), "p": float(p_bd_acc)},
            "llm_dist_vs_accuracy": {"rho": float(rho_ld_acc), "p": float(p_ld_acc)},
        },
        "pairs": pair_results,
    }
    out_path = Path(args.output_dir) / f"{args.model_short}_behavioral_prediction.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
