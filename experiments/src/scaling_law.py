"""
Phase 5: Modularity Scaling Law.

Measure functional modularity (Q-score) across Pythia model sizes
and correlate with downstream benchmark performance.
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

from activation_extraction import extract_activations
from module_discovery import iterd, objective


def modularity_q_score(A, neuron_assign, sample_assign, K):
    """
    Compute Newman modularity Q-score.

    Q measures how much more within-module activation exists compared
    to what would be expected by chance in a random assignment.
    """
    n_neurons, n_samples = A.shape
    total_activation = A.sum()
    if total_activation == 0:
        return 0.0

    Q = 0.0
    for k in range(K):
        u_mask = neuron_assign == k
        s_mask = sample_assign == k

        within = A[np.ix_(u_mask, s_mask)].sum()

        row_sum = A[u_mask, :].sum()
        col_sum = A[:, s_mask].sum()

        expected = (row_sum * col_sum) / total_activation
        Q += (within - expected) / total_activation

    return float(Q)


def run_scaling_analysis(
    model_names: list[str],
    stimuli_path: str,
    output_dir: str,
    K: int = 10,
    n_samples: int = 2000,
    device: str = "cuda",
):
    """
    Run modularity analysis across model sizes.

    For each model: extract activations → discover modules → compute Q-score → run benchmarks.
    """
    with open(stimuli_path) as f:
        all_samples = [json.loads(line)["text"] for line in f]
    samples = all_samples[:n_samples]

    results = []

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model_short = model_name.split("/")[-1]
        act_dir = os.path.join(output_dir, "activations")
        mod_dir = os.path.join(output_dir, "modules")
        os.makedirs(act_dir, exist_ok=True)
        os.makedirs(mod_dir, exist_ok=True)

        act_path = os.path.join(act_dir, f"{model_short}_activations.npy")
        if not os.path.exists(act_path):
            print("Extracting activations...")
            A, meta = extract_activations(
                model_name=model_name,
                samples=samples,
                output_dir=act_dir,
                max_length=256,
                batch_size=8,
                device=device,
            )
        else:
            print(f"Loading cached activations from {act_path}")
            A = np.load(act_path)

        print(f"Running IterD with K={K}...")
        neuron_assign, sample_assign, scores = iterd(A, K, max_iter=50, verbose=False)

        Q = modularity_q_score(A, neuron_assign, sample_assign, K)
        L = scores[-1]

        n_params = sum(
            p.numel() for p in
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).parameters()
        )

        module_sizes = [(neuron_assign == k).sum() for k in range(K)]
        size_entropy = stats.entropy([s / sum(module_sizes) for s in module_sizes if s > 0])

        entry = {
            "model": model_name,
            "n_params": int(n_params),
            "log_params": float(np.log10(n_params)),
            "Q_score": Q,
            "L_score": L,
            "module_size_entropy": float(size_entropy),
            "module_sizes": [int(s) for s in module_sizes],
            "K": K,
        }
        results.append(entry)

        print(f"  Params: {n_params:,}")
        print(f"  Q-score: {Q:.4f}")
        print(f"  L-score: {L:.4f}")
        print(f"  Module size entropy: {size_entropy:.4f}")

        np.savez(
            os.path.join(mod_dir, f"{model_short}_K{K}_modules.npz"),
            neuron_assign=neuron_assign,
            sample_assign=sample_assign,
        )

        torch.cuda.empty_cache()

    # Correlation analysis
    log_params = [r["log_params"] for r in results]
    q_scores = [r["Q_score"] for r in results]

    if len(results) >= 3:
        r, p = stats.pearsonr(log_params, q_scores)
        slope, intercept, _, _, stderr = stats.linregress(log_params, q_scores)
        fit = {"pearson_r": r, "p_value": p, "slope": slope, "intercept": intercept, "stderr": stderr}
    else:
        fit = {"note": "too few data points for regression"}

    output = {
        "models": results,
        "scaling_fit": fit,
        "config": {"K": K, "n_samples": n_samples},
    }

    output_path = os.path.join(output_dir, "scaling_law_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Scaling Law Results")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']:40s} params={r['n_params']:>12,}  Q={r['Q_score']:.4f}")
    if "pearson_r" in fit:
        print(f"\n  Q ~ {fit['slope']:.4f} × log10(params) + {fit['intercept']:.4f}")
        print(f"  Pearson r = {fit['pearson_r']:.4f}, p = {fit['p_value']:.4e}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--models", nargs="+", default=[
        "EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    ])
    args = parser.parse_args()

    run_scaling_analysis(args.models, args.stimuli, args.output_dir, args.K, args.n_samples)
