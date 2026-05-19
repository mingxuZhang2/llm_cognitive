"""
Phase 4: Atlas-Guided Pruning.

Compare atlas-guided task-specific pruning against standard methods
(Wanda, magnitude pruning, random pruning).
"""

import json
import os
import copy
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_neuron_importance_magnitude(model):
    """Baseline: importance = absolute weight magnitude of FFN neurons."""
    importances = []
    for layer in model.model.layers:
        w = layer.mlp.gate_proj.weight.data if hasattr(layer.mlp, 'gate_proj') else layer.mlp.fc1.weight.data
        imp = w.abs().mean(dim=1)  # mean across input dim → [ffn_dim]
        importances.append(imp.cpu().numpy())
    return np.concatenate(importances)


def get_neuron_importance_wanda(model, calibration_activations):
    """
    Wanda-style importance: |weight| × |activation|.

    Args:
        model: the LLM
        calibration_activations: [n_neurons] mean activation magnitudes from calibration set
    """
    weight_imp = get_neuron_importance_magnitude(model)
    return weight_imp * calibration_activations


def atlas_guided_importance(base_importance, neuron_assign, protect_module_id, protect_factor=3.0):
    """
    Atlas-guided importance: multiply base importance by protection factor
    for neurons in the protected module.

    Args:
        base_importance: [n_neurons] base importance scores
        neuron_assign: [n_neurons] module assignments
        protect_module_id: which module to protect
        protect_factor: how much to boost protected neurons (default 3x)
    """
    guided = base_importance.copy()
    mask = neuron_assign == protect_module_id
    guided[mask] *= protect_factor
    return guided


def prune_by_importance(model, importance_scores, sparsity, n_layers, ffn_dim):
    """
    Unstructured pruning: zero out neurons with lowest importance.

    Returns a new model (deep copy) with pruned weights.
    """
    n_to_prune = int(len(importance_scores) * sparsity)
    threshold_idx = np.argsort(importance_scores)[n_to_prune]
    threshold = importance_scores[threshold_idx]

    pruned_model = copy.deepcopy(model)
    n_pruned = 0

    for layer_idx, layer in enumerate(pruned_model.model.layers):
        mlp = layer.mlp
        gate = mlp.gate_proj if hasattr(mlp, 'gate_proj') else mlp.fc1
        up = mlp.up_proj if hasattr(mlp, 'up_proj') else None

        start = layer_idx * ffn_dim
        end = start + ffn_dim
        layer_imp = importance_scores[start:end]
        prune_mask = layer_imp < threshold

        with torch.no_grad():
            gate.weight.data[prune_mask] = 0.0
            if up is not None:
                up.weight.data[prune_mask] = 0.0

        n_pruned += prune_mask.sum()

    actual_sparsity = n_pruned / len(importance_scores)
    return pruned_model, actual_sparsity


def evaluate_model(model, tokenizer, benchmarks, batch_size=8):
    """Run lm-eval-harness benchmarks."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = lm_eval.simple_evaluate(model=lm, tasks=benchmarks, batch_size=batch_size)

    scores = {}
    for task in benchmarks:
        task_results = results["results"].get(task, {})
        acc = task_results.get("acc,none", task_results.get("acc_norm,none", None))
        if acc is not None:
            scores[task] = float(acc)
    return scores


def run_pruning_experiment(
    model_name: str,
    module_path: str,
    meta_path: str,
    activation_path: str,
    protect_module_id: int,
    protect_module_name: str,
    target_benchmarks: list[str],
    collateral_benchmarks: list[str],
    sparsity_levels: list[float],
    output_path: str,
    device: str = "cuda",
):
    """
    Run full pruning comparison experiment.

    Compares: atlas-guided, wanda-style, magnitude, random pruning.
    """
    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]

    modules = np.load(module_path)
    neuron_assign = modules["neuron_assign"]

    A = np.load(activation_path)
    calibration_act = A.mean(axis=1)  # mean activation across all samples
    calibration_act = np.abs(calibration_act)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    all_benchmarks = list(set(target_benchmarks + collateral_benchmarks))

    print("Evaluating baseline (no pruning)...")
    baseline_scores = evaluate_model(model, tokenizer, all_benchmarks)
    print(f"Baseline: {baseline_scores}")

    mag_importance = get_neuron_importance_magnitude(model)
    wanda_importance = get_neuron_importance_wanda(model, calibration_act)

    results = {
        "model": model_name,
        "protect_module": protect_module_name,
        "protect_module_id": protect_module_id,
        "baseline": baseline_scores,
        "experiments": [],
    }

    for sparsity in sparsity_levels:
        print(f"\n{'='*60}")
        print(f"Sparsity: {sparsity:.0%}")
        print(f"{'='*60}")

        methods = {
            "atlas_guided_wanda": atlas_guided_importance(wanda_importance, neuron_assign, protect_module_id),
            "atlas_guided_magnitude": atlas_guided_importance(mag_importance, neuron_assign, protect_module_id),
            "wanda": wanda_importance,
            "magnitude": mag_importance,
            "random": np.random.RandomState(42).random(len(mag_importance)),
        }

        for method_name, importance in methods.items():
            print(f"\n  Method: {method_name}")
            pruned_model, actual_sparsity = prune_by_importance(
                model, importance, sparsity, n_layers, ffn_dim
            )
            pruned_model = pruned_model.to(device)

            scores = evaluate_model(pruned_model, tokenizer, all_benchmarks)

            exp = {
                "sparsity": sparsity,
                "actual_sparsity": float(actual_sparsity),
                "method": method_name,
                "scores": scores,
                "target_avg": np.mean([scores.get(t, 0) for t in target_benchmarks]),
                "collateral_avg": np.mean([scores.get(t, 0) for t in collateral_benchmarks]),
            }
            results["experiments"].append(exp)

            target_str = ", ".join([f"{t}={scores.get(t, 0):.3f}" for t in target_benchmarks])
            print(f"    Target: {target_str}")
            print(f"    Collateral avg: {exp['collateral_avg']:.3f}")

            del pruned_model
            torch.cuda.empty_cache()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--activation_path", type=str, required=True)
    parser.add_argument("--protect_module", type=int, required=True)
    parser.add_argument("--protect_name", type=str, required=True)
    parser.add_argument("--target_benchmarks", nargs="+", required=True)
    parser.add_argument("--collateral_benchmarks", nargs="+", required=True)
    parser.add_argument("--sparsity", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_pruning_experiment(
        model_name=args.model,
        module_path=args.module_path,
        meta_path=args.meta_path,
        activation_path=args.activation_path,
        protect_module_id=args.protect_module,
        protect_module_name=args.protect_name,
        target_benchmarks=args.target_benchmarks,
        collateral_benchmarks=args.collateral_benchmarks,
        sparsity_levels=args.sparsity,
        output_path=args.output,
    )
