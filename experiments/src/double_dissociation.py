"""
Phase 3: Double dissociation experiments.

Ablate functional modules and measure task-specific performance changes.
The gold standard for causal functional specificity.
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModuleAblator:
    """Hook-based module ablation: zeros out neurons belonging to a specified module."""

    def __init__(self, model, neuron_assign, target_module_id, n_layers, ffn_dim):
        self.model = model
        self.hooks = []

        module_mask = (neuron_assign == target_module_id)
        self.layer_masks = {}
        for layer_idx in range(n_layers):
            start = layer_idx * ffn_dim
            end = start + ffn_dim
            layer_mask = module_mask[start:end]
            if layer_mask.any():
                self.layer_masks[layer_idx] = torch.tensor(layer_mask, dtype=torch.bool)

    def _make_hook(self, layer_idx):
        mask = self.layer_masks[layer_idx].to(self.model.device)
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
                act[:, :, mask] = 0.0
                return (act,) + output[1:]
            else:
                output[:, :, mask] = 0.0
                return output
        return hook_fn

    def attach(self):
        """Attach ablation hooks to the model."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx not in self.layer_masks:
                continue
            mlp = layer.mlp
            target = mlp.gate_proj if hasattr(mlp, 'gate_proj') else mlp.fc1
            h = target.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(h)
        return self

    def detach(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, *args):
        self.detach()


def run_ablation_benchmark(
    model_name: str,
    module_path: str,
    meta_path: str,
    target_module_id: int,
    benchmarks: list[str],
    output_path: str,
    device: str = "cuda",
):
    """
    Run benchmarks with a specific module ablated.

    Uses lm-eval-harness for standardized evaluation.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    with open(meta_path) as f:
        meta = json.load(f)

    modules = np.load(module_path)
    neuron_assign = modules["neuron_assign"]
    n_neurons_ablated = (neuron_assign == target_module_id).sum()

    print(f"Ablating module {target_module_id}: {n_neurons_ablated} neurons "
          f"({n_neurons_ablated / len(neuron_assign) * 100:.1f}%)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    ablator = ModuleAblator(
        model, neuron_assign, target_module_id,
        meta["n_layers"], meta["ffn_dim"]
    )

    # Baseline (no ablation)
    print("Running baseline (no ablation)...")
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    baseline_results = lm_eval.simple_evaluate(
        model=lm, tasks=benchmarks, batch_size=8
    )

    # Ablated
    print(f"Running with module {target_module_id} ablated...")
    with ablator:
        ablated_results = lm_eval.simple_evaluate(
            model=lm, tasks=benchmarks, batch_size=8
        )

    # Random control (same number of neurons, random selection)
    print("Running random ablation control...")
    rng = np.random.RandomState(42)
    random_assign = np.full_like(neuron_assign, -1)
    random_indices = rng.choice(len(neuron_assign), n_neurons_ablated, replace=False)
    random_assign[random_indices] = target_module_id

    random_ablator = ModuleAblator(
        model, random_assign, target_module_id,
        meta["n_layers"], meta["ffn_dim"]
    )
    with random_ablator:
        random_results = lm_eval.simple_evaluate(
            model=lm, tasks=benchmarks, batch_size=8
        )

    output = {
        "model": model_name,
        "ablated_module": target_module_id,
        "n_neurons_ablated": int(n_neurons_ablated),
        "pct_neurons_ablated": float(n_neurons_ablated / len(neuron_assign)),
        "baseline": {t: baseline_results["results"][t] for t in benchmarks},
        "ablated": {t: ablated_results["results"][t] for t in benchmarks},
        "random_control": {t: random_results["results"][t] for t in benchmarks},
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    for t in benchmarks:
        b_acc = baseline_results["results"][t].get("acc,none", baseline_results["results"][t].get("acc_norm,none", 0))
        a_acc = ablated_results["results"][t].get("acc,none", ablated_results["results"][t].get("acc_norm,none", 0))
        r_acc = random_results["results"][t].get("acc,none", random_results["results"][t].get("acc_norm,none", 0))
        print(f"  {t}: baseline={b_acc:.3f} → ablated={a_acc:.3f} (Δ={a_acc-b_acc:+.3f}) | random={r_acc:.3f}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--target_module", type=int, required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_ablation_benchmark(
        model_name=args.model,
        module_path=args.module_path,
        meta_path=args.meta_path,
        target_module_id=args.target_module,
        benchmarks=args.benchmarks,
        output_path=args.output,
    )
