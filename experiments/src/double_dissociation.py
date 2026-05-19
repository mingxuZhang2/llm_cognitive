"""
Phase 3: Double dissociation via perplexity measurement.

For each model:
1. Auto-detect which module = math, which = code (from sample composition)
2. Ablate math module → measure perplexity on math vs code stimuli
3. Ablate code module → measure perplexity on code vs math stimuli
4. Report effect sizes and dissociation pattern

No external benchmark downloads needed — uses the same stimuli that defined the modules.
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def identify_modules(module_path, stimuli_path, target_categories):
    """Auto-detect which module ID corresponds to each target category."""
    modules = np.load(module_path)
    neuron_assign = modules["neuron_assign"]
    sample_assign = modules["sample_assign"]

    with open(stimuli_path) as f:
        stimuli = [json.loads(line) for line in f]

    K = sample_assign.max() + 1
    module_map = {}

    for cat in target_categories:
        best_k, best_purity = -1, 0
        for k in range(K):
            s_indices = np.where(sample_assign == k)[0]
            cats = {}
            for i in s_indices:
                if i < len(stimuli):
                    cats[stimuli[i]["category"]] = cats.get(stimuli[i]["category"], 0) + 1
            total = sum(cats.values())
            if total > 0:
                purity = cats.get(cat, 0) / total
                if purity > best_purity:
                    best_purity = purity
                    best_k = k

        n_neurons = int((neuron_assign == best_k).sum()) if best_k >= 0 else 0
        module_map[cat] = {
            "module_id": best_k,
            "purity": best_purity,
            "n_neurons": n_neurons,
            "pct_neurons": n_neurons / len(neuron_assign) if len(neuron_assign) > 0 else 0,
        }

    return module_map, neuron_assign, sample_assign


def compute_perplexity_by_category(model, tokenizer, stimuli, category_indices, device="cuda", max_length=256):
    """Compute mean perplexity for samples in each category."""
    model.eval()
    results = {}

    for cat, indices in category_indices.items():
        if not indices:
            results[cat] = {"perplexity": float("inf"), "n_samples": 0}
            continue

        total_loss = 0.0
        total_tokens = 0
        n_valid = 0

        for idx in indices:
            text = stimuli[idx]["text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

            if inputs["input_ids"].shape[1] < 2:
                continue

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            if not torch.isnan(loss) and not torch.isinf(loss):
                n_tokens = inputs["input_ids"].shape[1] - 1
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
                n_valid += 1

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            ppl = np.exp(min(avg_loss, 100))
        else:
            ppl = float("inf")

        results[cat] = {"perplexity": float(ppl), "n_samples": n_valid, "avg_loss": float(avg_loss) if total_tokens > 0 else None}

    return results


class NeuronAblator:
    """Zero out FFN neurons belonging to a target module during forward pass."""

    def __init__(self, model, neuron_assign, target_module_id, n_layers, ffn_dim):
        self.hooks = []
        module_mask = (neuron_assign == target_module_id)

        layers = None
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layers = model.gpt_neox.layers

        self.targets = []
        for layer_idx in range(n_layers):
            start = layer_idx * ffn_dim
            end = start + ffn_dim
            layer_mask = module_mask[start:end]
            if layer_mask.any():
                layer = layers[layer_idx]
                mlp = layer.mlp
                if hasattr(mlp, 'gate_proj'):
                    target_module = mlp.gate_proj
                elif hasattr(mlp, 'dense_h_to_4h'):
                    target_module = mlp.dense_h_to_4h
                else:
                    continue
                device = next(target_module.parameters()).device
                self.targets.append((target_module, torch.tensor(layer_mask, dtype=torch.bool, device=device)))

    def __enter__(self):
        for target_module, mask in self.targets:
            def make_hook(m):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        output[0][:, :, m] = 0.0
                        return output
                    output[:, :, m] = 0.0
                    return output
                return hook_fn
            h = target_module.register_forward_hook(make_hook(mask))
            self.hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_dissociation(
    model_path: str,
    model_short: str,
    module_path: str,
    meta_path: str,
    stimuli_path: str,
    output_path: str,
    K: int = 8,
    device: str = "cuda",
):
    """Run full double dissociation experiment for one model."""
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Double Dissociation: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]

    with open(stimuli_path) as f:
        stimuli = [json.loads(line) for line in f]

    # Build category → sample indices mapping
    cat_indices = {}
    for i, s in enumerate(stimuli):
        cat = s["category"]
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    # Auto-detect math and code modules
    print("\n[Step 1] Identifying math and code modules...")
    module_map, neuron_assign, sample_assign = identify_modules(
        module_path, stimuli_path, ["math", "code"]
    )
    for cat, info in module_map.items():
        print(f"  {cat}: module {info['module_id']}, purity={info['purity']:.2f}, "
              f"{info['n_neurons']:,} neurons ({info['pct_neurons']:.1%})")

    math_mod = module_map["math"]["module_id"]
    code_mod = module_map["code"]["module_id"]

    if math_mod == code_mod:
        print("WARNING: Math and code mapped to same module! Dissociation not possible.")
        return None

    # Load model
    print(f"\n[Step 2] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Categories to measure
    measure_cats = {cat: indices[:50] for cat, indices in cat_indices.items() if len(indices) >= 5}

    # Baseline perplexity
    print(f"\n[Step 3] Baseline perplexity ({len(measure_cats)} categories)...")
    baseline = compute_perplexity_by_category(model, tokenizer, stimuli, measure_cats, device)
    for cat, r in sorted(baseline.items()):
        print(f"  {cat:20s}: PPL = {r['perplexity']:>10.2f} ({r['n_samples']} samples)")

    # Ablate math module
    print(f"\n[Step 4] Ablating MATH module (module {math_mod}, {module_map['math']['n_neurons']:,} neurons)...")
    ablator_math = NeuronAblator(model, neuron_assign, math_mod, n_layers, ffn_dim)
    with ablator_math:
        ablated_math = compute_perplexity_by_category(model, tokenizer, stimuli, measure_cats, device)
    for cat, r in sorted(ablated_math.items()):
        delta = r['perplexity'] - baseline[cat]['perplexity']
        pct = delta / baseline[cat]['perplexity'] * 100 if baseline[cat]['perplexity'] > 0 else 0
        marker = " <<<" if cat == "math" else ""
        print(f"  {cat:20s}: PPL = {r['perplexity']:>10.2f} (delta={delta:>+8.2f}, {pct:>+6.1f}%){marker}")

    # Ablate code module
    print(f"\n[Step 5] Ablating CODE module (module {code_mod}, {module_map['code']['n_neurons']:,} neurons)...")
    ablator_code = NeuronAblator(model, neuron_assign, code_mod, n_layers, ffn_dim)
    with ablator_code:
        ablated_code = compute_perplexity_by_category(model, tokenizer, stimuli, measure_cats, device)
    for cat, r in sorted(ablated_code.items()):
        delta = r['perplexity'] - baseline[cat]['perplexity']
        pct = delta / baseline[cat]['perplexity'] * 100 if baseline[cat]['perplexity'] > 0 else 0
        marker = " <<<" if cat == "code" else ""
        print(f"  {cat:20s}: PPL = {r['perplexity']:>10.2f} (delta={delta:>+8.2f}, {pct:>+6.1f}%){marker}")

    # Random ablation control (same size as math module)
    print(f"\n[Step 6] Random ablation control ({module_map['math']['n_neurons']:,} neurons)...")
    rng = np.random.RandomState(42)
    random_assign = np.full_like(neuron_assign, -1)
    random_indices = rng.choice(len(neuron_assign), module_map['math']['n_neurons'], replace=False)
    random_assign[random_indices] = 99
    ablator_rand = NeuronAblator(model, random_assign, 99, n_layers, ffn_dim)
    with ablator_rand:
        ablated_random = compute_perplexity_by_category(model, tokenizer, stimuli, measure_cats, device)
    for cat, r in sorted(ablated_random.items()):
        delta = r['perplexity'] - baseline[cat]['perplexity']
        pct = delta / baseline[cat]['perplexity'] * 100 if baseline[cat]['perplexity'] > 0 else 0
        print(f"  {cat:20s}: PPL = {r['perplexity']:>10.2f} (delta={delta:>+8.2f}, {pct:>+6.1f}%)")

    # Compute dissociation metrics
    print(f"\n{'='*70}")
    print("DOUBLE DISSOCIATION SUMMARY")
    print(f"{'='*70}")

    math_on_math = ablated_math["math"]["perplexity"] / baseline["math"]["perplexity"]
    math_on_code = ablated_math["code"]["perplexity"] / baseline["code"]["perplexity"]
    code_on_code = ablated_code["code"]["perplexity"] / baseline["code"]["perplexity"]
    code_on_math = ablated_code["math"]["perplexity"] / baseline["math"]["perplexity"]
    rand_on_math = ablated_random["math"]["perplexity"] / baseline["math"]["perplexity"]
    rand_on_code = ablated_random["code"]["perplexity"] / baseline["code"]["perplexity"]

    print(f"\n  PPL ratio (ablated / baseline):")
    print(f"                    Math stimuli    Code stimuli")
    print(f"  Ablate Math:      {math_on_math:>10.2f}x     {math_on_code:>10.2f}x")
    print(f"  Ablate Code:      {code_on_math:>10.2f}x     {code_on_code:>10.2f}x")
    print(f"  Random control:   {rand_on_math:>10.2f}x     {rand_on_code:>10.2f}x")

    dissociation = (math_on_math > math_on_code * 1.5) and (code_on_code > code_on_math * 1.5)
    specificity = (math_on_math > rand_on_math * 1.2) or (code_on_code > rand_on_code * 1.2)

    print(f"\n  Double dissociation: {'YES' if dissociation else 'NO'}")
    print(f"  Specificity vs random: {'YES' if specificity else 'NO'}")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    # Save results
    result = {
        "model": model_short,
        "K": K,
        "module_map": module_map,
        "baseline": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for c, r in baseline.items()},
        "ablated_math": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for c, r in ablated_math.items()},
        "ablated_code": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for c, r in ablated_code.items()},
        "ablated_random": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for c, r in ablated_random.items()},
        "dissociation_ratios": {
            "math_ablation_on_math": float(math_on_math),
            "math_ablation_on_code": float(math_on_code),
            "code_ablation_on_code": float(code_on_code),
            "code_ablation_on_math": float(code_on_math),
            "random_on_math": float(rand_on_math),
            "random_on_code": float(rand_on_code),
        },
        "double_dissociation": dissociation,
        "specificity": specificity,
        "elapsed_seconds": elapsed,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Results saved to {output_path}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_short", type=str, required=True)
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--stimuli", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--K", type=int, default=8)
    args = parser.parse_args()

    run_dissociation(
        model_path=args.model_path,
        model_short=args.model_short,
        module_path=args.module_path,
        meta_path=args.meta_path,
        stimuli_path=args.stimuli,
        output_path=args.output,
        K=args.K,
    )
