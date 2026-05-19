"""
Atlas-guided pruning: use causal attribution to do smarter pruning.

Strategy: protect neurons that are highly selective for ANY function.
Prune neurons with low causal importance across ALL categories.

Compare against baselines:
1. Random pruning
2. Magnitude pruning (smallest weights)
3. Atlas-guided pruning (protect functional neurons)

Evaluate: PPL across all categories at various sparsity levels.
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_ppl(model, tokenizer, stimuli, cat_indices, device="cuda", max_length=256):
    model.eval()
    results = {}
    for cat, indices in cat_indices.items():
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
                nt = inputs["input_ids"].shape[1] - 1
                total_loss += loss.item() * nt
                total_tokens += nt
                n_valid += 1
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            ppl = np.exp(min(avg_loss, 100))
        else:
            avg_loss, ppl = float("inf"), float("inf")
        results[cat] = {"perplexity": float(ppl), "avg_loss": float(avg_loss), "n_samples": n_valid}
    return results


def get_weight_magnitudes(model, n_layers, ffn_dim):
    """Get L2 norm of gate_proj weights per neuron."""
    def get_layers(m):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        return None

    layers = get_layers(model)
    magnitudes = np.zeros(n_layers * ffn_dim, dtype=np.float32)
    for layer_idx in range(n_layers):
        mlp = layers[layer_idx].mlp
        if hasattr(mlp, "gate_proj"):
            w = mlp.gate_proj.weight.data
        elif hasattr(mlp, "dense_h_to_4h"):
            w = mlp.dense_h_to_4h.weight.data
        else:
            continue
        mag = w.float().norm(dim=1).cpu().numpy()
        start = layer_idx * ffn_dim
        magnitudes[start:start + ffn_dim] = mag
    return magnitudes


def prune_and_measure(model, tokenizer, stimuli, prune_indices,
                      n_layers, ffn_dim, cat_indices, device="cuda", max_length=256):
    """Hook-based pruning: zero output for pruned neurons during forward pass."""
    def get_layers(m):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        return None

    layers = get_layers(model)
    hooks = []
    prune_set = set(int(n) for n in prune_indices)

    for layer_idx in range(n_layers):
        start = layer_idx * ffn_dim
        end = start + ffn_dim
        layer_neurons = [n - start for n in prune_set if start <= n < end]
        if not layer_neurons:
            continue
        mlp = layers[layer_idx].mlp
        if hasattr(mlp, "gate_proj"):
            target = mlp.gate_proj
        elif hasattr(mlp, "dense_h_to_4h"):
            target = mlp.dense_h_to_4h
        else:
            continue
        mask = torch.zeros(ffn_dim, dtype=torch.bool, device=device)
        mask[layer_neurons] = True

        def make_hook(m):
            def hook_fn(module, input, output):
                output[:, :, m] = 0.0
                return output
            return hook_fn
        hooks.append(target.register_forward_hook(make_hook(mask)))

    results = measure_ppl(model, tokenizer, stimuli, cat_indices, device, max_length)

    for h in hooks:
        h.remove()
    return results


def run_atlas_pruning(
    model_path, model_short, meta_path, attribution_path,
    stimuli_path, output_dir, sparsity_levels=None, device="cuda",
):
    t0 = time.time()
    if sparsity_levels is None:
        sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"{'='*70}")
    print(f"Atlas-Guided Pruning: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim

    with open(stimuli_path) as f:
        stimuli = [json.loads(line) for line in f]

    cat_indices = {}
    for i, s in enumerate(stimuli):
        cat = s["category"]
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    # Load attribution
    attr_data = np.load(attribution_path)
    categories = sorted(set(k.rsplit("_", 1)[0] for k in attr_data.files if k.endswith("_importance")))

    # Compute max importance across all categories
    all_importances = np.stack([attr_data[f"{cat}_importance"] for cat in categories])
    max_importance = all_importances.max(axis=0)
    max_selectivity = np.stack([np.abs(attr_data[f"{cat}_selectivity"]) for cat in categories]).max(axis=0)

    # Atlas score: important AND selective neurons are protected
    atlas_score = max_importance * (1 + max_selectivity)

    print(f"  Neurons: {n_neurons:,}")
    print(f"  Categories: {categories}")

    # Load model
    print(f"\n[Step 1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Weight magnitudes
    print(f"  Computing weight magnitudes...")
    magnitudes = get_weight_magnitudes(model, n_layers, ffn_dim)

    # Baseline
    print(f"\n[Step 2] Baseline perplexity...")
    baseline = measure_ppl(model, tokenizer, stimuli, cat_indices, device)
    baseline_avg_ppl = np.mean([baseline[c]["perplexity"] for c in cat_indices])
    print(f"  Average PPL: {baseline_avg_ppl:.2f}")
    for cat in sorted(baseline):
        print(f"    {cat:>12s}: PPL={baseline[cat]['perplexity']:>8.2f}")

    # Pruning orderings (ascending = prune first)
    rng = np.random.RandomState(42)
    methods = [
        ("Random", rng.permutation(n_neurons)),
        ("Magnitude", np.argsort(magnitudes)),
        ("Atlas-guided", np.argsort(atlas_score)),
    ]

    print(f"\n[Step 3] Pruning experiments...")
    all_results = {
        "model": model_short,
        "n_neurons": n_neurons,
        "baseline": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in r.items()} for c, r in baseline.items()},
        "sparsity_levels": sparsity_levels,
        "methods": {},
    }

    for method_name, order in methods:
        print(f"\n  === {method_name} pruning ===")
        method_results = []

        for sparsity in sparsity_levels:
            n_prune = int(n_neurons * sparsity)
            prune_indices = order[:n_prune]

            print(f"    Sparsity {sparsity:.0%} ({n_prune:,} neurons)...", end=" ")
            pruned = prune_and_measure(
                model, tokenizer, stimuli, prune_indices,
                n_layers, ffn_dim, cat_indices, device,
            )

            avg_ppl = np.mean([pruned[c]["perplexity"] for c in cat_indices])
            avg_ratio = avg_ppl / baseline_avg_ppl
            cat_ratios = {c: float(pruned[c]["perplexity"] / baseline[c]["perplexity"]) for c in cat_indices}

            print(f"avg PPL={avg_ppl:.2f} ({avg_ratio:.2f}x)")

            method_results.append({
                "sparsity": sparsity,
                "n_pruned": n_prune,
                "avg_ppl": float(avg_ppl),
                "avg_ratio": float(avg_ratio),
                "per_category": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                     for k, v in r.items()} for c, r in pruned.items()},
                "category_ratios": cat_ratios,
            })

        all_results["methods"][method_name] = method_results

    # Summary
    print(f"\n{'='*70}")
    print(f"PRUNING COMPARISON (avg PPL ratio vs baseline)")
    print(f"{'='*70}")
    header = f"  {'Sparsity':>10s}"
    for mn, _ in methods:
        header += f"  {mn:>14s}"
    print(header)

    for si, sparsity in enumerate(sparsity_levels):
        row = f"  {sparsity:>10.0%}"
        for mn, _ in methods:
            ratio = all_results["methods"][mn][si]["avg_ratio"]
            row += f"  {ratio:>14.2f}x"
        print(row)

    print(f"\n  Atlas advantage over Magnitude:")
    for si, sparsity in enumerate(sparsity_levels):
        mag_r = all_results["methods"]["Magnitude"][si]["avg_ratio"]
        atl_r = all_results["methods"]["Atlas-guided"][si]["avg_ratio"]
        adv = (mag_r - atl_r) / mag_r * 100
        print(f"    {sparsity:.0%}: Atlas {atl_r:.2f}x vs Magnitude {mag_r:.2f}x ({adv:+.1f}%)")

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed

    out_path = os.path.join(output_dir, f"{model_short}_atlas_pruning.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out_path} ({elapsed:.0f}s)")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_short", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--attribution_path", type=str, required=True)
    parser.add_argument("--stimuli", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sparsity", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    run_atlas_pruning(
        model_path=args.model_path,
        model_short=args.model_short,
        meta_path=args.meta_path,
        attribution_path=args.attribution_path,
        stimuli_path=args.stimuli,
        output_dir=args.output_dir,
        sparsity_levels=args.sparsity,
    )
