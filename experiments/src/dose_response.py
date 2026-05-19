"""
Dose-response curves for causal ablation.

For a single model (Qwen) and two functions (math, code):
  - Load pre-computed selectivity from multi_attribution.npz
  - For each ablation size (500, 1000, 2000, 5000, 10000, 20000):
    - Ablate the top-N most selective neurons for that function
    - Measure PPL across all 8 categories
  - Random ablation at each size as control

Outputs a JSON with the full dose-response data.
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_ppl(model, tokenizer, stimuli, cat_indices, device="cuda", max_length=256):
    """Measure PPL for each category."""
    model.eval()
    results = {}
    for cat, indices in cat_indices.items():
        total_loss = 0.0
        total_tokens = 0
        n_valid = 0
        for idx in indices:
            text = stimuli[idx]["text"]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss_val = outputs.loss
            if not torch.isnan(loss_val) and not torch.isinf(loss_val):
                nt = inputs["input_ids"].shape[1] - 1
                total_loss += loss_val.item() * nt
                total_tokens += nt
                n_valid += 1
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            ppl = np.exp(min(avg_loss, 100))
        else:
            avg_loss = float("inf")
            ppl = float("inf")
        results[cat] = {
            "perplexity": float(ppl),
            "avg_loss": float(avg_loss),
            "n_samples": n_valid,
        }
    return results


def ablate_and_measure(model, tokenizer, stimuli, neuron_indices,
                       n_layers, ffn_dim, cat_indices, device="cuda", max_length=256):
    """Ablate specific neurons and measure PPL across all categories."""
    def get_layers(m):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        return None

    layers = get_layers(model)
    hooks = []
    neuron_set = set(int(n) for n in neuron_indices)

    for layer_idx in range(n_layers):
        start = layer_idx * ffn_dim
        end = start + ffn_dim
        layer_neurons = [n - start for n in neuron_set if start <= n < end]
        if not layer_neurons:
            continue
        layer = layers[layer_idx]
        mlp = layer.mlp
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


def run_dose_response(
    model_path, model_short, meta_path, stimuli_path, attribution_path, output_dir,
    target_functions=("math", "code"),
    ablation_sizes=(500, 1000, 2000, 5000, 10000, 20000),
    device="cuda",
):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Dose-Response Curves: {model_short}")
    print(f"  Functions: {target_functions}")
    print(f"  Ablation sizes: {ablation_sizes}")
    print(f"{'='*70}")

    # ── Load metadata ────────────────────────────────────────────────────
    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim
    print(f"  n_layers={n_layers}, ffn_dim={ffn_dim}, n_neurons={n_neurons:,}")

    # ── Load stimuli ─────────────────────────────────────────────────────
    with open(stimuli_path) as f:
        stimuli = [json.loads(line) for line in f]
    cat_indices = {}
    for i, s in enumerate(stimuli):
        cat = s["category"]
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)
    categories = sorted(cat_indices.keys())
    print(f"  Categories: {categories}")
    print(f"  Samples per category: {', '.join(f'{c}={len(v)}' for c, v in sorted(cat_indices.items()))}")

    # ── Load pre-computed selectivity ────────────────────────────────────
    print(f"\n[Step 1] Loading pre-computed selectivity from {attribution_path}...")
    data = np.load(attribution_path)
    selectivity = {}
    for func in target_functions:
        key = f"{func}_selectivity"
        if key not in data:
            raise KeyError(f"Missing '{key}' in {attribution_path}. Available keys: {list(data.keys())}")
        selectivity[func] = data[key]
        n_high = (selectivity[func] > 0.3).sum()
        print(f"  {func}: loaded selectivity, {n_high:,} neurons with sel>0.3, max={selectivity[func].max():.4f}")

    # ── Pre-sort neurons for each function ───────────────────────────────
    sorted_neurons = {}
    for func in target_functions:
        sorted_neurons[func] = np.argsort(-selectivity[func])

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n[Step 2] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Baseline PPL ─────────────────────────────────────────────────────
    print(f"\n[Step 3] Measuring baseline perplexity...")
    baseline = measure_ppl(model, tokenizer, stimuli, cat_indices, device)
    for cat in categories:
        print(f"  {cat:>12s}: PPL={baseline[cat]['perplexity']:>8.2f}")

    # ── Dose-response: selective ablation ────────────────────────────────
    print(f"\n[Step 4] Dose-response ablation...")
    dose_results = {}

    for func in target_functions:
        dose_results[func] = {}
        for n_ablate in ablation_sizes:
            if n_ablate > n_neurons:
                print(f"  WARNING: n_ablate={n_ablate} > n_neurons={n_neurons}, skipping")
                continue
            top_neurons = sorted_neurons[func][:n_ablate]
            sel_min = selectivity[func][top_neurons[-1]]
            sel_max = selectivity[func][top_neurons[0]]
            pct = n_ablate / n_neurons * 100
            print(f"\n  Ablating top {n_ablate:>6,} {func}-selective neurons "
                  f"({pct:.2f}%, sel=[{sel_min:.4f}, {sel_max:.4f}])...")

            ablated = ablate_and_measure(
                model, tokenizer, stimuli, top_neurons,
                n_layers, ffn_dim, cat_indices, device,
            )
            dose_results[func][n_ablate] = ablated

            # Print damage profile
            for cat in categories:
                bl = baseline[cat]["avg_loss"]
                al = ablated[cat]["avg_loss"]
                delta = al - bl
                ratio = ablated[cat]["perplexity"] / baseline[cat]["perplexity"]
                marker = " <<<" if cat == func else ""
                print(f"    {cat:>12s}: delta={delta:>+.4f}, ratio={ratio:.2f}x{marker}")

    # ── Random ablation control at each size ─────────────────────────────
    print(f"\n[Step 5] Random ablation controls...")
    rng = np.random.RandomState(42)
    random_results = {}

    for n_ablate in ablation_sizes:
        if n_ablate > n_neurons:
            continue
        random_neurons = rng.choice(n_neurons, n_ablate, replace=False)
        pct = n_ablate / n_neurons * 100
        print(f"\n  Random ablation: {n_ablate:>6,} neurons ({pct:.2f}%)...")

        ablated = ablate_and_measure(
            model, tokenizer, stimuli, random_neurons,
            n_layers, ffn_dim, cat_indices, device,
        )
        random_results[n_ablate] = ablated

        for cat in categories:
            bl = baseline[cat]["avg_loss"]
            delta = ablated[cat]["avg_loss"] - bl
            print(f"    {cat:>12s}: delta={delta:>+.4f}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"DOSE-RESPONSE SUMMARY")
    print(f"{'='*70}")

    for func in target_functions:
        print(f"\n  Function: {func}")
        header = f"  {'N_ablate':>10s} | {'pct':>6s}"
        for cat in categories:
            header += f" | {cat:>8s}"
        header += " | rand_self"
        print(header)
        print(f"  {'-' * len(header)}")

        for n_ablate in ablation_sizes:
            if n_ablate not in dose_results[func]:
                continue
            pct = n_ablate / n_neurons * 100
            row = f"  {n_ablate:>10,} | {pct:>5.2f}%"
            for cat in categories:
                ratio = dose_results[func][n_ablate][cat]["perplexity"] / baseline[cat]["perplexity"]
                marker = "*" if cat == func else " "
                row += f" | {ratio:>7.2f}{marker}"
            # Random control ratio for the target function
            rand_ratio = random_results[n_ablate][func]["perplexity"] / baseline[func]["perplexity"]
            row += f" | {rand_ratio:>7.2f}"
            print(row)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ── Save results ─────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def serialize_ppl(d):
        return {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                     for kk, vv in v.items()} for k, v in d.items()}

    result = {
        "model": model_short,
        "n_neurons": n_neurons,
        "n_layers": n_layers,
        "ffn_dim": ffn_dim,
        "target_functions": list(target_functions),
        "ablation_sizes": list(ablation_sizes),
        "categories": categories,
        "baseline": serialize_ppl(baseline),
        "dose_response": {
            func: {
                str(n): serialize_ppl(dose_results[func][n])
                for n in ablation_sizes if n in dose_results[func]
            }
            for func in target_functions
        },
        "random_control": {
            str(n): serialize_ppl(random_results[n])
            for n in ablation_sizes if n in random_results
        },
        "elapsed_seconds": elapsed,
    }

    out_path = os.path.join(output_dir, f"{model_short}_dose_response.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dose-response curves for causal ablation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--model_short", type=str, required=True,
                        help="Short model name for output files")
    parser.add_argument("--meta_path", type=str, required=True,
                        help="Path to model meta JSON")
    parser.add_argument("--stimuli", type=str, required=True,
                        help="Path to stimuli JSONL")
    parser.add_argument("--attribution_path", type=str, required=True,
                        help="Path to multi_attribution.npz with pre-computed selectivity")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--functions", type=str, nargs="+", default=["math", "code"],
                        help="Target functions to build dose-response curves for")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[500, 1000, 2000, 5000, 10000, 20000],
                        help="Ablation sizes to test")
    args = parser.parse_args()

    run_dose_response(
        model_path=args.model_path,
        model_short=args.model_short,
        meta_path=args.meta_path,
        stimuli_path=args.stimuli,
        attribution_path=args.attribution_path,
        output_dir=args.output_dir,
        target_functions=tuple(args.functions),
        ablation_sizes=tuple(args.sizes),
    )
