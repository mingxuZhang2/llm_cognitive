"""
Causal attribution: identify neurons causally important for each task
using gradient × activation method, then do targeted ablation.

Instead of IterD's statistical co-activation clustering, this directly
measures each neuron's causal contribution to task-specific loss.

For each neuron i and sample j:
  importance(i, j) = |d_loss/d_act_i * act_i|  (mean across tokens)

Then:
  math_importance[i] = mean importance over math samples
  code_importance[i] = mean importance over code samples
  selectivity[i] = (math - code) / (math + code + eps)

Ablate top-N most selective neurons (not 15% of network, but 0.1-1%).
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_causal_attribution(
    model, tokenizer, stimuli, target_categories,
    n_layers, ffn_dim, device="cuda", max_length=256,
):
    """
    Compute gradient x activation importance for each neuron on each category.
    Returns dict: {category: importance_array[n_neurons]}, and sample counts.
    """
    n_neurons = n_layers * ffn_dim
    cat_importance = {cat: np.zeros(n_neurons, dtype=np.float64) for cat in target_categories}
    cat_counts = {cat: 0 for cat in target_categories}

    def get_layers(m):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        return None

    layers = get_layers(model)
    assert layers is not None, "Cannot find model layers"

    activations = {}
    hooks = []

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        mlp = layer.mlp
        if hasattr(mlp, "gate_proj"):
            target = mlp.gate_proj
        elif hasattr(mlp, "dense_h_to_4h"):
            target = mlp.dense_h_to_4h
        else:
            continue

        def make_hook(lidx):
            def fwd_hook(module, input, output):
                activations[lidx] = output
                output.retain_grad()
            return fwd_hook

        hooks.append(target.register_forward_hook(make_hook(layer_idx)))

    n_target = sum(1 for s in stimuli if s["category"] in target_categories)
    print(f"  Computing gradient x activation for {n_target} samples across {n_layers} layers...")

    for idx, sample in enumerate(stimuli):
        cat = sample["category"]
        if cat not in target_categories:
            continue

        text = sample["text"]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        if inputs["input_ids"].shape[1] < 2:
            continue

        model.zero_grad()
        activations.clear()

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()

        importance = np.zeros(n_neurons, dtype=np.float32)
        for layer_idx in range(n_layers):
            if layer_idx not in activations:
                continue
            act = activations[layer_idx]
            if act.grad is None:
                continue
            grad = act.grad
            # |grad * act| averaged across batch and sequence dimensions
            imp = (act.detach() * grad.detach()).abs().mean(dim=(0, 1))  # [ffn_dim]
            start = layer_idx * ffn_dim
            end = start + ffn_dim
            importance[start:end] = imp.cpu().float().numpy()

        cat_importance[cat] += importance.astype(np.float64)
        cat_counts[cat] += 1

        if (cat_counts[cat]) % 5 == 0:
            print(f"    {cat}: {cat_counts[cat]} samples processed")

    for h in hooks:
        h.remove()

    for cat in target_categories:
        if cat_counts[cat] > 0:
            cat_importance[cat] /= cat_counts[cat]

    return cat_importance, cat_counts


def compute_selectivity(imp_a, imp_b, eps=1e-10):
    """Selectivity index in [-1, 1]. Positive = more important for A."""
    return (imp_a - imp_b) / (imp_a + imp_b + eps)


def ablate_neurons_and_measure(
    model, tokenizer, stimuli, neuron_indices,
    n_layers, ffn_dim, device="cuda", max_length=256,
    categories=None,
):
    """Zero-ablate specific neurons and measure PPL by category."""
    if categories is None:
        cat_indices = {}
        for i, s in enumerate(stimuli):
            cat = s["category"]
            if cat not in cat_indices:
                cat_indices[cat] = []
            cat_indices[cat].append(i)
        categories = cat_indices
    else:
        cat_indices = categories

    def get_layers(m):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        return None

    layers = get_layers(model)
    hooks = []

    # Build per-layer masks
    neuron_set = set(neuron_indices.tolist() if isinstance(neuron_indices, np.ndarray) else neuron_indices)
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

    # Measure PPL
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
            avg_loss = float("inf")
            ppl = float("inf")
        results[cat] = {"perplexity": float(ppl), "avg_loss": float(avg_loss), "n_samples": n_valid}

    for h in hooks:
        h.remove()

    return results


def run_causal_dissociation(
    model_path, model_short, meta_path, stimuli_path, output_dir,
    ablation_sizes=None, device="cuda",
):
    """Full causal attribution + targeted ablation pipeline."""
    t0 = time.time()

    if ablation_sizes is None:
        ablation_sizes = [500, 1000, 2000, 5000]

    print(f"{'='*70}")
    print(f"Causal Attribution + Targeted Dissociation: {model_short}")
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

    # Load model
    print(f"\n[Step 1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Compute causal attribution
    print(f"\n[Step 2] Computing causal attribution (grad x act)...")
    target_cats = ["math", "code"]
    cat_importance, cat_counts = compute_causal_attribution(
        model, tokenizer, stimuli, target_cats, n_layers, ffn_dim, device
    )
    for cat in target_cats:
        imp = cat_importance[cat]
        print(f"  {cat}: {cat_counts[cat]} samples, importance range=[{imp.min():.6f}, {imp.max():.6f}], mean={imp.mean():.6f}")

    # Compute selectivity
    print(f"\n[Step 3] Computing selectivity scores...")
    selectivity = compute_selectivity(cat_importance["math"], cat_importance["code"])
    print(f"  Selectivity range: [{selectivity.min():.4f}, {selectivity.max():.4f}]")
    print(f"  Math-selective (>0.5): {(selectivity > 0.5).sum():,} neurons")
    print(f"  Code-selective (<-0.5): {(selectivity < -0.5).sum():,} neurons")
    print(f"  Neutral (-0.1 to 0.1): {((selectivity > -0.1) & (selectivity < 0.1)).sum():,} neurons")

    # Per-layer selectivity distribution
    print(f"\n  Per-layer selectivity (mean):")
    for l in range(n_layers):
        s = l * ffn_dim
        e = s + ffn_dim
        layer_sel = selectivity[s:e]
        math_count = (layer_sel > 0.3).sum()
        code_count = (layer_sel < -0.3).sum()
        bar_m = "#" * (math_count // 100)
        bar_c = "*" * (code_count // 100)
        print(f"    L{l:02d}: mean={layer_sel.mean():+.4f}, math(>0.3)={math_count:>5}, code(<-0.3)={code_count:>5} | {bar_m}{bar_c}")

    # Save attribution data
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f"{model_short}_causal_attribution.npz"),
        math_importance=cat_importance["math"].astype(np.float32),
        code_importance=cat_importance["code"].astype(np.float32),
        selectivity=selectivity.astype(np.float32),
    )

    # Baseline PPL
    print(f"\n[Step 4] Baseline perplexity...")
    measure_cats = {cat: indices[:50] for cat, indices in cat_indices.items() if len(indices) >= 5}
    baseline = ablate_neurons_and_measure(
        model, tokenizer, stimuli, np.array([], dtype=int),
        n_layers, ffn_dim, device, categories=measure_cats,
    )
    # Actually baseline = no ablation, so just compute normally
    baseline = {}
    model.eval()
    for cat, indices in measure_cats.items():
        total_loss = 0.0
        total_tokens = 0
        n_valid = 0
        for idx in indices:
            text = stimuli[idx]["text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
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
            avg_loss = float("inf")
            ppl = float("inf")
        baseline[cat] = {"perplexity": float(ppl), "avg_loss": float(avg_loss), "n_samples": n_valid}

    for cat in sorted(baseline):
        print(f"  {cat:>12s}: PPL={baseline[cat]['perplexity']:>8.2f}")

    # Targeted ablation at multiple sizes
    print(f"\n[Step 5] Targeted ablation experiments...")
    math_selective_order = np.argsort(-selectivity)  # most math-selective first
    code_selective_order = np.argsort(selectivity)   # most code-selective first

    all_results = {
        "model": model_short,
        "n_neurons": n_neurons,
        "n_layers": n_layers,
        "ffn_dim": ffn_dim,
        "baseline": baseline,
        "experiments": [],
    }

    for n_ablate in ablation_sizes:
        pct = n_ablate / n_neurons * 100
        print(f"\n  --- Ablating top {n_ablate:,} neurons ({pct:.2f}% of network) ---")

        # Ablate math-selective neurons
        math_neurons = math_selective_order[:n_ablate]
        print(f"  Ablating {n_ablate} most MATH-selective neurons...")
        print(f"    Selectivity range: [{selectivity[math_neurons[-1]]:.4f}, {selectivity[math_neurons[0]]:.4f}]")
        ablated_math = ablate_neurons_and_measure(
            model, tokenizer, stimuli, math_neurons,
            n_layers, ffn_dim, device, categories=measure_cats,
        )

        # Ablate code-selective neurons
        code_neurons = code_selective_order[:n_ablate]
        print(f"  Ablating {n_ablate} most CODE-selective neurons...")
        print(f"    Selectivity range: [{selectivity[code_neurons[0]]:.4f}, {selectivity[code_neurons[-1]]:.4f}]")
        ablated_code = ablate_neurons_and_measure(
            model, tokenizer, stimuli, code_neurons,
            n_layers, ffn_dim, device, categories=measure_cats,
        )

        # Random control
        rng = np.random.RandomState(42)
        random_neurons = rng.choice(n_neurons, n_ablate, replace=False)
        print(f"  Random ablation control ({n_ablate} neurons)...")
        ablated_random = ablate_neurons_and_measure(
            model, tokenizer, stimuli, random_neurons,
            n_layers, ffn_dim, device, categories=measure_cats,
        )

        # Report
        math_on_math = ablated_math["math"]["perplexity"] / baseline["math"]["perplexity"]
        math_on_code = ablated_math["code"]["perplexity"] / baseline["code"]["perplexity"]
        code_on_code = ablated_code["code"]["perplexity"] / baseline["code"]["perplexity"]
        code_on_math = ablated_code["math"]["perplexity"] / baseline["math"]["perplexity"]
        rand_on_math = ablated_random["math"]["perplexity"] / baseline["math"]["perplexity"]
        rand_on_code = ablated_random["code"]["perplexity"] / baseline["code"]["perplexity"]

        print(f"\n  PPL ratios (N={n_ablate:,}, {pct:.2f}%):")
        print(f"                      Math stim    Code stim")
        print(f"    Ablate Math-sel:  {math_on_math:>8.2f}x    {math_on_code:>8.2f}x")
        print(f"    Ablate Code-sel:  {code_on_math:>8.2f}x    {code_on_code:>8.2f}x")
        print(f"    Random:           {rand_on_math:>8.2f}x    {rand_on_code:>8.2f}x")

        dissociation = (math_on_math > math_on_code * 1.5) and (code_on_code > code_on_math * 1.5)
        specificity = (math_on_math > rand_on_math * 1.2) or (code_on_code > rand_on_code * 1.2)
        print(f"    Double dissociation: {'YES !!!' if dissociation else 'NO'}")
        print(f"    Specificity vs random: {'YES' if specificity else 'NO'}")

        # Log-PPL deltas for all categories
        print(f"\n    Log-PPL deltas (all categories):")
        print(f"    {'Category':>12s}  {'d(math-sel)':>11s}  {'d(code-sel)':>11s}  {'d(random)':>9s}")
        for cat in sorted(baseline):
            bl = baseline[cat]["avg_loss"]
            dm = ablated_math[cat]["avg_loss"] - bl
            dc = ablated_code[cat]["avg_loss"] - bl
            dr = ablated_random[cat]["avg_loss"] - bl
            marker = ""
            if cat == "math":
                marker = " <-"
            elif cat == "code":
                marker = " <-"
            print(f"    {cat:>12s}  {dm:>+11.4f}  {dc:>+11.4f}  {dr:>+9.4f}{marker}")

        exp_result = {
            "n_ablate": n_ablate,
            "pct_neurons": pct,
            "ablated_math": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for c, r in ablated_math.items()},
            "ablated_code": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for c, r in ablated_code.items()},
            "ablated_random": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for c, r in ablated_random.items()},
            "ratios": {
                "math_on_math": float(math_on_math),
                "math_on_code": float(math_on_code),
                "code_on_code": float(code_on_code),
                "code_on_math": float(code_on_math),
                "rand_on_math": float(rand_on_math),
                "rand_on_code": float(rand_on_code),
            },
            "double_dissociation": dissociation,
            "specificity": specificity,
        }
        all_results["experiments"].append(exp_result)

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed
    print(f"\n{'='*70}")
    print(f"Completed in {elapsed:.0f}s")

    out_path = os.path.join(output_dir, f"{model_short}_causal_dissociation.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_short", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--stimuli", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ablation_sizes", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    args = parser.parse_args()

    run_causal_dissociation(
        model_path=args.model_path,
        model_short=args.model_short,
        meta_path=args.meta_path,
        stimuli_path=args.stimuli,
        output_dir=args.output_dir,
        ablation_sizes=args.ablation_sizes,
    )
