"""
Multi-function dissociation: extend causal attribution to all 8 categories.

For each category X:
  1. Compute grad x act importance for X
  2. Selectivity_X = (imp_X - mean_imp_others) / (imp_X + mean_imp_others + eps)
  3. Ablate top-N most X-selective neurons
  4. Measure PPL across ALL categories

If ablating X-selective neurons damages X >> other categories for all X,
we have an N-way functional dissociation — a full functional atlas.
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_all_attributions(
    model, tokenizer, stimuli, categories,
    n_layers, ffn_dim, device="cuda", max_length=256,
):
    """Compute grad x act importance for every category."""
    n_neurons = n_layers * ffn_dim
    cat_importance = {cat: np.zeros(n_neurons, dtype=np.float64) for cat in categories}
    cat_counts = {cat: 0 for cat in categories}

    def get_layers(m):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers
        return None

    layers = get_layers(model)
    assert layers is not None

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

    total = sum(1 for s in stimuli if s["category"] in categories)
    print(f"  Computing attributions for {total} samples, {len(categories)} categories...")

    for idx, sample in enumerate(stimuli):
        cat = sample["category"]
        if cat not in categories:
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
            imp = (act.detach() * act.grad.detach()).abs().mean(dim=(0, 1))
            start = layer_idx * ffn_dim
            end = start + ffn_dim
            importance[start:end] = imp.cpu().float().numpy()

        cat_importance[cat] += importance.astype(np.float64)
        cat_counts[cat] += 1

    for h in hooks:
        h.remove()

    for cat in categories:
        if cat_counts[cat] > 0:
            cat_importance[cat] /= cat_counts[cat]

    return cat_importance, cat_counts


def compute_one_vs_rest_selectivity(cat_importance, categories):
    """For each category, selectivity = (imp_X - mean_others) / (imp_X + mean_others + eps)."""
    eps = 1e-10
    selectivity = {}
    for target in categories:
        imp_target = cat_importance[target]
        others = [cat_importance[c] for c in categories if c != target]
        imp_others = np.mean(others, axis=0)
        sel = (imp_target - imp_others) / (imp_target + imp_others + eps)
        selectivity[target] = sel
    return selectivity


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
        results[cat] = {"perplexity": float(ppl), "avg_loss": float(avg_loss), "n_samples": n_valid}
    return results


def ablate_and_measure(model, tokenizer, stimuli, neuron_indices,
                       n_layers, ffn_dim, cat_indices, device="cuda", max_length=256):
    """Ablate specific neurons and measure PPL."""
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


def run_multi_dissociation(
    model_path, model_short, meta_path, stimuli_path, output_dir,
    n_ablate=5000, device="cuda",
):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Multi-Function Dissociation: {model_short}")
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

    categories = sorted(cat_indices.keys())
    print(f"  Categories: {categories}")
    print(f"  Samples per category: {', '.join(f'{c}={len(v)}' for c, v in sorted(cat_indices.items()))}")

    # Load model
    print(f"\n[Step 1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Attribution for all categories
    print(f"\n[Step 2] Computing causal attribution for all {len(categories)} categories...")
    cat_importance, cat_counts = compute_all_attributions(
        model, tokenizer, stimuli, categories, n_layers, ffn_dim, device
    )
    for cat in categories:
        imp = cat_importance[cat]
        print(f"  {cat:>12s}: {cat_counts[cat]} samples, mean_imp={imp.mean():.6f}, max={imp.max():.6f}")

    # One-vs-rest selectivity
    print(f"\n[Step 3] Computing one-vs-rest selectivity...")
    selectivity = compute_one_vs_rest_selectivity(cat_importance, categories)
    for cat in categories:
        sel = selectivity[cat]
        n_high = (sel > 0.3).sum()
        print(f"  {cat:>12s}: selective(>0.3)={n_high:>6,} neurons, max_sel={sel.max():.4f}")

    # Save attribution
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_dict = {f"{cat}_importance": cat_importance[cat].astype(np.float32) for cat in categories}
    save_dict.update({f"{cat}_selectivity": selectivity[cat].astype(np.float32) for cat in categories})
    np.savez(os.path.join(output_dir, f"{model_short}_multi_attribution.npz"), **save_dict)

    # Baseline
    print(f"\n[Step 4] Baseline perplexity...")
    baseline = measure_ppl(model, tokenizer, stimuli, cat_indices, device)
    for cat in categories:
        print(f"  {cat:>12s}: PPL={baseline[cat]['perplexity']:>8.2f}, logPPL={baseline[cat]['avg_loss']:.3f}")

    # N-way ablation: for each category, ablate its top-N selective neurons
    print(f"\n[Step 5] N-way ablation (N={n_ablate:,}, {n_ablate/n_neurons*100:.2f}% of network)...")

    ablation_results = {}
    for target_cat in categories:
        sel = selectivity[target_cat]
        top_neurons = np.argsort(-sel)[:n_ablate]
        sel_range = f"[{sel[top_neurons[-1]]:.4f}, {sel[top_neurons[0]]:.4f}]"
        print(f"\n  Ablating {n_ablate} {target_cat.upper()}-selective neurons (sel range {sel_range})...")

        ablated = ablate_and_measure(
            model, tokenizer, stimuli, top_neurons,
            n_layers, ffn_dim, cat_indices, device,
        )
        ablation_results[target_cat] = ablated

        # Show damage profile
        damages = []
        for cat in categories:
            bl = baseline[cat]["avg_loss"]
            al = ablated[cat]["avg_loss"]
            delta = al - bl
            ratio = ablated[cat]["perplexity"] / baseline[cat]["perplexity"]
            damages.append((cat, delta, ratio))
            marker = " <<<" if cat == target_cat else ""
            print(f"    {cat:>12s}: logPPL delta={delta:>+.4f}, ratio={ratio:.2f}x{marker}")

    # Random control
    print(f"\n  Random ablation control ({n_ablate:,} neurons)...")
    rng = np.random.RandomState(42)
    random_neurons = rng.choice(n_neurons, n_ablate, replace=False)
    random_result = ablate_and_measure(
        model, tokenizer, stimuli, random_neurons,
        n_layers, ffn_dim, cat_indices, device,
    )
    for cat in categories:
        bl = baseline[cat]["avg_loss"]
        delta = random_result[cat]["avg_loss"] - bl
        print(f"    {cat:>12s}: logPPL delta={delta:>+.4f}")

    # Build dissociation matrix
    print(f"\n{'='*70}")
    print(f"DISSOCIATION MATRIX (PPL ratios)")
    print(f"{'='*70}")
    print(f"  Rows = which neurons ablated, Cols = which category measured")
    header = f"{'Ablated':>12s} |" + "".join(f" {c:>8s}" for c in categories) + " | random"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    dissoc_matrix = np.zeros((len(categories), len(categories)))
    for i, ablated_cat in enumerate(categories):
        row = f"  {ablated_cat:>12s} |"
        for j, measured_cat in enumerate(categories):
            ratio = ablation_results[ablated_cat][measured_cat]["perplexity"] / baseline[measured_cat]["perplexity"]
            dissoc_matrix[i, j] = ratio
            marker = "*" if i == j else " "
            row += f" {ratio:>7.2f}{marker}"
        rand_ratio = random_result[categories[i]]["perplexity"] / baseline[categories[i]]["perplexity"] if i < len(categories) else 0
        row += f" | {rand_ratio:.2f}"
        print(row)

    # Log-PPL version (more interpretable)
    print(f"\n{'='*70}")
    print(f"DISSOCIATION MATRIX (Log-PPL deltas)")
    print(f"{'='*70}")
    header2 = f"{'Ablated':>12s} |" + "".join(f" {c:>8s}" for c in categories)
    print(f"  {header2}")
    print(f"  {'-'*len(header2)}")

    logppl_matrix = np.zeros((len(categories), len(categories)))
    for i, ablated_cat in enumerate(categories):
        row = f"  {ablated_cat:>12s} |"
        for j, measured_cat in enumerate(categories):
            bl = baseline[measured_cat]["avg_loss"]
            al = ablation_results[ablated_cat][measured_cat]["avg_loss"]
            delta = al - bl
            logppl_matrix[i, j] = delta
            marker = "*" if i == j else " "
            row += f" {delta:>+7.4f}{marker}"
        print(row)

    # Analyze dissociation quality
    print(f"\n{'='*70}")
    print(f"DISSOCIATION ANALYSIS")
    print(f"{'='*70}")

    n_cats = len(categories)
    successes = 0
    total_pairs = 0

    for i, cat in enumerate(categories):
        on_target = logppl_matrix[i, i]
        off_target = np.delete(logppl_matrix[i], i)
        max_collateral = off_target.max()
        mean_collateral = off_target.mean()

        # Specificity: target damage > 1.5x max collateral
        specific = on_target > max_collateral * 1.5 and on_target > 0.05
        ratio_vs_max = on_target / (max_collateral + 1e-10)
        ratio_vs_mean = on_target / (mean_collateral + 1e-10)

        status = "PASS" if specific else "FAIL"
        print(f"  {cat:>12s}: target_delta={on_target:>+.4f}, max_collateral={max_collateral:>+.4f}, "
              f"ratio={ratio_vs_max:.1f}x, specificity: {status}")

        if specific:
            successes += 1

    # Pairwise dissociation check
    print(f"\n  Pairwise double dissociation:")
    pair_pass = 0
    pair_total = 0
    for i in range(n_cats):
        for j in range(i + 1, n_cats):
            a_on_a = logppl_matrix[i, i]
            a_on_b = logppl_matrix[i, j]
            b_on_b = logppl_matrix[j, j]
            b_on_a = logppl_matrix[j, i]
            dd = (a_on_a > a_on_b * 1.5 and a_on_a > 0.05) and (b_on_b > b_on_a * 1.5 and b_on_b > 0.05)
            pair_total += 1
            if dd:
                pair_pass += 1
            status = "YES" if dd else "no"
            if dd:
                print(f"    {categories[i]:>12s} vs {categories[j]:<12s}: {status}  "
                      f"({a_on_a:+.3f}/{a_on_b:+.3f}, {b_on_b:+.3f}/{b_on_a:+.3f})")

    print(f"\n  Summary: {successes}/{n_cats} categories show specificity")
    print(f"  Pairwise double dissociation: {pair_pass}/{pair_total} pairs pass")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    # Save results
    result = {
        "model": model_short,
        "n_neurons": n_neurons,
        "n_ablate": n_ablate,
        "pct_neurons": n_ablate / n_neurons * 100,
        "categories": categories,
        "baseline": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in r.items()} for c, r in baseline.items()},
        "ablation_results": {
            ablated: {measured: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                 for k, v in r.items()}
                      for measured, r in results_dict.items()}
            for ablated, results_dict in ablation_results.items()
        },
        "random_control": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                for k, v in r.items()} for c, r in random_result.items()},
        "dissociation_matrix_ppl_ratio": dissoc_matrix.tolist(),
        "dissociation_matrix_logppl_delta": logppl_matrix.tolist(),
        "n_specific_categories": successes,
        "n_pairwise_dissociations": pair_pass,
        "total_pairwise": pair_total,
        "elapsed_seconds": elapsed,
    }

    out_path = os.path.join(output_dir, f"{model_short}_multi_dissociation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_short", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--stimuli", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_ablate", type=int, default=5000)
    args = parser.parse_args()

    run_multi_dissociation(
        model_path=args.model_path,
        model_short=args.model_short,
        meta_path=args.meta_path,
        stimuli_path=args.stimuli,
        output_dir=args.output_dir,
        n_ablate=args.n_ablate,
    )
