"""
Method Triangulation: Compare 3 attribution methods to show functional atlas
findings are robust to methodological choice.

Methods:
  1. Gradient x Activation (existing method): |dL/d_act * act|
  2. Activation-Only (no gradient):           |act|
  3. Gradient-Only (no activation):           |dL/d_act|

For each method:
  - Compute per-category importance on medium stimuli (50/cat)
  - Compute one-vs-rest selectivity
  - Select top-5000 neurons per category
  - Ablate top-5000 selective neurons per category → 8x8 dissociation matrix

Cross-method comparisons:
  - Top-k Jaccard overlap per category between each pair of methods
  - Pearson correlation between dissociation matrices
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer


METHODS = ["grad_x_act", "act_only", "grad_only"]


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    return None


def get_mlp_target(mlp):
    """Get the gate/first linear projection of the MLP."""
    if hasattr(mlp, "gate_proj"):
        return mlp.gate_proj
    elif hasattr(mlp, "dense_h_to_4h"):
        return mlp.dense_h_to_4h
    return None


def compute_attributions(
    model, tokenizer, stimuli, categories,
    n_layers, ffn_dim, method, device="cuda", max_length=256,
):
    """
    Compute neuron importance for a given method.

    method: one of "grad_x_act", "act_only", "grad_only"
    """
    n_neurons = n_layers * ffn_dim
    cat_importance = {cat: np.zeros(n_neurons, dtype=np.float64) for cat in categories}
    cat_counts = {cat: 0 for cat in categories}

    layers = get_layers(model)
    assert layers is not None, "Could not find model layers"

    need_grad = method in ("grad_x_act", "grad_only")

    activations = {}
    hooks = []

    for layer_idx in range(n_layers):
        layer = layers[layer_idx]
        target = get_mlp_target(layer.mlp)
        if target is None:
            continue

        def make_hook(lidx):
            def fwd_hook(module, input, output):
                activations[lidx] = output
                if need_grad:
                    output.retain_grad()
            return fwd_hook

        hooks.append(target.register_forward_hook(make_hook(layer_idx)))

    total = sum(1 for s in stimuli if s["category"] in categories)
    print(f"    [{method}] Computing attributions for {total} samples...")

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

        if need_grad:
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
        else:
            # act_only: no backward needed
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])

        importance = np.zeros(n_neurons, dtype=np.float32)
        for layer_idx in range(n_layers):
            if layer_idx not in activations:
                continue
            act = activations[layer_idx]
            start = layer_idx * ffn_dim
            end = start + ffn_dim

            if method == "grad_x_act":
                if act.grad is None:
                    continue
                imp = (act.detach() * act.grad.detach()).abs().mean(dim=(0, 1))
            elif method == "act_only":
                imp = act.detach().abs().mean(dim=(0, 1))
            elif method == "grad_only":
                if act.grad is None:
                    continue
                imp = act.grad.detach().abs().mean(dim=(0, 1))
            else:
                raise ValueError(f"Unknown method: {method}")

            importance[start:end] = imp.cpu().float().numpy()

        cat_importance[cat] += importance.astype(np.float64)
        cat_counts[cat] += 1

        if (idx + 1) % 50 == 0:
            print(f"      Processed {idx + 1} samples...")

    for h in hooks:
        h.remove()

    for cat in categories:
        if cat_counts[cat] > 0:
            cat_importance[cat] /= cat_counts[cat]

    return cat_importance, cat_counts


def compute_one_vs_rest_selectivity(cat_importance, categories):
    """selectivity_X = (imp_X - mean_others) / (imp_X + mean_others + eps)"""
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
    """Ablate specific neurons and measure PPL across all categories."""
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
        target = get_mlp_target(layer.mlp)
        if target is None:
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


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def run_method_triangulation(
    model_path, model_short, meta_path, stimuli_path, output_dir,
    n_ablate=5000, device="cuda",
):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Method Triangulation: {model_short}")
    print(f"{'='*70}")

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim

    # Load stimuli
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
    print(f"  Total neurons: {n_neurons:,} ({n_layers} layers x {ffn_dim} ffn_dim)")
    print(f"  Top-k for selection: {n_ablate:,} ({n_ablate/n_neurons*100:.2f}%)")

    # Load model
    print(f"\n[Step 1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 2: Compute attributions for all 3 methods
    # =========================================================================
    all_importance = {}   # method -> {cat: importance_array}
    all_selectivity = {}  # method -> {cat: selectivity_array}
    all_top_neurons = {}  # method -> {cat: set of neuron indices}

    for method in METHODS:
        print(f"\n[Step 2/{method}] Computing attributions using {method}...")
        cat_imp, cat_counts = compute_attributions(
            model, tokenizer, stimuli, categories,
            n_layers, ffn_dim, method, device,
        )
        for cat in categories:
            print(f"    {cat:>12s}: {cat_counts[cat]} samples, "
                  f"mean_imp={cat_imp[cat].mean():.6f}, max={cat_imp[cat].max():.6f}")

        selectivity = compute_one_vs_rest_selectivity(cat_imp, categories)
        for cat in categories:
            n_high = (selectivity[cat] > 0.3).sum()
            print(f"    {cat:>12s}: selective(>0.3)={n_high:>6,} neurons, "
                  f"max_sel={selectivity[cat].max():.4f}")

        # Select top-k neurons per category
        top_neurons = {}
        for cat in categories:
            top_neurons[cat] = set(np.argsort(-selectivity[cat])[:n_ablate].tolist())

        all_importance[method] = cat_imp
        all_selectivity[method] = selectivity
        all_top_neurons[method] = top_neurons

        # Save attribution arrays
        save_dict = {}
        for cat in categories:
            save_dict[f"{cat}_importance"] = cat_imp[cat].astype(np.float32)
            save_dict[f"{cat}_selectivity"] = selectivity[cat].astype(np.float32)
        np.savez(
            os.path.join(output_dir, f"{model_short}_{method}_attribution.npz"),
            **save_dict,
        )

    # =========================================================================
    # Step 3: Cross-method top-k overlap (Jaccard)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"[Step 3] Cross-Method Top-{n_ablate} Neuron Overlap (Jaccard)")
    print(f"{'='*70}")

    method_pairs = [
        ("grad_x_act", "act_only"),
        ("grad_x_act", "grad_only"),
        ("act_only", "grad_only"),
    ]

    overlap_results = {}
    for m1, m2 in method_pairs:
        pair_key = f"{m1}_vs_{m2}"
        overlap_results[pair_key] = {}
        print(f"\n  {m1} vs {m2}:")
        jaccards = []
        for cat in categories:
            j = jaccard(all_top_neurons[m1][cat], all_top_neurons[m2][cat])
            overlap_results[pair_key][cat] = j
            jaccards.append(j)
            print(f"    {cat:>12s}: Jaccard={j:.4f}")
        mean_j = np.mean(jaccards)
        overlap_results[pair_key]["mean"] = float(mean_j)
        print(f"    {'MEAN':>12s}: Jaccard={mean_j:.4f}")

    # =========================================================================
    # Step 4: Baseline PPL
    # =========================================================================
    print(f"\n[Step 4] Baseline perplexity...")
    baseline = measure_ppl(model, tokenizer, stimuli, cat_indices, device)
    for cat in categories:
        print(f"  {cat:>12s}: PPL={baseline[cat]['perplexity']:>8.2f}")

    # =========================================================================
    # Step 5: Ablation for each method → dissociation matrix
    # =========================================================================
    all_dissoc_matrices = {}   # method -> 8x8 numpy array (log-PPL delta)
    all_ppl_matrices = {}      # method -> 8x8 numpy array (PPL ratio)
    all_ablation_results = {}  # method -> {ablated_cat: {measured_cat: PPL results}}
    all_specificity = {}       # method -> {cat: bool}
    all_pairwise = {}          # method -> (pass, total)

    for method in METHODS:
        print(f"\n{'='*70}")
        print(f"[Step 5/{method}] Ablation with {method}-selected neurons")
        print(f"{'='*70}")

        ablation_results = {}
        for target_cat in categories:
            top_neurons = np.array(sorted(all_top_neurons[method][target_cat]))
            sel = all_selectivity[method][target_cat]
            sel_vals = sel[top_neurons]
            print(f"\n  Ablating {n_ablate} {target_cat.upper()}-selective neurons "
                  f"(sel range [{sel_vals.min():.4f}, {sel_vals.max():.4f}])...")

            ablated = ablate_and_measure(
                model, tokenizer, stimuli, top_neurons,
                n_layers, ffn_dim, cat_indices, device,
            )
            ablation_results[target_cat] = ablated

            for cat in categories:
                bl = baseline[cat]["avg_loss"]
                al = ablated[cat]["avg_loss"]
                delta = al - bl
                ratio = ablated[cat]["perplexity"] / baseline[cat]["perplexity"]
                marker = " <<<" if cat == target_cat else ""
                print(f"    {cat:>12s}: logPPL delta={delta:>+.4f}, ratio={ratio:.2f}x{marker}")

        # Build dissociation matrices
        logppl_matrix = np.zeros((len(categories), len(categories)))
        ppl_matrix = np.zeros((len(categories), len(categories)))
        for i, ablated_cat in enumerate(categories):
            for j, measured_cat in enumerate(categories):
                bl = baseline[measured_cat]["avg_loss"]
                al = ablation_results[ablated_cat][measured_cat]["avg_loss"]
                logppl_matrix[i, j] = al - bl
                ppl_matrix[i, j] = (
                    ablation_results[ablated_cat][measured_cat]["perplexity"]
                    / baseline[measured_cat]["perplexity"]
                )

        all_dissoc_matrices[method] = logppl_matrix
        all_ppl_matrices[method] = ppl_matrix
        all_ablation_results[method] = ablation_results

        # Print dissociation matrix
        print(f"\n  Dissociation Matrix (PPL ratio) for {method}:")
        header = f"  {'Ablated':>12s} |" + "".join(f" {c:>8s}" for c in categories)
        print(header)
        print(f"  {'-'*len(header)}")
        for i, ablated_cat in enumerate(categories):
            row = f"  {ablated_cat:>12s} |"
            for j, measured_cat in enumerate(categories):
                marker = "*" if i == j else " "
                row += f" {ppl_matrix[i, j]:>7.2f}{marker}"
            print(row)

        # Specificity and pairwise analysis
        successes = 0
        for i, cat in enumerate(categories):
            on_target = logppl_matrix[i, i]
            off_target = np.delete(logppl_matrix[i], i)
            max_collateral = off_target.max()
            specific = on_target > max_collateral * 1.5 and on_target > 0.05
            if specific:
                successes += 1

        pair_pass = 0
        pair_total = 0
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                a_on_a = logppl_matrix[i, i]
                a_on_b = logppl_matrix[i, j]
                b_on_b = logppl_matrix[j, j]
                b_on_a = logppl_matrix[j, i]
                dd = (a_on_a > a_on_b * 1.5 and a_on_a > 0.05) and \
                     (b_on_b > b_on_a * 1.5 and b_on_b > 0.05)
                pair_total += 1
                if dd:
                    pair_pass += 1

        all_specificity[method] = successes
        all_pairwise[method] = (pair_pass, pair_total)
        print(f"\n  {method}: {successes}/{len(categories)} specific, "
              f"{pair_pass}/{pair_total} pairwise dissociations")

    # =========================================================================
    # Step 6: Cross-method dissociation matrix correlation
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"[Step 6] Cross-Method Dissociation Matrix Correlation")
    print(f"{'='*70}")

    matrix_corr = {}
    for m1, m2 in method_pairs:
        pair_key = f"{m1}_vs_{m2}"
        mat1 = all_dissoc_matrices[m1].flatten()
        mat2 = all_dissoc_matrices[m2].flatten()
        r, p = stats.pearsonr(mat1, mat2)
        rho, rho_p = stats.spearmanr(mat1, mat2)
        matrix_corr[pair_key] = {
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
        }
        print(f"  {m1} vs {m2}: Pearson r={r:.4f} (p={p:.2e}), Spearman rho={rho:.4f} (p={rho_p:.2e})")

    # Also correlate the PPL-ratio matrices
    ppl_matrix_corr = {}
    for m1, m2 in method_pairs:
        pair_key = f"{m1}_vs_{m2}"
        mat1 = all_ppl_matrices[m1].flatten()
        mat2 = all_ppl_matrices[m2].flatten()
        r, p = stats.pearsonr(mat1, mat2)
        ppl_matrix_corr[pair_key] = {"pearson_r": float(r), "pearson_p": float(p)}

    # =========================================================================
    # Step 7: Summary
    # =========================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for method in METHODS:
        pp, pt = all_pairwise[method]
        print(f"  {method:>15s}: {all_specificity[method]}/8 specific, {pp}/{pt} pairwise")
    print(f"\n  Neuron overlap (Jaccard, mean across categories):")
    for m1, m2 in method_pairs:
        pair_key = f"{m1}_vs_{m2}"
        print(f"    {m1} vs {m2}: {overlap_results[pair_key]['mean']:.4f}")
    print(f"\n  Dissociation matrix correlation (Pearson r):")
    for m1, m2 in method_pairs:
        pair_key = f"{m1}_vs_{m2}"
        print(f"    {m1} vs {m2}: r={matrix_corr[pair_key]['pearson_r']:.4f}")
    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # =========================================================================
    # Save results
    # =========================================================================
    result = {
        "model": model_short,
        "n_neurons": n_neurons,
        "n_layers": n_layers,
        "ffn_dim": ffn_dim,
        "n_ablate": n_ablate,
        "pct_neurons": n_ablate / n_neurons * 100,
        "categories": categories,
        "methods": METHODS,
        "baseline": {c: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in r.items()} for c, r in baseline.items()},
        "per_method": {},
        "overlap": overlap_results,
        "matrix_correlation_logppl": matrix_corr,
        "matrix_correlation_ppl_ratio": ppl_matrix_corr,
        "elapsed_seconds": elapsed,
    }

    for method in METHODS:
        pp, pt = all_pairwise[method]
        method_data = {
            "n_specific": all_specificity[method],
            "n_pairwise_pass": pp,
            "n_pairwise_total": pt,
            "dissociation_matrix_logppl_delta": all_dissoc_matrices[method].tolist(),
            "dissociation_matrix_ppl_ratio": all_ppl_matrices[method].tolist(),
            "ablation_results": {
                ablated: {measured: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                     for k, v in r.items()}
                          for measured, r in results_dict.items()}
                for ablated, results_dict in all_ablation_results[method].items()
            },
            "selectivity_summary": {
                cat: {
                    "n_selective_gt03": int((all_selectivity[method][cat] > 0.3).sum()),
                    "max_selectivity": float(all_selectivity[method][cat].max()),
                    "mean_importance": float(all_importance[method][cat].mean()),
                }
                for cat in categories
            },
        }
        result["per_method"][method] = method_data

    out_path = os.path.join(output_dir, f"{model_short}_method_triangulation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Method Triangulation for Functional Atlas")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_short", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--stimuli", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_ablate", type=int, default=5000)
    args = parser.parse_args()

    run_method_triangulation(
        model_path=args.model_path,
        model_short=args.model_short,
        meta_path=args.meta_path,
        stimuli_path=args.stimuli,
        output_dir=args.output_dir,
        n_ablate=args.n_ablate,
    )
