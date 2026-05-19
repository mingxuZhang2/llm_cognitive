"""
Predictive experiments: demonstrate the functional atlas has causal predictive power.

Experiment 1: Pathway decomposition
  - Separate math-only, reasoning-only, and shared (math∩reasoning) neurons
  - Ablate each set → predict differential effects

Experiment 2: Atlas-guided steering
  - Amplify specific functional pathways during inference
  - Predict: boosting math neurons improves reasoning (dependency)
  - Predict: boosting code neurons does NOT improve reasoning (no dependency)

Experiment 3: Instance-level vulnerability prediction
  - For each sample, predict which ablation will hurt it most based on
    its functional activation profile
"""

import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_ppl_per_sample(model, tokenizer, stimuli, indices, device="cuda", max_length=256):
    """Measure per-sample perplexity."""
    model.eval()
    results = []
    for idx in indices:
        text = stimuli[idx]["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        if inputs["input_ids"].shape[1] < 2:
            results.append({"idx": idx, "loss": float("inf"), "ppl": float("inf")})
            continue
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        ppl = np.exp(min(loss, 100))
        results.append({"idx": idx, "loss": float(loss), "ppl": float(ppl)})
    return results


def measure_ppl_by_category(model, tokenizer, stimuli, cat_indices, device="cuda", max_length=256):
    """Category-level PPL."""
    model.eval()
    results = {}
    for cat, indices in cat_indices.items():
        total_loss = 0.0
        total_tokens = 0
        for idx in indices:
            text = stimuli[idx]["text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            nt = inputs["input_ids"].shape[1] - 1
            total_loss += out.loss.item() * nt
            total_tokens += nt
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            results[cat] = {"avg_loss": float(avg_loss), "ppl": float(np.exp(min(avg_loss, 100)))}
        else:
            results[cat] = {"avg_loss": float("inf"), "ppl": float("inf")}
    return results


def apply_hooks(model, neuron_indices, n_layers, ffn_dim, device, mode="ablate", scale=1.5):
    """
    Apply forward hooks to modify neuron activations.
    mode='ablate': zero out neurons
    mode='amplify': multiply activations by scale
    mode='suppress': multiply activations by 1/scale
    """
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
        mlp = layers[layer_idx].mlp
        if hasattr(mlp, "gate_proj"):
            target = mlp.gate_proj
        elif hasattr(mlp, "dense_h_to_4h"):
            target = mlp.dense_h_to_4h
        else:
            continue

        mask = torch.zeros(ffn_dim, dtype=torch.bool, device=device)
        mask[layer_neurons] = True

        if mode == "ablate":
            def make_hook(m):
                def hook_fn(module, input, output):
                    output[:, :, m] = 0.0
                    return output
                return hook_fn
        elif mode == "amplify":
            def make_hook(m, s=scale):
                def hook_fn(module, input, output):
                    output[:, :, m] = output[:, :, m] * s
                    return output
                return hook_fn
        elif mode == "suppress":
            def make_hook(m, s=scale):
                def hook_fn(module, input, output):
                    output[:, :, m] = output[:, :, m] / s
                    return output
                return hook_fn

        hooks.append(target.register_forward_hook(make_hook(mask)))

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def experiment_1_pathway_decomposition(
    model, tokenizer, stimuli, cat_indices, attr_data,
    n_layers, ffn_dim, device, n_neurons_per_set=2000,
):
    """
    Decompose neurons into: math-only, reasoning-only, shared.
    Ablate each → predict differential effects.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: Pathway Decomposition (math-only / reasoning-only / shared)")
    print(f"{'='*70}")

    math_imp = attr_data["math_importance"]
    reasoning_imp = attr_data["reasoning_importance"]

    # Selectivity for math vs reasoning
    eps = 1e-10
    math_vs_reasoning = (math_imp - reasoning_imp) / (math_imp + reasoning_imp + eps)

    # Also need overall importance for both
    combined_imp = math_imp + reasoning_imp

    # Math-only: high math importance, low reasoning importance (selectivity >> 0)
    math_only_order = np.argsort(-math_vs_reasoning)  # most math-selective first
    # Filter: only neurons where math_imp > some threshold
    math_only_candidates = math_only_order[math_vs_reasoning[math_only_order] > 0.3]
    math_only = math_only_candidates[:n_neurons_per_set]

    # Reasoning-only: high reasoning importance, low math importance (selectivity << 0)
    reasoning_only_order = np.argsort(math_vs_reasoning)
    reasoning_only_candidates = reasoning_only_order[math_vs_reasoning[reasoning_only_order] < -0.3]
    reasoning_only = reasoning_only_candidates[:n_neurons_per_set]

    # Shared: high importance for BOTH (take neurons with high min(math, reasoning))
    min_imp = np.minimum(math_imp, reasoning_imp)
    shared_order = np.argsort(-min_imp)
    # Filter: both math and reasoning importance > median
    shared = shared_order[:n_neurons_per_set]

    print(f"  Math-only: {len(math_only)} neurons (selectivity > 0.3)")
    print(f"  Reasoning-only: {len(reasoning_only)} neurons (selectivity < -0.3)")
    print(f"  Shared: {len(shared)} neurons (high min(math_imp, reasoning_imp))")

    # Baseline
    baseline = measure_ppl_by_category(model, tokenizer, stimuli, cat_indices, device)

    # Ablation experiments
    results = {"baseline": baseline}

    for name, neurons in [("math_only", math_only), ("reasoning_only", reasoning_only), ("shared", shared)]:
        hooks = apply_hooks(model, neurons, n_layers, ffn_dim, device, mode="ablate")
        ablated = measure_ppl_by_category(model, tokenizer, stimuli, cat_indices, device)
        remove_hooks(hooks)

        results[name] = ablated

        print(f"\n  Ablate {name} ({len(neurons)} neurons):")
        for cat in ["math", "reasoning", "code", "language", "science"]:
            if cat in ablated and cat in baseline:
                delta = ablated[cat]["avg_loss"] - baseline[cat]["avg_loss"]
                ratio = ablated[cat]["ppl"] / baseline[cat]["ppl"]
                marker = " <<<" if (name == "math_only" and cat == "math") or \
                                    (name == "reasoning_only" and cat == "reasoning") or \
                                    (name == "shared" and cat in ["math", "reasoning"]) else ""
                print(f"    {cat:>12s}: delta_logPPL={delta:>+.4f}, ratio={ratio:.2f}x{marker}")

    # Check predictions
    print(f"\n  PREDICTION CHECK:")
    m_math = results["math_only"]["math"]["avg_loss"] - baseline["math"]["avg_loss"]
    m_reas = results["math_only"]["reasoning"]["avg_loss"] - baseline["reasoning"]["avg_loss"]
    r_math = results["reasoning_only"]["math"]["avg_loss"] - baseline["math"]["avg_loss"]
    r_reas = results["reasoning_only"]["reasoning"]["avg_loss"] - baseline["reasoning"]["avg_loss"]
    s_math = results["shared"]["math"]["avg_loss"] - baseline["math"]["avg_loss"]
    s_reas = results["shared"]["reasoning"]["avg_loss"] - baseline["reasoning"]["avg_loss"]

    print(f"    P1: math-only ablation hurts math > reasoning? "
          f"({m_math:+.4f} vs {m_reas:+.4f}) → {'YES' if m_math > m_reas else 'NO'}")
    print(f"    P2: reasoning-only ablation hurts reasoning > math? "
          f"({r_reas:+.4f} vs {r_math:+.4f}) → {'YES' if r_reas > r_math else 'NO'}")
    print(f"    P3: shared ablation hurts BOTH math AND reasoning? "
          f"(math={s_math:+.4f}, reas={s_reas:+.4f}) → {'YES' if s_math > 0.01 and s_reas > 0.01 else 'NO'}")

    return results


def experiment_2_steering(
    model, tokenizer, stimuli, cat_indices, attr_data,
    n_layers, ffn_dim, device, n_neurons=5000, scales=None,
):
    """
    Atlas-guided steering: amplify specific neurons during inference.

    Key prediction from spillover matrix:
    - reasoning depends on math → amplifying math neurons should help reasoning
    - reasoning does NOT depend on code → amplifying code neurons should NOT help reasoning
    """
    if scales is None:
        scales = [1.2, 1.5, 2.0, 3.0]

    print(f"\n{'='*70}")
    print("EXPERIMENT 2: Atlas-Guided Steering")
    print(f"{'='*70}")
    print("  Prediction: amplifying math neurons improves reasoning (dependency)")
    print("  Prediction: amplifying code neurons does NOT improve reasoning (no dependency)")

    # Get top-N neurons for each function
    categories_to_steer = ["math", "code", "language", "science"]
    top_neurons = {}
    for cat in categories_to_steer:
        sel = attr_data[f"{cat}_selectivity"]
        top_neurons[cat] = np.argsort(-sel)[:n_neurons]

    baseline = measure_ppl_by_category(model, tokenizer, stimuli, cat_indices, device)

    print(f"\n  Baseline:")
    for cat in sorted(baseline):
        print(f"    {cat:>12s}: logPPL={baseline[cat]['avg_loss']:.4f}, PPL={baseline[cat]['ppl']:.2f}")

    all_results = {"baseline": baseline, "steering_results": {}}

    for steer_cat in categories_to_steer:
        steer_results = {}
        for scale in scales:
            hooks = apply_hooks(model, top_neurons[steer_cat], n_layers, ffn_dim, device,
                              mode="amplify", scale=scale)
            steered = measure_ppl_by_category(model, tokenizer, stimuli, cat_indices, device)
            remove_hooks(hooks)
            steer_results[scale] = steered

        all_results["steering_results"][steer_cat] = steer_results

        print(f"\n  Amplify {steer_cat.upper()} neurons (x{scales}):")
        header = f"    {'Measured':>12s}"
        for s in scales:
            header += f"  x{s:<6}"
        print(header)

        for measured_cat in ["math", "code", "reasoning", "language", "science"]:
            if measured_cat not in baseline:
                continue
            row = f"    {measured_cat:>12s}"
            for s in scales:
                delta = steer_results[s][measured_cat]["avg_loss"] - baseline[measured_cat]["avg_loss"]
                row += f"  {delta:>+.4f}"
            # Mark if this is an interesting prediction
            if steer_cat == "math" and measured_cat == "reasoning":
                row += " ← PREDICT: helps"
            elif steer_cat == "code" and measured_cat == "reasoning":
                row += " ← PREDICT: no effect"
            print(row)

    # Summarize predictions
    print(f"\n  PREDICTION SUMMARY (effect on reasoning):")
    for steer_cat in categories_to_steer:
        best_scale = scales[1]  # moderate amplification
        delta = all_results["steering_results"][steer_cat][best_scale]["reasoning"]["avg_loss"] - baseline["reasoning"]["avg_loss"]
        direction = "HELPS (↓ loss)" if delta < -0.01 else "HURTS (↑ loss)" if delta > 0.01 else "NO EFFECT"
        expected = "helps" if steer_cat in ["math", "science", "language"] else "no effect"
        match = "✓" if (delta < -0.01 and expected == "helps") or (abs(delta) < 0.01 and expected == "no effect") else "?"
        print(f"    Amplify {steer_cat:>8s} → reasoning: {delta:>+.4f} ({direction}) [expected: {expected}] {match}")

    return all_results


def experiment_3_instance_prediction(
    model, tokenizer, stimuli, cat_indices, attr_data,
    n_layers, ffn_dim, device, n_neurons=5000,
):
    """
    Predict per-sample vulnerability from its functional profile.

    For each reasoning sample:
    1. Compute its activation profile on math vs code neurons
    2. Predict: high-math-activation samples are more vulnerable to math ablation
    3. Verify by measuring per-sample PPL change under ablation
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: Instance-Level Vulnerability Prediction")
    print(f"{'='*70}")

    math_sel = attr_data["math_selectivity"]
    code_sel = attr_data["code_selectivity"]
    math_neurons = np.argsort(-math_sel)[:n_neurons]
    code_neurons = np.argsort(-code_sel)[:n_neurons]

    # Get all sample indices
    all_indices = []
    for cat, indices in cat_indices.items():
        all_indices.extend(indices)

    # Baseline per-sample PPL
    baseline_samples = measure_ppl_per_sample(model, tokenizer, stimuli, all_indices, device)
    baseline_map = {r["idx"]: r["loss"] for r in baseline_samples}

    # Ablate math → per-sample PPL
    hooks = apply_hooks(model, math_neurons, n_layers, ffn_dim, device, mode="ablate")
    math_ablated_samples = measure_ppl_per_sample(model, tokenizer, stimuli, all_indices, device)
    remove_hooks(hooks)
    math_ablated_map = {r["idx"]: r["loss"] for r in math_ablated_samples}

    # Ablate code → per-sample PPL
    hooks = apply_hooks(model, code_neurons, n_layers, ffn_dim, device, mode="ablate")
    code_ablated_samples = measure_ppl_per_sample(model, tokenizer, stimuli, all_indices, device)
    remove_hooks(hooks)
    code_ablated_map = {r["idx"]: r["loss"] for r in code_ablated_samples}

    # Compute per-sample vulnerability
    print(f"\n  Per-category average vulnerability:")
    print(f"  {'Category':>12s}  {'math_abl_delta':>14s}  {'code_abl_delta':>14s}  {'more_vulnerable_to':>20s}")

    cat_vulnerabilities = {}
    for cat, indices in sorted(cat_indices.items()):
        math_deltas = [math_ablated_map[i] - baseline_map[i] for i in indices if i in baseline_map and baseline_map[i] < 50]
        code_deltas = [code_ablated_map[i] - baseline_map[i] for i in indices if i in baseline_map and baseline_map[i] < 50]

        avg_math_d = np.mean(math_deltas) if math_deltas else 0
        avg_code_d = np.mean(code_deltas) if code_deltas else 0

        more_vuln = "math ablation" if avg_math_d > avg_code_d else "code ablation"
        print(f"  {cat:>12s}  {avg_math_d:>+14.4f}  {avg_code_d:>+14.4f}  {more_vuln:>20s}")

        cat_vulnerabilities[cat] = {
            "math_ablation_delta": float(avg_math_d),
            "code_ablation_delta": float(avg_code_d),
        }

    # Key predictions
    print(f"\n  PREDICTIONS:")
    math_v_math = cat_vulnerabilities.get("math", {}).get("math_ablation_delta", 0)
    math_v_code = cat_vulnerabilities.get("math", {}).get("code_ablation_delta", 0)
    code_v_math = cat_vulnerabilities.get("code", {}).get("math_ablation_delta", 0)
    code_v_code = cat_vulnerabilities.get("code", {}).get("code_ablation_delta", 0)

    print(f"    P1: math samples more vulnerable to math ablation than code ablation? "
          f"({math_v_math:+.4f} vs {math_v_code:+.4f}) → {'YES' if math_v_math > math_v_code else 'NO'}")
    print(f"    P2: code samples more vulnerable to code ablation than math ablation? "
          f"({code_v_code:+.4f} vs {code_v_math:+.4f}) → {'YES' if code_v_code > code_v_math else 'NO'}")

    # Reasoning vulnerability: should be more vulnerable to math than code (from spillover)
    reas_v_math = cat_vulnerabilities.get("reasoning", {}).get("math_ablation_delta", 0)
    reas_v_code = cat_vulnerabilities.get("reasoning", {}).get("code_ablation_delta", 0)
    print(f"    P3: reasoning more vulnerable to math ablation than code ablation? "
          f"({reas_v_math:+.4f} vs {reas_v_code:+.4f}) → {'YES' if reas_v_math > reas_v_code else 'NO'}")
    print(f"         (predicted by spillover: math→reasoning=0.10, code→reasoning≈0)")

    return cat_vulnerabilities


def run_all_predictive_experiments(
    model_path, model_short, meta_path, attribution_path,
    stimuli_path, output_dir, device="cuda",
):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Predictive Experiments: {model_short}")
    print(f"{'='*70}")

    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]

    with open(stimuli_path) as f:
        stimuli = [json.loads(line) for line in f]

    cat_indices = {}
    for i, s in enumerate(stimuli):
        cat = s["category"]
        if cat not in cat_indices:
            cat_indices[cat] = []
        cat_indices[cat].append(i)

    attr_data = np.load(attribution_path)

    print(f"\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Experiment 1
    results["pathway_decomposition"] = experiment_1_pathway_decomposition(
        model, tokenizer, stimuli, cat_indices, attr_data, n_layers, ffn_dim, device
    )

    # Experiment 2
    results["steering"] = experiment_2_steering(
        model, tokenizer, stimuli, cat_indices, attr_data, n_layers, ffn_dim, device
    )

    # Experiment 3
    results["instance_prediction"] = experiment_3_instance_prediction(
        model, tokenizer, stimuli, cat_indices, attr_data, n_layers, ffn_dim, device
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"All experiments completed in {elapsed:.0f}s")

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert results to serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = os.path.join(output_dir, f"{model_short}_predictive_results.json")
    with open(out_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    print(f"Saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_short", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--attribution_path", type=str, required=True)
    parser.add_argument("--stimuli", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    run_all_predictive_experiments(
        model_path=args.model_path,
        model_short=args.model_short,
        meta_path=args.meta_path,
        attribution_path=args.attribution_path,
        stimuli_path=args.stimuli,
        output_dir=args.output_dir,
    )
