"""
Accuracy-based multi-function dissociation: extend causal ablation evaluation
to downstream task accuracy (not just perplexity).

For each category X:
  1. Compute grad x act importance on DISCOVERY stimuli
  2. Selectivity_X = (imp_X - mean_imp_others) / (imp_X + mean_imp_others + eps)
  3. Ablate top-N most X-selective neurons
  4. Measure ACCURACY + PPL on VALIDATION stimuli across ALL categories

answer_type handling:
  - multiple_choice: log-prob of option letters (A/B/C/D), pick highest
  - exact_match: generate up to 64 tokens, extract answer, compare to gold
  - completion: log-prob of gold completion given prompt
  - generation: mean log-prob of gold answer tokens given prompt

If ablating X-selective neurons damages X accuracy >> other categories for all X,
we have an N-way functional dissociation validated by task accuracy.
"""

import json
import math
import os
import re
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Attribution (reused from multi_function_dissociation.py)
# ---------------------------------------------------------------------------

def get_layers(model):
    """Return the transformer layer list regardless of architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise RuntimeError("Cannot find transformer layers in model architecture")


def get_gate_module(mlp):
    """Return the gate/up-projection module of an MLP block."""
    if hasattr(mlp, "gate_proj"):
        return mlp.gate_proj
    if hasattr(mlp, "dense_h_to_4h"):
        return mlp.dense_h_to_4h
    return None


def compute_all_attributions(
    model, tokenizer, stimuli, categories,
    n_layers, ffn_dim, device="cuda", max_length=256,
):
    """Compute grad x act importance for every category using DISCOVERY stimuli."""
    n_neurons = n_layers * ffn_dim
    cat_importance = {cat: np.zeros(n_neurons, dtype=np.float64) for cat in categories}
    cat_counts = {cat: 0 for cat in categories}

    layers = get_layers(model)
    activations = {}
    hooks = []

    for layer_idx in range(n_layers):
        gate = get_gate_module(layers[layer_idx].mlp)
        if gate is None:
            continue

        def make_hook(lidx):
            def fwd_hook(module, input, output):
                activations[lidx] = output
                output.retain_grad()
            return fwd_hook

        hooks.append(gate.register_forward_hook(make_hook(layer_idx)))

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


# ---------------------------------------------------------------------------
# Ablation hooks
# ---------------------------------------------------------------------------

def install_ablation_hooks(model, neuron_indices, n_layers, ffn_dim, device="cuda"):
    """Zero-out specified global neuron indices. Returns list of hooks to remove."""
    layers = get_layers(model)
    hooks = []
    neuron_set = set(int(n) for n in neuron_indices)

    for layer_idx in range(n_layers):
        start = layer_idx * ffn_dim
        end = start + ffn_dim
        layer_neurons = [n - start for n in neuron_set if start <= n < end]
        if not layer_neurons:
            continue

        gate = get_gate_module(layers[layer_idx].mlp)
        if gate is None:
            continue

        mask = torch.zeros(ffn_dim, dtype=torch.bool, device=device)
        mask[layer_neurons] = True

        def make_hook(m):
            def hook_fn(module, input, output):
                output[:, :, m] = 0.0
                return output
            return hook_fn

        hooks.append(gate.register_forward_hook(make_hook(mask)))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Accuracy evaluation helpers
# ---------------------------------------------------------------------------

# Token IDs for option letters — resolved per-tokenizer at runtime
_OPTION_TOKEN_CACHE = {}


def _get_option_token_ids(tokenizer):
    """Get token IDs for A, B, C, D option letters (handles various tokenizer styles)."""
    key = id(tokenizer)
    if key in _OPTION_TOKEN_CACHE:
        return _OPTION_TOKEN_CACHE[key]

    option_ids = {}
    for letter in ["A", "B", "C", "D"]:
        # Try multiple encodings — different tokenizers handle single-char differently
        candidates = set()
        for text in [letter, f" {letter}", f"{letter}"]:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                candidates.add(ids[-1])  # last token is the letter itself

        # Pick the single-character encoding if available
        direct = tokenizer.encode(letter, add_special_tokens=False)
        if direct:
            option_ids[letter] = direct[-1]
        elif candidates:
            option_ids[letter] = min(candidates)  # deterministic fallback
        else:
            raise ValueError(f"Cannot encode option letter '{letter}' with this tokenizer")

    _OPTION_TOKEN_CACHE[key] = option_ids
    return option_ids


def evaluate_multiple_choice(model, tokenizer, text, gold_answer, device="cuda", max_length=256):
    """
    Evaluate multiple-choice: compute log-prob of each option letter as next token.
    Returns (predicted_letter, correct, option_logprobs_dict).
    """
    option_ids = _get_option_token_ids(tokenizer)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # logits at last position predict the next token
    last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
    log_probs = torch.log_softmax(last_logits, dim=-1)

    option_logprobs = {}
    for letter in ["A", "B", "C", "D"]:
        tid = option_ids[letter]
        option_logprobs[letter] = log_probs[tid].item()

    predicted = max(option_logprobs, key=option_logprobs.get)
    # Normalize gold answer to single uppercase letter
    gold_letter = gold_answer.strip().upper()
    if len(gold_letter) > 1:
        gold_letter = gold_letter[0]  # take first char if e.g. "A)"

    correct = (predicted == gold_letter)
    return predicted, correct, option_logprobs


def extract_number(text):
    """Extract the final number from generated text (for math tasks like GSM8K)."""
    # Look for #### pattern first (GSM8K convention)
    match = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Otherwise take the last number in the text
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return None


def evaluate_exact_match(model, tokenizer, text, gold_answer, device="cuda",
                         max_length=256, max_new_tokens=64):
    """
    Generate up to max_new_tokens tokens, then check:
    - For numeric gold answers: extract last number from generation, compare.
    - For string gold answers: check if gold appears in generation (case-insensitive).
    Returns (generation, correct).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_len:]
    generation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    gold_str = str(gold_answer).strip()

    # Try numeric comparison first
    gold_num = None
    try:
        gold_num = float(gold_str.replace(",", ""))
    except ValueError:
        pass

    if gold_num is not None:
        pred_num = extract_number(generation)
        if pred_num is not None:
            correct = abs(pred_num - gold_num) < 1e-3
        else:
            correct = False
    else:
        # String containment (case-insensitive), also check exact match
        correct = gold_str.lower() in generation.lower()

    return generation, correct


def evaluate_completion_logprob(model, tokenizer, text, gold_answer, device="cuda", max_length=256):
    """
    Compute the log-probability of the gold completion given the prompt.
    Returns (mean_logprob, total_logprob, n_tokens).
    """
    # Tokenize prompt and completion separately, then concatenate
    prompt_ids = tokenizer.encode(text, add_special_tokens=True)
    completion_ids = tokenizer.encode(gold_answer, add_special_tokens=False)

    if not completion_ids:
        return 0.0, 0.0, 0

    # Truncate if needed — keep prompt + completion within max_length
    max_prompt = max_length - len(completion_ids)
    if max_prompt < 1:
        max_prompt = 1
        completion_ids = completion_ids[:max_length - 1]
    prompt_ids = prompt_ids[-max_prompt:]  # keep end of prompt if truncated

    full_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_tensor)
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Compute log-probs for each completion token
    log_probs = torch.log_softmax(logits, dim=-1)

    prompt_len = len(prompt_ids)
    total_logprob = 0.0
    n_tokens = 0

    for i, tid in enumerate(completion_ids):
        # The logit at position (prompt_len - 1 + i) predicts token at (prompt_len + i)
        pos = prompt_len - 1 + i
        if pos < logits.shape[0]:
            total_logprob += log_probs[pos, tid].item()
            n_tokens += 1

    mean_logprob = total_logprob / n_tokens if n_tokens > 0 else 0.0
    return mean_logprob, total_logprob, n_tokens


def evaluate_generation_logprob(model, tokenizer, text, gold_answer, device="cuda", max_length=256):
    """
    Compute mean log-prob of gold answer tokens given the prompt.
    Same as completion_logprob but framed as a generation quality metric.
    Returns (mean_logprob, total_logprob, n_tokens).
    """
    return evaluate_completion_logprob(model, tokenizer, text, gold_answer, device, max_length)


# ---------------------------------------------------------------------------
# PPL measurement (same as multi_function_dissociation.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Combined accuracy + PPL evaluation
# ---------------------------------------------------------------------------

def measure_accuracy_and_ppl(model, tokenizer, stimuli, cat_indices,
                             device="cuda", max_length=256):
    """
    Evaluate both accuracy and PPL for each category on the validation set.
    Each stimulus must have 'answer' and 'answer_type' fields.
    """
    model.eval()
    results = {}

    for cat, indices in cat_indices.items():
        n_correct = 0
        n_total = 0
        total_loss = 0.0
        total_tokens = 0
        n_valid_ppl = 0
        # For logprob-based answer_types, accumulate mean logprobs
        logprob_values = []

        for idx in indices:
            sample = stimuli[idx]
            text = sample["text"]
            gold_answer = sample.get("answer", "")
            answer_type = sample.get("answer_type", "generation")

            # --- PPL ---
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            if inputs["input_ids"].shape[1] >= 2:
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss_val = outputs.loss
                if not torch.isnan(loss_val) and not torch.isinf(loss_val):
                    nt = inputs["input_ids"].shape[1] - 1
                    total_loss += loss_val.item() * nt
                    total_tokens += nt
                    n_valid_ppl += 1

            # --- Accuracy ---
            if answer_type == "multiple_choice":
                _, correct, _ = evaluate_multiple_choice(
                    model, tokenizer, text, gold_answer, device, max_length
                )
                n_correct += int(correct)
                n_total += 1

            elif answer_type == "exact_match":
                _, correct = evaluate_exact_match(
                    model, tokenizer, text, gold_answer, device, max_length
                )
                n_correct += int(correct)
                n_total += 1

            elif answer_type == "completion":
                mean_lp, _, n_tok = evaluate_completion_logprob(
                    model, tokenizer, text, gold_answer, device, max_length
                )
                if n_tok > 0:
                    logprob_values.append(mean_lp)
                    # Threshold: if mean log-prob > -2.0, count as "correct"
                    # (model assigns reasonable probability to gold completion)
                    n_correct += int(mean_lp > -2.0)
                    n_total += 1

            elif answer_type == "generation":
                mean_lp, _, n_tok = evaluate_generation_logprob(
                    model, tokenizer, text, gold_answer, device, max_length
                )
                if n_tok > 0:
                    logprob_values.append(mean_lp)
                    n_correct += int(mean_lp > -2.0)
                    n_total += 1

        # Aggregate
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            ppl = np.exp(min(avg_loss, 100))
        else:
            avg_loss = float("inf")
            ppl = float("inf")

        cat_result = {
            "accuracy": float(accuracy),
            "n_correct": int(n_correct),
            "n_total": int(n_total),
            "perplexity": float(ppl),
            "avg_loss": float(avg_loss),
            "n_samples_ppl": int(n_valid_ppl),
        }
        if logprob_values:
            cat_result["mean_logprob"] = float(np.mean(logprob_values))

        results[cat] = cat_result

    return results


def ablate_and_measure_accuracy(model, tokenizer, stimuli, neuron_indices,
                                n_layers, ffn_dim, cat_indices,
                                device="cuda", max_length=256):
    """Ablate specific neurons and measure accuracy + PPL."""
    hooks = install_ablation_hooks(model, neuron_indices, n_layers, ffn_dim, device)
    results = measure_accuracy_and_ppl(model, tokenizer, stimuli, cat_indices, device, max_length)
    remove_hooks(hooks)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_accuracy_dissociation(
    model_path, model_short, meta_path,
    discovery_stimuli_path, validation_stimuli_path,
    output_dir, n_ablate=5000, device="cuda",
):
    t0 = time.time()
    print(f"{'='*70}")
    print(f"Accuracy Dissociation: {model_short}")
    print(f"{'='*70}")

    # --- Load metadata ---
    with open(meta_path) as f:
        meta = json.load(f)
    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]
    n_neurons = n_layers * ffn_dim

    # --- Load discovery stimuli (for attribution) ---
    with open(discovery_stimuli_path) as f:
        discovery_stimuli = [json.loads(line) for line in f]

    discovery_cat_indices = {}
    for i, s in enumerate(discovery_stimuli):
        cat = s["category"]
        if cat not in discovery_cat_indices:
            discovery_cat_indices[cat] = []
        discovery_cat_indices[cat].append(i)

    # --- Load validation stimuli (for accuracy measurement) ---
    with open(validation_stimuli_path) as f:
        validation_stimuli = [json.loads(line) for line in f]

    val_cat_indices = {}
    for i, s in enumerate(validation_stimuli):
        cat = s["category"]
        if cat not in val_cat_indices:
            val_cat_indices[cat] = []
        val_cat_indices[cat].append(i)

    # Categories must match between discovery and validation
    disc_cats = sorted(discovery_cat_indices.keys())
    val_cats = sorted(val_cat_indices.keys())
    categories = sorted(set(disc_cats) & set(val_cats))
    if len(categories) < len(disc_cats) or len(categories) < len(val_cats):
        print(f"  WARNING: category mismatch!")
        print(f"    Discovery: {disc_cats}")
        print(f"    Validation: {val_cats}")
        print(f"    Intersection: {categories}")

    print(f"  Categories: {categories}")
    print(f"  Discovery samples: {', '.join(f'{c}={len(discovery_cat_indices.get(c, []))}' for c in categories)}")
    print(f"  Validation samples: {', '.join(f'{c}={len(val_cat_indices.get(c, []))}' for c in categories)}")

    # Check answer_type distribution
    type_counts = {}
    for s in validation_stimuli:
        at = s.get("answer_type", "unknown")
        type_counts[at] = type_counts.get(at, 0) + 1
    print(f"  Validation answer_types: {type_counts}")

    # --- Load model ---
    print(f"\n[Step 1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Attribution on discovery set ---
    print(f"\n[Step 2] Computing causal attribution on DISCOVERY set ({len(discovery_stimuli)} samples)...")
    cat_importance, cat_counts = compute_all_attributions(
        model, tokenizer, discovery_stimuli, categories, n_layers, ffn_dim, device
    )
    for cat in categories:
        imp = cat_importance[cat]
        print(f"  {cat:>12s}: {cat_counts[cat]} samples, mean_imp={imp.mean():.6f}, max={imp.max():.6f}")

    # --- Selectivity ---
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
    np.savez(os.path.join(output_dir, f"{model_short}_accuracy_attribution.npz"), **save_dict)

    # --- Baseline accuracy + PPL on validation set ---
    print(f"\n[Step 4] Baseline accuracy + PPL on VALIDATION set ({len(validation_stimuli)} samples)...")
    baseline = measure_accuracy_and_ppl(model, tokenizer, validation_stimuli, val_cat_indices, device)
    print(f"\n  {'Category':>12s}  {'Accuracy':>8s}  {'PPL':>8s}  {'n_correct':>9s}  {'n_total':>7s}")
    print(f"  {'-'*55}")
    for cat in categories:
        b = baseline[cat]
        print(f"  {cat:>12s}  {b['accuracy']:>8.3f}  {b['perplexity']:>8.2f}  "
              f"{b['n_correct']:>9d}  {b['n_total']:>7d}")

    # --- N-way ablation ---
    print(f"\n[Step 5] N-way ablation (N={n_ablate:,}, {n_ablate/n_neurons*100:.2f}% of network)...")
    print(f"  Measuring accuracy + PPL on validation set after each ablation.")

    ablation_results = {}
    for target_cat in categories:
        sel = selectivity[target_cat]
        top_neurons = np.argsort(-sel)[:n_ablate]
        sel_range = f"[{sel[top_neurons[-1]]:.4f}, {sel[top_neurons[0]]:.4f}]"
        print(f"\n  Ablating {n_ablate} {target_cat.upper()}-selective neurons (sel range {sel_range})...")

        ablated = ablate_and_measure_accuracy(
            model, tokenizer, validation_stimuli, top_neurons,
            n_layers, ffn_dim, val_cat_indices, device,
        )
        ablation_results[target_cat] = ablated

        # Show accuracy damage profile
        for cat in categories:
            bl_acc = baseline[cat]["accuracy"]
            ab_acc = ablated[cat]["accuracy"]
            acc_drop = bl_acc - ab_acc
            bl_ppl = baseline[cat]["perplexity"]
            ab_ppl = ablated[cat]["perplexity"]
            ppl_ratio = ab_ppl / bl_ppl if bl_ppl > 0 else float("inf")
            marker = " <<<" if cat == target_cat else ""
            print(f"    {cat:>12s}: acc={ab_acc:.3f} (drop={acc_drop:>+.3f}), "
                  f"PPL={ab_ppl:.2f} (x{ppl_ratio:.2f}){marker}")

    # --- Random control ---
    print(f"\n  Random ablation control ({n_ablate:,} neurons)...")
    rng = np.random.RandomState(42)
    random_neurons = rng.choice(n_neurons, n_ablate, replace=False)
    random_result = ablate_and_measure_accuracy(
        model, tokenizer, validation_stimuli, random_neurons,
        n_layers, ffn_dim, val_cat_indices, device,
    )
    for cat in categories:
        bl_acc = baseline[cat]["accuracy"]
        rn_acc = random_result[cat]["accuracy"]
        acc_drop = bl_acc - rn_acc
        print(f"    {cat:>12s}: acc={rn_acc:.3f} (drop={acc_drop:>+.3f})")

    # --- Build dissociation matrices ---
    n_cats = len(categories)

    # Accuracy drop matrix: baseline_acc - ablated_acc (positive = damage)
    acc_dissoc_matrix = np.zeros((n_cats, n_cats))
    # PPL ratio matrix (for comparison)
    ppl_dissoc_matrix = np.zeros((n_cats, n_cats))

    for i, ablated_cat in enumerate(categories):
        for j, measured_cat in enumerate(categories):
            bl_acc = baseline[measured_cat]["accuracy"]
            ab_acc = ablation_results[ablated_cat][measured_cat]["accuracy"]
            acc_dissoc_matrix[i, j] = bl_acc - ab_acc

            bl_ppl = baseline[measured_cat]["perplexity"]
            ab_ppl = ablation_results[ablated_cat][measured_cat]["perplexity"]
            ppl_dissoc_matrix[i, j] = ab_ppl / bl_ppl if bl_ppl > 0 else float("inf")

    # --- Print accuracy dissociation matrix ---
    print(f"\n{'='*70}")
    print(f"ACCURACY DISSOCIATION MATRIX (accuracy DROP: baseline - ablated)")
    print(f"{'='*70}")
    header = f"{'Ablated':>12s} |" + "".join(f" {c:>8s}" for c in categories) + " | random"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for i, ablated_cat in enumerate(categories):
        row = f"  {ablated_cat:>12s} |"
        for j, measured_cat in enumerate(categories):
            drop = acc_dissoc_matrix[i, j]
            marker = "*" if i == j else " "
            row += f" {drop:>+7.3f}{marker}"
        rn_drop = baseline[ablated_cat]["accuracy"] - random_result[ablated_cat]["accuracy"]
        row += f" | {rn_drop:+.3f}"
        print(row)

    # --- Print PPL dissociation matrix ---
    print(f"\n{'='*70}")
    print(f"PPL DISSOCIATION MATRIX (PPL ratio: ablated / baseline)")
    print(f"{'='*70}")
    header2 = f"{'Ablated':>12s} |" + "".join(f" {c:>8s}" for c in categories) + " | random"
    print(f"  {header2}")
    print(f"  {'-'*len(header2)}")

    for i, ablated_cat in enumerate(categories):
        row = f"  {ablated_cat:>12s} |"
        for j, measured_cat in enumerate(categories):
            ratio = ppl_dissoc_matrix[i, j]
            marker = "*" if i == j else " "
            row += f" {ratio:>7.2f}{marker}"
        rn_ratio = random_result[ablated_cat]["perplexity"] / baseline[ablated_cat]["perplexity"] \
            if baseline[ablated_cat]["perplexity"] > 0 else 0
        row += f" | {rn_ratio:.2f}"
        print(row)

    # --- Dissociation analysis ---
    print(f"\n{'='*70}")
    print(f"ACCURACY DISSOCIATION ANALYSIS")
    print(f"{'='*70}")

    successes = 0
    for i, cat in enumerate(categories):
        on_target = acc_dissoc_matrix[i, i]
        off_target = np.delete(acc_dissoc_matrix[i], i)
        max_collateral = off_target.max()
        mean_collateral = off_target.mean()

        # Specificity: target accuracy drop > 1.5x max collateral AND drop > 0.05 (5%)
        specific = on_target > max_collateral * 1.5 and on_target > 0.05
        ratio_vs_max = on_target / (max_collateral + 1e-10)

        status = "PASS" if specific else "FAIL"
        print(f"  {cat:>12s}: target_drop={on_target:>+.3f}, max_collateral={max_collateral:>+.3f}, "
              f"ratio={ratio_vs_max:.1f}x, specificity: {status}")
        if specific:
            successes += 1

    # Pairwise double dissociation
    print(f"\n  Pairwise double dissociation (accuracy-based):")
    pair_pass = 0
    pair_total = 0
    for i in range(n_cats):
        for j in range(i + 1, n_cats):
            a_on_a = acc_dissoc_matrix[i, i]
            a_on_b = acc_dissoc_matrix[i, j]
            b_on_b = acc_dissoc_matrix[j, j]
            b_on_a = acc_dissoc_matrix[j, i]
            dd = (a_on_a > a_on_b * 1.5 and a_on_a > 0.05) and \
                 (b_on_b > b_on_a * 1.5 and b_on_b > 0.05)
            pair_total += 1
            if dd:
                pair_pass += 1
                print(f"    {categories[i]:>12s} vs {categories[j]:<12s}: YES  "
                      f"({a_on_a:+.3f}/{a_on_b:+.3f}, {b_on_b:+.3f}/{b_on_a:+.3f})")

    print(f"\n  Summary: {successes}/{n_cats} categories show accuracy specificity")
    print(f"  Pairwise double dissociation (accuracy): {pair_pass}/{pair_total} pairs pass")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    # --- Save results ---
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return str(obj)

    result = {
        "model": model_short,
        "n_neurons": n_neurons,
        "n_ablate": n_ablate,
        "pct_neurons": n_ablate / n_neurons * 100,
        "categories": categories,
        "discovery_stimuli": discovery_stimuli_path,
        "validation_stimuli": validation_stimuli_path,
        "n_discovery_samples": len(discovery_stimuli),
        "n_validation_samples": len(validation_stimuli),
        "baseline_accuracy": {c: baseline[c]["accuracy"] for c in categories},
        "baseline_ppl": {c: baseline[c]["perplexity"] for c in categories},
        "baseline": to_serializable(baseline),
        "ablation_results": to_serializable(ablation_results),
        "random_control": to_serializable(random_result),
        "accuracy_dissociation_matrix": acc_dissoc_matrix.tolist(),
        "ppl_dissociation_matrix": ppl_dissoc_matrix.tolist(),
        "n_specific_categories_accuracy": successes,
        "n_pairwise_dissociations_accuracy": pair_pass,
        "total_pairwise": pair_total,
        "elapsed_seconds": elapsed,
    }

    out_path = os.path.join(output_dir, f"{model_short}_accuracy_dissociation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Accuracy-based multi-function dissociation experiment"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model")
    parser.add_argument("--model_short", type=str, required=True,
                        help="Short model name for output files")
    parser.add_argument("--meta_path", type=str, required=True,
                        help="Path to model meta JSON (n_layers, ffn_dim)")
    parser.add_argument("--discovery_stimuli", type=str, required=True,
                        help="JSONL with answers, used for attribution computation")
    parser.add_argument("--validation_stimuli", type=str, required=True,
                        help="JSONL with answers, used for accuracy measurement")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--n_ablate", type=int, default=5000,
                        help="Number of neurons to ablate per category (default: 5000)")
    args = parser.parse_args()

    run_accuracy_dissociation(
        model_path=args.model_path,
        model_short=args.model_short,
        meta_path=args.meta_path,
        discovery_stimuli_path=args.discovery_stimuli,
        validation_stimuli_path=args.validation_stimuli,
        output_dir=args.output_dir,
        n_ablate=args.n_ablate,
    )
