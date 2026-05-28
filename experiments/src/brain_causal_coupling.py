#!/usr/bin/env python3
"""
Direction A: Does the brain's coupling structure predict LLM causal coupling?

Pipeline:
  1. Compute gradient × activation importance for each of 14 cognitive conditions
  2. Compute one-vs-rest selectivity per condition
  3. Ablate top-N neurons per condition → measure PPL on ALL 14 conditions
  4. Build 14×14 causal coupling matrix (how much ablating condition X damages condition Y)
  5. Correlate with brain RDM → if the brain predicts LLM causal interdependencies

Adapted from multi_function_dissociation.py for 14 cognitive conditions.
Requires GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Cannot find model layers")


def get_ffn_dim(model, n_layers):
    layers = get_model_layers(model)
    mlp = layers[0].mlp
    if hasattr(mlp, "gate_proj"):
        return mlp.gate_proj.out_features
    if hasattr(mlp, "dense_h_to_4h"):
        return mlp.dense_h_to_4h.out_features
    raise ValueError("Cannot determine FFN dim")


def compute_all_attributions(model, tokenizer, stimuli, conditions,
                             n_layers, ffn_dim, device="cuda", max_length=256):
    n_neurons = n_layers * ffn_dim
    cond_importance = {c: np.zeros(n_neurons, dtype=np.float64) for c in conditions}
    cond_counts = {c: 0 for c in conditions}

    layers = get_model_layers(model)
    activations = {}
    hooks = []

    for lidx in range(n_layers):
        mlp = layers[lidx].mlp
        target = mlp.gate_proj if hasattr(mlp, "gate_proj") else mlp.dense_h_to_4h

        def make_hook(idx):
            def fwd_hook(module, inp, out):
                activations[idx] = out
                out.retain_grad()
            return fwd_hook
        hooks.append(target.register_forward_hook(make_hook(lidx)))

    total = len(stimuli)
    print(f"  Computing attributions for {total} stimuli, {len(conditions)} conditions...")

    for idx, sample in enumerate(stimuli):
        cond = sample["condition"]
        if cond not in cond_importance:
            continue

        inputs = tokenizer(
            sample["text"], return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(device)
        if inputs["input_ids"].shape[1] < 2:
            continue

        model.zero_grad()
        activations.clear()

        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()

        importance = np.zeros(n_neurons, dtype=np.float32)
        for lidx in range(n_layers):
            if lidx not in activations or activations[lidx].grad is None:
                continue
            act = activations[lidx]
            imp = (act.detach() * act.grad.detach()).abs().mean(dim=(0, 1))
            start, end = lidx * ffn_dim, (lidx + 1) * ffn_dim
            importance[start:end] = imp.cpu().float().numpy()

        cond_importance[cond] += importance.astype(np.float64)
        cond_counts[cond] += 1

        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{total} samples processed")

    for h in hooks:
        h.remove()

    for c in conditions:
        if cond_counts[c] > 0:
            cond_importance[c] /= cond_counts[c]

    print(f"  Attribution counts: {cond_counts}")
    return cond_importance, cond_counts


def compute_selectivity(cond_importance, conditions):
    eps = 1e-10
    selectivity = {}
    for target in conditions:
        imp_target = cond_importance[target]
        others = [cond_importance[c] for c in conditions if c != target]
        imp_others = np.mean(others, axis=0)
        sel = (imp_target - imp_others) / (imp_target + imp_others + eps)
        selectivity[target] = sel
    return selectivity


def measure_ppl_per_condition(model, tokenizer, stimuli, conditions,
                              device="cuda", max_length=256):
    model.eval()
    results = {}
    for cond in conditions:
        cond_stims = [s for s in stimuli if s["condition"] == cond]
        total_loss = 0.0
        total_tokens = 0
        for s in cond_stims:
            inputs = tokenizer(
                s["text"], return_tensors="pt",
                truncation=True, max_length=max_length
            ).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            with torch.no_grad():
                out = model(**inputs, labels=inputs["input_ids"])
            if not torch.isnan(out.loss) and not torch.isinf(out.loss):
                nt = inputs["input_ids"].shape[1] - 1
                total_loss += out.loss.item() * nt
                total_tokens += nt
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            results[cond] = {"perplexity": float(np.exp(min(avg_loss, 100))),
                             "avg_loss": float(avg_loss),
                             "n_samples": len(cond_stims)}
        else:
            results[cond] = {"perplexity": float("inf"), "avg_loss": float("inf"),
                             "n_samples": 0}
    return results


def ablate_and_measure(model, tokenizer, stimuli, conditions,
                       selectivity, n_ablate, device="cuda", max_length=256):
    """For each condition: ablate top-N selective neurons, measure PPL on all conditions."""
    layers = get_model_layers(model)
    n_layers = len(layers)
    ffn_dim = get_ffn_dim(model, n_layers)
    ablation_results = {}

    for target_cond in conditions:
        print(f"\n  Ablating top {n_ablate} neurons for: {target_cond}")
        sel = selectivity[target_cond]
        top_neurons = np.argsort(-sel)[:n_ablate]

        # Group by layer
        layer_neurons = {}
        for n in top_neurons:
            l = n // ffn_dim
            pos = n % ffn_dim
            if l not in layer_neurons:
                layer_neurons[l] = []
            layer_neurons[l].append(pos)

        # Install ablation hooks
        ablation_hooks = []
        for lidx, positions in layer_neurons.items():
            mlp = layers[lidx].mlp
            target_module = mlp.gate_proj if hasattr(mlp, "gate_proj") else mlp.dense_h_to_4h
            pos_tensor = torch.tensor(positions, device=device)

            def make_ablation_hook(pos_t):
                def hook(module, inp, out):
                    out[:, :, pos_t] = 0.0
                    return out
                return hook
            ablation_hooks.append(
                target_module.register_forward_hook(make_ablation_hook(pos_tensor))
            )

        ppl = measure_ppl_per_condition(model, tokenizer, stimuli, conditions,
                                        device, max_length)
        ablation_results[target_cond] = ppl

        for h in ablation_hooks:
            h.remove()

        target_ppl = ppl[target_cond]["perplexity"]
        mean_other = np.mean([ppl[c]["perplexity"] for c in conditions if c != target_cond])
        print(f"    target PPL={target_ppl:.2f}, mean other PPL={mean_other:.2f}")

    return ablation_results


def build_coupling_matrix(baseline, ablation_results, conditions):
    """Build 14×14 causal coupling matrix.

    Entry [i, j] = log(PPL_j after ablating i / baseline PPL_j)
    = how much ablating condition i's neurons damages condition j
    """
    n = len(conditions)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, ci in enumerate(conditions):
        for j, cj in enumerate(conditions):
            base_ppl = baseline[cj]["perplexity"]
            abl_ppl = ablation_results[ci][cj]["perplexity"]
            if base_ppl > 0 and not np.isinf(base_ppl):
                matrix[i, j] = np.log(abl_ppl / base_ppl)
            else:
                matrix[i, j] = 0.0
    return matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_short", required=True)
    ap.add_argument("--stimuli", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--brain_rdm", required=True)
    ap.add_argument("--n_ablate", type=int, default=5000)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    from scipy.stats import spearmanr

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    n_layers = len(get_model_layers(model))
    ffn_dim = get_ffn_dim(model, n_layers)
    n_neurons = n_layers * ffn_dim
    print(f"  n_layers={n_layers}, ffn_dim={ffn_dim}, n_neurons={n_neurons}")

    # Load stimuli
    with open(args.stimuli) as f:
        stimuli = [json.loads(l) for l in f]
    conditions = sorted(set(s["condition"] for s in stimuli))
    print(f"  {len(stimuli)} stimuli, {len(conditions)} conditions: {conditions}")

    # Load brain RDM
    brain_data = np.load(args.brain_rdm, allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])
    assert sorted(conditions) == sorted(brain_conds), \
        f"Condition mismatch: {conditions} vs {brain_conds}"
    # Reorder brain to match our condition order
    brain_order = [brain_conds.index(c) for c in conditions]
    brain_rdm = brain_rdm[np.ix_(brain_order, brain_order)]

    # Step 1: Compute attributions
    print("\n" + "=" * 70)
    print("STEP 1: Gradient × Activation attribution")
    print("=" * 70)
    model.train()
    cond_importance, cond_counts = compute_all_attributions(
        model, tokenizer, stimuli, conditions,
        n_layers, ffn_dim, device, args.max_length
    )

    # Step 2: Compute selectivity
    print("\n" + "=" * 70)
    print("STEP 2: One-vs-rest selectivity")
    print("=" * 70)
    selectivity = compute_selectivity(cond_importance, conditions)
    for c in conditions:
        top5k = np.sort(selectivity[c])[-5000:]
        print(f"  {c:>20s}: top-5K selectivity mean={top5k.mean():.4f}, "
              f"max={top5k.max():.4f}")

    # Save attribution data
    os.makedirs(args.output_dir, exist_ok=True)
    attr_data = {}
    for c in conditions:
        attr_data[f"{c}_importance"] = cond_importance[c].astype(np.float32)
        attr_data[f"{c}_selectivity"] = selectivity[c].astype(np.float32)
    np.savez_compressed(
        os.path.join(args.output_dir, f"{args.model_short}_cognitive_attribution.npz"),
        **attr_data
    )

    # Step 3: Baseline PPL
    print("\n" + "=" * 70)
    print("STEP 3: Baseline PPL")
    print("=" * 70)
    model.eval()
    baseline = measure_ppl_per_condition(
        model, tokenizer, stimuli, conditions, device, args.max_length
    )
    for c in conditions:
        print(f"  {c:>20s}: PPL={baseline[c]['perplexity']:.2f}")

    # Step 4: Ablation (14 conditions × measure all 14)
    print("\n" + "=" * 70)
    print("STEP 4: Causal ablation (14×14)")
    print("=" * 70)
    ablation_results = ablate_and_measure(
        model, tokenizer, stimuli, conditions,
        selectivity, args.n_ablate, device, args.max_length
    )

    # Step 5: Build coupling matrix and correlate with brain
    print("\n" + "=" * 70)
    print("STEP 5: Causal coupling matrix vs brain RDM")
    print("=" * 70)

    coupling = build_coupling_matrix(baseline, ablation_results, conditions)
    print(f"\n  Causal coupling matrix (log PPL ratio):")
    print(f"  {'':>20s}", end="")
    for c in conditions:
        print(f" {c[:8]:>8s}", end="")
    print()
    for i, ci in enumerate(conditions):
        print(f"  {ci:>20s}", end="")
        for j in range(len(conditions)):
            print(f" {coupling[i,j]:>+8.3f}", end="")
        print()

    # The coupling matrix is ASYMMETRIC: coupling[i,j] = damage to j when ablating i
    # We want to test: does the brain's DISSIMILARITY predict the LLM's CAUSAL INDEPENDENCE?
    # High brain RDM(i,j) = brain says i and j are far apart → expect LOW coupling[i,j]
    # Low brain RDM(i,j) = brain says i and j are close → expect HIGH coupling[i,j]
    # So we expect NEGATIVE correlation between brain RDM and coupling matrix

    # Also test: does brain RDM predict coupling better than a random baseline?

    n = len(conditions)
    triu = np.triu_indices(n, k=1)

    # Symmetrize coupling: (C[i,j] + C[j,i]) / 2
    coupling_sym = (coupling + coupling.T) / 2
    # Exclude diagonal for correlation
    brain_vec = brain_rdm[triu]
    coupling_vec = coupling_sym[triu]

    rho_sym, p_sym = spearmanr(brain_vec, coupling_vec)
    print(f"\n  Brain RDM vs symmetric causal coupling:")
    print(f"    Spearman ρ = {rho_sym:+.4f}, p = {p_sym:.4f}")
    print(f"    (expect NEGATIVE: high brain dissimilarity → low causal coupling)")
    if rho_sym < 0 and p_sym < 0.05:
        print(f"    → SIGNIFICANT: brain coupling structure PREDICTS LLM causal interdependencies")
    elif rho_sym < 0:
        print(f"    → Correct direction but not significant")
    else:
        print(f"    → Wrong direction")

    # Also test asymmetric: each row of coupling predicts brain row
    print(f"\n  Per-condition (row-wise) brain vs coupling:")
    row_rhos = []
    for i, c in enumerate(conditions):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r, p = spearmanr(brain_rdm[i, mask], coupling[i, mask])
        row_rhos.append(r)
        print(f"    {c:>20s}: ρ={r:+.4f} (p={p:.3f})")
    print(f"    Mean row-wise ρ: {np.mean(row_rhos):+.4f}")

    # Random control: correlation of brain RDM with random coupling matrix
    rng = np.random.default_rng(2026)
    null_rhos = []
    for _ in range(10000):
        perm = rng.permutation(n)
        brain_perm = brain_rdm[np.ix_(perm, perm)]
        null_rhos.append(float(spearmanr(brain_perm[triu], coupling_vec)[0]))
    null_rhos = np.array(null_rhos)
    p_perm = float((np.sum(null_rhos <= rho_sym) + 1) / (len(null_rhos) + 1))
    print(f"\n  Permutation null: mean={np.mean(null_rhos):+.4f}, "
          f"95th={np.percentile(null_rhos, 5):+.4f}")
    print(f"  Permutation p (one-sided, negative) = {p_perm:.4f}")

    # Save everything
    elapsed = time.time() - t0
    result = {
        "model": args.model_short,
        "n_neurons": n_neurons,
        "n_ablate": args.n_ablate,
        "conditions": conditions,
        "baseline": baseline,
        "ablation_results": ablation_results,
        "coupling_matrix_logppl": coupling.tolist(),
        "coupling_matrix_symmetric": coupling_sym.tolist(),
        "brain_rdm": brain_rdm.tolist(),
        "rsa_brain_vs_coupling": {
            "symmetric_rho": float(rho_sym),
            "symmetric_p": float(p_sym),
            "permutation_p": float(p_perm),
            "row_wise_rhos": {c: float(r) for c, r in zip(conditions, row_rhos)},
            "row_wise_mean_rho": float(np.mean(row_rhos)),
            "null_mean": float(np.mean(null_rhos)),
            "null_std": float(np.std(null_rhos)),
        },
        "elapsed_seconds": elapsed,
    }
    out_path = os.path.join(args.output_dir, f"{args.model_short}_brain_causal_coupling.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
