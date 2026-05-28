#!/usr/bin/env python3
"""
Direction A: Does the brain's coupling structure predict LLM causal coupling?

Pipeline:
  1. Compute gradient × activation importance for each of 14 cognitive conditions
  2. Compute one-vs-rest selectivity per condition
  3. Ablate top-N neurons per condition → measure PPL on ALL 14 conditions
  4. Build 14×14 causal coupling matrix
  5. Correlate with brain RDM

Batched processing for multi-GPU speed. Each model on 1-2 GPUs.
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


def compute_all_attributions_batched(model, tokenizer, stimuli, conditions,
                                     n_layers, ffn_dim, device="cuda",
                                     max_length=256, batch_size=16):
    """Batched gradient × activation attribution."""
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
    n_batches = (total + batch_size - 1) // batch_size
    print(f"  Attribution: {total} stimuli, batch_size={batch_size}, "
          f"{n_batches} batches, {len(conditions)} conditions")

    for batch_start in range(0, total, batch_size):
        batch = stimuli[batch_start:batch_start + batch_size]
        texts = [s["text"] for s in batch]
        conds = [s["condition"] for s in batch]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(device)

        if inputs["input_ids"].shape[1] < 2:
            continue

        model.zero_grad()
        activations.clear()

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        outputs = model(**inputs, labels=labels)
        outputs.loss.backward()

        # Extract per-sample importance via attention mask weighting
        attn_mask = inputs["attention_mask"]  # (B, seq_len)
        for lidx in range(n_layers):
            if lidx not in activations or activations[lidx].grad is None:
                continue
            act = activations[lidx].detach()   # (B, seq_len, ffn_dim)
            grad = activations[lidx].grad.detach()
            imp = (act * grad).abs()  # (B, seq_len, ffn_dim)

            # Mask padding, average over valid tokens per sample
            mask_3d = attn_mask.unsqueeze(-1).to(imp.dtype)  # (B, seq_len, 1)
            imp_masked = imp * mask_3d
            valid_counts = mask_3d.sum(dim=1).clamp(min=1)  # (B, 1)
            imp_per_sample = imp_masked.sum(dim=1) / valid_counts  # (B, ffn_dim)

            start, end = lidx * ffn_dim, (lidx + 1) * ffn_dim
            imp_np = imp_per_sample.cpu().float().numpy()  # (B, ffn_dim)

            for b_idx in range(len(batch)):
                c = conds[b_idx]
                if c in cond_importance:
                    cond_importance[c][start:end] += imp_np[b_idx].astype(np.float64)

        for b_idx in range(len(batch)):
            c = conds[b_idx]
            if c in cond_counts:
                cond_counts[c] += 1

        batch_num = batch_start // batch_size + 1
        if batch_num % 10 == 0 or batch_num == n_batches:
            print(f"    batch {batch_num}/{n_batches}")

    for h in hooks:
        h.remove()

    for c in conditions:
        if cond_counts[c] > 0:
            cond_importance[c] /= cond_counts[c]

    print(f"  Counts: {cond_counts}")
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


def measure_ppl_batched(model, tokenizer, stimuli, conditions,
                        device="cuda", max_length=256, batch_size=32):
    """Batched PPL measurement per condition."""
    model.eval()
    results = {}
    for cond in conditions:
        cond_stims = [s for s in stimuli if s["condition"] == cond]
        if not cond_stims:
            results[cond] = {"perplexity": float("inf"), "avg_loss": float("inf"),
                             "n_samples": 0}
            continue

        total_loss = 0.0
        total_tokens = 0
        for bs in range(0, len(cond_stims), batch_size):
            batch = cond_stims[bs:bs + batch_size]
            texts = [s["text"] for s in batch]
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length
            ).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue

            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            with torch.no_grad():
                out = model(**inputs, labels=labels)

            # Per-token loss (model returns mean over non-ignored tokens)
            n_valid = (labels != -100).sum().item()
            if not torch.isnan(out.loss) and not torch.isinf(out.loss) and n_valid > 0:
                total_loss += out.loss.item() * n_valid
                total_tokens += n_valid

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
                       selectivity, n_ablate, device="cuda",
                       max_length=256, batch_size=32):
    """Ablate each condition's top neurons, measure PPL on all conditions."""
    layers = get_model_layers(model)
    n_layers = len(layers)
    ffn_dim = get_ffn_dim(model, n_layers)
    ablation_results = {}

    for target_cond in conditions:
        t0 = time.time()
        sel = selectivity[target_cond]
        top_neurons = np.argsort(-sel)[:n_ablate]

        layer_neurons = {}
        for n in top_neurons:
            l = n // ffn_dim
            pos = n % ffn_dim
            if l not in layer_neurons:
                layer_neurons[l] = []
            layer_neurons[l].append(pos)

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

        ppl = measure_ppl_batched(model, tokenizer, stimuli, conditions,
                                  device, max_length, batch_size)
        ablation_results[target_cond] = ppl

        for h in ablation_hooks:
            h.remove()

        elapsed = time.time() - t0
        target_ppl = ppl[target_cond]["perplexity"]
        print(f"  ablate {target_cond:>20s}: target PPL={target_ppl:>8.2f}  ({elapsed:.1f}s)")

    return ablation_results


def build_coupling_matrix(baseline, ablation_results, conditions):
    n = len(conditions)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, ci in enumerate(conditions):
        for j, cj in enumerate(conditions):
            base_ppl = baseline[cj]["perplexity"]
            abl_ppl = ablation_results[ci][cj]["perplexity"]
            if base_ppl > 0 and not np.isinf(base_ppl) and not np.isinf(abl_ppl):
                matrix[i, j] = np.log(abl_ppl / base_ppl)
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
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    from scipy.stats import spearmanr

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )

    n_layers = len(get_model_layers(model))
    ffn_dim = get_ffn_dim(model, n_layers)
    n_neurons = n_layers * ffn_dim
    print(f"  n_layers={n_layers}, ffn_dim={ffn_dim}, n_neurons={n_neurons}")

    with open(args.stimuli) as f:
        stimuli = [json.loads(l) for l in f]
    conditions = sorted(set(s["condition"] for s in stimuli))
    print(f"  {len(stimuli)} stimuli, {len(conditions)} conditions")

    brain_data = np.load(args.brain_rdm, allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])
    brain_order = [brain_conds.index(c) for c in conditions]
    brain_rdm = brain_rdm[np.ix_(brain_order, brain_order)]

    # Step 1: Attribution (batched)
    print("\n" + "=" * 70)
    print("STEP 1: Batched gradient × activation attribution")
    print("=" * 70)
    model.train()
    cond_importance, cond_counts = compute_all_attributions_batched(
        model, tokenizer, stimuli, conditions,
        n_layers, ffn_dim, device, args.max_length, args.batch_size
    )

    # Step 2: Selectivity
    print("\nSTEP 2: Selectivity")
    selectivity = compute_selectivity(cond_importance, conditions)
    for c in conditions:
        top5k = np.sort(selectivity[c])[-5000:]
        print(f"  {c:>20s}: top-5K sel mean={top5k.mean():.4f}")

    # Save attribution
    os.makedirs(args.output_dir, exist_ok=True)
    attr_data = {}
    for c in conditions:
        attr_data[f"{c}_importance"] = cond_importance[c].astype(np.float32)
        attr_data[f"{c}_selectivity"] = selectivity[c].astype(np.float32)
    np.savez_compressed(
        os.path.join(args.output_dir, f"{args.model_short}_cognitive_attribution.npz"),
        **attr_data
    )
    print(f"  Saved attribution NPZ  [{time.time()-t0:.0f}s elapsed]")

    # Step 3: Baseline PPL
    print("\nSTEP 3: Baseline PPL (batched)")
    model.eval()
    baseline = measure_ppl_batched(
        model, tokenizer, stimuli, conditions, device, args.max_length, args.batch_size * 2
    )
    for c in conditions:
        print(f"  {c:>20s}: PPL={baseline[c]['perplexity']:.2f}")

    # Step 4: Ablation (14 × 14)
    print("\nSTEP 4: Causal ablation (14 conditions × batched PPL)")
    ablation_results = ablate_and_measure(
        model, tokenizer, stimuli, conditions,
        selectivity, args.n_ablate, device, args.max_length, args.batch_size * 2
    )

    # Step 5: Coupling matrix vs brain RDM
    print("\n" + "=" * 70)
    print("STEP 5: Causal coupling matrix vs brain RDM")
    print("=" * 70)

    coupling = build_coupling_matrix(baseline, ablation_results, conditions)

    # Print matrix
    print(f"\n  Coupling matrix (log PPL ratio when ablating row → measuring col):")
    print(f"  {'':>18s}", end="")
    for c in conditions:
        print(f" {c[:7]:>7s}", end="")
    print()
    for i, ci in enumerate(conditions):
        print(f"  {ci:>18s}", end="")
        for j in range(len(conditions)):
            print(f" {coupling[i,j]:>+7.3f}", end="")
        print()

    n = len(conditions)
    triu = np.triu_indices(n, k=1)
    coupling_sym = (coupling + coupling.T) / 2
    brain_vec = brain_rdm[triu]
    coupling_vec = coupling_sym[triu]

    rho_sym, p_sym = spearmanr(brain_vec, coupling_vec)
    print(f"\n  Brain RDM vs symmetric causal coupling:")
    print(f"    Spearman ρ = {rho_sym:+.4f}, p = {p_sym:.4f}")
    print(f"    (expect NEGATIVE: high brain distance → low causal coupling)")

    # Row-wise
    row_rhos = []
    for i, c in enumerate(conditions):
        mask = np.ones(n, dtype=bool); mask[i] = False
        r, p = spearmanr(brain_rdm[i, mask], coupling[i, mask])
        row_rhos.append(r)
        print(f"    {c:>20s}: row ρ={r:+.4f}")
    print(f"    Mean row-wise ρ: {np.mean(row_rhos):+.4f}")

    # Permutation null
    rng = np.random.default_rng(2026)
    null_rhos = np.array([
        float(spearmanr(brain_rdm[np.ix_(p, p)][triu], coupling_vec)[0])
        for p in (rng.permutation(n) for _ in range(10000))
    ])
    p_perm = float((np.sum(null_rhos <= rho_sym) + 1) / (len(null_rhos) + 1))
    print(f"  Permutation null: mean={np.mean(null_rhos):+.4f}, p={p_perm:.4f}")

    elapsed = time.time() - t0
    result = {
        "model": args.model_short,
        "n_neurons": n_neurons,
        "n_ablate": args.n_ablate,
        "batch_size": args.batch_size,
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
        },
        "elapsed_seconds": elapsed,
    }
    out_path = os.path.join(args.output_dir, f"{args.model_short}_brain_causal_coupling.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
