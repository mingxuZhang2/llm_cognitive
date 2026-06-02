#!/usr/bin/env python3
"""
Clinical double-dissociation experiment: psychopathy vs autism analog in LLMs.

Scientific motivation:
  - Shamay-Tsoory 2009 (Brain): ventral PFC lesion -> impaired emotion, spared ToM
  - Baron-Cohen 1995: autism -> impaired ToM, spared basic emotion
  - Our analog: ablate emotion subspace -> emotion tasks collapse, social spared;
                ablate social subspace -> reverse

Pipeline:
  1. Compute gradient x activation importance for all 14 conditions (reuses
     machinery from brain_causal_coupling.py)
  2. Pool importance across 6 affective conditions -> rank -> top-5000 "emotion neurons"
  3. Pool importance across 8 social conditions -> rank -> top-5000 "social neurons"
  4. Three ablation conditions:
     - PSYCHOPATHY: ablate emotion neuron set -> measure log(PPL_abl/PPL_base) on all 14
     - AUTISM: ablate social neuron set -> measure same
     - RANDOM: ablate 5000 random neurons x 50 repeats -> null distribution
  5. Double-dissociation test: cross-selectivity ratios

Output: experiments/results/clinical_dissociation/clinical_dissociation.json
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

# ---------------------------------------------------------------------------
# Reused core functions from brain_causal_coupling.py
# ---------------------------------------------------------------------------

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
    """Batched gradient x activation attribution (identical to brain_causal_coupling.py)."""
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

        attn_mask = inputs["attention_mask"]
        for lidx in range(n_layers):
            if lidx not in activations or activations[lidx].grad is None:
                continue
            act = activations[lidx].detach()
            grad = activations[lidx].grad.detach()
            imp = (act * grad).abs()

            mask_3d = attn_mask.unsqueeze(-1).to(imp.dtype)
            imp_masked = imp * mask_3d
            valid_counts = mask_3d.sum(dim=1).clamp(min=1)
            imp_per_sample = imp_masked.sum(dim=1) / valid_counts

            start, end = lidx * ffn_dim, (lidx + 1) * ffn_dim
            imp_np = imp_per_sample.cpu().float().numpy()

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


def measure_ppl_batched(model, tokenizer, stimuli, conditions,
                        device="cuda", max_length=256, batch_size=32):
    """Batched PPL measurement per condition (identical to brain_causal_coupling.py)."""
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


# ---------------------------------------------------------------------------
# Clinical dissociation specific logic
# ---------------------------------------------------------------------------

AFFECTIVE = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
SOCIAL = ["belief", "mentalizing", "intention", "theory_of_mind",
          "empathy", "self_referential", "judgment", "moral"]


def pool_importance(cond_importance, condition_group):
    """Mean importance across a group of conditions -> single neuron-level vector."""
    vecs = [cond_importance[c] for c in condition_group if c in cond_importance]
    if not vecs:
        raise ValueError(f"No conditions found in group: {condition_group}")
    return np.mean(vecs, axis=0)


def select_top_neurons(importance_vec, n_top):
    """Return indices of the top-n neurons by importance."""
    return np.argsort(-importance_vec)[:n_top]


def install_ablation_hooks(model, neuron_indices, ffn_dim, device):
    """Install forward hooks that zero out the specified neurons.

    Returns a list of hook handles (caller must remove them after measurement).
    """
    layers = get_model_layers(model)
    # Group neurons by layer
    layer_neurons = {}
    for n in neuron_indices:
        l = int(n // ffn_dim)
        pos = int(n % ffn_dim)
        if l not in layer_neurons:
            layer_neurons[l] = []
        layer_neurons[l].append(pos)

    hooks = []
    for lidx, positions in layer_neurons.items():
        mlp = layers[lidx].mlp
        target_module = mlp.gate_proj if hasattr(mlp, "gate_proj") else mlp.dense_h_to_4h
        pos_tensor = torch.tensor(positions, device=device)

        def make_ablation_hook(pos_t):
            def hook(module, inp, out):
                out[:, :, pos_t] = 0.0
                return out
            return hook
        hooks.append(target_module.register_forward_hook(make_ablation_hook(pos_tensor)))

    return hooks


def compute_log_ppl_ratio(baseline, ablated, conditions):
    """log(PPL_ablated / PPL_baseline) per condition."""
    ratios = {}
    for c in conditions:
        base_ppl = baseline[c]["perplexity"]
        abl_ppl = ablated[c]["perplexity"]
        if base_ppl > 0 and not np.isinf(base_ppl) and not np.isinf(abl_ppl):
            ratios[c] = float(np.log(abl_ppl / base_ppl))
        else:
            ratios[c] = float("nan")
    return ratios


def compute_selectivity_ratio(log_ratios, target_group, other_group):
    """mean(effect on target) / mean(effect on other).

    Both values should be positive (ablation raises PPL). The ratio > 1 means
    the ablation hits the target group harder than the other group.
    """
    target_vals = [log_ratios[c] for c in target_group
                   if c in log_ratios and not np.isnan(log_ratios[c])]
    other_vals = [log_ratios[c] for c in other_group
                  if c in log_ratios and not np.isnan(log_ratios[c])]
    if not target_vals or not other_vals:
        return float("nan")
    mean_target = np.mean(target_vals)
    mean_other = np.mean(other_vals)
    if mean_other == 0:
        return float("inf") if mean_target > 0 else float("nan")
    return float(mean_target / mean_other)


def ablate_and_measure_set(model, tokenizer, stimuli, conditions,
                           neuron_indices, ffn_dim, device,
                           max_length=256, batch_size=32):
    """Ablate a fixed set of neurons and measure PPL on all conditions."""
    hooks = install_ablation_hooks(model, neuron_indices, ffn_dim, device)
    ppl = measure_ppl_batched(model, tokenizer, stimuli, conditions,
                              device, max_length, batch_size)
    for h in hooks:
        h.remove()
    return ppl


def print_summary_table(all_results):
    """Print a clear summary table across all models."""
    print("\n" + "=" * 90)
    print("CLINICAL DOUBLE-DISSOCIATION SUMMARY")
    print("=" * 90)

    # Header
    print(f"\n{'Model':<35s} {'Ablation':<14s} "
          f"{'Aff effect':>10s} {'Soc effect':>10s} {'Selectivity':>12s} {'p(rand)':>8s}")
    print("-" * 90)

    for model_name, res in all_results.items():
        for abl_type in ["psychopathy", "autism"]:
            abl = res[abl_type]
            if abl_type == "psychopathy":
                sel = abl["emotion_selectivity"]
                tgt_eff = abl["mean_affective_effect"]
                oth_eff = abl["mean_social_effect"]
            else:
                sel = abl["social_selectivity"]
                tgt_eff = abl["mean_social_effect"]
                oth_eff = abl["mean_affective_effect"]
            p_val = abl.get("p_vs_random", float("nan"))
            print(f"{model_name:<35s} {abl_type.upper():<14s} "
                  f"{tgt_eff:>+10.4f} {oth_eff:>+10.4f} {sel:>12.3f} {p_val:>8.4f}")

        dd = res["double_dissociation"]
        tag = "YES" if dd["confirmed"] else "NO"
        print(f"  -> Double dissociation: {tag}  "
              f"(emotion_sel={dd['emotion_selectivity']:.3f}, "
              f"social_sel={dd['social_selectivity']:.3f})")
        print()

    # Per-condition detail
    print("\nPER-CONDITION log(PPL_abl/PPL_base):")
    print(f"{'Model':<28s} {'Ablation':<14s}", end="")
    # Use the conditions from the first model
    first_res = next(iter(all_results.values()))
    conditions = first_res["conditions"]
    for c in conditions:
        print(f" {c[:6]:>6s}", end="")
    print()
    print("-" * (42 + 7 * len(conditions)))

    for model_name, res in all_results.items():
        for abl_type in ["psychopathy", "autism"]:
            abl = res[abl_type]
            print(f"{model_name:<28s} {abl_type.upper():<14s}", end="")
            for c in conditions:
                val = abl["log_ppl_ratios"].get(c, float("nan"))
                print(f" {val:>+6.3f}", end="")
            print()


def main():
    ap = argparse.ArgumentParser(
        description="Clinical double-dissociation: psychopathy vs autism analog in LLMs")
    ap.add_argument("--model_path", required=True, help="HF model path or local dir")
    ap.add_argument("--model_short", required=True, help="Short model name for output")
    ap.add_argument("--stimuli", required=True, help="Path to rsa_stimuli.jsonl")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--n_ablate", type=int, default=5000,
                    help="Number of neurons to ablate per set (default: 5000)")
    ap.add_argument("--n_random", type=int, default=50,
                    help="Number of random ablation repeats (default: 50)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026,
                    help="Random seed for random ablation control")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), "total_mem")
              else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Load model ----
    print(f"\nLoading model: {args.model_path}")
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

    # ---- Load stimuli ----
    with open(args.stimuli) as f:
        stimuli = [json.loads(l) for l in f]
    conditions = sorted(set(s["condition"] for s in stimuli))
    print(f"  {len(stimuli)} stimuli, {len(conditions)} conditions: {conditions}")

    # Verify condition groups
    for c in AFFECTIVE:
        assert c in conditions, f"Affective condition '{c}' not in stimuli"
    for c in SOCIAL:
        assert c in conditions, f"Social condition '{c}' not in stimuli"

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================================================================
    # STEP 1: Attribution
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Gradient x activation attribution")
    print("=" * 70)
    model.train()  # need gradients for attribution
    cond_importance, cond_counts = compute_all_attributions_batched(
        model, tokenizer, stimuli, conditions,
        n_layers, ffn_dim, device, args.max_length, args.batch_size
    )
    attr_time = time.time() - t0
    print(f"  Attribution done in {attr_time:.0f}s")

    # ==================================================================
    # STEP 2: Pool importance -> emotion & social neuron sets
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Pool importance into emotion & social neuron sets")
    print("=" * 70)

    emotion_imp = pool_importance(cond_importance, AFFECTIVE)
    social_imp = pool_importance(cond_importance, SOCIAL)

    emotion_neurons = select_top_neurons(emotion_imp, args.n_ablate)
    social_neurons = select_top_neurons(social_imp, args.n_ablate)

    overlap = len(set(emotion_neurons.tolist()) & set(social_neurons.tolist()))
    print(f"  Emotion neuron set: top-{args.n_ablate} (mean imp={emotion_imp[emotion_neurons].mean():.6f})")
    print(f"  Social neuron set:  top-{args.n_ablate} (mean imp={social_imp[social_neurons].mean():.6f})")
    print(f"  Overlap: {overlap} neurons ({100*overlap/args.n_ablate:.1f}%)")

    # ==================================================================
    # STEP 3: Baseline PPL
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Baseline PPL")
    print("=" * 70)
    model.eval()
    baseline = measure_ppl_batched(
        model, tokenizer, stimuli, conditions, device,
        args.max_length, args.batch_size * 2
    )
    for c in conditions:
        print(f"  {c:>20s}: PPL={baseline[c]['perplexity']:>8.2f}")

    # ==================================================================
    # STEP 4a: PSYCHOPATHY ablation (emotion neurons)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4a: PSYCHOPATHY ablation (ablate emotion neuron set)")
    print("=" * 70)
    t_psych = time.time()
    psych_ppl = ablate_and_measure_set(
        model, tokenizer, stimuli, conditions,
        emotion_neurons, ffn_dim, device, args.max_length, args.batch_size * 2
    )
    psych_ratios = compute_log_ppl_ratio(baseline, psych_ppl, conditions)
    psych_emo_sel = compute_selectivity_ratio(psych_ratios, AFFECTIVE, SOCIAL)
    print(f"  Elapsed: {time.time()-t_psych:.0f}s")
    for c in conditions:
        tag = "AFF" if c in AFFECTIVE else "SOC"
        print(f"  {c:>20s} [{tag}]: log(PPL_abl/base)={psych_ratios[c]:>+.4f}")
    mean_aff_psych = float(np.mean([psych_ratios[c] for c in AFFECTIVE]))
    mean_soc_psych = float(np.mean([psych_ratios[c] for c in SOCIAL]))
    print(f"  Mean affective effect: {mean_aff_psych:+.4f}")
    print(f"  Mean social effect:    {mean_soc_psych:+.4f}")
    print(f"  Emotion selectivity:   {psych_emo_sel:.3f}")

    # ==================================================================
    # STEP 4b: AUTISM ablation (social neurons)
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 4b: AUTISM ablation (ablate social neuron set)")
    print("=" * 70)
    t_aut = time.time()
    autism_ppl = ablate_and_measure_set(
        model, tokenizer, stimuli, conditions,
        social_neurons, ffn_dim, device, args.max_length, args.batch_size * 2
    )
    autism_ratios = compute_log_ppl_ratio(baseline, autism_ppl, conditions)
    autism_soc_sel = compute_selectivity_ratio(autism_ratios, SOCIAL, AFFECTIVE)
    print(f"  Elapsed: {time.time()-t_aut:.0f}s")
    for c in conditions:
        tag = "AFF" if c in AFFECTIVE else "SOC"
        print(f"  {c:>20s} [{tag}]: log(PPL_abl/base)={autism_ratios[c]:>+.4f}")
    mean_aff_aut = float(np.mean([autism_ratios[c] for c in AFFECTIVE]))
    mean_soc_aut = float(np.mean([autism_ratios[c] for c in SOCIAL]))
    print(f"  Mean affective effect: {mean_aff_aut:+.4f}")
    print(f"  Mean social effect:    {mean_soc_aut:+.4f}")
    print(f"  Social selectivity:    {autism_soc_sel:.3f}")

    # ==================================================================
    # STEP 4c: RANDOM ablation (null distribution)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"STEP 4c: RANDOM ablation (n={args.n_random} repeats, {args.n_ablate} neurons each)")
    print("=" * 70)
    rng = np.random.default_rng(args.seed)
    random_emo_sels = []
    random_soc_sels = []
    random_all_ratios = []

    for rep in range(args.n_random):
        t_rep = time.time()
        rand_neurons = rng.choice(n_neurons, size=args.n_ablate, replace=False)
        rand_ppl = ablate_and_measure_set(
            model, tokenizer, stimuli, conditions,
            rand_neurons, ffn_dim, device, args.max_length, args.batch_size * 2
        )
        rand_ratios = compute_log_ppl_ratio(baseline, rand_ppl, conditions)
        emo_sel = compute_selectivity_ratio(rand_ratios, AFFECTIVE, SOCIAL)
        soc_sel = compute_selectivity_ratio(rand_ratios, SOCIAL, AFFECTIVE)
        random_emo_sels.append(emo_sel)
        random_soc_sels.append(soc_sel)
        random_all_ratios.append(rand_ratios)

        if (rep + 1) % 5 == 0 or rep == 0 or rep == args.n_random - 1:
            print(f"  rep {rep+1:>3d}/{args.n_random}: "
                  f"emo_sel={emo_sel:.3f}, soc_sel={soc_sel:.3f} "
                  f"({time.time()-t_rep:.1f}s)")

    random_emo_sels = np.array(random_emo_sels)
    random_soc_sels = np.array(random_soc_sels)

    # p-value: fraction of random ablations with selectivity >= observed
    p_psych = float((np.sum(random_emo_sels >= psych_emo_sel) + 1) / (args.n_random + 1))
    p_autism = float((np.sum(random_soc_sels >= autism_soc_sel) + 1) / (args.n_random + 1))

    print(f"\n  Random null emotion selectivity: "
          f"mean={np.mean(random_emo_sels):.3f}, std={np.std(random_emo_sels):.3f}")
    print(f"  Random null social selectivity:  "
          f"mean={np.mean(random_soc_sels):.3f}, std={np.std(random_soc_sels):.3f}")
    print(f"  PSYCHOPATHY emotion_sel={psych_emo_sel:.3f}, p={p_psych:.4f}")
    print(f"  AUTISM social_sel={autism_soc_sel:.3f}, p={p_autism:.4f}")

    # ==================================================================
    # STEP 5: Double-dissociation test
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Double-dissociation test")
    print("=" * 70)

    dd_confirmed = (psych_emo_sel > 1.0) and (autism_soc_sel > 1.0)
    print(f"  Psychopathy ablation -> emotion selectivity = {psych_emo_sel:.3f} (>1? {psych_emo_sel>1})")
    print(f"  Autism ablation      -> social selectivity  = {autism_soc_sel:.3f} (>1? {autism_soc_sel>1})")
    print(f"  DOUBLE DISSOCIATION: {'CONFIRMED' if dd_confirmed else 'NOT CONFIRMED'}")

    elapsed = time.time() - t0

    # ==================================================================
    # Save results
    # ==================================================================
    result = {
        "model": args.model_short,
        "n_neurons": n_neurons,
        "n_layers": n_layers,
        "ffn_dim": ffn_dim,
        "n_ablate": args.n_ablate,
        "n_random": args.n_random,
        "seed": args.seed,
        "conditions": conditions,
        "affective_conditions": AFFECTIVE,
        "social_conditions": SOCIAL,
        "neuron_set_overlap": overlap,
        "neuron_set_overlap_pct": float(100 * overlap / args.n_ablate),
        "baseline": baseline,
        "psychopathy": {
            "description": "Ablate emotion neuron set (top-N by pooled affective importance)",
            "ablated_ppl": {c: psych_ppl[c] for c in conditions},
            "log_ppl_ratios": psych_ratios,
            "mean_affective_effect": mean_aff_psych,
            "mean_social_effect": mean_soc_psych,
            "emotion_selectivity": psych_emo_sel,
            "p_vs_random": p_psych,
        },
        "autism": {
            "description": "Ablate social neuron set (top-N by pooled social importance)",
            "ablated_ppl": {c: autism_ppl[c] for c in conditions},
            "log_ppl_ratios": autism_ratios,
            "mean_affective_effect": mean_aff_aut,
            "mean_social_effect": mean_soc_aut,
            "social_selectivity": autism_soc_sel,
            "p_vs_random": p_autism,
        },
        "random_null": {
            "description": f"{args.n_random} repeats of ablating {args.n_ablate} random neurons",
            "emotion_selectivity_mean": float(np.mean(random_emo_sels)),
            "emotion_selectivity_std": float(np.std(random_emo_sels)),
            "social_selectivity_mean": float(np.mean(random_soc_sels)),
            "social_selectivity_std": float(np.std(random_soc_sels)),
            "emotion_selectivity_all": [float(x) for x in random_emo_sels],
            "social_selectivity_all": [float(x) for x in random_soc_sels],
            # Per-condition mean effect across random repeats
            "mean_log_ppl_ratios": {
                c: float(np.mean([r[c] for r in random_all_ratios
                                  if not np.isnan(r.get(c, float("nan")))]))
                for c in conditions
            },
        },
        "double_dissociation": {
            "emotion_selectivity": psych_emo_sel,
            "social_selectivity": autism_soc_sel,
            "confirmed": dd_confirmed,
            "p_psychopathy": p_psych,
            "p_autism": p_autism,
        },
        "elapsed_seconds": elapsed,
    }

    out_path = os.path.join(args.output_dir,
                            f"{args.model_short}_clinical_dissociation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return result


def aggregate_and_print(output_dir):
    """If run after all 4 models: aggregate and print cross-model summary."""
    import glob
    files = sorted(glob.glob(os.path.join(output_dir, "*_clinical_dissociation.json")))
    if not files:
        return
    all_results = {}
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        all_results[d["model"]] = d

    if len(all_results) > 1:
        print_summary_table(all_results)

        # Cross-model summary
        n_confirmed = sum(1 for r in all_results.values()
                          if r["double_dissociation"]["confirmed"])
        print(f"\nDouble dissociation confirmed in {n_confirmed}/{len(all_results)} models")

        # Save aggregate
        agg_path = os.path.join(output_dir, "clinical_dissociation.json")
        agg = {
            "models": list(all_results.keys()),
            "per_model": {
                name: {
                    "double_dissociation": r["double_dissociation"],
                    "neuron_set_overlap": r["neuron_set_overlap"],
                    "neuron_set_overlap_pct": r["neuron_set_overlap_pct"],
                    "psychopathy_emotion_selectivity": r["psychopathy"]["emotion_selectivity"],
                    "autism_social_selectivity": r["autism"]["social_selectivity"],
                    "psychopathy_p": r["psychopathy"]["p_vs_random"],
                    "autism_p": r["autism"]["p_vs_random"],
                }
                for name, r in all_results.items()
            },
            "n_confirmed": n_confirmed,
            "all_confirmed": n_confirmed == len(all_results),
        }
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"Aggregate saved to {agg_path}")


if __name__ == "__main__":
    result = main()

    # Try to aggregate if other models already done
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args, _ = ap.parse_known_args()
    aggregate_and_print(args.output_dir)
