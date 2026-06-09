#!/usr/bin/env python3
"""
BrainCog-14: Evaluate a language model's social-emotional representational
geometry against the human brain.

This script is self-contained. It loads a HuggingFace causal language model,
runs the 14-condition stimulus set, extracts hidden-state representations,
builds a representational dissimilarity matrix (RDM), and compares it to the
brain-derived RDM using Spearman correlation.

Usage:
    python evaluate.py --model_path /path/to/model --layer 27

    # Sweep all layers to find the peak:
    python evaluate.py --model_path /path/to/model --layer all

    # Use a specific device:
    python evaluate.py --model_path /path/to/model --layer 27 --device cuda:1

Dependencies: torch, transformers, numpy, scipy

Reference results (Spearman rho on 91 upper-triangle pairs):
    Qwen2.5-7B-Instruct          0.739  (layer 27)
    Meta-Llama-3.1-8B-Instruct   0.727  (layer 31)
    Mistral-7B-Instruct-v0.3     0.730  (layer 14)
    gemma-2-9b-it                 0.735  (layer 21)
    Chance 95th percentile:       0.250

Citation:
    Zhang et al. (2026). The Brain as a Reference Frame for Language Models:
    Social-Emotional Representational Geometry. Preprint.

License: MIT (code), CC-BY-4.0 (data)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Configuration (locked benchmark recipe)
# ---------------------------------------------------------------------------
POOLING = "mean_all"          # mean over all non-pad tokens
CENTERING = True              # subtract condition-mean across the 14 conditions
DISTANCE = "cosine"           # 1 - cosine similarity
COMPARISON = "spearman"       # Spearman rank correlation on upper triangle
N_PERMUTATIONS = 5000         # condition-label shuffle permutation test
CHANCE_95TH = 0.25            # 95th percentile of the null distribution
MAX_LENGTH = 512              # max tokens per stimulus

AFFECTIVE = ["anger", "disgust", "fear", "happiness", "sadness", "valence"]
SOCIAL = ["belief", "empathy", "intention", "judgment", "mentalizing",
          "moral", "self_referential", "theory_of_mind"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_stimuli(jsonl_path: str | Path) -> list[dict]:
    """Load stimuli from JSONL file."""
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_brain_rdm(npz_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load the brain RDM and condition labels."""
    d = np.load(npz_path, allow_pickle=True)
    rdm = d["rdm"].astype(np.float64)
    conditions = list(d["conditions"])
    return rdm, conditions


@torch.no_grad()
def extract_mean_all(
    model,
    tokenizer,
    text: str,
    device: str,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Extract mean-pooled hidden states across all layers for one stimulus.

    Returns: numpy array of shape [n_layers+1, hidden_dim] (float64).
             Index 0 = embedding layer, index i = transformer layer i.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)

    # hidden_states is a tuple of (n_layers+1) tensors, each [1, seq_len, hidden]
    # Stack into [n_layers+1, seq_len, hidden], then mean over seq_len dim
    layers = torch.stack([h[0].float() for h in out.hidden_states], dim=0)
    mean_all = layers.mean(dim=1)  # [n_layers+1, hidden]
    return mean_all.cpu().numpy().astype(np.float64)


def center_across_conditions(condition_means: np.ndarray) -> np.ndarray:
    """Subtract the grand mean across conditions (axis 0).

    Input: [n_conditions, hidden_dim]
    Output: [n_conditions, hidden_dim], zero-mean across conditions.
    """
    return condition_means - condition_means.mean(axis=0, keepdims=True)


def cosine_rdm(X: np.ndarray) -> np.ndarray:
    """Compute cosine-distance RDM: 1 - cosine_similarity.

    Input: [n_conditions, hidden_dim]
    Output: [n_conditions, n_conditions] symmetric distance matrix.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    return 1.0 - Xn @ Xn.T


def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    """Extract the strict upper triangle of a square matrix."""
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    return rdm[iu]


def permutation_test(
    brain_ut: np.ndarray,
    model_rdm: np.ndarray,
    n_perms: int = N_PERMUTATIONS,
    seed: int = 42,
) -> tuple[float, float]:
    """Permutation test: shuffle condition labels, recompute Spearman rho.

    Returns: (rho, p_value)
    """
    n = model_rdm.shape[0]
    rho_obs, _ = spearmanr(brain_ut, upper_triangle(model_rdm))

    rng = np.random.default_rng(seed)
    n_exceed = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        shuffled = model_rdm[np.ix_(perm, perm)]
        rho_null, _ = spearmanr(brain_ut, upper_triangle(shuffled))
        if rho_null >= rho_obs:
            n_exceed += 1

    p_value = (n_exceed + 1) / (n_perms + 1)
    return rho_obs, p_value


def evaluate_layer(
    per_stim_acts: np.ndarray,
    stim_conditions: list[str],
    conditions_sorted: list[str],
    layer_idx: int,
    brain_rdm: np.ndarray,
    brain_conditions: list[str],
    run_permutation: bool = True,
) -> dict:
    """Evaluate one layer: condition-mean -> center -> cosine RDM -> Spearman.

    Args:
        per_stim_acts: [n_stimuli, n_layers+1, hidden_dim]
        stim_conditions: condition label for each stimulus
        conditions_sorted: the 14 condition names in sorted order
        layer_idx: which layer to evaluate
        brain_rdm: [14, 14] brain RDM
        brain_conditions: condition order in the brain RDM
        run_permutation: whether to run the full permutation test

    Returns: dict with rho, p, pass/fail, etc.
    """
    # Build condition-mean activations at this layer
    cond_to_stims = {
        c: [i for i, cc in enumerate(stim_conditions) if cc == c]
        for c in conditions_sorted
    }
    cond_means = np.stack([
        per_stim_acts[cond_to_stims[c], layer_idx, :].mean(axis=0)
        for c in conditions_sorted
    ])  # [14, hidden_dim]

    # Center and compute cosine RDM
    cond_means = center_across_conditions(cond_means)
    model_rdm = cosine_rdm(cond_means)

    # Reorder brain RDM to match condition order
    order = [brain_conditions.index(c) for c in conditions_sorted]
    brain_reordered = brain_rdm[np.ix_(order, order)]

    brain_ut = upper_triangle(brain_reordered)
    model_ut = upper_triangle(model_rdm)

    rho, _ = spearmanr(brain_ut, model_ut)

    result = {
        "layer": layer_idx,
        "rho": round(float(rho), 4),
        "pass": rho > CHANCE_95TH,
    }

    if run_permutation:
        _, p_value = permutation_test(brain_ut, model_rdm)
        result["p_value"] = round(float(p_value), 6)
        result["significant"] = p_value < 0.05

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BrainCog-14: Evaluate brain-LLM representational alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a single layer:
    python evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --layer 27

    # Sweep all layers to find the peak:
    python evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --layer all

    # Skip permutation test for fast screening:
    python evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --layer 27 --no-permutation
""",
    )
    parser.add_argument(
        "--model_path", required=True,
        help="HuggingFace model path or local directory",
    )
    parser.add_argument(
        "--layer", required=True,
        help="Layer index to evaluate (0-based, 0=embedding), or 'all' to sweep",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--no-permutation", action="store_true",
        help="Skip permutation test (faster, no p-value)",
    )
    parser.add_argument(
        "--stimuli", default=None,
        help="Path to stimuli JSONL (default: braincog14_stimuli.jsonl in same dir)",
    )
    parser.add_argument(
        "--brain-rdm", default=None,
        help="Path to brain RDM NPZ (default: braincog14_brain_rdm.npz in same dir)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--max-length", type=int, default=MAX_LENGTH,
        help=f"Max tokens per stimulus (default: {MAX_LENGTH})",
    )
    args = parser.parse_args()

    # Resolve data paths (default: same directory as this script)
    script_dir = Path(__file__).resolve().parent
    stimuli_path = Path(args.stimuli) if args.stimuli else script_dir / "braincog14_stimuli.jsonl"
    brain_rdm_path = Path(args.brain_rdm) if args.brain_rdm else script_dir / "braincog14_brain_rdm.npz"

    if not stimuli_path.exists():
        print(f"ERROR: Stimuli file not found: {stimuli_path}", file=sys.stderr)
        sys.exit(1)
    if not brain_rdm_path.exists():
        print(f"ERROR: Brain RDM file not found: {brain_rdm_path}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("BrainCog-14: Brain-derived benchmark for social-emotional geometry")
    print("=" * 70)

    print(f"\nLoading stimuli from {stimuli_path} ...")
    stimuli = load_stimuli(stimuli_path)
    conditions_sorted = sorted(set(s["condition"] for s in stimuli))
    assert len(conditions_sorted) == 14, f"Expected 14 conditions, got {len(conditions_sorted)}"
    print(f"  {len(stimuli)} stimuli across {len(conditions_sorted)} conditions")

    print(f"Loading brain RDM from {brain_rdm_path} ...")
    brain_rdm, brain_conditions = load_brain_rdm(brain_rdm_path)
    print(f"  {brain_rdm.shape[0]}x{brain_rdm.shape[1]} RDM, "
          f"{len(brain_conditions)} conditions")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {args.model_path}")
    print(f"  Device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Probe to get layer count
    probe = extract_mean_all(model, tokenizer, stimuli[0]["text"], args.device,
                             args.max_length)
    n_layers_total = probe.shape[0]  # includes embedding layer
    hidden_dim = probe.shape[1]
    print(f"  {n_layers_total} layers (embedding + {n_layers_total - 1} transformer), "
          f"hidden_dim={hidden_dim}")

    # Parse layer argument
    if args.layer.lower() == "all":
        layers_to_eval = list(range(n_layers_total))
    else:
        layer_idx = int(args.layer)
        if layer_idx < 0 or layer_idx >= n_layers_total:
            print(f"ERROR: Layer {layer_idx} out of range [0, {n_layers_total - 1}]",
                  file=sys.stderr)
            sys.exit(1)
        layers_to_eval = [layer_idx]

    # -----------------------------------------------------------------------
    # Extract hidden states
    # -----------------------------------------------------------------------
    print(f"\nExtracting hidden states for {len(stimuli)} stimuli ...")
    t0 = time.time()

    per_stim_acts = np.empty((len(stimuli), n_layers_total, hidden_dim),
                             dtype=np.float64)
    stim_conditions = []

    for i, item in enumerate(stimuli):
        per_stim_acts[i] = extract_mean_all(
            model, tokenizer, item["text"], args.device, args.max_length,
        )
        stim_conditions.append(item["condition"])
        if (i + 1) % 100 == 0 or i + 1 == len(stimuli):
            elapsed = time.time() - t0
            print(f"  [{i + 1:4d}/{len(stimuli)}]  {elapsed:.0f}s elapsed")

    print(f"  Extraction complete ({time.time() - t0:.0f}s)")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    run_perm = not args.no_permutation
    is_sweep = len(layers_to_eval) > 1

    # For a full sweep, skip permutation on individual layers (too slow),
    # run it only on the peak layer at the end
    results = []
    print(f"\nEvaluating {'all layers' if is_sweep else f'layer {layers_to_eval[0]}'} ...")

    for layer_idx in layers_to_eval:
        res = evaluate_layer(
            per_stim_acts, stim_conditions, conditions_sorted,
            layer_idx, brain_rdm, brain_conditions,
            run_permutation=(run_perm and not is_sweep),
        )
        results.append(res)
        status = "PASS" if res["pass"] else "FAIL"
        rho_str = f"{res['rho']:+.4f}"
        p_str = f"  p={res.get('p_value', 'N/A')}" if "p_value" in res else ""
        if is_sweep:
            print(f"  Layer {layer_idx:3d}: rho={rho_str}  [{status}]")
        else:
            print(f"  Layer {layer_idx}: rho={rho_str}{p_str}  [{status}]")

    # For a sweep, identify peak and run permutation on it
    if is_sweep:
        peak = max(results, key=lambda r: r["rho"])
        print(f"\n  Peak layer: {peak['layer']} (rho={peak['rho']:+.4f})")
        if run_perm:
            print(f"  Running permutation test on peak layer ({N_PERMUTATIONS} permutations) ...")
            peak_full = evaluate_layer(
                per_stim_acts, stim_conditions, conditions_sorted,
                peak["layer"], brain_rdm, brain_conditions,
                run_permutation=True,
            )
            peak.update(peak_full)
            print(f"  Peak: rho={peak['rho']:+.4f}, p={peak['p_value']:.6f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    if is_sweep:
        best = max(results, key=lambda r: r["rho"])
    else:
        best = results[0]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Model:            {args.model_path}")
    print(f"  Best layer:       {best['layer']}")
    print(f"  Spearman rho:     {best['rho']:+.4f}")
    if "p_value" in best:
        print(f"  Permutation p:    {best['p_value']:.6f}")
        print(f"  Significant:      {'Yes' if best['significant'] else 'No'} (alpha=0.05)")
    print(f"  Chance (95th):    {CHANCE_95TH}")
    print(f"  Verdict:          {'PASS' if best['pass'] else 'FAIL'}")
    print()
    print(f"  Reference: Qwen2.5-7B=0.739, Llama-3.1-8B=0.727, "
          f"Mistral-7B=0.730, Gemma-2-9B=0.735")
    print(f"  Config:    {POOLING} | centered={CENTERING} | "
          f"{DISTANCE} distance | {COMPARISON} comparison")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "benchmark": "BrainCog-14",
        "model": args.model_path,
        "config": {
            "pooling": POOLING,
            "centering": CENTERING,
            "distance": DISTANCE,
            "comparison": COMPARISON,
            "max_length": args.max_length,
            "n_permutations": N_PERMUTATIONS if run_perm else 0,
        },
        "best": best,
        "all_layers": results if is_sweep else None,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        # Print JSON to stdout for piping
        print(f"\nJSON output (pass --output <path> to save to file):")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
