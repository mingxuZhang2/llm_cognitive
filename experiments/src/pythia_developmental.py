#!/usr/bin/env python3
"""
Developmental emergence of brain-like cognitive structure during LLM training,
tracked via Pythia-2.8B-deduped checkpoints.

Neuroscience prediction: basic emotions emerge from birth/early infancy; Theory of
Mind appears ~age 4 (Wimmer & Perner 1983). If LLMs recapitulate this developmental
trajectory, emotion-related representational structure should appear BEFORE social-
cognition structure during training.

For each of ~20 log-spaced training checkpoints (step 0 = random init through
step 143,000 = final):
  1. Load Pythia-2.8b-deduped at that checkpoint.
  2. Forward-pass all 14-condition RSA stimuli; extract hidden states at ALL layers.
  3. Per layer: condition centroids → center → cosine RDM → compare to brain RDM
     (full Spearman, block-model, within-affective, within-social).
  4. Record peak-layer and layer-averaged metrics.

Streams checkpoints: load one model, extract, delete, load next.

Outputs:
  results/developmental_emergence/pythia_trajectory.json
  figures/pythia_developmental_curve.png
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
STIMULI_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "rsa_stimuli.jsonl"
BRAIN_RDM_PATH = BASE / "results" / "cognitive_rsa" / "brain_rdm.npz"
OUT_DIR = BASE / "results" / "developmental_emergence"
FIG_DIR = BASE / "figures"

MODEL_NAME = "EleutherAI/pythia-2.8b-deduped"

# ~20 log-spaced checkpoints across 143,000 training steps
CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000,
]

# Condition blocks (must match within_block_control.py)
AFFECTIVE = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
SOCIAL = {"belief", "mentalizing", "intention", "theory_of_mind",
          "empathy", "self_referential", "judgment", "moral"}

# Reference values from the 4-architecture headline (Qwen-7B)
REF_FULL = 0.73
REF_WITHIN_SOCIAL = 0.62
REF_WITHIN_AFFECTIVE = -0.11


# ── Utility functions (mirror the project's existing conventions) ──────────

def utri(M: np.ndarray) -> np.ndarray:
    """Upper triangle of a square matrix (excluding diagonal)."""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (no scipy import needed at runtime)."""
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def rdm_cosine_centered(act: np.ndarray) -> np.ndarray:
    """Center across conditions then compute cosine distance RDM.

    Matches the headline recipe: center → normalize → 1 - cos.
    """
    act = act - act.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - np.clip(a @ a.T, -1, 1)


def build_block_model(conditions: list[str]) -> np.ndarray:
    """Binary block-membership model: 0 if same block, 1 if different."""
    block = np.array([0 if c in AFFECTIVE else 1 for c in conditions])
    return (block[:, None] != block[None, :]).astype(float)


def submatrix_rho(
    brain: np.ndarray, llm: np.ndarray, idx: np.ndarray
) -> float:
    """Spearman on the upper triangle of a sub-matrix."""
    sub_b = brain[np.ix_(idx, idx)]
    sub_l = llm[np.ix_(idx, idx)]
    bv = utri(sub_b)
    lv = utri(sub_l)
    if len(bv) < 3:
        return float("nan")
    return spearman(bv, lv)


# ── Stimuli loading ───────────────────────────────────────────────────────

def load_stimuli(path: Path) -> tuple[list[str], list[str]]:
    """Load RSA stimuli JSONL.  Returns (texts, conditions)."""
    texts, conditions = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            conditions.append(rec["condition"])
    return texts, conditions


# ── Activation extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 256,
) -> np.ndarray:
    """Extract mean-all-pooled hidden states for each stimulus at every layer.

    Returns: np.ndarray of shape (n_stim, n_layers, hidden_dim), float32.
    """
    device = next(model.parameters()).device
    n_stim = len(texts)
    all_states = []

    for start in range(0, n_stim, batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq, hidden)
        # Layer 0 = embedding layer; layers 1..N = transformer blocks
        hidden = outputs.hidden_states  # tuple length = n_layers + 1

        attention_mask = enc["attention_mask"]  # (batch, seq)
        mask_f = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)

        # Mean-all pooling per layer (skip embedding layer 0)
        batch_layers = []
        for layer_idx in range(1, len(hidden)):
            h = hidden[layer_idx]  # (batch, seq, hidden)
            pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            batch_layers.append(pooled.cpu().float().numpy())

        # Stack: (n_layers, batch, hidden) -> transpose -> (batch, n_layers, hidden)
        stacked = np.stack(batch_layers, axis=0)  # (n_layers, batch, hidden)
        stacked = stacked.transpose(1, 0, 2)  # (batch, n_layers, hidden)
        all_states.append(stacked)

    return np.concatenate(all_states, axis=0)  # (n_stim, n_layers, hidden)


# ── Per-layer RSA analysis ────────────────────────────────────────────────

def analyze_layer(
    brain_rdm: np.ndarray,
    block_model: np.ndarray,
    layer_centroids: np.ndarray,
    conditions: list[str],
    aff_idx: np.ndarray,
    soc_idx: np.ndarray,
) -> dict:
    """Compute all RSA metrics for one layer's centroids vs brain RDM."""
    llm_rdm = rdm_cosine_centered(layer_centroids)

    full_rho = spearman(utri(brain_rdm), utri(llm_rdm))
    block_rho = spearman(utri(llm_rdm), utri(block_model))
    within_aff = submatrix_rho(brain_rdm, llm_rdm, aff_idx)
    within_soc = submatrix_rho(brain_rdm, llm_rdm, soc_idx)

    return {
        "full_rho": full_rho,
        "block_rho": block_rho,
        "within_aff_rho": within_aff,
        "within_soc_rho": within_soc,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track developmental emergence of brain-like structure in Pythia"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for forward passes (default: 8)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Max token length per stimulus (default: 256)"
    )
    parser.add_argument(
        "--checkpoints", type=str, default=None,
        help="Comma-separated checkpoint steps to run (default: all 20)"
    )
    parser.add_argument(
        "--save_rdms", action="store_true",
        help="Also save per-checkpoint peak-layer RDMs as .npz"
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Parse checkpoint list
    if args.checkpoints:
        checkpoints = [int(x.strip()) for x in args.checkpoints.split(",")]
    else:
        checkpoints = CHECKPOINTS

    # ── Load brain RDM ────────────────────────────────────────────────
    brain_data = np.load(BRAIN_RDM_PATH, allow_pickle=True)
    brain_rdm = brain_data["rdm"].astype(np.float64)
    brain_conds = list(brain_data["conditions"])
    n_conds = len(brain_conds)
    print(f"Brain RDM: {n_conds} conditions: {brain_conds}")

    block_model = build_block_model(brain_conds)
    aff_idx = np.array([i for i, c in enumerate(brain_conds) if c in AFFECTIVE])
    soc_idx = np.array([i for i, c in enumerate(brain_conds) if c in SOCIAL])
    print(f"  Affective ({len(aff_idx)}): {[brain_conds[i] for i in aff_idx]}")
    print(f"  Social    ({len(soc_idx)}): {[brain_conds[i] for i in soc_idx]}")

    # ── Load stimuli ──────────────────────────────────────────────────
    texts, stim_conditions = load_stimuli(STIMULI_PATH)
    unique_conds = sorted(set(stim_conditions))
    assert set(unique_conds) == set(brain_conds), (
        f"Stimulus conditions {unique_conds} != brain conditions {brain_conds}"
    )
    n_stim = len(texts)
    print(f"Stimuli: {n_stim} total, {len(unique_conds)} conditions")

    # Pre-compute stimulus→condition index mapping (in brain_conds order)
    cond_to_stim_idx: dict[str, list[int]] = {c: [] for c in brain_conds}
    for i, c in enumerate(stim_conditions):
        cond_to_stim_idx[c].append(i)

    # ── Stream through checkpoints ────────────────────────────────────
    trajectory: list[dict] = []
    wall_start = time.time()

    for ckpt_i, step in enumerate(checkpoints):
        t0 = time.time()
        revision = f"step{step}"
        print(f"\n{'='*70}")
        print(f"[{ckpt_i+1}/{len(checkpoints)}] Checkpoint step={step}  "
              f"(revision={revision})")
        print(f"{'='*70}")

        # Load model and tokenizer
        print(f"  Loading model {MODEL_NAME} @ {revision} ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        print(f"  Model loaded. Device: {next(model.parameters()).device}")

        # Extract hidden states: (n_stim, n_layers, hidden_dim)
        print(f"  Extracting hidden states for {n_stim} stimuli "
              f"(batch_size={args.batch_size}) ...")
        states = extract_hidden_states(
            model, tokenizer, texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        n_layers = states.shape[1]
        hidden_dim = states.shape[2]
        print(f"  Got states: {states.shape}  "
              f"({n_layers} layers, {hidden_dim} hidden dim)")

        # Free GPU memory immediately
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Per-layer analysis ────────────────────────────────────────
        # Build condition centroids: (n_conds, n_layers, hidden_dim)
        cond_means = np.zeros(
            (n_conds, n_layers, hidden_dim), dtype=np.float64
        )
        for ci, cond in enumerate(brain_conds):
            idx = cond_to_stim_idx[cond]
            cond_means[ci] = states[idx].astype(np.float64).mean(axis=0)

        del states
        gc.collect()

        best_full_rho = -2.0
        best_layer = 0
        layer_results = []

        for L in range(n_layers):
            centroids = cond_means[:, L, :]
            metrics = analyze_layer(
                brain_rdm, block_model, centroids, brain_conds, aff_idx, soc_idx
            )
            metrics["layer"] = L
            layer_results.append(metrics)

            if metrics["full_rho"] > best_full_rho:
                best_full_rho = metrics["full_rho"]
                best_layer = L

        # Layer-averaged metrics (mean across all layers)
        avg_full = float(np.mean([r["full_rho"] for r in layer_results]))
        avg_block = float(np.mean([r["block_rho"] for r in layer_results]))
        avg_aff = float(np.mean([r["within_aff_rho"] for r in layer_results]))
        avg_soc = float(np.mean([r["within_soc_rho"] for r in layer_results]))

        # Peak-layer results
        peak = layer_results[best_layer]

        record = {
            "step": step,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            # Peak-layer metrics
            "peak_layer": best_layer,
            "full_rho": peak["full_rho"],
            "block_rho": peak["block_rho"],
            "within_aff_rho": peak["within_aff_rho"],
            "within_soc_rho": peak["within_soc_rho"],
            # Layer-averaged metrics
            "avg_full_rho": avg_full,
            "avg_block_rho": avg_block,
            "avg_within_aff_rho": avg_aff,
            "avg_within_soc_rho": avg_soc,
            # Per-layer detail (for layer-depth profiles)
            "per_layer": [
                {
                    "layer": r["layer"],
                    "full_rho": r["full_rho"],
                    "block_rho": r["block_rho"],
                    "within_aff_rho": r["within_aff_rho"],
                    "within_soc_rho": r["within_soc_rho"],
                }
                for r in layer_results
            ],
        }
        trajectory.append(record)

        # Optionally save peak-layer RDM
        if args.save_rdms:
            peak_centroids = cond_means[:, best_layer, :]
            peak_rdm = rdm_cosine_centered(peak_centroids)
            rdm_path = OUT_DIR / f"pythia_rdm_step{step}.npz"
            np.savez_compressed(
                rdm_path,
                rdm=peak_rdm,
                conditions=np.array(brain_conds),
                step=step,
                peak_layer=best_layer,
            )

        elapsed = time.time() - t0
        print(f"\n  step={step:>6d}  peak_L={best_layer:>2d}  "
              f"full_rho={peak['full_rho']:+.3f}  "
              f"block_rho={peak['block_rho']:+.3f}  "
              f"w_aff={peak['within_aff_rho']:+.3f}  "
              f"w_soc={peak['within_soc_rho']:+.3f}  "
              f"({elapsed:.0f}s)")

    total_time = time.time() - wall_start
    print(f"\n{'='*70}")
    print(f"Total wall time: {total_time/60:.1f} min")

    # ── Save trajectory JSON ──────────────────────────────────────────
    output = {
        "model": MODEL_NAME,
        "n_stimuli": n_stim,
        "conditions": brain_conds,
        "affective": [brain_conds[i] for i in aff_idx],
        "social": [brain_conds[i] for i in soc_idx],
        "n_checkpoints": len(trajectory),
        "checkpoints": trajectory,
        "reference": {
            "note": "Qwen-7B headline values for comparison",
            "full_rho": REF_FULL,
            "within_social_rho": REF_WITHIN_SOCIAL,
            "within_affective_rho": REF_WITHIN_AFFECTIVE,
        },
        "total_wall_seconds": total_time,
    }
    out_path = OUT_DIR / "pythia_trajectory.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: Pythia-2.8B developmental trajectory (peak layer)")
    print(f"{'='*70}")
    header = (f"{'step':>8s}  {'peak_L':>6s}  {'full':>7s}  {'block':>7s}  "
              f"{'w_aff':>7s}  {'w_soc':>7s}")
    print(header)
    print("-" * len(header))
    for rec in trajectory:
        print(f"{rec['step']:>8d}  {rec['peak_layer']:>6d}  "
              f"{rec['full_rho']:>+7.3f}  {rec['block_rho']:>+7.3f}  "
              f"{rec['within_aff_rho']:>+7.3f}  {rec['within_soc_rho']:>+7.3f}")
    print("-" * len(header))
    print(f"{'Qwen ref':>8s}  {'--':>6s}  {REF_FULL:>+7.3f}  "
          f"{'--':>7s}  {REF_WITHIN_AFFECTIVE:>+7.3f}  {REF_WITHIN_SOCIAL:>+7.3f}")

    # ── Generate figure ───────────────────────────────────────────────
    plot_trajectory(trajectory)


def plot_trajectory(trajectory: list[dict]) -> None:
    """Generate the developmental curve figure."""
    steps = [r["step"] for r in trajectory]
    full = [r["full_rho"] for r in trajectory]
    block = [r["block_rho"] for r in trajectory]
    w_aff = [r["within_aff_rho"] for r in trajectory]
    w_soc = [r["within_soc_rho"] for r in trajectory]

    # Replace step=0 with 0.5 for log scale (can't plot log(0))
    steps_plot = [max(s, 0.5) for s in steps]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps_plot, full, "k-o", linewidth=2, markersize=5,
            label="Full RSA (91 pairs)", zorder=5)
    ax.plot(steps_plot, block, color="gray", linestyle="--", marker="s",
            linewidth=1.5, markersize=4,
            label="Block split (emotion vs social)", zorder=4)
    ax.plot(steps_plot, w_soc, "b-^", linewidth=2, markersize=5,
            label="Within-social (28 pairs)", zorder=5)
    ax.plot(steps_plot, w_aff, "r-v", linewidth=2, markersize=5,
            label="Within-affective (15 pairs)", zorder=5)

    # Reference lines from Qwen-7B headline
    ax.axhline(REF_FULL, color="black", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(steps_plot[-1] * 1.1, REF_FULL, "Qwen-7B full",
            fontsize=8, va="center", color="black", alpha=0.6)

    ax.axhline(REF_WITHIN_SOCIAL, color="blue", linestyle=":", alpha=0.4,
               linewidth=1)
    ax.text(steps_plot[-1] * 1.1, REF_WITHIN_SOCIAL, "Qwen-7B w-soc",
            fontsize=8, va="center", color="blue", alpha=0.6)

    ax.axhline(REF_WITHIN_AFFECTIVE, color="red", linestyle=":", alpha=0.4,
               linewidth=1)
    ax.text(steps_plot[-1] * 1.1, REF_WITHIN_AFFECTIVE, "Qwen-7B w-aff",
            fontsize=8, va="center", color="red", alpha=0.6)

    ax.axhline(0, color="gray", linestyle="-", alpha=0.2, linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)", fontsize=12)
    ax.set_ylabel("Spearman $\\rho$ vs brain RDM", fontsize=12)
    ax.set_title("Emergence of brain-like cognitive structure during training\n"
                 "(Pythia-2.8B-deduped)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)

    # Mark the random-init baseline
    if steps[0] == 0:
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.3)
        ax.text(0.6, ax.get_ylim()[1] * 0.95, "random init",
                fontsize=8, color="gray", rotation=90, va="top")

    plt.tight_layout()
    fig_path = FIG_DIR / "pythia_developmental_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
