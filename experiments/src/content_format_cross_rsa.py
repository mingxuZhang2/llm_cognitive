#!/usr/bin/env python3
"""
Content x Format Crossed RSA: extraction + variance decomposition.

Design: 4 social conditions x 4 formats x 15 items = 240 stimuli.

Phase 1 — Extract hidden states at the model's peak layer (mean_all pooling).
Phase 2 — Build 240x240 pairwise cosine distance matrix and decompose variance
           into content, format, and length predictors (OLS partial R²).
Phase 3 — Compare the 4-condition mean RDM to the corresponding brain sub-RDM.

This script answers: when content and format are fully crossed, does the model
cluster stimuli by cognitive content or by surface format?

GPU required. One model per run.
Usage:
  python src/content_format_cross_rsa.py \
      --model_path /path/to/model \
      --model_short Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
STIM_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "content_format_cross_stimuli.jsonl"
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "content_format_cross"

CONDITIONS = ["false_belief", "intention", "moral_judgment", "self_referential"]
FORMATS = ["narrative", "dialogue", "record", "list"]

PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}

# Mapping from our 4 conditions to the 14-condition brain RDM labels
CONDITION_TO_BRAIN = {
    "false_belief": "belief",        # closest Neurosynth map
    "intention": "intention",
    "moral_judgment": "judgment",     # normative judgment
    "self_referential": "self_referential",
}


def load_stimuli() -> list[dict]:
    """Load the 240 crossed stimuli."""
    if not STIM_PATH.exists():
        raise FileNotFoundError(
            f"Stimuli not found at {STIM_PATH}. "
            "Run content_format_cross_stimuli.py first."
        )
    with open(STIM_PATH) as f:
        stimuli = [json.loads(line) for line in f]
    print(f"Loaded {len(stimuli)} stimuli from {STIM_PATH.name}")
    return stimuli


def extract_hidden_states(model, tokenizer, stimuli, peak_layer, device,
                          batch_size=8) -> np.ndarray:
    """Extract mean_all-pooled hidden states at the peak layer."""
    all_hidden = []
    n_batches = (len(stimuli) + batch_size - 1) // batch_size

    print(f"  Extracting hidden states at layer {peak_layer} "
          f"({len(stimuli)} stimuli, {n_batches} batches)...")

    for bi in range(0, len(stimuli), batch_size):
        batch = stimuli[bi:bi + batch_size]
        texts = [s["text"] for s in batch]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # Hidden states at peak layer: (batch, seq, hidden_dim)
        h = out.hidden_states[peak_layer].float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)

        # Mean-all pooling (weighted by attention mask)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_hidden.append(pooled.cpu().numpy())

        if (bi // batch_size + 1) % 10 == 0:
            print(f"    batch {bi // batch_size + 1}/{n_batches}")

    return np.concatenate(all_hidden, axis=0)  # (n_stimuli, hidden_dim)


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_normed = X / norms
    sim = X_normed @ X_normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def variance_decomposition(dist_mat, stimuli):
    """OLS variance decomposition: predict pairwise distance from content, format, length."""
    n = len(stimuli)
    # Extract upper triangle indices
    triu_i, triu_j = np.triu_indices(n, k=1)
    y = dist_mat[triu_i, triu_j]

    # Build predictors
    conds = [s["condition"] for s in stimuli]
    fmts = [s["format"] for s in stimuli]
    lens = np.array([len(s["text"].split()) for s in stimuli], dtype=float)

    # Content RDM: same condition = 0, different = 1
    content_pred = np.array([
        0.0 if conds[i] == conds[j] else 1.0
        for i, j in zip(triu_i, triu_j)
    ])

    # Format RDM: same format = 0, different = 1
    format_pred = np.array([
        0.0 if fmts[i] == fmts[j] else 1.0
        for i, j in zip(triu_i, triu_j)
    ])

    # Length RDM: normalized absolute difference
    max_len = lens.max()
    length_pred = np.array([
        abs(lens[i] - lens[j]) / max_len if max_len > 0 else 0.0
        for i, j in zip(triu_i, triu_j)
    ])

    # OLS: y ~ content + format + length
    X = np.column_stack([content_pred, format_pred, length_pred])
    X_with_intercept = np.column_stack([np.ones(len(y)), X])

    # Full model R²
    beta_full = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    y_pred_full = X_with_intercept @ beta_full
    ss_res_full = np.sum((y - y_pred_full) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_full = 1.0 - ss_res_full / ss_tot if ss_tot > 0 else 0.0

    # Partial R² for each predictor: drop one, measure R² decrease
    partial_r2 = {}
    predictor_names = ["content", "format", "length"]
    for drop_idx, name in enumerate(predictor_names):
        keep = [i for i in range(3) if i != drop_idx]
        X_reduced = np.column_stack([np.ones(len(y)), X[:, keep]])
        beta_red = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
        y_pred_red = X_reduced @ beta_red
        ss_res_red = np.sum((y - y_pred_red) ** 2)
        r2_red = 1.0 - ss_res_red / ss_tot if ss_tot > 0 else 0.0
        partial_r2[name] = r2_full - r2_red

    # Also compute semi-partial correlations (Spearman)
    from scipy.stats import spearmanr as spr
    rho_content, p_content = spr(content_pred, y)
    rho_format, p_format = spr(format_pred, y)
    rho_length, p_length = spr(length_pred, y)

    return {
        "full_r2": float(r2_full),
        "content_partial_r2": float(partial_r2["content"]),
        "format_partial_r2": float(partial_r2["format"]),
        "length_partial_r2": float(partial_r2["length"]),
        "content_spearman": {"rho": float(rho_content), "p": float(p_content)},
        "format_spearman": {"rho": float(rho_format), "p": float(p_format)},
        "length_spearman": {"rho": float(rho_length), "p": float(p_length)},
        "n_pairs": int(len(y)),
    }


def condition_format_rdms(hidden_states, stimuli):
    """Build 4x4 condition and format mean RDMs."""
    conds = [s["condition"] for s in stimuli]
    fmts = [s["format"] for s in stimuli]

    # Condition centroids (across all formats)
    cond_centroids = {}
    for c in CONDITIONS:
        mask = [i for i, x in enumerate(conds) if x == c]
        centroid = hidden_states[mask].mean(axis=0)
        cond_centroids[c] = centroid

    # Center condition centroids
    grand_mean = np.mean(list(cond_centroids.values()), axis=0)
    for c in cond_centroids:
        cond_centroids[c] = cond_centroids[c] - grand_mean

    # 4x4 condition RDM (cosine)
    n_cond = len(CONDITIONS)
    cond_rdm = np.zeros((n_cond, n_cond))
    for i, c1 in enumerate(CONDITIONS):
        for j, c2 in enumerate(CONDITIONS):
            if i < j:
                v1, v2 = cond_centroids[c1], cond_centroids[c2]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                d = 1.0 - cos
                cond_rdm[i, j] = d
                cond_rdm[j, i] = d

    # Format centroids (across all conditions)
    fmt_centroids = {}
    for f in FORMATS:
        mask = [i for i, x in enumerate(fmts) if x == f]
        centroid = hidden_states[mask].mean(axis=0)
        fmt_centroids[f] = centroid

    # Center format centroids
    grand_mean_f = np.mean(list(fmt_centroids.values()), axis=0)
    for f in fmt_centroids:
        fmt_centroids[f] = fmt_centroids[f] - grand_mean_f

    # 4x4 format RDM (cosine)
    n_fmt = len(FORMATS)
    fmt_rdm = np.zeros((n_fmt, n_fmt))
    for i, f1 in enumerate(FORMATS):
        for j, f2 in enumerate(FORMATS):
            if i < j:
                v1, v2 = fmt_centroids[f1], fmt_centroids[f2]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                d = 1.0 - cos
                fmt_rdm[i, j] = d
                fmt_rdm[j, i] = d

    return cond_rdm, fmt_rdm, cond_centroids, fmt_centroids


def clustering_test(dist_mat, stimuli):
    """For each stimulus, is its nearest neighbor same-content or same-format?"""
    n = len(stimuli)
    conds = [s["condition"] for s in stimuli]
    fmts = [s["format"] for s in stimuli]

    content_nn = 0
    format_nn = 0
    both_nn = 0
    neither_nn = 0

    for i in range(n):
        dists = dist_mat[i].copy()
        dists[i] = np.inf  # exclude self
        nn = int(np.argmin(dists))

        same_content = conds[i] == conds[nn]
        same_format = fmts[i] == fmts[nn]

        if same_content and same_format:
            both_nn += 1
        elif same_content:
            content_nn += 1
        elif same_format:
            format_nn += 1
        else:
            neither_nn += 1

    total = content_nn + format_nn + both_nn + neither_nn
    return {
        "content_only": content_nn,
        "format_only": format_nn,
        "both": both_nn,
        "neither": neither_nn,
        "total": total,
        "content_pct": round(100.0 * (content_nn + both_nn) / total, 1),
        "format_pct": round(100.0 * (format_nn + both_nn) / total, 1),
    }


def cross_validated_content_tracking(hidden_states, stimuli):
    """Leave-one-format-out: compute content tracking stability.

    For each held-out format, build condition centroids from the OTHER 3 formats,
    then measure whether the held-out stimuli cluster by condition.
    """
    conds = [s["condition"] for s in stimuli]
    fmts = [s["format"] for s in stimuli]

    results = {}
    for held_out in FORMATS:
        # Train: everything except held_out format
        train_mask = [i for i, f in enumerate(fmts) if f != held_out]
        test_mask = [i for i, f in enumerate(fmts) if f == held_out]

        # Build condition centroids from training set
        centroids = {}
        for c in CONDITIONS:
            c_mask = [i for i in train_mask if conds[i] == c]
            centroids[c] = hidden_states[c_mask].mean(axis=0)

        # For each test stimulus, find nearest centroid
        correct = 0
        for i in test_mask:
            dists = {c: np.linalg.norm(hidden_states[i] - centroids[c])
                     for c in CONDITIONS}
            nearest = min(dists, key=dists.get)
            if nearest == conds[i]:
                correct += 1

        accuracy = correct / len(test_mask) if test_mask else 0.0
        results[held_out] = {
            "accuracy": round(accuracy, 4),
            "n_test": len(test_mask),
            "n_correct": correct,
        }

    mean_acc = np.mean([v["accuracy"] for v in results.values()])
    results["mean_accuracy"] = round(float(mean_acc), 4)
    return results


def brain_comparison(cond_rdm, model_short):
    """Compare 4-condition LLM RDM to corresponding brain sub-RDM."""
    brain_path = RSA_DIR / "brain_rdm.npz"
    if not brain_path.exists():
        print(f"  WARNING: brain_rdm.npz not found, skipping brain comparison")
        return {"error": "brain_rdm.npz not found"}

    brain = np.load(brain_path, allow_pickle=True)
    brain_conds = list(brain["conditions"])
    brain_rdm_full = brain["rdm"]

    # Extract the 4 corresponding brain conditions
    brain_labels = [CONDITION_TO_BRAIN[c] for c in CONDITIONS]
    brain_idx = [brain_conds.index(b) for b in brain_labels]
    brain_sub = brain_rdm_full[np.ix_(brain_idx, brain_idx)]

    # Upper triangle
    llm_tri = cond_rdm[np.triu_indices(4, k=1)]
    brain_tri = brain_sub[np.triu_indices(4, k=1)]

    rho, p = spearmanr(llm_tri, brain_tri)

    # Permutation p-value (only 4! = 24 permutations, so enumerate all)
    from itertools import permutations as perm_iter
    obs_rho = rho
    count = 0
    total = 0
    for perm in perm_iter(range(4)):
        perm = list(perm)
        perm_brain = brain_sub[np.ix_(perm, perm)]
        perm_tri = perm_brain[np.triu_indices(4, k=1)]
        r, _ = spearmanr(perm_tri, llm_tri)
        if r >= obs_rho:
            count += 1
        total += 1
    p_perm = count / total

    return {
        "rho_vs_brain_4cond": round(float(rho), 4),
        "p_spearman": round(float(p), 4),
        "p_perm_exact": round(float(p_perm), 4),
        "n_pairs": int(len(llm_tri)),
        "brain_conditions": brain_labels,
        "llm_rdm_tri": [round(float(x), 6) for x in llm_tri],
        "brain_rdm_tri": [round(float(x), 6) for x in brain_tri],
    }


def content_vs_format_centroid_test(hidden_states, stimuli):
    """For each condition-format cell, compute distances to:
    (a) same-content-different-format centroids
    (b) same-format-different-content centroids
    Report which is closer on average.
    """
    conds = [s["condition"] for s in stimuli]
    fmts = [s["format"] for s in stimuli]

    # Build cell centroids: (condition, format) -> centroid
    cell_centroids = {}
    for c in CONDITIONS:
        for f in FORMATS:
            mask = [i for i, (sc, sf) in enumerate(zip(conds, fmts))
                    if sc == c and sf == f]
            cell_centroids[(c, f)] = hidden_states[mask].mean(axis=0)

    # For each cell, compute mean distance to same-content and same-format
    same_content_dists = []
    same_format_dists = []

    for c in CONDITIONS:
        for f in FORMATS:
            v = cell_centroids[(c, f)]

            # Same content, different format
            for f2 in FORMATS:
                if f2 != f:
                    v2 = cell_centroids[(c, f2)]
                    d = 1.0 - np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2) + 1e-10)
                    same_content_dists.append(d)

            # Same format, different content
            for c2 in CONDITIONS:
                if c2 != c:
                    v2 = cell_centroids[(c2, f)]
                    d = 1.0 - np.dot(v, v2) / (np.linalg.norm(v) * np.linalg.norm(v2) + 1e-10)
                    same_format_dists.append(d)

    mean_same_content = float(np.mean(same_content_dists))
    mean_same_format = float(np.mean(same_format_dists))

    # Wilcoxon signed-rank is not applicable (different pair counts),
    # use Mann-Whitney U as a rough test
    from scipy.stats import mannwhitneyu
    u_stat, u_p = mannwhitneyu(same_content_dists, same_format_dists,
                                alternative='less')

    return {
        "mean_same_content_dist": round(mean_same_content, 6),
        "mean_same_format_dist": round(mean_same_format, 6),
        "content_closer": mean_same_content < mean_same_format,
        "ratio": round(mean_same_content / mean_same_format, 4) if mean_same_format > 0 else None,
        "mannwhitneyu_stat": float(u_stat),
        "mannwhitneyu_p": float(u_p),
        "n_same_content_pairs": len(same_content_dists),
        "n_same_format_pairs": len(same_format_dists),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Content x Format crossed RSA: extraction + analysis")
    ap.add_argument("--model_path", required=True, help="Path to HuggingFace model")
    ap.add_argument("--model_short", default=None, help="Short model name")
    args = ap.parse_args()

    model_short = args.model_short or Path(args.model_path).name
    peak_layer = PEAK.get(model_short, 15)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"CONTENT x FORMAT CROSSED RSA: {model_short}")
    print("=" * 70)
    print(f"Device: {device}, Peak layer: L{peak_layer}")

    t0 = time.time()

    # ── Load stimuli ──
    stimuli = load_stimuli()

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Extract hidden states
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 1: Extract hidden states")
    print(f"{'='*70}")
    hidden_states = extract_hidden_states(
        model, tokenizer, stimuli, peak_layer, device)
    print(f"  Shape: {hidden_states.shape}")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  Model unloaded, GPU memory freed.")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Compute RDMs and variance decomposition
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 2: Variance decomposition")
    print(f"{'='*70}")

    # 240x240 distance matrix
    dist_mat = cosine_distance_matrix(hidden_states.astype(np.float64))
    print(f"  Distance matrix: {dist_mat.shape}, "
          f"mean={dist_mat[np.triu_indices(len(stimuli), k=1)].mean():.4f}")

    # Variance decomposition
    vd = variance_decomposition(dist_mat, stimuli)
    print(f"\n  Variance decomposition (OLS partial R²):")
    print(f"    Content:  {vd['content_partial_r2']:.4f}")
    print(f"    Format:   {vd['format_partial_r2']:.4f}")
    print(f"    Length:   {vd['length_partial_r2']:.4f}")
    print(f"    Full R²:  {vd['full_r2']:.4f}")
    print(f"\n  Spearman correlations:")
    print(f"    Content:  rho={vd['content_spearman']['rho']:.4f} "
          f"(p={vd['content_spearman']['p']:.4e})")
    print(f"    Format:   rho={vd['format_spearman']['rho']:.4f} "
          f"(p={vd['format_spearman']['p']:.4e})")

    # Condition and format mean RDMs
    cond_rdm, fmt_rdm, cond_centroids, fmt_centroids = condition_format_rdms(
        hidden_states.astype(np.float64), stimuli)

    print(f"\n  4x4 Condition mean RDM:")
    for i, c in enumerate(CONDITIONS):
        row = "    " + f"{c:20s}"
        for j in range(4):
            row += f" {cond_rdm[i,j]:.4f}"
        print(row)

    print(f"\n  4x4 Format mean RDM:")
    for i, f in enumerate(FORMATS):
        row = "    " + f"{f:20s}"
        for j in range(4):
            row += f" {fmt_rdm[i,j]:.4f}"
        print(row)

    # Nearest-neighbor clustering test
    clust = clustering_test(dist_mat, stimuli)
    print(f"\n  Nearest-neighbor clustering test:")
    print(f"    Same-content NN: {clust['content_pct']:.1f}%")
    print(f"    Same-format NN:  {clust['format_pct']:.1f}%")
    print(f"    Content-only: {clust['content_only']}, "
          f"Format-only: {clust['format_only']}, "
          f"Both: {clust['both']}, Neither: {clust['neither']}")

    # Content vs format centroid distances
    cvf = content_vs_format_centroid_test(hidden_states.astype(np.float64), stimuli)
    print(f"\n  Cell centroid distances:")
    print(f"    Same-content, diff-format: {cvf['mean_same_content_dist']:.6f}")
    print(f"    Same-format, diff-content: {cvf['mean_same_format_dist']:.6f}")
    print(f"    Content closer: {cvf['content_closer']} "
          f"(ratio={cvf['ratio']})")

    # ══════════════════════════════════════════════════════════════════
    # Cross-validated content tracking
    # ══════════════════════════════════════════════════════════════════
    cv = cross_validated_content_tracking(
        hidden_states.astype(np.float64), stimuli)
    print(f"\n  Leave-one-format-out content classification:")
    for fmt in FORMATS:
        print(f"    Held-out {fmt:12s}: accuracy={cv[fmt]['accuracy']:.4f} "
              f"({cv[fmt]['n_correct']}/{cv[fmt]['n_test']})")
    print(f"    Mean accuracy: {cv['mean_accuracy']:.4f} (chance=0.25)")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Brain comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Brain comparison")
    print(f"{'='*70}")
    brain = brain_comparison(cond_rdm, model_short)
    if "error" not in brain:
        print(f"  4-condition RSA vs brain: rho={brain['rho_vs_brain_4cond']:.4f} "
              f"(p_perm={brain['p_perm_exact']:.4f})")
    else:
        print(f"  {brain['error']}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_short}")
    print(f"{'='*70}")
    print(f"  Content partial R²:   {vd['content_partial_r2']:.4f}")
    print(f"  Format partial R²:    {vd['format_partial_r2']:.4f}")
    print(f"  Content/Format ratio:  {vd['content_partial_r2']/max(vd['format_partial_r2'], 1e-10):.2f}x")
    print(f"  NN content cluster:    {clust['content_pct']:.1f}%")
    print(f"  NN format cluster:     {clust['format_pct']:.1f}%")
    print(f"  CV content accuracy:   {cv['mean_accuracy']:.4f}")
    if "error" not in brain:
        print(f"  Brain RSA (4 cond):    rho={brain['rho_vs_brain_4cond']:.4f}")
    print(f"  Total time:            {time.time()-t0:.0f}s")

    verdict = []
    if vd['content_partial_r2'] > vd['format_partial_r2']:
        verdict.append("Content explains MORE variance than format (content > format)")
    else:
        verdict.append("Format explains MORE variance than content (format > content)")
    if cv['mean_accuracy'] > 0.50:
        verdict.append(f"Content tracking is robust (CV accuracy={cv['mean_accuracy']:.2f}, chance=0.25)")
    if cvf['content_closer']:
        verdict.append("Cell centroids cluster by content, not format")
    print(f"\n  VERDICT:")
    for v in verdict:
        print(f"    -> {v}")

    # ══════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": model_short,
        "peak_layer": peak_layer,
        "n_stimuli": len(stimuli),
        "n_conditions": len(CONDITIONS),
        "n_formats": len(FORMATS),
        "conditions": CONDITIONS,
        "formats": FORMATS,
        "variance_decomposition": vd,
        "condition_rdm_4x4": cond_rdm.tolist(),
        "format_rdm_4x4": fmt_rdm.tolist(),
        "brain_comparison": brain,
        "clustering_test": clust,
        "content_vs_format_centroids": cvf,
        "cross_validated": cv,
        "verdict": verdict,
    }

    out_path = OUT_DIR / f"{model_short}_content_format_cross.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
