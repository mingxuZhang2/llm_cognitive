#!/usr/bin/env python3
"""
Functional Convergence Analysis across LLM Architectures
=========================================================

Computes a "Functional Convergence Index" quantifying how similar
the functional atlases are across 4 different model architectures.

Metrics:
  1. Dissociation matrix similarity (Pearson correlation)
  2. DAG edge consistency (universal/consistent/weak/absent)
  3. Layer-normalized functional profile comparison (cosine similarity)
  4. Functional hierarchy consistency (Spearman rank correlation)
  5. Overall convergence index with null model significance test

No GPU required. Uses only numpy/scipy.
"""

import json
import os
import sys
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine as cosine_dist
from scipy.interpolate import interp1d


# --- Configuration ---
MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]
CATEGORIES = ["code", "ethics", "factual_qa", "humanities",
              "language", "math", "reasoning", "science"]
N_CATS = len(CATEGORIES)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTI_DIR = os.path.join(BASE_DIR, "results", "scaled_multi")
LAYER_DIR = os.path.join(BASE_DIR, "results", "scaled_layer_stats")
DOSE_DIR = os.path.join(BASE_DIR, "results", "dose_response")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

SPILLOVER_THRESHOLD = 0.03  # For DAG edge detection
N_PERMUTATIONS = 1000       # Null model permutations
SEED = 42


def load_dissociation_matrices():
    """Load 8x8 dissociation matrices for all models.

    Returns both ppl_ratio (for correlation analysis) and logppl_delta
    (for DAG edge thresholding, since delta values have meaningful zero).
    """
    matrices_ratio = {}
    matrices_delta = {}
    for model in MODELS:
        path = os.path.join(MULTI_DIR, f"{model}_multi_dissociation.json")
        with open(path) as f:
            data = json.load(f)
        matrices_ratio[model] = np.array(data["dissociation_matrix_ppl_ratio"])
        matrices_delta[model] = np.array(data["dissociation_matrix_logppl_delta"])
    return matrices_ratio, matrices_delta


def load_layer_distributions():
    """Load top5000 layer distributions for all models."""
    layer_dists = {}
    for model in MODELS:
        path = os.path.join(LAYER_DIR, f"{model}_layer_stats.json")
        with open(path) as f:
            data = json.load(f)
        dists = {}
        for cat in CATEGORIES:
            raw = np.array(data["top5000_layer_dist"][cat], dtype=float)
            dists[cat] = raw
        layer_dists[model] = dists
    return layer_dists


def normalize_layer_dist_to_common_grid(layer_dists, n_bins=100):
    """
    Normalize layer distributions to probability distributions on a
    common grid (0-1 relative depth) so models with different layer
    counts are comparable.
    """
    normalized = {}
    for model, dists in layer_dists.items():
        normalized[model] = {}
        for cat, raw in dists.items():
            n_layers = len(raw)
            total = raw.sum()
            if total == 0:
                prob = np.zeros(n_bins)
            else:
                # Convert to probability distribution
                prob_orig = raw / total
                # Layer centers as relative depth [0, 1]
                centers = (np.arange(n_layers) + 0.5) / n_layers
                # Interpolate to common grid
                grid = np.linspace(0, 1, n_bins)
                interp_func = interp1d(centers, prob_orig, kind='linear',
                                       bounds_error=False, fill_value=0.0)
                prob = interp_func(grid)
                # Re-normalize after interpolation
                s = prob.sum()
                if s > 0:
                    prob = prob / s
            normalized[model][cat] = prob
    return normalized


# =============================================================
# Metric 1: Dissociation Matrix Similarity
# =============================================================
def compute_matrix_similarity(matrices):
    """
    For each model pair, compute Pearson correlation of:
    - Full 8x8 matrix (64 values)
    - Diagonal only (8 values)
    - Off-diagonal only (56 values)
    """
    model_pairs = list(combinations(MODELS, 2))
    full_corrs, diag_corrs, offdiag_corrs = [], [], []

    pair_details = {}
    for m1, m2 in model_pairs:
        mat1, mat2 = matrices[m1], matrices[m2]

        # Full matrix
        r_full, p_full = pearsonr(mat1.flatten(), mat2.flatten())
        full_corrs.append(r_full)

        # Diagonal
        d1, d2 = np.diag(mat1), np.diag(mat2)
        r_diag, p_diag = pearsonr(d1, d2)
        diag_corrs.append(r_diag)

        # Off-diagonal
        mask = ~np.eye(N_CATS, dtype=bool)
        od1, od2 = mat1[mask], mat2[mask]
        r_offdiag, p_offdiag = pearsonr(od1, od2)
        offdiag_corrs.append(r_offdiag)

        pair_details[f"{m1} vs {m2}"] = {
            "full_r": float(r_full),
            "full_p": float(p_full),
            "diagonal_r": float(r_diag),
            "diagonal_p": float(p_diag),
            "offdiag_r": float(r_offdiag),
            "offdiag_p": float(p_offdiag),
        }

    result = {
        "pair_details": pair_details,
        "full_matrix": {
            "mean_r": float(np.mean(full_corrs)),
            "std_r": float(np.std(full_corrs)),
            "all_r": [float(x) for x in full_corrs],
        },
        "diagonal": {
            "mean_r": float(np.mean(diag_corrs)),
            "std_r": float(np.std(diag_corrs)),
            "all_r": [float(x) for x in diag_corrs],
        },
        "off_diagonal": {
            "mean_r": float(np.mean(offdiag_corrs)),
            "std_r": float(np.std(offdiag_corrs)),
            "all_r": [float(x) for x in offdiag_corrs],
        },
    }
    return result, full_corrs


# =============================================================
# Metric 2: DAG Edge Consistency
# =============================================================
def compute_dag_edge_consistency(matrices_delta):
    """
    For each off-diagonal entry in the logppl_delta matrix, count in how
    many models it exceeds the spillover threshold. Classify edges as
    universal/consistent/weak/absent.

    Uses logppl_delta (not ppl_ratio) because delta values have a
    meaningful zero point, making thresholding interpretable.
    """
    # Binarize off-diagonal entries
    edge_counts = np.zeros((N_CATS, N_CATS), dtype=int)
    per_model_edges = {}

    for model in MODELS:
        mat = matrices_delta[model]
        mask = ~np.eye(N_CATS, dtype=bool)
        edges = (mat > SPILLOVER_THRESHOLD) & mask
        edge_counts += edges.astype(int)
        per_model_edges[model] = edges

    # Classify
    mask = ~np.eye(N_CATS, dtype=bool)
    n_offdiag = int(mask.sum())  # 56

    universal = int(np.sum((edge_counts == 4) & mask))   # present in 4/4
    consistent = int(np.sum((edge_counts == 3) & mask))   # present in 3/4
    weak = int(np.sum((edge_counts >= 1) & (edge_counts <= 2) & mask))
    absent = int(np.sum((edge_counts == 0) & mask))

    # Build edge detail table
    edge_details = {}
    for i in range(N_CATS):
        for j in range(N_CATS):
            if i == j:
                continue
            key = f"{CATEGORIES[i]}->{CATEGORIES[j]}"
            count = int(edge_counts[i, j])
            label = {4: "universal", 3: "consistent"}.get(count,
                     "weak" if count >= 1 else "absent")
            edge_details[key] = {
                "n_models": count,
                "classification": label,
                "models_present": [m for m in MODELS if per_model_edges[m][i, j]],
            }

    # Consistency score: fraction of edges that are universal or consistent
    consistency_score = (universal + consistent) / n_offdiag if n_offdiag > 0 else 0.0

    result = {
        "threshold": SPILLOVER_THRESHOLD,
        "total_off_diagonal_entries": n_offdiag,
        "universal_4_4": universal,
        "consistent_3_4": consistent,
        "weak_1_2": weak,
        "absent_0_4": absent,
        "consistency_score": float(consistency_score),
        "edge_details": edge_details,
    }
    return result


# =============================================================
# Metric 3: Layer-Normalized Functional Profile Comparison
# =============================================================
def compute_layer_profile_similarity(norm_dists):
    """
    For each category, compute pairwise cosine similarity of the
    normalized layer distributions across model pairs.
    """
    model_pairs = list(combinations(MODELS, 2))
    per_category = {}
    all_sims = []

    for cat in CATEGORIES:
        sims = []
        for m1, m2 in model_pairs:
            v1 = norm_dists[m1][cat]
            v2 = norm_dists[m2][cat]
            # Cosine similarity = 1 - cosine distance
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                sim = 0.0
            else:
                sim = 1.0 - cosine_dist(v1, v2)
            sims.append(sim)
        per_category[cat] = {
            "mean_cosine_sim": float(np.mean(sims)),
            "std_cosine_sim": float(np.std(sims)),
            "all_sims": [float(x) for x in sims],
        }
        all_sims.extend(sims)

    result = {
        "per_category": per_category,
        "overall_mean": float(np.mean(all_sims)),
        "overall_std": float(np.std(all_sims)),
    }
    return result, [per_category[c]["mean_cosine_sim"] for c in CATEGORIES]


# =============================================================
# Metric 4: Functional Hierarchy Consistency
# =============================================================
def compute_hierarchy_consistency(matrices):
    """
    Rank functions by diagonal self-effect. Compute Spearman rank
    correlation across model pairs.
    """
    model_pairs = list(combinations(MODELS, 2))
    rankings = {}

    for model in MODELS:
        diag = np.diag(matrices[model])
        # Rank by self-effect (higher = more modular)
        rank_order = np.argsort(-diag)  # descending
        rank_labels = [CATEGORIES[i] for i in rank_order]
        rankings[model] = {
            "diagonal_values": {CATEGORIES[i]: float(diag[i]) for i in range(N_CATS)},
            "rank_order": rank_labels,
        }

    spearman_corrs = []
    pair_details = {}
    for m1, m2 in model_pairs:
        d1 = np.diag(matrices[m1])
        d2 = np.diag(matrices[m2])
        rho, pval = spearmanr(d1, d2)
        spearman_corrs.append(rho)
        pair_details[f"{m1} vs {m2}"] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
        }

    result = {
        "model_rankings": rankings,
        "pair_details": pair_details,
        "mean_spearman": float(np.mean(spearman_corrs)),
        "std_spearman": float(np.std(spearman_corrs)),
        "all_spearman": [float(x) for x in spearman_corrs],
    }
    return result, spearman_corrs


# =============================================================
# Metric 5: Overall Convergence Index with Null Model
# =============================================================
def compute_permuted_metrics(matrices, norm_dists, rng):
    """
    Randomly permute category labels within each model and
    recompute the three sub-metrics.
    """
    perm_matrices = {}
    perm_norm_dists = {}

    for model in MODELS:
        perm = rng.permutation(N_CATS)
        # Permute rows and columns of dissociation matrix
        mat = matrices[model]
        perm_mat = mat[np.ix_(perm, perm)]
        perm_matrices[model] = perm_mat

        # Permute layer distributions
        perm_dists = {}
        cats_permuted = [CATEGORIES[i] for i in perm]
        for orig_idx, orig_cat in enumerate(CATEGORIES):
            # Category at position orig_idx gets the distribution of perm[orig_idx]
            perm_dists[orig_cat] = norm_dists[model][cats_permuted[orig_idx]]
        perm_norm_dists[model] = perm_dists

    # Recompute metrics 1, 3, 4
    model_pairs = list(combinations(MODELS, 2))

    # Metric 1: full matrix Pearson
    full_corrs = []
    for m1, m2 in model_pairs:
        r, _ = pearsonr(perm_matrices[m1].flatten(), perm_matrices[m2].flatten())
        full_corrs.append(r)
    m1_score = float(np.mean(full_corrs))

    # Metric 3: layer cosine similarity
    all_sims = []
    for cat in CATEGORIES:
        for m1, m2 in model_pairs:
            v1 = perm_norm_dists[m1][cat]
            v2 = perm_norm_dists[m2][cat]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 == 0 or n2 == 0:
                sim = 0.0
            else:
                sim = 1.0 - cosine_dist(v1, v2)
            all_sims.append(sim)
    m3_score = float(np.mean(all_sims))

    # Metric 4: Spearman of diagonals
    spearman_corrs = []
    for m1, m2 in model_pairs:
        d1 = np.diag(perm_matrices[m1])
        d2 = np.diag(perm_matrices[m2])
        rho, _ = spearmanr(d1, d2)
        spearman_corrs.append(rho)
    m4_score = float(np.mean(spearman_corrs))

    return m1_score, m3_score, m4_score


def normalize_to_01(value, metric_type):
    """
    Normalize a metric to [0, 1] range.
    - Pearson/Spearman correlations: range [-1, 1] -> [0, 1] via (x+1)/2
    - Cosine similarity: already in [0, 1] for non-negative distributions
    """
    if metric_type in ("pearson", "spearman"):
        return (value + 1.0) / 2.0
    elif metric_type == "cosine":
        return max(0.0, min(1.0, value))
    return value


def compute_overall_convergence(matrices, norm_dists,
                                m1_real, m3_real, m4_real):
    """
    Compute overall convergence index and null model significance.
    """
    # Normalize to [0, 1]
    m1_norm = normalize_to_01(m1_real, "pearson")
    m3_norm = normalize_to_01(m3_real, "cosine")
    m4_norm = normalize_to_01(m4_real, "spearman")

    real_index = (m1_norm + m3_norm + m4_norm) / 3.0

    # Null model: permute category labels
    rng = np.random.default_rng(SEED)
    null_indices = []
    null_m1, null_m3, null_m4 = [], [], []

    for _ in range(N_PERMUTATIONS):
        pm1, pm3, pm4 = compute_permuted_metrics(matrices, norm_dists, rng)
        null_m1.append(pm1)
        null_m3.append(pm3)
        null_m4.append(pm4)
        idx = (normalize_to_01(pm1, "pearson") +
               normalize_to_01(pm3, "cosine") +
               normalize_to_01(pm4, "spearman")) / 3.0
        null_indices.append(idx)

    null_indices = np.array(null_indices)
    null_mean = float(np.mean(null_indices))
    null_std = float(np.std(null_indices))

    if null_std > 0:
        z_score = (real_index - null_mean) / null_std
    else:
        z_score = float('inf')

    # P-value: fraction of null >= real
    p_value = float(np.mean(null_indices >= real_index))

    result = {
        "metric1_dissociation_similarity": {
            "raw": float(m1_real),
            "normalized_01": float(m1_norm),
        },
        "metric3_layer_profile_similarity": {
            "raw": float(m3_real),
            "normalized_01": float(m3_norm),
        },
        "metric4_hierarchy_consistency": {
            "raw": float(m4_real),
            "normalized_01": float(m4_norm),
        },
        "convergence_index": float(real_index),
        "null_model": {
            "n_permutations": N_PERMUTATIONS,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": float(z_score),
            "p_value": p_value,
            "null_metric1_mean": float(np.mean(null_m1)),
            "null_metric3_mean": float(np.mean(null_m3)),
            "null_metric4_mean": float(np.mean(null_m4)),
        },
    }
    return result


# =============================================================
# Main
# =============================================================
def main():
    print("=" * 70)
    print("FUNCTIONAL CONVERGENCE ANALYSIS ACROSS LLM ARCHITECTURES")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    matrices, matrices_delta = load_dissociation_matrices()
    layer_dists = load_layer_distributions()
    norm_dists = normalize_layer_dist_to_common_grid(layer_dists)
    print(f"  Models: {MODELS}")
    print(f"  Categories: {CATEGORIES}")
    print()

    # --- Metric 1 ---
    print("-" * 70)
    print("METRIC 1: Dissociation Matrix Similarity")
    print("-" * 70)
    m1_result, m1_full_corrs = compute_matrix_similarity(matrices)
    print(f"  Full matrix Pearson r:    {m1_result['full_matrix']['mean_r']:.4f} "
          f"+/- {m1_result['full_matrix']['std_r']:.4f}")
    print(f"  Diagonal Pearson r:       {m1_result['diagonal']['mean_r']:.4f} "
          f"+/- {m1_result['diagonal']['std_r']:.4f}")
    print(f"  Off-diagonal Pearson r:   {m1_result['off_diagonal']['mean_r']:.4f} "
          f"+/- {m1_result['off_diagonal']['std_r']:.4f}")
    print()
    for pair, detail in m1_result["pair_details"].items():
        print(f"    {pair}")
        print(f"      full r={detail['full_r']:.4f}  diag r={detail['diagonal_r']:.4f}"
              f"  offdiag r={detail['offdiag_r']:.4f}")
    print()

    # --- Metric 2 ---
    print("-" * 70)
    print("METRIC 2: DAG Edge Consistency")
    print("-" * 70)
    m2_result = compute_dag_edge_consistency(matrices_delta)
    print(f"  Spillover threshold: > {m2_result['threshold']}")
    print(f"  Total off-diagonal entries: {m2_result['total_off_diagonal_entries']}")
    print(f"  Universal (4/4 models):   {m2_result['universal_4_4']}")
    print(f"  Consistent (3/4 models):  {m2_result['consistent_3_4']}")
    print(f"  Weak (1-2 models):        {m2_result['weak_1_2']}")
    print(f"  Absent (0/4 models):      {m2_result['absent_0_4']}")
    print(f"  Consistency score:        {m2_result['consistency_score']:.4f}")
    print()

    # Show universal edges
    print("  Universal edges (present in all 4 models):")
    for edge, info in m2_result["edge_details"].items():
        if info["classification"] == "universal":
            print(f"    {edge}")
    print()

    # --- Metric 3 ---
    print("-" * 70)
    print("METRIC 3: Layer-Normalized Functional Profile Comparison")
    print("-" * 70)
    m3_result, m3_cat_means = compute_layer_profile_similarity(norm_dists)
    print(f"  Overall mean cosine sim:  {m3_result['overall_mean']:.4f} "
          f"+/- {m3_result['overall_std']:.4f}")
    print()
    print("  Per-category cosine similarity:")
    for cat in CATEGORIES:
        info = m3_result["per_category"][cat]
        print(f"    {cat:12s}: {info['mean_cosine_sim']:.4f} "
              f"+/- {info['std_cosine_sim']:.4f}")
    print()

    # --- Metric 4 ---
    print("-" * 70)
    print("METRIC 4: Functional Hierarchy Consistency")
    print("-" * 70)
    m4_result, m4_spearman_corrs = compute_hierarchy_consistency(matrices)
    print(f"  Mean Spearman rho:        {m4_result['mean_spearman']:.4f} "
          f"+/- {m4_result['std_spearman']:.4f}")
    print()
    print("  Model rankings (by self-effect, most modular first):")
    for model in MODELS:
        rank = m4_result["model_rankings"][model]["rank_order"]
        short_name = model.split("/")[-1][:30]
        print(f"    {short_name:30s}: {' > '.join(rank)}")
    print()
    for pair, detail in m4_result["pair_details"].items():
        print(f"    {pair}: rho={detail['spearman_rho']:.4f} "
              f"(p={detail['p_value']:.4f})")
    print()

    # --- Metric 5 ---
    print("-" * 70)
    print("METRIC 5: Overall Convergence Index")
    print("-" * 70)
    m1_real = m1_result["full_matrix"]["mean_r"]
    m3_real = m3_result["overall_mean"]
    m4_real = m4_result["mean_spearman"]

    print(f"  Computing null model ({N_PERMUTATIONS} permutations)...")
    m5_result = compute_overall_convergence(matrices, norm_dists,
                                            m1_real, m3_real, m4_real)

    print(f"\n  Component scores (normalized to [0,1]):")
    print(f"    Dissociation similarity: {m5_result['metric1_dissociation_similarity']['normalized_01']:.4f}"
          f"  (raw r={m5_result['metric1_dissociation_similarity']['raw']:.4f})")
    print(f"    Layer profile similarity:{m5_result['metric3_layer_profile_similarity']['normalized_01']:.4f}"
          f"  (raw cos={m5_result['metric3_layer_profile_similarity']['raw']:.4f})")
    print(f"    Hierarchy consistency:   {m5_result['metric4_hierarchy_consistency']['normalized_01']:.4f}"
          f"  (raw rho={m5_result['metric4_hierarchy_consistency']['raw']:.4f})")
    print()
    print(f"  *** CONVERGENCE INDEX: {m5_result['convergence_index']:.4f} ***")
    print()
    print(f"  Null model: mean={m5_result['null_model']['null_mean']:.4f} "
          f"+/- {m5_result['null_model']['null_std']:.4f}")
    print(f"  Z-score:    {m5_result['null_model']['z_score']:.2f}")
    print(f"  P-value:    {m5_result['null_model']['p_value']:.4f}")
    print()

    if m5_result['null_model']['z_score'] > 3:
        verdict = "STRONG convergence (z > 3, highly significant)"
    elif m5_result['null_model']['z_score'] > 2:
        verdict = "MODERATE convergence (z > 2, significant)"
    elif m5_result['null_model']['z_score'] > 1:
        verdict = "WEAK convergence (z > 1, marginal)"
    else:
        verdict = "NO significant convergence"
    print(f"  Verdict: {verdict}")
    print()

    # --- Assemble output ---
    output = {
        "models": MODELS,
        "categories": CATEGORIES,
        "metric1_dissociation_similarity": m1_result,
        "metric2_dag_edge_consistency": m2_result,
        "metric3_layer_profile_similarity": m3_result,
        "metric4_hierarchy_consistency": m4_result,
        "metric5_overall_convergence": m5_result,
    }

    # Remove verbose edge_details from JSON to keep file manageable
    # (keep the summary stats; edge_details has 56 entries)
    # Actually, keep them -- they're useful for the paper.

    out_path = os.path.join(OUTPUT_DIR, "convergence_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
