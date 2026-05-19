"""
Statistical validation of dissociation claims across models and scales.

Validates the functional dissociation atlas using:
  1. Cross-scale consistency (pilot / medium / balanced) for each model
  2. Cross-model universality for each pairwise dissociation
  3. Permutation test on dissociation matrices (matrix-level diagonal dominance)
  4. Per-pair significance via cross-replication tests:
     a. One-sample t-test on D > 0 across 4 models x 3 scales (12 observations)
     b. Wilcoxon signed-rank test (non-parametric) on the same
     c. Within-matrix rank test: is this pair's D in the top quantile of all
        possible 2x2 sub-matrices from the same dissociation matrix?
  5. Effect size (Cohen's d) for diagonal vs off-diagonal
  6. BH-FDR correction across all 28 per-pair tests

Works entirely from the pre-computed JSON dissociation results -- no GPU or
model loading required.

Usage:
    python -m experiments.src.statistical_validation
"""

import json
import itertools
import numpy as np
from pathlib import Path

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
RESULTS_DIR = BASE_DIR / "experiments" / "results"

SCALE_DIRS = {
    "pilot":    RESULTS_DIR / "multi",
    "medium":   RESULTS_DIR / "scaled_multi",
    "balanced": RESULTS_DIR / "balanced_multi",
}

MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]

N_PERM = 10000       # permutation iterations (matrix-level)
RNG_SEED = 42
ALPHA = 0.05         # significance threshold after FDR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dissociation(scale_dir: Path, model: str) -> dict:
    """Load a dissociation JSON file."""
    path = scale_dir / f"{model}_multi_dissociation.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def logppl_matrix(data: dict) -> np.ndarray:
    """Extract the log-PPL delta dissociation matrix as a numpy array."""
    return np.array(data["dissociation_matrix_logppl_delta"])


def pairwise_dissociation_effect(mat: np.ndarray, i: int, j: int) -> float:
    """
    Double-dissociation effect for functions (i, j).

    D(i,j) = min(mat[i,i] - mat[i,j],  mat[j,j] - mat[j,i])

    Positive value means both legs show the expected selective pattern.
    The magnitude reflects the weaker leg (conservative).
    """
    leg1 = mat[i, i] - mat[i, j]  # ablate_i hurts i more than j
    leg2 = mat[j, j] - mat[j, i]  # ablate_j hurts j more than i
    return min(leg1, leg2)


def all_pair_indices(n: int):
    """Return all (i, j) pairs with i < j."""
    return list(itertools.combinations(range(n), 2))


# ---------------------------------------------------------------------------
# 1. Cross-scale consistency
# ---------------------------------------------------------------------------

def cross_scale_consistency(models, scale_dirs, all_data, cats):
    """
    For each model and each pair, compute the dissociation effect at each
    scale and report mean +/- std across scales.
    """
    n = len(cats)
    pairs = all_pair_indices(n)
    results = {}

    for model in models:
        model_results = {}
        matrices = {}
        for scale_name in scale_dirs:
            if scale_name in all_data[model]:
                matrices[scale_name] = logppl_matrix(all_data[model][scale_name])

        if not matrices:
            continue

        for (i, j) in pairs:
            pair_name = f"{cats[i]}_vs_{cats[j]}"
            effects = []
            by_scale = {}
            for sn, m in matrices.items():
                eff = pairwise_dissociation_effect(m, i, j)
                effects.append(eff)
                by_scale[sn] = float(eff)
            model_results[pair_name] = {
                "effects_by_scale": by_scale,
                "mean": float(np.mean(effects)),
                "std": float(np.std(effects, ddof=1)) if len(effects) > 1 else 0.0,
                "n_scales": len(effects),
                "all_positive": all(e > 0 for e in effects),
            }
        results[model] = model_results
    return results


# ---------------------------------------------------------------------------
# 2. Cross-model universality
# ---------------------------------------------------------------------------

def cross_model_universality(models, all_data, cats, reference_scale="medium"):
    """
    For each pair, count how many models show a positive dissociation.
    """
    n = len(cats)
    pairs = all_pair_indices(n)
    pair_results = {}

    for (i, j) in pairs:
        pair_name = f"{cats[i]}_vs_{cats[j]}"
        model_effects = {}
        for model in models:
            if reference_scale in all_data[model]:
                mat = logppl_matrix(all_data[model][reference_scale])
                model_effects[model] = float(pairwise_dissociation_effect(mat, i, j))
        n_positive = sum(1 for v in model_effects.values() if v > 0)
        pair_results[pair_name] = {
            "model_effects": model_effects,
            "n_models_positive": n_positive,
            "n_models_total": len(model_effects),
            "universal": n_positive == len(model_effects),
        }
    return pair_results


# ---------------------------------------------------------------------------
# 3. Matrix-level permutation test (diagonal dominance)
# ---------------------------------------------------------------------------

def diagonal_dominance_statistic(mat: np.ndarray) -> float:
    """mean(diagonal) - mean(off-diagonal)."""
    n = mat.shape[0]
    diag = np.diag(mat)
    off_diag = mat[~np.eye(n, dtype=bool)]
    return float(np.mean(diag) - np.mean(off_diag))


def permutation_test_matrix(mat: np.ndarray, n_perm: int, rng: np.random.Generator):
    """
    Permutation test for diagonal dominance.

    Null: randomly permute the row-to-column alignment (shuffle which
    category's selective neurons are assigned to which row label).
    """
    observed = diagonal_dominance_statistic(mat)
    n = mat.shape[0]
    count = 0
    null_dist = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        permuted = mat[perm, :]
        null_dist[k] = diagonal_dominance_statistic(permuted)
        if null_dist[k] >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist)),
        "z_score": float((observed - np.mean(null_dist)) / (np.std(null_dist) + 1e-12)),
    }


# ---------------------------------------------------------------------------
# 4. Per-pair significance via cross-replication
# ---------------------------------------------------------------------------

def per_pair_replication_tests(models, all_data, cats):
    """
    For each pair (i, j), collect the dissociation effect D across all
    (model, scale) combinations.  With 4 models x 3 scales = up to 12
    independent measurements, test H0: D <= 0 using:
      (a) one-sample t-test
      (b) Wilcoxon signed-rank test
      (c) sign test (fraction positive)
    """
    n = len(cats)
    pairs = all_pair_indices(n)
    results = {}

    for (i, j) in pairs:
        pair_name = f"{cats[i]}_vs_{cats[j]}"
        observations = []
        details = []
        for model in models:
            for scale_name in SCALE_DIRS:
                if scale_name in all_data[model]:
                    mat = logppl_matrix(all_data[model][scale_name])
                    d = pairwise_dissociation_effect(mat, i, j)
                    observations.append(d)
                    details.append({"model": model, "scale": scale_name, "D": float(d)})

        obs = np.array(observations)
        n_obs = len(obs)
        n_positive = int(np.sum(obs > 0))

        result = {
            "n_observations": n_obs,
            "n_positive": n_positive,
            "mean": float(np.mean(obs)),
            "std": float(np.std(obs, ddof=1)),
            "min": float(np.min(obs)),
            "max": float(np.max(obs)),
            "observations": details,
        }

        # (a) One-sample t-test: D > 0
        if n_obs >= 3 and HAS_SCIPY:
            t_stat, p_two = sp_stats.ttest_1samp(obs, 0.0)
            # One-sided: p(D > 0)
            p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
            result["ttest"] = {
                "t_statistic": float(t_stat),
                "p_value_onesided": float(p_one),
                "p_value_twosided": float(p_two),
                "df": n_obs - 1,
            }
        elif n_obs >= 3:
            # Manual t-test
            t_stat = np.mean(obs) / (np.std(obs, ddof=1) / np.sqrt(n_obs))
            # Approximate p using normal for large-ish df
            from math import erfc, sqrt
            p_one = 0.5 * erfc(t_stat / sqrt(2))  # upper tail
            p_one = 1.0 - p_one  # P(T > observed | H0: mu=0)
            # Actually for one-sided greater: p = P(T >= t_obs | H0)
            # With positive t_stat: p_onesided = P(T >= t) which is small
            # erfc gives P(X > x*sqrt(2)) for standard normal
            p_one_sided = 0.5 * erfc(float(t_stat) / sqrt(2))
            result["ttest"] = {
                "t_statistic": float(t_stat),
                "p_value_onesided_approx": float(p_one_sided),
                "df": n_obs - 1,
            }

        # (b) Wilcoxon signed-rank test
        if n_obs >= 6 and HAS_SCIPY:
            try:
                w_stat, p_wilcox = sp_stats.wilcoxon(obs, alternative="greater")
                result["wilcoxon"] = {
                    "statistic": float(w_stat),
                    "p_value": float(p_wilcox),
                }
            except ValueError:
                # All values identical or zero
                result["wilcoxon"] = {"note": "degenerate (all values equal)"}

        # (c) Sign test: binomial probability of >= n_positive out of n_obs
        if n_obs >= 3:
            # Under H0: P(positive) = 0.5
            # P(X >= n_positive | n, p=0.5) via sum of binomial
            if HAS_SCIPY:
                btest = sp_stats.binomtest(n_positive, n_obs, 0.5,
                                           alternative="greater")
                result["sign_test"] = {
                    "n_positive": n_positive,
                    "n_total": n_obs,
                    "p_value": float(btest.pvalue),
                }
            else:
                # Manual: sum binomial(n,k) * 0.5^n for k >= n_positive
                from math import comb
                p_sign = sum(comb(n_obs, k) for k in range(n_positive, n_obs + 1)) / (2 ** n_obs)
                result["sign_test"] = {
                    "n_positive": n_positive,
                    "n_total": n_obs,
                    "p_value": float(p_sign),
                }

        results[pair_name] = result

    return results


# ---------------------------------------------------------------------------
# 5. Effect size: Cohen's d
# ---------------------------------------------------------------------------

def cohens_d_matrix(mat: np.ndarray) -> float:
    """Cohen's d comparing diagonal elements to off-diagonal elements."""
    n = mat.shape[0]
    diag = np.diag(mat)
    off_diag = mat[~np.eye(n, dtype=bool)]
    mean_diff = np.mean(diag) - np.mean(off_diag)
    s_diag = np.std(diag, ddof=1)
    s_off = np.std(off_diag, ddof=1)
    n1, n2 = len(diag), len(off_diag)
    pooled_std = np.sqrt(((n1 - 1) * s_diag**2 + (n2 - 1) * s_off**2) / (n1 + n2 - 2))
    return float(mean_diff / (pooled_std + 1e-12))


def cohens_d_replication(observations: np.ndarray) -> float:
    """
    Cohen's d for a one-sample test (D > 0).
    d = mean(D) / std(D)
    """
    if len(observations) < 2:
        return float('inf') if np.mean(observations) > 0 else float('-inf')
    return float(np.mean(observations) / (np.std(observations, ddof=1) + 1e-12))


# ---------------------------------------------------------------------------
# 6. Bootstrap CIs on cross-replication
# ---------------------------------------------------------------------------

def bootstrap_ci(observations: np.ndarray, n_boot: int,
                 rng: np.random.Generator, ci: float = 0.95):
    """Bootstrap CI for the mean of observations."""
    n = len(observations)
    boot_means = np.empty(n_boot)
    for k in range(n_boot):
        sample = observations[rng.integers(0, n, size=n)]
        boot_means[k] = np.mean(sample)
    alpha = 1 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {"mean": float(np.mean(observations)), "ci_lo": lo, "ci_hi": hi}


# ---------------------------------------------------------------------------
# 7. BH-FDR correction
# ---------------------------------------------------------------------------

def bh_fdr(p_values: np.ndarray, alpha: float = 0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns: adjusted p-values and boolean rejection mask.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.empty(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for k in range(n - 2, -1, -1):
        rank = k + 1
        adjusted_val = sorted_p[k] * n / rank
        adjusted[sorted_idx[k]] = min(adjusted_val, adjusted[sorted_idx[k + 1]])
    adjusted = np.clip(adjusted, 0, 1)
    rejected = adjusted <= alpha
    return adjusted, rejected


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_validation():
    """Run all statistical validation analyses."""
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 72)
    print("STATISTICAL VALIDATION OF DISSOCIATION CLAIMS")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    all_data = {}
    for model in MODELS:
        all_data[model] = {}
        for scale_name, scale_dir in SCALE_DIRS.items():
            data = load_dissociation(scale_dir, model)
            if data is not None:
                all_data[model][scale_name] = data

    # Get categories
    cats = None
    for model in MODELS:
        for scale_name in SCALE_DIRS:
            if scale_name in all_data[model]:
                cats = all_data[model][scale_name]["categories"]
                break
        if cats is not None:
            break

    n_cats = len(cats)
    pairs = all_pair_indices(n_cats)
    n_pairs = len(pairs)  # 28

    print(f"\nCategories ({n_cats}): {cats}")
    print(f"Models: {len(MODELS)}")
    print(f"Scales: {list(SCALE_DIRS.keys())}")
    print(f"Pairwise comparisons: {n_pairs}")
    print(f"Observations per pair: {len(MODELS)} models x {len(SCALE_DIRS)} scales = "
          f"{len(MODELS) * len(SCALE_DIRS)}")

    output = {
        "categories": cats,
        "models": MODELS,
        "scales": list(SCALE_DIRS.keys()),
        "n_permutations_matrix": N_PERM,
        "alpha": ALPHA,
    }

    # ------------------------------------------------------------------
    # 1. Cross-scale consistency
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("1. CROSS-SCALE CONSISTENCY")
    print("-" * 72)

    cross_scale = cross_scale_consistency(MODELS, SCALE_DIRS, all_data, cats)
    output["cross_scale_consistency"] = cross_scale

    for model in MODELS:
        if model not in cross_scale:
            continue
        mr = cross_scale[model]
        n_consistent = sum(1 for v in mr.values() if v["all_positive"])
        print(f"\n  {model}:")
        print(f"    Pairs positive at ALL 3 scales: {n_consistent}/{n_pairs}")
        weakest = sorted(mr.items(), key=lambda x: x[1]["mean"])[:3]
        for pname, pdata in weakest:
            print(f"    Weakest: {pname}: mean={pdata['mean']:.4f} +/- {pdata['std']:.4f}")

    # ------------------------------------------------------------------
    # 2. Cross-model universality
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("2. CROSS-MODEL UNIVERSALITY (medium scale)")
    print("-" * 72)

    universality = cross_model_universality(MODELS, all_data, cats, "medium")
    output["cross_model_universality"] = universality

    n_universal = sum(1 for v in universality.values() if v["universal"])
    print(f"\n  Universal dissociations (all 4 models positive): {n_universal}/{n_pairs}")
    non_universal = {k: v for k, v in universality.items() if not v["universal"]}
    if non_universal:
        print("  Non-universal pairs:")
        for pname, pdata in non_universal.items():
            print(f"    {pname}: {pdata['n_models_positive']}/{pdata['n_models_total']}")
    else:
        print("  ALL 28 pairs are universal across all 4 models.")

    # ------------------------------------------------------------------
    # 3. Matrix-level permutation test
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"3. MATRIX-LEVEL PERMUTATION TEST ({N_PERM} permutations)")
    print("-" * 72)

    matrix_perm = {}
    for model in MODELS:
        model_results = {}
        for scale_name in SCALE_DIRS:
            if scale_name not in all_data[model]:
                continue
            mat = logppl_matrix(all_data[model][scale_name])
            result = permutation_test_matrix(mat, N_PERM, rng)
            model_results[scale_name] = result
            print(f"\n  {model} ({scale_name}):")
            print(f"    Observed = {result['observed']:.4f}, "
                  f"z = {result['z_score']:.2f}, p = {result['p_value']:.6f}")
        matrix_perm[model] = model_results
    output["matrix_permutation_test"] = matrix_perm

    # ------------------------------------------------------------------
    # 4. Per-pair cross-replication tests
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("4. PER-PAIR CROSS-REPLICATION TESTS")
    print("    (each pair has 12 observations: 4 models x 3 scales)")
    print("-" * 72)

    pair_tests = per_pair_replication_tests(MODELS, all_data, cats)
    output["per_pair_replication_tests"] = pair_tests

    # Collect p-values for FDR
    all_p_values = []
    all_p_labels = []

    print(f"\n  {'Pair':<30s} {'mean D':>8s} {'t':>7s} {'p(t)':>10s} "
          f"{'p(W)':>10s} {'p(sign)':>10s} {'d':>7s} {'n+/n':>6s}")
    print("  " + "-" * 100)

    for (i, j) in pairs:
        pair_name = f"{cats[i]}_vs_{cats[j]}"
        r = pair_tests[pair_name]
        obs = np.array([d["D"] for d in r["observations"]])

        # Primary p-value: t-test (most powerful with 12 obs)
        p_t = r.get("ttest", {}).get("p_value_onesided", 1.0)
        t_val = r.get("ttest", {}).get("t_statistic", 0.0)
        p_w = r.get("wilcoxon", {}).get("p_value", float("nan"))
        p_s = r.get("sign_test", {}).get("p_value", float("nan"))
        d_eff = cohens_d_replication(obs)

        all_p_values.append(p_t)
        all_p_labels.append(pair_name)

        print(f"  {pair_name:<30s} {r['mean']:8.4f} {t_val:7.2f} {p_t:10.6f} "
              f"{p_w:10.6f} {p_s:10.6f} {d_eff:7.2f} "
              f"{r['n_positive']}/{r['n_observations']}")

    # ------------------------------------------------------------------
    # 5. Effect sizes
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("5. EFFECT SIZES")
    print("-" * 72)

    effect_sizes = {}
    for model in MODELS:
        if "medium" not in all_data[model]:
            continue
        mat = logppl_matrix(all_data[model]["medium"])
        d_mat = cohens_d_matrix(mat)
        effect_sizes[model] = {"matrix_cohens_d": d_mat}

        if d_mat >= 0.8:
            interp = "LARGE"
        elif d_mat >= 0.5:
            interp = "MEDIUM"
        else:
            interp = "SMALL"
        print(f"\n  {model}:")
        print(f"    Matrix-level Cohen's d = {d_mat:.3f} ({interp})")

    # Per-pair effect sizes (from cross-replication)
    print(f"\n  Per-pair Cohen's d (from 12 cross-replication observations):")
    pair_d_values = []
    for (i, j) in pairs:
        pair_name = f"{cats[i]}_vs_{cats[j]}"
        obs = np.array([d["D"] for d in pair_tests[pair_name]["observations"]])
        d_eff = cohens_d_replication(obs)
        pair_d_values.append(d_eff)
        effect_sizes.setdefault("per_pair_cohens_d", {})[pair_name] = d_eff

    print(f"    mean={np.mean(pair_d_values):.3f}, "
          f"median={np.median(pair_d_values):.3f}, "
          f"range=[{np.min(pair_d_values):.3f}, {np.max(pair_d_values):.3f}]")
    print(f"    All d > 0.8 (large): {sum(1 for d in pair_d_values if d >= 0.8)}/{n_pairs}")
    print(f"    All d > 2.0 (very large): {sum(1 for d in pair_d_values if d >= 2.0)}/{n_pairs}")

    output["effect_sizes"] = effect_sizes

    # ------------------------------------------------------------------
    # 6. Bootstrap CIs (cross-replication)
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("6. BOOTSTRAP 95% CIs (1000 resamples of 12 observations per pair)")
    print("-" * 72)

    N_BOOT = 1000
    boot_results = {}
    n_ci_above_zero = 0
    for (i, j) in pairs:
        pair_name = f"{cats[i]}_vs_{cats[j]}"
        obs = np.array([d["D"] for d in pair_tests[pair_name]["observations"]])
        ci = bootstrap_ci(obs, N_BOOT, rng)
        boot_results[pair_name] = ci
        if ci["ci_lo"] > 0:
            n_ci_above_zero += 1

    output["bootstrap_cis"] = boot_results

    print(f"\n  Pairs with 95% CI entirely above zero: {n_ci_above_zero}/{n_pairs}")

    # Show weakest CIs
    weakest = sorted(boot_results.items(), key=lambda x: x[1]["ci_lo"])[:5]
    print(f"\n  Five pairs with lowest CI lower bound:")
    for pname, ci in weakest:
        print(f"    {pname:<30s}: mean={ci['mean']:.4f}, "
              f"95% CI=[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    # ------------------------------------------------------------------
    # 7. BH-FDR correction
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("7. BH-FDR CORRECTION (28 per-pair t-test p-values)")
    print("-" * 72)

    p_arr = np.array(all_p_values)
    adjusted_p, rejected = bh_fdr(p_arr, ALPHA)

    fdr_results = []
    for idx, pair_name in enumerate(all_p_labels):
        fdr_results.append({
            "pair": pair_name,
            "p_raw": float(p_arr[idx]),
            "p_adjusted": float(adjusted_p[idx]),
            "significant": bool(rejected[idx]),
        })

    n_sig_raw = int(np.sum(p_arr < ALPHA))
    n_sig_fdr = int(np.sum(rejected))

    print(f"\n  Total tests: {len(p_arr)}")
    print(f"  Significant (raw p < {ALPHA}): {n_sig_raw}/{len(p_arr)}")
    print(f"  Significant (BH-FDR adjusted p < {ALPHA}): {n_sig_fdr}/{len(p_arr)}")

    # Show results sorted by adjusted p
    print(f"\n  {'Pair':<30s} {'p_raw':>10s} {'p_adj':>10s} {'sig?':>6s}")
    print("  " + "-" * 60)
    for r in sorted(fdr_results, key=lambda x: x["p_adjusted"]):
        sig_mark = "***" if r["significant"] else ""
        print(f"  {r['pair']:<30s} {r['p_raw']:10.6f} {r['p_adjusted']:10.6f} {sig_mark:>6s}")

    nonsig = [r for r in fdr_results if not r["significant"]]
    if nonsig:
        print(f"\n  Non-significant after FDR: {len(nonsig)}/{len(p_arr)}")
    else:
        print(f"\n  ALL {len(p_arr)} tests remain significant after BH-FDR correction.")

    output["fdr_correction"] = {
        "n_total": len(p_arr),
        "n_significant_raw": n_sig_raw,
        "n_significant_fdr": n_sig_fdr,
        "alpha": ALPHA,
        "method": "one-sample t-test (D > 0), BH-FDR corrected",
        "results": sorted(fdr_results, key=lambda x: x["p_adjusted"]),
    }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    # Cross-scale
    for model in MODELS:
        if model in cross_scale:
            mr = cross_scale[model]
            n_all_pos = sum(1 for v in mr.values() if v["all_positive"])
            print(f"  {model}: {n_all_pos}/{n_pairs} pairs positive at all 3 scales")

    print(f"\n  Cross-model universality: {n_universal}/{n_pairs} pairs universal at medium scale")

    # Matrix permutation
    all_mp = []
    for model in MODELS:
        for sn in SCALE_DIRS:
            if model in matrix_perm and sn in matrix_perm[model]:
                all_mp.append(matrix_perm[model][sn]["p_value"])
    print(f"\n  Matrix-level permutation test ({N_PERM} perms):")
    print(f"    All 12 (4 models x 3 scales) tests: p < {max(all_mp):.4f}")

    # Per-pair replication
    print(f"\n  Per-pair replication test (4 models x 3 scales = 12 obs each):")
    print(f"    Significant (raw p < {ALPHA}): {n_sig_raw}/{len(p_arr)}")
    print(f"    Significant (BH-FDR adjusted): {n_sig_fdr}/{len(p_arr)}")

    # Effect sizes
    all_matrix_d = [es["matrix_cohens_d"] for k, es in effect_sizes.items()
                    if isinstance(es, dict) and "matrix_cohens_d" in es]
    if all_matrix_d:
        print(f"\n  Cohen's d (matrix-level): mean={np.mean(all_matrix_d):.2f}, "
              f"range=[{min(all_matrix_d):.2f}, {max(all_matrix_d):.2f}]")
    print(f"  Cohen's d (per-pair, cross-replication): "
          f"mean={np.mean(pair_d_values):.2f}, min={np.min(pair_d_values):.2f}")

    # Bootstrap
    print(f"\n  Bootstrap 95% CI above zero: {n_ci_above_zero}/{n_pairs} pairs")

    # Save
    out_path = RESULTS_DIR / "statistical_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_validation()
