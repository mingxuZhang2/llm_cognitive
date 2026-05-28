#!/usr/bin/env python3
"""
Direction B: Does the brain's "cognitive reserve" theory predict which
LLMs are more robust to pruning?

Brain science: distributed representations = higher resilience to damage.
Test: models with more distributed selectivity (lower Gini, higher layer
entropy) should degrade less under pruning.

Uses existing attribution NPZ + pruning JSON data. No GPU needed.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, entropy as shannon_entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
ATTR_DIR = BASE / "results" / "scaled_multi"
PRUNE_DIR = BASE / "results" / "scaled_pruning"
OUT = BASE / "results" / "cognitive_reserve"
FIG = BASE / "figures"

MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]
SHORT = {
    "Qwen2.5-7B-Instruct": "Qwen 7B",
    "Meta-Llama-3.1-8B-Instruct": "Llama 8B",
    "Mistral-7B-Instruct-v0.3": "Mistral 7B",
    "gemma-2-9b-it": "Gemma 9B",
}
CATEGORIES = ["math", "code", "reasoning", "language", "science", "ethics",
              "factual_qa", "humanities"]

# FFN dimensions per model (neurons = n_layers * ffn_dim)
MODEL_SPECS = {
    "Qwen2.5-7B-Instruct":            {"n_layers": 28, "ffn_dim": 18944},
    "Meta-Llama-3.1-8B-Instruct":     {"n_layers": 32, "ffn_dim": 14336},
    "Mistral-7B-Instruct-v0.3":       {"n_layers": 32, "ffn_dim": 14336},
    "gemma-2-9b-it":                   {"n_layers": 42, "ffn_dim": 14336},
}


def gini_coefficient(x):
    """Gini coefficient of array x. 0 = perfectly equal, 1 = maximally unequal."""
    x = np.abs(x)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n)


def effective_number(x):
    """Effective number of neurons (inverse Herfindahl index)."""
    x = np.abs(x)
    total = x.sum()
    if total == 0:
        return 0.0
    p = x / total
    hhi = np.sum(p ** 2)
    return float(1.0 / hhi) if hhi > 0 else 0.0


def layer_entropy(selectivity, n_layers, ffn_dim):
    """Shannon entropy of selectivity distribution across layers."""
    per_layer = np.zeros(n_layers)
    for L in range(n_layers):
        start = L * ffn_dim
        end = start + ffn_dim
        per_layer[L] = np.abs(selectivity[start:end]).sum()
    total = per_layer.sum()
    if total == 0:
        return 0.0
    p = per_layer / total
    p = p[p > 0]
    return float(shannon_entropy(p, base=2))


def top_k_concentration(selectivity, k=5000):
    """Fraction of total selectivity captured by top-k neurons."""
    x = np.abs(selectivity)
    total = x.sum()
    if total == 0:
        return 0.0
    topk = np.sort(x)[-k:]
    return float(topk.sum() / total)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Part 1: Compute distribution metrics per model × category
    # ================================================================
    print("=" * 70)
    print("PART 1: Distribution metrics (cognitive reserve proxy)")
    print("=" * 70)

    dist_metrics = {}

    for model in MODELS:
        attr_path = ATTR_DIR / f"{model}_multi_attribution.npz"
        if not attr_path.exists():
            print(f"[skip] {attr_path}")
            continue
        attr = np.load(attr_path)
        specs = MODEL_SPECS[model]
        n_layers = specs["n_layers"]
        ffn_dim = specs["ffn_dim"]

        dist_metrics[model] = {}
        print(f"\n{SHORT[model]}:")

        for cat in CATEGORIES:
            sel = attr[f"{cat}_selectivity"]

            g = gini_coefficient(sel)
            eff_n = effective_number(sel)
            l_ent = layer_entropy(sel, n_layers, ffn_dim)
            max_ent = np.log2(n_layers)
            conc = top_k_concentration(sel, k=5000)

            dist_metrics[model][cat] = {
                "gini": g,
                "effective_n": eff_n,
                "layer_entropy": l_ent,
                "layer_entropy_norm": l_ent / max_ent if max_ent > 0 else 0,
                "top5k_concentration": conc,
                "n_neurons_total": len(sel),
            }

            print(f"  {cat:>12s}: gini={g:.4f}  eff_n={eff_n:.0f}  "
                  f"layer_ent={l_ent:.3f}/{max_ent:.2f}  top5k_conc={conc:.4f}")

        # Model-level aggregate
        agg = {}
        for metric in ["gini", "effective_n", "layer_entropy_norm", "top5k_concentration"]:
            vals = [dist_metrics[model][c][metric] for c in CATEGORIES]
            agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        dist_metrics[model]["_aggregate"] = agg
        print(f"  AGGREGATE: gini={agg['gini']['mean']:.4f}  "
              f"eff_n={agg['effective_n']['mean']:.0f}  "
              f"layer_ent_norm={agg['layer_entropy_norm']['mean']:.3f}")

    # ================================================================
    # Part 2: Load pruning resilience
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: Pruning resilience")
    print("=" * 70)

    prune_results = {}
    for model in MODELS:
        prune_path = PRUNE_DIR / f"{model}_atlas_pruning.json"
        if not prune_path.exists():
            print(f"[skip] {prune_path}")
            continue
        with open(prune_path) as f:
            p = json.load(f)

        prune_results[model] = {}
        baselines = p["baseline"]
        atlas_runs = p["methods"].get("Atlas-guided", p["methods"].get("atlas", []))
        sparsities = p["sparsity_levels"]

        print(f"\n{SHORT[model]}:")
        for cat in CATEGORIES:
            if cat not in baselines:
                continue
            base_ppl = baselines[cat]["perplexity"]
            ratios = []
            for sp in sparsities:
                found = False
                for run in atlas_runs:
                    if abs(run["sparsity"] - sp) < 0.01 and cat in run.get("per_category", {}):
                        pruned_ppl = run["per_category"][cat]["perplexity"]
                        ratios.append(pruned_ppl / base_ppl)
                        found = True
                        break
                if not found:
                    ratios.append(float("nan"))

            # Resilience = inverse of mean PPL ratio (lower ratio = more resilient)
            valid = [r for r in ratios if not np.isnan(r)]
            mean_ratio = np.mean(valid) if valid else float("nan")
            # Also: ratio at 50% sparsity specifically
            ratio_50 = ratios[-1] if len(ratios) == len(sparsities) else float("nan")

            prune_results[model][cat] = {
                "ppl_ratios": ratios,
                "mean_ppl_ratio": float(mean_ratio),
                "ratio_at_50pct": float(ratio_50),
                "baseline_ppl": float(base_ppl),
            }
            print(f"  {cat:>12s}: baseline PPL={base_ppl:.2f}  "
                  f"50% ratio={ratio_50:.3f}  mean ratio={mean_ratio:.3f}")

    # ================================================================
    # Part 3: Correlate distribution metrics with pruning resilience
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 3: Cognitive reserve predicts pruning resilience?")
    print("=" * 70)

    # Collect paired data: (distribution metric, pruning resilience) per model×category
    pairs = []
    for model in MODELS:
        if model not in dist_metrics or model not in prune_results:
            continue
        for cat in CATEGORIES:
            if cat not in dist_metrics[model] or cat not in prune_results[model]:
                continue
            dm = dist_metrics[model][cat]
            pr = prune_results[model][cat]
            if np.isnan(pr["ratio_at_50pct"]):
                continue
            pairs.append({
                "model": SHORT[model],
                "category": cat,
                "gini": dm["gini"],
                "effective_n": dm["effective_n"],
                "layer_entropy_norm": dm["layer_entropy_norm"],
                "top5k_concentration": dm["top5k_concentration"],
                "ppl_ratio_50": pr["ratio_at_50pct"],
                "mean_ppl_ratio": pr["mean_ppl_ratio"],
            })

    if not pairs:
        print("No paired data available!")
        return

    # Test each distribution metric vs pruning resilience
    metrics_to_test = [
        ("gini", "Gini coefficient", "positive",
         "High Gini = concentrated = LESS resilient → PPL ratio should be HIGH"),
        ("layer_entropy_norm", "Layer entropy (normalized)", "negative",
         "High entropy = distributed = MORE resilient → PPL ratio should be LOW"),
        ("top5k_concentration", "Top-5K concentration", "positive",
         "High concentration = less distributed → PPL ratio should be HIGH"),
        ("effective_n", "Effective number of neurons", "negative",
         "High eff_n = distributed → PPL ratio should be LOW"),
    ]

    print(f"\n  N = {len(pairs)} paired observations (model × category)")
    correlations = {}
    for metric, name, expected_sign, rationale in metrics_to_test:
        x = [p[metric] for p in pairs]
        y = [p["ppl_ratio_50"] for p in pairs]
        rho, p_val = spearmanr(x, y)
        sign_match = (expected_sign == "positive" and rho > 0) or \
                     (expected_sign == "negative" and rho < 0)
        correlations[metric] = {"rho": rho, "p": p_val, "expected": expected_sign,
                                "sign_match": sign_match}
        status = "✓ SUPPORTS" if sign_match else "✗ CONTRADICTS"
        print(f"\n  {name}:")
        print(f"    {rationale}")
        print(f"    Spearman ρ = {rho:+.4f}, p = {p_val:.4f}")
        print(f"    Expected {expected_sign} → {status}")

    # Model-level aggregate test
    print("\n\n  MODEL-LEVEL AGGREGATE TEST:")
    model_gini = []
    model_resilience = []
    for model in MODELS:
        if model not in dist_metrics or model not in prune_results:
            continue
        agg_gini = dist_metrics[model]["_aggregate"]["gini"]["mean"]
        cat_ratios = [prune_results[model][c]["ratio_at_50pct"]
                      for c in CATEGORIES if c in prune_results[model]
                      and not np.isnan(prune_results[model][c]["ratio_at_50pct"])]
        if cat_ratios:
            model_gini.append(agg_gini)
            model_resilience.append(np.mean(cat_ratios))
            print(f"    {SHORT[model]:>12s}: gini={agg_gini:.4f}  "
                  f"mean PPL ratio@50%={np.mean(cat_ratios):.3f}")

    if len(model_gini) >= 3:
        rho_model, p_model = spearmanr(model_gini, model_resilience)
        print(f"\n    Model-level: gini vs resilience ρ={rho_model:+.4f}  p={p_model:.4f}")
        print(f"    {'SUPPORTS' if rho_model > 0 else 'CONTRADICTS'} cognitive reserve theory")

    # ================================================================
    # Figures
    # ================================================================

    # Figure 1: Scatter plots of distribution metrics vs pruning resilience
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    model_colors = {"Qwen 7B": "tab:red", "Llama 8B": "tab:blue",
                    "Mistral 7B": "tab:green", "Gemma 9B": "tab:orange"}

    for ax, (metric, name, expected, _) in zip(axes.flat, metrics_to_test):
        for p in pairs:
            ax.scatter(p[metric], p["ppl_ratio_50"],
                       c=model_colors.get(p["model"], "gray"),
                       s=50, alpha=0.7, label=p["model"])
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        rho = correlations[metric]["rho"]
        p_val = correlations[metric]["p"]
        ax.set_xlabel(name)
        ax.set_ylabel("PPL ratio at 50% sparsity")
        ax.set_title(f"ρ={rho:+.3f}, p={p_val:.3f} (expect {expected})")
        ax.grid(alpha=0.3)

    plt.suptitle("Direction B: Cognitive reserve predicts pruning resilience?",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG / "cognitive_reserve_scatter.png", dpi=150)
    plt.close()
    print(f"\nSaved figure: {FIG / 'cognitive_reserve_scatter.png'}")

    # Figure 2: Model-level comparison
    if len(model_gini) >= 3:
        fig, ax = plt.subplots(figsize=(7, 5))
        for g, r, model in zip(model_gini, model_resilience, MODELS):
            ax.scatter(g, r, s=150, zorder=3)
            ax.annotate(SHORT[model], (g, r), fontsize=11,
                        ha="center", va="bottom", xytext=(0, 8),
                        textcoords="offset points")
        ax.set_xlabel("Mean Gini coefficient (higher = more concentrated)")
        ax.set_ylabel("Mean PPL ratio at 50% sparsity (higher = less resilient)")
        ax.set_title("Model-level: concentrated representations → worse pruning")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / "cognitive_reserve_model_level.png", dpi=150)
        plt.close()
        print(f"Saved figure: {FIG / 'cognitive_reserve_model_level.png'}")

    # Save all results
    final = {
        "distribution_metrics": {SHORT.get(m, m): dist_metrics[m] for m in dist_metrics},
        "pruning_resilience": {SHORT.get(m, m): prune_results[m] for m in prune_results},
        "correlations": correlations,
        "model_level": {
            "gini": model_gini,
            "resilience": model_resilience,
            "models": [SHORT[m] for m in MODELS if m in dist_metrics],
        },
        "n_pairs": len(pairs),
    }
    with open(OUT / "cognitive_reserve.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    print(f"\nSaved results: {OUT / 'cognitive_reserve.json'}")


if __name__ == "__main__":
    main()
