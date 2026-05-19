"""
Structural analysis of functional atlas to surface genuine findings.

1. Hub neurons: neurons important for 3+ functions ("rich club")
2. Functional interference: ablating X improves Y (resource competition)
3. Specialist/Generalist distribution: how specialized are neurons?
4. Layer-wise specialization gradient
5. Functional overlap: which function pairs share the most neurons?
"""

import json
import os
import numpy as np
from pathlib import Path
from scipy.stats import entropy

HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    pass


MODELS = [
    ("Qwen2.5-7B-Instruct", "Qwen-7B"),
    ("Meta-Llama-3.1-8B-Instruct", "LLaMA-8B"),
    ("Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ("gemma-2-9b-it", "Gemma-9B"),
]

CAT_ORDER = ["math", "code", "reasoning", "language", "science", "ethics", "factual_qa", "humanities"]


def load_attributions(attr_dir):
    """Load attribution npz files."""
    data = {}
    for model_full, model_short in MODELS:
        path = os.path.join(attr_dir, f"{model_full}_multi_attribution.npz")
        if os.path.exists(path):
            data[model_short] = np.load(path)
    return data


def load_dissociation(dissoc_dir):
    data = {}
    for model_full, model_short in MODELS:
        path = os.path.join(dissoc_dir, f"{model_full}_multi_dissociation.json")
        if os.path.exists(path):
            with open(path) as f:
                data[model_short] = json.load(f)
    return data


def load_layer_stats(stats_dir):
    data = {}
    for model_full, model_short in MODELS:
        path = os.path.join(stats_dir, f"{model_full}_layer_stats.json")
        if os.path.exists(path):
            with open(path) as f:
                data[model_short] = json.load(f)
    return data


def analysis_hub_neurons(attr_data, output_dir):
    """Find neurons important for multiple functions."""
    print("\n" + "="*70)
    print("FINDING 1: Hub Neurons — Multi-Function 'Rich Club'")
    print("="*70)

    if HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()

    for idx, (model_short, npz) in enumerate(attr_data.items()):
        cats = [c for c in CAT_ORDER if f"{c}_importance" in npz.files]
        n_neurons = len(npz[f"{cats[0]}_importance"])

        # For each neuron: count how many functions it's in top-5000 for
        top_sets = {}
        for cat in cats:
            sel = npz[f"{cat}_selectivity"]
            top_sets[cat] = set(np.argsort(-sel)[:5000])

        hub_count = np.zeros(n_neurons, dtype=int)
        for cat in cats:
            for n_idx in top_sets[cat]:
                hub_count[n_idx] += 1

        # Distribution
        count_dist = {}
        for c in range(0, len(cats)+1):
            count_dist[c] = int((hub_count == c).sum())

        print(f"\n  {model_short}: {n_neurons:,} neurons, {len(cats)} functions")
        print(f"    # functions served by each neuron:")
        for c in range(len(cats)+1):
            pct = count_dist.get(c, 0) / n_neurons * 100
            bar = "█" * int(pct)
            print(f"      {c} functions: {count_dist.get(c, 0):>7,} ({pct:>5.1f}%) {bar}")

        # Hub neurons (3+ functions)
        hubs_3plus = np.where(hub_count >= 3)[0]
        hubs_4plus = np.where(hub_count >= 4)[0]
        hubs_5plus = np.where(hub_count >= 5)[0]
        print(f"    Hub neurons (≥3 functions): {len(hubs_3plus):,} ({len(hubs_3plus)/n_neurons*100:.2f}%)")
        print(f"    Hub neurons (≥4 functions): {len(hubs_4plus):,} ({len(hubs_4plus)/n_neurons*100:.2f}%)")
        print(f"    Hub neurons (≥5 functions): {len(hubs_5plus):,} ({len(hubs_5plus)/n_neurons*100:.2f}%)")

        # Hub layer distribution
        n_layers = n_neurons // (n_neurons // len(npz[f"{cats[0]}_importance"]))
        # Infer from layer_stats
        ffn_dim = None
        for possible_layers in [28, 32, 42]:
            if n_neurons % possible_layers == 0:
                ffn_dim = n_neurons // possible_layers
                n_layers = possible_layers
                break
        if ffn_dim is None:
            ffn_dim = n_neurons // 28
            n_layers = 28

        hub_per_layer = np.zeros(n_layers)
        for h in hubs_3plus:
            hub_per_layer[h // ffn_dim] += 1

        print(f"    Hub concentration (≥3 func) per layer:")
        max_hub_layer = hub_per_layer.argmax()
        for l in range(n_layers):
            if hub_per_layer[l] > 0:
                bar = "█" * int(hub_per_layer[l] / 5)
                marker = " ← PEAK" if l == max_hub_layer else ""
                if hub_per_layer[l] > hub_per_layer.mean() * 1.5:
                    print(f"      L{l:02d}: {int(hub_per_layer[l]):>4} {bar}{marker}")

        # Total importance carried by hubs
        total_imp = sum(npz[f"{cat}_importance"].sum() for cat in cats)
        hub_imp = sum(npz[f"{cat}_importance"][hubs_3plus].sum() for cat in cats)
        print(f"    Hub importance: {hub_imp/total_imp*100:.1f}% of total (in {len(hubs_3plus)/n_neurons*100:.2f}% of neurons)")

        # Plot
        if HAS_MPL:
            ax = axes_flat[idx]
            counts = [count_dist.get(c, 0) for c in range(len(cats)+1)]
            ax.bar(range(len(cats)+1), counts, color=["#ccc"] + ["#4daf4a"]*2 + ["#e41a1c"]*(len(cats)-2))
            ax.set_xlabel("# functions served")
            ax.set_ylabel("# neurons")
            ax.set_title(model_short, fontweight="bold")
            ax.set_yscale("log")

    if HAS_MPL:
        fig.suptitle("Hub Neuron Distribution: How Many Functions Does Each Neuron Serve?",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "hub_neurons.png"), dpi=200, bbox_inches="tight")
        print(f"\n  Saved: hub_neurons.png")


def analysis_interference(dissoc_data, output_dir):
    """Find functional interference: ablating X improves Y."""
    print("\n" + "="*70)
    print("FINDING 2: Functional Interference — Resource Competition")
    print("="*70)
    print("  Negative spillover = ablating X IMPROVES Y = X steals resources from Y")

    all_matrices = {}
    for model_short, data in dissoc_data.items():
        cats = data["categories"]
        cat_idx = {c: cats.index(c) for c in CAT_ORDER if c in cats}
        ordered = [c for c in CAT_ORDER if c in cat_idx]
        order = [cat_idx[c] for c in ordered]
        matrix = np.array(data["dissociation_matrix_logppl_delta"])
        matrix = matrix[np.ix_(order, order)]
        all_matrices[model_short] = (ordered, matrix)

    # Find consistent negative spillovers
    print("\n  Consistent facilitation effects (ablate X → Y improves, across models):")
    n = len(CAT_ORDER)
    facilitation_counts = {}
    for model_short, (ordered, matrix) in all_matrices.items():
        for i in range(len(ordered)):
            for j in range(len(ordered)):
                if i != j and matrix[i, j] < -0.02:
                    key = (ordered[i], ordered[j])
                    if key not in facilitation_counts:
                        facilitation_counts[key] = []
                    facilitation_counts[key].append((model_short, matrix[i, j]))

    for (src, tgt), occurrences in sorted(facilitation_counts.items(), key=lambda x: -len(x[1])):
        if len(occurrences) >= 2:
            models_str = ", ".join([f"{m}({v:+.3f})" for m, v in occurrences])
            print(f"    Ablate {src:>12s} → {tgt:<12s} IMPROVES ({len(occurrences)}/4 models): {models_str}")

    # Asymmetric relationships: A hurts B but B helps A (or vice versa)
    print("\n  Asymmetric relationships (A→B positive, B→A negative):")
    for model_short, (ordered, matrix) in all_matrices.items():
        for i in range(len(ordered)):
            for j in range(i+1, len(ordered)):
                a_to_b = matrix[i, j]
                b_to_a = matrix[j, i]
                if (a_to_b > 0.03 and b_to_a < -0.02) or (a_to_b < -0.02 and b_to_a > 0.03):
                    print(f"    {model_short}: {ordered[i]:>12s} → {ordered[j]}: {a_to_b:+.3f}, "
                          f"{ordered[j]:>12s} → {ordered[i]}: {b_to_a:+.3f}")


def analysis_specialist_generalist(attr_data, output_dir):
    """Analyze specialist (1 function) vs generalist (many functions) neurons."""
    print("\n" + "="*70)
    print("FINDING 3: Specialist vs Generalist Neuron Distribution")
    print("="*70)

    for idx, (model_short, npz) in enumerate(attr_data.items()):
        cats = [c for c in CAT_ORDER if f"{c}_importance" in npz.files]
        n_neurons = len(npz[f"{cats[0]}_importance"])

        # For each neuron, compute its functional entropy
        # High entropy = generalist (uniform across functions)
        # Low entropy = specialist (concentrated on 1-2 functions)
        importances = np.stack([npz[f"{cat}_importance"] for cat in cats], axis=1)  # [n_neurons, n_cats]

        # Normalize to probability distribution per neuron
        imp_sum = importances.sum(axis=1, keepdims=True)
        imp_sum[imp_sum == 0] = 1
        imp_probs = importances / imp_sum

        # Shannon entropy per neuron
        neuron_entropy = np.array([entropy(imp_probs[i] + 1e-10) for i in range(n_neurons)])
        max_entropy = np.log(len(cats))

        # Normalized entropy (0 = pure specialist, 1 = pure generalist)
        norm_entropy = neuron_entropy / max_entropy

        # Statistics
        pct_specialist = (norm_entropy < 0.3).sum() / n_neurons * 100
        pct_moderate = ((norm_entropy >= 0.3) & (norm_entropy < 0.7)).sum() / n_neurons * 100
        pct_generalist = (norm_entropy >= 0.7).sum() / n_neurons * 100

        print(f"\n  {model_short}:")
        print(f"    Specialist (entropy < 0.3): {pct_specialist:.1f}%")
        print(f"    Moderate (0.3-0.7):         {pct_moderate:.1f}%")
        print(f"    Generalist (entropy > 0.7): {pct_generalist:.1f}%")
        print(f"    Mean entropy: {norm_entropy.mean():.3f}")

        # For specialists: what's their dominant function?
        dominant_func = np.argmax(importances, axis=1)
        specialists = np.where(norm_entropy < 0.3)[0]
        if len(specialists) > 0:
            func_counts = {}
            for s in specialists:
                func = cats[dominant_func[s]]
                func_counts[func] = func_counts.get(func, 0) + 1
            print(f"    Specialist breakdown (top functions):")
            for func, count in sorted(func_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"      {func:>12s}: {count:,} specialists")

        # Layer-wise entropy gradient
        ffn_dim = None
        for possible_layers in [28, 32, 42]:
            if n_neurons % possible_layers == 0:
                ffn_dim = n_neurons // possible_layers
                n_layers = possible_layers
                break
        if ffn_dim:
            layer_entropy = []
            for l in range(n_layers):
                s, e = l * ffn_dim, (l+1) * ffn_dim
                layer_entropy.append(norm_entropy[s:e].mean())
            print(f"    Layer-wise specialization gradient:")
            print(f"      Early layers (0-{n_layers//4}): avg entropy = {np.mean(layer_entropy[:n_layers//4]):.3f}")
            print(f"      Middle layers ({n_layers//4}-{3*n_layers//4}): avg entropy = {np.mean(layer_entropy[n_layers//4:3*n_layers//4]):.3f}")
            print(f"      Late layers ({3*n_layers//4}-{n_layers}): avg entropy = {np.mean(layer_entropy[3*n_layers//4:]):.3f}")

            # Is there a significant gradient?
            early = np.mean(layer_entropy[:n_layers//4])
            late = np.mean(layer_entropy[3*n_layers//4:])
            if early > late + 0.02:
                print(f"      → GRADIENT: early layers more GENERALIST, late layers more SPECIALIST")
            elif late > early + 0.02:
                print(f"      → GRADIENT: early layers more SPECIALIST, late layers more GENERALIST")
            else:
                print(f"      → No clear gradient")

        pass  # plots generated locally


def analysis_functional_overlap(attr_data, output_dir):
    """Which function pairs share the most top neurons? Jaccard similarity."""
    print("\n" + "="*70)
    print("FINDING 4: Functional Overlap — Shared Neural Substrate")
    print("="*70)

    for idx, (model_short, npz) in enumerate(attr_data.items()):
        cats = [c for c in CAT_ORDER if f"{c}_selectivity" in npz.files]
        n = len(cats)

        # Top-5000 neurons per function
        top_sets = {}
        for cat in cats:
            sel = npz[f"{cat}_selectivity"]
            top_sets[cat] = set(np.argsort(-sel)[:5000])

        # Jaccard similarity matrix
        jaccard = np.zeros((n, n))
        overlap_counts = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                intersection = len(top_sets[cats[i]] & top_sets[cats[j]])
                union = len(top_sets[cats[i]] | top_sets[cats[j]])
                jaccard[i, j] = intersection / union if union > 0 else 0
                overlap_counts[i, j] = intersection

        print(f"\n  {model_short}: Top-5000 neuron overlap (Jaccard similarity):")
        header = f"    {'':>12s}" + "".join(f" {c:>8s}" for c in cats)
        print(header)
        for i in range(n):
            row = f"    {cats[i]:>12s}"
            for j in range(n):
                if i == j:
                    row += f" {'---':>8s}"
                else:
                    row += f" {jaccard[i,j]:>8.3f}"
            print(row)

        # Highest overlaps
        print(f"    Top 5 overlapping pairs:")
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((cats[i], cats[j], jaccard[i, j], overlap_counts[i, j]))
        pairs.sort(key=lambda x: -x[2])
        for c1, c2, j, ov in pairs[:5]:
            print(f"      {c1:>12s} ∩ {c2:<12s}: Jaccard={j:.3f}, {ov} shared neurons")

        pass  # plots generated locally


def analysis_layer_specialization(attr_data, output_dir):
    """How does functional specialization change across depth?"""
    print("\n" + "="*70)
    print("FINDING 5: Layer-wise Specialization Gradient")
    print("="*70)

    for idx, (model_short, npz) in enumerate(attr_data.items()):
        cats = [c for c in CAT_ORDER if f"{c}_importance" in npz.files]
        n_neurons = len(npz[f"{cats[0]}_importance"])
        importances = np.stack([npz[f"{cat}_importance"] for cat in cats], axis=1)

        ffn_dim = None
        n_layers = None
        for possible_layers in [28, 32, 42]:
            if n_neurons % possible_layers == 0:
                ffn_dim = n_neurons // possible_layers
                n_layers = possible_layers
                break
        if not ffn_dim:
            continue

        # Per-layer metrics
        layer_max_selectivity = []  # max selectivity in each layer
        layer_mean_importance = []  # total importance per layer
        layer_gini = []  # Gini coefficient of importance distribution

        for l in range(n_layers):
            s, e = l * ffn_dim, (l+1) * ffn_dim
            layer_imp = importances[s:e]

            # Mean importance (how "active" is this layer?)
            layer_mean_importance.append(layer_imp.mean())

            # Max selectivity per neuron in this layer
            imp_sum = layer_imp.sum(axis=1, keepdims=True)
            imp_sum[imp_sum == 0] = 1
            probs = layer_imp / imp_sum
            max_prob = probs.max(axis=1)
            layer_max_selectivity.append(max_prob.mean())

            # Gini coefficient of total importance
            flat = layer_imp.sum(axis=1)
            flat_sorted = np.sort(flat)
            n_f = len(flat_sorted)
            cum = np.cumsum(flat_sorted)
            gini = 1 - 2 * cum.sum() / (n_f * cum[-1]) if cum[-1] > 0 else 0
            layer_gini.append(gini)

        print(f"\n  {model_short} ({n_layers} layers):")
        print(f"    {'Layer':>8s}  {'Importance':>10s}  {'Selectivity':>12s}  {'Gini':>8s}")
        for l in range(n_layers):
            marker = ""
            if layer_mean_importance[l] > np.mean(layer_mean_importance) * 1.5:
                marker = " ← HIGH IMP"
            print(f"    L{l:02d}       {layer_mean_importance[l]:>10.6f}  {layer_max_selectivity[l]:>12.4f}  {layer_gini[l]:>8.4f}{marker}")

        pass  # plots generated locally


def main(attr_dir, dissoc_dir, stats_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    attr_data = load_attributions(attr_dir)
    dissoc_data = load_dissociation(dissoc_dir)

    analysis_hub_neurons(attr_data, output_dir)
    analysis_interference(dissoc_data, output_dir)
    analysis_specialist_generalist(attr_data, output_dir)
    analysis_functional_overlap(attr_data, output_dir)
    analysis_layer_specialization(attr_data, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr_dir", type=str, required=True)
    parser.add_argument("--dissoc_dir", type=str, required=True)
    parser.add_argument("--stats_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.attr_dir, args.dissoc_dir, args.stats_dir, args.output_dir)
