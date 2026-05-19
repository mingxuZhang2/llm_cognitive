"""
Extract per-layer attribution statistics from multi_attribution.npz files.
Produces a small JSON with layer-wise neuron counts and selectivity scores
for visualization.
"""

import json
import os
import numpy as np
import sys
from pathlib import Path


def extract_stats(npz_path, meta_path, output_path):
    data = np.load(npz_path)
    with open(meta_path) as f:
        meta = json.load(f)

    n_layers = meta["n_layers"]
    ffn_dim = meta["ffn_dim"]

    categories = sorted(set(k.replace("_importance", "").replace("_selectivity", "")
                           for k in data.files if "_" in k))
    categories = sorted(set(k.rsplit("_", 1)[0] for k in data.files))
    # Deduplicate
    cats = []
    for k in data.files:
        if k.endswith("_selectivity"):
            cats.append(k.replace("_selectivity", ""))
    cats = sorted(set(cats))

    result = {
        "n_layers": n_layers,
        "ffn_dim": ffn_dim,
        "categories": cats,
        "layers": {},
    }

    for layer_idx in range(n_layers):
        start = layer_idx * ffn_dim
        end = start + ffn_dim
        layer_data = {}

        for cat in cats:
            sel_key = f"{cat}_selectivity"
            imp_key = f"{cat}_importance"
            if sel_key not in data:
                continue

            sel = data[sel_key][start:end]
            imp = data[imp_key][start:end]

            # Count neurons at different selectivity thresholds
            layer_data[cat] = {
                "n_selective_03": int((sel > 0.3).sum()),
                "n_selective_05": int((sel > 0.5).sum()),
                "n_selective_07": int((sel > 0.7).sum()),
                "mean_selectivity": float(sel.mean()),
                "max_selectivity": float(sel.max()),
                "mean_importance": float(imp.mean()),
                "top1pct_importance": float(np.percentile(imp, 99)),
            }

        result["layers"][str(layer_idx)] = layer_data

    # Also extract top-5000 neuron layer distribution per category
    result["top5000_layer_dist"] = {}
    for cat in cats:
        sel_key = f"{cat}_selectivity"
        if sel_key not in data:
            continue
        sel = data[sel_key]
        top_indices = np.argsort(-sel)[:5000]
        layer_counts = np.zeros(n_layers, dtype=int)
        for idx in top_indices:
            layer_counts[idx // ffn_dim] += 1
        result["top5000_layer_dist"][cat] = layer_counts.tolist()

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, required=True)
    parser.add_argument("--meta_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    models = [
        "Qwen2.5-7B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
        "gemma-2-9b-it",
    ]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for model in models:
        npz = os.path.join(args.npz_dir, f"{model}_multi_attribution.npz")
        meta = os.path.join(args.meta_dir, f"{model}_meta.json")
        out = os.path.join(args.output_dir, f"{model}_layer_stats.json")
        if os.path.exists(npz) and os.path.exists(meta):
            print(f"Processing {model}...")
            extract_stats(npz, meta, out)
        else:
            print(f"Skipping {model}: files not found")
