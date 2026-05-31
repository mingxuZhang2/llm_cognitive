#!/usr/bin/env python3
"""
Agent-state binding probe v3b — fix p >> n overfitting.

Key fix: PCA dimensionality reduction before probing.
Sweep PCA dims: [16, 32, 64, 128, 256, 512] to find sweet spot.
Also adds: permutation baseline (shuffle labels) to calibrate chance.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import random

# Import stimuli generation from v3
import sys
sys.path.insert(0, str(Path(__file__).parent))
from agent_state_probe_v3 import (
    PAIR_TEMPLATES, NAME_POOL, MENTAL_STATES,
    generate_condition_c, extract_agent_activations,
)


PCA_DIMS = [16, 32, 64, 128, 256, 512]


def run_probes_with_pca(X, y_labels, pca_dim, n_splits=5, n_permutations=100):
    """Run linear + MLP probes with PCA, plus permutation baseline."""
    ms_types = sorted(set(y_labels))
    y = np.array([ms_types.index(l) for l in y_labels])
    n_cv = min(n_splits, min(np.bincount(y)))
    if n_cv < 2:
        return {"linear": float("nan"), "mlp": float("nan"),
                "permutation_mean": float("nan"), "permutation_std": float("nan")}

    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)

    # PCA transform
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    actual_dim = min(pca_dim, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=actual_dim, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Linear probe
    clf_lin = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lin_scores = cross_val_score(clf_lin, X_pca, y, cv=skf, scoring="accuracy")

    # MLP probe
    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(min(64, actual_dim),),
        max_iter=1000, early_stopping=True, validation_fraction=0.15,
        random_state=42, learning_rate_init=0.001)
    mlp_scores = cross_val_score(clf_mlp, X_pca, y, cv=skf, scoring="accuracy")

    # Permutation baseline (shuffle labels, re-run linear)
    perm_accs = []
    rng_perm = np.random.RandomState(42)
    for _ in range(n_permutations):
        y_shuf = rng_perm.permutation(y)
        perm_scores = cross_val_score(clf_lin, X_pca, y_shuf, cv=skf,
                                       scoring="accuracy")
        perm_accs.append(float(perm_scores.mean()))

    return {
        "linear": float(lin_scores.mean()),
        "linear_std": float(lin_scores.std()),
        "mlp": float(mlp_scores.mean()),
        "mlp_std": float(mlp_scores.std()),
        "pca_dim": actual_dim,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "permutation_mean": float(np.mean(perm_accs)),
        "permutation_std": float(np.std(perm_accs)),
        "permutation_95th": float(np.percentile(perm_accs, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_per_pair", type=int, default=30)
    parser.add_argument("--layer_stride", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    np.random.seed(42)
    torch.manual_seed(42)

    stim_c = generate_condition_c(rng, n_per_pair=args.n_per_pair)
    print(f"Generated {len(stim_c)} stories ({len(stim_c)*2} agent probes)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    layers_to_test = list(range(0, n_layers + 1, args.layer_stride))
    if n_layers not in layers_to_test:
        layers_to_test.append(n_layers)

    all_results = {}

    for layer_idx in layers_to_test:
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}/{n_layers}")
        print(f"{'='*60}")

        agent_results = extract_agent_activations(
            model, tokenizer, stim_c, device, layer_idx)
        acts = np.array([r["activation"] for r in agent_results])
        labels = [r["mental_state"] for r in agent_results]
        print(f"  Activations: {acts.shape}")

        layer_data = {"layer": layer_idx}

        # Per pair type, sweep PCA dims
        pair_types = sorted(set(r["pair_type"] for r in agent_results))
        for pt in pair_types:
            items = [r for r in agent_results if r["pair_type"] == pt]
            X_pt = np.array([r["activation"] for r in items])
            y_pt = [r["mental_state"] for r in items]

            pt_results = {}
            for pca_dim in PCA_DIMS:
                res = run_probes_with_pca(X_pt, y_pt, pca_dim, n_permutations=200)
                pt_results[pca_dim] = res
                sig = ""
                if res["linear"] > res["permutation_95th"]:
                    sig = " *SIG*"
                print(f"  {pt:25s} PCA={res['pca_dim']:3d}  "
                      f"lin={res['linear']:.3f}  mlp={res['mlp']:.3f}  "
                      f"perm={res['permutation_mean']:.3f}±{res['permutation_std']:.3f}"
                      f"  (95th={res['permutation_95th']:.3f}){sig}")
            layer_data[pt] = pt_results

        # Overall 5-way with best PCA
        for pca_dim in [64, 128]:
            res = run_probes_with_pca(acts, labels, pca_dim, n_permutations=200)
            print(f"  OVERALL 5-way PCA={res['pca_dim']:3d}  "
                  f"lin={res['linear']:.3f}  mlp={res['mlp']:.3f}  "
                  f"perm={res['permutation_mean']:.3f}")
            layer_data[f"overall_pca{pca_dim}"] = res

        all_results[str(layer_idx)] = layer_data

    # Best results summary
    print(f"\n{'='*60}")
    print("BEST RESULT PER PAIR (linear, across all layers & PCA dims)")
    print(f"{'='*60}")
    pair_types = sorted(set(r["pair_type"] for r in agent_results))
    for pt in pair_types:
        best_acc = -1
        best_layer = best_pca = -1
        best_perm_95 = 0
        for lk, ld in all_results.items():
            if pt not in ld:
                continue
            for pca_dim, res in ld[pt].items():
                if res["linear"] > best_acc:
                    best_acc = res["linear"]
                    best_layer = ld["layer"]
                    best_pca = res["pca_dim"]
                    best_perm_95 = res["permutation_95th"]
        sig = "SIG" if best_acc > best_perm_95 else "n.s."
        print(f"  {pt:25s}  best={best_acc:.3f} (L{best_layer}, PCA={best_pca})  "
              f"perm_95={best_perm_95:.3f}  [{sig}]")

    print(f"\n{'='*60}")
    print("BEST RESULT PER PAIR (MLP, across all layers & PCA dims)")
    print(f"{'='*60}")
    for pt in pair_types:
        best_acc = -1
        best_layer = best_pca = -1
        best_perm_95 = 0
        for lk, ld in all_results.items():
            if pt not in ld:
                continue
            for pca_dim, res in ld[pt].items():
                if res["mlp"] > best_acc:
                    best_acc = res["mlp"]
                    best_layer = ld["layer"]
                    best_pca = res["pca_dim"]
                    best_perm_95 = res["permutation_95th"]
        sig = "SIG" if best_acc > best_perm_95 else "n.s."
        print(f"  {pt:25s}  best={best_acc:.3f} (L{best_layer}, PCA={best_pca})  "
              f"perm_95={best_perm_95:.3f}  [{sig}]")

    out_path = out_dir / f"{args.model_short}_agent_state_v3b.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model_short, "n_layers": n_layers,
                    "n_per_pair": args.n_per_pair,
                    "pca_dims_tested": PCA_DIMS,
                    "layers_tested": layers_to_test,
                    "results": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
