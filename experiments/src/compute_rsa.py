"""
Compute representational similarity (Spearman of upper-triangle RDM entries)
between brain RDM and LLM RDM per layer, with permutation null.

Inputs:
  --brain_rdm_npz       brain_rdm.npz (output of build_brain_rdm.py)
  --activations_npz     {model_short}_rsa_activations.npz
                          mean_activations [n_cond, n_layers+1, hidden_dim]
  --output_path         {model_short}_rsa.json
  --n_permutations      default 10000

Logic:
  For each layer L:
      llm_rdm_L = 1 - Pearson(mean_activations[:, L, :])
      triu_brain  = brain_rdm[triu_indices]
      triu_llm_L  = llm_rdm_L[triu_indices]
      observed_rho = Spearman(triu_brain, triu_llm_L)
      For 10K iterations:
        shuffle condition labels on llm side -> permuted rho
      p-value = (# permuted >= observed) / N

Outputs (JSON):
  model_short
  conditions
  n_layers
  per_layer:
    [{layer_index, layer_name, rho, p_value, pair_preserved_top, ...}]
  peak_layer
  llm_rdms     - saved separately as .npz alongside
  preserved_pairs - top pairs that are similar in both brain and LLM
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def llm_rdm_from_activations(act):
    """act: [n_cond, hidden_dim] -> [n_cond, n_cond] RDM (1 - Pearson)."""
    X = act - act.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    corr = Xn @ Xn.T
    return 1.0 - corr


def spearman_triu(rdm_a, rdm_b):
    n = rdm_a.shape[0]
    triu = np.triu_indices(n, k=1)
    a = rdm_a[triu]
    b = rdm_b[triu]
    rho, _ = spearmanr(a, b)
    return rho


def permutation_null(brain_rdm, llm_rdm, n_perm=10000, rng=None):
    """Permute condition labels on the LLM RDM, recompute Spearman, return null."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = brain_rdm.shape[0]
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        llm_perm = llm_rdm[np.ix_(perm, perm)]
        null[i] = spearman_triu(brain_rdm, llm_perm)
    return null


def run(brain_rdm_npz, activations_npz, output_path, n_permutations=10000):
    t0 = time.time()
    brain = np.load(brain_rdm_npz, allow_pickle=True)
    brain_rdm = brain["rdm"]
    brain_conds = list(brain["conditions"])
    print(f"Brain RDM: {brain_rdm.shape}, conditions={len(brain_conds)}")

    llm = np.load(activations_npz, allow_pickle=True)
    mean_act = llm["mean_activations"]  # [n_cond, n_layers+1, hidden_dim]
    llm_conds = list(llm["conditions"])
    layer_names = list(llm["layer_names"])
    model_short = str(llm["model_short"])
    print(f"LLM activations: {mean_act.shape}  model={model_short}")

    # Reorder LLM to match brain condition order
    assert sorted(llm_conds) == sorted(brain_conds), \
        f"condition mismatch: {set(llm_conds)} vs {set(brain_conds)}"
    order = [llm_conds.index(c) for c in brain_conds]
    mean_act = mean_act[order]
    conditions = brain_conds

    n_layers = mean_act.shape[1]
    n_perm = n_permutations
    rng = np.random.default_rng(20260525)

    print(f"\nComputing per-layer RSA (Spearman of triu(brain_rdm) vs triu(llm_rdm))")
    print(f"  n_permutations per layer: {n_perm}")
    print()
    per_layer = []
    llm_rdms = np.empty((n_layers, len(conditions), len(conditions)), dtype=np.float64)

    for L in range(n_layers):
        llm_rdm = llm_rdm_from_activations(mean_act[:, L, :].astype(np.float64))
        llm_rdms[L] = llm_rdm
        rho = spearman_triu(brain_rdm, llm_rdm)
        null = permutation_null(brain_rdm, llm_rdm, n_perm=n_perm, rng=rng)
        p = float(((null >= rho).sum() + 1) / (n_perm + 1))
        per_layer.append({
            "layer_index": L,
            "layer_name": layer_names[L],
            "rho": float(rho),
            "p_value": p,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
        })
        if L % max(1, n_layers // 10) == 0 or L == n_layers - 1:
            print(f"  layer {L:>3d} ({layer_names[L]:>10s}): rho={rho:+.4f}  "
                  f"p={p:.4g}  null mean={null.mean():+.3f} std={null.std():.3f}")

    # Peak layer
    peak = max(per_layer, key=lambda r: r["rho"])
    print(f"\nPeak alignment layer: {peak['layer_name']} (idx {peak['layer_index']})  "
          f"rho={peak['rho']:.4f}  p={peak['p_value']:.4g}")

    # Off-diagonal pair analysis at peak layer
    peak_rdm = llm_rdms[peak["layer_index"]]
    triu = np.triu_indices(len(conditions), k=1)
    pairs_data = []
    for i, j in zip(triu[0], triu[1]):
        pairs_data.append({
            "pair": [conditions[i], conditions[j]],
            "brain_rdm": float(brain_rdm[i, j]),
            "llm_rdm_peak": float(peak_rdm[i, j]),
        })

    # Pairs that are similar in BOTH brain and LLM (low RDM in both)
    pairs_data_sorted = sorted(pairs_data, key=lambda p: p["brain_rdm"] + p["llm_rdm_peak"])
    preserved_similar = pairs_data_sorted[:10]
    preserved_dissimilar = sorted(pairs_data,
        key=lambda p: -(p["brain_rdm"] + p["llm_rdm_peak"]))[:10]
    # Pairs where they DISAGREE most
    disagreement = sorted(pairs_data,
        key=lambda p: -abs(p["brain_rdm"] - p["llm_rdm_peak"]))[:10]

    print(f"\nTop 10 PRESERVED-SIMILAR pairs (low rdm in both brain & LLM peak):")
    for p in preserved_similar:
        print(f"  {p['pair'][0]:>16s} <-> {p['pair'][1]:<16s}  "
              f"brain={p['brain_rdm']:.3f}  llm={p['llm_rdm_peak']:.3f}")

    print(f"\nTop 10 PRESERVED-DISSIMILAR pairs (high rdm in both):")
    for p in preserved_dissimilar:
        print(f"  {p['pair'][0]:>16s} <-> {p['pair'][1]:<16s}  "
              f"brain={p['brain_rdm']:.3f}  llm={p['llm_rdm_peak']:.3f}")

    print(f"\nTop 10 DISAGREEMENT pairs (large |brain - llm|):")
    for p in disagreement:
        diff = p['brain_rdm'] - p['llm_rdm_peak']
        print(f"  {p['pair'][0]:>16s} <-> {p['pair'][1]:<16s}  "
              f"brain={p['brain_rdm']:.3f}  llm={p['llm_rdm_peak']:.3f}  diff={diff:+.3f}")

    # Save
    out_json = {
        "model_short": model_short,
        "conditions": conditions,
        "n_layers": n_layers,
        "n_permutations": n_perm,
        "per_layer": per_layer,
        "peak_layer": peak,
        "preserved_similar_pairs": preserved_similar,
        "preserved_dissimilar_pairs": preserved_dissimilar,
        "disagreement_pairs": disagreement,
        "elapsed_seconds": time.time() - t0,
    }
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved RSA JSON to {out_p}")

    # Save LLM RDMs as separate npz alongside
    rdm_path = out_p.with_suffix("").parent / f"{out_p.stem}_llm_rdms.npz"
    np.savez_compressed(rdm_path, llm_rdms=llm_rdms, conditions=np.array(conditions),
                        layer_names=np.array(layer_names))
    print(f"Saved LLM RDMs to {rdm_path}")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain_rdm_npz", required=True)
    ap.add_argument("--activations_npz", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--n_permutations", type=int, default=10000)
    args = ap.parse_args()
    run(args.brain_rdm_npz, args.activations_npz, args.output_path,
        n_permutations=args.n_permutations)


if __name__ == "__main__":
    main()
