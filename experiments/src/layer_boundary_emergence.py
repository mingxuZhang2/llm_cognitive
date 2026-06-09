#!/usr/bin/env python3
"""
Layer-wise boundary emergence: WHERE in the network does the emotion-social
boundary form, and how does it differ from within-block structure?

Uses pre-extracted per-layer RSA data from rsa_v2 results. For each model,
at each layer, decomposes the brain-LLM RSA into:
  1. Block-boundary component (binary emotion/social split)
  2. Within-social fine structure
  3. Within-affective fine structure

This tells us: does the boundary form early (embedding-like) or late (deep
processing)? Does within-social track the boundary or emerge independently?
Does within-affective ever align at any layer?

Output:
  results/mechanistic/layer_boundary_emergence.json
  figures/layer_boundary_emergence.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import lstsq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "mechanistic"
FIG_DIR = BASE / "figures"
OUT_DIR.mkdir(exist_ok=True)

AFF = ["anger", "fear", "disgust", "sadness", "happiness", "valence"]
MENT = ["judgment", "belief", "intention", "mentalizing", "moral", "empathy",
        "self_referential", "theory_of_mind"]
ORDER = AFF + MENT

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
MSHORT = {"Qwen2.5-7B-Instruct": "Qwen-7B", "Meta-Llama-3.1-8B-Instruct": "Llama-8B",
          "Mistral-7B-Instruct-v0.3": "Mistral-7B", "gemma-2-9b-it": "Gemma-9B"}
PEAK_LAYERS = {"Qwen2.5-7B-Instruct": 27, "Meta-Llama-3.1-8B-Instruct": 31,
               "Mistral-7B-Instruct-v0.3": 14, "gemma-2-9b-it": 21}

C_AFF, C_MENT, C_BLUE = "#e8743b", "#19a979", "#2e5cb8"


def triu_vec(mat):
    iu = np.triu_indices(mat.shape[0], 1)
    return mat[iu]


def reorder_rdm(rdm, conds, order):
    idx = [conds.index(c) for c in order]
    return rdm[np.ix_(idx, idx)]


def within_block_rho(rdm, ref_rdm, block_indices):
    sub = rdm[np.ix_(block_indices, block_indices)]
    ref_sub = ref_rdm[np.ix_(block_indices, block_indices)]
    sv = triu_vec(sub)
    rv = triu_vec(ref_sub)
    if np.std(sv) < 1e-12 or np.std(rv) < 1e-12:
        return 0.0
    return float(spearmanr(sv, rv)[0])


def main():
    print("=" * 70)
    print("LAYER-WISE BOUNDARY EMERGENCE")
    print("=" * 70)

    # Load brain RDM
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_rdm_raw, brain_conds = ns["rdm"], list(ns["conditions"])
    brain_rdm = reorder_rdm(brain_rdm_raw, brain_conds, ORDER)
    brain_vec = triu_vec(brain_rdm)

    # Binary block RDM
    block_labels = np.array([0 if c in AFF else 1 for c in ORDER]).reshape(-1, 1)
    block_rdm = squareform(pdist(block_labels, metric="euclidean"))
    block_vec = triu_vec(block_rdm)

    aff_idx = [ORDER.index(c) for c in AFF]
    soc_idx = [ORDER.index(c) for c in MENT]

    results = {}

    for model in MODELS:
        short = MSHORT[model]
        print(f"\n--- {short} ---")

        per_stim_path = RSA / f"{model}_rsa_v2_per_stim.npz"
        if not per_stim_path.exists():
            print(f"  SKIP: {per_stim_path.name} not found")
            continue

        print(f"  Loading {per_stim_path.name}...")
        data = np.load(per_stim_path, allow_pickle=True)
        stim_conditions = list(data["conditions"])
        pooling_names = list(data["pooling_names"])
        layer_names = list(data["layer_names"])
        # shape: (n_poolings, n_stim, n_layers, hidden_dim)
        all_acts = data["per_stim_activations"]

        pool_idx = pooling_names.index("mean_all")
        activations = all_acts[pool_idx]  # (n_stim, n_layers, hidden_dim)
        n_layers = activations.shape[1]

        print(f"  Pooling: mean_all (idx {pool_idx}), {n_layers} layers, "
              f"{activations.shape[0]} stimuli, dim={activations.shape[2]}")

        # For each layer, build condition-level RDM and decompose
        layer_results = []
        peak_layer = PEAK_LAYERS[model]

        for layer_i in range(n_layers):
            acts = activations[:, layer_i, :].astype(np.float32)  # (n_stim, hidden_dim)

            # Average per condition
            cond_means = {}
            for c in ORDER:
                mask = [i for i, sc in enumerate(stim_conditions) if sc == c]
                if mask:
                    cond_means[c] = acts[mask].mean(axis=0)

            if len(cond_means) < 14:
                layer_results.append(None)
                continue

            # Build RDM (centered, cosine)
            mat = np.stack([cond_means[c] for c in ORDER])
            mat = mat - mat.mean(axis=0, keepdims=True)  # center
            rdm = squareform(pdist(mat, metric="cosine"))
            llm_vec = triu_vec(rdm)

            # Full RSA
            full_rho = float(spearmanr(llm_vec, brain_vec)[0])

            # Block RSA
            block_rho = float(spearmanr(llm_vec, block_vec)[0])

            # Within-block RSA
            w_aff = within_block_rho(rdm, brain_rdm, aff_idx)
            w_soc = within_block_rho(rdm, brain_rdm, soc_idx)

            # Partial RSA (control block)
            X = np.column_stack([block_vec, np.ones(len(block_vec))])
            beta_l, _, _, _ = lstsq(X, llm_vec, rcond=None)
            beta_b, _, _, _ = lstsq(X, brain_vec, rcond=None)
            resid_l = llm_vec - X @ beta_l
            resid_b = brain_vec - X @ beta_b
            partial_rho = float(spearmanr(resid_l, resid_b)[0])

            layer_results.append({
                "layer": layer_i,
                "full_rho": full_rho,
                "block_rho": block_rho,
                "within_aff_rho": w_aff,
                "within_soc_rho": w_soc,
                "partial_rho": partial_rho,
            })

            if layer_i == peak_layer or layer_i % 8 == 0:
                marker = " <-- PEAK" if layer_i == peak_layer else ""
                print(f"  L{layer_i:02d}: full={full_rho:+.3f} block={block_rho:.3f} "
                      f"w_aff={w_aff:+.3f} w_soc={w_soc:+.3f} partial={partial_rho:+.3f}{marker}")

        valid = [lr for lr in layer_results if lr is not None]
        if valid:
            results[short] = {
                "n_layers": n_layers,
                "peak_layer": peak_layer,
                "per_layer": valid,
            }

    if not results:
        print("\nNo per-stim data available on login node. Need to run on SLURM with per-stim NPZ files.")
        print("Falling back to headline RDMs (peak layer only)...")

        # Fallback: just show peak-layer decomposition
        for model in MODELS:
            short = MSHORT[model]
            z = np.load(RSA / f"{model}_rdm14_headline.npz", allow_pickle=True)
            llm_rdm = reorder_rdm(z["rdm"], list(z["conditions"]), ORDER)
            llm_vec = triu_vec(llm_rdm)

            full_rho = float(spearmanr(llm_vec, brain_vec)[0])
            block_rho = float(spearmanr(llm_vec, block_vec)[0])
            w_aff = within_block_rho(llm_rdm, brain_rdm, aff_idx)
            w_soc = within_block_rho(llm_rdm, brain_rdm, soc_idx)

            print(f"  {short}: full={full_rho:+.3f} block={block_rho:.3f} "
                  f"w_aff={w_aff:+.3f} w_soc={w_soc:+.3f}")

        # Use layer_depth_profile.json if available
        ldp_path = RSA / "layer_depth_profile.json"
        if ldp_path.exists():
            ldp = json.load(open(ldp_path))
            print("\nUsing layer_depth_profile.json for layer-wise full RSA")
            for model_key in ldp:
                if isinstance(ldp[model_key], dict) and "per_layer_rho" in ldp[model_key]:
                    per_layer = ldp[model_key]["per_layer_rho"]
                    short = model_key
                    results[short] = {
                        "n_layers": len(per_layer),
                        "per_layer": [{"layer": i, "full_rho": r} for i, r in enumerate(per_layer)],
                        "source": "layer_depth_profile.json (full RSA only)",
                    }
                    print(f"  {short}: {len(per_layer)} layers, "
                          f"peak rho={max(per_layer):.3f} at L{np.argmax(per_layer)}")

        # Also check base_vs_instruct.json which has per-layer data
        bvi_path = RSA / "base_vs_instruct.json"
        if bvi_path.exists():
            bvi = json.load(open(bvi_path))
            for variant in ["instruct", "base"]:
                if variant in bvi and "per_layer_rho" in bvi[variant]:
                    per_layer = bvi[variant]["per_layer_rho"]
                    name = bvi[variant]["model"]
                    results[f"{name}"] = {
                        "n_layers": len(per_layer),
                        "per_layer": [{"layer": i, "full_rho": r} for i, r in enumerate(per_layer)],
                        "source": "base_vs_instruct.json",
                    }
                    print(f"  {name}: {len(per_layer)} layers, "
                          f"peak rho={max(per_layer):.3f} at L{np.argmax(per_layer)}")

    # ─── Visualization ───
    if results:
        print("\n--- Generating figures ---")

        n_models = len(results)
        fig, axes = plt.subplots(1, min(n_models, 4), figsize=(min(n_models, 4) * 4.5, 4),
                                 squeeze=False)
        axes = axes[0]

        for idx, (mname, mdata) in enumerate(list(results.items())[:4]):
            ax = axes[idx]
            layers = [d["layer"] for d in mdata["per_layer"]]
            full = [d["full_rho"] for d in mdata["per_layer"]]

            ax.plot(layers, full, "-", color=C_BLUE, lw=2, label="Full RSA")

            if "block_rho" in mdata["per_layer"][0]:
                block = [d["block_rho"] for d in mdata["per_layer"]]
                w_aff = [d["within_aff_rho"] for d in mdata["per_layer"]]
                w_soc = [d["within_soc_rho"] for d in mdata["per_layer"]]
                partial = [d["partial_rho"] for d in mdata["per_layer"]]

                ax.plot(layers, block, "--", color="#999", lw=1.5, label="Block split")
                ax.plot(layers, w_soc, "-", color=C_MENT, lw=1.5, label="Within-social")
                ax.plot(layers, w_aff, "-", color=C_AFF, lw=1.5, label="Within-aff")
                ax.plot(layers, partial, ":", color=C_BLUE, lw=1.5, label="Partial (beyond split)")

            if "peak_layer" in mdata:
                ax.axvline(mdata["peak_layer"], color="#ccc", ls="--", lw=0.8)

            ax.axhline(0, color="#ccc", lw=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Spearman rho" if idx == 0 else "")
            ax.set_title(mname, fontsize=10)
            ax.set_ylim(-0.8, 1.0)
            ax.grid(alpha=0.2)
            if idx == 0:
                ax.legend(fontsize=7, loc="lower left")

        fig.tight_layout()
        fig.savefig(FIG_DIR / "layer_boundary_emergence.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {FIG_DIR / 'layer_boundary_emergence.png'}")
        plt.close(fig)

    # Save results
    json.dump(results, open(OUT_DIR / "layer_boundary_emergence.json", "w"), indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nSaved: {OUT_DIR / 'layer_boundary_emergence.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
