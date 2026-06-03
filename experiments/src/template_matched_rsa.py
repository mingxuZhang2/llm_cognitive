#!/usr/bin/env python3
"""
Template-matched RSA: compute brain-LLM RSA using template-matched stimuli.

This is the FORMAT-CONFOUND control. If the brain-LLM alignment (rho ~ 0.73)
survives when all 14 conditions use identical sentence templates (only content
words differ), the signal cannot be attributed to format/source differences
between the original stimuli (Hadidi 2026 concern).

Pipeline:
  1. extract_rsa_activations_v2.py was run on rsa_stimuli_template_matched.jsonl
     -> {model}_tm_per_stim.npz
  2. This script loads those per-stim activations, computes per-condition mean
     at the headline peak layer (mean_all | centered | 1_cosine), builds the
     14x14 LLM RDM, and compares to brain_rdm.npz.

Outputs:
  results/template_matched_rsa/template_matched_rsa_results.json
  results/template_matched_rsa/{model}_tm_rdm14.npz

Usage:
  python src/template_matched_rsa.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# ── Paths ──
BASE = Path(__file__).resolve().parents[1]
RSA_DIR = BASE / "results" / "cognitive_rsa"
OUT_DIR = BASE / "results" / "template_matched_rsa"

# ── Headline recipe ──
POOLING = "mean_all"  # index 1 in extraction
NORMALIZATION = "centered"
DISTANCE = "1_cosine"

# Model configs: (short_name, peak_layer)
MODELS = [
    ("Qwen2.5-7B-Instruct", 27),
    ("Meta-Llama-3.1-8B-Instruct", 31),
    ("Mistral-7B-Instruct-v0.3", 14),
    ("gemma-2-9b-it", 21),
]


def normalize_centered(act):
    return act - act.mean(axis=0, keepdims=True)


def rdm_cosine(act):
    n = np.linalg.norm(act, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = act / n
    return 1.0 - Xn @ Xn.T


def triu_vals(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def permutation_pvalue(brain_tri, llm_tri, n_perm=10000, seed=42):
    """Permutation null: shuffle condition labels, recompute Spearman."""
    rng = np.random.default_rng(seed)
    n_cond = int((1 + np.sqrt(1 + 8 * len(brain_tri))) / 2)
    obs_rho, _ = spearmanr(brain_tri, llm_tri)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_cond)
        # Reconstruct RDM from triu, permute, extract triu
        brain_rdm = np.zeros((n_cond, n_cond))
        brain_rdm[np.triu_indices(n_cond, 1)] = brain_tri
        brain_rdm += brain_rdm.T
        perm_rdm = brain_rdm[np.ix_(perm, perm)]
        perm_tri = perm_rdm[np.triu_indices(n_cond, 1)]
        r, _ = spearmanr(perm_tri, llm_tri)
        if r >= obs_rho:
            count += 1
    return count / n_perm


def split_half_ceiling(per_stim_act, conds, unique_conds, n_splits=100,
                       seed=123):
    """LLM split-half noise ceiling at the headline recipe."""
    rng = np.random.default_rng(seed)
    cond_to_idx = {c: [] for c in unique_conds}
    for i, c in enumerate(conds):
        cond_to_idx[c].append(i)

    rhos = []
    for _ in range(n_splits):
        half_a = []
        half_b = []
        for c in unique_conds:
            idx = np.array(cond_to_idx[c])
            rng.shuffle(idx)
            cut = len(idx) // 2
            if cut < 1:
                return float("nan")
            half_a.append(per_stim_act[idx[:cut]].mean(axis=0))
            half_b.append(per_stim_act[idx[cut:]].mean(axis=0))

        a = normalize_centered(np.stack(half_a).astype(np.float64))
        b = normalize_centered(np.stack(half_b).astype(np.float64))
        rdm_a = rdm_cosine(a)
        rdm_b = rdm_cosine(b)
        r, _ = spearmanr(triu_vals(rdm_a), triu_vals(rdm_b))
        if np.isfinite(r):
            rhos.append(r)

    return float(np.mean(rhos)) if rhos else float("nan")


def per_template_rsa(per_stim_act, stim_templates, conds, unique_conds,
                     brain_tri):
    """Compute RSA separately for each template to check consistency."""
    templates = sorted(set(stim_templates))
    results = {}
    for tmpl in templates:
        mask = np.array([t == tmpl for t in stim_templates])
        sub_act = per_stim_act[mask]
        sub_conds = [c for c, m in zip(conds, mask) if m]

        cond_to_idx = {c: [] for c in unique_conds}
        for i, c in enumerate(sub_conds):
            cond_to_idx[c].append(i)

        cmean = np.stack([sub_act[cond_to_idx[c]].mean(axis=0).astype(np.float64)
                          for c in unique_conds])
        rdm = rdm_cosine(normalize_centered(cmean))
        r, _ = spearmanr(triu_vals(rdm), brain_tri)
        results[tmpl] = round(float(r), 4)
    return results


def process_model(model_short, peak_layer, brain_rdm, brain_tri):
    """Process one model's template-matched activations."""
    # extract_rsa_activations_v2.py saves as {model_short}_rsa_v2_per_stim.npz
    # SLURM passes model_short="${MODEL}_tm", so file is {model}_tm_rsa_v2_per_stim.npz
    npz_path = RSA_DIR / f"{model_short}_tm_rsa_v2_per_stim.npz"
    if not npz_path.exists():
        print(f"  SKIP {model_short}: {npz_path.name} not found")
        return None

    print(f"  Loading {npz_path.name} ...")
    z = np.load(npz_path, allow_pickle=True)
    per_stim = z["per_stim_activations"]  # [3, n_stim, n_layers, hidden]
    conds = list(z["conditions"])
    stim_ids = list(z["stim_ids"])
    pools = list(z["pooling_names"])

    pidx = pools.index(POOLING)
    unique_conds = sorted(set(conds))

    # Load template info from stimuli file
    stim_path = (BASE / "data" / "cognitive_stimuli" / "rsa"
                 / "rsa_stimuli_template_matched.jsonl")
    stim_templates = []
    if stim_path.exists():
        stim_data = {}
        with open(stim_path) as f:
            for line in f:
                d = json.loads(line)
                stim_data[d["id"]] = d.get("template", "unknown")
        stim_templates = [stim_data.get(sid, "unknown") for sid in stim_ids]
    else:
        stim_templates = ["unknown"] * len(stim_ids)

    # Condition-mean at peak layer
    cond_to_idx = {c: [] for c in unique_conds}
    for i, c in enumerate(conds):
        cond_to_idx[c].append(i)

    act_at_peak = per_stim[pidx, :, peak_layer, :].astype(np.float64)
    cmean = np.stack([act_at_peak[cond_to_idx[c]].mean(axis=0)
                      for c in unique_conds])
    rdm = rdm_cosine(normalize_centered(cmean))

    # Match condition order to brain RDM
    brain_conds = list(brain_rdm["conditions"])
    order = [brain_conds.index(c) for c in unique_conds]
    brain_re = brain_rdm["rdm"][np.ix_(order, order)]
    brain_tri_reord = triu_vals(brain_re)

    rho, _ = spearmanr(triu_vals(rdm), brain_tri_reord)

    # Permutation p-value
    p_perm = permutation_pvalue(brain_tri_reord, triu_vals(rdm))

    # Split-half noise ceiling
    ceiling = split_half_ceiling(act_at_peak, conds, unique_conds)

    # Per-template RSA
    tmpl_rsa = per_template_rsa(act_at_peak, stim_templates, conds,
                                unique_conds, brain_tri_reord)

    # Also compute RSA at all layers to find peak
    n_layers = per_stim.shape[2]
    layer_rhos = []
    for L in range(n_layers):
        act_L = per_stim[pidx, :, L, :].astype(np.float64)
        cm_L = np.stack([act_L[cond_to_idx[c]].mean(axis=0) for c in unique_conds])
        rdm_L = rdm_cosine(normalize_centered(cm_L))
        r, _ = spearmanr(triu_vals(rdm_L), brain_tri_reord)
        layer_rhos.append(round(float(r), 4))

    actual_peak = int(np.argmax(layer_rhos))

    # Save RDM
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"{model_short}_tm_rdm14.npz",
        rdm=rdm,
        conditions=np.array(unique_conds),
        peak_layer=peak_layer,
        config=f"{POOLING}|{NORMALIZATION}|{DISTANCE}|template_matched",
    )

    result = {
        "rho_at_headline_peak": round(float(rho), 4),
        "p_perm": round(float(p_perm), 5),
        "headline_peak_layer": peak_layer,
        "actual_peak_layer": actual_peak,
        "rho_at_actual_peak": layer_rhos[actual_peak],
        "ceiling": round(ceiling, 4),
        "rho_over_ceiling": round(float(rho) / ceiling, 3) if ceiling > 0 else None,
        "per_template_rho": tmpl_rsa,
        "layer_rhos": layer_rhos,
    }

    print(f"  {model_short}: rho={rho:+.4f} (p={p_perm:.4f}), "
          f"peak_L={peak_layer}, actual_peak_L={actual_peak} "
          f"(rho={layer_rhos[actual_peak]:+.4f}), "
          f"ceiling={ceiling:.3f}")
    for tmpl, tr in tmpl_rsa.items():
        print(f"    template '{tmpl}': rho={tr:+.4f}")

    return result


def load_headline_rdms_for_comparison():
    """Load headline RDMs from original stimuli for side-by-side comparison."""
    comparison = {}
    for model_short, _ in MODELS:
        path = RSA_DIR / f"{model_short}_rdm14_headline.npz"
        if path.exists():
            z = np.load(path, allow_pickle=True)
            comparison[model_short] = z["rdm"]
    return comparison


def main():
    t0 = time.time()
    print("=" * 70)
    print("Template-Matched RSA: format-confound control")
    print("=" * 70)

    # Load brain RDM
    brain = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    brain_conds = list(brain["conditions"])
    brain_tri = triu_vals(brain["rdm"])
    print(f"Brain RDM: {len(brain_conds)} conditions")

    # Load headline RDMs for comparison
    headline_rdms = load_headline_rdms_for_comparison()

    results = {"config": f"{POOLING}|{NORMALIZATION}|{DISTANCE}|template_matched",
               "models": {}}

    for model_short, peak_layer in MODELS:
        print(f"\n--- {model_short} (peak L{peak_layer}) ---")
        r = process_model(model_short, peak_layer, brain, brain_tri)
        if r is not None:
            # Add comparison to original-stimuli RSA
            if model_short in headline_rdms:
                orig_rdm = headline_rdms[model_short]
                orig_conds = list(np.load(
                    RSA_DIR / f"{model_short}_rdm14_headline.npz",
                    allow_pickle=True)["conditions"])
                order = [brain_conds.index(c) for c in orig_conds]
                brain_re = brain["rdm"][np.ix_(order, order)]
                orig_rho, _ = spearmanr(triu_vals(orig_rdm), triu_vals(brain_re))
                r["original_stimuli_rho"] = round(float(orig_rho), 4)
                print(f"  Original stimuli: rho={orig_rho:+.4f}")
                print(f"  Delta (template - original): "
                      f"{r['rho_at_headline_peak'] - orig_rho:+.4f}")

            results["models"][model_short] = r

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Template-Matched vs Original Stimuli RSA")
    print("=" * 70)
    print(f"{'Model':35s} {'TM rho':>10s} {'Orig rho':>10s} {'Delta':>10s} "
          f"{'p_perm':>8s} {'Ceiling':>8s}")
    print("-" * 82)
    for model_short, _ in MODELS:
        if model_short in results["models"]:
            r = results["models"][model_short]
            tm = r["rho_at_headline_peak"]
            orig = r.get("original_stimuli_rho", float("nan"))
            delta = tm - orig if np.isfinite(orig) else float("nan")
            print(f"{model_short:35s} {tm:+10.4f} {orig:+10.4f} {delta:+10.4f} "
                  f"{r['p_perm']:8.4f} {r['ceiling']:8.3f}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "template_matched_rsa_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
