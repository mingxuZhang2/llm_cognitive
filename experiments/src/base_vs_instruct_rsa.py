#!/usr/bin/env python3
"""
Base vs Instruct RSA comparison: how much brain-LLM alignment comes from
pretraining alone (Qwen2.5-1.5B) vs RLHF/instruction-tuning (Qwen2.5-1.5B-Instruct)?

Recipe (matches the headline):
  mean_all pooling | center across 14 conditions | cosine distance | sweep all layers

Loads per-stim activation NPZ files, computes condition-mean RDMs at every layer,
finds peak Spearman rho vs the Neurosynth brain RDM, and reports the ratio
base_peak_rho / instruct_peak_rho.

Output: results/cognitive_rsa/base_vs_instruct.json

If the base model NPZ does not exist, computes only the Instruct side and prints
instructions for extracting base activations (GPU job).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parents[1]
RSA_DIR = BASE_DIR / "results" / "cognitive_rsa"
N_PERM = 5000
SEED = 20260602


def normalize_centered(act: np.ndarray) -> np.ndarray:
    """Subtract mean across conditions (axis 0)."""
    return act - act.mean(axis=0, keepdims=True)


def rdm_cosine(act: np.ndarray) -> np.ndarray:
    """1 - cosine similarity matrix."""
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    xn = act / norms
    return 1.0 - xn @ xn.T


def triu(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def compute_layer_sweep(npz_path: str | Path, brain_rdm: np.ndarray,
                        brain_conds: list[str]) -> dict | None:
    """Sweep all layers, return peak rho, per-layer profile, noise ceiling, p-value."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)
    per_stim = data["per_stim_activations"]   # [3, n_stim, n_layers, hidden]
    conditions = list(data["conditions"])
    pool_names = list(data["pooling_names"])
    layer_names = list(data["layer_names"])
    pool_idx = pool_names.index("mean_all")

    unique_conds = sorted(set(conditions))
    stim_cond = np.array([unique_conds.index(c) for c in conditions])
    n_layers = per_stim.shape[2]

    # Condition means across stimuli: [n_cond, n_layers, hidden]
    cond_means = np.zeros((len(unique_conds), n_layers, per_stim.shape[-1]),
                          dtype=np.float32)
    for ci in range(len(unique_conds)):
        idxs = np.where(stim_cond == ci)[0]
        cond_means[ci] = per_stim[pool_idx, idxs].mean(axis=0)

    # Reorder to match brain condition order
    order = [unique_conds.index(c) for c in brain_conds]
    cond_means = cond_means[order]

    btr = triu(brain_rdm)

    # Layer sweep
    per_layer_rho = []
    peak_rdm = None
    for L in range(n_layers):
        act = cond_means[:, L, :].astype(np.float64)
        rdm = rdm_cosine(normalize_centered(act))
        rho, _ = spearmanr(btr, triu(rdm))
        per_layer_rho.append(float(rho))
        if peak_rdm is None or rho > per_layer_rho[int(np.argmax(per_layer_rho[:len(per_layer_rho)-1]))]:
            peak_rdm = rdm

    peak_L = int(np.argmax(per_layer_rho))
    obs_rho = per_layer_rho[peak_L]

    # Rebuild peak RDM cleanly
    act_peak = cond_means[:, peak_L, :].astype(np.float64)
    peak_rdm = rdm_cosine(normalize_centered(act_peak))

    # Permutation p-value
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(N_PERM):
        perm = rng.permutation(len(brain_conds))
        permuted = peak_rdm[np.ix_(perm, perm)]
        rho_p, _ = spearmanr(btr, triu(permuted))
        if rho_p >= obs_rho:
            count += 1
    p_val = (count + 1) / (N_PERM + 1)

    # Split-half noise ceiling at peak layer
    rng_c = np.random.default_rng(42)
    layer_act = per_stim[pool_idx, :, peak_L, :].astype(np.float32)
    rhos_ceil = []
    for _ in range(30):
        a = np.zeros((len(unique_conds), layer_act.shape[1]), dtype=np.float64)
        b = np.zeros_like(a)
        ok = True
        for ci in range(len(unique_conds)):
            idxs = np.where(stim_cond == ci)[0]
            rng_c.shuffle(idxs)
            cut = len(idxs) // 2
            if cut < 1 or len(idxs) - cut < 1:
                ok = False
                break
            a[ci] = layer_act[idxs[:cut]].mean(axis=0)
            b[ci] = layer_act[idxs[cut:]].mean(axis=0)
        if not ok:
            continue
        a = a[order]
        b = b[order]
        ra = rdm_cosine(normalize_centered(a))
        rb = rdm_cosine(normalize_centered(b))
        rho_c, _ = spearmanr(triu(ra), triu(rb))
        if np.isfinite(rho_c):
            rhos_ceil.append(rho_c)
    ceiling = float(np.mean(rhos_ceil)) if rhos_ceil else float("nan")

    return {
        "per_layer_rho": per_layer_rho,
        "peak_layer": peak_L,
        "peak_layer_name": layer_names[peak_L],
        "peak_rho": obs_rho,
        "noise_ceiling": ceiling,
        "rho_over_ceiling": obs_rho / ceiling if ceiling and ceiling > 0 else float("nan"),
        "p_value": p_val,
        "n_layers": n_layers,
        "n_stim": len(conditions),
        "n_cond": len(unique_conds),
    }


def main():
    # Load brain RDM
    brain = np.load(RSA_DIR / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain["rdm"]
    brain_conds = list(brain["conditions"])
    print(f"Brain RDM: {brain_rdm.shape}, {len(brain_conds)} conditions")

    instruct_npz = RSA_DIR / "Qwen2.5-1.5B-Instruct_rsa_v2_per_stim.npz"
    base_npz = RSA_DIR / "Qwen2.5-1.5B_rsa_v2_per_stim.npz"

    results = {"brain_conditions": brain_conds}

    # --- Instruct model ---
    print(f"\n{'='*60}")
    print("Computing Instruct model (Qwen2.5-1.5B-Instruct)...")
    print(f"{'='*60}")
    inst = compute_layer_sweep(instruct_npz, brain_rdm, brain_conds)
    if inst is None:
        print(f"ERROR: Instruct NPZ not found at {instruct_npz}")
        return
    results["instruct"] = {
        "model": "Qwen2.5-1.5B-Instruct",
        **inst,
    }
    print(f"  Peak layer: {inst['peak_layer']} ({inst['peak_layer_name']})")
    print(f"  Peak rho:   {inst['peak_rho']:+.4f}")
    print(f"  Ceiling:    {inst['noise_ceiling']:.4f}")
    print(f"  rho/ceil:   {inst['rho_over_ceiling']:.3f}")
    print(f"  p-value:    {inst['p_value']:.4g}")

    # --- Base model ---
    print(f"\n{'='*60}")
    print("Computing Base model (Qwen2.5-1.5B)...")
    print(f"{'='*60}")
    if not base_npz.exists():
        print(f"  Base NPZ not found: {base_npz}")
        print(f"  --> Need to extract base model activations via GPU job.")
        print(f"  --> Run: sbatch scripts/slurm/extract_base_1.5b.sh")
        print(f"  --> Then re-run this script.")
        results["base"] = None
        results["comparison"] = None
        results["note"] = (
            "Base model activations not yet extracted. "
            "Submit scripts/slurm/extract_base_1.5b.sh to extract them, "
            "then re-run this script."
        )
    else:
        base = compute_layer_sweep(base_npz, brain_rdm, brain_conds)
        if base is None:
            print("  ERROR: Base NPZ exists but could not be loaded.")
            return
        results["base"] = {
            "model": "Qwen2.5-1.5B",
            **base,
        }
        print(f"  Peak layer: {base['peak_layer']} ({base['peak_layer_name']})")
        print(f"  Peak rho:   {base['peak_rho']:+.4f}")
        print(f"  Ceiling:    {base['noise_ceiling']:.4f}")
        print(f"  rho/ceil:   {base['rho_over_ceiling']:.3f}")
        print(f"  p-value:    {base['p_value']:.4g}")

        # Also report base rho at the Instruct peak layer
        inst_peak_L = inst["peak_layer"]
        base_rho_at_inst_peak = base["per_layer_rho"][inst_peak_L]
        print(f"\n  Base rho at Instruct peak (L{inst_peak_L}): {base_rho_at_inst_peak:+.4f}")

        # Comparison
        ratio = base["peak_rho"] / inst["peak_rho"] if inst["peak_rho"] != 0 else float("nan")
        delta = inst["peak_rho"] - base["peak_rho"]
        results["comparison"] = {
            "base_peak_rho": base["peak_rho"],
            "instruct_peak_rho": inst["peak_rho"],
            "ratio_base_over_instruct": ratio,
            "delta_instruct_minus_base": delta,
            "base_rho_at_instruct_peak_layer": base_rho_at_inst_peak,
            "interpretation": (
                f"Base model captures {ratio*100:.1f}% of the Instruct alignment "
                f"(peak rho {base['peak_rho']:+.3f} vs {inst['peak_rho']:+.3f}). "
                f"RLHF/instruction-tuning adds delta={delta:+.3f}."
            ),
        }
        print(f"\n{'='*60}")
        print("COMPARISON:")
        print(f"  Base peak rho:      {base['peak_rho']:+.4f}  (L{base['peak_layer']})")
        print(f"  Instruct peak rho:  {inst['peak_rho']:+.4f}  (L{inst['peak_layer']})")
        print(f"  Ratio (base/inst):  {ratio:.3f}  ({ratio*100:.1f}% from pretraining)")
        print(f"  Delta (inst-base):  {delta:+.4f}")
        print(f"{'='*60}")

    # Save
    out_path = RSA_DIR / "base_vs_instruct.json"
    # Strip per_layer_rho for JSON (can be large); keep only summary
    for key in ("instruct", "base"):
        if results.get(key) and "per_layer_rho" in results[key]:
            results[key]["per_layer_rho_summary"] = {
                "min": min(results[key]["per_layer_rho"]),
                "max": max(results[key]["per_layer_rho"]),
                "mean": float(np.mean(results[key]["per_layer_rho"])),
            }
            # Keep full profile for plotting
            results[key]["per_layer_rho"] = [
                round(r, 4) for r in results[key]["per_layer_rho"]
            ]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
