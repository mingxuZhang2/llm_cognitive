"""
Scaling-axis analysis: compute brain-LLM RSA alignment for every Qwen2.5-Instruct
size at the v2 common configuration, then plot ρ vs log(parameter count).

Common config (frozen from v2 cross-model analysis):
  - pool          : mean_all
  - normalization : centered
  - distance      : 1_cosine
  - brain RDM     : v1 Neurosynth-anchored 14×14

For each size: load per-stim NPZ, build per-condition mean activations at each
layer, compute LLM RDM, report peak-layer Spearman vs brain plus split-half
noise ceiling.

Output:
  results/cognitive_rsa/scaling_summary.json
  figures/cognitive_rsa_scaling_curve.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


_EXP = Path(__file__).resolve().parents[1]
RES = _EXP / "results" / "cognitive_rsa"
FIG = _EXP / "figures"

QWEN_SIZES = [
    ("Qwen2.5-0.5B-Instruct",   0.494e9,  "0.5B"),
    ("Qwen2.5-1.5B-Instruct",   1.54e9,   "1.5B"),
    ("Qwen2.5-3B-Instruct",     3.09e9,   "3B"),
    ("Qwen2.5-7B-Instruct",     7.61e9,   "7B"),
    ("Qwen2.5-14B-Instruct",    14.7e9,   "14B"),
]


def normalize_centered(act):
    return act - act.mean(axis=0, keepdims=True)


def rdm_cosine(act):
    n = np.linalg.norm(act, axis=1, keepdims=True); n[n == 0] = 1.0
    return 1.0 - (act / n) @ (act / n).T


def triu(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def compute_for_model(model_short, brain_rdm, brain_conds):
    npz_path = RES / f"{model_short}_rsa_v2_per_stim.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    per_stim = data["per_stim_activations"]
    conditions = list(data["conditions"])
    pool_idx = list(data["pooling_names"]).index("mean_all")
    layer_names = list(data["layer_names"])
    unique_conditions = sorted(set(conditions))
    stim_cond = np.array([unique_conditions.index(c) for c in conditions])

    n_layers = per_stim.shape[2]
    cond_means = np.zeros((len(unique_conditions), n_layers, per_stim.shape[-1]),
                          dtype=np.float32)
    for c in range(len(unique_conditions)):
        idxs = np.where(stim_cond == c)[0]
        cond_means[c] = per_stim[pool_idx, idxs].mean(axis=0)
    order = [unique_conditions.index(c) for c in brain_conds]
    cond_means = cond_means[order]

    btr = triu(brain_rdm)
    per_layer_rho = []
    rdms = []
    for L in range(n_layers):
        act = cond_means[:, L, :].astype(np.float64)
        a_n = normalize_centered(act)
        rdm = rdm_cosine(a_n)
        rho, _ = spearmanr(btr, triu(rdm))
        per_layer_rho.append(rho)
        rdms.append(rdm)
    peak_L = int(np.argmax(per_layer_rho))

    # noise ceiling via split-half
    rng = np.random.default_rng(42)
    layer_act = per_stim[pool_idx, :, peak_L, :].astype(np.float32)
    rhos_ceil = []
    for _ in range(30):
        a = np.zeros((len(unique_conditions), layer_act.shape[1]), dtype=np.float64)
        b = np.zeros_like(a)
        ok = True
        for c in range(len(unique_conditions)):
            idxs = np.where(stim_cond == c)[0]
            rng.shuffle(idxs)
            cut = len(idxs) // 2
            if cut < 1 or len(idxs) - cut < 1:
                ok = False; break
            a[c] = layer_act[idxs[:cut]].mean(axis=0)
            b[c] = layer_act[idxs[cut:]].mean(axis=0)
        if not ok: continue
        a = a[order]; b = b[order]
        ra = rdm_cosine(normalize_centered(a))
        rb = rdm_cosine(normalize_centered(b))
        rho_c, _ = spearmanr(triu(ra), triu(rb))
        if np.isfinite(rho_c): rhos_ceil.append(rho_c)
    ceiling = float(np.mean(rhos_ceil)) if rhos_ceil else float("nan")

    # permutation null
    rng_p = np.random.default_rng(2026)
    peak_rdm = rdms[peak_L]
    obs_rho = per_layer_rho[peak_L]
    n_perm = 5000
    count = 0
    for _ in range(n_perm):
        perm = rng_p.permutation(len(brain_conds))
        permuted = peak_rdm[np.ix_(perm, perm)]
        rho_p, _ = spearmanr(btr, triu(permuted))
        if rho_p >= obs_rho: count += 1
    p_val = (count + 1) / (n_perm + 1)

    return {
        "model_short": model_short,
        "per_layer_rho": per_layer_rho,
        "peak_layer": peak_L,
        "peak_layer_name": layer_names[peak_L],
        "peak_rho": float(obs_rho),
        "noise_ceiling": ceiling,
        "rho_over_ceiling": float(obs_rho / ceiling) if ceiling and ceiling > 0 else float("nan"),
        "p_value": float(p_val),
    }


def main():
    brain = np.load(RES / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain["rdm"]
    brain_conds = list(brain["conditions"])
    print(f"Brain RDM {brain_rdm.shape}")

    rows = []
    for model_short, params, label in QWEN_SIZES:
        r = compute_for_model(model_short, brain_rdm, brain_conds)
        if r is None:
            print(f"  [skip] {label}: NPZ not present")
            continue
        r["params"] = params
        r["label"] = label
        rows.append(r)
        print(f"  {label:>5s}  peak L{r['peak_layer']:<3d}  "
              f"rho={r['peak_rho']:+.4f}  ceil={r['noise_ceiling']:.4f}  "
              f"ratio={r['rho_over_ceiling']:+.3f}  p={r['p_value']:.4g}")

    # ---- scaling plot ----
    params = np.array([r["params"] for r in rows])
    rhos = np.array([r["peak_rho"] for r in rows])
    ceils = np.array([r["noise_ceiling"] for r in rows])
    ratios = rhos / ceils

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(params, rhos, "o-", color="crimson", linewidth=2.5, markersize=12,
            label="Brain–LLM RSA (peak ρ)")
    ax.plot(params, ceils, "s--", color="gray", markersize=8, alpha=0.6,
            label="LLM noise ceiling (split-half)")
    for i, r in enumerate(rows):
        ax.annotate(r["label"], (r["params"], r["peak_rho"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Cognitive RSA alignment vs LLM scale (Qwen2.5-Instruct family)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(params, ratios * 100, "o-", color="darkblue", linewidth=2.5,
            markersize=12)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['label']}\n{r['rho_over_ceiling']*100:.0f}%",
                    (r["params"], ratios[i] * 100),
                    textcoords="offset points", xytext=(8, -8), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("ρ / noise ceiling (%)")
    ax.set_title("Fraction of achievable alignment captured")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = FIG / "cognitive_rsa_scaling_curve.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")

    # Save summary
    summary = {
        "models": [
            {"model": r["model_short"], "label": r["label"], "params": r["params"],
             "peak_layer": r["peak_layer"], "peak_layer_name": r["peak_layer_name"],
             "peak_rho": r["peak_rho"], "noise_ceiling": r["noise_ceiling"],
             "rho_over_ceiling": r["rho_over_ceiling"], "p_value": r["p_value"]}
            for r in rows
        ],
    }
    with open(RES / "scaling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {RES / 'scaling_summary.json'}")


if __name__ == "__main__":
    main()
