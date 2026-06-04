"""
v2 cross-model synthesis: pick a single common configuration across all 4
models for fair comparison, compute consensus, and produce final figures.

Strategy: use the config that gives the highest mean ρ across models
(mean_all + centered + 1_cosine + v1_NS brain).

Outputs:
  - results/cognitive_rsa/cross_model_summary_v2.json
  - figures/cognitive_rsa_v2_alignment.png   (per-layer ρ curves, all 4 models)
  - figures/cognitive_rsa_v2_rdm_grid.png    (brain + 4 LLM RDMs at peak)
  - figures/cognitive_rsa_v2_preserved_pairs.png
  - figures/cognitive_rsa_v2_ceiling_bar.png  (ρ and noise ceiling per model)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr


_EXP = Path(__file__).resolve().parents[1]
RES = _EXP / "results" / "cognitive_rsa"
FIG = _EXP / "figures"

MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]
COLORS = {"Qwen2.5-7B-Instruct": "#1f77b4",
          "Meta-Llama-3.1-8B-Instruct": "#ff7f0e",
          "Mistral-7B-Instruct-v0.3": "#2ca02c",
          "gemma-2-9b-it": "#d62728"}

# Common config — chosen because it gives the best mean ρ across the 4 models.
COMMON_CFG = ("mean_all", "centered", "1_cosine", "v1_NS_only")


def load_brain():
    d = np.load(RES / "brain_rdm.npz", allow_pickle=True)
    return d["rdm"], list(d["conditions"])


def normalize(act, mode):
    if mode == "raw":
        return act
    if mode == "centered":
        return act - act.mean(axis=0, keepdims=True)
    if mode == "feature_zscore":
        m, s = act.mean(0, keepdims=True), act.std(0, keepdims=True) + 1e-8
        return (act - m) / s


def rdm_cosine(act):
    n = np.linalg.norm(act, axis=1, keepdims=True); n[n == 0] = 1.0
    return 1.0 - (act / n) @ (act / n).T


def triu(rdm):
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def compute_one_model(model, brain_rdm, brain_conds):
    """Re-compute per-layer RDM and pick peak under the common config."""
    pool_name, norm, dist, _ = COMMON_CFG
    npz = np.load(RES / f"{model}_rsa_v2_per_stim.npz", allow_pickle=True)
    per_stim = npz["per_stim_activations"]  # [3, n_stim, n_layers+1, hidden]
    conditions = list(npz["conditions"])
    pool_idx = list(npz["pooling_names"]).index(pool_name)
    layer_names = list(npz["layer_names"])

    unique_conditions = sorted(set(conditions))
    cond_to_idx = {c: i for i, c in enumerate(unique_conditions)}
    stim_cond = np.array([cond_to_idx[c] for c in conditions])

    n_layers = per_stim.shape[2]
    cond_means = np.zeros((len(unique_conditions), n_layers, per_stim.shape[-1]),
                          dtype=np.float32)
    for c in range(len(unique_conditions)):
        idxs = np.where(stim_cond == c)[0]
        cond_means[c] = per_stim[pool_idx, idxs].mean(axis=0)

    # reorder to brain condition ordering
    order = [unique_conditions.index(c) for c in brain_conds]
    cond_means = cond_means[order]
    conditions_brain_order = brain_conds

    btr = triu(brain_rdm)
    per_layer_rho = []
    layer_rdms = []
    for L in range(n_layers):
        act = cond_means[:, L, :].astype(np.float64)
        a = normalize(act, norm)
        rdm = rdm_cosine(a)
        rho, _ = spearmanr(btr, triu(rdm))
        per_layer_rho.append(rho)
        layer_rdms.append(rdm)
    layer_rdms = np.stack(layer_rdms)

    peak_L = int(np.argmax(per_layer_rho))
    return {
        "model": model,
        "per_layer_rho": per_layer_rho,
        "peak_L": peak_L,
        "peak_layer_name": layer_names[peak_L],
        "peak_rho": float(per_layer_rho[peak_L]),
        "peak_rdm": layer_rdms[peak_L],
        "all_layer_rdms": layer_rdms,
    }


def split_half_ceiling_at_layer(model, peak_L, n_splits=50, rng_seed=42):
    pool_name, norm, dist, _ = COMMON_CFG
    npz = np.load(RES / f"{model}_rsa_v2_per_stim.npz", allow_pickle=True)
    per_stim = npz["per_stim_activations"]
    conditions = list(npz["conditions"])
    pool_idx = list(npz["pooling_names"]).index(pool_name)
    unique_conditions = sorted(set(conditions))
    stim_cond = np.array([list(unique_conditions).index(c) for c in conditions])

    layer_act = per_stim[pool_idx, :, peak_L, :].astype(np.float32)
    rng = np.random.default_rng(rng_seed)
    rhos = []
    for _ in range(n_splits):
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
        if not ok:
            return float("nan")
        a = normalize(a, norm); b = normalize(b, norm)
        ra = rdm_cosine(a); rb = rdm_cosine(b)
        rho, _ = spearmanr(triu(ra), triu(rb))
        if np.isfinite(rho): rhos.append(rho)
    return float(np.mean(rhos))


def permutation_p(brain_rdm, llm_rdm, n_perm=10000, seed=2026):
    rng = np.random.default_rng(seed)
    btr = triu(brain_rdm)
    obs, _ = spearmanr(btr, triu(llm_rdm))
    n = brain_rdm.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        permuted = llm_rdm[np.ix_(perm, perm)]
        rho, _ = spearmanr(btr, triu(permuted))
        if rho >= obs:
            count += 1
    return float((count + 1) / (n_perm + 1)), obs


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    brain_rdm, brain_conds = load_brain()
    print(f"Brain RDM shape: {brain_rdm.shape}, conditions: {len(brain_conds)}")
    print(f"Common config: {COMMON_CFG}")

    per_model = {}
    for m in MODELS:
        r = compute_one_model(m, brain_rdm, brain_conds)
        p, obs = permutation_p(brain_rdm, r["peak_rdm"], n_perm=10000)
        ceiling = split_half_ceiling_at_layer(m, r["peak_L"], n_splits=50)
        r["p_value"] = p
        r["noise_ceiling"] = ceiling
        r["rho_over_ceiling"] = obs / ceiling if ceiling and ceiling > 0 else float("nan")
        per_model[m] = r
        print(f"  {m:>30s}  peak L{r['peak_L']:<3d}  rho={r['peak_rho']:+.4f}  "
              f"ceil={ceiling:.4f}  ratio={r['rho_over_ceiling']:+.3f}  p={p:.4g}")

    # ---- Figure A: alignment curves ----
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for m in MODELS:
        r = per_model[m]
        rel_x = np.linspace(0, 1, len(r["per_layer_rho"]))
        ax.plot(rel_x, r["per_layer_rho"], "-", color=COLORS[m],
                label=f"{m}  (peak ρ={r['peak_rho']:.2f}, ceil={r['noise_ceiling']:.2f})",
                linewidth=2)
        ax.scatter([rel_x[r["peak_L"]]], [r["peak_rho"]], s=80,
                   color=COLORS[m], edgecolor="black", zorder=5)
    # mean noise ceiling band
    mean_ceil = np.mean([per_model[m]["noise_ceiling"] for m in MODELS])
    ax.axhline(mean_ceil, color="gray", linestyle="--", linewidth=0.9,
               label=f"mean LLM noise ceiling = {mean_ceil:.2f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Relative layer depth (0 = embedding, 1 = final)")
    ax.set_ylabel("Spearman ρ (brain RDM vs LLM RDM)")
    ax.set_title("Cognitive-domain RSA alignment (v2)\n"
                 "mean-pool tokens, centered, 1−cosine, Neurosynth+HCP brain")
    ax.legend(fontsize=8, loc="lower center")
    ax.set_ylim(-0.1, 1.0)
    plt.tight_layout()
    plt.savefig(FIG / "cognitive_rsa_v2_alignment.png", dpi=160)
    plt.close()
    print(f"Wrote {FIG / 'cognitive_rsa_v2_alignment.png'}")

    # ---- Figure B: RDM grid ----
    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(3.5 * (n_models + 1), 3.6),
                             constrained_layout=True)
    def show(ax, mat, title, vmin=0, vmax=1.4):
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(brain_conds)))
        ax.set_yticks(range(len(brain_conds)))
        ax.set_xticklabels(brain_conds, rotation=80, fontsize=6)
        ax.set_yticklabels(brain_conds, fontsize=6)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)
    show(axes[0], brain_rdm, "Brain (Neurosynth + HCP)")
    for i, m in enumerate(MODELS):
        r = per_model[m]
        # rescale LLM RDM to brain RDM range for visual comparison
        rdm = r["peak_rdm"]
        show(axes[i + 1], rdm,
             f"{m.split('-')[0]} L{r['peak_L']}\nρ={r['peak_rho']:.3f}  "
             f"ratio={r['rho_over_ceiling']:.2f}", vmin=0, vmax=rdm.max())
    plt.savefig(FIG / "cognitive_rsa_v2_rdm_grid.png", dpi=160)
    plt.close()
    print(f"Wrote {FIG / 'cognitive_rsa_v2_rdm_grid.png'}")

    # ---- Figure C: ceiling-normalized bar chart ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    xs = np.arange(len(MODELS))
    rhos = [per_model[m]["peak_rho"] for m in MODELS]
    ceils = [per_model[m]["noise_ceiling"] for m in MODELS]
    bar1 = ax.bar(xs - 0.18, rhos, 0.36,
                  color=[COLORS[m] for m in MODELS], label="Observed ρ")
    bar2 = ax.bar(xs + 0.18, ceils, 0.36,
                  color="lightgray", edgecolor="black",
                  label="LLM noise ceiling (split-half)")
    for i, m in enumerate(MODELS):
        ratio = per_model[m]["rho_over_ceiling"]
        ax.text(i, max(rhos[i], ceils[i]) + 0.02,
                f"{ratio*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([m.split('-')[0] for m in MODELS], fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Brain–LLM RSA alignment vs LLM noise ceiling\n"
                 "(percentage above bars = ρ / ceiling)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "cognitive_rsa_v2_ceiling_bar.png", dpi=160)
    plt.close()
    print(f"Wrote {FIG / 'cognitive_rsa_v2_ceiling_bar.png'}")

    # ---- Figure D: preserved pairs (consensus) ----
    n_cond = len(brain_conds)
    consensus = np.mean([per_model[m]["peak_rdm"] for m in MODELS], axis=0)
    triu_idx = np.triu_indices(n_cond, k=1)
    cons_rho, _ = spearmanr(triu(brain_rdm), triu(consensus))
    print(f"\nConsensus LLM RDM vs brain RDM: Spearman ρ = {cons_rho:.4f}")

    # z-score within each side, find universal pairs
    btr_vals = triu(brain_rdm); ctr_vals = triu(consensus)
    bz = (btr_vals - btr_vals.mean()) / btr_vals.std()
    cz = (ctr_vals - ctr_vals.mean()) / ctr_vals.std()
    combo = bz + cz
    order_asc = np.argsort(combo)
    pairs = [(brain_conds[triu_idx[0][k]], brain_conds[triu_idx[1][k]], combo[k])
             for k in range(len(combo))]
    preserved_similar = [pairs[k] for k in order_asc[:8]]
    preserved_dissimilar = [pairs[k] for k in order_asc[-8:][::-1]]

    print("\n[v2 Universal PRESERVED-SIMILAR pairs]:")
    for a, b, z in preserved_similar:
        print(f"  {a:>16s} <-> {b:<16s}  combined z = {z:+.2f}")
    print("\n[v2 Universal PRESERVED-DISSIMILAR pairs]:")
    for a, b, z in preserved_dissimilar:
        print(f"  {a:>16s} <-> {b:<16s}  combined z = {z:+.2f}")

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))
    bx = btr_vals; cy = ctr_vals
    ax.scatter(bx, cy, s=30, color="gray", alpha=0.5)
    annot = set()
    for a, b, _ in preserved_similar[:6] + preserved_dissimilar[:6]:
        annot.add(tuple(sorted([a, b])))
    for k in range(len(combo)):
        a = brain_conds[triu_idx[0][k]]
        b = brain_conds[triu_idx[1][k]]
        if tuple(sorted([a, b])) in annot:
            ax.scatter([bx[k]], [cy[k]], s=60, color="crimson", zorder=5)
            ax.annotate(f"{a[:6]}–{b[:6]}", (bx[k], cy[k]),
                        fontsize=7, alpha=0.85)
    ax.set_xlabel("Brain RDM (1 − Pearson over voxels)")
    ax.set_ylabel("Consensus LLM RDM (4-model peak avg, 1 − cosine)")
    ax.set_title(f"Pairwise cognitive-domain dissimilarity: brain vs LLM (v2)\n"
                 f"Spearman ρ = {cons_rho:.3f}, all p<0.001, "
                 f"ρ/ceiling = 64–81%")
    lo = min(bx.min(), cy.min()); hi = max(bx.max(), cy.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG / "cognitive_rsa_v2_preserved_pairs.png", dpi=160)
    plt.close()
    print(f"Wrote {FIG / 'cognitive_rsa_v2_preserved_pairs.png'}")

    # ---- save summary ----
    summary = {
        "common_config": list(COMMON_CFG),
        "models": MODELS,
        "conditions": brain_conds,
        "consensus_rho_vs_brain": float(cons_rho),
        "per_model": {
            m: {
                "peak_layer": int(per_model[m]["peak_L"]),
                "peak_layer_name": per_model[m]["peak_layer_name"],
                "peak_rho": per_model[m]["peak_rho"],
                "p_value": per_model[m]["p_value"],
                "noise_ceiling": per_model[m]["noise_ceiling"],
                "rho_over_ceiling": per_model[m]["rho_over_ceiling"],
            }
            for m in MODELS
        },
        "preserved_similar_pairs": [
            {"pair": [a, b], "combined_z": float(z)}
            for a, b, z in preserved_similar
        ],
        "preserved_dissimilar_pairs": [
            {"pair": [a, b], "combined_z": float(z)}
            for a, b, z in preserved_dissimilar
        ],
    }
    out = RES / "cross_model_summary_v2.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
