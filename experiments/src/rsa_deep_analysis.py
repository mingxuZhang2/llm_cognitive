#!/usr/bin/env python3
"""
Three deep analyses on cognitive RSA alignment (no GPU needed):

1. GAP ANALYSIS: Per-pair residual decomposition — which cognitive relationships
   do text-trained LLMs fail to capture?
2. CONFUSION ANALYSIS: Does RDM geometry predict stimulus-level confusion?
   (geometry → behavior test)
3. CAUSAL ABLATION: Virtual ablation of the affective-mentalistic boundary direction.
   Show boundary-selective collapse of brain-LLM alignment.

Output:
  results/cognitive_rsa/deep_analysis.json
  figures/rsa_deep_{gap,confusion,ablation}.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results" / "cognitive_rsa"
FIG = BASE / "figures"

MODELS_7B = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "gemma-2-9b-it",
]

AFFECTIVE = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}
MENTALISTIC = {"belief", "mentalizing", "intention", "theory_of_mind",
               "empathy", "self_referential", "judgment"}


def normalize_centered(act):
    return act - act.mean(axis=0, keepdims=True)

def rdm_cosine(act):
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    a = act / norms
    return 1.0 - a @ a.T

def triu_vec(mat):
    return mat[np.triu_indices(mat.shape[0], k=1)]

def pair_cat(c1, c2):
    def g(c):
        if c in AFFECTIVE: return "aff"
        if c in MENTALISTIC: return "ment"
        return "moral"
    g1, g2 = g(c1), g(c2)
    if g1 == g2: return f"within-{g1}"
    return "cross-" + "-".join(sorted([g1, g2]))


def load_model(model_short, brain_conds):
    npz_path = RES / f"{model_short}_rsa_v2_per_stim.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    per_stim = data["per_stim_activations"]
    conditions = list(data["conditions"])
    pool_idx = list(data["pooling_names"]).index("mean_all")

    unique_conds = sorted(set(conditions))
    stim_cond_raw = np.array([unique_conds.index(c) for c in conditions])
    order = [unique_conds.index(c) for c in brain_conds]
    inv = {order[i]: i for i in range(len(order))}
    stim_cond = np.array([inv[stim_cond_raw[s]] for s in range(len(conditions))])

    n_layers = per_stim.shape[2]
    n_conds = len(unique_conds)
    cond_means_all = np.zeros((n_conds, n_layers, per_stim.shape[-1]), dtype=np.float32)
    for c in range(n_conds):
        idx = np.where(stim_cond_raw == c)[0]
        cond_means_all[c] = per_stim[pool_idx, idx].mean(axis=0)
    cond_means_all = cond_means_all[order]

    brain_rdm = np.load(RES / "brain_rdm.npz", allow_pickle=True)["rdm"]
    btr = triu_vec(brain_rdm)

    best_rho, best_L = -1, 0
    rdms = []
    for L in range(n_layers):
        act = normalize_centered(cond_means_all[:, L, :].astype(np.float64))
        rdm = rdm_cosine(act)
        rho, _ = spearmanr(btr, triu_vec(rdm))
        rdms.append(rdm)
        if rho > best_rho:
            best_rho, best_L = rho, L

    peak_stim = per_stim[pool_idx, :, best_L, :].astype(np.float64)
    peak_cond_means = cond_means_all[:, best_L, :].astype(np.float64)

    del per_stim, data, cond_means_all

    return {
        "model": model_short,
        "peak_L": best_L,
        "peak_rho": best_rho,
        "peak_rdm": rdms[best_L],
        "peak_cond_means": peak_cond_means,
        "peak_stim_acts": peak_stim,
        "stim_cond": stim_cond,
    }


# ===================================================================
# PART 1: GAP ANALYSIS
# ===================================================================
def gap_analysis(models_data, brain_rdm, brain_conds):
    print("\n" + "=" * 70)
    print("PART 1: GAP — which condition pairs does the LLM get wrong?")
    print("=" * 70)

    n = len(brain_conds)
    triu_idx = np.triu_indices(n, k=1)
    pairs = [(brain_conds[i], brain_conds[j]) for i, j in zip(*triu_idx)]
    cats = [pair_cat(c1, c2) for c1, c2 in pairs]

    brain_vec = triu_vec(brain_rdm)
    brain_ranks = rankdata(brain_vec)

    all_res = []
    for md in models_data:
        llm_vec = triu_vec(md["peak_rdm"])
        llm_ranks = rankdata(llm_vec)
        all_res.append(brain_ranks - llm_ranks)

    mean_res = np.mean(all_res, axis=0)
    std_res = np.std(all_res, axis=0)
    mean_llm_ranks = np.mean([rankdata(triu_vec(md["peak_rdm"])) for md in models_data], axis=0)

    agreement = np.array([
        abs(sum(np.sign(all_res[m][i]) for m in range(len(all_res)))) / len(all_res)
        for i in range(len(pairs))
    ])

    sorted_idx = np.argsort(-np.abs(mean_res))

    print(f"\n{'Pair':<42s} {'Type':<20s} {'BrRk':>6s} {'LLMRk':>6s} {'Resid':>7s} {'Dir':<12s} {'Agree':>5s}")
    pair_results = []
    for ri, idx in enumerate(sorted_idx):
        c1, c2 = pairs[idx]
        cat = cats[idx]
        br, lr, r = brain_ranks[idx], mean_llm_ranks[idx], mean_res[idx]
        direction = "brain>LLM" if r > 0 else "LLM>brain"
        if ri < 20:
            print(f"  {c1}—{c2:<32s} {cat:<20s} {br:>6.0f} {lr:>6.0f} {r:>+7.1f} {direction:<12s} {agreement[idx]:>5.0%}")
        pair_results.append({
            "pair": f"{c1}—{c2}", "category": cat,
            "brain_rank": float(br), "llm_rank": float(lr),
            "residual": float(r), "std": float(std_res[idx]),
            "direction": direction, "agreement": float(agreement[idx]),
        })

    print(f"\nBy category:")
    cat_summary = {}
    for cat_name in sorted(set(cats)):
        ci = [i for i, c in enumerate(cats) if c == cat_name]
        abs_r = np.abs(mean_res[ci])
        signed_r = mean_res[ci]
        cat_summary[cat_name] = {
            "n": len(ci), "mean_abs_residual": float(np.mean(abs_r)),
            "mean_signed": float(np.mean(signed_r)),
        }
        print(f"  {cat_name:<25s}  n={len(ci):>2d}  |res|={np.mean(abs_r):>5.1f}  signed={np.mean(signed_r):>+5.1f}")

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    cm = {"within-aff": "#e74c3c", "within-ment": "#3498db", "within-moral": "#27ae60",
          "cross-aff-ment": "#9b59b6", "cross-aff-moral": "#e67e22", "cross-ment-moral": "#1abc9c"}
    for i in range(len(pairs)):
        ax1.scatter(brain_ranks[i], mean_llm_ranks[i], c=cm.get(cats[i], "gray"),
                    s=50, alpha=0.7, edgecolors="k", linewidth=0.3, zorder=2)
    mn, mx = 0, len(pairs) + 1
    ax1.plot([mn, mx], [mn, mx], "k--", alpha=0.3, zorder=1)
    for ri in range(7):
        idx = sorted_idx[ri]
        c1, c2 = pairs[idx]
        ax1.annotate(f"{c1[:5]}–{c2[:5]}", (brain_ranks[idx], mean_llm_ranks[idx]),
                     fontsize=6.5, textcoords="offset points", xytext=(5, 5))
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(fc=cm[k], label=k) for k in sorted(cm) if k in set(cats)],
               fontsize=7, loc="upper left")
    ax1.set_xlabel("Brain distance rank", fontsize=11)
    ax1.set_ylabel("LLM distance rank (mean 4 models)", fontsize=11)
    ax1.set_title("Per-pair rank agreement", fontsize=12)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.2)

    res_mat = np.zeros((n, n))
    for idx, (i, j) in enumerate(zip(*triu_idx)):
        res_mat[i, j] = mean_res[idx]
        res_mat[j, i] = mean_res[idx]
    vlim = np.max(np.abs(res_mat))
    im = ax2.imshow(res_mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([c[:7] for c in brain_conds], rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([c[:7] for c in brain_conds], fontsize=7)
    ax2.set_title("Residual heatmap (red = brain says more distant)", fontsize=11)
    plt.colorbar(im, ax=ax2, fraction=0.046)
    plt.tight_layout()
    out = FIG / "rsa_deep_gap.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")

    return {"pairs": pair_results, "category_summary": cat_summary,
            "n_unanimous": int(np.sum(agreement == 1.0)),
            "total_pairs": len(pairs)}


# ===================================================================
# PART 2: CONFUSION — does geometry predict behavioral errors?
# ===================================================================
def confusion_analysis(models_data, brain_rdm, brain_conds):
    print("\n" + "=" * 70)
    print("PART 2: CONFUSION — does RDM geometry predict classification errors?")
    print("=" * 70)

    n = len(brain_conds)
    brain_vec = triu_vec(brain_rdm)
    rng = np.random.default_rng(2026)
    n_splits = 50

    all_conf = []
    all_acc = []
    all_rho_bc = []

    for md in models_data:
        stim_acts = md["peak_stim_acts"]
        stim_cond = md["stim_cond"]
        conf_mats = []
        accs = []

        for _ in range(n_splits):
            train_idx, test_idx = [], []
            for c in range(n):
                ci = np.where(stim_cond == c)[0].copy()
                rng.shuffle(ci)
                cut = len(ci) // 2
                train_idx.extend(ci[:cut])
                test_idx.extend(ci[cut:])

            centroids = np.zeros((n, stim_acts.shape[1]))
            for c in range(n):
                ct = [i for i in train_idx if stim_cond[i] == c]
                if ct:
                    centroids[c] = stim_acts[ct].mean(axis=0)

            cent_mean = centroids.mean(axis=0)
            centroids_c = centroids - cent_mean
            c_norms = np.linalg.norm(centroids_c, axis=1, keepdims=True)
            c_norms[c_norms == 0] = 1.0
            centroids_cn = centroids_c / c_norms

            conf = np.zeros((n, n))
            correct = 0
            for i in test_idx:
                s = stim_acts[i] - cent_mean
                s_norm = np.linalg.norm(s)
                if s_norm == 0:
                    continue
                s_n = s / s_norm
                sims = centroids_cn @ s_n
                true_c = stim_cond[i]
                pred_c = int(np.argmax(sims))
                conf[true_c, pred_c] += 1
                if pred_c == true_c:
                    correct += 1

            row_sums = conf.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            conf /= row_sums
            conf_mats.append(conf)
            accs.append(correct / len(test_idx) if test_idx else 0)

        mean_conf = np.mean(conf_mats, axis=0)
        mean_acc = np.mean(accs)

        conf_dist = 1.0 - mean_conf
        conf_dist_sym = (conf_dist + conf_dist.T) / 2.0
        np.fill_diagonal(conf_dist_sym, 0)
        rho_bc, p_bc = spearmanr(brain_vec, triu_vec(conf_dist_sym))

        all_conf.append(mean_conf)
        all_acc.append(mean_acc)
        all_rho_bc.append((rho_bc, p_bc))

        print(f"  {md['model'][:25]:<25s}  acc={mean_acc:.1%}  brain-confusion ρ={rho_bc:.4f} (p={p_bc:.1e})")

    grand_conf = np.mean(all_conf, axis=0)
    grand_acc = np.mean(all_acc)
    grand_dist = 1.0 - grand_conf
    grand_dist_sym = (grand_dist + grand_dist.T) / 2.0
    np.fill_diagonal(grand_dist_sym, 0)
    grand_rho, grand_p = spearmanr(brain_vec, triu_vec(grand_dist_sym))

    print(f"\n  Grand mean accuracy: {grand_acc:.1%}")
    print(f"  Grand brain-confusion ρ = {grand_rho:.4f}  (p = {grand_p:.1e})")

    # Top confused pairs (off-diagonal, excluding self)
    print(f"\n  Top confused pairs (highest off-diagonal confusion probability):")
    for i in range(n):
        for j in range(n):
            if i == j:
                grand_conf[i, j] = 0
    top_conf_idx = np.argsort(-grand_conf.ravel())
    shown = 0
    for flat_idx in top_conf_idx:
        i, j = divmod(flat_idx, n)
        if i == j:
            continue
        print(f"    {brain_conds[i]:>18s} → {brain_conds[j]:<18s}  P={grand_conf[i,j]:.3f}  "
              f"brain_dist={brain_rdm[i,j]:.3f}")
        shown += 1
        if shown >= 10:
            break

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    diag_mask = grand_conf.copy()
    np.fill_diagonal(diag_mask, np.nan)
    im1 = ax1.imshow(grand_conf, cmap="YlOrRd", vmin=0, vmax=grand_conf.max())
    for i in range(n):
        ax1.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=True, color="white", zorder=2))
        ax1.text(i, i, f"{grand_conf[i,i]:.0%}" if np.isfinite(grand_conf[i,i]) else "",
                ha="center", va="center", fontsize=6, zorder=3)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([c[:7] for c in brain_conds], rotation=45, ha="right", fontsize=7)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([c[:7] for c in brain_conds], fontsize=7)
    ax1.set_xlabel("Predicted (nearest centroid)")
    ax1.set_ylabel("True condition")
    ax1.set_title(f"Cross-validated confusion matrix\n(4 models avg, accuracy={grand_acc:.1%})")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    bv = brain_vec
    cv = triu_vec(grand_dist_sym)
    ax2.scatter(bv, cv, s=50, alpha=0.6, c="#2980b9", edgecolors="k", linewidth=0.3)
    z = np.polyfit(bv, cv, 1)
    xl = np.linspace(bv.min(), bv.max(), 100)
    ax2.plot(xl, np.polyval(z, xl), "r--", alpha=0.7, linewidth=2)
    ax2.set_xlabel("Brain RDM distance", fontsize=11)
    ax2.set_ylabel("LLM confusion distance (1 − P(confused))", fontsize=11)
    ax2.set_title(f"Brain geometry predicts LLM confusion\nρ = {grand_rho:.3f}, p = {grand_p:.1e}", fontsize=12)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG / "rsa_deep_confusion.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Wrote {out}")

    return {
        "per_model": [{"model": md["model"], "accuracy": float(all_acc[i]),
                       "brain_confusion_rho": float(all_rho_bc[i][0]),
                       "brain_confusion_p": float(all_rho_bc[i][1])}
                      for i, md in enumerate(models_data)],
        "grand_accuracy": float(grand_acc),
        "grand_brain_confusion_rho": float(grand_rho),
        "grand_brain_confusion_p": float(grand_p),
    }


# ===================================================================
# PART 3: CAUSAL ABLATION — boundary-selective manipulation
# ===================================================================
def causal_ablation(models_data, brain_rdm, brain_conds):
    print("\n" + "=" * 70)
    print("PART 3: CAUSAL ABLATION — remove boundary, measure selective collapse")
    print("=" * 70)

    n = len(brain_conds)
    triu_idx = np.triu_indices(n, k=1)
    brain_vec = triu_vec(brain_rdm)
    pairs = [(brain_conds[i], brain_conds[j]) for i, j in zip(*triu_idx)]
    cats = [pair_cat(c1, c2) for c1, c2 in pairs]

    aff_idx = [i for i, c in enumerate(brain_conds) if c in AFFECTIVE]
    ment_idx = [i for i, c in enumerate(brain_conds) if c in MENTALISTIC]

    cat_pair_idx = {}
    for cat_name in ["within-aff", "within-ment", "cross-aff-ment"]:
        cat_pair_idx[cat_name] = [i for i, c in enumerate(cats) if c == cat_name]

    rng = np.random.default_rng(42)
    n_random = 2000

    all_results = []
    all_random_cat_deltas = {cat: [] for cat in cat_pair_idx}

    for md in models_data:
        cond_acts = md["peak_cond_means"]

        orig_act = normalize_centered(cond_acts)
        orig_rdm = rdm_cosine(orig_act)
        orig_rho, _ = spearmanr(brain_vec, triu_vec(orig_rdm))

        orig_cat_rho = {}
        for cat_name, ci in cat_pair_idx.items():
            if len(ci) >= 4:
                r, _ = spearmanr(brain_vec[ci], triu_vec(orig_rdm)[ci])
                orig_cat_rho[cat_name] = float(r)

        # Boundary direction: affective centroid → mentalistic centroid
        aff_c = cond_acts[aff_idx].mean(axis=0)
        ment_c = cond_acts[ment_idx].mean(axis=0)
        boundary_dir = ment_c - aff_c
        boundary_dir /= np.linalg.norm(boundary_dir)

        # Ablate boundary direction
        proj = np.outer(cond_acts @ boundary_dir, boundary_dir)
        ablated = cond_acts - proj
        abl_act = normalize_centered(ablated)
        abl_rdm = rdm_cosine(abl_act)
        abl_rho, _ = spearmanr(brain_vec, triu_vec(abl_rdm))

        abl_cat_rho = {}
        for cat_name, ci in cat_pair_idx.items():
            if len(ci) >= 4:
                r, _ = spearmanr(brain_vec[ci], triu_vec(abl_rdm)[ci])
                abl_cat_rho[cat_name] = float(r)

        boundary_delta = abl_rho - orig_rho

        # Random direction ablation null distribution
        D = cond_acts.shape[1]
        random_deltas = np.zeros(n_random)
        random_cat_deltas_local = {cat: np.zeros(n_random) for cat in cat_pair_idx}

        for ri in range(n_random):
            rd = rng.standard_normal(D)
            rd /= np.linalg.norm(rd)
            proj_r = np.outer(cond_acts @ rd, rd)
            abl_r = normalize_centered(cond_acts - proj_r)
            rdm_r = rdm_cosine(abl_r)
            rdm_r_vec = triu_vec(rdm_r)
            rho_r, _ = spearmanr(brain_vec, rdm_r_vec)
            random_deltas[ri] = rho_r - orig_rho
            for cat_name, ci in cat_pair_idx.items():
                if len(ci) >= 4:
                    r, _ = spearmanr(brain_vec[ci], rdm_r_vec[ci])
                    random_cat_deltas_local[cat_name][ri] = r - orig_cat_rho.get(cat_name, 0)

        p_val = float(np.mean(random_deltas <= boundary_delta))

        print(f"\n  {md['model']}:")
        print(f"    Original ρ        = {orig_rho:.4f}")
        print(f"    Boundary ablation = {abl_rho:.4f}  (Δ = {boundary_delta:+.4f})")
        print(f"    Random ablation   : mean Δ = {np.mean(random_deltas):+.4f} ± {np.std(random_deltas):.4f}")
        print(f"    p(random ≤ boundary) = {p_val:.4f}")
        print(f"    Per-category:")
        for cat_name in ["within-aff", "within-ment", "cross-aff-ment"]:
            if cat_name in orig_cat_rho:
                d = abl_cat_rho.get(cat_name, 0) - orig_cat_rho[cat_name]
                rm = np.mean(random_cat_deltas_local[cat_name])
                print(f"      {cat_name:<20s}  Δρ = {d:+.4f}  (random mean: {rm:+.4f})")
                all_random_cat_deltas[cat_name].append(
                    float(np.mean(random_cat_deltas_local[cat_name])))

        all_results.append({
            "model": md["model"],
            "orig_rho": float(orig_rho),
            "abl_rho": float(abl_rho),
            "boundary_delta": float(boundary_delta),
            "random_mean": float(np.mean(random_deltas)),
            "random_std": float(np.std(random_deltas)),
            "p_value": float(p_val),
            "orig_cat_rho": orig_cat_rho,
            "abl_cat_rho": abl_cat_rho,
        })

    # --- Multi-dimensional ablation: top-k PCA components of boundary ---
    print(f"\n  === Multi-dimensional boundary ablation (1st model) ===")
    md0 = models_data[0]
    cond_acts0 = md0["peak_cond_means"]
    aff_c0 = cond_acts0[aff_idx].mean(axis=0)
    ment_c0 = cond_acts0[ment_idx].mean(axis=0)

    aff_acts = cond_acts0[aff_idx] - cond_acts0.mean(axis=0)
    ment_acts = cond_acts0[ment_idx] - cond_acts0.mean(axis=0)
    combined = np.vstack([aff_acts, ment_acts])
    labels = np.array([0]*len(aff_idx) + [1]*len(ment_idx))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(combined, labels)
    lda_dir = lda.scalings_.ravel()
    lda_dir /= np.linalg.norm(lda_dir)

    proj_lda = np.outer(cond_acts0 @ lda_dir, lda_dir)
    abl_lda = normalize_centered(cond_acts0 - proj_lda)
    rdm_lda = rdm_cosine(abl_lda)
    rho_lda, _ = spearmanr(brain_vec, triu_vec(rdm_lda))
    orig_rho0 = all_results[0]["orig_rho"]
    print(f"    LDA boundary ablation: ρ = {rho_lda:.4f} (Δ = {rho_lda - orig_rho0:+.4f})")

    # PCA ablation: remove top-k PCs and compare
    U, S, Vt = np.linalg.svd(normalize_centered(cond_acts0), full_matrices=False)
    pca_deltas = []
    for k in range(1, min(14, len(S))):
        proj_k = cond_acts0 @ Vt[:k].T @ Vt[:k]
        abl_k = normalize_centered(cond_acts0 - proj_k)
        rdm_k = rdm_cosine(abl_k)
        rho_k, _ = spearmanr(brain_vec, triu_vec(rdm_k))
        pca_deltas.append({"k": k, "rho": float(rho_k), "delta": float(rho_k - orig_rho0)})
    print(f"    PCA ablation (top-k components removed):")
    for pd in pca_deltas[:5]:
        print(f"      k={pd['k']}:  ρ = {pd['rho']:.4f}  (Δ = {pd['delta']:+.4f})")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Boundary vs random Δρ per model
    ax = axes[0]
    labels_m = [r["model"].split("-")[0][:8] for r in all_results]
    bd = [r["boundary_delta"] for r in all_results]
    rm = [r["random_mean"] for r in all_results]
    rs = [r["random_std"] for r in all_results]
    x = np.arange(len(all_results))
    ax.bar(x - 0.18, bd, 0.32, color="#c0392b", label="Boundary ablation", zorder=2)
    ax.bar(x + 0.18, rm, 0.32, color="#bdc3c7", yerr=rs, capsize=4,
           label="Random direction (mean±σ)", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m, fontsize=9)
    ax.set_ylabel("Δρ (change in brain-LLM alignment)")
    ax.set_title("Boundary vs random ablation")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2, zorder=0)

    # Panel 2: Per-category selective collapse
    ax = axes[1]
    cat_labels = ["Within\nAffective", "Within\nMentalistic", "Cross\nBoundary"]
    cat_keys = ["within-aff", "within-ment", "cross-aff-ment"]
    cat_colors = ["#e74c3c", "#3498db", "#9b59b6"]
    bd_cat = []
    rd_cat = []
    for ck in cat_keys:
        deltas = [r["abl_cat_rho"].get(ck, 0) - r["orig_cat_rho"].get(ck, 0)
                  for r in all_results if ck in r["orig_cat_rho"]]
        bd_cat.append(np.mean(deltas) if deltas else 0)
        rd_cat.append(np.mean(all_random_cat_deltas[ck]) if all_random_cat_deltas[ck] else 0)
    x2 = np.arange(3)
    ax.bar(x2 - 0.15, bd_cat, 0.28, color=cat_colors, edgecolor="k", linewidth=0.5,
           label="Boundary ablation", zorder=2)
    ax.bar(x2 + 0.15, rd_cat, 0.28, color="#bdc3c7", edgecolor="k", linewidth=0.5,
           label="Random ablation", zorder=2)
    ax.set_xticks(x2)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Δρ")
    ax.set_title("Which relationships collapse?")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.2, zorder=0)

    # Panel 3: PCA ablation curve
    ax = axes[2]
    ks = [p["k"] for p in pca_deltas]
    rhos = [p["rho"] for p in pca_deltas]
    ax.plot(ks, rhos, "o-", color="#2c3e50", linewidth=2, markersize=8)
    ax.axhline(orig_rho0, color="gray", linestyle="--", alpha=0.5, label=f"Original ρ={orig_rho0:.3f}")
    ax.axhline(rho_lda, color="#c0392b", linestyle="--", alpha=0.7,
               label=f"LDA boundary ρ={rho_lda:.3f}")
    ax.set_xlabel("Number of PCA components removed")
    ax.set_ylabel("Brain-LLM ρ")
    ax.set_title("Alignment vs dimensionality reduction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG / "rsa_deep_ablation.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\nWrote {out}")

    return {
        "per_model": all_results,
        "lda_ablation_rho": float(rho_lda),
        "lda_ablation_delta": float(rho_lda - orig_rho0),
        "pca_ablation": pca_deltas,
    }


# ===================================================================
def main():
    brain_data = np.load(RES / "brain_rdm.npz", allow_pickle=True)
    brain_rdm = brain_data["rdm"]
    brain_conds = list(brain_data["conditions"])
    print(f"Brain RDM: {brain_rdm.shape[0]} conditions: {brain_conds}")

    print("\nLoading models...")
    models_data = []
    for model in MODELS_7B:
        md = load_model(model, brain_conds)
        if md is not None:
            models_data.append(md)
            print(f"  {model}: peak L{md['peak_L']}, ρ={md['peak_rho']:.4f}")
        else:
            print(f"  [skip] {model}: NPZ not found")
    if not models_data:
        print("No models available!")
        return

    gap = gap_analysis(models_data, brain_rdm, brain_conds)
    confusion = confusion_analysis(models_data, brain_rdm, brain_conds)
    ablation = causal_ablation(models_data, brain_rdm, brain_conds)

    output = {
        "gap_analysis": gap,
        "confusion_analysis": confusion,
        "causal_ablation": ablation,
    }
    out_json = RES / "deep_analysis.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
