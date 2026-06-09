#!/usr/bin/env python3
"""
WHY does enriched input improve emotion differentiation?

Analysis 1: Layer-wise within-affective RSA for natural vs template-matched stimuli.
  → At which layer does the divergence appear?

Analysis 2: PCA of emotion centroids under both conditions.
  → Short input = 1D blob? Rich input = multi-dimensional structure?

Analysis 3: Within-affective distance matrix comparison.
  → Which emotion PAIRS gain the most from enriched input?

CPU-only. Processes one model at a time from existing per-stim NPZ files.
"""
from __future__ import annotations
import gc, json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from itertools import combinations
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

AFF = ["anger", "disgust", "fear", "happiness", "sadness", "valence"]
AFF5 = ["anger", "disgust", "fear", "happiness", "sadness"]

PEAK = {
    "Qwen2.5-7B-Instruct": 27,
    "Meta-Llama-3.1-8B-Instruct": 31,
    "Mistral-7B-Instruct-v0.3": 14,
    "gemma-2-9b-it": 21,
}
MODELS = list(PEAK.keys())


def load_acts(npz_path, conditions_out):
    npz = np.load(npz_path, allow_pickle=True)
    conditions = list(npz["conditions"])
    pool_idx = list(npz["pooling_names"]).index("mean_all")
    acts = npz["per_stim_activations"][pool_idx]  # (n_stim, n_layers, hidden)
    conditions_out.extend(conditions)
    del npz
    return acts


def centroid_cos_dist(X1, X2):
    mu1, mu2 = X1.mean(0), X2.mean(0)
    cos = np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2) + 1e-10)
    return float(1 - cos)


def within_aff_rsa_at_layer(acts, conditions, layer, brain_aff_dists, aff_list):
    a = acts[:, layer, :].astype(np.float32)
    uniq = sorted(set(conditions))
    cond_stims = {c: a[[i for i, cc in enumerate(conditions) if cc == c]] for c in uniq}

    # Center
    all_centroids = [cond_stims[c].mean(0) for c in uniq if c in set(aff_list)]
    if not all_centroids:
        return None, None
    gm = np.mean(all_centroids, axis=0)
    for c in cond_stims:
        cond_stims[c] = cond_stims[c] - gm

    pairs = list(combinations(aff_list, 2))
    llm_dists = []
    for c1, c2 in pairs:
        if c1 in cond_stims and c2 in cond_stims:
            llm_dists.append(centroid_cos_dist(cond_stims[c1], cond_stims[c2]))
        else:
            return None, None

    rho, p = spearmanr(brain_aff_dists, llm_dists)
    return float(rho), float(p)


def analyze_model(model_name):
    nat_path = RSA / f"{model_name}_rsa_v2_per_stim.npz"
    tm_path = RSA / f"{model_name}_tm_rsa_v2_per_stim.npz"

    if not nat_path.exists() or not tm_path.exists():
        print(f"  SKIP {model_name}: missing files")
        return None

    # Brain within-affective distances
    brain = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_conds = list(brain["conditions"])
    brain_rdm = brain["rdm"]

    brain_aff5 = [brain_rdm[brain_conds.index(c1), brain_conds.index(c2)]
                  for c1, c2 in combinations(AFF5, 2)]

    peak = PEAK[model_name]

    # ── Load natural stimuli ──
    print(f"  Loading natural: {model_name}...", flush=True)
    nat_conds = []
    nat_acts = load_acts(nat_path, nat_conds)
    n_stim_nat, n_layers, hd = nat_acts.shape

    # ── Load template-matched ──
    print(f"  Loading TM: {model_name}...", flush=True)
    tm_conds = []
    tm_acts = load_acts(tm_path, tm_conds)
    n_stim_tm = tm_acts.shape[0]

    print(f"  Natural: {n_stim_nat} stim, TM: {n_stim_tm} stim, {n_layers} layers, dim={hd}")

    # ═══ ANALYSIS 1: Layer-wise within-aff RSA ═══
    print(f"\n  --- Analysis 1: Layer-wise within-aff RSA ---")
    layer_results = []
    for li in range(n_layers):
        rho_nat, p_nat = within_aff_rsa_at_layer(nat_acts, nat_conds, li, brain_aff5, AFF5)
        rho_tm, p_tm = within_aff_rsa_at_layer(tm_acts, tm_conds, li, brain_aff5, AFF5)
        layer_results.append({
            "layer": li,
            "natural_rho": rho_nat, "natural_p": p_nat,
            "tm_rho": rho_tm, "tm_p": p_tm,
        })
        if li % 5 == 0 or li == n_layers - 1:
            print(f"    L{li:2d}: natural={rho_nat:+.3f}, TM={rho_tm:+.3f}")

    # Find divergence point
    divergence_layer = None
    for lr in layer_results:
        if lr["tm_rho"] is not None and lr["natural_rho"] is not None:
            if lr["tm_rho"] - lr["natural_rho"] > 0.3:
                divergence_layer = lr["layer"]
                break

    print(f"\n    Peak layer: L{peak}")
    nat_peak = layer_results[peak]["natural_rho"]
    tm_peak = layer_results[peak]["tm_rho"]
    print(f"    At peak: natural={nat_peak:+.3f}, TM={tm_peak:+.3f}, Δ={tm_peak-nat_peak:+.3f}")
    if divergence_layer is not None:
        print(f"    First divergence (Δ>0.3): L{divergence_layer}")

    # ═══ ANALYSIS 2: PCA of emotion centroids ═══
    print(f"\n  --- Analysis 2: PCA of emotion centroids at peak layer ---")

    for label, acts, conds in [("Natural", nat_acts, nat_conds), ("TM", tm_acts, tm_conds)]:
        a = acts[:, peak, :].astype(np.float32)
        uniq = sorted(set(conds))
        cond_stims = {c: a[[i for i, cc in enumerate(conds) if cc == c]] for c in uniq}
        centroids_aff = np.array([cond_stims[c].mean(0) for c in AFF5])
        # Center
        centroids_aff = centroids_aff - centroids_aff.mean(0)

        pca = PCA(n_components=min(5, len(AFF5)))
        coords = pca.fit_transform(centroids_aff)
        var_explained = pca.explained_variance_ratio_

        # Effective dimensionality (participation ratio)
        pr = (np.sum(var_explained) ** 2) / np.sum(var_explained ** 2)

        print(f"\n    {label} stimuli:")
        print(f"      Variance explained: {' '.join(f'PC{i+1}={v:.1%}' for i, v in enumerate(var_explained[:4]))}")
        print(f"      Effective dimensionality (PR): {pr:.2f}")
        print(f"      PC1 dominance: {var_explained[0]:.1%}")

        # What emotions load on PC1?
        print(f"      PC1 loadings: ", end="")
        for i, emo in enumerate(AFF5):
            print(f"{emo}={coords[i,0]:+.3f}", end="  ")
        print()
        if len(var_explained) > 1:
            print(f"      PC2 loadings: ", end="")
            for i, emo in enumerate(AFF5):
                print(f"{emo}={coords[i,1]:+.3f}", end="  ")
            print()

    # ═══ ANALYSIS 3: Which pairs gain most from enrichment? ═══
    print(f"\n  --- Analysis 3: Which emotion pairs gain most from enrichment? ---")

    nat_a = nat_acts[:, peak, :].astype(np.float32)
    tm_a = tm_acts[:, peak, :].astype(np.float32)

    nat_stims = {c: nat_a[[i for i, cc in enumerate(nat_conds) if cc == c]] for c in sorted(set(nat_conds))}
    tm_stims = {c: tm_a[[i for i, cc in enumerate(tm_conds) if cc == c]] for c in sorted(set(tm_conds))}

    # Center
    nat_gm = np.mean([nat_stims[c].mean(0) for c in AFF5], 0)
    tm_gm = np.mean([tm_stims[c].mean(0) for c in AFF5], 0)
    for c in AFF5:
        nat_stims[c] = nat_stims[c] - nat_gm
        tm_stims[c] = tm_stims[c] - tm_gm

    pairs = list(combinations(AFF5, 2))
    pair_gains = []
    print(f"\n    {'Pair':>25s} | {'Natural dist':>12s} | {'TM dist':>12s} | {'Gain':>8s} | {'Brain dist':>10s}")
    print(f"    {'-'*75}")
    for c1, c2 in pairs:
        d_nat = centroid_cos_dist(nat_stims[c1], nat_stims[c2])
        d_tm = centroid_cos_dist(tm_stims[c1], tm_stims[c2])
        d_brain = brain_rdm[brain_conds.index(c1), brain_conds.index(c2)]
        gain = d_tm / d_nat if d_nat > 1e-6 else 0
        pair_gains.append({"pair": f"{c1}—{c2}", "nat": d_nat, "tm": d_tm,
                           "gain": gain, "brain": d_brain})
        print(f"    {c1+'—'+c2:>25s} | {d_nat:12.5f} | {d_tm:12.5f} | {gain:7.1f}× | {d_brain:10.4f}")

    # Does enrichment gain correlate with brain distance?
    gains = [p["gain"] for p in pair_gains]
    brains = [p["brain"] for p in pair_gains]
    rho_gain_brain, p_gain_brain = spearmanr(gains, brains)
    print(f"\n    Enrichment gain vs brain distance: ρ={rho_gain_brain:+.3f} (p={p_gain_brain:.3f})")
    print(f"    (positive = pairs that brain separates most gain most from enrichment)")

    del nat_acts, tm_acts
    gc.collect()

    return {
        "model": model_name,
        "peak_layer": peak,
        "layer_wise_rsa": layer_results,
        "divergence_layer": divergence_layer,
        "pair_enrichment": pair_gains,
        "gain_brain_correlation": {"rho": float(rho_gain_brain), "p": float(p_gain_brain)},
    }


def main():
    print("=" * 75)
    print("WHY DOES ENRICHED INPUT IMPROVE EMOTION DIFFERENTIATION?")
    print("=" * 75)

    all_results = {}
    for m in MODELS:
        print(f"\n{'='*75}")
        print(f"MODEL: {m}")
        print(f"{'='*75}")
        r = analyze_model(m)
        if r:
            all_results[m] = r

    # Cross-model summary
    print(f"\n{'='*75}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*75}")

    print(f"\n  Layer-wise: where does TM overtake natural?")
    for m in MODELS:
        if m in all_results:
            dl = all_results[m].get("divergence_layer")
            pk = all_results[m]["peak_layer"]
            lr = all_results[m]["layer_wise_rsa"]
            nat_pk = lr[pk]["natural_rho"]
            tm_pk = lr[pk]["tm_rho"]
            print(f"    {m:>35s}: diverge@L{dl}, peak@L{pk}: nat={nat_pk:+.3f} TM={tm_pk:+.3f}")

    print(f"\n  Enrichment gain vs brain distance:")
    for m in MODELS:
        if m in all_results:
            r = all_results[m]["gain_brain_correlation"]
            print(f"    {m:>35s}: ρ={r['rho']:+.3f} (p={r['p']:.3f})")

    json.dump(all_results, open(OUT / "emotion_input_mechanism.json", "w"),
              indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved: {OUT / 'emotion_input_mechanism.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
