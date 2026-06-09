#!/usr/bin/env python
"""
Cross-architecture Narratives brain-LLM RSA (the "same story to real brain & model" defense).

For each of the 4 architectures, build an LLM RDM from the Narratives story sentences
(aggregate per-sentence activations by their llm_labels condition, across the 12 target
stories -> per-condition centroid -> center -> cosine RDM) and compare it to the real-fMRI
Narratives brain RDM (results/narratives_brain_rdm/narratives_brain_rdm.npz, 13 conditions,
built from real BOLD on the SAME text). This does NOT use the Neurosynth maps, so it answers
the "Neurosynth is text-derived" confound.

Rigor on layer choice (avoid double-dipping the Narratives data):
  - PRIMARY = rho at the FROZEN Neurosynth-headline peak layer (Qwen-7B L27, Llama L31,
    Mistral L14, Gemma L21) -> a pre-registered layer, no circularity.
  - Also report best-layer rho (exploratory) and last-layer rho (what the prior multistory
    analysis used), plus the full per-layer sweep, for full transparency.
Permutation p at the frozen peak shuffles condition labels.

Light enough for the login node (loads only the small per-story LLM npz). No GPU.
Writes results/narratives_brain_rdm/narratives_cross_arch.json
"""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NARR = os.path.join(BASE, "results", "narratives_brain_rdm")
LLM_DIR = os.path.join(BASE, "results", "narratives_llm")
SEED = 20260525
N_PERM = 10000
MIN_SENTS = 5   # min LLM sentences for a condition centroid to be used

TARGET_STORIES = ["pieman", "tunnel", "notthefallintact", "black", "prettymouth",
                  "forgot", "sherlock", "merlin", "bronx", "21styear",
                  "slumlordreach", "lucy"]
# frozen peak layer from the Neurosynth headline (no double-dipping on Narratives)
FROZEN_PEAK = {"Qwen2.5-7B-Instruct": 27, "Meta-Llama-3.1-8B-Instruct": 31,
               "Mistral-7B-Instruct-v0.3": 14, "gemma-2-9b-it": 21}
MODELS = list(FROZEN_PEAK.keys())


def utri(M):
    iu = np.triu_indices_from(M, k=1); return M[iu]

def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

def rdm_cosine(act):
    n = np.linalg.norm(act, axis=1, keepdims=True); n[n == 0] = 1.0
    Xn = act / n
    return 1.0 - Xn @ Xn.T


def collect_model(model_short):
    """Return {condition: list-of-(n_layers,hidden) sentence activations} pooled over stories."""
    cond_acts = defaultdict(list)
    n_layers = None
    nfiles = 0
    for story in TARGET_STORIES:
        fn = os.path.join(LLM_DIR, f"{model_short}_{story}_narratives.npz")
        if not os.path.exists(fn):
            continue
        nfiles += 1
        d = np.load(fn, allow_pickle=True)
        acts = d["activations"]          # (n_sents, n_layers, hidden)
        labels = d["labels"]
        n_layers = acts.shape[1]
        for i, labs in enumerate(labels):
            for lab in labs:
                cond_acts[lab].append(acts[i])   # keep all layers; pick layer later
    return cond_acts, n_layers, nfiles


def llm_rdm_at_layer(cond_acts, conds, L):
    cm = np.stack([np.mean([a[L] for a in cond_acts[c]], axis=0).astype(np.float64)
                   for c in conds])
    cm -= cm.mean(axis=0, keepdims=True)
    return rdm_cosine(cm)


def main():
    brain = np.load(os.path.join(NARR, "narratives_brain_rdm.npz"), allow_pickle=True)
    brain_rdm = brain["rdm"].astype(float)
    brain_conds = [str(c) for c in brain["conditions"]]
    print(f"Narratives brain RDM: {brain_rdm.shape}, conditions: {brain_conds}\n")

    rng = np.random.default_rng(SEED)
    out = {"brain_conditions": brain_conds, "models": {}}

    print(f"{'model':28s} {'#cond':>5s} {'rho@frozen':>11s} {'p':>7s} {'rho@best(L)':>14s} {'rho@last':>9s}")
    print("-" * 80)
    for model in MODELS:
        cond_acts, n_layers, nfiles = collect_model(model)
        if nfiles == 0:
            print(f"{model:28s}  [no extractions yet]"); continue
        # conditions usable: in brain RDM AND >=MIN_SENTS LLM sentences
        conds = [c for c in brain_conds if len(cond_acts.get(c, [])) >= MIN_SENTS]
        ov_brain = [brain_conds.index(c) for c in conds]
        sub_brain = brain_rdm[np.ix_(ov_brain, ov_brain)]
        bz = utri(sub_brain)

        # full layer sweep
        sweep = []
        for L in range(n_layers):
            rdm = llm_rdm_at_layer(cond_acts, conds, L)
            sweep.append(spearman(bz, utri(rdm)))
        sweep = np.array(sweep)

        frozenL = min(FROZEN_PEAK[model], n_layers - 1)
        rho_frozen = float(sweep[frozenL])
        bestL = int(np.argmax(sweep)); rho_best = float(sweep[bestL])
        rho_last = float(sweep[-1])

        # permutation p at frozen layer (shuffle condition labels of the LLM RDM)
        rdm_frozen = llm_rdm_at_layer(cond_acts, conds, frozenL)
        lz = utri(rdm_frozen)
        k = len(conds)
        null = np.empty(N_PERM)
        for p in range(N_PERM):
            perm = rng.permutation(k)
            null[p] = spearman(bz, utri(rdm_frozen[np.ix_(perm, perm)]))
        pval = float((np.sum(null >= rho_frozen) + 1) / (N_PERM + 1))

        out["models"][model] = {
            "n_stories": nfiles, "n_conditions": k, "conditions": conds,
            "n_layers": int(n_layers), "frozen_peak_layer": frozenL,
            "rho_frozen_peak": rho_frozen, "perm_p_frozen": pval,
            "rho_best": rho_best, "best_layer": bestL, "rho_last": rho_last,
            "sentences_per_condition": {c: len(cond_acts[c]) for c in conds},
            "layer_sweep": [round(float(x), 4) for x in sweep],
        }
        print(f"{model:28s} {k:>5d} {rho_frozen:>+11.3f} {pval:>7.4f} "
              f"{rho_best:>+10.3f}(L{bestL:>2d}) {rho_last:>+9.3f}")

    vals = [m["rho_frozen_peak"] for m in out["models"].values()]
    if vals:
        out["summary"] = {"mean_rho_frozen_peak": round(float(np.mean(vals)), 3),
                          "min": round(float(np.min(vals)), 3),
                          "max": round(float(np.max(vals)), 3),
                          "n_models": len(vals)}
        print(f"\nmean rho @ frozen peak across {len(vals)} models: "
              f"{out['summary']['mean_rho_frozen_peak']:+.3f} "
              f"(range {out['summary']['min']:+.3f}..{out['summary']['max']:+.3f})")
        print("Reference: prior Qwen-only Narratives rho ~0.54-0.58; brain split-half ceiling ~0.76.")

    outpath = os.path.join(NARR, "narratives_cross_arch.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
