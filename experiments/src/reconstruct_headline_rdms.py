#!/usr/bin/env python3
"""
Reconstruct the HEADLINE LLM RDMs (mean_all | centered | 1_cosine | v1_NS_only,
per-model peak layer) from the per-stim activation npz, so that all downstream
validation (HCP/IBC/Kragel subsets) compares against the SAME geometry that
produced the rho~0.63 result -- not the stale llm_rdms.npz (last_tok/raw/pearson).

Saves results/cognitive_rsa/{model}_rdm14_headline.npz with rdm(14x14)+conditions,
and prints the reproduced Spearman vs Neurosynth (should match ~0.63).
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
CFG = "mean_all|centered|1_cosine|v1_NS_only"
MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]


def normalize_centered(act):              # subtract mean across conditions (axis 0)
    return act - act.mean(axis=0, keepdims=True)


def rdm_cosine(act):
    n = np.linalg.norm(act, axis=1, keepdims=True); n[n == 0] = 1.0
    Xn = act / n
    return 1.0 - Xn @ Xn.T


def main():
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    ns_rdm = ns["rdm"]; ns_conds = list(ns["conditions"])

    for model in MODELS:
        j = json.load(open(RSA / f"{model}_rsa_v2.json"))
        peak_L = j["results"][CFG]["peak_layer"]
        head_rho = j["results"][CFG]["rho"]
        uniq = j["conditions"]                       # sorted unique conditions

        z = np.load(RSA / f"{model}_rsa_v2_per_stim.npz", allow_pickle=True)
        per = z["per_stim_activations"]              # [pool, stim, layer, hidden]
        conds = list(z["conditions"])
        pools = list(z["pooling_names"])
        pidx = pools.index("mean_all")

        # condition-mean at peak layer
        cond_to_stims = {c: [i for i, cc in enumerate(conds) if cc == c] for c in uniq}
        cmean = np.stack([per[pidx, cond_to_stims[c], peak_L, :].astype(np.float64).mean(0)
                          for c in uniq])            # [14, hidden]
        rdm = rdm_cosine(normalize_centered(cmean))

        # verify vs Neurosynth in matching order
        order = [ns_conds.index(c) for c in uniq]
        ns_re = ns_rdm[np.ix_(order, order)]
        iu = np.triu_indices(len(uniq), 1)
        rho, _ = spearmanr(rdm[iu], ns_re[iu])
        print(f"{model:30s} peak L{peak_L:2d}  reproduced rho={rho:+.3f}  (json says {head_rho:+.3f})")

        np.savez_compressed(RSA / f"{model}_rdm14_headline.npz",
                            rdm=rdm, conditions=np.array(uniq),
                            peak_layer=peak_L, config=CFG)
    print("\nSaved *_rdm14_headline.npz for all models.")


if __name__ == "__main__":
    main()
