#!/usr/bin/env python
"""
Within-block / block-partial control for the brain-LLM RSA headline.

The headline rho ~= 0.73 between the 14x14 brain RDM and each LLM RDM could in
principle be carried almost entirely by the single emotion <-> social-cognition
2-block split (our own causal-ablation result shows removing that one axis sends
rho +0.73 -> -0.36). A hostile reviewer will ask: once you control for the
categorical 2-block structure, do brain and LLM still agree on the *within-block*
fine geometry, or is rho=0.73 just "both agree emotion != social" (1 bit)?

This script answers that directly, with no GPU and tiny matrices:
  1. Full rho(brain, LLM)                          -- reproduce the headline.
  2. rho(brain, block), rho(LLM, block)            -- how categorical each side is.
  3. partial rho(brain, LLM | block)               -- agreement beyond the split.
  4. within-affective rho (15 pairs)               -- emotion block internal geometry.
  5. within-social rho (28 pairs; 21 w/o moral)    -- social block internal geometry.
Each (3)-(5) gets a label-permutation p-value.

Run on the login node (RDMs are 14x14). Writes results/cognitive_rsa/within_block_control.json
"""
import json, os
import numpy as np
from scipy.stats import rankdata

RSA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cognitive_rsa")
SEED = 20260525
N_PERM = 10000

MODELS = {
    "Qwen2.5-7B":   "Qwen2.5-7B-Instruct_rdm14_headline.npz",
    "Llama-3.1-8B": "Meta-Llama-3.1-8B-Instruct_rdm14_headline.npz",
    "Mistral-7B":   "Mistral-7B-Instruct-v0.3_rdm14_headline.npz",
    "Gemma-2-9B":   "gemma-2-9b-it_rdm14_headline.npz",
}

AFFECTIVE = {"anger", "disgust", "fear", "happiness", "sadness", "valence"}
# everything else (incl. moral) is the social/mentalistic block

def utri(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]

def partial_spearman(x, y, z):
    """partial Spearman of x,y controlling z (rank then partial-Pearson)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return (rxy - rxz * ryz) / denom if denom > 0 else np.nan

def submatrix_rho(brain, llm, idx):
    return spearman(utri(brain[np.ix_(idx, idx)]), utri(llm[np.ix_(idx, idx)]))

def main():
    brain_npz = np.load(os.path.join(RSA_DIR, "brain_rdm.npz"), allow_pickle=True)
    conds = [str(c) for c in brain_npz["conditions"]]
    brain = brain_npz["rdm"].astype(float)
    n = len(conds)

    block = np.array([0 if c in AFFECTIVE else 1 for c in conds])
    aff_idx = np.where(block == 0)[0]
    soc_idx = np.where(block == 1)[0]
    soc_no_moral_idx = np.array([i for i in soc_idx if conds[i] != "moral"])

    # block-membership model RDM: 0 if same block, 1 if different
    blockM = (block[:, None] != block[None, :]).astype(float)
    z = utri(blockM)
    bz = utri(brain)

    rng = np.random.default_rng(SEED)
    out = {"conditions": conds,
           "affective": [conds[i] for i in aff_idx],
           "social":    [conds[i] for i in soc_idx],
           "rho_brain_block": float(spearman(bz, z)),
           "models": {}}

    print(f"conditions order: {conds}")
    print(f"affective block ({len(aff_idx)}): {[conds[i] for i in aff_idx]}")
    print(f"social block    ({len(soc_idx)}): {[conds[i] for i in soc_idx]}")
    print(f"\nrho(brain, block-model) = {out['rho_brain_block']:.3f}  "
          f"(how much the BRAIN RDM is just the 2-block split)\n")

    header = (f"{'model':<13} {'full':>7} {'LLM~blk':>8} {'partial|blk':>12} "
              f"{'within-aff':>16} {'within-soc':>16} {'soc(no moral)':>16}")
    print(header); print("-" * len(header))

    for name, fn in MODELS.items():
        llm = np.load(os.path.join(RSA_DIR, fn), allow_pickle=True)["rdm"].astype(float)
        lz = utri(llm)

        full = spearman(bz, lz)
        llm_block = spearman(lz, z)
        part = partial_spearman(bz, lz, z)
        waff = submatrix_rho(brain, llm, aff_idx)
        wsoc = submatrix_rho(brain, llm, soc_idx)
        wsoc_nm = submatrix_rho(brain, llm, soc_no_moral_idx)

        # permutation nulls: permute LLM condition labels.
        # - partial: full 14-label shuffle, block fixed to true brain order.
        # - within-block: shuffle labels WITHIN that block only.
        part_null = np.empty(N_PERM)
        waff_null = np.empty(N_PERM)
        wsoc_null = np.empty(N_PERM)
        wsoc_nm_null = np.empty(N_PERM)
        for p in range(N_PERM):
            perm = rng.permutation(n)
            lzp = utri(llm[np.ix_(perm, perm)])
            part_null[p] = partial_spearman(bz, lzp, z)
            # within-block: independent within-block shuffles
            pa = aff_idx[rng.permutation(len(aff_idx))]
            waff_null[p] = spearman(utri(brain[np.ix_(aff_idx, aff_idx)]),
                                    utri(llm[np.ix_(pa, pa)]))
            ps = soc_idx[rng.permutation(len(soc_idx))]
            wsoc_null[p] = spearman(utri(brain[np.ix_(soc_idx, soc_idx)]),
                                    utri(llm[np.ix_(ps, ps)]))
            psn = soc_no_moral_idx[rng.permutation(len(soc_no_moral_idx))]
            wsoc_nm_null[p] = spearman(utri(brain[np.ix_(soc_no_moral_idx, soc_no_moral_idx)]),
                                       utri(llm[np.ix_(psn, psn)]))

        def pval(obs, null):
            return float((np.sum(null >= obs) + 1) / (N_PERM + 1))

        rec = {
            "full_rho": float(full),
            "llm_block_rho": float(llm_block),
            "partial_rho_given_block": float(part),
            "partial_p": pval(part, part_null),
            "within_affective_rho": float(waff),
            "within_affective_p": pval(waff, waff_null),
            "within_social_rho": float(wsoc),
            "within_social_p": pval(wsoc, wsoc_null),
            "within_social_no_moral_rho": float(wsoc_nm),
            "within_social_no_moral_p": pval(wsoc_nm, wsoc_nm_null),
        }
        out["models"][name] = rec
        print(f"{name:<13} {full:>7.3f} {llm_block:>8.3f} "
              f"{part:>8.3f} p={rec['partial_p']:<.3f} "
              f"{waff:>8.3f} p={rec['within_affective_p']:<.3f} "
              f"{wsoc:>8.3f} p={rec['within_social_p']:<.3f} "
              f"{wsoc_nm:>8.3f} p={rec['within_social_no_moral_p']:<.3f}")

    # summary across models
    def mean(key):
        return float(np.mean([m[key] for m in out["models"].values()]))
    out["summary"] = {k: mean(k) for k in
                      ["full_rho", "llm_block_rho", "partial_rho_given_block",
                       "within_affective_rho", "within_social_rho",
                       "within_social_no_moral_rho"]}

    outpath = os.path.join(RSA_DIR, "within_block_control.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nmean across 4 models:")
    print(f"  full rho                 = {out['summary']['full_rho']:.3f}")
    print(f"  partial rho | block      = {out['summary']['partial_rho_given_block']:.3f}")
    print(f"  within-affective rho     = {out['summary']['within_affective_rho']:.3f}")
    print(f"  within-social rho        = {out['summary']['within_social_rho']:.3f}")
    print(f"  within-social (no moral) = {out['summary']['within_social_no_moral_rho']:.3f}")
    print(f"\nwrote {outpath}")

if __name__ == "__main__":
    main()
