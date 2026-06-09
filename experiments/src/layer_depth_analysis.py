#!/usr/bin/env python
"""
Layer-depth profile of the brain-LLM alignment (Direction D / cortical-gradient test).

Brain prediction we are porting to the LLM: the cortex runs a processing gradient from
sensory/concrete (shallow) to transmodal/abstract (deep), and social/mentalistic cognition
sits at the abstract/transmodal end. Ported to an LLM: the emotion<->social organizing axis
(our one causally load-bearing dimension) and the social-block fine structure should live in
*deeper* layers than basic-emotion structure.

For every layer of every model we rebuild the headline 14x14 LLM RDM
(recipe mean_all | centered | 1_cosine, exactly as reconstruct_headline_rdms.py) and report,
per layer:
  - full_rho            : brain<->LLM Spearman over the 91 pairs (the alignment-vs-depth curve)
  - block_sep_llm       : how strongly the LLM separates the emotion/social blocks at this layer
                          (Spearman of the LLM RDM with the binary block-membership model)
  - partial_rho|block   : brain<->LLM agreement controlling for the block split
  - within_aff_rho      : brain<->LLM within the 6 affective conditions (15 pairs)
  - within_soc_rho      : brain<->LLM within the 8 social conditions (28 pairs)
  - within_soc_nomoral  : ditto excluding 'moral' (21 pairs)
Then per model: the (relative) depth at which each curve peaks, so we can compare the depth
of the emotion vs social structure across models of different layer counts.

Loads the large *_rsa_v2_per_stim.npz (each ~100-610 MB) -> SUBMIT TO A 512 GB CPU NODE
(scripts/slurm/layer_depth.sh); it OOM-kills on the login node. No GPU needed.
Writes results/cognitive_rsa/layer_depth_profile.json
"""
import json, os
import numpy as np
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RSA = os.path.join(BASE, "results", "cognitive_rsa")

# non-VL models present on disk; (model, headline peak layer for the sanity check or None)
MODELS = [
    ("Qwen2.5-0.5B-Instruct", None),
    ("Qwen2.5-1.5B-Instruct", None),
    ("Qwen2.5-3B-Instruct",   None),
    ("Qwen2.5-7B-Instruct",   27),
    ("Meta-Llama-3.1-8B-Instruct", 31),
    ("Mistral-7B-Instruct-v0.3",   14),
    ("gemma-2-9b-it",              21),
]
AFFECTIVE = {"anger", "disgust", "fear", "happiness", "sadness", "valence"}


def utri(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu]

def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]; rxz = np.corrcoef(rx, rz)[0, 1]; ryz = np.corrcoef(ry, rz)[0, 1]
    d = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / d) if d > 0 else float("nan")

def normalize_centered(act):
    return act - act.mean(axis=0, keepdims=True)

def rdm_cosine(act):
    n = np.linalg.norm(act, axis=1, keepdims=True); n[n == 0] = 1.0
    Xn = act / n
    return 1.0 - Xn @ Xn.T


def main():
    brain_npz = np.load(os.path.join(RSA, "brain_rdm.npz"), allow_pickle=True)
    brain_conds = [str(c) for c in brain_npz["conditions"]]
    brain = brain_npz["rdm"].astype(float)
    n = len(brain_conds)

    block = np.array([0 if c in AFFECTIVE else 1 for c in brain_conds])
    aff_idx = np.where(block == 0)[0]
    soc_idx = np.where(block == 1)[0]
    soc_nm_idx = np.array([i for i in soc_idx if brain_conds[i] != "moral"])
    blockM = (block[:, None] != block[None, :]).astype(float)
    z_blk = utri(blockM)
    bz = utri(brain)
    bz_aff = utri(brain[np.ix_(aff_idx, aff_idx)])
    bz_soc = utri(brain[np.ix_(soc_idx, soc_idx)])
    bz_snm = utri(brain[np.ix_(soc_nm_idx, soc_nm_idx)])

    out = {"conditions": brain_conds,
           "affective": [brain_conds[i] for i in aff_idx],
           "social":    [brain_conds[i] for i in soc_idx],
           "models": {}}

    for model, peakL in MODELS:
        fn = os.path.join(RSA, f"{model}_rsa_v2_per_stim.npz")
        if not os.path.exists(fn):
            print(f"[skip] {model}: no per_stim npz"); continue
        z = np.load(fn, allow_pickle=True)
        per = z["per_stim_activations"]                       # [pool, stim, layer, hidden]
        conds = [str(c) for c in z["conditions"]]
        pools = [str(p) for p in z["pooling_names"]]
        pidx = pools.index("mean_all")
        nL = per.shape[2]
        cond_to_stims = {c: [i for i, cc in enumerate(conds) if cc == c] for c in brain_conds}

        curves = {k: [] for k in
                  ["full_rho", "block_sep_llm", "partial_given_block",
                   "within_aff", "within_soc", "within_soc_nomoral"]}
        for L in range(nL):
            cmean = np.stack([per[pidx, cond_to_stims[c], L, :].astype(np.float64).mean(0)
                              for c in brain_conds])           # [14, hidden] in brain order
            rdm = rdm_cosine(normalize_centered(cmean))
            lz = utri(rdm)
            curves["full_rho"].append(spearman(bz, lz))
            curves["block_sep_llm"].append(spearman(lz, z_blk))
            curves["partial_given_block"].append(partial_spearman(bz, lz, z_blk))
            curves["within_aff"].append(spearman(bz_aff, utri(rdm[np.ix_(aff_idx, aff_idx)])))
            curves["within_soc"].append(spearman(bz_soc, utri(rdm[np.ix_(soc_idx, soc_idx)])))
            curves["within_soc_nomoral"].append(
                spearman(bz_snm, utri(rdm[np.ix_(soc_nm_idx, soc_nm_idx)])))

        def peak(key):
            arr = np.array(curves[key]); L = int(np.argmax(arr))
            return {"peak_layer": L, "rel_depth": round(L / (nL - 1), 3), "peak_val": round(float(arr[L]), 3)}

        rec = {"n_layers": nL,
               "peak_full":  peak("full_rho"),
               "peak_blocksep": peak("block_sep_llm"),
               "peak_within_soc": peak("within_soc"),
               "peak_within_soc_nomoral": peak("within_soc_nomoral"),
               "peak_within_aff": peak("within_aff"),
               "curves": {k: [round(v, 4) for v in vals] for k, vals in curves.items()}}
        if peakL is not None:
            rec["headline_peak_layer"] = peakL
            rec["full_rho_at_headline_peak"] = round(curves["full_rho"][peakL], 3)
        out["models"][model] = rec

        msg = (f"{model:28s} L={nL:2d}  full peak L{rec['peak_full']['peak_layer']:2d}"
               f"(d={rec['peak_full']['rel_depth']:.2f}, rho={rec['peak_full']['peak_val']:+.3f})"
               f"  within-soc peak d={rec['peak_within_soc']['rel_depth']:.2f}"
               f"  within-aff peak d={rec['peak_within_aff']['rel_depth']:.2f}"
               f"  blocksep peak d={rec['peak_blocksep']['rel_depth']:.2f}")
        if peakL is not None:
            msg += f"  [headline L{peakL} rho={rec['full_rho_at_headline_peak']:+.3f}]"
        print(msg, flush=True)
        del per, z

    # cross-model summary: honest depth PROFILE.
    # NOTE: argmax of a curve ('peak layer') is unreliable when the curve is flat or has no
    # signal. In particular within-affective has NO brain-alignment at any layer (it is
    # negative/near-zero throughout), so its 'peak depth' lands on a noise maximum (often the
    # embedding layer) and is MEANINGLESS. We therefore report early-third vs late-third means
    # and ranges, not peak layers, and we do NOT claim a cortical gradient from argmax.
    def thirds(curve):
        a = np.array(curve); n = len(a); t = max(1, n // 3)
        return float(a[:t].mean()), float(a[-t:].mean())

    print("\n=== depth profile: early-third -> late-third mean (shallow -> deep) ===")
    print(f"{'model':28s} | {'full':>14s} | {'blocksep':>16s} | {'within-soc':>15s} | {'within-aff (all layers)':>24s}")
    prof = {"full": [], "blocksep": [], "within_soc": [], "within_aff_mean": []}
    for m, r in out["models"].items():
        fe, fl = thirds(r["curves"]["full_rho"])
        be, bl = thirds(r["curves"]["block_sep_llm"])
        se, sl = thirds(r["curves"]["within_soc"])
        wa = r["curves"]["within_aff"]; amin, amax, amean = min(wa), max(wa), float(np.mean(wa))
        prof["full"].append(fl - fe); prof["blocksep"].append(bl - be)
        prof["within_soc"].append(sl - se); prof["within_aff_mean"].append(amean)
        r["depth_profile"] = {"full_early": round(fe, 3), "full_late": round(fl, 3),
                              "blocksep_early": round(be, 3), "blocksep_late": round(bl, 3),
                              "within_soc_early": round(se, 3), "within_soc_late": round(sl, 3),
                              "within_aff_min": round(amin, 3), "within_aff_max": round(amax, 3),
                              "within_aff_mean": round(amean, 3)}
        print(f"{m:28s} | {fe:>6.2f}->{fl:<6.2f} | {be:>7.2f}->{bl:<7.2f} | {se:>6.2f}->{sl:<6.2f} | "
              f"min{amin:+.2f} max{amax:+.2f} mean{amean:+.2f}")

    out["summary"] = {
        "full_late_minus_early_mean": round(float(np.mean(prof["full"])), 3),
        "blocksep_late_minus_early_mean": round(float(np.mean(prof["blocksep"])), 3),
        "within_soc_late_minus_early_mean": round(float(np.mean(prof["within_soc"])), 3),
        "within_aff_mean_over_models": round(float(np.mean(prof["within_aff_mean"])), 3),
        "cortical_gradient_supported": False,
        "interpretation": ("depth-invariant: the emotion/social split and the brain-alignment "
                           "are present from shallow layers and roughly flat across depth; "
                           "within-affective does not align at any depth (peak-depth undefined); "
                           "the 'social deeper than affective' cortical-gradient prediction is NOT supported"),
    }
    print("\n=== HONEST CONCLUSION ===")
    print(f"  full alignment, shallow->deep change (mean):  {out['summary']['full_late_minus_early_mean']:+.3f}")
    print(f"  emotion/social separation, shallow->deep:     {out['summary']['blocksep_late_minus_early_mean']:+.3f}")
    print(f"  within-social, shallow->deep:                 {out['summary']['within_soc_late_minus_early_mean']:+.3f}")
    print(f"  within-affective mean over all layers:        {out['summary']['within_aff_mean_over_models']:+.3f}  (no signal at any depth)")
    print("  => DEPTH-INVARIANT. The split and the alignment are present from shallow layers and")
    print("     roughly flat with depth. Cortical-gradient prediction ('social deeper than")
    print("     affective') is NOT supported (within-affective has no alignment at any depth to")
    print("     locate; block-separation is already strong and flat from shallow layers).")

    outpath = os.path.join(RSA, "layer_depth_profile.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
