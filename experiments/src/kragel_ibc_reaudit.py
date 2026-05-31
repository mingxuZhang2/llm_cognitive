#!/usr/bin/env python3
"""
Re-audit Kragel & IBC controlled-fMRI validators after the HCP-ToM artifact lesson.

Two fixes vs the original scripts:
  1. Compare source RDMs against the CORRECT headline LLM RDMs
     ({model}_rdm14_headline.npz, mean_all|centered|1_cosine|peak), NOT the stale
     {model}_rsa_llm_rdms.npz (last_tok/raw/pearson) the old scripts used.
  2. Compare against the current PURE-Neurosynth brain_rdm.npz.

Outlier check (the HCP-ToM signature): for each condition, correlate its row in the
source RDM (Kragel/IBC) with the same condition's row in the Neurosynth RDM. A map
that is a bad outlier (like HCP social_tom_vs_random was, rho=-0.016) will have a
near-zero / negative row correlation -> flag it.

No GPU.
"""
import os, json
from pathlib import Path
from itertools import permutations
import numpy as np
import nibabel as nib
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
BRAIN = BASE / "data" / "brain_maps"
RSA = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "affective_validation" / "kragel_ibc_reaudit.json"
MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]


def rdm_from_vecs(vecs, mask):
    X = np.stack([v[mask] for v in vecs]).astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    Xn = X / n
    return 1.0 - Xn @ Xn.T


def upper(m):
    iu = np.triu_indices(m.shape[0], 1); return m[iu]


def rowvals(R, i): return np.delete(R[i], i)


def exact_perm_p(a, b):
    """Exact label-permutation p (|rho| >= obs) over rows/cols of b."""
    obs = spearmanr(upper(a), upper(b))[0]
    rhos = [spearmanr(upper(a), upper(b[np.ix_(p, p)]))[0] for p in permutations(range(a.shape[0]))]
    rhos = np.array(rhos)
    return float(obs), float((np.abs(rhos) >= abs(obs) - 1e-12).mean())


def llm_subset(model, conds, order_conds):
    z = np.load(RSA / f"{model}_rdm14_headline.npz", allow_pickle=True)
    lr, lc = z["rdm"], list(z["conditions"])
    idx = [lc.index(c) for c in conds]
    return lr[np.ix_(idx, idx)]


def ns_subset(conds):
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    nr, nc = ns["rdm"], list(ns["conditions"])
    idx = [nc.index(c) for c in conds]
    return nr[np.ix_(idx, idx)], nr, nc


# ---------------- Kragel ----------------
def load_img(path):
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float64)).ravel()


def audit_kragel():
    KDIR = BRAIN / "kragel_emotion"
    files = {"anger": "mean_3comp_angry_group_emotion_PLS_beta_BSz_10000it.img",
             "fear": "mean_3comp_fearful_group_emotion_PLS_beta_BSz_10000it.img",
             "sadness": "mean_3comp_sad_group_emotion_PLS_beta_BSz_10000it.img",
             "happiness": "mean_3comp_amused_group_emotion_PLS_beta_BSz_10000it.img"}
    conds = ["anger", "fear", "sadness", "happiness"]
    vecs = [load_img(KDIR / files[c]) for c in conds]
    stacked = np.stack(vecs)
    mask = np.all(np.isfinite(stacked) & (stacked != 0), axis=0)
    src = rdm_from_vecs(vecs, mask)
    return conds, src, int(mask.sum())


# ---------------- IBC ----------------
def group_mean_vec(cdir, ref):
    from nibabel.processing import resample_from_to
    files = sorted(f for f in os.listdir(cdir) if f.endswith(".nii.gz"))
    acc, n, ra, rs = None, 0, ref[0], ref[1]
    for f in files:
        img = nib.load(str(Path(cdir) / f))
        if rs is None:
            ra, rs = img.affine, img.shape[:3]
        if img.shape[:3] != rs or not np.allclose(img.affine, ra):
            img = resample_from_to(img, (rs, ra), order=1)
        d = np.nan_to_num(img.get_fdata(dtype=np.float64))
        acc = d if acc is None else acc + d
        n += 1
    return (acc / n).ravel(), (ra, rs)


def audit_ibc():
    IDIR = BRAIN / "ibc"
    CHOICE = [("face-shape", "fear"), ("reward-punishment", "valence"),
              ("belief-mechanistic_video", "belief"), ("mental-random", "theory_of_mind"),
              ("intention-control", "intention"), ("trusty-control", "judgment")]
    conds, vecs, ref = [], [], (None, None)
    for cdir, cond in CHOICE:
        v, ref = group_mean_vec(IDIR / cdir, ref)
        conds.append(cond); vecs.append(v)
    stacked = np.stack(vecs)
    mask = np.all(np.isfinite(stacked) & (stacked != 0), axis=0)
    src = rdm_from_vecs(vecs, mask)
    return conds, src, int(mask.sum())


def report(name, conds, src):
    print(f"\n{'='*64}\n{name}: {len(conds)} conditions {conds}\n{'='*64}")
    ns_sub, ns_full, ns_full_conds = ns_subset(conds)

    # outlier check: each condition's row in source vs Neurosynth subset
    print("Per-map outlier check (source row vs Neurosynth row; near-0/neg = bad map):")
    bad = []
    for i, c in enumerate(conds):
        r = spearmanr(rowvals(src, i), rowvals(ns_sub, i))[0]
        flag = "  <-- OUTLIER?" if r < 0.1 else ""
        if r < 0.1: bad.append(c)
        print(f"   {c:16s} row-rho = {r:+.3f}{flag}")

    # source vs Neurosynth (brain-brain)
    rho_ns, p_ns = exact_perm_p(src, ns_sub)
    print(f"\n[{name} vs Neurosynth]  rho={rho_ns:+.3f}  exact_p={p_ns:.3f}")

    # source vs each LLM (correct headline RDM)
    llm_rhos = {}
    for m in MODELS:
        L = llm_subset(m, conds, conds)
        rho, p = exact_perm_p(src, L)
        llm_rhos[m] = {"rho": rho, "exact_p": p}
        print(f"[{name} vs {m:28s}] rho={rho:+.3f}  exact_p={p:.3f}")
    mean_llm = float(np.mean([llm_rhos[m]["rho"] for m in MODELS]))
    print(f"-> mean LLM rho = {mean_llm:+.3f}")
    return {"conditions": conds, "src_vs_ns_rho": rho_ns, "src_vs_ns_p": p_ns,
            "outlier_flags": bad, "llm": llm_rhos, "mean_llm_rho": mean_llm}


def main():
    res = {}
    kc, ks, kv = audit_kragel()
    res["kragel"] = report("KRAGEL", kc, ks); res["kragel"]["voxels"] = kv
    ic, isrc, iv = audit_ibc()
    res["ibc"] = report("IBC", ic, isrc); res["ibc"]["voxels"] = iv

    print(f"\n{'#'*64}\nSUMMARY (corrected, headline LLM RDM + pure-NS brain)\n{'#'*64}")
    for k in ("kragel", "ibc"):
        r = res[k]
        print(f"  {k.upper():7s}: vs-Neurosynth {r['src_vs_ns_rho']:+.3f} | "
              f"mean-LLM {r['mean_llm_rho']:+.3f} | outliers={r['outlier_flags'] or 'none'}")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
