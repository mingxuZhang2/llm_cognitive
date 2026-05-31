#!/usr/bin/env python3
"""
Control: per-condition NOISE CEILING for the row-wise brain alignment, to test
whether the affective core's high alignment (rho~0.78) is "near ceiling" and
whether the mentalistic conditions' low alignment (rho~0.42) reflects genuine
mis-alignment vs. mere LLM-side noise.

Per condition c, the ceiling = LLM split-half self-consistency of c's RDM row:
  split each condition's stimuli into halves -> two 14x14 LLM RDMs (same headline
  recipe: mean_all | centered | 1_cosine | peak layer) -> rho between the two
  halves' row-c (13 off-diagonal entries). Average over K splits.

Interpretation:
  - high ceiling + high brain-align  -> genuinely brain-like (affective)
  - high ceiling + low  brain-align  -> reliably represented but NOT brain-like
  - low  ceiling                     -> condition is just noisy (align is unreliable)

Outputs per-condition: brain_align (from developmental_emergence 7B), ceiling,
align/ceiling ratio; aggregated by block, for all 4 models.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "affective_validation"
CFG = "mean_all|centered|1_cosine|v1_NS_only"

MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
AFF = {"anger", "fear", "disgust", "sadness", "happiness", "valence"}

def centered(a): return a - a.mean(axis=0, keepdims=True)
def cosine_rdm(a):
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1.0
    Xn = a / n; return 1.0 - Xn @ Xn.T

def main():
    ns = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    ns_rdm = ns["rdm"]; ns_conds = list(ns["conditions"])

    all_rows = []
    agg = {}
    for model in MODELS:
        j = json.load(open(RSA / f"{model}_rsa_v2.json"))
        peakL = j["results"][CFG]["peak_layer"]
        uniq = j["conditions"]
        z = np.load(RSA / f"{model}_rsa_v2_per_stim.npz", allow_pickle=True)
        per = z["per_stim_activations"]; conds = list(z["conditions"])
        pidx = list(z["pooling_names"]).index("mean_all")
        layer_act = per[pidx, :, peakL, :].astype(np.float64)   # [n_stim, hidden]
        cond_stims = {c: np.array([i for i, cc in enumerate(conds) if cc == c]) for c in uniq}

        # full-data LLM RDM + brain alignment per condition (matches Direction C)
        cmean = np.stack([layer_act[cond_stims[c]].mean(0) for c in uniq])
        llm_rdm = cosine_rdm(centered(cmean))
        order = [ns_conds.index(c) for c in uniq]
        ns_re = ns_rdm[np.ix_(order, order)]

        def rowvals(R, i): return np.delete(R[i], i)
        brain_align = {c: spearmanr(rowvals(llm_rdm, i), rowvals(ns_re, i))[0]
                       for i, c in enumerate(uniq)}

        # split-half ceiling per condition
        rng = np.random.default_rng(2026)
        K = 100
        ceil_acc = {c: [] for c in uniq}
        for _ in range(K):
            A = np.zeros((len(uniq), layer_act.shape[1])); B = np.zeros_like(A)
            ok = True
            for k, c in enumerate(uniq):
                idx = cond_stims[c].copy(); rng.shuffle(idx)
                h = len(idx) // 2
                if h < 1 or len(idx) - h < 1: ok = False; break
                A[k] = layer_act[idx[:h]].mean(0); B[k] = layer_act[idx[h:]].mean(0)
            if not ok: continue
            Ra = cosine_rdm(centered(A)); Rb = cosine_rdm(centered(B))
            for i, c in enumerate(uniq):
                r = spearmanr(rowvals(Ra, i), rowvals(Rb, i))[0]
                if np.isfinite(r): ceil_acc[c].append(r)
        ceiling = {c: float(np.mean(v)) for c, v in ceil_acc.items()}

        print(f"\n=== {model} (peak L{peakL}) ===")
        print(f"  {'cond':16s} {'brain_align':>11s} {'ceiling':>8s} {'ratio':>7s}  block")
        rows = []
        for c in sorted(uniq, key=lambda x: -brain_align[x]):
            blk = "AFF" if c in AFF else "ment"
            ratio = brain_align[c] / ceiling[c] if ceiling[c] > 0 else float("nan")
            print(f"  {c:16s} {brain_align[c]:>+11.3f} {ceiling[c]:>8.3f} {ratio:>+7.2f}  {blk}")
            rows.append({"model": model, "cond": c, "block": blk,
                         "brain_align": brain_align[c], "ceiling": ceiling[c], "ratio": ratio})
        all_rows += rows

        # aggregate
        def blkmean(key, blk):
            return float(np.mean([r[key] for r in rows if r["block"] == blk]))
        agg[model] = {
            "AFF":  {"align": blkmean("brain_align", "AFF"),  "ceiling": blkmean("ceiling", "AFF"),  "ratio": blkmean("ratio", "AFF")},
            "ment": {"align": blkmean("brain_align", "ment"), "ceiling": blkmean("ceiling", "ment"), "ratio": blkmean("ratio", "ment")},
        }
        a, m = agg[model]["AFF"], agg[model]["ment"]
        print(f"  -> AFF : align={a['align']:.2f} ceiling={a['ceiling']:.2f} ratio={a['ratio']:.2f}")
        print(f"  -> ment: align={m['align']:.2f} ceiling={m['ceiling']:.2f} ratio={m['ratio']:.2f}")

    print("\n" + "=" * 66)
    print("AGGREGATE across 4 models (mean)")
    print("=" * 66)
    for blk in ["AFF", "ment"]:
        al = np.mean([agg[m][blk]["align"] for m in MODELS])
        ce = np.mean([agg[m][blk]["ceiling"] for m in MODELS])
        ra = np.mean([agg[m][blk]["ratio"] for m in MODELS])
        print(f"  {blk:5s}: brain_align={al:+.3f}  ceiling={ce:.3f}  align/ceiling={ra:+.2f}")
    json.dump({"per_condition": all_rows, "aggregate": agg},
              open(OUT / "affective_ceiling_control.json", "w"), indent=2)
    print(f"\nSaved: {OUT/'affective_ceiling_control.json'}")

if __name__ == "__main__":
    main()
