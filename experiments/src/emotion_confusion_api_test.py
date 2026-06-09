#!/usr/bin/env python3
"""
Emotion confusion test: does representational compression cause behavioral confusion?

Core logic:
  Brain says anger-sadness are VERY different (d=0.59).
  LLM says anger-sadness are NEARLY IDENTICAL (d=0.02).
  → When advising someone who is angry vs sad about the SAME situation,
    does the model give less differentiated advice for emotion pairs
    than for social-cognition pairs?

Design:
  10 scenarios. For each:
    - 3 emotion framings: anger, sadness, fear (most compressed pairs in LLM)
    - 3 social framings: belief, empathy, judgment (well-separated in LLM)

  Measure: mean pairwise TF-IDF distance WITHIN emotion framings vs WITHIN social framings.
  If compression matters: emotion advice should be MORE similar (less differentiated).

  Also: per-pair analysis aligned to LLM RDM distances and brain RDM distances.
  If the model's behavioral confusion tracks LLM geometry (compressed) rather than
  brain geometry (differentiated), that's the smoking gun.

60 API calls (~$0.30).
"""
from __future__ import annotations
import os, json, time
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu

BASE = Path(__file__).resolve().parents[1]
RSA = BASE / "results" / "cognitive_rsa"
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

SITUATIONS = [
    "Your teenage child was caught shoplifting from a store",
    "You discovered your business partner has been secretly diverting funds",
    "Your elderly parent's caregiver has been neglecting their duties",
    "A close friend publicly mocked your career at a dinner party",
    "You found out your medical records were shared without your consent",
    "Your neighbor has been deliberately damaging your property",
    "A teacher singled out and humiliated your child in front of the class",
    "You discovered that your promotion was given to a less qualified person due to favoritism",
    "Your sibling revealed a family secret that you had trusted them to keep",
    "A contractor took your payment but never completed the work on your home",
]

EMOTION_FRAMINGS = {
    "anger": "You feel a burning rage about this. Your jaw is clenched and you want to confront the situation head-on.",
    "sadness": "You feel a deep, hollow sadness about this. Your energy is drained and the loss feels overwhelming.",
    "fear": "You feel gripped by anxiety and dread about this. Your mind races with worst-case scenarios about what could happen next.",
}

SOCIAL_FRAMINGS = {
    "belief": "You believe this was completely deliberate and calculated — a conscious choice to harm you.",
    "empathy": "You find yourself thinking about why the other person did this — what pressures or circumstances might have led them here.",
    "judgment": "You see this as a clear moral violation — a line was crossed that should never be crossed regardless of circumstances.",
}

PROMPT = """A person is going through this situation: "{situation}"

{framing}

As their counselor, what specific steps should they take? Give concrete, actionable advice in 3-4 sentences. Tailor your advice specifically to what they are experiencing."""


def call_api(client, prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return ""


def pairwise_dist(texts):
    """TF-IDF cosine distance between all pairs."""
    if any(t == "" for t in texts):
        return {}
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sim = cosine_similarity(tfidf)
    n = len(texts)
    dists = {}
    for i, j in combinations(range(n), 2):
        dists[(i, j)] = float(1 - sim[i, j])
    return dists


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("=" * 70)
    print("EMOTION CONFUSION TEST: Does compression cause behavioral confusion?")
    print("=" * 70)

    emo_keys = list(EMOTION_FRAMINGS.keys())
    soc_keys = list(SOCIAL_FRAMINGS.keys())
    responses = {}

    for si, sit in enumerate(SITUATIONS):
        for emo, framing in EMOTION_FRAMINGS.items():
            prompt = PROMPT.format(situation=sit, framing=framing)
            print(f"  emo:{emo}×s{si}", end="", flush=True)
            responses[("emo", emo, si)] = call_api(client, prompt)
            print(".", end="", flush=True)
            time.sleep(0.3)
        for soc, framing in SOCIAL_FRAMINGS.items():
            prompt = PROMPT.format(situation=sit, framing=framing)
            print(f"  soc:{soc}×s{si}", end="", flush=True)
            responses[("soc", soc, si)] = call_api(client, prompt)
            print(".", end="", flush=True)
            time.sleep(0.3)
    print()

    # Compute per-situation pairwise distances
    emo_pair_dists = {pair: [] for pair in combinations(emo_keys, 2)}
    soc_pair_dists = {pair: [] for pair in combinations(soc_keys, 2)}

    for si in range(len(SITUATIONS)):
        emo_texts = [responses[("emo", k, si)] for k in emo_keys]
        soc_texts = [responses[("soc", k, si)] for k in soc_keys]

        ed = pairwise_dist(emo_texts)
        sd = pairwise_dist(soc_texts)

        for (i, j), d in ed.items():
            pair = (emo_keys[i], emo_keys[j])
            emo_pair_dists[pair].append(d)
        for (i, j), d in sd.items():
            pair = (soc_keys[i], soc_keys[j])
            soc_pair_dists[pair].append(d)

    # Aggregate
    print(f"\n{'='*70}")
    print("RESULTS: Per-pair mean response distance (higher = more differentiated)")
    print(f"{'='*70}")

    print(f"\n  EMOTION pairs (should be LESS differentiated if compression matters):")
    all_emo_d = []
    for pair, dists in sorted(emo_pair_dists.items()):
        m = np.mean(dists)
        all_emo_d.extend(dists)
        print(f"    {pair[0]:>8s}—{pair[1]:<8s}: {m:.4f} ± {np.std(dists):.4f}")

    print(f"\n  SOCIAL pairs (should be MORE differentiated):")
    all_soc_d = []
    for pair, dists in sorted(soc_pair_dists.items()):
        m = np.mean(dists)
        all_soc_d.extend(dists)
        print(f"    {pair[0]:>8s}—{pair[1]:<8s}: {m:.4f} ± {np.std(dists):.4f}")

    emo_mean = np.mean(all_emo_d)
    soc_mean = np.mean(all_soc_d)

    print(f"\n  Overall emotion differentiation:  {emo_mean:.4f} ± {np.std(all_emo_d):.4f}")
    print(f"  Overall social differentiation:   {soc_mean:.4f} ± {np.std(all_soc_d):.4f}")
    print(f"  Ratio (emo/soc):                  {emo_mean/soc_mean:.3f}")

    # Paired test per situation
    emo_per_sit = [np.mean([d for dists in emo_pair_dists.values() for d in [dists[si]]])
                   for si in range(len(SITUATIONS))]
    soc_per_sit = [np.mean([d for dists in soc_pair_dists.values() for d in [dists[si]]])
                   for si in range(len(SITUATIONS))]

    stat, p = wilcoxon(soc_per_sit, emo_per_sit, alternative="greater")
    print(f"\n  Wilcoxon (social > emotion): stat={stat:.1f}, p={p:.4f}")
    print(f"  Sign: {sum(s > e for s, e in zip(soc_per_sit, emo_per_sit))}/{len(SITUATIONS)} situations")

    # Also Mann-Whitney on all distances pooled
    u, p_mw = mannwhitneyu(all_soc_d, all_emo_d, alternative="greater")
    print(f"  Mann-Whitney (pooled, social > emotion): U={u:.0f}, p={p_mw:.4f}")

    # Correlation with LLM and brain geometry
    print(f"\n{'='*70}")
    print("GEOMETRY COMPARISON")
    print(f"{'='*70}")

    brain = np.load(RSA / "brain_rdm.npz", allow_pickle=True)
    brain_conds = list(brain["conditions"])
    brain_rdm = brain["rdm"]

    # Map our 3 emotions to brain RDM indices
    all_keys = emo_keys + soc_keys
    behavioral_dists = []
    brain_dists = []
    llm_dists = []

    # Try loading Qwen headline RDM for LLM distances
    qwen_rdm = None
    rdm_path = RSA / "Qwen2.5-7B-Instruct_rdm14_headline.npz"
    if rdm_path.exists():
        z = np.load(rdm_path, allow_pickle=True)
        qwen_conds = list(z["conditions"])
        qwen_rdm = z["rdm"]

    # All 6 conditions are in brain_conds
    for pair in list(combinations(emo_keys, 2)) + list(combinations(soc_keys, 2)):
        c1, c2 = pair
        if c1 in brain_conds and c2 in brain_conds:
            bd = brain_rdm[brain_conds.index(c1), brain_conds.index(c2)]
            brain_dists.append(bd)

            if pair in emo_pair_dists:
                behavioral_dists.append(np.mean(emo_pair_dists[pair]))
            else:
                behavioral_dists.append(np.mean(soc_pair_dists[pair]))

            if qwen_rdm is not None and c1 in qwen_conds and c2 in qwen_conds:
                llm_dists.append(qwen_rdm[qwen_conds.index(c1), qwen_conds.index(c2)])

    if brain_dists and behavioral_dists:
        rho_brain, p_brain = spearmanr(brain_dists, behavioral_dists)
        print(f"  Behavioral distance vs brain distance: ρ={rho_brain:+.3f}, p={p_brain:.4f}")
        print(f"    (positive = behavior tracks brain geometry)")

    if llm_dists and behavioral_dists:
        rho_llm, p_llm = spearmanr(llm_dists, behavioral_dists)
        print(f"  Behavioral distance vs LLM distance:   ρ={rho_llm:+.3f}, p={p_llm:.4f}")
        print(f"    (positive = behavior tracks LLM geometry)")

    if brain_dists and llm_dists:
        print(f"\n  If behavioral confusion tracks LLM geometry (compressed emotions)")
        print(f"  rather than brain geometry (differentiated emotions),")
        print(f"  that demonstrates compression → behavioral confusion.")

    # Example responses
    print(f"\n{'='*70}")
    print(f"EXAMPLE: \"{SITUATIONS[0]}\"")
    print(f"{'='*70}")
    for k in emo_keys:
        print(f"\n  [EMOTION: {k.upper()}]")
        print(f"  {responses[('emo', k, 0)][:250]}")
    for k in soc_keys:
        print(f"\n  [SOCIAL: {k.upper()}]")
        print(f"  {responses[('soc', k, 0)][:250]}")

    # Save
    results = {
        "model": "deepseek-chat",
        "n_situations": len(SITUATIONS),
        "emotion_keys": emo_keys,
        "social_keys": soc_keys,
        "emotion_mean_dist": float(emo_mean),
        "social_mean_dist": float(soc_mean),
        "ratio": float(emo_mean / soc_mean),
        "wilcoxon_paired": {"stat": float(stat), "p": float(p)},
        "mann_whitney_pooled": {"U": float(u), "p": float(p_mw)},
        "emotion_pair_dists": {f"{k[0]}-{k[1]}": {"mean": float(np.mean(v)), "per_sit": v}
                               for k, v in emo_pair_dists.items()},
        "social_pair_dists": {f"{k[0]}-{k[1]}": {"mean": float(np.mean(v)), "per_sit": v}
                              for k, v in soc_pair_dists.items()},
        "geometry_correlation": {
            "behavioral_vs_brain": {"rho": float(rho_brain), "p": float(p_brain)} if brain_dists else None,
            "behavioral_vs_llm": {"rho": float(rho_llm), "p": float(p_llm)} if llm_dists else None,
        },
        "responses": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in responses.items()},
    }
    json.dump(results, open(OUT / "emotion_confusion_api_test.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT / 'emotion_confusion_api_test.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
