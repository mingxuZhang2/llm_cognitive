#!/usr/bin/env python3
"""
Implicit emotion test: does DeepSeek differentiate emotions WITHOUT labels?

Two conditions:
  A. LABELED:   "They feel very angry about it."
  B. IMPLICIT:  Describe the emotion through bodily/behavioral cues only,
                 NO emotion word.

If differentiation drops in condition B → the model relies on label→template
mapping, not genuine emotion understanding.

10 situations × 5 emotions × 2 conditions = 100 API calls (~$1).
"""
from __future__ import annotations
import os, json, time
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu, wilcoxon

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

SITUATIONS = [
    "My colleague took credit for my work in front of the whole team",
    "I found out my partner has been keeping a major secret from me",
    "I was passed over for a promotion I worked toward for two years",
    "My best friend shared my most private confession with others",
    "I discovered my trusted mentor had been manipulating me",
    "Someone broke into my apartment while I was at work",
    "My child came home saying older kids have been targeting them",
    "I found evidence of serious misconduct at my organization",
    "A family member gambled away money they borrowed from me",
    "I got unexpected medical test results",
]

LABELED = {
    "anger":     "They feel very angry about it.",
    "fear":      "They feel very afraid and anxious about it.",
    "disgust":   "They feel deeply disgusted and revolted by it.",
    "sadness":   "They feel deeply sad and heartbroken about it.",
    "happiness": "They feel surprisingly relieved and hopeful about how it might turn out.",
}

IMPLICIT = {
    "anger": (
        "Their jaw is clenched tight. They keep replaying what happened, "
        "fists balling up. There's a hot pressure building in their chest "
        "and they feel an overwhelming urge to confront someone about this."
    ),
    "fear": (
        "Their hands won't stop trembling. Their heart is racing and they keep "
        "glancing around nervously. They feel a knot in their stomach and an "
        "overwhelming urge to get somewhere safe and lock the door."
    ),
    "disgust": (
        "Their stomach is turning and they feel slightly nauseous. They keep "
        "wanting to push away from the situation, like something contaminated "
        "has touched them. Their lip curls when they think about it."
    ),
    "sadness": (
        "Their chest feels hollow and heavy. They can barely find the energy "
        "to move. Their eyes keep filling up and there's a lump in their throat "
        "that won't go away. Everything feels muted and far away."
    ),
    "happiness": (
        "They feel a weight lifting off their shoulders. There's a warm lightness "
        "in their chest and they catch themselves almost smiling. Their breathing "
        "has slowed down and they feel a quiet energy, like things might actually work out."
    ),
}

PROMPT_TEMPLATE = """A person is going through this situation: "{situation}"

{emotion_description}

As their counselor, give them specific, practical advice. What concrete steps should they take? 3-4 sentences."""


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


def compute_pairwise_distances(responses, keys, n_sit):
    """Mean pairwise TF-IDF cosine distance across situations."""
    n = len(keys)
    rdm = np.zeros((n, n))
    count = 0
    for si in range(n_sit):
        texts = [responses.get((k, si), "") for k in keys]
        if any(t == "" for t in texts):
            continue
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
        sim = cosine_similarity(tfidf)
        rdm += (1 - sim)
        count += 1
    return rdm / max(count, 1)


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("=" * 70)
    print("IMPLICIT EMOTION TEST: Label vs No-Label")
    print("=" * 70)

    emo_keys = list(LABELED.keys())
    labeled_resp = {}
    implicit_resp = {}

    # ── Condition A: Labeled ──
    print(f"\n--- Condition A: LABELED (5 × 10 = 50 calls) ---")
    for emo in emo_keys:
        for si, sit in enumerate(SITUATIONS):
            prompt = PROMPT_TEMPLATE.format(situation=sit, emotion_description=LABELED[emo])
            print(f"  L:{emo}×s{si}", end="", flush=True)
            labeled_resp[(emo, si)] = call_api(client, prompt)
            print(f".", end="", flush=True)
            time.sleep(0.3)
    print()

    # ── Condition B: Implicit ──
    print(f"\n--- Condition B: IMPLICIT (5 × 10 = 50 calls) ---")
    for emo in emo_keys:
        for si, sit in enumerate(SITUATIONS):
            prompt = PROMPT_TEMPLATE.format(situation=sit, emotion_description=IMPLICIT[emo])
            print(f"  I:{emo}×s{si}", end="", flush=True)
            implicit_resp[(emo, si)] = call_api(client, prompt)
            print(f".", end="", flush=True)
            time.sleep(0.3)
    print()

    # ── Compute RDMs ──
    n_sit = len(SITUATIONS)
    labeled_rdm = compute_pairwise_distances(labeled_resp, emo_keys, n_sit)
    implicit_rdm = compute_pairwise_distances(implicit_resp, emo_keys, n_sit)

    # ── Extract upper triangles ──
    pairs = list(combinations(range(len(emo_keys)), 2))
    labeled_dists = [labeled_rdm[i, j] for i, j in pairs]
    implicit_dists = [implicit_rdm[i, j] for i, j in pairs]

    # ── Results ──
    print(f"\n{'='*70}")
    print("LABELED RDM (with emotion words)")
    print(f"{'='*70}")
    for i, ei in enumerate(emo_keys):
        print(f"  {ei:>12}", end="")
        for j in range(len(emo_keys)):
            print(f" {labeled_rdm[i,j]:7.4f}", end="")
        print()

    print(f"\n{'='*70}")
    print("IMPLICIT RDM (bodily/behavioral cues only, no emotion words)")
    print(f"{'='*70}")
    for i, ei in enumerate(emo_keys):
        print(f"  {ei:>12}", end="")
        for j in range(len(emo_keys)):
            print(f" {implicit_rdm[i,j]:7.4f}", end="")
        print()

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  LABELED mean distance:   {np.mean(labeled_dists):.4f} ± {np.std(labeled_dists):.4f}")
    print(f"  IMPLICIT mean distance:  {np.mean(implicit_dists):.4f} ± {np.std(implicit_dists):.4f}")

    # Does differentiation INCREASE or DECREASE without labels?
    diff = np.array(implicit_dists) - np.array(labeled_dists)
    print(f"\n  Δ(implicit - labeled): {np.mean(diff):+.4f}")
    print(f"    If positive: MORE differentiated without labels")
    print(f"    If negative: LESS differentiated without labels (label-dependent)")

    # Paired test
    if len(labeled_dists) == len(implicit_dists):
        stat, p = wilcoxon(implicit_dists, labeled_dists)
        print(f"  Wilcoxon signed-rank: stat={stat:.1f}, p={p:.4f}")

    # Per-pair comparison
    print(f"\n  {'Pair':>25s} | {'Labeled':>8s} | {'Implicit':>8s} | {'Δ':>8s}")
    print(f"  {'-'*60}")
    for idx, (i, j) in enumerate(pairs):
        ld = labeled_dists[idx]
        id_ = implicit_dists[idx]
        d = id_ - ld
        arrow = "↑" if d > 0.01 else "↓" if d < -0.01 else "="
        print(f"  {emo_keys[i]+' — '+emo_keys[j]:>25s} | {ld:8.4f} | {id_:8.4f} | {d:+8.4f} {arrow}")

    # Correlation between labeled and implicit RDMs
    rho, p_rho = mannwhitneyu(labeled_dists, implicit_dists, alternative="two-sided")
    from scipy.stats import spearmanr
    rdm_rho, rdm_p = spearmanr(labeled_dists, implicit_dists)
    print(f"\n  RDM correlation (labeled vs implicit): ρ={rdm_rho:+.3f}, p={rdm_p:.4f}")
    print(f"    If high: same emotion structure preserved without labels")
    print(f"    If low: structure changes → label-dependent")

    # ── Example responses for one situation ──
    print(f"\n{'='*70}")
    print(f"EXAMPLE: Situation 0")
    print(f"\"{SITUATIONS[0]}\"")
    print(f"{'='*70}")
    for emo in emo_keys:
        print(f"\n  LABELED [{emo}]: {labeled_resp[(emo, 0)][:150]}...")
        print(f"  IMPLICIT [{emo}]: {implicit_resp[(emo, 0)][:150]}...")

    # ── Save ──
    results = {
        "model": "deepseek-chat",
        "n_situations": n_sit,
        "emotion_keys": emo_keys,
        "labeled_rdm": labeled_rdm.tolist(),
        "implicit_rdm": implicit_rdm.tolist(),
        "labeled_mean_dist": float(np.mean(labeled_dists)),
        "implicit_mean_dist": float(np.mean(implicit_dists)),
        "delta_mean": float(np.mean(diff)),
        "rdm_correlation": {"rho": float(rdm_rho), "p": float(rdm_p)},
        "labeled_responses": {f"{e}_{si}": labeled_resp[(e, si)]
                              for e in emo_keys for si in range(n_sit)},
        "implicit_responses": {f"{e}_{si}": implicit_resp[(e, si)]
                               for e in emo_keys for si in range(n_sit)},
    }
    json.dump(results, open(OUT / "implicit_emotion_api_test.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT / 'implicit_emotion_api_test.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
