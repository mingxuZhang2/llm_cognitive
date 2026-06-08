#!/usr/bin/env python3
"""
Test emotion compression in commercial models via DeepSeek API.

Same situation × different emotions → how much does the response change?
Same situation × different cognitive framings → how much does the response change?

If commercial models also compress within-affective:
  within-emotion response distance ≈ 0 (all emotions get same advice)
  within-social response distance >> 0 (different framings get different advice)

90 API calls total (~$0.50).
"""
from __future__ import annotations
import os, json, time
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu, spearmanr

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "mechanistic"
OUT.mkdir(exist_ok=True)

SITUATIONS = [
    "My colleague took credit for my work in front of the whole team",
    "I found out my partner has been keeping a major secret from me for months",
    "I was passed over for a promotion I had been working toward for two years",
    "My best friend shared my most private confession with mutual friends",
    "I discovered that a trusted mentor had been manipulating me for their benefit",
    "Someone broke into my apartment while I was at work",
    "My child came home from school saying older kids have been targeting them",
    "I found evidence of serious misconduct in the organization I work for",
    "A family member gambled away money they had borrowed from me",
    "I learned that my medical test results came back with something unexpected",
]

EMOTIONS = {
    "anger": "very angry",
    "fear": "very afraid and anxious",
    "disgust": "deeply disgusted and revolted",
    "sadness": "deeply sad and heartbroken",
    "happiness": "surprisingly relieved and hopeful about how this might turn out",
}

SOCIAL_FRAMINGS = {
    "belief_deliberate": "They believe this was completely deliberate and personal",
    "intention_confront": "They have decided to confront the person responsible face-to-face",
    "judgment_moral": "They see this as a serious moral violation that crosses a clear line",
    "empathy_others": "They are mainly worried about how this affects the other people involved",
}

EMOTION_PROMPT = """A person is going through this situation: "{situation}"

They feel {emotion} about it.

As their counselor, give them specific, practical advice. What concrete steps should they take? 3-4 sentences."""

SOCIAL_PROMPT = """A person is going through this situation: "{situation}"

{framing}.

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
                print(f" [retry {attempt+1}: {e}]", end="")
                time.sleep(2 ** attempt)
            else:
                print(f" [FAILED: {e}]")
                return ""


def compute_rdm(responses_dict, keys, n_situations):
    """Build RDM from response texts. responses_dict[(key, sit_idx)] = text."""
    n = len(keys)
    rdm = np.zeros((n, n))

    for si in range(n_situations):
        texts = [responses_dict.get((k, si), "") for k in keys]
        if any(t == "" for t in texts):
            continue
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
        sim = cosine_similarity(tfidf)
        rdm += (1 - sim)

    rdm /= n_situations
    return rdm


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("=" * 70)
    print("COMMERCIAL MODEL EMOTION COMPRESSION TEST (DeepSeek)")
    print("=" * 70)

    emo_responses = {}
    soc_responses = {}

    # ── Collect emotion responses ──
    print(f"\n--- Emotion responses (5 × 10 = 50 calls) ---")
    for emo_key, emo_phrase in EMOTIONS.items():
        for si, situation in enumerate(SITUATIONS):
            prompt = EMOTION_PROMPT.format(situation=situation, emotion=emo_phrase)
            print(f"  {emo_key}×sit{si}...", end="", flush=True)
            resp = call_api(client, prompt)
            emo_responses[(emo_key, si)] = resp
            print(f" {len(resp)}ch", flush=True)
            time.sleep(0.3)

    # ── Collect social framing responses ──
    print(f"\n--- Social framing responses (4 × 10 = 40 calls) ---")
    for soc_key, framing in SOCIAL_FRAMINGS.items():
        for si, situation in enumerate(SITUATIONS):
            prompt = SOCIAL_PROMPT.format(situation=situation, framing=framing)
            print(f"  {soc_key}×sit{si}...", end="", flush=True)
            resp = call_api(client, prompt)
            soc_responses[(soc_key, si)] = resp
            print(f" {len(resp)}ch", flush=True)
            time.sleep(0.3)

    # ── Compute RDMs ──
    emo_keys = list(EMOTIONS.keys())
    soc_keys = list(SOCIAL_FRAMINGS.keys())

    emo_rdm = compute_rdm(emo_responses, emo_keys, len(SITUATIONS))
    soc_rdm = compute_rdm(soc_responses, soc_keys, len(SITUATIONS))

    # ── Print RDMs ──
    print(f"\n{'='*70}")
    print("EMOTION BEHAVIORAL RDM (response dissimilarity, 0=identical)")
    print(f"{'='*70}")
    print(f"  {'':>12}", end="")
    for e in emo_keys:
        print(f" {e:>9}", end="")
    print()
    for i, ei in enumerate(emo_keys):
        print(f"  {ei:>12}", end="")
        for j in range(len(emo_keys)):
            print(f" {emo_rdm[i,j]:9.4f}", end="")
        print()

    print(f"\n{'='*70}")
    print("SOCIAL FRAMING BEHAVIORAL RDM")
    print(f"{'='*70}")
    print(f"  {'':>20}", end="")
    for s in soc_keys:
        print(f" {s[:9]:>9}", end="")
    print()
    for i, si_name in enumerate(soc_keys):
        print(f"  {si_name:>20}", end="")
        for j in range(len(soc_keys)):
            print(f" {soc_rdm[i,j]:9.4f}", end="")
        print()

    # ── Compare ──
    emo_dists = [emo_rdm[i, j] for i, j in combinations(range(len(emo_keys)), 2)]
    soc_dists = [soc_rdm[i, j] for i, j in combinations(range(len(soc_keys)), 2)]

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"  Within-emotion response distance:  mean={np.mean(emo_dists):.4f} ± {np.std(emo_dists):.4f} (n={len(emo_dists)} pairs)")
    print(f"  Within-social response distance:   mean={np.mean(soc_dists):.4f} ± {np.std(soc_dists):.4f} (n={len(soc_dists)} pairs)")

    if emo_dists and soc_dists:
        u, p = mannwhitneyu(emo_dists, soc_dists, alternative="less")
        print(f"  Mann-Whitney (emo < soc): U={u:.0f}, p={p:.4f}")
        ratio = np.mean(emo_dists) / np.mean(soc_dists) if np.mean(soc_dists) > 0 else 0
        print(f"  Ratio (emo/soc): {ratio:.3f}")

    # ── Per-pair details ──
    print(f"\n  Emotion pair details:")
    for i, j in combinations(range(len(emo_keys)), 2):
        print(f"    {emo_keys[i]:>10}—{emo_keys[j]:<10}: {emo_rdm[i,j]:.4f}")

    print(f"\n  Social pair details:")
    for i, j in combinations(range(len(soc_keys)), 2):
        print(f"    {soc_keys[i]:>20}—{soc_keys[j]:<20}: {soc_rdm[i,j]:.4f}")

    # ── Example responses (show one situation) ──
    print(f"\n{'='*70}")
    print(f"EXAMPLE: Situation 0 responses")
    print(f"Situation: \"{SITUATIONS[0]}\"")
    print(f"{'='*70}")
    for emo_key in emo_keys:
        print(f"\n  [{emo_key.upper()}]: {emo_responses[(emo_key, 0)][:200]}...")
    print()
    for soc_key in soc_keys:
        print(f"\n  [{soc_key.upper()}]: {soc_responses[(soc_key, 0)][:200]}...")

    # ── Save ──
    results = {
        "model": "deepseek-chat",
        "n_situations": len(SITUATIONS),
        "emotion_keys": emo_keys,
        "social_keys": soc_keys,
        "emotion_rdm": emo_rdm.tolist(),
        "social_rdm": soc_rdm.tolist(),
        "emotion_mean_dist": float(np.mean(emo_dists)),
        "social_mean_dist": float(np.mean(soc_dists)),
        "mann_whitney_p": float(p) if emo_dists and soc_dists else None,
        "responses_emotion": {f"{e}_{si}": emo_responses[(e, si)]
                              for e in emo_keys for si in range(len(SITUATIONS))},
        "responses_social": {f"{s}_{si}": soc_responses[(s, si)]
                             for s in soc_keys for si in range(len(SITUATIONS))},
    }
    json.dump(results, open(OUT / "commercial_emotion_test.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT / 'commercial_emotion_test.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
