#!/usr/bin/env python3
"""
Emotion specificity under ambiguity: does compression cause practical problems?

When emotions are explicitly labeled, models differentiate (DeepSeek test).
When emotions are described via rich bodily cues, models differentiate even
more (implicit test). But what happens with NATURALISTIC emotional input —
where the person describes a situation and their state ambiguously?

Three conditions per scenario:
  A. ANGER-explicit: "I feel angry about this"
  B. SADNESS-explicit: "I feel sad about this"
  C. AMBIGUOUS: "I'm having a strong emotional reaction to this"

If the model's internal representations compress emotions:
  - Its "ambiguous" response should be generic (equally distant from A and B)
  - It should NOT infer which specific emotion fits the context

Compared to SOCIAL COGNITION control:
  D. BELIEF-explicit: "I believe this was deliberate"
  E. EMPATHY-explicit: "I'm worried about how this affects others"
  F. AMBIGUOUS: "I have complicated thoughts about this situation"

Social ambiguous should be MORE specific (closer to one pole) because the
model's representations differentiate social-cognitive states.

10 scenarios × 6 conditions = 60 API calls (~$0.30).
"""
from __future__ import annotations
import os, json, time
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon, mannwhitneyu

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
    "I got unexpected medical test results that changed everything",
]

EMOTION_CONDITIONS = {
    "anger": "I feel angry and furious about this.",
    "sadness": "I feel deeply sad and heartbroken about this.",
    "ambiguous": "I'm having a strong emotional reaction to this and I'm not sure how to process it.",
}

SOCIAL_CONDITIONS = {
    "belief": "I believe this was completely deliberate and calculated.",
    "empathy": "I'm mainly worried about how this is affecting the other people involved.",
    "ambiguous": "I have complicated thoughts about this whole situation and I'm not sure what to think.",
}

PROMPT = """A person is going through this situation: "{situation}"

{condition}

As their counselor, give them specific, practical advice tailored to what they're experiencing. What concrete steps should they take? 3-4 sentences."""


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


def specificity_score(resp_a, resp_b, resp_ambig):
    """How much does the ambiguous response lean toward one pole?

    Returns |sim(ambig,A) - sim(ambig,B)|.
    Higher = model is more specific (infers which condition fits).
    Lower = model gives generic response (equally distant from both).
    """
    texts = [resp_a, resp_b, resp_ambig]
    if any(t == "" for t in texts):
        return 0.0, 0.0, 0.0
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sim = cosine_similarity(tfidf)
    sim_a_ambig = sim[0, 2]
    sim_b_ambig = sim[1, 2]
    specificity = abs(sim_a_ambig - sim_b_ambig)
    return float(specificity), float(sim_a_ambig), float(sim_b_ambig)


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("=" * 70)
    print("EMOTION SPECIFICITY UNDER AMBIGUITY")
    print("=" * 70)

    responses = {}

    # Collect all responses
    for domain, conditions in [("emotion", EMOTION_CONDITIONS), ("social", SOCIAL_CONDITIONS)]:
        for cond_key, cond_text in conditions.items():
            for si, sit in enumerate(SITUATIONS):
                prompt = PROMPT.format(situation=sit, condition=cond_text)
                tag = f"{domain}_{cond_key}"
                print(f"  {tag}×s{si}", end="", flush=True)
                responses[(tag, si)] = call_api(client, prompt)
                print(".", end="", flush=True)
                time.sleep(0.3)
    print()

    # Compute specificity scores
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    emo_specs = []
    soc_specs = []

    print(f"\n  {'Situation':>4s} | {'Emo specificity':>15s} | {'Soc specificity':>15s} | {'Emo sim(A,C)':>12s} {'sim(B,C)':>10s} | {'Soc sim(D,F)':>12s} {'sim(E,F)':>10s}")
    print(f"  {'-'*95}")

    for si in range(len(SITUATIONS)):
        e_spec, e_sim_a, e_sim_b = specificity_score(
            responses[("emotion_anger", si)],
            responses[("emotion_sadness", si)],
            responses[("emotion_ambiguous", si)],
        )
        s_spec, s_sim_d, s_sim_e = specificity_score(
            responses[("social_belief", si)],
            responses[("social_empathy", si)],
            responses[("social_ambiguous", si)],
        )
        emo_specs.append(e_spec)
        soc_specs.append(s_spec)
        print(f"  s{si:>3d} | {e_spec:>15.4f} | {s_spec:>15.4f} | {e_sim_a:>12.4f} {e_sim_b:>10.4f} | {s_sim_d:>12.4f} {s_sim_e:>10.4f}")

    emo_specs = np.array(emo_specs)
    soc_specs = np.array(soc_specs)

    print(f"\n  Emotion specificity: {emo_specs.mean():.4f} ± {emo_specs.std():.4f}")
    print(f"  Social specificity:  {soc_specs.mean():.4f} ± {soc_specs.std():.4f}")

    # Paired test: is social specificity > emotion specificity?
    stat, p = wilcoxon(soc_specs, emo_specs, alternative="greater")
    print(f"\n  Wilcoxon (social > emotion): stat={stat:.1f}, p={p:.4f}")
    print(f"  Paired differences: {(soc_specs - emo_specs).mean():+.4f}")
    print(f"  Sign: {np.sum(soc_specs > emo_specs)}/{len(emo_specs)} scenarios social > emotion")

    # Ambiguous response similarity to explicit poles
    print(f"\n  Interpretation:")
    print(f"    Low specificity = ambiguous response equidistant from both poles (generic)")
    print(f"    High specificity = ambiguous response leans toward one pole (context-sensitive)")

    if p < 0.05:
        print(f"\n  FINDING: Social ambiguous responses are MORE context-specific than")
        print(f"  emotion ambiguous responses (p={p:.4f}). Consistent with representational")
        print(f"  compression: the model's compressed emotion representations produce")
        print(f"  generic responses when the emotion is not explicitly labeled.")
    else:
        print(f"\n  RESULT: No significant difference (p={p:.4f}).")

    # Example responses
    print(f"\n{'='*70}")
    print(f"EXAMPLE: Situation 0")
    print(f"\"{SITUATIONS[0]}\"")
    print(f"{'='*70}")
    for key in ["emotion_anger", "emotion_sadness", "emotion_ambiguous",
                 "social_belief", "social_empathy", "social_ambiguous"]:
        print(f"\n  [{key.upper()}]:")
        print(f"  {responses[(key, 0)][:300]}")

    # Save
    results = {
        "model": "deepseek-chat",
        "n_situations": len(SITUATIONS),
        "emotion_specificity": {
            "mean": float(emo_specs.mean()),
            "std": float(emo_specs.std()),
            "per_scenario": emo_specs.tolist(),
        },
        "social_specificity": {
            "mean": float(soc_specs.mean()),
            "std": float(soc_specs.std()),
            "per_scenario": soc_specs.tolist(),
        },
        "wilcoxon_social_gt_emotion": {"stat": float(stat), "p": float(p)},
        "responses": {f"{k[0]}_{k[1]}": v for k, v in responses.items()},
    }
    json.dump(results, open(OUT / "emotion_specificity_test.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT / 'emotion_specificity_test.json'}")
    print("DONE")


if __name__ == "__main__":
    main()
