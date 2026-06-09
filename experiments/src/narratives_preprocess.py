#!/usr/bin/env python3
"""
Narratives fMRI dataset: segment stories into sentences with timestamps,
then assign cognitive condition labels.

Two annotation modes:
  1. keyword  — fast rule-based first pass (no GPU)
  2. llm      — use Qwen2.5-7B-Instruct on HPC3 with activation proximity
                to existing RSA condition centroids

Output:
  data/narratives/sentences.jsonl          — segmented sentences with timing
  data/narratives/annotated_sentences.jsonl — with cognitive labels (keyword mode)
  data/narratives/story_profiles.json      — per-story cognitive profile

Usage:
  python narratives_preprocess.py
"""
from __future__ import annotations
import json, re, csv
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1] / "data" / "narratives"
TRANSCRIPT_DIR = BASE / "transcripts"
OUT = BASE

STORIES = [
    "pieman", "piemanpni", "tunnel", "lucy", "prettymouth",
    "milkywayoriginal", "milkywayvodka", "milkywaysynonyms",
    "slumlordreach", "notthefallintact", "merlin", "sherlock",
    "21styear", "black", "bronx", "forgot",
    "shapesphysical", "shapessocial",
    "bigbang", "friends", "himym", "santa", "seinfeld",
    "shame", "upintheair", "vinny",
]

# Stories with near-duplicate content (keep only one per group for RSA)
DUPLICATE_GROUPS = {
    "pieman_group": ["pieman", "piemanpni"],
    "milkyway_group": ["milkywayoriginal", "milkywayvodka", "milkywaysynonyms"],
}
# Unique stories for RSA (drop duplicates)
RSA_STORIES = [s for s in STORIES if s not in
    {"piemanpni", "milkywayvodka", "milkywaysynonyms"}]

# Subject counts per story (from Nastase et al. 2021)
STORY_N = {
    "pieman": 82, "piemanpni": 47, "tunnel": 23, "lucy": 16,
    "prettymouth": 40, "milkywayoriginal": 18, "milkywayvodka": 18,
    "milkywaysynonyms": 18, "slumlordreach": 18, "notthefallintact": 56,
    "merlin": 36, "sherlock": 36, "21styear": 25, "black": 47,
    "bronx": 47, "forgot": 47, "shapesphysical": 59, "shapessocial": 59,
    "bigbang": 31, "friends": 31, "himym": 31, "santa": 31,
    "seinfeld": 31, "shame": 31, "upintheair": 31, "vinny": 31,
}

CONDITIONS = [
    "anger", "fear", "disgust", "sadness", "happiness", "valence",
    "belief", "mentalizing", "intention", "theory_of_mind",
    "empathy", "self_referential", "judgment", "moral",
]

KEYWORDS = {
    "anger": r"\b(angry|anger|furious|rage|enraged|infuriat|hostile|resent|irritat|outrag|mad)\b",
    "fear": r"\b(afraid|fear|scared|frighten|terrif|dread|panic|horror|startl|alarm|nervous|anxious|worry|worri)\b",
    "disgust": r"\b(disgust|gross|revolt|repuls|nauseate|sicken|vile|awful)\b",
    "sadness": r"\b(sad|grief|sorrow|mourn|devastat|heartbreak|cry|crying|cried|tears|depress|lonely|loneli|miserable|tragic|loss|lost|miss(?:ing|ed)?)\b",
    "happiness": r"\b(happy|joy|delight|elat|bliss|excit|thrill|laugh|ecsta|wonderful|amazing|love[ds]?|loving)\b",
    "valence": r"\b(good|bad|pleasant|unpleasant|horrible|terrible|great|nice|dreadful|painful|hurt)\b",
    "belief": r"\b(believ|thought? (?:that|he|she|it|they)|assum|expect|suppos|convinc|certain)\b",
    "mentalizing": r"\b(knew|know(?:s|n)?|realiz|understand|aware|recogniz|notic|sense[ds]?|perceiv)\b",
    "intention": r"\b(intend|plan(?:ned|s)?|try|tried|want(?:ed|s)?|goal|aim|decid|chose|choos|determin)\b",
    "theory_of_mind": r"\b(wonder|imagin|perspective|point of view|she thought|he thought|she felt|he felt|what (?:she|he) (?:was thinking|meant|wanted)|mind)\b",
    "empathy": r"\b(sympathy|compassion|comfort|care[ds]? (?:about|for)|console|tender|protective|concern)\b",
    "self_referential": r"\b(my(?:self)?|I (?:was|am|felt|thought|knew|had|wanted|could|would|did)|me |own )\b",
    "judgment": r"\b(wrong|right|should|ought|fair|unfair|appropriate|inappropriate|deserv|blame|accus)\b",
    "moral": r"\b(moral|ethic|guilt|innocent|responsib|justice|cruel|kind(?:ness)?|virtue|sin|wicked|evil)\b",
}


def read_align_csv(story: str) -> list[dict]:
    path = TRANSCRIPT_DIR / "gentle" / story / "align.csv"
    if not path.exists():
        return []
    words = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                try:
                    words.append({
                        "word": parts[0],
                        "onset": float(parts[2]),
                        "offset": float(parts[3]),
                    })
                except ValueError:
                    continue
    return words


def segment_into_sentences(words: list[dict], max_gap: float = 1.5) -> list[dict]:
    """Group words into sentence-like segments using punctuation and pauses."""
    if not words:
        return []

    sentences = []
    current_words = []

    for i, w in enumerate(words):
        current_words.append(w)

        text = w["word"]
        is_end = bool(re.search(r'[.!?]$', text))

        next_gap = 0
        if i + 1 < len(words):
            next_gap = words[i+1]["onset"] - w["offset"]

        long_pause = next_gap > max_gap
        long_sentence = len(current_words) >= 40

        if (is_end or long_pause or long_sentence) and len(current_words) >= 3:
            sent_text = " ".join(cw["word"] for cw in current_words)
            sentences.append({
                "text": sent_text,
                "onset": current_words[0]["onset"],
                "offset": current_words[-1]["offset"],
                "n_words": len(current_words),
            })
            current_words = []

    if current_words and len(current_words) >= 2:
        sent_text = " ".join(cw["word"] for cw in current_words)
        sentences.append({
            "text": sent_text,
            "onset": current_words[0]["onset"],
            "offset": current_words[-1]["offset"],
            "n_words": len(current_words),
        })

    return sentences


def annotate_keywords(text: str) -> dict[str, float]:
    """Assign cognitive condition scores based on keyword matching."""
    text_lower = text.lower()
    scores = {}
    for cond, pattern in KEYWORDS.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            scores[cond] = len(matches)
    return scores


def story_profile(sentences: list[dict]) -> dict:
    """Aggregate keyword annotations across all sentences in a story."""
    total_words = sum(s["n_words"] for s in sentences)
    cond_counts = defaultdict(int)
    cond_sentences = defaultdict(int)
    for s in sentences:
        for cond, count in s.get("conditions", {}).items():
            cond_counts[cond] += count
            cond_sentences[cond] += 1

    profile = {}
    for cond in CONDITIONS:
        profile[cond] = {
            "keyword_hits": cond_counts.get(cond, 0),
            "sentences_with_hits": cond_sentences.get(cond, 0),
            "density": cond_counts.get(cond, 0) / max(total_words, 1) * 1000,
        }
    return profile


def main():
    all_sentences = []
    story_profiles = {}

    for story in STORIES:
        words = read_align_csv(story)
        if not words:
            print(f"  [skip] {story}: no align.csv")
            continue

        sentences = segment_into_sentences(words)

        for i, s in enumerate(sentences):
            s["story"] = story
            s["sent_idx"] = i
            s["conditions"] = annotate_keywords(s["text"])

        all_sentences.extend(sentences)

        profile = story_profile(sentences)
        story_profiles[story] = {
            "n_subjects": STORY_N.get(story, 0),
            "n_sentences": len(sentences),
            "n_words": sum(s["n_words"] for s in sentences),
            "duration_sec": words[-1]["offset"] - words[0]["onset"] if words else 0,
            "conditions": profile,
            "in_rsa_set": story in RSA_STORIES,
        }

        n_labeled = sum(1 for s in sentences if s["conditions"])
        print(f"  {story:<25s}  {len(sentences):>4d} sents  {len(words):>5d} words  "
              f"{n_labeled:>4d} labeled ({n_labeled/max(len(sentences),1)*100:.0f}%)")

    # Save sentences
    out_sent = OUT / "sentences.jsonl"
    with open(out_sent, "w") as f:
        for s in all_sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_sentences)} sentences to {out_sent}")

    # Save story profiles
    out_prof = OUT / "story_profiles.json"
    with open(out_prof, "w") as f:
        json.dump(story_profiles, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_prof}")

    # Summary: condition coverage across all stories
    print(f"\n{'Condition':<20s} {'Total hits':>10s} {'Sentences':>10s} {'Stories':>8s}")
    for cond in CONDITIONS:
        total = sum(sp["conditions"][cond]["keyword_hits"]
                    for sp in story_profiles.values())
        sents = sum(sp["conditions"][cond]["sentences_with_hits"]
                    for sp in story_profiles.values())
        stories_with = sum(1 for sp in story_profiles.values()
                          if sp["conditions"][cond]["keyword_hits"] > 0)
        print(f"  {cond:<20s} {total:>10d} {sents:>10d} {stories_with:>8d}")

    # RSA-relevant stories summary
    print(f"\n=== RSA story set ({len(RSA_STORIES)} unique stories) ===")
    for story in RSA_STORIES:
        sp = story_profiles.get(story, {})
        n = sp.get("n_subjects", 0)
        dur = sp.get("duration_sec", 0)
        top_conds = sorted(sp.get("conditions", {}).items(),
                          key=lambda x: -x[1]["keyword_hits"])[:3]
        top_str = ", ".join(f"{c}({v['keyword_hits']})" for c, v in top_conds if v["keyword_hits"] > 0)
        print(f"  {story:<25s}  N={n:>3d}  {dur/60:>5.1f}min  top: {top_str}")


if __name__ == "__main__":
    main()
