"""
Aggregate the 4 cognitive domain stimuli (emotion, moral, tom, self) into a unified
{category, text} JSONL file compatible with multi_function_dissociation.py.

For the pilot, we sample 50 items per domain (balanced) and tag each with its domain
as the 'category' field. This lets us reuse the existing pipeline as-is.

A separate 'neutral_control' category is also built from low-arousal mid-valence
sentences and from physical/literal-interpretation control items, providing a
baseline that should engage none of the four cognitive modules.

Output: experiments/data/cognitive_stimuli/stimuli_cognitive_pilot.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STIM_DIR = ROOT / "data" / "cognitive_stimuli"
OUT_FILE = STIM_DIR / "stimuli_cognitive_pilot.jsonl"

N_PER_DOMAIN = 50
SEED = 20260525


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_emotion(rng: random.Random):
    """Sample 50 emotionally-loaded sentences from the Ekman localizer set.

    The localizer file has 50 items per emotion across 6 Ekman categories plus
    50 neutrals. We want emotionally-engaging content for the 'emotion' domain.
    """
    items = load_jsonl(STIM_DIR / "emotion" / "emotion_localizer_stimuli.jsonl")
    emotional = [x for x in items if x.get("emotion") != "neutral"]
    sampled = rng.sample(emotional, N_PER_DOMAIN)
    out = []
    for x in sampled:
        out.append({
            "category": "emotion",
            "text": x["text"],
            "source_id": x.get("id", ""),
            "source_label": x.get("emotion", ""),
        })
    return out


def sample_moral(rng: random.Random):
    """Sample 50 moral foundation vignettes (canonical Clifford 2015 MFV)."""
    items = load_jsonl(STIM_DIR / "moral" / "moral_foundations_vignettes.jsonl")
    # Stratify across foundations so we don't oversample one
    by_foundation = {}
    for x in items:
        f = x.get("foundation", "unknown")
        by_foundation.setdefault(f, []).append(x)
    # Round-robin pick from each foundation until we have N_PER_DOMAIN
    out = []
    foundations = sorted(by_foundation.keys())
    while len(out) < N_PER_DOMAIN:
        for f in foundations:
            pool = by_foundation[f]
            if not pool:
                continue
            pick = pool.pop(rng.randrange(len(pool)))
            out.append({
                "category": "moral",
                "text": pick["vignette_text"],
                "source_id": pick.get("id", ""),
                "source_label": pick.get("foundation", ""),
            })
            if len(out) >= N_PER_DOMAIN:
                break
    return out


def sample_tom(rng: random.Random):
    """Sample 50 theory-of-mind items, mixed across paradigms."""
    pool = []
    # false-belief stories (use false-belief condition only — true-belief is control)
    fb = [x for x in load_jsonl(STIM_DIR / "tom" / "false_belief.jsonl")
          if x.get("condition") == "false_belief"]
    for x in fb:
        pool.append({"text": f"{x['story']} {x['question']}",
                     "source_id": x.get("id", ""), "source_label": "false_belief"})
    # faux pas with present
    fp = [x for x in load_jsonl(STIM_DIR / "tom" / "faux_pas.jsonl")
          if x.get("has_faux_pas") is True]
    for x in fp:
        pool.append({"text": x["story"],
                     "source_id": x.get("id", ""), "source_label": "faux_pas"})
    # strange stories
    ss = load_jsonl(STIM_DIR / "tom" / "strange_stories.jsonl")
    for x in ss:
        pool.append({"text": f"{x['story']} {x.get('question', '')}".strip(),
                     "source_id": x.get("id", ""),
                     "source_label": x.get("mental_state_category", "strange")})
    rng.shuffle(pool)
    out = []
    for x in pool[:N_PER_DOMAIN]:
        out.append({
            "category": "tom",
            "text": x["text"],
            "source_id": x["source_id"],
            "source_label": x["source_label"],
        })
    return out


def sample_self(rng: random.Random):
    """Sample 50 self-referential items from the self_other triplet file.

    Use 'self' perspective items only — the 'other' and 'reality' triplets
    become candidates for control or future analyses.
    """
    items = [x for x in load_jsonl(STIM_DIR / "tom" / "self_other.jsonl")
             if x.get("perspective") == "self"]
    if len(items) < N_PER_DOMAIN:
        # if pool too small, fall back to other-perspective items too
        items += [x for x in load_jsonl(STIM_DIR / "tom" / "self_other.jsonl")
                  if x.get("perspective") == "other"]
    sampled = rng.sample(items, min(N_PER_DOMAIN, len(items)))
    out = []
    for x in sampled:
        text = f"{x['scenario']} {x['question']}".strip()
        out.append({
            "category": "self",
            "text": text,
            "source_id": x.get("id", ""),
            "source_label": x.get("perspective", ""),
        })
    return out


def sample_neutral(rng: random.Random):
    """Sample 50 neutral control sentences (low arousal, no moral/social content)."""
    items = load_jsonl(STIM_DIR / "emotion" / "emotion_localizer_stimuli.jsonl")
    neutrals = [x for x in items if x.get("emotion") == "neutral"]
    sampled = rng.sample(neutrals, N_PER_DOMAIN)
    out = []
    for x in sampled:
        out.append({
            "category": "neutral_control",
            "text": x["text"],
            "source_id": x.get("id", ""),
            "source_label": "neutral",
        })
    return out


def main():
    rng = random.Random(SEED)
    all_items = []
    all_items += sample_emotion(rng)
    all_items += sample_moral(rng)
    all_items += sample_tom(rng)
    all_items += sample_self(rng)
    all_items += sample_neutral(rng)

    # Assign global indices, deduplicate any accidental exact text matches
    seen = set()
    deduped = []
    for i, x in enumerate(all_items):
        key = x["text"].strip()
        if key in seen:
            continue
        seen.add(key)
        x["idx"] = i
        deduped.append(x)

    with open(OUT_FILE, "w") as f:
        for x in deduped:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # Report
    from collections import Counter
    counts = Counter(x["category"] for x in deduped)
    print(f"Wrote {len(deduped)} stimuli to {OUT_FILE}")
    for cat in ["emotion", "moral", "tom", "self", "neutral_control"]:
        print(f"  {cat:>18s}: {counts[cat]}")


if __name__ == "__main__":
    main()
