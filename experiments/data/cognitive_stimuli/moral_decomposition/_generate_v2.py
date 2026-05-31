"""
Generate v2 paired moral decomposition stimuli — extends to 200 pairs per condition.

This script produces 150 NEW pairs per condition (IDs 051-200), to be combined
with the existing 50 pairs (IDs 001-050) in decomposition_stimuli.jsonl.

Total new = 4 conditions x 150 pairs x 2 items = 1200 items.

Design constraints (verified post-generation):
- Within-pair word count delta <= 5
- Targets calibrated against Cushman/Turiel/Greene/Nichols literature
- No exact text overlap with existing 50 pairs
- >=15 distinct domains per condition
- 2-3 sentence scenarios
- Conventional violations have NO buried harm (no fire-drill confounds, etc.)

Seed: 20260525 (deterministic ordering only; no random text generation).
"""

import json
import random
from pathlib import Path

SEED = 20260525
random.seed(SEED)

OUT_DIR = Path(__file__).parent
OUT_PATH = OUT_DIR / "decomposition_stimuli_v2.jsonl"
EXISTING_PATH = OUT_DIR / "decomposition_stimuli.jsonl"


# =========================================================================
# Imports of condition pair lists (split across modules for readability)
# =========================================================================
from _v2_intent import INTENT_PAIRS_V2
from _v2_outcome import OUTCOME_PAIRS_V2
from _v2_norm import NORM_PAIRS_V2
from _v2_morality import VALENCE_PAIRS_V2


def word_count(s):
    return len(s.split())


def build_entry(condition, subcondition, pair_idx, ab, text, target, note):
    pair_id = f"{condition}_{pair_idx:03d}"
    item_id = f"{pair_id}{ab}"
    return {
        "id": item_id,
        "condition": condition,
        "subcondition": subcondition,
        "pair_id": pair_id,
        "text": text,
        "target_judgment": target,
        "notes": note,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing texts to verify no duplicates
    existing_texts = set()
    if EXISTING_PATH.exists():
        with EXISTING_PATH.open() as f:
            for line in f:
                rec = json.loads(line)
                existing_texts.add(rec["text"])
    print(f"Loaded {len(existing_texts)} existing texts for duplicate checking")

    entries = []

    # Intent (new pairs index 51-200)
    assert len(INTENT_PAIRS_V2) == 150, f"Expected 150 intent pairs, got {len(INTENT_PAIRS_V2)}"
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(INTENT_PAIRS_V2, 51):
        entries.append(build_entry("intent", "intentional", i, "a", txt_a, t_a, f"intentional harm; {note}"))
        entries.append(build_entry("intent", "accidental",   i, "b", txt_b, t_b, f"accidental harm, same outcome; {note}"))

    # Outcome
    assert len(OUTCOME_PAIRS_V2) == 150, f"Expected 150 outcome pairs, got {len(OUTCOME_PAIRS_V2)}"
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(OUTCOME_PAIRS_V2, 51):
        entries.append(build_entry("outcome", "completed", i, "a", txt_a, t_a, f"completed harm; {note}"))
        entries.append(build_entry("outcome", "attempted", i, "b", txt_b, t_b, f"attempted harm, same intent, no harm; {note}"))

    # Norm type
    assert len(NORM_PAIRS_V2) == 150, f"Expected 150 norm_type pairs, got {len(NORM_PAIRS_V2)}"
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(NORM_PAIRS_V2, 51):
        entries.append(build_entry("norm_type", "moral_violation",         i, "a", txt_a, t_a, f"moral violation (harm/fairness/rights); {note}"))
        entries.append(build_entry("norm_type", "conventional_violation",  i, "b", txt_b, t_b, f"conventional violation (etiquette/custom); {note}"))

    # Morality vs valence
    assert len(VALENCE_PAIRS_V2) == 150, f"Expected 150 morality pairs, got {len(VALENCE_PAIRS_V2)}"
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(VALENCE_PAIRS_V2, 51):
        entries.append(build_entry("morality", "moral_negative",    i, "a", txt_a, t_a, f"negative outcome with moral agent; {note}"))
        entries.append(build_entry("morality", "nonmoral_negative", i, "b", txt_b, t_b, f"equally negative outcome, natural cause; {note}"))

    # =====================================================================
    # Validation
    # =====================================================================
    print(f"\nTotal new entries: {len(entries)} (expected 1200)")
    assert len(entries) == 1200

    by_cond = {}
    for e in entries:
        by_cond.setdefault(e["condition"], []).append(e)
    print("\nPer-condition stats:")
    for c, lst in by_cond.items():
        wcs = [word_count(e["text"]) for e in lst]
        print(f"  {c}: {len(lst)} items | wc min={min(wcs)} med={sorted(wcs)[len(wcs)//2]} max={max(wcs)} mean={sum(wcs)/len(wcs):.2f}")

    # Within-pair word count delta
    print("\nLength-mismatch report (pairs with > 5 word difference):")
    mismatch_count = 0
    pairs = {}
    for e in entries:
        pairs.setdefault(e["pair_id"], []).append(e)
    pair_deltas = []
    for pid, items in pairs.items():
        if len(items) != 2:
            print(f"  WARNING: pair {pid} has {len(items)} items")
            continue
        wa, wb = word_count(items[0]["text"]), word_count(items[1]["text"])
        delta = abs(wa - wb)
        pair_deltas.append(delta)
        if delta > 5:
            print(f"  {pid}: {wa} vs {wb} words (diff={delta})")
            print(f"    a: {items[0]['text']}")
            print(f"    b: {items[1]['text']}")
            mismatch_count += 1
    print(f"  Total mismatches > 5 words: {mismatch_count}")
    print(f"  Mean pair delta: {sum(pair_deltas)/len(pair_deltas):.2f}")
    print(f"  Max pair delta: {max(pair_deltas)}")

    # Check intra-file duplicates
    seen_texts = set()
    dups = 0
    for e in entries:
        if e["text"] in seen_texts:
            print(f"  INTRA-FILE DUP: {e['id']}")
            dups += 1
        seen_texts.add(e["text"])
    print(f"\nIntra-file duplicate texts: {dups}")

    # Check overlap with existing file
    overlap_count = 0
    for e in entries:
        if e["text"] in existing_texts:
            print(f"  OVERLAP WITH v1: {e['id']}")
            overlap_count += 1
    print(f"Texts overlapping with existing v1 file: {overlap_count}")

    # ID uniqueness
    ids = [e["id"] for e in entries]
    print(f"Unique IDs: {len(set(ids))} / {len(ids)}")
    pair_ids = [e["pair_id"] for e in entries]
    print(f"Unique pair_ids: {len(set(pair_ids))} / 600 expected (each x2)")
    assert len(set(ids)) == 1200
    assert len(set(pair_ids)) == 600

    # Domain coverage per condition (from notes)
    print("\nDomain coverage (distinct tags per condition):")
    for c, lst in by_cond.items():
        domains = set()
        for e in lst:
            # parse domain from note (last "; X" segment)
            note = e["notes"]
            if ";" in note:
                dom = note.rsplit(";", 1)[-1].strip()
                domains.add(dom)
        print(f"  {c}: {len(domains)} distinct domain tags")

    # Target judgment summary
    print("\nTarget judgment summary by subcondition:")
    by_sub = {}
    for e in entries:
        key = (e["condition"], e["subcondition"])
        by_sub.setdefault(key, []).append(e["target_judgment"])
    for key, vals in sorted(by_sub.items()):
        print(f"  {key}: n={len(vals)} min={min(vals):.1f} mean={sum(vals)/len(vals):.2f} max={max(vals):.1f}")

    # Calibration sanity (a-b within each pair)
    print("\nWithin-pair target diff (a - b) summary:")
    for cond, lst in by_cond.items():
        diffs = []
        # group by pair_id
        pbp = {}
        for e in lst:
            pbp.setdefault(e["pair_id"], {})[e["id"][-1]] = e["target_judgment"]
        for pid, ab in pbp.items():
            if "a" in ab and "b" in ab:
                diffs.append(ab["a"] - ab["b"])
        print(f"  {cond}: min={min(diffs):+.2f} mean={sum(diffs)/len(diffs):+.2f} max={max(diffs):+.2f}")

    # Write JSONL
    with OUT_PATH.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
