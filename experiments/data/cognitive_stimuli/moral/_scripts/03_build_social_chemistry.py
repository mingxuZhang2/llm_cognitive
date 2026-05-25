"""Selectively sample 1000 high-quality Social Chemistry 101 RoTs.

Source: https://maxwellforbes.com/social-chemistry/ — distributed under
CC BY 4.0. We use the canonical TSV
    social-chem-101/social-chem-101.v1.0.tsv
from the publisher's Google Cloud zip.

Citation:
    Forbes, M., Hwang, J. D., Shwartz, V., Sap, M., & Choi, Y. (2020).
    Social Chemistry 101: Learning to Reason about Social and Moral Norms.
    EMNLP 2020. arXiv:2011.00620.

Quality filter (an entry must satisfy ALL):
  - split == "train"                  (the bulk, clean-annotated subset)
  - rot-bad == 0                      (annotator did not mark as bad)
  - rot-agree >= 3                    (>= 75% to >99% agreement bucket)
  - rot-categorization contains "morality-ethics" OR
        rot-moral-foundations is non-empty
  - rot-judgment is non-empty and starts with "it's "
        (sanity for a moral-judgment phrasing)
  - action and rot are non-empty

We then sample 1000 rows balanced over the five MFT foundations: care-harm,
fairness-cheating, loyalty-betrayal, authority-subversion, sanctity-degradation.
Each RoT may have multiple foundations; we duplicate it across foundations only
when balancing requires it (but we deduplicate the *output* IDs).

Output schema (one row per line):
    {
      "id":              "sc101_<rotid_hash>",
      "action":          str,    actor action (short verb-phrase)
      "rule_of_thumb":   str,    full RoT sentence
      "situation":       str,    original reddit/rocstories prompt
      "moral_judgment":  int,    action-moral-judgment in [-2, +2]
                                 (-2=very bad ... +2=very good)
      "moral_foundations": list[str],  e.g. ["care-harm","fairness-cheating"]
      "agreement_level": int,    rot-agree in {0,1,2,3,4}; we keep >=3
      "area":            str,    confessions / dearabby /
                                  rocstories / amitheasshole
      "source":          str
    }
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SRC_TSV = Path("/tmp/social_chem/social-chem-101/social-chem-101.v1.0.tsv")
OUT_PATH = Path(
    "/hpc2hdd/home/mzhang630/data/nature/experiments/data/"
    "cognitive_stimuli/moral/social_chemistry_sample.jsonl"
)
TARGET_TOTAL = 1000
SEED = 20250525

CITATION = (
    "Forbes, M., Hwang, J. D., Shwartz, V., Sap, M., & Choi, Y. (2020). "
    "Social Chemistry 101: Learning to Reason about Social and Moral Norms. "
    "EMNLP 2020. arXiv:2011.00620. License: CC BY 4.0."
)

MFT_FOUNDATIONS = [
    "care-harm",
    "fairness-cheating",
    "loyalty-betrayal",
    "authority-subversion",
    "sanctity-degradation",
]


def _parse_int(x: str) -> int | None:
    x = x.strip()
    if x == "" or x == "":
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _passes_quality(row: dict) -> bool:
    if row.get("split") != "train":
        return False
    if row.get("rot-bad") != "0":
        return False
    agree = _parse_int(row.get("rot-agree", ""))
    if agree is None or agree < 3:
        return False
    cats = row.get("rot-categorization", "")
    found = row.get("rot-moral-foundations", "").strip()
    if "morality-ethics" not in cats and not found:
        return False
    if not row.get("rot", "").strip() or not row.get("action", "").strip():
        return False
    judg = row.get("rot-judgment", "").lower().strip()
    if not judg.startswith("it's ") and not judg.startswith("it is "):
        return False
    return True


def main() -> None:
    rng = random.Random(SEED)

    bucket: dict[str, list[dict]] = defaultdict(list)
    unfounded: list[dict] = []  # passes quality but has no foundation tag
    with SRC_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if not _passes_quality(row):
                continue
            mj = _parse_int(row.get("action-moral-judgment", ""))
            if mj is None:
                continue
            agree = _parse_int(row["rot-agree"])
            rec = {
                "raw_rot_id": row["rot-id"],
                "action": row["action"].strip(),
                "rule_of_thumb": row["rot"].strip(),
                "situation": row.get("situation", "").strip(),
                "moral_judgment": mj,
                "moral_foundations": (
                    [s for s in row["rot-moral-foundations"].split("|") if s]
                ),
                "agreement_level": agree,
                "area": row.get("area", ""),
            }
            if rec["moral_foundations"]:
                # Bucket by primary foundation (first tag).
                primary = rec["moral_foundations"][0]
                if primary in MFT_FOUNDATIONS:
                    bucket[primary].append(rec)
                else:
                    unfounded.append(rec)
            else:
                unfounded.append(rec)

    print("After quality filter:")
    for f in MFT_FOUNDATIONS:
        print(f"  {f}: {len(bucket[f])}")
    print(f"  (no MFT primary tag): {len(unfounded)}")

    # Balanced sample: TARGET_TOTAL // 5 per foundation, then top up from any.
    per = TARGET_TOTAL // len(MFT_FOUNDATIONS)
    picks: list[dict] = []
    for f in MFT_FOUNDATIONS:
        rng.shuffle(bucket[f])
        picks.extend(bucket[f][:per])
    # If any foundation was short, top up from unfounded pool
    if len(picks) < TARGET_TOTAL:
        rng.shuffle(unfounded)
        picks.extend(unfounded[: TARGET_TOTAL - len(picks)])

    # Deduplicate by raw_rot_id, keep deterministic order.
    seen = set()
    out: list[dict] = []
    for rec in picks:
        rid = rec.pop("raw_rot_id")
        if rid in seen:
            continue
        seen.add(rid)
        rec_id = "sc101_" + hashlib.md5(rid.encode()).hexdigest()[:12]
        out.append({
            "id": rec_id,
            **rec,
            "source": CITATION,
        })

    # If we lost rows to dedupe, top up from leftover bucket entries.
    if len(out) < TARGET_TOTAL:
        leftover: list[dict] = []
        for f in MFT_FOUNDATIONS:
            leftover.extend(bucket[f])
        leftover.extend(unfounded)
        rng.shuffle(leftover)
        for rec in leftover:
            if len(out) >= TARGET_TOTAL:
                break
            rid = rec.pop("raw_rot_id", "")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            rec_id = "sc101_" + hashlib.md5(rid.encode()).hexdigest()[:12]
            out.append({
                "id": rec_id,
                **rec,
                "source": CITATION,
            })

    out = out[:TARGET_TOTAL]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH}  total={len(out)}")

    # Foundation distribution (primary tag)
    from collections import Counter
    c = Counter(
        (r["moral_foundations"][0] if r["moral_foundations"] else "<none>")
        for r in out
    )
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
