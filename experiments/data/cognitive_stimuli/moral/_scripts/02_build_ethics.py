"""Sample 200 examples per ETHICS subset into a unified JSONL.

Source: https://people.eecs.berkeley.edu/~hendrycks/ethics.tar (downloaded to
/tmp/ethics_extract/ethics/). Paper:

    Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., &
    Steinhardt, J. (2021). Aligning AI With Shared Human Values.
    International Conference on Learning Representations (ICLR).
    arXiv:2008.02275 / https://github.com/hendrycks/ethics

Per-subset schemas in the raw files (only train splits are used):
    commonsense  (cm_train.csv):
        label,input,is_short,edited
        label=1 means "is morally wrong", label=0 means "not wrong".
    deontology   (deontology_train.csv):
        label,scenario,excuse
        label=1 means the excuse is a reasonable justification, 0 not.
    justice      (justice_train.csv):
        label,scenario
        label=1 means the claim is justifiable, 0 not.
    virtue       (virtue_train.csv):
        label,scenario [SEP] trait
        label=1 means the trait is exhibited, 0 not.
    utilitarianism (util_train.csv):  HEADERLESS pairwise.
        First col = better scenario, second col = worse scenario.

For a unified moral/immoral binary, we expose `label` (0/1) verbatim and add a
`scenario_type` describing what label=1 means in that subset. For
utilitarianism the dataset is pairwise; we emit ONE row per pair with two
columns and `label=1` always (better preference), keeping the convention used
by Hendrycks' eval scripts.

Output schema (one row per line):
    {
      "id":            "ethics_<subset>_<n>",
      "text":          str,    formatted text for the model
      "label":         int,    0 or 1
      "subset":        str,    commonsense / deontology / justice /
                               virtue / utilitarianism
      "scenario_type": str,    short description of what label=1 means
      "source":        str
    }
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SRC_DIR = Path("/tmp/ethics_extract/ethics")
OUT_PATH = Path(
    "/hpc2hdd/home/mzhang630/data/nature/experiments/data/"
    "cognitive_stimuli/moral/ethics_sample.jsonl"
)
N_PER_SUBSET = 200
SEED = 20250525

CITATION = (
    "Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., & "
    "Steinhardt, J. (2021). Aligning AI With Shared Human Values. ICLR 2021. "
    "arXiv:2008.02275."
)


def _balanced_sample(rows: list[dict], n: int, rng: random.Random,
                     label_key: str = "label") -> list[dict]:
    """Sample n rows trying to balance labels 0/1."""
    by_label: dict[int, list[dict]] = {}
    for r in rows:
        by_label.setdefault(int(r[label_key]), []).append(r)
    half = n // 2
    picks = []
    for lab, group in by_label.items():
        rng.shuffle(group)
        take = min(half if len(by_label) > 1 else n, len(group))
        picks.extend(group[:take])
    # Fill if short
    flat = [r for grp in by_label.values() for r in grp]
    rng.shuffle(flat)
    seen_ids = {id(r) for r in picks}
    for r in flat:
        if len(picks) >= n:
            break
        if id(r) in seen_ids:
            continue
        picks.append(r)
        seen_ids.add(id(r))
    rng.shuffle(picks)
    return picks[:n]


def _sample_dicts(path: Path, n: int, rng: random.Random,
                  has_header: bool = True) -> list[dict]:
    with path.open() as f:
        if has_header:
            rows = list(csv.DictReader(f))
        else:
            rows = list(csv.reader(f))
    return rows, _balanced_sample(rows, n, rng) if has_header else None


def _load_pairwise(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                pairs.append((row[0], row[1]))
    return pairs


def main() -> None:
    rng = random.Random(SEED)
    out: list[dict] = []

    # ---- commonsense ---------------------------------------------------
    rows = list(csv.DictReader((SRC_DIR / "commonsense" / "cm_train.csv").open()))
    # cm_train has many long-form and short-form; keep short-form for
    # cleaner stimuli (is_short=True)
    short_rows = [r for r in rows if r.get("is_short", "").lower() == "true"]
    pick = _balanced_sample(short_rows, N_PER_SUBSET, rng)
    for i, r in enumerate(pick):
        out.append({
            "id": f"ethics_commonsense_{i:04d}",
            "text": r["input"].strip(),
            "label": int(r["label"]),
            "subset": "commonsense",
            "scenario_type": "first_person_act; label=1 means morally wrong",
            "source": CITATION,
        })

    # ---- deontology ----------------------------------------------------
    rows = list(csv.DictReader((SRC_DIR / "deontology" / "deontology_train.csv").open()))
    pick = _balanced_sample(rows, N_PER_SUBSET, rng)
    for i, r in enumerate(pick):
        text = (
            f"Question: {r['scenario'].strip()}\n"
            f"Excuse: {r['excuse'].strip()}"
        )
        out.append({
            "id": f"ethics_deontology_{i:04d}",
            "text": text,
            "label": int(r["label"]),
            "subset": "deontology",
            "scenario_type": "request_excuse_pair; label=1 means excuse is "
                             "a reasonable duty-based justification",
            "source": CITATION,
        })

    # ---- justice -------------------------------------------------------
    rows = list(csv.DictReader((SRC_DIR / "justice" / "justice_train.csv").open()))
    pick = _balanced_sample(rows, N_PER_SUBSET, rng)
    for i, r in enumerate(pick):
        out.append({
            "id": f"ethics_justice_{i:04d}",
            "text": r["scenario"].strip(),
            "label": int(r["label"]),
            "subset": "justice",
            "scenario_type": "desert_or_impartiality_claim; "
                             "label=1 means claim is justifiable",
            "source": CITATION,
        })

    # ---- virtue --------------------------------------------------------
    rows = list(csv.DictReader((SRC_DIR / "virtue" / "virtue_train.csv").open()))
    pick = _balanced_sample(rows, N_PER_SUBSET, rng)
    for i, r in enumerate(pick):
        # scenario field already includes "<scene> [SEP] <trait>"
        out.append({
            "id": f"ethics_virtue_{i:04d}",
            "text": r["scenario"].strip(),
            "label": int(r["label"]),
            "subset": "virtue",
            "scenario_type": "scene_trait_pair_with_[SEP]; "
                             "label=1 means trait is exhibited",
            "source": CITATION,
        })

    # ---- utilitarianism ------------------------------------------------
    pairs = _load_pairwise(SRC_DIR / "utilitarianism" / "util_train.csv")
    rng.shuffle(pairs)
    for i, (better, worse) in enumerate(pairs[:N_PER_SUBSET]):
        # Format: pair view; label=1 means scenario_A is the preferred
        # (more pleasant) experience.
        text = (
            f"Scenario A: {better.strip()}\n"
            f"Scenario B: {worse.strip()}"
        )
        out.append({
            "id": f"ethics_utilitarianism_{i:04d}",
            "text": text,
            "label": 1,  # always: by construction A is preferred
            "subset": "utilitarianism",
            "scenario_type": "pleasantness_pair; A is the more pleasant "
                             "experience by construction (label=1)",
            "source": CITATION,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH}  total={len(out)}")

    from collections import Counter
    c = Counter(r["subset"] for r in out)
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
