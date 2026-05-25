"""Build moral_foundations_vignettes.jsonl from the canonical Clifford 2015 CSV.

Source CSV: data/survey/vignettes.csv from peterkirgis/llm-moral-foundations,
which mirrors the 132-vignette stimulus set published in:

    Clifford, S., Iyengar, V., Cabeza, R., & Sinnott-Armstrong, W. (2015).
    Moral foundations vignettes: A standardized stimulus database of scenarios
    based on moral foundations theory. Behavior Research Methods, 47(4),
    1178-1198. https://doi.org/10.3758/s13428-014-0551-2

Schema we emit (one JSON object per line):
    {
      "id":               str,    "mfv_<NNN>"
      "vignette_text":    str,    third-person observer scenario
      "foundation":       str,    one of {care, fairness, loyalty,
                                   authority, sanctity, liberty,
                                   social_norms}
      "foundation_subtype": str|None,  Care/(e), Care/(p,a), Care/(p,h)
                                       if applicable
      "wrongness_mean":   float,  on Clifford's 0-4 "How Wrong" scale
      "wrongness_sd":     null,   not in published per-vignette tables;
                                   see README for details
      "foundation_percentages": {  human modal-foundation classifications
            "care": float, "fairness": float, "loyalty": float,
            "authority": float, "sanctity": float, "liberty": float,
            "not_wrong": float
        }
      "source":           str,    full citation
    }
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

SRC_CSV = Path("/tmp/mfv_classic.csv")
OUT_PATH = Path(
    "/hpc2hdd/home/mzhang630/data/nature/experiments/data/"
    "cognitive_stimuli/moral/moral_foundations_vignettes.jsonl"
)

CITATION = (
    "Clifford, S., Iyengar, V., Cabeza, R., & Sinnott-Armstrong, W. (2015). "
    "Moral foundations vignettes: A standardized stimulus database of "
    "scenarios based on moral foundations theory. Behavior Research Methods, "
    "47(4), 1178-1198. https://doi.org/10.3758/s13428-014-0551-2"
)


def _parse_foundation(raw: str) -> tuple[str, str | None]:
    """Map raw Foundation column to (canonical, subtype)."""
    raw = raw.strip()
    if raw.startswith("Care"):
        # Care (e) = emotional, Care (p, a) = physical animal,
        # Care (p, h) = physical human
        sub = re.search(r"\((.+)\)", raw)
        return "care", sub.group(1).strip() if sub else None
    if raw == "Social Norms":
        return "social_norms", None
    return raw.lower(), None


def _parse_pct(s: str) -> float:
    s = s.strip().rstrip("%").strip()
    return float(s) if s else 0.0


def main() -> None:
    rows = list(csv.DictReader(SRC_CSV.open()))
    if len(rows) != 132:
        raise RuntimeError(f"expected 132 vignettes, got {len(rows)}")

    out = []
    for i, row in enumerate(rows, 1):
        foundation, subtype = _parse_foundation(row["Foundation"])
        text = row["Scenario"].strip()
        rec = {
            "id": f"mfv_{i:03d}",
            "vignette_text": text,
            "foundation": foundation,
            "foundation_subtype": subtype,
            "wrongness_mean": float(row["Wrong"]),
            "wrongness_sd": None,
            "foundation_percentages": {
                "care": _parse_pct(row["Care"]),
                "fairness": _parse_pct(row["Fairness"]),
                "loyalty": _parse_pct(row["Loyalty"]),
                "authority": _parse_pct(row["Authority"]),
                "sanctity": _parse_pct(row["Sanctity"]),
                "liberty": _parse_pct(row["Liberty"]),
                "not_wrong": _parse_pct(row["Not Wrong"]),
            },
            "source": CITATION,
        }
        out.append(rec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Verify
    with OUT_PATH.open() as f:
        n = sum(1 for _ in f)
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()[:12]
    print(f"wrote {OUT_PATH}  rows={n}  sha={sha}")

    # Quick distribution
    from collections import Counter
    c = Counter(r["foundation"] for r in out)
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
