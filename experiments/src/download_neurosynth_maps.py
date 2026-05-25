"""Download Neurosynth association-test maps for cognitive terms of interest.

For each target term, runs an MKDAChi2 meta-analysis with the "association test"
(studies-containing-term vs. studies-not-containing-term) and writes the
z-statistic map (z_desc-association) as a NIfTI file.

Output structure (under brain_maps/neurosynth/):
    <slug>.nii.gz           — association z-stat map for the term
    <slug>_uniformity.nii.gz — uniformity z-stat (less corrected for base rate)
    manifest.json           — list of terms, study counts, file paths, failures

Usage:
    python download_neurosynth_maps.py
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
from nimare.dataset import Dataset
from nimare.meta.cbma.mkda import MKDAChi2


# Target term groups. Spellings adjusted to Neurosynth v7 vocab where possible.
# A "Neurosynth term" is the exact string from the vocab file. Some compound
# terms must be queried with their underscored / spaced form as stored.
TERMS_BY_GROUP = {
    "emotion": [
        ("anger",          "anger"),
        ("disgust",        "disgust"),
        ("fear",           "fear"),
        ("sad",            "sadness"),               # vocab uses "sad"
        ("happy",          "happiness"),             # vocab uses "happy"
        ("happy faces",    "happy_faces"),           # related expression
        ("emotion",        "emotion"),
        ("affective",      "affective"),
        ("valence",        "valence"),
        ("arousal",        "arousal"),
        ("amygdala",       "amygdala"),
    ],
    "moral": [
        ("moral",          "moral"),
        ("judgment",       "judgment"),              # closest term in vocab
        ("intention",      "intention"),             # vocab uses 'intention'/'intentional'
        ("intentional",    "intentional"),
        ("intentions",     "intentions"),
    ],
    "tom_social": [
        ("theory mind",    "theory_of_mind"),        # vocab uses 'theory mind'
        ("mentalizing",    "mentalizing"),
        ("social cognition", "social_cognition"),
        ("social",         "social"),
        ("social interaction", "social_interaction"),
        ("belief",         "belief"),                # ~ false belief proxy
        ("beliefs",        "beliefs"),
        ("empathy",        "empathy"),
    ],
    "self": [
        ("self",                "self"),
        ("self referential",    "self_referential"),
        ("self report",         "self_report"),
    ],
}

# Threshold for considering a study to "contain" a term in its abstract tfidf.
TFIDF_THRESHOLD = 0.001
MIN_STUDIES = 30  # warn below this; very few studies -> unreliable map

DATA_DIR = Path("/hpc2hdd/home/mzhang630/data/nature/experiments/data/brain_maps")
NS_DIR = DATA_DIR / "neurosynth"
RAW_DIR = NS_DIR / "raw"
DATASET_PATH = RAW_DIR / "neurosynth_dataset.pkl.gz"


def slugify(term: str) -> str:
    return term.replace(" ", "_").replace("-", "_").lower()


def run_term_meta(dset: Dataset, term_vocab: str, threshold: float = TFIDF_THRESHOLD):
    """Run MKDAChi2 for a single term. Returns (fit_result, n_in, n_out)."""
    col = f"terms_abstract_tfidf__{term_vocab}"
    if col not in dset.annotations.columns:
        raise KeyError(f"term column not in vocab: {term_vocab!r}")
    vals = dset.annotations[col].values
    mask = vals > threshold
    ids_in = list(dset.annotations[mask]["id"].values)
    ids_out = list(dset.annotations[~mask]["id"].values)
    if len(ids_in) < 2:
        raise ValueError(f"only {len(ids_in)} studies for term {term_vocab!r}")
    dset_in = dset.slice(ids_in)
    dset_out = dset.slice(ids_out)
    mkda = MKDAChi2(kernel__r=10)
    result = mkda.fit(dset_in, dset_out)
    return result, len(ids_in), len(ids_out)


def main():
    print(f"Loading Neurosynth Dataset from {DATASET_PATH}")
    dset = Dataset.load(str(DATASET_PATH))
    print(f"  total studies: {len(dset.ids)}")

    NS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "Neurosynth v7 abstracts, MKDA Chi-square association test (vs. all other studies)",
        "dataset_path": str(DATASET_PATH),
        "total_studies_in_corpus": len(dset.ids),
        "tfidf_threshold": TFIDF_THRESHOLD,
        "kernel_radius_mm": 10,
        "min_studies_warning": MIN_STUDIES,
        "groups": {},
        "failures": [],
        "low_count_warnings": [],
    }

    for group, terms in TERMS_BY_GROUP.items():
        print(f"\n=== Group: {group} ===")
        manifest["groups"][group] = []
        for term_vocab, out_name in terms:
            t0 = time.time()
            slug = slugify(out_name)
            assoc_path = NS_DIR / f"{slug}.nii.gz"
            unif_path = NS_DIR / f"{slug}_uniformity.nii.gz"
            try:
                result, n_in, n_out = run_term_meta(dset, term_vocab)
                # association test = z map of P(activation | term) vs P(activation | ~term)
                result.save_maps(
                    output_dir=str(NS_DIR),
                    prefix=f"{slug}_full",
                )
                # Save the two main maps explicitly with clean names
                img_assoc = result.get_map("z_desc-association", return_type="image")
                img_assoc.to_filename(str(assoc_path))
                img_unif = result.get_map("z_desc-uniformity", return_type="image")
                img_unif.to_filename(str(unif_path))
                rec = {
                    "term_vocab": term_vocab,
                    "out_name": out_name,
                    "slug": slug,
                    "n_studies_term": n_in,
                    "n_studies_other": n_out,
                    "association_map": str(assoc_path.relative_to(DATA_DIR)),
                    "uniformity_map": str(unif_path.relative_to(DATA_DIR)),
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if n_in < MIN_STUDIES:
                    rec["low_count_warning"] = True
                    manifest["low_count_warnings"].append(
                        {"term": term_vocab, "n_studies": n_in}
                    )
                manifest["groups"][group].append(rec)
                msg = f"  {term_vocab!r:30s} -> {n_in:4d} studies, {time.time()-t0:.1f}s"
                if n_in < MIN_STUDIES:
                    msg += "  [LOW COUNT]"
                print(msg)
            except Exception as e:
                tb = traceback.format_exc(limit=2)
                print(f"  {term_vocab!r:30s} -> FAILED ({type(e).__name__}: {e})")
                manifest["failures"].append(
                    {
                        "term_vocab": term_vocab,
                        "out_name": out_name,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": tb,
                    }
                )

    # Clean up intermediate map files (we only keep the two main z-stat maps)
    for f in NS_DIR.glob("*_full*.nii.gz"):
        f.unlink()

    out_manifest = NS_DIR / "manifest.json"
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {out_manifest}")
    n_ok = sum(len(v) for v in manifest["groups"].values())
    n_fail = len(manifest["failures"])
    print(f"Maps written: {n_ok}, failures: {n_fail}")


if __name__ == "__main__":
    main()
