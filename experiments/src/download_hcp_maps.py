"""Download HCP S1200 group-average task contrast z-stat maps from NeuroVault.

The canonical HCP task-fMRI group-average contrasts (Barch et al., 2013) are
mirrored without authentication in NeuroVault collection 457:
"The WU-Minn Human Connectome Project: An overview" (DOI 10.1016/j.neuroimage.2013.05.041).

The full HCP1200 per-subject contrasts (Volkmann etc.) require ConnectomeDB
registration and are NOT downloaded here.

Output:
    brain_maps/hcp/<short_name>.nii.gz   — z-stat map
    brain_maps/hcp/manifest.json         — list of downloads + metadata
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "brain_maps"
HCP_DIR = DATA_DIR / "hcp"

# Subset of the 48 HCP S1200 group-average contrasts that we care about for the
# emotion / moral / ToM / language / WM alignment study.
TARGET_CONTRASTS = [
    # NeuroVault image id, short slug, description
    # Theory of Mind (SOCIAL task)
    (3181, "social_tom",          "tfMRI SOCIAL TOM (Mental Interaction)"),
    (3179, "social_random",       "tfMRI SOCIAL RANDOM (Random Interaction)"),
    (3180, "social_tom_vs_random","tfMRI SOCIAL TOM minus RANDOM"),
    # Emotion (EMOTION task)
    (3129, "emotion_faces",       "tfMRI EMOTION FACES"),
    (3131, "emotion_shapes",      "tfMRI EMOTION SHAPES"),
    (3128, "emotion_faces_vs_shapes", "tfMRI EMOTION FACES minus SHAPES"),
    # Language (LANGUAGE task)
    (3143, "language_story",      "tfMRI LANGUAGE STORY"),
    (3141, "language_math",       "tfMRI LANGUAGE MATH"),
    (3142, "language_story_vs_math", "tfMRI LANGUAGE STORY minus MATH"),
    # Relational reasoning
    (3175, "relational_rel",      "tfMRI RELATIONAL Relational Processing"),
    (3173, "relational_match",    "tfMRI RELATIONAL Matching"),
    (8820, "relational_rel_vs_match", "tfMRI RELATIONAL REL minus MATCH"),
    # Working memory (WM task)
    (3195, "wm_2bk",              "tfMRI WM 2-back (all)"),
    (3189, "wm_0bk",              "tfMRI WM 0-back (all)"),
    (3190, "wm_2bk_vs_0bk",       "tfMRI WM 2BK minus 0BK"),
    # WM × face (useful for emotion-face contrast)
    (3192, "wm_2bk_face",         "tfMRI WM 2-back Face"),
    # Gambling (reward / punishment — useful for affective valence)
    (3137, "gambling_reward",     "tfMRI GAMBLING REWARD"),
    (3135, "gambling_punish",     "tfMRI GAMBLING PUNISH"),
    (3136, "gambling_reward_vs_punish", "tfMRI GAMBLING REWARD minus PUNISH"),
]

URL_TEMPLATE = "https://neurovault.org/media/images/457/{fname}"


def fname_for_image_id(img_id: int):
    """Look up the NeuroVault filename for an image id (via curl to bypass UA filter)."""
    out = subprocess.check_output(
        ["curl", "-s", "-A", UA, "--max-time", "30",
         f"https://neurovault.org/api/images/{img_id}/"],
        timeout=60,
    )
    meta = json.loads(out)
    f = meta.get("file")
    if not f:
        return None, meta
    return f.rsplit("/", 1)[-1], meta


def download(url: str, dest: Path) -> int:
    """Download via curl (NeuroVault's CDN 403s urllib's default UA)."""
    subprocess.check_call(
        ["curl", "-sSL", "-A", UA, "--max-time", "180",
         "-o", str(dest), url],
        timeout=200,
    )
    return dest.stat().st_size


def main():
    HCP_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "NeuroVault collection 457 — HCP S1200 group-average task fMRI contrasts",
        "doi": "10.1016/j.neuroimage.2013.05.041",
        "neurovault_collection": "https://neurovault.org/collections/457/",
        "license": "HCP Open Access Data Use Terms (https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)",
        "note": (
            "Group-average z-stat maps (zstat1) from FSL FEAT GLM. "
            "These are the canonical Barch et al. 2013 task contrasts available "
            "without ConnectomeDB registration. The per-subject contrasts and "
            "raw timeseries are NOT included here — they require ConnectomeDB."
        ),
        "downloads": [],
        "failures": [],
    }

    for img_id, slug, desc in TARGET_CONTRASTS:
        t0 = time.time()
        try:
            fname, meta = fname_for_image_id(img_id)
            url = URL_TEMPLATE.format(fname=fname)
            dest = HCP_DIR / f"{slug}.nii.gz"
            size = download(url, dest)
            elapsed = time.time() - t0
            manifest["downloads"].append({
                "neurovault_image_id": img_id,
                "slug": slug,
                "description": desc,
                "original_filename": fname,
                "saved_as": str(dest.relative_to(DATA_DIR)),
                "size_bytes": size,
                "contrast_definition": meta.get("contrast_definition"),
                "cognitive_paradigm": meta.get("cognitive_paradigm_cogatlas"),
                "elapsed_s": round(elapsed, 2),
            })
            print(f"  {slug:35s} ({size/1e6:.2f} MB, {elapsed:.1f}s)")
        except Exception as e:
            print(f"  {slug:35s} FAILED: {e}")
            manifest["failures"].append({"slug": slug, "image_id": img_id, "error": str(e)})

    out_path = HCP_DIR / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {out_path}")
    print(f"  downloads: {len(manifest['downloads'])}, failures: {len(manifest['failures'])}")


if __name__ == "__main__":
    main()
