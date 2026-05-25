"""
Brain alignment: compare LLM layer-wise norm_type contrast against
Neurosynth meta-analytic maps for "moral", "judgment", "theory_of_mind",
and "self" terms.

For each Neurosynth term we compute a 1-D ANTERIOR–POSTERIOR (Y-axis)
"spatial fingerprint" by binning voxel activation along the Y MNI coordinate.
This collapses the 3-D map to a 1-D curve that runs from posterior (occipital,
early visual) to anterior (prefrontal). Higher-order moral / social cognition
should peak ANTERIORLY (vmPFC) with secondary peaks around the temporo-parietal
junction (mid-posterior).

For the LLM we compute a 1-D LAYER fingerprint by summing the positive contrast
magnitude per layer of the norm_type_selectivity vector, then normalizing.

If brain moral processing peaks at ANTERIOR coordinates while LLM moral-vs-
conventional contrast peaks at LATE layers (also abstract / integrative),
that's the qualitative parallel for the alignment claim. If brain peaks in
MIDDLE position (association cortex) and LLM peaks in middle layers, also good.

We then compute the Spearman rank correlation between the LLM layer profile
and the brain AP profile (after interpolating to a common axis length).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
NEUROSYNTH = ROOT / "data" / "brain_maps" / "neurosynth"
CONTRAST = ROOT / "results" / "contrast_pilot"
META_DIR = ROOT / "activations"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Terms to extract from Neurosynth (association test maps)
BRAIN_TERMS = ["moral", "judgment", "theory_mind", "mentalizing",
                "intention", "self", "emotion", "anger"]
MODELS = ["Qwen2.5-7B-Instruct", "Meta-Llama-3.1-8B-Instruct",
          "Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"]
SHORT = {"Qwen2.5-7B-Instruct": "Qwen 7B",
         "Meta-Llama-3.1-8B-Instruct": "Llama 8B",
         "Mistral-7B-Instruct-v0.3": "Mistral 7B",
         "gemma-2-9b-it": "Gemma 9B"}


def brain_ap_profile(nii_path: Path, n_bins=20):
    """Collapse a 3D Neurosynth association map to a 1-D anterior-posterior
    profile by binning along the Y axis of MNI space.

    Returns: (bin_centers_mm, mean_pos_value_per_bin)
    """
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    affine = img.affine
    # Only positive activations contribute (association maps are signed but we
    # care about regions preferentially activated by the term).
    pos = np.clip(data, 0, None)
    nx, ny, nz = pos.shape
    # Per-voxel MNI Y coordinates
    Y, X, Z = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz), indexing="ij")
    voxels = np.stack([X, Y, Z, np.ones_like(X)], axis=-1)
    mni = voxels @ affine.T  # shape (ny, nx, nz, 4)
    y_coords = mni[..., 1]  # MNI y-axis (front-back)

    flat_y = y_coords.flatten()
    flat_v = pos.transpose(1, 0, 2).flatten()  # match meshgrid ordering
    # discard zero-activation voxels for stability
    mask = flat_v > 0
    flat_y = flat_y[mask]
    flat_v = flat_v[mask]
    if len(flat_v) == 0:
        return np.linspace(-80, 80, n_bins), np.zeros(n_bins)

    bins = np.linspace(-80, 80, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    sums, _ = np.histogram(flat_y, bins=bins, weights=flat_v)
    counts, _ = np.histogram(flat_y, bins=bins)
    profile = np.divide(sums, counts + 1e-6, out=np.zeros_like(sums), where=counts > 0)
    return centers, profile


def llm_layer_profile(model_short, condition="norm_type", topk=5000):
    """Return per-layer sum of contrast magnitude for the top-k positive
    neurons of the requested condition."""
    npz = np.load(CONTRAST / f"{model_short}_contrast_attribution.npz")
    sel = npz[f"{condition}_selectivity"]
    with open(META_DIR / f"{model_short}_cognitive_pilot_meta.json") as f:
        m = json.load(f)
    n_layers, ffn_dim = m["n_layers"], m["ffn_dim"]
    top = np.argsort(-sel)[:topk]
    profile = np.zeros(n_layers)
    for n in top:
        profile[n // ffn_dim] += sel[n]
    # Normalize to share
    if profile.sum() > 0:
        profile = profile / profile.sum()
    return profile, n_layers


def main():
    # Brain AP profiles
    brain_profiles = {}
    for term in BRAIN_TERMS:
        path = NEUROSYNTH / f"{term}.nii.gz"
        if not path.exists():
            print(f"  missing: {path}")
            continue
        centers, prof = brain_ap_profile(path, n_bins=20)
        brain_profiles[term] = (centers, prof)
        print(f"  brain '{term}': peak at Y={centers[np.argmax(prof)]:.0f} mm "
              f"(max value {prof.max():.3f})")

    # LLM layer profiles
    llm_profiles = {}
    for m in MODELS:
        prof, n_layers = llm_layer_profile(m, "norm_type")
        depth = np.arange(n_layers) / max(n_layers - 1, 1)
        llm_profiles[m] = (depth, prof, n_layers)
        peak_layer = int(np.argmax(prof))
        print(f"  LLM '{SHORT[m]}': peak at layer {peak_layer}/{n_layers} "
              f"(normalized depth {peak_layer/(n_layers-1):.2f}, share {prof[peak_layer]:.3f})")

    # ---------- Figure 1: brain AP profiles ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors
    for i, term in enumerate(BRAIN_TERMS):
        if term not in brain_profiles:
            continue
        centers, prof = brain_profiles[term]
        # normalize for shape comparison
        norm_prof = prof / (prof.max() + 1e-9)
        ax.plot(centers, norm_prof, marker="o", markersize=4, linewidth=1.5,
                color=colors[i % 10], label=term, alpha=0.85)
    ax.set_xlabel("MNI Y-coordinate (mm) — posterior (occipital) → anterior (prefrontal)")
    ax.set_ylabel("Normalized mean positive activation")
    ax.set_title("Neurosynth association maps — anterior-posterior profile (normalized)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "brain_ap_profiles.png", dpi=140)
    plt.close()

    # ---------- Figure 2: side-by-side LLM layer profile vs brain "moral" ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: brain moral / TOM / convention proxy
    ax = axes[0]
    for i, term in enumerate(["moral", "theory_mind", "mentalizing", "intention"]):
        if term not in brain_profiles:
            continue
        centers, prof = brain_profiles[term]
        ax.plot(centers, prof / (prof.max() + 1e-9), marker="o", markersize=4,
                linewidth=1.5, label=term, alpha=0.85)
    ax.set_xlabel("MNI Y (mm) — posterior → anterior")
    ax.set_ylabel("Normalized activation")
    ax.set_title("Brain anterior-posterior profile (Neurosynth moral / mentalizing)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: LLM layer profile per model
    ax = axes[1]
    for m in MODELS:
        depth, prof, _ = llm_profiles[m]
        ax.plot(depth, prof / (prof.max() + 1e-9), marker="o", markersize=4,
                linewidth=1.5, label=SHORT[m], alpha=0.85)
    ax.set_xlabel("Normalized layer depth (0=input, 1=output)")
    ax.set_ylabel("Normalized share of top-5000 norm_type contrast")
    ax.set_title("LLM norm_type contrast layer profile (4 architectures)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Brain vs LLM spatial fingerprint of moral cognition\n"
        "Brain: anterior peaks for moral/ToM/intention. LLM: middle-layer peaks for norm_type contrast.",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_DIR / "brain_vs_llm_moral.png", dpi=140)
    plt.close()

    # ---------- Quantitative comparison: cross-correlation ----------
    # Interpolate both profiles to 50-bin common axis.
    print("\n" + "=" * 80)
    print("CROSS-CORRELATION: brain AP profile vs LLM layer profile")
    print("=" * 80)
    common = np.linspace(0, 1, 50)

    # Normalize brain AP coords to 0-1 (posterior to anterior)
    print(f"\n{'LLM model':>12s} | " + " ".join(f"{t[:10]:>10s}" for t in BRAIN_TERMS))
    print(f"  {'-'*12} +" + "-+".join(["-"*10]*len(BRAIN_TERMS)))
    for m in MODELS:
        depth, prof, n_layers = llm_profiles[m]
        llm_interp = np.interp(common, depth, prof)
        row = f"  {SHORT[m]:<12s} |"
        for term in BRAIN_TERMS:
            if term not in brain_profiles:
                row += f" {'NA':>10s}"
                continue
            centers, bp = brain_profiles[term]
            # Map MNI Y from posterior (-80) to anterior (+80) onto 0-1
            bp_x = (centers - centers.min()) / (centers.max() - centers.min())
            brain_interp = np.interp(common, bp_x, bp)
            r, _ = stats.spearmanr(llm_interp, brain_interp)
            row += f" {r:>+10.3f}"
        print(row)

    print(f"\n  Saved figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
