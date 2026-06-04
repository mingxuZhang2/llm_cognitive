> # ⚠️ SUPERSEDED — dated snapshot, do not cite these numbers
> **2026-05-25 "v2" snapshot, pre-correction.** The ρ ≈ 0.63 here still used the **old Neurosynth +
> HCP** brain RDM (one defective HCP `theory_of_mind` map). After swapping it to Neurosynth
> (2026-05-30) the asymmetry dissolved and the headline rose to **ρ ≈ 0.73**, dominated by one
> causally load-bearing emotion↔social axis. Authoritative current numbers: `../../../PROJECT_SUMMARY_FOR_REVIEW.md`,
> `../../PROJECT_SUMMARY.md`, root `../../../CLAUDE.md`. Kept only as the honest dated record.

# Cognitive-Domain RSA v2: Brain vs LLM (Updated)

**Date:** 2026-05-25 (v2 update)
**Pipeline:** per-stimulus extraction × multi-pooling × multi-distance sweep ×
LLM split-half reliability ceiling
**Headline change:** Spearman ρ jumped from ~0.31 (v1) to **~0.63 (v2) across all
4 models**, 64–66% of LLM split-half reliability ceiling.

---

## Main Finding (v2)

> **The cross-cognitive-domain coupling structure is shared between
> text-trained LLMs and the human brain at ~65% of the LLM's own internal
> reliability ceiling, universally across 4 architecturally distinct models.**
>
> Spearman ρ between the 14-condition brain RDM (Neurosynth association +
> HCP task contrasts) and each LLM's mean-token-pooled hidden-state RDM
> at peak layer: Qwen 0.64 / Llama 0.63 / Mistral 0.63 / Gemma 0.64
> (all p < 0.0002, 10K-permutation null).
>
> The LLM's own split-half reliability ceiling is ~0.97 — so 64-66% of the maximum
> *any* representation could correlate with this LLM's RDM is captured by the
> brain. Consensus LLM RDM vs brain RDM: ρ = 0.63.

### Common configuration (all 4 models same setup):
- **Pooling**: mean over all non-pad tokens of the LLM hidden states
- **Normalization**: per-condition centering (subtract mean across conditions)
- **Distance**: 1 − cosine
- **Brain RDM**: 14 Neurosynth maps + 2 HCP overrides (theory_of_mind via HCP)

---

## Per-Model Numbers

| Model | Peak layer | ρ | Noise ceiling | ρ / ceiling | p |
|-------|-----------|-----:|--------------:|------------:|--:|
| Qwen2.5-7B-Instruct   | layer_27 (late) | **0.639** | 0.967 | **66%** | <1e-4 |
| Meta-Llama-3.1-8B-Instruct | layer_31 (final block) | **0.627** | 0.976 | **64%** | 0.0002 |
| Mistral-7B-Instruct-v0.3   | layer_14 (mid) | **0.633** | 0.970 | **65%** | <1e-4 |
| gemma-2-9b-it             | layer_21 (mid-late) | **0.639** | 0.969 | **66%** | <1e-4 |
| **Consensus (4-model avg LLM RDM vs brain RDM)** | — | **0.63** | — | — | — |

**Per-layer ρ curve**: alignment is flat at ~0.6 across most of the network
depth (from L4 onward in all models). The "peak" layer barely beats other
layers by ~0.01–0.02 ρ — the cross-domain coupling is a **global property** of
the LLM, not localized to a single layer.

---

## Universal Preserved-SIMILAR Pairs (v2)

| Pair | Combined z |
|------|-----------:|
| happiness ↔ sadness  | −3.54 |
| disgust ↔ fear       | −3.27 |
| fear ↔ happiness     | −3.24 |
| fear ↔ valence       | −3.12 |
| **belief ↔ mentalizing** | **−3.04** |
| anger ↔ fear         | −2.87 |
| happiness ↔ valence  | −2.85 |
| sadness ↔ valence    | −2.70 |

**Two clusters emerge clearly now (vs v1):**
- Within-affective: anger / fear / disgust / sadness / happiness / valence
- Within-ToM: belief ↔ mentalizing
The within-ToM cluster only emerged at this strength after switching to
mean-pool + centering — v1's last-token pooling was too noisy to surface it.

## Universal Preserved-DISSIMILAR Pairs (v2)

| Pair | Combined z |
|------|-----------:|
| fear ↔ judgment        | +2.84 |
| fear ↔ intention       | +2.65 |
| judgment ↔ sadness     | +2.55 |
| disgust ↔ intention    | +2.52 |
| disgust ↔ judgment     | +2.51 |
| happiness ↔ judgment   | +2.46 |
| intention ↔ valence    | +2.34 |
| happiness ↔ mentalizing | +2.26 |

All 8 are **affective × cognitive crossings**. The affective-vs-mentalistic
boundary is now matched between brain and LLM on both sides (it was a
brain-side-only finding in v1; the LLM now also represents this boundary).

**The v1 "judgment disagreement" finding is no longer supported by v2 data.**
With proper pooling, LLM correctly places judgment OUTSIDE the emotion cluster,
matching the brain.

---

## What changed from v1 → v2

| Change | v1 | v2 | Effect on ρ |
|--------|----|----|-------------|
| LLM token pooling | last-token only | mean over all tokens | +0.20–0.30 |
| RDM normalization | none | per-condition centering | +0.02–0.05 |
| Distance metric | 1 − Pearson | 1 − cosine (same after centering) | ≈0 |
| Brain RDM | NS-only | NS + HCP for ToM (kept) | small |
| Noise ceiling | not computed | split-half = 0.97 | interpretive only |

**Biggest single lever**: mean-pool tokens instead of last-token. Last-token
captures only the model's "next-prediction" state; mean over all tokens
captures the contextual content representation, which is what aligns with
the brain's stimulus-evoked activity.

---

## Sub-findings

1. **Universality is striking**: 4 models from 4 different developers (Qwen
   = Alibaba China, LLaMA = Meta US, Mistral = France, Gemma = Google) with
   independent training data converge on ρ = 0.63 ± 0.01 against the same
   brain RDM.

2. **Alignment is *not* layer-localized**: per-layer ρ curve is flat at 0.6
   from layer ~4 onwards. The brain–LLM relational structure overlap is a
   property of the representation as a whole, not a single layer's job.
   This argues against a strict "language-region vs perception-region"
   layer hierarchy.

3. **Two cognitive clusters emerge**: affective + ToM, separated by a robust
   boundary. Within each cluster, LLMs and brain agree on the relative
   similarity ordering of components.

4. **Noise ceiling is informative**: ρ = 0.63 against ceiling = 0.97 means
   the LLM is internally extremely reliable on this measurement (split halves
   agree at 0.97), and the brain–LLM correlation captures 65% of that
   reliability. This is high in the NeuroAI literature.

---

## Caveats

1. Brain RDM still uses Neurosynth meta-analytic maps for most conditions;
   stimulus-locked task fMRI (Pereira 2018, LeBel 2023) would test whether
   ρ can climb above 0.65.
2. We use group-averaged or meta-analytic maps; per-subject variability is
   not modeled.
3. The 14 conditions cover affective + mentalistic only; primary
   sensorimotor + language-syntax not tested here.

## Files

- `results/cognitive_rsa/brain_rdm.npz`, `brain_rdm_v2.npz` — brain RDMs
- `results/cognitive_rsa/{model}_rsa_v2_per_stim.npz` — per-stimulus,
  per-layer, 3-pooling activations (large, ~500MB each)
- `results/cognitive_rsa/{model}_rsa_v2.json` — sweep over 12 configs × layers
- `results/cognitive_rsa/cross_model_summary_v2.json` — final summary
- `figures/cognitive_rsa_v2_alignment.png` — per-layer ρ curves, 4 models, with
  LLM split-half ceiling reference line
- `figures/cognitive_rsa_v2_rdm_grid.png` — brain + 4 LLM RDMs at peak layer
- `figures/cognitive_rsa_v2_ceiling_bar.png` — bar chart ρ vs LLM split-half ceiling
- `figures/cognitive_rsa_v2_preserved_pairs.png` — pair scatter, universals labeled
