> # ⚠️ SUPERSEDED — dated snapshot, do not cite these numbers
> **2026-05-25 snapshot, pre-correction.** Uses the **old Neurosynth + HCP** brain RDM and the
> **old ρ ≈ 0.31** recipe. One defective brain map (HCP `theory_of_mind`, row-corr −0.016) inflated
> the divergence narrative. **Current headline: ρ ≈ 0.73** on the pure-Neurosynth RDM, dominated by
> one causally load-bearing emotion↔social axis (within-block detail: social aligns, affective
> unresolved — see within_block_control). Authoritative current numbers:
> `../../../PROJECT_SUMMARY_FOR_REVIEW.md`, `../../PROJECT_SUMMARY.md`, root `../../../CLAUDE.md`.
> Kept only as the honest dated record of the 0.31 → 0.63 → 0.73 progression.

# Cognitive-Domain RSA: Brain vs LLM

**Date:** 2026-05-25
**Pipeline:** RSA between human brain RDM (Neurosynth + HCP) and LLM RDM (4 models)
**Conditions:** 14 cognitive domains (5 basic emotions + valence + 7 social/mentalistic + judgment)
**Stimuli:** 712 (30-60 per condition)
**Distance:** 1 − Pearson over voxels (brain) / hidden states (LLM)
**Null:** 10K label permutations per layer

---

## Main Finding (Novel — Not Previously Reported)

> **The cross-cognitive-domain coupling matrix is preserved between human brain
> and text-trained LLMs, even though the spatial localization of any single
> domain differs.**
>
> Across 4 architecturally distinct LLMs (Qwen / LLaMA / Mistral / Gemma), the
> off-diagonal pairwise dissimilarity structure between 14 cognitive domains
> correlates significantly with the brain's (Spearman ρ in [0.29, 0.35],
> p < 0.025 against label-shuffled null, peak layer ρ_consensus = 0.34).
> The basic-affective cluster (5 Ekman emotions + valence) is universally
> separated from the mentalistic-conceptual cluster (belief / mentalizing /
> self-referential / intention) in both brain and every LLM tested — despite
> LLMs receiving no neural supervision and despite their peak alignment
> layer varying widely (Qwen L7, Gemma L7, Mistral L13, Llama L22).

### Why this is novel
- Prior NeuroAI alignment work compares **one domain at a time** (language, vision, math).
- Nobody has reported a systematic **cross-domain coupling matrix** comparison.
- The finding inverts our earlier failed spatial-correspondence framing: spatial
  layout diverges (brain bilateral DMN vs LLM unimodal middle layers), but the
  **relational organization** of cognitive functions is conserved.

---

## Per-Model RSA Results

| Model | Peak layer | Peak ρ | Peak p | Layers p<0.05 / total |
|-------|------------|--------|--------|-----------------------|
| Qwen2.5-7B-Instruct       | layer_6  (early)     | **0.312** | 0.017 | 20 / 29 |
| Meta-Llama-3.1-8B-Instruct | layer_21 (late)      | **0.328** | 0.015 | 31 / 33 |
| Mistral-7B-Instruct-v0.3  | layer_12 (mid)       | **0.351** | 0.009 | 29 / 33 |
| gemma-2-9b-it             | layer_6  (early)     | **0.289** | 0.024 | (fewer) |
| **Consensus (4-model avg LLM RDM)** | — | **0.342** | — | — |

---

## Universally Preserved-SIMILAR Pairs

(Low dissimilarity in brain AND in every LLM consensus; combined z-score < −2.5)

| Pair | Combined z |
|------|-----------:|
| happiness ↔ sadness | −3.91 |
| disgust ↔ fear      | −3.58 |
| fear ↔ happiness    | −3.48 |
| anger ↔ fear        | −3.15 |
| fear ↔ valence      | −3.08 |
| fear ↔ sadness      | −3.05 |
| disgust ↔ happiness | −2.80 |
| anger ↔ disgust     | −2.76 |

**All 8 universal-similar pairs are within the basic-affective family.**
The valence dimension co-clusters with the discrete emotions in both systems.

---

## Universally Preserved-DISSIMILAR Pairs

(High dissimilarity in brain AND every LLM consensus; combined z-score > +2.0)

| Pair | Combined z |
|------|-----------:|
| belief ↔ happiness        | +2.50 |
| belief ↔ disgust          | +2.48 |
| happiness ↔ mentalizing   | +2.35 |
| belief ↔ fear             | +2.34 |
| disgust ↔ mentalizing     | +2.24 |
| fear ↔ mentalizing        | +2.23 |
| disgust ↔ self_referential | +2.05 |
| mentalizing ↔ sadness     | +2.04 |

**All 8 universal-dissimilar pairs are basic-affective ↔ mentalistic crossings.**
The affective-vs-mentalistic boundary is the most robust topological invariant
shared between brain and LLM.

---

## Robust Disagreements (Brain−LLM)

(Pairs where brain says "very dissimilar" but LLM says "very similar".)

| Pair | Brain RDM | LLM RDM (Gemma peak) | Δ |
|------|----------:|---------------------:|---:|
| fear ↔ judgment        | 1.20 | 0.08 | +1.11 |
| judgment ↔ sadness     | 1.13 | 0.08 | +1.05 |
| happiness ↔ judgment   | 1.12 | 0.09 | +1.03 |
| disgust ↔ judgment     | 1.12 | 0.10 | +1.02 |
| anger ↔ judgment       | 1.00 | 0.08 | +0.92 |

**LLMs collapse "judgment" toward the emotion cluster; the brain separates it
into deliberative-PFC territory.** This is the only systematic structural
disagreement and a clear candidate for follow-up: text-trained models may have
learned moral/normative judgment as emotionally laden, missing the
prefrontal-deliberation distinction the brain maintains.

---

## Method Notes

- **Brain RDM**: 14 cognitive maps (12 Neurosynth association z + 2 HCP task contrasts)
  resampled to MNI 2mm anchor grid, masked to ~900k voxels finite-in-all-maps,
  flattened to voxel vectors, pairwise 1 − Pearson → 14×14 RDM.
- **LLM RDM (per layer)**: 14 stimulus-condition mean activations at last-token,
  pairwise 1 − Pearson → 14×14 RDM per layer.
- **Alignment statistic**: Spearman ρ on the 91 upper-triangle entries.
- **Null**: shuffle the 14 condition labels on the LLM side, recompute ρ; 10K iterations.
- **Consensus LLM RDM**: average peak-layer LLM RDM across 4 models; combined z-score
  identifies pairs ranked similarly in both brain and consensus.

## Caveats

1. Neurosynth association maps reflect **abstract co-occurrence**, not stimulus-locked
   activation. A confirmatory analysis with HCP or LangLoc task data on the same
   stimuli would strengthen the brain-side reliability.
2. LLM mean activations may compress within-condition variability; a
   stimulus-level RDM (instead of condition-mean) would test fine-grained alignment.
3. The 14 conditions cover affective + mentalistic; primary sensorimotor and
   language-syntax domains are not included.

## Files

- `results/cognitive_rsa/brain_rdm.npz` — 14×14 brain RDM
- `results/cognitive_rsa/{model}_rsa.json` — per-model per-layer ρ + p, pair lists
- `results/cognitive_rsa/{model}_rsa_llm_rdms.npz` — full per-layer LLM RDMs
- `results/cognitive_rsa/cross_model_summary.json` — consensus + universal pairs
- `figures/cognitive_rsa_alignment.png` — per-layer ρ curves, 4 models, with null band
- `figures/cognitive_rsa_rdm_grid.png` — brain + 4 LLM RDMs side-by-side
- `figures/cognitive_rsa_preserved_pairs.png` — pairwise scatter, universal pairs labeled
