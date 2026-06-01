# Project Status — Brain-LLM Representational Alignment

> **2026-05-31.** This document was rewritten after the HCP-ToM artifact correction. The
> previous status ("LLMs compress human mental-state geometry / brains separate minds, language
> models compress them / extraction failure") was built on a single defective brain map and is
> **retracted**. See `PROJECT_SUMMARY.md` §header and `present/index.html` for the corrected
> story; this file is the short current-status view.

## One-line summary

A text-only LLM reproduces the human brain's relational organization of **both emotion and
social cognition** (RSA ρ ≈ 0.73, near noise ceiling), carried by a single brain-like axis —
the emotion ↔ social-cognition boundary — that is causally load-bearing.

---

## Where the project stands

**Core finding (solid, recomputed against the corrected RDM):**
- ρ ≈ 0.73 across 4 architectures (Qwen .739 / Llama .727 / Mistral .730 / Gemma .735),
  permutation p ≈ 0.0001–0.0002.
- Per-block **row-wise** (split-dominated, not a within-block test): affective 78%, mentalistic
  87%; every condition aligns 0.67–0.85 except empathy (0.24, underpowered n=32).
- **Within-block control:** dominated by the single emotion↔social split; beyond it a significant
  residual survives (partial ρ≈0.36, 4/4 p≤0.001), living in the within-social ordering
  (ρ≈0.52–0.65). Within-affective ordering does **not** align (ρ≈−0.11) but is underpowered
  (n=6) → no claim. Net: one load-bearing axis + beyond-split residual, not a rich 14-way match.
- Scale-invariant (0.5B–7B flat: 0.752/0.754/0.747/0.739, Δρ=−0.013, all ≈78% of ceiling),
  present in base models (~91%), survives confound control (~80% retained after partialling
  word-embedding + concept-name + length, ρ 0.74→0.59, p=0.0002). Bootstrap 95% CI
  [0.719, 0.759]; max-stat p=0.0002 (peak-layer-corrected).
- Independent stimulus-locked fMRI (Narratives, N=91): ρ ≈ 0.56, 71–76% of brain ceiling.
- **Causal axis:** removing the emotion↔social boundary direction (≈PC1, cosine 0.9999) inverts
  ρ from +0.73 to −0.36 (4/4 models, p<0.0001; 2000 random-direction controls ≈ 0).
- **Behavioral prediction:** brain geometry predicts LLM confusion (ρ=0.24, p=0.02) and internal
  causal coupling (Direction A, 3/4 models, LOO/leave-2-out stable).

**What was retracted (artifact of the HCP-ToM map):** the emotion-vs-social *asymmetry*; the
"theory-of-mind is anti-aligned" claim; the "mentalistic collapse / LLMs compress mental-state
geometry"; the "capacity-but-extraction-failure" framing; and the title "Brains separate minds;
language models compress them." On the corrected RDM the within-mentalistic residual fell from
19.2 to 13.8 and the top "misaligned" pairs (all previously theory_of_mind) disappeared.

---

## The strategic frame (why this matters)

Use the brain as a **predictive reference frame for LLMs**: established neuroscience built on the
emotion/social-cognition separation becomes a battery of testable LLM predictions.
- Lesion double-dissociation (Shamay-Tsoory 2009) → ablate the LLM's emotion subspace, hit
  emotion tasks selectively, spare social tasks. → **Direction A, in progress (3/4 models).**
- Dual-process moral cognition + lesion→behavior (Greene 2001; Koenigs 2007) → suppressing the
  emotion axis should shift moral judgment toward utilitarian/analytical. → **steering result
  (Exp 7) is consistent; quantify next.**
- Cortical processing gradient places social cognition at the abstract end (Margulies 2016) →
  social cognition should sit at greater LLM layer-depth than emotion. → **layer data available.**
- Developmental order: affect early, theory-of-mind ~age 4 → emotion structure should form
  before social structure along training. → **needs training checkpoints.**

---

## Open questions / next steps

1. **Layer-depth test** (cortical-gradient prediction) — cheapest next win; data in hand.
2. **Quantify the moral-judgment steering** (dual-process prediction) on a proper utilitarian-vs-
   deontological battery.
3. **Empathy condition** is underpowered (n=32) — rebuild with a real empathy-induction set.
4. **Do not over-sell pure-Neurosynth:** related social terms (ToM/mentalizing/belief) come from
   overlapping literature and are inherently similar — the robust claims are the high overall ρ
   and the emotion↔social boundary, which hold in every version and on independent fMRI.

---

## Quantitative results in one table (corrected)

| Result | Number | Status |
|---|---|---|
| Cross-architecture RSA (4 models) | ρ = 0.727–0.739 | ✓ |
| Matrix permutation p (per model) | 0.0001–0.0002 | ✓ |
| Per-block row-wise (split-dominated) | aff 78%, ment 87% | ✓ |
| Within-block residual (partial out split) | partial ρ≈0.36, 4/4 p≤0.001 | ✓ |
| → within-social ordering | ρ≈0.52–0.65 (sig.) | ✓ |
| → within-affective ordering | ρ≈−0.11, n=6 underpowered | no claim |
| Stimulus-locked fMRI (N=91) | ρ ≈ 0.54–0.58 | ✓ |
| Scale invariance (1.5B vs 7B) | 0.754 vs 0.739 | ✓ |
| Base vs instruct | base ≈ 91% | ✓ |
| Partial RSA (embedding+name+length) | ρ 0.74→0.59 (80%), p=2e-4 | ✓ |
| Untrained baseline | ≈ 0 (ns) | ✓ |
| Boundary-axis ablation | +0.73 → −0.36 (4/4, p<1e-4) | ✓ |
| Boundary ≈ PC1 | cosine 0.9999 | ✓ |
| Brain predicts LLM confusion | ρ = 0.24, p = 0.02 | ✓ |
| Direction A (causal coupling) | 3/4 models, LOO stable | ✓ |
| Brain-derived steering | emotional ↔ analytical | ✓ |
| Emotion geometry | valence-dominant | ✓ Descriptive |
| Empathy condition | ρ = 0.24 (n=32) | ⚠ underpowered |

---

Code: https://github.com/mingxuZhang2/llm_cognitive (branch `cognitive-atlas`).
Canonical overview: `../CLAUDE.md`. Detailed experiments: `PROJECT_SUMMARY.md`.
Briefing deck: `present/index.html`.
