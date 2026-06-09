> # ⚠️ SUPERSEDED — dated snapshot, do not cite these numbers
> **2026-05-26 snapshot, pre-correction.** Built on the **old Neurosynth + HCP** brain RDM, so the
> per-size ρ here (0.665 … 0.639) are the old values. The *scale-invariance conclusion is unchanged
> and confirmed*, but the corrected per-size numbers are **0.752 / 0.754 / 0.747 / 0.739** (Δρ
> = −0.013). Authoritative current numbers: `../../../PROJECT_SUMMARY_FOR_REVIEW.md`,
> `../../PROJECT_SUMMARY.md`. Kept only as the honest dated record.

# RSA Scaling Analysis: 4-point Qwen2.5-Instruct curve

**Date:** 2026-05-26
**Config:** mean-pool tokens, centered, 1−cosine, 14-condition Neurosynth+HCP brain RDM

---

## Main Finding (Scaling Axis)

> **Brain–LLM cross-cognitive-domain RSA alignment is essentially scale-invariant
> within the Qwen2.5-Instruct family. The structure is fully formed at 0.5 B
> parameters; scaling to 15× larger (7 B) yields ρ change of just −0.026.**

### Per-size results

| Size | Params | Peak layer | Peak ρ | Noise ceiling | ρ / ceiling |
|------|-------:|-----------:|-------:|--------------:|------------:|
| **0.5B**  | 494M   | L21 / 24 | **0.665** | 0.965 | **68.9 %** |
| **1.5B**  | 1.54B  | L26 / 28 | **0.655** | 0.964 | **67.9 %** |
| **3B**    | 3.09B  | L29 / 36 | **0.657** | 0.962 | **68.3 %** |
| **7B**    | 7.61B  | L27 / 28 | **0.639** | 0.969 | **65.9 %** |

(14B and 32B downloads in progress; will extend the curve later.)

### Interpretation

This is the OPPOSITE of typical scaling-emergence findings:
- Math, code, reasoning capabilities all show steep emergence curves between
  1B → 10B → 100B parameters
- The cross-cognitive-domain coupling structure does NOT — it is captured
  identically by a 0.5 B parameter model and a 7 B model
- Both models capture ~68% of their (essentially identical) LLM split-half reliability ceilings

### Why this is novel

This decouples "what brain-aligned representations encode" from
"what large LLMs uniquely do":
1. The 14-condition cross-domain coupling matrix is a **basic statistical
   structure of natural language**, learned by any sufficiently competent
   language model (≥ 0.5 B parameters).
2. It is NOT a capability that scales — unlike Q&A accuracy, chain-of-thought,
   math, code.
3. This suggests the relational organization of cognitive functions is
   established early in language learning (analogous to early childhood
   acquisition, not late-adolescence reasoning maturation).

### Peak layer scales with depth

| Size | n_layers | peak_layer | relative depth |
|------|---------:|-----------:|---------------:|
| 0.5B | 24 | 21 | 0.88 |
| 1.5B | 28 | 26 | 0.93 |
| 3B   | 36 | 29 | 0.81 |
| 7B   | 28 | 27 | 0.96 |

All peak in the last ~15% of network depth — the alignment is consistently
a property of late representations, scaled proportionally to network size.

---

## Caveats

1. Only Qwen2.5 family tested; cross-architecture scaling (e.g., LLaMA family)
   not yet measured. But cross-architecture at 7-9B in v2 shows ρ = 0.63 ± 0.01
   for Qwen/Llama/Mistral/Gemma, consistent with scale-invariance.
2. No size below 0.5 B tested — there may be a phase transition below this
   (e.g., 100 M might lose the structure).
3. Instruct-tuned models only; base models may differ.
4. Single brain RDM (14 conditions); test on additional cognitive batteries
   would strengthen the claim.

## Files

- `results/cognitive_rsa/scaling_summary.json` — per-size summary
- `figures/cognitive_rsa_scaling_curve.png` — 4-point scaling figure
- `src/rsa_scaling_analysis.py` — analysis script
- `scripts/auto_pipeline_size.sh` — sync + SLURM helper
