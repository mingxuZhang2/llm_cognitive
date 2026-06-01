# The Brain as a Reference Frame for LLMs

Using the human brain as a **predictive reference frame** for the internal organization of
Large Language Models, via Representational Similarity Analysis (RSA). Direction matters: we
use established neuroscience as *testable predictions about LLMs*, not LLMs as a model of the
brain.

## Key Finding

A text-only LLM reproduces the human brain's **relational organization of both emotion and
social cognition**: RSA between each model's internal geometry and meta-analytic fMRI maps
(Neurosynth, 14 conditions → 91 condition-pairs) gives **ρ ≈ 0.73**, near the noise ceiling.
This alignment is:
- **Universal** across 4 architectures (Qwen 0.739 / Llama 0.727 / Mistral 0.730 / Gemma 0.735)
- **Scale-invariant** from 0.5B to 7B parameters
- **Dominated by one causally load-bearing axis** — the alignment rides on the single
  emotion ↔ social-cognition division (removing it sends ρ +0.73 → −0.36). Beyond that
  categorical split a significant residual brain–LLM agreement remains (partial ρ ≈ 0.36,
  all 4 models p ≤ 0.001), concentrated in the within-social ordering. The within-affective
  fine structure does not resolve at the current condition count (n=6) — no claim is made
  about it. We frame the result as *one shared organizing axis + a significant beyond-split
  residual*, **not** a rich fine-grained match across both families.
- **Causally predictive** — the brain RDM predicts the LLM's internal causal coupling
  between functions (Direction A; 3/4 models significant)

> **Note (2026-05-30 correction):** an earlier framing claimed the alignment "reduces to a
> single emotion-reasoning dimension" with a social-cognition *divergence*. That asymmetry
> was an artifact of one bad brain map (HCP theory-of-mind); on a pure-Neurosynth RDM it
> dissolves and ρ rises 0.63 → 0.73. Any figure/number describing divergence, anti-aligned
> ToM, or a single-dimension reduction predates this fix and is superseded. See
> `experiments/present/index.html` for the corrected briefing.

## Repository Structure

```
experiments/
  src/                    — All analysis scripts
  scripts/slurm/          — HPC3 SLURM job scripts
  results/                — Experimental results (JSON, NPZ)
  figures/                — Generated visualizations
  data/                   — Stimuli and fMRI data
    cognitive_stimuli/    — Curated cognitive task stimuli
    narratives/           — Narratives fMRI dataset annotations
```

## Detailed Summary

See [experiments/PROJECT_SUMMARY.md](experiments/PROJECT_SUMMARY.md) for:
- Complete experimental methods and results
- All 11 experiments with tables and findings
- Related work analysis
- Code structure and reproduction instructions

## Quick Start

```bash
# Core RSA analysis (requires pre-computed NPZ files)
cd experiments
python src/rsa_deep_analysis.py          # Gap + confusion + causal ablation
python src/rsa_scaling_analysis.py       # Scaling curve
python src/narratives_brain_rdm.py       # Brain RDM from fMRI
python src/emotion_geometry.py           # Emotion space analysis
```

## Models Tested
- Qwen2.5-Instruct: 0.5B, 1.5B, 3B, 7B (scaling) + 1.5B base
- Meta-Llama-3.1-8B-Instruct
- Mistral-7B-Instruct-v0.3
- gemma-2-9b-it

## Brain Data
- Neurosynth meta-analytic maps (14 cognitive conditions)
- Narratives fMRI dataset (Nastase et al. 2021): 91-258 subjects, stimulus-locked
