# LLM Cognitive Atlas: Brain-LLM Representational Alignment

Comparing the internal cognitive organization of Large Language Models with the human brain using Representational Similarity Analysis (RSA).

## Key Finding

Brain-LLM cognitive alignment reduces to **a single representational dimension** — the emotion-reasoning boundary — which is:
- **Universal** across 4 architectures (Qwen/Llama/Mistral/Gemma)
- **Scale-invariant** from 0.5B to 7B parameters
- **Present in base models** (91% of alignment exists before RLHF)
- **Causally manipulable** (removing this one direction inverts brain-AI alignment from ρ=+0.64 to ρ=−0.36)
- **Validated on real fMRI** (N=91 subjects, stimulus-locked, ρ=0.56, 73% of brain noise ceiling)

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
