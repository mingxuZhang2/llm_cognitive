# Experiments: Brain as a Reference Frame for LLMs

> Working-directory guide for `experiments/`. The canonical project overview is the **root
> `../CLAUDE.md`**; the review-facing summary is `../PROJECT_SUMMARY_FOR_REVIEW.md`; the detailed
> log is `PROJECT_SUMMARY.md`. **Read those for the science.** This file tells an agent how the
> code in *this directory* is laid out and how to run it.

## What this directory does (current)

Representational Similarity Analysis (RSA) between the **human brain** and **LLMs**, in the
brain-as-reference-frame direction (brain geometry → predict LLM organization). The headline:
a text-only LLM reproduces the brain's relational geometry of **both emotion and social
cognition** (RSA ρ ≈ 0.73, near noise ceiling, 4 architectures, scale-invariant 0.5B–7B, carried
by one causally load-bearing emotion↔social boundary axis). See the overview docs above.

> **Legacy v1 code is still present in this directory.** The original project (v1, archived at
> git tag `v1-ai-categories`) was an *AI-task functional atlas* — neuron attribution + double
> dissociation + atlas-guided pruning over 8 engineering categories (math/code/reasoning/…). Those
> files (`double_dissociation.py`, `multi_function_dissociation.py`, `atlas_pruning.py`,
> `module_discovery.py`, `dose_response.py`, `method_triangulation.py`, `accuracy_dissociation.py`,
> `scaling_law.py`, the `phase0–6_*.sh` SLURM scripts, `PLAN.md`, `analysis_findings.md`,
> `results/{dissociation,multi,scaled_multi,pruning,dose_response,accuracy,subcategory,...}`) are
> **kept for the v1 record — they are NOT the current pipeline.** Do not extend them. The RSA
> pipeline below is the live project.

---

## RSA pipeline (the live project)

**1. Stimuli** — 14 cognitive conditions (6 affective + 7 mentalistic/social + 1 moral),
`data/cognitive_stimuli/rsa/rsa_conditions_manifest.json` (60/condition target).
Builders: `prepare_cognitive_stimuli.py`, `prepare_emotion_stimuli.py`, `build_rsa_stimuli.py`.

**2. Brain RDM** — `build_brain_rdm.py` → `results/cognitive_rsa/brain_rdm.npz`.
Each condition = one Neurosynth meta-analytic map → resample to MNI anchor → mask voxels
finite+nonzero in ≥7/14 maps → flatten → **1 − Pearson** → 14×14 RDM. All 14 maps are Neurosynth
(the ToM map was corrected from HCP on 2026-05-30 — see `tom_source_check.py`). Map downloaders:
`download_neurosynth_maps.py`, `download_hcp_maps.py` (HCP only for the demoted validator).

**3. LLM RDM** — extract hidden states (`extract_rsa_activations_v2.py`, GPU) →
`reconstruct_headline_rdms.py` → `results/cognitive_rsa/{model}_rdm14_headline.npz`.
Recipe: `mean_all | centered | 1_cosine | v1_NS_only` at peak layer (Qwen L27, Llama L31,
Mistral L14, Gemma L21).

**4. Compare** — Spearman ρ on the 91 upper-triangle pairs; permutation null shuffles condition
labels. `rsa_cross_model_v2.py` (cross-arch ρ), `rsa_scaling_analysis.py` (scaling + noise
ceiling), `rsa_deep_analysis.py` (gap + confusion + one-axis causal ablation),
`confirmatory_rsa.py` (discovery/confirmation split, max-stat perm, bootstrap, CV-ablation).

**5. Confound controls** — `baseline_controls.py` (GloVe / TF-IDF / condition-name / length),
`fix_all_holes.py` (partial RSA + LOO / leave-2-out stability),
`affective_ceiling_control.py` (per-block alignment vs LLM split-half ceiling).

**6. Independent real-fMRI validation (Narratives, N=91, same text to brain & LLM)** —
`narratives_preprocess.py` → `narratives_annotate.py` → `narratives_brain_rdm.py`
(`_glm.py` GLM variant) → `extract_narratives_llm.py` (GPU) → compare.
Regional / MVPA variants: `regional_rsa*.py`.

**7. Mechanism / prediction / predictive program** —
`cognitive_steering.py` (brain-derived activation steering), `emotion_geometry.py` (emotion-space
PCA), `brain_causal_coupling.py` (**Direction A** — brain RDM predicts LLM causal coupling, GPU),
`coupling_dissociation_analysis.py` (Direction A **reanalysis** — per-condition double dissociation
from coupling matrices; 4/4 models DD, all Wilcoxon p < 0.007, 14/14 conditions block-selective),
`cognitive_reserve.py` (Direction B), `developmental_emergence.py` (Direction C).

**Demoted validators (directional supplement only — see overview docs):**
`kragel_ibc_reaudit.py`, `kragel_affective_validation.py`, `ibc_boundary_validation.py`,
`hcp_boundary_validation.py` → `results/affective_validation/`.

---

## Key files (live pipeline)

| File | Purpose |
|---|---|
| `src/build_brain_rdm.py` | Build the 14-map Neurosynth brain RDM (1−Pearson). |
| `src/reconstruct_headline_rdms.py` | Rebuild `{model}_rdm14_headline.npz` (headline recipe). |
| `src/rsa_cross_model_v2.py` | 4-model cross-architecture ρ. |
| `src/rsa_scaling_analysis.py` | Scaling curve + noise ceiling (Qwen 0.5B–7B). |
| `src/rsa_deep_analysis.py` | Gap + confusion (geometry→behavior) + one-axis causal ablation. |
| `src/confirmatory_rsa.py` | Discovery/confirmation split, max-stat perm p, bootstrap CI, CV-ablation. |
| `src/baseline_controls.py` | GloVe / TF-IDF / condition-name / length baselines + partial RSA. |
| `src/fix_all_holes.py` | Partial RSA + LOO / leave-2-out stability. |
| `src/tom_source_check.py` | Diagnostic that found the HCP-ToM artifact. |
| `src/brain_causal_coupling.py` | Direction A: brain RDM predicts LLM causal coupling (GPU). |
| `src/coupling_dissociation_analysis.py` | Direction A reanalysis: per-condition double dissociation from coupling matrices. |
| `src/clinical_dissociation.py` | Psychopathy vs autism double-dissociation (ablate emotion/social neuron sets, GPU). |
| `src/subspace_dissociation.py` | **Subspace double-dissociation** (project out emotion/social PCA subspaces from per-stim acts, CPU). Cleaner replacement for clinical_dissociation.py. |
| `src/robustness_gauntlet.py` | **Anti-spurious-alignment gauntlet** (locked-pipeline, LOO, LOMO-CV, stim sub-sampling). Pre-empts Hadidi et al. 2026. All 4/4 pass; rho survives at 0.67+ under every degradation. |
| `src/paraphrase_invariance.py` | **Paraphrase-invariance tests** (split-half stability, LOSO jackknife, cross-source invariance). Shows RSA signal is content-driven, not surface-form-dependent. All 4 models: split-half 95% CI stays above +0.65, LOSO max drop <0.007, cross-source sub-pools agree within 0.03. |
| `src/prospective_prediction.py` | **Prospective prediction battery** (5 tests: coupling asymmetry, within-block coupling, distinctiveness, vulnerable pairs, boundary sensitivity). CPU-only from existing data. |
| `src/steering_controls.py` | **Steering control conditions** (random, sentiment, PC1) to prove brain-derived axis specificity. Same 30 prompts/5 alphas as brain-axis; generates per-control + combined brain-vs-random ranking sheets. GPU. |
| `src/narratives_*.py` | Narratives fMRI pipeline (independent stimulus-locked validation). |
| `present/build_present.py` | Regenerate the plain-language briefing deck (`present/index.html`). |
| `benchmark/evaluate.py` | **BrainCog-14** self-contained benchmark evaluation script (any HF causal LM). |
| `benchmark/README.md` | BrainCog-14 benchmark documentation, conditions, recipe, interpretation guide. |

## Benchmark release: BrainCog-14

`benchmark/` is a **self-contained release package** for the brain-derived social-emotional
geometry benchmark. It contains everything needed to evaluate any HuggingFace causal LM:
- `evaluate.py` — one-file evaluation script (torch + transformers + numpy + scipy only)
- `braincog14_brain_rdm.npz` — 14x14 brain RDM from Neurosynth
- `braincog14_stimuli.jsonl` — 712 stimuli (14 conditions)
- `braincog14_config.json` — locked evaluation recipe
- `baselines.json` — reference results (4 architectures, scaling, GloVe, null)
- `README.md` — full documentation

Usage: `python benchmark/evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --layer 27`

## Results layout (live pipeline)

- `results/cognitive_rsa/` — `brain_rdm.npz`, `{model}_rdm14_headline.npz`, `deep_analysis.json`,
  `scaling_summary.json`, `confirmatory_rsa.json`, `baseline_controls.json`; archived old RDM
  `brain_rdm_hcptom.npz`; dated `FINDINGS*.md` snapshots (**superseded — banners point to current**).
- `results/affective_validation/` — ceiling control, Kragel/IBC/HCP re-audits, ToM source check.
- `results/narratives_brain_rdm/`, `results/developmental_emergence/`, `results/cognitive_reserve/`,
  `results/brain_causal_coupling/`, `results/clinical_dissociation/`,
  `results/robustness_checks/`, `results/emotion_geometry/`,
  `results/next_token/`, `results/specificity_*/` — supporting experiments.

## Running (this cluster — `/hpc2hdd`, NOT the HPC3 in code comments)

- The **login node is memory-saturated** (~5 GB free of 503 GB). Scripts that load the large
  `*_rsa_v2_per_stim.npz` (each ~100–580 MB, float64-expanded) get **OOM-killed (exit 137)**.
  Submit those to **SLURM**: partition `i64m512u` (64-core, 512 GB), account `root`. GPU
  partition: `i64m1tga800u`.
- conda: `source /hpc2hdd/home/mzhang630/miniconda3/etc/profile.d/conda.sh; conda activate base`
  (base has numpy/scipy/sklearn/nibabel/matplotlib). **Do not** combine `set -e` with
  `source ~/.bashrc` (bashrc early-returns non-interactively → conda never inits → job dies in ~2 s).
- Template: `scripts/slurm/rsa_recompute_local.sh`. Logs → `logs/` (gitignored).
  Submit `sbatch …`; watch `squeue -u $USER`.
- Small jobs (tiny RDM NPZ, permutation from `*_rdm14_headline.npz`) are fine on the login node.
- The `phase0–6_*.sh` and `*_dissociation*.sh` / `*_pruning*.sh` SLURM scripts target the **old
  HPC3** (`/data/user/mzhang630`, partition `acd_u`) and are v1 — they won't run here as-is.

## Data note
Large activation/attribution NPZ (`*_rsa_v2_per_stim.npz`, `*_multi_attribution.npz`,
`*_contrast_attribution.npz`, …) are **gitignored** — GPU-reproducible intermediates kept locally,
excluded from GitHub (each >100 MB exceeds the GitHub limit). Re-extract with the `extract_*` scripts.

## Reproduce the headline
```bash
python src/build_brain_rdm.py            # corrected 14-map Neurosynth brain RDM
python src/reconstruct_headline_rdms.py  # LLM headline RDMs (needs extracted activations)
python src/rsa_cross_model_v2.py         # cross-architecture ρ
python src/rsa_deep_analysis.py          # gap + confusion + one-axis causal ablation
python src/baseline_controls.py          # confound baselines + partial RSA
python src/fix_all_holes.py              # partial RSA + LOO stability
# Heavy CPU recomputes (load large per-stim NPZ) → sbatch scripts/slurm/rsa_recompute_local.sh
```
