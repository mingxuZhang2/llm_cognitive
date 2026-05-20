# Experiment Code: Causal Functional Modules in LLMs

## Pipeline
1. Phase 1: `src/activation_extraction.py` — Extract FFN activations
2. Phase 2: `src/module_discovery.py` — IterD dual partitioning
3. Phase 3: `src/double_dissociation.py` — Ablation + benchmark evaluation
4. Phase 4: Atlas-guided pruning (TODO)
5. Phase 5: Modularity scaling law (TODO)
6. Phase 6: Brain comparison (TODO)

## Running on HPC3
- SLURM scripts in `scripts/slurm/`
- Submit: `sbatch scripts/slurm/phase1_extract_activations.sh`
- Logs: `/data/user/mzhang630/logs/`

## Stimulus Preparation
- `src/stimulus_preparation.py` — Original: downloads from HuggingFace, creates stimuli_full.jsonl (3133 samples, unbalanced)
- `src/prepare_balanced_stimuli.py` — Creates balanced subsets from stimuli_full.jsonl:
  - `stimuli_balanced.jsonl` — 152/category (matches smallest category = science), 1216 total
  - `stimuli_medium.jsonl` — 50/category, 400 total (for fast iteration)
- 8 categories: code, ethics, factual_qa, humanities, language, math, reasoning, science

## Scaled Experiments
- `scripts/slurm/scaled_multi_dissociation.sh` — 8-way dissociation on medium set (50/cat) across 4 models
  - Output: `/data/user/mzhang630/data/nature_exp/results/scaled_multi/`

## Dose-Response Experiment
- `src/dose_response.py` — Dose-response curves for causal ablation
  - Loads pre-computed selectivity from `multi_attribution.npz` (no re-computation)
  - Sweeps ablation sizes: 500, 1000, 2000, 5000, 10000, 20000 neurons
  - Tests math and code functions; also runs random ablation controls
  - Output: `{model_short}_dose_response.json`
- `scripts/slurm/dose_response.sh` — SLURM script for Qwen on HPC3
  - Output: `/data/user/mzhang630/data/nature_exp/results/dose_response/`

## Subcategory Dissociation (Format-Control Experiment)
- `src/prepare_subcategory_stimuli.py` — Splits 8 broad categories into 12 format-matched subcategories
  - Addresses criticism: categories might be distinguished by surface features, not cognitive functions
  - Splits: science -> science_factual / science_explanation (both MC, different cognitive demand);
    humanities -> history / philosophy; math -> arithmetic / word_problem;
    language -> procedural / activity_narration; code, reasoning, factual_qa, ethics unchanged
  - Reads stimuli_full.jsonl, outputs stimuli_subcategory.jsonl with category=subcategory
  - 581 samples total, ~50/subcat (31 for science_explanation)
- `scripts/slurm/subcategory_dissociation.sh` — 12-way dissociation on subcategory stimuli across 4 models
  - Uses existing multi_function_dissociation.py (reads categories from stimuli automatically)
  - Output: `/data/user/mzhang630/data/nature_exp/results/subcategory/`
  - SLURM job 306991 submitted 2026-05-20

## Accuracy-Based Dissociation
- `src/accuracy_dissociation.py` — Extends causal dissociation to downstream task ACCURACY
  - Supports 4 answer_types: multiple_choice (log-prob of A/B/C/D), exact_match (generate + extract),
    completion (log-prob of gold completion), generation (mean log-prob of gold answer)
  - Uses separate DISCOVERY stimuli (for attribution) and VALIDATION stimuli (for accuracy)
  - Outputs: 8x8 accuracy drop matrix + 8x8 PPL ratio matrix + baseline accuracies + random control
  - Output per model: `{model_short}_accuracy_dissociation.json`
- `scripts/slurm/accuracy_dissociation.sh` — SLURM array job for 4 models
  - Discovery: `stimuli/stimuli_medium_discovery.jsonl`, Validation: `stimuli/stimuli_medium_validation.jsonl`
  - Output: `/data/user/mzhang630/data/nature_exp/results/accuracy/`
  - Time: ~6-8h per model (generation for exact_match is slower than PPL-only)

## Statistical Validation
- `src/statistical_validation.py` — Formal statistical validation of all dissociation claims
  - No GPU needed; works from pre-computed JSON dissociation results
  - Tests: cross-scale consistency, cross-model universality, matrix-level permutation test,
    per-pair cross-replication (t-test, Wilcoxon, sign test), Cohen's d, bootstrap CIs, BH-FDR
  - Uses 12 independent observations per pair (4 models x 3 scales)
  - Output: `results/statistical_validation.json`
  - Run: `python -m experiments.src.statistical_validation`

## Method Triangulation (Robustness to Attribution Method)
- `src/method_triangulation.py` — Compares 3 attribution methods on one model (Qwen)
  - Method 1: Gradient x Activation (existing): |dL/d_act * act|
  - Method 2: Activation-Only: |act| (no gradient, simplest baseline)
  - Method 3: Gradient-Only: |dL/d_act| (gradient without activation weighting)
  - Per method: importance → selectivity → top-5000 neurons → 8x8 ablation dissociation matrix
  - Cross-method: Jaccard overlap of top-k neurons, Pearson/Spearman correlation of dissociation matrices
  - Output: `{model_short}_method_triangulation.json` + per-method `_attribution.npz`
- `scripts/slurm/method_triangulation.sh` — Single GPU job for Qwen on medium stimuli
  - Output: `/data/user/mzhang630/data/nature_exp/results/method_triangulation/`
  - SLURM job 307966 submitted 2026-05-20

## Configs
- `configs/models.yaml` — Model paths and specs
- `configs/tasks.yaml` — Stimulus categories, dissociation pairs, pruning configs
