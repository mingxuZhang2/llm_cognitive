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

## Configs
- `configs/models.yaml` — Model paths and specs
- `configs/tasks.yaml` — Stimulus categories, dissociation pairs, pruning configs
