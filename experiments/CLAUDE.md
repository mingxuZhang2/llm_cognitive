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

## Configs
- `configs/models.yaml` — Model paths and specs
- `configs/tasks.yaml` — Stimulus categories, dissociation pairs, pruning configs
