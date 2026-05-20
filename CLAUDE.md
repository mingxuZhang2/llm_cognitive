# Nature Paper Project: LLM Functional Atlas

## Project Goal
Build a causal functional atlas of Large Language Models — mapping which neurons implement which cognitive functions, validated by double dissociation, with predictive power for controlling model behavior. Targeting Nature Machine Intelligence.

## Repository Structure
- `IDEA_SYNTHESIS.md` -- Original paper idea synthesis
- `experiments/` -- All experimental code and results
  - `src/` -- Python scripts for each experiment phase
  - `scripts/slurm/` -- SLURM job scripts for HPC3
  - `results/` -- JSON results from all experiments
  - `figures/` -- Generated visualizations
  - `analysis_findings.md` -- Comprehensive findings document (8 key findings)
- Research docs: `research_paper_analysis.md`, `research_neuroscience_methods.md`, etc.
- Feasibility studies: `feasibility_*.md`

## Current Experimental Status (May 20, 2026)
- Branch: experiments

### Completed Experiments
1. **8-way Functional Dissociation** (pilot: 15/cat, scaled: 50/cat)
   - 4 models × 8 categories × causal attribution (grad × act)
   - Pilot: 27-28/28 pairwise dissociation
   - Scaled (50/cat): ALL 8/8 specificity, 28/28 pairwise (3 models confirmed, Gemma pending)
   - Key: only ~1% of neurons (5000) needed per function

2. **Predictive Experiments** (pilot)
   - Pathway decomposition: 12/12 predictions correct (math-only, reasoning-only, shared)
   - Atlas-guided steering: dependency DAG confirmed (4/4 code→reasoning = no effect)
   - Instance-level prediction: 11/12 correct

3. **Atlas-Guided Pruning** (pilot)
   - 50% sparsity: atlas 1.11-3.24x vs random 3-3948x vs magnitude 13-11898x
   - No phase transition for atlas method (linear degradation)

4. **Structural Analysis** (pilot attribution data)
   - Hub neurons extremely rare (<50 across 500K+ neurons)
   - Science-humanities uniquely bidirectional coupling (layer-colocalized)
   - Reasoning is emergent coalition (3x weaker self-effect than math)

### Pending Experiments
- **Job 306948**: Balanced multi-dissociation (152/cat, 1216 total) — 4 models
- **Job 306949**: Scaled predictive experiments (50/cat) — 4 models

7. **Accuracy-Based Dissociation** (code ready, awaiting stimuli)
   - `src/accuracy_dissociation.py` — Downstream task accuracy under causal ablation
   - Supports 4 answer_types: multiple_choice, exact_match, completion, generation
   - Uses separate discovery (attribution) and validation (accuracy) stimuli sets
   - Outputs: 8x8 accuracy drop matrix + PPL matrix + baselines + random control
   - SLURM: `scripts/slurm/accuracy_dissociation.sh` (array 0-3, 4 models)

5. **Convergence Analysis** (local, no GPU)
   - `src/convergence_analysis.py` — Functional Convergence Index across 4 architectures
   - 5 metrics: dissociation matrix similarity, DAG edge consistency, layer profile similarity, hierarchy consistency, overall index with null model
   - Results: `results/convergence_analysis.json`
   - **Convergence Index = 0.86** (z=5.64 vs null, p<0.001) — strong cross-architecture convergence
   - Math most conserved layer profile (cos=0.94), factual_qa most variable (cos=0.67)
   - 9 universal spillover edges, ethics is universal hub
   - Mistral and Gemma have identical modularity rankings (rho=1.0)

6. **Statistical Validation** (local, no GPU)
   - `src/statistical_validation.py` — Formal tests on all 28 pairwise dissociations
   - Matrix-level permutation test (10K perms): all 12 (4 models x 3 scales) p < 0.0002
   - Per-pair one-sample t-test (12 obs = 4 models x 3 scales): ALL 28/28 p < 1.1e-5
   - After BH-FDR correction: ALL 28/28 remain significant (p_adj < 0.05)
   - Cohen's d: matrix-level mean=5.97; per-pair mean=3.73, min=2.02 (all "very large")
   - Bootstrap 95% CIs: all 28 pairs above zero
   - Cross-model universality: 28/28 pairs positive in all 4 models
   - Cross-scale consistency: 27-28/28 pairs positive at all 3 scales per model
   - Results: `results/statistical_validation.json`

### Key Findings (from analysis_findings.md)
1. **Universal 3-layer functional hierarchy**: language/code → math/science → reasoning/ethics
2. **Science-humanities knowledge integration zone**: bidirectional, near-symmetric, layer-colocalized
3. **Reasoning as emergent coalition**: weakest self-effect, depends on math+science+language
4. **Atlas pruning eliminates phase transition**: never >10x degradation at 50% sparsity
5. **Steering confirms DAG non-linearly**: 10-20x damage escalation from mild to strong amplification
6. **Shared hub neurons catastrophically important**: <50 neurons → model collapse when ablated
7. **Competitive inhibition**: code suppresses ethics, ethics suppresses humanities
8. **Modularity gradient**: formal (math) → distributed (reasoning), parallels brain cortex

## HPC3 Configuration
- SSH: `ssh -i /hpc2hdd/home/mzhang630/data/id_rsa -o StrictHostKeyChecking=no mzhang630@hpc3login.hpc.hkust-gz.edu.cn`
- Base dir: `/data/user/mzhang630/data/nature_exp`
- Conda env: `alphasteer`
- SLURM: partition=acd_u, account=d_yings_team
- Models: Qwen2.5-7B, LLaMA-3.1-8B, Mistral-7B, Gemma-2-9b (snapshot paths in SLURM scripts)

## Stimuli
- `stimuli_pilot.jsonl` — 120 samples (15/category, balanced)
- `stimuli_medium.jsonl` — 400 samples (50/category, balanced)
- `stimuli_balanced.jsonl` — 1216 samples (152/category, balanced)
- `stimuli_full.jsonl` — 3133 samples (unbalanced, original)
- 8 categories: math, code, reasoning, language, science, ethics, factual_qa, humanities
