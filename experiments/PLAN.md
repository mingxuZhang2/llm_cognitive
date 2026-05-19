# Experiment Plan: Causal Functional Modules in LLMs

## HPC3 Environment
- GPU: NVIDIA H100 80GB HBM3, 8 per node, 41 nodes
- Partition: `acd_u`, 7-day limit
- Conda: `/data/user/mzhang630/miniconda3`
- Data dir: `/data/user/mzhang630/data/nature_exp/`

---

## Phase 0: Environment Setup (~1 day)

Create conda env `funcatlas` with:
- PyTorch 2.3+ (CUDA 12.1)
- transformers, accelerate, datasets
- scikit-learn, scipy, numpy
- SAELens (for SAE-based feature extraction, optional)
- lm-eval-harness (for benchmark evaluation)

Models to download (all on HuggingFace, open-weight):

| Model | Size | VRAM (fp16) | Notes |
|-------|------|-------------|-------|
| Qwen2.5-7B-Instruct | 7B | ~14GB | ULCMOD baseline |
| Meta-Llama-3-8B-Instruct | 8B | ~16GB | Most popular open model |
| Mistral-7B-Instruct-v0.3 | 7B | ~14GB | Different training data |
| google/gemma-2-9b-it | 9B | ~18GB | Different architecture details |
| Pythia suite (70M→12B) | various | varies | For scaling law (has 154 checkpoints) |

All fit on a single H100 80GB for inference. No multi-GPU needed for extraction.

---

## Phase 1: Activation Extraction (~3 days)

### Goal
Extract FFN intermediate activations from all models across a diverse stimulus set.

### Stimulus Set Design
Instead of ULCMOD's 7 categories (too coarse), use **25 fine-grained cognitive task categories**:

**Language & Communication (6)**
1. Grammar correction
2. Translation (en↔zh)
3. Summarization
4. Creative writing (poetry, fiction)
5. Formal/technical writing
6. Dialogue/conversation

**Reasoning & Logic (5)**
7. Mathematical word problems (GSM8K-style)
8. Formal mathematics (MATH-style, proofs)
9. Logical reasoning (syllogisms, puzzles)
10. Commonsense reasoning
11. Causal reasoning

**Code & Technical (4)**
12. Code generation (Python)
13. Code debugging
14. Code explanation
15. Algorithm design

**Knowledge & Facts (4)**
16. Factual Q&A (science)
17. Factual Q&A (history/geography)
18. Current events / news analysis
19. Domain expertise (law, medicine)

**Higher Cognition (4)**
20. Ethical reasoning / moral judgment
21. Analogical reasoning
22. Planning and task decomposition
23. Metacognition / self-reflection

**Specialized (2)**
24. Role-playing / persona adoption
25. Humor / sarcasm understanding

**Per category**: 400 samples → total: 10,000 samples
**Source**: Mix of Infinity-Instruct, MMLU, GSM8K, HumanEval, custom-crafted stimuli

### Extraction Protocol
For each model and each sample:
1. Forward pass with `output_hidden_states=True`
2. For each layer L, extract FFN intermediate activations (after first linear, before activation function)
3. Compute per-neuron activation: mean(|activation|) across tokens
4. Z-score normalize across samples

### Output
Activation matrix A[n_neurons × n_samples] per model
- Qwen2.5-7B: 28 layers × 11,008 neurons = 308,224 neurons × 10,000 samples ≈ 12GB fp32
- Similar scale for other 7B models

### SLURM Config
```
1 node, 1 GPU, ~6h per model
5 models × 6h = 30h total (can parallelize across nodes → 6h wall time)
```

---

## Phase 2: Module Discovery (~2 days)

### Goal
Discover functional modules in each model using dual partitioning.

### Method A: ULCMOD Replication (Primary)
- Implement IterD algorithm (greedy alternating neuron/sample assignment)
- Objective: L(F) = ξ(F) × B(F)
- K values to test: 5, 10, 15, 20, 25
- Initialization: K-Means on PCA-reduced activations

### Method B: SAE Feature Clustering (Secondary, for robustness)
- Train lightweight SAEs on residual stream activations at each layer
- Cluster SAE features by co-activation patterns
- Compare module structure with Method A

### Cross-Model Alignment
After discovering modules independently in each model:
- Use Hungarian algorithm to find optimal module-to-module correspondence across models
- Metric: cosine similarity of module activation profiles on shared stimuli
- Quantify cross-model consistency (adjusted Rand index, normalized mutual information)

### Output
- Module assignments F = {(S_k, U_k)} for each model at each K
- Cross-model alignment matrix
- Module semantic labels (verified by LLM-based labeling + human spot-check)

### SLURM Config
```
1 node, CPU-only (clustering is not GPU-intensive), ~4h per model per K
With 5 models × 5 K values = ~100h CPU (parallelize → ~8h wall time)
```

---

## Phase 3: Double Dissociation (~3 days)

### Goal
Causally validate functional specificity via the neuroscience gold standard.

### Dissociation Pairs

**Pair 1: Math vs Code**
- Module: "Math" module (discovered in Phase 2, expected K≈10-15)
- Module: "Code" module
- Ablation: zero out all neurons in the target module
- Benchmarks:
  - Math: GSM8K (8-shot), MATH (4-shot)
  - Code: HumanEval (0-shot), MBPP (3-shot)
- Prediction: Math ablation → GSM8K/MATH ↓↓, HumanEval/MBPP →

**Pair 2: Factual Knowledge vs Logical Reasoning**
- Module: "Knowledge" module
- Module: "Reasoning" module  
- Benchmarks:
  - Knowledge: TriviaQA, NaturalQuestions
  - Reasoning: LogiQA, ARC-Challenge
- Prediction: Knowledge ablation → TriviaQA ↓↓, LogiQA →

**Pair 3: Language/Humanities vs Math**
- Module: "Language" module
- Module: "Math" module
- Benchmarks:
  - Language: MMLU-humanities, HellaSwag
  - Math: GSM8K, MATH

### Ablation Protocol
For each module ablation:
1. Identify all neurons assigned to the target module
2. Create a hook that zeros out these neurons' activations during forward pass
3. Run full benchmark suite
4. Record: (a) target benchmark scores, (b) non-target benchmark scores, (c) general perplexity

### Statistical Framework
- Use Crawford & Garthwaite (2005) single-case dissociation test
- Report effect sizes (Cohen's d) for each ablation on each benchmark
- Significance threshold: p < 0.01, Bonferroni-corrected

### Control Conditions
1. **Random ablation** (same # neurons, random selection): expect smaller, non-specific deficits
2. **Size-matched ablation** (same # neurons from non-target module): expect different deficit pattern
3. **Graduated ablation** (25%, 50%, 75%, 100% of module): expect dose-response curve

### Output
- Causal functional mapping matrix: (module × benchmark) effect sizes
- Double dissociation plots (the "money figure")
- Dose-response curves

### SLURM Config
```
3 pairs × 2 directions × 4 control conditions = 24 ablation experiments per model
Each experiment: 1 GPU, ~2h for full benchmark suite
5 models × 24 experiments = 120 runs → parallelize across 8 GPUs → ~30h wall time
```

---

## Phase 4: Atlas-Guided Pruning (~3 days)

### Goal
Demonstrate practical AI value: atlas knowledge enables better task-specific model compression.

### Experimental Design

**Scenario**: "I need a model that's great at math but don't care about other capabilities"

**Our method: Atlas-Guided Pruning (AGP)**
1. From Phase 2, identify which neurons belong to the "Math" module
2. Assign importance scores: neurons in math module get score × 2 (protect), others get base score
3. Prune using magnitude + atlas-guided score, at sparsity levels: 20%, 30%, 40%, 50%, 60%, 70%
4. Evaluate on GSM8K, MATH (target) and HumanEval, MMLU, HellaSwag (non-target)

**Baselines**:
1. **Wanda** (Sun et al., ICML 2023): pruning by weight × activation magnitude
2. **SparseGPT** (Frantar & Alistarh, ICML 2023): one-shot weight reconstruction
3. **Magnitude Pruning**: prune by absolute weight value
4. **Random Pruning**: random unstructured pruning

**Repeat for 3 scenarios**:
- Preserve math → prune rest
- Preserve code → prune rest
- Preserve language → prune rest

### Metrics
- Primary: target task accuracy at each sparsity level
- Secondary: non-target task accuracy (collateral damage)
- Efficiency: memory footprint, inference speed

### Output
- Sparsity vs accuracy curves for all methods
- Pareto front showing atlas-guided dominance at high sparsity
- Collateral damage comparison

### SLURM Config
```
3 scenarios × 6 sparsity levels × 5 methods = 90 configs per model
2 models (LLaMA-8B, Qwen-7B) × 90 = 180 runs
Each: 1 GPU, ~1h → 180 GPU-hours → 8 GPUs parallel → ~23h wall time
```

---

## Phase 5: Modularity Scaling Law (~2 days)

### Goal
Show that functional modularity correlates with model capability across scales.

### Models
Use Pythia suite (all checkpoints publicly available):
- Pythia-70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B

### Protocol
For each model:
1. Extract activations on standard stimulus set (subset of 2,000 samples)
2. Run module discovery at K=10
3. Compute modularity Q-score: Q = (within-module activation) / (total activation) - expected
4. Run quick benchmark suite: MMLU (5-shot), GSM8K (8-shot), HellaSwag (10-shot)

### Analysis
- Plot Q-score vs model size (log scale)
- Plot Q-score vs benchmark performance
- Fit: Q ~ α × log(params) + β (test if log-linear)
- Correlation analysis with confidence intervals

### SLURM Config
```
8 models, smallest ones on CPU, larger ones on 1 GPU each
~1h per model → 8h total, parallelize → 2h wall time
```

---

## Phase 6: Brain Network Comparison (~1 day)

### Goal
Quantitative graph-theoretic comparison with fMRI brain network data.

### Brain Data (Public)
- Human Connectome Project (HCP): functional connectivity matrices
- OR: use published modularity Q-scores from Sporns et al.

### LLM Network Construction
From Phase 2 module assignments:
1. Build module co-activation graph: nodes = modules, edges = shared activation patterns
2. Compute: modularity Q-score, clustering coefficient, path length, small-worldness σ
3. Compute: participation coefficient per module (hub-ness)

### Comparison
- Compare LLM graph metrics to published human brain values
- Use permutation test: is LLM Q-score significantly different from random networks?
- Report: which LLM graph properties match brain, which diverge

### SLURM Config
```
CPU-only, ~2h total
```

---

## Timeline Summary

| Phase | What | Duration | GPU-hours | Dependencies |
|-------|------|----------|-----------|-------------|
| 0 | Setup | 1 day | 0 | - |
| 1 | Activation extraction | 3 days | 30 | Phase 0 |
| 2 | Module discovery | 2 days | ~10 | Phase 1 |
| 3 | Double dissociation | 3 days | 240 | Phase 2 |
| 4 | Atlas-guided pruning | 3 days | 180 | Phase 2 |
| 5 | Modularity scaling | 2 days | 16 | Phase 0 |
| 6 | Brain comparison | 1 day | 0 | Phase 2 |
| **Total** | | **~2-3 weeks compute** | **~476 H100-hours** | |

Phases 3, 4, 5 can run in parallel after Phase 2 completes.
With 8 GPUs available, effective wall time ≈ **10-12 days** for all experiments.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Double dissociation fails (no clean separation) | Report graded dissociation with effect sizes; may indicate polysemanticity |
| Modules differ across models | Report both universal and model-specific modules; diversity itself is interesting |
| Atlas-guided pruning doesn't beat baselines | Try hybrid: atlas weights + Wanda/SparseGPT scores |
| Modularity doesn't correlate with capability | Report as negative result; try alternative modularity metrics |
| Activation extraction OOMs on 12B | Use gradient checkpointing or batch processing |

---

## Key Design Decisions

1. **FFN neurons (ULCMOD) vs SAE features**: Start with FFN neurons (direct replication + extension), add SAE as robustness check
2. **Instruct vs Base models**: Use Instruct models (following ULCMOD), but add one base model comparison to test if instruction tuning creates vs sharpens modules
3. **K selection**: Test multiple K values, report the K that maximizes downstream classification accuracy (I(F) metric)
4. **Stimulus set**: Use 25 categories × 400 samples = 10,000, much larger and more diverse than ULCMOD's 7 × 1,200 = 8,400
