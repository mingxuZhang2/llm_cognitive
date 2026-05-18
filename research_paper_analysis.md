# Deep Analysis: "Discovering Decoupled Functional Modules in Large Language Models"

**Prepared for Dr. Zhang | May 18, 2026**

---

## Bibliographic Record

| Field | Value |
|---|---|
| Title | Discovering Decoupled Functional Modules in Large Language Models |
| Authors | Yanke Yu, Jin Li, Ying Sun (HKUST-GZ); Ping Li, Zhefeng Wang, Yi Zheng (Huawei Technologies Ltd.) |
| Venue | AAAI-26 — Technical Track on NLP VI — **Oral Presentation** |
| Published | March 14, 2026 |
| Pages | 34503–34511, Vol. 40, No. 41 |
| DOI | https://doi.org/10.1609/aaai.v40i41.40749 |
| arXiv | 2603.17823 (submitted March 18, 2026) |
| Code | https://github.com/rank-Yu/llm-modules |

---

## 1. CORE CONTRIBUTION: What Did They Discover?

### The Central Claim

This paper makes one foundational claim: **large language models spontaneously develop decoupled, interpretable functional modules in their feedforward network (FFN) neurons**, and these modules can be discovered systematically without supervision.

The word "decoupled" is load-bearing. Prior interpretability work (knowledge neurons, skill neurons, circuit analysis) found neurons that contribute to specific tasks, but these neurons overlapped heavily across tasks — they were shared, entangled, and polysemantic. This paper argues instead that LLMs secretly contain *mutually exclusive* functional regions: neurons assigned primarily to one function, samples assigned primarily to one function, and these two assignments align into coherent "modules."

### What the "Decoupled Functional Modules" Actually Are

A functional module is a pair `(S_k, U_k)` where:
- `U_k` is a set of neurons (FFN intermediate activations) across all layers of the model
- `S_k` is a set of input samples that preferentially activate exactly those neurons
- The relationship is **exclusive**: each neuron belongs to exactly one module, each sample belongs to exactly one module
- The relationship is **coherent**: neurons in `U_k` activate strongly for samples in `S_k` and weakly for all others

This is a **dual partitioning** of both the neuron space and the sample space simultaneously. It is not neuron-level attribution — it is a global organizational structure of the model.

### The Discovered Module Semantics (Key Finding)

Across K=5 to K=20 module configurations, every discovered module maps to a recognizable cognitive function:
- Algorithmic Programming
- Mathematics
- Creative/Formal/Professional Writing
- Information Retrieval
- Translation & Linguistic processing
- Knowledge / Science
- Ethical Reasoning
- Role-Playing

This emergence is **not supervised**. The framework has no knowledge of these categories. The categories are confirmed post-hoc by feeding representative samples from each discovered module into an LLM for semantic labeling.

---

## 2. METHODOLOGY: How Did They Find These Modules?

### Step 1: Activation Extraction

For each input sample `s_j` passed through model layers, the activation of FFN neuron `u_i` is computed as:

```
A_{i,j} = (1 / T_j) * sum_t |a_{i,j,t}|
```

where `T_j` is the number of tokens in sample `j` and `a_{i,j,t}` is the raw activation at token `t`. This collapses the token dimension by taking the mean absolute value. Critically, z-score normalization is applied across samples to ensure neurons with different scales are comparable.

The result is an **activation matrix A** of shape [n_neurons x n_samples], where each entry measures how strongly a given neuron responds to a given sample.

**Implementation detail**: Only FFN intermediate activations are used, not attention heads, embeddings, or output logits. For Qwen2.5-7B with 28 layers and thousands of FFN neurons per layer, this matrix is very large, which is why the pipeline script `p2_get_activation.py` is a separate expensive step.

### Step 2: Problem Formulation — Neuron-Sample Dual Partitioning

The core innovation is framing this as a *joint* clustering problem over both rows (neurons) and columns (samples) of `A`. Given K desired modules, find:

```
F = {(S_1, U_1), ..., (S_K, U_K)}
```

subject to:
- **Completeness**: union of all S_k = S (all samples covered), union of all U_k = U (all neurons covered)
- **Exclusivity**: no sample appears in two modules, no neuron appears in two modules
- **Non-Emptiness**: every module has at least one sample and one neuron

### Step 3: The Objective Function L(F)

The composite objective to maximize is:

```
L(F) = xi(F) * B(F)
```

**Activation Modularity xi(F)**:
```
xi(F) = [sum_k sum_{u_i in U_k, s_j in S_k} A_{i,j}] / [sum_k |U_k| * |S_k|]
```
This is the mean activation restricted to within-module pairs. Maximizing this forces neurons and samples in the same module to strongly co-activate.

**Balance Score B(F)**:
```
B(F) = K / sum_k (1 / (|U_k| * |S_k|))
```
This is the harmonic mean of module "sizes" (product of neuron count and sample count). Maximizing this prevents the degenerate solution of one giant module and K-1 singleton modules.

The product `L(F) = xi(F) * B(F)` combines compactness (high co-activation within modules) with coverage (roughly balanced module sizes).

### Step 4: The IterD Algorithm

The IterD (Iterative Decoupling) algorithm solves this NP-hard combinatorial problem greedily:

**Initialization**: Run standard K-Means on PCA-reduced activation vectors to get starting partitions `F_0`.

**Repeat until convergence**:

1. **Neuron Assignment Step** (samples fixed):
   For each neuron `u`, compute the change in `L(F)` if `u` is moved from its current module to module `k`, for all k. Immediately reassign `u` to the `argmax_k` module. This "greedy immediate reassignment" (not batch update) is important — it means each subsequent neuron reassignment already benefits from previous reassignments.

2. **Sample Assignment Step** (neurons fixed):
   Same procedure: for each sample `s`, immediately reassign to the module maximizing `L(F)`.

**Convergence**: when a full pass through all neurons and all samples produces zero reassignments.

The computational cost per iteration is O(|U| * K + |S| * K) per pass, which is tractable. The paper runs this through `p4_iter.py`.

### Step 5: Evaluation

Two evaluation protocols:

**Intrinsic (L(F), xi(F), B(F))**: Direct measurement of the objective function value on discovered modules. IterD is compared against K-Means, Mini-Batch K-Means, Agglomerative Clustering, Spectral Clustering, and Spectral Co-clustering.

**Extrinsic — Function Module Informativeness I(F)**:
For each sample `s_j`, build a K-dimensional feature vector:
```
x_{s_j, k} = (1 / |U_k|) * sum_{u_i in U_k} A_{i,j}
```
(how strongly sample j activates each module k). Train a classifier (Logistic Regression or SVM) on `{x_{s_j}, y_j}` where `y_j` is the ground-truth functional category. Higher classification accuracy means the discovered modules encode functionally meaningful information.

---

## 3. KEY FINDINGS (Quantitative and Qualitative)

### Quantitative Results

**Table 1: Objective Function L(F) — IterD vs. Baselines**

| Model | K | IterD L(F) | K-Means L(F) | Best Baseline L(F) |
|---|---|---|---|---|
| Qwen2.5-1.5B | 5 | **31.4** | ~14.2 | ~14.2 |
| Qwen2.5-3B | 5 | **56.9** | ~26.1 | ~26.1 |
| Qwen2.5-7B | 5 | **64.6** | **29.6** | 29.6 |
| Qwen2.5-7B | 20 | **9.0** | ~4.1 | ~4.1 |

IterD achieves roughly **2x the objective score** of the best competing method across all configurations. Higher xi (activation concentration within modules) and high B (balanced sizes) are both achieved simultaneously.

**Table 2: Downstream Classification (I(F)) — Accuracy / F1**

| Model | K | Classifier | IterD Acc | IterD F1 |
|---|---|---|---|---|
| Qwen2.5-1.5B | 10 | LogReg | **0.7257** | **0.7259** |
| Qwen2.5-1.5B | 20 | LogReg | **0.8200** | **0.8200** |
| Qwen2.5-7B | 20 | SVC | **0.8214** | **0.8210** |

Modules discovered by IterD provide feature representations that classify functional categories with ~82% accuracy using only a linear classifier — substantially above all baselines.

**Semantic Disentanglement Scores**:
The cosine similarity between module activation vectors for different functional categories is very low (near 0), except for semantically related pairs. Notably, "Linguistic" and "Translation" categories show similarity 0.168, reflecting genuine semantic overlap, while "Math" and "Creative Writing" show near-zero similarity.

### Qualitative Findings (The Most Surprising Results)

**Finding 1: Function Hierarchy (most novel)**
Increasing K doesn't just add more modules — it *refines* existing ones in a hierarchically consistent way:
- K=10: One module labeled "Algorithmic Programming"
- K=15: Splits into "Algorithmic Programming" + "Software Architecture"
- K=20: Further splits into "Code Analysis & Debugging" + "Software & System Programming" + "Algorithmic Design"

Similarly, "Writing" at K=10 becomes "Creative Writing," "Formal Writing," and "Professional Writing" at K=20. This hierarchical consistency was **not enforced** by the algorithm — it emerges naturally.

**Finding 2: Spatial Arrangement (striking)**
When the discovered modules are plotted by their average layer depth (which layers their neurons concentrate in), a clear spatial geography emerges:
- "Information Retrieval" neurons peak in **layers 1-2** (near the input)
- "Mathematics" and "Algorithmic Programming" neurons concentrate in **mid-to-late layers (15-32)**
- "Writing" and "Translation" neurons are distributed across **middle layers**
- "Non-English processing" (translation) concentrates in **early AND late layers** with a gap in the middle
- "Role-Playing" notably peaks at **layer 26** specifically

This spatial structure implies the model computes a functional pipeline: surface/retrieval first, complex reasoning last.

**Finding 3: Central Cognitive Hubs**
Complex cross-domain functions (Translation, Linguistics) physically reside at the intersection point between other module clusters in the layer-space visualization. This echoes the neuroscience concept of "connector hubs" — brain regions that bridge specialized networks.

**Finding 4: Semantic Locality**
Semantically related functions (Programming, Mathematics, Science) cluster together in the model's architecture. Their neurons share overlapping layer ranges and their module activation vectors show higher cosine similarity to each other than to dissimilar functions.

**Finding 5: Function Comprehensiveness**
No matter what K is chosen, the full range of 7 ground-truth functional categories is recoverable from the discovered modules. Even at K=5, each discovered module maps to one (or a merged pair) of the ground truth functions. No important function is "lost."

---

## 4. NEUROSCIENCE PARALLELS

### The Core Analogy

The paper explicitly draws its motivation from the neuroscience of functional brain regions. The abstract states: "Inspired by the human brain that contains highly specialized and decoupled function modules."

Two foundational neuroscience papers are cited:
1. **Bullmore & Sporns (2009)** — Brain Graphs: complex brain networks and their structural properties. Established that the brain has small-world topology with modular clusters.
2. **Meunier et al. (2010)** — Hierarchical modularity in human cortical networks. Demonstrated that brain modules have nested hierarchical structure.

### Specific Analogies Made

| Brain Observation | LLM Analogy in Paper |
|---|---|
| Broca's/Wernicke's areas handle language; visual cortex handles vision — specialized, minimally overlapping | Discovered modules handle Coding vs. Math vs. Writing with minimal cross-activation |
| Prefrontal cortex (late-stage integration) handles abstract reasoning | Complex reasoning (Math, Programming) localizes to late layers |
| Primary sensory cortices (early processing) handle low-level perception | Information Retrieval activates in early layers |
| Connector hubs in default mode network bridge specialized regions | Linguistics/Translation modules sit at the spatial intersection of other modules |
| Hierarchical cortical organization (V1 → V4 → IT cortex in vision) | Function hierarchy: coarse modules at low K refine into specializations at high K |
| Different brain regions have distinct functional profiles even across similar stimuli | Different modules show distinct layer distributions even for related tasks |

### What the Paper Does NOT Claim

The paper is careful not to claim the LLM *is* a brain. It uses the brain analogy as motivation and as a qualitative interpretive frame, but does not attempt to map specific LLM layers to specific brain regions in a quantitative way.

---

## 5. LIMITATIONS AND GAPS

### Explicitly Acknowledged Limitations

1. **Hard/exclusive assignment constraint**: Each neuron and sample belongs to exactly ONE module. The paper acknowledges this "simplifies the problem but potentially misses multi-functional aspects." In reality, a neuron may contribute to multiple functions (polysemanticity). The hard assignment forces a choice.

2. **FFN-only scope**: Activation analysis is restricted to FFN intermediate activations. Attention heads, embeddings, layer norms, and residual streams are not analyzed. This is a significant gap since circuits research (Elhage et al. 2021, Conmy et al. 2023) has established that attention heads are often the primary carriers of relational reasoning.

3. **Cross-model generalization not explored**: All experiments are on Qwen2.5 (1.5B, 3B, 7B). Whether the discovered module structure is universal (also present in LLaMA, Mistral, GPT-4o architectures) is untested.

4. **Modules not causally validated**: The paper demonstrates that discovered modules are statistically coherent and informative for classification, but does not perform causal ablation experiments. If you zero out neurons in the "Mathematics" module, does math performance specifically degrade while coding performance is preserved? This causal test is absent.

### Implicit Gaps (Not Discussed)

5. **No dynamics analysis**: The module structure is computed on a fixed trained model. How do modules form during training? Are they present from early training or do they emerge late? This is a natural follow-up.

6. **Instruction tuning vs. pretraining**: All tested models are Instruct-finetuned (Qwen2.5-*-Instruct). Does the module structure differ for base models? Does instruction finetuning *create* the modules or merely sharpen pre-existing ones?

7. **Dataset dependency**: The Infinity-Instruct dataset was used with exactly 7 functional categories. If different or more categories were used, would different modules emerge? The module structure might be partially an artifact of the training data distribution.

8. **Attention mechanism**: No analysis of whether attention heads also organize into functional modules, or how the discovered FFN modules interact with specific attention patterns.

9. **No module-based intervention experiments**: If the model is modified to strengthen module separation (e.g., through auxiliary training losses), does interpretability improve or task performance improve?

10. **Scalability to large models**: Experiments stop at 7B parameters. The neuron count and layer depth at 70B or 405B scale may make IterD computationally prohibitive without approximations.

11. **No overlap/soft assignment variant**: The paper mentions soft assignment as future work but provides no experiments with it.

12. **Biological fidelity of analogy**: The paper cites brain modularity as inspiration but never quantitatively validates how similar the discovered LLM structure actually is to brain organization metrics (e.g., modularity Q-score, participation coefficient, within-module degree).

---

## 6. REPRODUCIBILITY

### Models Used
- Qwen2.5-1.5B-Instruct (HuggingFace)
- Qwen2.5-3B-Instruct (HuggingFace)
- Qwen2.5-7B-Instruct (HuggingFace)

All publicly available and runnable on 1-2 A800 GPUs (80GB VRAM each). The 7B model fits on a single A800 for inference.

### Dataset
- **Infinity-Instruct** (Li et al. 2025): 8,400 samples selected from a large instruction-following dataset
- 1,200 samples per category x 7 categories: Coding, Mathematics, Linguistic, Knowledge, Translation, Ethical, Writing
- 7,000 training / 1,400 test split
- Category assignments used as ground truth labels for the downstream evaluation only (not for module discovery)

### Pipeline (6 Python scripts, run sequentially via `bash run.sh`)
1. `p1_generate_dataset.py` — Sample selection from Infinity-Instruct
2. `p2_get_activation.py` — Forward pass, extract FFN intermediate activations, z-score normalize
3. `p3_clustering.py` — Baseline comparisons (K-Means etc. on PCA-reduced activations)
4. `p4_iter.py` — Run IterD algorithm (main contribution)
5. `p5_result_objective.py` — Compute L(F), xi(F), B(F) metrics
6. `p6_result_prediction.py` — Train classifier on module features, report accuracy/F1

### Software Dependencies
- Python 3.10+
- PyTorch (for model inference)
- HuggingFace Transformers
- scikit-learn (for baseline clustering and classifiers)
- Standard numerical stack (numpy, scipy)

### Reproducibility Assessment
- Code is available (9 stars, 2 forks, 5 commits — very new, minimally adopted as of May 2026)
- Configuration via `.env.example` (user sets `LLM_PATH_BASE`)
- No random seed documentation found
- No GPU memory requirements documented
- No runtime estimates provided
- Hyperparameter K is user-specified; no guidance on selection

**Verdict**: Nominally reproducible with modest effort. The main bottleneck is storing FFN activations for 8,400 samples across all layers — for Qwen2.5-7B this is roughly 8400 x 28_layers x ~11264_neurons = ~2.6 billion float32 values (~10GB). Manageable on Dr. Zhang's A800 system.

---

## 7. CITATION CONTEXT AND RELATED WORK LANDSCAPE

### Direct Citations (Papers the Authors Build Upon)

**Foundational Neuron Interpretability:**
- Dai et al. (2022), ACL — Knowledge Neurons in Pretrained Transformers: introduced the idea that specific FFN neurons store factual knowledge, identifiable via gradient attribution. ULCMOD's insight that FFN neurons have functional specialization directly extends this.
- Wang et al. (2022), EMNLP — Finding Skill Neurons in Pre-trained Transformers: showed that after prompt tuning, a small set of neurons become highly predictive of task labels. ULCMOD differs by not requiring any task-specific fine-tuning.
- Voita et al. (2023) — Analyzing linguistic features encoded in individual neurons.

**Circuit/Mechanistic Interpretability:**
- Elhage et al. (2021), Anthropic — A Mathematical Framework for Transformer Circuits: established the circuit paradigm for understanding attention heads and MLPs as composable algorithms. ULCMOD takes a different (population-level clustering) approach.
- Conmy et al. (2023) — Towards Automated Circuit Discovery for Mechanistic Interpretability: automated identification of task-relevant circuits.

**Sparse Autoencoders:**
- Bricken et al. (2023), Anthropic — Towards Monosemanticity: used SAEs to find monosemantic features in MLP layers. ULCMOD is unsupervised and module-level rather than feature-level.
- Templeton et al. (2024), Anthropic — Scaling Monosemanticity (Claude Sonnet): large-scale SAE analysis. SAEs find distributed features; ULCMOD finds discrete population-level modules.

**Knowledge Localization:**
- Meng et al. (2022) — ROME/MEMIT: locating factual associations to specific MLP layers, and editing them. ULCMOD's layer distribution findings (different functions in different layers) relate to this line.

**Neuroscience:**
- Bullmore & Sporns (2009), Nature Reviews Neuroscience — Complex Brain Networks
- Meunier et al. (2010) — Hierarchical modularity in human cortical networks

**Modularity in LLMs:**
- Emergent Modularity (arXiv:2310.10908, 2023) — found implicit modularity in pretrained transformers via neuron co-activation patterns, proposed EMoE. Key predecessor.

### Closely Related Concurrent/Follow-Up Work

**Brain-Inspired Exploration of Functional Networks (arXiv:2502.20408, Liu et al., 2025)**
Uses Independent Component Analysis (ICA) on FFN outputs to find functional networks. Key difference: ICA allows overlapping (soft) assignments; ULCMOD enforces hard partitioning. ICA requires no predefined K. Their finding that ~10% of neurons sustain performance complements ULCMOD's discovery that these neurons cluster into identifiable functional modules.

**Unraveling Cognitive Patterns through Module Communities (arXiv:2508.18192, 2025)**
Uses Louvain community detection on a tripartite network (skills, datasets, LLM modules). Finding: community structure does NOT align with predefined cognitive categories — contradicting ULCMOD's finding that modules do align. The discrepancy may be methodological (network-level vs. neuron-level analysis) or may reflect that soft communities differ fundamentally from hard neuron partitions.

**Unlocking Emergent Modularity (arXiv:2310.10908, 2023)**
Demonstrated that FFN neurons form implicit modular clusters through co-activation. Directly cited and is the closest predecessor. ULCMOD extends this by: (1) joint partitioning of both neurons and samples, (2) a principled objective function, (3) explicit discovery of the sample-level functional categories.

**LLM Language Network (arXiv:2411.02280, 2024)**
Neuroscientific approach using fMRI-inspired methods to identify "language-selective" units in LLMs. Finds that a small subset of units is causally relevant for language tasks. Complements ULCMOD but focuses on language vs. non-language rather than within-language function specialization.

**On the Analogy between Human Brain and LLMs (arXiv:2511.06519)**
Finds ~5,228 neurons (1.1% of total) encode part-of-speech information in Llama 3 via gradient attribution and SVM validation. Different scope (syntactic not functional), but confirms sparse specialization in FFN layers.

**Modular Reasoning with Brain-Like Specialization (arXiv:2506.13331, 2025)**
Architectural approach: designs MoE systems with explicit brain-like specialization. Evaluates whether induced specialization improves reasoning. Provides the missing causal link that ULCMOD lacks — if you enforce modules, performance changes in predictable ways.

### Who Would Cite This Paper (Predicted Citation Trajectory)

Given it was published March 2026 with an Oral designation at AAAI-26, it is likely to accumulate citations from:
1. LLM interpretability papers needing a "module discovery" baseline
2. LLM pruning/compression papers (functional modules naturally define which neurons to prune together)
3. Knowledge editing papers (knowing which module owns a fact enables targeted edits)
4. MoE architecture papers (functional modules inform expert routing design)
5. Continual learning papers (training on new tasks might modify specific modules)

---

## 8. SYNTHESIS: WHERE THIS PAPER SITS IN THE FIELD

### The Research Landscape It Inhabits

The paper occupies a specific niche at the intersection of three research traditions:

1. **Mechanistic Interpretability** (Anthropic, DeepMind): works at the level of features, circuits, attention heads. Methods: activation patching, causal tracing, SAEs. Granularity: fine. Does not attempt global organizational structure.

2. **Neuroscience-Inspired LLM Analysis**: ICA, fMRI analogies, network science. Methods: community detection, correlation analysis. Granularity: coarse. Does not typically achieve interpretable module labeling.

3. **Model Modularity / MoE**: MoE architecture, emergent modularity. Methods: expert routing, clustering. Granularity: architectural. Usually requires modification of training.

ULCMOD is unique in combining: (a) neuron-level granularity, (b) unsupervised discovery on pretrained models (no modification), (c) dual partitioning of both neurons and samples, (d) explicit semantic labeling of discovered modules, (e) hierarchical consistency across K values.

### What Makes This Paper Nature-Worthy (or Not)

**Strengths relative to Nature standards:**
- The finding of spontaneous, decoupled, semantically coherent functional modules in a trained neural network is genuinely surprising and conceptually significant
- The analogy to brain functional organization is meaningful and the hierarchical structure finding is elegant
- The discovery is clean enough to be visualized compellingly (block-diagonal co-activation heatmaps)

**Weaknesses relative to Nature standards:**
- Three models, all from one model family (Qwen2.5), severely limits generalizability claims
- No causal validation: discovering correlation between neuron groups and function is not the same as proving those neurons cause that function
- The method is essentially co-clustering with a custom objective function — the methodological novelty, while solid, is not transformative
- The paper does not connect its findings to any practical application (compression, editing, safety) — it remains purely descriptive

---

## 9. POTENTIAL EXTENSIONS (FOR DR. ZHANG'S NATURE-LEVEL PAPER)

Based on the gaps identified above, the following extensions would substantially increase impact:

### Extension 1: Causal Validation via Lesioning (High Priority)
Ablate neurons in specific discovered modules and measure function-specific performance degradation. Hypothesis: ablating the "Mathematics" module degrades math benchmarks (GSM8K, MATH) significantly more than coding benchmarks (HumanEval), and vice versa. This would provide causal evidence that the modules are not just correlational clusters.

### Extension 2: Cross-Architecture Generalization (Critical for Nature)
Replicate the analysis on LLaMA-3 (8B, 70B), Mistral, Gemma, and ideally a GPT-4o-class model (through API activation extraction if possible). If the same functional module structure appears across architectures and scales, this becomes a universal law of LLM organization, not a Qwen-specific artifact.

### Extension 3: Module Formation During Training
Track module formation across training checkpoints. At what point do modules emerge? Are they present after pretraining, or only after instruction finetuning? This would answer whether modularity is a product of supervised instruction data or of self-supervised pretraining.

### Extension 4: Soft/Overlapping Assignment
Extend the hard partitioning to soft assignments (each neuron has a distribution over modules, each sample has a distribution over modules). This would better capture polysemantic neurons and would likely connect to SAE feature decompositions.

### Extension 5: Functional Module-Guided Applications
Demonstrate that knowing the module structure enables: (a) targeted pruning that preserves specific capabilities, (b) targeted knowledge editing that does not corrupt adjacent functions, (c) capability isolation for safety (identifying which module governs harmful reasoning).

### Extension 6: Attention Head Module Discovery
Apply the same dual-partitioning framework to attention head activations. Attention heads and FFN neurons may form complementary or overlapping functional modules, and the full picture requires both.

### Extension 7: Quantitative Brain Comparison
Use established graph-theoretic measures (modularity Q-score, participation coefficient, small-worldness) to quantitatively compare LLM module structure to human brain network structure from fMRI data. This would rigorously test whether the brain analogy holds beyond visual similarity.

---

## 10. QUICK REFERENCE SUMMARY

| Aspect | Detail |
|---|---|
| Core idea | Joint clustering of LLM neurons + input samples into mutually exclusive functional modules |
| Algorithm | IterD (greedy iterative reassignment alternating neurons and samples) |
| Objective | L(F) = Activation Modularity x Balance Score |
| Models | Qwen2.5-1.5B/3B/7B-Instruct |
| Dataset | 8,400 samples from Infinity-Instruct (7 functional categories) |
| Main metric | L(F) score + downstream classification accuracy I(F) |
| Best result | 82% classification accuracy using only K-dimensional module activation features |
| Biggest finding | Discovered modules are hierarchically consistent, spatially organized, and semantically coherent — without any supervision |
| Key gap | No causal validation; FFN-only; single model family; no intervention experiments |
| Code status | Available, 5 commits, minimally adopted as of May 2026 |
| Venue | AAAI-26 Oral, AAAI Technical Track NLP VI, pages 34503-34511 |

---

*Analysis prepared by Research Assistant | May 18, 2026*
*Sources: arXiv:2603.17823, AAAI-2026 Proceedings DOI:10.1609/aaai.v40i41.40749, GitHub:rank-Yu/llm-modules, and related literature survey*
