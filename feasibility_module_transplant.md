# Feasibility Report: Module/Neuron Transplantation Between LLMs

**For Dr. Zhang | May 19, 2026**
**Exhaustive Literature Search Results**

---

## EXECUTIVE SUMMARY

Module/neuron transplantation between LLMs is an **active and rapidly growing research area** (2024-2026), with multiple groups independently converging on very similar ideas. However, **no existing work performs transplantation at the level of causally-validated functional modules** (i.e., transplanting a "math module" identified via double dissociation from one LLM into another). The closest papers operate at coarser granularity (full layers) or narrower scope (safety neurons only). This represents both a **validation** (the direction is hot) and an **opportunity** (our specific angle — functional-module-level transplantation guided by atlas knowledge — remains open).

---

## 1. DIRECTLY RELEVANT PAPERS: NEURON/MODULE TRANSPLANTATION

### 1.1 CNT: Safety-oriented Function Reuse across LLMs via Cross-Model Neuron Transfer
- **Paper**: arXiv:2603.18449 (March 2026)
- **What they do**: Transfer a minimal subset of neurons from a donor LLM to a target LLM to add or remove safety-related functions. Operates at the individual neuron level.
- **Key results**: Targeted safety functionality transfer with <1% performance degradation on other tasks. Evaluated on 7 popular LLMs. Supports function addition AND deletion.
- **How close to our idea**: **VERY CLOSE**. This is neuron-level cross-model transplantation. But limited to safety functions only — not general cognitive capabilities (math, code, reasoning).
- **Gap we fill**: We transplant *functional modules identified via neuroscience methods* (not just safety neurons identified via activation filtering).

### 1.2 ARM: Role-Conditioned Neuron Transplantation for Training-Free Generalist LLM Agent Merging
- **Paper**: arXiv:2601.07309 (January 2026, Fudan University)
- **What they do**: 3-step framework: (1) construct merged backbones, (2) role-conditioned activation analysis to select neurons, (3) neuron transplantation for fine-grained refinement. Training-free.
- **Key results**: Improves merging from static NLP tasks to multi-turn agent scenarios. Enhances generalization across interactive environments without gradient-based optimization.
- **How close to our idea**: **VERY CLOSE**. Uses activation-guided neuron selection + transplantation. But focused on *agent environments*, not general cognitive function transfer.
- **Gap we fill**: We transplant causally-validated functional modules (double dissociation validated), not just activation-selected neurons.

### 1.3 Beyond Learning: A Training-Free Alternative to Model Adaptation (Model Transplantation)
- **Paper**: arXiv:2602.16189 (February 2026, Korea University)
- **What they do**: Identify modules showing consistent activation changes under specific tasks. Directly transplant internal layers from a source model into a target model. Training-free.
- **Key results**: Transplanting activation-selected modules can improve underperforming models by up to 2x the target baseline, with >100% gap-based recovery in some cases.
- **How close to our idea**: **VERY CLOSE at the layer level**. This IS model transplantation. But operates at the layer level (replacing entire transformer blocks), not at the functional neuron population level.
- **Gap we fill**: We transplant *specific neuron populations within layers* identified as belonging to a cognitive function, not entire layers.

### 1.4 Neural Organ Transplantation (NOT): Checkpoint-Based Modular Adaptation for Transformer Models
- **Paper**: arXiv:2601.13580 (January 2026)
- **What they do**: Extract contiguous layer subsets ("donor organs") from pretrained models, train them independently on domain-specific data, save as standalone checkpoint files, then transplant into compatible recipient models.
- **Key results**: Tested on GPT-2, TinyLlama, and GPT-OSS (124M–20B parameters). No access to original training data needed.
- **How close to our idea**: **MODERATELY CLOSE**. Explicitly uses the "organ transplant" metaphor. But "organs" = contiguous layer blocks, not functionally-defined neuron populations.
- **Gap we fill**: Our "organs" are defined by functional role (math, code, language), not by layer position.

### 1.5 Model Fusion via Neuron Transplantation
- **Paper**: arXiv:2502.06849 (February 2025, ECML-PKDD 2024)
- **What they do**: Fuse an ensemble of models by pruning insignificant neurons and transplanting important neurons from ensemble members into vacant spaces.
- **Key results**: Performance quickly recovers with fine-tuning. Outperforms OT-fusion and individual ensemble members. Faster and more memory-efficient than alternatives.
- **How close to our idea**: **MODERATE**. Neuron-level transplantation, but within an ensemble (same architecture, same task), not between different models with different capabilities.
- **Code**: https://github.com/masterbaer/neuron-transplantation

### 1.6 Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction
- **Paper**: arXiv:2601.16034 (January 2026)
- **What they do**: Transfer refusal interventions from donor to target models spanning diverse architectures (Dense to MoE). Uses "Concept Atom Registry" for shared semantic basis between models. Includes weight-SVD stability guard to prevent collateral damage.
- **Key results**: Cross-architecture refusal transfer without target-side supervision.
- **How close to our idea**: **CONCEPTUALLY CLOSE**. Transfers a specific functional circuit (refusal) across architectures. But limited to one function (refusal) and uses activation-space reconstruction rather than direct weight transplantation.

### 1.7 Who Transfers Safety? Identifying and Targeting Cross-Lingual Shared Safety Neurons
- **Paper**: arXiv:2602.01283 (February 2026)
- **What they do**: Identify cross-lingual shared safety neurons (SS-Neurons). Show that pruning these neurons in a base model reliably induces safety degradation in fine-tuned and distilled sibling models.
- **Key results**: Attack success rates jump from <5% to >75% when SS-Neurons are pruned. Safety neurons form a generalizable, attackable core across model variants.
- **Relevance**: Demonstrates that specific neurons encode transferable functions across model variants — a prerequisite for transplantation.

---

## 2. MODEL STITCHING (Connecting Layers from Different Models)

### 2.1 Revisiting Model Stitching to Compare Neural Representations
- **Paper**: Bansal et al., NeurIPS 2021
- **What they do**: Connect bottom-layers of model A to top-layers of model B via a trainable affine "stitch" layer. Measure representational compatibility.
- **Key finding**: Good networks of the same architecture but trained differently produce similar internal representations. Model stitching reveals aspects that other similarity measures miss.
- **Relevance**: Foundational work showing that representations from different models are "compatible" — a prerequisite for transplantation.

### 2.2 Transferring Linear Features Across Language Models With Model Stitching
- **Paper**: arXiv:2506.06609 (NeurIPS 2025 Spotlight)
- **What they do**: Learn affine mappings between residual streams of different-scale LMs. Transfer SAE weights, probes, and steering vectors between models.
- **Key results**: Small and large models learn highly similar representation spaces. Transferred SAEs as initialization → 50% cheaper SAE training. Steering vectors transfer faithfully.
- **How close to our idea**: **IMPORTANT ENABLING WORK**. Shows that feature-level representations transfer between models — not just at the layer level but at the SAE feature level. Does not transplant weights/neurons directly.

### 2.3 Revisiting Model Stitching In the Foundation Model Era
- **Paper**: arXiv:2603.12433 (March 2026, Amazon + Ohio State)
- **What they do**: Test whether heterogeneous Vision Foundation Models are stitchable. Feature-matching loss at penultimate layer enables reliable stitching across vision tasks.
- **Relevance**: Extends stitching to heterogeneous models — relevant to cross-architecture transplantation feasibility.

---

## 3. MODEL MERGING (Context: How Existing Methods Differ from Functional Transplantation)

### 3.1 Standard Merging Methods (SLERP, TIES, DARE, Task Arithmetic)
- **Granularity**: Parameter-level or layer-level. These operate on ENTIRE weight matrices or task vectors, not on functionally-identified neuron subsets.
- **SLERP**: Spherical interpolation between two models' full weight spaces.
- **Task Arithmetic**: Treats weight differences from a shared pretrained base as composable vectors. Algebraic operations (addition/subtraction) on full task vectors.
- **TIES-Merging**: Trims low-value parameters and resolves sign conflicts — still parameter-level, not function-level.
- **DARE**: Sparsifies task vectors via random dropout — preprocessing before merging, not functional selection.
- **Key limitation**: None of these methods select neurons based on *what function they serve*. They apply uniform operations across all parameters or layers.

### 3.2 MIN-Merging: Merge the Important Neurons
- **Paper**: arXiv:2510.17890 (October 2025, Chinese Academy of Sciences)
- **What they do**: Router-based framework that selectively merges the most *important* neurons to reduce parameter conflicts.
- **Key results**: 83.3% on NLP GLUE merges, exceeding fine-tuned upper bounds. Enhanced in-domain + retained out-of-domain.
- **How close**: Operates at neuron level but selects by *importance* (activation magnitude), not by *functional role*.

### 3.3 NeuroMerging: "To See a World in a Spark of Neuron"
- **Paper**: arXiv:2503.05320 (EMNLP 2025)
- **What they do**: Decompose task-specific representations into neuronal subspaces (sensitivity + adaptability). Merge these subspaces separately.
- **Key results**: Superior performance on multi-task benchmarks. First study relying on neuronal mechanisms for model merging.
- **How close**: **CLOSE in spirit** — uses neuron-level functional decomposition. But merges within the same model rather than transplanting between different models.

### 3.4 Transport and Merge: Cross-Architecture Merging for LLMs
- **Paper**: arXiv:2602.05495 (February 2026)
- **What they do**: Optimal transport to align activations and infer cross-neuron correspondences between heterogeneous models. Transport matrices guide direct weight-space fusion.
- **Key results**: First cross-architecture merging method for LLMs — different sizes and potentially different architectures.
- **How close**: **IMPORTANT for feasibility** — shows that neuron correspondence CAN be established between different architectures. But still merges/averages weights rather than transplanting specific functional components.

### 3.5 FrankenMerging / Passthrough
- **What**: Concatenate layers from different models (e.g., two 7B models → 9B model). Examples: Goliath-120B, SOLAR-10.7B.
- **Granularity**: Entire layers (very coarse). No functional selection.
- **Key limitation**: Experimental, recipe-based. No understanding of what functions each layer contributes.

### 3.6 Git Re-Basin: Merging Models modulo Permutation Symmetries
- **Paper**: Ainsworth et al. 2022 (ICLR 2023)
- **What they do**: Permute units of one model to align with another, revealing that loss landscapes contain (nearly) a single basin after accounting for permutation symmetries.
- **Relevance**: Foundational for understanding WHY weight transplantation is feasible — models trained independently can be aligned via permutation.

### 3.7 Model Merging Surveys
- **ACM Computing Surveys 2026**: "Model Merging in LLMs, MLLMs, and Beyond" (Yang et al.) — comprehensive taxonomy.
- **arXiv March 2026**: "Model Merging in the Era of Large Language Models" — FUSE taxonomy covering Foundations, Unification Strategies, Scenarios, Ecosystem.
- **Key gap noted in surveys**: Most merging methods operate at model-level, layer-level, or parameter-level. Neuron-level functional merging is identified as an emerging frontier.

---

## 4. CROSS-MODEL CAPABILITY TRANSFER (Related but Distinct Methods)

### 4.1 Cross-LoRA: Data-Free LoRA Transfer across Heterogeneous LLMs
- **Paper**: arXiv:2508.05232 (2025)
- **What they do**: Transfer LoRA adapters between different base models without retraining. Uses SVD-based subspace alignment.
- **Relevance**: Shows capability transfer is possible at the adapter level without data access. But transfers the *adapter*, not specific functional neurons.

### 4.2 LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation
- **Paper**: arXiv:2501.16559 (ICLR 2025)
- **What they do**: Training-free transfer of LoRA parameters across source and target models.
- **Relevance**: Demonstrates cross-model weight transfer. But LoRA = low-rank perturbation, not functional module.

### 4.3 GraftLLM: Knowledge Grafting via Modular SkillPacks
- **Paper**: arXiv:2505.18502 (2025)
- **What they do**: Store source model capabilities as modular "SkillPacks" (parameter deltas via distillation). Graft into target models. Module-aware adaptive compression.
- **Key results**: Outperforms existing techniques in knowledge transfer, fusion, and continual learning. Supports forget-free learning.
- **How close**: **CONCEPTUALLY ALIGNED** — modular capability transfer. But SkillPacks are obtained via distillation (training), not via functional module identification.

### 4.4 Activation Space Interventions Can Be Transferred Between Large Language Models
- **Paper**: arXiv:2503.04429 (ICML 2025)
- **What they do**: Use autoencoders to learn mappings between activation spaces of different models. Transfer steering vectors for safety tasks (backdoor removal, refusal).
- **Key results**: Successful cross-family transfer (Llama, Qwen, Gemma). Smaller models can align larger ones.
- **Relevance**: Shows activation-level interventions transfer. Operates in activation space, not weight space.

### 4.5 XTransplant: Cross-lingual Latent Transplantation
- **Paper**: arXiv:2412.12686 (December 2024)
- **What they do**: During inference, extract activations from a specific module in one layer for one language and transplant them into the corresponding module for another language's input.
- **Key results**: Mutually beneficial effects on multilingual capability. Attention modules support understanding; feed-forward modules capture culture-specific knowledge.
- **How close**: Uses the word "transplantation" explicitly. Operates on activations (not weights) between languages within the same model.

### 4.6 NTT Learning Transfer: Reusing Past Learning Trajectories
- **Paper**: ICLR 2024
- **What they do**: Transfer learning trajectories between different models by exploiting permutation symmetry in parameter space.
- **Key results**: First algorithm to solve the "learning transfer problem." Reduces retraining cost significantly.
- **Relevance**: Shows that weight-level correspondences can be established between different models through permutation alignment — foundational for transplantation.

---

## 5. FUNCTIONAL MODULE DISCOVERY IN LLMs (The Identification Step)

### 5.1 ULCMOD: Unsupervised LLM Cross-layer Module Discovery
- **Paper**: arXiv:2603.17823 (2026, HKUST + Huawei)
- **What they do**: Disentangle neurons into functional modules while discovering topics. Modules show function comprehensiveness, hierarchy, and spatial arrangement.
- **Relevance**: **Direct enabler** for our transplantation work — you need to identify modules before transplanting them.

### 5.2 Brain-Inspired Exploration of Functional Networks and Key Neurons in LLMs
- **Paper**: arXiv:2502.20408 (2025)
- **What they do**: Use ICA (from neuroimaging) to discover functional networks in LLMs. Show that inhibiting key networks impairs capability; amplifying enhances it.
- **Relevance**: Identifies functional neuron groups that could be transplanted.

### 5.3 Pruning LLMs by Identifying and Preserving Functional Networks
- **Paper**: arXiv:2508.05239 (August 2025)
- **What they do**: Treat LLM as a "digital brain," decompose into functional networks, prune by preserving key neurons.
- **Code**: https://github.com/WhatAboutMyStar/LLM_ACTIVATION
- **Relevance**: Shows functional networks can be identified and preserved — halfway to transplantation (identification without transfer).

### 5.4 Spontaneous Functional Differentiation in LLMs: A Brain-Like Intelligence Economy
- **Paper**: arXiv:2603.29735 (March 2026)
- **Key finding**: LLMs exhibit brain-like energy economy — middle layers act as reasoning cores, early/late layers as memory. Organization emerges via phase transitions.

### 5.5 Brain-like Functional Organization within Large Language Models
- **Paper**: arXiv:2410.19542 (October 2024, ICLR 2025 submission)
- **Key finding**: Sub-groups of artificial neurons mirror functional brain networks. Organization evolves with model sophistication.

### 5.6 Finding Skill Neurons in Pre-trained Transformer Language Models
- **Paper**: EMNLP 2022
- **Key finding**: Specific neurons are highly predictive of task performance ("skill neurons"). They are task-specific, generated during pretraining, and crucial for handling tasks.
- **Relevance**: **Foundational** — skill neurons are exactly what you would transplant.

### 5.7 CRANE: Causal Relevance Analysis of Language-Specific Neurons
- **Paper**: arXiv:2601.04664 (January 2026)
- **What they do**: Redefine language specificity in terms of functional necessity via neuron-level interventions.
- **Relevance**: Causal identification of language-specific neurons — transplantation candidates.

### 5.8 Unraveling the Cognitive Patterns of LLMs through Module Communities
- **Paper**: arXiv:2508.18192 (August 2025)
- **What they do**: Network-based framework linking cognitive skills, architectures, and datasets. Find LLMs exhibit unique module communities with emergent skill patterns partially mirroring biological brains.

---

## 6. ENABLING TECHNOLOGIES

### 6.1 Sparse Autoencoders (SAEs) for Feature Identification
- SAEs decompose neural activations into interpretable features. Cross-model SAE transfer demonstrated (NeurIPS 2025).
- **Relevance**: SAE features could define the "functional modules" more precisely than raw neuron activations.

### 6.2 Activation Patching / Causal Tracing (Meng et al. 2022)
- Transplant activations from clean run to corrupted run to identify causal components.
- **Relevance**: Foundational technique for identifying which components are causally responsible for specific behaviors — precursor to knowing what to transplant.

### 6.3 Git Re-Basin (Ainsworth et al. 2023)
- Permutation alignment makes independently trained models compatible for merging.
- **Relevance**: Solves the alignment problem — necessary for transplanting neurons between models that were trained independently.

### 6.4 Grafting: Architecture Editing via Activation Distillation
- **Paper**: arXiv:2506.05340 (NeurIPS 2025 Oral, Stanford + Liquid AI)
- **What they do**: Replace specific operators (attention heads, MLPs) in pretrained transformers with new architectures via activation distillation. Requires <2% of pretraining compute.
- **Relevance**: Demonstrates that individual components can be replaced without catastrophic failure.

### 6.5 Optimal Transport for Cross-Architecture Neuron Correspondence
- Transport and Merge (arXiv:2602.05495) and DOTResize (arXiv:2507.04517) use optimal transport to map neuron correspondences between heterogeneous models.
- **Relevance**: Provides the mathematical machinery for establishing which neurons in Model A correspond to which neurons in Model B.

### 6.6 Representation Engineering: Steering Vectors Transfer
- **Paper**: Turner et al. 2023; Oozeer et al. ICML 2025
- Steering vectors derived from one model can be transferred to another via learned activation-space mappings.
- **Relevance**: Shows functional interventions (not just features) transfer between models.

---

## 7. ANALYSIS: WHAT EXISTS VS. WHAT WE PROPOSE

### 7.1 The Existing Landscape (Taxonomy of Granularity)

| Granularity | What Gets Transferred | Example Methods | Functional Awareness |
|-------------|----------------------|-----------------|---------------------|
| **Full model** | All weights averaged | SLERP, simple averaging | None |
| **Layer-level** | Entire transformer blocks | FrankenMerging, SOLAR DUS, NOT | None (positional only) |
| **Task vector level** | Weight deltas from fine-tuning | Task Arithmetic, TIES, DARE | Implicit (task-defined) |
| **Neuron-level (importance)** | High-activation neurons | MIN-Merging, NeuroMerging | Importance, not function |
| **Neuron-level (safety)** | Safety-specific neurons | CNT, SS-Neurons | Safety function only |
| **Neuron-level (role)** | Role-conditioned neurons | ARM | Agent role, not cognitive function |
| **Activation-level** | Activations (not weights) | XTransplant, steering vectors | Behavior-defined |
| **Feature-level** | SAE features | Stitching + SAE transfer | Interpretable but not causal |
| **🎯 Functional-module-level** | Causally-validated cognitive modules | **NOT YET DONE** | Full causal + functional |

### 7.2 The Gap

**Nobody has combined:**
1. Identification of functional modules via neuroscience methods (ICA, localizers, double dissociation)
2. Cross-model neuron correspondence mapping (optimal transport, permutation alignment)
3. Surgical transplantation of specific functional neuron populations
4. Validation that the transplanted capability actually works in the recipient model

Each of these steps has been done independently, but never as an integrated pipeline.

### 7.3 Key Distinctions from Closest Work

| Paper | What They Transfer | How They Select | Cross-Model? | Causal Validation? |
|-------|-------------------|----------------|-------------|-------------------|
| CNT (2026) | Safety neurons | Activation analysis | Yes | Partial (safety metrics) |
| ARM (2026) | Role-conditioned neurons | Activation + role prompts | Yes (same architecture) | No |
| Beyond Learning (2026) | Entire layers | Activation discrepancy | Yes (same family) | No |
| NOT (2026) | Layer blocks (checkpoints) | Positional | Yes (same architecture) | No |
| **Our proposal** | **Functional cognitive modules** | **Double dissociation + atlas** | **Yes (cross-architecture)** | **Yes (gold standard)** |

---

## 8. TECHNICAL FEASIBILITY ASSESSMENT

### 8.1 What Makes Transplantation Plausible (Positive Evidence)

1. **Permutation symmetry**: Git Re-Basin shows models can be aligned in weight space. NTT Learning Transfer shows trajectories can be reused. This means neuron correspondences ARE findable.

2. **Shared representation spaces**: NeurIPS 2025 spotlight shows models of different scales share similar feature spaces. SAEs transfer between models. This means functional roles OVERLAP.

3. **Safety neuron transfer works**: CNT demonstrates functional neuron-level transfer with <1% collateral damage. If it works for safety, it can work for math/code/reasoning.

4. **Layer-level transplantation works**: "Beyond Learning" gets 2x improvement by transplanting layers. Functional modules are just a more precise version of this.

5. **Steering vectors transfer**: ICML 2025 paper shows activation-level interventions transfer across model families (Llama, Qwen, Gemma).

### 8.2 Key Technical Challenges

1. **Neuron correspondence problem**: Different models have different numbers of neurons, different layer structures. How do you map "math neuron #347 in Llama" to "the corresponding neuron in Qwen"?
   - **Solution exists**: Optimal transport (Transport and Merge), permutation alignment (Git Re-Basin), activation mapping (SAE transfer).

2. **Functional module boundaries**: Modules aren't cleanly separable — neurons participate in multiple functions.
   - **Mitigation**: Use soft assignments (participation coefficients). Transplant the top-k most function-specific neurons.

3. **Scale mismatch**: Models have different widths and depths.
   - **Solution exists**: Cross-architecture merging via optimal transport handles dimension mismatches. Weight subcloning handles size differences.

4. **Integration shock**: Transplanted neurons may disrupt the recipient model's internal coordination.
   - **Evidence for optimism**: CNT achieves <1% degradation. NOT works across scales. "Beyond Learning" shows immediate improvement.
   - **Mitigation**: Light fine-tuning post-transplant (demonstrated in Neuron Transplantation fusion paper).

5. **Cross-architecture transfer**: LLaMA → Qwen is harder than LLaMA → LLaMA.
   - **Partial solution**: Transport and Merge handles cross-architecture. Universal refusal circuits transfer across Dense → MoE.

### 8.3 Feasibility Verdict

**FEASIBLE with caveats.** Functional-module-level transplantation is technically achievable using existing building blocks (module identification + neuron correspondence + weight transfer). The key innovation is the *integration* — using atlas-derived functional module definitions to guide transplantation, validated by whether the transplanted function actually works.

**Expected difficulty ranking:**
1. **Easiest**: Same architecture, same scale (e.g., LLaMA-3-8B → LLaMA-3-8B-Instruct) — essentially what CNT already does
2. **Moderate**: Same architecture, different scale (e.g., LLaMA-3-8B → LLaMA-3-70B) — needs dimension mapping
3. **Hard**: Different architecture, similar scale (e.g., LLaMA-3-8B → Mistral-7B) — needs full OT alignment
4. **Very hard**: Different architecture, different scale (e.g., Qwen-2.5-7B → LLaMA-3-70B) — combines all challenges

---

## 9. STRATEGIC IMPLICATIONS FOR THE NATURE PAPER

### 9.1 Module Transplantation as Act 6 / Act 7

Module transplantation could serve as the **ultimate validation** of the functional atlas:

> "If the atlas correctly identifies the 'math module,' then transplanting it from Model A to Model B should transfer mathematical capability — the strongest possible test of functional specificity."

This would be the **neuroscience equivalent of a brain region transplant experiment** — which has never been done computationally at the functional module level.

### 9.2 Potential Additional Act for the Paper

**Act 7: Module Transplantation — The Acid Test**
- Take the math module from LLaMA-3-8B (identified via double dissociation in Act 2)
- Transplant it into Qwen-2.5-7B (using OT-based neuron correspondence from Transport and Merge)
- Test whether Qwen's math performance improves
- Control: transplant a random set of neurons of the same size → should NOT improve math
- Control: transplant the code module → should improve code, NOT math
- **If this works, it is an extraordinary result** — proving that functional modules are not just descriptively real but operationally transferable.

### 9.3 Risk Assessment for Transplantation Experiments

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Transplantation fails entirely | Medium | Low (paper strong without it) | Keep as supplementary result |
| Transplantation works within architecture but not across | Medium-High | Low | Report as-is — still interesting |
| Transplantation requires fine-tuning to work | High | Low | Expected; fine-tuning budget is small |
| Transplantation works perfectly without fine-tuning | Low | Very High | If this happens, it's a separate Nature paper |
| Scooped on transplantation specifically | Medium | Medium | CNT/ARM are close but not at functional-module level |

### 9.4 Competitive Landscape Warning

The field is moving FAST. Key groups to watch:
- **Korea University** (Beyond Learning / Model Transplantation) — actively working on layer-level transplantation
- **Fudan University** (ARM) — neuron-level transplantation for agents
- **CNT authors** (arXiv:2603.18449) — neuron-level safety transplantation
- **HKUST + Huawei** (ULCMOD) — functional module discovery (the identification half)
- **NTT** — learning trajectory transfer (foundational theory)

None of them are combining functional atlas + transplantation. But the convergence is clear — someone will do it within 12-18 months.

---

## 10. ALL RELEVANT PAPERS (Comprehensive Reference List)

### Neuron/Module Transplantation
1. CNT: Safety-oriented Function Reuse across LLMs via Cross-Model Neuron Transfer — [arXiv:2603.18449](https://arxiv.org/abs/2603.18449)
2. ARM: Role-Conditioned Neuron Transplantation for Training-Free Generalist LLM Agent Merging — [arXiv:2601.07309](https://arxiv.org/abs/2601.07309)
3. Beyond Learning: A Training-Free Alternative to Model Adaptation — [arXiv:2602.16189](https://arxiv.org/abs/2602.16189)
4. Neural Organ Transplantation (NOT): Checkpoint-Based Modular Adaptation — [arXiv:2601.13580](https://arxiv.org/abs/2601.13580)
5. Model Fusion via Neuron Transplantation — [arXiv:2502.06849](https://arxiv.org/abs/2502.06849)
6. Universal Refusal Circuits Across LLMs — [arXiv:2601.16034](https://arxiv.org/abs/2601.16034)
7. Who Transfers Safety? Cross-Lingual Shared Safety Neurons — [arXiv:2602.01283](https://arxiv.org/abs/2602.01283)

### Model Stitching
8. Revisiting Model Stitching to Compare Neural Representations (Bansal et al., NeurIPS 2021) — [arXiv:2106.07682](https://arxiv.org/abs/2106.07682)
9. Transferring Linear Features Across Language Models With Model Stitching (NeurIPS 2025 Spotlight) — [arXiv:2506.06609](https://arxiv.org/abs/2506.06609)
10. Revisiting Model Stitching In the Foundation Model Era — [arXiv:2603.12433](https://arxiv.org/abs/2603.12433)

### Model Merging (Neuron-Aware)
11. MIN-Merging: Merge the Important Neurons — [arXiv:2510.17890](https://arxiv.org/abs/2510.17890)
12. NeuroMerging: To See a World in a Spark of Neuron (EMNLP 2025) — [arXiv:2503.05320](https://arxiv.org/abs/2503.05320)
13. Transport and Merge: Cross-Architecture Merging for LLMs — [arXiv:2602.05495](https://arxiv.org/abs/2602.05495)
14. Git Re-Basin: Merging Models modulo Permutation Symmetries (ICLR 2023) — [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)
15. Model Fusion via Neuron Interpolation — [arXiv:2507.00037](https://arxiv.org/abs/2507.00037)
16. DOTResize: Reducing LLM Width via Discrete OT-based Neuron Merging — [arXiv:2507.04517](https://arxiv.org/abs/2507.04517)

### Cross-Model Capability Transfer
17. Cross-LoRA: Data-Free LoRA Transfer across Heterogeneous LLMs — [arXiv:2508.05232](https://arxiv.org/abs/2508.05232)
18. LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation (ICLR 2025) — [arXiv:2501.16559](https://arxiv.org/abs/2501.16559)
19. GraftLLM: Knowledge Grafting of LLMs via Modular SkillPacks — [arXiv:2505.18502](https://arxiv.org/abs/2505.18502)
20. Activation Space Interventions Transfer Between LLMs (ICML 2025) — [arXiv:2503.04429](https://arxiv.org/abs/2503.04429)
21. Cross-lingual Latent Transplantation (XTransplant) — [arXiv:2412.12686](https://arxiv.org/abs/2412.12686)
22. Transferring Learning Trajectories of Neural Networks (ICLR 2024) — [arXiv:2305.14122](https://arxiv.org/abs/2305.14122)
23. Trans-LoRA: Data-free Transferable Parameter Efficient Finetuning — [arXiv:2405.17258](https://arxiv.org/abs/2405.17258)

### Functional Module Discovery in LLMs
24. ULCMOD: Discovering Decoupled Functional Modules in LLMs — [arXiv:2603.17823](https://arxiv.org/abs/2603.17823)
25. Brain-Inspired Exploration of Functional Networks and Key Neurons in LLMs — [arXiv:2502.20408](https://arxiv.org/abs/2502.20408)
26. Pruning LLMs by Identifying and Preserving Functional Networks — [arXiv:2508.05239](https://arxiv.org/abs/2508.05239)
27. Spontaneous Functional Differentiation in LLMs: A Brain-Like Intelligence Economy — [arXiv:2603.29735](https://arxiv.org/abs/2603.29735)
28. Brain-like Functional Organization within Large Language Models — [arXiv:2410.19542](https://arxiv.org/abs/2410.19542)
29. Finding Skill Neurons in Pre-trained Transformer Language Models (EMNLP 2022) — [arXiv:2211.07349](https://arxiv.org/abs/2211.07349)
30. CRANE: Causal Relevance Analysis of Language-Specific Neurons — [arXiv:2601.04664](https://arxiv.org/abs/2601.04664)
31. Unraveling the Cognitive Patterns of LLMs through Module Communities — [arXiv:2508.18192](https://arxiv.org/abs/2508.18192)
32. NeuronMoE: Neuron-Guided MoE for Multilingual LLM Extension — [arXiv:2603.05046](https://arxiv.org/abs/2603.05046)
33. A Brain-like Synergistic Core in LLMs Drives Behaviour and Learning — [arXiv:2601.06851](https://arxiv.org/abs/2601.06851)
34. The Transfer Neurons Hypothesis — [arXiv:2509.17030](https://arxiv.org/abs/2509.17030)
35. Flash Interpretability: Decoding Specialised Feature Neurons in LLMs — [arXiv:2501.02688](https://arxiv.org/abs/2501.02688)

### Knowledge Editing (Related Paradigm)
36. ROME: Locating and Editing Factual Associations in GPT (NeurIPS 2022) — [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
37. Precise Localization of Memories: Fine-grained Neuron-level Knowledge Editing — [arXiv:2503.01090](https://arxiv.org/abs/2503.01090)
38. Knowledge Editing of LLMs with Knowledge Neuronal Ensemble — [arXiv:2412.20637](https://arxiv.org/abs/2412.20637)

### Architecture and Component Transfer
39. Neural Network Surgery with Sets (2019) — [arXiv:1912.06719](https://arxiv.org/abs/1912.06719)
40. SOLAR 10.7B: Scaling LLMs with Depth Up-Scaling — [arXiv:2312.15166](https://arxiv.org/abs/2312.15166)
41. Exploring Diffusion Transformer Designs via Grafting (NeurIPS 2025 Oral) — [arXiv:2506.05340](https://arxiv.org/abs/2506.05340)
42. Attention Editing: Cross-Architecture Attention Conversion — [arXiv:2604.05688](https://arxiv.org/abs/2604.05688)
43. Weight Subcloning: Direct Initialization from Larger Pretrained Models — [arXiv:2312.09299](https://arxiv.org/abs/2312.09299)
44. Transfer Learning Between Different Architectures Via Weights Injection — [arXiv:2101.02757](https://arxiv.org/abs/2101.02757)

### Lottery Ticket / Subnetwork Transfer
45. The Elastic Lottery Ticket Hypothesis — [arXiv:2103.16547](https://arxiv.org/abs/2103.16547)
46. Robust Tickets Can Transfer Better — [arXiv:2304.11834](https://arxiv.org/abs/2304.11834)
47. The Lottery Ticket Hypothesis for Pre-trained BERT Networks (NeurIPS 2020) — [arXiv:2007.12223](https://arxiv.org/abs/2007.12223)

### Surveys
48. Model Merging in LLMs, MLLMs, and Beyond (ACM Computing Surveys, 2026) — [arXiv:2408.07666](https://arxiv.org/abs/2408.07666)
49. Model Merging in the Era of LLMs (March 2026) — [arXiv:2603.09938](https://arxiv.org/abs/2603.09938)
50. A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of LLMs — [arXiv:2503.05613](https://arxiv.org/abs/2503.05613)

---

## 11. BOTTOM LINE

### What exists:
- Layer-level transplantation between models (works, training-free)
- Neuron-level transplantation for safety functions (works, <1% collateral damage)
- Neuron-level transplantation for agent roles (works, training-free)
- Functional module discovery in LLMs (multiple methods)
- Cross-architecture neuron correspondence mapping (optimal transport)
- Representation/feature transfer between model scales (SAEs, steering vectors)

### What does NOT exist:
- **Transplantation of causally-validated cognitive functional modules** (math module, code module, reasoning module) between different LLM architectures
- **Atlas-guided transplantation** — using a functional atlas to select WHAT to transplant
- **Double-dissociation validation** of transplanted capabilities
- **Systematic comparison** of transplantation success across different function types

### Our unique contribution:
The integration of **functional atlas** (Acts 1-4 of the paper) with **module transplantation** creates a closed-loop validation: discover modules → validate causally → transplant → validate transfer. This is the **neurosurgery paradigm** applied to AI — precision intervention guided by functional mapping.

---

*Report compiled May 19, 2026 | Based on exhaustive web search across 30+ queries and 50+ papers*
