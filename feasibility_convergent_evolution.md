# Feasibility Study: Convergent Functional Organization Across Different AI Architectures

**Prepared for Dr. Zhang | May 19, 2026**
**Research Question: Has anyone compared whether Transformer, Mamba (SSM), RWKV, or other fundamentally different architectures develop the same functional modules/specialization when trained on language?**

---

## EXECUTIVE SUMMARY

**The specific angle -- "convergent evolution of functional modules across fundamentally different architectures" -- has NOT been done as a unified study.** However, the pieces are falling into place rapidly. Multiple 2024-2026 papers each address a fragment of this question, creating both a clear opportunity and a narrowing window.

**Risk level: MEDIUM-HIGH for scooping on individual pieces, but LOW for the unified "functional atlas across architectures" framing.** No one has combined functional module discovery + cross-architecture comparison + causal validation + brain alignment into a single paper across Transformer/Mamba/RWKV.

---

## PART 1: DIRECTLY RELEVANT PAPERS (Closest to Our Angle)

### 1.1 "Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures"
- **Authors**: Autumn 2024; accepted at ICLR 2025/2026
- **Link**: https://arxiv.org/abs/2410.06672
- **What they did**: Compared Transformers (Pythia) and Mamba using Sparse Autoencoders (SAEs) to extract interpretable features from both architectures. Also compared induction circuits across the two.
- **Key findings**:
  - More than 30% of Pythia features find their matched feature in Mamba with very high similarity (cross-architecture SAE similarity average: 0.74)
  - Induction circuits in Mamba are structurally analogous to those in Transformers
  - A nuanced "Off-by-One motif" in Mamba: token information is written into the SSM state at the next position, unlike Transformers
  - At the **neuron level**, comparison between Mamba and Pythia shows almost zero similarity (only 1-5% of neurons are universal even within Transformers of different seeds)
  - Features extracted by SAEs are much more universal than raw neurons
- **CRITICAL IMPLICATION**: Features are universal across architectures; neurons are not. This means comparing "functional modules" (as clusters of SAE features rather than raw neurons) is the correct approach.
- **Gap**: They did NOT study RWKV. They did NOT map functional modules (they studied individual features and one circuit). They did NOT do brain alignment. They did NOT do double dissociation.

### 1.2 "Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders"
- **Authors**: Michael Lan et al., October 2024 (updated May 2025)
- **Link**: https://arxiv.org/abs/2410.06981
- **What they did**: Trained SAEs on different LLMs' residual-stream activations, aligned monosemantic features via activation correlation, compared matched feature spaces with SVCCA and RSA.
- **Key findings**: Significant similarities in SAE feature spaces across various LLMs. Semantically meaningful subspaces show even higher similarity across models.
- **Gap**: Focused on different-sized models within Transformer family, not cross-architecture comparison.

### 1.3 "Semantic Convergence: Investigating Shared Representations Across Scaled LLMs"
- **Authors**: ACL-SRW 2025 poster (June 2025)
- **Link**: https://arxiv.org/abs/2507.22918
- **What they did**: SAE dictionary-learning on Gemma-2-2B and Gemma-2-9B, aligned features via activation correlation, compared with SVCCA and RSA.
- **Key findings**: Middle layers yield strongest overlap; early and late layers show far less similarity. Despite fourfold size differences, models share strongly aligned internal representations.
- **Gap**: Same architecture family (Gemma-2), different sizes only.

### 1.4 "The Platonic Representation Hypothesis"
- **Authors**: Huh et al., ICML 2024
- **Link**: https://arxiv.org/abs/2405.07987
- **What they did**: Argued that neural networks trained with different objectives on different data and modalities are converging to a shared statistical model of reality.
- **Key findings**: Convergence is measurable via kernel alignment (CKA), model stitching, and mutual nearest-neighbor analysis. Convergence driven by scaling model capacity and data/task diversity.
- **CRITICAL IMPLICATION**: Provides the theoretical framework for why convergent functional organization might occur across architectures. This is the "why" paper; what's missing is the "how" at the functional module level for language models of different architectures.
- **Gap**: Very broad (vision + language + multimodal). Does NOT study specific functional modules. Does NOT compare Transformer vs. SSM internals.

---

## PART 2: MECHANISTIC INTERPRETABILITY OF NON-TRANSFORMER ARCHITECTURES

### 2.1 Mamba Interpretability (Rapidly Growing Field)

**a) "Locating and Editing Factual Associations in Mamba"**
- **Authors**: Sen Sharma, Atkinson, Bau (Northeastern); 2024
- **Link**: https://arxiv.org/abs/2404.03646
- **Key finding**: Factual recall in Mamba can be localized to specific components in middle layers, similar to how Transformers store factual knowledge. "Despite significant differences in architectural approach, the two architectures share many similarities."
- **Methods used**: Causal tracing, activation patching, rank-one model editing, attention-knockout adapted to Mamba.

**b) "Investigating the Indirect Object Identification Circuit in Mamba"**
- **Authors**: Ensign & Garriga-Alonso (FAR.AI); ICML 2024 MI Workshop
- **Link**: https://arxiv.org/abs/2407.14008
- **Key finding**: Circuit-based mechanistic interpretability tools work for Mamba. Layer 39 is a key bottleneck; convolutions in layer 39 shift names one position forward; name entities are stored linearly in the SSM state.
- **Method**: Positional Edge Attribution Patching adapted for Mamba.

**c) "The Hidden Attention of Mamba Models"**
- **Authors**: ACL 2025
- **Link**: https://arxiv.org/abs/2403.01590
- **Key finding**: Mamba's selective state-space layers can be reformulated as a form of attention. Mamba models give rise to three orders of magnitude more "attention matrices" than Transformers. This enables applying attention-based interpretability tools to Mamba.

**d) "Mamba Knockout for Unraveling Factual Information Flow"**
- **Authors**: ACL 2025
- **Link**: https://arxiv.org/abs/2505.24244
- **Key finding**: Extended Attention Knockout to Mamba-1 and Mamba-2. "Some phenomena vary between Mamba models and Transformer-based models, while others appear universally across all models inspected -- hinting that these may be inherent to LLMs in general."

**e) "Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks"**
- **Authors**: February 2026
- **Link**: https://arxiv.org/abs/2602.22719
- **Key finding**: Identified activation subspace bottlenecks in Mamba using mechanistic interpretability. Since Mamba lacks explicit attention heads, they interpret activation subspaces by projecting hidden states onto attention-weighted vectors derived from implicit attention matrices.

**f) "Mechanistic Evaluation of Transformers and State Space Models"**
- **Authors**: Submitted to NeurIPS 2025
- **Link**: https://arxiv.org/abs/2505.15105
- **Key finding**: "Architectures with similar accuracy may still have substantive differences" at the mechanistic level. Transformers learn key-value associations via induction across context; SSMs compute them at the last state using a single layer. Different mechanisms, similar outputs.

**g) "Characterizing Mamba's Selective Memory using Auto-Encoders"**
- **Authors**: IJCNLP 2025
- **Link**: https://arxiv.org/abs/2512.15653
- **Key finding**: Mamba exhibits strong recency bias and significant information loss on math-related tokens, rare tokens, and non-standard language. Auto-encoder reconstruction reveals what the SSM state retains vs. forgets.

**h) "Lost in State Space: Probing Frozen Mamba Representations"**
- **Authors**: May 2026
- **Link**: https://arxiv.org/abs/2605.00253
- **Key finding**: Frozen Mamba-130M representations show severe anisotropy (cosine similarity 0.9999) and representational collapse in raw SSM states. This is a critical technical warning for anyone trying to extract functional modules from Mamba hidden states.

### 2.2 RWKV Interpretability (Much Less Studied)

- **GitHub**: https://github.com/UnstoppableCurry/RWKV-LM-Interpretability-Research -- Interpretability analysis of RWKV language model outliers and distillation attempts
- RWKV-v5/v6 employ matrix-valued states (not just vectors), increasing representational capacity
- The decay rates in RWKV's time-mixing mechanism are input-dependent, allowing flexible temporal modeling
- **NO published papers found that apply SAE, circuit analysis, or causal tracing to RWKV specifically**
- **This is a significant gap and opportunity**

### 2.3 "Does Transformer Interpretability Transfer to RNNs?"
- **Authors**: AAAI 2025
- **Link**: https://arxiv.org/abs/2404.05971
- **Key finding**: Representation-based interpretability tools (contrastive activation addition, tuned lens, eliciting latent knowledge) work "out-of-the-box" for RNNs (including RWKV and Mamba), with similar but not identical performance.
- **Critical caveat**: "This work did not explore mechanistic or circuit-based interpretability tools" -- future work is needed on whether circuit analysis transfers.

---

## PART 3: BRAIN ALIGNMENT ACROSS ARCHITECTURES

### 3.1 "Scaling and Context Steer LLMs Along the Same Computational Path as the Human Brain"
- **Authors**: NeurIPS 2025 Spotlight
- **Link**: https://arxiv.org/abs/2512.01591
- **What they did**: Examined 22 LLMs (including Transformers, Mamba, RecurrentGemma) using MEG recordings of participants listening to audiobooks.
- **Key finding**: Brain-LLM alignment is consistent across Transformers AND recurrent architectures. Temporal alignment between model layers and MEG responses observed in ALL studied models, including Mamba-1.4B.
- **CRITICAL IMPLICATION**: Different architectures converge on brain-like computational dynamics. This validates our hypothesis at the representation level.

### 3.2 "Revenge of the Fallen? Recurrent Models Match Transformers at Predicting Human Language Comprehension Metrics"
- **Authors**: COLM 2024
- **Link**: https://arxiv.org/abs/2404.19178
- **Key finding**: Mamba and RWKV match or exceed comparably sized Transformers at predicting human language comprehension (reading times, N400 responses). Recurrent models outperform Transformers on N400 datasets.
- **CRITICAL IMPLICATION**: Different architectures achieve similar brain alignment through different mechanisms.

### 3.3 "Shared Functional Specialization in Transformer-Based Language Models and the Human Brain"
- **Authors**: Kumar et al., Nature Communications 2024
- **Link**: https://www.nature.com/articles/s41467-024-49173-5
- **What they did**: Deconstructed Transformer circuit computations into functionally-specialized "transformations" and showed they predict brain activity in specific cortical regions.
- **Gap**: Only studied Transformers (BERT). Did NOT compare with Mamba/RWKV.

### 3.4 "The Mind's Transformer: Computational Neuroanatomy of LLM-Brain Alignment"
- **Authors**: ICLR 2026
- **Link**: https://openreview.net/forum?id=PgIlCCNxdB
- **Key finding**: First systematic computational neuroanatomy of transformer blocks across 21 LLMs. Different computational stages within a single transformer block map to anatomically distinct brain systems (early attention -> sensory cortices; later FFN -> association areas).
- **Gap**: Only studied Transformers.

### 3.5 "Training-Driven Representational Geometry Modularization Predicts Brain Alignment in Language Models"
- **Authors**: February 2026
- **Link**: https://arxiv.org/abs/2602.07539
- **Key finding**: Layers self-organize into stable low- and high-complexity clusters during training, with the low-complexity module better predicting human language network activity. Alignment follows heterogeneous spatial-temporal trajectories: rapid in temporal regions, delayed in frontal areas.
- **Gap**: Only studied Pythia (Transformer family).

---

## PART 4: FUNCTIONAL ORGANIZATION / "BRAIN-LIKE MODULES" IN LLMs

### 4.1 "Brain-like Functional Organization within Large Language Models"
- **Authors**: October 2024
- **Link**: https://arxiv.org/abs/2410.19542
- **What they did**: Applied ICA (Independent Component Analysis) to LLM neuron activation patterns (similar to fMRI analysis), then mapped resulting sub-groups to functional brain networks via voxel-wise encoding models.
- **Key finding**: BERT and Llama 1-3 exhibit brain-like functional architecture, with sub-groups of artificial neurons mirroring functional brain networks. Organization improves with model sophistication.
- **Gap**: Only Transformer-based models (BERT, Llama). Did NOT study Mamba or RWKV.

### 4.2 "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models"
- **Authors**: Liu et al., February 2025
- **Link**: https://arxiv.org/abs/2502.20408
- **What they did**: Applied fMRI-inspired methods to identify functional networks in LLMs (GPT2, Qwen2.5-3B, ChatGLM3-6B). Showed task-specific functional networks emerge and are indispensable for performance.
- **Key finding**: LLMs exhibit recurrent functional networks like the human brain. Inhibiting key functional networks severely impairs the model. Amplifying them enhances performance.
- **Gap**: Only Transformer-based models. No Mamba/RWKV.

### 4.3 "FNF: Functional Network Fingerprint for Large Language Models"
- **Authors**: January 2026
- **Link**: https://arxiv.org/abs/2601.22692
- **Key finding**: Models sharing a common origin exhibit highly consistent functional network patterns. Functional networks can serve as "fingerprints" for model provenance detection.
- **Implication**: Functional networks are stable enough to serve as identifiers, suggesting they are robust emergent properties.

### 4.4 "TopoLM: Brain-like Spatio-Functional Organization in a Topographic Language Model"
- **Authors**: Rathi et al., October 2024
- **Link**: https://arxiv.org/abs/2410.11516
- **What they did**: Added a spatial correlation loss to a Transformer LM, creating topographic organization. The resulting "TopoLM" develops semantically interpretable clusters matching functional organization in the brain's language system.
- **Implication**: Functional organization can be *encouraged* in LLMs and matches brain patterns, but this was engineered, not spontaneous.

### 4.5 "Dynamics of Specialization in Neural Modules Under Resource Constraints"
- **Authors**: Nature Communications, January 2025
- **Link**: https://www.nature.com/articles/s41467-024-55188-9
- **Key finding**: Functional specialization in neural networks is NOT automatic. It requires: (1) meaningfully separable environmental features, (2) strong resource constraints, and (3) appropriate information flow dynamics. Findings are "qualitatively similar across several different variations of network architectures."
- **CRITICAL IMPLICATION**: Specialization depends on task structure and resource constraints, not architecture. This supports the convergent evolution hypothesis.

### 4.6 "Brain-Machine Convergent Evolution" (PNAS 2024)
- **Authors**: Simony, Grossman, Malach; PNAS 121(41):e2319709121
- **Link**: https://www.pnas.org/doi/10.1073/pnas.2319709121
- **What they did**: Extended the concept of convergent evolution from biology to brain-machine comparisons.
- **Key argument**: Parallels between artificial and neuronal networks are informative *precisely because* these systems are so different -- convergent evolution logic. Finding similar functional organization in different architectures solving the same problem is evidence of optimal solutions.
- **CRITICAL IMPLICATION**: Provides the conceptual framework for our "convergent functional organization" story.

---

## PART 5: TECHNICAL BARRIERS TO CROSS-ARCHITECTURE COMPARISON

### 5.1 Architectural Asymmetries: What Components Exist in Each Architecture?

| Component | Transformer | Mamba (SSM) | RWKV |
|-----------|------------|-------------|------|
| **Attention heads** | Yes (explicit) | No (but "hidden attention" can be extracted) | No (uses receptance-weighted gating) |
| **FFN/MLP layers** | Yes (separate from attention) | Fused -- SSM + MLP-style projections in one block | Channel-mixing block (analogous to FFN) |
| **Residual stream** | Yes | Yes | Yes |
| **Position encoding** | Explicit (RoPE, absolute, etc.) | Implicit via SSM dynamics + convolution | Implicit via time-decay mechanism |
| **Per-token hidden state** | Yes (residual stream) | Yes (residual stream) + recurrent state | Yes (residual stream) + recurrent state |
| **Recurrent state** | No | Yes (d_model x d_state matrix per layer) | Yes (matrix-valued state in v5/v6) |
| **Neurons / activations** | Clear per-neuron activations in FFN | Activations in linear projections, gates, SSM | Activations in time-mixing, channel-mixing |

### 5.2 Key Technical Challenges

**Challenge 1: No Direct Component-to-Component Mapping**
Transformers have a clean separation between attention (communication) and FFN (computation). Mamba fuses these into a single block. RWKV has time-mixing (communication) and channel-mixing (computation) but with different mechanisms. There is no one-to-one mapping of components.

**Proposed solution**: Work at the **feature level** (via SAEs on residual stream activations), not the component level. The residual stream is the one structural element shared by all three architectures. SAE features extracted from the residual stream are the natural "common currency" for cross-architecture comparison, as demonstrated by the "Towards Universality" paper.

**Challenge 2: Mamba's Representational Collapse in Frozen States**
The "Lost in State Space" paper (May 2026) found severe anisotropy and representational collapse in frozen Mamba-130M representations. This means naive probing of Mamba hidden states may fail.

**Proposed solution**: Use SAE features or ICA on activation patterns rather than raw hidden states. Focus on the residual stream outputs, not the internal SSM states.

**Challenge 3: Mamba's "Off-by-One" Positional Shift**
The "Towards Universality" paper found that Mamba writes token information into the state at the *next* position, creating a systematic positional offset compared to Transformers.

**Proposed solution**: Account for this shift when aligning features across architectures. Use position-aware correlation methods.

**Challenge 4: RWKV Interpretability is Nearly Unexplored**
Very few papers apply mechanistic interpretability to RWKV. No SAE analysis, no circuit discovery, no functional module mapping exists for RWKV.

**Proposed solution**: This is simultaneously a barrier and an opportunity. Being first to do RWKV interpretability at the functional module level would be highly novel. Start with established methods (SAE on residual stream, probing classifiers) before attempting more complex circuit analysis.

**Challenge 5: Scale Mismatches**
Available Mamba models are relatively small (130M-2.8B). RWKV has larger models (up to 14B). Transformer models span a huge range. Functional specialization may not emerge at small scales.

**Proposed solution**: Use comparable model sizes across architectures. Mamba-2.8B, RWKV-7B (or 3B), and a Transformer of similar size (e.g., Pythia-2.8B, Llama-3-3B).

**Challenge 6: Different Training Data**
Models trained on different data may develop different functional organization due to data distribution, not architecture.

**Proposed solution**: Ideally, train all three architectures on the same data. Pythia and Mamba share some training heritage. Alternatively, show that functional modules are consistent *despite* different training data, which would strengthen the convergent evolution argument.

---

## PART 6: ASSESSMENT -- HAS THE SPECIFIC ANGLE BEEN DONE?

### What EXISTS:
1. SAE feature universality across Transformer and Mamba (30%+ matching features) -- "Towards Universality"
2. Functional modules discovered in Transformer-based LLMs (BERT, Llama, GPT-2) -- multiple papers
3. Brain alignment validated across Transformer AND Mamba/RecurrentGemma -- NeurIPS 2025 spotlight
4. Factual knowledge localization compared between Transformer and Mamba -- "Locating and Editing"
5. Brain-like functional organization mapped in Transformers -- Nature Comms 2024, multiple 2025 papers
6. Conceptual framework for brain-machine convergent evolution -- PNAS 2024
7. Theoretical argument for convergent representations -- "Platonic Representation Hypothesis" (ICML 2024)

### What DOES NOT EXIST (our opportunity):
1. **Functional module discovery applied to Mamba and RWKV** (not just Transformers)
2. **Systematic comparison of functional modules across 3+ architecture families** (Transformer vs. Mamba vs. RWKV)
3. **Causal validation (double dissociation) of functional modules in ANY non-Transformer architecture**
4. **Brain alignment of functional modules (not just overall representations) compared across architectures**
5. **RWKV interpretability at the functional module level** -- essentially unexplored
6. **A unified "functional atlas" validated across architectures** -- nobody has produced this
7. **The convergent evolution framing applied to functional modules specifically** (vs. individual features or circuits)

### Novelty Assessment:
- The "Towards Universality" paper comes closest but studies individual features and one circuit, NOT functional modules
- The brain alignment papers compare architectures but NOT at the functional module level
- The functional organization papers study only Transformers
- **The gap at the intersection -- functional modules + multiple architectures + causal validation + brain alignment -- is WIDE OPEN**

---

## PART 7: RISK ASSESSMENT FOR SCOOPING

### HIGH risk:
- Someone extending "Towards Universality" to include functional module clustering (rather than just individual SAE features)
- Someone applying "Brain-like Functional Organization" (currently Transformers only) to Mamba

### MEDIUM risk:
- SAE training on RWKV (no one seems to be working on this publicly)
- Double dissociation experiments on Mamba functional modules

### LOW risk:
- The full unified study: Functional Atlas across 3+ architectures + double dissociation + brain alignment
- The "convergent evolution" framing at the functional module level (as opposed to feature level)

### Recommended timeline:
The field is moving fast. The "Towards Universality" paper (ICLR 2025/2026), NeurIPS 2025 spotlight on cross-architecture brain alignment, and the growing Mamba interpretability literature suggest this space will be crowded within 12-18 months. A 6-9 month execution timeline is advisable.

---

## PART 8: KEY REFERENCES (Complete List)

### Cross-Architecture Universality & Convergence
1. "Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures" -- ICLR 2025/2026. https://arxiv.org/abs/2410.06672
2. "Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders" -- Lan et al., 2024. https://arxiv.org/abs/2410.06981
3. "Semantic Convergence: Investigating Shared Representations Across Scaled LLMs" -- ACL-SRW 2025. https://arxiv.org/abs/2507.22918
4. "The Platonic Representation Hypothesis" -- Huh et al., ICML 2024. https://arxiv.org/abs/2405.07987
5. "A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations" -- Chughtai et al., ICML 2023. https://arxiv.org/abs/2302.03025
6. "Does Transformer Interpretability Transfer to RNNs?" -- AAAI 2025. https://arxiv.org/abs/2404.05971

### Mamba Mechanistic Interpretability
7. "Locating and Editing Factual Associations in Mamba" -- Sen Sharma et al., 2024. https://arxiv.org/abs/2404.03646
8. "Investigating the Indirect Object Identification Circuit in Mamba" -- Ensign & Garriga-Alonso, 2024. https://arxiv.org/abs/2407.14008
9. "The Hidden Attention of Mamba Models" -- ACL 2025. https://arxiv.org/abs/2403.01590
10. "Mamba Knockout for Unraveling Factual Information Flow" -- ACL 2025. https://arxiv.org/abs/2505.24244
11. "Interpreting and Steering State-Space Models via Activation Subspace Bottlenecks" -- Feb 2026. https://arxiv.org/abs/2602.22719
12. "Mechanistic Evaluation of Transformers and State Space Models" -- NeurIPS 2025 submission. https://arxiv.org/abs/2505.15105
13. "Characterizing Mamba's Selective Memory using Auto-Encoders" -- IJCNLP 2025. https://arxiv.org/abs/2512.15653
14. "Lost in State Space: Probing Frozen Mamba Representations" -- May 2026. https://arxiv.org/abs/2605.00253
15. GitHub: mamba-sae (SAE training for Mamba) -- https://github.com/joelburget/mamba-sae

### Brain Alignment Across Architectures
16. "Scaling and Context Steer LLMs Along the Same Computational Path as the Human Brain" -- NeurIPS 2025 Spotlight. https://arxiv.org/abs/2512.01591
17. "Revenge of the Fallen? Recurrent Models Match Transformers at Predicting Human Language Comprehension Metrics" -- COLM 2024. https://arxiv.org/abs/2404.19178
18. "Shared Functional Specialization in Transformer-Based Language Models and the Human Brain" -- Nature Communications 2024. https://www.nature.com/articles/s41467-024-49173-5
19. "The Mind's Transformer: Computational Neuroanatomy of LLM-Brain Alignment" -- ICLR 2026. https://openreview.net/forum?id=PgIlCCNxdB
20. "Training-Driven Representational Geometry Modularization Predicts Brain Alignment" -- Feb 2026. https://arxiv.org/abs/2602.07539
21. "Brain-Language Model Alignment: Insights into the Platonic Hypothesis" -- 2025. https://arxiv.org/abs/2510.17833

### Functional Organization in LLMs
22. "Brain-like Functional Organization within Large Language Models" -- Oct 2024. https://arxiv.org/abs/2410.19542
23. "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models" -- Liu et al., Feb 2025. https://arxiv.org/abs/2502.20408
24. "FNF: Functional Network Fingerprint for Large Language Models" -- Jan 2026. https://arxiv.org/abs/2601.22692
25. "Pruning Large Language Models by Identifying and Preserving Functional Networks" -- 2025. https://arxiv.org/abs/2508.05239
26. "TopoLM: Brain-like Spatio-Functional Organization in a Topographic Language Model" -- Oct 2024. https://arxiv.org/abs/2410.11516
27. "Neuron-Level Knowledge Attribution in Large Language Models" -- EMNLP 2024. https://arxiv.org/abs/2312.12141
28. "Emergent Specialization: Rare Token Neurons in Language Models" -- 2025. https://arxiv.org/abs/2505.12822

### Conceptual Frameworks
29. "Brain-Machine Convergent Evolution: Why Finding Parallels Between Brain and Artificial Systems is Informative" -- PNAS 2024. https://www.pnas.org/doi/10.1073/pnas.2319709121
30. "Dynamics of Specialization in Neural Modules Under Resource Constraints" -- Nature Communications, Jan 2025. https://www.nature.com/articles/s41467-024-55188-9
31. "Brain-like Functional Specialization Emerges Spontaneously in Deep Neural Networks" -- 2021. https://www.biorxiv.org/content/10.1101/2021.07.05.451192

---

## PART 9: STRATEGIC RECOMMENDATIONS FOR THE PAPER

### Recommended Approach:

1. **Common currency for comparison**: Use SAE features extracted from the residual stream (the one shared structural element across all architectures). Do NOT try to compare attention heads to SSM components directly.

2. **Architecture selection**: Transformer (Pythia-2.8B or Llama-3-3B), Mamba (2.8B), RWKV-v6 (3B or 7B). If possible, include a hybrid (Jamba or Mamba-3) as a control.

3. **Module discovery method**: Apply ICA or NMF to activation patterns (as in papers #22, #23), or cluster SAE features (as conceptually suggested by #1). The SAE approach is more principled and allows direct cross-architecture feature matching.

4. **Convergent evolution narrative**: Frame as analogous to PNAS 2024 (#29): different architectures evolving similar functional solutions is evidence of optimality, just as different species evolving similar structures (eyes, wings) is evidence of adaptive optima.

5. **Key experiment that nobody has done**: Train SAEs on Transformer, Mamba, and RWKV. Cluster SAE features into functional modules. Show that corresponding modules exist across all three. Ablate modules and show double dissociation (function-specific impairment). Map modules to brain regions and show similar brain alignment across architectures.

### What This Adds Beyond Existing Work:
- "Towards Universality" showed feature-level similarity between 2 architectures. We would show **module-level organization** across 3+ architectures.
- "Brain-like Functional Organization" showed brain-aligned modules in Transformers. We would show this is **architecture-invariant**.
- "Scaling and Context" showed brain alignment is cross-architectural. We would show this extends to **causal, functional modules**, not just overall representations.
- The PNAS convergent evolution paper provided the theory. We would provide the **first comprehensive empirical validation for language models**.

---

## BOTTOM LINE

**The convergent functional organization angle is feasible and highly novel.** The pieces exist -- feature universality, Mamba interpretability tools, brain alignment across architectures, functional module discovery in Transformers -- but nobody has assembled them into a unified study. The specific combination of "functional modules + 3 architectures + causal validation + brain alignment" has NOT been done. This is the key gap that makes our proposed paper Nature-level.

**Biggest risk**: The "Towards Universality" team or the "Brain-like Functional Organization" team extending their work to cover this ground. Execution speed matters.

**Biggest technical challenge**: RWKV interpretability is nearly unexplored. Mamba hidden states can suffer from representational collapse. Working at the SAE-feature level on the residual stream is the safest methodological path.
