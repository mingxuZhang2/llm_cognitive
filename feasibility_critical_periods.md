# Feasibility Analysis: Critical Periods and Phase Transitions in Functional Module Formation During LLM Training

## Executive Summary

This is an **active and rapidly growing research area** with substantial existing work, but a critical gap remains: **no paper has systematically tracked the formation of functional modules (brain-region-like specialized subsystems) across LLM pretraining checkpoints and identified critical periods specifically for module formation**. Individual pieces of this puzzle exist across ~30+ papers, but they have not been integrated. Below is an exhaustive accounting.

---

## 1. DIRECTLY RELEVANT PAPERS (Highest Overlap with Proposed Work)

### 1.1 Papers Tracking Internal Organization Changes Across Training Checkpoints

**"LLM Circuit Analyses Are Consistent Across Training and Scale"** (Tigges, Hanna et al., NeurIPS 2024)
- **What they did:** Tracked circuit evolution in Pythia models (70M-2.8B) across 300B training tokens. Examined Successor Heads, Induction Heads, Copy Suppression Heads, and Name-Mover Heads.
- **Key finding:** Critical attention head types emerge at remarkably similar token counts (2-8B tokens) regardless of model size. The overarching algorithm remains consistent even when specific implementing heads change.
- **Gap relative to our work:** Tracked individual circuit *types*, not the formation of spatial/functional *modules* (i.e., brain-region-like organized subsystems). No critical period analysis for modular organization.

**"Tracking the Feature Dynamics in LLM Training: A Mechanistic Study"** (Xu, Wang, Wang, Dec 2024)
- **What they did:** Trained SAEs on sequential training checkpoints (SAE-Track). Studied semantic evolution of features, feature formation processes, and directional drift of feature vectors.
- **Key finding:** Feature formation involves geometric convergence into localized regions. Feature directions undergo three-phase adjustments, with drift persisting even after semantic formation. Full stabilization occurs only late in training.
- **Gap:** Tracked individual *features*, not organized functional *modules*. No critical period framing.

**"The Birth of Knowledge: Emergent Features across Time, Space, and Scale in Large Language Models"** (Sawmya, Adler, Shavit, May 2025)
- **What they did:** Used SAEs to study when and where semantic concepts emerge across training checkpoints (time), transformer layers (space), and model sizes (scale).
- **Key finding:** Clear temporal and scale-specific thresholds for feature emergence. Unexpected semantic reactivation: early-layer features re-emerge at later layers.
- **Gap:** Tracked feature emergence timing, but not *modular organization* of features into functional groups. No critical period analysis.

**"Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining"** (Bayazit, Mueller, Bosselut, Sep 2025)
- **What they did:** Used sparse crosscoders to learn a joint feature space across model checkpoints. Tracked feature emergence, maintenance, and discontinuation in Pythia, BLOOM, and OLMo.
- **Key finding:** Crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Shared features signal maintained concepts; unique features mark emerging/vanishing concepts.
- **Gap:** Tracked individual linguistic features, not functional modules. No critical period analysis for organizational structure.

**"How LLMs Learn: Tracing Internal Representations with Sparse Autoencoders"** (Mar 2025)
- **What they did:** Trained SAEs at multiple checkpoints. Tracked progression from random fragments to language-specific meanings to higher-level cross-lingual semantics.
- **Key finding:** Features progress from token-level to concept-level representations, forming increasingly abstract knowledge structures.
- **Gap:** Characterized learning trajectory but not module formation or critical periods.

### 1.2 Papers on Functional Specialization Emergence

**"How Syntax Specialization Emerges in Language Models"** (arXiv:2505.19548, May 2025)
- **What they did:** Introduced a Syntactic Selectivity Index (SSI) to track how syntactic sensitivity emerges across training. Studied the developmental trajectory of syntax-specific internal specialization.
- **Key finding:** Identified a clear developmental trajectory with syntactic sensitivity emerging gradually, concentrating in specific layers, and exhibiting a **"critical period" of rapid internal specialization**. Consistent across architectures and initializations; influenced by model scale and training data.
- **OVERLAP ALERT:** This paper comes closest to our proposed angle. However, it tracks ONE type of specialization (syntax) using a selectivity index, not multi-function modular organization. It identifies a critical period for syntax specialization specifically, not for broader functional module formation.

**"Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs"** (Chen et al., ICLR 2024, arXiv:2309.07311)
- **What they did:** Studied Syntactic Attention Structure (SAS) in MLMs during training. Identified a brief window where models abruptly acquire SAS, concurrent with a steep loss drop.
- **Key finding:** SAS acquisition is abrupt and causally necessary for subsequent linguistic capabilities. Manipulating SAS during training demonstrates its necessity.
- **Gap:** Studied ONE functional property (syntactic attention) emergence, not formation of multiple interacting functional modules.

**"Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficient"** (Wang et al., Timaeus Research, Oct 2024)
- **What they did:** Applied refined Local Learning Coefficients (rLLCs) from singular learning theory to track how attention heads differentiate and specialize during training in a 2-layer attention-only transformer.
- **Key finding:** Heads initially have similar rLLC curves that progressively diverge. Data structure drives specialization. Discovered a previously unidentified multigram circuit.
- **OVERLAP ALERT:** This paper directly studies how functional components differentiate during training. But it is limited to a toy 2-layer model. Does not study modular *grouping* or brain-like regional organization.

**"Emergent Specialization: Rare Token Neurons in Language Models"** (Liu, ICML 2025, arXiv:2505.12822)
- **What they did:** Identified neurons with exceptionally strong influence on rare token prediction. Studied their emergence dynamics.
- **Key finding:** Rare token neurons exhibit a three-phase organization (plateau, power-law, rapid decay) emerging dynamically during training. They form coordinated subnetworks with selective co-activation.
- **Gap:** Studied one type of neuron specialization and its subnetwork, not broad modular structure formation.

**"Start Making Sense(s): A Developmental Probe of Attention Specialization Using Lexical Ambiguity"** (Riviere & Trott, TACL forthcoming, Nov 2025)
- **What they did:** Used lexical ambiguity to probe attention head specialization across Pythia checkpoints. Identified inflection points in disambiguation performance.
- **Key finding:** Found specific heads whose attention patterns covary with disambiguation performance across development, with identifiable inflection points.
- **Gap:** Probed one functional capability, not modular organization.

**"Specialization of softmax attention heads: insights from the high-dimensional single-location model"** (Sagitova, Duranthon, Zdeborova, 2026)
- **What they did:** Theoretical model of multi-head attention specialization dynamics.
- **Key finding:** Head specialization emerges in distinct stages: initial unspecialized phase followed by multi-stage specialization where heads sequentially align with latent signal directions. Many heads remain redundant.
- **Gap:** Theoretical framework, not empirical tracking of module formation in real LLMs.

### 1.3 Papers on Emergent Modularity in LLMs

**"Unlocking Emergent Modularity in Large Language Models"** (Qiu et al., NAACL 2024, arXiv:2310.10908)
- **What they did:** Demonstrated that implicit modular structures exist in standard pre-trained transformers (Emergent Modularity). Found strong correlation between neuron activation and specific tasks.
- **Key finding:** Modular structures spontaneously exhibit during the **early pre-training phase**. Function-based neuron grouping is present with neurons having similar functions usually co-activated.
- **CRITICAL:** This paper demonstrates that modular structure *exists* in LLMs and notes it appears early in pretraining, but does NOT systematically track its formation dynamics or identify critical periods.

**"Discovering Decoupled Functional Modules in Large Language Models"** (Yu et al., 2025, arXiv:2603.17823)
- **What they did:** Proposed ULCMOD framework to discover functional modules in LLMs via unsupervised cross-layer module discovery.
- **Key finding:** Discovered modules show function comprehensiveness, function hierarchy, and clear spatial arrangement within LLMs.
- **Gap:** Analyzes module structure in trained models; does not track formation across training.

**"Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models"** (Liu et al., Feb 2025, arXiv:2502.20408)
- **What they did:** Applied functional brain network (FBN) methodology from neuroscience to explore functional networks within LLMs.
- **Key finding:** LLMs exhibit certain recurring functional networks that are indispensable for performance. Inhibiting key functional networks severely impairs capabilities; amplifying can enhance performance.
- **Gap:** Applied neuroscience methodology to *trained* LLMs but did not track formation dynamics during training.

**"Spontaneous Functional Differentiation in Large Language Models: A Brain-Like Intelligence Economy"** (Zhang et al., CAS, Mar 2026, arXiv:2603.29735)
- **What they did:** Studied how synergistic cores and functional differentiation emerge in LLMs.
- **Key finding:** LLMs spontaneously develop synergistic cores where information integration exceeds individual parts, similar to the human brain. This organization is dynamic and emerges as a physical phase transition as task difficulty increases. Ablating synergistic components causes catastrophic performance loss.
- **OVERLAP:** Studies functional differentiation with brain analogies, but focuses on inference-time dynamics rather than training-time module formation.

---

## 2. CRITICAL PERIOD RESEARCH IN NEURAL NETWORKS

### 2.1 Critical Learning Periods (General)

**"Critical Learning Periods in Deep Neural Networks"** (Achille, Rovere, Soatto, ICLR 2019)
- **Foundational paper.** Showed deep networks exhibit critical periods analogous to biological systems. Early visual deficits (e.g., blur) cause irreversible performance loss. First few epochs create strong connections that are "locked in."
- **Key finding:** Information rises rapidly early in training then decreases, preventing redistribution. Critical periods arise from information processing, not biochemistry.

**"Critical Learning Periods for Multisensory Integration in Deep Networks"** (Kleinman et al., 2023)
- Extended Achille et al. to show DNNs also have critical learning periods for multisensory integration, with deficits early in training affecting both representations and behavior.

**"On the Occurrence of Critical Learning Periods in Neural Networks"** (Pawlak, Oct 2025, arXiv:2510.09687)
- **What they did:** Showed critical learning periods can be avoided through cyclic learning rate schedules.
- **Key finding:** Critical period effects emerge from deficit data + plasticity loss, and can be mitigated by hyperparameter adjustment.

**"One Period to Rule Them All: Identifying Critical Learning Periods in Deep Networks"** (Fukase et al., Jun 2025, arXiv:2506.15954)
- Systematic approach for identifying critical periods. Used for efficient training (2.5x speedup). Focused on data augmentation and regularization effectiveness windows.

### 2.2 Critical Periods Specific to Language Models

**"Investigating Critical Period Effects in Language Acquisition through Neural Language Models"** (Constantinescu et al., TACL 2025, arXiv:2407.19325)
- **What they did:** Tested whether LMs exhibit critical period effects for L2 acquisition analogous to humans. Varied age of L2 exposure.
- **Key finding:** LMs do NOT naturally show critical period effects when L2 exposure is delayed. However, critical periods CAN be engineered via Elastic Weight Consolidation (EWC) regularization simulating plasticity decrease.
- **Important distinction:** This studied critical periods for *task performance* (language acquisition), not for *internal functional organization*.

### 2.3 Loss of Plasticity

**"Loss of plasticity in deep continual learning"** (Dohare et al., Nature 2024)
- Deep learning methods gradually lose plasticity in continual learning, eventually learning no better than shallow networks. Highly relevant context for understanding why critical periods might exist.

---

## 3. PHASE TRANSITION RESEARCH DURING LLM TRAINING

**"Triple Phase Transitions: Understanding the Learning Dynamics of Large Language Models from a Neuroscience Perspective"** (Nakagi et al., Feb 2025, arXiv:2502.20779)
- **What they did:** Studied three interconnected perspectives: LLM-brain similarity, internal states, and downstream task performance across training.
- **Key finding:** Three phase transitions emerge: (1) brain alignment surges with instruction following, (2) LLMs diverge from brain during performance stagnation, (3) brain alignment reoccurs with task mastery.
- **OVERLAP:** Explicitly uses neuroscience framing for training dynamics. But tracks brain *alignment* trajectories, not internal module formation per se.

**"Evidence of Phase Transitions in Small Transformer-Based Language Models"** (Hong et al., Nov 2025, arXiv:2511.12768)
- Phase-transition-like reorganizations occur even in small transformers and arise surprisingly early, with lexical coherence serving as an order parameter.

**"Phase Transitions in Large Language Models and the O(N) Model"** (Sun & Haghighat, Jan 2025, arXiv:2501.16241)
- Reformulated Transformer as O(N) model from statistical physics. Identified two distinct phase transitions (temperature-dependent and parameter-dependent).

**"In-context Learning and Induction Heads"** (Olsson et al., Anthropic, 2022)
- Demonstrated a previously unknown phase change in transformer training: induction heads form abruptly concurrent with sharp increases in in-context learning.

**"Progress measures for grokking via mechanistic interpretability"** (Nanda et al., ICLR 2023)
- Identified three continuous training phases in grokking: memorization, circuit formation, and cleanup. The generalizing circuit forms gradually before becoming dominant.

---

## 4. DEVELOPMENTAL TRAJECTORY & LEARNING ORDER RESEARCH

**"The Grammar-Learning Trajectories of Neural Language Models"** (arXiv:2109.06096, 2021)
- Models with different initialization/architecture/data acquire linguistic phenomena in a similar order. Learning trajectory is approximately one-dimensional.

**"What do Language Models Learn and When? The Implicit Curriculum Hypothesis"** (Liu et al., Apr 2026, arXiv:2604.08510)
- **Key finding:** Pretraining follows a compositional and predictable "implicit curriculum." Emergence orderings are strikingly consistent (rho=0.81 across 45 model pairs). Composite tasks emerge after component tasks.

**"Language acquisition: do children and language models follow similar learning stages?"** (Evanson et al., ACL Findings 2023)
- Some but not all learning stages shared between children and LMs. LMs learn skills in systematic but parallel order.

**"Is neural language acquisition similar to natural? A chronological probing study"** (Voloshina et al., 2022)
- Linguistic information acquired in first 600K steps. Morphology/syntax show similar patterns; discourse significantly different.

**"Vocabulary embeddings organize linguistic structure early in language model training"** (Papadimitriou & Prince, Oct 2025)
- Vocabulary embedding geometry converges to semantic/syntactic structure early in training (~15% of training). High-frequency/function words stabilize faster.

**"Loss Landscape Degeneracy and Stagewise Development in Transformers"** (Feb 2024)
- In-context learning emerges in discrete developmental stages. Local Learning Coefficient detects hidden developmental stages invisible in loss curves.

---

## 5. MODULARITY IN NEURAL NETWORKS (Not LLM-Specific)

**"Growing Brains: Co-emergence of Anatomical and Functional Modularity in Recurrent Neural Networks"** (Liu, Khona, Fiete, Tegmark, NeurIPS 2023)
- Applied brain-inspired modular training (BIMT) to RNNs. Functional and anatomical clustering co-emerge during training on compositional tasks. Functionally similar neurons become spatially localized.
- **Gap:** RNNs on cognitive tasks, not LLMs on language.

**"Brain-like functional specialization emerges spontaneously in deep neural networks"** (Dobs et al., Science Advances 2022)
- Networks optimized for multiple visual tasks spontaneously segregate into separate specialized subsystems, paralleling brain functional specialization. No built-in task-specific inductive biases needed.
- **Gap:** Vision CNNs, not language models.

**"Dynamics of specialization in neural modules under resource constraints"** (Bena & Goodman, Nature Communications 2025)
- Structural modularity does not guarantee functional specialization. Specialization requires separable environments AND resource constraints. Specialization varies dynamically across time.
- **Gap:** Simple artificial networks, not LLMs.

**"Quantifying Local Specialization in Deep Neural Networks"** (Hod et al., 2021)
- Proposed importance and coherence proxies for local specialization. Graph-based partitioning can reveal modularity. Methods for understanding modular structure.

---

## 6. BRAIN-LLM ALIGNMENT & NEUROSCIENCE-INSPIRED INTERPRETABILITY

**"Shared functional specialization in transformer-based language models and the human brain"** (Kumar et al., Nature Communications 2024)
- Individual attention heads differentially predict brain activity in specific cortical regions. Establishes direct parallel between LLM components and brain functional specialization.

**"Training-Driven Representational Geometry Modularization Predicts Brain Alignment in Language Models"** (Feb 2026, arXiv:2602.07539)
- **HIGHLY RELEVANT.** Tracked geometric modularization across Pythia training. Found layers self-organize into stable low- and high-complexity clusters. Low-complexity module better predicts human brain activity. Brain alignment follows heterogeneous spatial-temporal trajectories: rapid in temporal regions, delayed in frontal areas.
- **This is very close to our concept** but frames it as geometry modularization predicting brain alignment, not as a standalone study of functional module formation and critical periods.

**"Stroke Lesions as a Rosetta Stone for Language Model Interpretability"** (Fridriksson et al., Feb 2026, arXiv:2602.04074)
- Used human lesion-symptom mapping as external reference for LLM interpretability. LLM error profiles sufficiently similar to human profiles that predicted lesions corresponded to actual lesions.

**"Emergence of a High-Dimensional Abstraction Phase in Language Transformers"** (ICLR 2024)
- A distinct phase with high intrinsic dimensionality corresponds to the first full linguistic abstraction. Earlier onset predicts better language modeling.

---

## 7. ADDITIONAL RELEVANT WORK

**"Emergent Structures and Training Dynamics in Large Language Models"** (Teehan et al., BigScience Workshop 2022)
- Survey paper that **explicitly identified the research gap**: "lack of sufficient research on the emergence of functional units---subsections of the network where related functions are grouped or organized---within large language models."
- **This 2022 call for research has still not been fully answered as of 2026.**

**"Linear Predictability of Attention Heads in Large Language Models"** (Mar 2026, arXiv:2603.13314)
- Inter-head linear structure (QKV reconstruction from peer heads) is absent at initialization and rises rapidly during pretraining. Evidence that organizational structure emerges through training.

**Pythia Model Suite** (Biderman et al., ICML 2023)
- 16 models (70M-12B) with 154 checkpoints each, trained on identical data in identical order. Designed specifically for studying learning dynamics and interpretability across training.

**"How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training"** (Feb 2025)
- Knowledge circuit evolution exhibits a distinct phase shift from formation to optimization, following a deep-to-shallow pattern during continual pretraining.

---

## 8. GAP ANALYSIS: What Has and Has NOT Been Done

### What HAS been done:
1. **Individual circuit/feature emergence timing** -- well-studied (induction heads, SAS, rare token neurons, individual SAE features)
2. **Phase transitions in loss/capability** -- well-studied (grokking, emergent abilities, loss bumps)
3. **Critical periods for task performance** -- studied (Achille et al., Constantinescu et al.)
4. **Existence of modular structure in trained LLMs** -- demonstrated (EMoE, ULCMOD, functional networks)
5. **Brain-LLM alignment across training** -- studied (Triple Phase Transitions, geometry modularization)
6. **Individual attention head specialization dynamics** -- studied (rLLC paper, softmax attention theory)
7. **Learning order/implicit curriculum** -- well-studied

### What has NOT been done (THE GAP):
1. **Systematic tracking of when functional MODULES (not individual features/circuits) form during pretraining** -- No paper has tracked the emergence of brain-region-like organized subsystems across training checkpoints using module-discovery methods applied at each checkpoint.
2. **Critical periods specifically for modular organization** -- No paper asks: "Is there a critical window during which the modular architecture of the LLM is established, after which it becomes locked in?"
3. **Multi-function module formation dynamics** -- Papers track individual capabilities (syntax, semantics, rare tokens). Nobody tracks how MULTIPLE functional modules simultaneously differentiate, interact, and stabilize.
4. **Applying neuroscience developmental methodology (critical period theory) to LLM internal organization** -- The brain-LLM alignment papers study similarity to the brain at static points or track alignment; they don't apply the *developmental neuroscience* framework of critical periods to module formation.
5. **Perturbation experiments during module formation** -- Analogous to biological critical period experiments (e.g., deprivation during critical windows), nobody has tested whether disrupting training during specific windows permanently alters modular organization.

### The closest paper:
**"Training-Driven Representational Geometry Modularization Predicts Brain Alignment in Language Models"** (2026) comes closest by tracking geometric modularization across Pythia training and finding self-organization into clusters. However, it (a) frames this as brain alignment prediction rather than developmental biology, (b) does not identify critical periods, and (c) does not perform perturbation experiments.

---

## 9. FEASIBILITY ASSESSMENT

### Strengths of the Proposed Research Direction:
- **Clear, unfilled gap** confirmed by exhaustive search
- **Infrastructure exists**: Pythia, OLMo checkpoints make this immediately feasible
- **Methods exist**: SAEs, rLLCs, module discovery (ULCMOD, EMoE), brain alignment metrics
- **Conceptual frameworks exist**: Critical period theory (Achille), developmental stages (Hoogland), brain-LLM alignment
- **High novelty**: Combining module-discovery + training dynamics + critical period perturbation experiments = entirely new
- **Nature-level potential**: Bridges AI interpretability, developmental neuroscience, and statistical physics

### Risks:
- "Training-Driven Representational Geometry Modularization" (2026) is very close and could be extended quickly by its authors
- "How Syntax Specialization Emerges in Language Models" (2025) already uses "critical period" language for one functional dimension
- Multiple groups are converging on this area from different directions -- speed matters

### Recommendation:
**The gap is real and substantial.** The specific combination of (1) tracking multi-function module formation dynamics, (2) identifying critical periods for modular organization, and (3) conducting perturbation experiments during those windows has NOT been done. This is a strong, publishable research direction. However, the field is converging quickly, so execution speed is important.

---

## Sources

### Directly Tracking Training Dynamics
- [LLM Circuit Analyses Are Consistent Across Training and Scale](https://arxiv.org/abs/2407.10827) (Tigges et al., NeurIPS 2024)
- [Tracking the Feature Dynamics in LLM Training](https://arxiv.org/abs/2412.17626) (Xu et al., Dec 2024)
- [The Birth of Knowledge: Emergent Features across Time, Space, and Scale](https://arxiv.org/abs/2505.19440) (Sawmya et al., May 2025)
- [Crosscoding Through Time](https://arxiv.org/abs/2509.05291) (Bayazit et al., Sep 2025)
- [How LLMs Learn: Tracing Internal Representations with SAEs](https://arxiv.org/abs/2503.06394) (Mar 2025)

### Functional Specialization Emergence
- [How Syntax Specialization Emerges in Language Models](https://arxiv.org/abs/2505.19548) (May 2025)
- [Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions](https://arxiv.org/abs/2309.07311) (Chen et al., ICLR 2024)
- [Differentiation and Specialization of Attention Heads via rLLC](https://arxiv.org/abs/2410.02984) (Wang et al., Oct 2024)
- [Emergent Specialization: Rare Token Neurons](https://arxiv.org/abs/2505.12822) (Liu, ICML 2025)
- [Start Making Sense(s): Developmental Probe of Attention Specialization](https://arxiv.org/abs/2511.21974) (Riviere & Trott, Nov 2025)
- [Specialization of softmax attention heads](https://arxiv.org/pdf/2603.03993) (Sagitova et al., 2026)

### Emergent Modularity in LLMs
- [Unlocking Emergent Modularity in Large Language Models](https://arxiv.org/abs/2310.10908) (Qiu et al., NAACL 2024)
- [Discovering Decoupled Functional Modules in LLMs](https://arxiv.org/pdf/2603.17823) (Yu et al., 2025)
- [Brain-Inspired Exploration of Functional Networks and Key Neurons](https://arxiv.org/abs/2502.20408) (Liu et al., Feb 2025)
- [Spontaneous Functional Differentiation in LLMs](https://arxiv.org/abs/2603.29735) (Zhang et al., Mar 2026)

### Critical Periods
- [Critical Learning Periods in Deep Neural Networks](https://arxiv.org/pdf/1711.08856) (Achille et al., ICLR 2019)
- [Investigating Critical Period Effects in Language Acquisition through Neural LMs](https://arxiv.org/abs/2407.19325) (Constantinescu et al., TACL 2025)
- [On the Occurrence of Critical Learning Periods in Neural Networks](https://arxiv.org/abs/2510.09687) (Pawlak, Oct 2025)
- [One Period to Rule Them All](https://arxiv.org/abs/2506.15954) (Fukase et al., Jun 2025)
- [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7) (Dohare et al., Nature 2024)

### Phase Transitions
- [Triple Phase Transitions](https://arxiv.org/abs/2502.20779) (Nakagi et al., Feb 2025)
- [Evidence of Phase Transitions in Small Transformers](https://arxiv.org/abs/2511.12768) (Hong et al., Nov 2025)
- [Phase Transitions in LLMs and the O(N) Model](https://arxiv.org/abs/2501.16241) (Sun & Haghighat, Jan 2025)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Olsson et al., Anthropic 2022)
- [Progress measures for grokking](https://arxiv.org/abs/2301.05217) (Nanda et al., ICLR 2023)

### Developmental Trajectories
- [The Grammar-Learning Trajectories of Neural Language Models](https://arxiv.org/abs/2109.06096) (2021)
- [What do Language Models Learn and When?](https://arxiv.org/abs/2604.08510) (Liu et al., Apr 2026)
- [Language acquisition: do children and LMs follow similar learning stages?](https://aclanthology.org/2023.findings-acl.773/) (Evanson et al., ACL 2023)
- [Is neural language acquisition similar to natural?](https://arxiv.org/abs/2207.00560) (Voloshina et al., 2022)
- [Vocabulary embeddings organize linguistic structure early](https://arxiv.org/abs/2510.07613) (Papadimitriou & Prince, Oct 2025)
- [Loss Landscape Degeneracy and Stagewise Development](https://arxiv.org/abs/2402.02364) (Feb 2024)

### Modularity (Non-LLM)
- [Growing Brains: Co-emergence of Anatomical and Functional Modularity](https://arxiv.org/abs/2310.07711) (Liu et al., NeurIPS 2023)
- [Brain-like functional specialization emerges spontaneously](https://www.science.org/doi/10.1126/sciadv.abl8913) (Dobs et al., Science Advances 2022)
- [Dynamics of specialization in neural modules under resource constraints](https://www.nature.com/articles/s41467-024-55188-9) (Bena & Goodman, Nature Communications 2025)
- [Quantifying Local Specialization in Deep Neural Networks](https://arxiv.org/abs/2110.08058) (Hod et al., 2021)

### Brain-LLM Alignment
- [Shared functional specialization in transformer-based LMs and the human brain](https://www.nature.com/articles/s41467-024-49173-5) (Kumar et al., Nature Communications 2024)
- [Training-Driven Representational Geometry Modularization Predicts Brain Alignment](https://arxiv.org/abs/2602.07539) (Feb 2026)
- [Stroke Lesions as a Rosetta Stone for Language Model Interpretability](https://arxiv.org/abs/2602.04074) (Fridriksson et al., Feb 2026)
- [Emergence of a High-Dimensional Abstraction Phase](https://arxiv.org/abs/2405.15471) (ICLR 2024)

### Infrastructure
- [Pythia: A Suite for Analyzing LLMs Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., ICML 2023)
- [Emergent Structures and Training Dynamics in LLMs](https://aclanthology.org/2022.bigscience-1.11/) (Teehan et al., BigScience 2022)
- [Linear Predictability of Attention Heads in LLMs](https://arxiv.org/abs/2603.13314) (Mar 2026)
- [How much pretraining data do LMs need to learn syntax?](https://arxiv.org/abs/2109.03160) (Perez-Mayos et al., EMNLP 2021)
- [Emergence of Minimal Circuits for IOI](https://arxiv.org/abs/2510.25013) (Adhikari, Oct 2025)
- [How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective](https://arxiv.org/pdf/2502.11196) (Feb 2025)
