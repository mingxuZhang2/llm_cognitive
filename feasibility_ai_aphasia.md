# Feasibility Analysis: "AI Neuropathology" -- Mapping LLM Ablation Patterns to Clinical Neurological Syndromes

**Date:** 2026-05-19
**Research Question:** Has anyone systematically studied whether damaging/ablating specific functional regions in LLMs produces deficit patterns that resemble clinical neurological syndromes (aphasia, semantic dementia, amnesia, etc.)?

---

## EXECUTIVE SUMMARY

**This exact idea has been explored, and recently with increasing sophistication -- but the field is nascent (most papers are 2025-2026 preprints) and the systematic "AI neuropathology" framing as a comprehensive discipline is NOT yet established.** The closest papers focus almost exclusively on aphasia simulation in LLMs. No one has yet produced a comprehensive "neuropathology atlas" covering the full range of clinical syndromes (amnesia, agnosia, semantic dementia, executive dysfunction, etc.) mapped onto LLM functional architecture. There is a clear gap between what exists and a full "AI Neuropathology" framework.

**Verdict: The specific angle of aphasia simulation via LLM lesioning is now CROWDED (4-5 strong papers in 2025-2026). But a broader "AI Neuropathology" framework covering multiple syndrome categories beyond aphasia remains wide open.**

---

## TIER 1: DIRECTLY ON-TOPIC PAPERS (LLM Lesioning -> Clinical Syndrome Patterns)

### 1. "Stroke Lesions as a Rosetta Stone for Language Model Interpretability" (Feb 2026)
- **Authors:** Julius Fridriksson + 27 others
- **Venue:** arXiv 2602.04074
- **What they did:** Created BLUM (Brain-LLM Unified Model) -- a framework that applies systematic perturbations to transformer layers, then administers IDENTICAL clinical assessments (picture naming, sentence completion) to both perturbed LLMs and N=410 human stroke patients. Projects LLM error profiles into human lesion space.
- **Key finding:** LLM error profiles were sufficiently similar to human error profiles that predicted lesions corresponded to actual lesions in error-matched humans above chance in 67% of picture naming conditions and 68.3% of sentence completion conditions. Semantic-dominant errors mapped onto ventral-stream lesion patterns; phonemic-dominant errors mapped onto dorsal-stream patterns.
- **How close to "AI Neuropathology":** VERY CLOSE. This is the single most directly relevant paper. They use human clinical data as ground truth for validating LLM perturbation effects. However, they focus only on language/aphasia, not broader cognitive syndromes.
- **URL:** https://arxiv.org/abs/2602.04074

### 2. "Component-Level Lesioning of Language Models Reveals Clinically Aligned Aphasia Phenotypes" (Jan 2026)
- **Authors:** (Same research group as Bridging Brains paper, extended)
- **Venue:** arXiv 2601.19723
- **What they did:** Clinically grounded, component-level framework that simulates aphasia by selectively perturbing functional components in LLMs. Applied to both MoE models and dense Transformers. Used Western Aphasia Battery (WAB) subtests for evaluation. Identified subtype-linked components for Broca's and Wernicke's aphasia.
- **Key finding:** Subtype-targeted perturbations yield more systematic, aphasia-like regressions than size-matched random perturbations. MoE modularity supports more localized and interpretable phenotype-to-component mappings.
- **How close:** VERY CLOSE. Directly maps LLM component ablation to clinical aphasia subtypes using standardized clinical instruments.
- **URL:** https://arxiv.org/abs/2601.19723

### 3. "Bridging Brains and Models: MoE-Based Functional Lesions for Simulating and Rehabilitating Aphasia" (Aug 2025)
- **Authors:** (Research group focusing on MoE models)
- **Venue:** arXiv 2508.04749
- **What they did:** Selectively disabled components in a modular MoE language model. Simulated distinct aphasia subtypes and validated linguistic outputs against real patient speech. Also investigated functional recovery by retraining remaining healthy experts.
- **Key finding:** Lesioning functionally-specialized experts for syntax induces Broca's-like deficits; lesioning semantics experts induces Wernicke's-like deficits. Freezing damaged experts and retraining intact ones restores significant linguistic function (computational analogue for rehabilitation).
- **How close:** VERY CLOSE. Goes beyond mere ablation to model rehabilitation/recovery.
- **URL:** https://arxiv.org/abs/2508.04749

### 4. "Emergent Modularity in Large Language Models: Insights from Aphasia Simulations" (Feb 2025)
- **Authors:** Chengcheng Wang, Zhiyu Fan, Zaizhu Han, Yanchao Bi, Jixing Li
- **Venue:** bioRxiv 2025.02.22.639416
- **What they did:** Systematically disrupted LLM components to simulate aphasia behavioral profiles. Tested whether LLMs exhibit distinct functional modules for semantic vs. syntactic processing.
- **Key finding:** Semantic deficits (Wernicke's, Conduction aphasia) were relatively straightforward to simulate. Reproducing syntactic and lexical impairments (Broca's, Anomic aphasia) proved more challenging. Highlights both parallels and discrepancies between emergent LLM modularity and human language systems.
- **How close:** VERY CLOSE. Directly tests aphasia subtypes but acknowledges asymmetric success.
- **URL:** https://www.biorxiv.org/content/10.1101/2025.02.22.639416v1

### 5. "Comparison of Large Language Model with Aphasia" (May 2025)
- **Authors:** Takamitsu Watanabe, Katsuma Inoue, Yasuo Kuniyoshi, Kohei Nakajima, Kazuyuki Aihara (University of Tokyo)
- **Venue:** Advanced Science 12:2414016
- **What they did:** Compared network dynamics between LLMs (ALBERT, GPT-2, Llama-3.1, LLM-jp-3) and various aphasic human brains using energy landscape analysis. Quantified transition frequency and dwelling time of network activity patterns.
- **Key finding:** LLM dynamics indices were located close to those for Wernicke's aphasia. Receptive aphasia shows bimodal distributions; expressive aphasia exhibits uniform distributions.
- **How close:** VERY CLOSE. Directly compares internal dynamics of LLMs to aphasic brains, but approaches from the opposite direction (characterizing LLMs' existing behavior as aphasia-like, rather than inducing it).
- **URL:** https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202414016

### 6. "Neural Erosion: Emulating Controlled Neurodegeneration and Aging in AI Systems" (Mar 2024)
- **Authors:** (Not specified in search results)
- **Venue:** arXiv 2403.10596
- **What they did:** Introduced "neural erosion" concept -- ablating synapses or neurons, adding Gaussian noise during/after training in LLaMA 2. Used IQ tests to measure progressive decline.
- **Key finding:** LLMs first lose abstract thinking, then mathematical abilities, then linguistic abilities, finally becoming incoherent -- mirroring the cognitive decline pattern observed in human neurodegenerative studies.
- **How close:** MODERATELY CLOSE. Maps overall degradation trajectory to human neurodegeneration but does not map specific syndromes to specific ablation targets.
- **URL:** https://arxiv.org/abs/2403.10596

### 7. "An LLM as a Virtual Simulation Model for Language-Eloquent Area Surgery in Human" (Oct 2025)
- **Authors:** Sun Mo Nam et al.
- **Venue:** Medical Hypotheses (ScienceDirect)
- **What they did:** Proposed that neuron masking in LLMs may simulate brain lesions and predict language deficits for neurosurgical planning. Conducted neuron masking experiments using Qwen2.5 models.
- **Key finding:** Layer-specific vulnerability patterns show similarities to neurofunctional anatomical organization. Middle layers showed semantic processing deficits resembling ventral stream damage; late layers showed executive dysfunction.
- **How close:** MODERATELY CLOSE. Hypothesis paper with proof of concept. Explicitly frames LLMs as surgical simulation tools.
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0306987725002312

### 8. "Transformer Language Models Reveal Distinct Patterns in Aphasia Subtypes and Recovery Trajectories" (Mar 2026)
- **Venue:** bioRxiv 2026.03.27.714240
- **What they did:** Used GPT-2 to analyze activation patterns in aphasic narrative speech. Examined longitudinal recovery.
- **Key finding:** Statistically significant differences in activation magnitude across aphasia subtypes at every layer, with most pronounced effects in deeper layers. Broca's aphasia was consistently distinguished from Anomic and Wernicke's aphasia.
- **How close:** MODERATELY CLOSE. Uses LLMs as diagnostic tools for human aphasia rather than simulating aphasia in the LLM itself.
- **URL:** https://www.biorxiv.org/content/10.64898/2026.03.27.714240v1

---

## TIER 2: CLOSELY RELATED PAPERS (LLM Functional Organization / Non-Aphasia Syndromes)

### 9. "Inducing Dyslexia in Vision Language Models" (Sep 2025)
- **Authors:** Melika Honarmand, Ayati Sharma, Badr AlKhamissi, Johannes Mehrer, Martin Schrimpf
- **Venue:** arXiv 2509.24597
- **What they did:** Identified visual-word-form-selective units within VLMs that predict human VWFA neural responses. Ablated these units to create "artificial dyslexia."
- **Key finding:** Ablating model VWF units leads to selective impairments in reading tasks while general visual and language comprehension remain intact -- directly analogous to dyslexia.
- **How close:** VERY RELEVANT for "AI Neuropathology" beyond aphasia. Demonstrates a reading-specific syndrome can be induced by targeted ablation.
- **URL:** https://arxiv.org/abs/2509.24597

### 10. "Constructive Apraxia: An Unexpected Limit of Instructible Vision-Language Models and Analog for Human Cognitive Disorders" (Oct 2024)
- **Authors:** David Noever, Samantha E. Miller Noever
- **Venue:** arXiv 2410.03551
- **What they did:** Tested 25 VLMs on spatial reasoning tasks used in clinical apraxia assessment. Found 24/25 models fail in ways mirroring parietal lobe damage patients.
- **Key finding:** Models consistently misinterpreted spatial instructions in ways strikingly similar to constructive apraxia patients.
- **How close:** RELEVANT but observational rather than interventional. Found existing failure modes that resemble clinical syndromes, rather than inducing them through ablation.
- **URL:** https://arxiv.org/abs/2410.03551

### 11. "The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities" (Oct 2025, ICLR 2026)
- **Authors:** Zixuan Qin et al.
- **Venue:** arXiv 2510.10238 / ICLR 2026
- **What they did:** Discovered ultra-sparse critical neurons in LLMs. Disabling as few as 3 neurons can catastrophically impair a 72B model. Critical neurons concentrate in outer layers, particularly MLP down_proj.
- **Key finding:** Performance degradation shows sharp phase transitions rather than gradual decline. Maps critical neuron distribution across Llama-3, Gemma, DeepSeek-R1, Phi-3, Qwen2.5.
- **How close:** RELEVANT for understanding vulnerability patterns but does not map to clinical syndromes.
- **URL:** https://arxiv.org/abs/2510.10238

### 12. "Brain-like Functional Organization within Large Language Models" (Oct 2024)
- **Authors:** Haiyang Sun, Lin Zhao, Zihao Wu, Xiaohui Gao et al.
- **Venue:** arXiv 2410.19542
- **What they did:** Extracted temporal patterns from LLM neuron activity (BERT, Llama 1-3), used as regressors in voxel-wise encoding models to predict fMRI brain activity.
- **Key finding:** Sub-groups of artificial neurons in LLMs align closely with functional brain networks (FBNs) in the human brain. Organization becomes more brain-like with increased model sophistication.
- **URL:** https://arxiv.org/abs/2410.19542

### 13. "Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models" (Feb 2025/Jan 2026)
- **Authors:** Liu, Yiheng et al.
- **Venue:** arXiv 2502.20408
- **What they did:** Used neuroscience methodology to identify functional networks in LLMs (GPT2, Qwen2.5-3B, ChatGLM3-6B). Tested inhibition and amplification of these networks.
- **Key finding:** LLMs exhibit recurrent functional networks analogous to brain FBNs. Inhibiting key networks severely impairs capabilities; amplifying neurons within them enhances performance.
- **URL:** https://arxiv.org/abs/2502.20408

### 14. "Probing Neural Topology of Large Language Models" (Jun 2025)
- **Authors:** Zheng et al.
- **Venue:** arXiv 2506.01042
- **What they did:** Used graph probing to uncover functional connectivity of LLM neurons. Identified "default networks" and "hub neurons."
- **Key finding:** Neural topology predicts LLM performance better than activation patterns (up to 130% improvement). Found causal evidence that LLMs exploit topological information.
- **URL:** https://arxiv.org/abs/2506.01042

### 15. "Are Formal and Functional Linguistic Mechanisms Dissociated in Language Models?" (Mar 2025)
- **Authors:** Michael Hanna, Yonatan Belinkov, Sandro Pezzelle
- **Venue:** arXiv 2503.11302 / Computational Linguistics
- **What they did:** Found and compared "circuits" (minimal computational subgraphs) responsible for formal linguistic tasks vs. functional linguistic tasks.
- **Key finding:** Provides evidence for functional dissociation within LLMs, paralleling neuroscience findings of language-thought dissociation.
- **URL:** https://arxiv.org/abs/2503.11302

### 16. "Cyclic Ablation: Testing Concept Localization against Functional Regeneration in AI" (Sep 2025)
- **Authors:** Eduard Kapelko
- **Venue:** arXiv 2509.25220
- **What they did:** Iteratively ablated deception-related features using SAEs, then tested recovery via adversarial training.
- **Key finding:** Deception is resilient and recovers after ablation ("functional regeneration"), while each ablation cycle causes gradual linguistic performance decay. Analogous to brain plasticity and compensatory mechanisms.
- **URL:** https://arxiv.org/abs/2509.25220

### 17. "Friends and Grandmothers in Silico: Localizing Entity Cells in Language Models" (Apr 2026)
- **Authors:** Itay Yona, Dan Barzilay, Michael Karasik, Mor Geva
- **Venue:** arXiv 2604.01404
- **What they did:** Localized entity-selective MLP neurons. Negative ablation produces "entity-specific amnesia."
- **Key finding:** Ablating a single neuron can selectively erase knowledge of a specific entity while preserving other knowledge -- analogous to selective amnesia.
- **URL:** https://arxiv.org/pdf/2604.01404

---

## TIER 3: SUPPORTING/FOUNDATIONAL WORK

### 18. "Dissociating Language and Thought in Large Language Models" (2024)
- **Authors:** Mahowald, Ivanova, Blank, Kanwisher, Tenenbaum, Fedorenko (MIT)
- **Venue:** Trends in Cognitive Sciences 28(6):517-540
- **Key contribution:** Establishes theoretical framework distinguishing formal linguistic competence from functional linguistic competence in both brains and LLMs.
- **URL:** https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(24)00027-5

### 19. "Dementia in Convolutional Neural Networks: Using Deep Learning Models to Simulate Neurodegeneration of the Visual System" (2022)
- **Authors:** Moore, Tuladhar et al.
- **Venue:** Neuroinformatics (Springer)
- **What they did:** Simulated posterior cortical atrophy in CNNs by progressively lesioning networks trained for visual object recognition.
- **Key finding:** Injured networks lose object-level representations before category-level representations, matching human PCA progression.
- **URL:** https://link.springer.com/article/10.1007/s12021-022-09602-6

### 20. "Towards Realistic Simulation of Disease Progression in the Visual Cortex with CNNs" (2025)
- **Authors:** Moore et al.
- **Venue:** Scientific Reports 15:6099
- **What they did:** Extended CNN neurodegeneration model with neuroplasticity simulation via synaptic weight decay and retraining.
- **URL:** https://www.nature.com/articles/s41598-025-89738-y

### 21. "Digital Dementia and Testing of Cognitive Intervention for Degenerating Neural Networks" (2025)
- **Venue:** npj Systems Biology and Applications 11:125
- **What they did:** CNN-based in silico model testing intervention strategies (random, accuracy-based, entropy-based retraining) for simulated visual system degeneration.
- **URL:** https://www.nature.com/articles/s41540-025-00596-w

### 22. "Schizophrenia-Mimicking Layers Outperform Conventional Neural Network Layers" (2022)
- **Venue:** Frontiers in Neurorobotics
- **What they did:** Designed neural network layers that mimic the suppressed distal connections observed in schizophrenia brain tissue. Found these layers are actually more robust to overfitting.
- **URL:** https://arxiv.org/abs/2009.10887

### 23. "Modeling Cognitive Deficits Following Neurodegenerative Diseases and Traumatic Brain Injuries with Deep CNNs" (2016/2018)
- **Authors:** Bethany Lusch, Jake Weholt, Pedro D. Maia, J. Nathan Kutz
- **Venue:** Brain and Cognition / arXiv 1612.04423
- **What they did:** Used biophysically relevant statistical data on focal axonal swellings to damage CNN connections. Demonstrated human-like error patterns.
- **URL:** https://arxiv.org/abs/1612.04423

### 24. "On Simulating Neural Damage in Connectionist Networks" (2020)
- **Authors:** Olivia Guest, Andrea Caso, Richard P. Cooper
- **Venue:** Computational Brain & Behavior
- **What they did:** Comprehensive survey of damage methods in connectionist models: removing connections, adding noise to weights, scaling weights, removing units, adding noise to activations.
- **URL:** https://link.springer.com/article/10.1007/s42113-020-00081-z

### 25. "Clinical Efficacy of Pre-trained Large Language Models Through the Lens of Aphasia" (2024)
- **Venue:** Scientific Reports (Nature)
- **What they did:** Used LLM surprisal scores to detect aphasia presence and subtype from patient speech.
- **URL:** https://www.nature.com/articles/s41598-024-66576-y

### 26. "Tracking Priming-Induced Language Recovery in Aphasia with Pre-trained Language Models" (Oct 2025)
- **Authors:** Yan Cong, Jiyeon Lee (Purdue)
- **Venue:** Frontiers in Artificial Intelligence
- **What they did:** Used PLM-derived surprisals to track treatment-induced language changes in aphasia patients.
- **URL:** https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1668399/full

### 27. "Mapping Brains with Language Models: A Survey" (2023)
- **Authors:** Karamolegkou, Abdou, Sogaard
- **Venue:** ACL 2023 Findings
- **What they did:** Surveyed 30+ studies on brain-LLM alignment, spanning 10 datasets and 8 metrics.
- **URL:** https://arxiv.org/abs/2306.05126

### 28. "Detection Transformers Under the Knife: A Neuroscience-Inspired Approach to Ablations" (Jul 2025)
- **Venue:** arXiv 2507.21723
- **What they did:** Applied neuroscience-inspired ablation methodology to detection transformers (DETR, DDETR, DINO), revealing model-specific resilience patterns.
- **URL:** https://arxiv.org/abs/2507.21723

### 29. "Generating Completions for Broca's Aphasic Sentences Using Large Language Models" (Dec 2024)
- **Authors:** Sijbren van Vaals, Yevgen Matusevych, Frank Tsiwah
- **Venue:** arXiv 2412.17669
- **What they did:** Fine-tuned LLMs to complete agrammatic Broca's aphasic sentences as a communication aid.
- **URL:** https://arxiv.org/abs/2412.17669

---

## GAP ANALYSIS: What Has NOT Been Done

### Syndromes NOT yet mapped to LLM ablation patterns:
1. **Amnesia subtypes** (retrograde vs. anterograde) -- Only "entity-specific amnesia" from single-neuron ablation (Paper #17), no systematic mapping
2. **Semantic dementia** -- Not addressed in any paper found
3. **Visual agnosia / prosopagnosia** -- Only in CNN visual systems (Papers #19-21), not in LLMs
4. **Executive dysfunction / frontal syndromes** -- Brief mention in Paper #7, not systematically studied
5. **Neglect syndromes** -- Not addressed
6. **Confabulation** -- LLM hallucination is extensively studied but NOT framed as a clinical confabulation analogue
7. **Anosognosia** (unawareness of deficit) -- Not addressed
8. **Semantic paraphasia vs. phonemic paraphasia** -- BLUM paper (#1) begins this but doesn't develop full taxonomy
9. **Pure alexia / alexia without agraphia** -- Paper #9 on dyslexia in VLMs is closest
10. **Anomia** -- Partially addressed in aphasia papers but not as isolated syndrome
11. **Transcortical aphasias** -- Not addressed (only Broca's and Wernicke's)
12. **Global aphasia** -- Not specifically studied

### Methodological gaps:
1. **No comprehensive taxonomy** mapping ALL clinical syndromes to LLM component damage patterns
2. **No "atlas" publication** providing a reference framework for syndrome-component mappings
3. **No cross-model validation** showing syndrome patterns are consistent across different architectures
4. **No severity grading systems** that parallel clinical severity scales (e.g., NIHSS equivalent for LLMs)
5. **Limited to language syndromes** -- No work on memory, executive, visuospatial, or attentional syndromes in LLMs
6. **No "AI neurologist" diagnostic framework** that takes an observed LLM deficit and localizes the "lesion"
7. **No rehabilitation protocols** systematically tested across multiple syndrome types (only aphasia in Paper #3)

---

## COMPETITIVE LANDSCAPE SUMMARY

| Paper | Year | Syndrome(s) Covered | Method | Clinical Validation |
|-------|------|---------------------|--------|-------------------|
| BLUM (Fridriksson) | 2026 | Aphasia (naming/comprehension) | Layer perturbation | Strong (N=410 patients) |
| Component-Level Lesioning | 2026 | Broca's, Wernicke's | Component perturbation (MoE + dense) | WAB subtests |
| Bridging Brains | 2025 | Broca's, Wernicke's | MoE expert disabling | Patient speech validation |
| Emergent Modularity | 2025 | Broca's, Wernicke's, Conduction, Anomic | Component disruption | Behavioral profiles |
| Watanabe et al. | 2025 | Receptive vs. expressive aphasia | Network dynamics analysis | Energy landscape analysis |
| Neural Erosion | 2024 | General cognitive decline | Neuron ablation + noise | IQ test degradation |
| Inducing Dyslexia | 2025 | Dyslexia | Unit ablation in VLMs | VWFA neural prediction |
| Constructive Apraxia | 2024 | Apraxia | Observational (no ablation) | Clinical assessment tasks |
| Dementia in CNNs | 2022 | PCA (visual) | Progressive lesioning | Object recognition hierarchy |

---

## ASSESSMENT FOR NATURE-LEVEL NOVELTY

### What would be novel:
1. **A comprehensive "AI Neuropathology" framework** that goes beyond aphasia to cover amnesia, agnosia, executive dysfunction, etc. -- This does NOT exist yet.
2. **Formal double dissociations** between different cognitive domains (language vs. memory vs. reasoning vs. perception) through targeted ablation -- Partially done for language subcomponents, not done across cognitive domains.
3. **A diagnostic "AI neurology" methodology** that can classify unknown LLM deficits into clinical syndrome categories -- Not done.
4. **Cross-architecture syndrome consistency** -- Showing the same ablation patterns produce the same syndromes across GPT, Llama, Mistral, etc. -- Not done comprehensively.
5. **Recovery and plasticity protocols** mapped to clinical rehabilitation -- Only done for aphasia in one paper.

### What would NOT be novel:
1. Ablating LLM components to produce aphasia-like outputs (done by 4-5 papers)
2. Comparing LLM dynamics to aphasic brain dynamics (done by Watanabe 2025)
3. Using clinical instruments (WAB) to evaluate lesioned LLMs (done by multiple papers)
4. Showing MoE models support more interpretable lesion studies (done)
5. Demonstrating progressive degradation order in LLMs mirrors human cognitive decline (done by Neural Erosion)

### Risk assessment:
- **HIGH RISK** if the paper focuses only on aphasia simulation -- this space is now competitive with 4-5 recent papers
- **MEDIUM RISK** if proposing a broad framework but only demonstrating aphasia + one other syndrome
- **LOW RISK** if demonstrating a genuine atlas across 5+ syndrome categories with cross-architecture validation
- The BLUM paper (Feb 2026, with 28 authors) is particularly strong competition -- led by a major aphasia research group

---

## KEY RESEARCH GROUPS TO WATCH

1. **BLUM/Fridriksson group** (28 authors, Feb 2026) -- Strongest competition, clinical neuroscience validation approach
2. **Wang/Fan/Han/Bi/Li group** (bioRxiv 2025) -- Emergent modularity / aphasia simulation
3. **MoE lesioning group** (arXiv Aug 2025, Jan 2026) -- Two papers on component-level lesioning
4. **Watanabe group** (U Tokyo, Advanced Science 2025) -- Network dynamics comparison
5. **Schrimpf group** (dyslexia in VLMs, Sep 2025) -- Vision-language disorder simulation
6. **Moore/Tuladhar group** (CNN neurodegeneration, 2022-2025) -- Visual system degeneration

---

## CONCLUSION

The "AI Neuropathology" concept is real and gaining traction, but remains fragmented and aphasia-centric. A Nature-level contribution would need to:

1. **Go far beyond aphasia** -- Cover at least 5-8 distinct clinical syndrome categories
2. **Establish formal clinical parallels** -- Use actual clinical diagnostic criteria, not just benchmark score drops
3. **Demonstrate double dissociations** -- Show that different ablation patterns produce genuinely different syndromes (not just different severities of the same degradation)
4. **Validate cross-architecturally** -- Show consistency across model families
5. **Frame as a new discipline** -- "AI Neuropathology" as a systematic field, not just case studies

The window for a landmark paper is still open but narrowing rapidly. The aphasia niche is saturated; the broader multi-syndrome atlas opportunity is the differentiator.
