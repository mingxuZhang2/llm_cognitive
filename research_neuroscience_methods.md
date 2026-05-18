# Neuroscience Methods Applied to Large Language Models: A Systematic Survey for Nature-Level Research

**Prepared for:** Dr. Zhang  
**Date:** May 2026  
**Purpose:** Developing a Nature-level paper idea at the intersection of neuroscience methodology and LLM interpretability

---

## Executive Summary

This survey maps 10 foundational neuroscience methodologies to their emerging parallels in LLM research. The central thesis is that the toolkit neuroscientists have developed over 150 years to understand animal brains — lesion studies, functional imaging, connectomics, optogenetics, single-neuron recording, developmental studies, comparative anatomy, plasticity research, brain atlases, and split-brain experiments — maps almost perfectly onto the methods emerging in mechanistic interpretability and LLM analysis. A Nature-level paper opportunity lies in *systematically formalizing this correspondence* and using it to import underutilized neuroscience methods into LLM research, or to propose a unified theoretical framework.

---

## Method 1: Lesion Studies and Ablation

### Neuroscience Background

Lesion studies are among the oldest and most productive methods in neuroscience. Pierre Paul Broca (1861) and Carl Wernicke (1874) established that focal brain damage produces specific language deficits, founding the field of neuropsychology. In these studies, patients with brain damage from stroke, tumor, or surgery show selective impairments. By correlating the location of the lesion with the specific cognitive deficit, researchers infer the function of the damaged region.

Key discoveries from lesion research include:
- Broca's area (inferior frontal gyrus) is critical for speech production
- Wernicke's area (superior temporal gyrus) is critical for language comprehension
- The hippocampus (patient H.M., bilateral removal 1953) is essential for forming new declarative memories
- The amygdala mediates fear responses and emotional memory
- The prefrontal cortex supports working memory and executive function

Modern voxel-based lesion-symptom mapping (VLSM) allows computational analysis of thousands of patients to build probabilistic maps of cognitive function in the brain. Critically, lesion studies establish *causal necessity* — if damage to region X consistently produces deficit Y, then region X is necessary for function Y.

However, a key lesson from 30 years of modern lesion research is that simple localizationism is wrong. Chronic Broca's aphasia requires damage beyond Broca's area alone, implicating distributed networks. This has pushed the field toward network lesion analysis.

**Key neuroscience papers:**
- Broca, P. (1861). *Remarks on the seat of the faculty of articulate language*. Bulletin de la Société Anatomique, 6, 330–357.
- Dronkers, N.F., Plaisant, O., Iba-Zizen, M.T., & Cabanis, E.A. (2007). Paul Broca's historic cases: high resolution MR imaging of the brains of Leborgne and Lelong. *Brain*, 130(5), 1432–1441.
- Baldo, J.V., Dronkers, N.F., et al. (2018). What do language disorders reveal about brain–language relationships? *Journal of the International Neuropsychological Society*, 24(9), 872–882.

### LLM Parallel: Neuron/Layer Ablation and Knockout Experiments

In LLMs, ablation studies systematically deactivate individual neurons, attention heads, or entire layers to identify components causally necessary for specific capabilities. This directly mirrors the lesion paradigm: instead of a stroke destroying tissue, researchers zero out weights or activation values and observe the resulting "deficit" in model behavior.

Key LLM ablation findings:
- Critically, only a handful of "critical neurons" govern core language functions. One 2025 study found that deactivating as few as 3 neurons in a 72B-parameter model can drive perplexity up by 20 orders of magnitude — analogous to the focal, devastating effects of small strokes.
- In circuit analysis (Wang et al., 2022), ablating specific attention heads in GPT-2 small causes selective deficits in the indirect object identification task, mirroring double dissociation logic in neuropsychology.
- The "LLM Language Network" paper (AlKhamissi et al., 2024) found that ablating the 1% most language-selective units causes catastrophic language deficits, while ablating random units has minimal effect — a direct analog of focal vs. distributed lesion effects.
- Code LLMs contain language-specific neurons: ablating them selectively impairs one programming language while sparing others (Neuron-Guided Code LLM Interpretation, 2024).

**Key LLM papers:**
- Wang, K., Variengien, A., Conmy, A., et al. (2022). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 small. *arXiv:2211.00593*. [ICLR 2023]
- AlKhamissi, B., Tuckute, G., Bosselut, A., & Schrimpf, M. (2024). The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units. *arXiv:2411.02280*. [NAACL 2025]
- Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities. *arXiv:2510.10238* (2025).

### Nature Paper Novelty/Impact: HIGH

**Opportunity:** The neuroscience field has developed sophisticated methodologies beyond simple single-region lesion studies — notably, double dissociation, network lesion analysis, and lesion subtraction analysis. These refined approaches have not been systematically applied to LLMs. A Nature paper could formalize a rigorous "double dissociation" framework for LLMs, where two components are shown to produce opposite deficits across two tasks, providing much stronger causal evidence than single ablations. This would parallel the gold standard of clinical neuropsychology.

---

## Method 2: Functional Imaging (fMRI, PET, EEG)

### Neuroscience Background

Functional magnetic resonance imaging (fMRI) revolutionized cognitive neuroscience by enabling non-invasive measurement of brain activity in living humans during cognitive tasks. fMRI measures the blood-oxygen-level-dependent (BOLD) signal, which reflects regional neural activity. Since its development in the early 1990s (Ogawa et al., 1990), fMRI has produced a comprehensive map of which brain regions activate during language processing, memory retrieval, decision-making, and social cognition.

Key principles:
- Subtraction logic: Activity during task minus activity during baseline isolates task-specific processing
- Encoding models: Brain activity can be predicted from stimulus features (e.g., word embeddings)
- Decoding: Brain patterns can be used to predict what stimulus the subject experienced
- Representational similarity analysis (RSA): Comparing the geometry of neural and model representations

EEG and magnetoencephalography (MEG) add millisecond temporal resolution, revealing when, not just where, processing occurs. PET tracers can measure neurotransmitter systems and metabolic activity.

**Key neuroscience papers:**
- Ogawa, S., Lee, T.M., Kay, A.R., & Tank, D.W. (1990). Brain magnetic resonance imaging with contrast dependent on blood oxygenation. *PNAS*, 87(24), 9868–9872.
- Fedorenko, E., Hsieh, P.J., Nieto-Castañón, A., Whitfield-Gabrieli, S., & Kanwisher, N. (2010). New method for fMRI investigations of language: defining ROIs functionally in individual subjects. *Journal of Neurophysiology*, 104(2), 1177–1194.
- Wehbe, L., Murphy, B., Talukdar, P., et al. (2014). Simultaneously uncovering the patterns of brain regions involved in different story reading subprocesses. *PLoS ONE*, 9(11), e112575.

### LLM Parallel: Activation Pattern Analysis and Neural Encoding Models

The connection between fMRI and LLMs has become one of the most productive intersections in cognitive neuroscience. Researchers use LLM representations as "encoding models" to predict fMRI BOLD responses to the same stimuli. If a particular LLM layer's embeddings better predict brain activity in a particular region, this suggests those layers and brain regions perform analogous computations.

Key findings from fMRI-LLM research:
- LLM representations predict brain activity significantly better than traditional NLP features, and prediction accuracy scales logarithmically with model size (scaling laws for brain encoding; Schrimpf et al., 2021).
- Intermediate LLM layers (not the final output layer) best predict activity in the language-selective cortex, suggesting early and middle processing stages correspond to linguistic comprehension regions.
- A 2025 study found that brain left lateralization in language processing follows a scaling law with model parameters — larger models more strongly predict left-hemisphere dominance.
- The fMRI-LM work (2024) trained a foundational model to jointly model fMRI tokens and text, enabling bidirectional brain-to-language decoding.
- GPT-2's attention patterns correlate with eye-tracking data during reading (Toneva & Wehbe, 2019), linking LLM processing to human reading behavior.

**Key LLM-neuroscience bridge papers:**
- Schrimpf, M., Blank, I., Tuckute, G., et al. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. *PNAS*, 118(45). https://www.pnas.org/doi/10.1073/pnas.2105646118
- Caucheteux, C., & King, J.R. (2022). Brains and algorithms partially converge in natural language processing. *Communications Biology*, 5, 134.
- Schrimpf, M., et al. (2024). Scaling laws for language encoding models in fMRI. *NeurIPS 2024*. PMC11258918.
- Do Large Language Models Think Like the Brain? (2025). *arXiv:2505.22563*

### Nature Paper Novelty/Impact: HIGH

**Opportunity:** While encoding model research is active, the reverse direction is underexplored: using *LLMs as a virtual "subject"* whose internal representations serve as ground truth for testing theories about brain computation. Specifically, neuroscientists have developed representational similarity analysis (RSA) and partial RSA to disentangle low-level from high-level representations. Applying these tools to rigorously test whether specific LLM layers share not just predictive accuracy but actual representational *geometry* with specific brain regions would be novel. A second opportunity is studying *task-specific* functional segregation in LLMs using the fMRI localizer paradigm — the method used in "The LLM Language Network" paper shows this is highly publishable.

---

## Method 3: Connectomics

### Neuroscience Background

Connectomics is the comprehensive mapping of all synaptic connections in a nervous system — producing a "wiring diagram" or connectome. The field emerged from the landmark mapping of the 302-neuron C. elegans connectome (White et al., 1986), and has since scaled to mapping millions of neurons in mouse and human cortex.

The field now combines electron microscopy (EM) for nanometer-resolution imaging with deep learning-based reconstruction algorithms. Landmark achievements:
- Complete Drosophila brain connectome: 139,255 neurons and 50 million synapses (FlyWire Consortium, 2024)
- 1 cubic millimeter of human temporal cortex reconstructed (2024) — the H01 dataset from Google/Harvard
- MICrONS: 75,000 functionally characterized mouse visual cortex neurons co-registered with EM reconstruction of 200,000 cells and 0.5 billion synapses

The MICrONS dataset's key finding is that neurons with similar functional tuning preferentially connect to one another ("like-to-like" connectivity), establishing a clear structure-function relationship at the level of individual synapses.

Connectomics enables asking questions that were previously impossible: What is the exact wiring pattern of the circuit implementing a computation? Are there canonical circuit motifs that repeat across regions? How does connectivity predict function?

Nature named EM-based connectomics its "Method of the Year" for 2025.

**Key neuroscience papers:**
- White, J.G., Southgate, E., Thomson, J.N., & Brenner, S. (1986). The structure of the nervous system of the nematode Caenorhabditis elegans. *Philosophical Transactions of the Royal Society B*, 314(1165), 1–340.
- MICrONS Consortium (2025). Functional connectomics spanning multiple areas of mouse visual cortex. *Nature*, 626. https://www.nature.com/articles/s41586-025-08790-w
- Connectome-seq (2026). High-throughput mapping of neuronal connectivity at single-synapse resolution via barcode sequencing. *Nature Methods*. https://www.nature.com/articles/s41592-026-03026-9

### LLM Parallel: Weight Connectivity Analysis and Circuit Discovery

In transformers, the equivalent of a connectome is the complete mapping of information flow through the computational graph — from input tokens, through attention heads and MLP layers, to output logits. The "transformer circuits" research program (Elhage et al., 2021) introduced the mathematical framework for treating transformers as computational circuits, analogous to neural circuits.

Key tools for LLM "connectomics":
- Attention weight matrices reveal which token positions each head "reads from" (analogous to axonal inputs)
- OV (output-value) circuits describe how attention heads transform and route information
- QK (query-key) circuits describe the pattern-matching that determines attention weights
- Path patching traces causal information flow along specific pathways through the network
- Knowledge circuits (Yao et al., NeurIPS 2024) map the exact multi-layer circuit through which factual knowledge is stored and retrieved

The "A Mathematical Framework for Transformer Circuits" paper established that even simple 2-layer attention-only transformers can be fully decomposed into understandable computational circuits, setting the stage for the connectomics-inspired analysis of larger models.

A critical parallel: just as the MICrONS project found structure-function correspondence (like-to-like connectivity), mechanistic interpretability has found that attention heads cluster into functional groups with interpretable roles (name movers, inhibition heads, backup heads), suggesting structured functional organization.

**Key LLM papers:**
- Elhage, N., Nanda, N., Olsson, C., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/index.html
- Olsson, C., Elhage, N., Nanda, N., et al. (2022). In-context Learning and Induction Heads. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Yao, Y., Zhang, N., et al. (2024). Knowledge Circuits in Pretrained Transformers. *NeurIPS 2024*.

### Nature Paper Novelty/Impact: HIGH

**Opportunity:** The connectomics field has developed sophisticated graph-theoretic tools for analyzing wiring diagrams — motif analysis, hub detection, rich-club analysis, and community detection. These tools have barely been applied to LLM weight matrices. A Nature paper could apply network neuroscience methods to transformer weight graphs, asking: Are there "hub" attention heads analogous to brain hub regions? Do transformers show small-world network properties? Do circuit motifs (e.g., feedforward loops, recurrent loops) appear systematically? This would be the first systematic "network neuroscience" analysis of LLM connectivity.

---

## Method 4: Optogenetics

### Neuroscience Background

Optogenetics, developed by Karl Deisseroth and colleagues starting in 2005 (Nature Methods "Method of the Year" 2010), enables researchers to precisely activate or silence specific neurons by expressing light-sensitive proteins (channelrhodopsins). Unlike lesion studies (which produce permanent damage) or pharmacology (which affects many cell types non-specifically), optogenetics provides:
- Cell-type specificity: Only neurons expressing the opsin respond to light
- Millisecond temporal precision: Neurons can be activated or silenced in precise patterns
- Reversibility: Light is turned on and off without tissue damage
- Causal directionality: Activation produces behavior, demonstrating that activation is *sufficient* for a function

Key optogenetics discoveries:
- Stimulating specific hypothalamic neurons is sufficient to trigger feeding, aggression, or fear
- Activating place cells during sleep can reactivate and strengthen memories
- Optogenetic silencing of dopaminergic neurons prevents reward learning
- Specific amygdala circuits are necessary and sufficient for conditioned fear

Optogenetics resolves a fundamental limitation of lesion studies: it distinguishes between neurons necessary for a function and neurons sufficient to drive a function. It also enables studying the *timing* of neural contributions with precision impossible before.

**Key neuroscience papers:**
- Boyden, E.S., Zhang, F., Bamberg, E., Nagel, G., & Deisseroth, K. (2005). Millisecond-timescale, genetically targeted optical control of neural activity. *Nature Neuroscience*, 8, 1263–1268.
- Deisseroth, K. (2010). Optogenetics. *Nature Methods*, 8, 26–29. https://web.stanford.edu/group/dlab/media/papers/deisserothnature2010.pdf
- Ramirez, S., Liu, X., Lin, P.A., et al. (2013). Creating a false memory in the hippocampus. *Science*, 341(6144), 387–391.

### LLM Parallel: Activation Patching and Causal Interventions

Activation patching (also called "causal tracing") is the closest LLM analog to optogenetics. The technique works by:
1. Running the model on a "clean" input (baseline)
2. Running the model on a "corrupted" input (equivalent to silencing a neuron)
3. Selectively restoring ("patching in") specific internal activations from the clean run into the corrupted run
4. Observing which restorations recover the original behavior

This is precisely optogenetics logic: identify the minimal set of components whose *activation* is *sufficient* to restore a behavior.

The ROME paper (Meng et al., NeurIPS 2022) used causal tracing to localize factual knowledge storage to early-to-middle MLP layers at subject token positions in GPT-style models, then used this to perform targeted knowledge editing. This parallels optogenetic tagging followed by targeted manipulation.

More recent work includes:
- Activation steering (adding linear vectors to activations to steer behavior) is analogous to continuous optogenetic stimulation at controlled levels
- "Representation engineering" inserts control vectors into residual stream activations to modulate emotional tone, honesty, and reasoning
- Path patching (a more precise variant) traces individual activation pathways rather than full layer outputs

**Key LLM papers:**
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*. https://arxiv.org/abs/2202.05262
- Conmy, A., Mavor-Parker, A.N., Lynch, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*.
- Zou, A., Phan, L., Chen, S., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.

### Nature Paper Novelty/Impact: HIGH

**Opportunity:** The most powerful application of optogenetics is *gain-of-function* experiments — not just silencing but actively driving circuits to create behaviors. The LLM equivalent (activation patching/steering) has been used mostly for interpretation, not for systematically *constructing* circuits that produce targeted behaviors from scratch. A Nature paper could use the optogenetics framework to perform systematic "circuit construction" in LLMs: identify minimal sufficient circuits for specific reasoning capabilities through a principled program of gain-of-function activation patching. This would parallel the seminal Ramirez et al. (2013) false memory paper — arguably one of the most cited and admired experiments in modern neuroscience.

---

## Method 5: Electrophysiology and Single-Neuron Recording

### Neuroscience Background

Electrophysiology — recording the electrical activity of individual neurons — has been fundamental to neuroscience for over 70 years. Single-unit recordings in awake, behaving animals (pioneered by David Hubel and Torsten Wiesel in the visual cortex, Nobel Prize 1981) revealed that individual neurons are selective for specific stimulus features:
- Simple cells in V1 respond to oriented edges at specific locations
- Complex cells respond to oriented edges regardless of precise position
- Place cells in the hippocampus fire when animals occupy specific spatial locations
- Concept cells (Jennifer Aniston neurons, Quiroga et al., 2005) in human temporal cortex respond to specific high-level concepts

Modern multi-electrode arrays can record hundreds to thousands of neurons simultaneously. Neuropixels probes (Jun et al., 2017) enable recording from thousands of neurons across multiple brain areas simultaneously.

The key insight from single-neuron recording is that individual neurons carry interpretable, selective information — they are tuned detectors for specific features of the world. However, the "grandmother cell" hypothesis (one neuron = one concept) is wrong; most neurons are broadly tuned and population codes are essential.

**Key neuroscience papers:**
- Hubel, D.H., & Wiesel, T.N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. *Journal of Physiology*, 160, 106–154.
- Quiroga, R.Q., Reddy, L., Kreiman, G., Koch, C., & Fried, I. (2005). Invariant visual representation by single neurons in the human brain. *Nature*, 435, 1102–1107.
- Jun, J.J., Steinmetz, N.A., Siegle, J.H., et al. (2017). Fully integrated silicon probes for high-density recording of neural activity. *Nature*, 551, 232–236.

### LLM Parallel: Individual Neuron Analysis and Knowledge Neurons

The analog of single-neuron recording in LLMs is analyzing the activation of individual MLP neurons (or individual attention head output dimensions) across diverse inputs. Just as Quiroga et al. found "Jennifer Aniston neurons," LLM researchers have found neurons selective for specific concepts:

- "Knowledge neurons" (Dai et al., ACL 2022): Specific feed-forward network neurons in BERT and GPT models activate strongly when the model processes factual associations (e.g., "the Eiffel Tower is in [Paris]"). Suppressing these neurons degrades factual recall by ~29%.
- Anthropic's monosemanticity work found neurons in Claude Sonnet that respond specifically to the Golden Gate Bridge, specific programming constructs, or emotional states.
- Sparse autoencoders (Cunningham et al., 2023; Templeton et al., 2024) decompose polysemantic neurons into monosemantic features — solving the LLM equivalent of the polysemantic neuron problem in neuroscience (where neurons respond to multiple, apparently unrelated stimuli).
- The "Brain-Inspired Exploration" paper (2025) introduced methods directly inspired by functional neuroimaging analysis to identify "key neurons" in LLMs, finding functionally specialized units analogous to concept cells.

The key parallel: just as V1 neurons are tuned to oriented edges (local, low-level features) while temporal cortex neurons encode abstract concepts, LLM neurons in earlier layers encode syntactic/positional features while later layers encode semantic and conceptual information.

**Key LLM papers:**
- Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). Knowledge Neurons in Pretrained Transformers. *ACL 2022*. https://aclanthology.org/2022.acl-long.581/
- Cunningham, H., Ewart, A., Riggs, L., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *arXiv:2309.08600*. [ICLR 2024]
- Templeton, A., Conerly, T., Marcus, J., et al. (2024). Scaling and evaluating sparse autoencoders. *Anthropic Research Blog*.

### Nature Paper Novelty/Impact: HIGH

**Opportunity:** The neuroscience analogy suggests an important gap: while individual "knowledge neurons" have been found, nobody has conducted a systematic large-scale "neural survey" of an LLM analogous to a full electrophysiology study across cognitive domains. Such a study would characterize the tuning properties of LLM neurons across thousands of diverse stimuli (linguistic, mathematical, factual, social), build "tuning curves," identify "receptive fields" in semantic space, and test whether neurons show orientation tuning, contrast sensitivity, or other classical neuroscience phenomena in a linguistic sense. This would be directly analogous to Hubel and Wiesel's systematic mapping of V1, which produced a Nobel Prize.

---

## Method 6: Developmental Neuroscience

### Neuroscience Background

Developmental neuroscience studies how the brain self-organizes during embryogenesis, childhood, and adolescence. The brain develops through well-defined stages: neurogenesis, migration, differentiation, synaptogenesis, synaptic pruning, and myelination. Critical periods — sensitive windows during which specific experiences are necessary for normal development — are a key discovery of developmental neuroscience (Hubel & Wiesel's work on monocular deprivation).

Key discoveries:
- The "outside-in" rule: cortical layers develop from deep (layer VI) to superficial (layer II/III)
- Synaptic pruning peaks in adolescence and eliminates up to 50% of synaptic connections
- Critical periods represent windows of heightened plasticity, after which experience can no longer fully shape circuits
- Language acquisition follows a developmental trajectory with critical periods (Lenneberg's hypothesis)
- Premature closure of the critical period prevents normal circuit formation; delayed closure allows abnormal plasticity

Modern developmental neuroscience uses longitudinal imaging, genetic tools, and organoids to study brain development across time.

**Key neuroscience papers:**
- Hubel, D.H., & Wiesel, T.N. (1970). The period of susceptibility to the physiological effects of unilateral eye closure in kittens. *Journal of Physiology*, 206, 419–436.
- Bourgeois, J.P., Goldman-Rakic, P.S., & Rakic, P. (1994). Synaptogenesis in the prefrontal cortex of rhesus monkeys. *Cerebral Cortex*, 4(1), 78–96.
- Tau, G.Z., & Peterson, B.S. (2010). Normal development of brain circuits. *Neuropsychopharmacology*, 35(1), 147–168.

### LLM Parallel: Training Dynamics, Grokking, and Emergence of Capabilities

LLM training has striking parallels to brain development: both involve a system reorganizing from undifferentiated initial states toward specialized, efficient representations. Key phenomena:

**Grokking (Power et al., 2022):** Small transformers trained on modular arithmetic first memorize training data (achieving 100% train accuracy) and then, after many more steps, suddenly "generalize" — achieving high test accuracy in a rapid phase transition. This mirrors how children may initially memorize linguistic patterns before suddenly "grokking" the underlying grammar rule (the U-shaped learning curve).

**Phase transitions and emergence:** LLMs exhibit sudden emergence of capabilities at specific scale thresholds. Below a critical parameter count, a capability (e.g., multi-step reasoning) is absent; above the threshold, it appears abruptly. This parallels brain critical periods: below a developmental threshold, a circuit does not function; above it, it rapidly matures.

**Training stage organization:** Experiments show that during training, transformers develop hierarchical representations progressively — syntactic structures emerge early, semantic representations emerge later — paralleling the outside-in cortical layer development.

**Grokking in pretraining (2026):** Recent work provides the first evidence that grokking occurs in large-scale LLM pretraining, with different data domains entering their generalization phase asynchronously — analogous to how different cortical areas mature on different developmental timelines.

**Key LLM papers:**
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. *arXiv:2201.02177*. [ICLR 2022 Workshop]
- Wei, J., Tay, Y., Bommasani, R., et al. (2022). Emergent Abilities of Large Language Models. *Transactions on Machine Learning Research*.
- Where to find Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test. *arXiv:2506.21551* (2025/2026).

### Nature Paper Novelty/Impact: HIGH

**Opportunity:** The parallel between LLM training dynamics and brain development has been noted informally but never rigorously formalized. A Nature paper could draw tight quantitative analogies: Does the ratio of "grokking delay" to "memorization time" in LLMs correspond to the ratio of critical period closure to synaptogenesis in specific brain regions? Are there "sensitive windows" during LLM training where interventions (data poisoning, fine-tuning) have disproportionate long-term impact — analogous to critical period effects? Most ambitiously, could one define a "developmental trajectory" metric for LLMs — an equivalent of the neurodevelopmental index used clinically — that predicts final model capabilities from early training checkpoints? This is both scientifically fundamental and practically important for efficient training.

---

## Method 7: Comparative Neuroscience

### Neuroscience Background

Comparative neuroscience studies brains across species to understand which neural structures are conserved, which are uniquely elaborated in some lineages, and what computational principles are universal versus species-specific. The phylogenetic comparison reveals design principles that evolution found repeatedly useful.

Key findings from comparative neuroscience:
- The six-layered neocortex is a mammalian innovation; birds achieve similar cognitive sophistication through a different laminar organization
- Primate brains (including human) scale more efficiently than rodent brains: adding neurons requires proportionally less space in primates (Herculano-Houzel, 2009)
- The human brain has 86 billion neurons but is a "scaled-up primate brain" rather than a fundamentally different structure
- Specific prefrontal regions (e.g., area 10, frontopolar cortex) are disproportionately enlarged in humans relative to other primates, correlating with prospective memory and abstract reasoning
- Cetacean (whale/dolphin) brains have independently evolved large spindle neurons similar to human von Economo neurons, suggesting convergent evolution of certain computations

The comparative approach allows "evolution experiments" — natural variants of neural design that reveal which features are computationally essential.

**Key neuroscience papers:**
- Herculano-Houzel, S. (2009). The human brain in numbers: a linearly scaled-up primate brain. *Frontiers in Human Neuroscience*, 3, 31.
- Herculano-Houzel, S., Collins, C.E., Wong, P., & Kaas, J.H. (2007). Cellular scaling rules for primate brains. *PNAS*, 104(9), 3562–3567.
- Charvet, C.J., & Finlay, B.L. (2018). Comparing neurodevelopmental time tables across mammalian species. *European Journal of Neuroscience*, 47(7), 709–720.

### LLM Parallel: Cross-Scale and Cross-Architecture Analysis

Comparative neuroscience maps directly onto studying how LLM properties vary across model scales and architectures. Just as neuroscientists compare mouse → monkey → human brains, ML researchers compare 125M → 7B → 70B → 405B parameter models.

Key findings from comparative LLM analysis:
- Brain encoding accuracy scales logarithmically with model parameters: each order-of-magnitude increase in parameters improves fMRI prediction by ~4.4% (scaling laws for fMRI encoding, NeurIPS 2024)
- Some capabilities (multi-step reasoning, in-context learning) emerge abruptly at specific scale thresholds, analogous to evolutionarily discontinuous neural innovations
- Circuit structures found in small models (e.g., induction heads in 2-layer models) persist and scale to large models, suggesting conserved computational primitives — analogous to conserved cortical circuits across species
- Different architectures (BERT encoder vs. GPT decoder vs. T5 encoder-decoder) show different strengths, analogous to how different brain organization plans (mammalian neocortex vs. avian pallium) achieve similar cognitive outcomes through different structural means

**Key LLM papers:**
- Schrimpf, M., et al. (2024). Scaling laws for language encoding models in fMRI. *NeurIPS 2024*. PMC11258918. https://pmc.ncbi.nlm.nih.gov/articles/PMC11258918/
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
- Tigges, C., et al. (2024). LLM Circuit Analyses Are Consistent Across Training and Scale. *NeurIPS 2024*.

### Nature Paper Novelty/Impact: MEDIUM-HIGH

**Opportunity:** Comparative neuroscience has developed rigorous phylogenetic statistics to test whether differences between species reflect independent evolution or shared ancestry. These tools (phylogenetic comparative methods, ancestral state reconstruction) could be applied to LLM "phylogenies" — comparing model families (GPT-2 → GPT-3 → GPT-4; LLaMA-7B → 13B → 70B) to understand which computational strategies are "ancestral" (conserved from small models) and which are "derived" (emergent at scale). A Nature paper could formalize the "comparative LLM" approach with rigorous methods, providing a principled framework for understanding what makes large models qualitatively different from smaller ones. The connection to brain evolution scaling laws (primate advantage, neuronal density) is particularly compelling.

---

## Method 8: Brain Plasticity and Adaptation

### Neuroscience Background

Brain plasticity refers to the ability of the nervous system to reorganize — changing the strength of synaptic connections, forming new connections, or even rewiring functional regions — in response to experience, learning, or injury. Key forms:
- Hebbian plasticity: "Neurons that fire together, wire together" (Hebb, 1949)
- Long-term potentiation (LTP) and long-term depression (LTD): Cellular mechanisms of synaptic strengthening and weakening
- Homeostatic plasticity: Stabilizing mechanism that scales synaptic weights to maintain balanced activity
- Cortical remapping: After limb amputation, the somatosensory cortex region formerly representing the lost limb is taken over by adjacent regions (phantom limb, Ramachandran)
- Cross-modal plasticity: Blind individuals recruit visual cortex for tactile and auditory processing

The stability-plasticity dilemma is a fundamental challenge: the brain must remain plastic enough to learn new information while stable enough to retain old memories. Synaptic consolidation, sleep-dependent memory consolidation, and neuromodulatory systems (dopamine, norepinephrine) help manage this trade-off.

**Key neuroscience papers:**
- Bliss, T.V.P., & Lømo, T. (1973). Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path. *Journal of Physiology*, 232(2), 331–356.
- Ramachandran, V.S., Rogers-Ramachandran, D., & Stewart, M. (1992). Perceptual correlates of massive cortical reorganization. *Science*, 258(5085), 1159–1160.
- Abraham, W.C. (2008). Metaplasticity: tuning synapses and networks for plasticity. *Nature Reviews Neuroscience*, 9, 387.

### LLM Parallel: Fine-Tuning, Pruning, and Model Editing

Brain plasticity maps directly onto several LLM adaptation paradigms:

**Fine-tuning as synaptic plasticity:** Parameter fine-tuning updates model weights analogously to LTP/LTD. The stability-plasticity dilemma manifests as catastrophic forgetting: fine-tuned models overwrite pre-trained knowledge. Continual learning methods (EWC, rehearsal-based methods) are computational analogs of neuromodulatory stabilization mechanisms.

**Model editing (ROME, MEMIT) as targeted synaptic modification:** Rank-one model editing (Meng et al., 2022) directly modifies specific MLP weight matrices to update factual associations — analogous to targeted synaptic modification during memory reconsolidation. The "Knowledge Neuronal Ensemble" paper (2024) shows that editing a coordinated ensemble of knowledge-encoding neurons is more effective than editing individual neurons, paralleling the brain's distributed memory engrams.

**Pruning as synaptic pruning:** Neural network pruning (removing low-magnitude weights) directly parallels adolescent synaptic pruning in the brain. "Optimal Brain Pruning" methods (derived from "Optimal Brain Damage" by LeCun et al., 1990) explicitly invoke this analogy. Research shows that pruned LLMs often perform comparably to dense models at the same computational cost, mirroring how synaptic pruning in development refines and sharpens cognitive function.

**Cross-modal plasticity and LoRA:** Low-rank adaptation (LoRA) fine-tuning uses low-rank matrices to efficiently adapt pre-trained representations to new tasks, analogous to how cross-modal plasticity exploits existing synaptic architecture rather than creating entirely new connections.

**Key LLM papers:**
- Meng, K., et al. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS 2022*.
- Meng, K., et al. (2023). Mass-Editing Memory in a Transformer. *ICLR 2023* (MEMIT).
- Yao, Y., et al. (2024). Knowledge Editing for Large Language Model with Knowledge Neuronal Ensemble. *arXiv:2412.20637*.
- Hu, E., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

### Nature Paper Novelty/Impact: MEDIUM-HIGH

**Opportunity:** The neuroscience of plasticity offers a specific unexplored insight: *metaplasticity* — "the plasticity of plasticity" — which refers to how prior history of activity changes the threshold for future plasticity. In LLMs, the analog would be studying how prior fine-tuning history changes the *susceptibility* to subsequent fine-tuning: does prior fine-tuning on task A make the model more or less easily fine-tuned on task B? Does the order of training data affect the architecture of learned circuits? A Nature paper could formally characterize LLM metaplasticity, drawing on the BCM theory of synaptic modification (a key neuroscience framework) to predict and explain fine-tuning dynamics. This connects the fields of continual learning, curriculum learning, and mechanistic interpretability in a neuroscience-grounded framework.

---

## Method 9: Brodmann Areas and Brain Atlases

### Neuroscience Background

Korbinian Brodmann (1909) systematically mapped the human cerebral cortex into 52 numbered areas based on cytoarchitecture — the microscopic cellular organization of each region. This atlas has remained foundational for over 100 years as a standard reference for locating brain regions. Modern extensions include:
- The Human Connectome Project (HCP) parcellation: 360 cortical areas defined by combining myelin content, cortical thickness, resting-state connectivity, and task activation maps
- The Allen Brain Atlas: Gene expression maps across the entire mouse and human brain
- The Julich Brain Atlas: Probabilistic cytoarchitectonic maps aligned to MNI standard space
- MIDB Precision Brain Atlas (2024): 53,273 individual-specific network maps from 9,900+ individuals, enabling personalized brain atlases

These atlases serve as coordinate systems for neuroscience: researchers can specify a location in brain space, and the atlas provides a probability estimate for which functional region that location belongs to. Atlases enable integration of findings across labs and species.

The key conceptual contribution of brain atlases: they impose a *principled coordinate system* on biological complexity, enabling cumulative science. Without the Brodmann atlas, neuroscience findings from different labs would be difficult to compare.

**Key neuroscience papers:**
- Brodmann, K. (1909). Vergleichende Lokalisationslehre der Grosshirnrinde. Leipzig: Johann Ambrosius Barth.
- Glasser, M.F., Coalson, T.S., Robinson, E.C., et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature*, 536, 171–178.
- Hawrylycz, M.J., Lein, E.S., Guillozet-Bongaarts, A.L., et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. *Nature*, 489, 391–399.

### LLM Parallel: Functional Atlases of Transformer Space

The most direct LLM parallel to Brodmann areas would be a systematic "atlas" of transformer functional regions — a principled mapping of which layer ranges, attention heads, and MLP neurons support which functional capabilities.

Existing work toward an LLM atlas:
- The "LLM Language Network" paper (AlKhamissi et al., 2024/2025) identified language-selective units in 18 LLMs using the neuroscience localizer paradigm — essentially the first step toward a functional atlas.
- Mechanistic interpretability research has informally mapped functional roles: early layers handle positional/syntactic information, middle layers handle factual associations, late layers handle task-specific output formatting.
- The "Brain-Inspired Exploration of Functional Networks and Key Neurons in LLMs" paper (2025) directly applies functional neuroimaging analysis methods to construct something analogous to a functional parcellation of LLM "space."
- ROME's causal tracing found that factual knowledge retrieval is concentrated in early-to-middle MLP layers at subject token positions — a specific "address" in transformer coordinate space.

However, no comprehensive, principled atlas has been constructed across multiple capabilities, model families, and scales — the equivalent of the HCP multimodal parcellation.

**Key LLM papers:**
- AlKhamissi, B., et al. (2024). The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units. *arXiv:2411.02280*.
- Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models. *arXiv:2502.20408* (2025).
- Henighan, T., et al. (2020). Scaling Laws for Autoregressive Generative Modeling. *arXiv:2010.14701*.

### Nature Paper Novelty/Impact: VERY HIGH

**Opportunity:** This is arguably the highest-impact opportunity in this survey. A Nature paper could construct the first systematic "Functional Atlas of the Transformer" — a comprehensive map of which transformer components (layer ranges, attention head clusters, MLP neuron populations) are necessary and sufficient for specific capabilities (language, factual knowledge, reasoning, arithmetic, social inference, code generation). Like the Brodmann atlas, this would provide a coordinate system and reference standard for the entire field. The atlas would be built using the neuroscience localizer paradigm (AlKhamissi et al.'s approach) extended across 50+ functional domains and validated across multiple LLM families. This would be a landmark reference paper likely to be cited by every future mechanistic interpretability study.

---

## Method 10: Split-Brain Experiments

### Neuroscience Background

Split-brain research arose from a surgical treatment for severe epilepsy: severing the corpus callosum (the major white matter tract connecting the two cerebral hemispheres) to prevent seizure propagation. Pioneered by Roger Sperry and Michael Gazzaniga (Nobel Prize to Sperry, 1981), split-brain experiments revealed that the two hemispheres function as semi-independent cognitive systems:
- Information presented only to one hemisphere (via divided visual field presentations) cannot be verbally reported by the other hemisphere
- The left hemisphere is dominant for language and produces verbal confabulations to explain actions driven by the right hemisphere
- The right hemisphere excels at spatial and holistic processing
- The left hemisphere's "interpreter" module constructs post-hoc narratives to explain behavior it did not control

Key insight: normal brain function depends critically on interhemispheric integration. Severing connections between brain modules reveals their independent computational specializations and the critical role of connection pathways.

More recent split-brain research (2025) finds that even a small number of remaining corpus callosum fibers (as little as 1 cm) can enable near-normal integration, suggesting remarkable capacity for compensatory information routing.

**Key neuroscience papers:**
- Sperry, R.W. (1968). Hemisphere deconnection and unity in conscious awareness. *American Psychologist*, 23(10), 723–733.
- Gazzaniga, M.S. (2005). Forty-five years of split-brain research and still going strong. *Nature Reviews Neuroscience*, 6, 653–659.
- Pinto, Y., et al. (2025). New findings: Even minimal fiber connections can unify consciousness. UCSB Research. https://news.ucsb.edu/2025/022246/new-findings-split-brain-science-even-minimal-fiber-connections-can-unify-consciousness

### LLM Parallel: Modular Disconnection Experiments

Split-brain logic applied to LLMs would involve systematically severing connections between components and observing the resulting "split" behaviors. Unlike full ablation (which removes a component entirely), disconnection experiments preserve both components while removing their communication.

Existing related work:
- Probing studies that insert linear "decoders" at specific intermediate layers effectively study what information is available at that layer — analogous to probing what the split hemisphere "knows"
- Residual stream decomposition in transformer circuits analyzes which attention heads' outputs are "read" by which later heads — mapping the communication pathways between modules
- Early work on "split-brain" LLM architectures uses two models operating in parallel, studying how they coordinate
- Attention head "blocking" experiments that prevent specific heads from attending to specific positions are a partial analog of corpus callosum sectioning

The deeper analogy from split-brain research is the *interpreter module* phenomenon: the left hemisphere confabulates explanations for behavior it did not cause. This maps to LLMs generating confident but incorrect explanations of their own reasoning — a form of "cognitive confabulation." A split-brain analysis could test whether LLM reasoning traces are genuine mechanistic explanations or post-hoc confabulations generated by a separate "interpreter" module.

**Key LLM papers:**
- Logit Lens / Tuned Lens (Nostalgebraist, 2020; Belrose et al., 2023): Reads out intermediate layer predictions to study progressive information accumulation.
- Anthropic (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Anthropic Research*.
- Turpin, M., et al. (2023). Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. *NeurIPS 2023*.

### Nature Paper Novelty/Impact: HIGH

**Opportunity:** The split-brain analogy points to an underexplored and scientifically rich question: the relationship between LLM *reasoning traces* and actual internal computations. Gazzaniga's interpreter work showed that verbal reports can be confabulations disconnected from the actual computational process. Turpin et al. (2023) showed LLMs do this too — their stated reasoning does not always correspond to what actually determines their outputs. A Nature paper could perform systematic "split-brain" experiments in LLMs: (1) identify the components that generate chain-of-thought explanations vs. the components that generate the final answer; (2) test whether these can be dissociated; (3) characterize what each "hemisphere" knows independently; (4) identify conditions under which LLM reasoning is faithful vs. confabulatory. This would be a landmark result for LLM interpretability and AI safety.

---

## Synthesis and Thematic Analysis

### Theme 1: Causal vs. Correlational Methods (High Priority)

The most important lesson from neuroscience methodology is the emphasis on causal inference. Correlational methods (fMRI, EEG, electrophysiology) identify *where* and *when* activity occurs, but only causal methods (lesions, optogenetics) establish *necessity* and *sufficiency*. LLM interpretability is increasingly making this transition — from correlational probing studies to causal ablation and activation patching — but has not yet adopted the full rigor of neuroscience experimental design (double dissociation, necessity vs. sufficiency distinction, within-subject controls).

### Theme 2: Multi-Scale Analysis

Neuroscience operates simultaneously at molecular, cellular, circuit, region, and systems levels — and understanding requires integration across levels. LLM interpretability has analogous scales: individual neurons, attention heads, circuits (groups of heads), layers (regions), and the full model (system). Current work tends to focus on one scale at a time; Nature-level work would integrate across scales.

### Theme 3: Standardization and Atlas-Building

Neuroscience accumulated decades of scattered findings before atlas-building projects (Brodmann, HCP, Allen Brain Atlas) enabled cumulative science. LLM interpretability is at the pre-atlas stage. The highest-leverage intervention for the field — and the highest-impact paper opportunity — is constructing the first principled functional atlas.

### Theme 4: Development Over Time

Both neuroscience and LLM research suffer from over-focus on static "adult" systems. Developmental neuroscience reveals computational principles invisible in fully-trained systems. The parallel grokking/training dynamics literature suggests LLM development contains rich computational structure that has barely been characterized.

### Theme 5: Comparative/Evolutionary Logic

The comparative neuroscience approach of "evolution as experiment" has been enormously productive. The analogous approach for LLMs — treating the diversity of model sizes, architectures, and training procedures as natural experiments — has been used for scaling laws but not for mechanistic interpretability questions.

---

## Research Landscape Overview

### State of the Field (2026)

The intersection of neuroscience methodology and LLM research has undergone explosive growth from 2022-2026. The field now has:
- Established methods: activation patching, circuit analysis, sparse autoencoders, probing classifiers
- Maturing methods: knowledge neurons, functional localization, training dynamics analysis
- Emerging methods: systematic atlasing, comparative interpretability, developmental analysis
- Largely unexplored: split-brain logic, metaplasticity, multi-scale integration, atlas construction

### Key Research Groups

- Anthropic Interpretability Team (Neel Nanda, Chris Olah): Transformer circuits, sparse autoencoders, monosemanticity
- MIT Brain and Cognitive Sciences (Martin Schrimpf, Greta Tuckute, Ev Fedorenko): Brain-LLM alignment, functional localization
- NYU / Princeton: Knowledge neurons, factual editing
- DeepMind / Google: Scaling laws, circuit analysis consistency across scale
- Multiple academic groups: Connectomics-inspired weight analysis, developmental dynamics

### Research Gaps (Highest Priority for Dr. Zhang)

1. **No comprehensive LLM Functional Atlas exists** — the single highest-impact gap
2. **Split-brain / confabulation experiments** are conceptually rich but empirically underdeveloped
3. **Developmental/grokking dynamics** lack formal connection to neurodevelopmental theory
4. **Comparative interpretability** across architectures lacks rigorous phylogenetic methodology
5. **Multi-scale integration** — connecting neuron-level, circuit-level, and layer-level findings — is entirely missing

---

## Recommendations for Dr. Zhang

### Priority Reading (Start Here)

1. **AlKhamissi et al. (2024)** — "The LLM Language Network" — Best existing example of directly importing neuroscience experimental design into LLM research. NAACL 2025. arXiv:2411.02280
2. **Elhage et al. (2021)** — "A Mathematical Framework for Transformer Circuits" — Foundation for connectomics-style LLM analysis. transformer-circuits.pub
3. **Meng et al. (2022)** — ROME paper — Best example of optogenetics-style causal intervention in LLMs. NeurIPS 2022. arXiv:2202.05262
4. **Wang et al. (2022)** — IOI circuit paper — Best example of lesion/circuit analysis. ICLR 2023. arXiv:2211.00593
5. **Schrimpf et al. (2024)** — Scaling laws for fMRI encoding — Key bridge between fMRI methods and LLMs. NeurIPS 2024.

### Highest-Impact Paper Opportunity

Based on this survey, the highest-impact Nature paper opportunity is:

**"A Functional Atlas of the Transformer: Systematic Mapping of Computational Regions Across Capabilities, Architectures, and Scales"**

This would:
- Use the neuroscience localizer paradigm (AlKhamissi et al.'s method) extended to 50+ functional domains
- Apply the lesion/ablation framework to establish causal necessity of identified regions
- Use activation patching (optogenetics analog) to establish sufficiency
- Compare atlas structure across multiple model families and scales (comparative neuroscience)
- Provide the field's first principled coordinate system — analogous to the Brodmann atlas

This paper would be analogous to Brodmann's 1909 atlas — a reference work that every subsequent interpretability paper cites, and a landmark that defines the field's development for decades.

### Second-Best Opportunity

**"LLM Split-Brain: Dissociating Explanation Generation from Answer Computation in Large Language Models"**

This would:
- Systematically test whether chain-of-thought reasoning traces are mechanistically faithful or confabulatory
- Identify the specific components generating explanations vs. those generating answers
- Test Gazzaniga's "interpreter module" hypothesis in LLMs
- Provide direct relevance to AI safety and LLM trustworthiness

---

## Appendix: Full Citation List

### Core Neuroscience Papers Referenced

| Paper | Year | Method | Key Contribution |
|-------|------|--------|-----------------|
| Broca (1861) | 1861 | Lesion | Localized speech production to left inferior frontal gyrus |
| Hubel & Wiesel (1962) | 1962 | Electrophysiology | Orientation selectivity in V1 |
| White et al. (1986) | 1986 | Connectomics | First complete connectome (C. elegans) |
| Bliss & Lømo (1973) | 1973 | Plasticity | Discovery of LTP |
| Boyden et al. (2005) | 2005 | Optogenetics | First optogenetic neuronal control |
| Quiroga et al. (2005) | 2005 | Electrophysiology | Concept cells ("Jennifer Aniston neurons") |
| Deisseroth (2010) | 2010 | Optogenetics | Optogenetics review, Method of Year |
| Herculano-Houzel (2009) | 2009 | Comparative | Human brain = scaled-up primate brain |
| Glasser et al. (2016) | 2016 | Atlas | HCP 360-area multimodal cortical parcellation |
| Jun et al. (2017) | 2017 | Electrophysiology | Neuropixels probe for large-scale recording |
| MICrONS (2025) | 2025 | Connectomics | 75,000 functionally characterized + EM mouse cortex |

### Core LLM Papers Referenced

| Paper | Year | Method | Key Contribution |
|-------|------|--------|-----------------|
| Elhage et al. (2021) | 2021 | Circuits | Mathematical framework for transformer circuits |
| Dai et al. (2022, ACL) | 2022 | Neuron analysis | Knowledge neurons in FFN layers |
| Wang et al. (2022) | 2022 | Circuits/Lesion | IOI circuit in GPT-2 small |
| Meng et al. (2022) | 2022 | Activation patch | ROME: causal tracing + factual editing |
| Power et al. (2022) | 2022 | Development | Grokking phenomenon |
| Cunningham et al. (2023) | 2023 | Neuron analysis | Sparse autoencoders for LLM features |
| Schrimpf et al. (2024) | 2024 | fMRI-LLM | Scaling laws for brain encoding |
| AlKhamissi et al. (2024) | 2024 | Functional local. | LLM Language Network, NAACL 2025 |
| Yao et al. (2024) | 2024 | Circuits | Knowledge circuits in pretrained transformers |
| Tigges et al. (2024) | 2024 | Comparative | Circuit analyses consistent across scale |
| arXiv:2502.20408 (2025) | 2025 | Brain-inspired | Functional networks and key neurons in LLMs |
| arXiv:2506.21551 (2026) | 2026 | Development | Grokking in LLM pretraining |

---

*Survey conducted May 2026. All papers verified against arXiv, ACL Anthology, NeurIPS proceedings, and Nature/Nature family journals. For papers behind paywalls, arXiv preprint versions are listed where available.*
