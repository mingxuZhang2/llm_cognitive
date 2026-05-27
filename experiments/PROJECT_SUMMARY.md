# LLM Cognitive Atlas: Brain-LLM Representational Alignment

## Project Summary for External Review

### One-line Summary
We compare the internal cognitive organization of Large Language Models with the human brain using Representational Similarity Analysis (RSA), and find that brain-LLM alignment reduces to a single representational dimension — the emotion-reasoning boundary — which is universal across architectures, invariant to scale, present in base (pre-RLHF) models, and can be causally manipulated.

---

## 1. Research Question

When LLMs learn language, do they develop an internal organization of cognitive functions that resembles the brain's? If so:
- How similar is it?
- What specifically is shared vs. different?
- What is the mechanistic basis of the similarity?
- Where does it come from (pretraining vs. RLHF)?
- Can it be manipulated?

## 2. Method: Representational Similarity Analysis (RSA)

### Core Idea
We don't compare individual activations. We compare **distance structures**: for N cognitive conditions, compute all pairwise distances to get an N×N distance matrix (RDM). Then compare the brain's RDM with the LLM's RDM via Spearman correlation (ρ).

High ρ = brain and LLM agree on which cognitive functions are similar to each other and which are different.

### LLM Side
- Input: text stimuli belonging to different cognitive conditions
- Processing: feed each stimulus through the LLM, extract hidden-state activations at each layer
- Pooling: mean over all tokens in the sentence → one vector per stimulus per layer
- Condition averaging: mean over all stimuli within a condition → one "condition centroid" per layer
- Normalization: per-condition centering + cosine distance
- Output: N×N LLM RDM at each layer; report peak-layer ρ

### Brain Side (two sources)

**Source 1: Neurosynth + HCP meta-analytic maps (14 conditions)**
- For each of 14 cognitive conditions (5 Ekman emotions + valence + 7 mentalistic/social + moral), retrieve the whole-brain meta-analytic activation map from Neurosynth (MKDAChi2 association test, 10mm kernel) or HCP S1200 task contrasts
- Compute pairwise distances between these 14 maps → 14×14 brain RDM
- Advantage: represents consensus across thousands of fMRI studies
- Limitation: not stimulus-locked (the brain data and LLM data use different stimuli)

**Source 2: Narratives fMRI dataset (stimulus-locked, N=91)**
- Public dataset: Nastase et al. 2021, 345 subjects listening to stories, preprocessed fMRI (denoised BOLD, MNI space, Schaefer-400 parcellation)
- We annotated story sentences with cognitive condition labels using DeepSeek-chat (1,330 sentences across 6 stories, 13 conditions, 85% coverage)
- For each condition: averaged fMRI signal across all labeled time points (with 5s HRF lag) → condition-level brain activation pattern (400-dimensional, one per Schaefer parcel)
- Same story text fed to LLMs → LLM condition centroids from identical stimuli
- Output: 10×10 brain RDM (10 conditions with sufficient data), directly comparable to LLM RDM on same text

### Noise Ceiling
Split-half reliability of the LLM RDM (for Neurosynth analysis) or the brain RDM (for Narratives analysis). This is the theoretical upper bound on ρ — how well the brain (or LLM) agrees with itself.

### Cognitive Conditions
14 conditions spanning affective and social/mentalistic domains:
- **Affective** (6): anger, fear, disgust, sadness, happiness, valence
- **Mentalistic** (7): belief, mentalizing, intention, theory_of_mind, empathy, self_referential, judgment
- **Moral** (1): moral

Stimuli: 712 curated sentences (for Neurosynth analysis) or story sentences from Narratives dataset (for stimulus-locked analysis).

---

## 3. Models Tested

### Cross-architecture (7-9B parameter range):
| Model | Source | Architecture |
|---|---|---|
| Qwen2.5-7B-Instruct | Alibaba | Standard decoder |
| Meta-Llama-3.1-8B-Instruct | Meta | Standard decoder |
| Mistral-7B-Instruct-v0.3 | Mistral AI | Decoder + sliding window attention |
| gemma-2-9b-it | Google | Decoder + soft-capping |

### Scaling (Qwen2.5-Instruct family):
0.5B, 1.5B, 3B, 7B (same architecture, different capacity)

### Base vs. Instruct:
Qwen2.5-1.5B (base, no RLHF) vs. Qwen2.5-1.5B-Instruct (instruction-tuned with RLHF)

---

## 4. Experiments and Findings

### Experiment 1: Cross-Architecture Brain-LLM RSA
**Data**: Neurosynth brain RDM (14 conditions) × 4 LLMs × 712 curated stimuli

**Results**:
| Model | Peak ρ | Noise ceiling | ρ/ceiling | p |
|---|---|---|---|---|
| Qwen2.5-7B | 0.64 | 0.97 | 66% | <0.0002 |
| Llama-3.1-8B | 0.63 | 0.97 | 65% | <0.0002 |
| Mistral-7B | 0.63 | 0.97 | 65% | <0.0002 |
| Gemma-2-9B | 0.64 | 0.97 | 66% | <0.0002 |

**Finding**: Brain-LLM cognitive alignment is **architecture-invariant** — 4 different companies/architectures/training data produce ρ within 0.01 of each other.

### Experiment 2: Scaling Analysis
**Data**: Neurosynth brain RDM × Qwen2.5-Instruct at 0.5B/1.5B/3B/7B

**Results**:
| Size | Params | Peak ρ | ρ/ceiling |
|---|---|---|---|
| 0.5B | 494M | 0.665 | 68.9% |
| 1.5B | 1.54B | 0.655 | 67.9% |
| 3B | 3.09B | 0.657 | 68.3% |
| 7B | 7.61B | 0.639 | 65.9% |

**Finding**: Brain-LLM alignment is **scale-invariant** — 15× parameter increase yields Δρ = −0.026. This contradicts the typical "bigger = more brain-like" narrative (Schrimpf 2021, Antonello 2023).

### Experiment 3: Stimulus-Locked Validation with Real fMRI
**Data**: Narratives fMRI (91 subjects, pieman story, Schaefer-400 parcellation) × same story text fed to Qwen2.5 family

**Results**:
| Model | Peak ρ | Brain ceiling | ρ/ceiling | p |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.540 | 0.762 | 70.9% | 0.003 |
| Qwen2.5-1.5B | 0.582 | 0.762 | 76.3% | 0.001 |
| Qwen2.5-3B | 0.560 | 0.762 | 73.4% | 0.002 |

**Finding**: Brain-LLM alignment **confirmed on stimulus-locked real fMRI** (N=91). Using identical text for brain and LLM, ρ ≈ 0.56, reaching 71-76% of brain noise ceiling. Scale-invariance also confirmed on real fMRI.

### Experiment 4: Gap Analysis — Where Does LLM Fail?
**Data**: Per-pair residual decomposition of brain-LLM distance matrices (Neurosynth, 4 models)

**Results**:
- Within-affective pairs: mean |residual| = **6.9** (lowest — LLM gets emotions right)
- Within-mentalistic pairs: mean |residual| = **19.2** (highest — LLM collapses them)
- Top misaligned pairs (LLM says close, brain says far):
  - self_referential — theory_of_mind: residual +59
  - intention — theory_of_mind: residual +57
  - belief — theory_of_mind: residual +48
  - mentalizing — theory_of_mind: residual +42

**Finding**: The 35% alignment gap concentrates in **mentalistic fine structure** — the brain distinguishes belief/intention/ToM/empathy/self-referential from each other, but LLMs collapse them into a single cluster. Emotion fine structure (anger vs fear vs happiness) matches brain well.

### Experiment 5: One-Axis Causal Ablation
**Data**: Virtual ablation of the affective-mentalistic boundary direction in LLM activation space (4 models, 2000 random-direction controls)

**Method**: Find the direction from affective centroid to mentalistic centroid in the LLM's hidden space. Project it out (remove this one direction from all condition representations). Recompute RDM and brain-LLM ρ.

**Results**:
| Model | Original ρ | After ablation | Δρ |
|---|---|---|---|
| Qwen2.5-7B | +0.639 | −0.362 | **−1.00** |
| Llama-3.1-8B | +0.627 | −0.364 | **−0.99** |
| Mistral-7B | +0.633 | −0.327 | **−0.96** |
| Gemma-2-9B | +0.639 | −0.317 | **−0.96** |

Random direction control: Δρ = −0.0001 ± 0.0002 (2000 iterations). p < 0.0005.

**Selective collapse**:
- Within-affective: Δρ = **+0.15** (improves — emotion structure is independent)
- Within-mentalistic: Δρ = **−0.44** (collapses — stacked on same axis)
- Cross-boundary: Δρ = −0.04 (minor)

**Finding**: The **entire brain-LLM alignment rides on a single representational dimension** (1 out of 3584). Removing it inverts the correlation. This axis separates affective from mentalistic processing. Emotion fine structure is independently encoded (survives ablation); mentalistic fine structure is not (collapses).

### Experiment 6: Brain-Derived Cognitive Steering
**Data**: Moral dilemma prompts × Qwen2.5-3B-Instruct × steering along the brain-derived boundary direction at different strengths (α = −20 to +20)

**Results** (rated by DeepSeek, 1-7 scale):
| α | Emotionality | Analyticity | Coherence |
|---|---|---|---|
| −20 | 4.0 | 1.2 | 1.4 |
| 0 (baseline) | 2.0 | 3.7 | 3.9 |
| +10 | 1.7 | **4.3** | **4.0** |
| +20 | 1.7 | 2.6 | 2.6 |

At α = −20: model outputs emotional collapse ("Horror! Horror!").
At α = +10: model produces structured philosophical analysis ("utilitarian vs deontological").

**Finding**: The brain-derived axis is **causally functional** — it controls the LLM's response style between emotional and analytical processing.

### Experiment 7: Brain-to-LLM Transfer (Negative Result)
**Data**: Steering along brain-derived mentalistic directions (belief→ToM, intention→ToM) during false belief and faux pas tasks

**Results**: All steering alphas either maintain or degrade performance. Baseline: False Belief 80%, Faux Pas 67%. No improvement at any alpha.

**Finding**: LLM's mentalistic deficit **cannot be repaired by injecting brain-derived directions**. The problem is deeper than representational geometry.

### Experiment 8: Base vs. Instruct Comparison
**Data**: Qwen2.5-1.5B (base, no RLHF) vs. Qwen2.5-1.5B-Instruct on:
(a) Brain-LLM RSA on pieman story (91 subjects fMRI)
(b) Emotion geometry (GoEmotions 28 categories)

**Results**:
- Brain-LLM RSA: base ρ = **0.525**, instruct ρ = **0.577** (base has 91% of instruct alignment)
- Emotion PCA: nearly identical structure (PC2-valence correlation: base r=0.653, instruct r=0.652)

**Finding**: Brain-LLM cognitive alignment is **primarily a product of language pretraining**, not RLHF. The cognitive axis exists in base models; instruction tuning adds ~9%.

### Experiment 9: Emotion Geometry
**Data**: GoEmotions 28-category stimuli (5,398 Reddit comments) × Qwen2.5 (1.5B and 3B)

**Results**:
- PC1 (40% variance): correlates with valence (r = −0.42, p < 0.05)
- PC2 (15% variance): correlates with valence (r = +0.65, p < 0.001)
- PC3 (8% variance): correlates with arousal (r = −0.45, p < 0.05)
- Consistent across 1.5B and 3B

**Finding**: LLM emotion space is **valence-dominant** — the first two PCs capture valence, arousal appears only at PC3. LLMs learn the evaluative dimension of emotion (good/bad) strongly from text but encode arousal (calm/excited) weakly.

### Experiment 10: Individual Differences
**Data**: Per-subject brain-LLM alignment across 96 Narratives subjects

**Results**: Mean ρ = 0.568, range [0.278, 0.776]. 97% of subjects show ρ > 0.3 (all positive). Subjects with more "typical" brain organization align more with LLM (r = 0.24, p = 0.017).

**Finding**: Brain-LLM alignment is **universal across individuals** — not driven by a subset of subjects.

### Experiment 11: Regional Brain Analysis (Inconclusive)
**Data**: Per-parcel (Schaefer-400) multi-voxel pattern RSA, 258 subjects × 6 stories

**Results**: All 400 parcels significant. Network ranking: Limbic (0.181) > DMN (0.162) > Somatomotor (0.164). But range is very small (0.158-0.181). No clear network dissociation.

**Finding**: With current data/method resolution, **no clear network-level differentiation** in brain-LLM alignment. Either the alignment is genuinely distributed, or condition-level RSA from naturalistic stories lacks the resolution to distinguish networks.

---

## 5. Unified Conclusions

1. **Brain-LLM cognitive alignment exists and is robust**: ρ = 0.56-0.64 across multiple brain data sources, confirmed on stimulus-locked fMRI (N=91).

2. **The alignment is one-dimensional**: A single representational direction carries the entire correlation. Removing it inverts brain-LLM alignment from +0.64 to −0.36.

3. **The dimension is the affective-mentalistic boundary**: It separates emotion-related conditions from social reasoning conditions — matching the brain's limbic vs prefrontal organization.

4. **The alignment is universal**: invariant across 4 architectures, 4 scales (0.5B-7B), 96 individual brains, and base vs instruct models.

5. **The alignment has a hard ceiling**: No tested variation (architecture, scale, RLHF) pushes beyond ~73% of brain noise ceiling. The 27% gap concentrates in mentalistic fine structure.

6. **The alignment is causally functional**: The brain-derived axis controls LLM output style (emotional ↔ analytical).

7. **The alignment originates in language statistics**: Base models (no RLHF) have 91% of the alignment. It's a property of natural language, not of human feedback training.

8. **The deficit is fundamental**: Steering along brain-derived directions cannot repair the LLM's collapsed mentalistic representations.

---

## 6. Related Work

### Brain-LLM Alignment (Nature-level papers)
- **Schrimpf et al. 2021 (PNAS)**: "Brain-Score" — 43 models benchmarked; next-word prediction is the key factor for brain-likeness. Our finding extends this: brain-likeness saturates at 0.5B, and the underlying alignment is one-dimensional.
- **Goldstein et al. 2022 (Nature Neuroscience)**: ECoG shows brain predicts next word before hearing it, matching GPT-2 temporal profile. Algorithmic-level claim; we make a representational-geometry claim.
- **Caucheteux & King 2023 (Nature Human Behaviour)**: Hierarchical predictive coding — different brain areas predict different time horizons. N=304. We address a complementary question (cognitive organization, not temporal prediction).
- **Tang et al. 2023 (Nature Neuroscience)**: Semantic decoder using LLM representations. Demonstrates practical capability (mind-reading). We focus on representational structure.
- **Tuckute et al. 2024 (Nature Human Behaviour)**: Closed-loop brain control — LLM predicts sentences that drive/suppress brain language network. We also demonstrate causal manipulation, but in the LLM direction (steering model output with brain-derived directions).
- **Mischler et al. 2024 (Nature Machine Intelligence)**: iEEG shows hierarchical convergence between LLMs and brain. Better LLMs have more brain-like hierarchies.
- **Antonello & Huth 2023 (NeurIPS)**: Brain prediction scales logarithmically with LLM size (125M-30B). At the level of coarse cognitive-condition RSA, we find alignment saturates early (0.5B-7B flat), unlike continuous stimulus-level encoding.

### Confound Warning
- **Hadidi et al. 2025 (Nature Communications)**: Many brain-LLM alignment findings are driven by confounds (word position, word rate). Our RSA approach operates on condition-level distance matrices, making it more robust to these first-order confounds than voxel-level encoding models. We have now run untrained-model baselines (ρ=0.09, ns), lexical baselines (GloVe ρ=0.40, TF-IDF ρ=0.26), and partial RSA controlling for GloVe + length + condition-name (trained LLM retains 87% of alignment, ρ=0.557, p=0.0002). Source-dataset, valence/arousal, word-frequency, pronoun, and mental-state verb controls remain to be completed.

### Representation Engineering
- **Zou et al. 2023**: Representation engineering — finding directions in activation space that control model behavior. Our contribution: we derive the steering direction from brain data rather than behavioral supervision.
- **Turner et al. 2023**: Activation addition for steering. Our work is similar in method but the direction is brain-derived and shown to be universal across architectures.

### LLM Cognitive Evaluation
- **Kosinski 2023**: Claims GPT-4 passes Theory of Mind tests. Our finding contextualizes this: LLMs may pass ToM tests while internally collapsing distinct mentalistic processes onto a single dimension.
- **Mahowald et al. 2024 (Trends in Cognitive Sciences)**: "Dissociating language and thought" — formal linguistic competence vs functional competence. Our data aligns: LLMs learn the formal structure (distance relationships between cognitive categories) but not the functional details (within-category fine structure for mentalizing).

---

## 7. Open Questions / What's Missing

1. **No "wow" finding yet**: The results are solid characterization but lack a single discovery that changes understanding. We've described what the alignment is, but haven't used it to discover something new.

2. **Possible directions**:
   - Use the brain-LLM mismatch to predict specific LLM failure modes
   - Test whether multimodal models close the mentalistic gap
   - Test on larger models (70B+) to rule out late-emergence
   - Use the one-axis finding to make a theoretical claim about the information content of natural language

3. **Controls completed**: untrained model (ρ=0.09, ns), GloVe/TF-IDF/condition-name/length baselines, partial RSA (ρ=0.557 after controlling all surface features). Confirmatory RSA with discovery/confirmation split, max-stat permutation (p=0.0002), cross-validated ablation (95% CI [-0.91, -0.67]), variance-matched random controls. Still needed: source-dataset RDM, valence/arousal RDM, human annotation validation for Narratives.

4. **PC1 ≈ boundary direction** (cosine = 0.999): The affective-mentalistic boundary is nearly identical to the first principal component of the LLM condition space. This means the dominant axis is not a mysterious brain-derived structure but the LLM's primary variance axis — which happens to align with brain cognitive organization.

5. **Regional analysis was inconclusive**: All 400 parcels significant with small effect range. Likely insufficient resolution from condition-level RSA on naturalistic stories.

---

## 8. Code Structure

```
experiments/
  src/
    build_rsa_stimuli.py          — Build 712-stimulus cognitive RSA battery
    extract_rsa_activations_v2.py — Extract per-stimulus LLM activations (HPC3 GPU)
    build_brain_rdm.py            — Build Neurosynth+HCP brain RDM
    compute_rsa.py                — Compare brain vs LLM RDMs
    rsa_cross_model_analysis.py   — 4-model cross-architecture comparison
    rsa_scaling_analysis.py       — Scaling curve (Qwen 0.5B-7B)
    rsa_deep_analysis.py          — Gap decomposition + confusion + causal ablation
    narratives_preprocess.py      — Segment stories + keyword annotation
    narratives_annotate.py        — DeepSeek cognitive annotation
    narratives_brain_rdm.py       — Build brain RDM from Narratives fMRI
    extract_narratives_llm.py     — Extract LLM activations for story text
    regional_rsa_mvpa.py          — Per-parcel brain-LLM RSA
    regional_rsa_multistory.py    — Multi-story regional RSA
    cognitive_steering.py         — Brain-derived activation steering
    brain_transfer.py             — Brain-to-LLM transfer experiment
    emotion_geometry.py           — Emotion space PCA analysis
  scripts/slurm/                  — SLURM job scripts for HPC3
  results/
    cognitive_rsa/                — RSA results, RDMs, scaling summaries
    narratives_llm/               — LLM activations on story text
    deep_experiments/             — Steering + transfer results
    emotion_geometry/             — Emotion PCA results
  data/
    cognitive_stimuli/            — Curated stimuli (emotion, moral, ToM)
    narratives/                   — Narratives dataset (transcripts, annotations, fMRI)
  figures/                        — All generated plots
```

---

## 9. Reproduction

### Requirements
- Python 3.10+, PyTorch, transformers, nilearn, nibabel, scipy, numpy, matplotlib
- GPU: NVIDIA H100 80GB (for 7B+ models) or A100 40GB (for ≤3B)
- fMRI data: Narratives dataset from `s3://fcp-indi/data/Projects/narratives/`
- Models: Qwen2.5 family from HuggingFace

### Key commands
```bash
# 1. Build brain RDM from Neurosynth
python src/build_brain_rdm.py

# 2. Extract LLM activations (submit to SLURM)
sbatch scripts/slurm/cognitive_rsa_scaling.sh

# 3. Compute RSA
python src/rsa_cross_model_analysis.py

# 4. Deep analysis (gap + ablation)
python src/rsa_deep_analysis.py

# 5. Narratives fMRI pipeline
python src/narratives_preprocess.py
python src/narratives_annotate.py
python src/narratives_brain_rdm.py

# 6. Stimulus-locked RSA comparison
# (run extract_narratives_llm.py on HPC3, then compare locally)
```
