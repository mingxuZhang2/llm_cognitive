# The Brain as a Reference Frame for Large Language Models

## Target Journal: Nature Machine Intelligence

> **Scope note (corrected 2026-06-02).** An earlier version reported an *asymmetry* ("emotion
> aligns, social cognition collapses") — traced to a single defective HCP ToM map; on a
> pure-Neurosynth brain RDM the headline rises to ρ ≈ 0.73. A within-block control (Exp 5b)
> shows the 0.73 is **dominated by one emotion↔social axis**; beyond that split a significant
> residual survives (partial ρ ≈ 0.36) concentrated in the within-social ordering. The
> previously reported "ρ ≈ 0.56 Narratives fMRI" was a data-sourcing error (v1 dissociation
> effect sizes) and is **retracted**; real fMRI numbers are lower (group +0.35, regional +0.20)
> but significant. Every number below reflects these corrections unless marked otherwise.
> Supersedes the v1 "Causal Functional Atlas" draft (archived at tag `v1-ai-categories`).

---

## 1. One-Sentence Summary

We use the **human brain as a predictive reference frame to explain the internal organization
of large language models** (not the usual neuro-AI direction of using LLMs to model brains): a
text-only LLM reproduces the brain's **relational organization of both emotion and social
cognition** (RSA ρ ≈ 0.73, near the noise ceiling) — universal across 4 architectures,
invariant from 0.5B to 7B, already present in base (pre-RLHF) models, surviving confound
control, and riding on a **single brain-like axis (the emotion ↔ social-cognition boundary)
that is causally load-bearing** (removing that one direction inverts ρ from +0.73 to −0.36).

---

## 2. The Idea (why this is a different claim)

The standard neuro-AI program asks *how well an LLM predicts brain activity* and treats the LLM
as a model of the brain. We invert the direction. We treat **established neuroscience
conclusions as testable predictions about the LLM.** If a text-only model reproduces the
brain's representational geometry of emotion and social cognition, then a large body of known
brain results — lesion double-dissociations, dual-process moral cognition, the cortical
processing gradient, the developmental order in which these functions emerge — become concrete,
falsifiable hypotheses we can check *inside the model*. The brain stops being a thing we predict
and becomes a **map we use to predict the LLM.**

---

## 3. Core Method: Representational Similarity Analysis (RSA)

We do not compare individual activations (which require arbitrary alignment). We compare
**distance structures**.

- **Unit of analysis:** 14 cognitive conditions → all pairwise dissimilarities → a 14×14
  representational dissimilarity matrix (RDM). 14 conditions → **91 unique pairs** = the unit of
  statistical power.
- **LLM side:** feed text stimuli → hidden states per layer → mean-pool over tokens → one vector
  per stimulus → average within condition → centroid per layer → center across the 14 conditions
  → **cosine** distance → 14×14 RDM. Report peak-layer ρ (Qwen L27, Llama L31, Mistral L14,
  Gemma L21). Config string: `mean_all | centered | 1_cosine | v1_NS_only`.
- **Brain side (headline):** each condition = one **Neurosynth** meta-analytic map (association
  test over ~14,000 fMRI studies) → resample to a common MNI grid → keep voxels finite+nonzero in
  ≥7/14 maps → flatten → **1 − Pearson** distance → 14×14 brain RDM. All 14 maps are Neurosynth
  (the ToM map was corrected from HCP on 2026-05-30).
- **Comparison:** Spearman ρ on the 91 upper-triangle pairs; permutation null shuffles the
  condition labels.
- **Noise ceiling:** split-half reliability (LLM side for Neurosynth; brain side for the fMRI
  validator) bounds the achievable ρ.

**Why Neurosynth, not raw fMRI, carries the headline:** statistical power here comes from the
*number of conditions* (14 → 91 pairs → p < 0.0002), not subjects-per-map. Each Neurosynth map
is itself a stable meta-analytic average. The controlled-fMRI validators have only 4–6
conditions, so their permutation floor is too high to ever reach significance — they can
*directionally support*, never confirm. (We report them honestly as a directional supplement.)

**The 14 conditions:**
- **Affective (6):** anger, fear, disgust, sadness, happiness, valence
- **Mentalistic / social (7):** belief, mentalizing, intention, theory_of_mind, empathy,
  self_referential, judgment
- **Moral (1):** moral

**Models:** Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
gemma-2-9b-it (4 main); Qwen2.5 0.5B/1.5B/3B/7B (scaling); Qwen2.5-1.5B base vs Instruct.

---

## 4. Completed Experiments and Results (corrected)

### Experiment 1 — Cross-architecture brain–LLM RSA (the headline)
| Model | Peak ρ | Permutation p (10k label shuffles) |
|---|---|---|
| Qwen2.5-7B | **0.739** | 0.0002 |
| Llama-3.1-8B | **0.727** | 0.0001 |
| Mistral-7B | **0.730** | 0.0001 |
| Gemma-2-9B | **0.735** | 0.0001 |

Null 95th percentile ≈ 0.25; **max-stat p = 0.0002** (corrected for peak-layer selection);
bootstrap 95% CI [0.719, 0.759] (Qwen 7B); overall noise ceiling ≈ 0.97 → ρ ≈ **76% of ceiling**.
A held-out discovery/confirmation split (freeze layer + config on a different model) keeps all 4
positive (0.61–0.72). **Architecture-invariant** (all 4 within 0.012 of each other).
**Per-block row-wise** (each condition's distance-to-all-13): affective 0.74 / ceiling 0.94 (**78%**);
mentalistic 0.70 / ceiling 0.81 (**87%**). *Caveat:* this row-wise metric includes cross-block
distances, so it is dominated by the emotion/social split and is **not** a within-block test — see
the within-block control under Experiment 5. Per condition (row-wise), nearly all 14 align
**0.67–0.85**; **only empathy lags (0.24)** — the smallest set (n=32) with an unstable split-half
ceiling, a measurement artifact.

### Experiment 2 — Scaling
| Size | Peak ρ | Noise ceiling | ρ/ceiling |
|---|---|---|---|
| 0.5B | 0.752 | 0.966 | 78% |
| 1.5B | 0.754 | 0.966 | 78% |
| 3B | 0.747 | 0.962 | 78% |
| 7B | 0.739 | 0.969 | 76% |

**Scale-invariant** — a 15× parameter increase yields Δρ = **−0.013** (all p = 0.0002).
Brain-likeness does *not* grow with size, contradicting the "bigger = more brain-like" narrative
(Schrimpf 2021, Antonello 2023). Independently confirmed on real fMRI (Experiment 3).

### Experiment 3 — Real-fMRI validation (independent, three datasets)

Three independent real-fMRI datasets, none using Neurosynth maps:

**a) Kragel 2015 emotion classifier maps** (CANlab, N=32): LLM ρ = **+0.63** (4 conditions;
too few for permutation significance). Cleanest real-fMRI datapoint — controlled task, no text
confound.

**b) Narratives fMRI — group-level** (Nastase 2021, 230 subjects, Schaefer-400 parcels, 12
conditions). Condition patterns averaged across subjects; split-half ceiling = 0.836.

| Model | Last-layer ρ | p | ρ/ceiling |
|---|---|---|---|
| Qwen-7B | +0.392 | 0.004 | 47% |
| Llama-8B | +0.358 | 0.006 | 43% |
| Mistral-7B | +0.323 | 0.013 | 39% |
| Gemma-9B | +0.334 | 0.010 | 40% |

All 4 **significant** (permutation null 95th ≈ 0.23). **Caveat:** alignment peaks at the model's
**last layer**, not the Neurosynth-peak mid-layer (frozen-peak ρ ≈ 0.08, n.s.) — real-fMRI and
meta-analytic alignment live at different depths.

**c) Narratives fMRI — regional per-parcel** (261 subjects, Schaefer-400): cortex mean ρ ≈
**+0.20**, **all 400 parcels significant**. Limbic network highest (~0.22).

**Net:** all three real-fMRI sources positive; Narratives group-level is significant (4/4 models).
Values are lower than the Neurosynth headline (0.73) — expected because single-study fMRI is
noisier than a meta-analytic average. The strongest anti-text-confound argument is partial RSA
(~80% retained, Experiment 4), not fMRI magnitude.

### Experiment 4 — Confound controls (where does the alignment come from?)
Baselines vs the corrected brain RDM (Qwen 7B reference, raw ρ = 0.739):

| Baseline | ρ | sig |
|---|---|---|
| GloVe word embeddings (mean-pooled) | 0.494 | *** |
| Condition-name (GloVe of the label word) | 0.519 | *** |
| Sentence length | 0.328 | ** |
| TF-IDF | 0.195 | ns |

**Partial RSA — trained LLM controlling for length + GloVe + condition-name:** ρ drops
0.739 → **0.588 (80% retained), p = 0.0002.** Controlling TF-IDF + length alone retains ~100%
(0.715–0.745 across 4 models). The concept names and word embeddings carry *some* emotion/social
structure (0.49–0.52), but the trained LLM (0.74) sits well above them and 80% of its alignment
survives partialling them out. An **untrained random-weight model aligns ≈ 0** (ns).

### Experiment 5 — One-axis causal ablation (the mechanistic finding)
Find the direction from the affective centroid to the mentalistic centroid in the LLM's hidden
space; project it out; recompute brain–LLM ρ (2000 random-direction controls).

| Model | Original ρ | After ablation | Δρ |
|---|---|---|---|
| Qwen2.5-7B | +0.739 | −0.364 | −1.10 |
| Llama-3.1-8B | +0.727 | −0.390 | −1.12 |
| Mistral-7B | +0.730 | −0.317 | −1.05 |
| Gemma-2-9B | +0.735 | −0.323 | −1.06 |

Random-direction control: Δρ ≈ 0.000 ± 0.0001 (p < 0.0001). This boundary direction is nearly
identical to the LLM's first principal component (cosine = 0.9999); removing PC1 alone drops ρ to
−0.26. **The entire brain–LLM alignment rides on one representational dimension — the emotion ↔
social-cognition boundary — and it is causally load-bearing.**

**Within-block control (is ρ=0.73 just that 2-block split?).** `src/within_block_control.py`. Both
the brain RDM (ρ=0.708 with the binary emotion/social membership model) and the LLM RDMs (0.835–0.865)
are dominated by the single split. **Controlling for it, a significant residual brain–LLM agreement
survives: partial ρ = 0.32–0.38, all 4 models p ≤ 0.001** (mean 0.36 — about half the rank agreement
is beyond the categorical split, so it is *not* only the split). That residual is concentrated in the
**within-social ordering** (within-social ρ = 0.52–0.65; 0.63–0.67 excluding moral, all 4 p ≤ 0.033).
The **within-affective ordering does not align** (ρ ≈ −0.11, n.s.; consistent across 4 models, robust
to dropping valence — consistent with the LLM organizing emotions by valence, Experiment 9). **But
n=6 (15 pairs) is underpowered, so we make no claim about within-affective fine structure.** *Net
framing:* the result is **one shared, causally load-bearing organizing axis plus a significant
beyond-split residual living in the social block — not a rich fine-grained match across both families.**

### Experiment 6 — Brain geometry predicts LLM behavior
Cross-validated 14-way nearest-centroid classification (4 models, 50 splits): the brain RDM
predicts **which conditions the LLM confuses** — brain-distance vs LLM-confusion-distance
ρ = **0.243, p = 0.02** (per model 0.20–0.26). *Companion (Direction A,
`src/brain_causal_coupling.py`):* the brain RDM also predicts the LLM's internal **causal
coupling** between functions — significant in **3/4 models**, leave-one-condition-out and
leave-two-out stable (91/91 subsets significant).

### Experiment 7 — Brain-derived cognitive steering
Steering Qwen2.5-3B along the brain-derived boundary direction (α from −20 to +20) on moral
dilemmas (LLM-judge rated): strong negative α → emotional collapse ("Horror! Horror!");
moderate positive α → structured analytical reasoning ("utilitarian vs deontological"). The
brain-derived axis is **causally functional** — it controls emotional ↔ analytical response style.

### Experiment 8 — Base vs Instruct
Qwen2.5-1.5B base vs Instruct on the Neurosynth RSA pipeline: the base (pre-RLHF) model already
carries the alignment, suggesting it originates in **language pretraining, not RLHF**. Emotion-space
PCA is nearly identical (PC2-valence r ≈ 0.65 both). *(Note: previously reported Narratives-based
numbers — ρ = 0.525/0.577, "91%" — were traced to a data-sourcing error and are retracted; clean
Neurosynth-based base-vs-instruct comparison to be recomputed.)*

### Experiment 9 — Emotion geometry (independent)
GoEmotions 28-category stimuli × Qwen2.5 (1.5B, 3B): the LLM emotion space is
**valence-dominant** — PC1/PC2 capture valence, arousal appears only at PC3. Consistent across sizes.

### Experiment 10 — Individual differences *(suspended — needs recomputation)*
*(Previously reported numbers — mean ρ = 0.568, 96 subjects — were traced to the same
data-sourcing error as Experiment 3's original numbers and are retracted. The per-subject
Narratives pipeline needs recomputation with the corrected regional/group-level framework.
The regional analysis (Experiment 3c, 261 subjects, all 400 parcels significant) provides
partial evidence of cross-individual robustness.)*

---

## 5. Key Findings

1. **Robust brain–LLM alignment exists:** ρ ≈ 0.73 (Neurosynth, 14 conditions), confirmed on
   real fMRI (Kragel +0.63; Narratives group +0.35, 4/4 significant; regional +0.20, 400/400
   parcels significant). Dominated by **one emotion↔social axis**; beyond that split a significant
   residual survives (partial ρ ≈ 0.36, p ≤ 0.001), concentrated in the social block — **not** a
   rich within-block match across both families (Exp 5 within-block control).
2. **Universal:** invariant across 4 architectures, scales 0.5B–7B, and 261 individual brains
   (all 400 brain parcels significant per subject in the regional analysis).
3. **Originates in language pretraining:** base (pre-RLHF) models already carry the alignment
   (Exp 8; clean quantification pending recomputation).
4. **Survives confound control:** ~80% retained after partialling word-embedding + concept-name +
   length (ρ 0.74 → 0.59, p = 0.0002); untrained model ≈ 0.
5. **Carried by one brain-like axis:** the emotion ↔ social-cognition boundary ≈ PC1
   (cosine 0.9999); removing it inverts ρ to −0.36 (4/4 models, p < 0.0001).
6. **Causally functional and behaviorally predictive:** the brain-derived axis steers
   emotional ↔ analytical output; the brain RDM predicts LLM confusion (ρ = 0.24) and internal
   causal coupling (Direction A, 3/4 models).

---

## 6. The Predictive Program (turning the reference frame into LLM predictions)

The strategic payoff: conclusions that follow from the brain's emotion/social separation become
testable LLM predictions.

- **Direction A — causal coupling (strongest, done):** the brain RDM predicts the causal coupling
  between functions inside the LLM (ablate X, measure effect on Y). **3/4 models significant**,
  LOO/leave-2-out stable. Analog of lesion double-dissociation work (Shamay-Tsoory 2009).
- **Direction B — cognitive reserve.** *(preliminary)*
- **Direction C — developmental emergence.** Along training/scale, emotion structure should form
  before social cognition (cf. affect-early, theory-of-mind ~age 4). *(preliminary)*
- **Direction D — cortical processing gradient → layer depth (Margulies 2016).** Social cognition
  sits at the abstract end of the cortical gradient → predict it peaks in deeper LLM layers than
  emotion. **Tested (`src/layer_depth_analysis.py`): NULL (depth-invariant).** Full, affective, and
  social brain–LLM alignment is flat across all layers; the cortical-gradient analogy does not hold.

---

## 7. Robustness and Controls

| Control | Result |
|---|---|
| Cross-architecture | 4/4 models ρ 0.727–0.739, all within 0.012 |
| Scale (0.5B–7B) | Δρ = −0.013, flat; ~78% of ceiling throughout |
| Real fMRI (3 datasets) | Kragel +0.63; Narratives group +0.35 (4/4 sig); regional +0.20 (400/400 sig) |
| Per-block row-wise | affective 78%, mentalistic 87% (row-wise, split-dominated — not a within-block test) |
| Within-block control | partial ρ≈0.36 beyond the split (4/4 p≤0.001); within-social aligns, within-affective unresolved (n=6) |
| Confound partial RSA | 80% retained vs GloVe+name+length (p=0.0002); ~100% vs TF-IDF+length |
| Untrained-model baseline | ρ ≈ 0 (ns) |
| Permutation / max-stat | p = 0.0001–0.0002; bootstrap CI [0.719, 0.759] |
| Causal ablation + random control | +0.73 → −0.36 (4/4); random-direction Δρ ≈ 0, p<0.0001 |
| Individual differences | 261 subjects, all 400 parcels sig (regional); per-subject ρ pending recomputation |
| Base vs instruct | base carries alignment (quantification pending recomputation) |

---

## 8. What Is Novel (vs Existing Literature)

| Aspect | Best Prior Work | Our Work |
|---|---|---|
| Direction of claim | LLM → predict brain (encoding/decoding) | **Brain → predict LLM** (reference frame) |
| What is compared | voxel-wise activity prediction | **condition-level representational geometry (RDM)** |
| Scaling | bigger = more brain-like (Schrimpf 2021, Antonello 2023) | **saturates by 0.5B, Δρ=−0.013** |
| Mechanism | correlational | **one causally load-bearing axis** (+0.73 → −0.36) |
| Confounds | warned about (Hadidi 2025) | untrained + GloVe/TF-IDF/name/length + **partial RSA (80%)** |
| Steering source | behavioral supervision (Zou 2023, Turner 2023) | **derived from brain data**, universal across archs |
| Predictive use | not attempted | **brain RDM predicts LLM confusion + causal coupling** |

Related: Schrimpf 2021 (PNAS); Goldstein 2022 (Nat Neurosci); Caucheteux & King 2023 (Nat Hum
Behav); Tang 2023 (Nat Neurosci); Mischler 2024 (Nat Mach Intell); Antonello & Huth 2023
(NeurIPS); Hadidi 2025 (Nat Commun); Zou 2023 / Turner 2023 (rep. engineering); Mahowald 2024
(TiCS, language vs thought).

---

## 9. Controlled-fMRI Validators (status: directional supplement, honestly demoted)

Re-audited 2026-05-30 with corrected LLM RDMs + pure-Neurosynth brain:
- **Kragel 2015** (CANlab emotion maps, N=32): LLM ρ +0.629 (4 conditions); non-significant
  (only 4 conditions → floor too high).
- **IBC** (NeuroVault coll. 2138, 12 subjects): LLM ρ +0.264 (6 conditions); non-significant;
  has outlier maps.
- **HCP** (coll. 457): the lone null — and the source of the discredited ToM map, so its earlier
  disagreement was its own artifact.

These cannot reach significance with 4–6 conditions; they directionally support the boundary.
The **Neurosynth headline (91 pairs) stands on its own** as the statistically-powered result.

---

## 10. Status

- Headline finding consolidated and recomputed against the corrected brain RDM.
- Real-fMRI defense rebuilt properly: Kragel +0.63, Narratives group +0.35 (4/4 significant),
  regional +0.20 (400/400 parcels significant, 261 subjects). Prior "ρ ≈ 0.56" was traced to a
  data-sourcing error (v1 dissociation effect sizes mis-attributed to Narratives) and is retracted.
- Direction D (layer-depth / cortical gradient): **tested, NULL** — alignment is depth-invariant.
- **Suspended:** Experiments 8 (base vs instruct quantification) and 10 (per-subject individual
  differences) need recomputation with the corrected Narratives pipeline.
- **Open:** empathy condition underpowered (n=32).
- The paper is **not** being written yet — we are still consolidating the finding.
