# The Brain as a Reference Frame for Large Language Models

## Target Journal: Nature Machine Intelligence

> **Scope note (corrected 2026-05-31).** An earlier version of this project reported an
> *asymmetry* — "emotion aligns, social cognition collapses; brains separate minds, language
> models compress them." That asymmetry was traced to a **single defective brain map** (the
> lone non-Neurosynth map, HCP `theory_of_mind`, orthogonal to its Neurosynth counterpart,
> row-correlation −0.016). On a pure-Neurosynth brain RDM the asymmetry dissolves, the headline
> *rises* (ρ 0.63 → 0.73), and *both* emotion and social cognition align near the noise ceiling.
> Every number below is recomputed against the corrected RDM unless marked "(independent)".
> This document supersedes the earlier "Causal Functional Atlas of LLMs" draft (the v1
> AI-task-category work, archived at git tag `v1-ai-categories`).

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

### Experiment 3 — Stimulus-locked validation on real fMRI (independent)
| Model | Peak ρ | Brain ceiling | ρ/ceiling | p |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.540 | 0.762 | 70.9% | 0.003 |
| Qwen2.5-1.5B | 0.582 | 0.762 | 76.3% | 0.001 |
| Qwen2.5-3B | 0.560 | 0.762 | 73.4% | 0.002 |

Narratives fMRI (Nastase et al. 2021, N=91): the **same story text** is fed to brain and LLM →
directly comparable RDMs on identical stimuli (no Neurosynth maps; unaffected by the ToM
correction). ρ ≈ 0.56, reaching **71–76% of the brain noise ceiling**. Confirms the alignment is
genuine and stimulus-driven, not an artifact of meta-analytic maps. Scale-invariance reproduced.

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

### Experiment 8 — Base vs Instruct (independent)
Brain–LLM RSA on the pieman story (N=91): base ρ = 0.525, instruct ρ = 0.577 — **base models
already carry ~91% of the alignment.** Emotion-space PCA is nearly identical (PC2-valence
r ≈ 0.65 both). The alignment is primarily a product of **language pretraining, not RLHF.**

### Experiment 9 — Emotion geometry (independent)
GoEmotions 28-category stimuli × Qwen2.5 (1.5B, 3B): the LLM emotion space is
**valence-dominant** — PC1/PC2 capture valence, arousal appears only at PC3. Consistent across sizes.

### Experiment 10 — Individual differences (independent)
Per-subject brain–LLM alignment across 96 Narratives subjects: mean ρ = 0.568, range
[0.278, 0.776]; **97% of subjects ρ > 0.3** (all positive). The alignment is **universal across
individuals**, not driven by a subset.

---

## 5. Key Findings

1. **Robust brain–LLM alignment exists:** ρ ≈ 0.73 (Neurosynth) and ≈ 0.56 (stimulus-locked real
   fMRI, N=91), near the noise ceiling, over the 14-condition set spanning emotion and social
   cognition. It is **dominated by one emotion↔social axis**; beyond that split a significant
   residual survives (partial ρ ≈ 0.36, p ≤ 0.001), concentrated in the social block — **not** a
   rich within-block match across both families (Exp 5 within-block control).
2. **Universal:** invariant across 4 architectures, scales 0.5B–7B, 96 individual brains, and
   base vs instruct models.
3. **Originates in language pretraining:** base models carry ~91% of the alignment; RLHF adds little.
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
  sits at the abstract end of the cortical gradient → predict it peaks in *deeper* LLM layers than
  emotion. **Cheapest next test — per-layer data already on disk, no new GPU runs needed.**

---

## 7. Robustness and Controls

| Control | Result |
|---|---|
| Cross-architecture | 4/4 models ρ 0.727–0.739, all within 0.012 |
| Scale (0.5B–7B) | Δρ = −0.013, flat; ~78% of ceiling throughout |
| Independent real fMRI (Narratives, N=91) | ρ ≈ 0.56, 71–76% of brain ceiling, identical text |
| Per-block row-wise | affective 78%, mentalistic 87% (row-wise, split-dominated — not a within-block test) |
| Within-block control | partial ρ≈0.36 beyond the split (4/4 p≤0.001); within-social aligns, within-affective unresolved (n=6) |
| Confound partial RSA | 80% retained vs GloVe+name+length (p=0.0002); ~100% vs TF-IDF+length |
| Untrained-model baseline | ρ ≈ 0 (ns) |
| Permutation / max-stat | p = 0.0001–0.0002; bootstrap CI [0.719, 0.759] |
| Causal ablation + random control | +0.73 → −0.36 (4/4); random-direction Δρ ≈ 0, p<0.0001 |
| Individual differences | 97% of 96 subjects ρ > 0.3, all positive |
| Base vs instruct | base = 91% of instruct → pretraining, not RLHF |

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

- Headline finding consolidated and recomputed against the corrected brain RDM; all docs, README,
  and the plain-language briefing deck (`experiments/present/index.html`) reflect the corrected story.
- Code synced to GitHub (`main` / `cognitive-atlas` at the corrected commit); large activation/
  attribution NPZ are GPU-reproducible intermediates, kept locally, excluded from the repo.
- **Open:** empathy condition is underpowered (n=32) — rebuild with a proper empathy-induction set;
  run Direction D (layer-depth / cortical-gradient) as the next cheap empirical test.
- The paper is **not** being written yet — we are still consolidating the finding.
