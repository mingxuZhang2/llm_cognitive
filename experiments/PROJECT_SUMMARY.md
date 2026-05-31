# Brain-LLM Representational Alignment: Project Summary

## Project Summary for External Review

> **Corrected 2026-05-31.** An earlier version of this document reported an *asymmetry*
> ("emotion aligns, social cognition collapses; brains separate minds, language models
> compress them"). That asymmetry was traced to a single defective brain map (the lone
> non-Neurosynth map, HCP `theory_of_mind`, orthogonal to its Neurosynth counterpart,
> row-correlation −0.016). On a pure-Neurosynth brain RDM the asymmetry dissolves, the
> headline rises (ρ 0.63 → 0.73), and *both* emotion and social cognition align near the
> noise ceiling. All numbers below are recomputed against the corrected RDM unless marked
> "(independent)".

### One-line Summary
A text-only LLM reproduces the human brain's **relational organization of both emotion and
social cognition** (RSA ρ ≈ 0.73, near noise ceiling). The alignment is universal across 4
architectures, invariant to scale (0.5B–7B), present in base (pre-RLHF) models, survives
confound control (~80% retained after partialling word-embedding + concept-name + length),
and rides on a **single brain-like axis — the emotion ↔ social-cognition boundary — that is
causally load-bearing** (removing this one direction inverts ρ from +0.73 to −0.36).

---

## 1. Research Question

When an LLM learns language, does it develop an internal organization of cognitive functions
that resembles the human brain's? If so: how similar, what specifically is shared, what is
the mechanistic basis, where does it come from (pretraining vs RLHF), and can it be
manipulated? Framing: we use the brain as a **predictive reference frame** — established
neuroscience becomes testable predictions about the LLM.

## 2. Method: Representational Similarity Analysis (RSA)

### Core idea
We do not compare individual activations. We compare **distance structures**: for N cognitive
conditions, compute all pairwise distances → an N×N representational dissimilarity matrix
(RDM). Compare the brain's RDM to the LLM's RDM via Spearman ρ on the upper triangle
(14 conditions → 91 pairs). High ρ = brain and LLM agree on which functions are alike.

### LLM side
Feed text stimuli through the LLM → hidden states per layer → mean-pool over tokens → one
vector per stimulus → average within condition → condition centroid per layer → center across
the 14 conditions → cosine-distance RDM. Report peak-layer ρ (Qwen L27, Llama L31, Mistral
L14, Gemma L21). Reconstruct with `src/reconstruct_headline_rdms.py`.

### Brain side (two independent sources)
**Source 1 — Neurosynth meta-analytic maps (14 conditions) — THE HEADLINE.** Each condition =
one Neurosynth association-test map (meta-analysis over ~14,000 fMRI studies). Build: resample
each map to a common MNI grid → keep voxels finite+nonzero in ≥7/14 maps → flatten →
1−Pearson distance → 14×14 brain RDM (`src/build_brain_rdm.py`, `brain_rdm.npz`). All 14 maps
are now Neurosynth (the ToM map was corrected from HCP on 2026-05-30). Statistical power comes
from the **number of conditions** (91 pairs), and each map is itself a stable meta-analytic
average.

**Source 2 — Narratives fMRI (stimulus-locked, N=91) — INDEPENDENT VALIDATION.** Nastase et
al. 2021; subjects listen to stories; condition-labelled story sentences → averaged BOLD (5 s
HRF lag, Schaefer-400) → brain RDM. The **same story text** is fed to the LLM → directly
comparable RDM on identical stimuli. This pipeline does not use the 14 Neurosynth maps and was
unaffected by the ToM correction.

### Noise ceiling
Split-half reliability (LLM side for Source 1; brain side for Source 2) — the upper bound on ρ.

### The 14 cognitive conditions
- **Affective (6):** anger, fear, disgust, sadness, happiness, valence
- **Mentalistic / social (7):** belief, mentalizing, intention, theory_of_mind, empathy,
  self_referential, judgment
- **Moral (1):** moral

---

## 3. Models Tested
- **Cross-architecture (7–9B):** Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-Instruct,
  Mistral-7B-Instruct-v0.3, gemma-2-9b-it.
- **Scaling (Qwen2.5-Instruct):** 0.5B, 1.5B, 3B, 7B.
- **Base vs Instruct:** Qwen2.5-1.5B (base) vs Qwen2.5-1.5B-Instruct.

---

## 4. Experiments and Findings (corrected)

### Experiment 1 — Cross-architecture brain-LLM RSA
| Model | Peak ρ | Permutation p (10k label shuffles) |
|---|---|---|
| Qwen2.5-7B | **0.739** | 0.0002 |
| Llama-3.1-8B | **0.727** | 0.0001 |
| Mistral-7B | **0.730** | 0.0001 |
| Gemma-2-9B | **0.735** | 0.0001 |

Null 95th percentile ≈ 0.25; observed ρ ≈ 0.73; **max-stat p = 0.0002** (corrected for
peak-layer selection); bootstrap 95% CI [0.719, 0.759] (Qwen 7B); overall noise ceiling ≈ 0.97,
so ρ ≈ 76% of ceiling. A held-out discovery/confirmation split (freeze the layer + config
chosen on a different model) keeps all 4 positive (0.61–0.72). **Architecture-invariant** (all
4 within 0.012 of each other). Per-block alignment near ceiling: **affective 0.74 / ceiling
0.94 (78%); mentalistic 0.70 / ceiling 0.81 (87%)** — both blocks align, no asymmetry. Per-condition,
nearly all 14 conditions align 0.67–0.85; only empathy lags (0.24), and empathy is the
smallest set (n=32) with an unstable split-half ceiling — a measurement artifact, not a
divergence.

### Experiment 2 — Scaling
| Size | Peak ρ | Noise ceiling | ρ/ceiling |
|---|---|---|---|
| 0.5B | 0.752 | 0.966 | 78% |
| 1.5B | 0.754 | 0.966 | 78% |
| 3B | 0.747 | 0.962 | 78% |
| 7B | 0.739 | 0.969 | 76% |

**Scale-invariant** — a 15× parameter increase yields Δρ = −0.013 (all p=0.0002). Brain-likeness
does not grow with size, contradicting the "bigger = more brain-like" narrative (Schrimpf 2021,
Antonello 2023). Independently confirmed on real fMRI (Experiment 3).

### Experiment 3 — Stimulus-locked validation on real fMRI (independent)
| Model | Peak ρ | Brain ceiling | ρ/ceiling | p |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.540 | 0.762 | 70.9% | 0.003 |
| Qwen2.5-1.5B | 0.582 | 0.762 | 76.3% | 0.001 |
| Qwen2.5-3B | 0.560 | 0.762 | 73.4% | 0.002 |

Using **identical text** for brain and LLM (Narratives, N=91), ρ ≈ 0.56, reaching 71–76% of
the brain noise ceiling. Confirms the alignment is genuine and stimulus-driven, not an artifact
of meta-analytic maps. Scale-invariance reproduced.

### Experiment 4 — Confound controls (where does the alignment come from?)
Baselines against the corrected brain RDM (Qwen 7B reference, raw ρ=0.739):
| Baseline | ρ | sig |
|---|---|---|
| GloVe word embeddings (mean-pooled) | 0.494 | *** |
| Condition-name (GloVe of the label word) | 0.519 | *** |
| Sentence length | 0.328 | ** |
| TF-IDF | 0.195 | ns |

**Partial RSA — trained LLM controlling for length + GloVe + condition-name:** ρ drops
0.739 → **0.588 (80% retained), p=0.0002**. Controlling TF-IDF + length alone retains ~100%
(0.715–0.745 across 4 models). **Interpretation:** the concept names and word embeddings
themselves carry some emotion/social structure (GloVe 0.49, name 0.52), but the trained LLM
(0.74) sits well above them, and 80% of its alignment survives after partialling them out —
the alignment is more than concept-name semantics. An untrained random-weight model shows ≈0
alignment (ns).

### Experiment 5 — One-axis causal ablation (the mechanistic finding)
Find the direction from the affective centroid to the mentalistic centroid in the LLM's hidden
space; project it out; recompute brain-LLM ρ. (2000 random-direction controls.)
| Model | Original ρ | After ablation | Δρ |
|---|---|---|---|
| Qwen2.5-7B | +0.739 | −0.364 | −1.10 |
| Llama-3.1-8B | +0.727 | −0.390 | −1.12 |
| Mistral-7B | +0.730 | −0.317 | −1.05 |
| Gemma-2-9B | +0.735 | −0.323 | −1.06 |

Random-direction control: Δρ ≈ 0.000 ± 0.0001 (p < 0.0001). This boundary direction is nearly
identical to the LLM's first principal component (cosine = 0.9999); removing PC1 alone
(k=1 PCA ablation) drops ρ to −0.26. **The entire brain-LLM alignment rides on one
representational dimension — the emotion ↔ social-cognition boundary — and it is causally
load-bearing.** After ablation, within-affective ordering is unaffected/slightly improved
(Δρ +0.10 to +0.23) while within-mentalistic ordering degrades (Δρ −0.59 to −0.81): the
affective block carries independent fine structure, whereas the mentalistic block's
brain-alignment is more entangled with the global boundary axis. (This is a structural
property, not a deficit — the mentalistic block aligns at 87% of ceiling, Experiment 1.)

### Experiment 6 — Brain geometry predicts LLM behavior
Cross-validated 14-way nearest-centroid classification of stimuli (4 models, 50 splits):
the brain RDM predicts **which conditions the LLM confuses** — brain-distance vs
LLM-confusion-distance ρ = **0.243, p = 0.02** (per model 0.20–0.26). Geometry → behavior:
conditions the brain places close are the ones the LLM mixes up. (Companion result —
Direction A, `src/brain_causal_coupling.py` — the brain RDM also predicts the LLM's internal
*causal* coupling between functions: significant in 3/4 models, leave-one-condition-out and
leave-two-out stable, 91/91 leave-two-out subsets significant.)

### Experiment 7 — Brain-derived cognitive steering
Steering Qwen2.5-3B along the brain-derived boundary direction (α from −20 to +20) on moral
dilemmas (rated by an LLM judge): at strong negative α the model collapses into emotional
output ("Horror! Horror!"); at moderate positive α it produces structured analytical reasoning
("utilitarian vs deontological"). The brain-derived axis is **causally functional** — it
controls the model's emotional ↔ analytical response style.

### Experiment 8 — Base vs Instruct (independent)
Brain-LLM RSA on pieman story (N=91): base ρ = 0.525, instruct ρ = 0.577 — **base models
already have ~91% of the alignment.** Emotion-space PCA is nearly identical (PC2-valence
r ≈ 0.65 in both). The alignment is primarily a product of **language pretraining**, not RLHF.

### Experiment 9 — Emotion geometry (independent)
GoEmotions 28-category stimuli × Qwen2.5 (1.5B, 3B): the LLM emotion space is
**valence-dominant** — PC1/PC2 capture valence, arousal appears only at PC3. Consistent across
sizes.

### Experiment 10 — Individual differences (independent)
Per-subject brain-LLM alignment across 96 Narratives subjects: mean ρ = 0.568, range
[0.278, 0.776]; 97% of subjects ρ > 0.3 (all positive). The alignment is **universal across
individuals**, not driven by a subset.

---

## 5. Unified Conclusions

1. **Broad brain-LLM alignment exists and is robust:** ρ ≈ 0.73 (Neurosynth) and ≈ 0.56
   (stimulus-locked real fMRI, N=91), near noise ceiling, both emotion and social-cognition
   blocks.
2. **Universal:** invariant across 4 architectures, scales (0.5B–7B), 96 individual brains, and
   base vs instruct models.
3. **Originates in language pretraining:** base models carry ~91% of the alignment.
4. **Survives confound control:** ~80% retained after partialling word-embedding + concept-name
   + length (ρ 0.74 → 0.59, p=0.0002); untrained model ≈ 0.
5. **Carried by one brain-like axis:** the emotion ↔ social-cognition boundary ≈ PC1
   (cosine 0.9999); removing it inverts ρ to −0.36 (4/4 models, p<0.0001).
6. **Causally functional and behaviorally predictive:** the brain-derived axis steers
   emotional ↔ analytical output; the brain RDM predicts LLM confusion (ρ=0.24) and internal
   causal coupling (Direction A, 3/4 models).

---

## 6. Related Work
- **Schrimpf et al. 2021 (PNAS)** Brain-Score / next-word prediction. We extend: condition-level
  brain-likeness *saturates by 0.5B* and reduces to one dominant brain-like axis.
- **Goldstein et al. 2022 (Nat Neurosci)**, **Caucheteux & King 2023 (Nat Hum Behav)**,
  **Tang et al. 2023 (Nat Neurosci)**, **Mischler et al. 2024 (Nat Mach Intell)**,
  **Antonello & Huth 2023 (NeurIPS)** — encoding / temporal-prediction / decoding work; we make
  a representational-geometry claim and find early saturation rather than log-linear scaling.
- **Hadidi et al. 2025 (Nat Commun)** confound warning. Our RSA operates on condition-level
  distance matrices and we add untrained-model, GloVe/TF-IDF/name/length baselines and partial
  RSA (80% retained), making it robust to first-order confounds.
- **Zou et al. 2023 / Turner et al. 2023** representation engineering — we derive the steering
  direction from *brain data* rather than behavioral supervision, and show it is universal
  across architectures.
- **Mahowald et al. 2024 (TiCS)** language vs thought — our data show LLMs learn brain-like
  *relational* structure across both emotion and social cognition from text alone.

---

## 7. Open Questions / Next
1. **Turn the reference frame into predictions:** known neuroscience built on the emotion/social
   separation (lesion double-dissociations, dual-process moral cognition, cortical processing
   gradient, developmental order) becomes a battery of testable LLM predictions. Direction A
   (causal coupling) is the first; layer-depth (gradient) and training-checkpoint
   (developmental order) are next.
2. **Empathy condition is underpowered** (n=32); rebuild with a proper empathy-induction set.

---

## 8. Code Structure
```
experiments/
  src/
    build_brain_rdm.py            — Build 14-map Neurosynth brain RDM (1−Pearson)
    reconstruct_headline_rdms.py  — Rebuild {model}_rdm14_headline.npz (headline recipe)
    compute_rsa_v2.py             — Full RSA sweep (poolings/centerings/distances/layers)
    rsa_cross_model_v2.py         — 4-model cross-architecture comparison
    rsa_scaling_analysis.py       — Scaling curve + noise ceiling (Qwen 0.5B–7B)
    rsa_deep_analysis.py          — Gap + confusion (geometry→behavior) + causal ablation
    confirmatory_rsa.py           — Discovery/confirmation split, max-stat perm, bootstrap, CV ablation
    baseline_controls.py          — GloVe / TF-IDF / condition-name / length baselines
    fix_all_holes.py              — Partial RSA + LOO / leave-2-out stability
    tom_source_check.py           — Diagnostic that found the HCP-ToM artifact
    affective_ceiling_control.py  — Per-block alignment vs LLM noise ceiling
    brain_causal_coupling.py      — Direction A: brain RDM predicts LLM causal coupling
    narratives_*.py               — Narratives fMRI pipeline (stimulus-locked validation)
    cognitive_steering.py         — Brain-derived activation steering
    emotion_geometry.py           — Emotion-space PCA
  results/cognitive_rsa/          — brain_rdm.npz, {model}_rdm14_headline.npz, deep_analysis.json,
                                    scaling_summary.json, confirmatory_rsa.json, baseline_controls.json
  results/affective_validation/   — ceiling control, Kragel/IBC/HCP audits, tom_source_check
  present/index.html              — plain-language briefing deck (corrected story)
```

## 9. Reproduction
```bash
python src/build_brain_rdm.py            # corrected 14-map Neurosynth brain RDM
python src/reconstruct_headline_rdms.py  # LLM headline RDMs
python src/rsa_cross_model_v2.py         # cross-architecture ρ
python src/rsa_deep_analysis.py          # gap + confusion + causal ablation
python src/baseline_controls.py          # confound baselines + partial RSA
python src/fix_all_holes.py              # partial RSA + LOO stability
# Narratives (independent): narratives_preprocess.py → narratives_annotate.py →
#   narratives_brain_rdm.py → extract_narratives_llm.py (GPU) → compare
```
Heavy CPU recomputes that load the large per-stim NPZ should be submitted to a 512 GB compute
node (`scripts/slurm/rsa_recompute_local.sh`), not run on the login node.
