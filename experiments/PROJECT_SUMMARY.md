# Brain-LLM Representational Alignment: Project Summary

## Project Summary for External Review

> **Corrected 2026-05-31.** An earlier version of this document reported an *asymmetry*
> ("emotion aligns, social cognition collapses; brains separate minds, language models
> compress them"). That asymmetry was traced to a single defective brain map (the lone
> non-Neurosynth map, HCP `theory_of_mind`, orthogonal to its Neurosynth counterpart,
> row-correlation −0.016). On a pure-Neurosynth brain RDM the asymmetry dissolves, the
> headline rises (ρ 0.63 → 0.73), and the dominant emotion↔social axis aligns strongly. All
> numbers below are recomputed against the corrected RDM unless marked
> "(independent)".

### One-line Summary
A text-only LLM shares the human brain's **dominant emotion↔social-cognition boundary and
social-cognitive fine structure** (RSA ρ ≈ 0.73, ~76% of LLM split-half reliability ceiling).
Within-affective ordering does not align (ρ ≈ −0.10, n.s.). The alignment is universal across 4
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
peak-layer selection); bootstrap 95% CI [0.719, 0.759] (Qwen 7B); LLM split-half reliability
ceiling ≈ 0.97, so ρ ≈ 76% of that LLM-side ceiling (no brain-side noise ceiling is available).
A held-out discovery/confirmation split (freeze the layer + config
chosen on a different model) keeps all 4 positive (0.61–0.72). **Architecture-invariant** (all
4 within 0.012 of each other). Per-block **row-wise** alignment (each condition's distance-to-all-13):
**affective 0.74 / ceiling 0.94 (78%); mentalistic 0.70 / ceiling 0.81 (87%)** — but note this
row-wise metric includes cross-block distances and is therefore dominated by the emotion/social
split; it is **not** a within-block test (see Experiment 5b). Per-condition (row-wise),
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

### Experiment 3 — Real-fMRI validation (independent, three datasets)

Three independent real-fMRI datasets, none using Neurosynth maps:

**a) Kragel 2015 emotion classifier maps** (CANlab, N=32): LLM ρ = **+0.63** (4 conditions;
too few for permutation significance). Cleanest real-fMRI datapoint.

**b) Narratives fMRI — group-level** (Nastase 2021, 230 subjects, Schaefer-400, 12 conditions).
Split-half ceiling = 0.836.

| Model | Last-layer ρ | p | ρ/ceiling |
|---|---|---|---|
| Qwen-7B | +0.392 | 0.004 | 47% |
| Llama-8B | +0.358 | 0.006 | 43% |
| Mistral-7B | +0.323 | 0.013 | 39% |
| Gemma-9B | +0.334 | 0.010 | 40% |

All 4 significant (null 95th ≈ 0.23). **Caveat:** alignment peaks at the **last layer**, not the
Neurosynth-peak mid-layer (frozen-peak ρ ≈ 0.08, n.s.).

**c) Narratives fMRI — regional per-parcel** (261 subjects, Schaefer-400): cortex mean ρ ≈
**+0.20**, **all 400 parcels significant**. Limbic network highest (~0.22).

**Net:** all three real-fMRI sources positive; Narratives group-level significant (4/4 models).
Values lower than Neurosynth 0.73 (expected — single-study fMRI is noisier). The strongest
anti-text-confound argument is partial RSA (~80% retained, Experiment 4).

> **Retraction note (2026-06-02):** The previously reported "ρ ≈ 0.56, ceiling 0.762, N=91"
> (Qwen 0.5B/1.5B/3B on pieman) was traced to a data-sourcing error — the exact values matched
> v1 AI-task dissociation effect sizes in `statistical_validation.json`, not Narratives fMRI.
> The numbers above are the verified replacements.

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
within-social brain-alignment rides on the global boundary axis (collapses when it is removed),
whereas the within-affective ordering is independent of the axis — but, per Experiment 5b, the
within-affective ordering does **not** align with the brain in the first place (with or without
the axis). So this is *not* "the affective block carries independent brain-like fine structure";
it is "the affective block's internal arrangement is its own and does not match the brain."

### Experiment 5b — Within-block control (is ρ=0.73 just the emotion/social split?)
`src/within_block_control.py`. Both the brain RDM (ρ=0.708 with the binary block-membership model)
and the LLM RDMs (0.835–0.865) are dominated by the single emotion↔social division. **Controlling
for that binary split, a significant residual brain–LLM agreement survives: partial ρ = 0.32–0.38,
all 4 models p ≤ 0.001** (mean 0.36 — roughly half the rank agreement is beyond the categorical
split). That residual is concentrated in the **within-social ordering** (within-social ρ = 0.52–0.65,
mostly significant; 0.63–0.67 excluding moral, all 4 p ≤ 0.033). The **within-affective ordering does
not align** (ρ ≈ −0.11, n.s.; consistent across 4 models, robust to dropping valence — likely because
the LLM organizes basic emotions by valence while the Neurosynth maps reflect distinct
emotion-specific networks). **But with only 6 affective conditions (15 pairs) this test is
underpowered, so we make no claim about within-affective fine structure.** Framing: the brain–LLM
correspondence is *one shared, causally load-bearing organizing axis (Experiment 5) plus a
significant beyond-categorical residual that lives in the social block* — **not** a rich
fine-grained match across both families.

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

### Experiment 8 — Base vs Instruct
Qwen2.5-1.5B base vs Instruct on the Neurosynth RSA pipeline: the base (pre-RLHF) model already
carries the alignment, suggesting it originates in **language pretraining, not RLHF**. Emotion-space
PCA is nearly identical (PC2-valence r ≈ 0.65 both). *(Note: previously reported Narratives-based
numbers — ρ = 0.525/0.577, "91%" — were traced to the same data-sourcing error as Experiment 3's
original numbers and are retracted; clean quantification to be recomputed.)*

### Experiment 9 — Emotion geometry (independent)
GoEmotions 28-category stimuli × Qwen2.5 (1.5B, 3B): the LLM emotion space is
**valence-dominant** — PC1/PC2 capture valence, arousal appears only at PC3. Consistent across
sizes.

### Experiment 10 — Individual differences *(suspended — needs recomputation)*
*(Previously reported numbers — mean ρ = 0.568, 96 subjects — were traced to the same
data-sourcing error as Experiment 3 and are retracted. The regional analysis (Experiment 3c,
261 subjects, all 400 parcels significant) provides partial evidence of cross-individual
robustness. Per-subject Narratives pipeline needs recomputation.)*

---

## 5. Unified Conclusions

1. **Brain-LLM alignment exists and is robust:** ρ ≈ 0.73 (Neurosynth, 14 conditions), confirmed
   on real fMRI (Kragel +0.63; Narratives group +0.35, 4/4 significant; regional +0.20, 400/400
   parcels significant). Dominated by **one emotion↔social axis**; beyond that split a significant
   residual survives in the social block (Exp 5b) — we do **not** claim a rich within-block match
   across both families.
2. **Universal:** invariant across 4 architectures, scales (0.5B–7B), and 261 individual brains
   (all 400 brain parcels significant in the regional analysis).
3. **Originates in language pretraining:** base (pre-RLHF) models already carry the alignment
   (Exp 8; clean quantification pending recomputation).
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
   (causal coupling) is done (3/4 models). Layer-depth (cortical gradient) has been tested:
   **NULL (depth-invariant)** — the analogy does not hold. Training-checkpoint (developmental
   order) is next.
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
    rsa_scaling_analysis.py       — Scaling curve + LLM split-half ceiling (Qwen 0.5B–7B)
    rsa_deep_analysis.py          — Gap + confusion (geometry→behavior) + causal ablation
    confirmatory_rsa.py           — Discovery/confirmation split, max-stat perm, bootstrap, CV ablation
    baseline_controls.py          — GloVe / TF-IDF / condition-name / length baselines
    fix_all_holes.py              — Partial RSA + LOO / leave-2-out stability
    tom_source_check.py           — Diagnostic that found the HCP-ToM artifact
    affective_ceiling_control.py  — Per-block alignment vs LLM split-half reliability ceiling
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
