# Project Status: LLM Cognitive Atlas — For GPT Pro Review

## What This Document Is

This is a complete, honest status report of our project. We list all findings (positive and negative), all controls, and the specific puzzles we're stuck on. We're asking GPT Pro: given these results, what is the best scientific story, and how should we interpret the puzzling negative results?

---

## 1. Core Established Findings

### Finding 1: Brain-LLM cognitive RSA alignment exists and is robust

We compared the representational distance matrices (RDMs) of 14 cognitive conditions between human brain data and LLMs. Brain RDMs come from two sources: Neurosynth meta-analytic maps and stimulus-locked Narratives fMRI (N=91).

| Analysis | ρ | p | Notes |
|---|---|---|---|
| Raw RSA (Neurosynth, 4 architectures) | 0.60–0.64 | <0.0002 | Qwen/Llama/Mistral/Gemma all identical |
| Partial RSA (controlling GloVe + length + condition-name) | 0.496 | 0.0002 | 78% retained after confound control |
| Stimulus-locked fMRI (Narratives, N=91) | 0.54–0.58 | <0.003 | Same text for brain and LLM |
| Untrained model (random weights) | 0.09 | 0.39 | Not an architectural artifact |
| Max-stat permutation (correcting layer selection) | — | 0.0004 | Survives multiple comparison correction |
| Bootstrap 95% CI | [0.606, 0.661] | — | Narrow, stable |

### Finding 2: The alignment is dominated by one principal axis

The first principal component (PC1) of the LLM's condition space is nearly identical to the affective–mentalistic boundary direction (cosine similarity = 0.9999). Removing this one direction from 3584-dimensional space:

| Model | Original ρ | After ablation | Δρ |
|---|---|---|---|
| Qwen-7B | +0.639 | −0.362 | −1.00 |
| Llama-8B | +0.627 | −0.364 | −0.99 |
| Mistral-7B | +0.633 | −0.327 | −0.96 |
| Gemma-9B | +0.639 | −0.317 | −0.96 |

Controls:
- Random direction ablation: Δρ = −0.0001 (no effect, N=200 PCA-subspace directions)
- Valence direction ablation: Δρ = +0.008 (no effect)
- Cross-validated (50 split-halves, axis defined on train, tested on held-out): mean Δρ = −0.80, 95% CI [−0.87, −0.63], all 50 negative

### Finding 3: Asymmetric preservation — affective fine structure preserved, mentalistic collapsed

Within the LLM's cognitive RDM:
- **Within-affective pairs** (anger-fear, anger-sadness, etc.): mean |residual| = 6.9 — LLM matches brain well
- **Within-mentalistic pairs** (belief-ToM, intention-empathy, etc.): mean |residual| = 19.2 — LLM collapses them

After boundary ablation:
- Within-affective alignment **improves** (Δρ = +0.15) — independently encoded
- Within-mentalistic alignment **collapses** (Δρ = −0.44) — stacked on same axis

This predicts behavioral failures: LLM classification accuracy on within-mentalistic pairs (56%) is significantly lower than within-affective (71%) and cross-domain (68%) pairs (Kruskal-Wallis H=12.91, p=0.0016).

### Finding 4: Scale-invariant and architecture-invariant

| Size | ρ (Neurosynth) | ρ (stimulus-locked fMRI) |
|---|---|---|
| 0.5B | 0.665 | 0.540 |
| 1.5B | 0.655 | 0.582 |
| 3B | 0.657 | 0.560 |
| 7B | 0.639 | — |

Δρ across 15× parameter range = −0.026. Coarse cognitive geometry saturates at small model scale.

### Finding 5: Pretraining origin, not RLHF

Qwen2.5-1.5B base (no instruction tuning): ρ = 0.525
Qwen2.5-1.5B instruct: ρ = 0.577
Base model has 91% of the alignment. The cognitive axis exists before RLHF.

### Finding 6: Multimodal training does NOT fix the mentalistic gap

Qwen2.5-VL-3B (vision-language model, trained on images+video+text) vs Qwen2.5-3B (text-only), same text input:

- Text-only RDM vs VL RDM correlation: ρ = 0.998 (nearly identical)
- Within-mentalistic residual: text-only = 20.2, VL = 20.3 (no change)
- Within-affective residual: text-only = 7.7, VL = 7.4 (no change)

Seeing images and videos does not help the model distinguish belief from intention from ToM.

---

## 2. Hypotheses We Tested and Their Outcomes

### Hypothesis A: "Mentalistic collapse is due to lexical similarity"
**Result: REJECTED**

GloVe distances: within-affective = 0.74, within-mentalistic = 0.91. Mentalistic conditions use MORE distinct vocabulary, not less. Yet LLMs collapse them anyway.

### Hypothesis B: "Mentalistic collapse is due to similar next-token predictions"  
**Result: REJECTED**

Jensen-Shannon divergence of next-token distributions:
- Within-affective JSD = 0.294 (most similar predictions)
- Within-mentalistic JSD = 0.575 (more different predictions)

JSD vs brain-LLM residual: ρ = −0.009, p = 0.93 (no correlation).

LLM collapses mentalistic pairs DESPITE them having MORE different next-token predictions than affective pairs. The training objective does not explain the collapse.

However: JSD vs LLM distance: ρ = +0.50, p < 0.0001. The prediction objective DOES shape the LLM's own geometry — just not in a way that explains the brain-LLM mismatch.

### Hypothesis C: "Brain-derived steering can repair mentalistic deficits"
**Result: REJECTED**

Steering along brain-derived belief→ToM, intention→ToM directions at various alphas: all degrade or maintain performance on false belief (baseline 80%) and faux pas (baseline 67%) tasks. The deficit is not a missing direction.

### Hypothesis D: "Scale/architecture/RLHF/multimodal training can close the gap"
**Result: ALL REJECTED**

| Intervention | Effect on mentalistic gap |
|---|---|
| 15× more parameters (0.5B→7B) | No change |
| Different architecture (4 models) | No change |
| RLHF/instruction tuning | Negligible (+9%) |
| Multimodal training (VLM) | No change (Δresidual = +0.1) |

---

## 3. What We're Stuck On

### Puzzle 1: What CAUSES the mentalistic collapse?

We've ruled out:
- Lexical similarity (mentalistic words are MORE distinct)
- Next-token prediction similarity (mentalistic continuations are MORE different)
- Insufficient model capacity (scale-invariant)
- Insufficient training data diversity (4 companies' data)
- Insufficient modality (VLM doesn't help)
- Insufficient alignment training (RLHF doesn't help)

So WHY does the LLM collapse belief/intention/ToM/empathy/self-referential/judgment into one cluster, when:
- The input words are different
- The predicted continuations are different
- The brain treats them as different
- The model is big enough

**We cannot explain the mechanism.**

### Puzzle 2: What would the "killer finding" be?

We have a solid characterization: brain-LLM alignment is one-dimensional, scale-invariant, controlled for confounds, and has an asymmetric gap (affective preserved, mentalistic collapsed). But:

- It's not a new brain discovery (we use brain as reference, don't discover brain mechanisms)
- It's not a new capability (we can't fix the gap)
- It's not a prediction that changes understanding (behavioral prediction works categorically but not quantitatively)

GPT Pro previously suggested two paths to Nature-level:
1. Show the gap predicts specific LLM failures on unseen tasks → partially confirmed (p=0.0016 categorical), but not quantitatively
2. Show specific training modifies the gap → tested VLM, negative result

**What scientific claim can we make that is both supported by the data and genuinely novel?**

### Puzzle 3: Why is JSD inversely related to what we expected?

Within-affective pairs have the LOWEST JSD (most similar next-token predictions) but the BEST brain-LLM alignment. Within-mentalistic pairs have HIGHER JSD but WORSE alignment. This is backwards from the simple prediction objective hypothesis.

One possible interpretation: the LLM preserves affective distinctions DESPITE similar predictions because emotion distinctions are socially/pragmatically important and reinforced through many different textual contexts. Mentalistic distinctions are collapsed DESPITE different predictions because the model lacks the computational mechanism (simulation? recursive modeling?) needed to build separate representations for different types of mental state reasoning.

But this is speculation. We don't have evidence for it.

### Puzzle 4: Is the story "solid characterization" enough for a strong publication?

The current findings, properly controlled, may be publishable as:
> "Language pretraining induces a brain-aligned cognitive geometry in LLMs, dominated by a single affective–mentalistic axis. Affective fine structure is preserved but mentalistic fine structure is systematically collapsed — a gap robust to scale, architecture, alignment training, and multimodal experience."

Is this Nature-family worthy, or is this a Cognition / PNAS / eLife level paper? What would elevate it?

---

## 4. Summary of All Quantitative Results

| Experiment | Key number | Status |
|---|---|---|
| Cross-architecture RSA | ρ = 0.60–0.64 (4 models) | ✓ Confirmed |
| Partial RSA (confound-controlled) | ρ = 0.496, 78% retained | ✓ Confirmed |
| Stimulus-locked fMRI (N=91) | ρ = 0.54–0.58 | ✓ Confirmed |
| Untrained baseline | ρ = 0.09 (ns) | ✓ Clean |
| Max-stat permutation | p = 0.0004 | ✓ Significant |
| CV ablation | Δρ = −0.80, CI [−0.87, −0.63] | ✓ Robust |
| Scale invariance (0.5B–7B) | Δρ = −0.026 | ✓ Flat |
| Base vs instruct | Base has 91% | ✓ Pretraining origin |
| VLM vs text-only | RDM correlation = 0.998 | ✓ No multimodal effect |
| Behavioral prediction (categorical) | p = 0.0016 | ✓ Mentalistic harder |
| Behavioral prediction (continuous) | ρ = −0.03, ns | ✗ Not quantitative |
| Lexical similarity explains gap? | Mentalistic MORE distinct | ✗ Rejected |
| Next-token JSD explains gap? | ρ = −0.009, ns | ✗ Rejected |
| Steering fixes ToM? | All alphas degrade | ✗ Rejected |
| Regional brain dissociation? | All parcels significant | ✗ Inconclusive |

---

## 5. Repository Structure

All code is at https://github.com/mingxuZhang2/llm_cognitive

Key scripts:
- `src/baseline_controls.py` — Untrained, GloVe, TF-IDF, length baselines + partial RSA
- `src/confirmatory_rsa.py` — Discovery/confirmation split, max-stat permutation, CV ablation, matched-variance controls
- `src/rsa_deep_analysis.py` — Gap decomposition + confusion + one-axis causal ablation
- `src/rsa_scaling_analysis.py` — Scale invariance (0.5B–7B)
- `src/next_token_similarity.py` — JSD mechanistic test
- `src/predict_behavioral_failure.py` — Behavioral prediction from RDM mismatch
- `src/emotion_geometry.py` — Emotion space PCA (valence/arousal)
- `src/narratives_brain_rdm.py` — Stimulus-locked fMRI pipeline
- `src/cognitive_steering.py` — Brain-derived activation steering
- `src/brain_transfer.py` — Brain-to-LLM transfer (negative result)
