# Project Status: LLM Cognitive Atlas — Round 4 for GPT Pro Review

## One-Line Summary

LLMs learn a brain-aligned coarse cognitive map but compress human mental-state fine structure — not because they lack the representational capacity (linear probes achieve 100%), but because they fail to extract mental-state distinctions from naturalistic text where the brain succeeds.

---

## 1. The Complete Story (All Experiments)

### Chapter 1: Brain-LLM alignment exists and is robust

We compared representational distance matrices (RDMs) of 14 cognitive conditions between human brain data and LLMs.

| Analysis | ρ | p | 
|---|---|---|
| Raw RSA (4 architectures, Neurosynth) | 0.60–0.64 | <0.0002 |
| Partial RSA (controlling GloVe + length + condition-name) | **0.496** | **0.0002** |
| Stimulus-locked fMRI (Narratives, N=91, same text) | 0.54–0.58 | <0.003 |
| Untrained model (random weights) | 0.09 | 0.39 (ns) |
| Max-stat permutation (correcting layer selection) | — | 0.0004 |
| Bootstrap 95% CI | [0.606, 0.661] | — |

### Chapter 2: The alignment is one-dimensional

PC1 of the LLM condition space ≈ affective–mentalistic boundary (cosine = 0.9999). Removing this one direction from 3584-dim space flips ρ from +0.64 to −0.36 (all 4 architectures, p<0.0005). Cross-validated on 50 split-halves: mean Δρ = −0.80, 95% CI [−0.87, −0.63].

Controls: valence direction ablation Δρ = +0.008 (no effect). PCA-subspace random directions Δρ = −0.0004 (no effect).

### Chapter 3: Affective fine structure preserved, mentalistic collapsed

| Pair category | Mean |residual| | Brain-LLM match |
|---|---|---|
| Within-affective (anger-fear, etc.) | 6.9 | Good |
| Within-mentalistic (belief-ToM, etc.) | 19.2 | Poor |
| Cross-domain | 15.0 | Moderate |

After boundary ablation: within-affective **improves** (Δρ = +0.15), within-mentalistic **collapses** (Δρ = −0.44).

Behavioral validation: LLM classification accuracy on within-mentalistic pairs (56%) significantly lower than within-affective (71%) and cross-domain (68%). Kruskal-Wallis H=12.91, **p=0.0016**.

### Chapter 4: Nothing fixes the mentalistic gap

| Intervention | Effect on mentalistic fine structure |
|---|---|
| 15× more parameters (0.5B→7B) | No change |
| Different architecture (4 models) | No change |
| RLHF/instruction tuning | Negligible (base has 91%) |
| Multimodal training (VLM) | No change (RDM corr = 0.998) |
| Activation steering | Cannot repair (all alphas degrade) |

### Chapter 5: The collapse is NOT due to simple explanations

| Hypothesis | Test | Result |
|---|---|---|
| Lexical similarity | GloVe distances | **REJECTED**: mentalistic words are MORE distinct (0.91 vs 0.74) |
| Next-token prediction similarity | Jensen-Shannon divergence | **REJECTED**: mentalistic continuations are MORE different (JSD 0.58 vs 0.29) |
| Training objective determines collapse | JSD vs brain-LLM residual | **REJECTED**: ρ = −0.009, p = 0.93 |

### Chapter 6: The KEY finding — LLMs HAVE the capacity but FAIL to extract it

**Agent-state factorization probe**: We created controlled synthetic stimuli varying ONLY the mental state type ("Alice believes/wants/intends/knows/doesn't know that X"). Linear probes on LLM hidden states:

| Probe | Qwen 3B | Qwen 1.5B |
|---|---|---|
| All 10 pairwise (belief vs desire, belief vs intention, etc.) | **100%** | **100%** |

**Both models perfectly distinguish belief from desire from intention from knowledge from ignorance when the input is controlled.**

But in our RSA analysis with natural stimuli (from diverse sources), these same conditions collapse. This means:

> **The mentalistic collapse is not a representational capacity limitation. The model HAS separable representations for different mental states. The collapse happens because natural text contains other dominant features (topic, format, style, source) that overwhelm the mental-state signal. The brain manages to extract mental-state distinctions from natural text; the LLM does not.**

### Chapter 7: Prospective prediction (preliminary)

Pre-registered high-risk pairs (brain-far, LLM-close) vs low-risk pairs (brain & LLM agree) on held-out classification tasks:

| | Qwen 3B |
|---|---|
| HIGH-RISK (belief-ToM, intention-ToM, empathy-intention) | 53% accuracy |
| LOW-RISK (anger-happiness, fear-moral) | 90% accuracy |
| p (one-sided) | 0.069 |

Direction correct but underpowered (only 5 items per pair, 3 vs 2 pairs with templates). Needs scaling up.

---

## 2. The Emerging Scientific Claim

### Old framing (too weak):
"LLMs and brains have similar cognitive organization"

### Current framing:

> **LLMs acquire the vocabulary and surface patterns of social cognition, and can represent distinct mental states when explicitly prompted. But they fail to spontaneously extract human-like mental-state geometry from naturalistic text — a capacity that the brain possesses. This extraction failure is not addressable by scale, architecture, alignment training, multimodal experience, or simple representational intervention. It represents a qualitative gap between statistical language learning and human social cognition.**

### Why this matters:

1. **For AI safety**: A model can appear socially competent (passing ToM benchmarks, generating empathetic responses) while internally compressing distinct mental states. It knows "believes" ≠ "intends" as vocabulary, but doesn't build a human-like state space for tracking who believes what vs who intends what in context.

2. **For cognitive science**: The brain has a mechanism for extracting mental-state geometry from the same noisy naturalistic input where LLMs fail. This mechanism is not about vocabulary (LLMs have that) or prediction (LLMs do that) — it's something else, possibly related to simulation, embodied interaction, or developmental social experience.

3. **For ML/NLP**: ToM benchmark accuracy may not reflect genuine mental-state understanding. A model can score well on explicit ToM questions while having a collapsed representational geometry that would fail under distribution shift.

---

## 3. What We Still Don't Know

### Open Question 1: Why does the brain succeed at extraction where LLMs fail?

The brain processes the same naturalistic text (Narratives fMRI) and successfully separates mentalistic conditions (brain RDM has large within-mentalistic distances). What mechanism does the brain use that LLMs lack?

Candidates:
- Simulation-based processing (running internal models of other agents)
- Developmental social experience (not just text exposure)
- Recursive self-other modeling
- Embodied/emotional grounding of mental state concepts

### Open Question 2: Can training be designed to restore extraction?

The probe result (100% on controlled stimuli) suggests the representational capacity exists. The question is whether a training objective can teach the model to USE this capacity in naturalistic context.

Candidates to test:
- Explicit agent-state prediction during training
- Counterfactual social reasoning tasks
- Interactive multi-agent environments
- Auxiliary loss for mental-state factorization

### Open Question 3: Does this dissociation (capacity vs extraction) generalize?

We showed it for mentalistic fine structure. Does the same pattern hold for other cognitive domains? Are there other cases where LLMs have latent capacity that's not expressed in naturalistic processing?

---

## 4. Caveats and Limitations

1. **Agent-state probe uses simple templates**. "Alice believes X" vs "Alice intends Y" — the probe may be detecting verb choice, not deep mental state understanding. More naturalistic controlled stimuli needed.

2. **Prospective prediction is underpowered**. Only 5 items per pair, 3 high-risk vs 2 low-risk pairs with templates. Needs 50+ items per pair for mixed-effects regression.

3. **Only Qwen family for scaling + probe**. Other families (Llama, Mistral, Gemma) tested for RSA but not for probe or prospective prediction.

4. **Neurosynth brain RDM is meta-analytic**, not stimulus-locked for the RSA stimuli. Stimulus-locked validation uses only pieman story (N=91) with DeepSeek annotation (no human validation).

5. **14 cognitive conditions from heterogeneous sources**. GPT Pro flagged this as a confound risk. Partial RSA controls for some surface features but not all (source-dataset RDM, format RDM not yet done).

6. **VLM comparison uses text-only input**. We showed the VL model's text pathway hasn't changed, but we haven't tested it with actual visual social scenes.

---

## 5. Suggested Paper Structure

**Title**: "Brains separate minds; language models compress them"

**Or**: "Large language models compress human mental-state geometry despite intact representational capacity"

**Key figures**:
1. Brain RDM vs LLM RDM + overall RSA with controls
2. PC1/boundary ablation → alignment inversion + selective collapse
3. Affective preserved vs mentalistic collapsed (residual map)
4. Negative controls: scale, architecture, RLHF, VLM, lexical, JSD, steering
5. Agent-state probe: 100% on controlled stimuli → capacity exists but extraction fails
6. (Needed) Prospective failure prediction at scale, or training intervention that restores extraction

**Abstract draft**:

> We report that large language models learn a brain-aligned coarse cognitive map — a dominant axis separating affective from mentalistic processing that matches human brain organization (RSA ρ=0.50 after confound control, validated on stimulus-locked fMRI with N=91). However, LLMs systematically compress the fine structure of human mental-state representations: belief, intention, theory of mind, empathy, and self-reference are collapsed into a single cluster, while affective conditions (anger, fear, happiness) maintain brain-like geometry. This compression persists across model scale (0.5B–7B), architecture (4 families), instruction tuning, and passive multimodal training, and is not explained by lexical similarity or next-token prediction distributions. Critically, linear probes reveal that LLMs possess separable representations for distinct mental states in controlled settings (100% pairwise accuracy), indicating the compression is not a capacity limitation but an extraction failure: LLMs cannot recover mental-state distinctions from naturalistic text where the brain succeeds. These findings suggest that current language models acquire the surface patterns of social cognition without building a human-like geometry of mental states, with implications for the reliability of AI systems in social reasoning contexts.

---

## 6. All Quantitative Results in One Table

| Experiment | Key number | Status |
|---|---|---|
| Cross-architecture RSA | ρ = 0.60–0.64 (4 models) | ✓ |
| Partial RSA (confound-controlled) | ρ = 0.496, 78% retained | ✓ |
| Stimulus-locked fMRI (N=91) | ρ = 0.54–0.58 | ✓ |
| Untrained baseline | ρ = 0.09 (ns) | ✓ |
| Max-stat permutation | p = 0.0004 | ✓ |
| CV ablation | Δρ = −0.80, CI [−0.87, −0.63] | ✓ |
| Scale invariance (0.5B–7B) | Δρ = −0.026 | ✓ |
| Base vs instruct | Base has 91% | ✓ |
| VLM vs text-only | RDM corr = 0.998 | ✓ |
| Behavioral (categorical) | p = 0.0016 | ✓ |
| Behavioral (continuous) | ρ = −0.03 (ns) | ✗ |
| Lexical explains gap? | Mentalistic MORE distinct | ✗ Rejected |
| JSD explains gap? | ρ = −0.009 (ns) | ✗ Rejected |
| Steering fixes ToM? | All degrade | ✗ Rejected |
| **Agent-state probe** | **10/10 = 100%** | **✓ NEW** |
| Prospective prediction | HIGH 53% vs LOW 90% (p=0.069) | Marginal |
| Regional brain dissociation | All parcels significant | ✗ Inconclusive |
| Emotion geometry (PCA) | PC1-2 = valence, PC3 = arousal | ✓ Descriptive |

---

## 7. Repository

Code: https://github.com/mingxuZhang2/llm_cognitive

Key new scripts since last review:
- `src/agent_state_probe.py` — Mental state factorization probe (the key new finding)
- `src/prospective_prediction.py` — Pre-registered failure prediction
- `src/next_token_similarity.py` — JSD mechanistic test (rejected hypothesis)
- `src/predict_behavioral_failure.py` — Behavioral prediction from RDM mismatch
