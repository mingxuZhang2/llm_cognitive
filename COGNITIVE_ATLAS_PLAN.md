# Cognitive Atlas: Project Plan (v2)

**Decision date:** 2026-05-25
**Target venue:** Nature Machine Intelligence
**Branch:** cognitive-atlas (v1 archived at tag v1-ai-categories)

## Core scientific claim (the make-or-break finding)

**Moral cognition in LLMs is compositionally organized.** It is not a single module — it emerges from interaction of three more basic cognitive components:
- Affective evaluation (parallels human vmPFC/amygdala)
- Mental-state reasoning (parallels human TPJ/mPFC)
- Norm integration (parallels human lateral PFC)

Ablating each component selectively impairs different aspects of moral judgment. This mirrors human dual-process moral cognition (Greene 2001 Science) and provides the first causal compositional evidence in artificial systems.

## Methodology (unchanged from v1)

- Gradient × Activation causal attribution
- Top-k neuron selection by one-vs-rest selectivity
- Zero ablation (forward hooks on gate_proj)
- Double dissociation matrix
- Cross-architecture validation (Qwen, LLaMA, Mistral, Gemma)

## Cognitive domains (new)

Replacing v1's 8 AI-task categories with 4 human cognitive domains:

1. **Emotion processing** — VAD (valence/arousal/dominance), discrete emotions
2. **Moral cognition** — moral foundations, wrongness judgment
3. **Theory of Mind / Social cognition** — false belief, intent attribution
4. **Self-cognition** — self-other distinction, metacognition (optional Phase 2)

## Stimulus design principles

Every category MUST have:
- A target condition (engaging the function)
- A matched control condition (controlling for confounds)
- Sufficient samples (50-100 per condition for statistical power)

**Critical confound controls for moral decomposition:**

| Condition pair | What it dissociates |
|---|---|
| Intentional harm vs Accidental harm | Intent (controls outcome) |
| Attempted harm vs Completed harm | Outcome (controls intent) |
| Moral violation vs Conventional violation | Norm type (controls negativity) |
| Negative non-moral vs Moral negative | Morality (controls valence) |

If "moral neurons" are just emotion neurons, they will respond equally to all negative content. If they are genuine moral neurons, they will preferentially encode intent/agency/norm violation.

## Five target findings (in priority order)

1. **Functional separation** — emotion/moral/ToM neurons are causally separable
2. **Moral compositional structure** ← THE CORE FINDING
3. **Layer hierarchy** — shallow=affect → mid=mental states → deep=moral integration
4. **Behavioral alignment** — neuron activations predict human ratings
5. **Brain data alignment** — RSA with Neurosynth/HCP/MOFOMIC fMRI

## Data preparation tracks

- `data/cognitive_stimuli/emotion/` — ANEW/Warriner VAD + GoEmotions
- `data/cognitive_stimuli/moral/` — Moral Foundations Vignettes + ETHICS + Social Chemistry 101
- `data/cognitive_stimuli/moral_decomposition/` — Custom 4×2 factorial control stimuli
- `data/cognitive_stimuli/tom/` — false belief + faux pas + indirect requests
- `data/brain_maps/` — Neurosynth meta-analytic maps + HCP contrasts + MOFOMIC (if accessible)

## Code reuse from v1

Existing methods that carry over directly:
- `src/multi_function_dissociation.py` — main pipeline (will need stimulus loader update)
- `src/activation_extraction.py` — activation extraction (unchanged)
- `src/statistical_validation.py` — permutation + t-test + BH-FDR (unchanged)
- `src/convergence_analysis.py` — cross-architecture (unchanged)

New code needed:
- Stimulus generation/validation for moral decomposition
- RSA analysis against brain maps
- Behavioral alignment regression

## Risk acknowledgment

The compositional finding may not hold — moral neurons might just be emotion neurons. If pilot shows this, we adjust scope rather than fabricate results. Dr. Zhang explicitly accepts this risk.
