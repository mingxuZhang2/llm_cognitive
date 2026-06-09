# The Brain as a Reference Frame for LLMs

Using the human brain as a **predictive reference frame** to explain and predict
the internal organization of Large Language Models. Direction matters: we treat
established neuroscience conclusions as *testable predictions about LLMs* — not
the other way around. Targeting Nature Machine Intelligence.

---

## Three Main Findings

### Finding 1: Consistency — text alone produces brain-like cognitive topology

A text-only LLM recovers a brain-like affect-mentalizing axis and social-cognitive
fine structure (14 conditions, 91 pairwise distances). The alignment is dominated by
the emotion-social boundary plus within-social ordering; within-affective fine structure
does not align (rho ~ -0.10, n.s., n=6 underpowered).

| Evidence | Result |
|---|---|
| Cross-architecture RSA (4 models) | **rho = 0.73**, all p < 0.0002 |
| Scale invariance (0.5B - 7B) | Flat: 0.739 - 0.754, ceiling ~78% |
| Base vs Instruct (Qwen 1.5B) | **99.3%** of alignment from pretraining; RLHF adds nothing |
| Confound control (partial RSA) | 80% retained after partialling GloVe + name + length |
| Real fMRI (3 independent datasets) | Kragel +0.63; Narratives group +0.35 (4/4 sig, p<0.013); regional +0.20 (400/400 parcels sig, 261 subjects) |
| Within-block control | Dominated by one emotion-social axis; beyond-split residual rho=0.36 (4/4 p<0.001), concentrated in within-social ordering |

**What it means for LLMs:** Language pretraining doesn't just learn word co-occurrence — it
learns the relational structure between cognitive functions, matching the brain's organization.
This provides a new interpretability tool: use the brain's known "map" to navigate LLM
representations.

**What it means for cognitive science:** A system with no body, no evolution, and no
development produces brain-like organization from text alone. This constrains theories about
*why* the brain is organized this way — it's not unique to biological substrates.

### Finding 2: Predictability — neuroscience conclusions transfer to LLMs ("so what")

Known neuroscience results, derived from the brain's emotion/social separation, successfully
predict LLM internal structure.

| Neuroscience conclusion | LLM prediction | Result |
|---|---|---|
| Lesion double-dissociation (Shamay-Tsoory 2009) | Ablating emotion/social neurons should produce selective damage | **Per-condition: 4/4 models significant** (Wilcoxon p=0.0001-0.007); ablate-emotion affects emotion 3.6x more than social; ablate-social affects social 6.2x more |
| Subspace separability | Projecting out the social subspace should selectively destroy social alignment | **Within-social rho collapses from +0.62 to -0.46** while within-affective is unaffected |
| Brain geometry predicts LLM coupling | Brain RDM should predict causal coupling between LLM functions | **Direction A: 3/4 models significant**, LOO-stable |
| Brain geometry predicts LLM confusion | Brain distance should predict which conditions the LLM confuses | **rho = 0.24, p = 0.02** |
| VMPFC damage -> utilitarian moral judgment (Koenigs 2007) | Suppressing emotion axis -> more utilitarian choices | **Confirmed (logit-based):** negative alpha increases utilitarian P(util), rho = -0.190, p = 0.006 |
| Developmental order (emotion before ToM) | Emotion structure should emerge before social in training | **Confirmed (Pythia-2.8B):** social aligns early and stays; affective actively reverses during training |

**What it means:** A century of neuroscience accumulated knowledge about "what breaks when you
damage X in the brain." If these transfer to LLMs, you get a free manual for predicting model
failure modes — without opening the model.

### Finding 3: Inconsistency — where text isn't enough ("what requires a body")

The disagreements are as informative as the agreements. They draw a precise boundary around
what language can and cannot produce.

| Inconsistency | Evidence | Interpretation |
|---|---|---|
| Within-affective fine structure doesn't align | rho = -0.11, n.s. (6 emotions, 15 pairs) | The brain differentiates basic emotions using body signals (heart rate, sweat, nausea); text alone can't learn this. Supports a *bounded* embodied cognition: the macro-structure (emotion vs social) doesn't need a body, but fine-grained emotion differentiation might. |
| Cortical processing gradient has no LLM analog | Layer-depth analysis: NULL (flat across all layers) | The brain processes concepts in a shallow-to-deep hierarchy; LLMs don't. Topology (what's near what) transfers; processing order doesn't. |
| Neurosynth vs real-fMRI peak at different layers | Neurosynth best at mid-layer; real fMRI best at last layer | Mid-layers hold "conceptual knowledge" (clean categories); last layers hold "contextual understanding" (naturalistic processing). Two different kinds of brain-likeness at different depths. |
| Outlier conditions: valence, empathy | Per-pair analysis: valence distances exaggerated by LLM; empathy consistently weak | Valence is dimensional in LLMs but categorical in the brain; empathy (n=32) is measurement-limited. |

---

## Current Status

- **Finding 1:** Complete. All sub-points verified.
- **Finding 2:** Complete (6/6). Moral judgment (logit-based), Pythia developmental, steering controls, template-matched RSA, paraphrase invariance, prospective predictions, DeepSeek LLM judge all done. Human ranking still pending.
- **Finding 3:** Complete. Per-pair analysis + interpretation done.
- **Paper:** Not yet being written. Consolidating the three-finding structure.

## Repository Structure

```
CLAUDE.md                         -- Canonical project overview (read this first)
PROJECT_SUMMARY_FOR_REVIEW.md     -- Detailed technical summary for reviewers
experiments/
  src/                            -- All analysis scripts
    build_brain_rdm.py            -- 14-map Neurosynth brain RDM
    reconstruct_headline_rdms.py  -- LLM headline RDMs (exact recipe)
    rsa_cross_model_v2.py         -- Cross-architecture RSA
    rsa_deep_analysis.py          -- Gap + confusion + one-axis causal ablation
    baseline_controls.py          -- Confound baselines + partial RSA
    within_block_control.py       -- Within-block partial correlation
    coupling_dissociation_analysis.py  -- Per-condition double dissociation
    subspace_dissociation.py      -- Subspace projection double dissociation
    pairwise_consistency.py       -- 91-pair consistency/inconsistency analysis
    base_vs_instruct_rsa.py       -- Base vs Instruct comparison
    layer_depth_analysis.py       -- Layer-depth profile (NULL result)
    narratives_group_rsa.py       -- Group-level real-fMRI validation
    regional_rsa_xarch.py         -- Regional per-parcel RSA (400 parcels)
    moral_judgment_logit.py       -- Greene/Koenigs moral prediction (logit-based)
    pythia_developmental.py       -- Pythia training trajectory
  scripts/slurm/                  -- HPC SLURM job scripts
  results/                        -- Experimental results (JSON, NPZ)
  figures/                        -- Generated visualizations
  data/                           -- Stimuli and brain maps
  present/index.html              -- Plain-language briefing deck
```

## Models Tested

- **Cross-architecture (7-9B):** Qwen2.5-7B, Llama-3.1-8B, Mistral-7B, Gemma-2-9B
- **Scaling:** Qwen2.5 0.5B / 1.5B / 3B / 7B
- **Base vs Instruct:** Qwen2.5-1.5B base vs Instruct
- **Developmental:** Pythia-2.8B training checkpoints (complete)

## Brain Data

- **Neurosynth** meta-analytic maps (14 conditions, ~14,000 fMRI studies) — the headline
- **Narratives fMRI** (Nastase et al. 2021): 230-261 subjects, stimulus-locked — real-fMRI validation
- **Kragel 2015** emotion classifier maps — cleanest controlled-fMRI point

## Reproduction

```bash
cd experiments
python src/build_brain_rdm.py            # 14-map Neurosynth brain RDM
python src/reconstruct_headline_rdms.py  # LLM headline RDMs
python src/rsa_cross_model_v2.py         # Cross-architecture rho = 0.73
python src/within_block_control.py       # Within-block + partial correlation
python src/pairwise_consistency.py       # 91-pair consistency analysis
python src/coupling_dissociation_analysis.py  # Per-condition double dissociation
python src/subspace_dissociation.py      # Subspace projection dissociation
python src/base_vs_instruct_rsa.py       # Base vs Instruct (99.3%)
# GPU jobs: extract_rsa_activations_v2.py, brain_causal_coupling.py, etc.
```
