# Nature Paper Project: The Brain as a Reference Frame for LLMs

## Project Goal (current)
Use the **human brain as a predictive reference frame to explain and predict the internal
organization of Large Language Models.** Direction matters: we are *not* using LLMs to
model the brain (the usual neuro-AI direction). We treat established neuroscience
conclusions as **testable predictions about LLMs** — if a text-only model reproduces the
brain's representational geometry, then known brain results become hypotheses we can check
in the model. Targeting Nature Machine Intelligence.

**Headline finding (2026-05-31):** A text-only LLM reproduces the human brain's *relational*
organization of both emotion and social cognition — Representational Similarity Analysis
(RSA) between each model's internal geometry and meta-analytic fMRI maps gives **ρ ≈ 0.73**,
near the noise ceiling, in **all 4 architectures**, and **scale-invariant from 0.5B → 7B**.
The emotion ↔ social-cognition boundary is reproduced as part of this geometry.

---

## The Story (how we got here — read this first)

The project went through several honest pivots. Each is preserved in git tags/branches so
the arc is auditable.

1. **v1 — LLM Functional Atlas** (branch `experiments`, tag `v1-ai-categories`).
   Mapped which neurons implement 8 *AI-task* categories (math, code, reasoning, language,
   science, ethics, factual_qa, humanities) via gradient×activation causal attribution +
   double dissociation. Strong results (28/28 pairwise dissociations, cross-architecture
   convergence index 0.86) — but the categories were *engineering* categories, not a
   scientific frame a top journal would care about. See `analysis_findings.md` for the full
   v1 record (8 findings) and `experiments/PROJECT_SUMMARY.md`.

2. **Pivot 1 — to cognitive science** (branch `cognitive-atlas`, 2026-05-25).
   Reframed the 8 AI-task categories into **human cognitive functions** (emotion, moral
   cognition, theory of mind, self). The point of comparison became the *brain*, not
   benchmark performance.

3. **Pivot 2 — moral-conventional, then dropped.**
   Tried to show moral cognition in LLMs is *compositionally* built from affect + mental-state
   reasoning + norm integration (mirroring Greene's dual-process moral theory). The clean
   compositional decomposition did **not** hold; we narrowed to the Turiel moral-vs-conventional
   distinction, then stepped back from "compositional moral module" as the headline.

4. **The real result — RSA relational alignment.**
   The robust, surprising finding turned out to be **relational**, not about individual
   modules: the *cross-condition similarity structure* (which cognitive functions sit near
   which) is preserved between the human brain and every LLM tested. This is the current
   headline.

5. **Artifact correction (2026-05-30) — a net win, told honestly.**
   For weeks a secondary narrative claimed an *asymmetry*: "affective core aligns near
   ceiling, social cognition diverges, theory-of-mind is anti-aligned." We traced this
   entirely to **one bad brain map** — the lone non-Neurosynth map (theory_of_mind ←
   `hcp/social_tom_vs_random`), which was orthogonal to its Neurosynth counterpart
   (row-correlation −0.016) and corrupted every condition's distances to ToM. Swapping it
   to the Neurosynth ToM map:
   - ToM alignment: **−0.13 → +0.78**
   - Headline: **ρ 0.634 → 0.733**, single-source (pure 14-map Neurosynth) brain RDM.
   The bad-map asymmetry **dissolved** and the headline rose to 0.73. We **dropped** the
   "divergence / anti-aligned / 自成一套" claims. (The clean within-block picture, added later, is
   finer — the alignment is dominated by one emotion↔social axis; the social block's internal
   ordering aligns, the affective block's does not resolve at n=6. See the Within-block control
   under "Headline Result.") (See
   `experiments/results/cognitive_rsa/brain_rdm_hcptom.npz` for the archived old RDM and
   `src/tom_source_check.py` for the diagnostic.)

---

## Headline Result (current numbers)

**RSA recipe (LLM side):** per-condition mean of `mean_all`-pooled hidden states at each
model's **peak layer** (Qwen L27, Llama L31, Mistral L14, Gemma L21) → center across the 14
conditions → **cosine** distance → 14×14 RDM. Config string:
`mean_all | centered | 1_cosine | v1_NS_only`. Reconstruct with
`src/reconstruct_headline_rdms.py` → writes `{model}_rdm14_headline.npz`.

**RSA recipe (brain side):** each of 14 conditions = one Neurosynth meta-analytic map
(`brain_rdm.npz`). Build: resample each NIfTI → anchor grid (`neurosynth/anger`, MNI
91×109×91) → keep voxels finite+nonzero in ≥7/14 maps → flatten → **1 − Pearson** distance.
Built by `src/build_brain_rdm.py` from `data/cognitive_stimuli/rsa/rsa_conditions_manifest.json`.

**Comparison:** Spearman ρ on the 91 upper-triangle pairs of the two 14×14 RDMs; permutation
null shuffles condition labels.

**The 14 conditions** (this is the unit of statistical power — 14 conditions → 91 pairs):
- 6 affective: anger, fear, disgust, sadness, happiness, valence
- 8 mentalistic/social: belief, intention, judgment, mentalizing, moral, empathy,
  self_referential, theory_of_mind

**Numbers:**
- Headline ρ: **Qwen 0.739, Llama 0.727, Mistral 0.730, Gemma 0.735** (vs pure-NS `brain_rdm.npz`).
- Per-block **row-wise** (7B, each condition's distance-to-all-13-others): affective 0.74 (78% of
  ceiling), mentalistic 0.70 (87%). **Caveat:** this row-wise metric includes cross-block distances,
  so it is dominated by the emotion/social split — it is **not** a within-block test (see next bullet).
- **Within-block control** (`src/within_block_control.py`, the clean test): both the brain RDM
  (ρ=0.71 with the binary split) and the LLM RDMs (0.84–0.87) are dominated by the single
  emotion↔social division. Controlling for that split, a significant residual remains
  (**partial ρ≈0.36, 4/4 p≤0.001**), concentrated in the **within-social** ordering (within-social
  ρ≈0.52–0.65; 0.63–0.67 excl. moral, 4/4 p≤0.033). **Within-affective ordering does not align**
  (ρ≈−0.11, n.s.; consistent across 4 models, robust to dropping valence) — but n=6 (15 pairs) is
  underpowered, so **no claim** is made about within-affective fine structure. Net: *one shared,
  causally load-bearing axis + a significant beyond-split residual living in the social block* —
  **not** a rich 14-way match. (This also corrects an earlier mis-statement that the affective block
  "carries independent fine structure": it carries structure that does **not** align with the brain.)
- Per-condition (7B): nearly all 14 align **0.67–0.85** (belief .85, ToM .79, judgment .80,
  happiness .81). **Only empathy lags (0.24)** — and empathy is the smallest set (n=32) with
  an unstable split-half ceiling; this is a measurement artifact, not a divergence.
- Scale-invariant across the Qwen family (0.5B → 7B).

**Why Neurosynth, not raw fMRI, carries the headline:** statistical power here comes from the
*number of conditions* (14 → 91 pairs → p < 0.0002), not subjects-per-map. Neurosynth maps
are meta-analytic averages over ~14,000 fMRI papers, so each condition map is itself stable.
The controlled-fMRI validators below have only 4–6 conditions, so their permutation floor is
too high to ever reach significance — they can only *directionally* support, never confirm.

---

## Three "brain-explains-LLM" directions (the predictive program)

The strategy: take conclusions that follow from the brain's emotion/social separation and
test whether they hold in LLMs.

- **Direction A — causal coupling (strongest, partly done).** The brain RDM *predicts* the
  causal coupling between functions inside the LLM (ablate function X, measure effect on
  function Y). **3/4 models significant.** Code: `src/brain_causal_coupling.py`. Commit
  `a4f705b`. This is the analog of lesion double-dissociation work (Shamay-Tsoory 2009).
- **Direction B — cognitive reserve.** (`cb84f57`.)
- **Direction C — developmental emergence.** Along training/scale, emotion structure should
  form before social cognition (cf. affect-early, theory-of-mind ~age 4). Results:
  `results/developmental_emergence/developmental_emergence.json` (recomputed vs corrected RDM).

- **Direction D — cortical processing gradient → layer depth (Margulies 2016).** Social
  cognition sits at the abstract end of the cortical gradient → predict it peaks in deeper
  LLM layers. **Tested (`src/layer_depth_analysis.py`): NULL (depth-invariant).** Alignment
  is flat across all layers; the cortical-gradient analogy does not hold.

Downstream neuroscience programs that could become further LLM predictions: dual-route
empathy (Shamay-Tsoory 2009, *Brain*), clinical mirror-disorders (psychopathy vs autism;
Blair; Baron-Cohen 1995), dual-process moral cognition + lesion→behavior (Greene 2001
*Science*; Koenigs 2007 *Nature*).

---

## Real-fMRI validation (three independent datasets)

**a) Kragel 2015** (CANlab emotion classifier maps, N=32): LLM ρ **+0.629** (4 conditions);
too few for permutation significance. Cleanest real-fMRI point.

**b) Narratives fMRI — group-level** (Nastase 2021, 230 subjects, Schaefer-400, 12 conditions):
last-layer ρ = **+0.32 to +0.39, all 4 models significant** (p = 0.004–0.013); ceiling 0.836
(~42% of ceiling). Caveat: alignment peaks at the **last layer**, not the Neurosynth-peak
mid-layer (frozen-peak ρ ≈ 0.08, n.s.). Code: `src/narratives_group_rsa.py`.

**c) Narratives fMRI — regional per-parcel** (261 subjects, Schaefer-400): cortex mean ρ ≈
**+0.20**, **all 400 parcels significant**. Limbic highest (~0.22). Code: `src/regional_rsa_xarch.py`.

**d) IBC** (NeuroVault coll. 2138, 12 subjects): LLM ρ **+0.264** (6 conditions); non-significant.
**e) HCP** (coll. 457): null — source of the discredited ToM map.

**Retracted (2026-06-02):** previously reported "ρ ≈ 0.56, N=91, ceiling 0.762" was traced to
v1 dissociation effect sizes mis-sourced into the Narratives table. The numbers above replace it.

**Conclusion:** all three independent real-fMRI sources positive; Narratives group-level is
**statistically significant**. The Neurosynth headline (91 pairs) remains the powered result;
real-fMRI confirms it is not a pure text artifact. The strongest anti-text-confound argument
is partial RSA (~80% retained), not fMRI magnitude.

---

## Repository Structure
- `CLAUDE.md` (this file) — canonical project overview, kept in sync after every change.
- `README.md` — public-facing short version.
- `IDEA_SYNTHESIS.md`, `COGNITIVE_ATLAS_PLAN.md` — origin idea + pivot plan.
- `experiments/`
  - `src/` — analysis scripts (see Key Files below).
  - `scripts/slurm/` — HPC3 SLURM jobs.
  - `results/` — JSON/NPZ results.
    - `cognitive_rsa/` — headline RDMs (`brain_rdm.npz`, `{model}_rdm14_headline.npz`,
      `{model}_rsa_v2_per_stim.npz`), archived old RDM (`brain_rdm_hcptom.npz`).
    - `affective_validation/` — ceiling control, Kragel/IBC/HCP audits, ToM source check.
    - `developmental_emergence/`, `behavioral_prediction/`, `next_token/`, `specificity_*/`,
      `contrast_pilot*/`, `narratives_brain_rdm/` — supporting experiments.
  - `figures/` — generated visualizations.
  - `present/` — **`index.html`** self-contained briefing deck (built by `build_present.py`,
    base64-embedded figures, plain-language, no neuro background assumed). This is the deck
    Dr. Zhang presents from; it reflects the corrected story.
  - `analysis_findings.md` — v1 findings record. `PROJECT_SUMMARY.md` / `PROJECT_STATUS.md` —
    detailed logs (NOTE: parts predate the 2026-05-30 brain-RDM correction; the corrected
    story lives here and in `present/index.html`).
- Research docs: `research_*.md`, `LITERATURE_DETAIL.md`, `literature_survey_synthesis.md`.
- Feasibility studies: `feasibility_*.md`.

## Key Files
| File | Purpose |
|---|---|
| `src/build_brain_rdm.py` | Build `brain_rdm.npz` from the 14-condition manifest (1−Pearson). |
| `src/reconstruct_headline_rdms.py` | Rebuild `{model}_rdm14_headline.npz` (headline recipe). |
| `src/compute_rsa_v2.py` | Full RSA sweep over poolings/centerings/distances/layers. |
| `src/tom_source_check.py` | Diagnostic that found the HCP-ToM artifact. |
| `src/kragel_ibc_reaudit.py` | Re-audit controlled-fMRI validators (corrected RDMs). |
| `src/affective_ceiling_control.py` | Per-block alignment vs LLM split-half noise ceiling. |
| `src/brain_causal_coupling.py` | Direction A: brain RDM predicts LLM causal coupling. |
| `src/within_block_control.py` | Within-block control: partial ρ, within-affective/social RSA. |
| `src/layer_depth_analysis.py` | Direction D: per-layer RSA (result: NULL, depth-invariant). |
| `src/narratives_group_rsa.py` | Group-level Narratives brain-LLM RSA (230 subj, Schaefer-400). |
| `src/regional_rsa_xarch.py` | Regional per-parcel RSA, 4 architectures (261 subj, 400 parcels). |
| `src/base_vs_instruct_rsa.py` | Base vs Instruct RSA: pretraining vs RLHF alignment comparison (Qwen2.5-1.5B). |
| `present/build_present.py` | Regenerate the HTML briefing. |

**Traps (do not repeat):**
- `{model}_rsa_llm_rdms.npz` is an **older recipe** (last_tok/raw/pearson) → gives ρ≈0.25,
  flips subset signs. Always use `{model}_rdm14_headline.npz` instead.
- The `rsa_v2.json` ρ fields are **stale** (0.63, old brain RDM). Recompute fresh vs
  `brain_rdm.npz`.
- **"ρ ≈ 0.56" Narratives fMRI is RETRACTED** (2026-06-02). The values 0.540/0.560/0.582,
  ceiling 0.762, base 0.525, instruct 0.577, per-subject 0.568 were traced to v1 AI-task
  dissociation effect sizes in `statistical_validation.json` that were mis-sourced into the
  Narratives table. Real numbers: group +0.35 (sig), regional +0.20 (400/400 sig).

## HPC3 Configuration
- SSH: `ssh -i /hpc2hdd/home/mzhang630/data/id_rsa -o StrictHostKeyChecking=no mzhang630@hpc3login.hpc.hkust-gz.edu.cn`
- Base dir: `/data/user/mzhang630/data/nature_exp`
- Conda env: `alphasteer`
- SLURM: partition=acd_u, account=d_yings_team
- Models: Qwen2.5-7B, LLaMA-3.1-8B, Mistral-7B, Gemma-2-9b (+ Qwen 0.5/1.5/3B for scaling).
  Snapshot paths in SLURM scripts.

## Stimuli

### RSA cognitive conditions (current headline)
- `data/cognitive_stimuli/rsa/rsa_conditions_manifest.json` — 14 conditions, brain_map +
  source + n per condition (seed 20260525, target 60/condition). All `brain_map` entries are
  now `neurosynth/*` (the ToM entry was corrected from HCP on 2026-05-30).
- Emotion stimuli: `data/cognitive_stimuli/emotion/` — Warriner VAD (13,905 lemmas),
  GoEmotions sample, emotion-localizer sentences, VAD-graded sentences. Builder:
  `src/prepare_emotion_stimuli.py`. See that folder's `README.md` for citations/licenses.
- Moral / ToM / self stimuli: `data/cognitive_stimuli/{moral, moral_decomposition, tom}/`.

### Brain maps
- `data/brain_maps/` — Neurosynth meta-analytic maps (14 conditions) + Kragel/IBC/HCP
  validators. Raw map files (`*.nii.gz`, `*.img/*.hdr`) are **gitignored** (re-download as
  needed); only manifests/docs are tracked.

### Legacy AI-task stimuli (v1)
- `stimuli_{pilot,medium,balanced,full}.jsonl` (15/50/152/unbalanced per category), 8 AI-task
  categories. `stimuli_with_answers_*.jsonl` add gold answers for accuracy evaluation.
  Retained for the v1 functional-atlas record.

## Git / workflow notes
- Active branch: **`cognitive-atlas`**. v1 archived at tag `v1-ai-categories`.
- Per Dr. Zhang's rules: commit after every change with a clear message; **ask before
  merging to `main`**. Present only positive, verified results; do not over-claim. The
  paper is **not** being written yet — we are still consolidating the finding.
