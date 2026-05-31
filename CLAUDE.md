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
   The asymmetry **dissolved** — both emotion and social cognition now align near ceiling.
   We **dropped** the "divergence / anti-aligned / 自成一套" claims. (See
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
- Per-block (7B): affective 0.74 (ceiling 0.94 → **78%**), mentalistic 0.70 (ceiling 0.81 →
  **87%**) — *both* near the noise ceiling; no asymmetry.
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

Downstream neuroscience programs that could become further LLM predictions: dual-route
empathy (Shamay-Tsoory 2009, *Brain*), clinical mirror-disorders (psychopathy vs autism;
Blair; Baron-Cohen 1995), dual-process moral cognition + lesion→behavior (Greene 2001
*Science*; Koenigs 2007 *Nature*), cortical processing gradient placing social cognition at
the abstract end (Margulies 2016 *PNAS*) → maps onto LLM layer depth.

---

## Controlled-fMRI validators (status: directional supplement, demoted)

Re-audited 2026-05-30 with the corrected headline LLM RDMs + pure-NS brain
(`src/kragel_ibc_reaudit.py` → `results/affective_validation/kragel_ibc_reaudit.json`):
- **Kragel 2015** (CANlab emotion classifier maps, N=32): LLM ρ **+0.629** (4 conditions),
  vs-Neurosynth −0.03; non-significant (only 4 conditions).
- **IBC** (NeuroVault coll. 2138, 12 subjects, multi-task contrasts): LLM ρ **+0.264**
  (6 conditions), vs-Neurosynth +0.18; non-significant; has outlier maps (valence row −1.0).
- **HCP** (coll. 457, group average): the lone null — and the source of the discredited ToM
  map, so its earlier disagreement was its own artifact.

**Conclusion:** demote Kragel/IBC/HCP from "validation" to "directional supplement." The
Neurosynth headline stands on its own (it is the statistically-powered result).

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
| `present/build_present.py` | Regenerate the HTML briefing. |

**Traps (do not repeat):**
- `{model}_rsa_llm_rdms.npz` is an **older recipe** (last_tok/raw/pearson) → gives ρ≈0.25,
  flips subset signs. Always use `{model}_rdm14_headline.npz` instead.
- The `rsa_v2.json` ρ fields are **stale** (0.63, old brain RDM). Recompute fresh vs
  `brain_rdm.npz`.

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
