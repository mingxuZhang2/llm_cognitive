# Three Findings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build out the paper's three-finding structure: (1) Consistency, (2) Predictability
("so what"), (3) Inconsistency ("what requires a body"). Each finding gets concrete
sub-experiments that answer the advisor's "so what" question.

**Architecture:** All new experiments build on the existing 14-condition RSA infrastructure.
LLM RDMs come from `{model}_rdm14_headline.npz` (peak-layer centroids, cosine distance).
Brain RDM from `brain_rdm.npz` (14-map Neurosynth, 1-Pearson). Ablation uses
gradient x activation importance (as in `brain_causal_coupling.py`). Behavioral tests use
generation + scoring. Pythia checkpoints streamed one at a time to avoid disk pressure.

**Tech Stack:** PyTorch, transformers, scipy, nibabel, numpy. SLURM on /hpc2hdd
(CPU: i64m512u, GPU: i64m1tga800u). conda base env.

**Models (unless noted):** Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-Instruct,
Mistral-7B-Instruct-v0.3, gemma-2-9b-it (the 4 headline architectures).

---

## Phase 1: Quick Wins (no new GPU, ~1 day)

### Task 1: Per-Pair Consistency/Inconsistency Analysis

**Purpose:** Finding 3 foundation. Systematically identify which of the 91 condition pairs
are consistent vs inconsistent between brain and LLM. Feeds the "what requires a body"
narrative.

**Files:**
- Create: `src/pairwise_consistency.py`
- Read: `results/cognitive_rsa/brain_rdm.npz`, `results/cognitive_rsa/*_rdm14_headline.npz`
- Output: `results/cognitive_rsa/pairwise_consistency.json`, `figures/pairwise_scatter.png`,
  `figures/residual_heatmap.png`

**What it computes:**
1. For each of the 91 upper-triangle pairs, extract brain distance and LLM distance (averaged
   across 4 models, also per-model).
2. Compute signed residual = rank(brain_dist) - rank(LLM_dist) for each pair.
3. Group pairs by type: within-affective (15), within-social (28), cross-block (48).
4. Report: top-5 most consistent pairs, top-5 most inconsistent, per-type statistics.
5. Visualize: scatter plot (brain vs LLM distance, colored by type, annotated outliers) +
   residual heatmap (14x14, signed residual intensity).

**Step 1:** Write `src/pairwise_consistency.py` — loads brain_rdm + 4 headline RDMs, computes
per-pair residuals, groups by type, saves JSON + figures.

**Step 2:** Run on login node (tiny computation, <1 min):
```bash
cd /hpc2hdd/home/mzhang630/data/nature/experiments
python src/pairwise_consistency.py
```

**Step 3:** Inspect output, commit.

**Expected output:** The cross-block pairs (emotion vs social) should be highly consistent
(they carry the big axis). Within-social pairs should be moderately consistent (partial
rho ~0.6). Within-affective pairs should be the most inconsistent (rho ~ -0.11). Specific
pairs like (anger, fear) vs (belief, theory_of_mind) will tell us exactly WHERE the
disagreement lives.

---

### Task 2: Recompute Base vs Instruct on Neurosynth

**Purpose:** Finding 1 gap-fill. The previously reported base/instruct numbers (0.525/0.577)
were from the broken Narratives pipeline. Recompute on the headline Neurosynth pipeline.

**Files:**
- Create: `src/base_vs_instruct_rsa.py`
- Read: `results/cognitive_rsa/brain_rdm.npz`,
  `results/cognitive_rsa/Qwen2.5-1.5B-Instruct_rsa_v2_per_stim.npz` (if it exists),
  or extract fresh from Qwen2.5-1.5B (base) model
- Output: `results/cognitive_rsa/base_vs_instruct.json`

**What it computes:**
1. Load Qwen2.5-1.5B (base, NOT Instruct) per-stim activations, or extract fresh if needed.
2. Build headline RDM (mean_all | centered | 1_cosine) at each layer.
3. Compare to brain_rdm.npz -> report peak rho for base vs instruct.
4. Compute: instruct peak rho (from existing scaling data), base peak rho, ratio.

**Decision point:** Check if base model activations already exist in the cache. If not,
need a GPU job to extract (~30 min). If they exist, this is a login-node task.

**Step 1:** Check cache, write script, run/submit.
**Step 2:** Report clean numbers, commit.

---

## Phase 2: Clinical Double Dissociation (~3 days, GPU needed)

### Task 3: Block-Level Ablation — PPL (Simple Version)

**Purpose:** Finding 2 core. Test the "psychopathy vs autism" analog: ablating the
emotion subspace produces psychopathy-like selective damage (emotion tasks collapse, social
tasks spared), and vice versa. This is the neuroscience double-dissociation transferred to LLMs.

**Neuroscience prediction being tested:**
- Shamay-Tsoory 2009 (Brain): ventral PFC lesion -> impaired emotion recognition, spared ToM
- Baron-Cohen 1995: autism -> impaired ToM, spared basic emotion

**Files:**
- Create: `src/clinical_dissociation.py`
- Create: `scripts/slurm/clinical_dissociation.sh`
- Read: `src/brain_causal_coupling.py` (reuse attribution + ablation machinery)
- Output: `results/clinical_dissociation/clinical_dissociation.json`,
  `figures/double_dissociation_heatmap.png`

**What it computes:**
1. For each model, load the per-condition gradient x activation importance (from Direction A,
   or recompute if not cached as tensors).
2. Pool neuron importance: emotion_neurons = top-5000 by mean importance across
   {anger, fear, disgust, sadness, happiness, valence}. social_neurons = same for the
   8 social conditions.
3. Three ablation conditions:
   a) Ablate emotion_neurons -> measure PPL on all 14 conditions
   b) Ablate social_neurons -> measure PPL on all 14 conditions
   c) Ablate random neurons (same count) -> baseline x 100 shuffles
4. Report: PPL increase ratio per condition per ablation. The double dissociation is:
   (a) hits emotion >> social; (b) hits social >> emotion.
5. Statistical test: paired t-test or Wilcoxon on effect sizes within vs across block.
6. Sanity check: existing coupling matrix already hints at this
   (ablate_emotion on_emotion=0.198 vs on_social=0.049; ablate_social on_social=0.122 vs
   on_emotion=0.007 for Qwen).

**Step 1:** Write `src/clinical_dissociation.py`. Reuse the attribution and ablation
functions from `brain_causal_coupling.py` (import or copy the core logic).

**Step 2:** Write SLURM script (GPU, ~2h per model, 4 models = array job).

**Step 3:** Submit: `sbatch scripts/slurm/clinical_dissociation.sh`

**Step 4:** Inspect results, generate double-dissociation figure, commit.

---

### Task 4: Clinical Dissociation — Behavioral Tests (Deep Version)

**Purpose:** Finding 2 deeper evidence. Instead of just PPL, measure actual behavioral
damage on clinical-psychology-inspired tasks. "After ablating emotion neurons, can the
model still recognize emotions? Still pass ToM tests? How does it judge moral dilemmas?"

**Files:**
- Create: `src/clinical_behavioral_battery.py`
- Create: `scripts/slurm/clinical_behavioral.sh`
- Read: `data/cognitive_stimuli/emotion/emotion_localizer_stimuli.jsonl` (350 items)
- Read: `data/cognitive_stimuli/tom/false_belief.jsonl` (100 items)
- Read: `data/cognitive_stimuli/moral/moral_dilemmas.jsonl` (30 items)
- Output: `results/clinical_dissociation/behavioral_battery.json`

**Behavioral battery (4 domains):**

a) **Emotion recognition** (tests emotion capacity):
   - Input: emotion-laden sentences from emotion_localizer
   - Prompt: "What emotion is expressed? Choose: anger, fear, disgust, sadness, happiness, neutral"
   - Metric: classification accuracy
   - Prediction: emotion-ablated model drops sharply; social-ablated model is spared

b) **Theory of Mind** (tests social cognition):
   - Input: false belief stories from tom/false_belief.jsonl
   - Prompt: use the existing question field ("Where will Sally look?")
   - Metric: accuracy (belief-consistent = correct for false belief)
   - Prediction: social-ablated model drops; emotion-ablated model is spared

c) **Moral judgment** (tests both — Greene's dual process):
   - Input: moral_dilemmas.jsonl (personal vs impersonal classified)
   - Prompt: "Which option would you choose? Respond with just 'A' or 'B'."
   - Metric: utilitarian choice rate, split by personal/impersonal
   - Prediction: emotion-ablated -> more utilitarian on personal dilemmas (Koenigs 2007)

d) **Empathy/perspective-taking** (tests social-emotional overlap):
   - Input: subset of faux_pas.jsonl + self_other.jsonl
   - Prompt: "Does [character] realize they said something awkward?"
   - Metric: accuracy on faux pas detection
   - Prediction: social-ablated drops; emotion-ablated may also drop (empathy needs both)

**Conditions:** intact model, emotion-ablated, social-ablated, random-ablated (control).
4 conditions x 4 models x 4 tasks.

**Step 1:** Write `src/clinical_behavioral_battery.py` with the generation + scoring pipeline.

**Step 2:** Write SLURM script (GPU, generation-heavy, ~3-4h per model).

**Step 3:** Submit, collect results.

**Step 4:** Produce the key table:

```
                    Intact  Emotion-ablated  Social-ablated  Random-ctrl
Emotion recog.      85%     45% (!!!)        82%             83%
ToM false belief    78%     75%              40% (!!!)       76%
Moral (personal)    30%U    65%U (!!!)       35%U            32%U
Faux pas            72%     55%              35% (!!!)       70%
```

(Numbers are illustrative. The pattern matters: selective damage.)

**Step 5:** Commit.

---

## Phase 3: Moral Judgment Quantification (~1 day, GPU)

### Task 5: Greene/Koenigs Moral Prediction

**Purpose:** Finding 2. Test a specific, famous neuroscience result in LLMs:
"VMPFC damage -> more utilitarian judgments on emotionally salient moral dilemmas"
(Koenigs et al. 2007, Nature). Our analog: suppressing the emotion axis -> utilitarian shift.

**Neuroscience prediction:**
- Greene 2001 (Science): personal moral dilemmas engage emotion -> deontological
- Koenigs 2007 (Nature): VMPFC lesion patients make more utilitarian choices
- Our prediction: steering along negative-emotion direction -> more utilitarian

**Files:**
- Create: `src/moral_judgment_test.py`
- Read: `data/cognitive_stimuli/moral/moral_dilemmas.jsonl`
- Read: `src/cognitive_steering.py` (reuse steering machinery)
- Output: `results/moral_judgment/moral_judgment.json`, `figures/moral_steering_curve.png`

**What it computes:**
1. Load each model, compute the emotion-social boundary direction (as in steering code).
2. For each moral dilemma (classified as personal or impersonal):
   a) Generate response at alpha = {-20, -10, -5, 0, 5, 10, 20}
   b) Extract: forced choice (A or B) -> map to utilitarian/deontological
3. Report: utilitarian rate vs steering strength, split by personal/impersonal.
4. Key prediction: on PERSONAL dilemmas, negative alpha (suppress emotion) increases
   utilitarian rate. On IMPERSONAL dilemmas, effect should be weaker.
5. Statistical test: logistic regression of (utilitarian choice ~ alpha * dilemma_type).
6. Visualize: steering curve (x=alpha, y=utilitarian %, two lines for personal/impersonal).

**Step 1:** Write script, submit GPU job (~1h per model).
**Step 2:** Analyze, generate figure, commit.

---

## Phase 4: Pythia Developmental Trajectory (~3 days, GPU)

### Task 6: Pythia Checkpoint Extraction

**Purpose:** Finding 2. Test the developmental prediction: "emotion structure forms before
social cognition in training" (cf. basic emotions develop before ToM in child development).

**Neuroscience prediction:**
- Basic emotions: present from birth/early infancy
- Theory of mind: ~age 4 (Wimmer & Perner 1983)
- Prediction: in LLM training, emotion-block alignment with brain should rise BEFORE
  social-block alignment

**Files:**
- Create: `src/pythia_developmental.py`
- Create: `scripts/slurm/pythia_developmental.sh`
- Output: `results/developmental_emergence/pythia_trajectory.json`,
  `figures/pythia_developmental_curve.png`

**Model:** Pythia-2.8B (EleutherAI/pythia-2.8b-deduped). 2.8B parameters, 32 layers,
2560 hidden dim. Good balance of size and speed.

**Checkpoints (~20, log-spaced):**
step 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000,
16000, 32000, 64000, 100000, 143000

**What it computes:**
1. For each checkpoint:
   a) Load model from HF with `revision=f"step{step}"`
   b) Feed 14-condition stimuli (same as headline), extract hidden states
   c) Build LLM RDM (mean_all | centered | cosine) at each layer
   d) Compute vs brain: full rho, within-affective rho, within-social rho,
      emotion-social split strength (binary model correlation)
   e) Save per-checkpoint summary, free model
2. Track across training: 4 curves (full rho, split strength, within-aff, within-soc).
3. Key question: does the emotion-social SPLIT appear before within-social structure?
   Does within-affective ever appear?

**Resource estimate:**
- Per checkpoint: ~60 stim x 14 cond x 1 forward pass = ~840 passes, ~10 min on A800
- 20 checkpoints: ~3.5h
- Model loading: ~2 min per checkpoint = ~40 min
- Total: ~4-5h GPU time

**Step 1:** Write extraction + analysis script (stream checkpoints, don't store all).
**Step 2:** Write SLURM script (single GPU job, A800, 8h wall time).
**Step 3:** Submit, collect results.
**Step 4:** Generate developmental curve figure. The key visual: x-axis = training step
(log scale), y-axis = rho. Four lines: full, split, within-social, within-affective.
**Step 5:** Commit.

**If emotion structure demonstrably appears before social structure:** this is a strong
positive result that mirrors the developmental neuroscience prediction.

**If they appear simultaneously:** still informative — means the distinction is NOT ordered.

**If social appears first:** contradicts the prediction, must report honestly.

---

## Phase 5: Synthesis and Figures

### Task 7: Organize Results into Three-Finding Structure

**Purpose:** Take all completed experiments and organize them into the paper's structure.

**Files:**
- Update: `PROJECT_SUMMARY_FOR_REVIEW.md`
- Update: `CLAUDE.md`
- Create: `experiments/figures/` — publication-quality versions of all key figures

**Finding 1 (Consistency) sub-points:**
- a) rho = 0.73, 4 archs, scale-invariant (existing)
- b) Dominated by emotion-social axis + significant residual (existing)
- c) Not text artifact: partial RSA 80% (existing)
- d) Real fMRI: 3 datasets positive, 1 significant (existing)
- e) Base vs Instruct (Task 2 — recomputed)

**Finding 2 (Predictability = "so what") sub-points:**
- a) Brain predicts LLM causal coupling (Direction A, existing)
- b) Clinical double dissociation (Task 3 + 4)
- c) Moral judgment: Koenigs prediction confirmed in LLM (Task 5)
- d) Developmental trajectory: Pythia (Task 6)

**Finding 3 (Inconsistency) sub-points:**
- a) Systematic per-pair analysis (Task 1)
- b) Within-affective doesn't align -> embodiment hypothesis
- c) Layer depth: NULL -> no processing hierarchy analog
- d) Different peak layers (Neurosynth mid-layer vs real-fMRI last-layer)

---

## Execution Order and Dependencies

```
Phase 1 (Day 1, no GPU):
  Task 1 (per-pair analysis) ─┐
  Task 2 (base vs instruct)  ─┴─> commit

Phase 2 (Day 2-4, GPU):
  Task 3 (PPL dissociation) ──> Task 4 (behavioral battery)
                                  [Task 4 depends on Task 3 confirming signal exists]

Phase 3 (Day 3-4, GPU, parallel with Phase 2):
  Task 5 (moral judgment)

Phase 4 (Day 4-6, GPU, parallel):
  Task 6 (Pythia developmental)

Phase 5 (Day 7):
  Task 7 (synthesis)
```

Tasks 1-2 are independent and can run in parallel (Day 1).
Tasks 3, 5, 6 are independent GPU jobs and can run in parallel if slots are available.
Task 4 depends on Task 3 (need to confirm signal before running the expensive behavioral battery).
Task 7 depends on all others.

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Pythia-2.8B may not show the 14-condition structure at all (too small/different training) | Run a quick sanity check at final checkpoint first; if rho ~ 0, downgrade to Discussion |
| Behavioral battery results may be noisy (generation is stochastic) | Run 3 seeds per condition, report mean + CI |
| Disk space (95% used) | Stream Pythia checkpoints; delete after extraction |
| GPU queue contention | Submit jobs early; Task 1-2 need no GPU |
| Moral dilemmas only 30 items | Supplement with additional trolley-problem variants if needed |
| Base model activations may not exist in cache | Check first; if missing, quick GPU extraction (~30 min) |
