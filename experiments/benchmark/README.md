# BrainCog-14

**A brain-derived benchmark for social-emotional representational geometry in language models.**

BrainCog-14 tests whether a language model's internal representational geometry shares
the human brain's dominant affect-mentalizing axis and social-cognitive fine structure.
It provides 14 cognitive conditions (6 affective, 8 social/mentalistic), a reference brain
RDM built from Neurosynth meta-analytic fMRI maps, and a locked evaluation recipe. A model
"passes" if the Spearman correlation between its representational dissimilarity matrix (RDM)
and the brain RDM exceeds the permutation null. Four architectures (Qwen, Llama, Mistral,
Gemma) achieve rho ~ 0.73 (~76% of the LLM split-half reliability ceiling), and this
alignment is scale-invariant from 0.5B to 7B parameters.

**Limitation:** The alignment is dominated by the emotion-social boundary plus within-social
ordering. Within-affective fine structure does not align (rho ~ -0.10, n.s., n=6
underpowered), so the match should not be interpreted as a rich 14-way correspondence.
No brain-side noise ceiling is available; the ceiling ratio refers to the LLM's own
split-half reliability.

---

## The 14 Conditions

### Affective (6)
| Condition   | Description | N stimuli |
|-------------|-------------|-----------|
| anger       | Anger-eliciting scenarios and statements | 50 |
| disgust     | Disgust-eliciting scenarios and statements | 50 |
| fear        | Fear-eliciting scenarios and statements | 50 |
| happiness   | Joy/happiness-eliciting scenarios and statements | 50 |
| sadness     | Sadness-eliciting scenarios and statements | 50 |
| valence     | Valence-graded sentences (positive/negative affect) | 60 |

### Social / Mentalistic (8)
| Condition       | Description | N stimuli |
|-----------------|-------------|-----------|
| belief          | False-belief and belief-attribution scenarios | 60 |
| empathy         | Empathic concern and care-related scenarios | 32 |
| intention       | Intention-reading and indirect request scenarios | 60 |
| judgment        | Moral/commonsense judgment scenarios | 60 |
| mentalizing     | Strange-stories and mental-state inference tasks | 40 |
| moral           | Moral dilemmas and norm violations | 60 |
| self_referential | Self-referential processing and self/other distinction | 30 |
| theory_of_mind  | Theory of mind: perspective-taking and faux pas | 60 |

**Total: 712 stimuli across 14 conditions (30-60 per condition).**

---

## How to Evaluate a Model

### Quick Start

```bash
# Evaluate at a known good layer:
python evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --layer 27

# Sweep all layers to find the peak:
python evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --layer all

# Fast screening without permutation test:
python evaluate.py --model_path Qwen/Qwen2.5-7B-Instruct --layer 27 --no-permutation
```

### Step-by-Step Recipe

The evaluation recipe is locked to ensure reproducibility:

1. **Load stimuli.** Read `braincog14_stimuli.jsonl` (712 items, each with a `condition`
   label and `text` field).

2. **Extract hidden states.** For each stimulus, run a forward pass through the model and
   collect the hidden state at the target layer. Use **mean-all pooling**: average over all
   token positions (not just the last token).

3. **Compute condition means.** For each of the 14 conditions, average the hidden states of
   all stimuli belonging to that condition. Result: a 14 x hidden_dim matrix.

4. **Center.** Subtract the grand mean across the 14 conditions (zero-center along the
   condition axis). This removes the shared activation baseline and isolates relational
   structure.

5. **Build the model RDM.** Compute pairwise **cosine distance** (1 - cosine similarity)
   between all 14 condition centroids. Result: a 14 x 14 representational dissimilarity
   matrix.

6. **Compare to brain.** Load `braincog14_brain_rdm.npz` and extract the 91 upper-triangle
   values from both the brain and model RDMs. Compute the **Spearman rank correlation**
   between them.

7. **Significance test.** Permutation test with 5000 iterations: shuffle the 14 condition
   labels, recompute the Spearman rho, count how often the shuffled rho exceeds the observed
   rho. The 95th percentile of the null distribution is approximately 0.25.

### Configuration Summary

| Parameter   | Value |
|-------------|-------|
| Pooling     | `mean_all` (mean over all non-pad tokens) |
| Centering   | Yes (subtract condition grand mean) |
| Distance    | Cosine (1 - cosine similarity) |
| Comparison  | Spearman rank correlation |
| N pairs     | 91 (upper triangle of 14 x 14) |
| Permutations | 5000 (condition-label shuffle) |
| Max tokens  | 512 per stimulus |

---

## Interpreting Results

| Spearman rho | Interpretation |
|--------------|----------------|
| < 0.25       | **Not above chance.** The model's representational geometry does not reproduce the brain's social-emotional organization. |
| 0.25 - 0.50  | **Above chance.** Comparable to static-embedding baselines (GloVe: 0.49). The model captures some lexical-level structure but not deep representational geometry. |
| 0.50 - 0.70  | **Strong alignment.** Exceeds lexical baselines. After partialling out all text confounds (GloVe + condition name + sentence length), ~80% of alignment is retained, suggesting genuine representational structure beyond surface statistics. |
| > 0.70       | **Strong alignment.** Comparable to reference LLMs (0.73, ~76% of LLM split-half reliability ceiling). The model shares the brain's dominant emotion-social boundary and social-cognitive fine structure. Note: within-affective ordering does not align (rho ~ -0.10), so this reflects a shared axis plus within-social structure, not a full 14-way match. |

### What Does High Alignment Mean?

High rho means the model's internal representational distances between cognitive conditions
(e.g., how far apart "anger" and "fear" are vs. "anger" and "belief") match the distances
observed in the human brain's meta-analytic activation maps. The dominant axis driving this
alignment is the **emotion-vs-social cognition boundary** -- the same division that organizes
the brain's limbic and default-mode networks.

### What Does Low Alignment Mean?

Low rho means the model organizes its representations of emotion and social cognition
differently from the brain. This could indicate: (a) the model has not learned the
emotion/social boundary, (b) it represents these concepts through a different geometry, or
(c) it lacks depth in social-emotional processing (e.g., a code-only model).

---

## Reference Results

### Cross-Architecture (7-9B scale)

| Model | rho | p-value | Peak layer |
|-------|-----|---------|------------|
| Qwen2.5-7B-Instruct | 0.739 | < 0.001 | 27 |
| Meta-Llama-3.1-8B-Instruct | 0.727 | < 0.001 | 31 |
| Mistral-7B-Instruct-v0.3 | 0.730 | < 0.001 | 14 |
| gemma-2-9b-it | 0.735 | < 0.001 | 21 |

### Scale Invariance (Qwen2.5 family)

| Model | Params | rho | Peak layer |
|-------|--------|-----|------------|
| Qwen2.5-0.5B-Instruct | 0.5B | 0.752 | 12 |
| Qwen2.5-1.5B-Instruct | 1.5B | 0.754 | 3 |
| Qwen2.5-3B-Instruct | 3.1B | 0.747 | 29 |
| Qwen2.5-7B-Instruct | 7.6B | 0.739 | 27 |

### Baselines

| Baseline | rho | Notes |
|----------|-----|-------|
| GloVe-300d (mean pooled) | 0.494 | Static word embeddings; lexical co-occurrence |
| TF-IDF (5000 features) | 0.195 | Bag-of-words; not significant |
| Condition label (GloVe) | 0.519 | Embedding of condition name only; label leakage bound |
| Sentence length | 0.328 | Surface confound |
| Partial RSA (all confounds removed) | 0.588 | LLM rho after partialling out GloVe + name + length |
| Permutation null (95th percentile) | 0.250 | Chance level |

---

## Files

| File | Description |
|------|-------------|
| `evaluate.py` | Self-contained evaluation script |
| `braincog14_brain_rdm.npz` | Brain-side 14x14 RDM (Neurosynth, 1-Pearson distance) |
| `braincog14_stimuli.jsonl` | 712 stimuli (14 conditions) |
| `braincog14_config.json` | Locked evaluation configuration |
| `baselines.json` | Reference results and baselines |
| `README.md` | This documentation |

### Brain RDM Details

The brain RDM (`braincog14_brain_rdm.npz`) was built from 14 Neurosynth meta-analytic maps
(Yarkoni et al., 2011). Each condition corresponds to one term-based meta-analysis aggregating
activation coordinates across ~14,000 published fMRI studies. Maps were resampled to a common
MNI152 2mm grid (91x109x91 voxels), masked to voxels active in at least 7 of 14 maps
(902,629 voxels retained), and pairwise distance was computed as 1 - Pearson correlation.

NPZ fields: `rdm` (14x14 float64), `conditions` (14 strings, sorted alphabetically),
`n_voxels`, `grid_shape`, `distance`, `anchor_map`, `benchmark`, `description`, `source`.

---

## Dependencies

```
torch >= 2.0
transformers >= 4.30
numpy >= 1.24
scipy >= 1.10
```

No other packages are required. The evaluation script does not import from the parent
project -- it is fully self-contained.

---

## Citation

```bibtex
@article{zhang2026braincog14,
  title   = {The Brain as a Reference Frame for Language Models:
             Social-Emotional Representational Geometry},
  author  = {Zhang, Mingxu and others},
  journal = {Preprint},
  year    = {2026},
  note    = {Preprint available at [URL]}
}
```

---

## License

- **Data** (brain RDM, stimuli): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code** (evaluate.py): [MIT](https://opensource.org/licenses/MIT)

Brain maps derived from [Neurosynth](https://neurosynth.org/) (Yarkoni et al., 2011,
*Nature Methods*). Stimuli drawn from published datasets (GoEmotions, Moral Foundations
Vignettes, false-belief tasks, etc.) -- see individual `source` fields in the stimuli file.
