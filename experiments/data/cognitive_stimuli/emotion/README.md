# Emotion stimuli for the LLM functional atlas

Generated on 2026-05-25 for the emotion module of the Nature paper project.
All artifacts in this directory are produced by
`experiments/src/prepare_emotion_stimuli.py` from the raw sources listed
under "Sources" below.

Re-build with:

```bash
python -m experiments.src.prepare_emotion_stimuli
```

(no GPU required; deterministic, seed = `20260525`).

---

## Files

| File | Rows | Description |
| --- | ---: | --- |
| `warriner_vad.csv` | 13 905 | Word-level valence / arousal / dominance norms |
| `goemotions_sample.jsonl` | 5 398 unique | Balanced sentence-level emotion labels (28-way) |
| `emotion_localizer_stimuli.jsonl` | 350 | High-clarity sentences for the Ekman-6 + neutral localizer |
| `vad_graded_sentences.jsonl` | 200 | Sentence-level continuous V/A targets for regression |

Raw inputs kept for reproducibility:

| File | Source |
| --- | --- |
| `warriner_raw.csv` | JULIELab/XANEW GitHub mirror (3.5 MB CSV) |
| `goemotions_train.tsv`, `goemotions_dev.tsv`, `goemotions_test.tsv` | google-research/google-research master |
| `goemotions_labels.txt`, `goemotions_ekman.json`, `goemotions_sentiment.json` | google-research/google-research master |

---

## 1. `warriner_vad.csv` (word-level VAD norms)

Columns: `word, valence, arousal, dominance`.

* Source: **Warriner, Kuperman, Brysbaert (2013)**, *Norms of valence, arousal, and
  dominance for 13,915 English lemmas*, Behavior Research Methods.
  doi:10.3758/s13428-012-0314-x
* Each rating is the mean across the full annotator pool (`V.Mean.Sum`,
  `A.Mean.Sum`, `D.Mean.Sum` in the original Springer supplement) on a 1–9
  scale.  We do **not** keep the sex / age / education / SD columns; if you
  need them, see `warriner_raw.csv`.
* Words are lowercased and deduplicated (`drop_duplicates(subset="word",
  keep="first")`), which drops 10 of the original 13 915 rows -> **13 905
  unique lemmas**.

### Manual download (if `warriner_raw.csv` is missing)

The Springer supplement is gated behind a download link; the easiest open
mirror is JULIELab/XANEW:

```bash
curl -L -o warriner_raw.csv \
    https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv
```

Alternative mirrors (if the above changes): OSF copy at
https://osf.io/x96ud/ (search "Warriner"), or the original supplement at
https://link.springer.com/article/10.3758/s13428-012-0314-x.

---

## 2. `goemotions_sample.jsonl` (sentence-level GoEmotions)

* Source: **Demszky, Movshovitz-Attias, Ko, Cowen, Nemade, Ravi (2020)**,
  *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020.
  License: Apache-2.0 (Reddit user content; see Google's data card).
* Format (one JSON object per line):

  ```json
  {"id": "ef7hr99",
   "text": "...",
   "emotion_labels": ["admiration"],
   "valence": 1,
   "is_neutral": false}
  ```

* `emotion_labels` is the original multi-label vector (1–N labels per
  comment); `valence` is derived from GoEmotions' own
  `sentiment_dict.json` as +1 (all labels positive), -1 (all labels
  negative), or 0 (mixed / ambiguous / neutral); `is_neutral` is `True`
  iff the only label is `neutral`.

### Sampling procedure

For each of the 28 GoEmotions labels we drew up to **200** unique examples
from the union of the train/dev/test splits.  Because GoEmotions is
multi-label, the same comment can appear under multiple buckets — we
deduplicate the final file by `comment_id`, which is why the row count is
5 398 rather than 28 x 200 = 5 600.  Pool-limited categories
(`grief`: 96, `pride`: 142, `relief`: 182, `nervousness`: 193) were
sampled exhaustively.

### Network caveat

`datasets.load_dataset("go_emotions", "simplified")` was attempted but the
HPC3 head node cannot reach `huggingface.co` (`HF_ENDPOINT` is set to the
`hf-mirror.com` proxy, but the proxy does not surface the `go_emotions`
config either).  We therefore download the raw TSVs directly from
`raw.githubusercontent.com/google-research/google-research/master/goemotions/data/`
which mirrors exactly what the HuggingFace builder ingests.  See
`prepare_emotion_stimuli.iter_goemotions` for the parser.

---

## 3. `emotion_localizer_stimuli.jsonl` (Ekman localizer set)

50 sentences x 6 Ekman basic emotions + 50 neutral = **350** stimuli.
Format:

```json
{"id": "emo_loc_0007",
 "text": "Get out of my sight before I lose my temper.",
 "emotion": "anger",
 "intensity": 5,
 "source": "curated"}
```

* `emotion` ∈ {`anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`,
  `neutral`}.
* `intensity` is a 1–5 first-author estimate (5 = prototypical / very
  intense).  GoEmotions-sourced rows default to intensity `3` (single-label
  Reddit comments — moderate-to-high clarity but not hand-tuned).
* `source` ∈ {`curated`, `goemotions`}.  Curated entries (15 per
  emotion + 20 neutral, **110 total**) are author-written prototypical
  sentences with high agreement when piloted internally.  GoEmotions
  entries (35 per emotion + 30 neutral, **240 total**) are *single-label*
  Reddit comments (single-label is the strongest annotator-agreement proxy
  available in the dataset).  We additionally filter by length
  (15–200 chars), drop comments with `[NAME]` / `[RELIGION]` masks, and
  deduplicate by lowercased text.
* GoEmotions labels are mapped to Ekman categories via
  `goemotions_ekman.json` (e.g. `nervousness` -> `fear`,
  `disappointment` -> `sadness`).  `goemotions_label` is recorded on the
  row for traceability.

Recommended usage: use the **curated** subset as the primary localizer
(strong, balanced, low style variance) and the GoEmotions subset as a
broader generalization probe.

---

## 4. `vad_graded_sentences.jsonl` (continuous VAD regression set)

200 hand-curated sentences covering the V x A plane.  Format:

```json
{"id": "vad_0000",
 "text": "Everything feels grey and hollow this morning.",
 "target_valence": 2.0,
 "target_arousal": 2.0}
```

* `target_valence`, `target_arousal` ∈ [1, 9] mimic Warriner's
  metric so neuron activations can be regressed against the same scale as
  the word-level norms.
* Coverage (counted in the bins used by Russell's circumplex
  visualisations):

  | Valence band | # sentences |
  | --- | ---: |
  | 1.0–2.9 (negative) | 27 |
  | 3.0–4.4 (mild negative) | 55 |
  | 4.5–5.4 (neutral) | 34 |
  | 5.5–7.4 (mild positive) | 32 |
  | 7.5–9.0 (positive) | 52 |

  | Arousal band | # sentences |
  | --- | ---: |
  | 1.0–2.9 (low) | 34 |
  | 3.0–4.9 (mid-low) | 64 |
  | 5.0–6.4 (mid-high) | 68 |
  | 6.5–9.0 (high) | 34 |

* **Important caveat.**  These targets are first-author point estimates,
  not normed ratings.  They are intended as *initial* targets for the
  regression pipeline; the structure of the V x A grid (the fact that
  all four quadrants are populated) is the load-bearing property.  Plan
  to collect 5+ external raters per sentence before publishing
  regression results.

---

## Citations

```bibtex
@article{warriner2013norms,
  title   = {Norms of valence, arousal, and dominance for 13,915 English lemmas},
  author  = {Warriner, Amy Beth and Kuperman, Victor and Brysbaert, Marc},
  journal = {Behavior Research Methods},
  volume  = {45},
  number  = {4},
  pages   = {1191--1207},
  year    = {2013},
  doi     = {10.3758/s13428-012-0314-x}
}

@inproceedings{demszky2020goemotions,
  title     = {{GoEmotions}: A Dataset of Fine-Grained Emotions},
  author    = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo
               and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for
               Computational Linguistics},
  year      = {2020},
  url       = {https://aclanthology.org/2020.acl-main.372}
}
```

---

## Known limitations / open issues

1. **VAD-graded targets are unrated.**  Treat them as author point
   estimates; the file should be re-derived from external ratings before
   the regression results enter the paper's main claims.
2. **GoEmotions is Reddit-sourced.**  Style is informal and U.S./English
   centric; reasoning about emotion across cultures or registers should
   rely on the curated subset and Warriner norms.
3. **Pool-limited categories.**  `grief` (96), `pride` (142),
   `relief` (182), `nervousness` (193) hit fewer than 200 unique
   examples in `goemotions_sample.jsonl`; the localizer is unaffected
   because Ekman aggregation collapses these into joy / fear / sadness.
4. **Word-level Warriner intersection with model vocabularies is not
   filtered here.**  Each model script should tokenize and report
   coverage at use time (Qwen / LLaMA / Mistral / Gemma BPE will differ).
