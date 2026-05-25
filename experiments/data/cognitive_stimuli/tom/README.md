# Theory of Mind and Social Cognition Stimuli

Curated stimuli set for probing LLM neurons that support Theory of Mind (ToM) and social cognition. Items are organized into five canonical paradigms drawn from developmental and clinical neuropsychology.

## Files and counts

| File | Total | Condition breakdown |
|---|---|---|
| `false_belief.jsonl` | 100 | 50 first-order false-belief + 50 true-belief controls |
| `faux_pas.jsonl` | 60 | 31 with faux pas + 29 without |
| `indirect_requests.jsonl` | 60 | 30 indirect requests + 30 literal controls |
| `strange_stories.jsonl` | 40 | 13 mental-state categories |
| `self_other.jsonl` | 90 | 30 self + 30 other + 30 reality, paired by triplet |
| **Total** | **350** | |

## Schemas

### `false_belief.jsonl`
Sally-Anne style location-change false-belief tasks plus matched true-belief controls.
Fields:
- `id`: stable identifier (`fb_NNN` for false-belief, `fb_tc_NNN` for true-belief control)
- `story`: narrative containing the object transfer
- `question`: where will the protagonist look for the object?
- `belief_consistent_answer`: location the protagonist believes (in false-belief case = original location)
- `reality_consistent_answer`: location where the object actually is
- `condition`: `"false_belief"` or `"true_belief_control"`

In false-belief items the protagonist leaves while the object is moved. In true-belief controls the protagonist witnesses the move, so belief and reality coincide.

### `faux_pas.jsonl`
Stone, Baron-Cohen & Knight (1998) Faux Pas Recognition Test paradigm. Each story has a character who says something unintentionally hurtful (faux pas) or not (control).
Fields:
- `id`: `fp_NNN`
- `story`: full narrative
- `has_faux_pas`: boolean
- `faux_pas_speaker`: who committed the faux pas (or `null`)
- `faux_pas_listener`: who was the target (or `null`)
- `contextual_info`: explanation of why it does/does not constitute a faux pas

### `indirect_requests.jsonl`
Pragmatic inference: utterances that on the surface describe a state of affairs, but function as requests for action.
Fields:
- `id`: `ir_NNN`
- `context`: brief situational description
- `utterance`: the surface form
- `interpretation_type`: `"indirect_request"` or `"literal"`
- `inferred_meaning`: the intended interpretation given the context

### `strange_stories.jsonl`
Happé (1994) Strange Stories paradigm. Each story tests inference of a particular non-literal mental state.
Fields:
- `id`: `ss_NNN`
- `story`: narrative
- `question`: why did the character say or do that?
- `mental_state_category`: one of `lie`, `white_lie`, `joke`, `pretend`, `misunderstanding`, `persuasion`, `sarcasm`, `figure_of_speech`, `double_bluff`, `appearance_reality`, `kindness`, `principle`, `absentmindedness`
- `correct_answer`: target mental-state attribution
- `distractor_answer`: a salient but incorrect interpretation (typically literal or competing)

### `self_other.jsonl`
Triplet items varying which perspective is queried while holding the scenario fixed. The same scenario is asked once for the participant's belief (self), once for another agent's belief (other), and once for reality.
Fields:
- `id`: `so_self_NNN`, `so_other_NNN`, `so_reality_NNN`
- `scenario`: situation in which self, other, and reality diverge
- `question`: perspective-specific probe
- `perspective`: `"self"` | `"other"` | `"reality"`
- `correct_answer`: target answer for that perspective

Triplets share the trailing `NNN`, so item `so_self_001`, `so_other_001`, and `so_reality_001` refer to the same scenario.

## Sources and curation methodology

No internet access was available in this environment, so the items were curated to match the published paradigms rather than reproduced from official datasets. References that informed the construction:

- **False-belief**: Wimmer & Perner (1983) `Cognition` 13:103-128; Baron-Cohen, Leslie & Frith (1985) `Cognition` 21:37-46. Each item follows the canonical Sally-Anne structure: agent A places an object, leaves, agent B moves it, A returns. Controls match the same structure but with A present during the move.
- **Faux pas**: Stone, Baron-Cohen & Knight (1998) `J. Cognitive Neuroscience` 10:640-656. Items model the original test's structure (a speaker says something they would not have said had they known a contextual fact known to the protagonist) along with affect-matched controls of similar length.
- **Indirect requests**: Searle (1975) Indirect Speech Acts; Holtgraves (1998); the FANToM and SocialIQA paradigms. Each item is paired (across the file) with a literal control of comparable length and surface complexity.
- **Strange stories**: Happé (1994) `J. Autism Dev. Disorders` 24:129-154. Items span the canonical 12 mental-state types plus three additional (`appearance_reality`, `kindness`, `principle`, `absentmindedness`) used in follow-up versions.
- **Self-other**: derived from the dual-belief / divergent perspective paradigm (Perner & Wimmer, 1985; Frith & Frith, 2003), structured as a 3-way triplet to enable within-scenario perspective contrasts.

## Design notes

- Length matched within file across conditions (sentence count and mean character length comparable).
- Character names, settings, and objects varied across items to avoid lexical confounds.
- Answers (where applicable) are short and unambiguous to support exact-match evaluation under causal ablation.
- All items written in present tense (false-belief, indirect requests) or simple past (strange stories, faux pas) for tense consistency within file.

## Intended use

These stimuli are intended as inputs to the LLM functional atlas attribution pipeline (see project root `CLAUDE.md`). Recommended pairings:
- `false_belief.jsonl`: contrast false-belief vs. true-belief-control logits/activations.
- `faux_pas.jsonl`: probe detection of social-norm violation (yes/no judgment).
- `indirect_requests.jsonl`: probe pragmatic inference (target = inferred meaning vs. literal).
- `strange_stories.jsonl`: probe non-literal mental-state attribution; use `mental_state_category` for sub-condition analysis.
- `self_other.jsonl`: probe perspective-taking; analyze self vs. other vs. reality channels separately or contrastively.
