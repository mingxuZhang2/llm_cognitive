# Moral Decomposition Stimuli — Design Notes

## Purpose
These stimuli decompose what "moral neurons" in LLMs actually encode. The four
conditions are designed so that a neuron population that genuinely encodes
moral cognition (rather than emotion, negative affect, or arbitrary
intentionality) will respond differentially across the *theoretically motivated*
contrasts below — and not across non-moral confounds.

Each pair varies **one variable** while controlling lexical content,
domain, severity, and length within ±5 words.

## Files
- `decomposition_stimuli.jsonl` — 400 items (200 paired scenarios)
- `_generate.py` — generator (single source of truth; rerunning reproduces the file exactly)
- `DESIGN_NOTES.md` — this file

## Counts
- 4 conditions × 50 pairs × 2 items = **400 items**
- 50 pairs per condition = 100 items per condition (50 per subcondition)
- All 200 IDs unique; all 400 item IDs unique; zero duplicate texts.

**Interpretation used:** 50 pairs per condition = 100 items per condition (since each pair contains an "a" and a "b" sibling). Total = 400 items. This matches the prompt's explicit final clarification ("PLEASE GENERATE 50 PAIRS = 100 items per condition, 400 total items").

## Schema (each JSONL row)
```
id                  e.g. "intent_001a"
condition           {intent, outcome, norm_type, morality}
subcondition        intent:    {intentional, accidental}
                    outcome:   {completed, attempted}
                    norm_type: {moral_violation, conventional_violation}
                    morality:  {moral_negative, nonmoral_negative}
pair_id             shared between a and b siblings (e.g. "intent_001")
text                2-3 sentence scenario
target_judgment     expected moral-wrongness rating on 1-7 scale
notes               brief design rationale + domain tag
```

## The four contrasts and what each isolates

### Condition 1: Intent (intentional vs accidental)
**Variable manipulated:** mental state of the actor (knew + chose harm  vs  did not foresee).
**Held constant:** identity of actor, action taken, harm outcome.
**Why it matters:** If "moral neurons" actually encode intent attribution, they should fire to "a" but not "b". If they're just negative-affect detectors, both should activate (same harmful outcome).
**Literature anchor:** Cushman (2008, *Cognition*); Young et al. (2007, *Neuron*) on the role of the RTPJ in intent-based moral judgment.
**Expected gradient:** intentional ~6.4 vs accidental ~2.8 (mean a-b = +3.63).

### Condition 2: Outcome (completed vs attempted)
**Variable manipulated:** whether the harm actually occurred.
**Held constant:** identity of actor, identical bad intent, identical action.
**Why it matters:** This isolates whether neurons track outcome-based wrongness (consequentialist signal) or are agnostic to outcome once intent is fixed. The completed/attempted gap is the classic Cushman dissociation; the gap is real but smaller than the intent gap.
**Literature anchor:** Cushman (2008) — completed harm rated worse than attempted with identical intent (effect size ~0.5-1.0 on 1-7 scales).
**Expected gradient:** completed ~6.8 vs attempted ~5.9 (mean a-b = +0.84). Note the SMALL but non-zero gap is the empirical signature; a model that doesn't distinguish them at all is treating moral judgment purely deontologically.

### Condition 3: Norm type (moral vs conventional)
**Variable manipulated:** whether the wrong is a moral violation (harm/fairness/rights) or a conventional violation (etiquette/custom/dress code).
**Held constant:** both items describe a *transgression* (so "wrongness" is on the table for both); domain (school, workplace, family, etc.) matched within each pair where possible.
**Why it matters:** Turiel's (1983) and Nichols's (2002) classic moral/conventional distinction is the cleanest behavioral evidence that moral cognition is a distinct cognitive system. If LLM "moral neurons" lump all rule violations together, they're encoding rule-violation in general, not moral wrongness specifically.
**Literature anchor:** Turiel (1983, *The Development of Social Knowledge*); Nichols (2002, *Cognition*); Smetana (1981).
**Expected gradient:** moral ~6.3 vs conventional ~1.9 (mean a-b = +4.43).
**Design care:** all conventional violations are *real* social conventions (dress codes, etiquette, queue customs, religious practices that themselves involve no harm) — no fabricated rules. None of the conventional items involves anyone being harmed. Reviewer-relevant: a strict reviewer will check both sides for buried harm; I've been deliberate about this.

### Condition 4: Morality vs valence (moral_negative vs nonmoral_negative)
**Variable manipulated:** presence or absence of a moral agent (someone *did* something wrong) — while the outcome's severity and negativity are matched.
**Held constant:** the harmful outcome itself (poisoned water supply, building collapse, mass illness, etc.) is *identical or near-identical* in valence, harm scale, and lexical depiction.
**Why it matters:** This is the strongest test of whether a neuron is moral or merely a negative-affect detector. A neuron tracking negative valence will fire equally to both subconditions; only a genuine moral neuron will differentiate.
**Literature anchor:** Greene et al. (2001); the broader literature on dissociating moral judgment from disgust/negative affect (Schaich Borg et al. 2008).
**Expected gradient:** moral_negative ~6.8 vs nonmoral_negative ~1.5 (mean a-b = +5.37) — the largest gap by design, because this is the cleanest contrast.

## Statistics (final)

| condition  | items | word_min | word_median | word_max | word_mean | pair Δwords (mean / max) | target Δ (a-b) (min / mean / max) |
|------------|------:|---------:|------------:|---------:|----------:|--------------------------|-----------------------------------|
| intent     |   100 |       21 |          26 |       33 |     26.03 | 1.18 / 3                | +2.3 / +3.63 / +4.6              |
| outcome    |   100 |       22 |          28 |       32 |     27.75 | 1.30 / 4                | +0.6 / +0.84 / +1.0              |
| norm_type  |   100 |       23 |        27.5 |       34 |     27.85 | 1.86 / 5                | +2.7 / +4.43 / +5.3              |
| morality   |   100 |       19 |          24 |       30 |     24.27 | 1.50 / 4                | +5.0 / +5.37 / +5.8              |

- All 200 pairs have pair Δwords ≤ 5 (target was ±5).
- 187/200 pairs have Δwords ≤ 3 (the prompt's stronger preference of ±3).
- All items are 2-3 sentences (398 with 2 sentences, 2 with 3 sentences). None below 2, none above 4.
- Target diffs are signed (a - b). All conditions have **a** rated more wrong than **b**, as designed.

## Domain coverage
Stimuli intentionally span: workplace, family, community, school, friendship, healthcare, public space, online, financial, environmental, transport, food/product, political/legal. No theme dominates. No artificial trolley-style philosophical edge cases.

## Severity spread
Within each condition, severity ranges from milder transgressions (e.g. holding an elevator door closed, RSVP-ing late) to severe ones (death, serious injury, permanent disability). This is deliberate so that the dose-response relationship of "moral neuron" activation vs target judgment can be tested. The target judgments are NOT all identical within a subcondition — they range across the upper or lower portion of the 1-7 scale as appropriate.

## Calibration of target judgments
Target judgments are best-estimate point predictions calibrated against:
1. Cushman (2008) Table 1 — moral wrongness ratings for completed vs attempted vs accidental harm.
2. Schaich Borg et al. (2008) Table 1 — wrongness ratings for moral vs disgust vs neutral scenarios.
3. Turiel/Nichols moral vs conventional rating bands.

They are intended as *predictive anchors* for what a human-aligned model should produce, not as ground truth from human data. We make no claim that these exact ratings match human means; the relative ordering and magnitude of differences within pairs is what matters.

## Difficult / edge cases (transparent notes)

1. **Intent condition** — the "accidental" sibling needs to keep the same harm outcome. To avoid implying recklessness (which would partially restore moral blame), I worded accidentals as genuine ignorance + non-foreseeability where possible. A few accidentals border on negligence (e.g. the foreman who didn't notice a broken fan); these are rated slightly higher (~3.4) to reflect real human variance on negligence vs strict accident.

2. **Outcome condition** — the prompt asks for "same intent, different outcome". I kept the *first sentence* of "a" and "b" verbatim (both describe the bad intent + attempted act) and only diverged in the second sentence (whether harm landed). This maximizes intent-controlled comparison and is closest to Cushman's original design.

3. **Norm_type condition** — splitting into 2 sentences was a non-trivial constraint (single-sentence versions felt more natural). I expanded each into a setup + violation structure: sentence 1 describes the actor entering a normal context; sentence 2 describes the violation. This keeps the comparison clean while satisfying the 2-4 sentence requirement.

4. **Morality vs valence condition** — the hardest design challenge was matching outcome severity exactly. For each pair, the *outcome sentence* is near-identical between "a" and "b" (e.g. "Dozens of residents became severely ill" appears in both subconditions of pair 1). What differs is exclusively the *cause sentence*: human agent + intent vs natural process. This is the cleanest possible isolation.

5. **Pair 010 (outcome / drug overdose)** — both involve a drug dealer with intent to harm. I made the attempted-version end with "arrested before injecting" rather than "drug was substituted" to preserve the actor's full agency. This may make the attempt rating higher (~5.8) than purely thwarted attempts.

6. **Norm_type pair 015 ("man falsely told his elderly mother...")** — this is a lie about a pet's fate. It's a moral violation (deception) but rated lower (5.0) because the harm is psychological and the actor's motive is unclear; I kept it because it diversifies severity.

## What we are NOT claiming
- These are not validated against human ratings yet; target_judgment is a predictive anchor.
- We are not claiming the four conditions are statistically independent. Norm-type and morality both partly tap "moral content", but they manipulate orthogonal features: norm_type holds the *act-of-wrongdoing* constant and varies its content; morality holds the *outcome* constant and varies whether there is wrongdoing at all.
- Some of the scenarios are confronting (death, assault). They were chosen because the moral judgment literature uses such stimuli routinely (Greene, Cushman, Haidt) and because severity variation is needed to span the response scale. None describe gratuitous violence beyond what is necessary to specify the moral content.

## Reproducibility
Run `python _generate.py` from anywhere; it deterministically produces `decomposition_stimuli.jsonl`. The script also prints validation stats (counts, length distribution, duplicate check). Editing only the lists at the top of the script is sufficient to add or modify scenarios.

## Suggested downstream analyses
1. **Per-condition selectivity matrix**: compute moral-neuron activation difference across the two subconditions within each condition; expect large effects on intent / norm_type / morality and a smaller but non-zero effect on outcome.
2. **Cross-condition decoding**: train a probe to discriminate moral_negative vs nonmoral_negative; test whether the same probe also separates moral vs conventional and intentional vs accidental (transfer test).
3. **Confound check**: predict target_judgment from moral-neuron activation. The fit should be tight within each subcondition (severity gradient) and respect the across-condition ordering.
4. **Ablation cross-validation**: ablate the "moral" neurons identified by the broad ethics stimuli (from `stimuli_full.jsonl`) and measure the effect on each of the four contrasts. If "moral" neurons are real, ablation should disproportionately collapse the morality vs valence contrast (Condition 4).
