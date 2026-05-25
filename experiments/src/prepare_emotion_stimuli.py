"""Prepare emotion stimuli for the LLM functional atlas (emotion module).

Produces four artifacts under
``experiments/data/cognitive_stimuli/emotion/``:

1. ``warriner_vad.csv``           -- 13,915 English lemmas with V/A/D means (Warriner et al. 2013)
2. ``goemotions_sample.jsonl``    -- balanced 200/emotion sample from GoEmotions (Demszky et al. 2020)
3. ``emotion_localizer_stimuli.jsonl`` -- 50 sentences x 6 Ekman emotions + 50 neutral = 350 stimuli
4. ``vad_graded_sentences.jsonl`` -- 200 sentences with continuous valence/arousal targets

The script assumes the raw sources have been pre-downloaded into the same
directory (see ``README.md``):

* ``warriner_raw.csv`` -- the JULIELab/XANEW mirror of the Warriner spreadsheet
* ``goemotions_{train,dev,test}.tsv`` -- the Google-Research GoEmotions release
* ``goemotions_labels.txt``, ``goemotions_ekman.json``, ``goemotions_sentiment.json``

Run:

    python -m experiments.src.prepare_emotion_stimuli

(no GPU needed; pure CSV/JSONL processing).
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
EMO_DIR = REPO / "experiments" / "data" / "cognitive_stimuli" / "emotion"
EMO_DIR.mkdir(parents=True, exist_ok=True)

WARRINER_RAW = EMO_DIR / "warriner_raw.csv"
WARRINER_OUT = EMO_DIR / "warriner_vad.csv"

GE_TRAIN = EMO_DIR / "goemotions_train.tsv"
GE_DEV = EMO_DIR / "goemotions_dev.tsv"
GE_TEST = EMO_DIR / "goemotions_test.tsv"
GE_LABELS = EMO_DIR / "goemotions_labels.txt"
GE_EKMAN = EMO_DIR / "goemotions_ekman.json"
GE_SENT = EMO_DIR / "goemotions_sentiment.json"

GOEMO_SAMPLE = EMO_DIR / "goemotions_sample.jsonl"
LOCALIZER_OUT = EMO_DIR / "emotion_localizer_stimuli.jsonl"
VAD_GRADED_OUT = EMO_DIR / "vad_graded_sentences.jsonl"

SEED = 20260525  # today's date, deterministic
PER_EMOTION_SAMPLE = 200
LOCALIZER_PER_EMOTION = 50
LOCALIZER_NEUTRAL = 50
VAD_GRADED_TOTAL = 200


# ---------------------------------------------------------------------------
# 1. Warriner VAD
# ---------------------------------------------------------------------------

def build_warriner() -> None:
    """Reduce the raw 65-column Warriner table to (word, valence, arousal, dominance)."""
    if not WARRINER_RAW.exists():
        raise FileNotFoundError(f"Missing {WARRINER_RAW}; see README for download steps.")

    df = pd.read_csv(WARRINER_RAW)
    # JULIELab mirror keeps the original column names:
    # Word, V.Mean.Sum, A.Mean.Sum, D.Mean.Sum
    keep = df[["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]].copy()
    keep.columns = ["word", "valence", "arousal", "dominance"]
    keep["word"] = keep["word"].astype(str).str.strip().str.lower()
    keep = keep.dropna(subset=["word", "valence", "arousal", "dominance"])
    # Some entries may duplicate after lowercasing -- keep the first.
    keep = keep.drop_duplicates(subset=["word"], keep="first")
    keep = keep.sort_values("word").reset_index(drop=True)
    keep.to_csv(WARRINER_OUT, index=False)
    print(f"[warriner] wrote {len(keep)} lemmas -> {WARRINER_OUT}")


# ---------------------------------------------------------------------------
# 2. GoEmotions parsing
# ---------------------------------------------------------------------------

def load_goemotions_labels() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Return (label list, idx->sentiment(+1/-1/0), idx->Ekman category)."""
    labels = [ln.strip() for ln in GE_LABELS.read_text().splitlines() if ln.strip()]

    sent_map = json.loads(GE_SENT.read_text())
    sentiment_of: Dict[str, str] = {}
    for k in sent_map["positive"]:
        sentiment_of[k] = "positive"
    for k in sent_map["negative"]:
        sentiment_of[k] = "negative"
    for k in sent_map["ambiguous"]:
        sentiment_of[k] = "ambiguous"
    sentiment_of["neutral"] = "neutral"

    ekman_map = json.loads(GE_EKMAN.read_text())
    ekman_of: Dict[str, str] = {}
    for ekman_cat, members in ekman_map.items():
        for m in members:
            ekman_of[m] = ekman_cat
    ekman_of["neutral"] = "neutral"

    return labels, sentiment_of, ekman_of


def iter_goemotions(path: Path) -> Iterable[Tuple[str, List[int], str]]:
    """Yield (text, label_indices, comment_id) for every example in a GoEmotions TSV."""
    with path.open() as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) != 3:
                continue
            text, label_str, comment_id = row
            label_ids = [int(x) for x in label_str.split(",") if x != ""]
            yield text, label_ids, comment_id


def sentiment_signum(label_names: List[str], sentiment_of: Dict[str, str]) -> int:
    """Map a set of emotion labels to +1 / -1 / 0 sentiment.

    Rules:
    * neutral-only -> 0
    * if all non-neutral labels are positive -> +1
    * if all non-neutral labels are negative -> -1
    * mixed or ambiguous -> 0
    """
    non_neutral = [lbl for lbl in label_names if lbl != "neutral"]
    if not non_neutral:
        return 0
    sentiments = {sentiment_of.get(lbl, "ambiguous") for lbl in non_neutral}
    if sentiments == {"positive"}:
        return 1
    if sentiments == {"negative"}:
        return -1
    return 0


def build_goemotions_sample() -> None:
    """Sample up to ``PER_EMOTION_SAMPLE`` examples per emotion (balanced)."""
    labels, sentiment_of, _ = load_goemotions_labels()
    rng = random.Random(SEED)

    # Collect every (text, [label_names], id, valence) once.
    bucket: Dict[str, List[dict]] = defaultdict(list)
    seen_ids: set = set()
    for split_path in (GE_TRAIN, GE_DEV, GE_TEST):
        for text, label_ids, comment_id in iter_goemotions(split_path):
            if comment_id in seen_ids:
                continue
            seen_ids.add(comment_id)
            label_names = [labels[i] for i in label_ids]
            entry = {
                "id": comment_id,
                "text": text,
                "emotion_labels": label_names,
                "valence": sentiment_signum(label_names, sentiment_of),
                "is_neutral": label_names == ["neutral"],
            }
            for lbl in label_names:
                bucket[lbl].append(entry)

    rng_state = random.Random(SEED)
    sampled: List[dict] = []
    seen = set()
    for emotion in labels:
        pool = bucket.get(emotion, [])
        rng_state.shuffle(pool)
        taken = 0
        for ex in pool:
            if ex["id"] in seen:
                continue
            seen.add(ex["id"])
            sampled.append(ex)
            taken += 1
            if taken == PER_EMOTION_SAMPLE:
                break
        print(f"[goemotions] {emotion:>15s}: pool={len(pool):5d}, sampled={taken}")

    rng_state.shuffle(sampled)
    with GOEMO_SAMPLE.open("w") as fh:
        for ex in sampled:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[goemotions] wrote {len(sampled)} rows -> {GOEMO_SAMPLE}")


# ---------------------------------------------------------------------------
# 3. Localizer: 50 sentences x 6 Ekman emotions + 50 neutral
# ---------------------------------------------------------------------------

# Curated high-clarity sentences for each Ekman emotion.  Each list has >=15
# items so that the GoEmotions pool only needs to contribute ~35 examples per
# category; this guards against pool exhaustion for rarer emotions (grief,
# disgust) and keeps the localizer stylistically diverse.

CURATED: Dict[str, List[Tuple[str, int]]] = {
    "anger": [
        ("How dare you speak to me like that.", 5),
        ("I am absolutely furious about what they did.", 5),
        ("This is completely unacceptable and I will not tolerate it.", 5),
        ("I cannot believe how badly they treated us.", 4),
        ("Stop interrupting me right now.", 4),
        ("That kind of behavior makes my blood boil.", 5),
        ("I am sick of being lied to by these people.", 4),
        ("Get out of my sight before I lose my temper.", 5),
        ("Their selfishness is enraging beyond words.", 5),
        ("I have had enough of their excuses.", 4),
        ("You broke your promise and I am livid.", 5),
        ("The way they ignored us was infuriating.", 4),
        ("I refuse to be treated like this any longer.", 4),
        ("It makes me want to scream when I hear that.", 4),
        ("Their dishonesty is utterly outrageous.", 5),
    ],
    "disgust": [
        ("The smell coming from the trash made me gag.", 5),
        ("I cannot believe people would eat something so revolting.", 5),
        ("Watching him chew with his mouth open is repulsive.", 4),
        ("The mold inside the fridge was sickening.", 5),
        ("That story about the bug in his soup turned my stomach.", 5),
        ("Their dishonest behavior is morally disgusting.", 4),
        ("The bathroom was so filthy I refused to enter.", 5),
        ("Spitting on the sidewalk is absolutely vile.", 4),
        ("I felt nauseous after smelling the spoiled milk.", 5),
        ("Cockroaches in the kitchen are utterly repulsive.", 5),
        ("The way he licked his fingers in public was gross.", 4),
        ("That decaying carcass on the road made me retch.", 5),
        ("Their cruelty toward the animals was sickening.", 5),
        ("The dish smelled so foul I had to push it away.", 5),
        ("I cannot stand the slimy texture of raw oysters.", 4),
    ],
    "fear": [
        ("I heard footsteps behind me in the empty parking lot.", 4),
        ("My heart pounded as the shadow approached the door.", 5),
        ("I am terrified that something terrible will happen tonight.", 5),
        ("The dark hallway made me freeze in my tracks.", 4),
        ("Every creak in the old house made my skin crawl.", 4),
        ("I am scared to walk home alone after midnight.", 4),
        ("The thought of public speaking makes my hands shake.", 4),
        ("I jumped when the door slammed shut by itself.", 4),
        ("My stomach dropped when I saw the unknown number calling.", 4),
        ("The howling wind outside frightened the children.", 4),
        ("I dread getting the test results back tomorrow.", 4),
        ("Watching the news about the outbreak terrifies me.", 5),
        ("A wave of panic washed over me in the crowded subway.", 5),
        ("My pulse raced as the turbulence shook the plane.", 5),
        ("I was paralyzed with fear when I saw the snake.", 5),
    ],
    "joy": [
        ("I just found out I got into my dream school.", 5),
        ("Holding my newborn for the first time was pure bliss.", 5),
        ("We laughed so hard our sides hurt all evening.", 5),
        ("The surprise party brought tears of happiness to my eyes.", 5),
        ("I cannot stop smiling about the good news.", 4),
        ("Dancing in the rain with my friends was magical.", 5),
        ("Seeing my parents reunite after years made me beam.", 5),
        ("I am so proud of how far our team has come.", 4),
        ("That warm sunshine on my face felt absolutely wonderful.", 4),
        ("Hearing my best friend say yes filled me with joy.", 5),
        ("The little girl squealed with delight at the puppy.", 5),
        ("My heart soared when the offer letter arrived.", 5),
        ("It was the most joyful wedding I have ever attended.", 5),
        ("I jumped for joy when the package finally arrived.", 4),
        ("Reading their thank-you note made my whole day brighter.", 4),
    ],
    "sadness": [
        ("I miss my grandmother every single day since she passed.", 5),
        ("Watching the empty chair at dinner broke my heart.", 5),
        ("I cried myself to sleep after reading the letter.", 5),
        ("Nothing feels the same since my dog died.", 5),
        ("The funeral left everyone in quiet, heavy silence.", 5),
        ("I sat alone in the rain, feeling utterly forgotten.", 4),
        ("Hearing the goodbye made tears slide down her cheeks.", 4),
        ("He has not smiled once since the accident.", 4),
        ("The diagnosis left the whole family in tears.", 5),
        ("It is so painful to walk past her old room.", 4),
        ("I have been crying off and on all afternoon.", 4),
        ("Losing the championship after years of work was crushing.", 4),
        ("My chest ached as I read the farewell note.", 4),
        ("I felt hollow when I learned they had moved away.", 4),
        ("She wept quietly at the edge of the grave.", 5),
    ],
    "surprise": [
        ("I cannot believe you remembered my birthday after all these years.", 4),
        ("Wait, you mean to tell me he was the suspect all along?", 5),
        ("Out of nowhere, my long-lost cousin appeared at the door.", 5),
        ("Whoa, the building was twice as tall as I expected.", 4),
        ("I never imagined the tiny shop would have that book.", 3),
        ("She opened the box and gasped in astonishment.", 4),
        ("The plot twist completely caught me off guard.", 4),
        ("I had no idea you spoke five languages fluently.", 4),
        ("It blew my mind when the results came back positive.", 4),
        ("Suddenly the lights came on and everyone shouted surprise.", 5),
        ("To my amazement, the old engine started on the first try.", 4),
        ("I was stunned that they chose me out of hundreds.", 5),
        ("Hold on, when did you learn to play the violin?", 4),
        ("The puppy popped out of the gift bag and I shrieked.", 4),
        ("I cannot believe how much you have grown since last summer.", 3),
    ],
}

NEUTRAL_CURATED: List[str] = [
    "The library closes at nine on weekdays and at six on weekends.",
    "The recipe calls for two cups of flour and one teaspoon of salt.",
    "He filled out the form and handed it back to the receptionist.",
    "Trains depart from platform four every fifteen minutes.",
    "The textbook covers vector spaces in the third chapter.",
    "She placed the documents into the manila folder.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Add the tomatoes to the saucepan and simmer for ten minutes.",
    "The shipment is expected to arrive on Thursday morning.",
    "Most apartments in the building have two bedrooms.",
    "The committee meeting is scheduled for next Tuesday at three.",
    "Please staple the pages together before submitting them.",
    "The library has copies of the report on the second floor.",
    "He took the elevator up to the seventh floor.",
    "The package weighs about two kilograms.",
    "Office hours are listed on the syllabus.",
    "The new policy takes effect at the beginning of next month.",
    "She walked to the bus stop and waited for the next bus.",
    "The thermostat is set to twenty-one degrees Celsius.",
    "Please remember to recycle paper in the blue bin.",
]


def collect_goemotions_for_localizer(
    rng: random.Random,
    ekman_of: Dict[str, str],
    needed_per_emotion: int = 35,
    neutral_needed: int = 30,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """Return single-label, high-confidence GoEmotions examples per Ekman class + neutral pool."""
    labels, _, _ = load_goemotions_labels()
    per_ekman: Dict[str, List[dict]] = defaultdict(list)
    neutrals: List[dict] = []
    seen_ids: set = set()
    seen_text: set = set()

    for split_path in (GE_TRAIN, GE_DEV, GE_TEST):
        for text, label_ids, comment_id in iter_goemotions(split_path):
            if comment_id in seen_ids:
                continue
            seen_ids.add(comment_id)
            # Quality filter: keep tractable Reddit comments only.
            stripped = text.strip()
            if len(stripped) < 15 or len(stripped) > 200:
                continue
            if "[NAME]" in stripped or "[RELIGION]" in stripped:
                continue
            if stripped.lower() in seen_text:
                continue
            seen_text.add(stripped.lower())
            label_names = [labels[i] for i in label_ids]
            if len(label_names) != 1:
                continue  # single-label = highest annotator agreement proxy
            lbl = label_names[0]
            entry = {
                "id": comment_id,
                "text": stripped,
                "emotion_labels": label_names,
                "source": "goemotions",
            }
            if lbl == "neutral":
                neutrals.append(entry)
                continue
            ek = ekman_of.get(lbl)
            if ek is None or ek == "neutral":
                continue
            per_ekman[ek].append(entry)

    rng.shuffle(neutrals)
    for ek in list(per_ekman):
        rng.shuffle(per_ekman[ek])
        per_ekman[ek] = per_ekman[ek][: needed_per_emotion * 3]  # over-sample for selection later

    return per_ekman, neutrals[: neutral_needed * 3]


def build_localizer() -> None:
    rng = random.Random(SEED + 1)
    _, _, ekman_of = load_goemotions_labels()
    ge_pool, ge_neutral = collect_goemotions_for_localizer(rng, ekman_of)

    out: List[dict] = []
    idx = 0
    ekman_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    for emotion in ekman_emotions:
        curated = CURATED[emotion]
        rng.shuffle(curated)
        per_emo: List[dict] = []
        # Curated entries first (they are hand-validated high-confidence).
        for sent, intensity in curated:
            per_emo.append(
                {
                    "id": f"emo_loc_{idx:04d}",
                    "text": sent,
                    "emotion": emotion,
                    "intensity": intensity,
                    "source": "curated",
                }
            )
            idx += 1
            if len(per_emo) == LOCALIZER_PER_EMOTION:
                break
        # Fill with GoEmotions single-label items (intensity heuristic = 3).
        ge_iter = iter(ge_pool.get(emotion, []))
        while len(per_emo) < LOCALIZER_PER_EMOTION:
            try:
                ex = next(ge_iter)
            except StopIteration:
                break
            per_emo.append(
                {
                    "id": f"emo_loc_{idx:04d}",
                    "text": ex["text"],
                    "emotion": emotion,
                    "intensity": 3,
                    "source": "goemotions",
                    "goemotions_label": ex["emotion_labels"][0],
                }
            )
            idx += 1
        out.extend(per_emo)
        print(f"[localizer] {emotion:>10s}: {len(per_emo)} sentences "
              f"(curated={sum(1 for x in per_emo if x['source']=='curated')}, "
              f"goemotions={sum(1 for x in per_emo if x['source']=='goemotions')})")

    # Neutral pool: curated + GoEmotions single-label neutral.
    neutral_entries: List[dict] = []
    for sent in NEUTRAL_CURATED:
        neutral_entries.append(
            {
                "id": f"emo_loc_{idx:04d}",
                "text": sent,
                "emotion": "neutral",
                "intensity": 1,
                "source": "curated",
            }
        )
        idx += 1
        if len(neutral_entries) == LOCALIZER_NEUTRAL:
            break
    for ex in ge_neutral:
        if len(neutral_entries) >= LOCALIZER_NEUTRAL:
            break
        neutral_entries.append(
            {
                "id": f"emo_loc_{idx:04d}",
                "text": ex["text"],
                "emotion": "neutral",
                "intensity": 1,
                "source": "goemotions",
                "goemotions_label": "neutral",
            }
        )
        idx += 1
    out.extend(neutral_entries)
    print(f"[localizer]    neutral: {len(neutral_entries)} sentences "
          f"(curated={sum(1 for x in neutral_entries if x['source']=='curated')}, "
          f"goemotions={sum(1 for x in neutral_entries if x['source']=='goemotions')})")

    with LOCALIZER_OUT.open("w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[localizer] wrote {len(out)} sentences -> {LOCALIZER_OUT}")


# ---------------------------------------------------------------------------
# 4. VAD-graded sentences (200 items, continuous valence x arousal)
# ---------------------------------------------------------------------------

# Each entry: (text, target_valence on 1-9, target_arousal on 1-9).
# Anchors taken to span the V x A plane: 5 valence bands (1-2, 3-4, 5, 6-7, 8-9)
# crossed with 4 arousal bands (1-2, 3-4, 5-6, 7-9), with ~10 sentences per cell.

# The list is hand-curated so it can be cited as "the authors' stimuli" without
# claiming Warriner-level normative ratings.  Targets are best-effort point
# estimates that future raters can refine; the structure (V x A coverage) is
# the load-bearing property.

VAD_GRADED: List[Tuple[str, float, float]] = [
    # ----- very negative valence (1-2) x very low arousal (1-2) -----
    ("Everything feels grey and hollow this morning.", 2.0, 2.0),
    ("Nothing seems worth doing anymore.", 1.8, 2.0),
    ("I sat alone in the dim apartment for hours.", 2.2, 1.8),
    ("Time drags on with no purpose at all.", 2.0, 2.0),
    ("The rain has soaked everything in dull sorrow.", 2.2, 2.2),
    # ----- very negative x moderate arousal (3-4) -----
    ("My friend has been ignoring me all week.", 2.2, 3.5),
    ("The doctor's tone made me deeply uneasy.", 2.0, 4.0),
    ("I keep replaying the argument in my head.", 2.2, 3.8),
    ("Reading the bad review left a bitter aftertaste.", 2.5, 3.5),
    ("Their disappointment in me is hard to bear.", 2.0, 3.8),
    # ----- very negative x mid arousal (5-6) -----
    ("I cannot stop crying about what happened.", 1.8, 5.5),
    ("My stomach churned as the rejection sank in.", 2.0, 5.5),
    ("Every reminder of him makes my chest ache.", 2.2, 5.5),
    ("She broke down sobbing at the front desk.", 1.8, 6.0),
    ("Reading the diagnosis left me trembling.", 1.8, 6.0),
    # ----- very negative x high arousal (7-9) -----
    ("A masked figure stepped suddenly out of the alley.", 1.8, 8.5),
    ("The plane plunged through violent turbulence.", 1.8, 8.0),
    ("I screamed as the car spun off the road.", 1.5, 9.0),
    ("Flames raced up the staircase faster than we could run.", 1.5, 9.0),
    ("They forced open the door and rushed inside shouting.", 2.0, 8.5),
    # ----- mildly negative (3-4) x low arousal (1-2) -----
    ("The waiting room smelled faintly of antiseptic.", 3.5, 2.0),
    ("My coffee went cold while I worked.", 4.0, 1.8),
    ("The lecture dragged on past the closing bell.", 3.5, 2.0),
    ("Another rainy commute, another delay.", 3.5, 2.0),
    ("My phone battery died right before the alarm.", 3.5, 2.5),
    # ----- mildly negative x moderate arousal (3-4) -----
    ("My boss flagged three mistakes in my report.", 3.5, 4.0),
    ("I missed the last train by less than a minute.", 3.5, 4.5),
    ("The dentist warned me about a cavity forming.", 3.5, 4.0),
    ("She rolled her eyes when I told the story.", 3.5, 4.0),
    ("My password expired again at the worst moment.", 3.8, 3.8),
    # ----- mildly negative x mid arousal (5-6) -----
    ("I argued with my brother on the phone tonight.", 3.5, 5.5),
    ("The presentation crashed in front of everyone.", 3.2, 6.0),
    ("My bike got stolen from outside the library.", 3.0, 5.5),
    ("They left without saying goodbye and it stung.", 3.2, 5.5),
    ("The deadline keeps slipping and tempers are rising.", 3.5, 5.8),
    # ----- mildly negative x high arousal (7-9) -----
    ("A wasp got trapped inside my helmet on the highway.", 3.0, 8.0),
    ("My laptop fell from the table during the meeting.", 3.2, 7.5),
    ("The dog charged at me growling without warning.", 2.8, 8.0),
    ("I almost dropped the antique vase down the stairs.", 3.0, 7.5),
    ("Sparks flew from the outlet and the lights cut out.", 2.8, 8.0),
    # ----- neutral (around 5) x low arousal (1-2) -----
    ("The book is on the second shelf to the left.", 5.0, 1.8),
    ("Office hours are listed on the syllabus.", 5.0, 1.5),
    ("Buses run every twenty minutes on weekdays.", 5.0, 1.8),
    ("The recipe yields six servings.", 5.0, 1.8),
    ("Please staple the pages before turning them in.", 5.0, 2.0),
    ("The trains depart from platform four.", 5.0, 1.8),
    ("Most apartments have two bedrooms and a balcony.", 5.0, 2.0),
    ("Water boils at one hundred degrees Celsius.", 5.0, 1.8),
    ("Add the tomatoes and simmer for ten minutes.", 5.0, 2.0),
    ("The package weighs about two kilograms.", 5.0, 1.8),
    # ----- neutral x moderate arousal (3-4) -----
    ("The supervisor reviewed the budget line by line.", 5.0, 3.5),
    ("I parallel-parked between two delivery trucks.", 5.0, 4.0),
    ("She filled out the customs form on the plane.", 5.0, 3.5),
    ("He kept glancing at the clock during the meeting.", 5.0, 3.8),
    ("The technician calibrated each sensor in turn.", 5.0, 3.5),
    # ----- neutral x mid arousal (5-6) -----
    ("The traffic light turned yellow just as we approached.", 5.0, 5.5),
    ("He answered the cold call but kept his tone formal.", 5.0, 5.2),
    ("The buzzer rang twice before anyone reacted.", 5.0, 5.5),
    ("A debate broke out about the new policy.", 5.0, 5.8),
    ("The referee waved the players back onto the field.", 5.0, 5.5),
    # ----- neutral x high arousal (7-9) -----
    ("The fire drill alarm blared through the corridors.", 5.0, 7.5),
    ("A motorcade of sirens raced past the office window.", 5.0, 7.5),
    ("The earthquake drill required everyone under desks within seconds.", 5.0, 7.5),
    ("All hands worked through the night to relaunch the server.", 5.0, 7.2),
    ("The pilot announced an emergency landing procedure.", 5.0, 7.8),
    # ----- mildly positive (6-7) x low arousal (1-2) -----
    ("Reading by the window with a warm tea felt nice.", 6.5, 2.0),
    ("Sunlight slanted softly across the kitchen floor.", 6.5, 2.0),
    ("The cat curled up on my lap and purred.", 7.0, 2.2),
    ("I watched the snow fall in quiet, fat flakes.", 6.5, 2.0),
    ("The hot bath melted the day off my shoulders.", 6.8, 2.0),
    # ----- mildly positive x moderate arousal (3-4) -----
    ("A small bonus showed up in my paycheck today.", 6.5, 4.0),
    ("My garden basil finally sprouted its first leaves.", 6.8, 3.5),
    ("He waved hello from across the cafe with a grin.", 6.5, 3.8),
    ("The recipe came out better than I expected.", 6.8, 3.8),
    ("I finished the crossword puzzle on the first try.", 6.5, 4.0),
    # ----- mildly positive x mid arousal (5-6) -----
    ("The community garden harvest came in beautifully today.", 6.5, 5.0),
    ("She got a callback after the audition tonight.", 7.0, 5.5),
    ("Their first podcast episode broke ten thousand listeners.", 6.8, 5.5),
    ("The team rallied and tied the game in the last quarter.", 6.8, 6.0),
    ("My short story made the longlist for the prize.", 7.0, 5.8),
    # ----- mildly positive x high arousal (7-9) -----
    ("We zipped through the canyon on mountain bikes.", 7.0, 7.5),
    ("The crowd cheered as the underdog scored a goal.", 7.0, 7.5),
    ("I leapt across the rocks at the waterfall.", 6.8, 7.5),
    ("The roller coaster threw us into bright laughter.", 7.0, 7.8),
    ("He sprinted the last hundred meters and pumped his fists.", 7.0, 8.0),
    # ----- very positive (8-9) x low arousal (1-2) -----
    ("Watching my baby sleep is the most peaceful feeling.", 8.5, 2.0),
    ("Grandma's old afghan still smells like her perfume.", 8.5, 2.0),
    ("Holding my dog at the end of a long day is bliss.", 8.5, 2.2),
    ("A quiet anniversary dinner with my partner is enough.", 8.5, 2.2),
    ("Listening to the rain together felt deeply tender.", 8.5, 2.0),
    # ----- very positive x moderate arousal (3-4) -----
    ("I just heard back that I got the dream apartment.", 8.5, 4.0),
    ("They named the new baby after my late grandmother.", 8.5, 4.0),
    ("My best friend is finally moving back to town.", 8.5, 4.2),
    ("The scan came back clean and we both cried with relief.", 8.5, 4.2),
    ("My poem will be published in the autumn issue.", 8.5, 4.0),
    # ----- very positive x mid arousal (5-6) -----
    ("I finally got the acceptance email from the program.", 8.5, 5.5),
    ("Our team won the regional finals after a year of work.", 8.5, 6.0),
    ("She said yes and the whole restaurant clapped.", 9.0, 6.0),
    ("The clinical trial worked, the tumour is shrinking.", 8.5, 6.0),
    ("After ten years apart, my brother walked through the door.", 9.0, 6.0),
    # ----- very positive x high arousal (7-9) -----
    ("Confetti rained down as we crossed the marathon finish line.", 8.5, 8.5),
    ("Fireworks burst overhead as the trophy was lifted.", 8.5, 8.5),
    ("My whole family jumped and screamed when the test came back positive.", 8.8, 8.5),
    ("The crowd erupted when the last out was called.", 8.5, 8.5),
    ("We surfed the wave together, laughing the whole way.", 8.5, 8.0),
    # ----- additional fillers to reach 200 -----
    # Slightly more granular neutral & off-axis points.
    ("She refilled her water bottle at the office fountain.", 5.0, 2.0),
    ("He locked his bike to the rack outside the gym.", 5.0, 2.2),
    ("The committee approved the routine motion unanimously.", 5.2, 3.0),
    ("I confirmed my appointment on the patient portal.", 5.0, 2.5),
    ("The printer jammed twice but eventually finished.", 4.5, 3.5),
    ("My new prescription glasses arrived in the mail.", 6.0, 3.0),
    ("My dog rolled in something foul in the park.", 3.0, 4.5),
    ("The pasta water boiled over onto the stove.", 4.0, 4.0),
    ("My niece read her first book to me out loud.", 8.0, 4.5),
    ("We finally repaired the leaky kitchen faucet.", 6.0, 3.0),
    ("A rude customer kept yelling at the cashier.", 2.5, 6.5),
    ("My laptop fan started shrieking during the call.", 3.5, 5.0),
    ("The neighbor's stereo blasted past midnight again.", 3.0, 5.5),
    ("Our flight was delayed three hours on the tarmac.", 3.0, 5.0),
    ("My favorite playlist shuffled the perfect song.", 7.0, 4.0),
    ("The barista remembered my regular order today.", 6.5, 3.5),
    ("I aced the quiz I had been dreading all week.", 7.5, 5.5),
    ("The startup pitch was cut short and they looked crushed.", 3.0, 5.0),
    ("My grant proposal was rejected without comments.", 2.5, 5.0),
    ("The hike took longer than expected but we made it.", 6.0, 4.5),
    ("A pigeon flew straight into our window with a thud.", 4.0, 6.0),
    ("Lightning struck the tree just outside the window.", 3.5, 8.0),
    ("The smoke detector kept chirping at three in the morning.", 3.0, 5.5),
    ("Their wedding vows had everyone reaching for tissues.", 8.5, 5.5),
    ("Reading bedtime stories to my son makes me melt.", 8.0, 3.5),
    ("The toddler giggled uncontrollably at the puppet show.", 8.0, 5.5),
    ("My therapist's words finally clicked for me today.", 7.5, 4.5),
    ("The exam proctor announced ten minutes remaining.", 5.0, 5.5),
    ("The intern fumbled the introduction in front of the CEO.", 3.5, 6.0),
    ("A skunk wandered into the campground at dusk.", 3.5, 5.5),
    ("My headphones finally arrived after a month of waiting.", 6.5, 3.5),
    ("The cat knocked my favorite mug off the shelf.", 3.5, 5.0),
    ("Our community raised enough money for the new playground.", 8.0, 5.0),
    ("The volunteer crew cleaned the riverbank all weekend.", 7.0, 3.5),
    ("Power went out across the whole block at midnight.", 4.0, 6.0),
    ("My homemade bread came out airy and golden.", 7.5, 3.5),
    ("The toast burned and the kitchen filled with smoke.", 3.5, 5.5),
    ("My old college roommate sent a surprise birthday card.", 7.5, 3.5),
    ("The choir hit a perfect harmony on the final note.", 8.0, 5.5),
    ("The principal called us into the office unexpectedly.", 4.0, 6.0),
    ("Their tiny puppy peed on my work bag.", 4.0, 5.0),
    ("My grandmother taught me to fold dumplings today.", 8.0, 3.5),
    ("The investigator's questions kept circling back to me.", 3.5, 6.5),
    ("A hummingbird hovered just outside the window.", 7.5, 4.0),
    ("We watched the meteor shower from the rooftop.", 8.0, 5.0),
    ("The first frost killed the tomatoes overnight.", 4.0, 3.5),
    ("My audit found three missing receipts.", 4.0, 4.0),
    ("Our toddler said her first complete sentence today.", 8.5, 5.0),
    ("The judge ruled in our favor after months of waiting.", 8.0, 5.5),
    ("My package was stolen from the porch again.", 3.0, 5.0),
    ("The orchestra tuned their instruments before the concert.", 5.5, 3.5),
    ("The marathon official handed me a finisher's medal.", 8.5, 6.0),
    ("My professor wrote 'excellent work' across the top page.", 8.0, 4.0),
    ("The interviewer kept asking trick questions about my résumé.", 4.0, 5.5),
    ("My garden gnome was knocked over by the wind.", 4.5, 3.0),
    ("Their cat finally let me pet her after months.", 7.0, 3.5),
    ("Our weekly board game night turned into a shouting match.", 4.0, 6.0),
    ("My favorite mug fell and shattered on the tiles.", 3.5, 5.0),
    ("I helped an elderly woman carry her groceries home.", 7.5, 3.0),
    ("The hotel lost our reservation on arrival.", 3.0, 5.5),
    ("Our flight got upgraded to first class for free.", 8.0, 5.5),
    ("The midnight thunderstorm shook the entire house.", 4.0, 7.0),
    ("The toddler dropped my phone in the toilet.", 3.0, 5.5),
    ("My favorite indie band played my request encore.", 8.5, 6.5),
    ("My oldest friend called out of nowhere to chat.", 7.5, 3.5),
    ("My car broke down on the freeway in rush hour.", 2.5, 6.0),
    ("The keynote speaker forgot the punchline mid-joke.", 4.5, 4.5),
    ("Our team's new logo got rave reviews from the board.", 7.5, 4.5),
    ("The conference Wi-Fi kept dropping out during my talk.", 3.5, 5.5),
    ("My pottery wheel collapsed and ruined three pieces.", 3.0, 5.5),
    ("My garden's first sunflower finally bloomed today.", 8.0, 3.5),
    ("The audience gave the cast a standing ovation.", 8.5, 6.5),
    ("My neighbor passed away peacefully in her sleep.", 3.5, 3.5),
    ("My bonsai survived its first year on the windowsill.", 7.0, 2.5),
    ("Our quiet hike was interrupted by a snake on the trail.", 4.0, 6.5),
    ("My piano recital ended in thunderous applause.", 8.5, 6.5),
    ("The librarian shushed the whole reading room.", 4.5, 3.5),
    ("My favorite cafe closed for good last week.", 3.5, 4.0),
    ("A double rainbow stretched across the valley after the storm.", 8.0, 5.0),
    ("We discovered our reservation had been double booked.", 3.5, 5.0),
    ("The neighbor's child sang a lullaby to my baby.", 7.5, 3.0),
    ("The dishwasher flooded the entire kitchen overnight.", 2.5, 6.0),
    ("My midnight snack turned into a feast with my roommate.", 7.0, 4.5),
    ("My online order arrived smashed beyond repair.", 2.5, 4.5),
    ("My team beat our quarterly goal three weeks early.", 8.0, 5.5),
    ("The waiting list email finally said we were in.", 8.0, 5.0),
    ("A wasp landed on my arm and I froze.", 4.0, 6.5),
    ("My old letters made me weep on the floor for hours.", 3.0, 5.5),
    ("My toddler took her very first independent steps.", 9.0, 6.0),
    ("My dog learned to ring the bell to go outside.", 7.5, 4.0),
    ("The lecture hall finally got the projector working again.", 5.5, 3.0),
    ("My partner remembered our first date anniversary.", 8.5, 4.5),
    ("My morning coffee tipped right into my keyboard.", 3.0, 5.0),
    ("The investor passed on our pitch after two months.", 3.0, 4.5),
    ("My hiking buddy slipped and twisted her ankle.", 3.0, 6.0),
    ("The volunteer choir broke into the chorus together.", 7.5, 5.5),
    ("The school principal praised my essay over the loudspeaker.", 8.0, 5.0),
    ("My neighbor's tree fell on our shared fence overnight.", 3.0, 5.5),
    ("My cousin's wedding speech had everyone in tears.", 8.0, 5.5),
    ("My online date never showed up to the restaurant.", 2.5, 5.5),
    ("The barista wrote a sweet note on my coffee cup.", 7.5, 3.5),
    ("My ancient laptop finally died mid-presentation.", 3.0, 6.0),
    ("Our garden harvest filled the kitchen with squash.", 7.0, 3.5),
    ("My passport vanished an hour before departure.", 2.0, 7.5),
    ("Our daughter's painting won the school art prize.", 8.5, 5.0),
    ("My cat sneezed and knocked over the water glass.", 5.5, 4.0),
    ("The doctor said the cast can come off next week.", 7.5, 4.0),
    ("My old high school crush messaged out of nowhere.", 6.5, 5.0),
    ("The intercom blared a tornado warning across the office.", 3.0, 8.0),
    ("Our anniversary dinner ended with surprise champagne from the chef.", 8.5, 5.0),
    ("The faulty alarm clock made me sleep through the interview.", 2.5, 6.5),
]


def build_vad_graded() -> None:
    out: List[dict] = []
    for i, (text, v, a) in enumerate(VAD_GRADED):
        out.append(
            {
                "id": f"vad_{i:04d}",
                "text": text,
                "target_valence": float(v),
                "target_arousal": float(a),
            }
        )
    if len(out) < VAD_GRADED_TOTAL:
        raise RuntimeError(
            f"VAD_GRADED currently has {len(out)} entries, need at least {VAD_GRADED_TOTAL}"
        )
    out = out[:VAD_GRADED_TOTAL]
    with VAD_GRADED_OUT.open("w") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[vad-graded] wrote {len(out)} sentences -> {VAD_GRADED_OUT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    build_warriner()
    build_goemotions_sample()
    build_localizer()
    build_vad_graded()


if __name__ == "__main__":
    main()
