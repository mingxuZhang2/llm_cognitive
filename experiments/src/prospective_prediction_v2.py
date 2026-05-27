#!/usr/bin/env python3
"""
Experiment: Scaled Prospective Failure Prediction (v2).

Uses brain-derived Mental-state Collapse Score (MCS) to predict which
condition pairs will cause held-out social reasoning failures.

MCS(pair) = brain_distance(pair) - LLM_distance(pair)
High MCS = brain says different but LLM says similar → model will confuse them.

Improvements over v1:
  - 50 items per pair (v1 had 5)
  - 7 HIGH + 7 LOW pairs selected from data (v1 had 3+2, hardcoded)
  - Template-based generation with randomised slots for diversity
  - Spearman correlation + Mann-Whitney + mixed-effects regression
  - MCS computed from actual brain & LLM RDMs (not hardcoded)

Usage:
  python prospective_prediction_v2.py \
    --model_path /path/to/model \
    --model_short Qwen2.5-7B-Instruct \
    --rsa_dir results/cognitive_rsa \
    --brain_rdm_path results/cognitive_rsa/brain_rdm.npz \
    --output_dir results/prospective_v2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr, mannwhitneyu

# ---------------------------------------------------------------------------
# 14 cognitive conditions and human-readable descriptions
# ---------------------------------------------------------------------------
CONDITIONS = [
    "anger", "belief", "disgust", "empathy", "fear", "happiness",
    "intention", "judgment", "mentalizing", "moral", "sadness",
    "self_referential", "theory_of_mind", "valence",
]

CONDITION_DESCRIPTIONS = {
    "anger": "anger, hostility, or rage",
    "belief": "someone holding a specific factual belief (which may be wrong)",
    "disgust": "revulsion, contamination, or disgust",
    "empathy": "feeling with or for another person's emotional state",
    "fear": "threat, danger, or anxiety",
    "happiness": "joy, delight, or positive affect",
    "intention": "making plans, deciding to act, or having goals",
    "judgment": "evaluating right/wrong, fairness, or making a judgment call",
    "mentalizing": "thinking about what is going on in someone's mind",
    "moral": "ethical dilemmas, moral principles, right vs wrong",
    "sadness": "grief, loss, or melancholy",
    "self_referential": "thinking about oneself, self-reflection, self-awareness",
    "theory_of_mind": "tracking what different people know/believe, false belief",
    "valence": "general emotional valence (positive vs negative feeling)",
}

# ---------------------------------------------------------------------------
# Slot fillers for template instantiation
# ---------------------------------------------------------------------------
NAMES_MALE = [
    "James", "Daniel", "Marco", "David", "Nathan", "Carlos", "Ahmed",
    "Liam", "Raj", "Sam", "Oliver", "Felix", "Isaac", "Kenji",
    "Thomas", "Ethan", "Gabriel", "Victor", "Leo", "Hugo",
]
NAMES_FEMALE = [
    "Sarah", "Maria", "Priya", "Emma", "Chen", "Sofia", "Amira",
    "Olivia", "Nadia", "Zoe", "Hannah", "Maya", "Aisha", "Yuki",
    "Clara", "Diana", "Eva", "Leila", "Ingrid", "Rosa",
]
NAMES_ALL = NAMES_MALE + NAMES_FEMALE

SETTINGS = [
    "at work", "at school", "at a family gathering", "in a public park",
    "at the grocery store", "on the bus", "at a restaurant",
    "during a meeting", "at a community event", "at a neighbour's house",
    "in the hospital waiting room", "at the playground", "at a party",
    "in the office break room", "at a volunteer centre",
]

OBJECTS = [
    "a book", "a phone", "a letter", "a gift", "a wallet",
    "a photo", "a report", "a cake", "a package", "an umbrella",
]

FOODS = [
    "soup", "cake", "pasta", "sandwiches", "cookies",
    "salad", "bread", "coffee", "pie", "fruit",
]

ANIMALS = [
    "dog", "cat", "bird", "rabbit", "hamster",
    "horse", "goldfish", "turtle", "parrot", "kitten",
]

HOBBIES = [
    "painting", "playing guitar", "gardening", "running",
    "reading", "cooking", "photography", "hiking",
    "singing", "writing", "swimming", "cycling",
]

FACTS_TRUE = [
    "the Earth orbits the Sun",
    "water boils at 100 degrees Celsius",
    "humans need oxygen to survive",
    "the capital of France is Paris",
    "light travels faster than sound",
]

FACTS_FALSE = [
    "the store closes at 6pm (it actually closes at 5pm)",
    "the meeting is on Tuesday (it was moved to Wednesday)",
    "the train leaves at 8am (the schedule changed to 9am)",
    "the recipe calls for two eggs (it actually needs three)",
    "the deadline is next Friday (it was pushed to Thursday)",
]

MORAL_DILEMMAS = [
    "returning extra change a cashier gave by mistake",
    "reporting a friend's dishonest behaviour at work",
    "telling a painful truth versus a comforting lie",
    "keeping a found wallet with cash inside",
    "sharing limited medical supplies fairly",
]


# ---------------------------------------------------------------------------
# SCENARIO TEMPLATES — 7-10 per condition pair
#
# Each template:
#   "scenario": str with {name}, {name2}, {setting} etc. placeholders
#   "correct": which condition is the correct answer ("a" or "b")
#   "question": the A/B question (placeholders for condition descriptions)
#
# We will build templates for ALL 91 pairs.  For pairs not covered by
# hand-crafted templates we fall back to a systematic generator.
# ---------------------------------------------------------------------------

def _make_question(desc_a: str, desc_b: str) -> str:
    return (
        f"Does this primarily involve "
        f"(A) {desc_a}, or "
        f"(B) {desc_b}?"
    )


# ---- Master template bank, keyed by frozenset({cond_a, cond_b}) ----------

TEMPLATE_BANK: dict[frozenset, list[dict]] = {}


def _register(cond_a: str, cond_b: str, templates: list[dict]):
    """Register templates for a pair. 'correct' field uses 'a' or 'b'."""
    TEMPLATE_BANK[frozenset({cond_a, cond_b})] = templates


# -------  belief vs theory_of_mind  ----------------------------------------
_register("belief", "theory_of_mind", [
    {"scenario": "{name} is convinced that {false_fact}. Nobody has corrected {obj_pronoun} yet.",
     "correct": "a"},
    {"scenario": "{name} told {name2} that the meeting is at 3pm. Actually {name} knows it was moved to 4pm but forgot to update {name2}. {name2} is about to leave for the 3pm slot.",
     "correct": "b"},
    {"scenario": "{name} firmly believes that {fact_true}, and argues about it {setting}.",
     "correct": "a"},
    {"scenario": "{name} hid a toy in the drawer. While {name} was away, {name2} moved it to the cupboard. {name} comes back looking for the toy.",
     "correct": "b"},
    {"scenario": "The child was certain that Santa Claus delivered all the presents, even after seeing {name2} wrapping them.",
     "correct": "a"},
    {"scenario": "{name2} told {name} the shop was open, not knowing it had just closed. {name} headed out confidently.",
     "correct": "b"},
    {"scenario": "{name} assumed the exam was tomorrow because no one corrected {obj_pronoun}.",
     "correct": "a"},
    {"scenario": "{name} pretended to like the {food}, not realising {name2} could tell from {name}'s expression. {name2} said nothing.",
     "correct": "b"},
    {"scenario": "{name} was absolutely sure the bus runs on Sundays, though the schedule changed last month.",
     "correct": "a"},
    {"scenario": "{name} left a surprise {object} for {name2}. {name3} saw it and told {name2}, but {name} doesn't know {name2} already knows.",
     "correct": "b"},
])

# -------  intention vs theory_of_mind  -------------------------------------
_register("intention", "theory_of_mind", [
    {"scenario": "{name} decided to hide the {object} in the closet so {name2} wouldn't find it before the birthday.",
     "correct": "a"},
    {"scenario": "{name} didn't realise {name2} could see through the window as {name} snuck the {food}.",
     "correct": "b"},
    {"scenario": "The company planned to expand into three new markets by next quarter.",
     "correct": "a"},
    {"scenario": "{name} pretended to be surprised at the party, not realising {name2} had seen {name} peek at the guest list earlier.",
     "correct": "b"},
    {"scenario": "{name} aimed to finish the project by Friday and scheduled every task carefully.",
     "correct": "a"},
    {"scenario": "{name2} thought the gift was from {name}, but actually {name3} had sent it. Neither {name} nor {name2} knows the mix-up.",
     "correct": "b"},
    {"scenario": "{name} strategically chose to sit near the door so {subj_pronoun} could leave early.",
     "correct": "a"},
    {"scenario": "{name} lied about being sick to skip the event. {name2} believed it, but {name3} saw {name} {hobby} later that day.",
     "correct": "b"},
    {"scenario": "{name} mapped out a detailed budget to save for a new car within two years.",
     "correct": "a"},
    {"scenario": "{name} told {name2} the {food} was homemade. {name2} later found the packaging but {name} doesn't know {name2} discovered the truth.",
     "correct": "b"},
])

# -------  self_referential vs theory_of_mind  ------------------------------
_register("self_referential", "theory_of_mind", [
    {"scenario": "{name} spent the evening reflecting on how {subj_pronoun} had changed since graduating.",
     "correct": "a"},
    {"scenario": "{name} overheard {name2} talking about {name} behind {name}'s back. {name2} doesn't know {name} heard.",
     "correct": "b"},
    {"scenario": "{name} journaled about {subj_pronoun_possessive} strengths and weaknesses before the job interview.",
     "correct": "a"},
    {"scenario": "{name} thought the surprise was ruined, but {name2} had set up a decoy to make {name} think everything was normal.",
     "correct": "b"},
    {"scenario": "Looking in the mirror, {name} wondered whether {subj_pronoun} was living up to {subj_pronoun_possessive} own values.",
     "correct": "a"},
    {"scenario": "{name} told {name2} a white lie. {name2} pretended to believe it, but privately asked {name3} for the truth.",
     "correct": "b"},
    {"scenario": "During meditation, {name} examined {subj_pronoun_possessive} own patterns of anxiety and what triggered them.",
     "correct": "a"},
    {"scenario": "{name} and {name2} each think the other brought dessert. Neither realises no one actually brought it.",
     "correct": "b"},
])

# -------  mentalizing vs theory_of_mind  -----------------------------------
_register("mentalizing", "theory_of_mind", [
    {"scenario": "{name} tried to figure out why {name2} seemed upset during dinner, reading every subtle expression.",
     "correct": "a"},
    {"scenario": "{name} knows the {object} is in the box, but {name2} still thinks it's on the shelf because {name2} didn't see {name} move it.",
     "correct": "b"},
    {"scenario": "The therapist carefully considered what emotional state was driving the patient's outburst.",
     "correct": "a"},
    {"scenario": "{name} planned a surprise. {name2} accidentally overheard the plan, but {name} doesn't know that {name2} knows.",
     "correct": "b"},
    {"scenario": "{name} watched {name2}'s hesitation and guessed that {name2} was feeling conflicted about the decision.",
     "correct": "a"},
    {"scenario": "{name3} told {name} that {name2} believes the office is closed on Monday. In fact it's open, and {name3} forgot to correct {name2}.",
     "correct": "b"},
    {"scenario": "The detective studied the suspect's body language, trying to understand the suspect's emotional motivation.",
     "correct": "a"},
    {"scenario": "{name2} was told the parcel arrived, but actually it was returned to sender. {name} knows the truth but hasn't said anything.",
     "correct": "b"},
])

# -------  empathy vs intention  --------------------------------------------
_register("empathy", "intention", [
    {"scenario": "Watching {name2} struggle with heavy bags {setting}, {name} felt a pang of concern.",
     "correct": "a"},
    {"scenario": "{name} chose to volunteer at the shelter every Saturday for the next three months.",
     "correct": "b"},
    {"scenario": "{name}'s heart broke seeing the abandoned {animal} shivering in the rain.",
     "correct": "a"},
    {"scenario": "They strategically allocated funds to maximise impact in underserved communities.",
     "correct": "b"},
    {"scenario": "{name} couldn't stop tearing up after hearing about the family's loss.",
     "correct": "a"},
    {"scenario": "{name} set a strict schedule: wake at 5am, exercise, study, then commute.",
     "correct": "b"},
    {"scenario": "Seeing {name2} wince in pain after the fall, {name} felt an ache in {subj_pronoun_possessive} own chest.",
     "correct": "a"},
    {"scenario": "{name} carefully planned every step of the fundraiser, assigning roles and deadlines.",
     "correct": "b"},
    {"scenario": "{name} felt deeply moved listening to {name2}'s account of losing a parent.",
     "correct": "a"},
    {"scenario": "{name} resolved to learn {hobby} this year and signed up for weekly lessons.",
     "correct": "b"},
])

# -------  belief vs mentalizing  -------------------------------------------
_register("belief", "mentalizing", [
    {"scenario": "{name} was completely sure that {false_fact}. No amount of evidence could change {subj_pronoun_possessive} mind.",
     "correct": "a"},
    {"scenario": "{name} studied {name2}'s face, trying to figure out what {name2} was really thinking about the proposal.",
     "correct": "b"},
    {"scenario": "Despite all the data, {name} stubbornly held that the project would succeed.",
     "correct": "a"},
    {"scenario": "The teacher noticed {name}'s silence and wondered whether {name} was confused, bored, or distracted.",
     "correct": "b"},
    {"scenario": "{name} assumed the restaurant was still on the corner where it had always been.",
     "correct": "a"},
    {"scenario": "{name} could tell something was off with {name2}, and tried to work out whether {name2} was angry or just tired.",
     "correct": "b"},
    {"scenario": "{name} believed with certainty that {name2} would arrive on time, as always.",
     "correct": "a"},
    {"scenario": "The negotiator read the room carefully, sensing the other side's anxiety behind their calm exterior.",
     "correct": "b"},
])

# -------  judgment vs moral  -----------------------------------------------
_register("judgment", "moral", [
    {"scenario": "{name} evaluated the proposals and decided which one was the fairest allocation of resources.",
     "correct": "a"},
    {"scenario": "{name} faced a dilemma: {moral_dilemma}.",
     "correct": "b"},
    {"scenario": "The panel rated each candidate's portfolio, weighing creativity against technical skill.",
     "correct": "a"},
    {"scenario": "{name} wrestled with whether it was right to break a promise to prevent someone from getting hurt.",
     "correct": "b"},
    {"scenario": "{name} assessed whether the penalty was proportional to the offence.",
     "correct": "a"},
    {"scenario": "The whistleblower debated the ethics of exposing company secrets that could harm employees but help the public.",
     "correct": "b"},
    {"scenario": "The referee reviewed the play and judged whether it was a foul or a clean tackle.",
     "correct": "a"},
    {"scenario": "Should {name} lie to protect a friend, or tell the truth even if it hurts the friendship?",
     "correct": "b"},
    {"scenario": "{name} compared the two job offers, weighing salary against work-life balance to decide which was better.",
     "correct": "a"},
    {"scenario": "{name} discovered that the company was cutting corners on safety. Reporting it could cost people their jobs, but staying silent could endanger lives.",
     "correct": "b"},
])

# -------  anger vs happiness  ----------------------------------------------
_register("anger", "happiness", [
    {"scenario": "{name} slammed the door after the argument, face red with fury.",
     "correct": "a"},
    {"scenario": "{name} laughed with delight when the {animal} licked {subj_pronoun_possessive} face.",
     "correct": "b"},
    {"scenario": "The crowd erupted in cheers when the team scored the winning goal.",
     "correct": "b"},
    {"scenario": "{name} clenched {subj_pronoun_possessive} fists, barely containing the rage at the injustice.",
     "correct": "a"},
    {"scenario": "The children's faces lit up with pure joy when they saw the holiday decorations.",
     "correct": "b"},
    {"scenario": "{name} shouted at the driver who had just cut in front of {obj_pronoun} {setting}.",
     "correct": "a"},
    {"scenario": "{name2} grinned ear to ear after receiving the good news about the promotion.",
     "correct": "b"},
    {"scenario": "After being mocked publicly, {name} felt a boiling hostility rising inside.",
     "correct": "a"},
])

# -------  fear vs happiness  -----------------------------------------------
_register("fear", "happiness", [
    {"scenario": "{name}'s hands trembled as footsteps echoed behind {obj_pronoun} in the dark alley.",
     "correct": "a"},
    {"scenario": "{name} beamed with pride holding the award, surrounded by cheering colleagues.",
     "correct": "b"},
    {"scenario": "{name} froze when {subj_pronoun} saw the snake coiled on the path.",
     "correct": "a"},
    {"scenario": "The whole family burst into laughter during the game {setting}.",
     "correct": "b"},
    {"scenario": "The turbulence made several passengers grip their armrests in terror.",
     "correct": "a"},
    {"scenario": "{name} danced around the kitchen after learning the loan was approved.",
     "correct": "b"},
    {"scenario": "{name} jumped back with a gasp when the fireworks exploded unexpectedly.",
     "correct": "a"},
    {"scenario": "{name} couldn't stop smiling after {name2} proposed {setting}.",
     "correct": "b"},
])

# -------  anger vs belief  -------------------------------------------------
_register("anger", "belief", [
    {"scenario": "{name} threw the phone across the room after reading the insulting message.",
     "correct": "a"},
    {"scenario": "{name} was convinced that the bus still stopped at the old route, despite the signs.",
     "correct": "b"},
    {"scenario": "{name}'s voice shook with rage as {subj_pronoun} confronted the manager about the unfair treatment.",
     "correct": "a"},
    {"scenario": "{name} assumed the pharmacy was open until 9pm because it always had been.",
     "correct": "b"},
    {"scenario": "After being lied to repeatedly, {name} was seething with anger.",
     "correct": "a"},
    {"scenario": "{name} was certain the password hadn't changed, and kept typing it in.",
     "correct": "b"},
    {"scenario": "{name} kicked the chair in frustration after the third rejection.",
     "correct": "a"},
    {"scenario": "{name} held the firm conviction that hard work alone determines success.",
     "correct": "b"},
])

# -------  sadness vs intention  --------------------------------------------
_register("sadness", "intention", [
    {"scenario": "{name} sat alone {setting}, tears streaming down {subj_pronoun_possessive} face after the funeral.",
     "correct": "a"},
    {"scenario": "{name} outlined a five-step plan to get promoted within the year.",
     "correct": "b"},
    {"scenario": "A wave of grief washed over {name} as {subj_pronoun} packed up {name2}'s belongings.",
     "correct": "a"},
    {"scenario": "{name} deliberately chose the harder assignment because it aligned with {subj_pronoun_possessive} career goals.",
     "correct": "b"},
    {"scenario": "{name} stared at the old photograph, overwhelmed by how much {subj_pronoun} missed those days.",
     "correct": "a"},
    {"scenario": "{name} booked the flight, reserved the hotel, and mapped out every day of the trip.",
     "correct": "b"},
    {"scenario": "Hearing the song reminded {name} of {name2}, and the loss felt as fresh as ever.",
     "correct": "a"},
    {"scenario": "{name} committed to running a marathon and created a detailed training schedule.",
     "correct": "b"},
])

# -------  fear vs moral  ---------------------------------------------------
_register("fear", "moral", [
    {"scenario": "{name}'s heart pounded as the elevator lurched to a stop between floors.",
     "correct": "a"},
    {"scenario": "The committee debated whether it was ethical to use the experimental treatment without full consent.",
     "correct": "b"},
    {"scenario": "{name} bolted upright in bed, terrified by a noise downstairs.",
     "correct": "a"},
    {"scenario": "Should a doctor prioritise a younger patient over an elderly one when resources are scarce?",
     "correct": "b"},
    {"scenario": "{name} was too frightened to cross the bridge after the earthquake.",
     "correct": "a"},
    {"scenario": "{name} agonised over whether to return the extra money the ATM dispensed.",
     "correct": "b"},
    {"scenario": "The spider on the ceiling made {name} back away slowly, pulse racing.",
     "correct": "a"},
    {"scenario": "Is it ever acceptable to break a law to prevent greater harm?",
     "correct": "b"},
])

# -------  happiness vs theory_of_mind  -------------------------------------
_register("happiness", "theory_of_mind", [
    {"scenario": "{name} grinned from ear to ear after the surprise announcement {setting}.",
     "correct": "a"},
    {"scenario": "{name} realised that {name2} didn't know the party was cancelled, and wondered whether to tell {obj_pronoun2}.",
     "correct": "b"},
    {"scenario": "The whole room was glowing with warmth and laughter during the celebration.",
     "correct": "a"},
    {"scenario": "{name} figured out that {name2} still believed the old rumour, even though it had been debunked.",
     "correct": "b"},
    {"scenario": "{name} felt pure elation as the results were announced.",
     "correct": "a"},
    {"scenario": "{name} knew the secret, but pretended not to, while {name2} tried to figure out if {name} knew.",
     "correct": "b"},
    {"scenario": "{name} was overjoyed to reunite with {name2} after years apart.",
     "correct": "a"},
    {"scenario": "{name2} doesn't realise that {name} already opened the {object} and knows what's inside.",
     "correct": "b"},
])

# -------  disgust vs mentalizing  ------------------------------------------
_register("disgust", "mentalizing", [
    {"scenario": "{name} gagged at the sight of the rotting food left in the break-room fridge.",
     "correct": "a"},
    {"scenario": "{name} carefully analysed {name2}'s expression, trying to determine whether {name2} was offended or just tired.",
     "correct": "b"},
    {"scenario": "The smell from the waste bin was so overwhelming that {name} had to leave the room.",
     "correct": "a"},
    {"scenario": "The counsellor listened intently, working to understand the complex mix of shame and defiance in the client's voice.",
     "correct": "b"},
    {"scenario": "{name} recoiled when {subj_pronoun} stepped in something unpleasant on the pavement.",
     "correct": "a"},
    {"scenario": "{name} wondered whether {name2} was genuinely happy about the news or just putting on a brave face.",
     "correct": "b"},
    {"scenario": "Opening the container of weeks-old leftovers, {name} felt a wave of nausea.",
     "correct": "a"},
    {"scenario": "The diplomat tried to read the delegates' true positions behind their polite smiles.",
     "correct": "b"},
])

# -------  anger vs disgust  ------------------------------------------------
_register("anger", "disgust", [
    {"scenario": "{name} was furious after discovering that {name2} had lied about the report.",
     "correct": "a"},
    {"scenario": "{name} recoiled at the sight of mould covering the entire wall.",
     "correct": "b"},
    {"scenario": "{name} yelled in rage when the car was towed without warning.",
     "correct": "a"},
    {"scenario": "The smell of rotten eggs made {name} cover {subj_pronoun_possessive} nose and back away.",
     "correct": "b"},
    {"scenario": "After being betrayed by a close friend, {name} was seething with hostile fury.",
     "correct": "a"},
    {"scenario": "{name} felt nauseated watching the documentary about unsanitary food processing.",
     "correct": "b"},
    {"scenario": "{name} punched the wall after hearing the unfair verdict.",
     "correct": "a"},
    {"scenario": "The bathroom was in such a filthy state that {name} refused to enter.",
     "correct": "b"},
])

# -------  anger vs fear  ---------------------------------------------------
_register("anger", "fear", [
    {"scenario": "{name} screamed at the unfair referee, face contorted with rage.",
     "correct": "a"},
    {"scenario": "{name} heard a loud crash downstairs and felt {subj_pronoun_possessive} blood run cold.",
     "correct": "b"},
    {"scenario": "{name} was livid that {name2} had taken credit for {name}'s work.",
     "correct": "a"},
    {"scenario": "{name} backed slowly away from the growling stray {animal}.",
     "correct": "b"},
    {"scenario": "Slamming the table, {name} demanded an explanation for the mistake.",
     "correct": "a"},
    {"scenario": "A sudden gust of wind shook the windows, and {name} jumped in alarm.",
     "correct": "b"},
    {"scenario": "{name} stormed out of the room after the argument, seething.",
     "correct": "a"},
    {"scenario": "{name}'s hands were shaking as {subj_pronoun} opened the letter from the hospital.",
     "correct": "b"},
])

# -------  empathy vs sadness  ----------------------------------------------
_register("empathy", "sadness", [
    {"scenario": "{name} felt a deep ache of compassion listening to {name2} describe the hardship.",
     "correct": "a"},
    {"scenario": "Alone on the bench {setting}, {name} wept silently, overwhelmed by loneliness.",
     "correct": "b"},
    {"scenario": "{name} could feel {name2}'s pain as if it were {subj_pronoun_possessive} own.",
     "correct": "a"},
    {"scenario": "{name} stared out the window in a grey mood, missing {name2} who had moved away.",
     "correct": "b"},
    {"scenario": "Watching the family lose their home, {name} was moved to donate immediately.",
     "correct": "a"},
    {"scenario": "{name} lay in bed unable to sleep, consumed by grief over the end of the relationship.",
     "correct": "b"},
    {"scenario": "{name} held {name2}'s hand and whispered, 'I understand what you're going through.'",
     "correct": "a"},
    {"scenario": "Tears fell as {name} walked past the empty room that once belonged to {name2}.",
     "correct": "b"},
])

# -------  empathy vs mentalizing  ------------------------------------------
_register("empathy", "mentalizing", [
    {"scenario": "{name} felt {name2}'s devastation as though it were {subj_pronoun_possessive} own, tears welling up.",
     "correct": "a"},
    {"scenario": "{name} observed {name2}'s body language and deduced that {name2} was hiding nervousness.",
     "correct": "b"},
    {"scenario": "Hearing about the accident, {name} was overwhelmed with compassion and sorrow for the victims.",
     "correct": "a"},
    {"scenario": "The psychologist carefully mapped the patient's cognitive patterns from the session notes.",
     "correct": "b"},
    {"scenario": "{name}'s eyes filled with tears seeing {name2} struggle {setting}.",
     "correct": "a"},
    {"scenario": "{name} puzzled over why {name2} had suddenly changed the subject, suspecting discomfort.",
     "correct": "b"},
    {"scenario": "{name} embraced {name2}, saying, 'I feel your pain, and I'm here for you.'",
     "correct": "a"},
    {"scenario": "From the hesitation in {name2}'s voice, {name} inferred that {name2} had doubts about the plan.",
     "correct": "b"},
])

# -------  fear vs disgust  -------------------------------------------------
_register("fear", "disgust", [
    {"scenario": "{name} froze in place when {subj_pronoun} heard a growl from the dark basement.",
     "correct": "a"},
    {"scenario": "Opening the old container, {name} was hit by a putrid smell that made {obj_pronoun} gag.",
     "correct": "b"},
    {"scenario": "{name} broke into a cold sweat as the plane hit severe turbulence.",
     "correct": "a"},
    {"scenario": "{name} couldn't look at the plate after finding a hair baked into the {food}.",
     "correct": "b"},
    {"scenario": "The idea of walking across the glass floor 50 storeys up terrified {name}.",
     "correct": "a"},
    {"scenario": "{name} turned away in revulsion from the overflowing rubbish bin.",
     "correct": "b"},
    {"scenario": "{name} woke up screaming from a nightmare about being chased.",
     "correct": "a"},
    {"scenario": "The sight of the insects crawling on the {food} made {name} feel physically ill.",
     "correct": "b"},
])

# -------  sadness vs happiness  --------------------------------------------
_register("sadness", "happiness", [
    {"scenario": "{name} sat quietly {setting}, weighed down by a profound sense of loss.",
     "correct": "a"},
    {"scenario": "{name} was beaming with excitement after hearing the wonderful news.",
     "correct": "b"},
    {"scenario": "The melancholy music reminded {name} of everything that had gone wrong.",
     "correct": "a"},
    {"scenario": "Everyone at the table raised their glasses and cheered in pure delight.",
     "correct": "b"},
    {"scenario": "{name} struggled to get out of bed, feeling hollow after the breakup.",
     "correct": "a"},
    {"scenario": "{name} twirled around the room, giddy with joy at the acceptance letter.",
     "correct": "b"},
    {"scenario": "A deep gloom settled over {name} as {subj_pronoun} revisited old memories.",
     "correct": "a"},
    {"scenario": "{name2} surprised {name} with flowers, and {name} smiled the widest smile.",
     "correct": "b"},
])

# -------  anger vs empathy  ------------------------------------------------
_register("anger", "empathy", [
    {"scenario": "{name} lashed out at the waiter for getting the order wrong, voice full of hostility.",
     "correct": "a"},
    {"scenario": "{name} felt a surge of compassion seeing {name2} cry about the lost job.",
     "correct": "b"},
    {"scenario": "Furious at the noise, {name} banged on the neighbour's door.",
     "correct": "a"},
    {"scenario": "{name} gently held {name2}'s hand, sharing the weight of {name2}'s sorrow.",
     "correct": "b"},
    {"scenario": "{name} was incensed when the project was unfairly cancelled.",
     "correct": "a"},
    {"scenario": "Watching the elderly woman struggle with her groceries, {name} felt a warm wave of concern.",
     "correct": "b"},
    {"scenario": "{name} ranted for ten minutes about the rude customer, still fuming.",
     "correct": "a"},
    {"scenario": "{name} listened with deep care as {name2} described the difficult divorce.",
     "correct": "b"},
])

# -------  intention vs self_referential  -----------------------------------
_register("intention", "self_referential", [
    {"scenario": "{name} mapped out a detailed career plan for the next five years.",
     "correct": "a"},
    {"scenario": "{name} spent the evening pondering who {subj_pronoun} really was and what mattered most.",
     "correct": "b"},
    {"scenario": "{name} scheduled three gym sessions a week and set reminders.",
     "correct": "a"},
    {"scenario": "Looking back on the past decade, {name} reflected on how much {subj_pronoun} had grown.",
     "correct": "b"},
    {"scenario": "{name} decided to take the overnight flight so {subj_pronoun} could arrive a day early.",
     "correct": "a"},
    {"scenario": "{name} asked {subj_pronoun_possessive}self whether {subj_pronoun} was truly happy or just going through the motions.",
     "correct": "b"},
    {"scenario": "The team agreed on a strategy: launch the product in Q2 and expand in Q3.",
     "correct": "a"},
    {"scenario": "During therapy, {name} explored {subj_pronoun_possessive} own insecurities and how they shaped {subj_pronoun_possessive} relationships.",
     "correct": "b"},
])

# -------  moral vs self_referential  ---------------------------------------
_register("moral", "self_referential", [
    {"scenario": "{name} agonised over whether it was ethical to report a colleague's minor infraction.",
     "correct": "a"},
    {"scenario": "{name} sat quietly and considered how {subj_pronoun_possessive} own childhood shaped {subj_pronoun_possessive} adult personality.",
     "correct": "b"},
    {"scenario": "Is it right to break a promise to one friend in order to protect another?",
     "correct": "a"},
    {"scenario": "{name} examined {subj_pronoun_possessive} own biases and wondered how they influenced decisions.",
     "correct": "b"},
    {"scenario": "The jury debated the morality of the defendant's actions in the context of poverty.",
     "correct": "a"},
    {"scenario": "{name} journaled about {subj_pronoun_possessive} identity and sense of purpose after turning forty.",
     "correct": "b"},
    {"scenario": "{name} questioned whether using the leaked data, even for a good cause, was morally defensible.",
     "correct": "a"},
    {"scenario": "{name} reflected on {subj_pronoun_possessive} own strengths and weaknesses while preparing for the interview.",
     "correct": "b"},
])

# -------  anger vs valence  ------------------------------------------------
_register("anger", "valence", [
    {"scenario": "{name} pounded the desk in fury after the third rejection letter.",
     "correct": "a"},
    {"scenario": "{name} noticed a shift in {subj_pronoun_possessive} overall emotional tone from positive to negative as the week dragged on.",
     "correct": "b"},
    {"scenario": "{name} hurled accusations at {name2} in a fit of rage.",
     "correct": "a"},
    {"scenario": "The day started pleasantly, but by evening {name}'s mood had soured noticeably.",
     "correct": "b"},
    {"scenario": "After being insulted {setting}, {name} felt a white-hot flash of hostility.",
     "correct": "a"},
    {"scenario": "{name} rated the film as emotionally negative overall, though some scenes were uplifting.",
     "correct": "b"},
    {"scenario": "{name} was still boiling hours after the confrontation.",
     "correct": "a"},
    {"scenario": "The counsellor asked {name} to describe {subj_pronoun_possessive} general emotional state this week as positive, negative, or neutral.",
     "correct": "b"},
])

# -------  disgust vs fear  (same as fear vs disgust, reversed labels) ------
# Already registered above as ("fear", "disgust").

# -------  empathy vs self_referential  -------------------------------------
_register("empathy", "self_referential", [
    {"scenario": "{name} felt {name2}'s heartbreak as if it were {subj_pronoun_possessive} own.",
     "correct": "a"},
    {"scenario": "{name} pondered whether {subj_pronoun} was living up to {subj_pronoun_possessive} own ideals.",
     "correct": "b"},
    {"scenario": "Tears rolled down {name}'s cheeks as {name2} described the loss.",
     "correct": "a"},
    {"scenario": "{name} asked {subj_pronoun_possessive}self, 'What kind of person do I want to become?'",
     "correct": "b"},
    {"scenario": "{name} was deeply moved by the stranger's story of resilience.",
     "correct": "a"},
    {"scenario": "In the quiet of the morning, {name} reflected on how {subj_pronoun_possessive} values had shifted over the years.",
     "correct": "b"},
    {"scenario": "Seeing the child crying alone, {name} felt an overwhelming urge to comfort them.",
     "correct": "a"},
    {"scenario": "{name} meditated on {subj_pronoun_possessive} own emotional responses, trying to understand why {subj_pronoun} reacted so strongly.",
     "correct": "b"},
])

# -------  fear vs sadness  -------------------------------------------------
_register("fear", "sadness", [
    {"scenario": "{name} cowered under the blankets as lightning struck nearby.",
     "correct": "a"},
    {"scenario": "{name} cried quietly at the memorial, missing {name2} terribly.",
     "correct": "b"},
    {"scenario": "The dark corridor made {name}'s pulse race with dread.",
     "correct": "a"},
    {"scenario": "A heavy melancholy settled over {name} as the holidays approached without {name2}.",
     "correct": "b"},
    {"scenario": "{name} backed away from the edge of the cliff, dizzy with vertigo.",
     "correct": "a"},
    {"scenario": "{name} sat in the empty house, overcome by loneliness after {name2} left.",
     "correct": "b"},
    {"scenario": "The warning siren sent a jolt of panic through the whole building.",
     "correct": "a"},
    {"scenario": "{name} flipped through old photos with a bittersweet ache of loss.",
     "correct": "b"},
])

# -------  judgment vs belief  ----------------------------------------------
_register("judgment", "belief", [
    {"scenario": "{name} weighed the evidence and concluded the proposal was unfair to the junior staff.",
     "correct": "a"},
    {"scenario": "{name} was utterly convinced the deadline was Friday, not Thursday.",
     "correct": "b"},
    {"scenario": "The panel evaluated each submission against strict quality criteria.",
     "correct": "a"},
    {"scenario": "{name} believed without question that the supplement would cure {subj_pronoun_possessive} cold.",
     "correct": "b"},
    {"scenario": "{name} assessed whether the punishment fit the crime.",
     "correct": "a"},
    {"scenario": "Despite the forecast, {name} was sure it wouldn't rain today.",
     "correct": "b"},
    {"scenario": "After careful deliberation, the judge ruled the evidence inadmissible.",
     "correct": "a"},
    {"scenario": "{name} maintained the unwavering belief that the earth was only 6000 years old.",
     "correct": "b"},
])

# -------  disgust vs sadness  ----------------------------------------------
_register("disgust", "sadness", [
    {"scenario": "{name} gagged at the rotting smell coming from the drain.",
     "correct": "a"},
    {"scenario": "{name} felt a wave of sorrow watching the moving tribute at the ceremony.",
     "correct": "b"},
    {"scenario": "The mouldy bread made {name} throw the whole bag away in revulsion.",
     "correct": "a"},
    {"scenario": "A deep gloom fell over {name} on the anniversary of {name2}'s passing.",
     "correct": "b"},
    {"scenario": "{name} nearly vomited seeing the unsanitary conditions in the kitchen.",
     "correct": "a"},
    {"scenario": "{name} sat alone on the bench, eyes red and swollen from crying.",
     "correct": "b"},
    {"scenario": "The sight of cockroaches in the cupboard made {name} feel physically sick.",
     "correct": "a"},
    {"scenario": "{name} listened to the sad ballad and felt tears well up.",
     "correct": "b"},
])

# -------  valence vs happiness  --------------------------------------------
_register("valence", "happiness", [
    {"scenario": "{name} noticed {subj_pronoun_possessive} overall mood had been generally positive all week.",
     "correct": "a"},
    {"scenario": "{name} burst into ecstatic laughter after winning the competition.",
     "correct": "b"},
    {"scenario": "The therapist asked {name} to rate {subj_pronoun_possessive} emotional state from very negative to very positive.",
     "correct": "a"},
    {"scenario": "{name} jumped up and down with pure joy when the acceptance email arrived.",
     "correct": "b"},
    {"scenario": "{name} reflected that the overall emotional tone of the trip had been mildly pleasant.",
     "correct": "a"},
    {"scenario": "{name2} presented {name} with the award, and {name} beamed with unmistakable delight.",
     "correct": "b"},
    {"scenario": "The movie left {name} feeling generally upbeat, though no single scene was memorable.",
     "correct": "a"},
    {"scenario": "{name} couldn't contain {subj_pronoun_possessive} grin after {name2}'s hilarious prank.",
     "correct": "b"},
])

# -------  valence vs sadness  ----------------------------------------------
_register("valence", "sadness", [
    {"scenario": "{name} rated the experience as emotionally negative on the survey form.",
     "correct": "a"},
    {"scenario": "{name} wept at the funeral, devastated by the loss.",
     "correct": "b"},
    {"scenario": "Overall, {name} described {subj_pronoun_possessive} week as having a mostly negative emotional tone.",
     "correct": "a"},
    {"scenario": "The emptiness of the apartment after {name2} moved out filled {name} with grief.",
     "correct": "b"},
    {"scenario": "{name} felt the general emotional temperature of the room shift from neutral to mildly unpleasant.",
     "correct": "a"},
    {"scenario": "{name} cried alone in the car, unable to shake the heavy feeling of loss.",
     "correct": "b"},
    {"scenario": "The report described the community's overall sentiment as predominantly negative after the announcement.",
     "correct": "a"},
    {"scenario": "Hearing the news about {name2}'s diagnosis, {name} was consumed by sorrow.",
     "correct": "b"},
])

# -------  moral vs judgment  (already registered as judgment vs moral) ------
# -------  intention vs belief  ---------------------------------------------
_register("intention", "belief", [
    {"scenario": "{name} sat down and carefully planned every step of the house renovation.",
     "correct": "a"},
    {"scenario": "{name} was absolutely certain the library closed at 8pm, though it now closes at 7.",
     "correct": "b"},
    {"scenario": "{name} decided to switch careers and enrolled in evening classes.",
     "correct": "a"},
    {"scenario": "No matter what anyone said, {name} believed the old remedy would work.",
     "correct": "b"},
    {"scenario": "{name} set a goal to save a specific amount each month for two years.",
     "correct": "a"},
    {"scenario": "{name} assumed the traffic would be light on a holiday, and didn't check the map.",
     "correct": "b"},
    {"scenario": "The team resolved to submit the application before the end of the month.",
     "correct": "a"},
    {"scenario": "{name} clung to the belief that the company would never lay people off.",
     "correct": "b"},
])

# -------  mentalizing vs self_referential  ---------------------------------
_register("mentalizing", "self_referential", [
    {"scenario": "{name} studied {name2}'s hesitant smile, trying to decipher the emotions behind it.",
     "correct": "a"},
    {"scenario": "{name} spent the weekend journaling about {subj_pronoun_possessive} own life goals and values.",
     "correct": "b"},
    {"scenario": "The negotiator tried to figure out whether the other party was bluffing.",
     "correct": "a"},
    {"scenario": "{name} asked {subj_pronoun_possessive}self whether {subj_pronoun} was truly passionate about {subj_pronoun_possessive} work or just comfortable.",
     "correct": "b"},
    {"scenario": "{name} noticed {name2} fidgeting and guessed {name2} was anxious about the results.",
     "correct": "a"},
    {"scenario": "During a quiet walk, {name} reflected on how {subj_pronoun_possessive} childhood shaped {subj_pronoun_possessive} adult relationships.",
     "correct": "b"},
    {"scenario": "The coach watched the player's reactions closely to gauge confidence levels.",
     "correct": "a"},
    {"scenario": "{name} considered whether {subj_pronoun_possessive} own impatience was holding {obj_pronoun} back.",
     "correct": "b"},
])

# -------  intention vs moral  ----------------------------------------------
_register("intention", "moral", [
    {"scenario": "{name} planned to reorganise the entire filing system over the weekend.",
     "correct": "a"},
    {"scenario": "Is it right to sacrifice one person's privacy to protect many from harm?",
     "correct": "b"},
    {"scenario": "{name} scheduled meetings for every day next week to push the deal through.",
     "correct": "a"},
    {"scenario": "{name} wrestled with the ethics of accepting a gift from a client.",
     "correct": "b"},
    {"scenario": "{name} resolved to learn a new language and downloaded three apps.",
     "correct": "a"},
    {"scenario": "The board debated whether maximising shareholder value justified cutting employee benefits.",
     "correct": "b"},
    {"scenario": "The architect drew up blueprints to expand the building by two storeys.",
     "correct": "a"},
    {"scenario": "Should the journalist publish the story, knowing it would ruin the politician's career but serve the public interest?",
     "correct": "b"},
])

# -------  anger vs sadness  ------------------------------------------------
_register("anger", "sadness", [
    {"scenario": "{name} threw {subj_pronoun_possessive} keys across the room in a burst of fury.",
     "correct": "a"},
    {"scenario": "{name} sat in the dim room, quietly weeping over the loss.",
     "correct": "b"},
    {"scenario": "{name} yelled at {name2} for breaking the promise, voice shaking with rage.",
     "correct": "a"},
    {"scenario": "A profound melancholy settled over {name} as {subj_pronoun} read the goodbye letter.",
     "correct": "b"},
    {"scenario": "{name} was so angry {subj_pronoun} couldn't speak, just pacing back and forth.",
     "correct": "a"},
    {"scenario": "{name} held the old {object} and cried, remembering {name2}.",
     "correct": "b"},
    {"scenario": "The driver honked furiously and shouted at the pedestrian.",
     "correct": "a"},
    {"scenario": "The news of the school's closure left the community in deep sorrow.",
     "correct": "b"},
])

# -------  disgust vs happiness  --------------------------------------------
_register("disgust", "happiness", [
    {"scenario": "{name} recoiled at the foul-smelling liquid leaking from the bin.",
     "correct": "a"},
    {"scenario": "{name} was positively glowing with joy after the birth announcement.",
     "correct": "b"},
    {"scenario": "The mouldy sandwich made {name} grimace and push {subj_pronoun_possessive} lunch away.",
     "correct": "a"},
    {"scenario": "{name} clapped and cheered when {name2} crossed the finish line first.",
     "correct": "b"},
    {"scenario": "{name} was revolted by the film's graphic depiction of filth.",
     "correct": "a"},
    {"scenario": "A rush of delight filled {name} as the test came back positive for the good news.",
     "correct": "b"},
    {"scenario": "The cockroach on the counter made {name} shriek in revulsion.",
     "correct": "a"},
    {"scenario": "{name} couldn't stop smiling at the adorable {animal} video.",
     "correct": "b"},
])

# -------  valence vs fear  -------------------------------------------------
_register("valence", "fear", [
    {"scenario": "{name} described the overall emotional tone of the week as quite negative.",
     "correct": "a"},
    {"scenario": "{name} clutched the armrest as the roller coaster plummeted, heart pounding.",
     "correct": "b"},
    {"scenario": "The survey measured participants' general affective state on a positive-negative scale.",
     "correct": "a"},
    {"scenario": "A shadow moved at the edge of {name}'s vision, and {subj_pronoun} felt a spike of terror.",
     "correct": "b"},
    {"scenario": "{name} reflected that the day had an overall pleasant emotional quality.",
     "correct": "a"},
    {"scenario": "{name} was paralysed with dread as the deadline loomed and nothing was ready.",
     "correct": "b"},
    {"scenario": "{name} checked the mood tracker and noted a gradual shift from neutral to slightly positive.",
     "correct": "a"},
    {"scenario": "{name} ran from the building when the fire alarm sounded, gasping with panic.",
     "correct": "b"},
])

# -------  mentalizing vs moral  --------------------------------------------
_register("mentalizing", "moral", [
    {"scenario": "{name} watched {name2}'s microexpressions closely, trying to understand {name2}'s true feelings.",
     "correct": "a"},
    {"scenario": "The philosophers debated whether lying is ever morally permissible.",
     "correct": "b"},
    {"scenario": "{name} could sense that {name2} was holding back — there was something {name2} wasn't saying.",
     "correct": "a"},
    {"scenario": "Is it ethical to use someone's personal data without their knowledge to prevent a crime?",
     "correct": "b"},
    {"scenario": "The teacher tried to figure out whether the student was confused, defiant, or simply disengaged.",
     "correct": "a"},
    {"scenario": "{name} questioned whether profiting from the situation made it inherently wrong.",
     "correct": "b"},
    {"scenario": "{name} read between the lines of {name2}'s email, inferring frustration beneath the polite tone.",
     "correct": "a"},
    {"scenario": "Should {name} report the violation, even though it would destroy {name2}'s livelihood?",
     "correct": "b"},
])

# -------  disgust vs empathy  ----------------------------------------------
_register("disgust", "empathy", [
    {"scenario": "The pungent odour from the drains made {name} hold {subj_pronoun_possessive} breath and walk faster.",
     "correct": "a"},
    {"scenario": "{name} was moved to tears by {name2}'s story of struggling alone after the layoff.",
     "correct": "b"},
    {"scenario": "{name} nearly gagged at the sight of spoiled milk in the mug.",
     "correct": "a"},
    {"scenario": "Seeing {name2} wince after the fall, {name} felt an ache in {subj_pronoun_possessive} own chest.",
     "correct": "b"},
    {"scenario": "The matted, foul-smelling rug made {name} refuse to sit down.",
     "correct": "a"},
    {"scenario": "{name} listened intently, sharing {name2}'s grief over the miscarriage.",
     "correct": "b"},
    {"scenario": "{name} scraped the unidentifiable substance off {subj_pronoun_possessive} shoe in disgust.",
     "correct": "a"},
    {"scenario": "{name} put an arm around {name2}, genuinely feeling the weight of {name2}'s burden.",
     "correct": "b"},
])

# -------  judgment vs intention  -------------------------------------------
_register("judgment", "intention", [
    {"scenario": "The panel scored each design on originality, feasibility, and aesthetics.",
     "correct": "a"},
    {"scenario": "{name} resolved to wake at 5am every day to prepare for the exam.",
     "correct": "b"},
    {"scenario": "{name} determined that the settlement was fair to both parties.",
     "correct": "a"},
    {"scenario": "{name} drafted a step-by-step plan to relocate the office within six months.",
     "correct": "b"},
    {"scenario": "The critic assessed the novel's structure and pronounced it exceptional.",
     "correct": "a"},
    {"scenario": "{name} decided to propose to {name2} and started shopping for rings.",
     "correct": "b"},
    {"scenario": "After reviewing the data, {name} judged the experiment's results to be inconclusive.",
     "correct": "a"},
    {"scenario": "{name} committed to daily {hobby} and blocked out time on the calendar.",
     "correct": "b"},
])

# -------  belief vs empathy  -----------------------------------------------
_register("belief", "empathy", [
    {"scenario": "{name} remained absolutely convinced that the meeting was at 2pm, despite the rescheduling.",
     "correct": "a"},
    {"scenario": "{name} felt a wave of compassion wash over {obj_pronoun} as {name2} described the hardship.",
     "correct": "b"},
    {"scenario": "No evidence could shake {name}'s conviction that the product was safe.",
     "correct": "a"},
    {"scenario": "{name} teared up listening to the stranger's account of fleeing the disaster.",
     "correct": "b"},
    {"scenario": "{name} assumed the shop would be open, just as it had been every Sunday for years.",
     "correct": "a"},
    {"scenario": "Holding {name2}'s hand, {name} felt every tremor of {name2}'s anxiety.",
     "correct": "b"},
    {"scenario": "{name} believed the rumour without question, even though the source was unreliable.",
     "correct": "a"},
    {"scenario": "{name} was deeply moved watching the documentary about families separated by conflict.",
     "correct": "b"},
])

# -------  judgment vs empathy  ---------------------------------------------
_register("judgment", "empathy", [
    {"scenario": "The committee rated the proposals against a rubric and ranked them from best to worst.",
     "correct": "a"},
    {"scenario": "{name} was overcome with compassion watching {name2} struggle with the heavy bags.",
     "correct": "b"},
    {"scenario": "{name} assessed whether the contractor's quote was reasonable for the scope of work.",
     "correct": "a"},
    {"scenario": "{name} held {name2} close, absorbing the full weight of {name2}'s grief.",
     "correct": "b"},
    {"scenario": "The reviewer concluded that the paper's methods were flawed.",
     "correct": "a"},
    {"scenario": "{name} felt {name2}'s loneliness as keenly as if it were {subj_pronoun_possessive} own.",
     "correct": "b"},
    {"scenario": "{name} weighed the pros and cons and decided the plan was not cost-effective.",
     "correct": "a"},
    {"scenario": "Seeing the elderly woman fall {setting}, {name} felt a sharp pang of concern.",
     "correct": "b"},
])

# -------  moral vs empathy  ------------------------------------------------
_register("moral", "empathy", [
    {"scenario": "{name} debated whether it was ethically right to break confidentiality to prevent harm.",
     "correct": "a"},
    {"scenario": "{name} couldn't hold back tears watching {name2} recount the ordeal.",
     "correct": "b"},
    {"scenario": "The ethics board reviewed whether the study adequately protected participants' rights.",
     "correct": "a"},
    {"scenario": "{name} was overwhelmed with fellow-feeling for the displaced families.",
     "correct": "b"},
    {"scenario": "Is it moral to sacrifice one person's autonomy for the greater good?",
     "correct": "a"},
    {"scenario": "{name} felt {name2}'s pain so acutely that {subj_pronoun} had to step away to compose {obj_pronoun}self.",
     "correct": "b"},
    {"scenario": "The activist argued that the policy was morally indefensible.",
     "correct": "a"},
    {"scenario": "{name} sat with {name2} in silence, sharing the heaviness of the moment.",
     "correct": "b"},
])

# -------  valence vs anger  (same as anger vs valence, reversed) -----------
# Already registered.

# -------  fear vs valence  (same as valence vs fear, reversed) -------------
# Already registered.

# -------  valence vs disgust  ----------------------------------------------
_register("valence", "disgust", [
    {"scenario": "{name} described the overall emotional quality of the experience as slightly negative.",
     "correct": "a"},
    {"scenario": "{name} gagged at the rancid smell from the forgotten leftovers.",
     "correct": "b"},
    {"scenario": "Participants rated their general affect after the experiment on a positivity scale.",
     "correct": "a"},
    {"scenario": "The slimy texture of the unfamiliar dish made {name} push it away in revulsion.",
     "correct": "b"},
    {"scenario": "{name} reflected that the overall emotional tenor of the visit was moderately positive.",
     "correct": "a"},
    {"scenario": "{name} covered {subj_pronoun_possessive} mouth at the sight of the decomposing animal by the roadside.",
     "correct": "b"},
    {"scenario": "The app tracked {name}'s daily emotional valence on a sliding scale from negative to positive.",
     "correct": "a"},
    {"scenario": "{name} felt physically ill at the grimy state of the public restroom.",
     "correct": "b"},
])

# -------  judgment vs fear  ------------------------------------------------
_register("judgment", "fear", [
    {"scenario": "{name} carefully evaluated the candidates and ranked them on competence.",
     "correct": "a"},
    {"scenario": "{name} froze in terror when the lights suddenly went out.",
     "correct": "b"},
    {"scenario": "The appraiser judged the property to be overvalued by twenty percent.",
     "correct": "a"},
    {"scenario": "{name}'s stomach dropped as the car skidded on the icy road.",
     "correct": "b"},
    {"scenario": "After deliberation, the panel concluded the response time was inadequate.",
     "correct": "a"},
    {"scenario": "{name} backed slowly out of the room after spotting the wasp's nest.",
     "correct": "b"},
    {"scenario": "{name} assessed the restaurant's hygiene and rated it below average.",
     "correct": "a"},
    {"scenario": "A loud crash from outside made {name} jump with alarm.",
     "correct": "b"},
])

# -------  judgment vs sadness  ---------------------------------------------
_register("judgment", "sadness", [
    {"scenario": "The critics evaluated the performance and pronounced it technically brilliant.",
     "correct": "a"},
    {"scenario": "{name} sat silently {setting}, weighed down by an aching sense of loss.",
     "correct": "b"},
    {"scenario": "{name} compared the two offers and judged the first to be more equitable.",
     "correct": "a"},
    {"scenario": "Tears streamed down {name}'s face as {subj_pronoun} read the farewell note.",
     "correct": "b"},
    {"scenario": "The inspector rated the building's safety as unsatisfactory.",
     "correct": "a"},
    {"scenario": "{name} felt a hollow emptiness after learning of {name2}'s departure.",
     "correct": "b"},
    {"scenario": "After careful review, {name} concluded the manuscript was publishable.",
     "correct": "a"},
    {"scenario": "The memorial service left everyone in the room in quiet grief.",
     "correct": "b"},
])

# -------  mentalizing vs intention  ----------------------------------------
_register("mentalizing", "intention", [
    {"scenario": "{name} observed {name2}'s darting eyes and guessed {name2} was hiding something.",
     "correct": "a"},
    {"scenario": "{name} drafted a detailed project plan with milestones for each quarter.",
     "correct": "b"},
    {"scenario": "The interviewer tried to read the applicant's true attitude behind the rehearsed answers.",
     "correct": "a"},
    {"scenario": "{name} aimed to complete the marathon and began a strict training regimen.",
     "correct": "b"},
    {"scenario": "{name} could tell from {name2}'s tone that {name2} was more upset than {name2} let on.",
     "correct": "a"},
    {"scenario": "{name} set a clear goal to learn {hobby} before the end of the year.",
     "correct": "b"},
    {"scenario": "The detective studied the witness's reactions to determine credibility.",
     "correct": "a"},
    {"scenario": "The committee resolved to allocate extra funding and outlined the next steps.",
     "correct": "b"},
])

# -------  belief vs self_referential  --------------------------------------
_register("belief", "self_referential", [
    {"scenario": "{name} was firmly convinced that the deadline was next week, not this week.",
     "correct": "a"},
    {"scenario": "{name} spent a quiet hour thinking about {subj_pronoun_possessive} own identity and purpose.",
     "correct": "b"},
    {"scenario": "Despite contrary evidence, {name} maintained the belief that the old policy was better.",
     "correct": "a"},
    {"scenario": "{name} examined {subj_pronoun_possessive} own motivations, wondering if {subj_pronoun} was acting out of genuine care or guilt.",
     "correct": "b"},
    {"scenario": "{name} assumed the pharmacy was still on the corner, as it had been for years.",
     "correct": "a"},
    {"scenario": "Looking in the mirror, {name} reflected on how much {subj_pronoun} had changed since college.",
     "correct": "b"},
    {"scenario": "{name} was certain the recipe called for two cups, not three.",
     "correct": "a"},
    {"scenario": "{name} wrote a letter to {subj_pronoun_possessive} future self, describing {subj_pronoun_possessive} current hopes and fears.",
     "correct": "b"},
])

# -------  disgust vs valence  (same as valence vs disgust, reversed) -------
# Already registered.

# -------  mentalizing vs belief  (same as belief vs mentalizing, reversed) --
# Already registered.

# -------  empathy vs theory_of_mind  ---------------------------------------
_register("empathy", "theory_of_mind", [
    {"scenario": "{name} felt a deep swell of compassion watching {name2} struggle alone.",
     "correct": "a"},
    {"scenario": "{name} realised that {name2} didn't know the plan had been changed, while {name3} did.",
     "correct": "b"},
    {"scenario": "{name}'s eyes welled up in sympathy as {name2} described the ordeal.",
     "correct": "a"},
    {"scenario": "{name} hid the truth from {name2}, not knowing that {name3} had already told {name2} everything.",
     "correct": "b"},
    {"scenario": "{name} was moved by a stranger's story and offered to help without being asked.",
     "correct": "a"},
    {"scenario": "{name} told {name2} the restaurant was open, not knowing it had closed. {name3} knew but said nothing.",
     "correct": "b"},
    {"scenario": "Watching the documentary, {name} felt the refugees' anguish as if it were {subj_pronoun_possessive} own.",
     "correct": "a"},
    {"scenario": "{name} was trying to figure out whether {name2} knew about the surprise or was genuinely in the dark.",
     "correct": "b"},
])

# -------  moral vs mentalizing  (same as mentalizing vs moral, reversed) ----
# Already registered.

# -------  disgust vs belief  -----------------------------------------------
_register("disgust", "belief", [
    {"scenario": "{name} retched at the sight of maggots in the bin.",
     "correct": "a"},
    {"scenario": "{name} was utterly convinced the meeting was at noon, though it had been moved to 1pm.",
     "correct": "b"},
    {"scenario": "The putrid smell of the compost pile made {name} gag.",
     "correct": "a"},
    {"scenario": "No matter what the data showed, {name} held firm to the original hypothesis.",
     "correct": "b"},
    {"scenario": "{name} refused to touch the slimy handrail.",
     "correct": "a"},
    {"scenario": "{name} believed the rumour instantly, without seeking any verification.",
     "correct": "b"},
    {"scenario": "The contaminated water made everyone who saw it grimace.",
     "correct": "a"},
    {"scenario": "{name} was sure the museum was free on Tuesdays, even though the policy had changed.",
     "correct": "b"},
])

# -------  disgust vs intention  --------------------------------------------
_register("disgust", "intention", [
    {"scenario": "{name} turned away from the plate when {subj_pronoun} saw a hair in the {food}.",
     "correct": "a"},
    {"scenario": "{name} made a detailed plan to redecorate the living room by the end of the month.",
     "correct": "b"},
    {"scenario": "The foul odour from the dumpster behind the restaurant made {name} lose {subj_pronoun_possessive} appetite.",
     "correct": "a"},
    {"scenario": "{name} resolved to apply for ten jobs this week and started drafting cover letters.",
     "correct": "b"},
    {"scenario": "{name} was revolted by the state of the shared kitchen at the hostel.",
     "correct": "a"},
    {"scenario": "The project lead outlined every milestone for the next two quarters.",
     "correct": "b"},
    {"scenario": "Seeing the rat scurry across the floor, {name} shrieked in revulsion.",
     "correct": "a"},
    {"scenario": "{name} set a strict savings target and automated monthly transfers.",
     "correct": "b"},
])

# -------  disgust vs self_referential  -------------------------------------
_register("disgust", "self_referential", [
    {"scenario": "{name} gagged when {subj_pronoun} opened the bin lid and the stench hit.",
     "correct": "a"},
    {"scenario": "{name} lay awake reflecting on {subj_pronoun_possessive} own childhood and how it shaped who {subj_pronoun} is today.",
     "correct": "b"},
    {"scenario": "The grimy texture of the old sponge made {name} throw it away immediately.",
     "correct": "a"},
    {"scenario": "{name} journaled about {subj_pronoun_possessive} personal growth over the past year.",
     "correct": "b"},
    {"scenario": "{name} was sickened by the squalid state of the abandoned building.",
     "correct": "a"},
    {"scenario": "{name} meditated on {subj_pronoun_possessive} own sense of identity after the major life change.",
     "correct": "b"},
    {"scenario": "The sour smell of the spoiled milk made {name} wrinkle {subj_pronoun_possessive} nose.",
     "correct": "a"},
    {"scenario": "{name} explored {subj_pronoun_possessive} own emotional patterns in therapy, asking why certain things triggered {obj_pronoun}.",
     "correct": "b"},
])

# -------  disgust vs theory_of_mind  ---------------------------------------
_register("disgust", "theory_of_mind", [
    {"scenario": "{name} winced at the repulsive film of grease on the counter.",
     "correct": "a"},
    {"scenario": "{name} knew the gift was second-hand, but {name2} believed it was brand new. {name} kept quiet.",
     "correct": "b"},
    {"scenario": "The rotten vegetables in the fridge made {name} close the door in revulsion.",
     "correct": "a"},
    {"scenario": "{name} pretended not to see {name2}'s mistake, but {name2} later found out {name} had noticed.",
     "correct": "b"},
    {"scenario": "{name} was disgusted by the unsanitary state of the gym changing room.",
     "correct": "a"},
    {"scenario": "{name} and {name2} each believe the other booked the tickets. Neither actually did.",
     "correct": "b"},
    {"scenario": "Stepping in the puddle of something sticky made {name} shudder.",
     "correct": "a"},
    {"scenario": "{name} told {name2} the {food} was fresh, knowing it was from yesterday. {name2} doesn't suspect.",
     "correct": "b"},
])

# -------  sadness vs valence  (same as valence vs sadness, reversed) --------
# Already registered.

# -------  sadness vs self_referential  -------------------------------------
_register("sadness", "self_referential", [
    {"scenario": "{name} cried softly at the graveside, overwhelmed by grief.",
     "correct": "a"},
    {"scenario": "{name} spent the evening reflecting on {subj_pronoun_possessive} own strengths and weaknesses.",
     "correct": "b"},
    {"scenario": "A heavy sorrow filled {name} when the old {object} brought back memories of {name2}.",
     "correct": "a"},
    {"scenario": "{name} considered who {subj_pronoun} was becoming, and whether that person matched {subj_pronoun_possessive} aspirations.",
     "correct": "b"},
    {"scenario": "{name} lay awake feeling the ache of loneliness after {name2} left.",
     "correct": "a"},
    {"scenario": "During the retreat, {name} explored {subj_pronoun_possessive} own core values and personal narrative.",
     "correct": "b"},
    {"scenario": "The departure of the beloved teacher left the whole class in tears.",
     "correct": "a"},
    {"scenario": "{name} wrote a deeply personal letter examining {subj_pronoun_possessive} own fears and hopes.",
     "correct": "b"},
])

# -------  sadness vs belief  -----------------------------------------------
_register("sadness", "belief", [
    {"scenario": "{name} wept uncontrollably at the news of {name2}'s passing.",
     "correct": "a"},
    {"scenario": "{name} remained absolutely sure the post office was open on Saturdays.",
     "correct": "b"},
    {"scenario": "A deep gloom hung over {name} for weeks after the breakup.",
     "correct": "a"},
    {"scenario": "Despite the evidence, {name} believed the vaccine was unnecessary.",
     "correct": "b"},
    {"scenario": "{name} felt a stab of sorrow every time {subj_pronoun} walked past {name2}'s old office.",
     "correct": "a"},
    {"scenario": "{name} was certain the keys were in the drawer, even after checking twice.",
     "correct": "b"},
    {"scenario": "The community mourned together after the flood destroyed their homes.",
     "correct": "a"},
    {"scenario": "{name} assumed the conference was still in March, unaware it had been postponed.",
     "correct": "b"},
])

# -------  sadness vs mentalizing  ------------------------------------------
_register("sadness", "mentalizing", [
    {"scenario": "{name} stared at the empty chair, feeling the ache of absence.",
     "correct": "a"},
    {"scenario": "{name} watched {name2}'s face carefully, trying to read {name2}'s true emotions.",
     "correct": "b"},
    {"scenario": "Grief washed over {name} at the mention of {name2}'s name.",
     "correct": "a"},
    {"scenario": "The therapist tried to decode what the patient's silence really meant.",
     "correct": "b"},
    {"scenario": "{name} couldn't shake the heavy sadness after the memorial.",
     "correct": "a"},
    {"scenario": "{name} analysed {name2}'s tone of voice, sensing hidden frustration.",
     "correct": "b"},
    {"scenario": "Listening to the mournful music, {name} felt tears forming.",
     "correct": "a"},
    {"scenario": "From {name2}'s posture, {name} inferred that {name2} was feeling defeated.",
     "correct": "b"},
])

# -------  sadness vs moral  ------------------------------------------------
_register("sadness", "moral", [
    {"scenario": "{name} sobbed at the sight of the abandoned {animal} in the rain.",
     "correct": "a"},
    {"scenario": "The committee debated whether rationing essential supplies was morally justified.",
     "correct": "b"},
    {"scenario": "An overwhelming melancholy followed {name} through the anniversary week.",
     "correct": "a"},
    {"scenario": "Is it ethically right to withhold difficult news to spare someone's feelings?",
     "correct": "b"},
    {"scenario": "{name} grieved the loss of the friendship, which had ended in silence.",
     "correct": "a"},
    {"scenario": "{name} questioned whether keeping the extra change was theft or just luck.",
     "correct": "b"},
    {"scenario": "The empty playground on a rainy day filled {name} with wistful sorrow.",
     "correct": "a"},
    {"scenario": "Should a company recall a product that might harm a few people, even if most are fine?",
     "correct": "b"},
])

# -------  sadness vs theory_of_mind  ---------------------------------------
_register("sadness", "theory_of_mind", [
    {"scenario": "{name} wept quietly at the train station, missing {name2} already.",
     "correct": "a"},
    {"scenario": "{name} told {name2} the show was sold out, but actually {name} had bought the last tickets. {name2} doesn't know.",
     "correct": "b"},
    {"scenario": "The loss hit {name} hardest at night, when the silence was deafening.",
     "correct": "a"},
    {"scenario": "{name} realised that {name2} thought the office was still open, though it had closed early.",
     "correct": "b"},
    {"scenario": "{name} broke down reading the letter from {name2}, who had passed away months ago.",
     "correct": "a"},
    {"scenario": "{name2} assumed {name} hadn't seen the email, but {name} had read it and chosen not to reply.",
     "correct": "b"},
    {"scenario": "The haunting melody brought back memories, and {name} felt a wave of sorrow.",
     "correct": "a"},
    {"scenario": "{name} gave {name2} directions to the old building, forgetting it had been demolished. {name2} doesn't know.",
     "correct": "b"},
])

# -------  sadness vs disgust  (already registered as disgust vs sadness) ----
# Already registered.

# -------  happiness vs belief  ---------------------------------------------
_register("happiness", "belief", [
    {"scenario": "{name} burst into joyful tears when the acceptance letter arrived.",
     "correct": "a"},
    {"scenario": "{name} was certain the bank closed at 4pm, even though it now closes at 3pm.",
     "correct": "b"},
    {"scenario": "The whole room erupted in cheers and hugs when the results were announced.",
     "correct": "a"},
    {"scenario": "No one could convince {name} that the old route was no longer the fastest.",
     "correct": "b"},
    {"scenario": "{name} beamed with delight as {name2} walked through the door.",
     "correct": "a"},
    {"scenario": "{name} held the unshakeable conviction that the investment would pay off.",
     "correct": "b"},
    {"scenario": "{name} was so happy {subj_pronoun} danced around the kitchen after the phone call.",
     "correct": "a"},
    {"scenario": "Despite all signs to the contrary, {name} believed the project was on track.",
     "correct": "b"},
])

# -------  happiness vs mentalizing  ----------------------------------------
_register("happiness", "mentalizing", [
    {"scenario": "{name} grinned uncontrollably as {subj_pronoun} unwrapped the gift.",
     "correct": "a"},
    {"scenario": "{name} tried to decipher whether {name2}'s smile was genuine or forced.",
     "correct": "b"},
    {"scenario": "The crowd broke into spontaneous, joyful applause at the finale.",
     "correct": "a"},
    {"scenario": "{name} studied {name2}'s reaction carefully, sensing hidden anxiety beneath the cheer.",
     "correct": "b"},
    {"scenario": "{name} radiated pure delight holding the newborn for the first time.",
     "correct": "a"},
    {"scenario": "{name} noticed {name2}'s voice waver and suspected {name2} was masking disappointment.",
     "correct": "b"},
    {"scenario": "{name2} surprised {name} with a party, and {name} was overcome with joy.",
     "correct": "a"},
    {"scenario": "The investigator tried to determine whether the witness was lying or genuinely confused.",
     "correct": "b"},
])

# -------  happiness vs intention  ------------------------------------------
_register("happiness", "intention", [
    {"scenario": "{name} was brimming with joy after the unexpected reunion {setting}.",
     "correct": "a"},
    {"scenario": "{name} drew up a detailed plan to renovate the kitchen by summer.",
     "correct": "b"},
    {"scenario": "Laughter and cheers filled the room as the team celebrated the victory.",
     "correct": "a"},
    {"scenario": "{name} resolved to read one book a week and set up a reading list.",
     "correct": "b"},
    {"scenario": "{name} couldn't stop grinning after the surprise proposal.",
     "correct": "a"},
    {"scenario": "The director outlined a clear strategy for the product launch next quarter.",
     "correct": "b"},
    {"scenario": "Pure elation washed over {name} as the final whistle blew.",
     "correct": "a"},
    {"scenario": "{name} carefully budgeted {subj_pronoun_possessive} expenses and set a monthly savings target.",
     "correct": "b"},
])

# -------  happiness vs self_referential  -----------------------------------
_register("happiness", "self_referential", [
    {"scenario": "{name} laughed with unbridled joy as the children played {setting}.",
     "correct": "a"},
    {"scenario": "{name} spent a quiet hour reflecting on {subj_pronoun_possessive} personal growth and evolving values.",
     "correct": "b"},
    {"scenario": "The news of the scholarship filled {name} with radiant happiness.",
     "correct": "a"},
    {"scenario": "{name} examined {subj_pronoun_possessive} own habits and asked whether they aligned with {subj_pronoun_possessive} goals.",
     "correct": "b"},
    {"scenario": "{name} danced with delight after receiving the good news.",
     "correct": "a"},
    {"scenario": "{name} wrote in a journal about {subj_pronoun_possessive} sense of self and where {subj_pronoun} was heading.",
     "correct": "b"},
    {"scenario": "Everyone at the surprise party cheered, and {name} was overcome with happiness.",
     "correct": "a"},
    {"scenario": "In the mirror, {name} assessed how {subj_pronoun} had changed over the past five years.",
     "correct": "b"},
])

# -------  happiness vs moral  ----------------------------------------------
_register("happiness", "moral", [
    {"scenario": "{name} was overjoyed when the adoption was finally approved.",
     "correct": "a"},
    {"scenario": "The board debated the ethics of using AI to replace customer-service staff.",
     "correct": "b"},
    {"scenario": "{name} beamed as {name2} accepted the proposal {setting}.",
     "correct": "a"},
    {"scenario": "Should a company prioritise profits or employee well-being in a downturn?",
     "correct": "b"},
    {"scenario": "The classroom erupted in happy cheers when the teacher announced the field trip.",
     "correct": "a"},
    {"scenario": "{name} wrestled with whether downloading the leaked document was morally defensible.",
     "correct": "b"},
    {"scenario": "{name} felt a rush of joy watching the sunrise from the summit.",
     "correct": "a"},
    {"scenario": "Is it right to expose a friend's secret to protect someone else from harm?",
     "correct": "b"},
])

# -------  happiness vs empathy  (already registered as anger vs empathy-
# style; need separate) ---
_register("happiness", "empathy", [
    {"scenario": "{name} was absolutely thrilled to receive the award in front of everyone.",
     "correct": "a"},
    {"scenario": "{name} felt {name2}'s distress so strongly it brought {name} to tears.",
     "correct": "b"},
    {"scenario": "A wave of pure delight swept through {name} at the concert's finale.",
     "correct": "a"},
    {"scenario": "Listening to {name2}'s story of hardship, {name} was moved to deep compassion.",
     "correct": "b"},
    {"scenario": "{name} danced with joy after the long-awaited call.",
     "correct": "a"},
    {"scenario": "{name} reached out to comfort {name2}, feeling every bit of {name2}'s pain.",
     "correct": "b"},
    {"scenario": "The surprise gift left {name} speechless with happiness.",
     "correct": "a"},
    {"scenario": "Seeing the injured child, {name} felt an overwhelming urge to help, eyes glistening.",
     "correct": "b"},
])

# -------  happiness vs judgment  -------------------------------------------
_register("happiness", "judgment", [
    {"scenario": "{name} was giddy with joy as the confetti fell at the New Year celebration.",
     "correct": "a"},
    {"scenario": "{name} carefully weighed the merits of each candidate before making a decision.",
     "correct": "b"},
    {"scenario": "The team leapt to their feet in ecstatic celebration after the final goal.",
     "correct": "a"},
    {"scenario": "The appraiser assessed the artwork's quality and assigned a fair market value.",
     "correct": "b"},
    {"scenario": "{name} was so happy {subj_pronoun} hugged every person in the room.",
     "correct": "a"},
    {"scenario": "The editor evaluated each submission for clarity, originality, and rigour.",
     "correct": "b"},
    {"scenario": "{name} glowed with contentment during the family picnic {setting}.",
     "correct": "a"},
    {"scenario": "After reviewing all options, {name} concluded that only one met the quality threshold.",
     "correct": "b"},
])

# -------  fear vs empathy  -------------------------------------------------
_register("fear", "empathy", [
    {"scenario": "{name} was petrified as the elevator jolted to a halt between floors.",
     "correct": "a"},
    {"scenario": "{name} felt a deep ache of compassion listening to {name2}'s story of loss.",
     "correct": "b"},
    {"scenario": "The sudden blackout sent a wave of panic through the crowd.",
     "correct": "a"},
    {"scenario": "{name} was moved to tears by the stranger's account of survival.",
     "correct": "b"},
    {"scenario": "{name}'s legs shook as {subj_pronoun} approached the high ledge.",
     "correct": "a"},
    {"scenario": "{name} held {name2}'s hand, genuinely sharing {name2}'s grief.",
     "correct": "b"},
    {"scenario": "The howl of wind outside made {name} pull the covers over {subj_pronoun_possessive} head in dread.",
     "correct": "a"},
    {"scenario": "Watching {name2} struggle, {name} was filled with sympathetic pain.",
     "correct": "b"},
])

# -------  fear vs belief  --------------------------------------------------
_register("fear", "belief", [
    {"scenario": "{name} was paralysed with fear when the floorboards creaked at midnight.",
     "correct": "a"},
    {"scenario": "{name} held the firm conviction that the old road was still the fastest route.",
     "correct": "b"},
    {"scenario": "The earthquake drill sent real tremors of anxiety through the office.",
     "correct": "a"},
    {"scenario": "Despite the GPS, {name} believed the turn should have been earlier.",
     "correct": "b"},
    {"scenario": "{name} felt pure dread looking down from the observation deck.",
     "correct": "a"},
    {"scenario": "{name} assumed the water was safe to drink because it always had been.",
     "correct": "b"},
    {"scenario": "The siren blared and {name} ran for cover, heart hammering.",
     "correct": "a"},
    {"scenario": "{name} clung to the belief that the company would never close, despite the layoffs.",
     "correct": "b"},
])

# -------  fear vs intention  -----------------------------------------------
_register("fear", "intention", [
    {"scenario": "{name} cowered behind the door after hearing the intruder.",
     "correct": "a"},
    {"scenario": "{name} outlined a plan to run a half-marathon and hired a coach.",
     "correct": "b"},
    {"scenario": "A bolt of lightning made {name} scream and cover {subj_pronoun_possessive} ears.",
     "correct": "a"},
    {"scenario": "{name} committed to learning three new recipes each month.",
     "correct": "b"},
    {"scenario": "{name} broke into a cold sweat as the turbulence got worse.",
     "correct": "a"},
    {"scenario": "{name} decided to move abroad and began gathering visa documents.",
     "correct": "b"},
    {"scenario": "The rustling in the bushes made {name} quicken {subj_pronoun_possessive} pace nervously.",
     "correct": "a"},
    {"scenario": "The team set a target date and assigned each task to a specific member.",
     "correct": "b"},
])

# -------  fear vs self_referential  ----------------------------------------
_register("fear", "self_referential", [
    {"scenario": "{name} trembled as the plane hit a pocket of turbulence.",
     "correct": "a"},
    {"scenario": "{name} reflected deeply on how {subj_pronoun_possessive} own insecurities shaped {subj_pronoun_possessive} decisions.",
     "correct": "b"},
    {"scenario": "The footsteps behind {name} in the dark parking garage made {subj_pronoun_possessive} blood freeze.",
     "correct": "a"},
    {"scenario": "{name} asked {subj_pronoun_possessive}self what kind of legacy {subj_pronoun} wanted to leave behind.",
     "correct": "b"},
    {"scenario": "{name} jumped when the door slammed shut in the wind.",
     "correct": "a"},
    {"scenario": "During the long walk, {name} contemplated {subj_pronoun_possessive} own strengths and limitations.",
     "correct": "b"},
    {"scenario": "The spider dangling above the bed made {name} shriek.",
     "correct": "a"},
    {"scenario": "{name} sat with {subj_pronoun_possessive} journal, exploring what truly motivated {obj_pronoun} in life.",
     "correct": "b"},
])

# -------  fear vs mentalizing  ---------------------------------------------
_register("fear", "mentalizing", [
    {"scenario": "{name} bolted upright at the sound of shattering glass downstairs.",
     "correct": "a"},
    {"scenario": "{name} watched {name2}'s reaction closely, trying to read whether {name2} was pleased or just polite.",
     "correct": "b"},
    {"scenario": "The rickety bridge swaying over the gorge terrified {name}.",
     "correct": "a"},
    {"scenario": "From {name2}'s clipped replies, {name} inferred that {name2} was irritated.",
     "correct": "b"},
    {"scenario": "{name} gasped as the car ahead swerved dangerously.",
     "correct": "a"},
    {"scenario": "{name} noticed {name2}'s tight smile and guessed {name2} was masking disappointment.",
     "correct": "b"},
    {"scenario": "{name} lay awake terrified by the approaching storm.",
     "correct": "a"},
    {"scenario": "The teacher sensed the student's confusion from the puzzled look.",
     "correct": "b"},
])

# -------  fear vs theory_of_mind  ------------------------------------------
_register("fear", "theory_of_mind", [
    {"scenario": "{name} screamed when the shadow moved across the wall at night.",
     "correct": "a"},
    {"scenario": "{name} knew the road was closed, but {name2} was driving there, unaware of the detour.",
     "correct": "b"},
    {"scenario": "The sudden power outage plunged the room into darkness, and {name} panicked.",
     "correct": "a"},
    {"scenario": "{name} told {name2} the exam was easy, not knowing {name2} had a completely different version.",
     "correct": "b"},
    {"scenario": "{name} was terrified of the {animal} that appeared on the trail.",
     "correct": "a"},
    {"scenario": "{name} believed {name2} was at home, but {name2} had actually left an hour ago.",
     "correct": "b"},
    {"scenario": "The ground shook and {name} dove under the table in terror.",
     "correct": "a"},
    {"scenario": "{name2} thinks the surprise party is tomorrow, but {name} moved it to today.",
     "correct": "b"},
])

# -------  disgust vs moral  ------------------------------------------------
_register("disgust", "moral", [
    {"scenario": "{name} retched at the sight of the overflowing sewage drain.",
     "correct": "a"},
    {"scenario": "The panel debated whether mandatory vaccination was ethically justifiable.",
     "correct": "b"},
    {"scenario": "Opening the bin, {name} was hit by a wave of nauseating stench.",
     "correct": "a"},
    {"scenario": "Should a journalist expose corruption if it means violating a source's privacy?",
     "correct": "b"},
    {"scenario": "{name} backed away in revulsion from the mouldy, dripping wall.",
     "correct": "a"},
    {"scenario": "{name} agonised over whether returning the found money was the right thing to do.",
     "correct": "b"},
    {"scenario": "The contaminated pond made {name} cover {subj_pronoun_possessive} nose and look away.",
     "correct": "a"},
    {"scenario": "Is it moral to deceive a patient about a terminal diagnosis to reduce their suffering?",
     "correct": "b"},
])

# -------  disgust vs judgment  ---------------------------------------------
_register("disgust", "judgment", [
    {"scenario": "{name} grimaced at the sour smell of the expired {food}.",
     "correct": "a"},
    {"scenario": "After careful review, {name} rated the application as below the required standard.",
     "correct": "b"},
    {"scenario": "The greasy residue on the plate made {name} push the dish away.",
     "correct": "a"},
    {"scenario": "The panel weighed each contestant's performance against the scoring rubric.",
     "correct": "b"},
    {"scenario": "{name} was revolted by the unclean conditions in the hostel.",
     "correct": "a"},
    {"scenario": "{name} concluded that the vendor's price was unreasonable after comparing quotes.",
     "correct": "b"},
    {"scenario": "The mouldy ceiling tiles made {name} question the building's safety.",
     "correct": "a"},
    {"scenario": "The judge assessed whether the evidence met the threshold for conviction.",
     "correct": "b"},
])

# -------  anger vs judgment  -----------------------------------------------
_register("anger", "judgment", [
    {"scenario": "{name} was shaking with rage after the confrontation {setting}.",
     "correct": "a"},
    {"scenario": "{name} carefully weighed the two competing bids and chose the better one.",
     "correct": "b"},
    {"scenario": "{name} hurled insults across the room in a fit of fury.",
     "correct": "a"},
    {"scenario": "After reviewing the evidence, {name} concluded the claim was unsubstantiated.",
     "correct": "b"},
    {"scenario": "The unfair dismissal left {name} seething with hostile anger.",
     "correct": "a"},
    {"scenario": "The panel assessed each entry against strict criteria and ranked them.",
     "correct": "b"},
    {"scenario": "{name} punched the wall in frustration after the argument with {name2}.",
     "correct": "a"},
    {"scenario": "{name} determined that the proposed settlement was fair to all parties.",
     "correct": "b"},
])

# -------  anger vs intention  ----------------------------------------------
_register("anger", "intention", [
    {"scenario": "{name} was seething with fury after the blatant lie.",
     "correct": "a"},
    {"scenario": "{name} mapped out a six-month plan to learn {hobby} from scratch.",
     "correct": "b"},
    {"scenario": "Rage flashed across {name}'s face as the insult landed.",
     "correct": "a"},
    {"scenario": "{name} deliberately chose the harder route because it was more efficient.",
     "correct": "b"},
    {"scenario": "{name} slammed the laptop shut in frustration after losing the file.",
     "correct": "a"},
    {"scenario": "The committee set a clear objective and assigned action items for the next sprint.",
     "correct": "b"},
    {"scenario": "{name} was so angry {subj_pronoun} could barely speak.",
     "correct": "a"},
    {"scenario": "{name} resolved to cook dinner every night this month and prepared a menu.",
     "correct": "b"},
])

# -------  anger vs mentalizing  --------------------------------------------
_register("anger", "mentalizing", [
    {"scenario": "{name} hurled {subj_pronoun_possessive} keys at the wall in a rage after the phone call.",
     "correct": "a"},
    {"scenario": "{name} watched {name2}'s posture shift and sensed hidden unease.",
     "correct": "b"},
    {"scenario": "Fury erupted in {name} when {subj_pronoun} read the unfair review.",
     "correct": "a"},
    {"scenario": "The manager read between the lines of {name2}'s email, picking up on subtle resentment.",
     "correct": "b"},
    {"scenario": "{name} stormed out of the meeting, too angry to continue.",
     "correct": "a"},
    {"scenario": "{name} wondered whether {name2} was being sarcastic or genuinely supportive.",
     "correct": "b"},
    {"scenario": "{name} ripped up the notice in a burst of hostility.",
     "correct": "a"},
    {"scenario": "The counsellor observed the client's clenched jaw and inferred deep frustration.",
     "correct": "b"},
])

# -------  anger vs moral  --------------------------------------------------
_register("anger", "moral", [
    {"scenario": "{name} was fuming after being cut off in traffic for the third time.",
     "correct": "a"},
    {"scenario": "The ethics committee debated whether the study's design respected participants' autonomy.",
     "correct": "b"},
    {"scenario": "{name} punched the pillow, overwhelmed by rage at the betrayal.",
     "correct": "a"},
    {"scenario": "Is it right to break the law to prevent greater injustice?",
     "correct": "b"},
    {"scenario": "After the public humiliation, {name} was consumed by white-hot anger.",
     "correct": "a"},
    {"scenario": "{name} wrestled with whether exposing the fraud was the morally correct action.",
     "correct": "b"},
    {"scenario": "{name} shouted at the receptionist, unable to contain the hostility.",
     "correct": "a"},
    {"scenario": "Should a doctor refuse to treat a patient whose actions caused harm to others?",
     "correct": "b"},
])

# -------  anger vs self_referential  ---------------------------------------
_register("anger", "self_referential", [
    {"scenario": "{name} threw {subj_pronoun_possessive} bag down in a fit of fury at the cancelled flight.",
     "correct": "a"},
    {"scenario": "{name} spent the morning reflecting on {subj_pronoun_possessive} own emotional triggers.",
     "correct": "b"},
    {"scenario": "{name} yelled at {name2} in a moment of blind rage.",
     "correct": "a"},
    {"scenario": "{name} asked {subj_pronoun_possessive}self whether {subj_pronoun} was making decisions based on {subj_pronoun_possessive} values or {subj_pronoun_possessive} fears.",
     "correct": "b"},
    {"scenario": "The simmering hostility finally boiled over, and {name} lashed out.",
     "correct": "a"},
    {"scenario": "{name} examined {subj_pronoun_possessive} own biases before writing the review.",
     "correct": "b"},
    {"scenario": "{name} kicked over the bin in frustration.",
     "correct": "a"},
    {"scenario": "{name} journaled about {subj_pronoun_possessive} sense of purpose and where it came from.",
     "correct": "b"},
])

# -------  anger vs theory_of_mind  -----------------------------------------
_register("anger", "theory_of_mind", [
    {"scenario": "{name} snapped at {name2} in a burst of irritation.",
     "correct": "a"},
    {"scenario": "{name} realised {name2} didn't know the meeting had been cancelled, and debated whether to tell.",
     "correct": "b"},
    {"scenario": "{name} was livid that {subj_pronoun_possessive} contribution was not acknowledged.",
     "correct": "a"},
    {"scenario": "{name} figured out that {name2} still believed the old deadline, which had been extended.",
     "correct": "b"},
    {"scenario": "The hostile exchange left {name} trembling with rage.",
     "correct": "a"},
    {"scenario": "{name} knew the truth about the surprise, but acted ignorant so {name2} wouldn't know {name} knew.",
     "correct": "b"},
    {"scenario": "{name} was furious at the unfair verdict.",
     "correct": "a"},
    {"scenario": "{name2} thinks {name} is still at work, but {name} left early. {name2} has no idea.",
     "correct": "b"},
])

# -------  valence vs empathy  ----------------------------------------------
_register("valence", "empathy", [
    {"scenario": "{name} rated {subj_pronoun_possessive} overall emotional state for the week as moderately positive.",
     "correct": "a"},
    {"scenario": "{name} felt a surge of compassion watching {name2} break down in tears.",
     "correct": "b"},
    {"scenario": "The survey asked participants to mark their general mood on a scale from very negative to very positive.",
     "correct": "a"},
    {"scenario": "{name} was moved to tears by the refugee's story of resilience.",
     "correct": "b"},
    {"scenario": "{name} noticed {subj_pronoun_possessive} overall affect had shifted from neutral to mildly positive over the afternoon.",
     "correct": "a"},
    {"scenario": "{name} held {name2}'s hand, sharing every moment of {name2}'s distress.",
     "correct": "b"},
    {"scenario": "The app logged {name}'s daily emotional valence as slightly below average.",
     "correct": "a"},
    {"scenario": "Seeing the family reunited after the ordeal, {name} was overwhelmed with fellow-feeling.",
     "correct": "b"},
])

# -------  valence vs mentalizing  ------------------------------------------
_register("valence", "mentalizing", [
    {"scenario": "{name} described {subj_pronoun_possessive} day's overall emotional tone as mildly negative.",
     "correct": "a"},
    {"scenario": "{name} studied {name2}'s tone, trying to determine whether {name2} was genuinely fine or putting on a front.",
     "correct": "b"},
    {"scenario": "The therapist asked {name} to place {subj_pronoun_possessive} current mood on a positivity scale.",
     "correct": "a"},
    {"scenario": "From {name2}'s averted gaze, {name} sensed that {name2} was withholding something.",
     "correct": "b"},
    {"scenario": "{name} logged {subj_pronoun_possessive} emotional valence as neutral in the wellness app.",
     "correct": "a"},
    {"scenario": "The manager observed {name2}'s hunched shoulders and inferred low morale.",
     "correct": "b"},
    {"scenario": "Overall, {name} felt the meeting had a generally positive emotional atmosphere.",
     "correct": "a"},
    {"scenario": "{name} tried to read the subtext of {name2}'s carefully worded response.",
     "correct": "b"},
])

# -------  valence vs intention  --------------------------------------------
_register("valence", "intention", [
    {"scenario": "{name} characterised {subj_pronoun_possessive} overall emotional state this morning as slightly negative.",
     "correct": "a"},
    {"scenario": "{name} set a clear target to finish the proposal by end of day.",
     "correct": "b"},
    {"scenario": "The group's collective mood was measured as generally positive on the exit survey.",
     "correct": "a"},
    {"scenario": "{name} resolved to start jogging three times a week and bought new trainers.",
     "correct": "b"},
    {"scenario": "{name} reflected that the day had felt emotionally neutral overall.",
     "correct": "a"},
    {"scenario": "{name} planned every step of the event, from invitations to cleanup.",
     "correct": "b"},
    {"scenario": "The counsellor noted that {name}'s baseline emotional tone was mildly positive.",
     "correct": "a"},
    {"scenario": "{name} decided to reorganise the workshop and created a detailed timeline.",
     "correct": "b"},
])

# -------  valence vs belief  -----------------------------------------------
_register("valence", "belief", [
    {"scenario": "{name} noticed {subj_pronoun_possessive} mood had been consistently mildly negative all week.",
     "correct": "a"},
    {"scenario": "{name} was absolutely certain the bank branch was still open on Saturdays.",
     "correct": "b"},
    {"scenario": "The diary entry described {name}'s overall emotional tone as pleasant.",
     "correct": "a"},
    {"scenario": "Despite evidence, {name} maintained the conviction that the old method was superior.",
     "correct": "b"},
    {"scenario": "Participants rated their general affective state after the experiment.",
     "correct": "a"},
    {"scenario": "{name} assumed the train schedule hadn't changed since last year.",
     "correct": "b"},
    {"scenario": "{name} described the week's emotional climate as somewhat positive overall.",
     "correct": "a"},
    {"scenario": "{name} was sure the store still accepted the old coupon.",
     "correct": "b"},
])

# -------  valence vs self_referential  -------------------------------------
_register("valence", "self_referential", [
    {"scenario": "{name} rated {subj_pronoun_possessive} overall mood for the day as mildly positive on the wellness app.",
     "correct": "a"},
    {"scenario": "{name} sat quietly and reflected on {subj_pronoun_possessive} own core values and sense of identity.",
     "correct": "b"},
    {"scenario": "The exit survey measured the group's collective emotional tone as neutral.",
     "correct": "a"},
    {"scenario": "{name} wrote in {subj_pronoun_possessive} journal about how {subj_pronoun} had grown as a person over the past year.",
     "correct": "b"},
    {"scenario": "{name} described the concert's overall emotional atmosphere as overwhelmingly positive.",
     "correct": "a"},
    {"scenario": "{name} examined {subj_pronoun_possessive} own motivations, wondering if they were genuine.",
     "correct": "b"},
    {"scenario": "The therapist noted that {name}'s baseline affective state had improved from negative to neutral.",
     "correct": "a"},
    {"scenario": "{name} pondered who {subj_pronoun} was becoming and whether it matched {subj_pronoun_possessive} ideals.",
     "correct": "b"},
])

# -------  valence vs moral  ------------------------------------------------
_register("valence", "moral", [
    {"scenario": "{name} noted {subj_pronoun_possessive} general emotional state had been mildly negative all afternoon.",
     "correct": "a"},
    {"scenario": "The panel debated whether the policy was morally justifiable given its unequal impact.",
     "correct": "b"},
    {"scenario": "Participants' overall affect was measured on a positive-to-negative slider.",
     "correct": "a"},
    {"scenario": "Is it ethical to sacrifice individual rights for collective safety?",
     "correct": "b"},
    {"scenario": "{name} described the event's emotional tone as moderately pleasant.",
     "correct": "a"},
    {"scenario": "{name} struggled with whether accepting the inheritance was morally right.",
     "correct": "b"},
    {"scenario": "The mood diary showed a consistently neutral emotional valence for the week.",
     "correct": "a"},
    {"scenario": "Should a government surveil citizens to prevent terrorism?",
     "correct": "b"},
])

# -------  valence vs judgment  ---------------------------------------------
_register("valence", "judgment", [
    {"scenario": "{name} described {subj_pronoun_possessive} overall feeling about the semester as mildly positive.",
     "correct": "a"},
    {"scenario": "{name} carefully evaluated each candidate's portfolio and ranked them by quality.",
     "correct": "b"},
    {"scenario": "The feedback form asked attendees to rate their general emotional response to the event.",
     "correct": "a"},
    {"scenario": "After deliberation, the committee determined the proposal failed to meet minimum standards.",
     "correct": "b"},
    {"scenario": "{name} reflected that the weekend had a generally upbeat emotional character.",
     "correct": "a"},
    {"scenario": "The reviewer assessed the manuscript's strengths and weaknesses and gave a verdict.",
     "correct": "b"},
    {"scenario": "{name}'s mood tracker showed a slow drift from negative to neutral over the month.",
     "correct": "a"},
    {"scenario": "The inspector judged the facilities to be satisfactory overall.",
     "correct": "b"},
])

# -------  valence vs theory_of_mind  ---------------------------------------
_register("valence", "theory_of_mind", [
    {"scenario": "{name} assessed {subj_pronoun_possessive} overall emotional tone as moderately negative.",
     "correct": "a"},
    {"scenario": "{name} realised {name2} still believed the old schedule, even though it had been updated.",
     "correct": "b"},
    {"scenario": "The wellness check showed participants' general affect was mildly positive.",
     "correct": "a"},
    {"scenario": "{name} sent {name2} an apology, not knowing {name2} never received the original message and was confused.",
     "correct": "b"},
    {"scenario": "{name} described the meeting's emotional atmosphere as flat and neutral.",
     "correct": "a"},
    {"scenario": "{name2} assumes {name} is at the airport, but {name} is still at home packing.",
     "correct": "b"},
    {"scenario": "{name} rated the film's emotional impact as mildly positive overall.",
     "correct": "a"},
    {"scenario": "{name} knows the party is a surprise, but {name2} overheard the planning and is pretending not to know.",
     "correct": "b"},
])

# -------  self_referential vs belief  (already registered as belief vs self_referential) ----
# -------  self_referential vs moral  (already registered as moral vs self_referential) -----
# -------  judgment vs mentalizing  -----------------------------------------
_register("judgment", "mentalizing", [
    {"scenario": "The hiring committee scored each candidate against a standardised rubric.",
     "correct": "a"},
    {"scenario": "{name} watched {name2}'s fidgeting and guessed {name2} was more anxious than {name2} let on.",
     "correct": "b"},
    {"scenario": "{name} evaluated the two proposals and determined which offered better value.",
     "correct": "a"},
    {"scenario": "The therapist noticed a catch in {name2}'s voice and probed deeper.",
     "correct": "b"},
    {"scenario": "After inspecting the work, {name} judged it to meet professional standards.",
     "correct": "a"},
    {"scenario": "{name} read {name2}'s body language, inferring that {name2} disagreed despite saying otherwise.",
     "correct": "b"},
    {"scenario": "The panel assessed the grant applications on scientific merit and feasibility.",
     "correct": "a"},
    {"scenario": "{name} could tell {name2} was lying from the way {name2} avoided eye contact.",
     "correct": "b"},
])

# -------  judgment vs self_referential  ------------------------------------
_register("judgment", "self_referential", [
    {"scenario": "{name} weighed the evidence and concluded the offer was fair.",
     "correct": "a"},
    {"scenario": "{name} reflected on {subj_pronoun_possessive} own character and whether {subj_pronoun} was living authentically.",
     "correct": "b"},
    {"scenario": "The critic assessed the performance and gave it a rating of eight out of ten.",
     "correct": "a"},
    {"scenario": "{name} meditated on {subj_pronoun_possessive} own fears, tracing them back to childhood experiences.",
     "correct": "b"},
    {"scenario": "The board evaluated the report's quality and approved it for publication.",
     "correct": "a"},
    {"scenario": "{name} journaled about what kind of parent {subj_pronoun} wanted to be.",
     "correct": "b"},
    {"scenario": "{name} compared the bids and judged the lowest one to be unrealistically cheap.",
     "correct": "a"},
    {"scenario": "{name} examined {subj_pronoun_possessive} own biases before entering the discussion.",
     "correct": "b"},
])

# -------  judgment vs theory_of_mind  --------------------------------------
_register("judgment", "theory_of_mind", [
    {"scenario": "The panel scored each essay on originality and argumentation.",
     "correct": "a"},
    {"scenario": "{name} discovered that {name2} didn't know the rules had changed, while everyone else did.",
     "correct": "b"},
    {"scenario": "{name} determined that the contract terms were unfavourable and recommended renegotiation.",
     "correct": "a"},
    {"scenario": "{name} told {name2} the test was straightforward, not knowing {name2} had a different version.",
     "correct": "b"},
    {"scenario": "After reviewing the proposals, {name} concluded only two met the eligibility criteria.",
     "correct": "a"},
    {"scenario": "{name} bought a gift for {name2}, not realising {name2} had already bought the same item.",
     "correct": "b"},
    {"scenario": "The reviewer rated the submission's methodology as rigorous.",
     "correct": "a"},
    {"scenario": "{name} assumed {name2} knew the meeting was postponed, but {name2} showed up on the original date.",
     "correct": "b"},
])

# -------  mentalizing vs empathy  (already registered as empathy vs mentalizing) ----
# -------  mentalizing vs happiness  (already registered as happiness vs mentalizing) --
# -------  mentalizing vs sadness  (already registered as sadness vs mentalizing) ------
# -------  mentalizing vs fear  (already registered as fear vs mentalizing) -----
# -------  mentalizing vs disgust  (already registered as disgust vs mentalizing) ---
# -------  mentalizing vs judgment  (already registered) ---
# -------  mentalizing vs valence  (already registered as valence vs mentalizing) ---

# -------  self_referential vs empathy  (already registered as empathy vs self_referential) --
# -------  self_referential vs intention  (already registered as intention vs self_referential) --
# -------  self_referential vs sadness  (already registered as sadness vs self_referential) ---
# -------  self_referential vs happiness  (already registered as happiness vs self_referential) --
# -------  self_referential vs fear  (already registered as fear vs self_referential) ----
# -------  self_referential vs disgust  (already registered as disgust vs self_referential) ---
# -------  self_referential vs mentalizing  (already registered as mentalizing vs self_referential) ---
# -------  self_referential vs theory_of_mind  (already registered) ---
# -------  self_referential vs valence  (already registered as valence vs self_referential) ---
# -------  self_referential vs judgment  (already registered as judgment vs self_referential) ---

# -------  moral vs belief  -------------------------------------------------
_register("moral", "belief", [
    {"scenario": "{name} debated whether keeping the secret was the right thing to do.",
     "correct": "a"},
    {"scenario": "{name} was completely convinced the meeting was in room 201, not 202.",
     "correct": "b"},
    {"scenario": "The review board questioned whether the experiment's methods were ethical.",
     "correct": "a"},
    {"scenario": "Despite the announcement, {name} believed the old policy still applied.",
     "correct": "b"},
    {"scenario": "Is it right to sacrifice transparency for efficiency in governance?",
     "correct": "a"},
    {"scenario": "{name} assumed without question that the product was safe.",
     "correct": "b"},
    {"scenario": "{name} wrestled with the morality of accepting the inheritance from a questionable source.",
     "correct": "a"},
    {"scenario": "{name} held the firm belief that the local team would win, despite all odds.",
     "correct": "b"},
])

# -------  moral vs intention  (already registered as intention vs moral) ----
# -------  moral vs fear  (already registered as fear vs moral) ----
# -------  moral vs happiness  (already registered as happiness vs moral) ----
# -------  moral vs sadness  (already registered as sadness vs moral) ----
# -------  moral vs disgust  (already registered as disgust vs moral) ----
# -------  moral vs theory_of_mind  ----------------------------------------
_register("moral", "theory_of_mind", [
    {"scenario": "The ethics commission debated whether informed consent was truly obtained.",
     "correct": "a"},
    {"scenario": "{name} knew {name2} was about to walk into a trap, but {name2} had no idea.",
     "correct": "b"},
    {"scenario": "Is it morally permissible to deceive someone for their own protection?",
     "correct": "a"},
    {"scenario": "{name} assumed {name2} understood the terms, but {name2} had a completely different interpretation.",
     "correct": "b"},
    {"scenario": "{name} questioned whether the company's cost-cutting was ethical.",
     "correct": "a"},
    {"scenario": "{name} told {name2} one thing and {name3} another. Neither knows the other's version.",
     "correct": "b"},
    {"scenario": "Should individual freedom be limited to protect communal well-being?",
     "correct": "a"},
    {"scenario": "{name2} thought the package was a gift, but {name} had actually sent it by mistake. {name} doesn't know {name2} received it.",
     "correct": "b"},
])

# -------  moral vs valence  (already registered as valence vs moral) --------
# -------  moral vs judgment  (already registered as judgment vs moral) ------
# -------  moral vs mentalizing  (already registered as mentalizing vs moral) -

# -------  empathy vs belief  (already registered as belief vs empathy) ------
# -------  empathy vs fear  (already registered as fear vs empathy) ----------
# -------  empathy vs happiness  (already registered as happiness vs empathy) -
# -------  empathy vs judgment  (already registered as judgment vs empathy) ---
# -------  empathy vs moral  (already registered as moral vs empathy) --------
# -------  empathy vs disgust  (already registered as disgust vs empathy) ----
# -------  empathy vs valence  (already registered as valence vs empathy) ----


# ---------------------------------------------------------------------------
# Template instantiation engine
# ---------------------------------------------------------------------------

def _pick_gendered_name(rng: random.Random, used: set) -> tuple[str, str, str, str, str]:
    """Return (name, subj_pronoun, obj_pronoun, subj_pronoun_possessive, gender) avoiding used."""
    available_m = [n for n in NAMES_MALE if n not in used]
    available_f = [n for n in NAMES_FEMALE if n not in used]
    available = available_m + available_f
    if not available:
        available = NAMES_ALL[:]
    name = rng.choice(available)
    used.add(name)
    if name in NAMES_FEMALE:
        return name, "she", "her", "her", "f"
    return name, "he", "him", "his", "m"


def instantiate_template(template: dict, rng: random.Random, cond_a: str, cond_b: str) -> dict:
    """Fill a template with random slot values and return a complete item."""
    used_names: set[str] = set()
    n1, sp1, op1, pp1, _ = _pick_gendered_name(rng, used_names)
    n2, sp2, op2, pp2, _ = _pick_gendered_name(rng, used_names)
    n3, sp3, op3, pp3, _ = _pick_gendered_name(rng, used_names)

    scenario_template = template["scenario"]

    replacements = {
        "{name}": n1,
        "{name2}": n2,
        "{name3}": n3,
        "{subj_pronoun}": sp1,
        "{obj_pronoun}": op1,
        "{subj_pronoun_possessive}": pp1,
        "{obj_pronoun2}": op2,
        "{setting}": rng.choice(SETTINGS),
        "{object}": rng.choice(OBJECTS),
        "{food}": rng.choice(FOODS),
        "{animal}": rng.choice(ANIMALS),
        "{hobby}": rng.choice(HOBBIES),
        "{fact_true}": rng.choice(FACTS_TRUE),
        "{false_fact}": rng.choice(FACTS_FALSE),
        "{moral_dilemma}": rng.choice(MORAL_DILEMMAS),
    }

    scenario = scenario_template
    for k, v in replacements.items():
        scenario = scenario.replace(k, v)

    correct_label = template["correct"]  # "a" or "b"
    desc_a = CONDITION_DESCRIPTIONS[cond_a]
    desc_b = CONDITION_DESCRIPTIONS[cond_b]
    question = _make_question(desc_a, desc_b)

    if correct_label == "a":
        correct_cond = cond_a
        foil_cond = cond_b
    else:
        correct_cond = cond_b
        foil_cond = cond_a

    return {
        "scenario": scenario,
        "question": question,
        "correct_answer": "A" if correct_label == "a" else "B",
        "requires": correct_cond,
        "foil": foil_cond,
        "cond_a": cond_a,
        "cond_b": cond_b,
    }


def generate_items_for_pair(
    cond_a: str,
    cond_b: str,
    n_items: int,
    rng: random.Random,
) -> list[dict]:
    """Generate n_items for a condition pair using templates."""
    key = frozenset({cond_a, cond_b})
    templates = TEMPLATE_BANK.get(key, [])

    if not templates:
        # Fallback: create minimal generic templates
        templates = _generate_fallback_templates(cond_a, cond_b)

    # Split templates by correct label for balancing
    a_templates = [t for t in templates if t["correct"] == "a"]
    b_templates = [t for t in templates if t["correct"] == "b"]

    items = []
    n_a = n_items // 2
    n_b = n_items - n_a

    # Generate n_a items where A is correct
    for i in range(n_a):
        if a_templates:
            t = a_templates[i % len(a_templates)]
        else:
            t = templates[i % len(templates)]
        items.append(instantiate_template(t, rng, cond_a, cond_b))

    # Generate n_b items where B is correct
    for i in range(n_b):
        if b_templates:
            t = b_templates[i % len(b_templates)]
        else:
            t = templates[(i + n_a) % len(templates)]
        items.append(instantiate_template(t, rng, cond_a, cond_b))

    rng.shuffle(items)
    return items


def _generate_fallback_templates(cond_a: str, cond_b: str) -> list[dict]:
    """Generate simple fallback templates for pairs without hand-crafted ones."""
    desc_a = CONDITION_DESCRIPTIONS[cond_a]
    desc_b = CONDITION_DESCRIPTIONS[cond_b]

    templates = [
        {"scenario": f"{{name}} experienced something that primarily involved {desc_a} "
                     f"{{setting}}.",
         "correct": "a"},
        {"scenario": f"{{name}} encountered a situation centered on {desc_b} "
                     f"while talking with {{name2}}.",
         "correct": "b"},
        {"scenario": f"The event at {{setting}} was mainly about {desc_a} "
                     f"for {{name}}.",
         "correct": "a"},
        {"scenario": f"{{name2}} described a situation to {{name}} that clearly centered on "
                     f"{desc_b}.",
         "correct": "b"},
    ]
    return templates


# ---------------------------------------------------------------------------
# RDM loading and MCS computation
# ---------------------------------------------------------------------------

def load_brain_rdm(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load brain RDM. Returns (rdm, conditions)."""
    data = np.load(path, allow_pickle=True)
    return data["rdm"], list(data["conditions"])


def _compute_rdm_from_v2_per_stim(
    npz_path: str | Path,
    peak_layer: int,
    conditions: list[str],
) -> np.ndarray:
    """Compute LLM RDM from v2 per-stimulus activations (our actual format)."""
    data = np.load(npz_path, allow_pickle=True)
    per_stim = data["per_stim_activations"]  # [3, n_stim, n_layers, dim]
    stim_conds = list(data["conditions"])
    pooling_names = list(data["pooling_names"])
    pool_idx = pooling_names.index("mean_all")
    n_layers = per_stim.shape[2]
    layer = min(peak_layer, n_layers - 1)

    unique_conds = sorted(set(stim_conds))
    centroids = np.zeros((len(unique_conds), per_stim.shape[-1]), dtype=np.float64)
    for ci, c in enumerate(unique_conds):
        mask = np.array([sc == c for sc in stim_conds])
        centroids[ci] = per_stim[pool_idx, mask, layer, :].mean(axis=0)

    order = [unique_conds.index(c) for c in conditions]
    act = centroids[order]
    act = act - act.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    act_norm = act / norms
    return 1.0 - np.clip(act_norm @ act_norm.T, -1, 1)


def compute_llm_rdm_from_activations(
    npz_path: str | Path,
    peak_layer: int,
    conditions: list[str],
) -> np.ndarray:
    """Compute LLM RDM at peak_layer using mean_all pooling (cosine distance)."""
    data = np.load(npz_path, allow_pickle=True)
    mean_act = data["mean_activations"]  # (n_cond, n_layers, hidden_dim)
    npz_conds = list(data["conditions"])
    layer_names = list(data["layer_names"])

    # Find layer index
    layer_key = f"layer_{peak_layer}"
    if layer_key in layer_names:
        layer_idx = layer_names.index(layer_key)
    else:
        # Use the last available layer
        layer_idx = len(layer_names) - 1
        print(f"  Warning: layer_{peak_layer} not found, using {layer_names[layer_idx]}")

    # Reorder conditions to match brain RDM ordering
    order = [npz_conds.index(c) for c in conditions]
    act = mean_act[order, layer_idx, :].astype(np.float64)

    # Center
    act = act - act.mean(axis=0, keepdims=True)

    # Cosine distance RDM
    norms = np.linalg.norm(act, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    act_norm = act / norms
    rdm = 1.0 - np.clip(act_norm @ act_norm.T, -1, 1)
    return rdm


def compute_llm_rdm_from_llm_rdms_npz(
    npz_path: str | Path,
    peak_layer: int,
    conditions: list[str],
) -> np.ndarray:
    """Load pre-computed LLM RDMs. Returns RDM at peak layer."""
    data = np.load(npz_path, allow_pickle=True)
    llm_rdms = data["llm_rdms"]  # (n_layers, n_cond, n_cond)
    npz_conds = list(data["conditions"])
    layer_names = list(data["layer_names"])

    layer_key = f"layer_{peak_layer}"
    if layer_key in layer_names:
        layer_idx = layer_names.index(layer_key)
    else:
        layer_idx = len(layer_names) - 1
        print(f"  Warning: layer_{peak_layer} not found, using {layer_names[layer_idx]}")

    rdm = llm_rdms[layer_idx]

    # Reorder conditions to match brain order
    order = [npz_conds.index(c) for c in conditions]
    rdm = rdm[np.ix_(order, order)]
    return rdm


def compute_mcs(brain_rdm: np.ndarray, llm_rdm: np.ndarray, conditions: list[str]) -> dict:
    """Compute Mental-state Collapse Score for all 91 pairs.

    MCS(pair) = brain_distance(pair) - llm_distance(pair)
    High MCS → brain says far, LLM says close → high risk of confusion.
    """
    n = len(conditions)
    mcs_dict = {}
    for i, j in combinations(range(n), 2):
        pair_name = f"{conditions[i]}_vs_{conditions[j]}"
        brain_d = float(brain_rdm[i, j])
        llm_d = float(llm_rdm[i, j])
        mcs_dict[pair_name] = {
            "mcs": brain_d - llm_d,
            "brain_dist": brain_d,
            "llm_dist": llm_d,
            "cond_a": conditions[i],
            "cond_b": conditions[j],
        }
    return mcs_dict


def select_pairs(mcs_dict: dict, n_pairs: int) -> tuple[list, list]:
    """Select top-n HIGH-MCS and bottom-n LOW-MCS pairs."""
    sorted_pairs = sorted(mcs_dict.items(), key=lambda x: x[1]["mcs"], reverse=True)

    high_mcs = sorted_pairs[:n_pairs]
    low_mcs = sorted_pairs[-n_pairs:]

    return high_mcs, low_mcs


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def classify_item(
    model, tokenizer, item: dict, device: torch.device,
) -> dict:
    """Ask the model to answer the A/B question for one item."""
    prompt = f"""{item['scenario']}

{item['question']}

Answer with just the letter A or B:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip().upper()

    answered_a = "A" in response[:3] and "B" not in response[:3]
    answered_b = "B" in response[:3] and "A" not in response[:3]

    correct = item["correct_answer"]
    is_correct = (correct == "A" and answered_a) or (correct == "B" and answered_b)

    return {
        "scenario": item["scenario"],
        "correct_answer": correct,
        "model_response": response[:10],
        "is_correct": is_correct,
        "requires": item["requires"],
        "foil": item["foil"],
    }


# ---------------------------------------------------------------------------
# Lexical distance between condition names (for mixed-effects control)
# ---------------------------------------------------------------------------

def compute_lexical_distance(cond_a: str, cond_b: str) -> float:
    """Normalised edit distance between condition name strings."""
    a = cond_a.replace("_", " ")
    b = cond_b.replace("_", " ")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(n, m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scaled prospective failure prediction (v2): "
                    "brain-derived MCS predicts social reasoning errors",
    )
    parser.add_argument("--model_path", required=True, help="Path to HF model")
    parser.add_argument("--model_short", required=True, help="Short model name")
    parser.add_argument("--rsa_dir", default="results/cognitive_rsa",
                        help="Directory with RSA NPZ files")
    parser.add_argument("--brain_rdm_path", default="results/cognitive_rsa/brain_rdm.npz",
                        help="Path to brain_rdm.npz")
    parser.add_argument("--output_dir", default="results/prospective_v2",
                        help="Output directory")
    parser.add_argument("--peak_layer", type=int, default=26,
                        help="Peak RSA layer index")
    parser.add_argument("--n_pairs", type=int, default=7,
                        help="Number of HIGH and LOW pairs each")
    parser.add_argument("--items_per_pair", type=int, default=50,
                        help="Items per pair")
    args = parser.parse_args()

    rng = random.Random(42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load brain RDM and compute LLM RDM
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Load RDMs and compute MCS")
    print("=" * 70)

    brain_rdm, brain_conds = load_brain_rdm(args.brain_rdm_path)
    print(f"  Brain RDM: {brain_rdm.shape}, conditions: {brain_conds}")

    rsa_dir = Path(args.rsa_dir)

    # Try to load LLM RDM from available NPZ formats
    v2_npz = rsa_dir / f"{args.model_short}_rsa_v2_per_stim.npz"
    act_npz = rsa_dir / f"{args.model_short}_rsa_activations.npz"
    llm_rdms_npz = rsa_dir / f"{args.model_short}_rsa_llm_rdms.npz"

    if v2_npz.exists():
        print(f"  Loading per-stimulus RSA v2 from {v2_npz}")
        llm_rdm = _compute_rdm_from_v2_per_stim(
            v2_npz, args.peak_layer, brain_conds,
        )
    elif act_npz.exists():
        print(f"  Loading LLM activations from {act_npz}")
        llm_rdm = compute_llm_rdm_from_activations(
            act_npz, args.peak_layer, brain_conds,
        )
    elif llm_rdms_npz.exists():
        print(f"  Loading pre-computed LLM RDMs from {llm_rdms_npz}")
        llm_rdm = compute_llm_rdm_from_llm_rdms_npz(
            llm_rdms_npz, args.peak_layer, brain_conds,
        )
    else:
        raise FileNotFoundError(
            f"No RSA data found for {args.model_short} in {rsa_dir}. "
            f"Expected {v2_npz}, {act_npz}, or {llm_rdms_npz}."
        )

    print(f"  LLM RDM: {llm_rdm.shape}")

    # ------------------------------------------------------------------
    # Step 2: Compute MCS and select pairs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Compute MCS and select pairs")
    print("=" * 70)

    mcs_dict = compute_mcs(brain_rdm, llm_rdm, brain_conds)

    high_mcs_pairs, low_mcs_pairs = select_pairs(mcs_dict, args.n_pairs)

    print(f"\n  TOP {args.n_pairs} HIGH-MCS pairs (brain-far, LLM-close -> high risk):")
    for name, info in high_mcs_pairs:
        print(f"    {name:40s}  MCS={info['mcs']:.4f}  "
              f"(brain={info['brain_dist']:.3f}, llm={info['llm_dist']:.3f})")

    print(f"\n  BOTTOM {args.n_pairs} LOW-MCS pairs (both agree -> low risk):")
    for name, info in low_mcs_pairs:
        print(f"    {name:40s}  MCS={info['mcs']:.4f}  "
              f"(brain={info['brain_dist']:.3f}, llm={info['llm_dist']:.3f})")

    # ------------------------------------------------------------------
    # Step 3: Generate items
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Generate items")
    print("=" * 70)

    all_pair_items = {}
    for name, info in high_mcs_pairs + low_mcs_pairs:
        items = generate_items_for_pair(
            info["cond_a"], info["cond_b"],
            args.items_per_pair, rng,
        )
        all_pair_items[name] = items
        n_a = sum(1 for it in items if it["correct_answer"] == "A")
        n_b = sum(1 for it in items if it["correct_answer"] == "B")
        has_templates = frozenset({info["cond_a"], info["cond_b"]}) in TEMPLATE_BANK
        print(f"  {name:40s}  {len(items)} items (A={n_a}, B={n_b})  "
              f"{'hand-crafted' if has_templates else 'FALLBACK'}")

    # ------------------------------------------------------------------
    # Step 4: Load model and run items
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Load model and run items")
    print("=" * 70)

    print(f"  Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}")

    per_pair_results = []
    t0 = time.time()

    for pair_idx, (name, info) in enumerate(high_mcs_pairs + low_mcs_pairs):
        group = "HIGH" if pair_idx < args.n_pairs else "LOW"
        items = all_pair_items[name]

        item_results = []
        correct_count = 0
        for item in items:
            result = classify_item(model, tokenizer, item, device)
            item_results.append(result)
            if result["is_correct"]:
                correct_count += 1

        acc = correct_count / len(items) if items else 0.0
        lex_dist = compute_lexical_distance(info["cond_a"], info["cond_b"])

        pair_result = {
            "pair": name,
            "cond_a": info["cond_a"],
            "cond_b": info["cond_b"],
            "mcs": info["mcs"],
            "brain_dist": info["brain_dist"],
            "llm_dist": info["llm_dist"],
            "group": group,
            "accuracy": acc,
            "correct": correct_count,
            "total": len(items),
            "lexical_distance": lex_dist,
            "items": item_results,
        }
        per_pair_results.append(pair_result)

        elapsed = time.time() - t0
        print(f"  [{pair_idx + 1:2d}/{len(high_mcs_pairs) + len(low_mcs_pairs)}] "
              f"{group:4s}  {name:40s}  acc={acc:.0%} ({correct_count}/{len(items)})  "
              f"[{elapsed:.0f}s]")

    # ------------------------------------------------------------------
    # Step 5: Analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Analysis")
    print("=" * 70)

    high_results = [r for r in per_pair_results if r["group"] == "HIGH"]
    low_results = [r for r in per_pair_results if r["group"] == "LOW"]

    high_accs = [r["accuracy"] for r in high_results]
    low_accs = [r["accuracy"] for r in low_results]

    high_mean = float(np.mean(high_accs)) if high_accs else 0.0
    low_mean = float(np.mean(low_accs)) if low_accs else 0.0

    print(f"\n  HIGH-MCS mean accuracy: {high_mean:.1%}  (n={len(high_accs)} pairs)")
    print(f"  LOW-MCS  mean accuracy: {low_mean:.1%}  (n={len(low_accs)} pairs)")
    print(f"  Difference: {low_mean - high_mean:.1%} (LOW - HIGH)")

    # Mann-Whitney U test (HIGH < LOW)
    if len(high_accs) >= 2 and len(low_accs) >= 2:
        U, mw_p = mannwhitneyu(high_accs, low_accs, alternative="less")
        print(f"\n  Mann-Whitney U test (HIGH < LOW): U={U:.1f}, p={mw_p:.4f}")
    else:
        U, mw_p = float("nan"), float("nan")
        print("\n  Not enough pairs for Mann-Whitney U test")

    # Spearman correlation: MCS vs accuracy across all 14 pairs
    all_mcs = [r["mcs"] for r in per_pair_results]
    all_accs = [r["accuracy"] for r in per_pair_results]

    if len(all_mcs) >= 3:
        sp_rho, sp_p = spearmanr(all_mcs, all_accs)
        print(f"  Spearman (MCS vs accuracy): rho={sp_rho:.3f}, p={sp_p:.4f}")
    else:
        sp_rho, sp_p = float("nan"), float("nan")
        print("  Not enough pairs for Spearman correlation")

    # Effect size: Cohen's d
    if len(high_accs) >= 2 and len(low_accs) >= 2:
        pooled_std = np.sqrt(
            ((len(high_accs) - 1) * np.var(high_accs, ddof=1)
             + (len(low_accs) - 1) * np.var(low_accs, ddof=1))
            / (len(high_accs) + len(low_accs) - 2)
        )
        if pooled_std > 0:
            cohens_d = (low_mean - high_mean) / pooled_std
        else:
            cohens_d = float("nan")
        print(f"  Cohen's d (LOW - HIGH): {cohens_d:.2f}")
    else:
        cohens_d = float("nan")

    # Logistic mixed-effects regression (optional)
    mixed_effects_result = None
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import logit as logit_model

        # Build trial-level dataframe
        import pandas as pd
        rows = []
        for r in per_pair_results:
            for item in r["items"]:
                rows.append({
                    "correct": int(item["is_correct"]),
                    "mcs": r["mcs"],
                    "lexical_distance": r["lexical_distance"],
                    "pair": r["pair"],
                })
        df = pd.DataFrame(rows)

        # Simple logistic regression (statsmodels GLM as fallback for mixed effects)
        df["intercept"] = 1.0
        X = df[["intercept", "mcs", "lexical_distance"]].values
        y = df["correct"].values

        try:
            glm = sm.GLM(y, X, family=sm.families.Binomial())
            glm_result = glm.fit()
            mixed_effects_result = {
                "method": "logistic_regression_GLM",
                "coef_intercept": float(glm_result.params[0]),
                "coef_mcs": float(glm_result.params[1]),
                "coef_lexical_distance": float(glm_result.params[2]),
                "pvalue_mcs": float(glm_result.pvalues[1]),
                "pvalue_lexical_distance": float(glm_result.pvalues[2]),
                "aic": float(glm_result.aic),
                "n_observations": int(len(df)),
            }
            print(f"\n  Logistic regression (GLM):")
            print(f"    coef(MCS) = {mixed_effects_result['coef_mcs']:.4f}, "
                  f"p = {mixed_effects_result['pvalue_mcs']:.4f}")
            print(f"    coef(lexical_dist) = {mixed_effects_result['coef_lexical_distance']:.4f}, "
                  f"p = {mixed_effects_result['pvalue_lexical_distance']:.4f}")
        except Exception as e:
            print(f"  GLM fitting failed: {e}")

        # Try mixed effects if available
        try:
            from statsmodels.regression.mixed_linear_model import MixedLM
            # Linear mixed model on correct ~ MCS + lexical_distance + (1|pair)
            md = MixedLM.from_formula(
                "correct ~ mcs + lexical_distance",
                groups="pair",
                data=df,
            )
            mdf = md.fit(reml=False)
            mixed_effects_result = {
                "method": "linear_mixed_effects",
                "coef_intercept": float(mdf.fe_params["Intercept"]),
                "coef_mcs": float(mdf.fe_params["mcs"]),
                "coef_lexical_distance": float(mdf.fe_params["lexical_distance"]),
                "pvalue_mcs": float(mdf.pvalues["mcs"]),
                "pvalue_lexical_distance": float(mdf.pvalues["lexical_distance"]),
                "aic": float(mdf.aic),
                "n_observations": int(len(df)),
                "n_groups": int(mdf.nobs),
                "random_intercept_var": float(mdf.cov_re.iloc[0, 0]),
            }
            print(f"\n  Linear Mixed Effects (correct ~ MCS + lex_dist + (1|pair)):")
            print(f"    coef(MCS) = {mixed_effects_result['coef_mcs']:.4f}, "
                  f"p = {mixed_effects_result['pvalue_mcs']:.4f}")
            print(f"    coef(lexical_dist) = {mixed_effects_result['coef_lexical_distance']:.4f}, "
                  f"p = {mixed_effects_result['pvalue_lexical_distance']:.4f}")
            print(f"    Random intercept var = {mixed_effects_result['random_intercept_var']:.4f}")
        except Exception as e:
            print(f"  Mixed effects model not available or failed: {e}")

    except ImportError:
        print("\n  statsmodels not available; skipping regression analysis")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    mcs_values = {name: float(info["mcs"]) for name, info in mcs_dict.items()}

    # Strip items for cleaner per_pair_results in output (keep items)
    output = {
        "model": args.model_short,
        "peak_layer": args.peak_layer,
        "n_pairs_per_group": args.n_pairs,
        "items_per_pair": args.items_per_pair,
        "mcs_values": mcs_values,
        "high_mcs_pairs": [name for name, _ in high_mcs_pairs],
        "low_mcs_pairs": [name for name, _ in low_mcs_pairs],
        "per_pair_results": per_pair_results,
        "summary": {
            "high_mcs_mean_acc": high_mean,
            "low_mcs_mean_acc": low_mean,
            "acc_difference": low_mean - high_mean,
            "mann_whitney_U": float(U) if not np.isnan(U) else None,
            "mann_whitney_p": float(mw_p) if not np.isnan(mw_p) else None,
            "mcs_acc_spearman_rho": float(sp_rho) if not np.isnan(sp_rho) else None,
            "mcs_acc_spearman_p": float(sp_p) if not np.isnan(sp_p) else None,
            "cohens_d": float(cohens_d) if not np.isnan(cohens_d) else None,
            "mixed_effects": mixed_effects_result,
        },
    }

    out_path = out_dir / f"{args.model_short}_prospective_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Model: {args.model_short}")
    print(f"  Pairs: {args.n_pairs} HIGH + {args.n_pairs} LOW")
    print(f"  Items per pair: {args.items_per_pair}")
    print(f"  HIGH-MCS accuracy: {high_mean:.1%}")
    print(f"  LOW-MCS  accuracy: {low_mean:.1%}")
    if not np.isnan(mw_p):
        sig = "***" if mw_p < 0.001 else "**" if mw_p < 0.01 else "*" if mw_p < 0.05 else "n.s."
        print(f"  Mann-Whitney p = {mw_p:.4f} {sig}")
    if not np.isnan(sp_rho):
        sig = "***" if sp_p < 0.001 else "**" if sp_p < 0.01 else "*" if sp_p < 0.05 else "n.s."
        print(f"  Spearman rho = {sp_rho:.3f}, p = {sp_p:.4f} {sig}")
    print("=" * 70)


if __name__ == "__main__":
    main()
