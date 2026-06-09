#!/usr/bin/env python3
"""
Template-matched stimuli for the 14-condition RSA.

Motivation: the original 14-condition stimuli come from diverse sources
(GoEmotions, moral dilemmas, false-belief stories, ...) with wildly different
formats and lengths (emotion ~11 words vs mentalizing ~61 words). A reviewer
(citing Hadidi 2026) could argue that the RSA alignment reflects format/style
differences between sources, not genuine cognitive content. Template-matched
stimuli use IDENTICAL sentence templates for all 14 conditions, so the only
difference is the cognitive content words.

Design:
  - 4 sentence templates that work across all 14 conditions
  - 15 scenario snippets per condition per template = 60 stimuli per condition
  - Sentence length constrained to +/- 20% across conditions
  - Condition label words are AVOIDED in the stimuli (e.g., anger stimuli say
    "outrage", "fury", "rage" -- never "anger")
  - Every stimulus is a complete, natural English sentence

Output: data/cognitive_stimuli/rsa/rsa_stimuli_template_matched.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 4 sentence templates
# Each takes:  person, scenario, outcome
# ──────────────────────────────────────────────────────────────────────

TEMPLATES = {
    "narrative": (
        "When {person} {scenario}, {person_pronoun} experienced "
        "a strong sense of {outcome}."
    ),
    "observation": (
        "The researcher noted that {person} showed clear signs of "
        "{outcome} during the {scenario_short} episode."
    ),
    "third_person": (
        "{person} was in a situation where {person_pronoun} {scenario}. "
        "This involved {outcome_long}."
    ),
    "report": (
        "According to the case report, {person} {scenario}, "
        "which led to a pronounced experience of {outcome}."
    ),
}

# ──────────────────────────────────────────────────────────────────────
# Named persons (rotated across stimuli for variety)
# ──────────────────────────────────────────────────────────────────────

PERSONS = [
    ("Alex", "they"),
    ("Jordan", "they"),
    ("Morgan", "they"),
    ("Taylor", "they"),
    ("Casey", "they"),
    ("Riley", "they"),
    ("Sam", "they"),
    ("Quinn", "they"),
    ("Dana", "they"),
    ("Robin", "they"),
    ("Jamie", "they"),
    ("Avery", "they"),
    ("Blake", "they"),
    ("Cameron", "they"),
    ("Drew", "they"),
]


# ──────────────────────────────────────────────────────────────────────
# Per-condition scenario sets
#
# Each entry: (scenario, scenario_short, outcome, outcome_long)
#   scenario      — fills "When {person} {scenario}"
#   scenario_short — fills "during the {scenario_short} episode"
#   outcome       — fills "a strong sense of {outcome}"
#   outcome_long  — fills "This involved {outcome_long}"
#
# RULE: never use the condition label word itself.
# ──────────────────────────────────────────────────────────────────────

CONDITION_SCENARIOS = {
    # ── AFFECTIVE (6) ──
    "anger": [
        ("discovered that a coworker had stolen credit for their project",
         "workplace betrayal",
         "outrage and indignation",
         "an intense surge of outrage and a desire for confrontation"),
        ("found out their landlord had illegally entered their apartment",
         "privacy violation",
         "fury and violation",
         "a deep fury at the breach of personal boundaries"),
        ("was told that their insurance claim had been fraudulently denied",
         "insurance dispute",
         "rage and injustice",
         "mounting rage at the blatant unfairness of the decision"),
        ("learned their neighbor had poisoned their garden out of spite",
         "neighbor conflict",
         "wrath and resentment",
         "a wrathful reaction to deliberate destruction of their property"),
        ("saw someone kick a stray dog on the sidewalk",
         "animal cruelty witness",
         "indignation and hostility",
         "fierce indignation and an urge to intervene immediately"),
        ("realized the mechanic had charged triple the agreed price",
         "price gouging",
         "outrage and bitterness",
         "bitter outrage at being deliberately overcharged"),
        ("caught their roommate reading their private journal",
         "privacy intrusion",
         "fury and betrayal",
         "intense fury at the violation of their most private thoughts"),
        ("received a parking ticket despite being legally parked",
         "unjust penalty",
         "irritation and resentment",
         "seething irritation at an authority figure acting unfairly"),
        ("witnessed a bully taunting a younger child at the park",
         "bullying witness",
         "indignation and protectiveness",
         "fierce indignation rising from witnessing cruelty to the vulnerable"),
        ("was falsely accused of cheating on an important exam",
         "false accusation",
         "rage and humiliation",
         "explosive rage combined with the sting of an unjust accusation"),
        ("found their bicycle deliberately vandalized overnight",
         "vandalism discovery",
         "fury and frustration",
         "a surge of fury at the senseless destruction of their property"),
        ("learned that a trusted friend had been spreading rumors about them",
         "social betrayal",
         "wrath and disillusionment",
         "deep-seated wrath at the betrayal of someone they had trusted"),
        ("was cut off in traffic by a driver who then made rude gestures",
         "road confrontation",
         "hostility and aggressive impulses",
         "a flash of hostility and aggressive impulses toward the other driver"),
        ("discovered their employer had been underpaying them for months",
         "wage theft",
         "outrage and exploitation",
         "burning outrage at systematic exploitation by their employer"),
        ("watched a politician blatantly lie during a public debate",
         "political dishonesty",
         "indignation and contempt",
         "sharp indignation at the brazen dishonesty of the speaker"),
    ],

    "fear": [
        ("heard strange footsteps in the empty house late at night",
         "nighttime intrusion",
         "dread and alarm",
         "a gripping dread that someone had broken into their home"),
        ("noticed the airplane shaking violently during severe turbulence",
         "flight turbulence",
         "terror and helplessness",
         "overwhelming terror as the plane lurched and passengers screamed"),
        ("received an anonymous threatening letter at their doorstep",
         "threatening message",
         "panic and vulnerability",
         "a wave of panic and acute awareness of their vulnerability"),
        ("saw a large snake coiled on the path directly ahead",
         "snake encounter",
         "fright and paralysis",
         "an instinctive freeze response and visceral fright"),
        ("felt the building sway during a sudden earthquake",
         "earthquake",
         "alarm and survival urgency",
         "primal alarm and a desperate urge to find shelter"),
        ("woke up unable to move or speak in a dark room",
         "sleep paralysis",
         "horror and entrapment",
         "suffocating horror at being conscious but completely immobilized"),
        ("noticed a car following them through several turns on a dark road",
         "pursuit",
         "dread and hypervigilance",
         "mounting dread and hypervigilant scanning for escape routes"),
        ("was swimming and felt something large brush against their leg underwater",
         "water threat",
         "terror and revulsion",
         "a jolt of terror and frantic movement toward the shore"),
        ("smelled smoke and heard fire alarms blaring in the hotel corridor",
         "fire emergency",
         "panic and disorientation",
         "raw panic coupled with desperate disorientation about escape routes"),
        ("looked down from the observation deck of an extremely tall building",
         "height exposure",
         "vertigo and dread",
         "paralyzing vertigo and an irrational dread of falling"),
        ("was told by their doctor that the test results were abnormal",
         "medical scare",
         "alarm and existential anxiety",
         "piercing alarm and a flood of existential anxiety about health"),
        ("walked into a dark alley and saw several figures blocking the exit",
         "ambush threat",
         "terror and entrapment",
         "instant terror and a frantic assessment of escape options"),
        ("heard a tornado siren while home alone with no basement",
         "severe storm",
         "dread and helplessness",
         "overwhelming dread at having no safe shelter available"),
        ("woke up in an unfamiliar room and could not remember how they got there",
         "disorientation",
         "confusion and alarm",
         "a disorienting wave of alarm and confused vulnerability"),
        ("saw their child run toward a busy intersection without looking",
         "child danger",
         "sheer panic and protective urgency",
         "a surge of sheer panic driving immediate protective action"),
    ],

    "disgust": [
        ("opened a container of food that had been rotting for weeks",
         "spoiled food",
         "revulsion and nausea",
         "visceral revulsion and an immediate gagging reflex"),
        ("stepped barefoot into something slimy on the bathroom floor",
         "tactile contamination",
         "repulsion and queasiness",
         "acute repulsion at unexpected contact with something foul"),
        ("watched a documentary showing maggots infesting a wound",
         "parasite footage",
         "nausea and aversion",
         "overwhelming nausea and a compulsive urge to look away"),
        ("discovered mold spreading across the walls of their bedroom",
         "mold infestation",
         "revulsion and contamination concern",
         "deep revulsion mixed with alarm about contamination exposure"),
        ("smelled raw sewage flooding into the street after a pipe burst",
         "sewage exposure",
         "repulsion and physical discomfort",
         "intense repulsion causing involuntary recoiling from the stench"),
        ("found a dead rat decomposing inside their kitchen cabinet",
         "carcass discovery",
         "nausea and horror",
         "stomach-turning nausea at the sight and smell of decomposition"),
        ("sat down on a bus seat covered with an unknown sticky substance",
         "surface contamination",
         "repulsion and contamination anxiety",
         "immediate repulsion and anxious thoughts about what they had touched"),
        ("watched someone spit a large amount of phlegm into a drinking fountain",
         "public contamination",
         "revulsion and outrage",
         "a wave of revulsion combined with indignation at the behavior"),
        ("noticed cockroaches scattering across the restaurant table during dinner",
         "pest encounter",
         "repulsion and appetite loss",
         "intense repulsion that instantly eliminated any desire to eat"),
        ("smelled a coworker who clearly had not bathed in many days",
         "body odor",
         "aversion and social discomfort",
         "strong aversion combined with awkward social discomfort"),
        ("saw someone pick their nose and then handle shared food",
         "hygiene violation",
         "revulsion and avoidance",
         "sharp revulsion and an immediate desire to avoid that food"),
        ("bit into a piece of fruit and found half a worm inside",
         "food contamination",
         "nausea and shock",
         "a sickening wave of nausea combined with startled shock"),
        ("walked through a public restroom that had not been cleaned for days",
         "unsanitary facility",
         "repulsion and breath-holding",
         "overpowering repulsion forcing shallow breathing through the mouth"),
        ("watched someone vomit on the seat next to them on a bus",
         "public illness",
         "nausea and contagion worry",
         "sympathetic nausea and instinctive withdrawal from contamination"),
        ("opened a drain and found it clogged with a mass of wet hair and slime",
         "drain blockage",
         "revulsion and reluctant handling",
         "tactile revulsion at having to deal with the slimy obstruction"),
    ],

    "sadness": [
        ("received the news that their childhood best friend had passed away",
         "bereavement",
         "grief and loss",
         "profound grief and an aching sense of irreversible loss"),
        ("watched their elderly parent struggle to remember their name",
         "cognitive decline witness",
         "sorrow and helplessness",
         "deep sorrow at watching a loved one slowly fade away"),
        ("returned to their childhood home and found it demolished",
         "home demolition",
         "melancholy and nostalgia",
         "a heavy melancholy for a place that existed only in memory"),
        ("found old letters written by a deceased grandparent they never met",
         "posthumous discovery",
         "wistfulness and longing",
         "a bittersweet wistfulness for a relationship that could never be"),
        ("saw their dog waiting by the door for an owner who would never return",
         "animal loyalty",
         "heartbreak and empathic sorrow",
         "a wrenching heartbreak at witnessing faithful and futile devotion"),
        ("listened to the song that used to play during a lost relationship",
         "nostalgic trigger",
         "longing and emotional pain",
         "a piercing longing that reopened old emotional wounds"),
        ("learned that the local orphanage was closing due to lack of funding",
         "institutional loss",
         "despair and compassion",
         "a heavy despair for the children who would lose their home"),
        ("walked through an abandoned neighborhood they once knew as vibrant",
         "community decay",
         "desolation and reflection",
         "a desolate feeling at the contrast between past vitality and present ruin"),
        ("held the hand of a friend who was crying over a failed marriage",
         "companionship in grief",
         "empathic sorrow and tenderness",
         "an aching empathic sorrow shared with someone in deep pain"),
        ("read a diary entry they wrote during the worst period of their life",
         "past suffering recall",
         "retrospective anguish and compassion for their former self",
         "a wave of retrospective anguish upon revisiting old wounds"),
        ("stood at the grave of a teacher who had changed their life",
         "mentor loss",
         "gratitude mixed with grief",
         "an overwhelming mixture of gratitude and grief for an irreplaceable mentor"),
        ("missed the last chance to say goodbye to a dying relative",
         "missed farewell",
         "regret and sorrow",
         "crushing regret and unresolved sorrow over the lost opportunity"),
        ("watched the final sunset of a trip they knew they could never repeat",
         "fleeting experience",
         "bittersweet longing and transience",
         "a bittersweet awareness that the beautiful moment was already passing"),
        ("realized their closest friendship had slowly faded without either noticing",
         "drifting apart",
         "quiet grief and resignation",
         "a quiet grief for a bond that dissolved through sheer neglect"),
        ("saw a homeless person sleeping in the cold holding a photograph",
         "destitution witness",
         "compassionate sorrow and helplessness",
         "deep compassionate sorrow and a helpless wish to ease their suffering"),
    ],

    "happiness": [
        ("received the acceptance letter from the university of their dreams",
         "achievement notification",
         "elation and pride",
         "overwhelming elation and a sense of hard-earned accomplishment"),
        ("watched their child take their very first steps across the room",
         "milestone witness",
         "joy and wonder",
         "pure joy and amazement at witnessing a developmental milestone"),
        ("reunited with a sibling after ten years of living on different continents",
         "family reunion",
         "warmth and euphoria",
         "a flood of warmth and euphoria upon embracing a long-lost loved one"),
        ("finished the marathon they had spent an entire year training for",
         "goal completion",
         "triumph and exhilaration",
         "exhilarating triumph and physical relief at crossing the finish line"),
        ("was surprised with a birthday party by people they thought had forgotten",
         "surprise celebration",
         "delight and gratitude",
         "a rush of delight and gratitude at being remembered and celebrated"),
        ("adopted a rescue dog and watched it explore its new home with excitement",
         "pet adoption",
         "affection and contentment",
         "tender affection and deep contentment at providing a loving home"),
        ("received an unexpected compliment from someone they deeply respected",
         "peer recognition",
         "pride and validation",
         "a warm glow of pride and the feeling of being truly valued"),
        ("danced with friends at a wedding celebration late into the night",
         "social celebration",
         "euphoria and connection",
         "carefree euphoria and a feeling of belonging among close friends"),
        ("witnessed a beautiful sunrise after camping overnight in the mountains",
         "nature experience",
         "awe and serenity",
         "a quiet awe and serene contentment in the presence of natural beauty"),
        ("opened a gift that showed the giver had remembered an offhand wish",
         "thoughtful gesture",
         "touched gratitude and warmth",
         "deeply touched gratitude at the attentiveness of someone who cared"),
        ("heard their favorite song playing unexpectedly in a foreign city",
         "serendipitous moment",
         "nostalgia and delight",
         "a spontaneous burst of delight at the coincidence and familiarity"),
        ("learned that a project they poured their heart into had won an award",
         "creative recognition",
         "fulfillment and elation",
         "deep fulfillment at seeing personal effort publicly recognized"),
        ("shared a meal with close friends after a long period of isolation",
         "social reconnection",
         "warmth and belonging",
         "a comforting warmth and renewed sense of belonging after loneliness"),
        ("saw strangers helping each other during a difficult community event",
         "collective kindness",
         "uplift and faith in people",
         "an uplifting feeling and restored faith in human goodness"),
        ("looked through photographs from a trip that exceeded all expectations",
         "memory revisiting",
         "nostalgia and satisfaction",
         "a satisfied glow while reliving moments of unexpected adventure"),
    ],

    "valence": [
        ("received mixed feedback that included both praise and harsh criticism",
         "ambiguous evaluation",
         "conflicted emotional processing about the overall tone",
         "a complex emotional evaluation weighing positive against negative elements"),
        ("watched a bittersweet film that ended with both a reunion and a departure",
         "mixed narrative",
         "layered emotional evaluation of the experience",
         "a nuanced emotional assessment combining uplift with poignancy"),
        ("tasted a dish that was simultaneously too salty and unexpectedly delicious",
         "sensory contrast",
         "ambivalent positive-negative appraisal",
         "an ambivalent appraisal oscillating between pleasure and displeasure"),
        ("read a letter of recommendation that was lukewarm despite kind words",
         "veiled negativity",
         "unease from evaluating hidden sentiment",
         "subtle unease as they parsed the true evaluative tone beneath the surface"),
        ("attended a farewell dinner that was both celebratory and somber",
         "transitional event",
         "blended positive and negative emotional tone",
         "a blended emotional tone that was simultaneously uplifting and heavy"),
        ("heard a joke that was funny but also slightly offensive to a friend",
         "social tension",
         "evaluative conflict between amusement and discomfort",
         "an evaluative conflict as amusement clashed with social sensitivity"),
        ("received a promotion that required relocating far from family",
         "gain-loss tradeoff",
         "evaluative tension between opportunity and sacrifice",
         "a tense internal evaluation balancing professional gain against personal cost"),
        ("saw a street performer whose act was technically brilliant but emotionally hollow",
         "aesthetic judgment",
         "nuanced evaluative response weighing skill against feeling",
         "a nuanced evaluative response dissociating technical admiration from emotional engagement"),
        ("revisited a childhood place that had changed in both good and bad ways",
         "nostalgia contrast",
         "layered positive-negative evaluation of the changes",
         "a layered evaluative process comparing cherished memories with altered reality"),
        ("completed a difficult negotiation where both sides made painful concessions",
         "compromise outcome",
         "muted satisfaction amid residual dissatisfaction",
         "muted evaluative satisfaction tempered by awareness of what was given up"),
        ("watched a sunset that was beautiful but reminded them of a painful goodbye",
         "triggered memory",
         "conflicting evaluative tone between beauty and sorrow",
         "a conflicting evaluative experience as beauty triggered painful associations"),
        ("received a gift they did not want from someone who meant well",
         "intention mismatch",
         "evaluative dissonance between gratitude and disappointment",
         "evaluative dissonance as genuine gratitude coexisted with private disappointment"),
        ("listened to a eulogy that was both deeply moving and uncomfortably honest",
         "tonal complexity",
         "multi-layered evaluative response to the speech",
         "a multi-layered evaluative response appreciating honesty while flinching at its sting"),
        ("finished a book that was beautifully written but had a devastating ending",
         "narrative evaluation",
         "evaluative tension about the overall experience",
         "a mixed evaluative response as literary admiration contended with emotional distress"),
        ("ended a relationship that was unhealthy but still deeply cherished",
         "necessary loss",
         "evaluative clarity mixed with emotional pain",
         "a difficult evaluative clarity acknowledging necessity while mourning what was lost"),
    ],

    # ── SOCIAL / MENTALISTIC (8) ──

    "belief": [
        ("was told by a colleague that the office had moved to a new building",
         "informed relocation",
         "firm conviction about a factual claim",
         "forming a firm conviction based on testimony that might be outdated"),
        ("read a news article reporting that a major company was going bankrupt",
         "media report",
         "acceptance of a stated fact as true",
         "accepting a factual claim as true based on a presumably reliable source"),
        ("overheard someone say the restaurant on the corner had closed permanently",
         "secondhand information",
         "adopted assumption about a state of affairs",
         "adopting an assumption about reality based on casually overheard testimony"),
        ("was taught in school that a certain historical event happened in 1066",
         "educational instruction",
         "learned conviction about a historical fact",
         "holding a learned conviction transmitted through formal education"),
        ("checked the weather app and saw it predicted sunshine all day",
         "forecast reliance",
         "trust in a predictive claim",
         "trusting an algorithmically generated prediction as a reliable representation of reality"),
        ("remembered a friend saying the meeting was scheduled for Tuesday",
         "recalled testimony",
         "retained factual assumption from memory",
         "relying on a retained assumption from a previous conversation"),
        ("noticed a sign stating the museum was free on Thursdays",
         "posted notice",
         "accepted claim based on official signage",
         "accepting a factual claim on the authority of an institutional posting"),
        ("was assured by the mechanic that the car was safe to drive",
         "expert assurance",
         "trust in an expert assertion",
         "placing trust in a domain expert's assertion about a technical matter"),
        ("heard a rumor that the company was planning major layoffs",
         "rumor exposure",
         "tentative acceptance of an unverified claim",
         "tentatively accepting an unverified claim while remaining partially skeptical"),
        ("saw a documentary claiming a widely accepted theory was wrong",
         "counter-evidence",
         "cognitive conflict between old and new claims",
         "experiencing cognitive conflict between an established conviction and new counter-evidence"),
        ("found a handwritten note saying the spare key was under the mat",
         "written instruction",
         "trust in a specific factual instruction",
         "trusting a written instruction about a specific factual arrangement"),
        ("was warned by a neighbor that the tap water was unsafe to drink",
         "safety warning",
         "precautionary conviction based on a warning",
         "forming a precautionary conviction based on a community member's warning"),
        ("received an email confirming their appointment was at ten in the morning",
         "formal confirmation",
         "reinforced certainty about a scheduled event",
         "experiencing reinforced certainty from an official written confirmation"),
        ("recalled reading that a certain vitamin prevented colds",
         "health claim recall",
         "retained health-related conviction",
         "holding a retained health-related conviction from a previously read source"),
        ("noticed the map showed a shortcut through the park to the station",
         "navigational reference",
         "spatial assumption based on a map",
         "forming a spatial assumption about reality from a cartographic representation"),
    ],

    "mentalizing": [
        ("tried to figure out why their quiet colleague had suddenly become talkative",
         "behavioral change interpretation",
         "active mental-state reasoning about another person",
         "actively reasoning about what internal change could explain the behavioral shift"),
        ("wondered what their boss was really thinking during the awkward silence",
         "social ambiguity",
         "effortful interpretation of hidden mental states",
         "effortfully interpreting what thoughts might underlie an ambiguous social signal"),
        ("watched a stranger on the bus smile at their phone and imagined the reason",
         "spontaneous inference",
         "spontaneous attribution of thoughts to a stranger",
         "spontaneously attributing specific thoughts and feelings to an unknown person"),
        ("noticed their friend ordered a salad instead of the usual burger and wondered why",
         "routine deviation",
         "reasoning about motives behind a choice",
         "reasoning about what internal motive or preference shift drove the unusual choice"),
        ("observed a child hesitate before entering the classroom on the first day",
         "behavioral cue reading",
         "inferring emotional state from observable hesitation",
         "inferring a complex internal emotional state from subtle behavioral hesitation"),
        ("listened to a coworker's carefully worded email and sensed hidden frustration",
         "communication decoding",
         "detecting concealed emotions in language",
         "detecting concealed emotional states embedded within carefully neutral language"),
        ("saw their partner pause before answering a simple question about their day",
         "micro-hesitation",
         "reading significance into a brief behavioral pause",
         "reading psychological significance into a momentary hesitation in conversation"),
        ("noticed a student in the back row avoiding eye contact during the discussion",
         "avoidance behavior",
         "interpreting avoidance as a mental-state signal",
         "interpreting gaze avoidance as a signal of discomfort or disengagement"),
        ("wondered why a usually punctual friend had arrived late without explanation",
         "unexplained deviation",
         "hypothesizing about unspoken reasons",
         "generating hypotheses about unspoken reasons behind atypical behavior"),
        ("realized their neighbor had been unusually polite after a recent argument",
         "compensatory behavior",
         "inferring reconciliatory intent from changed behavior",
         "inferring reconciliatory intent or guilt from a strategic shift in social behavior"),
        ("sensed that the interviewer had already made a decision before the final question",
         "premature closure detection",
         "reading a foregone conclusion from behavioral cues",
         "detecting a foregone internal conclusion from the interviewer's diminished engagement"),
        ("watched a friend carefully choose words when describing their new partner",
         "selective disclosure",
         "interpreting deliberate word choice as a mental-state signal",
         "interpreting careful word selection as a signal about underlying attitudes or concerns"),
        ("noticed the waiter's forced smile and suspected the shift had been difficult",
         "affective masking detection",
         "seeing through a social mask to the state underneath",
         "perceiving the gap between a performed social expression and the true internal state"),
        ("overheard a couple speaking in clipped tones and inferred recent tension",
         "relational inference",
         "inferring relationship dynamics from communication style",
         "inferring recent relational tension from the prosodic and lexical quality of an exchange"),
        ("observed a politician's micro-expression flash before the composed response",
         "micro-expression reading",
         "detecting brief involuntary affective leakage",
         "detecting a brief involuntary emotional display that contradicted the verbal message"),
    ],

    "intention": [
        ("noticed their roommate leaving a mop and bucket near the dirty kitchen floor",
         "implied action plan",
         "inferred goal-directed planning in another person",
         "inferring that the arrangement of objects signaled an upcoming planned action"),
        ("watched a stranger at the store pick up and compare two brands carefully",
         "deliberation observation",
         "recognition of deliberate evaluative behavior",
         "recognizing that the careful comparison reflected a deliberate decision process"),
        ("saw a colleague printing and collating pages before the unannounced meeting",
         "preparation cues",
         "inference of planned action from preparatory behavior",
         "inferring a planned presentation from observed preparatory activity"),
        ("overheard a neighbor asking about fence regulations at the hardware store",
         "information seeking",
         "inference of a future project from inquiry behavior",
         "inferring a future construction project from the specific nature of the inquiry"),
        ("noticed their friend had packed hiking boots for a trip described as relaxing",
         "gear mismatch",
         "inference of undisclosed plans from equipment choices",
         "inferring undisclosed adventure plans from the mismatch between stated and actual preparation"),
        ("watched a driver slow down and signal well before the upcoming turn",
         "telegraphed action",
         "recognition of clearly signaled upcoming behavior",
         "recognizing that the signaling communicated a firm decision to change direction"),
        ("saw a parent carefully arranging surprise decorations while the child was out",
         "covert preparation",
         "detection of concealed goal-directed preparation",
         "detecting an organized plan being carried out secretly for a future reveal"),
        ("noticed their manager scheduling individual meetings with every team member",
         "systematic pattern",
         "inference of a larger plan from a systematic behavioral pattern",
         "inferring a larger organizational decision from the systematic scheduling pattern"),
        ("observed a chess player studying the board for an unusually long time before moving",
         "strategic deliberation",
         "recognition of deep strategic planning",
         "recognizing that the extended deliberation signaled a complex strategic calculation"),
        ("saw a student secretly rehearsing a speech in the hallway before class",
         "private rehearsal",
         "detection of planned performance through practice behavior",
         "detecting an upcoming planned performance from covert rehearsal activity"),
        ("watched a negotiator carefully arrange documents before the discussion began",
         "tactical arrangement",
         "inference of strategic preparation from object placement",
         "inferring strategic preparation and a rehearsed approach from the deliberate arrangement"),
        ("noticed their sibling had been researching flight prices without mentioning travel",
         "covert research",
         "inference of undisclosed plans from browsing behavior",
         "inferring secret travel plans from observed but unacknowledged search activity"),
        ("observed a gardener marking specific spots in the yard with small flags",
         "spatial planning",
         "recognition of a staged execution plan",
         "recognizing that the flagging indicated a detailed plan for staged planting or excavation"),
        ("saw a friend slowly clear their schedule for the upcoming weekend",
         "schedule clearing",
         "inference of anticipated activity from deliberate availability creation",
         "inferring that deliberate schedule clearing signaled anticipation of an upcoming commitment"),
        ("watched a cook lay out all ingredients on the counter before beginning",
         "mise en place",
         "recognition of systematic preparation for a complex task",
         "recognizing that the organized layout reflected a systematic plan for the coming procedure"),
    ],

    "theory_of_mind": [
        ("realized their friend did not know the surprise party had been canceled",
         "false-state awareness",
         "awareness that another person held an outdated representation",
         "recognizing that another person's mental representation diverged from current reality"),
        ("understood that the child thought the candy was still in the original box",
         "classic transferred-content scenario",
         "tracking a person's outdated representation of object location",
         "tracking another mind's outdated representation after an unwitnessed change"),
        ("figured out that their coworker misunderstood the sarcastic comment as sincere",
         "communicative misfire",
         "recognizing a gap between intended and received meaning",
         "recognizing that the listener constructed a meaning opposite to the speaker's intent"),
        ("noticed the tourist was heading to the old museum location that had since moved",
         "outdated knowledge detection",
         "detecting someone acting on obsolete information",
         "detecting that another person was acting on a representation that no longer matched reality"),
        ("grasped that the doctor was withholding bad news to spare the patient worry",
         "strategic omission",
         "understanding another's protective communicative strategy",
         "understanding that selective information withholding served the other person's emotional management"),
        ("saw that the new employee did not realize the email had been sent to everyone",
         "audience miscalculation",
         "detecting a person's mistaken model of who knows what",
         "detecting that the sender's mental model of the audience was narrower than reality"),
        ("recognized that the storyteller was exaggerating to entertain rather than inform",
         "pragmatic intent detection",
         "distinguishing performative from literal communicative goals",
         "distinguishing between a performative communicative goal and a truth-conveying one"),
        ("understood why the child pointed to the empty cookie jar with a guilty expression",
         "behavioral-mental linking",
         "linking behavioral cues to an internal state the child was trying to conceal",
         "linking observable guilt cues to the child's awareness of their own transgression"),
        ("realized the interviewer was testing them with a question that had no right answer",
         "meta-communicative awareness",
         "recognizing a hidden evaluative purpose behind a question",
         "recognizing that the question's true purpose was to observe the reasoning process, not the answer"),
        ("figured out that both friends each thought the other had made the restaurant reservation",
         "mutual false assumption",
         "tracking two people's incompatible assumptions simultaneously",
         "simultaneously tracking two minds each holding a false assumption about the other's actions"),
        ("understood that the vendor raised the price because they sensed the buyer was wealthy",
         "strategic inference",
         "recognizing that someone modeled and exploited another's perceived traits",
         "recognizing that the vendor modeled the buyer's economic status and adjusted strategy accordingly"),
        ("noticed the teacher simplified the explanation because the student looked confused",
         "responsive calibration",
         "recognizing adaptive communication based on another's comprehension state",
         "recognizing that the speaker dynamically adjusted their communication to match the listener's state"),
        ("grasped that the magician wanted the audience to look at the wrong hand",
         "misdirection awareness",
         "understanding deliberate manipulation of another's attention focus",
         "understanding that the performer deliberately engineered a false attentional focus in the audience"),
        ("realized their elderly relative was pretending not to need help out of pride",
         "concealed need detection",
         "seeing through a deliberate self-presentation to the state beneath",
         "perceiving that a deliberately maintained self-presentation concealed an actual need"),
        ("understood that the negotiator's aggressive opening was a tactic, not genuine hostility",
         "strategic display recognition",
         "distinguishing performed from genuine emotional display",
         "distinguishing a strategically performed emotional display from a genuinely felt internal state"),
    ],

    "empathy": [
        ("watched a friend try to hold back tears while describing a family loss",
         "grief sharing",
         "shared emotional resonance with another's suffering",
         "a deep shared emotional resonance that mirrored the other person's pain"),
        ("saw a child fall on the playground and immediately felt a pang in their own chest",
         "vicarious pain",
         "automatic somatic resonance with observed distress",
         "an automatic somatic echo of the observed physical pain and emotional upset"),
        ("listened to a stranger on the phone describe being lonely and felt their own mood drop",
         "mood contagion",
         "involuntary affective mirroring of another's distress",
         "involuntary affective contagion as the stranger's loneliness resonated internally"),
        ("noticed a colleague fighting back frustration during a harsh review and felt tense themselves",
         "vicarious tension",
         "somatic mirroring of observed emotional suppression",
         "tension that arose from somatically mirroring the colleague's suppressed frustration"),
        ("comforted a neighbor who had just received devastating medical news",
         "crisis support",
         "deep emotional attunement with someone in crisis",
         "a deep emotional attunement calibrated to the severity of the neighbor's distress"),
        ("saw an elderly person struggling with heavy bags and felt a pull to help",
         "compassionate urge",
         "compassion-driven motivation from observing another's difficulty",
         "a compassion-driven motivational pull arising from observing vulnerability and struggle"),
        ("felt their eyes water while watching a documentary about displaced families",
         "media-evoked concern",
         "emotional contagion from a portrayed suffering narrative",
         "emotional contagion triggered by a vivid portrayal of real human displacement and loss"),
        ("sensed a friend's embarrassment when they misspoke in public and felt it too",
         "vicarious embarrassment",
         "shared self-conscious affect from witnessing social mishap",
         "a shared flush of self-conscious discomfort upon witnessing the friend's public mistake"),
        ("held the hand of a patient who was anxious before surgery and absorbed their worry",
         "bedside absorption",
         "affective absorption of another's pre-procedural anxiety",
         "absorbing the patient's anxiety through physical contact and emotional attunement"),
        ("read about a humanitarian crisis and felt overwhelmed by the scale of human suffering",
         "distant suffering",
         "overwhelming concern for large-scale human hardship",
         "an overwhelming wave of concern that extended beyond any single person to collective suffering"),
        ("winced when their teammate twisted an ankle during the game",
         "vicarious injury",
         "reflexive physical co-experience of observed injury",
         "a reflexive physical wince that mirrored the teammate's sudden pain"),
        ("matched the slow, somber pace of a grieving friend during a walk",
         "behavioral synchrony",
         "unconscious behavioral alignment with another's emotional state",
         "unconscious behavioral synchrony driven by attunement to the friend's grief"),
        ("felt a wave of relief alongside a student who finally passed the difficult exam",
         "shared relief",
         "co-experienced positive affect following another's release from stress",
         "a shared wave of positive affect that mirrored the student's release from sustained worry"),
        ("teared up watching a parent reunite with a child after years of separation",
         "reunion witness",
         "emotional co-experience of observed joy and relief",
         "an overwhelming emotional co-experience triggered by witnessing reunion after prolonged separation"),
        ("sensed the loneliness of a new coworker eating alone and invited them to join",
         "social isolation detection",
         "perspective-taking that motivated prosocial intervention",
         "detecting isolation through perspective-taking and being moved to act on the resulting concern"),
    ],

    "self_referential": [
        ("reflected on how their personality had changed over the past decade",
         "temporal self-comparison",
         "introspective continuity assessment across time",
         "a sustained introspective process comparing past and present versions of the self"),
        ("considered whether their career truly reflected their core values",
         "value-action alignment check",
         "evaluative self-reflection on personal authenticity",
         "an evaluative self-reflection probing the alignment between inner values and outer choices"),
        ("thought about which of their traits they inherited from each parent",
         "origin tracing",
         "introspective attribution of personal characteristics",
         "an introspective attribution process linking current traits to their developmental origins"),
        ("asked themselves whether they were the kind of person who would speak up for a stranger",
         "hypothetical self-test",
         "self-concept evaluation through a hypothetical moral scenario",
         "testing their own self-concept against a hypothetical scenario requiring personal courage"),
        ("wondered how differently their life would have turned out with different choices",
         "counterfactual self-narrative",
         "inward-directed counterfactual reasoning about personal trajectory",
         "engaging in counterfactual personal reasoning about how alternate choices would have reshaped identity"),
        ("recalled how they used to react to conflict and compared it to how they react now",
         "behavioral self-tracking",
         "self-monitoring of personal growth over time",
         "self-monitoring that compared past and present behavioral responses to track personal growth"),
        ("examined whether their discomfort with the situation was about principles or pride",
         "motive self-scrutiny",
         "introspective disambiguation of personal motives",
         "an introspective effort to disambiguate whether the discomfort stemmed from principle or ego"),
        ("considered what they would want written in their own biography",
         "legacy reflection",
         "inward narrative construction about personal legacy",
         "a deeply personal narrative exercise envisioning how one's life story should be told"),
        ("debated internally whether they truly enjoyed their hobby or just did it out of habit",
         "preference authenticity check",
         "introspective probing of genuine versus habitual preference",
         "probing whether a longstanding habit still reflected genuine preference or mere inertia"),
        ("reflected on why certain criticisms bothered them more than others",
         "sensitivity mapping",
         "introspective analysis of emotional vulnerabilities",
         "an introspective analysis mapping which criticisms exposed genuine personal vulnerabilities"),
        ("thought about what kind of friend they had been during a past crisis",
         "retrospective role evaluation",
         "self-evaluation of past interpersonal performance",
         "a retrospective self-evaluation assessing their adequacy in a past support role"),
        ("assessed whether their political views had shifted since college and why",
         "ideological self-audit",
         "personal tracking of conviction evolution over time",
         "a personal audit tracking how and why deeply held convictions had evolved"),
        ("wondered whether their fear of public speaking said something deeper about them",
         "symptom-to-identity inference",
         "inward interpretation of a behavioral pattern",
         "interpreting a specific behavioral tendency as a window into deeper aspects of personality"),
        ("considered whether the image they projected at work was authentic or performed",
         "authenticity audit",
         "introspective evaluation of social self-presentation",
         "evaluating the gap between their public professional persona and private sense of self"),
        ("reflected on how a childhood experience had shaped their adult attachment style",
         "developmental self-theory",
         "introspective causal reasoning about personal development",
         "constructing a causal self-narrative linking early experience to current relational patterns"),
    ],

    "judgment": [
        ("evaluated whether the landlord's decision to evict the struggling tenant was justified",
         "fairness evaluation",
         "normative evaluation of another's decision",
         "a normative evaluation weighing the landlord's rights against the tenant's vulnerability"),
        ("considered whether a friend was right to cut off contact with a toxic family member",
         "relational boundary assessment",
         "evaluative reasoning about interpersonal boundary decisions",
         "evaluative reasoning about whether the severity of toxicity warranted the relational severance"),
        ("assessed whether the company's policy of monitoring employee emails was reasonable",
         "institutional policy evaluation",
         "normative assessment of an institutional practice",
         "a normative assessment balancing organizational security against individual privacy"),
        ("weighed whether the coach's decision to bench the star player was the right call",
         "strategic decision evaluation",
         "evaluative analysis of a consequential leadership decision",
         "an evaluative analysis of whether the coaching decision served the team's overall interests"),
        ("debated whether the journalist was right to publish the leaked documents",
         "press ethics evaluation",
         "normative reasoning about competing obligations",
         "normative reasoning balancing the public's right to know against potential harm from disclosure"),
        ("questioned whether the school's zero-tolerance policy was too harsh for the situation",
         "policy proportionality",
         "evaluative assessment of proportionality in rule application",
         "an evaluative assessment of whether the punishment's severity matched the violation's gravity"),
        ("decided that the neighbor's complaint about noise was reasonable given the circumstances",
         "dispute adjudication",
         "forming a considered verdict about a social dispute",
         "forming a considered evaluative verdict by weighing competing claims in a social dispute"),
        ("assessed whether a charity was spending its donations responsibly or wastefully",
         "organizational audit",
         "evaluative scrutiny of institutional resource allocation",
         "evaluative scrutiny of whether the charity's allocation of resources matched its stated mission"),
        ("evaluated whether the teacher was fair in giving the same grade to unequal work",
         "equity assessment",
         "normative evaluation of fairness in differential treatment",
         "a normative evaluation asking whether equality of outcome was just despite inequality of input"),
        ("considered whether the referee's controversial call changed the outcome of the match",
         "consequential ruling assessment",
         "evaluating the downstream consequences of an official decision",
         "evaluating whether the official's decision materially altered the outcome it was meant to adjudicate"),
        ("weighed whether a doctor's decision to override a patient's wishes was appropriate",
         "autonomy-beneficence evaluation",
         "normative assessment of paternalistic intervention",
         "a normative assessment of whether beneficent override of patient autonomy was justified"),
        ("assessed whether the company was right to prioritize profits over environmental concerns",
         "stakeholder tradeoff evaluation",
         "evaluative reasoning about competing stakeholder interests",
         "evaluative reasoning about whether shareholder obligations should outweigh ecological responsibility"),
        ("decided whether the student's excuse for missing the deadline was credible enough",
         "credibility evaluation",
         "evaluative assessment of the plausibility of a claim",
         "an evaluative assessment probing the credibility and plausibility of the presented excuse"),
        ("questioned whether the government's response to the crisis was adequate or negligent",
         "institutional response evaluation",
         "normative evaluation of an authority's crisis response",
         "a normative evaluation of whether the institutional response met reasonable standards of adequacy"),
        ("evaluated whether a friend's apology was sincere enough to warrant reconciliation",
         "sincerity assessment",
         "evaluative probing of communicative authenticity",
         "an evaluative probing of whether the apology's emotional quality warranted restored trust"),
    ],

    "moral": [
        ("discovered that a colleague had been taking supplies home without permission",
         "workplace theft discovery",
         "ethical evaluation of a transgression against shared resources",
         "a process of ethical evaluation weighing the severity of taking communal property without consent"),
        ("learned that a friend had lied to protect someone from hearing a painful truth",
         "protective deception",
         "ethical reasoning about whether deception for protection is permissible",
         "ethical reasoning about whether the benevolent motive justified the deceptive means"),
        ("witnessed a bystander ignore a person who had collapsed on the street",
         "omission of aid",
         "ethical evaluation of failure to act in an emergency",
         "an ethical evaluation of the bystander's failure to render aid to a person in distress"),
        ("found out that a company had been dumping waste illegally to save costs",
         "environmental violation",
         "ethical condemnation of profit-driven harm to the commons",
         "ethical condemnation of prioritizing profit over the shared environment and public health"),
        ("heard that a parent had lied about their address to get their child into a better school",
         "boundary manipulation",
         "ethical reasoning about rule-breaking motivated by parental concern",
         "weighing the ethical tension between fraudulent means and the motive to benefit a child"),
        ("saw a store owner charge different prices to customers based on their appearance",
         "discriminatory pricing",
         "ethical evaluation of differential treatment based on perceived status",
         "an ethical evaluation of whether discriminatory pricing constituted an unjust violation of equality"),
        ("considered the case of a whistleblower who exposed wrongdoing but broke a confidentiality oath",
         "competing obligations",
         "ethical reasoning about conflicting duties of loyalty and transparency",
         "ethical reasoning about whether the duty to expose wrongdoing overrode the duty of confidentiality"),
        ("reflected on whether it was acceptable to keep excess change given by a cashier in error",
         "minor windfall",
         "ethical evaluation of benefiting from another's mistake",
         "an ethical evaluation of whether passively benefiting from another's error constituted dishonesty"),
        ("evaluated a situation where someone copied a student's homework and both received credit",
         "academic integrity",
         "ethical assessment of complicity in rule violation",
         "ethical assessment of shared culpability when one party copies and the other tacitly allows it"),
        ("thought about whether a doctor should honor a terminally ill patient's wish to end treatment",
         "end-of-life autonomy",
         "ethical deliberation about autonomy versus preservation of life",
         "deep ethical deliberation balancing respect for autonomy against the imperative to preserve life"),
        ("considered whether redistributing wealth through heavy taxation was just or oppressive",
         "distributive tension",
         "ethical reasoning about fairness in resource redistribution",
         "ethical reasoning about whether compulsory redistribution served justice or violated individual liberty"),
        ("reflected on whether using animals for medical research could be ethically justified",
         "cross-species obligation",
         "ethical evaluation of instrumental use of sentient beings",
         "an ethical evaluation weighing potential human benefit against the suffering of sentient creatures"),
        ("learned that an aid worker had diverted supplies to their own community first",
         "partiality in distribution",
         "ethical evaluation of impartial duty versus personal loyalty",
         "an ethical evaluation of whether personal loyalty justified deviating from impartial distribution norms"),
        ("debated whether it was permissible to break a promise to attend a friend's event for a work emergency",
         "promise-breaking",
         "ethical reasoning about the conditions under which commitments may be broken",
         "ethical reasoning about whether the severity of the competing obligation excused the broken promise"),
        ("considered whether publicly shaming someone for bad behavior was an acceptable form of accountability",
         "public accountability",
         "ethical evaluation of punitive social exposure",
         "an ethical evaluation of whether public shaming served accountability or constituted disproportionate harm"),
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────

def generate_stimuli(seed: int = 20260603) -> list[dict]:
    """Generate template-matched stimuli for all 14 conditions."""
    rng = random.Random(seed)
    stimuli = []
    stim_id = 0

    conditions = sorted(CONDITION_SCENARIOS.keys())

    for condition in conditions:
        scenarios = CONDITION_SCENARIOS[condition]
        assert len(scenarios) >= 15, (
            f"{condition} has only {len(scenarios)} scenarios (need >= 15)")

        person_pool = list(PERSONS)
        rng.shuffle(person_pool)

        for tmpl_name, tmpl_str in TEMPLATES.items():
            for i, (scenario, scenario_short, outcome, outcome_long) in enumerate(scenarios):
                person, pronoun = person_pool[i % len(person_pool)]

                text = tmpl_str.format(
                    person=person,
                    person_pronoun=pronoun,
                    scenario=scenario,
                    scenario_short=scenario_short,
                    outcome=outcome,
                    outcome_long=outcome_long,
                )

                stimuli.append({
                    "id": f"tm_{condition}_{stim_id:04d}",
                    "condition": condition,
                    "text": text,
                    "template": tmpl_name,
                    "source": "template_matched",
                })
                stim_id += 1

    return stimuli


def validate_stimuli(stimuli: list[dict]) -> dict:
    """Check quality constraints: length balance, no label leakage."""
    import re
    from collections import defaultdict

    # Condition label words to ban from stimuli text (matched at word boundaries)
    LABEL_WORDS = {
        "anger": {"anger", "angry"},
        "fear": {"fear", "feared", "fearful", "fearing"},
        "disgust": {"disgust", "disgusted", "disgusting"},
        "sadness": {"sadness", "sad"},
        "happiness": {"happiness", "happy"},
        "valence": {"valence"},
        "belief": {"belief", "beliefs", "believe", "believed", "believing"},
        "mentalizing": {"mentalizing", "mentalize", "mentalized"},
        "intention": {"intention", "intentions", "intend", "intended", "intending"},
        "theory_of_mind": {"theory of mind"},
        "empathy": {"empathy", "empathize", "empathized", "empathetic", "empathic"},
        "self_referential": {"self-referential", "self referential"},
        "judgment": {"judgment", "judgement", "judge", "judging", "judged"},
        "moral": {"moral", "morals", "morality", "immoral", "morally"},
    }

    cond_lengths = defaultdict(list)
    label_leaks = []
    for s in stimuli:
        words = s["text"].split()
        cond_lengths[s["condition"]].append(len(words))

        text_lower = s["text"].lower()
        for banned in LABEL_WORDS.get(s["condition"], set()):
            # Use word-boundary matching to avoid false positives
            # (e.g., "ambivalence" should not match "valence")
            pattern = r'\b' + re.escape(banned) + r'\b'
            if re.search(pattern, text_lower):
                label_leaks.append((s["id"], s["condition"], banned))

    # Check length balance: per-condition mean should be within +/- 20% of grand mean
    all_means = {c: statistics.mean(lens) for c, lens in cond_lengths.items()}
    grand_mean = statistics.mean(all_means.values())
    length_violations = {}
    for c, m in all_means.items():
        dev = abs(m - grand_mean) / grand_mean
        if dev > 0.20:
            length_violations[c] = {"mean": m, "deviation": f"{dev:.1%}"}

    return {
        "n_conditions": len(cond_lengths),
        "n_stimuli_per_condition": {c: len(v) for c, v in sorted(cond_lengths.items())},
        "grand_mean_words": round(grand_mean, 1),
        "per_condition_mean_words": {c: round(m, 1) for c, m in sorted(all_means.items())},
        "length_violations": length_violations,
        "label_leaks": label_leaks,
        "total_stimuli": len(stimuli),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate template-matched stimuli for 14-condition RSA")
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSONL path (default: rsa_stimuli_template_matched.jsonl)")
    ap.add_argument("--validate-only", action="store_true",
                    help="Print validation stats without saving")
    args = ap.parse_args()

    stimuli = generate_stimuli(seed=args.seed)

    # ── Validation ──
    report = validate_stimuli(stimuli)
    print(f"Generated {report['total_stimuli']} stimuli across "
          f"{report['n_conditions']} conditions")
    print(f"Grand mean word count: {report['grand_mean_words']}")
    print(f"\nPer-condition counts and mean word lengths:")
    for c in sorted(report["n_stimuli_per_condition"].keys()):
        n = report["n_stimuli_per_condition"][c]
        m = report["per_condition_mean_words"][c]
        print(f"  {c:20s}  n={n:3d}  mean_words={m:5.1f}")

    if report["length_violations"]:
        print(f"\nWARNING: {len(report['length_violations'])} conditions "
              f"exceed ±20% length tolerance:")
        for c, info in report["length_violations"].items():
            print(f"  {c}: mean={info['mean']:.1f} ({info['deviation']})")

    if report["label_leaks"]:
        print(f"\nWARNING: {len(report['label_leaks'])} label word leaks found:")
        for sid, cond, word in report["label_leaks"][:20]:
            print(f"  {sid} ({cond}): contains '{word}'")
        if len(report["label_leaks"]) > 20:
            print(f"  ... and {len(report['label_leaks']) - 20} more")

    if args.validate_only:
        return

    # ── Save ──
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (Path(__file__).resolve().parents[1]
                    / "data" / "cognitive_stimuli" / "rsa"
                    / "rsa_stimuli_template_matched.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for s in stimuli:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved to {out_path}")

    # ── Also save a manifest ──
    manifest = {
        "seed": args.seed,
        "n_templates": len(TEMPLATES),
        "templates": list(TEMPLATES.keys()),
        "target_per_condition": 60,
        "conditions": {}
    }
    for c in sorted(report["n_stimuli_per_condition"].keys()):
        manifest["conditions"][c] = {
            "n_total": report["n_stimuli_per_condition"][c],
            "source": "template_matched",
            "mean_word_length": report["per_condition_mean_words"][c],
        }
    manifest_path = out_path.parent / "rsa_template_matched_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
