#!/usr/bin/env python3
"""
Template-matched stimuli V2 for the 14-condition RSA.

Motivation: V1 template-matched stimuli used SINGLE-SENTENCE templates (20-28 words)
that were too restrictive for social cognition conditions. Social conditions like
false belief NEED multi-character information asymmetry, ToM NEEDS multi-perspective
tracking, and intention NEEDS indirect evidence of hidden goals. The V1 templates
destroyed this cognitive content, causing within-social RSA to collapse from 0.55
to 0.12 -- an artifact of content destruction, not format control.

V2 design:
  - 4 multi-sentence templates (3-4 sentences each, 35-50 words)
  - Rich enough to express social cognition's structural requirements
  - ALL 14 conditions use the SAME 4 templates -- same structure, same voice,
    same tense -- only the cognitive content differs
  - 15 scenario tuples per condition per template = 60 stimuli per condition
  - 840 total stimuli (14 conditions x 4 templates x 15 scenarios)
  - Gender-neutral names, third person, past tense, neutral tone
  - NO condition label words in stimuli
  - NO dialogue quotes, first-person introspection, or evaluative normative language

Output: data/cognitive_stimuli/rsa/rsa_stimuli_template_matched_v2.jsonl
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 4 multi-sentence templates
# Each takes: person, setup, event, consequence, experience
# All produce 3-4 sentences targeting 35-50 words
# ──────────────────────────────────────────────────────────────────────

TEMPLATES = {
    "vignette": (
        "{person} was {setup}. {event}. {consequence}. "
        "{person} found that the experience centrally involved {experience}."
    ),
    "report": (
        "According to available information, {person} was {setup}. "
        "Subsequently, {event}. As a result, {consequence}. "
        "The notable aspect was {experience}."
    ),
    "sequence": (
        "{person} encountered the following situation. First, {person} was {setup}. "
        "Then, {event}. This led to a state where {consequence}, "
        "which involved {experience}."
    ),
    "summary": (
        "Background: {person} was {setup}. "
        "Development: {event}. "
        "Current state: {consequence}. "
        "Central experience: {experience}."
    ),
}

# ──────────────────────────────────────────────────────────────────────
# Named persons (rotated across stimuli for variety)
# ──────────────────────────────────────────────────────────────────────

PERSONS = [
    "Alex", "Jordan", "Morgan", "Taylor", "Casey",
    "Riley", "Sam", "Quinn", "Dana", "Robin",
    "Jamie", "Avery", "Blake", "Cameron", "Drew",
]

# ──────────────────────────────────────────────────────────────────────
# Per-condition scenario tuples
#
# Each entry: (setup, event, consequence, experience)
#   setup       — initial context / background
#   event       — what happened / changed
#   consequence — the resulting state
#   experience  — the central cognitive/emotional process
#
# RULES:
# 1. Never use the condition label word itself
# 2. Social conditions must preserve their cognitive structure:
#    - belief: person holds WRONG information (information asymmetry)
#    - theory_of_mind: multiple agents with DIFFERENT knowledge states
#    - intention: hidden goals inferred from indirect behavior
#    - mentalizing: reading hidden mental states from observable behavior
#    - empathy: resonating with another's emotional state
#    - self_referential: introspection about own mental states
#    - judgment: evaluating right/wrong in ethically tense situations
#    - moral: ethical reasoning from philosophical perspective
# 3. Emotion conditions must have correct diagnostic appraisals:
#    - anger: injustice / other-blame
#    - sadness: irreversible loss
#    - fear: future threat / imminent danger
#    - disgust: contamination / boundary violation
#    - happiness: goal achievement / social bonding
#    - valence: mixed positive-negative evaluation
# ──────────────────────────────────────────────────────────────────────

CONDITION_SCENARIOS = {
    # ── AFFECTIVE (6) ──

    "anger": [
        (
            "working on a team project for several months",
            "a colleague presented the completed work to management as solely their own creation",
            "the credit for months of collaborative effort was publicly attributed to one person",
            "a surge of indignation at the deliberate misattribution of shared labor",
        ),
        (
            "renting an apartment with a strict no-entry agreement",
            "the landlord entered the unit without notice and rearranged personal belongings",
            "private spaces had been invaded despite clear contractual boundaries",
            "fury at a trusted authority figure violating established personal boundaries",
        ),
        (
            "filing a legitimate insurance claim after documented property damage",
            "the insurance company denied the claim citing fabricated technicalities",
            "a valid request was rejected through bureaucratic manipulation",
            "mounting outrage at institutional bad faith and deliberate obstruction",
        ),
        (
            "tending a garden that had taken years to cultivate",
            "a neighbor deliberately poured herbicide over the flower beds overnight",
            "years of careful horticultural work were destroyed in a single act of spite",
            "wrath directed at the intentional destruction of something painstakingly built",
        ),
        (
            "walking through a neighborhood park on a quiet afternoon",
            "a passerby kicked a stray animal that was resting near a bench",
            "a defenseless creature was subjected to unprovoked cruelty in a public space",
            "fierce indignation at witnessing violence directed at a vulnerable being",
        ),
        (
            "picking up a vehicle after agreeing on a repair cost",
            "the mechanic presented a bill that was triple the original written estimate",
            "a binding price agreement was unilaterally broken to extract additional payment",
            "bitter hostility at being exploited by someone in a position of specialized knowledge",
        ),
        (
            "keeping a private journal in a locked drawer at home",
            "a roommate picked the lock and read through the personal entries",
            "deeply private reflections were accessed without any consent or justification",
            "a visceral sense of violation at the intrusion into protected inner life",
        ),
        (
            "parked legally in a marked space with a valid permit displayed",
            "an enforcement officer issued a citation despite the clearly visible permit",
            "a fine was imposed in contradiction of the observable evidence",
            "seething resentment at an authority figure acting in bad faith",
        ),
        (
            "supervising children at a community playground",
            "an older child repeatedly taunted and shoved a much younger one while adults watched",
            "a power imbalance was being exploited openly without intervention from bystanders",
            "a protective rage at the combination of cruelty and collective inaction",
        ),
        (
            "sitting for an important certification exam after months of preparation",
            "the proctor publicly accused the test-taker of copying without any evidence",
            "a reputation was damaged in front of peers through a baseless allegation",
            "explosive indignation at the injustice of being branded a cheater without cause",
        ),
        (
            "locking a bicycle securely outside a workplace building each morning",
            "someone cut the lock and scratched obscenities into the frame overnight",
            "functional property was deliberately ruined as an act of targeted vandalism",
            "a surge of frustration at senseless destructiveness aimed at personal belongings",
        ),
        (
            "confiding personal struggles to a close and trusted companion over many years",
            "the companion had been sharing those confidences as gossip throughout the social circle",
            "private vulnerabilities were transformed into public entertainment",
            "deep-seated wrath at the systematic betrayal of longstanding trust",
        ),
        (
            "driving cautiously through an intersection during heavy traffic",
            "another driver cut across lanes at high speed and then made a threatening gesture",
            "a dangerous maneuver was paired with hostile provocation on a public road",
            "a flash of aggressive impulse toward the reckless driver",
        ),
        (
            "employed at a company for over two years with consistent performance",
            "payroll records revealed systematic underpayment across every pay period",
            "a pattern of wage manipulation had been concealed by the employer for months",
            "burning resentment at prolonged and deliberate financial exploitation",
        ),
        (
            "watching a public official address a crowd at a civic event",
            "the speaker made claims that directly contradicted documented records accessible to everyone",
            "verifiable facts were distorted in front of an audience that could check them",
            "sharp contempt at the brazenness of deliberate public dishonesty",
        ),
    ],

    "fear": [
        (
            "alone in a house late at night with all doors locked",
            "heavy footsteps echoed from the supposedly empty floor above",
            "the sound pattern suggested someone uninvited was moving through the building",
            "a gripping dread that the secured space had been breached by an intruder",
        ),
        (
            "seated on a commercial flight at cruising altitude over open ocean",
            "the aircraft shuddered violently and dropped altitude without warning",
            "passengers were thrown against restraints as objects fell from overhead compartments",
            "overwhelming terror at being trapped in an uncontrollable descent",
        ),
        (
            "returning home after a routine day at work",
            "an envelope with no return address contained a handwritten warning of physical harm",
            "an anonymous source had identified the home address and communicated a specific threat",
            "a wave of dread at the realization of being targeted by an unknown adversary",
        ),
        (
            "hiking alone on a narrow trail through dense vegetation",
            "a large coiled serpent appeared directly on the path less than two meters ahead",
            "the route forward was blocked by an animal capable of delivering a lethal strike",
            "an instinctive freeze response combined with visceral alarm at proximity to danger",
        ),
        (
            "working on the third floor of an office building on an ordinary weekday",
            "the entire structure began swaying laterally as ceiling fixtures fell",
            "the ground had become unreliable and the building's structural integrity was uncertain",
            "primal alarm and a desperate urgency to locate shelter in an unstable environment",
        ),
        (
            "waking from sleep in a dark and silent room",
            "full consciousness returned but neither limbs nor vocal cords would respond to commands",
            "the body remained completely immobilized while the mind was fully alert",
            "suffocating horror at being conscious but entirely trapped inside a paralyzed body",
        ),
        (
            "driving alone on an unlit rural highway after midnight",
            "the same set of headlights appeared in the mirror after every turn for several kilometers",
            "a vehicle was maintaining deliberate pace through multiple route changes",
            "mounting hypervigilance and frantic scanning for escape routes or populated areas",
        ),
        (
            "swimming at moderate depth in murky lake water far from shore",
            "something large and solid brushed firmly against both legs from below",
            "an unseen creature of substantial size was moving directly underneath",
            "a jolt of panic and frantic movement toward the distant shore",
        ),
        (
            "sleeping in a hotel room on the eighth floor during a business trip",
            "the corridor filled with acrid smoke and fire alarms triggered simultaneously",
            "an evacuation was required from a high floor with limited knowledge of exit routes",
            "raw panic coupled with disorientation about the safest path out",
        ),
        (
            "standing on a glass observation platform extending from a tall structure",
            "looking directly down revealed a vertical drop of several hundred meters to the ground",
            "the transparent floor created an unobstructed view of the lethal distance below",
            "paralyzing vertigo and an irrational conviction of imminent falling",
        ),
        (
            "waiting in a clinic examination room for routine test results",
            "the physician re-entered with an unusually grave expression and requested a longer appointment",
            "the medical professional's demeanor suggested the results contained something serious",
            "piercing alarm and a flood of catastrophic speculation about personal health",
        ),
        (
            "walking through an unfamiliar part of a city after dark",
            "the only exit from a narrow passage was blocked by several figures who did not move aside",
            "retreat was as constrained as forward progress in an isolated location",
            "instant terror at being cornered by strangers in an unmonitored space",
        ),
        (
            "home alone during a severe weather event with sirens sounding outside",
            "the warning system upgraded to indicate a tornado was forming in the immediate area",
            "the dwelling had no basement or reinforced interior room for adequate shelter",
            "overwhelming helplessness at having no safe refuge from an approaching natural force",
        ),
        (
            "regaining consciousness in an unfamiliar room with no memory of arrival",
            "the surroundings contained no recognizable objects and the exit was not immediately visible",
            "the gap between last memory and current situation could not be accounted for",
            "a disorienting wave of vulnerability at the total absence of situational understanding",
        ),
        (
            "watching a young child run across a grassy area toward a busy road",
            "the child showed no awareness of the fast-moving vehicles in both lanes",
            "the distance was too great for immediate physical intervention to be certain",
            "a surge of protective panic driving desperate acceleration toward the child",
        ),
    ],

    "disgust": [
        (
            "cleaning out a refrigerator that had been unplugged for several weeks",
            "opening a sealed container released a wave of putrid gas from decomposing organic matter",
            "the olfactory assault triggered an immediate gagging reflex",
            "visceral revulsion at the sensory properties of advanced biological decay",
        ),
        (
            "walking barefoot through a bathroom in dim lighting",
            "one foot landed directly in a cold slimy substance of unknown origin on the tile",
            "the tactile properties of the substance indicated something organic and decomposing",
            "acute physical recoil at unexpected skin contact with an unidentified foul material",
        ),
        (
            "watching an educational documentary about tropical parasitic infections",
            "close-up footage showed larvae actively feeding inside an open wound on a limb",
            "the visual detail of organisms consuming living tissue was sustained and explicit",
            "overwhelming nausea and a compulsive need to avert the gaze from the screen",
        ),
        (
            "investigating a persistent musty odor in a bedroom",
            "pulling back the wallpaper revealed extensive colonies of black growth covering the surface",
            "the living space had been hosting active fungal proliferation behind the visible surfaces",
            "deep revulsion combined with alarm about prolonged exposure to biological contamination",
        ),
        (
            "walking along a residential street after heavy rainfall",
            "a ruptured pipe had flooded the pavement with raw sewage from the municipal system",
            "the roadway was covered in effluent and the stench was detectable from a distance",
            "intense repulsion causing involuntary breath-holding and a wide detour",
        ),
        (
            "retrieving a pan from a kitchen cabinet that had been closed for months",
            "behind the pan lay the partially liquefied remains of a large rodent",
            "the combination of visual decomposition and concentrated odor was immediate",
            "stomach-turning nausea at the proximity of advanced carcass putrefaction",
        ),
        (
            "taking a seat on a crowded public transit vehicle during a commute",
            "the seat surface was coated with a warm sticky residue of indeterminate origin",
            "clothing and skin had already made contact before the substance was noticed",
            "immediate physical recoil and anxious fixation on what the substance might contain",
        ),
        (
            "filling a water bottle at a public drinking fountain in a park",
            "another person approached and ejected a large volume of mucus directly into the basin",
            "the shared water source was visibly contaminated by a deliberate act",
            "a wave of revulsion at witnessing the fouling of a communal resource",
        ),
        (
            "seated at a restaurant table waiting for a meal to arrive",
            "several large insects scattered rapidly across the table surface when a plate was set down",
            "the dining surface was actively hosting a pest population in a food-service environment",
            "intense aversion that instantly eliminated any remaining appetite",
        ),
        (
            "working in close proximity to a colleague in a shared office space",
            "the colleague had clearly not maintained basic hygiene for an extended period",
            "the concentrated body odor in the confined space was inescapable and intensifying",
            "strong physical aversion combined with social discomfort at the unavoidable proximity",
        ),
        (
            "waiting in line at a buffet where food was served with communal utensils",
            "the person ahead used their fingers to excavate their nasal passage then handled the serving spoon",
            "the shared food-contact surface was contaminated by an unsanitary hand",
            "sharp revulsion and an immediate decision to avoid the contaminated dish entirely",
        ),
        (
            "biting into a piece of ripe fruit taken from a bowl on the counter",
            "the interior of the fruit contained the bisected remains of a large larva",
            "part of the organism had already been in the mouth before the visual discovery",
            "a sickening wave of nausea combined with frantic oral expulsion of the contents",
        ),
        (
            "entering a public restroom facility at a highway rest stop",
            "every surface was coated in accumulated filth and the floor had standing liquid waste",
            "the facility had clearly not been maintained for a prolonged period",
            "overpowering physical repulsion forcing shallow mouth-breathing and rapid exit",
        ),
        (
            "sitting on a bus next to a passenger who appeared unwell",
            "the passenger vomited copiously onto the seat and floor in the immediate vicinity",
            "the ejected material was in close physical proximity with no way to increase distance",
            "sympathetic nausea and an instinctive withdrawal from the zone of contamination",
        ),
        (
            "attempting to clear a slow-draining sink in a shared bathroom",
            "removing the drain cover exposed a compacted mass of wet hair matted with gelatinous residue",
            "the blockage required manual removal and the texture was viscous and fibrous",
            "tactile revulsion at the prospect of handling the slimy organic obstruction",
        ),
    ],

    "sadness": [
        (
            "maintaining a close bond with a childhood companion for over two decades",
            "a phone call brought the news that the companion had died suddenly overnight",
            "a relationship that had been a constant presence was permanently and irreversibly ended",
            "profound grief at the finality of a loss that permitted no further contact",
        ),
        (
            "visiting an elderly parent who had been experiencing cognitive decline",
            "the parent looked up and asked who the visitor was with genuine confusion",
            "the person who had once been the most familiar face was no longer recognized",
            "deep sorrow at watching a fundamental connection dissolve through neurological erosion",
        ),
        (
            "returning to the neighborhood where all formative years were spent",
            "the family home had been demolished and replaced with a commercial structure",
            "the physical anchor of all childhood memories no longer existed in any form",
            "a heavy melancholy for a place that now persisted only in private recollection",
        ),
        (
            "sorting through boxes of belongings inherited from a grandparent who died before they met",
            "a bundle of handwritten letters expressed deep hopes for the grandchild's future",
            "the words reached across death from someone who wanted the connection but never had it",
            "a bittersweet longing for a relationship that was wanted on both sides but never realized",
        ),
        (
            "watching a pet sit by the front door every evening at the same time",
            "the animal was waiting for a family member who had moved away permanently",
            "the creature's routine reflected a bond it could not understand had ended",
            "a wrenching ache at witnessing faithful devotion directed toward an empty threshold",
        ),
        (
            "hearing a particular piece of music playing in a public space",
            "the melody had been closely associated with a relationship that ended years earlier",
            "the auditory trigger reactivated a web of memories that had been carefully filed away",
            "a piercing longing that reopened emotional wounds previously thought healed",
        ),
        (
            "learning about the closing of a local institution that served vulnerable children",
            "the funding had been withdrawn and no alternative placement existed for the residents",
            "children who had found stability in that setting would lose their only consistent home",
            "a heavy sense of helplessness at witnessing structural abandonment of dependent lives",
        ),
        (
            "walking through a neighborhood that had once been vibrant with community life",
            "the storefronts were boarded and the streets showed years of accumulated neglect",
            "the contrast between remembered vitality and present decay was stark and total",
            "a desolate feeling at confronting the irreversible deterioration of a once-thriving place",
        ),
        (
            "sitting beside a close companion who was weeping after a relationship ended",
            "the companion could not speak but held on tightly while the tears continued",
            "there was no remedy to offer and the only available response was physical presence",
            "an aching resonance with another person's pain while having no means to relieve it",
        ),
        (
            "discovering an old personal journal from the most difficult period of life",
            "the entries described daily suffering in raw language that the present self had forgotten",
            "the written record preserved a version of the self that had endured sustained distress",
            "retrospective anguish at revisiting documented evidence of past personal suffering",
        ),
        (
            "standing at the grave of a teacher who had provided pivotal guidance years earlier",
            "the inscription was simple and the surrounding plot suggested few visitors",
            "the person whose influence had shaped a life trajectory was now largely unremembered",
            "an overwhelming mixture of gratitude and grief for someone who could never be thanked again",
        ),
        (
            "arriving at a hospital room to say farewell to a dying relative",
            "the relative had passed twenty minutes before the arrival",
            "the opportunity for a final exchange of words was permanently missed",
            "crushing regret at the irreversibility of a missed final moment of connection",
        ),
        (
            "watching the last evening light of a trip that could never be repeated",
            "the awareness that the experience was already ending colored every remaining minute",
            "the present moment was simultaneously at its most vivid and most transient",
            "a bittersweet recognition that the most valued experiences cannot be preserved",
        ),
        (
            "realizing during a routine evening that contact with a once-close companion had lapsed",
            "neither party had initiated communication in over a year despite living in the same city",
            "a meaningful bond had dissolved through mutual neglect without a defined ending",
            "quiet grief at the recognition that a valued connection simply evaporated",
        ),
        (
            "passing a person sleeping on a cold sidewalk wrapped in a thin blanket",
            "the person was clutching a worn photograph and showed signs of prolonged exposure",
            "the scale of human need visible in that single scene exceeded any available response",
            "deep compassionate sorrow at witnessing suffering that no individual action could resolve",
        ),
    ],

    "happiness": [
        (
            "checking an application portal after months of waiting",
            "the screen displayed a formal notice of acceptance to the top-choice program",
            "years of sustained effort had culminated in the desired outcome",
            "elation at the confirmation that persistent work produced the intended result",
        ),
        (
            "sitting on a living room floor with a toddler who had been trying to stand",
            "the child released the furniture edge and took several independent steps across the room",
            "a developmental milestone was reached in an ordinary domestic moment",
            "pure delight and wonder at witnessing a new physical capability emerge",
        ),
        (
            "waiting at an airport arrivals gate after a decade of separation from a sibling",
            "the sibling emerged from the corridor and the two embraced without speaking",
            "a prolonged period of geographic separation ended in physical reunion",
            "a flood of warmth and relief at the restoration of a fundamental family bond",
        ),
        (
            "crossing the finish line of a marathon after a full year of training",
            "the timing clock showed a result that exceeded the original goal",
            "a sustained physical challenge was completed beyond expectations",
            "exhilarating triumph combined with the physical release of completed effort",
        ),
        (
            "entering a dark room on what seemed like a forgotten occasion",
            "the lights came on to reveal a gathering of people who had organized a celebration",
            "people who had seemed inattentive had actually coordinated a collective gesture",
            "a rush of gratitude at the discovery of being valued and remembered by a community",
        ),
        (
            "bringing home a rescue animal that had spent months in a shelter",
            "the animal explored every room and then settled contentedly in a chosen spot",
            "a creature that had known instability chose to relax in the new environment",
            "tender affection and deep contentment at providing safety to a previously displaced being",
        ),
        (
            "finishing a presentation at a professional gathering",
            "a highly respected figure in the field approached with specific and genuine praise",
            "recognition came from someone whose opinion carried substantial weight",
            "a warm glow of validation at being acknowledged by an admired peer",
        ),
        (
            "attending a celebration with close companions late into the evening",
            "the group danced and laughed without any self-consciousness or social performance",
            "the boundaries of individual separateness dissolved in collective enjoyment",
            "carefree euphoria and an experience of genuine belonging among trusted people",
        ),
        (
            "waking before dawn after camping overnight at a mountain summit",
            "the first light revealed an unobstructed panorama of layered ridgelines and open sky",
            "the natural spectacle was available only to those who had made the physical effort",
            "quiet awe and serene fulfillment in the presence of earned natural beauty",
        ),
        (
            "unwrapping a gift from someone who was not known for attentiveness",
            "the package contained an item that referenced a casual comment made months earlier",
            "the giver had registered and remembered an offhand expression of private desire",
            "being deeply touched by evidence that someone had listened with unusual care",
        ),
        (
            "walking through an unfamiliar city in a foreign country",
            "a familiar song from home began playing from a window above the street",
            "an unexpected intersection of the distant and the familiar occurred in a random moment",
            "spontaneous delight at the improbability and warmth of the coincidence",
        ),
        (
            "waiting for the announcement of results from a professional competition",
            "the project that had consumed months of dedicated creative work received the top award",
            "external judges independently confirmed the quality of sustained personal investment",
            "deep fulfillment at seeing private effort publicly recognized by qualified evaluators",
        ),
        (
            "meeting close companions for a meal after an extended period of enforced isolation",
            "the conversation flowed easily and the atmosphere felt restored to its former quality",
            "the social fabric that isolation had strained proved to be intact and resilient",
            "comforting warmth and a renewed sense of connection after prolonged solitude",
        ),
        (
            "observing a scene where strangers coordinated to assist someone in difficulty",
            "multiple people who did not know each other independently offered help without hesitation",
            "collective prosocial behavior emerged spontaneously from unrelated individuals",
            "an uplifting feeling of restored confidence in cooperative human tendencies",
        ),
        (
            "looking through photographs from a recent journey that exceeded all plans",
            "each image triggered a detailed memory of an unexpected positive experience",
            "the record confirmed that the actual experience surpassed prior expectations",
            "a satisfied glow of reliving moments that were better than anything anticipated",
        ),
    ],

    "valence": [
        (
            "receiving a detailed performance review after a productive quarter",
            "the feedback contained effusive praise for one area alongside pointed criticism of another",
            "the overall assessment resisted simple categorization as positive or negative",
            "a complex internal process of weighing genuinely positive elements against genuinely negative ones",
        ),
        (
            "watching a film that built toward a dual resolution",
            "the ending reunited two characters while simultaneously separating two others permanently",
            "the narrative delivered reward and cost in the same concluding sequence",
            "a layered response combining uplift with poignancy that could not be reduced to either alone",
        ),
        (
            "tasting a dish at a restaurant that had received contradictory reviews",
            "the flavors were simultaneously overpowering in one dimension and exquisite in another",
            "the sensory experience resisted a simple positive or negative classification",
            "an oscillating appraisal between genuine pleasure and genuine displeasure",
        ),
        (
            "reading a recommendation letter written by a former supervisor",
            "the language was superficially complimentary but the specific word choices implied reservation",
            "the true evaluative stance was concealed beneath a veneer of conventional praise",
            "uneasy parsing of the genuine sentiment hidden beneath surface-level positive framing",
        ),
        (
            "attending a farewell gathering for a colleague leaving for an opportunity abroad",
            "the speeches alternated between celebrating the achievement and mourning the departure",
            "the occasion was structurally both a celebration and a loss",
            "a blended internal tone that was simultaneously uplifting and heavy",
        ),
        (
            "hearing a comedic remark that drew laughter from most of the group",
            "the humor relied on a characterization that was recognizably unkind to one member present",
            "amusement and social discomfort co-occurred without either overriding the other",
            "an unresolved tension between genuine amusement and awareness of interpersonal cost",
        ),
        (
            "receiving a career advancement that required immediate geographic relocation",
            "the professional gain was substantial but the personal cost of leaving family was equally real",
            "the opportunity and the sacrifice were inseparable components of the same decision",
            "a tense internal weighing of professional opportunity against personal attachment",
        ),
        (
            "observing a street performer execute a technically brilliant but emotionally flat routine",
            "the physical skill was objectively impressive while the artistic content felt hollow",
            "admiration for craft and disappointment in expression coexisted without resolution",
            "a nuanced internal response that separated technical appreciation from emotional engagement",
        ),
        (
            "revisiting a childhood location that had undergone extensive renovation",
            "some changes were genuine improvements while others erased irreplaceable original features",
            "the place was simultaneously better in function and diminished in personal significance",
            "a layered assessment comparing positive practical changes against lost personal associations",
        ),
        (
            "completing a prolonged negotiation where agreement was reached",
            "both parties made concessions that each privately considered painful",
            "the resolution removed the dispute but left residual awareness of what was surrendered",
            "muted satisfaction tempered by the persistent awareness of what the agreement cost",
        ),
        (
            "watching a sunset over a landscape associated with a painful farewell years earlier",
            "the visual beauty was identical to the scene present during the original difficult moment",
            "the sensory experience was objectively pleasant and associatively painful simultaneously",
            "a conflicting internal state as aesthetic appreciation triggered painful memory activation",
        ),
        (
            "receiving a thoughtfully wrapped present from a caring acquaintance",
            "the gift itself was unwanted but the effort and consideration behind it were genuine",
            "authentic gratitude for the gesture coexisted with private disappointment at the object",
            "an internal dissonance between appreciation for the giver and dissatisfaction with the gift",
        ),
        (
            "listening to a memorial address that combined candid assessment with deep affection",
            "the speaker praised and criticized the deceased with equal specificity and evident sincerity",
            "the speech was moving because of its honesty and uncomfortable for the same reason",
            "a multi-layered response where appreciation of truthfulness competed with discomfort at its content",
        ),
        (
            "finishing a novel that was beautifully constructed but ended with devastating loss",
            "the literary craftsmanship and the emotional devastation were products of the same narrative skill",
            "admiration for the writing and distress at its content were functionally inseparable",
            "an internal tension where artistic admiration contended with genuine emotional distress",
        ),
        (
            "ending a relationship that was both deeply valued and consistently harmful",
            "the decision was clearly necessary based on evidence but the attachment remained strong",
            "rational clarity about the correct choice coexisted with genuine mourning for what was lost",
            "a difficult internal state where evaluative certainty and emotional pain operated simultaneously",
        ),
    ],

    # ── SOCIAL / MENTALISTIC (8) ──

    "belief": [
        (
            "told by a colleague that the weekly meeting had moved to a new room",
            "the colleague had been misinformed and the meeting actually remained in the original location",
            "the person proceeded toward the wrong room while the actual meeting began elsewhere",
            "holding a confident expectation that no longer matched the actual state of affairs",
        ),
        (
            "reading a news article that reported a major corporation had declared insolvency",
            "the article was based on an unverified rumor and the corporation subsequently denied it",
            "the person continued to operate under the reported claim days after it was retracted",
            "maintaining an accepted factual claim that had been superseded by a correction",
        ),
        (
            "overhearing a neighbor mention that the corner bakery had closed permanently",
            "the bakery had actually only closed for renovation and was scheduled to reopen",
            "the person began walking to a more distant bakery each morning unnecessarily",
            "acting on secondhand testimony that provided an inaccurate picture of current reality",
        ),
        (
            "taught in a school curriculum that a specific historical event occurred in a particular year",
            "subsequent scholarly research had revised the date but the textbooks were never updated",
            "the learned date remained fixed in memory while the academic record had changed",
            "retaining an educational claim that the field of knowledge had since corrected",
        ),
        (
            "checking a weather application that predicted clear conditions for the entire day",
            "the forecast algorithm had missed an approaching storm system visible on satellite",
            "outdoor plans were made based on a prediction that a different data source contradicted",
            "trusting an automated forecast that diverged from the physical atmospheric conditions",
        ),
        (
            "remembering that a friend had said the dinner reservation was for seven o'clock",
            "the friend had actually said eight and later confirmed this in a message that went unread",
            "the person arrived an hour early and waited alone at the restaurant",
            "relying on a recalled statement that differed from the actual spoken words",
        ),
        (
            "reading a posted sign that stated a museum offered free admission on Thursdays",
            "the policy had changed the previous month and the sign had not been updated",
            "the person arrived on Thursday expecting free entry and was asked to pay the standard fee",
            "accepting institutional signage as current when the information it displayed was outdated",
        ),
        (
            "assured by a mechanic that a vehicle was safe for a long road trip",
            "the mechanic had missed a critical brake component failure during the inspection",
            "the vehicle was driven at highway speed with a safety defect the owner did not know existed",
            "placing trust in a specialist assessment that contained an undetected error",
        ),
        (
            "hearing from multiple acquaintances that the company was planning major layoffs",
            "the rumor originated from a misinterpreted internal memo about restructuring office space",
            "the person updated their resume and began job searching based on the spreading claim",
            "tentatively accepting an unverified social consensus that had no factual foundation",
        ),
        (
            "watching a documentary that presented evidence against a well-established scientific position",
            "the documentary selectively omitted data that supported the original position",
            "the viewer's previously stable understanding was destabilized by partial evidence",
            "experiencing tension between a longstanding conviction and persuasive but incomplete counter-evidence",
        ),
        (
            "finding a handwritten note in a new apartment that said the spare key was under a mat",
            "the previous tenant had retrieved the key when moving out months earlier",
            "the person told visitors about the spare key that no longer existed at that location",
            "trusting a written instruction whose factual content had changed since it was written",
        ),
        (
            "warned by a neighbor that the local tap water was contaminated after pipe construction",
            "the water authority had cleared the water as safe two weeks before the warning was given",
            "bottled water was purchased for months based on a precautionary claim that was already outdated",
            "forming a health-related conviction from a community warning that no longer reflected conditions",
        ),
        (
            "receiving a confirmation email stating an appointment was scheduled for ten in the morning",
            "the administrative system had a time-zone error and the actual slot was at noon",
            "the person arrived two hours early and waited based on the erroneous confirmation",
            "experiencing reinforced certainty from an official communication that contained a systematic error",
        ),
        (
            "recalling that a health article recommended a specific supplement for preventing illness",
            "the article had since been retracted due to flawed methodology in the underlying study",
            "the supplement continued to be purchased based on claims that the scientific community had withdrawn",
            "maintaining a retained health-related conviction from a source that was no longer considered valid",
        ),
        (
            "checking a map application that displayed a shortcut through a park to a train station",
            "the park path had been closed for construction and rerouted since the map was last updated",
            "the route shown on the device did not correspond to the physical paths available on the ground",
            "navigating based on a cartographic representation that no longer matched the actual terrain",
        ),
    ],

    "theory_of_mind": [
        (
            "organizing a surprise celebration for a friend on an upcoming weekend",
            "the event was quietly canceled by the host but no one informed the guest of honor",
            "the friend continued preparing for the occasion that no longer existed",
            "tracking two simultaneous realities where the friend's expectation diverged from the actual situation",
        ),
        (
            "watching a child hide a toy in a red box while a sibling was in the room",
            "the sibling left, and the child moved the toy to a blue box without the sibling seeing",
            "the sibling would return expecting the toy to be where they last saw it placed",
            "representing two distinct knowledge states about the same object's location simultaneously",
        ),
        (
            "present when one colleague made a sarcastic remark to another at a meeting",
            "the second colleague interpreted the sarcasm as a genuine compliment and responded with thanks",
            "the speaker's intended meaning and the listener's received meaning were opposite",
            "recognizing the gap between what was communicated and what was understood by the recipient",
        ),
        (
            "noticing a tourist consulting an outdated printed guide at a bus stop",
            "the museum the tourist was heading to had relocated to a different district last year",
            "the tourist was navigating confidently toward a destination that no longer existed at that address",
            "detecting that another person was acting on information that had become obsolete without their awareness",
        ),
        (
            "observing a physician reviewing test results before entering a patient room",
            "the physician paused and then entered with a composed expression, discussing only minor findings",
            "certain information was deliberately withheld from the patient's awareness",
            "understanding that selective omission served a protective function in the other person's communication",
        ),
        (
            "present when a new employee sent a candid message intended for one person",
            "the message was accidentally sent to the entire department distribution list",
            "the sender continued working unaware that dozens of people had read the private content",
            "detecting that the sender's model of who possessed the information was narrower than reality",
        ),
        (
            "listening to a colleague recount a weekend event with increasingly dramatic details",
            "the embellishments grew with each retelling while the factual core remained the same",
            "the speaker's communicative purpose was entertainment rather than accurate reporting",
            "distinguishing between the performative goal of the narration and its literal informational content",
        ),
        (
            "observing a child standing near an open and empty container that had held treats",
            "the child's face displayed a mixture of avoidance and flushing when asked about the contents",
            "the child was simultaneously concealing an action and unable to fully suppress the evidence of it",
            "linking the observable behavioral cues to the child's internal awareness of a transgression",
        ),
        (
            "participating in a job interview where the final question had no factually correct answer",
            "the interviewer watched the reasoning process rather than evaluating the conclusion",
            "the stated purpose of the question and its actual evaluative function were different",
            "recognizing that the question was designed to reveal process rather than to obtain specific content",
        ),
        (
            "learning that two roommates each assumed the other had paid the monthly rent",
            "neither had contacted the landlord, and both were surprised when a late notice arrived",
            "each roommate held a false assumption about what the other had done",
            "simultaneously representing two incompatible assumptions held by two different people",
        ),
        (
            "watching a vendor at a market quote a price after looking at a customer's clothing",
            "the quoted price was substantially higher than what previous customers had been charged",
            "the pricing decision was calibrated to the vendor's assessment of the customer's resources",
            "recognizing that one person modeled another's economic status and adjusted strategy based on that model",
        ),
        (
            "sitting in a classroom where a teacher simplified an explanation after scanning the room",
            "the teacher's language shifted to more concrete terms when several students showed confusion",
            "the instructional approach was dynamically adjusted based on reading the audience's comprehension",
            "recognizing that the speaker recalibrated communication based on an ongoing assessment of the listener's state",
        ),
        (
            "attending a stage performance that relied heavily on misdirection",
            "the performer's gestures consistently directed the audience's attention away from the operative hand",
            "the audience's focus was being systematically engineered to land on uninformative elements",
            "understanding that the performer deliberately constructed a false attentional target for the observers",
        ),
        (
            "visiting an elderly relative who insisted on carrying heavy items without assistance",
            "the relative's posture and pace showed clear physical strain despite verbal assurances of capability",
            "the verbal self-report and the physical evidence were in direct contradiction",
            "perceiving that a deliberately maintained self-presentation of capability concealed an actual need for support",
        ),
        (
            "observing a negotiation where one party opened with an aggressive demand and rigid posture",
            "the same party later conceded the exact points they had most aggressively defended initially",
            "the opening position bore no relation to the party's actual priorities or limits",
            "distinguishing a strategically performed stance from the person's genuine negotiating position",
        ),
    ],

    "mentalizing": [
        (
            "noticing that a usually reserved colleague had become unusually talkative over several days",
            "the behavioral shift had no obvious external cause visible to the observer",
            "the change in social behavior suggested an internal shift that had not been communicated",
            "actively reasoning about what internal change could explain the sustained behavioral departure",
        ),
        (
            "sitting across from a supervisor during a meeting that had gone silent for an extended pause",
            "the supervisor's expression was neutral but the silence persisted beyond social convention",
            "the absence of verbal content made the underlying mental state ambiguous and unreadable",
            "effortfully interpreting what thoughts or evaluations might underlie the sustained social ambiguity",
        ),
        (
            "observing a stranger on public transit who smiled suddenly while looking at a personal device",
            "the smile appeared genuine and was directed at content the observer could not see",
            "the visible emotional response was disconnected from any shared environmental stimulus",
            "spontaneously attributing specific internal states to an unknown person based on a single expression",
        ),
        (
            "eating at a restaurant with a companion who ordered an atypical meal without explanation",
            "the companion had consistently chosen the same item for years and the departure was abrupt",
            "a long-established behavioral pattern was broken in a context where it had been highly stable",
            "reasoning about what internal preference shift or new information drove the unexpected deviation",
        ),
        (
            "watching a child pause at the entrance of a new classroom on the first morning",
            "the child's body oriented toward the door but the feet did not move forward",
            "the visible conflict between approach and avoidance was playing out in posture and movement",
            "inferring a complex internal state from the observable tension between contradictory motor impulses",
        ),
        (
            "reading a carefully worded email from a colleague about a collaborative project",
            "the language was technically neutral but the sentence structure was unusually stiff and formal",
            "the communicative style had shifted in a way that the content alone did not explain",
            "detecting a concealed emotional state embedded within deliberately controlled language",
        ),
        (
            "asking a partner a simple question about the events of the day",
            "the partner paused noticeably before delivering a straightforward factual answer",
            "the hesitation was disproportionate to the simplicity of the question asked",
            "reading psychological significance into a brief temporal gap between question and response",
        ),
        (
            "leading a group discussion while scanning the room for engagement cues",
            "one participant in the back consistently avoided direct eye contact during exchanges",
            "the gaze pattern was systematically different from the rest of the group",
            "interpreting sustained gaze avoidance as a signal of internal discomfort or active disengagement",
        ),
        (
            "expecting a reliably punctual friend who arrived significantly late without offering a reason",
            "the friend behaved normally upon arrival but did not address the departure from routine",
            "the unexplained deviation created a gap between observed behavior and known behavioral baseline",
            "generating hypotheses about unspoken reasons that would explain atypical conduct from a predictable person",
        ),
        (
            "noticing that a neighbor had become markedly more polite in the days following a heated dispute",
            "the politeness had no explicit reference to the earlier conflict and appeared unsolicited",
            "the behavioral change was temporally linked to the dispute but not verbally connected to it",
            "inferring reconciliatory intent or unresolved guilt from a strategic shift in interpersonal manner",
        ),
        (
            "participating in an interview where the interviewer's questions became shorter and less engaged",
            "the shift occurred midway through the conversation without any explicit feedback being given",
            "the reduction in engagement intensity suggested a conclusion had been reached before the end",
            "detecting a foregone internal decision from diminishing behavioral investment in the interaction",
        ),
        (
            "listening to a friend describe a new romantic partner in an unusually selective manner",
            "specific topics were elaborated while others were consistently redirected or abbreviated",
            "the pattern of disclosure and avoidance formed a systematic shape across the conversation",
            "interpreting deliberate selectivity in self-disclosure as a signal about underlying private assessments",
        ),
        (
            "being served by a restaurant worker whose smile did not extend beyond the mouth",
            "the vocal tone was professionally warm but the facial expression lacked congruent activation",
            "the performed social expression and the underlying state appeared to be mismatched",
            "perceiving the gap between an outward social display and the likely internal state it was masking",
        ),
        (
            "overhearing a conversation between two people conducted entirely in short clipped phrases",
            "the topic was mundane but the prosodic quality suggested tension between the speakers",
            "the communication style carried relational information that the literal content did not",
            "inferring recent interpersonal tension from the paralinguistic quality of an overheard exchange",
        ),
        (
            "watching a public figure respond to a question during a live broadcast",
            "a brief facial configuration appeared and disappeared before the composed verbal response began",
            "the involuntary micro-expression and the subsequent controlled response were in opposition",
            "detecting brief involuntary emotional leakage that contradicted the deliberate verbal message",
        ),
    ],

    "intention": [
        (
            "entering a shared kitchen where a roommate had placed cleaning supplies near a stained surface",
            "the supplies were arranged in sequence of use but the cleaning had not yet started",
            "the physical arrangement of objects indicated a planned sequence of actions not yet executed",
            "inferring an upcoming deliberate action from the preparatory spatial arrangement of tools",
        ),
        (
            "observing a shopper in a store holding two competing products side by side at eye level",
            "the shopper rotated each item to read the specifications and placed one back repeatedly",
            "the systematic comparison behavior indicated a structured decision process in progress",
            "recognizing that the careful evaluative behavior reflected a deliberate cost-benefit calculation",
        ),
        (
            "arriving at the office to find a colleague printing and organizing materials before any meeting was announced",
            "the colleague had also reserved a conference room and arranged seating for several people",
            "the preparatory activities were coordinated toward an event that had not been communicated",
            "inferring a planned presentation from the convergence of multiple preparatory behaviors",
        ),
        (
            "standing in a hardware store while a neighbor asked detailed questions about fence regulations",
            "the neighbor also purchased measuring tape and boundary markers during the same visit",
            "the combination of regulatory inquiry and material purchase pointed toward a specific project",
            "inferring a future construction project from the specific pattern of information-seeking and purchasing",
        ),
        (
            "noticing that a travel companion had packed rugged outdoor boots for a trip described as relaxing",
            "the companion's bag also contained trail maps that were not part of the stated itinerary",
            "the physical preparation contradicted the verbally described plan for the journey",
            "inferring undisclosed adventure plans from the mismatch between stated and actual preparation",
        ),
        (
            "watching a driver begin signaling and reducing speed well before any visible turn or intersection",
            "the signaling continued consistently despite no immediate reason being visible to followers",
            "the advance preparation communicated a firm decision about an upcoming directional change",
            "recognizing that the sustained early signaling reflected a committed navigational decision",
        ),
        (
            "returning home earlier than expected to find a family member arranging decorations in another room",
            "the decorations were concealed from the common areas and a date was marked on a hidden calendar",
            "the secrecy and temporal specificity indicated an organized plan for a future event",
            "detecting a concealed goal-directed plan being executed in preparation for a specific occasion",
        ),
        (
            "receiving an email from a manager requesting brief individual meetings with every team member",
            "the meetings were scheduled back-to-back on the same day with identical fifteen-minute slots",
            "the systematic scheduling pattern suggested a coordinated information-gathering or delivery exercise",
            "inferring a larger organizational decision from the systematic structure of the meeting schedule",
        ),
        (
            "watching a chess opponent study the board in silence for an unusually extended period",
            "the opponent's eyes tracked repeatedly between several specific positions on the board",
            "the extended deliberation and systematic visual scanning indicated deep strategic calculation",
            "recognizing that the prolonged analysis reflected a complex multi-move strategic plan being formulated",
        ),
        (
            "passing through a hallway where a student was quietly rehearsing spoken content from memory",
            "the student was gesturing to an invisible audience and repeating specific phrases",
            "the private rehearsal combined vocal and gestural practice directed at a future public situation",
            "detecting a planned performance from the pattern of covert preparation observed by chance",
        ),
        (
            "observing a negotiator arrange documents in a specific order before a discussion began",
            "the arrangement placed certain figures on top and positioned reference materials for quick access",
            "the physical setup of materials suggested a pre-planned sequence of argumentative points",
            "inferring a rehearsed strategic approach from the deliberate organization of persuasive materials",
        ),
        (
            "noticing that a sibling had been searching for airline prices to a specific destination",
            "the browser history showed repeated comparisons across dates without any mention of travel plans",
            "the research pattern indicated sustained interest in a trip that had not been discussed",
            "inferring undisclosed travel plans from a pattern of covert but systematic research behavior",
        ),
        (
            "watching a gardener place small colored flags at specific positions across a yard",
            "the flags formed a geometric pattern with consistent spacing and orientation",
            "the spatial markers indicated a detailed plan for staged planting or landscaping work",
            "recognizing that the flagging pattern reflected a pre-designed spatial execution plan",
        ),
        (
            "noticing that a friend had declined multiple social invitations for an upcoming weekend",
            "the friend offered vague excuses but had not mentioned any alternative commitments",
            "the pattern of deliberate availability creation suggested anticipation of something unannounced",
            "inferring that systematic schedule clearing signaled an anticipated but undisclosed upcoming event",
        ),
        (
            "observing a cook arrange every ingredient and tool on the counter before beginning any preparation",
            "each item was positioned in the order it would be needed during the cooking sequence",
            "the organized layout reflected a complete mental representation of the procedure to follow",
            "recognizing that the systematic pre-arrangement indicated a fully planned sequential process",
        ),
    ],

    "empathy": [
        (
            "sitting with a friend who was describing a recent family loss",
            "the friend's voice broke mid-sentence and tears began falling despite efforts to maintain composure",
            "the friend's distress became visible through the failure of emotional suppression",
            "a deep shared resonance where the observer's own chest tightened in response to the other's pain",
        ),
        (
            "watching a young child trip and fall hard onto a concrete playground surface",
            "the child's face contorted in pain and a cry of shock followed the impact",
            "the physical injury and the child's distress response were both clearly visible",
            "an automatic somatic echo where the observer felt a pang in their own body at the witnessed impact",
        ),
        (
            "overhearing a stranger's phone conversation in a quiet public space",
            "the stranger described weeks of isolation and the absence of any social contact",
            "the loneliness in the stranger's voice was raw and unperformed",
            "an involuntary drop in the observer's own mood that mirrored the stranger's described isolation",
        ),
        (
            "present during a performance review where a colleague received unexpectedly harsh feedback",
            "the colleague suppressed visible frustration but tension was evident in clenched hands and rigid posture",
            "the colleague's effort to contain the emotional response was observable to someone paying attention",
            "a tension arising in the observer's own body that mirrored the colleague's suppressed reaction",
        ),
        (
            "sitting beside a neighbor who had just received devastating medical news in a phone call",
            "the neighbor ended the call and sat in silence staring at the floor",
            "the magnitude of the news was visible in the neighbor's sudden physical stillness",
            "a deep attunement to the severity of the situation calibrated by the neighbor's visible shock",
        ),
        (
            "observing an elderly person struggling to carry multiple heavy bags across a parking lot",
            "the person's pace was slow and the bags were visibly straining their grip and balance",
            "the physical difficulty and vulnerability were evident in every labored step",
            "a pull to intervene that arose from witnessing vulnerability and effort in someone less able",
        ),
        (
            "watching a documentary that followed displaced families through several months of relocation",
            "the footage showed children arriving at unfamiliar shelters carrying single bags of possessions",
            "the scale of loss was made concrete through the specific details of individual family experiences",
            "a contagion of distress triggered by vivid portrayal of real displacement and personal upheaval",
        ),
        (
            "present when a companion misspoke during a formal group introduction",
            "the companion's face flushed immediately and the room went briefly quiet",
            "the social misstep was witnessed by the entire group simultaneously",
            "a shared flush of self-conscious discomfort that arose from witnessing the companion's public exposure",
        ),
        (
            "holding the hand of a patient who was waiting to be taken into a surgical procedure",
            "the patient's grip tightened and their breathing became shallow and rapid",
            "the patient's pre-procedural anxiety was transmitted through physical contact and visible tension",
            "absorbing the patient's distress through the combination of touch and emotional attunement",
        ),
        (
            "reading a detailed report about a large-scale humanitarian crisis affecting millions",
            "the account included specific descriptions of individual families separated by the events",
            "the statistical magnitude and the individual human detail both registered simultaneously",
            "an overwhelming wave of concern that extended from the specific stories to the collective scale of suffering",
        ),
        (
            "watching a teammate twist an ankle during a competitive game",
            "the teammate crumpled to the ground and gripped the injured joint",
            "the acute physical pain was expressed through the teammate's immediate physical response",
            "a reflexive wince in the observer's own body that mirrored the teammate's sudden injury",
        ),
        (
            "walking alongside a companion who had just learned of a significant personal loss",
            "the observer's own walking pace unconsciously slowed to match the companion's heavy stride",
            "the behavioral synchrony occurred without deliberate decision or conscious awareness",
            "an automatic alignment of physical behavior driven by attunement to the companion's emotional state",
        ),
        (
            "present when a student received the result of an exam they had been dreading for weeks",
            "the student read the passing mark and exhaled visibly as tension drained from their posture",
            "the release from sustained anxiety was visible as a whole-body relaxation response",
            "a co-experienced wave of relief that mirrored the student's release from prolonged stress",
        ),
        (
            "watching a parent and child reunite at an airport after years of separation",
            "the two ran toward each other and held on without speaking for an extended moment",
            "the intensity of the reunion was visible in the duration and force of the physical contact",
            "an overwhelming co-experience of witnessed reunion that produced tears in the observer",
        ),
        (
            "noticing a new colleague eating alone in a corner of a crowded workplace cafeteria",
            "the colleague's posture was closed and their gaze avoided the surrounding groups",
            "the social isolation was evident in the physical contrast between the individual and the groups",
            "an inferential recognition of the colleague's isolation that produced a motivating pull to include them",
        ),
    ],

    "self_referential": [
        (
            "looking at old photographs from a decade earlier and comparing them to the present",
            "the contrast between past and present demeanor, interests, and social circle was striking",
            "the person depicted in the photographs felt both continuous with and distant from the current self",
            "a sustained introspective process comparing past and present versions of personal identity",
        ),
        (
            "reviewing career decisions made over the past several years during a quiet evening",
            "the trajectory of professional choices was laid out against a set of privately held principles",
            "some choices aligned with core values while others appeared driven by external pressure",
            "an evaluative self-reflection probing the alignment between inner commitments and actual life direction",
        ),
        (
            "noticing specific behavioral tendencies and wondering which parent they originally came from",
            "certain gestures and reactions were traceable to one parent while others resembled the other",
            "the observed personal traits appeared to be composites of inherited and developed characteristics",
            "an introspective attribution process connecting current personality features to their developmental origins",
        ),
        (
            "imagining a scenario where a stranger on the street needed help from a bystander",
            "the imagined scenario tested whether the self-concept included the kind of person who would act",
            "the answer was uncertain and the uncertainty itself revealed something about self-knowledge limits",
            "testing a private self-concept against a hypothetical situation requiring action under social pressure",
        ),
        (
            "lying awake at night tracing how different past decisions could have led to different outcomes",
            "the branching paths of alternate choices formed a map of lives that could have been",
            "the current life was revealed as one of many possible trajectories rooted in contingent decisions",
            "counterfactual reasoning directed inward about how alternate choices would have reshaped personal identity",
        ),
        (
            "remembering a past response to interpersonal conflict and comparing it to a recent one",
            "the earlier response was reactive and escalatory while the recent one was measured and contained",
            "the contrast between the two responses suggested a change in personal regulation capacity",
            "self-monitoring that compared past and present behavioral responses to gauge personal development",
        ),
        (
            "feeling discomfort during a professional disagreement and stopping to examine the source",
            "the discomfort could have originated from a principled objection or from wounded personal pride",
            "the two possible sources had different implications for how seriously to take the reaction",
            "an introspective effort to distinguish whether a strong reaction stemmed from principle or from ego",
        ),
        (
            "thinking about what the central themes of a personal biography would include",
            "the exercise required selecting which experiences and values defined the overall narrative",
            "the selection process revealed which elements of life were considered most identity-constituting",
            "a deeply inward narrative exercise about which aspects of lived experience define the self",
        ),
        (
            "considering whether a longstanding recreational activity still provided genuine satisfaction",
            "the activity had become routine and automatic rather than actively chosen each time",
            "the distinction between genuine preference and behavioral inertia was no longer clear",
            "probing whether continued engagement reflected authentic desire or simple habitual continuation",
        ),
        (
            "receiving several criticisms in one week and noticing that only certain ones lingered",
            "the criticisms that persisted in awareness targeted specific aspects of performance and character",
            "the pattern of which comments stung and which were dismissed formed a map of vulnerable areas",
            "an introspective analysis identifying which aspects of self-regard were most susceptible to external challenge",
        ),
        (
            "recalling a period when a close companion experienced a crisis and needed support",
            "the quality of support provided during that period was now uncertain in retrospect",
            "whether the response was adequate or fell short was an open question with personal implications",
            "a retrospective self-evaluation assessing personal adequacy in a past role of interpersonal responsibility",
        ),
        (
            "noticing that political positions held since early adulthood had shifted substantially",
            "the shifts tracked changes in personal experience and social context rather than any single argument",
            "deeply held convictions had evolved through gradual accumulation rather than dramatic conversion",
            "a personal audit tracing how and why foundational positions changed over time",
        ),
        (
            "experiencing recurring avoidance of public speaking situations across different life contexts",
            "the avoidance was consistent enough to suggest a stable trait rather than situational discomfort",
            "a single behavioral pattern appeared to be a surface expression of a deeper personality characteristic",
            "interpreting a persistent behavioral tendency as a window into a more fundamental aspect of personal makeup",
        ),
        (
            "catching a reflection in a window and comparing the professional persona projected outward",
            "the image triggered awareness of the gap between the presented version and the private experience",
            "the two versions of the self operated in different social contexts with different degrees of authenticity",
            "evaluating the distance between a public professional persona and an internal private sense of self",
        ),
        (
            "reflecting on how an early childhood experience shaped the way close relationships were approached",
            "the connection between the early event and the current relational pattern was not immediately obvious",
            "the causal link became visible only through sustained inward examination of personal history",
            "constructing a causal self-narrative that connected early formative experience to current attachment patterns",
        ),
    ],

    "judgment": [
        (
            "learning that a landlord evicted a tenant who had fallen behind on rent during a medical emergency",
            "the eviction was legally permissible but occurred during the tenant's period of greatest vulnerability",
            "the exercise of a legal right caused severe hardship to someone in a medically compromised state",
            "a normative evaluation weighing the landlord's contractual rights against the tenant's situational vulnerability",
        ),
        (
            "hearing that a friend had completely severed contact with a family member after years of conflict",
            "the family member had repeatedly violated personal boundaries despite multiple requests to stop",
            "the severance eliminated both the source of harm and any possibility of future repair",
            "evaluative reasoning about whether the pattern of violation warranted permanent relational termination",
        ),
        (
            "reading about an employer that monitored all employee digital communications on company devices",
            "the monitoring was disclosed in the employment contract but covered personal messages sent during breaks",
            "organizational security measures extended into territory that employees reasonably considered private",
            "a normative assessment balancing institutional security interests against individual privacy expectations",
        ),
        (
            "watching a coach bench the highest-scoring player minutes before a championship-deciding moment",
            "the benched player had violated a team conduct rule that other players had followed",
            "the team lost the championship but the conduct standard was applied consistently",
            "evaluating whether maintaining a principle was the correct decision when it caused a consequential loss",
        ),
        (
            "learning that a journalist published classified documents that revealed governmental misconduct",
            "the publication exposed genuine wrongdoing but also compromised the safety of named individuals",
            "the public benefit of disclosure and the private cost to identified persons were both concrete",
            "normative reasoning about whether the value of exposed truth outweighed the harm from compromised identities",
        ),
        (
            "hearing that a school expelled a student under a zero-tolerance policy for a minor first offense",
            "the offense was technically within the policy scope but clearly below the severity the policy targeted",
            "the punishment was procedurally correct but disproportionate to the specific infraction",
            "an evaluative assessment of whether rule consistency justified a severity mismatch in this particular case",
        ),
        (
            "mediating between two neighbors where one complained about noise during permitted hours",
            "the noise source was a musical instrument practiced within the posted quiet-hour exceptions",
            "both the complaint and the activity were legitimate under existing community guidelines",
            "forming a considered assessment by weighing competing but individually valid claims in a social dispute",
        ),
        (
            "reviewing financial records of a charitable organization against its stated objectives",
            "a substantial percentage of donations was allocated to administrative overhead rather than direct services",
            "the allocation pattern was legal but diverged significantly from the organization's public messaging",
            "evaluative scrutiny of whether institutional resource distribution matched the purpose donors believed they were supporting",
        ),
        (
            "comparing two student submissions that received identical marks despite visibly different quality",
            "one submission demonstrated original analysis while the other met minimum requirements mechanically",
            "the grading system treated unequal work as equivalent by applying the same threshold standard",
            "a normative evaluation asking whether equality of assessment was appropriate given inequality of effort and quality",
        ),
        (
            "reviewing a referee's call that overturned a score in the final minute of a closely contested match",
            "the replay showed the call was technically defensible but relied on an interpretation other officials might not have made",
            "the outcome of the entire competition pivoted on a single ambiguous enforcement decision",
            "evaluating whether a consequential official decision fell within or exceeded the acceptable range of interpretive discretion",
        ),
        (
            "reading about a physician who administered a treatment over a patient's explicit objection",
            "the treatment prevented serious harm but was delivered against the patient's clearly stated refusal",
            "the medical outcome was beneficial but the process violated the patient's expressed autonomous choice",
            "a normative assessment of whether preventing harm justified overriding a competent individual's explicit refusal of care",
        ),
        (
            "following a corporate decision to relocate a factory, eliminating local employment to reduce costs",
            "the relocation increased shareholder returns while devastating the economic base of a dependent community",
            "the financial and social consequences of the same decision fell on entirely different populations",
            "evaluative reasoning about whether fiduciary obligations to investors should outweigh economic obligations to affected workers",
        ),
        (
            "considering whether a student's documented personal crisis constituted a sufficient reason for a late submission",
            "the documentation was credible but the deadline extension would require recalibrating the assessment for other students",
            "accommodating one student's circumstances would create a precedent with consequences for fairness to others",
            "an evaluative assessment probing whether the credibility of the excuse justified differential treatment",
        ),
        (
            "evaluating whether a government's delayed response to a natural disaster reflected negligence or resource constraints",
            "independent reports indicated that both factors contributed but their relative weights were disputed",
            "the assessment required separating institutional capability from institutional willingness",
            "a normative evaluation of whether the institutional response met reasonable standards given available information about constraints",
        ),
        (
            "assessing whether a former friend's apology after a serious breach was adequate for restoring the relationship",
            "the apology addressed the surface behavior but did not acknowledge the underlying pattern it represented",
            "the verbal acknowledgment was present but the structural understanding of the harm was absent",
            "evaluative probing of whether a partial acknowledgment warranted restored trust or indicated insufficient understanding",
        ),
    ],

    "moral": [
        (
            "discovering that a colleague had been taking shared office supplies home regularly",
            "the items were individually minor but the cumulative amount over months was substantial",
            "shared resources were being systematically diverted for private use without acknowledgment",
            "an ethical evaluation of whether the pattern of taking communal property constituted a meaningful transgression",
        ),
        (
            "learning that a friend had fabricated an excuse to protect someone from receiving painful information",
            "the fabrication successfully prevented immediate distress but was discovered later",
            "the protective outcome was achieved through means that violated the norm of honest communication",
            "ethical reasoning about whether a benevolent motive could justify the use of deceptive means",
        ),
        (
            "witnessing a crowd walk past a person who had collapsed on a public sidewalk",
            "multiple individuals glanced at the fallen person and continued walking without pausing",
            "no one in a position to provide basic assistance chose to act despite the visible emergency",
            "an ethical evaluation of the collective failure to render aid in a situation where intervention was feasible",
        ),
        (
            "reading an investigative report about a company disposing of industrial waste in a waterway",
            "the disposal saved the company substantial processing costs while contaminating a shared natural resource",
            "private financial savings were obtained by externalizing toxic costs onto the surrounding community",
            "ethical condemnation of a profit structure that transferred harm to people who did not benefit from the savings",
        ),
        (
            "hearing that a parent falsified a residential address to enroll their child in a higher-performing school district",
            "the deception was successful and the child received substantially better educational resources",
            "the improved outcome for one child was obtained through a method that undermined the allocation system for all children",
            "weighing the ethical tension between a parent's protective motivation and the fraudulent means employed",
        ),
        (
            "observing a merchant apply different prices to different customers based on their outward appearance",
            "customers who appeared affluent were charged more for identical items than those who appeared less so",
            "the pricing structure treated people unequally based on a superficial assessment of their economic status",
            "an ethical evaluation of whether differential treatment based on perceived wealth constituted an unjust practice",
        ),
        (
            "reading about an employee who exposed institutional corruption while violating a signed confidentiality agreement",
            "the exposure led to criminal prosecution of the corrupt officials but also to legal action against the employee",
            "fulfilling one obligation required violating another and both carried serious consequences",
            "ethical reasoning about whether the obligation to expose wrongdoing outweighed a contractually binding promise of secrecy",
        ),
        (
            "receiving excess change from a cashier who was clearly exhausted and overworked",
            "the amount was noticeable and the error was unambiguous upon even brief inspection",
            "retaining the overpayment would benefit the recipient while the cost would fall on the worker",
            "an ethical evaluation of whether passively benefiting from another person's error constituted a form of dishonesty",
        ),
        (
            "learning that one student copied another's assignment and both received credit for the work",
            "the student who did the original work was aware of the copying and did not object",
            "the tacit permission created shared complicity in the violation of academic standards",
            "an ethical assessment of distributed responsibility when one party actively violated rules and the other knowingly allowed it",
        ),
        (
            "reading about a terminally ill patient who requested that all treatment be discontinued",
            "the medical team confirmed that the patient was fully competent and the prognosis was certain",
            "honoring the request would hasten death while continued treatment would extend a state the patient described as unbearable",
            "deep ethical deliberation balancing respect for individual autonomy against the imperative to sustain biological life",
        ),
        (
            "debating a policy proposal to fund public services through substantially increased taxation on high earners",
            "the policy would measurably reduce inequality while substantially reducing the disposable income of a minority",
            "the redistributive benefit and the individual burden fell on non-overlapping populations",
            "ethical reasoning about whether compulsory redistribution served collective fairness or violated individual entitlement",
        ),
        (
            "reviewing arguments about whether using animals in medical research could produce net positive outcomes",
            "the research had produced treatments that saved human lives but required procedures that caused animal suffering",
            "the benefit to one species was obtained through the instrumental use and suffering of another sentient species",
            "an ethical evaluation weighing measurable human benefit against the imposed suffering of beings capable of distress",
        ),
        (
            "learning that an aid worker distributed emergency supplies to their own community before others in greater need",
            "the diversion delayed delivery to the most critically affected populations by several days",
            "personal loyalty to a known community took precedence over the impartial distribution mandate of the role",
            "an ethical evaluation of whether personal attachment justified departing from the professional duty of impartial allocation",
        ),
        (
            "considering whether it was acceptable to break a commitment to attend a friend's important event for a work emergency",
            "the work emergency was genuine and the friend's event was a milestone that could not be rescheduled",
            "honoring one obligation required definitively failing the other with no available compromise",
            "ethical reasoning about the conditions under which a competing obligation could excuse a broken personal commitment",
        ),
        (
            "reading about a social media campaign that publicly identified a person who had committed a minor civic offense",
            "the public exposure resulted in the person losing employment and receiving threats from strangers",
            "the accountability mechanism produced consequences that far exceeded the severity of the original act",
            "an ethical evaluation of whether public exposure for accountability purposes constituted proportionate or disproportionate response",
        ),
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────

def capitalize_sentences(text: str) -> str:
    """Capitalize the first letter after each sentence-ending punctuation or colon."""
    result = []
    capitalize_next = True
    for char in text:
        if capitalize_next and char.isalpha():
            result.append(char.upper())
            capitalize_next = False
        else:
            result.append(char)
        if char in ".!?:":
            capitalize_next = True
    return "".join(result)


def generate_stimuli(seed: int = 20260611) -> list[dict]:
    """Generate V2 template-matched stimuli for all 14 conditions."""
    rng = random.Random(seed)
    stimuli = []
    item_id = 0

    conditions = sorted(CONDITION_SCENARIOS.keys())

    for condition in conditions:
        scenarios = CONDITION_SCENARIOS[condition]
        assert len(scenarios) == 15, (
            f"{condition} has {len(scenarios)} scenarios (need exactly 15)")

        person_pool = list(PERSONS)
        rng.shuffle(person_pool)

        for tmpl_name, tmpl_str in TEMPLATES.items():
            for i, (setup, event, consequence, experience) in enumerate(scenarios):
                person = person_pool[i % len(person_pool)]

                text = tmpl_str.format(
                    person=person,
                    setup=setup,
                    event=event,
                    consequence=consequence,
                    experience=experience,
                )

                # Capitalize the first letter of every sentence
                text = capitalize_sentences(text)

                stimuli.append({
                    "text": text,
                    "condition": condition,
                    "template": tmpl_name,
                    "item_id": item_id,
                })
                item_id += 1

    return stimuli


def validate_stimuli(stimuli: list[dict]) -> dict:
    """Check quality constraints: length balance, no label leakage."""

    # Condition label words to ban (matched at word boundaries)
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
    tmpl_lengths = defaultdict(list)
    label_leaks = []
    for s in stimuli:
        words = s["text"].split()
        n_words = len(words)
        cond_lengths[s["condition"]].append(n_words)
        tmpl_lengths[s["template"]].append(n_words)

        text_lower = s["text"].lower()
        for banned in LABEL_WORDS.get(s["condition"], set()):
            pattern = r'\b' + re.escape(banned) + r'\b'
            if re.search(pattern, text_lower):
                label_leaks.append((s["item_id"], s["condition"], banned))

    # Per-condition stats
    cond_means = {c: statistics.mean(lens) for c, lens in cond_lengths.items()}
    cond_stds = {c: statistics.stdev(lens) if len(lens) > 1 else 0.0
                 for c, lens in cond_lengths.items()}
    grand_mean = statistics.mean(cond_means.values())

    # Per-template stats
    tmpl_means = {t: statistics.mean(lens) for t, lens in tmpl_lengths.items()}
    tmpl_stds = {t: statistics.stdev(lens) if len(lens) > 1 else 0.0
                 for t, lens in tmpl_lengths.items()}

    # Length violations (> 20% deviation from grand mean)
    length_violations = {}
    for c, m in cond_means.items():
        dev = abs(m - grand_mean) / grand_mean
        if dev > 0.20:
            length_violations[c] = {"mean": round(m, 1), "deviation": f"{dev:.1%}"}

    return {
        "n_conditions": len(cond_lengths),
        "n_stimuli_per_condition": {c: len(v) for c, v in sorted(cond_lengths.items())},
        "total_stimuli": len(stimuli),
        "grand_mean_words": round(grand_mean, 1),
        "per_condition_mean_words": {c: round(m, 1) for c, m in sorted(cond_means.items())},
        "per_condition_std_words": {c: round(s, 1) for c, s in sorted(cond_stds.items())},
        "per_template_mean_words": {t: round(m, 1) for t, m in sorted(tmpl_means.items())},
        "per_template_std_words": {t: round(s, 1) for t, s in sorted(tmpl_stds.items())},
        "length_violations": length_violations,
        "label_leaks": label_leaks,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate V2 template-matched stimuli for 14-condition RSA")
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--output", type=str, default=None,
                    help="Output JSONL path (default: rsa_stimuli_template_matched_v2.jsonl)")
    ap.add_argument("--validate-only", action="store_true",
                    help="Print validation stats without saving")
    args = ap.parse_args()

    stimuli = generate_stimuli(seed=args.seed)

    # ── Validation ──
    report = validate_stimuli(stimuli)
    print(f"Generated {report['total_stimuli']} stimuli across "
          f"{report['n_conditions']} conditions")
    print(f"Grand mean word count: {report['grand_mean_words']}")

    print(f"\nPer-condition counts, mean and std word lengths:")
    for c in sorted(report["n_stimuli_per_condition"].keys()):
        n = report["n_stimuli_per_condition"][c]
        m = report["per_condition_mean_words"][c]
        s = report["per_condition_std_words"][c]
        print(f"  {c:20s}  n={n:3d}  mean_words={m:5.1f}  std={s:4.1f}")

    print(f"\nPer-template mean and std word lengths:")
    for t in sorted(report["per_template_mean_words"].keys()):
        m = report["per_template_mean_words"][t]
        s = report["per_template_std_words"][t]
        print(f"  {t:15s}  mean_words={m:5.1f}  std={s:4.1f}")

    if report["length_violations"]:
        print(f"\nWARNING: {len(report['length_violations'])} conditions "
              f"exceed +/-20% length tolerance:")
        for c, info in report["length_violations"].items():
            print(f"  {c}: mean={info['mean']} ({info['deviation']})")

    if report["label_leaks"]:
        print(f"\nWARNING: {len(report['label_leaks'])} label word leaks found:")
        for item_id, cond, word in report["label_leaks"][:30]:
            print(f"  item {item_id} ({cond}): contains '{word}'")
        if len(report["label_leaks"]) > 30:
            print(f"  ... and {len(report['label_leaks']) - 30} more")
    else:
        print(f"\nNo label word leaks detected.")

    if args.validate_only:
        return

    # ── Save ──
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (Path(__file__).resolve().parents[1]
                    / "data" / "cognitive_stimuli" / "rsa"
                    / "rsa_stimuli_template_matched_v2.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for s in stimuli:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved to {out_path}")

    # ── Also save a manifest ──
    manifest = {
        "version": "v2",
        "seed": args.seed,
        "n_templates": len(TEMPLATES),
        "templates": list(TEMPLATES.keys()),
        "target_per_condition": 60,
        "design": "4 multi-sentence templates x 15 scenarios x 14 conditions = 840 stimuli",
        "word_target": "35-50 words per stimulus",
        "conditions": {},
    }
    for c in sorted(report["n_stimuli_per_condition"].keys()):
        manifest["conditions"][c] = {
            "n_total": report["n_stimuli_per_condition"][c],
            "source": "template_matched_v2",
            "mean_word_length": report["per_condition_mean_words"][c],
            "std_word_length": report["per_condition_std_words"][c],
        }
    manifest_path = out_path.parent / "rsa_template_matched_v2_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
