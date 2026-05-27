#!/usr/bin/env python3
"""
Experiment B v2: Agent-state factorization probe — VERB-CONFOUND REMOVED.

Upgrades over v1 (agent_state_probe.py):
  v1 used templates like "Alice believes that X" — a linear probe could achieve
  100% by detecting the verb "believes"/"wants" without understanding mental
  states. This v2 creates controlled stories where mental states must be
  inferred from NARRATIVE CONTEXT, not from explicit verb signals.

Four experimental conditions:
  A) Explicit verbs (baseline/sanity check, like v1)
  B) Implicit context (KEY TEST) — no mental-state verbs
  C) Agent-indexed — multi-agent stories, probe must decode per-agent state
  D) Recombination — train on subset, test on novel combinations

The key comparison is Condition A vs Condition B: if B drops significantly,
the v1 probe was detecting verbs, not mental states.

Usage:
  python agent_state_probe_v2.py \
    --model_path /path/to/model \
    --model_short Qwen2.5-7B-Instruct \
    --output_dir /path/to/results/
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import random

# ============================================================
# Name and setting pools for diverse stimulus generation
# ============================================================
NAME_POOL = [
    "Sarah", "Tom", "Lisa", "James", "Maria", "David", "Priya", "Carlos",
    "Yuki", "Omar", "Nadia", "Kevin", "Aisha", "Marcus", "Lin", "Rachel",
    "Dmitri", "Fatima", "Jorge", "Elena", "Wei", "Amara", "Pavel", "Zara",
    "Kenji", "Sofia", "Ibrahim", "Hannah", "Raj", "Clara",
]

SETTINGS = [
    "workplace", "school", "home", "hospital", "airport", "restaurant",
    "laboratory", "courtroom", "farm", "theater", "library", "market",
    "office", "workshop", "park", "station", "museum", "studio",
]

# Forbidden mental-state verbs/phrases for Condition B validation
FORBIDDEN_PATTERNS = [
    r"\bbelieves?\b", r"\bthinks?\b", r"\bassumes?\b", r"\bconvinced\b",
    r"\bwants?\b", r"\bhopes?\b", r"\bwishes?\b", r"\blongs?\b", r"\bdesires?\b",
    r"\bplans?\b", r"\bintends?\b", r"\bintending\b", r"\bdecided\b",
    r"\bknows?\b", r"\baware\b", r"\blearned\b", r"\bfound out\b",
    r"\bdoesn'?t know\b", r"\bunaware\b", r"\bhasn'?t learned\b",
    r"\bno idea\b", r"\brealized?\b", r"\bexpects?\b",
    r"\bsuspects?\b", r"\bimagines?\b",
]
FORBIDDEN_RE = re.compile("|".join(FORBIDDEN_PATTERNS), re.IGNORECASE)

# ============================================================
# Condition A: Explicit verbs (baseline, same design as v1)
# ============================================================
EXPLICIT_OBJECTS = [
    "the meeting is at 3pm", "the package has arrived", "the store is closed",
    "the test is tomorrow", "the flight is delayed", "the project is cancelled",
    "the key is under the mat", "the password has changed",
    "the deadline has moved", "the restaurant is fully booked",
    "the report is finished", "the bus leaves at noon",
    "the gate is locked", "the system is down", "the results are ready",
]
DESIRE_OBJECTS = [
    "a promotion", "a vacation", "a new laptop", "more free time",
    "better grades", "a quiet evening", "recognition from peers",
    "a fresh start", "more responsibility", "a creative outlet",
    "a window seat", "an early finish", "a bigger lab", "a second chance",
    "a warm meal",
]
ACTION_OBJECTS = [
    "finish the report", "call the client", "book the flight",
    "start the experiment", "resign from the position", "apply for the grant",
    "confront the colleague", "reorganize the team", "cancel the subscription",
    "move to a new city", "rewrite the proposal", "fix the engine",
    "study for the exam", "negotiate the contract", "launch the website",
]

EXPLICIT_TEMPLATES = {
    "belief": [
        "{agent} believes that {obj}.",
        "{agent} is convinced that {obj}.",
        "{agent} thinks that {obj}.",
        "{agent} assumes that {obj}.",
    ],
    "desire": [
        "{agent} wants {obj}.",
        "{agent} hopes for {obj}.",
        "{agent} wishes for {obj}.",
        "{agent} longs for {obj}.",
    ],
    "intention": [
        "{agent} plans to {obj}.",
        "{agent} intends to {obj}.",
        "{agent} has decided to {obj}.",
        "{agent} is going to {obj}.",
    ],
    "knowledge": [
        "{agent} knows that {obj}.",
        "{agent} is aware that {obj}.",
        "{agent} has learned that {obj}.",
        "{agent} found out that {obj}.",
    ],
    "ignorance": [
        "{agent} doesn't know that {obj}.",
        "{agent} is unaware that {obj}.",
        "{agent} hasn't learned that {obj}.",
        "{agent} has no idea that {obj}.",
    ],
}


def generate_condition_a(rng, n_per_state=80):
    """Condition A: explicit mental-state verbs (baseline/sanity check)."""
    stimuli = []
    for ms_type, templates in EXPLICIT_TEMPLATES.items():
        for i in range(n_per_state):
            agent = rng.choice(NAME_POOL)
            template = templates[i % len(templates)]
            if ms_type == "desire":
                obj = rng.choice(DESIRE_OBJECTS)
            elif ms_type == "intention":
                obj = rng.choice(ACTION_OBJECTS)
            else:
                obj = rng.choice(EXPLICIT_OBJECTS)
            text = template.format(agent=agent, obj=obj)
            stimuli.append({"text": text, "mental_state": ms_type,
                            "condition": "A", "agent": agent})
    rng.shuffle(stimuli)
    return stimuli


# ============================================================
# Condition B: Implicit context (KEY TEST — no mental-state verbs)
# ============================================================
# Each template pool has ~20 templates per mental state, ensuring diversity.
# Templates use {name}, {place}, {item}, {number}, {detail} slots for variety.

IMPLICIT_BELIEF_TEMPLATES = [
    # False belief: character acts on outdated/wrong information
    "{name} checked the schedule posted last Monday and headed to Room {number}. The schedule had been updated on Wednesday to move everything to Room {number2}.",
    "The old sign said the pharmacy closed at nine. {name} rushed through traffic to arrive at eight-fifty, but the new hours had been six o'clock since March.",
    "{name} followed the recipe from the 1998 edition, measuring out two cups of flour. The corrected printing called for three.",
    "The map in {name}'s glove compartment showed a bridge over the river. {name} drove thirty miles to cross it. The bridge had been demolished two years ago.",
    "{name} brought an umbrella after seeing the morning forecast. By noon the sky was clear; the forecast had been corrected at ten.",
    "{name} set the oven to 350 degrees, following the label on the box. The manufacturer had issued a recall — the correct temperature was 275.",
    "According to the memo {name} read, the client meeting was on Thursday. An email sent after {name} left had moved it to Friday.",
    "{name} walked to the east parking lot to find the car. A valet had moved it to the west lot an hour earlier.",
    "The textbook {name} studied from said the element had an atomic weight of 58. The value had been revised to 59 in the latest edition.",
    "{name} called the old phone number listed in the directory. The office had changed numbers six months ago.",
    "{name} packed warm clothes for {place} after reading a travel blog from January. {place} was now in the middle of a summer heat wave.",
    "{name} arrived at the gate printed on the boarding pass. A gate change announcement had played while {name} was in the restroom.",
    "Following the GPS route cached from last week, {name} turned left onto {place} Road. The road had been closed for construction since yesterday.",
    "{name} sent the payment to the old account number from the invoice. The vendor had updated the account details in a follow-up letter that never arrived.",
    "{name} returned the library book to the drop box on Oak Street. The library had moved to the new building on Pine Avenue last month.",
    "{name} prepared the presentation using the figures from the Q2 draft. The final Q2 report had revised every number downward by fifteen percent.",
    "{name} knocked on apartment {number}, the address from the invitation. The host had moved to apartment {number2} across the hall.",
    "The tide table pinned to {name}'s wall was from last year. {name} launched the boat at what should have been high tide, only to find exposed mud flats.",
    "{name} drove to the old clinic location on River Street after checking an outdated website. The clinic had relocated to the medical campus downtown.",
    "{name} ordered the daily special listed on the chalkboard from this morning. The kitchen had changed it to a different dish at noon.",
]

IMPLICIT_DESIRE_TEMPLATES = [
    # Desire: character shows wanting through behavior — seeking, reaching, lingering, choosing
    "Every morning, {name} stood outside the bakery before it opened, watching the pastries through the glass. {name} saved every spare coin in a jar on the shelf.",
    "{name} circled the jewelry display case three times, eyes fixed on the silver bracelet. {name} asked the clerk to hold it until payday.",
    "{name} spent every lunch break browsing real estate listings for houses near the ocean, bookmarking the same coastal cottage again and again.",
    "At the bookstore, {name} picked up the leather-bound atlas, turned it over, checked the price, put it back, then picked it up once more.",
    "{name} kept the concert poster taped above the desk for months, checking ticket prices every few days even though they kept rising.",
    "{name} lingered at the pet shelter, sitting on the floor with the gray kitten for over an hour before the staff gently said it was closing time.",
    "{name} pressed against the fence, watching the vintage motorcycle in the yard. {name} came back every weekend to see if a for-sale sign had appeared.",
    "{name} flipped through the course catalog for the third time, underlining the same photography elective with a highlighter.",
    "Whenever the topic of {place} came up, {name}'s voice changed. {name} kept a folder of flight deals and hotel reviews on the desktop.",
    "At the farmer's market, {name} sampled the aged cheddar twice, then stood near the stall pretending to look at something else.",
    "{name} stared at the job posting for the {place} office, refreshing the page daily even after the deadline had passed.",
    "{name} left the bicycle shop empty-handed but turned back at the door, asking the owner one more question about the red touring model.",
    "{name} arranged a collection of paint swatches along the kitchen wall, stepping back, rearranging, stepping back again, unable to stop comparing shades.",
    "{name} sat in the parked car outside the music store for ten minutes before going in to play the grand piano, then stayed until closing.",
    "The telescope in the catalog had been on {name}'s screen for weeks. {name} measured the shelf at home to make sure it would fit.",
    "{name} signed up for every open-house tour in the neighborhood, always asking about the garden size before anything else.",
    "During the auction, {name} gripped the paddle tightly, leaning forward each time the antique clock was mentioned, only to lower the paddle at the last second.",
    "{name} traced the route to {place} on a paper map, marking rest stops and scenic overlooks with small red dots.",
    "{name} returned to the same bench in the park every evening, watching the sunset over the lake and staying until the last light faded.",
    "{name} kept the graduate program brochure in the nightstand drawer and read the course descriptions before falling asleep.",
]

IMPLICIT_INTENTION_TEMPLATES = [
    # Intention: character prepares, gathers resources, sets things in motion toward a future goal
    "{name} cleared the desk, organized the notes by date, and drafted three different opening statements. The alarm was set for five in the morning.",
    "{name} bought a new suitcase, renewed the passport, and started a daily language lesson on the phone. A one-way ticket sat in the email drafts folder.",
    "{name} measured the walls, sketched a layout on graph paper, and ordered paint samples in four shades of blue.",
    "{name} printed five copies of the resume, ironed the gray suit, and rehearsed answers to common interview questions in front of the mirror.",
    "{name} filled the car with gas, loaded the camping gear, and downloaded offline maps for the mountain trails near {place}.",
    "{name} registered the domain name, purchased a hosting package, and uploaded a placeholder page reading 'Coming Soon.'",
    "{name} enrolled in an evening welding course, bought safety goggles, and cleared space in the garage for a workbench.",
    "{name} signed the lease paperwork, scheduled the movers for Saturday, and forwarded the mail to the new address on {place} Street.",
    "{name} collected soil samples from the backyard, ordered {number} seed packets, and built raised beds out of cedar planks.",
    "{name} wrote a letter to the department head, attached the supporting documents, and handed the sealed envelope to the front desk.",
    "{name} stacked {number} textbooks on the kitchen table, created a color-coded study schedule, and turned off all phone notifications.",
    "{name} opened a savings account, set up automatic transfers of two hundred dollars a month, and cut the credit card in half.",
    "{name} drafted a business proposal, contacted three potential investors, and booked a conference room for next Tuesday.",
    "{name} downloaded architectural blueprints, met with two contractors, and applied for a building permit at city hall.",
    "{name} shipped boxes of belongings to {place}, arranged a farewell dinner with friends, and gave notice at the apartment.",
    "{name} filed the application, gathered three letters of recommendation, and scheduled a portfolio review for the following week.",
    "{name} compiled a list of {number} suppliers, negotiated bulk pricing, and reserved warehouse space near the port.",
    "{name} dusted off the old guitar, replaced the strings, and taped a practice schedule to the wall — one hour every evening.",
    "{name} interviewed four candidates, checked their references, and prepared an offer letter for the strongest one.",
    "{name} backed up every file, wiped the hard drive, installed a fresh operating system, and created a new administrator account.",
]

IMPLICIT_KNOWLEDGE_TEMPLATES = [
    # Knowledge: character has verified information and acts confidently/correctly
    "After reading the lab results twice and cross-checking the reference values, {name} adjusted the dosage precisely to {number} milligrams.",
    "{name} typed the six-digit security code from memory, opened the vault on the first try, and retrieved the file labeled '{detail}.'",
    "Without hesitation, {name} bypassed the main corridor and took the service elevator directly to the server room on floor {number}.",
    "{name} quoted the statute number, cited the relevant precedent, and corrected the opposing counsel's misreading of the clause.",
    "At the intersection, {name} turned right without checking the GPS. The shortcut saved twelve minutes, exactly as {name} had calculated.",
    "{name} identified the bird by its call alone — a Swainson's thrush — and pointed to the exact branch where it was perched.",
    "The mechanic had missed it, but {name} went straight to the timing belt, removed it, and revealed the hairline crack underneath.",
    "{name} answered each of the seven trivia questions about {place} without pausing, including the founding year and the original name of the main street.",
    "When the fire alarm triggered, {name} led the group past the jammed main exit and through the side door that opened onto the loading dock.",
    "{name} recalibrated the telescope by adjusting the declination axis by exactly {number} arc-seconds, centering the target star on the first attempt.",
    "At the tasting, {name} identified the wine as a 2016 Barolo from the Serralunga vineyard without looking at the label.",
    "{name} walked directly to the fuse box, flipped the third breaker on the second row, and the lights came back on throughout the building.",
    "The customs officer asked about the import regulations. {name} produced the correct form, already filled out, with the tariff code written in the margin.",
    "{name} mixed the epoxy in a 5:1 ratio, applied it to the fracture line, and clamped it at exactly the angle needed for the joint to hold under load.",
    "{name} told the paramedics the patient's blood type, listed the current medications, and gave the exact time the symptoms started.",
    "{name} disassembled the watch movement, replaced the worn escape wheel with the correct part, and had it running again within an hour.",
    "The password had been changed overnight, but {name} entered the new one correctly — the IT bulletin from yesterday had included it.",
    "{name} navigated the old quarter of {place} without a map, naming each alley and courtyard from memory as they passed.",
    "{name} adjusted the pH of the solution to 7.4 in a single step, using exactly the amount of buffer calculated beforehand.",
    "{name} rattled off the train schedule — platform {number}, departure at 14:22, arriving at {place} at 16:05 — without looking anything up.",
]

IMPLICIT_IGNORANCE_TEMPLATES = [
    # Ignorance: character is missing critical information, behavior shows confusion/mistakes
    "{name} arrived at the airport in shorts and sandals, suitcase full of summer clothes. The departure board showed 'Helsinki — Boarding.'",
    "{name} stood at the crossroads for ten minutes, turning the map sideways and then upside down, before asking a passing jogger which way to the station.",
    "The printer jammed. {name} opened every panel, pressed every visible button, and finally unplugged it entirely, staring at the blinking error code.",
    "{name} wandered through the hospital corridors, reading door numbers aloud, doubling back twice, and finally stopping a nurse to ask for the radiology department.",
    "At the formal dinner, {name} picked up the dessert fork for the salad course. The host glanced across the table but said nothing.",
    "{name} added salt instead of sugar to the batter and only noticed after tasting the first baked muffin.",
    "{name} clicked 'Reply All' and sent the private complaint about the manager to the entire department. The inbox filled with silence.",
    "{name} tried to board the express train with a local ticket. The conductor shook her head and pointed at the printed fare zone.",
    "{name} plugged the {detail} adapter into the wrong port, tried three different cables, and spent an hour troubleshooting before noticing the label above the correct socket.",
    "At the auction, {name} scratched an ear and accidentally placed a bid of twelve thousand dollars on a painting meant as a joke lot.",
    "{name} greeted the new colleague by the wrong name three times before someone quietly handed over a name badge.",
    "{name} mixed bleach and ammonia while cleaning the bathroom. The sharp smell hit within seconds, and {name} stumbled backward coughing.",
    "The rental car had a push-button start. {name} spent five minutes searching for a keyhole in the steering column.",
    "{name} showed up on Monday for the appointment scheduled on Tuesday and sat in the empty waiting room for half an hour before checking the confirmation email.",
    "{name} ordered in halting French at the cafe in {place}. The waiter replied in perfect English and gently corrected the pronunciation.",
    "{name} put diesel in the gasoline car. The engine sputtered and died two blocks from the station.",
    "{name} pressed 'F' for Fahrenheit on the lab thermostat, not seeing the small 'C' toggle. The incubation ran twenty degrees too hot.",
    "{name} wore a full suit to the beach barbecue, standing stiffly among guests in swimsuits and flip-flops.",
    "{name} attempted to pay with a credit card at the cash-only market stall, patting every pocket before the vendor pointed to the sign.",
    "During the fire drill, {name} headed for the elevators. A colleague caught {name} by the arm and redirected toward the stairwell.",
]


def _fill_template(rng, template, name=None):
    """Fill in a template's placeholders with randomized content."""
    if name is None:
        name = rng.choice(NAME_POOL)
    text = template.replace("{name}", name)
    text = text.replace("{place}", rng.choice([
        "Portland", "Lisbon", "Kyoto", "Marrakech", "Zurich", "Buenos Aires",
        "Vancouver", "Dublin", "Hanoi", "Nairobi", "Stockholm", "Cusco",
        "Bruges", "Reykjavik", "Osaka", "Santiago", "Florence", "Accra",
    ]))
    text = text.replace("{number2}", str(rng.randint(200, 450)))
    text = text.replace("{number}", str(rng.randint(2, 99)))
    text = text.replace("{item}", rng.choice([
        "a wrench", "a voltmeter", "a notebook", "a stethoscope", "a compass",
        "a calculator", "a magnifying glass", "a thermometer", "a level",
    ]))
    text = text.replace("{detail}", rng.choice([
        "HDMI", "USB-C", "Ethernet", "DisplayPort", "VGA", "Thunderbolt",
        "quarterly-report", "case-summary", "intake-form", "lab-protocol",
    ]))
    return text, name


IMPLICIT_POOLS = {
    "belief": IMPLICIT_BELIEF_TEMPLATES,
    "desire": IMPLICIT_DESIRE_TEMPLATES,
    "intention": IMPLICIT_INTENTION_TEMPLATES,
    "knowledge": IMPLICIT_KNOWLEDGE_TEMPLATES,
    "ignorance": IMPLICIT_IGNORANCE_TEMPLATES,
}


def generate_condition_b(rng, n_per_state=80):
    """Condition B: implicit context — mental state inferred from narrative."""
    stimuli = []
    for ms_type, pool in IMPLICIT_POOLS.items():
        for i in range(n_per_state):
            template = pool[i % len(pool)]
            text, name = _fill_template(rng, template)
            # Validate: no forbidden mental-state verbs
            if FORBIDDEN_RE.search(text):
                raise ValueError(
                    f"FORBIDDEN mental-state verb found in Condition B stimulus "
                    f"(state={ms_type}):\n  {text}"
                )
            stimuli.append({"text": text, "mental_state": ms_type,
                            "condition": "B", "agent": name})
    rng.shuffle(stimuli)
    return stimuli


# ============================================================
# Condition C: Agent-indexed (multi-agent stories)
# ============================================================
# Each story has 2 agents with DIFFERENT mental states about the SAME situation.
# 5 pair types × 12 items each = 60 stories.

AGENT_INDEXED_TEMPLATES = {
    ("belief", "knowledge"): [
        "The package was delivered to the neighbor's house yesterday. {a1} keeps checking the front porch every hour. {a2} saw the delivery truck pull up next door and signed for it.",
        "The department budget was cut by thirty percent last week. {a1} submitted a request for new equipment at full price. {a2} already revised the budget to fit the reduced allocation.",
        "The highway exit to {place} was permanently closed last month. {a1} merged into the right lane a mile ahead, ready to take it. {a2} stayed in the center lane and prepared for the detour route.",
        "The café switched to a new supplier and changed its menu. {a1} walked in and ordered the old house blend by name. {a2} glanced at the updated menu board and ordered the new roast.",
        "The company relocated the server room to the second floor over the weekend. {a1} headed to the basement with a toolbox. {a2} took the elevator up and badged into the new room.",
        "The landlord replaced the front-door lock yesterday. {a1} stood at the door trying the old key over and over. {a2} used the new key, opened the door on the first attempt, and walked in.",
        "The school moved the exam from Tuesday to Thursday in an email sent last night. {a1} arrived on Tuesday morning with sharpened pencils. {a2} spent Tuesday reviewing notes, calm and unhurried.",
        "The trailhead parking lot was closed for repaving since Friday. {a1} drove forty minutes to the lot and found barriers across the entrance. {a2} parked at the alternate entrance on the south side.",
        "The pharmacy switched the generic brand of the medication. {a1} brought back the new pills, confused by the different color and shape. {a2} recognized the new packaging from the notice stapled to the bag.",
        "The theater changed the showtime from seven to eight. {a1} arrived at six-thirty and sat in the empty lobby, checking the time. {a2} strolled in at seven-forty-five and found a seat without rushing.",
        "The office Wi-Fi password changed overnight. {a1} restarted the laptop three times trying to connect with the old one. {a2} typed the new password from the slip the IT desk had handed out.",
        "The community garden reassigned plots at the start of the season. {a1} watered and weeded someone else's plot for an hour before noticing the new numbered stakes. {a2} walked directly to the reassigned plot.",
    ],
    ("belief", "ignorance"): [
        "The building's south stairwell was blocked for painting. {a1} confidently took the south stairs and hit a locked door on the third landing. {a2} wandered the hallway, tried both stairwells, and eventually took the elevator.",
        "The town abolished its parallel-parking requirement for driving tests last year. {a1} practiced parallel parking for a week before the test. {a2} had never driven before and asked the examiner what the test involved.",
        "The restaurant's kitchen caught fire and the place closed. {a1} made a reservation online through the old website. {a2} showed up, looked at the boarded-up windows, and asked a passerby what happened.",
        "A tree fell across the running trail during the storm. {a1} set out on the trail at dawn, fully expecting a clear path. {a2} reached the fork and stood looking at both branches, unsure which way the trail went.",
        "The professor changed the textbook edition for the new semester. {a1} bought the old edition from a classmate and highlighted every chapter. {a2} showed up to class empty-handed, asking which textbook to get.",
        "The bus route was rerouted around the construction on Main Street. {a1} waited at the old stop on Main Street for twenty minutes. {a2} paced up and down the block looking for any bus stop sign.",
        "The pool's schedule was split into lap swim and open swim this month. {a1} showed up during lap swim with floaties and a beach ball. {a2} asked the lifeguard if the pool was even open today.",
        "The concert venue moved from the arena to the smaller club. {a1} drove to the arena and walked toward the main entrance. {a2} looked at both addresses on the phone, unsure which one to trust.",
        "The quarterly report format changed from spreadsheet to slide deck. {a1} submitted a detailed spreadsheet like last quarter. {a2} emailed the manager to ask what format to use.",
        "The museum's free-admission day shifted from Sunday to Wednesday. {a1} came on Sunday expecting free entry. {a2} arrived on a random Thursday and asked the front desk if there was a discount.",
        "The company switched from Slack to Teams overnight. {a1} typed messages into Slack all morning, wondering why no one responded. {a2} opened the laptop and stared at the new Teams icon, clicking it tentatively.",
        "The train station added a second platform last month. {a1} went to platform one as always, but the train now left from platform two. {a2} looked up at the departure boards, unsure which platform to go to.",
    ],
    ("desire", "intention"): [
        "{a1} pressed against the window of the guitar shop, staring at the sunburst acoustic for the fourth week in a row. {a2} walked in, put three hundred dollars on the counter, and asked to try it.",
        "{a1} scrolled through photos of the cottage in {place} every evening, saving each new listing to a growing folder. {a2} booked a viewing for Saturday, arranged time off work, and mapped the drive.",
        "{a1} watched the open-water swimmers from the pier each morning, towel draped over one shoulder, never stepping off the dock. {a2} signed up for the coached swim session, bought goggles, and reserved a locker.",
        "{a1} lingered in front of the pastry case, counting coins quietly. {a2} handed the cashier a bill and pointed to the chocolate éclair.",
        "{a1} re-read the art school brochure for the fifth time, running a finger along the course descriptions. {a2} filled out the application, attached a portfolio, and mailed it certified.",
        "{a1} stood by the adoption board, reading the same dog's profile over and over, petting its photo. {a2} filled out the adoption paperwork, paid the fee, and scheduled the home visit.",
        "{a1} bookmarked the mountain-bike trail guide and checked the weather for the next six weekends. {a2} loaded the bike onto the car rack, packed lunch, and left before dawn.",
        "{a1} sat in the bleachers during every rehearsal, mouthing the lyrics from the audience. {a2} signed up for auditions, practiced the solo after work, and printed the sheet music.",
        "{a1} traced the outline of {place} on the globe and sighed each time a travel show mentioned it. {a2} renewed the passport, purchased the guidebook, and started budgeting for airfare.",
        "{a1} stared at the empty lot next door, sketching imaginary garden layouts in a notebook. {a2} hired a landscaper, ordered topsoil, and rented a rototiller.",
        "{a1} kept returning to the same jewelry store, asking to see the engagement ring under better light. {a2} confirmed the ring size, placed a deposit, and arranged pickup for Friday.",
        "{a1} added the marathon to the calendar year after year and never laced up the running shoes. {a2} started a twelve-week training program, set daily mileage targets, and bought new shoes.",
    ],
    ("knowledge", "ignorance"): [
        "{a1} walked straight to the fuse box in the basement, flipped the third switch on the left, and the lights came back on. {a2} fumbled along the dark hallway, trying every light switch twice.",
        "{a1} quoted the building code from memory and told the inspector exactly which section applied. {a2} flipped through a thick manual, searching for the right page.",
        "At the wine tasting, {a1} identified the grape variety, the vintage, and the region after one sip. {a2} swirled the glass, took a hesitant sip, and guessed it was 'some kind of red.'",
        "{a1} entered the server credentials without looking anything up and restarted the service in under a minute. {a2} called the help desk and waited on hold for twenty minutes.",
        "The patient described the symptoms. {a1} named the condition immediately and outlined the treatment steps. {a2} opened the diagnostic manual and began cross-referencing the symptom list.",
        "{a1} navigated the old quarter of {place} without a map, pointing out landmarks as they passed. {a2} stopped at every intersection to consult the phone, turning it in circles.",
        "{a1} wired the circuit correctly on the first attempt and tested it with a multimeter. {a2} connected the wires in the wrong order and blew a fuse.",
        "When the hike turned steep, {a1} shifted weight forward and used the zigzag technique to maintain traction. {a2} slipped twice on the gravel and sat down to catch a breath.",
        "{a1} disarmed the alarm by pressing the four-digit code within seconds of entering. {a2} stared at the keypad, tried a birthday, and triggered the siren.",
        "At customs, {a1} produced the completed declaration form and the receipt for every item. {a2} rummaged through a bag of crumpled papers, asking the officer which form to fill out.",
        "{a1} set up the telescope, aligned it to Polaris, and tracked Jupiter across the sky. {a2} squinted through the eyepiece at a blur and asked which direction to point it.",
        "{a1} identified the mushroom species on sight and placed only the safe ones in the basket. {a2} picked every mushroom that looked edible and asked {a1} to check them.",
    ],
    ("belief", "desire"): [
        "The vintage bookshop was actually closed for renovation. {a1} walked up to the door and pulled the handle, stepping back in surprise when it didn't open. {a2} stood across the street, gazing at the window display and pressing fingertips to the glass.",
        "The results of the promotion were posted. {a1} scanned the list quickly, nodded at a name partway down, and went back to work. {a2} read the list slowly, lingering on every name, shoulders dropping slightly at the end.",
        "A rare comet was visible that night. {a1} set up the telescope at the coordinates published in the astronomy bulletin. {a2} sat on the rooftop with a blanket, gazing up at the sky, searching for something bright.",
        "The last ferry to {place} had already left at five. {a1} arrived at five-fifteen and walked toward the empty dock, checking the posted schedule. {a2} stood at the railing overlooking the harbor, watching the boats come and go, lingering until dark.",
        "The hiring committee already made their choice. {a1} wrote a congratulations email to the selected candidate. {a2} refreshed the application portal every few hours, reading the job description one more time.",
        "The gallery exhibit ended last week. {a1} drove to the gallery, parked, and walked to the front door before seeing the 'Exhibit Closed' sign. {a2} kept the exhibit catalog on the nightstand, flipping through the pages each evening.",
        "The final score of the match was three to one. {a1} told everyone at work that the home team won easily. {a2} replayed the highlights on the phone over and over, pausing on every near-miss by the losing side.",
        "The scholarship committee announced the winners. {a1} checked the list once and closed the browser tab. {a2} copied the announcement into a document, underlining each criterion, comparing it against a personal checklist.",
        "The mountain pass was snowed in for the season. {a1} packed the car with chains and drove to the base, planning to cross. {a2} pinned a photo of the pass above the desk and traced the road with a fingertip each morning.",
        "The antique clock at the estate sale was sold before noon. {a1} asked the organizer for the clock and was directed to the buyer. {a2} circled the empty display table twice, picking up the description card and reading it again.",
        "The adoption agency said the puppy had already been placed. {a1} nodded, thanked the staff, and left. {a2} asked for a photo of the puppy to keep, holding it carefully with both hands.",
        "The garden plot rental was full for the season. {a1} submitted a late application and received the waitlist notice. {a2} walked past the community garden every day, stopping to watch others tend their plots.",
    ],
}


def generate_condition_c(rng, n_per_pair=12):
    """Condition C: agent-indexed multi-agent stories."""
    stimuli = []
    for (ms_a, ms_b), templates in AGENT_INDEXED_TEMPLATES.items():
        for i in range(n_per_pair):
            template = templates[i % len(templates)]
            # Pick two distinct names
            names = rng.sample(NAME_POOL, 2)
            a1_name, a2_name = names[0], names[1]
            text = template.replace("{a1}", a1_name).replace("{a2}", a2_name)
            text = text.replace("{place}", rng.choice([
                "Portland", "Lisbon", "Kyoto", "Marrakech", "Zurich",
                "Buenos Aires", "Vancouver", "Dublin", "Hanoi", "Nairobi",
            ]))
            text = text.replace("{number}", str(rng.randint(2, 99)))
            stimuli.append({
                "text": text,
                "pair_type": f"{ms_a}+{ms_b}",
                "agent1": a1_name, "agent1_state": ms_a,
                "agent2": a2_name, "agent2_state": ms_b,
                "condition": "C",
            })
    rng.shuffle(stimuli)
    return stimuli


# ============================================================
# Condition D: Recombination / generalization test
# ============================================================
def generate_condition_d_novel(rng, n_per_state=20):
    """Generate NOVEL items for Condition D test set — new names + new content."""
    # Use names NOT in the main pool
    novel_names = [
        "Lucien", "Theodora", "Kiran", "Beatriz", "Soren", "Leila",
        "Idris", "Ingrid", "Rohan", "Celeste", "Thiago", "Maren",
        "Kofi", "Astrid", "Darian", "Noemi", "Haruto", "Ximena",
    ]
    novel_belief = [
        "{name} followed the directions scribbled on a napkin from last summer. The road it described now ended at a construction fence.",
        "{name} brought canned food for the shelter drive. The collection had switched to winter clothing two weeks ago.",
        "{name} applied the coupon from the magazine. The store clerk pointed out it had expired three months earlier.",
        "{name} tuned the radio to 94.7 for the morning show. The station had moved to 101.3 after a frequency reallocation.",
        "{name} parked in the employee lot using the old badge. Security had reprogrammed the gate last Friday.",
    ]
    novel_desire = [
        "{name} ran a thumb over the telescope listing in the catalog every night before bed, dog-earing the same page.",
        "{name} spent hours comparing flight prices to Patagonia, opening and closing the booking page without clicking 'confirm.'",
        "{name} kept a cutout of the vintage typewriter pinned to the corkboard, dusting it off every few days.",
        "{name} visited the animal rescue four Saturdays in a row, always sitting with the same one-eared tabby.",
        "{name} paused at the marina every jog, watching the sailboats rock and counting the masts.",
    ]
    novel_intention = [
        "{name} rented a storage unit, packed the winter gear, and listed the apartment for sublease starting next month.",
        "{name} ordered a soldering iron, downloaded the circuit diagrams, and cleared the workbench for the build.",
        "{name} wrote resignation letters in three drafts, sealed the final one, and placed it in the briefcase.",
        "{name} hired a tutor, blocked out study hours on the calendar, and ordered every recommended textbook.",
        "{name} got quotes from three moving companies, measured every piece of furniture, and labeled boxes by room.",
    ]
    novel_knowledge = [
        "{name} recited the alloy composition — seventy-two percent copper, twenty-eight percent zinc — and selected the correct brazing rod without hesitation.",
        "{name} corrected the pharmacist's decimal point, citing the maximum dosage guidelines from the latest formulary edition.",
        "{name} calibrated the spectrometer to the exact wavelength of the hydrogen-alpha line on the first try.",
        "{name} rattled off the tidal schedule for the next three days, accurate to the minute, while adjusting the mooring lines.",
        "{name} pointed out the load-bearing wall on the blueprint before the contractor could mark it wrong.",
    ]
    novel_ignorance = [
        "{name} tried to use chopsticks to eat soup, looking around the table for cues before switching to a spoon.",
        "{name} walked into the silent meditation session talking loudly on the phone. Every head turned.",
        "{name} handed the taxi driver euros in a country that used a different currency. The driver stared at the bills.",
        "{name} wore steel-toed boots to the yoga class and stood on the mat looking at everyone else's bare feet.",
        "{name} fed chocolate to the neighbor's dog, not seeing the vet's warning tag on the collar.",
    ]
    novel_pools = {
        "belief": novel_belief,
        "desire": novel_desire,
        "intention": novel_intention,
        "knowledge": novel_knowledge,
        "ignorance": novel_ignorance,
    }
    stimuli = []
    for ms_type, pool in novel_pools.items():
        for i in range(n_per_state):
            template = pool[i % len(pool)]
            name = rng.choice(novel_names)
            text, _ = _fill_template(rng, template, name=name)
            stimuli.append({"text": text, "mental_state": ms_type,
                            "condition": "D_novel", "agent": name})
    rng.shuffle(stimuli)
    return stimuli


# ============================================================
# Activation extraction
# ============================================================
def extract_activations(model, tokenizer, texts, device, layer=-1,
                        batch_size=8, max_length=256):
    """Extract mean-pooled activations from a specified layer."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[layer]  # (batch, seq, hidden)
        for b in range(len(batch)):
            mask = inputs["attention_mask"][b].bool()
            mean_act = hs[b][mask].float().mean(dim=0).cpu().numpy()
            all_acts.append(mean_act)
        # Free GPU memory
        del inputs, out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.array(all_acts)


def extract_agent_activations(model, tokenizer, stories, device, layer=-1,
                              max_length=256):
    """Extract activations at each agent's LAST mention in a multi-agent story.

    For each story, we find the last token span of each agent name and take
    the mean activation over that span. This forces the probe to decode
    agent-specific mental states.
    """
    results = []
    for story in stories:
        text = story["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[layer][0]  # (seq, hidden)
        token_ids = inputs["input_ids"][0]

        for agent_key, state_key in [("agent1", "agent1_state"),
                                     ("agent2", "agent2_state")]:
            agent_name = story[agent_key]
            agent_tokens = tokenizer.encode(agent_name, add_special_tokens=False)
            # Find last occurrence of the agent name tokens in the sequence
            seq = token_ids.tolist()
            last_pos = -1
            for pos in range(len(seq) - len(agent_tokens) + 1):
                if seq[pos:pos + len(agent_tokens)] == agent_tokens:
                    last_pos = pos
            if last_pos >= 0:
                span = hs[last_pos:last_pos + len(agent_tokens)]
                act = span.float().mean(dim=0).cpu().numpy()
            else:
                # Fallback: use mean of entire sequence
                act = hs.float().mean(dim=0).cpu().numpy()
            results.append({
                "activation": act,
                "mental_state": story[state_key],
                "agent": story[agent_key],
                "pair_type": story["pair_type"],
            })

        del inputs, out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


# ============================================================
# Probing helpers
# ============================================================
MENTAL_STATES = ["belief", "desire", "ignorance", "intention", "knowledge"]


def run_pairwise_probes(X, y_labels, n_splits=5):
    """Run pairwise binary classification probes for all pairs."""
    ms_types = sorted(set(y_labels))
    pair_results = []
    for ms_a, ms_b in combinations(ms_types, 2):
        idx_a = [i for i, l in enumerate(y_labels) if l == ms_a]
        idx_b = [i for i, l in enumerate(y_labels) if l == ms_b]
        Xp = np.vstack([X[idx_a], X[idx_b]])
        yp = np.array([0] * len(idx_a) + [1] * len(idx_b))
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        n_cv = min(n_splits, min(len(idx_a), len(idx_b)))
        if n_cv < 2:
            pair_results.append({"ms_a": ms_a, "ms_b": ms_b,
                                 "accuracy": float("nan")})
            continue
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, Xp, yp, cv=skf, scoring="accuracy")
        pair_results.append({"ms_a": ms_a, "ms_b": ms_b,
                             "accuracy": float(scores.mean()),
                             "std": float(scores.std())})
    return pair_results


def run_five_way_probe(X, y_labels, n_splits=5):
    """Run 5-way classification probe."""
    ms_types = sorted(set(y_labels))
    y = np.array([ms_types.index(l) for l in y_labels])
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Agent-state factorization probe v2 — verb-confound removed")
    parser.add_argument("--model_path", required=True, help="Path to HF model")
    parser.add_argument("--model_short", required=True, help="Short model name")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--n_per_condition", type=int, default=80,
                        help="Items per mental state per condition (A & B)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which layer to extract (-1 = last)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    rng = random.Random(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # 1. Generate all stimuli
    # ------------------------------------------------------------------
    print("=" * 70)
    print("GENERATING STIMULI")
    print("=" * 70)

    stim_a = generate_condition_a(rng, n_per_state=args.n_per_condition)
    print(f"Condition A (explicit):       {len(stim_a)} items")

    stim_b = generate_condition_b(rng, n_per_state=args.n_per_condition)
    print(f"Condition B (implicit):       {len(stim_b)} items")

    stim_c = generate_condition_c(rng, n_per_pair=12)
    print(f"Condition C (agent-indexed):  {len(stim_c)} stories "
          f"({len(stim_c) * 2} agent probes)")

    stim_d_novel = generate_condition_d_novel(rng, n_per_state=20)
    print(f"Condition D novel test items: {len(stim_d_novel)} items")

    # Condition D: split A+B into train (80%) and test-seen (20%)
    combined_ab = stim_a + stim_b
    rng.shuffle(combined_ab)
    # Stratified split
    by_state = {}
    for s in combined_ab:
        by_state.setdefault(s["mental_state"], []).append(s)
    train_items, test_seen_items = [], []
    for ms, items in by_state.items():
        cutoff = int(len(items) * 0.8)
        train_items.extend(items[:cutoff])
        test_seen_items.extend(items[cutoff:])
    rng.shuffle(train_items)
    rng.shuffle(test_seen_items)
    print(f"Condition D train:            {len(train_items)} items")
    print(f"Condition D test-seen:        {len(test_seen_items)} items")
    print(f"Condition D test-novel:       {len(stim_d_novel)} items")

    # Print some examples
    print("\n--- Condition B examples (one per mental state) ---")
    shown = set()
    for s in stim_b:
        ms = s["mental_state"]
        if ms not in shown:
            print(f"  [{ms}] {s['text'][:120]}...")
            shown.add(ms)
        if len(shown) == 5:
            break

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"LOADING MODEL: {args.model_path}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    layer_idx = args.layer if args.layer != -1 else n_layers
    print(f"  Extracting from layer {layer_idx} / {n_layers}")

    # ------------------------------------------------------------------
    # 3. Extract activations
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EXTRACTING ACTIVATIONS")
    print(f"{'=' * 70}")

    print("  Condition A (explicit)...")
    texts_a = [s["text"] for s in stim_a]
    acts_a = extract_activations(model, tokenizer, texts_a, device,
                                 layer=layer_idx)
    labels_a = [s["mental_state"] for s in stim_a]
    print(f"    Shape: {acts_a.shape}")

    print("  Condition B (implicit)...")
    texts_b = [s["text"] for s in stim_b]
    acts_b = extract_activations(model, tokenizer, texts_b, device,
                                 layer=layer_idx)
    labels_b = [s["mental_state"] for s in stim_b]
    print(f"    Shape: {acts_b.shape}")

    print("  Condition C (agent-indexed)...")
    agent_results_c = extract_agent_activations(
        model, tokenizer, stim_c, device, layer=layer_idx)
    acts_c = np.array([r["activation"] for r in agent_results_c])
    labels_c = [r["mental_state"] for r in agent_results_c]
    print(f"    Shape: {acts_c.shape}")

    # Condition D: train set activations (from combined A+B train split)
    print("  Condition D train set...")
    texts_d_train = [s["text"] for s in train_items]
    acts_d_train = extract_activations(model, tokenizer, texts_d_train, device,
                                       layer=layer_idx)
    labels_d_train = [s["mental_state"] for s in train_items]
    print(f"    Shape: {acts_d_train.shape}")

    print("  Condition D test-seen set...")
    texts_d_seen = [s["text"] for s in test_seen_items]
    acts_d_seen = extract_activations(model, tokenizer, texts_d_seen, device,
                                      layer=layer_idx)
    labels_d_seen = [s["mental_state"] for s in test_seen_items]
    print(f"    Shape: {acts_d_seen.shape}")

    print("  Condition D test-novel set...")
    texts_d_novel = [s["text"] for s in stim_d_novel]
    acts_d_novel = extract_activations(model, tokenizer, texts_d_novel, device,
                                       layer=layer_idx)
    labels_d_novel = [s["mental_state"] for s in stim_d_novel]
    print(f"    Shape: {acts_d_novel.shape}")

    # ------------------------------------------------------------------
    # 4. Run probes
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RUNNING PROBES")
    print(f"{'=' * 70}")

    # --- Condition A: Explicit ---
    print("\n--- Condition A: Explicit verbs (baseline) ---")
    a_five_way, a_five_std = run_five_way_probe(acts_a, labels_a)
    print(f"  5-way accuracy: {a_five_way:.1%} +/- {a_five_std:.1%} "
          f"(chance={1/5:.1%})")
    a_pairs = run_pairwise_probes(acts_a, labels_a)
    a_accs = [p["accuracy"] for p in a_pairs]
    print(f"  Pairwise mean: {np.mean(a_accs):.1%}, "
          f"min: {np.min(a_accs):.1%}, max: {np.max(a_accs):.1%}")
    for p in sorted(a_pairs, key=lambda x: x["accuracy"]):
        print(f"    {p['ms_a']:>12s} vs {p['ms_b']:<12s}  {p['accuracy']:.1%}")

    # --- Condition B: Implicit ---
    print("\n--- Condition B: Implicit context (KEY TEST) ---")
    b_five_way, b_five_std = run_five_way_probe(acts_b, labels_b)
    print(f"  5-way accuracy: {b_five_way:.1%} +/- {b_five_std:.1%} "
          f"(chance={1/5:.1%})")
    b_pairs = run_pairwise_probes(acts_b, labels_b)
    b_accs = [p["accuracy"] for p in b_pairs]
    print(f"  Pairwise mean: {np.mean(b_accs):.1%}, "
          f"min: {np.min(b_accs):.1%}, max: {np.max(b_accs):.1%}")
    for p in sorted(b_pairs, key=lambda x: x["accuracy"]):
        print(f"    {p['ms_a']:>12s} vs {p['ms_b']:<12s}  {p['accuracy']:.1%}")

    # --- Condition C: Agent-indexed ---
    print("\n--- Condition C: Agent-indexed multi-agent ---")
    c_five_way, c_five_std = run_five_way_probe(acts_c, labels_c)
    print(f"  Overall 5-way accuracy: {c_five_way:.1%} +/- {c_five_std:.1%}")

    # Per pair-type accuracy
    pair_types = sorted(set(r["pair_type"] for r in agent_results_c))
    c_per_pair = []
    for pt in pair_types:
        idx = [i for i, r in enumerate(agent_results_c) if r["pair_type"] == pt]
        if len(idx) < 4:
            c_per_pair.append({"pair_type": pt, "accuracy": float("nan")})
            continue
        Xp = acts_c[idx]
        yp = [labels_c[i] for i in idx]
        ms_in_pair = sorted(set(yp))
        if len(ms_in_pair) < 2:
            c_per_pair.append({"pair_type": pt, "accuracy": float("nan")})
            continue
        y_bin = np.array([ms_in_pair.index(l) for l in yp])
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        n_cv = min(5, min(sum(y_bin == 0), sum(y_bin == 1)))
        if n_cv < 2:
            c_per_pair.append({"pair_type": pt, "accuracy": float("nan")})
            continue
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, Xp, y_bin, cv=skf, scoring="accuracy")
        acc = float(scores.mean())
        c_per_pair.append({"pair_type": pt, "accuracy": acc})
        print(f"    {pt}: {acc:.1%}")

    # --- Condition D: Recombination ---
    print("\n--- Condition D: Recombination / generalization ---")
    ms_types = sorted(set(labels_d_train))
    y_train = np.array([ms_types.index(l) for l in labels_d_train])
    y_seen = np.array([ms_types.index(l) for l in labels_d_seen])
    y_novel = np.array([ms_types.index(l) for l in labels_d_novel])

    clf_d = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf_d.fit(acts_d_train, y_train)

    d_train_acc = float(clf_d.score(acts_d_train, y_train))
    d_seen_acc = float(clf_d.score(acts_d_seen, y_seen))
    d_novel_acc = float(clf_d.score(acts_d_novel, y_novel))
    print(f"  Train accuracy:      {d_train_acc:.1%}")
    print(f"  Test-seen accuracy:  {d_seen_acc:.1%}")
    print(f"  Test-novel accuracy: {d_novel_acc:.1%}")

    # ------------------------------------------------------------------
    # 5. Key comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("KEY COMPARISON: EXPLICIT vs IMPLICIT")
    print(f"{'=' * 70}")

    explicit_mean = float(np.mean(a_accs))
    implicit_mean = float(np.mean(b_accs))
    drop = explicit_mean - implicit_mean

    if drop > 0.15:
        interpretation = (
            f"Large drop ({drop:.1%}): v1 probe likely exploited verb cues. "
            f"Implicit probe accuracy ({implicit_mean:.1%}) reflects true "
            f"mental-state representation quality."
        )
    elif drop > 0.05:
        interpretation = (
            f"Moderate drop ({drop:.1%}): verb cues contributed partially. "
            f"The model has genuine but weaker mental-state representations."
        )
    else:
        interpretation = (
            f"Minimal drop ({drop:.1%}): mental-state decoding is robust "
            f"even without verb cues. The model genuinely represents mental states."
        )
    print(f"  Explicit (A) mean pairwise: {explicit_mean:.1%}")
    print(f"  Implicit (B) mean pairwise: {implicit_mean:.1%}")
    print(f"  Drop (A - B):               {drop:.1%}")
    print(f"  Interpretation: {interpretation}")

    # Per-pair comparison
    print("\n  Per-pair comparison (A vs B):")
    a_dict = {(p["ms_a"], p["ms_b"]): p["accuracy"] for p in a_pairs}
    b_dict = {(p["ms_a"], p["ms_b"]): p["accuracy"] for p in b_pairs}
    for pair_key in sorted(a_dict.keys()):
        a_val = a_dict[pair_key]
        b_val = b_dict.get(pair_key, float("nan"))
        d = a_val - b_val
        print(f"    {pair_key[0]:>12s} vs {pair_key[1]:<12s}  "
              f"A={a_val:.1%}  B={b_val:.1%}  drop={d:+.1%}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    output = {
        "model": args.model_short,
        "layer": layer_idx,
        "n_per_condition": args.n_per_condition,
        "mental_states": MENTAL_STATES,

        "condition_A_explicit": {
            "five_way_accuracy": a_five_way,
            "five_way_std": a_five_std,
            "pairwise_probes": a_pairs,
        },
        "condition_B_implicit": {
            "five_way_accuracy": b_five_way,
            "five_way_std": b_five_std,
            "pairwise_probes": b_pairs,
        },
        "condition_C_agent_indexed": {
            "overall_accuracy": c_five_way,
            "overall_std": c_five_std,
            "per_pair_accuracy": c_per_pair,
        },
        "condition_D_recombination": {
            "train_accuracy": d_train_acc,
            "test_seen_accuracy": d_seen_acc,
            "test_novel_accuracy": d_novel_acc,
        },

        "key_comparison": {
            "explicit_mean_pairwise": explicit_mean,
            "implicit_mean_pairwise": implicit_mean,
            "drop": drop,
            "interpretation": interpretation,
        },
    }

    out_path = out_dir / f"{args.model_short}_agent_state_probe_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Also save stimuli for reproducibility / inspection
    stim_path = out_dir / f"{args.model_short}_agent_state_probe_v2_stimuli.json"
    all_stim = {
        "condition_A": [{"text": s["text"], "mental_state": s["mental_state"],
                         "agent": s["agent"]} for s in stim_a],
        "condition_B": [{"text": s["text"], "mental_state": s["mental_state"],
                         "agent": s["agent"]} for s in stim_b],
        "condition_C": [{k: v for k, v in s.items()} for s in stim_c],
        "condition_D_novel": [{"text": s["text"],
                               "mental_state": s["mental_state"],
                               "agent": s["agent"]} for s in stim_d_novel],
    }
    with open(stim_path, "w") as f:
        json.dump(all_stim, f, indent=2)
    print(f"Saved stimuli to {stim_path}")


if __name__ == "__main__":
    main()
