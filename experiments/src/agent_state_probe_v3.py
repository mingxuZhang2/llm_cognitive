#!/usr/bin/env python3
"""
Experiment: Agent-state binding probe v3 — controls for v2 artifacts.

Fixes over v2:
  1. Counter-balanced agent assignment (randomly swap who gets which state)
  2. Larger sample size (30+ per pair, not 12)
  3. MLP probe alongside linear probe
  4. Multi-layer sweep (every 4th layer)
  5. Separate "sentence-level" and "agent-level" extraction for condition C

Core question: Is belief-specific binding failure real or artifact?
  - If MLP also fails → info not encoded
  - If MLP succeeds → info exists but is nonlinear
  - If counter-balanced version changes results → v2 had positional confound
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random


NAME_POOL = [
    "Sarah", "Tom", "Lisa", "James", "Maria", "David", "Priya", "Carlos",
    "Yuki", "Omar", "Nadia", "Kevin", "Aisha", "Marcus", "Lin", "Rachel",
    "Dmitri", "Fatima", "Jorge", "Elena", "Wei", "Amara", "Pavel", "Zara",
    "Kenji", "Sofia", "Ibrahim", "Hannah", "Raj", "Clara",
]

MENTAL_STATES = ["belief", "desire", "ignorance", "intention", "knowledge"]

# ============================================================
# Condition C templates — each template is a FUNCTION that takes
# (name_a, name_b) and returns text. Agent-state assignment is
# done OUTSIDE the template so we can counter-balance.
# ============================================================
# Template format: callable(a1_name, a2_name) -> str
# a1 always has state_1 of the pair, a2 has state_2.
# Counter-balancing swaps assignment at call time.

PAIR_TEMPLATES = {
    ("belief", "knowledge"): [
        lambda a, b: f"The package was delivered to the neighbor's house yesterday. {a} keeps checking the front porch every hour. {b} saw the delivery truck pull up next door and signed for it.",
        lambda a, b: f"The department budget was cut by thirty percent last week. {a} submitted a request for new equipment at full price. {b} already revised the budget to fit the reduced allocation.",
        lambda a, b: f"The highway exit was permanently closed last month. {a} merged into the right lane a mile ahead, ready to take it. {b} stayed in the center lane and prepared for the detour route.",
        lambda a, b: f"The café switched to a new supplier and changed its menu. {a} walked in and ordered the old house blend by name. {b} glanced at the updated menu board and ordered the new roast.",
        lambda a, b: f"The company relocated the server room to the second floor over the weekend. {a} headed to the basement with a toolbox. {b} took the elevator up and badged into the new room.",
        lambda a, b: f"The landlord replaced the front-door lock yesterday. {a} stood at the door trying the old key over and over. {b} used the new key and opened the door on the first attempt.",
        lambda a, b: f"The school moved the exam from Tuesday to Thursday. {a} arrived on Tuesday morning with sharpened pencils. {b} spent Tuesday reviewing notes, calm and unhurried.",
        lambda a, b: f"The trailhead parking lot was closed for repaving since Friday. {a} drove forty minutes to the lot and found barriers across the entrance. {b} parked at the alternate entrance on the south side.",
        lambda a, b: f"The pharmacy switched the generic brand of the medication. {a} brought back the new pills, confused by the different shape. {b} recognized the new packaging from the notice stapled to the bag.",
        lambda a, b: f"The theater changed the showtime from seven to eight. {a} arrived at six-thirty and sat in the empty lobby. {b} strolled in at seven-forty-five and found a seat without rushing.",
        lambda a, b: f"The office Wi-Fi password changed overnight. {a} restarted the laptop three times trying to connect with the old one. {b} typed the new password from the slip the IT desk had handed out.",
        lambda a, b: f"The community garden reassigned plots at the start of the season. {a} watered someone else's plot for an hour before noticing the new stakes. {b} walked directly to the reassigned plot.",
        lambda a, b: f"The meeting room was changed from 301 to 405 in a late-night email. {a} sat in room 301 waiting for everyone. {b} went straight to 405 and set up the projector.",
        lambda a, b: f"The library moved its reference section to the third floor last week. {a} spent twenty minutes searching the old location on the second floor. {b} took the stairs to the third floor and found the book in minutes.",
        lambda a, b: f"The train schedule was revised, moving the 8am departure to 7:30am. {a} arrived at 7:50 and watched the train pull away. {b} boarded at 7:25 and settled into a window seat.",
        lambda a, b: f"The restaurant switched from table service to counter ordering. {a} sat at a table waiting for a server. {b} walked up to the counter and placed an order.",
        lambda a, b: f"The company changed the expense report form last month. {a} submitted using the old form and got a rejection email. {b} downloaded the new form and submitted correctly the first time.",
        lambda a, b: f"The park entrance fee increased from five to ten dollars. {a} handed the attendant a five-dollar bill. {b} held out a ten and walked through.",
        lambda a, b: f"The gym moved the spin class from Studio A to Studio B. {a} set up a bike in Studio A and waited alone. {b} walked into Studio B and joined the warm-up.",
        lambda a, b: f"The vaccination clinic relocated from the east wing to the main lobby. {a} followed the old signs to the east wing and found it empty. {b} checked the updated website and went to the lobby.",
        lambda a, b: f"The weekly team call was rescheduled from Monday to Wednesday. {a} dialed into the conference line on Monday and heard silence. {b} blocked Wednesday afternoon on the calendar.",
        lambda a, b: f"The store moved the electronics section from the second floor to the basement. {a} rode the escalator up and wandered past empty shelves. {b} headed downstairs and found the new display.",
        lambda a, b: f"The hotel reassigned their room from 512 to 718 due to maintenance. {a} swiped the key at 512 and the light turned red. {b} went to 718 and unpacked.",
        lambda a, b: f"The dentist's office changed its phone number last month. {a} called the old number and reached a disconnected line. {b} used the number on the new appointment card.",
    ],
    ("belief", "ignorance"): [
        lambda a, b: f"The south stairwell was blocked for painting. {a} took the south stairs and hit a locked door. {b} wandered the hallway, tried both stairwells, and eventually took the elevator.",
        lambda a, b: f"The driving test no longer requires parallel parking. {a} practiced parallel parking for a week before the test. {b} had never driven before and asked the examiner what the test involved.",
        lambda a, b: f"The restaurant closed after a kitchen fire. {a} made a reservation through the old website. {b} showed up, looked at the boarded windows, and asked a passerby what happened.",
        lambda a, b: f"A tree fell across the running trail. {a} set out on the trail expecting a clear path. {b} reached the fork and stood looking at both branches, unsure which way to go.",
        lambda a, b: f"The professor changed the textbook edition. {a} bought the old edition and highlighted every chapter. {b} showed up to class empty-handed, asking which book to get.",
        lambda a, b: f"The bus route was rerouted around construction. {a} waited at the old stop for twenty minutes. {b} paced up and down the block looking for any bus sign.",
        lambda a, b: f"The pool schedule was split into lap swim and open swim. {a} showed up during lap swim with floaties. {b} asked the lifeguard if the pool was even open today.",
        lambda a, b: f"The concert moved from the arena to a smaller club. {a} drove to the arena and walked toward the entrance. {b} looked at both addresses on the phone, unsure which to trust.",
        lambda a, b: f"The report format changed from spreadsheet to slide deck. {a} submitted a detailed spreadsheet like last quarter. {b} emailed the manager to ask what format to use.",
        lambda a, b: f"The museum's free day shifted from Sunday to Wednesday. {a} came on Sunday expecting free entry. {b} arrived on Thursday and asked if there was a discount.",
        lambda a, b: f"The company switched from Slack to Teams overnight. {a} typed messages into Slack all morning wondering why no one responded. {b} stared at the new Teams icon, clicking it tentatively.",
        lambda a, b: f"The station added a second platform last month. {a} went to platform one as always, but the train now left from platform two. {b} looked at the departure boards, unsure which platform to go to.",
        lambda a, b: f"The office dress code relaxed to business casual last week. {a} wore the usual formal suit and tie. {b} looked around at what everyone else was wearing, unsure what to put on.",
        lambda a, b: f"The pharmacy moved from the ground floor to the mezzanine. {a} walked to the old location and stood in front of a shuttered counter. {b} asked the security guard where to find the pharmacy.",
        lambda a, b: f"The seminar was cancelled but the room booking wasn't updated. {a} showed up with notes ready and sat in the empty room. {b} walked past the room, peeked in, and kept going looking for signs.",
        lambda a, b: f"The parking meter now only takes cards, not coins. {a} fed quarters into the slot and wondered why the screen stayed blank. {b} tapped the machine, looked for instructions, then asked a nearby driver.",
        lambda a, b: f"The annual charity run changed its route this year. {a} ran the old course and ended up at a dead end. {b} stood at the starting line asking other runners which direction to go.",
        lambda a, b: f"The building's recycling bins were moved to the loading dock. {a} carried sorted recycling to the old spot in the hallway and found nothing. {b} held the bag and asked the janitor where to put it.",
        lambda a, b: f"The conference keynote was moved from the ballroom to the auditorium. {a} found a seat in the ballroom and waited as it stayed empty. {b} checked the conference app, refreshed twice, and still wasn't sure where to go.",
        lambda a, b: f"The neighborhood grocery started closing at nine instead of eleven. {a} headed out at ten expecting to shop. {b} arrived at nine-thirty, found the door locked, and asked a neighbor if the store was still in business.",
        lambda a, b: f"The tennis court booking switched to an online system. {a} showed up at seven like always and found someone else playing. {b} stood at the gate looking for a sign-up sheet that wasn't there.",
        lambda a, b: f"The doctor's office changed from walk-in to appointment-only. {a} arrived and took a number, sitting in the waiting room. {b} walked in and asked the receptionist if they needed to come back later.",
        lambda a, b: f"The ferry terminal changed the boarding gate from B to D. {a} lined up at gate B as always. {b} read the overhead signs, walked back and forth between gates, and finally asked a staff member.",
        lambda a, b: f"The building's elevator was converted to key-card access. {a} pressed the button repeatedly and wondered why nothing happened. {b} stood in front of the panel looking for instructions.",
    ],
    ("desire", "intention"): [
        lambda a, b: f"{a} pressed against the window of the guitar shop, staring at the sunburst acoustic for the fourth week. {b} walked in, put three hundred dollars on the counter, and asked to try it.",
        lambda a, b: f"{a} scrolled through photos of the cottage every evening, saving each listing to a growing folder. {b} booked a viewing for Saturday, arranged time off, and mapped the drive.",
        lambda a, b: f"{a} watched the open-water swimmers from the pier each morning, never stepping off the dock. {b} signed up for the coached session, bought goggles, and reserved a locker.",
        lambda a, b: f"{a} lingered in front of the pastry case, counting coins quietly. {b} handed the cashier a bill and pointed to the chocolate éclair.",
        lambda a, b: f"{a} re-read the art school brochure for the fifth time, running a finger along the course descriptions. {b} filled out the application, attached a portfolio, and mailed it.",
        lambda a, b: f"{a} stood by the adoption board, reading the same dog's profile over and over. {b} filled out the paperwork, paid the fee, and scheduled a home visit.",
        lambda a, b: f"{a} bookmarked the mountain-bike trail guide and checked the weather for six weekends. {b} loaded the bike onto the car, packed lunch, and left before dawn.",
        lambda a, b: f"{a} sat in the bleachers during every rehearsal, mouthing the lyrics from the audience. {b} signed up for auditions, practiced the solo, and printed the sheet music.",
        lambda a, b: f"{a} traced the outline of the country on the globe and sighed each time a travel show mentioned it. {b} renewed the passport, purchased the guidebook, and started budgeting for airfare.",
        lambda a, b: f"{a} stared at the empty lot next door, sketching imaginary garden layouts in a notebook. {b} hired a landscaper, ordered topsoil, and rented a rototiller.",
        lambda a, b: f"{a} kept returning to the jewelry store, asking to see the engagement ring under better light. {b} confirmed the ring size, placed a deposit, and arranged pickup for Friday.",
        lambda a, b: f"{a} added the marathon to the calendar year after year and never laced up. {b} started a twelve-week training program and bought new shoes.",
        lambda a, b: f"{a} flipped through the real estate listings every Sunday, circling the same lakeside cabin. {b} called the agent, scheduled three tours, and pre-approved a mortgage.",
        lambda a, b: f"{a} watched cooking shows every night, bookmarking recipes but never shopping for ingredients. {b} enrolled in culinary school, ordered a knife set, and arranged the kitchen.",
        lambda a, b: f"{a} gazed at the telescope in the catalog, measuring the shelf to see if it would fit. {b} placed the order, cleared the balcony, and downloaded a star-charting app.",
        lambda a, b: f"{a} pressed replay on the piano concerto recording every morning, eyes closed and fingers tapping. {b} rented a practice room, hired an instructor, and committed to daily sessions.",
        lambda a, b: f"{a} kept a folder of scuba diving photos on the desktop, reorganizing them by location. {b} booked the certification course, bought a wetsuit, and scheduled the pool sessions.",
        lambda a, b: f"{a} paused at the antique shop window each morning, staring at the brass compass on display. {b} went inside, negotiated a price, and left with the compass wrapped in cloth.",
        lambda a, b: f"{a} saved screenshots of tiny houses and pinned them to a board above the desk. {b} purchased a plot of land, hired a builder, and filed the permit application.",
        lambda a, b: f"{a} attended every open mic night at the café, sitting in the back row and clapping the loudest. {b} signed up for a slot, rehearsed a five-minute set, and brought a guitar.",
        lambda a, b: f"{a} dog-eared the same page of the pottery catalog every time, running a thumb over the kiln photo. {b} registered for the ceramics workshop, bought clay, and rented studio time.",
        lambda a, b: f"{a} stood at the harbor watching sailboats come and go, the wind pulling at a loosely held scarf. {b} enrolled in sailing lessons, passed the safety exam, and reserved a boat for the weekend.",
        lambda a, b: f"{a} read the same gardening blog every evening, zooming in on photos of Japanese maples. {b} ordered three saplings, dug holes in the backyard, and mixed compost into the soil.",
        lambda a, b: f"{a} kept a clipping of the vintage motorcycle ad tucked inside a notebook. {b} contacted the seller, test-rode the bike, and transferred the money.",
    ],
    ("knowledge", "ignorance"): [
        lambda a, b: f"{a} walked straight to the fuse box, flipped the third switch, and the lights came back on. {b} fumbled along the dark hallway, trying every light switch twice.",
        lambda a, b: f"{a} quoted the building code from memory and told the inspector exactly which section applied. {b} flipped through a thick manual, searching for the right page.",
        lambda a, b: f"At the wine tasting, {a} identified the grape, vintage, and region after one sip. {b} swirled the glass and guessed it was 'some kind of red.'",
        lambda a, b: f"{a} entered the server credentials and restarted the service in under a minute. {b} called the help desk and waited on hold for twenty minutes.",
        lambda a, b: f"The patient described the symptoms. {a} named the condition immediately. {b} opened the diagnostic manual and began cross-referencing.",
        lambda a, b: f"{a} navigated the old quarter without a map, pointing out landmarks. {b} stopped at every intersection to consult the phone.",
        lambda a, b: f"{a} wired the circuit correctly on the first attempt and tested it. {b} connected the wires wrong and blew a fuse.",
        lambda a, b: f"On the steep hike, {a} used the zigzag technique to maintain traction. {b} slipped twice on the gravel and sat down to rest.",
        lambda a, b: f"{a} disarmed the alarm by pressing the four-digit code within seconds. {b} tried a birthday and triggered the siren.",
        lambda a, b: f"At customs, {a} produced the completed form and receipts. {b} rummaged through crumpled papers asking which form to fill out.",
        lambda a, b: f"{a} set up the telescope, aligned it to Polaris, and tracked Jupiter. {b} squinted through the eyepiece at a blur and asked which direction to point.",
        lambda a, b: f"{a} identified the mushroom species on sight and picked only safe ones. {b} picked everything edible-looking and asked to have them checked.",
        lambda a, b: f"{a} recited the bus schedule from memory and arrived at the platform thirty seconds before departure. {b} stood at the wrong stop for fifteen minutes then asked a stranger.",
        lambda a, b: f"{a} adjusted the darkroom enlarger to the exact f-stop and exposure time for a perfect print. {b} guessed at the settings and produced a washed-out image.",
        lambda a, b: f"{a} tuned the guitar by ear in under a minute. {b} downloaded three different tuning apps and still couldn't get the low E right.",
        lambda a, b: f"{a} assembled the flat-pack bookshelf without looking at the instructions. {b} laid out all the pieces, stared at the diagram, and attached the back panel upside down.",
        lambda a, b: f"In the kitchen, {a} seasoned the dish from memory and nailed the balance. {b} measured every spice with shaking hands and added too much salt.",
        lambda a, b: f"{a} filled out the visa application from memory, listing every required document. {b} called the embassy three times to clarify which forms to submit.",
        lambda a, b: f"{a} named every constellation visible and their brightest stars. {b} pointed vaguely at the sky and called everything the Big Dipper.",
        lambda a, b: f"{a} drove to the mechanic and described the exact part needed by catalog number. {b} called and said the car was 'making a weird noise' and couldn't describe it further.",
        lambda a, b: f"{a} typed the terminal commands from memory and deployed in one pass. {b} copied commands from a tutorial, hit two errors, and asked for help in the group chat.",
        lambda a, b: f"{a} opened the sewing machine, rethreaded the bobbin correctly, and resumed sewing. {b} stared at the tangle of thread, tried pulling it, and made it worse.",
        lambda a, b: f"{a} read the tide chart and launched the kayak at the optimal window. {b} dragged the kayak out at random and struggled against the current.",
        lambda a, b: f"{a} explained the rental contract clause by clause to the new tenant. {b} signed without reading and later asked what the deposit terms meant.",
    ],
    ("belief", "desire"): [
        lambda a, b: f"The vintage bookshop was closed for renovation. {a} pulled the door handle, stepping back in surprise. {b} stood across the street, gazing at the window display and pressing fingertips to the glass.",
        lambda a, b: f"The promotion results were posted. {a} scanned the list, nodded at a name, and went back to work. {b} read the list slowly, lingering on every name, shoulders dropping at the end.",
        lambda a, b: f"A rare comet was visible that night. {a} set up the telescope at the published coordinates. {b} sat on the rooftop with a blanket, gazing up, searching for something bright.",
        lambda a, b: f"The last ferry had already left at five. {a} arrived at five-fifteen and walked toward the empty dock. {b} stood at the railing overlooking the harbor, watching boats come and go until dark.",
        lambda a, b: f"The hiring committee already made their choice. {a} wrote a congratulations email to the selected candidate. {b} refreshed the portal every few hours, reading the job description one more time.",
        lambda a, b: f"The gallery exhibit ended last week. {a} drove there and found the 'Exhibit Closed' sign. {b} kept the catalog on the nightstand, flipping through pages each evening.",
        lambda a, b: f"The match ended three to one. {a} told everyone the home team won easily. {b} replayed highlights on the phone, pausing on every near-miss by the losing side.",
        lambda a, b: f"The scholarship winners were announced. {a} checked the list once and closed the browser. {b} copied the announcement into a document, underlining each criterion.",
        lambda a, b: f"The mountain pass was snowed in. {a} packed chains and drove to the base, planning to cross. {b} pinned a photo of the pass above the desk and traced the road each morning.",
        lambda a, b: f"The antique clock at the estate sale was sold before noon. {a} asked the organizer and was directed to the buyer. {b} circled the empty display table twice, picking up the description card.",
        lambda a, b: f"The garden plot rental was full for the season. {a} submitted a late application and got the waitlist notice. {b} walked past the community garden every day, stopping to watch others tend plots.",
        lambda a, b: f"The limited-edition print sold out in minutes. {a} refreshed the page and saw 'Sold Out,' then closed the laptop. {b} saved the preview image as a wallpaper and checked resale sites daily.",
        lambda a, b: f"The cooking class was cancelled due to low enrollment. {a} showed up at the kitchen and found the note on the door. {b} kept the recipe list from the syllabus pinned to the fridge.",
        lambda a, b: f"The old cinema was being demolished. {a} drove past and saw the wrecking crew already at work. {b} collected ticket stubs from past showings and arranged them in a scrapbook.",
        lambda a, b: f"The vintage car auction ended yesterday. {a} called the auction house and was told the lot was sold. {b} kept the catalog open on the desk, circling the same blue convertible.",
        lambda a, b: f"The community band stopped accepting new members. {a} emailed the director and received a polite rejection. {b} practiced scales alone every evening, leaving the window open.",
        lambda a, b: f"The flight to Lisbon was fully booked. {a} called the airline and was told there were no seats left. {b} kept the travel guide on the bedside table, reading a new chapter each night.",
        lambda a, b: f"The workshop registration closed early. {a} submitted the form and got a 'deadline passed' error. {b} rewatched the instructor's old videos, taking notes in the margin.",
        lambda a, b: f"The rare orchid was sold to another buyer. {a} asked the nursery owner and was given the buyer's name. {b} visited the greenhouse weekly, photographing similar varieties.",
        lambda a, b: f"The apprenticeship position was filled internally. {a} received the rejection letter and filed it away. {b} walked past the workshop every morning, slowing down at the open door.",
        lambda a, b: f"The lakeside cabin was already rented for the whole summer. {a} called the owner and was given dates for autumn instead. {b} downloaded photos of the cabin and set one as a phone background.",
        lambda a, b: f"The art residency only accepted ten artists this year. {a} read the acceptance list, noted who got in, and moved on. {b} kept the application prompt taped to the studio wall, rereading it.",
        lambda a, b: f"The telescope model was discontinued by the manufacturer. {a} searched online and confirmed it was no longer available. {b} saved the product page as a PDF and emailed it to a friend with the subject line 'someday.'",
        lambda a, b: f"The used bookstore closed permanently last Sunday. {a} arrived Monday and read the sign on the door. {b} still walked the same route past the storefront, glancing at the dark window.",
    ],
}


def generate_condition_c(rng, n_per_pair=30):
    """Generate counter-balanced multi-agent stories."""
    stimuli = []
    for (ms_a, ms_b), templates in PAIR_TEMPLATES.items():
        for i in range(n_per_pair):
            template_fn = templates[i % len(templates)]
            names = rng.sample(NAME_POOL, 2)
            # Counter-balance: 50% of the time, swap which agent gets which state
            swap = rng.random() < 0.5
            if swap:
                text = template_fn(names[1], names[0])
                stimuli.append({
                    "text": text,
                    "pair_type": f"{ms_a}+{ms_b}",
                    "agent1": names[1], "agent1_state": ms_a,
                    "agent2": names[0], "agent2_state": ms_b,
                    "condition": "C", "swapped": True,
                })
            else:
                text = template_fn(names[0], names[1])
                stimuli.append({
                    "text": text,
                    "pair_type": f"{ms_a}+{ms_b}",
                    "agent1": names[0], "agent1_state": ms_a,
                    "agent2": names[1], "agent2_state": ms_b,
                    "condition": "C", "swapped": False,
                })
    rng.shuffle(stimuli)
    return stimuli


# ============================================================
# Activation extraction
# ============================================================
def extract_agent_activations(model, tokenizer, stories, device, layer_idx,
                              max_length=256):
    """Extract activations at each agent's LAST mention."""
    results = []
    for story in stories:
        text = story["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[layer_idx][0]
        token_ids = inputs["input_ids"][0].tolist()

        for agent_key, state_key in [("agent1", "agent1_state"),
                                     ("agent2", "agent2_state")]:
            agent_name = story[agent_key]
            agent_tokens = tokenizer.encode(agent_name, add_special_tokens=False)
            last_pos = -1
            for pos in range(len(token_ids) - len(agent_tokens) + 1):
                if token_ids[pos:pos + len(agent_tokens)] == agent_tokens:
                    last_pos = pos
            if last_pos >= 0:
                span = hs[last_pos:last_pos + len(agent_tokens)]
                act = span.float().mean(dim=0).cpu().numpy()
            else:
                act = hs.float().mean(dim=0).cpu().numpy()
            results.append({
                "activation": act,
                "mental_state": story[state_key],
                "agent": story[agent_key],
                "pair_type": story["pair_type"],
                "swapped": story["swapped"],
            })
        del inputs, out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def extract_sentence_activations(model, tokenizer, stories, device, layer_idx,
                                 max_length=256, batch_size=8):
    """Extract mean-pooled activations for entire stories (not agent-specific)."""
    all_acts = []
    texts = [s["text"] for s in stories]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states[layer_idx]
        for b in range(len(batch)):
            mask = inputs["attention_mask"][b].bool()
            act = hs[b][mask].float().mean(dim=0).cpu().numpy()
            all_acts.append(act)
        del inputs, out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.array(all_acts)


# ============================================================
# Probing: linear + MLP
# ============================================================
def run_probe(X, y_labels, probe_type="linear", n_splits=5):
    """Run classification probe. Returns (accuracy, std)."""
    ms_types = sorted(set(y_labels))
    y = np.array([ms_types.index(l) for l in y_labels])
    if probe_type == "linear":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
        ])
    else:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128), max_iter=1000,
                early_stopping=True, validation_fraction=0.15,
                random_state=42, learning_rate_init=0.001,
            )),
        ])
    n_cv = min(n_splits, min(np.bincount(y)))
    if n_cv < 2:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def run_pairwise_probe(X, y_labels, probe_type="linear", n_splits=5):
    """Binary probe for each pair of mental states."""
    ms_types = sorted(set(y_labels))
    results = []
    for ms_a, ms_b in combinations(ms_types, 2):
        idx_a = [i for i, l in enumerate(y_labels) if l == ms_a]
        idx_b = [i for i, l in enumerate(y_labels) if l == ms_b]
        Xp = np.vstack([X[idx_a], X[idx_b]])
        yp = np.array([0] * len(idx_a) + [1] * len(idx_b))
        if probe_type == "linear":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
            ])
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=1000,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=42,
                )),
            ])
        n_cv = min(n_splits, min(len(idx_a), len(idx_b)))
        if n_cv < 2:
            results.append({"ms_a": ms_a, "ms_b": ms_b, "accuracy": float("nan")})
            continue
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, Xp, yp, cv=skf, scoring="accuracy")
        results.append({
            "ms_a": ms_a, "ms_b": ms_b,
            "accuracy": float(scores.mean()), "std": float(scores.std()),
        })
    return results


def probe_per_pair_type(agent_results, probe_type="linear"):
    """For each pair type in condition C, run binary probe."""
    pair_types = sorted(set(r["pair_type"] for r in agent_results))
    results = []
    for pt in pair_types:
        items = [r for r in agent_results if r["pair_type"] == pt]
        X = np.array([r["activation"] for r in items])
        y_labels = [r["mental_state"] for r in items]
        ms_types = sorted(set(y_labels))
        if len(ms_types) < 2:
            results.append({"pair_type": pt, "accuracy": float("nan"),
                            "probe": probe_type})
            continue
        y = np.array([ms_types.index(l) for l in y_labels])
        if probe_type == "linear":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
            ])
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=1000,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=42,
                )),
            ])
        n_cv = min(5, min(np.bincount(y)))
        if n_cv < 2:
            results.append({"pair_type": pt, "accuracy": float("nan"),
                            "probe": probe_type})
            continue
        skf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
        acc = float(scores.mean())
        results.append({
            "pair_type": pt, "accuracy": acc, "std": float(scores.std()),
            "n_samples": len(items), "probe": probe_type,
        })
        # Also check swapped vs non-swapped
        swapped_items = [r for r in items if r["swapped"]]
        non_swapped = [r for r in items if not r["swapped"]]
        results[-1]["n_swapped"] = len(swapped_items)
        results[-1]["n_non_swapped"] = len(non_swapped)
    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Agent-state binding probe v3 — artifact controls")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_short", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_per_pair", type=int, default=30)
    parser.add_argument("--layer_stride", type=int, default=4,
                        help="Sweep every N-th layer")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate stimuli
    print("=" * 70)
    print("GENERATING STIMULI")
    print("=" * 70)
    stim_c = generate_condition_c(rng, n_per_pair=args.n_per_pair)
    n_stories = len(stim_c)
    n_swapped = sum(1 for s in stim_c if s["swapped"])
    print(f"Condition C: {n_stories} stories ({n_stories * 2} agent probes)")
    print(f"  Counter-balanced: {n_swapped} swapped, {n_stories - n_swapped} original")

    for pt in sorted(set(s["pair_type"] for s in stim_c)):
        ct = sum(1 for s in stim_c if s["pair_type"] == pt)
        sw = sum(1 for s in stim_c if s["pair_type"] == pt and s["swapped"])
        print(f"  {pt}: {ct} stories ({sw} swapped)")

    # Load model
    print(f"\n{'=' * 70}")
    print(f"LOADING MODEL: {args.model_path}")
    print(f"{'=' * 70}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    # Layer sweep
    layers_to_test = list(range(0, n_layers + 1, args.layer_stride))
    if n_layers not in layers_to_test:
        layers_to_test.append(n_layers)
    print(f"Testing layers: {layers_to_test}")

    all_layer_results = {}

    for layer_idx in layers_to_test:
        print(f"\n{'=' * 70}")
        print(f"LAYER {layer_idx} / {n_layers}")
        print(f"{'=' * 70}")

        # Extract agent-level activations
        print("  Extracting agent-level activations...")
        agent_results = extract_agent_activations(
            model, tokenizer, stim_c, device, layer_idx)
        acts = np.array([r["activation"] for r in agent_results])
        labels = [r["mental_state"] for r in agent_results]
        print(f"  Shape: {acts.shape}")

        layer_result = {"layer": layer_idx}

        # Linear probe: per pair type
        print("  Linear probe (per pair)...")
        linear_pairs = probe_per_pair_type(agent_results, "linear")
        for p in linear_pairs:
            print(f"    {p['pair_type']:25s} linear={p['accuracy']:.3f}")
        layer_result["linear_per_pair"] = linear_pairs

        # MLP probe: per pair type
        print("  MLP probe (per pair)...")
        mlp_pairs = probe_per_pair_type(agent_results, "mlp")
        for p in mlp_pairs:
            print(f"    {p['pair_type']:25s} MLP={p['accuracy']:.3f}")
        layer_result["mlp_per_pair"] = mlp_pairs

        # Overall 5-way probes
        lin_5way, lin_5std = run_probe(acts, labels, "linear")
        mlp_5way, mlp_5std = run_probe(acts, labels, "mlp")
        print(f"  Overall 5-way: linear={lin_5way:.3f}, MLP={mlp_5way:.3f}")
        layer_result["linear_5way"] = lin_5way
        layer_result["linear_5way_std"] = lin_5std
        layer_result["mlp_5way"] = mlp_5way
        layer_result["mlp_5way_std"] = mlp_5std

        all_layer_results[layer_idx] = layer_result

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: LINEAR vs MLP across layers")
    print(f"{'=' * 70}")
    for layer_idx in layers_to_test:
        r = all_layer_results[layer_idx]
        print(f"\nLayer {layer_idx}:")
        lin_pairs = {p["pair_type"]: p["accuracy"] for p in r["linear_per_pair"]}
        mlp_pairs_d = {p["pair_type"]: p["accuracy"] for p in r["mlp_per_pair"]}
        for pt in sorted(lin_pairs.keys()):
            la = lin_pairs[pt]
            ma = mlp_pairs_d.get(pt, float("nan"))
            delta = ma - la if not (np.isnan(ma) or np.isnan(la)) else float("nan")
            marker = ""
            if not np.isnan(delta) and delta > 0.1:
                marker = " *** MLP > LINEAR"
            print(f"  {pt:25s}  lin={la:.3f}  mlp={ma:.3f}  delta={delta:+.3f}{marker}")

    # Find best layer for each pair
    print(f"\n{'=' * 70}")
    print("BEST LAYER PER PAIR (MLP)")
    print(f"{'=' * 70}")
    pair_types = sorted(set(r["pair_type"] for r in agent_results))
    for pt in pair_types:
        best_layer = -1
        best_acc = -1
        for layer_idx in layers_to_test:
            for p in all_layer_results[layer_idx]["mlp_per_pair"]:
                if p["pair_type"] == pt and p["accuracy"] > best_acc:
                    best_acc = p["accuracy"]
                    best_layer = layer_idx
        print(f"  {pt:25s}  best_layer={best_layer}  mlp_acc={best_acc:.3f}")

    # Save
    output = {
        "model": args.model_short,
        "n_layers": n_layers,
        "n_per_pair": args.n_per_pair,
        "layers_tested": layers_to_test,
        "layer_results": {str(k): v for k, v in all_layer_results.items()},
    }
    out_path = out_dir / f"{args.model_short}_agent_state_v3.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    stim_path = out_dir / f"{args.model_short}_agent_state_v3_stimuli.json"
    stim_export = [{k: v for k, v in s.items()} for s in stim_c]
    with open(stim_path, "w") as f:
        json.dump(stim_export, f, indent=2)
    print(f"Saved stimuli to {stim_path}")


if __name__ == "__main__":
    main()
