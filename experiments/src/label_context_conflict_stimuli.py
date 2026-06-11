#!/usr/bin/env python3
"""
Label-Context Conflict Stimuli for emotion representation causal test.

Design: 5 emotions x 3 conditions x 10 items = 150 stimuli.

Key finding context: within-affective RSA is rho=-0.11 under short stimuli but
rho=+0.81 under template-matched (richer) stimuli, suggesting emotion
representations are input-driven. This experiment tests the CAUSAL direction:
when emotion label and situational context CONFLICT, does the model follow the
label or the context?

5 emotions: anger, sadness, fear, disgust, happiness

3 conditions:
  label_only:   Explicit emotion label + generic scenario (label determines emotion)
  context_only: Rich diagnostic appraisal situation, ZERO emotion words
  conflict:     Emotion label says X, context describes Y's diagnostic appraisal

Conflict pairings (fixed):
  anger label   + sadness context  (irreversible loss + claim of anger)
  sadness label + anger context    (injustice/betrayal + claim of sadness)
  fear label    + happiness context (achievement/safety + claim of fear)
  disgust label + fear context     (threat/danger + claim of disgust)
  happiness label + disgust context (contamination/violation + claim of happiness)

Output: data/cognitive_stimuli/rsa/label_context_conflict_stimuli.jsonl

Usage:
  python src/label_context_conflict_stimuli.py
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parents[1]
OUT_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "label_context_conflict_stimuli.jsonl"

# ═══════════════════════════════════════════════════════════════════════
# Emotion label phrases (used in label_only and conflict conditions)
# ═══════════════════════════════════════════════════════════════════════

EMOTION_PHRASES = {
    "anger":     "absolutely furious and full of rage",
    "sadness":   "deeply heartbroken and full of grief",
    "fear":      "terrified and filled with dread",
    "disgust":   "repulsed and completely sickened",
    "happiness": "overjoyed and filled with pure delight",
}

# ═══════════════════════════════════════════════════════════════════════
# LABEL_ONLY: Generic scenarios + explicit emotion label
# Rule: scenario is generic (could fit multiple emotions); label determines it
# Format: "[Scenario]. I feel [emotion_phrase]."
# ═══════════════════════════════════════════════════════════════════════

LABEL_ONLY_SCENARIOS = {
    "anger": [
        "My team lead reassigned my project today without any discussion.",
        "A coworker brought up my past mistake during the department meeting.",
        "My neighbor left a note about my parking again this morning.",
        "The company changed the vacation policy right before my planned trip.",
        "My roommate used my kitchen supplies and left everything out.",
        "A client sent a long email questioning every detail of my work.",
        "My manager scheduled a mandatory meeting during my lunch break.",
        "Someone at the gym took the equipment I had been waiting for.",
        "My landlord raised the rent with only a week's notice.",
        "A friend showed up two hours late to our planned dinner.",
    ],
    "sadness": [
        "I got a call from my hometown this morning with some news.",
        "My old college friend reached out after years of silence.",
        "I found a box of letters in the back of my closet today.",
        "My manager pulled me aside for a private conversation after work.",
        "I drove past my old neighborhood on the way home from work.",
        "A song came on the radio that I hadn't heard in years.",
        "My coworker mentioned she's moving to another city next month.",
        "I received a package addressed to someone who used to live here.",
        "My weekend plans fell through at the last minute.",
        "I noticed the coffee shop where I used to study has closed down.",
    ],
    "fear": [
        "My phone rang at three in the morning from an unknown number.",
        "My boss asked me to come to his office first thing tomorrow.",
        "I noticed an unfamiliar car parked outside my house all evening.",
        "The weather forecast changed dramatically for my travel day.",
        "My doctor's office called and asked me to come in for a follow-up.",
        "I heard a loud noise downstairs while I was reading in bed.",
        "My bank sent me an alert about unusual account activity.",
        "A message from HR appeared in my inbox marked as urgent.",
        "The pilot made an unscheduled announcement during the flight.",
        "My building superintendent knocked on my door late at night.",
    ],
    "disgust": [
        "I opened the container my coworker left in the shared fridge.",
        "Someone on the bus sat down right next to me during rush hour.",
        "I noticed something on the restaurant table after sitting down.",
        "My neighbor's yard had changed a lot since I last looked.",
        "I received a forwarded message from an acquaintance I barely know.",
        "The public restroom at the train station was the only option available.",
        "A street vendor offered me a sample of something I couldn't identify.",
        "My coworker showed me a video on their phone during break.",
        "I found something unexpected at the bottom of my bag.",
        "The hotel room I checked into had a particular smell to it.",
    ],
    "happiness": [
        "I got an email notification from my application portal this morning.",
        "My friend sent me a text saying she had something to tell me.",
        "I checked my mailbox and found an envelope with my name on it.",
        "My manager called a team meeting to make an announcement.",
        "I logged into my account and saw the latest status update.",
        "A colleague stopped by my desk with news about the project review.",
        "I received a voicemail from someone I hadn't spoken to in months.",
        "My partner suggested we go somewhere special this weekend.",
        "The results from last week's evaluation came back today.",
        "I got a notification that something I'd been waiting for had arrived.",
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# CONTEXT_ONLY: Rich diagnostic appraisal situations, ZERO emotion words
# Rule: no emotion vocabulary at all (not even subtle: upset, troubled,
#       thrilled, uncomfortable, etc.). Only situational descriptions/actions.
# Each context must be clearly diagnostic of ONE specific emotion.
# ═══════════════════════════════════════════════════════════════════════

CONTEXT_ONLY = {
    # ANGER: injustice, blocked goals, other-blame
    "anger": [
        "My colleague presented my research findings as their own at the board meeting. When I tried to speak up, my manager said it was a team effort and moved on. I spent six months on that analysis.",
        "The landlord raised our rent by forty percent and posted an eviction notice the same day we filed a maintenance complaint. The broken heater has been reported three times.",
        "My supervisor promoted a less experienced hire who started two months ago. I trained that person and have been here for four years with no advancement.",
        "Someone keyed my car in the parking lot and left a note saying I deserved it for taking their spot. The spot has no assigned name on it.",
        "I discovered my business partner had been redirecting client payments to a personal account. When I confronted them, they denied everything despite the bank records.",
        "My professor gave me a failing grade on the paper and admitted he hadn't read past the first page. He told me to resubmit with no extension on the deadline.",
        "My insurance company denied my claim for the third time, citing a clause they never mentioned when I signed up. The representative hung up when I asked for a supervisor.",
        "A driver cut me off and then brake-checked me, causing me to swerve into the next lane. When I honked, they made an obscene gesture and drove away.",
        "My coworker took the last slot for the training program I needed for my certification. She admitted she only signed up because she wanted a day away from the office.",
        "The contractor we hired demolished the wrong wall in our renovation and refused to take responsibility. He said our instructions were unclear, but we have the signed plan.",
    ],
    # SADNESS: irreversible loss, helplessness
    "sadness": [
        "I received a call from the hospital that my grandfather passed away during the night. I had been planning to visit him next weekend.",
        "My daughter packed the last of her boxes and drove away to start her new life across the country. The house has never been this quiet.",
        "The veterinarian told me there was nothing more they could do. I sat in the waiting room holding the leash with the collar still warm.",
        "I found out that the scholarship program I had been working toward for three years was permanently discontinued due to budget cuts.",
        "My closest friend from childhood sent a short message saying she no longer wanted to be in contact. No explanation, no argument beforehand.",
        "The store where my late mother used to take me every Saturday was demolished to make room for a parking garage. I drove past the empty lot.",
        "My partner told me over dinner that the relationship had run its course. They had already arranged to move out by the end of the week.",
        "I opened a drawer and found a birthday card my father had written before he passed. It was dated two years ago and never given to me.",
        "The community garden where I spent every summer was sold to a developer. All the plots were cleared overnight without any notice to the members.",
        "My mentor at work retired and on his last day told me he had been diagnosed with a terminal condition. He said he wanted to spend his remaining time at home.",
    ],
    # FEAR: future threat, uncertainty, lack of control
    "fear": [
        "The doctor looked at the scan results for a long time before asking me to sit down. She said she wanted to run more tests before drawing any conclusions.",
        "I woke up at two in the morning to the sound of someone trying the handle of my front door. The porch light had been unscrewed.",
        "My company announced a round of layoffs affecting thirty percent of staff. My department was specifically mentioned, and individual meetings start tomorrow.",
        "A car followed me for eight blocks after I left the grocery store. Every turn I made, the same headlights appeared in the mirror.",
        "The turbulence on the flight became so severe that the overhead bins opened and bags fell into the aisle. The captain came on but said nothing for several minutes.",
        "I received a letter from the IRS stating that my returns from the past three years were being audited. The deadline for document submission is in five days.",
        "Walking home, I realized the street was completely deserted and the lights on the block had gone out. I heard footsteps behind me matching my pace.",
        "My child's school called to say there was a lockdown situation in progress. They could not provide details and told me to stay away from the building.",
        "The earthquake stopped and then started again with a deeper vibration. The ceiling cracked above the doorframe where I was standing.",
        "A message appeared on my work computer saying all files had been encrypted and to contact an unknown address. My passwords had been changed.",
    ],
    # DISGUST: contamination, boundary violation, moral violation
    "disgust": [
        "I opened the lunch container that had been left in the office fridge for three weeks. The contents had turned into a dark liquid with white spots covering the surface.",
        "The bathroom drain backed up and raw sewage began pooling across the tile floor. I was barefoot and stepped directly into it before I could react.",
        "I bit into a piece of fruit and noticed the other half contained a cluster of larvae. Several had already broken through the skin of the fruit.",
        "My landlord installed a camera in the hallway pointed directly at my front door without telling any tenant. He admitted to reviewing the footage regularly.",
        "The restaurant kitchen was visible from our table. I watched the cook handle raw chicken and then immediately prepare my salad with the same unwashed hands.",
        "I found out that a manager at our company had been accessing employees' personal medical records and sharing them with department heads during performance reviews.",
        "Someone had smeared a substance across the seat of the public bus before I sat down. I only realized it when I stood up and my clothing stuck to the surface.",
        "I discovered that the supplement company I trusted had been filling capsules with sawdust and industrial fillers instead of the listed ingredients.",
        "The apartment I moved into had a cockroach infestation behind the walls. At night, they crawled across the kitchen counter and into any container left open.",
        "A company executive was caught using charity funds meant for children's hospitals to finance personal vacations. He had been doing this for over a decade.",
    ],
    # HAPPINESS: goal achievement, acceptance, safety
    "happiness": [
        "I opened the email and read that my application for the research fellowship had been accepted. The selection committee noted my proposal was among the top three out of four hundred.",
        "After months of rehabilitation following the accident, I stood up from the wheelchair and took twelve steps across the room on my own.",
        "My daughter ran across the stage at her graduation ceremony and pointed at me in the audience. She mouthed the words thank you.",
        "The test results came back and the doctor confirmed the treatment had worked. The condition that had defined the last two years of my life was in full remission.",
        "I walked into the surprise party that my friends had spent weeks organizing. Every person who mattered to me was in that room, including people who had flown in from other cities.",
        "After three years of night classes and weekend studying, I received the certification that qualified me for the position I had been working toward.",
        "My estranged brother showed up at my door and said he wanted to rebuild what we had lost. We talked until sunrise for the first time in seven years.",
        "The arbitration ruling came back entirely in my favor after eighteen months of proceedings. The organization publicly acknowledged the error and issued a formal correction.",
        "I submitted my manuscript expecting another round of revisions, but the editor responded that it had been accepted for publication without further changes.",
        "The adoption agency called to say the process was finalized. My partner and I drove to the center and held our child for the first time as a family.",
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# CONFLICT: Label says X, context describes Y's diagnostic appraisal
#
# Pairings (label_emotion -> context_emotion):
#   anger    -> sadness  (irreversible loss + claim of anger)
#   sadness  -> anger    (injustice/betrayal + claim of sadness)
#   fear     -> happiness (achievement/safety + claim of fear)
#   disgust  -> fear     (threat/danger + claim of disgust)
#   happiness -> disgust  (contamination/violation + claim of happiness)
# ═══════════════════════════════════════════════════════════════════════

CONFLICT_PAIRINGS = {
    "anger":     "sadness",    # context is sadness-diagnostic
    "sadness":   "anger",      # context is anger-diagnostic
    "fear":      "happiness",  # context is happiness-diagnostic
    "disgust":   "fear",       # context is fear-diagnostic
    "happiness": "disgust",    # context is disgust-diagnostic
}

# Context for conflict condition: 10 items per label emotion.
# The context must clearly describe the PARTNER emotion's diagnostic appraisal.

CONFLICT_CONTEXTS = {
    # label=anger, context=sadness (irreversible loss situations)
    "anger": [
        "My grandmother's house, where I spent every summer as a child, was torn down last week. The lot is empty now and there is nothing left of what it once was.",
        "After fifteen years together, my closest companion at work retired and moved overseas. His desk was cleared by the time I arrived on Monday.",
        "The community library where I learned to read was closed permanently due to funding cuts. The building will be converted into commercial offices.",
        "I got a letter that the rescue shelter where I volunteered for a decade is shutting down. All remaining animals will be transferred to facilities in other counties.",
        "The garden that my late wife planted twenty years ago was destroyed overnight by a construction crew that came to the wrong address.",
        "My son told me he has accepted a job on the other side of the world and will not be coming back for at least five years.",
        "The family photograph album was ruined when the basement flooded. Every picture from my parents' wedding through my childhood is gone.",
        "My college roommate, the only person who knew me during that time, passed away last Tuesday. I found out through a social media post.",
        "The neighborhood I grew up in has been entirely rebuilt. Not a single house, tree, or storefront from my childhood remains.",
        "I learned that the teacher who changed my life in high school died before I ever got the chance to write him the letter I had been planning.",
    ],
    # label=sadness, context=anger (injustice/betrayal situations)
    "sadness": [
        "My coworker took full credit for the proposal I spent three months developing and received a bonus for it. My name was removed from every version of the document.",
        "The dealership sold me a car they knew had been in a major accident. They forged the vehicle history report and charged full price.",
        "My neighbor built a fence two feet onto my property and filed a complaint against me when I pointed it out. The city sided with him without reviewing the survey.",
        "I found out my business partner has been billing clients for work I completed and pocketing the difference. This has been going on for over a year.",
        "The airline lost my luggage and offered a thirty-dollar voucher as compensation. When I asked to speak to a supervisor, they ended the call.",
        "My landlord entered my apartment while I was at work and went through my personal files. When I confronted them, they claimed it was a routine inspection.",
        "Someone filed a complaint against me at work using fabricated evidence. The investigation proceeded without ever asking for my side of the story.",
        "My professor accused me of plagiarism on a paper I wrote entirely from scratch. He refused to look at my drafts or revision history.",
        "The contractor we hired for our kitchen overcharged us by twelve thousand dollars and disappeared before finishing the backsplash and cabinet trim.",
        "My ex spread private information about me to mutual friends after I asked them to keep our separation respectful. Several people have contacted me about details only my ex would know.",
    ],
    # label=fear, context=happiness (achievement/safety/acceptance situations)
    "fear": [
        "I received the acceptance letter from the graduate program I had applied to three times. This was the last attempt I planned to make.",
        "The specialist called and confirmed that the biopsy results were completely clear. After six months of monitoring, no further treatment is needed.",
        "My company announced that my team's project won the annual innovation award. My manager named me specifically as the driving force behind the effort.",
        "After two years of physical therapy following the accident, I ran a full mile without stopping this morning for the first time.",
        "My daughter called from across the country to tell me she got the teaching position she wanted. She sounded more confident than I have ever heard her.",
        "The bank approved our mortgage application after we were denied twice before. The house we wanted is still available and the seller accepted our offer.",
        "My estranged sister called and said she wanted to come to Thanksgiving this year. She apologized for the years of distance and asked if the invitation still stood.",
        "The jury returned a verdict fully in our favor after eight months of litigation. The other party was required to cover all costs and issue a public retraction.",
        "I opened my performance review and saw the highest rating I have received in twelve years at the company. A promotion was recommended effective next quarter.",
        "My partner planned a weekend away for our anniversary and arranged for everything, including convincing my parents to watch the children. Every detail was exactly what I would have chosen.",
    ],
    # label=disgust, context=fear (threat/danger/uncertainty situations)
    "disgust": [
        "I heard the lock on my front door being tampered with at one in the morning. Through the peephole I could see a figure crouching by the frame.",
        "The MRI results showed a mass that the radiologist could not immediately identify. I have been referred to a specialist for further evaluation next week.",
        "Walking to my car in the parking garage, I noticed the same person from the elevator standing at the end of my row. They began walking toward me.",
        "The fire alarm went off in my hotel at midnight. Smoke was visible in the corridor and the stairwell was packed with people moving in both directions.",
        "My company received an anonymous threat directed at our office building. Security evacuated the floor, and we were told to wait outside until further notice.",
        "The road ahead washed out during a flash flood while I was driving alone. Water began rising around the tires and the car started to drift sideways.",
        "I found a series of anonymous notes slipped under my apartment door over the past week. Each one contained specific details about my daily routine.",
        "The ceiling in my son's bedroom cracked open and a large section of plaster fell onto his bed. He had been sleeping there ten minutes earlier.",
        "My phone was remotely accessed and someone changed the passwords to all my accounts. I received a message from an unknown number saying they had my financial records.",
        "During a solo hike, I rounded a bend and came face to face with a large animal blocking the trail. It did not move and maintained direct eye contact.",
    ],
    # label=happiness, context=disgust (contamination/violation situations)
    "happiness": [
        "I noticed the restaurant's kitchen had grease caked on every surface and the floor was sticky with residue. The cook was handling raw ingredients and cooked plates with the same pair of gloves.",
        "The apartment I toured had a dark stain spreading across the bathroom ceiling and a persistent odor that the landlord attributed to the pipes. Mold was visible behind the toilet.",
        "A colleague forwarded me screenshots showing that our department head had been reading our private messages in the company chat and sharing excerpts with senior leadership.",
        "I learned that the daycare my children attended had been cited for repeated sanitation violations. Inspectors found spoiled food being served and unchanged diaper stations.",
        "The water in my glass at the restaurant had a film on the surface and small particles floating in it. When I brought it to the waiter's attention, he said it was the ice.",
        "I discovered that the vendor supplying our office snacks had been repacking expired products with new labels. Several coworkers had been eating them for months.",
        "The hotel room had stains on the sheets and hair on the pillow. When I pulled back the comforter, I found a used bandage stuck to the fitted sheet.",
        "A company I trusted with my data sold my personal information to marketing firms without consent. My medical history and purchase records were included in the data set.",
        "The public pool I took my children to had a cloudy green tint. A maintenance worker told me the filtration system had been broken for a week but the pool was never closed.",
        "I found out that the charity I had donated to for years was a front operation. The director used contributions to fund a personal lifestyle while none of the promised aid was delivered.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# Build all 150 stimuli
# ═══════════════════════════════════════════════════════════════════════

def build_stimuli() -> list[dict]:
    stimuli = []
    item_id = 0

    emotions = ["anger", "sadness", "fear", "disgust", "happiness"]

    # --- CONDITION 1: label_only ---
    for emo in emotions:
        scenarios = LABEL_ONLY_SCENARIOS[emo]
        assert len(scenarios) == 10, f"label_only {emo}: need 10, got {len(scenarios)}"
        phrase = EMOTION_PHRASES[emo]
        for scenario in scenarios:
            text = f"{scenario} I feel {phrase}."
            stimuli.append({
                "text": text,
                "emotion": emo,
                "condition": "label_only",
                "item_id": item_id,
                "conflict_context_emotion": None,
            })
            item_id += 1

    # --- CONDITION 2: context_only ---
    for emo in emotions:
        contexts = CONTEXT_ONLY[emo]
        assert len(contexts) == 10, f"context_only {emo}: need 10, got {len(contexts)}"
        for ctx in contexts:
            stimuli.append({
                "text": ctx,
                "emotion": emo,
                "condition": "context_only",
                "item_id": item_id,
                "conflict_context_emotion": None,
            })
            item_id += 1

    # --- CONDITION 3: conflict ---
    for label_emo in emotions:
        ctx_emo = CONFLICT_PAIRINGS[label_emo]
        contexts = CONFLICT_CONTEXTS[label_emo]
        assert len(contexts) == 10, f"conflict {label_emo}: need 10, got {len(contexts)}"
        phrase = EMOTION_PHRASES[label_emo]
        for ctx in contexts:
            text = f"{ctx} I feel {phrase}."
            stimuli.append({
                "text": text,
                "emotion": label_emo,
                "condition": "conflict",
                "item_id": item_id,
                "conflict_context_emotion": ctx_emo,
            })
            item_id += 1

    return stimuli


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════

# Words that should NOT appear in context_only stimuli
BANNED_EMOTION_WORDS = {
    "angry", "anger", "furious", "rage", "enraged", "outraged", "irate",
    "mad", "livid", "fuming", "infuriated", "wrathful",
    "sad", "sadness", "heartbroken", "grief", "grieving", "mourning",
    "devastated", "despondent", "sorrowful", "melancholy", "depressed",
    "afraid", "fear", "scared", "terrified", "frightened", "panicked",
    "anxious", "dread", "alarmed", "petrified", "horror",
    "disgusted", "disgust", "repulsed", "revolted", "nauseated", "sickened",
    "grossed", "appalled", "repelled",
    "happy", "happiness", "joyful", "delighted", "elated", "ecstatic",
    "thrilled", "overjoyed", "euphoric", "blissful", "cheerful", "glad",
    "excited", "pleased",
    # Subtle emotion words that might leak
    "upset", "troubled", "distressed", "uncomfortable", "uneasy", "worried",
    "frustrated", "annoyed", "irritated", "miserable", "hopeless", "gloomy",
    "nervous", "tense", "stressed", "overwhelmed", "distraught",
    "apprehensive", "agitated", "bitter", "resentful", "hostile",
    "lonely", "forlorn",
}


def validate(stimuli: list[dict]):
    """Validate stimuli quality."""
    from collections import defaultdict
    import re

    cond_emo_counts = defaultdict(lambda: defaultdict(int))
    issues = []

    for s in stimuli:
        cond_emo_counts[s["condition"]][s["emotion"]] += 1

        # Check context_only for emotion word leakage
        if s["condition"] == "context_only":
            text_lower = s["text"].lower()
            words = set(re.findall(r"[a-z]+", text_lower))
            leaks = words & BANNED_EMOTION_WORDS
            if leaks:
                issues.append(
                    f"  LEAK in context_only/{s['emotion']} item {s['item_id']}: "
                    f"{leaks}"
                )

        # Check word counts
        wc = len(s["text"].split())
        if s["condition"] == "label_only" and not (10 <= wc <= 35):
            issues.append(
                f"  LENGTH label_only/{s['emotion']} item {s['item_id']}: "
                f"{wc} words (target 15-25)"
            )
        if s["condition"] == "context_only" and not (15 <= wc <= 65):
            issues.append(
                f"  LENGTH context_only/{s['emotion']} item {s['item_id']}: "
                f"{wc} words (target 30-50)"
            )
        if s["condition"] == "conflict" and not (20 <= wc <= 70):
            issues.append(
                f"  LENGTH conflict/{s['emotion']} item {s['item_id']}: "
                f"{wc} words (target 35-55)"
            )

    return cond_emo_counts, issues


def main():
    stimuli = build_stimuli()

    # Validate
    cond_emo_counts, issues = validate(stimuli)

    print("=" * 70)
    print("LABEL-CONTEXT CONFLICT STIMULI")
    print("=" * 70)
    print(f"\nTotal stimuli: {len(stimuli)}")
    print(f"\nBreakdown by condition x emotion:")
    for cond in ["label_only", "context_only", "conflict"]:
        print(f"\n  {cond}:")
        for emo in ["anger", "sadness", "fear", "disgust", "happiness"]:
            n = cond_emo_counts[cond][emo]
            print(f"    {emo:12s}: {n}")

    # Word count statistics
    for cond in ["label_only", "context_only", "conflict"]:
        wcs = [len(s["text"].split()) for s in stimuli if s["condition"] == cond]
        print(f"\n  {cond} word counts: "
              f"min={min(wcs)}, max={max(wcs)}, "
              f"mean={sum(wcs)/len(wcs):.1f}")

    # Conflict pairings summary
    print("\nConflict pairings:")
    for s in stimuli:
        if s["condition"] == "conflict" and s["item_id"] == min(
                x["item_id"] for x in stimuli
                if x["condition"] == "conflict" and x["emotion"] == s["emotion"]):
            print(f"  label={s['emotion']:10s}  context={s['conflict_context_emotion']}")

    if issues:
        print(f"\nWARNINGS ({len(issues)}):")
        for iss in issues:
            print(iss)
    else:
        print("\nAll validation checks PASSED.")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for s in stimuli:
            f.write(json.dumps(s) + "\n")
    print(f"\nSaved {len(stimuli)} stimuli to {OUT_PATH}")


if __name__ == "__main__":
    main()
