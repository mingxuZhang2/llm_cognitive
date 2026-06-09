#!/usr/bin/env python3
"""
2×2 Factorial: Situational richness × Embodied content.

Disambiguates four explanations for why enriched input recovers emotion geometry:
  (a) More tokens / SNR
  (b) Richer situational context (appraisal)
  (c) Embodied information (bodily states, action tendencies)
  (d) Emotion synonym vocabulary

Design: 5 emotions × 4 conditions × 15 stimuli = 300 stimuli, matched token count.
All conditions ~20-25 words. NO emotion labels or synonyms anywhere.

Conditions:
  A. minimal_neutral:  Short situation + generic affect ("strong negative reaction")
  B. minimal_embodied: Short situation + embodied description (body + action tendency)
  C. rich_neutral:     Rich situation + generic affect
  D. rich_embodied:    Rich situation + embodied description

Embodied descriptions derived from:
  - Ekman & Levenson (1983) autonomic specificity profiles
  - Nummenmaa et al. (2014) bodily sensation maps
  - Frijda (1986) action tendency theory

Generates stimuli to JSONL, then extracts activations and computes within-aff RSA.
Step 1 (this script, CPU): generate stimuli.
Step 2 (GPU script): extract activations.
Step 3 (CPU): compute RSA per condition.
"""
from __future__ import annotations
import json, random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "data" / "cognitive_stimuli" / "rsa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["anger", "fear", "disgust", "sadness", "happiness"]

# ═══════════════════════════════════════════════════════════════════
# EMBODIED PROFILES per emotion
# Sources: Ekman & Levenson 1983, Nummenmaa 2014, Frijda 1986
# ═══════════════════════════════════════════════════════════════════

EMBODIED = {
    "anger": [
        "Their jaw clenched tight, heat rising in their face and neck, fists balling up involuntarily, every muscle coiling with the urge to confront and push back.",
        "A hot pressure surged through their chest and up into their face, their teeth grinding, hands trembling with a barely contained impulse to strike out.",
        "Their nostrils flared, shoulders squared, a burning tension spreading from their stomach to their clenched fists, body leaning forward ready to act.",
        "Blood rushed to their face, neck veins taut, their whole upper body rigid and forward-leaning, hands opening and closing with restless aggressive energy.",
        "Their brow contracted hard, jaw locked, a fiery tightness gripping their chest while their legs braced as if preparing to charge.",
    ],
    "fear": [
        "Their heart hammered against their ribs, hands gone cold and clammy, stomach dropping, every instinct screaming to flee or find somewhere safe to hide.",
        "A cold wave washed down their spine, breath catching in their throat, pupils wide, muscles frozen between the urge to run and the inability to move.",
        "Their palms went slick with sweat, legs trembling, chest constricted so tight they could barely breathe, eyes darting for the nearest exit.",
        "Goosebumps spread across their arms, stomach churning with nausea, their body pulling inward and shrinking as if trying to become invisible.",
        "Their throat closed up, fingers numb and tingling, heart racing so fast they felt dizzy, every sense sharpened to detect the next threat.",
    ],
    "disgust": [
        "Their upper lip curled back, stomach turning with a wave of nausea, whole body recoiling as if something contaminated had brushed against their skin.",
        "A sour taste rose in the back of their throat, face scrunching involuntarily, hands pushing away from the source as if to create maximum distance.",
        "Their nose wrinkled, gorge rising, skin crawling with an overwhelming urge to wash their hands and put physical distance between themselves and the source.",
        "A queasy heaving gripped their stomach, shoulders hunching forward, mouth pressing shut tight to block out the sensation, body twisting away.",
        "Their diaphragm contracted in a gag reflex, face contorting, every fiber pulling back and away as if proximity itself was contaminating.",
    ],
    "sadness": [
        "Their chest felt hollow and impossibly heavy, shoulders slumping, eyes stinging as a thick lump formed in their throat, all energy draining away.",
        "A leaden weight settled behind their sternum, breathing shallow and slow, limbs heavy as if moving through water, vision blurring with unshed tears.",
        "Their whole body sagged, chin dropping toward their chest, a deep ache radiating outward from their core, too exhausted to even raise their head.",
        "Everything felt muted and far away, their muscles slack, a persistent sting behind their eyes, chest so tight with heaviness they could barely inhale.",
        "Their lower lip trembled, arms hanging limp at their sides, a cold emptiness spreading through their torso, legs barely supporting their weight.",
    ],
    "happiness": [
        "Warmth spread through their chest, shoulders dropping away from tension, a lightness lifting their whole posture, lungs expanding with an easy deep breath.",
        "Their cheeks lifted involuntarily into a wide smile, a fizzing energy tingling through their limbs, body bouncing slightly with barely contained buoyancy.",
        "A rush of warmth flooded from their core outward, posture straightening, eyes brightening, breath coming easy and deep as muscles they did not know were tense finally released.",
        "Their heart rate slowed to a calm steady beat, face relaxed and open, a gentle expansive warmth radiating from their chest down through their limbs.",
        "Every muscle softened, a quiet glow settling behind their sternum, the corners of their mouth turning up, body feeling light and grounded at the same time.",
    ],
}

GENERIC_AFFECT = [
    "They had a very strong emotional reaction to what happened, one that stayed with them and was impossible to simply set aside or move past easily.",
    "They experienced an intense feeling that was hard to put into words, a reaction so strong it colored everything around them for a long while afterward.",
    "A powerful emotional response washed over them in that moment, lingering in a way that made it hard to focus on anything else for the rest of the day.",
    "They felt something very strongly, a reaction that took over completely, filling their entire awareness and making everything else seem distant and unimportant.",
    "An overwhelming feeling gripped them, impossible to ignore or push aside, the kind of reaction that changes how a person sees everything around them afterward.",
]

# ═══════════════════════════════════════════════════════════════════
# SITUATIONS per emotion (rich vs minimal)
# Situations chosen to naturally evoke each emotion without naming it
# ═══════════════════════════════════════════════════════════════════

SITUATIONS_RICH = {
    "anger": [
        "When they discovered a colleague had deliberately taken credit for their six months of research in front of the entire department",
        "When they found out their landlord had illegally entered their apartment and gone through their personal belongings",
        "When they learned that someone had been spreading false and damaging rumors about them to their closest friends",
        "When the insurance company denied their legitimate claim for the third time using a technicality",
        "When they caught their trusted employee stealing from the cash register on the security camera",
        "When the mechanic charged them triple the quoted price and refused to release their car",
        "When their neighbor poisoned the tree they had planted with their late grandmother twenty years ago",
        "When the school administration ignored their repeated reports that their child was being bullied",
        "When they realized their financial advisor had been making unauthorized trades that lost their retirement savings",
        "When a driver intentionally cut them off and then brake-checked them on the highway with their children in the car",
        "When their business partner secretly filed paperwork transferring shared assets solely into their own name",
        "When a customer screamed obscenities at them in front of a packed restaurant for a kitchen delay beyond their control",
        "When they found that their identity had been stolen by someone they had personally helped and trusted",
        "When the contractor demolished the wrong wall in their home and refused to accept responsibility",
        "When their sibling excluded them from a family decision about their aging parents care without any discussion",
    ],
    "fear": [
        "When they heard heavy unfamiliar footsteps in their house at three in the morning while home alone",
        "When the turbulence became so violent that oxygen masks dropped and passengers started screaming",
        "When the doctor said the scan showed something unexpected and they needed to come in immediately",
        "When they realized their brakes were not responding while driving downhill toward a busy intersection",
        "When their small boat started taking on water two miles from shore with no other vessels in sight",
        "When an anonymous letter arrived threatening their family with specific details about their daily routine",
        "When the earthquake shook their building so hard that cracks appeared in the walls around them",
        "When they got separated from their group in an unfamiliar city after dark with a dead phone",
        "When the lab results came back positive for the genetic marker their parent had died from",
        "When they heard gunshots in the building and realized the exits were in the direction of the sound",
        "When they woke up unable to move or speak with a dark figure standing at the foot of their bed",
        "When the rope bridge they were crossing over a gorge began swaying and the planks started cracking",
        "When their child did not come home from school and was not answering their phone for three hours",
        "When the wildfire crested the ridge and they could see flames advancing toward their neighborhood",
        "When the MRI revealed a mass and the radiologist left the room without saying anything",
    ],
    "disgust": [
        "When they opened the container in the back of the fridge and found it crawling with maggots",
        "When they discovered the restaurant kitchen had a severe cockroach infestation visible from the dining area",
        "When they learned the public pool they had been swimming in had failed its health inspection for sewage contamination",
        "When they found mold growing thick and black behind the wallpaper throughout their bedroom",
        "When they accidentally stepped barefoot into a pile of rotting food waste that had been left in the alley",
        "When they watched footage of the factory that processed the food they had been eating daily for years",
        "When the plumber showed them the corroded pipe that had been leaking sewage into their drinking water",
        "When they realized the hotel room sheets had visible stains and hair from a previous guest",
        "When they found out their expensive organic produce had been sprayed with banned pesticides",
        "When they discovered the meat they had just served their family had been recalled for bacterial contamination",
        "When they saw their roommate handling raw chicken and then touching all the clean dishes without washing",
        "When the dentist showed them the decay that had been spreading under an old filling for months",
        "When they learned the herbal supplement they trusted contained undisclosed animal byproducts and heavy metals",
        "When they noticed the surgical instruments for their procedure had visible residue from a previous patient",
        "When they realized the vendor at the market had been repackaging expired products with new labels",
    ],
    "sadness": [
        "When they received the call that their childhood best friend had passed away unexpectedly overnight",
        "When they watched their elderly parent struggle to remember their name for the first time",
        "When they returned to their childhood home and found it demolished and replaced with a parking lot",
        "When they sat alone in the empty apartment after their partner of twelve years had moved out",
        "When their beloved dog that had been with them through everything finally stopped breathing at the vet",
        "When they found old letters from their late mother and recognized the handwriting they would never see again",
        "When the adoption agency called to say the birth parents had changed their mind after two years of waiting",
        "When they realized their youngest child no longer needed them and was building a life entirely of their own",
        "When they cleaned out their fathers workshop after the funeral finding unfinished projects he would never complete",
        "When the last friend from their original group moved away and they realized that chapter was truly over",
        "When they stood at the spot where their family home once was now just an empty foundation overgrown with weeds",
        "When they watched the video from last Christmas knowing it was the last one they would all be together",
        "When the school where they had taught for thirty years closed its doors permanently",
        "When they held the ultrasound photo of the baby they had lost",
        "When they looked through their wedding album alone on what would have been their twentieth anniversary",
    ],
    "happiness": [
        "When they opened the envelope and read that they had been accepted into the program they had dreamed of for years",
        "When their child took their first steps across the living room and reached out to grab their hands",
        "When they finished the marathon they had trained eighteen months for and saw their family cheering at the finish",
        "When the test came back negative after weeks of difficult waiting and sleepless nights",
        "When their estranged sibling showed up at their door after five years and said they wanted to make things right",
        "When they stood on the summit after a grueling climb and saw the valley stretched out below them in golden light",
        "When the audience gave them a standing ovation after the performance they had poured their soul into",
        "When they walked into the surprise party and saw every person they loved gathered in one room",
        "When they received the offer for the job that would let them finally do the work they believed in",
        "When their rescue dog who had been wary of people for months finally climbed into their lap and fell asleep",
        "When they paid off the last of their debt and saw a zero balance for the first time in fifteen years",
        "When they heard their song playing on the radio for the first time and realized strangers were listening",
        "When the transplant coordinator called to say a match had been found after three years on the waiting list",
        "When they woke up on the first morning of retirement and realized the day was entirely their own",
        "When their student who had struggled all year burst into tears after passing the final exam they once thought impossible",
    ],
}

SITUATIONS_MINIMAL = {
    "anger": [
        "When someone at work treated them very unfairly",
        "When a person they trusted violated their privacy",
        "When someone deliberately damaged their reputation",
        "When an institution repeatedly refused their valid request",
        "When they caught someone they trusted being dishonest",
        "When a service provider took advantage of them financially",
        "When someone destroyed something meaningful to them",
        "When authorities ignored their legitimate concerns",
        "When a professional they relied on cost them their savings",
        "When a stranger endangered them and their family on purpose",
        "When a partner in a shared endeavor cheated them",
        "When someone publicly humiliated them without cause",
        "When a person they helped personally betrayed their trust",
        "When a hired worker caused damage and denied responsibility",
        "When a family member excluded them from an important decision",
    ],
    "fear": [
        "When they heard unexpected sounds in their home late at night",
        "When their vehicle of transportation hit severe dangerous conditions",
        "When a medical professional said they needed urgent follow-up tests",
        "When a piece of critical equipment failed in a dangerous situation",
        "When they were stranded far from help with no communication",
        "When they received a direct personal threat from an unknown source",
        "When a natural event caused visible structural damage around them",
        "When they became lost in an unfamiliar place after dark alone",
        "When medical test results suggested a serious hereditary condition",
        "When they heard signs of immediate danger nearby with no clear escape",
        "When they woke and found themselves unable to move or call for help",
        "When a structure they depended on began failing beneath them",
        "When someone they were responsible for could not be reached for hours",
        "When a large-scale emergency began moving in their direction",
        "When a specialist abruptly left during a procedure without explanation",
    ],
    "disgust": [
        "When they found their stored food had become severely contaminated",
        "When they saw clear evidence of infestation where they had been eating",
        "When they learned a place they used regularly had failed health standards",
        "When they discovered hidden contamination in their living space",
        "When they made physical contact with decaying organic waste unexpectedly",
        "When they saw how a product they consumed daily was actually processed",
        "When they learned their water supply had been contaminated with waste",
        "When they found clear evidence a space they used was deeply unsanitary",
        "When they learned a trusted product contained undisclosed harmful substances",
        "When they found out food they served their family had been recalled for contamination",
        "When they saw someone handle contaminated and clean items without any separation",
        "When a professional showed them hidden decay in their own body",
        "When they discovered a health product they took contained banned substances",
        "When they noticed unsterile conditions in a medical setting before a procedure",
        "When they realized a vendor had been disguising expired products as fresh",
    ],
    "sadness": [
        "When they learned that someone very close to them had passed away suddenly",
        "When they watched a parent begin to lose the ability to recognize them",
        "When they returned to a meaningful place and found it completely gone",
        "When they were alone after a long relationship ended permanently",
        "When a beloved companion animal they had raised reached the end of its life",
        "When they found personal items from someone they had lost and could never see again",
        "When a long-awaited plan they had invested in emotionally fell through at the last moment",
        "When they realized someone they had always cared for no longer needed them",
        "When they sorted through the belongings of someone who would never return",
        "When the last close connection from an important period of their life departed",
        "When they visited the empty site of a place that had once been central to their life",
        "When they watched a recording from a gathering knowing it was the final one",
        "When an institution that had been part of their identity for decades closed permanently",
        "When they held a keepsake representing a future that would never come to be",
        "When they spent a significant anniversary alone that was once shared with someone",
    ],
    "happiness": [
        "When they received confirmation of acceptance into something they had long hoped for",
        "When they witnessed a major milestone of someone they had raised and nurtured",
        "When they completed a demanding long-term physical goal with loved ones watching",
        "When they received reassuring results after an extended period of deep worry",
        "When an important relationship they thought was lost was unexpectedly restored",
        "When they reached a personal summit and could see the full scope of their achievement",
        "When they received enthusiastic recognition for creative work they deeply cared about",
        "When they were surrounded unexpectedly by all the people who mattered most to them",
        "When they were offered the opportunity to do work aligned with their deepest values",
        "When a being they had patiently cared for finally showed complete trust in them",
        "When they achieved financial freedom after many years of disciplined effort",
        "When they experienced their personal creation reaching and moving complete strangers",
        "When life-saving help arrived after they had nearly given up hope of receiving it",
        "When they experienced the first day of a new chapter entirely on their own terms",
        "When someone they had guided through difficulty achieved something once thought impossible",
    ],
}


def build_stimulus(emotion, situation, embodied_desc, is_embodied, token_target=22):
    """Build a single stimulus, aiming for ~token_target words."""
    if is_embodied:
        text = f"{situation}, {embodied_desc.lower()}"
    else:
        generic = random.choice(GENERIC_AFFECT)
        text = f"{situation}. {generic}"
    return text


def main():
    random.seed(42)
    stimuli = []

    for emo in EMOTIONS:
        rich_sits = SITUATIONS_RICH[emo]
        min_sits = SITUATIONS_MINIMAL[emo]
        emb_descs = EMBODIED[emo]

        for i in range(15):
            # A: minimal_neutral
            stimuli.append({
                "condition": emo,
                "factorial": "minimal_neutral",
                "text": build_stimulus(emo, min_sits[i], None, False),
            })
            # B: minimal_embodied
            stimuli.append({
                "condition": emo,
                "factorial": "minimal_embodied",
                "text": build_stimulus(emo, min_sits[i], emb_descs[i % len(emb_descs)], True),
            })
            # C: rich_neutral
            stimuli.append({
                "condition": emo,
                "factorial": "rich_neutral",
                "text": build_stimulus(emo, rich_sits[i], None, False),
            })
            # D: rich_embodied
            stimuli.append({
                "condition": emo,
                "factorial": "rich_embodied",
                "text": build_stimulus(emo, rich_sits[i], emb_descs[i % len(emb_descs)], True),
            })

    # Verify no emotion labels leaked
    BANNED = {"anger", "angry", "furious", "fury", "rage", "outrage", "wrath", "indignation",
              "fear", "afraid", "scared", "terrified", "terror", "dread", "panic", "fright",
              "disgust", "disgusted", "disgusting", "revulsion", "revolted", "repulsed",
              "sadness", "sad", "grief", "grieving", "sorrow", "sorrowful", "melancholy",
              "happiness", "happy", "joy", "joyful", "elated", "elation", "delight", "delighted",
              "anxious", "anxiety", "depressed", "depression", "upset", "frustrated", "frustration"}

    leaks = 0
    for s in stimuli:
        words = set(s["text"].lower().split())
        found = words & BANNED
        if found:
            print(f"  LEAK in {s['condition']}/{s['factorial']}: {found}")
            print(f"    {s['text'][:100]}")
            leaks += 1

    # Stats
    from collections import Counter
    import numpy as np
    cond_counts = Counter((s["condition"], s["factorial"]) for s in stimuli)
    print(f"\nGenerated {len(stimuli)} stimuli ({leaks} label leaks)")
    print(f"\nCounts per cell:")
    for (emo, fac), n in sorted(cond_counts.items()):
        lens = [len(s["text"].split()) for s in stimuli
                if s["condition"] == emo and s["factorial"] == fac]
        print(f"  {emo:>12s} × {fac:<20s}: n={n:2d}, len={np.mean(lens):.1f}±{np.std(lens):.1f}")

    # Overall length stats per factorial condition
    print(f"\nLength per factorial condition:")
    for fac in ["minimal_neutral", "minimal_embodied", "rich_neutral", "rich_embodied"]:
        lens = [len(s["text"].split()) for s in stimuli if s["factorial"] == fac]
        print(f"  {fac:<20s}: {np.mean(lens):.1f} ± {np.std(lens):.1f} words")

    out_path = OUT_DIR / "embodied_factorial_stimuli.jsonl"
    with open(out_path, "w") as f:
        for s in stimuli:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
