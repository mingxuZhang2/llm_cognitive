"""
Generate paired moral decomposition stimuli.

4 conditions × 50 pairs × 2 items = 400 items total.

Each pair varies ONE variable while controlling others (length, lexical content,
domain, severity). Target judgments are calibrated against cognitive psychology
literature on moral judgment (Cushman 2008, Young et al. 2007, Greene et al. 2009,
Turiel 1983, Nichols 2002).
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent
OUT_PATH = OUT_DIR / "decomposition_stimuli.jsonl"


# =========================================================================
# CONDITION 1: INTENTIONAL vs ACCIDENTAL (50 pairs, varying intent only)
# =========================================================================
# Literature: Cushman (2008) shows intent dominates judgment for moral violations.
# Intentional harm rated ~6-7; accidental same outcome rated ~2-4.

INTENT_PAIRS = [
    # 1 - peanut allergy (workplace)
    ("Sarah knew her colleague was severely allergic to peanuts. She added peanut oil to his lunch as revenge for a workplace argument. He had a severe allergic reaction and was hospitalized.",
     "Sarah did not know her colleague was severely allergic to peanuts. She added peanut oil to his lunch as a friendly gesture at lunch. He had a severe allergic reaction and was hospitalized.",
     6.7, 2.3, "workplace/poisoning"),
    # 2 - icy steps (community)
    ("Daniel saw the icy steps and deliberately failed to salt them, hoping his estranged uncle would slip. His uncle fell on the steps and broke his hip the next morning.",
     "Daniel did not see the icy steps and forgot to salt them, not thinking his uncle would visit. His uncle fell on the steps and broke his hip the next morning.",
     6.4, 2.6, "family/negligence"),
    # 3 - medication swap (healthcare)
    ("A nurse intentionally swapped a patient's blood pressure pill with a sedative to harm him. The patient suffered a dangerous drop in blood pressure and required emergency care.",
     "A nurse accidentally swapped a patient's blood pressure pill with a sedative from a similar bottle. The patient suffered a dangerous drop in blood pressure and required emergency care.",
     6.8, 3.2, "healthcare/medication"),
    # 4 - online doxxing (online)
    ("Megan deliberately posted her ex-boyfriend's home address on a hostile forum to expose him. Strangers began harassing him at his home that same week.",
     "Megan accidentally posted her ex-boyfriend's home address on a hostile forum while sharing a photo. Strangers began harassing him at his home that same week.",
     6.5, 3.0, "online/doxxing"),
    # 5 - shoplifting setup (school)
    ("Tyler placed stolen merchandise in another student's backpack to frame him before security checks. The student was suspended and barred from the school store.",
     "Tyler accidentally placed merchandise in another student's backpack while distracted at the counter. The student was suspended and barred from the school store.",
     6.6, 2.4, "school/framing"),
    # 6 - house fire (public space)
    ("Owen lit a cigarette and tossed it onto his neighbor's porch on purpose, hoping to start a fire. The porch caught fire and the neighbor's home was badly damaged.",
     "Owen lit a cigarette and dropped it onto his neighbor's porch by mistake while walking past. The porch caught fire and the neighbor's home was badly damaged.",
     6.7, 3.0, "community/arson"),
    # 7 - withheld message (friendship)
    ("Linda intentionally withheld an urgent message about a job interview from her roommate out of envy. The roommate missed the interview and lost the opportunity.",
     "Linda inadvertently overlooked an urgent message about a job interview meant for her roommate. The roommate missed the interview and lost the opportunity.",
     5.8, 2.4, "friendship/withholding"),
    # 8 - allergic dessert (family)
    ("Hannah baked her cousin a dessert with hidden walnuts after learning of his allergy, hoping to cause a reaction. He went into anaphylaxis and was rushed to the hospital.",
     "Hannah baked her cousin a dessert with walnuts after forgetting his allergy entirely, not realizing the risk. He went into anaphylaxis and was rushed to the hospital.",
     6.7, 2.7, "family/poisoning"),
    # 9 - misrouted package (online/financial)
    ("Brett deliberately rerouted a customer's package to a vacant lot after a refund dispute. The customer lost the goods and the company refused to compensate.",
     "Brett accidentally rerouted a customer's package to a vacant lot due to a system error. The customer lost the goods and the company refused to compensate.",
     5.5, 2.2, "financial/delivery"),
    # 10 - spilled coffee (workplace)
    ("Greg knocked hot coffee onto his coworker's laptop on purpose, angry about a promotion he missed. The laptop was destroyed and a week of unsaved work was lost.",
     "Greg knocked hot coffee onto his coworker's laptop while turning around with a full mug. The laptop was destroyed and a week of unsaved work was lost.",
     5.7, 2.0, "workplace/property"),
    # 11 - chemical exposure (workplace)
    ("A foreman deliberately disabled a ventilation fan in the paint shop to punish a worker. The worker inhaled toxic fumes and was hospitalized for two weeks.",
     "A foreman did not notice the ventilation fan in the paint shop had broken that morning. The worker inhaled toxic fumes and was hospitalized for two weeks.",
     6.7, 3.0, "workplace/safety"),
    # 12 - withheld inhaler (school)
    ("Caleb deliberately hid the school nurse's asthma inhaler from a classmate he disliked during gym. The classmate had an attack on the field and collapsed.",
     "Caleb accidentally pocketed the school nurse's asthma inhaler with his own belongings during gym. The classmate had an attack on the field and collapsed.",
     6.6, 2.5, "school/health"),
    # 13 - false tip (community)
    ("Phyllis filed a false police report about her quiet neighbor to harass him into moving. Officers raided his home and his elderly mother was traumatized.",
     "Phyllis filed a confused police report about her quiet neighbor based on a misheard call. Officers raided his home and his elderly mother was traumatized.",
     6.5, 2.9, "community/false-report"),
    # 14 - drained battery (healthcare)
    ("A technician deliberately drained the battery on a hospital monitor in a rival's ward. A patient's deterioration went unnoticed and emergency intervention was delayed.",
     "A technician unwittingly drained the battery on a hospital monitor while running a routine test. A patient's deterioration went unnoticed and emergency intervention was delayed.",
     6.8, 3.4, "healthcare/equipment"),
    # 15 - shouted slur (public space)
    ("Glen shouted a slur at a stranger on the subway to humiliate her in front of others. The stranger left the train in tears and reported feeling unsafe.",
     "Glen shouted a slur at a stranger on the subway, unaware she was within earshot from across the car. The stranger left the train in tears and reported feeling unsafe.",
     6.3, 4.0, "public/harassment"),
    # 16 - missed dose (family/healthcare)
    ("Bart deliberately skipped giving his grandmother her evening dose, hoping she would deteriorate. She had a seizure that night and was admitted to the ICU.",
     "Bart absent-mindedly forgot to give his grandmother her evening dose during the family dinner. She had a seizure that night and was admitted to the ICU.",
     6.7, 3.0, "family/caregiving"),
    # 17 - dog bite (community)
    ("Roy unleashed his aggressive dog at a jogger he disliked on the trail to scare her. The dog bit the jogger badly and she needed stitches and rabies shots.",
     "Roy's leash snapped while walking his aggressive dog and the dog ran at a passing jogger. The dog bit the jogger badly and she needed stitches and rabies shots.",
     6.6, 3.3, "community/animal"),
    # 18 - locked door (workplace)
    ("Tara deliberately locked the storeroom from outside while a coworker was inside as a prank. The coworker was trapped for four hours and had a panic attack.",
     "Tara absent-mindedly locked the storeroom from outside while a coworker was still inside checking inventory. The coworker was trapped for four hours and had a panic attack.",
     5.7, 2.4, "workplace/prank"),
    # 19 - false rumor (friendship)
    ("Beatrice spread a deliberate lie that her best friend had cheated on a partner to ruin her reputation. The friend was dumped and shunned by their social circle.",
     "Beatrice repeated a rumor without checking facts that her best friend had cheated on a partner. The friend was dumped and shunned by their social circle.",
     6.3, 3.4, "friendship/rumor"),
    # 20 - mislabeled vial (healthcare)
    ("A lab tech deliberately mislabeled a vial of contagious culture so an annoying colleague would mishandle it. The colleague contracted an infection requiring hospitalization.",
     "A lab tech accidentally mislabeled a vial of contagious culture during a hectic morning shift. The colleague contracted an infection requiring hospitalization.",
     6.6, 3.4, "healthcare/labs"),
    # 21 - deleted files (workplace)
    ("Greta deliberately deleted six months of her coworker's project files after a heated argument. The coworker missed a major deadline and was placed on probation.",
     "Greta inadvertently deleted six months of her coworker's project files while cleaning up shared drives. The coworker missed a major deadline and was placed on probation.",
     6.0, 2.6, "workplace/property"),
    # 22 - tripped jogger (public space)
    ("Hank stuck out his foot to trip a passing jogger on the boardwalk as a cruel joke. The jogger fell hard onto the planks and broke her wrist.",
     "Hank shifted his stance and his foot caught a passing jogger on the boardwalk by mistake. The jogger fell hard onto the planks and broke her wrist.",
     6.0, 2.0, "public/physical"),
    # 23 - leaked secret (friendship)
    ("Karina deliberately leaked her friend's HIV status to a mutual acquaintance to embarrass her. The friend was ostracized at work and lost two close relationships.",
     "Karina inadvertently leaked her friend's HIV status in a passing remark to a mutual acquaintance. The friend was ostracized at work and lost two close relationships.",
     6.6, 3.7, "friendship/privacy"),
    # 24 - dropped baby (family)
    ("Vanessa deliberately let go of her baby cousin while holding her over the railing to scare the family. The baby fell, hit her head, and suffered a concussion.",
     "Vanessa lost her grip on her baby cousin while shifting her weight near the railing. The baby fell, hit her head, and suffered a concussion.",
     6.9, 2.8, "family/childcare"),
    # 25 - swapped wires (workplace/safety)
    ("An electrician deliberately swapped two live wires in a rival contractor's installation to embarrass him. A homeowner received a serious electric shock the next day.",
     "An electrician accidentally swapped two live wires during a rushed installation while a rival contractor watched. A homeowner received a serious electric shock the next day.",
     6.7, 3.4, "workplace/safety"),
    # 26 - rigged scale (financial)
    ("A grocer deliberately tampered with the produce scale to overcharge elderly customers each week. Many regulars were defrauded out of significant sums over months.",
     "A grocer inadvertently miscalibrated the produce scale during a routine reset early in the week. Many regulars were defrauded out of significant sums over months.",
     6.5, 2.4, "financial/fraud"),
    # 27 - unbuckled belt (family)
    ("Curt deliberately unbuckled his daughter's car seat belt before a freeway drive as a thrill. She was thrown from the seat in a minor collision and broke her arm.",
     "Curt did not notice his daughter's car seat belt was unbuckled before a freeway drive. She was thrown from the seat in a minor collision and broke her arm.",
     6.9, 3.0, "family/childcare"),
    # 28 - dumped solvent (environmental)
    ("A factory manager deliberately dumped solvent into the river behind the plant to save disposal costs. Local fish died and a downstream village's water was tainted.",
     "A factory manager did not realize stored solvent had leaked into the river behind the plant. Local fish died and a downstream village's water was tainted.",
     6.6, 3.2, "environmental/pollution"),
    # 29 - false alarm (school)
    ("Brent deliberately pulled the school's fire alarm during a final exam to ruin classmates' scores. A student with mobility issues was hurt in the stampede.",
     "Brent accidentally pulled the school's fire alarm while leaning against the wall during a final exam. A student with mobility issues was hurt in the stampede.",
     6.0, 2.3, "school/disruption"),
    # 30 - hot oil (family)
    ("Wendy deliberately tipped a pan of hot oil toward her sister's arm during a kitchen argument. Her sister sustained second-degree burns and needed skin grafts.",
     "Wendy lost her grip on a pan of hot oil while turning toward her sister during dinner prep. Her sister sustained second-degree burns and needed skin grafts.",
     6.8, 2.7, "family/violence"),
    # 31 - sent malware (online)
    ("Curtis deliberately sent his rival a file laced with malware disguised as a tax document. The rival's computer was compromised and confidential data was stolen.",
     "Curtis unknowingly forwarded a file laced with malware that arrived disguised as a tax document. The rival's computer was compromised and confidential data was stolen.",
     6.6, 2.6, "online/cybercrime"),
    # 32 - parked on hose (community)
    ("Marco deliberately parked on the fire hydrant hose during the fire next door to annoy responders. The blaze spread and the family lost most of their belongings.",
     "Marco parked his car on the fire hydrant hose without seeing it during the emergency next door. The blaze spread and the family lost most of their belongings.",
     6.4, 2.4, "community/safety"),
    # 33 - tainted sample (workplace)
    ("A chemist deliberately added a contaminant to a competitor's lab sample to discredit her work. Her published findings were retracted and her career was set back years.",
     "A chemist accidentally added a contaminant to a competitor's lab sample by reusing an unclean pipette. Her published findings were retracted and her career was set back years.",
     6.4, 2.7, "workplace/sabotage"),
    # 34 - cheated partner (friendship)
    ("Jonas deliberately set up a fake dating profile to lure his friend's wife into an affair as revenge. Their marriage ended and the couple's children were devastated.",
     "Jonas inadvertently created a profile that his friend's wife mistook as flirtation while he was joking. Their marriage ended and the couple's children were devastated.",
     6.7, 3.3, "friendship/betrayal"),
    # 35 - elevator (public space)
    ("Sheila deliberately held the elevator doors closed as a man was rushing in to taunt him. He missed an urgent appointment and lost a major contract.",
     "Sheila pressed the close-door button without noticing the man rushing in toward the elevator. He missed an urgent appointment and lost a major contract.",
     5.4, 1.9, "public/inconsiderate"),
    # 36 - hot water (community)
    ("Travis deliberately turned the apartment building's water heater past safe limits to scare a tenant. An elderly resident was scalded in the shower the next morning.",
     "Travis adjusted the apartment building's water heater past safe limits while doing routine checks. An elderly resident was scalded in the shower the next morning.",
     6.5, 3.2, "community/safety"),
    # 37 - cut brakes (community)
    ("Owen deliberately cut his neighbor's bicycle brakes to scare him on a steep ride home. The neighbor crashed at high speed and broke his collarbone.",
     "Owen damaged his neighbor's bicycle brakes while moving the bike in the shared garage. The neighbor crashed at high speed and broke his collarbone.",
     6.7, 2.7, "community/sabotage"),
    # 38 - kicked dog (community)
    ("Frances deliberately kicked a small dog that wandered near her yard out of irritation. The dog's leg was broken and the owner needed costly veterinary care.",
     "Frances stumbled and her foot struck a small dog that wandered near her yard. The dog's leg was broken and the owner needed costly veterinary care.",
     6.3, 2.4, "community/animal"),
    # 39 - hidden contract (financial)
    ("A landlord deliberately buried a costly hidden clause in a tenant's lease to extract penalties later. The tenant lost her deposit and most of two months' wages.",
     "A landlord copied a costly hidden clause into a tenant's lease without reading it carefully. The tenant lost her deposit and most of two months' wages.",
     5.9, 2.6, "financial/contract"),
    # 40 - left unattended (family)
    ("Audrey deliberately left her toddler unattended near the pool to teach her husband a lesson. The toddler nearly drowned and required CPR from a neighbor.",
     "Audrey did not realize her toddler had wandered near the pool while she answered the door. The toddler nearly drowned and required CPR from a neighbor.",
     6.8, 3.5, "family/childcare"),
    # 41 - photocopied test (school)
    ("Renee deliberately photocopied a stolen exam and slipped it to a struggling classmate she resented. The classmate was caught cheating and expelled from the program.",
     "Renee accidentally photocopied an exam left in the tray and slipped it to a struggling classmate. The classmate was caught cheating and expelled from the program.",
     5.9, 2.4, "school/cheating"),
    # 42 - tripped stroller (public space)
    ("Cal deliberately stuck a foot in a passing stroller's wheel to startle a young mother at the market. The stroller flipped and the toddler hit her head on the floor.",
     "Cal's foot caught a passing stroller's wheel as he stepped backward at the busy market. The stroller flipped and the toddler hit her head on the floor.",
     6.6, 2.5, "public/childcare"),
    # 43 - misrouted ambulance (healthcare)
    ("A dispatcher deliberately redirected an ambulance away from a person she resented from school. The patient's heart attack went untreated for critical minutes.",
     "A dispatcher mistakenly redirected an ambulance away from a person during a chaotic shift. The patient's heart attack went untreated for critical minutes.",
     6.9, 3.6, "healthcare/dispatch"),
    # 44 - vandalized car (community)
    ("Hugh deliberately keyed his neighbor's brand-new car over a parking dispute that had simmered for weeks. The neighbor faced a costly repair and a sleepless night.",
     "Hugh dropped his keychain and it dragged across his neighbor's brand-new car when he turned. The neighbor faced a costly repair and a sleepless night.",
     5.7, 2.0, "community/property"),
    # 45 - dirty needle (healthcare)
    ("A phlebotomist deliberately reused a needle on a patient she disliked from a prior visit. The patient contracted a serious blood-borne infection within weeks.",
     "A phlebotomist accidentally reused a needle in the rush of a short-staffed afternoon shift. The patient contracted a serious blood-borne infection within weeks.",
     6.9, 3.7, "healthcare/infection"),
    # 46 - smashed instrument (school)
    ("Roger deliberately smashed his classmate's violin before the recital because she had outshone him. She missed the audition and lost her chance at a scholarship.",
     "Roger tripped over his classmate's violin before the recital while reaching for his music stand. She missed the audition and lost her chance at a scholarship.",
     6.0, 2.3, "school/property"),
    # 47 - leaked exam (school)
    ("Penny deliberately leaked the graduate entrance exam questions online to discredit the testing committee. Thousands of applicants had their scores invalidated.",
     "Penny mistakenly uploaded the graduate entrance exam questions to a public folder during file sorting. Thousands of applicants had their scores invalidated.",
     6.3, 3.0, "school/cheating"),
    # 48 - shoved elder (public space)
    ("Drew deliberately shoved an elderly man aside in the crowded subway car to claim a seat. The man fell hard against a pole and broke two ribs.",
     "Drew lurched into an elderly man as the crowded subway car braked suddenly. The man fell hard against a pole and broke two ribs.",
     6.4, 1.8, "public/physical"),
    # 49 - drank stash (family)
    ("Lara deliberately drank her recovering father's hidden alcohol stash and replaced it with water to mock him. He relapsed badly that weekend and crashed his car.",
     "Lara unknowingly drank her recovering father's hidden alcohol stash, thinking it was a cooking ingredient. He relapsed badly that weekend and crashed his car.",
     6.4, 2.6, "family/addiction"),
    # 50 - cut rope (community)
    ("Vince deliberately cut a climber's safety rope during a recreational climb to scare him. The climber fell ten meters and suffered serious spinal injuries.",
     "Vince nicked a climber's safety rope with his own gear while passing on the route. The climber fell ten meters and suffered serious spinal injuries.",
     6.9, 3.2, "community/recreation"),
]


# =========================================================================
# CONDITION 2: ATTEMPTED vs COMPLETED (50 pairs, varying outcome only)
# =========================================================================
# Literature: Cushman (2008) shows OUTCOME has a smaller but real effect.
# Completed harm rated ~6.5-7; attempted same intent rated ~5-6.

OUTCOME_PAIRS = [
    # 1 - rifle
    ("Marcus loaded the rifle, aimed it at his rival, and pulled the trigger with the intent to kill. The bullet struck his rival in the chest and he died at the scene.",
     "Marcus loaded the rifle, aimed it at his rival, and pulled the trigger with the intent to kill. The gun jammed and no bullet fired; his rival escaped unharmed.",
     6.8, 5.8, "violence/firearm"),
    # 2 - poison drink
    ("Helen dissolved a lethal dose of rat poison into her husband's bedtime tea. He drank it, collapsed within the hour, and died on the way to the hospital.",
     "Helen dissolved a lethal dose of rat poison into her husband's bedtime tea. He never drank it because the cup was knocked over by the cat.",
     6.9, 5.9, "family/poisoning"),
    # 3 - bomb mailing
    ("Jacob mailed a homemade pipe bomb to a journalist who had criticized him. The package detonated when opened and the journalist lost three fingers.",
     "Jacob mailed a homemade pipe bomb to a journalist who had criticized him. Postal screening detected the device and disarmed it before delivery.",
     7.0, 6.4, "violence/bombing"),
    # 4 - drug a drink
    ("A man slipped a date-rape drug into a stranger's drink at the bar with the intent to assault her. The drug took effect and he assaulted her in the back room.",
     "A man slipped a date-rape drug into a stranger's drink at the bar with the intent to assault her. She did not drink it because a friend swapped the glass.",
     7.0, 6.3, "violence/assault"),
    # 5 - drowning attempt (childcare)
    ("Greta tried to drown her stepchild by holding her under the bath water during an argument. The child stopped struggling and died before paramedics arrived.",
     "Greta tried to drown her stepchild by holding her under the bath water during an argument. A neighbor heard the noise and intervened in time to save the child.",
     7.0, 6.4, "family/violence"),
    # 6 - sniper miss (violence)
    ("A gunman opened fire at a senator from a rooftop with intent to assassinate him. The senator was struck in the chest and pronounced dead at the hospital.",
     "A gunman opened fire at a senator from a rooftop with intent to assassinate him. The senator was unharmed because the shooter missed by inches.",
     6.9, 6.3, "violence/firearm"),
    # 7 - stab attack (community)
    ("Roy lunged at a stranger in the parking lot with a knife, intending to kill him over a feud. He stabbed the stranger in the chest, who later died of the wounds.",
     "Roy lunged at a stranger in the parking lot with a knife, intending to kill him over a feud. He missed entirely and a bystander tackled him to the ground.",
     6.9, 6.0, "violence/assault"),
    # 8 - tampered brakes
    ("Megan tampered with her ex-husband's brakes hoping to cause a fatal crash on his commute. The brakes failed on the freeway and he died in the resulting collision.",
     "Megan tampered with her ex-husband's brakes hoping to cause a fatal crash on his commute. A mechanic noticed the damage at a routine check and fixed it in time.",
     6.9, 6.0, "family/sabotage"),
    # 9 - fraud transfer
    ("Carl wired his employer's funds into his own shell account, intending to steal half a million dollars. The transfer cleared and the company suffered a catastrophic loss.",
     "Carl wired his employer's funds into his own shell account, intending to steal half a million dollars. The bank flagged the transfer and reversed it before settlement.",
     6.4, 5.5, "financial/fraud"),
    # 10 - drug overdose
    ("A drug dealer sold a heroin batch he knew was laced with fentanyl to harm a difficult client. The client overdosed and died alone in his apartment that night.",
     "A drug dealer sold a heroin batch he knew was laced with fentanyl to harm a difficult client. The client never used it because he was arrested before injecting.",
     6.7, 5.8, "violence/drugs"),
    # 11 - school shooting
    ("A teenager brought a loaded handgun to school intending to shoot a teacher who had failed him. He shot the teacher in the hallway and she was pronounced dead.",
     "A teenager brought a loaded handgun to school intending to shoot a teacher who had failed him. He was tackled by a janitor before he could draw the weapon.",
     7.0, 6.4, "violence/school"),
    # 12 - arson
    ("Vince poured gasoline around a rival's home at midnight and lit it to burn the building down. The house was destroyed and an elderly resident died of smoke inhalation.",
     "Vince poured gasoline around a rival's home at midnight and lit it to burn the building down. A passing officer extinguished the flames before they took hold.",
     6.9, 6.1, "violence/arson"),
    # 13 - laxative spite
    ("Brett laced his coworker's coffee with strong laxatives intending to ruin her client presentation. She was incapacitated during the meeting and lost the major contract.",
     "Brett laced his coworker's coffee with strong laxatives intending to ruin her client presentation. She switched mugs with a colleague and never drank the laced coffee.",
     5.7, 4.8, "workplace/sabotage"),
    # 14 - online scam
    ("A scammer designed a phishing email targeting elderly retirees in a small town's mailing list. Dozens fell for the scam and lost their life savings overnight.",
     "A scammer designed a phishing email targeting elderly retirees in a small town's mailing list. A spam filter intercepted every message before any was opened.",
     6.4, 5.5, "financial/fraud"),
    # 15 - hacking attempt
    ("A hacker deployed ransomware against a hospital network with the intent to extort large payment. Patient records were locked and surgeries were canceled for two days.",
     "A hacker deployed ransomware against a hospital network with the intent to extort large payment. The intrusion was detected and quarantined before any encryption began.",
     6.7, 5.8, "online/cybercrime"),
    # 16 - poisoned candy (family)
    ("Karina baked her stepson rat-poisoned cookies during an inheritance dispute with his father. He ate several, fell gravely ill, and required intensive care for weeks.",
     "Karina baked her stepson rat-poisoned cookies during an inheritance dispute with his father. He never ate them because his father threw the batch out unopened.",
     6.9, 6.0, "family/poisoning"),
    # 17 - mugging attempt
    ("Drew approached a tourist in the alley with a knife and demanded all her cash and jewelry. He stabbed her arm when she resisted and fled with her purse and rings.",
     "Drew approached a tourist in the alley with a knife and demanded all her cash and jewelry. A patrol officer rounded the corner and he fled before anything was taken.",
     6.6, 5.7, "violence/robbery"),
    # 18 - hit and run (community)
    ("Hank revved his engine and aimed his car at a pedestrian he had argued with on the sidewalk. The pedestrian was struck head-on and died at the hospital that night.",
     "Hank revved his engine and aimed his car at a pedestrian he had argued with on the sidewalk. The pedestrian leapt aside in time and Hank crashed into a pole instead.",
     7.0, 6.2, "community/vehicle"),
    # 19 - spreading disease (healthcare)
    ("A nurse intentionally pricked herself with a contaminated needle and then drew a patient's blood. The patient contracted a deadly disease and died within a year.",
     "A nurse intentionally pricked herself with a contaminated needle and then drew a patient's blood. The patient was unaffected because the test sample was discarded promptly.",
     6.9, 6.0, "healthcare/violence"),
    # 20 - sabotaged climbing (community)
    ("Karl loosened the bolts on a climbing wall before his rival's lead climb to cause a deadly fall. The bolts gave way mid-route and his rival fell and broke her back.",
     "Karl loosened the bolts on a climbing wall before his rival's lead climb to cause a deadly fall. A safety inspector noticed the tampering and re-tightened them in time.",
     6.9, 6.1, "community/sabotage"),
    # 21 - elevator wires
    ("Tim cut the emergency cable on the building's elevator hoping his unpopular boss would fall to his death. The cable failed during use and his boss died of the impact.",
     "Tim cut the emergency cable on the building's elevator hoping his unpopular boss would fall to his death. A backup brake activated and prevented any harm to passengers.",
     7.0, 6.2, "workplace/violence"),
    # 22 - acid attack
    ("A jilted lover threw acid at his ex-girlfriend's face on the street to disfigure her for life. She suffered severe burns and permanent damage to her sight.",
     "A jilted lover threw acid at his ex-girlfriend's face on the street to disfigure her for life. She ducked behind a friend and was completely unharmed by the splash.",
     7.0, 6.3, "violence/assault"),
    # 23 - rigged ballot (financial/political)
    ("A campaign aide rigged absentee ballots to flip an election outcome for his preferred candidate. The fraud succeeded and an entire mayoral race was decided by it.",
     "A campaign aide rigged absentee ballots to flip an election outcome for his preferred candidate. An auditor detected the irregularity before the ballots were counted.",
     6.5, 5.6, "political/fraud"),
    # 24 - tainted product (financial)
    ("A manufacturer knowingly shipped baby formula contaminated with melamine to boost protein readings. Hundreds of infants fell critically ill before recalls began.",
     "A manufacturer knowingly shipped baby formula contaminated with melamine to boost protein readings. A whistleblower halted distribution before any cans reached shelves.",
     6.9, 6.0, "financial/product"),
    # 25 - drowning push (community)
    ("Reggie pushed a swimmer off a dock at night, intending to let him drown while drunk friends laughed. The swimmer hit his head, sank, and was found dead at dawn.",
     "Reggie pushed a swimmer off a dock at night, intending to let him drown while drunk friends laughed. A passing kayaker pulled the swimmer out before he could go under.",
     6.9, 6.0, "community/violence"),
    # 26 - bomb threat (school)
    ("A student planted a homemade explosive in the school cafeteria intending to harm peers. The device detonated and four classmates were severely injured.",
     "A student planted a homemade explosive in the school cafeteria intending to harm peers. A janitor noticed wires and bomb squad disabled the device safely.",
     7.0, 6.3, "school/violence"),
    # 27 - kidnap attempt (community)
    ("Wade grabbed a child outside a school gate, intending to abduct her into his van. He drove off with her and police found her bound in his basement a week later.",
     "Wade grabbed a child outside a school gate, intending to abduct her into his van. A teacher screamed and tackled him and she ran free without injury.",
     7.0, 6.4, "violence/kidnap"),
    # 28 - rigged scale (financial)
    ("A jeweler swapped a real diamond for a cubic zirconia in a customer's resetting and pocketed the real stone. The fraud cost the customer a priceless family heirloom.",
     "A jeweler swapped a real diamond for a cubic zirconia in a customer's resetting and pocketed the real stone. A second appraiser caught the swap before pickup.",
     6.3, 5.4, "financial/theft"),
    # 29 - sabotage plane (workplace)
    ("A mechanic drained brake fluid from a small plane hoping the pilot, his rival, would crash. The pilot crashed on landing and was killed by the impact.",
     "A mechanic drained brake fluid from a small plane hoping the pilot, his rival, would crash. A pre-flight check caught the issue and the pilot landed without harm.",
     6.9, 6.1, "workplace/violence"),
    # 30 - online harassment (online)
    ("A stalker posted a victim's home address with threats on a public forum hoping someone would attack her. A reader showed up and assaulted her at home that night.",
     "A stalker posted a victim's home address with threats on a public forum hoping someone would attack her. The post was reported and removed within minutes of upload.",
     6.7, 5.8, "online/stalking"),
    # 31 - assault attempt (workplace)
    ("Grant attacked his boss with a hammer in the break room intending to cause serious injury. He struck her on the head and she was hospitalized with a fractured skull.",
     "Grant attacked his boss with a hammer in the break room intending to cause serious injury. A coworker disarmed him before the first swing connected at all.",
     6.9, 6.1, "workplace/violence"),
    # 32 - destroyed evidence (legal)
    ("A defendant set fire to a warehouse to destroy crucial evidence in a fraud case against him. The records burned to ash and the prosecution's case collapsed.",
     "A defendant set fire to a warehouse to destroy crucial evidence in a fraud case against him. The fire was extinguished in time and the records were preserved.",
     6.3, 5.4, "legal/cover-up"),
    # 33 - vehicle sabotage (community)
    ("Owen punctured a stranger's brake line at a parking lot intending to cause a deadly crash. The stranger lost control on the freeway and was killed in the collision.",
     "Owen punctured a stranger's brake line at a parking lot intending to cause a deadly crash. The leak was small and the driver coasted safely to a service station.",
     6.9, 6.0, "community/sabotage"),
    # 34 - kidnap and ransom (community)
    ("A criminal seized a child outside her home intending to ransom her to a wealthy family. The child was held for a week before she was finally released for payment.",
     "A criminal seized a child outside her home intending to ransom her to a wealthy family. Police intercepted the kidnapper minutes after the grab and rescued her.",
     7.0, 6.3, "violence/kidnap"),
    # 35 - mass spam (online)
    ("A scammer launched a wire-fraud campaign targeting elderly veterans across the country one weekend. Hundreds were defrauded and the total loss exceeded a million dollars.",
     "A scammer launched a wire-fraud campaign targeting elderly veterans across the country one weekend. Spam filters intercepted the entire batch before any user saw it.",
     6.4, 5.5, "online/fraud"),
    # 36 - food tampering (public space)
    ("A worker intentionally introduced a pathogen into a salad bar at a busy lunch buffet downtown. Dozens of patrons fell seriously ill and several required hospitalization.",
     "A worker intentionally introduced a pathogen into a salad bar at a busy lunch buffet downtown. A health inspector closed the buffet minutes before it opened to customers.",
     6.8, 6.0, "public/violence"),
    # 37 - sword attack (community)
    ("An attacker swung a machete at a bystander outside a temple to wound him for his beliefs. He struck the bystander across the arm, severing tendons and nerves.",
     "An attacker swung a machete at a bystander outside a temple to wound him for his beliefs. The bystander stepped back and the blade missed him entirely.",
     7.0, 6.2, "violence/assault"),
    # 38 - drowning baby (family)
    ("A mother held her newborn under the bathwater after a postpartum delusion convinced her to harm him. The infant stopped moving and was pronounced dead at the hospital.",
     "A mother held her newborn under the bathwater after a postpartum delusion convinced her to harm him. Her partner walked in, pulled the baby out, and revived him in time.",
     6.9, 6.0, "family/violence"),
    # 39 - financial fraud (financial)
    ("An accountant cooked the books to embezzle a million dollars from her elderly client over two years. The client was bankrupted and lost her home and savings.",
     "An accountant cooked the books to embezzle a million dollars from her elderly client over two years. An auditor caught the irregularities before any funds left the trust.",
     6.5, 5.6, "financial/embezzle"),
    # 40 - cyberattack (online)
    ("A foreign agent deployed a malware payload designed to disable a city's emergency response systems. The 911 system went dark for hours and several callers died waiting.",
     "A foreign agent deployed a malware payload designed to disable a city's emergency response systems. A defender isolated the breach before any production system was hit.",
     6.9, 6.0, "online/infrastructure"),
    # 41 - hospital lock (healthcare)
    ("An attacker locked the doors of a maternity ward and started a small fire intending to cause panic. Two newborns suffered smoke injuries before the door was forced open.",
     "An attacker locked the doors of a maternity ward and started a small fire intending to cause panic. A nurse smelled smoke and put it out before any infant was harmed.",
     7.0, 6.2, "healthcare/violence"),
    # 42 - destroyed art (community)
    ("A vandal threw acid on a priceless painting at a public gallery to protest the museum's funding sources. The artwork was destroyed beyond restoration and the gallery shuttered.",
     "A vandal threw acid on a priceless painting at a public gallery to protest the museum's funding sources. A glass cover intercepted the splash and the artwork was unharmed.",
     5.9, 5.0, "community/vandalism"),
    # 43 - elderly fraud (financial)
    ("A caregiver siphoned the savings out of her elderly client's accounts pretending to handle bills. The client lost her home and was forced into a state nursing facility.",
     "A caregiver siphoned the savings out of her elderly client's accounts pretending to handle bills. The bank flagged the transfers and reversed them before any harm was done.",
     6.6, 5.7, "financial/elder-abuse"),
    # 44 - drugged drink (school)
    ("A student crushed sedatives into a classmate's drink at a party hoping to leave him helpless. The classmate passed out and woke up with no memory of the entire night.",
     "A student crushed sedatives into a classmate's drink at a party hoping to leave him helpless. The classmate sniffed the drink, became suspicious, and poured it out untouched.",
     6.6, 5.7, "school/drugs"),
    # 45 - explosive device (workplace)
    ("A disgruntled employee planted a small explosive under a manager's desk to cause harm. The device detonated and the manager lost his hearing in one ear permanently.",
     "A disgruntled employee planted a small explosive under a manager's desk to cause harm. A security sweep found the device and bomb squad disarmed it cleanly.",
     6.9, 6.0, "workplace/violence"),
    # 46 - poison the dog (community)
    ("A neighbor laced meatballs with antifreeze and tossed them over a fence to kill a barking dog. The dog ate the bait, suffered seizures, and died at the emergency vet.",
     "A neighbor laced meatballs with antifreeze and tossed them over a fence to kill a barking dog. The owner noticed the bait and disposed of it before the dog could reach it.",
     6.7, 5.8, "community/animal"),
    # 47 - rigged scaffolding (workplace)
    ("A foreman intentionally loosened scaffolding bolts at a job site so a hated worker would fall. The worker fell from the second story and fractured his pelvis and spine.",
     "A foreman intentionally loosened scaffolding bolts at a job site so a hated worker would fall. A safety officer caught the rigging at morning inspection and repaired it.",
     6.9, 6.0, "workplace/violence"),
    # 48 - virus release (online)
    ("A bioterror sympathizer released a manufactured virus into a busy subway tunnel during rush hour. Dozens of commuters contracted the illness and three later died of it.",
     "A bioterror sympathizer released a manufactured virus into a busy subway tunnel during rush hour. The aerosol device malfunctioned and no viable particles entered the air.",
     7.0, 6.2, "violence/bioterror"),
    # 49 - knife in dark (community)
    ("An assailant stabbed a jogger from behind on a dark trail intending to kill her over a grudge. The jogger collapsed from blood loss and died before paramedics arrived.",
     "An assailant stabbed a jogger from behind on a dark trail intending to kill her over a grudge. Her thick winter coat blunted the strike and she escaped uninjured.",
     7.0, 6.1, "violence/assault"),
    # 50 - destroyed prototype (workplace)
    ("A spy smashed a rival firm's billion-dollar prototype with a hammer in the secure lab. The device was destroyed and the company's flagship product launch was canceled.",
     "A spy smashed a rival firm's billion-dollar prototype with a hammer in the secure lab. The reinforced casing held and the device was undamaged by the attack.",
     6.4, 5.5, "workplace/sabotage"),
]


# =========================================================================
# CONDITION 3: MORAL vs CONVENTIONAL violation (50 pairs)
# =========================================================================
# Literature: Turiel (1983), Nichols (2002). Moral violations rated wrong
# regardless of rules; conventional violations rated wrong only by social
# agreement. Moral typically ~5-7, conventional ~2-4.

NORM_PAIRS = [
    # 1 - school (lunch theft vs raising hand)
    ("During a school break, one student took another student's lunch from his locker. She threw the lunch in the trash so the other child would go hungry that day.",
     "During a school break, one student stood up to address the class. She spoke without raising her hand first, as the teacher had instructed at the start of the year.",
     6.0, 2.4, "school"),
    # 2 - workplace (stealing wallet vs dress code)
    ("A new employee saw a coworker's wallet left at her desk. He took the cash from the wallet during the morning meeting downstairs.",
     "A new employee arrived for his first day at the office. He wore casual sneakers and shorts, even though the dress code clearly required business attire.",
     6.2, 2.5, "workplace"),
    # 3 - restaurant (food in pocket vs tipping)
    ("A diner visited a buffet restaurant downtown. He secretly slipped expensive food into his bag, intending to walk out without paying the buffet fee at the register.",
     "A diner finished his meal at a restaurant downtown. He chose to leave the table without leaving any tip for the waiter who had served him.",
     5.4, 2.6, "public-restaurant"),
    # 4 - public transit (cutting line vs queueing custom)
    ("A commuter waited at the boarding line for the morning train. He pushed an elderly rider out of his way so he could grab the last available seat.",
     "A commuter waited at the boarding line for the morning train. He stepped in front of others instead of waiting his turn, against the customary order.",
     5.5, 2.8, "public-transit"),
    # 5 - school (cheat vs gum)
    ("A student took a final exam in class. He copied a classmate's answers off her test, intending to claim her work as his own for the grade.",
     "A student attended a regular lecture in class. He chewed bubble gum loudly throughout, even though chewing gum had been clearly forbidden by every teacher.",
     5.7, 2.3, "school"),
    # 6 - public space (push vs hat in church)
    ("A man arrived at a busy park with his takeaway lunch. He shoved a stranger off a park bench so he could sit down on a sunny afternoon outdoors.",
     "A man attended a wedding service in a church. He wore his hat throughout the ceremony, violating the longstanding custom of removing headwear in the chapel.",
     5.4, 2.0, "public-religious"),
    # 7 - family (steal from grandma vs no elbows)
    ("A teenager visited her grandmother for the afternoon. She took several twenty-dollar bills from the older woman's purse while she napped on the couch.",
     "A teenager sat down for a family meal at home. She put her elbows squarely on the dinner table, against the table manners her parents had always emphasized.",
     5.7, 1.8, "family"),
    # 8 - online (harass vs language register)
    ("A user joined a public forum online. He posted threatening private photos of his ex-girlfriend to embarrass her in front of her friends online.",
     "A user joined a formal academic forum online. He posted using casual slang and lowercase grammar, against the polished standard English clearly expected there.",
     6.6, 2.0, "online"),
    # 9 - community (steal mail vs flag etiquette)
    ("A man approached his neighbor's mailbox at the curb. He opened the box and pocketed several letters that contained checks and personal correspondence over a weekend.",
     "A man raised a flag at his home for a public holiday. He flew the national flag upside down, against the established etiquette of correct orientation on flagpoles.",
     6.0, 2.4, "community"),
    # 10 - friendship (steal phone vs eye contact)
    ("A college student visited her friend's apartment for a quiet study session. She took her friend's expensive phone off the coffee table and pocketed it before leaving.",
     "A college student visited her friend's apartment for dinner with the parents. She avoided making eye contact with the parents, against the social custom expected of young guests.",
     5.9, 2.2, "friendship"),
    # 11 - workplace (sexual harassment vs casual Friday)
    ("A manager interacted with a junior employee in the break room. He groped her after she had repeatedly told him to keep his hands to himself at work.",
     "A manager came to the office every Friday in casual attire. He wore a faded band T-shirt, against the company's longstanding rule that all Fridays remain business-casual.",
     6.9, 1.8, "workplace"),
    # 12 - school (bully vs name)
    ("A child played near the playground swing during recess. He pushed a smaller classmate off the swing so hard that the smaller one scraped his face on the concrete.",
     "A child raised her hand to ask a question in front of the entire class. She called her teacher by his first name, against the school custom of addressing staff by last name.",
     6.0, 2.0, "school"),
    # 13 - healthcare (faked test vs scrubs)
    ("A doctor examined a wealthy patient who wanted disability benefits. He signed off on a fabricated blood test for the patient, who was not actually ill at all.",
     "A doctor did his usual rounds in the hospital ward. He wore casual jeans instead of medical scrubs, against the hospital's clinical dress code for clinicians.",
     6.4, 2.4, "healthcare"),
    # 14 - financial (lie on taxes vs salutation)
    ("An accountant prepared her client's tax return for the year. She filed the return with fabricated business deductions, intending to defraud the government of tax revenue.",
     "An accountant wrote her usual work emails to clients. She signed each message with a first-name salutation, against the firm's etiquette of using titles and surnames.",
     6.0, 1.9, "financial"),
    # 15 - family (lie cancer vs dress)
    ("A man spoke with his elderly mother about her old dog. He falsely told her the dog had died of cancer, when in fact he had given the dog away to a stranger.",
     "A man attended his sister's wedding in a formal black suit. He wore brown shoes with the suit, against the etiquette of matching the leather color of accessories.",
     5.0, 1.7, "family"),
    # 16 - community (vandalism vs lawn)
    ("A teenager grew angry over a property line dispute at dusk. He smashed three of his neighbor's car windows with a baseball bat after a long argument outside.",
     "A teenager spent his Sunday morning doing yard work at home. He mowed his lawn very short, against the neighborhood's longstanding custom of keeping yards untouched on Sundays.",
     6.1, 1.7, "community"),
    # 17 - online (revenge porn vs caps)
    ("An ex-partner uploaded private intimate videos of his former girlfriend. He posted them to a porn site under her full name to humiliate her publicly online.",
     "An ex-partner posted across multiple group chats every evening. He used all capital letters in his messages, against the netiquette norm of avoiding the appearance of shouting.",
     6.9, 2.0, "online"),
    # 18 - school (cheat exam vs sit posture)
    ("A graduate student wanted a competitive scholarship she could not earn. She paid another person to take her exam under her name so she would qualify falsely.",
     "A graduate student attended a formal seminar in the lecture hall. She sat with one leg over her chair, against the school's posture guidelines for seminar attendance.",
     6.3, 1.7, "school"),
    # 19 - workplace (skim vs greeting)
    ("A cashier worked the register at a small store. She skimmed twenty dollars from the register every shift, intending to pocket the cash without management noticing.",
     "A cashier greeted customers at a small store. She did not use the standard scripted phrase on entry, against the chain's policy of opening every interaction warmly.",
     5.7, 1.9, "workplace"),
    # 20 - friendship (gossip secret vs vegetarian)
    ("A friend learned a confidential medical diagnosis from another friend. She leaked it to a mutual acquaintance after promising to keep the matter completely private forever.",
     "A friend hosted a casual dinner party for several guests. She served meat without asking guests about preferences, against the etiquette of accommodating dietary customs.",
     5.7, 2.2, "friendship"),
    # 21 - public (graffiti vs music)
    ("A man held a grudge against a specific neighbor in the area. He spray-painted insulting messages about her across the side of her house in the middle of the night.",
     "A man parked his car outside his own home late at night. He played loud music from the car at midnight, against the neighborhood's longstanding quiet-hours custom.",
     5.9, 2.5, "community"),
    # 22 - school (hit teacher vs no chair)
    ("A student argued heatedly with his math teacher during class. He threw a chair at her, hitting her on the shoulder and bruising her badly that morning.",
     "A student attended the math lecture in his usual classroom. He sat on the floor instead of in his assigned seat, against the school's classroom seating policy.",
     6.5, 1.8, "school"),
    # 23 - public (pickpocket vs queue talk)
    ("A pickpocket worked in a busy plaza one afternoon. He lifted a tourist's full wallet from her bag while she was distracted looking at directions on her phone.",
     "A tourist waited in a long museum queue one afternoon. She chatted loudly on her phone the whole time, against the soft-voice custom expected indoors.",
     6.2, 2.0, "public"),
    # 24 - online (catfish vs handle)
    ("A man created a fake online profile of a famous singer. He used the profile to lure a young fan into sending him intimate photos over a private chat thread.",
     "A man created a profile on a professional networking site. He used an unconventional handle there, against the convention of using a real legal name on the platform.",
     6.8, 1.7, "online"),
    # 25 - workplace (forge signature vs slacks)
    ("An employee wanted to authorize a payment that benefited her own consulting firm. She forged her supervisor's signature on the contract to push the payment through.",
     "An employee dressed for a client-facing meeting at the firm. She wore relaxed-fit slacks instead of tailored trousers, against the firm's longstanding policy on meeting attire.",
     6.4, 1.6, "workplace"),
    # 26 - family (slap kid vs no prayer)
    ("A parent grew furious during his eight-year-old son's tantrum. He slapped the boy repeatedly across the face, leaving visible red marks on the child's cheeks.",
     "A parent sat down for family dinner with his children one evening. He did not say grace beforehand, against the household's longstanding tradition of starting each meal with prayer.",
     6.4, 1.6, "family"),
    # 27 - community (run over cat vs lawn flag)
    ("A driver saw a stray cat sitting in the road on his commute. He intentionally ran over the cat for fun and laughed about it with his passenger as he drove on.",
     "A driver decorated his front lawn during a rival team's home week. He mounted a sports flag for his own team on the lawn, against the neighborhood's good-natured ribbing custom.",
     6.4, 1.5, "community"),
    # 28 - school (push down stairs vs uniform)
    ("A bully approached a smaller student at the end of the school day. He pushed the smaller one down a stairwell, causing him to sprain his wrist on the railing.",
     "A student attended classes in her standard school uniform shirt. She wore a non-uniform jacket over it all day, against the school's policy on standardized outerwear.",
     6.5, 1.7, "school"),
    # 29 - workplace (sexist remark vs casual phone)
    ("A senior partner addressed his female associate in front of the team. He mocked her appearance and told her to wear shorter skirts to client meetings.",
     "An associate carried her phone with a casual sticker-covered case at the office. She used the case in client-facing meetings, against the firm's understated aesthetic norm.",
     6.6, 1.6, "workplace"),
    # 30 - public (rob elder vs shoes off)
    ("A robber approached an elderly woman waiting at a bus stop. He pushed her to the pavement and snatched her purse before disappearing into a side street.",
     "A guest arrived at a host's home for a casual evening visit. He kept his shoes on indoors, against the household custom of removing footwear at the entrance.",
     6.8, 1.7, "public"),
    # 31 - community (smash window vs lights)
    ("A vandal resented the youth programs run at the local community center. She smashed all the windows of the center late one evening to protest the work done there.",
     "A homeowner kept her holiday decorations up well into the new year. She left the lights on her house past the customary takedown date that most neighbors observed.",
     6.0, 1.5, "community"),
    # 32 - online (impersonation vs emoji)
    ("A user created a fake online dating profile of a real woman. She used the profile to lure men into giving up money to a fraud they thought was her.",
     "A user wrote business emails for the firm every day. She added casual emojis to every message, against the conventional tone expected in formal professional correspondence.",
     6.5, 1.8, "online"),
    # 33 - friendship (stole inheritance vs gift wrap)
    ("A close friend housesat for another friend over the weekend. He stole a valuable family heirloom from the apartment and pawned it for cash that same week.",
     "A close friend attended another friend's birthday party at her home. He brought an unwrapped gift, against the custom of presenting wrapped presents at celebrations.",
     6.4, 1.6, "friendship"),
    # 34 - public (drive on sidewalk vs walking left)
    ("A driver grew impatient with downtown traffic on his commute. He swerved his car onto a busy sidewalk, hitting a pedestrian who was knocked down and badly bruised.",
     "A walker headed down a wide sidewalk on her commute. She chose the left side of the walkway rather than the right, against the local custom of pedestrians keeping to one side.",
     6.5, 1.7, "public"),
    # 35 - school (set fire vs gym clothes)
    ("A student went into the school bathroom one cold rainy morning. He lit a small fire that spread to the trash bins, forcing an evacuation that frightened students into the rain.",
     "A student attended a physical education session in the gym. He wore street clothes during the class instead of the required uniform, against the school's policy on PE attire.",
     6.5, 1.6, "school"),
    # 36 - workplace (extortion vs nameplate)
    ("A manager pulled his junior accountant aside before her bonus arrived. He threatened to fire her unless she handed over her year-end bonus directly to him in cash.",
     "An employee personalized her cubicle to feel more at home. She customized her nameplate with cartoons and stickers, against the office's policy on standardized desk identification.",
     6.7, 1.5, "workplace"),
    # 37 - healthcare (denied care vs accent)
    ("A clinic owner saw a homeless man bleeding heavily at the front desk. He refused emergency treatment because the man could not pay upfront cash for the visit.",
     "A nurse explained a procedure to a patient at the bedside. She used a regional accent throughout, against the hospital's preference for neutral professional speech with patients.",
     6.7, 1.5, "healthcare"),
    # 38 - public (push elder off train vs squeeze in)
    ("A commuter rushed to catch the downtown train as the doors began closing. He shoved an elderly passenger out of his way so he could squeeze in before the train departed.",
     "A commuter chose not to wait for the next train at the station. He squeezed onto an already-packed train, against the city's transit etiquette for crowded cars.",
     5.8, 2.3, "public"),
    # 39 - family (lock in basement vs talk during)
    ("A man's teenage daughter came home twenty minutes past her curfew. He locked her in the dark basement for two days as punishment for the late arrival.",
     "A man attended Thanksgiving dinner with his family at home. He spoke during the family prayer, against his household's longstanding custom of silent reflection in that moment.",
     6.7, 1.5, "family"),
    # 40 - financial (Ponzi vs handshake)
    ("An investment advisor ran a Ponzi scheme over several quiet years at his firm. He defrauded hundreds of retirees out of their entire life savings during that period.",
     "An investment advisor greeted his clients during their meetings at the firm. He waved instead of shaking hands, against the customary norm in the industry's meeting greetings.",
     6.9, 1.6, "financial"),
    # 41 - community (poison dog vs decor)
    ("A man had a small disagreement with his neighbor about her dog's barking. He poisoned the dog with antifreeze, killing the family pet within a few hours that day.",
     "A man wanted to repaint the front door of his home in a historic district. He chose a bright orange color, against the area's longstanding custom of muted traditional colors.",
     6.8, 1.5, "community"),
    # 42 - online (death threat vs reply-all)
    ("A user grew angry at a journalist over a controversial article. He sent a credible death threat naming the victim's home address to the journalist online.",
     "A user replied to a department email about a casual social event at work. He used reply-all to the entire team, against the office's netiquette of limiting replies to the sender.",
     6.9, 1.6, "online"),
    # 43 - school (steal phone vs locker decor)
    ("A student walked past another student's locker between classes. He rifled through it and pocketed her phone and headphones while pretending to look for his own books.",
     "A student personalized her own locker at school. She decorated it with stickers and posters inside and out, against the school's policy on plain unadorned lockers in the hallways.",
     6.0, 1.5, "school"),
    # 44 - public (drunk drive crash vs jaywalk)
    ("A drunk driver sped through an intersection on his way home. He ran a red light at high speed and slammed into a bicyclist crossing legally on the green light.",
     "A pedestrian crossed the street on a quiet residential block one afternoon. She crossed outside the marked crosswalk, against the city's traffic convention for safe crossing.",
     6.7, 2.3, "public"),
    # 45 - friendship (slash tires vs RSVP)
    ("A jealous friend learned about an upcoming wedding the night before the ceremony. She slashed all four tires on her best friend's new car, knowing the car was needed for the ceremony.",
     "A close friend received a wedding invitation in the mail many weeks before the event. He RSVPed very late, against the social custom of replying with attendance plans well before the deadline.",
     6.3, 1.7, "friendship"),
    # 46 - workplace (sabotage drug vs eyebrows)
    ("A pharmaceutical worker wanted to derail a competitor's FDA approval bid. She swapped active drug doses for placebos in the competitor's clinical trial that quarter.",
     "A pharmaceutical worker prepared for camera-facing executive meetings at the firm. She did not maintain neatly groomed eyebrows, against the firm's grooming guideline for executives.",
     6.6, 1.5, "workplace"),
    # 47 - community (assault vs no parade)
    ("A man argued heatedly with his neighbor over a fence line in the yard. He punched the neighbor in the face, knocking out two of his front teeth during the dispute.",
     "A homeowner stayed home during the annual community parade in the summer. He did not attend the parade, against the neighborhood's longstanding custom of communal participation each summer.",
     6.6, 1.5, "community"),
    # 48 - online (deepfake vs hashtag)
    ("A user created sexual deepfake images of a classmate using online tools. He shared them in a private chat with several boys at her high school for a laugh.",
     "A user posted vacation photos for an influencer brand campaign that quarter. She left off the required brand hashtag, against the partner's contracted etiquette norm for sponsored posts.",
     6.8, 1.6, "online"),
    # 49 - family (forge will vs not eat)
    ("A son visited his dying mother in her last weeks of life. He forged her signature on a new will to cut his sister out of the inheritance entirely from the shared estate.",
     "A son attended his mother's Sunday dinner with the rest of the family. He did not eat the food she prepared for him, against his family's longstanding tradition of finishing the plate.",
     6.5, 1.6, "family"),
    # 50 - public (kick wheelchair vs hat indoors)
    ("A man walked past a stranger's wheelchair at the top of a steep ramp in a public mall. He kicked the wheels of the chair, causing the rider to tip and bruise her shoulder.",
     "A man sat down to dine in a formal restaurant one evening. He kept his baseball cap on at the table, against the etiquette norm of removing hats in fine dining establishments.",
     6.8, 1.7, "public"),
]


# =========================================================================
# CONDITION 4: MORAL-NEGATIVE vs NONMORAL-NEGATIVE (50 pairs)
# =========================================================================
# Literature: Greene et al., Young et al. — distinguishes pure valence from
# moral content. Both negative; only one has moral agent.

VALENCE_PAIRS = [
    # 1 - water supply
    ("A man deliberately poisoned the town's water supply because he was angry at the local government. Dozens of residents became severely ill.",
     "A natural arsenic deposit contaminated the town's water supply after weeks of heavy rains. Dozens of residents became severely ill.",
     6.9, 1.2, "community/water"),
    # 2 - building collapse
    ("A contractor cut steel beams from a high-rise to save costs, ignoring engineer warnings. The building collapsed and many residents died.",
     "An earthquake of unprecedented magnitude shook the high-rise built to current codes. The building collapsed and many residents died.",
     6.8, 1.5, "community/structural"),
    # 3 - wildfire
    ("An arsonist set fires across the dry hills to retaliate against a neighbor. The wildfires consumed homes and dozens of residents had to flee.",
     "A lightning strike during the dry season ignited fires across the hills. The wildfires consumed homes and dozens of residents had to flee.",
     6.6, 1.3, "environmental/fire"),
    # 4 - power outage
    ("A hacker disabled the regional power grid as part of a politically motivated cyberattack on the country. Hospitals lost power and several patients died.",
     "A solar storm overwhelmed the regional power grid during a rare geomagnetic event over the country. Hospitals lost power and several patients died.",
     6.7, 1.5, "infrastructure/cyber"),
    # 5 - poisoned formula
    ("A factory worker laced infant formula with melamine to inflate protein readings for higher pay. Hundreds of babies fell critically ill.",
     "A rare bacterial contaminant entered infant formula during a cooling system failure at the plant. Hundreds of babies fell critically ill.",
     6.9, 1.7, "product/contamination"),
    # 6 - airplane crash
    ("A mechanic skipped multiple maintenance steps and falsified records to clock out early. A passenger jet crashed and many lives were lost.",
     "A flock of large birds struck both engines of a passenger jet during takeoff. The plane crashed and many lives were lost.",
     6.8, 1.4, "transport/aviation"),
    # 7 - school injuries
    ("A teenager opened fire on classmates in the school cafeteria one Monday morning. Multiple students were killed and many more were wounded.",
     "A tornado tore through the school cafeteria one Monday morning. Multiple students were killed and many more were wounded.",
     7.0, 1.3, "school/disaster"),
    # 8 - cancer cluster
    ("A factory CEO knowingly discharged carcinogens into the local groundwater for years. Many residents in nearby neighborhoods developed cancer.",
     "A rare radon vein under the local groundwater leached into nearby wells for years. Many residents in nearby neighborhoods developed cancer.",
     6.8, 1.4, "environmental/cancer"),
    # 9 - fatal traffic
    ("A drunk driver ran a red light at high speed and slammed into a family car at the intersection. Both parents were killed instantly.",
     "A sudden microburst tore a tree across the intersection just as a family car drove through. Both parents were killed instantly.",
     6.9, 1.3, "transport/accident"),
    # 10 - epidemic
    ("A bioterrorist released a manufactured virus in a crowded transit station to spread illness. Hundreds in the city fell critically ill within days.",
     "A novel zoonotic virus crossed from bats into humans in a crowded transit district that month. Hundreds in the city fell critically ill within days.",
     6.9, 1.3, "biological/disease"),
    # 11 - financial ruin
    ("A CEO ran a Ponzi scheme that took the entire savings of thousands of retirees who trusted his firm. They were left destitute and homeless.",
     "A once-in-a-century market crash wiped out the entire pensions of thousands of retirees who had saved diligently. They were left destitute and homeless.",
     6.7, 1.4, "financial/loss"),
    # 12 - food poisoning
    ("A restaurant manager knowingly served undercooked chicken to a packed dining room to clear the backlog of orders. Many diners contracted severe food poisoning.",
     "A previously unknown bacterial strain contaminated fresh chicken before it reached the restaurant supply chain that week. Many diners contracted severe food poisoning.",
     6.4, 1.4, "public/food"),
    # 13 - bridge collapse
    ("A bridge inspector accepted bribes to certify a crumbling overpass without doing the required structural tests. The overpass collapsed during rush hour and killed many drivers.",
     "An undetected seismic shift cracked the foundations of an overpass over decades despite regular inspections. The overpass collapsed during rush hour and killed many drivers.",
     6.7, 1.5, "infrastructure/collapse"),
    # 14 - drowning
    ("A swim coach intentionally locked the pool gates during practice to scare a student who could not climb out. The student panicked and drowned in the deep end.",
     "A sudden electrical fault closed the pool gates during practice while a student was still in the water. The student panicked and drowned in the deep end.",
     7.0, 1.5, "public/water"),
    # 15 - infant injury
    ("A nanny intentionally shook a crying infant in her care until the baby went limp. The infant suffered permanent brain damage and could not walk later.",
     "A sudden severe seizure shook a crying infant in his crib without any warning. The infant suffered permanent brain damage and could not walk later.",
     6.9, 1.3, "family/infant"),
    # 16 - mass injury (concert)
    ("A militant detonated an improvised bomb at a packed outdoor concert downtown. Hundreds of fans were injured and many never recovered fully.",
     "A massive lightning strike split a stage scaffold at a packed outdoor concert downtown. Hundreds of fans were injured and many never recovered fully.",
     7.0, 1.4, "public/disaster"),
    # 17 - oil spill
    ("An oil executive ordered tank inspections to be skipped to save costs ahead of an audit. A storage tank ruptured and coastal wildlife was devastated.",
     "A rare meteorite strike split open a storage tank during a routine quiet shift at the terminal. Coastal wildlife was devastated by the resulting spill.",
     6.7, 1.5, "environmental/spill"),
    # 18 - workplace injuries
    ("A foreman ordered workers to ignore safety protocols on a high scaffold to meet a deadline ahead of a client visit. Multiple workers fell and were severely injured.",
     "A rare burst of seismic vibration rattled a high scaffold during a quiet shift ahead of a client visit. Multiple workers fell and were severely injured.",
     6.6, 1.5, "workplace/safety"),
    # 19 - dam failure
    ("A dam manager falsified inspection reports for years and ignored cracks in the spillway. The dam burst and washed away most of a downstream village.",
     "An extraordinary monsoon dumped record rainfall onto an aging dam over a few hours. The dam burst and washed away most of a downstream village.",
     6.8, 1.4, "environmental/dam"),
    # 20 - elderly neglect
    ("A nursing home owner cut nutritional staff to maximize profit margins, leaving residents without meals. Several elderly residents starved and one died.",
     "A devastating supply-chain collapse stopped all food deliveries to a nursing home for many days. Several elderly residents starved and one died.",
     6.9, 1.5, "healthcare/elder"),
    # 21 - mining disaster
    ("A mine owner removed key safety ventilation to cut costs despite engineer warnings. A methane explosion killed many trapped miners underground.",
     "An undetected pocket of methane intersected the active tunnel during normal operations. A methane explosion killed many trapped miners underground.",
     6.8, 1.6, "workplace/mining"),
    # 22 - house fire
    ("A landlord disabled the smoke detectors and locked the fire escapes to discourage subletting. A fire broke out at night and entire families perished.",
     "A rare lightning strike to a transformer started a fire in the building at night despite working alarms. A fire broke out at night and entire families perished.",
     6.9, 1.5, "community/fire"),
    # 23 - school bus
    ("A school bus driver took the wheel drunk on a winter morning despite many warnings. The bus skidded off the icy road and killed several children.",
     "A patch of black ice formed unexpectedly on the road one winter morning despite normal forecasts. The bus skidded off the icy road and killed several children.",
     7.0, 1.4, "transport/school"),
    # 24 - hospital deaths
    ("An administrator cut nurse staffing to dangerously low levels to boost the hospital's quarterly profits ahead of an audit. Many patients died from preventable complications.",
     "A virulent novel pathogen overwhelmed the hospital's wards over a few terrible days ahead of an audit. Many patients died from preventable complications.",
     6.8, 1.4, "healthcare/staffing"),
    # 25 - bullied to death
    ("A group of students relentlessly bullied a quiet classmate in person and online for months without remorse. The classmate took her own life early one morning.",
     "A rare aggressive cancer struck a quiet classmate without any warning and progressed quickly. The classmate died from the illness early one morning.",
     7.0, 1.2, "school/death"),
    # 26 - apartment shooting
    ("A gunman opened fire in a quiet apartment courtyard one summer evening targeting residents at random. Several residents were killed and many more were wounded.",
     "A massive freak hailstorm tore through a quiet apartment courtyard one summer evening without warning. Several residents were killed and many more were wounded.",
     7.0, 1.5, "violence/community"),
    # 27 - financial scam (online)
    ("A cybercriminal ran a romance scam targeting widowed seniors and drained the entire savings of dozens of trusting victims. Many victims lost their homes and dignity.",
     "An unprecedented currency collapse erased the entire savings of dozens of widowed seniors over a few weeks of chaos. Many victims lost their homes and dignity.",
     6.7, 1.5, "online/fraud"),
    # 28 - drunk boating
    ("A boat captain knowingly piloted a ferry while drunk in heavy fog across a busy lake at night. The ferry collided with another vessel and many drowned.",
     "A sudden dense fog and unforecasted squall hit a ferry crossing a busy lake at night. The ferry collided with another vessel and many drowned.",
     6.8, 1.4, "transport/water"),
    # 29 - elder abuse
    ("A caregiver routinely struck and verbally abused her bedridden client in the privacy of his home. The client suffered chronic pain and trauma until he died.",
     "A degenerative neurological disease progressively wracked the body of a bedridden client in his home. The client suffered chronic pain until he died.",
     6.9, 1.2, "family/elder"),
    # 30 - chemical leak
    ("A plant manager covered up a chemical leak and lied to inspectors to keep the plant running. Many workers and nearby residents were poisoned by the fumes.",
     "A rare seismic micro-fracture cracked a chemical storage seal during a routine quiet shift. Many workers and nearby residents were poisoned by the fumes.",
     6.8, 1.5, "environmental/chemical"),
    # 31 - building fire
    ("A building owner blocked the fire exits with stored furniture to maximize the rental space. A small fire trapped residents inside and several were killed.",
     "An electrical fault deep inside a wall ignited a small fire that filled the corridors with smoke. Residents were trapped inside and several were killed.",
     6.9, 1.5, "community/fire"),
    # 32 - patient infection
    ("A surgeon refused to wash his hands between back-to-back cases despite repeated warnings from his team. Several patients contracted dangerous bloodstream infections.",
     "A novel drug-resistant bacterial strain colonized the surgical suite that quarter despite proper hygiene. Several patients contracted dangerous bloodstream infections.",
     6.7, 1.6, "healthcare/infection"),
    # 33 - online radicalization
    ("A network of operatives radicalized lonely teenagers through curated extremist videos pushed into their feeds. Several teens later carried out violent attacks at home.",
     "A complex neurological condition warped the perceptions of several lonely teenagers in unrelated towns. Several teens later carried out violent acts at home.",
     6.8, 1.6, "online/violence"),
    # 34 - dog attack
    ("A breeder trained his dogs to attack on command and unleashed them on a man during a feud. The man was severely mauled and required reconstructive surgery.",
     "An untreated rabies infection drove a pack of feral dogs to attack a man crossing a remote field at dusk. The man was severely mauled and required reconstructive surgery.",
     6.8, 1.5, "community/animal"),
    # 35 - factory fire
    ("A factory owner locked emergency exits to prevent unauthorized breaks during long sweatshop shifts. A fire broke out and dozens of workers died unable to escape.",
     "A short circuit in the lighting wires sparked a fire late at night in the otherwise empty factory. The fire spread and dozens of workers later perished.",
     6.9, 1.5, "workplace/fire"),
    # 36 - tainted alcohol
    ("A bootlegger sold methanol-laced spirits to unsuspecting partygoers to maximize his profits during the holiday. Several drinkers went blind and a few died.",
     "A rare yeast strain converted a batch of homemade spirits into methanol during the holiday season. Several drinkers went blind and a few died.",
     6.7, 1.5, "product/alcohol"),
    # 37 - hospital fire
    ("A maintenance worker disabled the hospital's sprinkler system to bypass an audit involving plumbing repairs. A small fire spread quickly and several patients died.",
     "An ultra-rare valve failure disabled the hospital's sprinkler system during a routine quiet morning. A small fire spread quickly and several patients died.",
     6.8, 1.6, "healthcare/fire"),
    # 38 - kidnapping
    ("An organized crime group abducted children from quiet neighborhoods to traffic them across the border for forced labor. Most of the children were never recovered.",
     "A massive flash flood swept children playing in dry riverbeds in quiet neighborhoods across the border. Most of the children were never recovered.",
     7.0, 1.4, "violence/trafficking"),
    # 39 - drug overdose
    ("A pharmaceutical executive concealed addiction risks of a new painkiller from doctors and patients alike. Thousands of users died from overdoses across the country.",
     "A genetically unusual liver enzyme caused fatal painkiller buildups in a population of unsuspecting patients. Thousands of users died from overdoses across the country.",
     6.9, 1.4, "healthcare/drugs"),
    # 40 - acid burn
    ("An attacker hurled a flask of industrial acid at his ex-fiancée's face on her way home from work. She suffered severe burns and permanent loss of vision.",
     "An industrial acid spill from a passing truck splashed onto a pedestrian walking home from work after a road accident. She suffered severe burns and permanent loss of vision.",
     7.0, 1.6, "violence/burn"),
    # 41 - water poisoning
    ("A factory manager ordered untreated effluent dumped into the river that supplied a nearby village's drinking water. Many villagers became chronically ill.",
     "A natural algal bloom released potent toxins into the river that supplied a nearby village's drinking water. Many villagers became chronically ill.",
     6.7, 1.4, "environmental/water"),
    # 42 - subway crash
    ("A subway operator falsified his sleep records and worked a triple shift in heavy traffic conditions. He missed a signal and the train collided with another in the tunnel.",
     "A rare lightning surge fried a key signal in heavy traffic conditions in the busy subway tunnel. The signal failed and the train collided with another in the tunnel.",
     6.7, 1.5, "transport/subway"),
    # 43 - infant deaths
    ("A maternity manager turned off neonatal monitor alarms at night so staff could rest between rounds. Several premature infants died from undetected complications.",
     "A sudden cluster of genetic anomalies caused several premature infants to deteriorate at night in the unit. Several premature infants died from undetected complications.",
     7.0, 1.5, "healthcare/neonatal"),
    # 44 - school collapse
    ("A school official used substandard cement during renovations to pocket the savings for personal expenses. A wing of the school collapsed and students inside were killed.",
     "An ultra-rare clay liquefaction beneath the school occurred after weeks of unusual rainfall and groundwater changes. A wing of the school collapsed and students inside were killed.",
     6.9, 1.5, "school/structural"),
    # 45 - workplace assault
    ("A man brought a knife to the office and stabbed his former supervisor in front of stunned coworkers in the lobby. The supervisor died of his injuries at the hospital.",
     "A massive ceiling tile dislodged and struck a supervisor in front of stunned coworkers in the office lobby. The supervisor died of his injuries at the hospital.",
     7.0, 1.4, "workplace/violence"),
    # 46 - drug shortage
    ("A drug manufacturer halted production of a life-saving medication to manipulate prices and squeeze more profit out of the contract. Many patients died waiting for their doses.",
     "A natural disaster destroyed the sole factory producing a life-saving medication and disrupted the global supply chain. Many patients died waiting for their doses.",
     6.8, 1.6, "healthcare/drugs"),
    # 47 - airline near-miss
    ("An air traffic controller intentionally misdirected a flight during a labor dispute to make a point about staffing. The plane skimmed terrain and barely landed safely.",
     "A rare radar ghost from solar interference misdirected a flight during otherwise calm afternoon conditions. The plane skimmed terrain and barely landed safely.",
     6.5, 1.4, "transport/atc"),
    # 48 - food chain contamination
    ("An agribusiness executive bribed inspectors to ignore widespread pesticide overuse across the entire crop. Many shoppers and their children became chronically ill.",
     "An invasive fungus released natural toxins across the entire crop during an unusually warm growing season. Many shoppers and their children became chronically ill.",
     6.8, 1.5, "food/pesticide"),
    # 49 - house collapse
    ("A landlord ignored a tenant's repeated written warnings about a sagging ceiling and shifting walls. The ceiling collapsed at night and several tenants were crushed.",
     "An extremely rare termite super-colony hollowed out the load-bearing beams over decades unnoticed. The ceiling collapsed at night and several tenants were crushed.",
     6.9, 1.5, "community/structural"),
    # 50 - boating accident
    ("A boat operator overloaded a small passenger ferry by stuffing in extra paying passengers despite the safety placard. The ferry capsized in the river and many drowned.",
     "A sudden underwater landslide triggered a freak wave on the river beneath a small passenger ferry. The ferry capsized in the river and many drowned.",
     6.7, 1.5, "transport/water"),
]


# =========================================================================
# BUILD OUTPUT
# =========================================================================

def word_count(s):
    return len(s.split())


def build_entry(condition, subcondition, pair_idx, ab, text, target, note):
    pair_id = f"{condition}_{pair_idx:03d}"
    item_id = f"{pair_id}{ab}"
    return {
        "id": item_id,
        "condition": condition,
        "subcondition": subcondition,
        "pair_id": pair_id,
        "text": text,
        "target_judgment": target,
        "notes": note,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []

    # Intent
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(INTENT_PAIRS, 1):
        entries.append(build_entry("intent", "intentional", i, "a", txt_a, t_a, f"intentional harm; {note}"))
        entries.append(build_entry("intent", "accidental",   i, "b", txt_b, t_b, f"accidental harm, same outcome; {note}"))

    # Outcome
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(OUTCOME_PAIRS, 1):
        entries.append(build_entry("outcome", "completed", i, "a", txt_a, t_a, f"completed harm; {note}"))
        entries.append(build_entry("outcome", "attempted", i, "b", txt_b, t_b, f"attempted harm, same intent, no harm; {note}"))

    # Norm type
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(NORM_PAIRS, 1):
        entries.append(build_entry("norm_type", "moral_violation",         i, "a", txt_a, t_a, f"moral violation (harm/fairness/rights); {note}"))
        entries.append(build_entry("norm_type", "conventional_violation",  i, "b", txt_b, t_b, f"conventional violation (etiquette/custom); {note}"))

    # Morality (vs valence)
    for i, (txt_a, txt_b, t_a, t_b, note) in enumerate(VALENCE_PAIRS, 1):
        entries.append(build_entry("morality", "moral_negative",    i, "a", txt_a, t_a, f"negative outcome with moral agent; {note}"))
        entries.append(build_entry("morality", "nonmoral_negative", i, "b", txt_b, t_b, f"equally negative outcome, natural cause; {note}"))

    # Quality checks
    print(f"Total entries: {len(entries)}")
    by_cond = {}
    for e in entries:
        by_cond.setdefault(e["condition"], []).append(e)
    for c, lst in by_cond.items():
        wcs = [word_count(e["text"]) for e in lst]
        print(f"  {c}: {len(lst)} items, word count min={min(wcs)} max={max(wcs)} mean={sum(wcs)/len(wcs):.1f}")

    # Check pair length balance
    print("\nLength-mismatch report (pairs with > 5 word difference):")
    mismatch_count = 0
    pairs = {}
    for e in entries:
        pairs.setdefault(e["pair_id"], []).append(e)
    for pid, items in pairs.items():
        if len(items) != 2:
            print(f"  WARNING: pair {pid} has {len(items)} items")
            continue
        wa, wb = word_count(items[0]["text"]), word_count(items[1]["text"])
        if abs(wa - wb) > 5:
            print(f"  {pid}: {wa} vs {wb} words (diff={abs(wa-wb)})")
            mismatch_count += 1
    print(f"  Total mismatches > 5 words: {mismatch_count}")

    # Check duplicates
    seen_texts = set()
    dups = 0
    for e in entries:
        if e["text"] in seen_texts:
            print(f"  DUP: {e['id']}")
            dups += 1
        seen_texts.add(e["text"])
    print(f"\nDuplicate texts: {dups}")

    # Write JSONL
    with OUT_PATH.open("w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
