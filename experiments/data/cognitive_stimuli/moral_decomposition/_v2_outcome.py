"""Outcome condition v2 — 150 NEW pairs (IDs 051-200).

Each tuple: (text_a_completed, text_b_attempted, target_a, target_b, domain_note).

Both sentences share the same first sentence (identical intent + act); they
diverge ONLY in the second sentence (whether harm landed).

Per Cushman (2008), completed harm rated only slightly higher than attempted
when intent is held constant. Expected diffs ~0.6-1.0 on 1-7 scale.

Domains: failed crimes, intercepted attacks, prevented disasters, near-misses,
sabotage attempts caught, fraud caught at last minute, etc.
"""

OUTCOME_PAIRS_V2 = [
    # 51 - rooftop sniper at protest
    ("A militant set up a sniper rifle above a peaceful protest with intent to shoot the speaker. He fired and the speaker was struck in the chest and pronounced dead.",
     "A militant set up a sniper rifle above a peaceful protest with intent to shoot the speaker. Police snipers spotted him on a thermal scan and detained him before he could shoot.",
     6.9, 6.1, "violence/sniper"),
    # 52 - razor in apple
    ("Greta hid razor blades in apples she handed out to neighborhood children on Halloween night. Two children cut their mouths badly and were rushed to the emergency room.",
     "Greta hid razor blades in apples she handed out to neighborhood children on Halloween night. A vigilant parent X-rayed the apples and notified police before any child bit in.",
     6.9, 6.0, "community/poisoning"),
    # 53 - tampered baby crib
    ("Reggie loosened the slats on his ex-wife's baby crib with intent to harm his estranged daughter. The slats gave way during the night and the baby fell, suffering a concussion.",
     "Reggie loosened the slats on his ex-wife's baby crib with intent to harm his estranged daughter. The wife noticed the missing screws at bath time and replaced the crib before bedtime.",
     7.0, 6.2, "family/violence"),
    # 54 - swapped insulin
    ("A pharmacist swapped a diabetic teen's insulin for a saline solution to spite the teen's parents. The teen's blood sugar spiked at school and she collapsed into a diabetic coma.",
     "A pharmacist swapped a diabetic teen's insulin for a saline solution to spite the teen's parents. A nurse cross-checked the vial color at the school clinic and replaced it in time.",
     6.9, 6.0, "healthcare/sabotage"),
    # 55 - bombed mosque
    ("A bomber planted explosives outside a crowded mosque during prayers intending to massacre worshippers. The bomb detonated and dozens of worshippers were killed in the blast.",
     "A bomber planted explosives outside a crowded mosque during prayers intending to massacre worshippers. A worshipper noticed the package and security defused it before any detonation.",
     7.0, 6.4, "violence/terrorism"),
    # 56 - poisoned office water
    ("A disgruntled employee dripped a lethal toxin into the office water cooler intending to harm coworkers. Several coworkers drank from it and developed acute organ failure overnight.",
     "A disgruntled employee dripped a lethal toxin into the office water cooler intending to harm coworkers. A janitor smelled the chemical and shut the cooler before anyone drank from it.",
     6.9, 6.0, "workplace/poisoning"),
    # 57 - cut bus brake line
    ("A vandal cut the brake line of a city bus parked overnight intending to cause a deadly crash. The brakes failed at a steep hill and the bus collided with cars, killing two riders.",
     "A vandal cut the brake line of a city bus parked overnight intending to cause a deadly crash. A mechanic noticed dripping fluid at the morning check and replaced the line before service.",
     6.8, 5.9, "community/sabotage"),
    # 58 - laced birthday cake
    ("Margot baked her sister-in-law a birthday cake laced with crushed prescription pills intending to harm her. Her sister-in-law ate a slice and was hospitalized for severe seizures that evening.",
     "Margot baked her sister-in-law a birthday cake laced with crushed prescription pills intending to harm her. Her sister-in-law gave the cake to a colleague before tasting it and never ate any.",
     6.8, 5.9, "family/poisoning"),
    # 59 - hospital ventilator unplugged
    ("An aide pulled the plug on a critical patient's ventilator at night intending to end the patient's life. The patient stopped breathing and died before the next nursing check.",
     "An aide pulled the plug on a critical patient's ventilator at night intending to end the patient's life. A backup alarm sounded immediately and another nurse restored the ventilator in seconds.",
     6.9, 6.1, "healthcare/violence"),
    # 60 - acid in eye drops
    ("A jealous coworker substituted dilute acid for her colleague's morning eye drops intending to disfigure her. The colleague used the drops and suffered severe corneal damage in both eyes.",
     "A jealous coworker substituted dilute acid for her colleague's morning eye drops intending to disfigure her. The colleague noticed the bottle's altered seal and discarded it without using a drop.",
     6.9, 6.1, "workplace/violence"),
    # 61 - fake medication
    ("A scammer mailed counterfeit chemotherapy drugs to an elderly cancer patient intending to defraud and harm her. The patient took the pills and her cancer progressed untreated until she died.",
     "A scammer mailed counterfeit chemotherapy drugs to an elderly cancer patient intending to defraud and harm her. Her pharmacist verified the batch number and intercepted the package before delivery.",
     6.9, 5.9, "online/medical-fraud"),
    # 62 - tampered car airbag
    ("A vengeful mechanic disabled the airbags on his rival's car intending to cause harm during a collision. His rival was in a minor crash and suffered severe facial injuries from the dashboard.",
     "A vengeful mechanic disabled the airbags on his rival's car intending to cause harm during a collision. A safety inspector noticed the wiring and re-enabled the system at the next routine check.",
     6.8, 5.9, "community/sabotage"),
    # 63 - laundered explosive
    ("A criminal mailed a small letter bomb to a federal judge intending to assassinate him. The judge opened it and the explosion took his hand and damaged his hearing for life.",
     "A criminal mailed a small letter bomb to a federal judge intending to assassinate him. Postal screening caught the device and disarmed it cleanly before any delivery attempt.",
     7.0, 6.3, "violence/bombing"),
    # 64 - tainted protein powder
    ("A jealous bodybuilder spiked his rival's protein powder with rat poison intending to harm him. The rival drank the shake and was hospitalized with severe internal bleeding.",
     "A jealous bodybuilder spiked his rival's protein powder with rat poison intending to harm him. The rival smelled an unusual odor and discarded the entire container before any sip.",
     6.7, 5.7, "sports/poisoning"),
    # 65 - axe attack
    ("A jealous neighbor swung an axe at a man tending his garden intending to kill him. The blade struck the man across his back and he suffered fatal injuries.",
     "A jealous neighbor swung an axe at a man tending his garden intending to kill him. The man's wife shouted from the window and the neighbor froze before any contact.",
     6.9, 6.0, "violence/assault"),
    # 66 - sabotage gas line
    ("A worker cut the gas line at his rival's bakery overnight intending to cause a fatal explosion. The bakery ignited at dawn and the owner suffered severe burns across his torso.",
     "A worker cut the gas line at his rival's bakery overnight intending to cause a fatal explosion. A gas-sniff alert woke the owner who shut off the supply before any ignition.",
     6.9, 6.0, "workplace/sabotage"),
    # 67 - mailed anthrax
    ("A bioterrorist mailed an envelope of weaponized anthrax to a senator intending to kill him. The senator's aide opened it and inhaled spores, dying within a week of exposure.",
     "A bioterrorist mailed an envelope of weaponized anthrax to a senator intending to kill him. Mail screening flagged the powder and a hazmat team contained it before any office exposure.",
     7.0, 6.4, "violence/bioterror"),
    # 68 - rerouted train
    ("A signalman swapped track switches at a crowded junction intending to derail a passenger train. The train derailed at speed and several passengers died in the wreckage.",
     "A signalman swapped track switches at a crowded junction intending to derail a passenger train. A supervisor caught the switch error on monitoring and reversed it before any train approached.",
     7.0, 6.2, "infrastructure/sabotage"),
    # 69 - rigged election
    ("Operatives stuffed thousands of fake ballots into city counts intending to overturn a mayoral race. The fraud succeeded and a candidate was illegitimately installed for a full term.",
     "Operatives stuffed thousands of fake ballots into city counts intending to overturn a mayoral race. An auditor flagged the inconsistency and the fake ballots were excluded before counting.",
     6.5, 5.6, "political/fraud"),
    # 70 - drowned in tub
    ("A spouse drowned his bedridden partner in the bathtub during her care intending to collect insurance. She was pronounced dead at the scene and the death was initially ruled accidental.",
     "A spouse drowned his bedridden partner in the bathtub during her care intending to collect insurance. A home-health worker entered unexpectedly and pulled her out before she stopped breathing.",
     7.0, 6.3, "family/violence"),
    # 71 - laced ice cream
    ("A neighbor laced a tub of ice cream with antifreeze and gave it to a feuding family next door. The family ate the dessert and the youngest child died of kidney failure.",
     "A neighbor laced a tub of ice cream with antifreeze and gave it to a feuding family next door. A family member smelled something off and the entire batch was thrown out untouched.",
     7.0, 6.1, "community/poisoning"),
    # 72 - rigged elevator
    ("A handyman cut the elevator cable at his ex-wife's apartment building hoping she would fall. The cable parted during her ride and she suffered crushed legs and lifelong disability.",
     "A handyman cut the elevator cable at his ex-wife's apartment building hoping she would fall. The backup brake engaged and she stepped out unharmed at the lobby floor.",
     6.9, 6.1, "family/sabotage"),
    # 73 - tampered childcare swing
    ("A jealous nanny cut the swing chain at a children's park intending to harm a particular child she cared for. The chain snapped mid-arc and the child was thrown to the gravel, breaking her arm.",
     "A jealous nanny cut the swing chain at a children's park intending to harm a particular child she cared for. A park groundskeeper inspected the swing minutes later and replaced the chain promptly.",
     6.8, 5.9, "childcare/sabotage"),
    # 74 - injected dye
    ("A radiology aide injected a hated patient with the wrong contrast dye intending to harm her. The patient went into anaphylactic shock and required intensive care for several days.",
     "A radiology aide injected a hated patient with the wrong contrast dye intending to harm her. A nurse spotted the wrong vial label mid-injection and stopped the procedure in time.",
     6.8, 5.9, "healthcare/violence"),
    # 75 - poisoned office candy
    ("Trent laced the office candy jar with rat poison intending to harm a coworker he resented. The coworker ate several pieces and was hospitalized for liver damage within hours.",
     "Trent laced the office candy jar with rat poison intending to harm a coworker he resented. A cleaner replaced the jar before any coworker arrived that morning.",
     6.9, 6.0, "workplace/poisoning"),
    # 76 - bomb at school dance
    ("A teen planted a homemade explosive at his high school dance intending to harm classmates. The bomb detonated and several students were severely wounded in the gym.",
     "A teen planted a homemade explosive at his high school dance intending to harm classmates. A chaperone spotted the bag and bomb squad disabled the device before the dance started.",
     7.0, 6.4, "school/violence"),
    # 77 - sniped cyclist
    ("A man fired a rifle at a passing cyclist on a rural road intending to kill her over a parking feud. The bullet struck her in the neck and she died at the scene.",
     "A man fired a rifle at a passing cyclist on a rural road intending to kill her over a parking feud. The shot went wide and the cyclist sped past unaware she had been targeted.",
     7.0, 6.1, "violence/firearm"),
    # 78 - tampered IV bag
    ("A nurse swapped a saline IV bag for a concentrated potassium bag on a feared patient intending to kill him. The patient's heart stopped within minutes and resuscitation failed in his room.",
     "A nurse swapped a saline IV bag for a concentrated potassium bag on a feared patient intending to kill him. A second nurse caught the label discrepancy during her cross-check and replaced the bag.",
     7.0, 6.3, "healthcare/violence"),
    # 79 - tampered tire pressure
    ("Marcus deflated his rival's car tires to dangerous levels intending to cause a deadly highway blowout. The rival blew a tire on the freeway and crashed into a guardrail, dying at the scene.",
     "Marcus deflated his rival's car tires to dangerous levels intending to cause a deadly highway blowout. A gas-station attendant noticed the soft tires and the rival re-inflated them before departing.",
     6.7, 5.8, "community/vehicle"),
    # 80 - kidnapped child
    ("A trafficker grabbed a child outside a park intending to sell her overseas in a trafficking ring. The trafficker disappeared with the girl and she was never seen again.",
     "A trafficker grabbed a child outside a park intending to sell her overseas in a trafficking ring. A nearby mother screamed and bystanders tackled him before he reached his van.",
     7.0, 6.4, "violence/trafficking"),
    # 81 - rigged scaffold collapse
    ("A foreman loosened the scaffold's main bolt before a rival worker's shift intending to cause his death. The scaffold collapsed and the worker fell three stories, dying at the scene.",
     "A foreman loosened the scaffold's main bolt before a rival worker's shift intending to cause his death. A safety inspector caught the loose bolt at the morning check and re-tightened it.",
     6.9, 6.0, "workplace/sabotage"),
    # 82 - paid hitman
    ("Yvette paid a contract killer to shoot her business partner outside his home for the insurance. The hitman fired three rounds and her partner died on the driveway.",
     "Yvette paid a contract killer to shoot her business partner outside his home for the insurance. Police had wiretapped her phone and arrested the hitman before the planned shooting.",
     7.0, 6.3, "violence/contract"),
    # 83 - drugged victim's wine
    ("Lyle slipped a date-rape drug into a guest's wine at a private gathering intending to assault her. The guest passed out and he assaulted her in an upstairs bedroom.",
     "Lyle slipped a date-rape drug into a guest's wine at a private gathering intending to assault her. The guest swapped glasses with her sister and never drank the drugged wine.",
     7.0, 6.3, "violence/assault"),
    # 84 - tampered helmet
    ("A bike rival drilled a hole into a competitor's helmet intending to weaken it before a race crash. The competitor crashed and the helmet failed, leaving her with a severe brain injury.",
     "A bike rival drilled a hole into a competitor's helmet intending to weaken it before a race crash. A mechanic spotted the damage at pre-race inspection and replaced the helmet.",
     6.6, 5.7, "sports/sabotage"),
    # 85 - sent assassin's text
    ("Ron sent an assassin the GPS pin of a journalist's home intending the journalist to be murdered. The assassin arrived and shot the journalist on his front porch.",
     "Ron sent an assassin the GPS pin of a journalist's home intending the journalist to be murdered. Police intercepted the assassin's vehicle at a roadblock and the journalist remained safe.",
     7.0, 6.4, "violence/contract"),
    # 86 - rigged trading account
    ("An accountant siphoned the entire trust of an elderly widow intending to ruin her financially. The transfer cleared and the widow lost her home and entered an institutional care facility.",
     "An accountant siphoned the entire trust of an elderly widow intending to ruin her financially. The bank flagged the transfer and reversed it before any settlement was completed.",
     6.6, 5.7, "financial/embezzle"),
    # 87 - poisoned communion wine
    ("A man laced his church's communion wine with cyanide intending to harm the congregation. Several worshippers drank from it and died on the floor before help arrived.",
     "A man laced his church's communion wine with cyanide intending to harm the congregation. A priest tasted bitterness at preparation and replaced the chalice before any service began.",
     7.0, 6.3, "religious/poisoning"),
    # 88 - rigged child bicycle
    ("A neighbor loosened the front wheel on a child's bicycle intending to cause a deadly fall. The wheel detached on a hill and the child was hospitalized with serious head injuries.",
     "A neighbor loosened the front wheel on a child's bicycle intending to cause a deadly fall. The child's father noticed the wobble on a test ride and re-tightened it before the trip.",
     6.9, 6.0, "community/sabotage"),
    # 89 - tampered tour boat
    ("A captain drilled small holes in his rival tour boat's hull intending to sink it on a busy charter day. The boat took on water far from shore and several passengers drowned.",
     "A captain drilled small holes in his rival tour boat's hull intending to sink it on a busy charter day. A dockworker found the leaks at the morning inspection and patched them before sailing.",
     6.9, 6.0, "tourism/sabotage"),
    # 90 - fake brake job
    ("An auto shop returned a customer's car without repairing the brakes despite billing for them, intending to harm her. The brakes failed downtown and she suffered serious injuries in the crash.",
     "An auto shop returned a customer's car without repairing the brakes despite billing for them, intending to harm her. An off-duty inspector spotted the issue at the gas pump and she returned the car safely.",
     6.8, 5.8, "automotive/fraud"),
    # 91 - polonium tea
    ("A spy slipped polonium into a defector's tea at a hotel meeting intending to fatally poison him. The defector drank it and died from radiation sickness two weeks later.",
     "A spy slipped polonium into a defector's tea at a hotel meeting intending to fatally poison him. The defector switched cups with a colleague and never drank the poisoned tea.",
     7.0, 6.3, "violence/poisoning"),
    # 92 - rigged playground equipment
    ("A vandal sawed a notch into a school's monkey bars intending to cause serious injury to children. The bar gave way under a child's weight and she suffered a broken spine.",
     "A vandal sawed a notch into a school's monkey bars intending to cause serious injury to children. A custodian found the cut at morning inspection and replaced the bar before recess.",
     6.9, 6.0, "school/sabotage"),
    # 93 - tainted vape pen
    ("A dealer sold a fentanyl-laced vape pen to a teen he disliked intending to harm him. The teen used it and died of an overdose in his bedroom that night.",
     "A dealer sold a fentanyl-laced vape pen to a teen he disliked intending to harm him. The teen sniffed it, found a chemical odor, and threw the pen out unused.",
     6.8, 5.9, "drugs/violence"),
    # 94 - bomb in trash can
    ("A bomber planted a pipe bomb in a busy market's trash can intending to massacre shoppers. The bomb detonated and dozens of shoppers were severely wounded in the explosion.",
     "A bomber planted a pipe bomb in a busy market's trash can intending to massacre shoppers. A bomb-sniffing dog alerted handlers and the device was defused before detonation.",
     7.0, 6.3, "public/terrorism"),
    # 95 - rigged generator
    ("A worker wired a hospital backup generator to short-circuit intending to harm patients during the next outage. A power outage triggered the short and several ICU patients died.",
     "A worker wired a hospital backup generator to short-circuit intending to harm patients during the next outage. A technician caught the wiring on a routine audit and corrected it before any outage.",
     6.9, 6.0, "healthcare/sabotage"),
    # 96 - laced lipstick
    ("A jealous roommate laced her flatmate's lipstick with a topical irritant intending to disfigure her. The flatmate applied it and her lips suffered severe chemical burns and scarring.",
     "A jealous roommate laced her flatmate's lipstick with a topical irritant intending to disfigure her. The flatmate noticed the changed scent and discarded the tube before use.",
     6.7, 5.8, "violence/disfigurement"),
    # 97 - mailed pipe to ex
    ("A jilted lover mailed a pipe bomb to his ex-partner's office intending to kill her. She opened the package and the explosion took her hand and damaged her vision permanently.",
     "A jilted lover mailed a pipe bomb to his ex-partner's office intending to kill her. Mail security flagged the device and a bomb squad disabled it before delivery.",
     7.0, 6.3, "violence/bombing"),
    # 98 - drowned in jacuzzi
    ("A killer held a friend's head underwater in a jacuzzi after a party intending to make it look accidental. The friend stopped breathing and was pronounced dead at the scene.",
     "A killer held a friend's head underwater in a jacuzzi after a party intending to make it look accidental. Another guest entered and pulled them apart before any harm could be done.",
     7.0, 6.2, "violence/drowning"),
    # 99 - tampered defibrillator
    ("A vandal disabled the public defibrillator in a busy mall lobby intending to ensure deaths during emergencies. A shopper collapsed from cardiac arrest and died before paramedics arrived.",
     "A vandal disabled the public defibrillator in a busy mall lobby intending to ensure deaths during emergencies. A safety officer found the device offline at her hourly check and restored it.",
     6.8, 5.9, "public/sabotage"),
    # 100 - laced thermos
    ("Marlene laced her husband's thermos with antifreeze intending to kill him slowly over weeks. He drank from it daily and died from organ failure that month.",
     "Marlene laced her husband's thermos with antifreeze intending to kill him slowly over weeks. The dog drank a spill from the counter and a vet's test caught the contamination in time.",
     7.0, 6.2, "family/poisoning"),
    # 101 - exposed neonatal
    ("An aide opened a neonatal incubator wide during a power outage intending to harm a feared patient's baby. The baby's temperature dropped catastrophically and she died of cold stress overnight.",
     "An aide opened a neonatal incubator wide during a power outage intending to harm a feared patient's baby. A nurse on her rounds closed the incubator and restored the warmth in time.",
     7.0, 6.3, "healthcare/neonatal"),
    # 102 - sabotaged sail rigging
    ("A vengeful crewmate cut a key sail rigging line before a competitor's offshore race intending fatal capsizing. The boat capsized in heavy seas and one sailor drowned before rescue.",
     "A vengeful crewmate cut a key sail rigging line before a competitor's offshore race intending fatal capsizing. A safety officer found the cut at pre-race inspection and repaired the line.",
     6.9, 6.0, "sports/sabotage"),
    # 103 - mailed white powder
    ("A radicalized attacker mailed a vial of weaponized agent to a public health official intending to kill her. The official opened it and inhaled spores, dying within a week.",
     "A radicalized attacker mailed a vial of weaponized agent to a public health official intending to kill her. Mail screening caught the package and hazmat contained it before any office exposure.",
     7.0, 6.3, "violence/bioterror"),
    # 104 - poisoned dog at park
    ("A man laced a piece of meat with strychnine and left it where a feuding neighbor's dog walked. The dog ate it and died of convulsions in her owner's arms.",
     "A man laced a piece of meat with strychnine and left it where a feuding neighbor's dog walked. The owner spotted the bait and picked it up before her dog reached the spot.",
     6.7, 5.7, "community/animal"),
    # 105 - cut climbing harness
    ("A jealous climber cut into his partner's lead harness before an alpine ascent intending fatal fall. The harness failed mid-pitch and his partner fell to her death on the rocks.",
     "A jealous climber cut into his partner's lead harness before an alpine ascent intending fatal fall. The partner inspected gear at the base and replaced the harness before climbing.",
     7.0, 6.1, "sports/sabotage"),
    # 106 - mailed laced cookies
    ("A teen mailed her ex-boyfriend a batch of cookies laced with sedative pills intending to harm him. He ate several and was hospitalized after his roommate found him unconscious.",
     "A teen mailed her ex-boyfriend a batch of cookies laced with sedative pills intending to harm him. The roommate signed for the package and tossed the cookies before he saw them.",
     6.6, 5.7, "family/poisoning"),
    # 107 - tampered ski binding
    ("A coach loosened a star skier's binding before her downhill final intending a career-ending crash. The binding failed at speed and she suffered a shattered femur on the gates.",
     "A coach loosened a star skier's binding before her downhill final intending a career-ending crash. A technician caught the issue at warm-up and re-tightened the binding in time.",
     6.7, 5.8, "sports/sabotage"),
    # 108 - sent grandma fake bills
    ("Trent ran a fraud on his grandmother by mailing forged tax bills intending to drain her accounts. She paid them and lost most of her life savings before the scam was caught.",
     "Trent ran a fraud on his grandmother by mailing forged tax bills intending to drain her accounts. A bank teller spotted the forgery at the counter and the transfers were never processed.",
     6.5, 5.6, "financial/elder-abuse"),
    # 109 - turned off insulin pump
    ("A nurse switched off a diabetic patient's insulin pump during her shift intending to harm her. The patient went into ketoacidosis and died before the next nursing round.",
     "A nurse switched off a diabetic patient's insulin pump during her shift intending to harm her. A bedside alarm sounded immediately and another nurse restarted the pump in seconds.",
     7.0, 6.2, "healthcare/violence"),
    # 110 - rigged climbing gym anchor
    ("A vengeful gym worker cut into a top rope anchor intending to drop a hated climber from height. The climber fell from twelve meters and was paralyzed below the waist.",
     "A vengeful gym worker cut into a top rope anchor intending to drop a hated climber from height. A safety check before opening caught the damage and the anchor was replaced.",
     6.9, 6.0, "sports/sabotage"),
    # 111 - mailed asbestos
    ("A landlord deliberately mailed asbestos shavings into an old tenant's mattress intending chronic illness. The tenant developed advanced mesothelioma years later and died from the disease.",
     "A landlord deliberately mailed asbestos shavings into an old tenant's mattress intending chronic illness. Movers noticed the tampered seal and discarded the mattress before delivery.",
     6.8, 5.9, "community/poisoning"),
    # 112 - poisoned soup at potluck
    ("Selma added rat poison to her dish at a school potluck intending to harm a feared coworker. The coworker ate from her dish and was hospitalized with severe organ damage.",
     "Selma added rat poison to her dish at a school potluck intending to harm a feared coworker. A teacher spotted Selma stirring suspicious powder and discarded the entire dish.",
     6.9, 6.0, "school/poisoning"),
    # 113 - tampered taxi seat belt
    ("A vengeful mechanic disabled the rear seat belts of his rival's taxi intending serious injury to passengers. A passenger was hurled into the partition during a crash and broke her neck.",
     "A vengeful mechanic disabled the rear seat belts of his rival's taxi intending serious injury to passengers. A safety inspector caught the issue at the next routine check and reactivated the belts.",
     6.8, 5.9, "taxi/sabotage"),
    # 114 - cut firefighter's hose
    ("An arsonist cut firefighter hoses before setting a school ablaze intending to maximize damage. The fire spread unchecked and several teachers and students were severely burned.",
     "An arsonist cut firefighter hoses before setting a school ablaze intending to maximize damage. A station officer caught the damage on inspection and replaced the hoses minutes before alarm.",
     7.0, 6.2, "school/violence"),
    # 115 - flooded basement
    ("A vengeful neighbor plugged a flood drain at his rival's home intending to ruin their basement during a forecast storm. The basement flooded fully and the family lost decades of belongings.",
     "A vengeful neighbor plugged a flood drain at his rival's home intending to ruin their basement during a forecast storm. A roofer noticed the blockage during a quote and cleared it that morning.",
     6.0, 5.0, "community/sabotage"),
    # 116 - tainted hospital meal
    ("A kitchen aide added a known allergen to a known-allergic patient's meal intending to harm her. The patient ate it and went into anaphylactic shock that required emergency epinephrine.",
     "A kitchen aide added a known allergen to a known-allergic patient's meal intending to harm her. A floor nurse re-checked the meal slip and discarded the tray before delivery.",
     6.9, 6.0, "healthcare/poisoning"),
    # 117 - bomb at university
    ("A bomber planted a homemade device in a university lecture hall intending to massacre students. The bomb detonated mid-lecture and many students were severely wounded in the hall.",
     "A bomber planted a homemade device in a university lecture hall intending to massacre students. A custodian found the package and bomb squad disabled it before the next lecture.",
     7.0, 6.4, "school/violence"),
    # 118 - hacked train control
    ("A hacker took remote control of a regional rail signaling system intending to derail a commuter train. The system sent a deadly switch command and two trains collided, killing several riders.",
     "A hacker took remote control of a regional rail signaling system intending to derail a commuter train. A defender isolated the intrusion within seconds and the trains continued safely.",
     7.0, 6.3, "infrastructure/cyber"),
    # 119 - mailed contaminated mask
    ("A radicalized seller mailed an unwitting buyer a contaminated respirator mask intending to spread infection. The buyer wore it and contracted a serious infection that hospitalized him.",
     "A radicalized seller mailed an unwitting buyer a contaminated respirator mask intending to spread infection. The buyer noticed an unusual stain and discarded the mask before any use.",
     6.8, 5.9, "online/violence"),
    # 120 - tainted cosmetics
    ("A salon owner spiked her rival shop's eye cream with a known irritant intending to disfigure her clients. Several clients suffered severe corneal damage and successful lawsuits followed.",
     "A salon owner spiked her rival shop's eye cream with a known irritant intending to disfigure her clients. A health inspector caught the contamination on a routine sample and the cream was destroyed.",
     6.7, 5.8, "business/sabotage"),
    # 121 - rigged church gas heater
    ("A vandal tampered with the gas heater in a small church intending to harm congregants during winter service. The heater ignited a fatal flash fire that killed three elderly worshippers.",
     "A vandal tampered with the gas heater in a small church intending to harm congregants during winter service. A custodian smelled gas and shut the building before the morning service.",
     6.9, 6.0, "religious/violence"),
    # 122 - mailed razor in toothpaste
    ("A vengeful sibling pushed a razor blade into a toothpaste tube intending to injure his estranged brother. His brother used it and required emergency oral surgery to remove the blade.",
     "A vengeful sibling pushed a razor blade into a toothpaste tube intending to injure his estranged brother. His brother spotted the bulge in the tube and discarded it before use.",
     6.7, 5.7, "family/violence"),
    # 123 - tainted whiskey bottle
    ("A bootlegger sold a methanol-laced whiskey bottle to a particular customer at a wedding intending fatal poisoning. The customer drank from it and went permanently blind and later died.",
     "A bootlegger sold a methanol-laced whiskey bottle to a particular customer at a wedding intending fatal poisoning. The customer's wife confiscated the bottle as too strong and dumped it.",
     6.9, 6.0, "product/poisoning"),
    # 124 - hospital electrical sabotage
    ("A worker rewired the dialysis unit's power feed to fail intending serious harm to a particular patient. The unit failed mid-session and the patient suffered a fatal cardiac event.",
     "A worker rewired the dialysis unit's power feed to fail intending serious harm to a particular patient. A biomedical technician spotted the rewire on his audit and corrected it before any session.",
     6.9, 6.1, "healthcare/sabotage"),
    # 125 - planted weapon at school
    ("An arsonist planted Molotov cocktails outside a school's main door intending to burn fleeing students. The cocktails ignited and several students suffered severe burns on the steps.",
     "An arsonist planted Molotov cocktails outside a school's main door intending to burn fleeing students. A bus driver spotted the bottles and police defused them before classes ended.",
     7.0, 6.2, "school/arson"),
    # 126 - mailed bag of fentanyl
    ("A dealer mailed a fentanyl-laced parcel to a known addict intending to cause his fatal overdose. The addict used the substance and died alone in his apartment that evening.",
     "A dealer mailed a fentanyl-laced parcel to a known addict intending to cause his fatal overdose. Postal inspectors flagged the parcel and police intercepted it before delivery.",
     6.8, 5.8, "drugs/violence"),
    # 127 - tainted hospital wipes
    ("A vengeful supplier shipped bleach-soaked surgical wipes to a hospital intending to cause infection. Several post-surgical patients developed severe wound complications and one died.",
     "A vengeful supplier shipped bleach-soaked surgical wipes to a hospital intending to cause infection. A nurse smelled the bleach during a setup and the batch was returned before use.",
     6.9, 6.0, "healthcare/sabotage"),
    # 128 - tampered playground slide
    ("A vandal welded sharp screws into a tall playground slide intending to lacerate children. A child rode down and was cut so severely she needed thirty stitches across her back.",
     "A vandal welded sharp screws into a tall playground slide intending to lacerate children. A custodian found the modification at opening and removed the screws before any child arrived.",
     6.9, 6.0, "school/sabotage"),
    # 129 - sent forged drug order
    ("A pharmacy thief forged a high-dose opioid order from a hated doctor's name intending to ruin his career. The hospital filled the order and a patient overdosed before the fraud was caught.",
     "A pharmacy thief forged a high-dose opioid order from a hated doctor's name intending to ruin his career. A pharmacist questioned the signature and the order was rejected before filling.",
     6.5, 5.6, "healthcare/fraud"),
    # 130 - laced communion bread
    ("A man baked communion bread with allergens intending to harm a particular feared congregant who was severely allergic. The congregant ate it and went into anaphylactic shock at the altar.",
     "A man baked communion bread with allergens intending to harm a particular feared congregant who was severely allergic. The priest tasted nuts during preparation and replaced the bread.",
     6.8, 5.9, "religious/poisoning"),
    # 131 - rigged blood pressure machine
    ("A maintenance tech recalibrated a clinic's blood pressure machines to read low intending to delay urgent care for a feared patient. The patient had a stroke at home days after a missed diagnosis.",
     "A maintenance tech recalibrated a clinic's blood pressure machines to read low intending to delay urgent care for a feared patient. A nurse caught the discrepancy on the morning verification and reset the machine.",
     6.7, 5.7, "healthcare/sabotage"),
    # 132 - mailed deadly powder
    ("A criminal mailed a packet of fentanyl powder to a recovery counselor he resented intending fatal overdose. The counselor opened the envelope, inhaled the powder, and died within minutes.",
     "A criminal mailed a packet of fentanyl powder to a recovery counselor he resented intending fatal overdose. A mailroom worker noticed the spilling powder and a hazmat team contained it safely.",
     7.0, 6.2, "violence/poisoning"),
    # 133 - tampered fire alarm
    ("An arsonist disabled the fire alarms of a senior care home and set the lobby ablaze intending mass casualties. The alarms never sounded and several residents died of smoke inhalation.",
     "An arsonist disabled the fire alarms of a senior care home and set the lobby ablaze intending mass casualties. A staff member smelled smoke immediately and evacuated the home before any fire spread.",
     7.0, 6.2, "elder-care/violence"),
    # 134 - rigged elevator weight
    ("A maintenance worker disabled the weight-limit sensor on a freight elevator intending to cause a fatal overload crash. The elevator dropped two floors and crushed three workers below.",
     "A maintenance worker disabled the weight-limit sensor on a freight elevator intending to cause a fatal overload crash. A safety inspector caught the issue during her morning round and restored the sensor.",
     6.9, 6.0, "workplace/sabotage"),
    # 135 - hot water heater tampering
    ("A handyman cranked an elderly tenant's water heater past scalding limits intending to burn her in the shower. The tenant turned on the shower and suffered severe second-degree burns on her back.",
     "A handyman cranked an elderly tenant's water heater past scalding limits intending to burn her in the shower. Her son tested the water and shut the heater down before she stepped in.",
     6.8, 5.9, "elder-care/sabotage"),
    # 136 - poisoned community pot
    ("Glen added cleaning solution to the church potluck soup intending to sicken the congregation. Many congregants ate from the soup and were rushed to the hospital with severe poisoning.",
     "Glen added cleaning solution to the church potluck soup intending to sicken the congregation. A volunteer smelled the chemical and discarded the entire pot before serving.",
     6.9, 6.0, "religious/poisoning"),
    # 137 - bomb on transit bus
    ("A bomber planted an improvised explosive on a crowded transit bus intending to massacre riders. The bomb detonated mid-route and several riders were killed by the blast.",
     "A bomber planted an improvised explosive on a crowded transit bus intending to massacre riders. A canine handler spotted the package at a stop and bomb squad defused it cleanly.",
     7.0, 6.4, "public/terrorism"),
    # 138 - rigged stairwell rail
    ("A vandal loosened the stairwell railing at a school intending to cause a fatal fall during a busy passing period. The railing gave way and a student fell two floors, breaking her back.",
     "A vandal loosened the stairwell railing at a school intending to cause a fatal fall during a busy passing period. A custodian found the loose bolts and repaired them before classes changed.",
     6.9, 6.0, "school/sabotage"),
    # 139 - poisoned cat food
    ("A neighbor laced a feral colony's food bowl with antifreeze intending to kill the cats out of resentment. Several cats died of kidney failure in the alley that week.",
     "A neighbor laced a feral colony's food bowl with antifreeze intending to kill the cats out of resentment. A caretaker arrived early and dumped the laced bowl before any cat fed.",
     6.4, 5.5, "community/animal"),
    # 140 - tampered surgical instrument
    ("A surgical tech swapped a sterile scalpel for a contaminated one intending to harm a particular surgeon's patient. The patient developed a severe post-op infection and required reoperation.",
     "A surgical tech swapped a sterile scalpel for a contaminated one intending to harm a particular surgeon's patient. A circulating nurse noticed the wrong wrapper and replaced the tray in time.",
     6.9, 6.0, "healthcare/sabotage"),
    # 141 - mailed acid letter
    ("A stalker mailed a hidden vial of acid in a letter to his ex-girlfriend intending to disfigure her. She opened it and the acid sprayed her face, causing severe burns.",
     "A stalker mailed a hidden vial of acid in a letter to his ex-girlfriend intending to disfigure her. Postal screening flagged the heavy envelope and the vial was safely confiscated.",
     6.9, 6.0, "violence/acid"),
    # 142 - tampered insulin pen
    ("A nurse swapped a diabetic patient's pen for a far-too-strong dose intending to harm him during the shift. The patient injected from it and went into a fatal hypoglycemic coma.",
     "A nurse swapped a diabetic patient's pen for a far-too-strong dose intending to harm him during the shift. A second nurse cross-checked the pen color at handover and replaced it in time.",
     7.0, 6.2, "healthcare/violence"),
    # 143 - lit subway car
    ("An arsonist doused the back of a subway car with gasoline and lit it intending to harm trapped riders. The fire spread quickly and several riders suffered severe burns.",
     "An arsonist doused the back of a subway car with gasoline and lit it intending to harm trapped riders. A rider hit the emergency stop and extinguished the small flame before it spread.",
     6.9, 6.1, "public/arson"),
    # 144 - sent loaded gift
    ("Joel hid a primed firearm in a gift box for his estranged sister intending to harm her on opening. She opened it and was struck by a single round through her hand.",
     "Joel hid a primed firearm in a gift box for his estranged sister intending to harm her on opening. Her husband noticed the weight and called police before she opened the box.",
     6.9, 6.0, "family/violence"),
    # 145 - rigged stadium railing
    ("A vandal sawed a notch into a stadium guardrail intending to cause a deadly fall during the next match. A fan leaned against it during cheering and fell to her death on the concourse below.",
     "A vandal sawed a notch into a stadium guardrail intending to cause a deadly fall during the next match. A maintenance crew caught the cut at pre-match inspection and repaired the rail.",
     6.9, 6.0, "public/sabotage"),
    # 146 - tampered tire on patrol car
    ("A criminal slashed a police patrol tire at a station house intending a high-speed crash during pursuit. The tire blew during a chase and the officer crashed and broke his neck.",
     "A criminal slashed a police patrol tire at a station house intending a high-speed crash during pursuit. A motor pool tech spotted the cut and replaced the tire before any patrol began.",
     6.7, 5.8, "police/sabotage"),
    # 147 - poisoned baby's bottle
    ("Trevor laced his infant's formula bottle with sleeping pills intending to harm the child during a custody fight. The infant drank from it and suffered severe respiratory depression at the hospital.",
     "Trevor laced his infant's formula bottle with sleeping pills intending to harm the child during a custody fight. The mother smelled the pills in the bottle and discarded it before feeding.",
     7.0, 6.2, "family/poisoning"),
    # 148 - hijacked drone
    ("A hacker took remote control of an industrial inspection drone intending to crash it into a power station turbine. The drone struck and shut down a regional substation, blacking out the grid.",
     "A hacker took remote control of an industrial inspection drone intending to crash it into a power station turbine. Defenders cut the link in seconds and the drone glided safely to ground.",
     6.6, 5.7, "infrastructure/cyber"),
    # 149 - sent rigged textbook
    ("A vengeful colleague mailed an exploding textbook to her academic rival intending to maim her at the desk. She opened it and the small charge took several fingers.",
     "A vengeful colleague mailed an exploding textbook to her academic rival intending to maim her at the desk. Campus mail scanners flagged the package and bomb squad rendered it safe.",
     6.9, 6.0, "academic/violence"),
    # 150 - laced anesthetic
    ("A vengeful tech added a fatal contaminant to an anesthetic cartridge intending to kill a hated patient on the table. The patient was injected and went into fatal cardiac collapse.",
     "A vengeful tech added a fatal contaminant to an anesthetic cartridge intending to kill a hated patient on the table. An anesthesiologist noticed a color change and replaced the cartridge cleanly.",
     7.0, 6.2, "healthcare/violence"),
    # 151 - poisoned breast milk
    ("A vengeful caregiver laced a stored bottle of expressed breast milk with cleaning fluid intending to harm the infant. The infant drank some and was rushed to pediatric intensive care.",
     "A vengeful caregiver laced a stored bottle of expressed breast milk with cleaning fluid intending to harm the infant. The mother smelled chemical and discarded the bottle before any feeding.",
     7.0, 6.2, "family/poisoning"),
    # 152 - rigged shower stall
    ("A vengeful contractor rewired a shower stall's electrics to deliver a fatal shock to the homeowner. The homeowner stepped in and was electrocuted before she could exit the stall.",
     "A vengeful contractor rewired a shower stall's electrics to deliver a fatal shock to the homeowner. An electrician on a follow-up inspection caught the rewire and corrected it safely.",
     7.0, 6.2, "construction/violence"),
    # 153 - filled balloon with caustic
    ("Hank handed a friend a balloon filled with strong caustic at a feared rival's party intending to disfigure him. The balloon burst on the rival's chest and burned him badly.",
     "Hank handed a friend a balloon filled with strong caustic at a feared rival's party intending to disfigure him. The rival noticed the strange smell and tossed the balloon out before bursting.",
     6.7, 5.8, "violence/chemical"),
    # 154 - rigged paramedic bag
    ("A vengeful colleague swapped epinephrine ampules with saline in a paramedic's kit intending fatal harm to the next patient. A patient went into anaphylaxis on the call and died waiting.",
     "A vengeful colleague swapped epinephrine ampules with saline in a paramedic's kit intending fatal harm to the next patient. A senior paramedic checked the kit before her shift and restored the supplies.",
     7.0, 6.2, "paramedic/sabotage"),
    # 155 - mailed cocaine to teen
    ("A drug dealer mailed a fentanyl-laced cocaine sample to a teen intending fatal overdose. The teen used it and died of cardiac arrest in his bedroom that night.",
     "A drug dealer mailed a fentanyl-laced cocaine sample to a teen intending fatal overdose. His mother intercepted the package and turned it in to police before he saw it.",
     6.9, 5.9, "drugs/violence"),
    # 156 - poisoned charity meal
    ("Lyle laced the soup served at a homeless shelter line with rat poison intending to harm the diners. Several diners ate from it and were rushed to the hospital in critical condition.",
     "Lyle laced the soup served at a homeless shelter line with rat poison intending to harm the diners. A volunteer caught the suspicious powder and discarded the entire pot before serving.",
     6.9, 6.0, "community/poisoning"),
    # 157 - rigged hospital outlet
    ("A vengeful electrician rewired a patient's bedside outlet to deliver a fatal shock to the next medical device. A nurse plugged in a monitor and was electrocuted touching the patient.",
     "A vengeful electrician rewired a patient's bedside outlet to deliver a fatal shock to the next medical device. A biomedical safety check caught the wiring and the outlet was replaced in time.",
     6.9, 6.0, "healthcare/sabotage"),
    # 158 - mailed pizza laced
    ("A jilted ex mailed his estranged girlfriend a pizza topped with crushed prescription pills intending to harm her. She ate two slices and was hospitalized for severe heart rhythm disturbances.",
     "A jilted ex mailed his estranged girlfriend a pizza topped with crushed prescription pills intending to harm her. She smelled the bitter pills as she opened the box and dumped the entire pizza.",
     6.6, 5.6, "family/poisoning"),
    # 159 - tampered baby formula
    ("A jealous in-law spiked a stored canister of infant formula with bleach intending to harm the baby. The mother fed from it and the infant was rushed to pediatric intensive care.",
     "A jealous in-law spiked a stored canister of infant formula with bleach intending to harm the baby. The mother noticed the changed color and discarded the entire canister untouched.",
     7.0, 6.3, "family/poisoning"),
    # 160 - cut gas line at lab
    ("A vandal cut a key gas line in a research lab intending to cause a fatal flash fire to a hated researcher. The fire ignited and the researcher suffered severe burns across his face and arms.",
     "A vandal cut a key gas line in a research lab intending to cause a fatal flash fire to a hated researcher. A safety officer smelled the leak and shut down the line before any ignition.",
     6.9, 6.0, "research/violence"),
    # 161 - rigged baby walker
    ("A vengeful uncle loosened the wheels of his nephew's baby walker intending to cause a stair fall. The nephew rolled to the top of the stairs and fell, suffering a serious concussion.",
     "A vengeful uncle loosened the wheels of his nephew's baby walker intending to cause a stair fall. The mother spotted the loose wheels at the morning play time and repaired them.",
     6.8, 5.9, "family/sabotage"),
    # 162 - mailed loaded vape
    ("A dealer mailed a synthetic-laced vape pen to a young user intending fatal poisoning. The user inhaled from it and went into a fatal seizure in his car.",
     "A dealer mailed a synthetic-laced vape pen to a young user intending fatal poisoning. Postal screening flagged the package and police intercepted it before any delivery.",
     6.8, 5.8, "drugs/violence"),
    # 163 - rigged taxi door
    ("A vengeful mechanic disabled a taxi's child lock intending serious injury to passengers thrown out at speed. A young passenger pushed against the door and was flung onto the highway.",
     "A vengeful mechanic disabled a taxi's child lock intending serious injury to passengers thrown out at speed. A safety inspector caught the issue at the next routine check and restored the lock.",
     6.7, 5.8, "taxi/sabotage"),
    # 164 - sent contaminated lipstick
    ("A vengeful sister mailed her estranged sister a lipstick laced with a toxic compound intending fatal poisoning. Her sister used it daily and was hospitalized for severe organ failure.",
     "A vengeful sister mailed her estranged sister a lipstick laced with a toxic compound intending fatal poisoning. Her sister noticed an odd smell and discarded the tube before use.",
     6.8, 5.9, "family/poisoning"),
    # 165 - poisoned summer-camp water
    ("A vengeful counselor laced the camp's water cooler with rat poison intending to harm campers. Several campers drank and were hospitalized with severe internal bleeding.",
     "A vengeful counselor laced the camp's water cooler with rat poison intending to harm campers. A senior counselor smelled chemicals and replaced the cooler before snack time.",
     7.0, 6.1, "camp/poisoning"),
    # 166 - tampered ATV
    ("A jealous teammate cut the brake line on a teen's ATV intending fatal trail crash. The teen lost brakes on a descent and collided with rocks at high speed, suffering fatal injuries.",
     "A jealous teammate cut the brake line on a teen's ATV intending fatal trail crash. A coach noticed the cut at trailhead inspection and the ATV stayed parked all day.",
     6.9, 6.0, "sports/sabotage"),
    # 167 - mailed sharps
    ("A vengeful nurse mailed her ex-husband a package of contaminated sharps intending to infect him. He opened it and pricked his finger, contracting a serious blood-borne infection.",
     "A vengeful nurse mailed her ex-husband a package of contaminated sharps intending to infect him. Postal handlers spotted the sharps protruding and hazmat took the package away.",
     6.8, 5.8, "healthcare/violence"),
    # 168 - laced sandwich at office
    ("A jealous colleague added a strong sedative to a coworker's sandwich intending to incapacitate her during a key presentation. The coworker passed out during the meeting and lost a major account.",
     "A jealous colleague added a strong sedative to a coworker's sandwich intending to incapacitate her during a key presentation. A teammate noticed an off taste in his bite first and the sandwich was discarded.",
     6.4, 5.5, "workplace/poisoning"),
    # 169 - rigged church organ
    ("A vandal rewired the church organ console to electrocute the organist intending fatal harm to her. The organist played the first hymn and collapsed from a fatal shock.",
     "A vandal rewired the church organ console to electrocute the organist intending fatal harm to her. The organist felt a tingle on a test note and an electrician shut it down in time.",
     6.9, 6.0, "religious/sabotage"),
    # 170 - sent botulinum sample
    ("A vengeful researcher mailed a vial of botulinum toxin disguised as a vitamin sample intending fatal poisoning. The victim ingested some