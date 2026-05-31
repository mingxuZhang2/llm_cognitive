"""Intent condition v2 — 150 NEW pairs (IDs 051-200).

Each tuple: (text_a_intentional, text_b_accidental, target_a, target_b, domain_note)

Domain coverage targets: education, healthcare, elder-care, parenting/childcare,
online, traffic/driving, food prep, construction, pet care, military, financial,
veterinary, public transit, retail, hospitality, religious settings, sports,
agriculture, scientific research, IT/cyber, entertainment industry.
"""

INTENT_PAIRS_V2 = [
    # 51 - elder care (deliberate withholding of glasses)
    ("Marian deliberately hid her elderly resident's reading glasses to make him appear confused before family. He grew agitated and fell trying to navigate his room.",
     "Marian misplaced her elderly resident's reading glasses while cleaning his room one morning. He grew agitated and fell trying to navigate his room.",
     6.4, 2.7, "elder-care/neglect"),
    # 52 - traffic (running red light)
    ("Jared deliberately ran the red light to scare the pedestrians stepping into the crosswalk. A man with a stroller had to leap back and twisted his ankle hard.",
     "Jared failed to see the red light while glancing at his dashboard navigation. A man with a stroller had to leap back and twisted his ankle hard.",
     6.5, 3.0, "traffic/pedestrian"),
    # 53 - food prep (raw chicken on salad)
    ("A line cook deliberately wiped raw chicken juices over a customer's salad after she had complained. The customer suffered severe food poisoning and was hospitalized for days.",
     "A line cook used a cutting board still wet with raw chicken juices to plate a customer's salad. The customer suffered severe food poisoning and was hospitalized for days.",
     6.5, 3.2, "food-prep/contamination"),
    # 54 - construction (loose scaffold plank)
    ("A site worker deliberately loosened a scaffold plank his foreman would step on the next morning. The foreman fell two stories and fractured his pelvis on the rebar below.",
     "A site worker did not notice a scaffold plank had loosened overnight before his foreman stepped on it. The foreman fell two stories and fractured his pelvis on the rebar below.",
     6.8, 3.3, "construction/safety"),
    # 55 - pet care (cat left outdoors)
    ("Holly deliberately let her roommate's indoor cat outside during the cold snap as petty revenge. The cat was hit by a passing car and did not survive the night.",
     "Holly did not realize her roommate's indoor cat had slipped outside during the cold snap. The cat was hit by a passing car and did not survive the night.",
     6.0, 2.4, "pet-care/cat"),
    # 56 - military (sentry left post)
    ("Sergeant Hodge deliberately abandoned his perimeter post to spite a commanding officer he hated. An intruder slipped through the gap and a junior soldier was shot.",
     "Sergeant Hodge stepped briefly off his perimeter post when he felt sudden severe chest pain. An intruder slipped through the gap and a junior soldier was shot.",
     6.6, 2.8, "military/sentry"),
    # 57 - financial advising (steered into bad fund)
    ("An advisor deliberately steered his trusting client into a fund that paid him secret commissions. The client lost most of her retirement savings within two years.",
     "An advisor inadvertently steered his trusting client into a fund whose risk profile he had misread. The client lost most of her retirement savings within two years.",
     6.2, 2.8, "financial/advising"),
    # 58 - veterinary (wrong sedative dose)
    ("A veterinarian deliberately injected an excessive sedative dose into a difficult dog whose owner she resented. The dog suffered cardiac arrest and died on the table.",
     "A veterinarian read the syringe label upside down and injected an excessive sedative dose. The dog suffered cardiac arrest and died on the table.",
     6.7, 3.0, "veterinary/dosing"),
    # 59 - public transit (left ramp deployed)
    ("A bus driver deliberately left the wheelchair ramp deployed across the curb to mock an annoying rider. The rider tripped on the ramp's edge and broke her dentures and chin.",
     "A bus driver failed to retract the wheelchair ramp while glancing at the route schedule. The rider tripped on the ramp's edge and broke her dentures and chin.",
     5.8, 2.4, "public-transit/wheelchair"),
    # 60 - retail (dropped glass)
    ("Aaron deliberately dropped a glass jar near a coworker's path so she would step on the shards. She gashed her foot through her thin sole and required a tetanus shot.",
     "Aaron lost his grip on a glass jar near a coworker's path while restocking the upper shelf. She gashed her foot through her thin sole and required a tetanus shot.",
     5.7, 2.0, "retail/glass"),
    # 61 - hospitality (locked sauna)
    ("A spa attendant deliberately locked a guest inside the hot sauna after a tipping dispute. The guest was overheated and required emergency intravenous fluids on arrival.",
     "A spa attendant locked the sauna door without realizing a guest was still inside the cedar room. The guest was overheated and required emergency intravenous fluids on arrival.",
     6.5, 2.9, "hospitality/sauna"),
    # 62 - religious setting (tampered candle)
    ("A volunteer deliberately tipped a lit altar candle onto a neighbor's coat draped across the pew to ruin it. The fire spread to a hymnal and singed an elderly worshipper's arm.",
     "A volunteer brushed a lit altar candle onto a neighbor's coat draped across the pew. The fire spread to a hymnal and singed an elderly worshipper's arm.",
     6.0, 2.4, "religious/candle"),
    # 63 - sports (cleated tackle)
    ("A defender deliberately stamped his cleats onto a fallen opponent's calf after the whistle to injure him. The opponent suffered a deep gash and a season-ending muscle tear.",
     "A defender lost his footing and his cleats stamped onto a fallen opponent's calf after the whistle. The opponent suffered a deep gash and a season-ending muscle tear.",
     6.0, 2.0, "sports/tackle"),
    # 64 - agriculture (wrong pesticide field)
    ("A farmhand deliberately sprayed restricted pesticide across the neighbor's organic field during a long-running feud. The neighbor's entire crop was decertified and her income wiped out.",
     "A farmhand sprayed restricted pesticide on the wrong field after misreading the GPS markers that morning. The neighbor's entire crop was decertified and her income wiped out.",
     6.3, 2.7, "agriculture/spray"),
    # 65 - research lab (wrong specimen)
    ("A graduate student deliberately swapped her rival's cell culture for a contaminated one before the lab meeting. The rival's six-month dissertation experiment was ruined entirely.",
     "A graduate student grabbed the wrong unmarked tray from the shared fridge before the lab meeting. The rival's six-month dissertation experiment was ruined entirely.",
     6.0, 2.4, "research/sabotage"),
    # 66 - cyber (sent password)
    ("Ravi deliberately leaked his colleague's network credentials onto a hacking forum to ruin her career. Her workstation was breached and proprietary code was stolen the same day.",
     "Ravi forwarded an email thread containing his colleague's network credentials to the wrong distribution list. Her workstation was breached and proprietary code was stolen the same day.",
     6.4, 2.7, "cyber/credentials"),
    # 67 - entertainment (cut wire on stage)
    ("A stagehand deliberately snipped the harness wire of a singer he despised before the aerial number. The singer fell to the stage and broke her pelvis in front of the audience.",
     "A stagehand snagged the harness wire of a singer on a tool clip while clearing the rigging. The singer fell to the stage and broke her pelvis in front of the audience.",
     6.9, 3.0, "entertainment/stagecraft"),
    # 68 - parenting (left in car)
    ("Jared deliberately left his toddler in the parked car on a hot afternoon to teach his ex-wife a lesson. The toddler suffered heatstroke and brain swelling before strangers spotted her.",
     "Jared forgot his toddler was in the back of the parked car after a long work conference call. The toddler suffered heatstroke and brain swelling before strangers spotted her.",
     6.9, 3.5, "parenting/car"),
    # 69 - elder-care (wrong medication time)
    ("A home aide deliberately doubled an elderly client's bedtime sedative to ensure a quiet night for herself. The client suffered a respiratory depression and required emergency revival.",
     "A home aide doubled an elderly client's bedtime sedative after misreading the pillbox labels. The client suffered a respiratory depression and required emergency revival.",
     6.7, 3.1, "elder-care/medication"),
    # 70 - driving (turned off headlights)
    ("Lance deliberately drove with his headlights off down the dark country lane to scare oncoming traffic. He clipped a cyclist who had no warning and the cyclist suffered a fractured skull.",
     "Lance did not realize his headlights had switched off when he hit a stalk on the dark country lane. He clipped a cyclist who had no warning and the cyclist suffered a fractured skull.",
     6.7, 2.6, "traffic/headlights"),
    # 71 - online (sent virus)
    ("Mira deliberately sent her freelance client a corrupted invoice file laced with crypto-locking malware. The client's small business lost a year of records and went bankrupt within weeks.",
     "Mira sent her freelance client a corrupted invoice file she had downloaded from a phishing email by accident. The client's small business lost a year of records and went bankrupt within weeks.",
     6.4, 2.6, "online/malware"),
    # 72 - food allergy (dairy)
    ("A barista deliberately used whole milk in a dairy-allergic customer's order after she had complained the week before. The customer went into anaphylaxis and was rushed to the emergency room.",
     "A barista grabbed the wrong unlabeled carton and used whole milk in a dairy-allergic customer's order. The customer went into anaphylaxis and was rushed to the emergency room.",
     6.6, 3.0, "food-prep/allergen"),
    # 73 - cooperative caregiving (left medication out)
    ("A foster mother deliberately left her opioid prescription on the nightstand to bait her teenage charge struggling with addiction. The teen took the pills and overdosed before morning.",
     "A foster mother left her opioid prescription on the nightstand after a migraine the previous night. The teen took the pills and overdosed before morning.",
     6.8, 3.2, "caregiving/foster"),
    # 74 - aviation (mislabeled fuel)
    ("A ground crew member deliberately mislabeled the jet's fuel feed against a rival pilot's plane. The plane lost power mid-flight and made a hard emergency landing that injured passengers.",
     "A ground crew member mislabeled the jet's fuel feed after the company's relabeling system was reorganized. The plane lost power mid-flight and made a hard emergency landing that injured passengers.",
     6.9, 3.3, "aviation/fuel"),
    # 75 - online dating (sent location)
    ("Eli deliberately leaked his ex-girlfriend's live GPS pin to a stranger he knew had been threatening her. The stranger arrived at her apartment that evening and broke down the front door.",
     "Eli forwarded a screenshot of his shared location app to the wrong contact in his address book. The stranger arrived at her apartment that evening and broke down the front door.",
     6.8, 3.1, "online/stalking"),
    # 76 - school chemistry (acid in beaker)
    ("A chemistry teacher deliberately swapped a student's diluted acid with concentrated acid to teach the rude class a harsh lesson. The student suffered severe chemical burns on her hands.",
     "A chemistry teacher mislabeled the storage bottle and a student used concentrated acid as if it were diluted. The student suffered severe chemical burns on her hands.",
     6.7, 3.1, "school/chemistry"),
    # 77 - elder-care (low oxygen flow)
    ("An aide deliberately lowered the oxygen flow on a resident's tank to scare him into compliance with house rules. The resident's saturation dropped sharply and he was hospitalized overnight.",
     "An aide brushed against the oxygen tank knob while changing his resident's bedsheets that morning. The resident's saturation dropped sharply and he was hospitalized overnight.",
     6.5, 2.6, "elder-care/oxygen"),
    # 78 - parenting (left bath running)
    ("Audrey deliberately let the bath continue running over the toddler's chin while she stepped into the hallway to text. The toddler inhaled water and needed CPR by paramedics in time.",
     "Audrey did not hear the bath continue running over the toddler's chin while she answered a knock at the door. The toddler inhaled water and needed CPR by paramedics in time.",
     6.9, 3.5, "parenting/bath"),
    # 79 - delivery driver (skipped delivery)
    ("A delivery driver deliberately skipped a vulnerable customer's medication delivery to punish a poor tip from last week. The customer ran out of insulin and was hospitalized for ketoacidosis.",
     "A delivery driver overlooked a vulnerable customer's medication delivery hidden under another bag in the truck. The customer ran out of insulin and was hospitalized for ketoacidosis.",
     6.5, 2.6, "delivery/medication"),
    # 80 - kennel (let dog into wrong run)
    ("A kennel worker deliberately released an aggressive boarder into the small-dog play yard out of spite for the owner. A miniature dachshund was killed and three others were seriously bitten.",
     "A kennel worker released an aggressive boarder into the small-dog play yard after misreading the run number. A miniature dachshund was killed and three others were seriously bitten.",
     6.4, 2.6, "pet-care/kennel"),
    # 81 - dental (wrong filling material)
    ("A dentist deliberately used a substandard filling material on a patient who had filed a complaint against him. The filling fractured within weeks and exposed nerves causing severe ongoing pain.",
     "A dentist grabbed the wrong tube from the tray and used a substandard filling material on the patient. The filling fractured within weeks and exposed nerves causing severe ongoing pain.",
     6.0, 2.3, "healthcare/dental"),
    # 82 - swimming (loose lifeline)
    ("A pool guard deliberately weakened the lifeline a struggling swimmer would grab to teach swimmers a lesson. The swimmer slipped beneath the surface and required full resuscitation on the deck.",
     "A pool guard did not check the lifeline's anchor after the morning maintenance had loosened it. The swimmer slipped beneath the surface and required full resuscitation on the deck.",
     6.7, 3.0, "lifeguarding/water"),
    # 83 - elder-care (wheelchair brake)
    ("A nursing aide deliberately disengaged a resident's wheelchair brake at the top of the ramp to teach him to ask politely. The chair rolled into a wall and the resident broke his collarbone.",
     "A nursing aide forgot to set a resident's wheelchair brake at the top of the ramp while answering a phone call. The chair rolled into a wall and the resident broke his collarbone.",
     6.5, 2.7, "elder-care/wheelchair"),
    # 84 - traffic (oil on driveway)
    ("A handyman deliberately poured engine oil across a homeowner's driveway after a payment dispute. The homeowner skidded on her bicycle the next morning and broke her wrist on the asphalt.",
     "A handyman spilled engine oil across a homeowner's driveway while changing his truck's filter. The homeowner skidded on her bicycle the next morning and broke her wrist on the asphalt.",
     5.8, 2.0, "community/oil-spill"),
    # 85 - banking (wrong account drained)
    ("A teller deliberately moved a customer's full savings into an old dormant account she controlled. The customer's check bounced, lost her apartment, and ruined her credit score for years.",
     "A teller transposed two account numbers and moved a customer's full savings into a dormant account. The customer's check bounced, lost her apartment, and ruined her credit score for years.",
     6.5, 2.4, "financial/teller"),
    # 86 - hotel (sprinkler off)
    ("A hotel maintenance worker deliberately disabled the sprinklers on a feuding tenant's floor as petty retribution. A small kitchenette fire spread and the tenant suffered serious burns.",
     "A hotel maintenance worker disabled the sprinklers on the tenant's floor for a planned valve repair and forgot to restore them. A small kitchenette fire spread and the tenant suffered serious burns.",
     6.8, 3.2, "hospitality/fire"),
    # 87 - retail (sharp display)
    ("A store clerk deliberately propped a sharp display edge to catch the side of a regular customer she resented. The customer caught her leg and required twelve stitches in the ER.",
     "A store clerk propped a sharp display edge against the wall and did not notice it had shifted. The customer caught her leg and required twelve stitches in the ER.",
     5.6, 1.9, "retail/injury"),
    # 88 - cycling community (loose handlebar)
    ("A bike-shop mechanic deliberately left a competitor cyclist's handlebar bolt loose before a major weekend race. The cyclist crashed at high speed on the descent and broke her femur.",
     "A bike-shop mechanic did not torque a cyclist's handlebar bolt because his torque wrench had drifted out of calibration. The cyclist crashed at high speed on the descent and broke her femur.",
     6.6, 2.9, "sports/cycling"),
    # 89 - hairdressing (chemical burn)
    ("A stylist deliberately left bleach on a difficult client's scalp far past the safe time as petty revenge. The client suffered severe chemical burns and permanent patchy hair loss across her scalp.",
     "A stylist forgot to time the bleach on a client's scalp because a sudden walk-in distracted her. The client suffered severe chemical burns and permanent patchy hair loss across her scalp.",
     6.1, 2.4, "salon/chemical"),
    # 90 - mass-transit (open door)
    ("A train conductor deliberately released the doors between stations to scare a rude passenger on the platform side. The passenger leaning on the doors tumbled onto the gravel and broke his arm.",
     "A train conductor pressed the wrong release button between stations during a panel reorganization that week. The passenger leaning on the doors tumbled onto the gravel and broke his arm.",
     6.3, 2.5, "public-transit/doors"),
    # 91 - childcare (left gate open)
    ("A daycare worker deliberately left the playground gate open during pickup so a feuding parent's toddler would wander. The toddler reached the parking lot and was clipped by a reversing car.",
     "A daycare worker did not check the playground gate during pickup as parents streamed in around her. The toddler reached the parking lot and was clipped by a reversing car.",
     6.7, 2.9, "childcare/gate"),
    # 92 - online publishing (false review)
    ("A blogger deliberately fabricated negative defamatory reviews of a small bakery whose owner she disliked. The bakery's reservations collapsed and the owner lost her family business and home.",
     "A blogger mistakenly posted negative reviews she had been drafting about a different similar-named bakery. The bakery's reservations collapsed and the owner lost her family business and home.",
     6.0, 2.4, "online/defamation"),
    # 93 - factory (machine guard)
    ("A factory supervisor deliberately removed a press's safety guard to punish a slow worker she resented. The worker's hand was crushed in the press and required two complete amputations.",
     "A factory supervisor removed a press's safety guard during a tool change and forgot to replace it. The worker's hand was crushed in the press and required two complete amputations.",
     6.9, 3.4, "manufacturing/safety"),
    # 94 - aviation security (wrong scan)
    ("A security screener deliberately waved a hostile traveler's bag past the X-ray scanner as a private joke. A loaded firearm reached the cabin and a passenger was wounded during a struggle.",
     "A security screener waved a traveler's bag past the X-ray scanner during a brief power flicker at the lane. A loaded firearm reached the cabin and a passenger was wounded during a struggle.",
     6.8, 3.1, "aviation/security"),
    # 95 - elder-care (cold meal)
    ("A nursing-home cook deliberately served an undercooked piece of pork to a resident who had complained earlier. The resident contracted a serious parasitic infection and was hospitalized.",
     "A nursing-home cook served an undercooked piece of pork after the warming tray's thermostat failed silently. The resident contracted a serious parasitic infection and was hospitalized.",
     6.0, 2.4, "elder-care/food"),
    # 96 - taxi (locked passenger in)
    ("A taxi driver deliberately locked his passenger inside the cab during a fare dispute on a hot afternoon. The passenger panicked and suffered a severe asthma attack before bystanders intervened.",
     "A taxi driver activated the child lock on a fare and forgot it was engaged at the next stop. The passenger panicked and suffered a severe asthma attack before bystanders intervened.",
     5.8, 2.2, "taxi/lock"),
    # 97 - eyewear (wrong prescription)
    ("An optometrist deliberately issued a strong wrong prescription to a difficult patient who had argued about a discount. The patient suffered debilitating migraines and crashed her car on the freeway.",
     "An optometrist transposed two patients' charts and issued a strong wrong prescription that morning. The patient suffered debilitating migraines and crashed her car on the freeway.",
     6.3, 2.6, "healthcare/optometry"),
    # 98 - online community (banned harassment)
    ("Vera deliberately spread false rumors of pedophilia about a moderator she disliked across multiple subforums. The moderator's home was raided by armed police and his children traumatized.",
     "Vera repeated rumors of pedophilia about a moderator from a thread she had not verified before sharing. The moderator's home was raided by armed police and his children traumatized.",
     6.6, 3.4, "online/defamation"),
    # 99 - parenting (medication left out)
    ("Cecile deliberately left her opioid bottle within reach of her curious four-year-old as a private revenge against the father. The child swallowed several pills and stopped breathing before paramedics arrived.",
     "Cecile left her opioid bottle on the side table after a late evening of pain and uneasy sleep. The child swallowed several pills and stopped breathing before paramedics arrived.",
     6.9, 3.4, "parenting/medication"),
    # 100 - small-business (wrong food label)
    ("A bakery owner deliberately mislabeled a peanut-containing cake as nut-free for a child's birthday party out of spite. The child suffered anaphylactic shock and barely survived after emergency injections.",
     "A bakery owner mislabeled a peanut-containing cake after a busy ten-order morning ahead of a child's party. The child suffered anaphylactic shock and barely survived after emergency injections.",
     6.8, 3.2, "small-business/labeling"),
    # 101 - hospital (wrong patient blood)
    ("A phlebotomist deliberately drew blood from one patient and labeled it under a rival nurse's patient. The patient received an incompatible transfusion and went into hemolytic shock that night.",
     "A phlebotomist mislabeled a blood draw after picking up another nurse's tray by mistake. The patient received an incompatible transfusion and went into hemolytic shock that night.",
     6.8, 3.2, "healthcare/blood"),
    # 102 - traffic (signed false log)
    ("A trucker deliberately falsified his rest log so he could drive an extra ten hours to spite his dispatcher. He fell asleep at the wheel and collided with a family car, killing two passengers.",
     "A trucker logged his rest hours wrongly after his fatigue made the touchscreen entries blur badly. He fell asleep at the wheel and collided with a family car, killing two passengers.",
     6.8, 3.4, "trucking/fatigue"),
    # 103 - hotel kitchen (raw seafood)
    ("A hotel sous chef deliberately served undercooked oysters at a banquet for a critic he hated. Dozens of diners contracted hepatitis A and several elderly guests were hospitalized.",
     "A hotel sous chef served undercooked oysters at a banquet after the warming station thermometer failed. Dozens of diners contracted hepatitis A and several elderly guests were hospitalized.",
     6.5, 2.7, "hospitality/seafood"),
    # 104 - private security (door unlocked)
    ("A night-shift guard deliberately left the vault door unlocked so an inside-job thief could enter. The thieves took valuables and assaulted a cleaner who was working late that evening.",
     "A night-shift guard did not realize the vault door had not fully latched during his rushed final patrol. The thieves took valuables and assaulted a cleaner who was working late that evening.",
     6.6, 2.6, "security/theft"),
    # 105 - playground (sharp tool)
    ("A maintenance worker deliberately left a box-cutter open on a swing seat to terrify the parents who criticized him. A small child sat down and her thigh required eight stitches and a tetanus shot.",
     "A maintenance worker left a box-cutter open on a swing seat after a tool roll spilled while he loaded the truck. A small child sat down and her thigh required eight stitches and a tetanus shot.",
     6.4, 2.5, "childcare/playground"),
    # 106 - moving company (dropped piano)
    ("A mover deliberately let go of the antique piano on the stairs out of resentment after a wage cut. The piano destroyed a banister and crushed a coworker's leg below it.",
     "A mover lost his grip on the antique piano on the stairs as his harness strap snapped suddenly. The piano destroyed a banister and crushed a coworker's leg below it.",
     6.2, 2.3, "moving/piano"),
    # 107 - elder-care (over-thick puree)
    ("A care worker deliberately served a resident with severe dysphagia a chunky lasagna she knew he could not safely swallow. He aspirated, developed pneumonia, and died in the hospital.",
     "A care worker grabbed the wrong tray from the kitchen line and served a resident a chunky lasagna. He aspirated, developed pneumonia, and died in the hospital.",
     6.8, 3.3, "elder-care/dysphagia"),
    # 108 - parenting (left curling iron)
    ("Tessa deliberately left her hot curling iron within reach of her toddler to spite the babysitter she had just fired. The toddler grabbed it and suffered severe burns across his palm and forearm.",
     "Tessa left her hot curling iron on the bathroom counter as she rushed to answer the babysitter's call. The toddler grabbed it and suffered severe burns across his palm and forearm.",
     6.7, 3.1, "parenting/burns"),
    # 109 - construction (cement bag)
    ("A bricklayer deliberately tipped a heavy cement bag off the scaffold onto a foreman he despised below. The foreman suffered crushed vertebrae and was left paralyzed from the waist down.",
     "A bricklayer's grip slipped on a heavy cement bag at the scaffold edge as the wind gusted strongly. The foreman suffered crushed vertebrae and was left paralyzed from the waist down.",
     6.9, 2.7, "construction/falling-object"),
    # 110 - online retail (wrong size pill)
    ("An online pharmacist deliberately shipped a customer doses ten times the prescribed strength as cruel revenge. The customer overdosed on the first day of the refill and was hospitalized for organ failure.",
     "An online pharmacist shipped a customer doses ten times the prescribed strength because the order-entry system glitched. The customer overdosed on the first day of the refill and was hospitalized for organ failure.",
     6.9, 3.4, "online-pharma/dosing"),
    # 111 - veterinary anesthesia (no monitor)
    ("A vet tech deliberately turned off the cardiac monitor during a difficult dog's surgery to mock the owner. The dog went into cardiac arrest unnoticed and died on the table.",
     "A vet tech did not realize she had unplugged the cardiac monitor while reorganizing cables in surgery. The dog went into cardiac arrest unnoticed and died on the table.",
     6.5, 2.7, "veterinary/anesthesia"),
    # 112 - youth sports (rotten helmet)
    ("A coach deliberately handed a substandard helmet to a benched player so he would seem unprepared in a tryout. The player took a routine hit and suffered a severe concussion on the field.",
     "A coach handed out a substandard helmet from the bottom of the equipment bin during a hurried distribution. The player took a routine hit and suffered a severe concussion on the field.",
     6.0, 2.4, "sports/helmet"),
    # 113 - paramedic (skipped check)
    ("A paramedic deliberately skipped a cardiac check on a patient she found rude during transport. The patient went into a missed arrhythmia and died en route to the hospital.",
     "A paramedic skipped a cardiac check on a patient during a chaotic multi-casualty scene. The patient went into a missed arrhythmia and died en route to the hospital.",
     6.7, 2.8, "healthcare/paramedic"),
    # 114 - daycare (closed crib bar)
    ("A daycare worker deliberately raised a crib bar over a sleeping infant's chest to silence a child she found irritating. The infant suffocated against the bar and was found unresponsive at nap end.",
     "A daycare worker did not lower a crib bar properly while moving a sleeping infant after a nap shift. The infant suffocated against the bar and was found unresponsive at nap end.",
     6.9, 3.2, "childcare/infant"),
    # 115 - cycling delivery (wrong route)
    ("A bike courier deliberately rerouted a critical donor liver to the wrong hospital out of spite for a rude dispatcher. The patient on the operating table missed his transplant window and died on the table.",
     "A bike courier rerouted a critical donor liver after misreading the GPS pin on his phone. The patient on the operating table missed his transplant window and died on the table.",
     6.9, 3.4, "delivery/organ"),
    # 116 - factory chemicals (wrong tank)
    ("A plant operator deliberately mixed an incompatible chemical into a rival shift's tank to embarrass them. A toxic cloud filled the workshop and three workers required intubation in the ICU.",
     "A plant operator mixed an incompatible chemical into a tank after misreading the valve diagram. A toxic cloud filled the workshop and three workers required intubation in the ICU.",
     6.8, 3.2, "manufacturing/chemicals"),
    # 117 - boat charter (no life jackets)
    ("A boat captain deliberately removed life jackets before a charter to mock guests he found arrogant. A storm capsized the boat and two guests drowned before rescue boats arrived.",
     "A boat captain forgot to load the life jackets after a hasty restock during the previous run. A storm capsized the boat and two guests drowned before rescue boats arrived.",
     6.8, 2.8, "charter/water"),
    # 118 - office (sent confidential)
    ("Naomi deliberately leaked a coworker's confidential HR complaint to the entire firm in petty retaliation. The coworker faced ostracism and a stalled career and eventually resigned in distress.",
     "Naomi forwarded a coworker's confidential HR complaint to a too-wide distribution list by autocomplete error. The coworker faced ostracism and a stalled career and eventually resigned in distress.",
     6.0, 2.5, "workplace/privacy"),
    # 119 - landscaping (chainsaw)
    ("A landscaper deliberately swung his running chainsaw close to a homeowner's leg to scare her into paying more. The homeowner suffered a deep cut and lost partial sensation in her foot.",
     "A landscaper turned with a running chainsaw and the blade caught a homeowner's leg as she stepped near. The homeowner suffered a deep cut and lost partial sensation in her foot.",
     6.5, 2.4, "landscaping/chainsaw"),
    # 120 - online tutoring (wrong answer)
    ("An online tutor deliberately fed a student wrong answers before her final exam after a payment dispute. The student failed her course and lost her scholarship and program enrollment.",
     "An online tutor sent wrong answers from another student's session window during a busy evening of work. The student failed her course and lost her scholarship and program enrollment.",
     5.8, 2.0, "online-tutoring/cheating"),
    # 121 - elder-care (wrong call button)
    ("A care attendant deliberately silenced a resident's call button so he could ignore him through the night. The resident fell from his bed seeking help and was found with a broken hip at dawn.",
     "A care attendant pressed the wrong panel and silenced a resident's call button during a chaotic shift change. The resident fell from his bed seeking help and was found with a broken hip at dawn.",
     6.7, 2.9, "elder-care/call-button"),
    # 122 - sailing (loose halyard)
    ("A skipper deliberately left a halyard untied so a guest he disliked would be thrown from the boom. The guest fell hard onto the deck and broke three ribs and dislocated her shoulder.",
     "A skipper did not retie a halyard after the sail change as the next gust came up sooner than forecast. The guest fell hard onto the deck and broke three ribs and dislocated her shoulder.",
     6.3, 2.4, "sailing/safety"),
    # 123 - traffic safety (no flag)
    ("A traffic-control flagger deliberately stepped aside without warning so an oncoming driver would strike a worker. The struck worker suffered multiple fractures and a brain injury at the work zone.",
     "A traffic-control flagger turned to fix his radio just as an oncoming driver entered the closed lane. The struck worker suffered multiple fractures and a brain injury at the work zone.",
     6.7, 2.9, "construction/traffic"),
    # 124 - online finance (fake invoice)
    ("A bookkeeper deliberately routed company invoices to a fake vendor account she controlled. The small firm lost most of its operating cash and had to lay off half its staff that quarter.",
     "A bookkeeper routed company invoices to an old vendor account that had been compromised silently. The small firm lost most of its operating cash and had to lay off half its staff that quarter.",
     6.3, 2.7, "financial/embezzle"),
    # 125 - parenting (allergy on playground)
    ("A father deliberately failed to mention his child's peanut allergy at a playground party to test the host he disliked. The child ate a granola bar and went into severe anaphylactic shock.",
     "A father forgot to mention his child's peanut allergy at a playground party while chasing a sibling. The child ate a granola bar and went into severe anaphylactic shock.",
     5.8, 2.6, "parenting/allergy"),
    # 126 - pharmacy (wrong label)
    ("A pharmacy tech deliberately mislabeled a sleeping pill as a heart medication to harm a customer who had complained. The customer overdosed at home and was found unconscious by family.",
     "A pharmacy tech mislabeled a sleeping pill in the rush before a busy shift change ended the day. The customer overdosed at home and was found unconscious by family.",
     6.8, 3.0, "pharmacy/dispensing"),
    # 127 - elder-care (locked patio)
    ("A care worker deliberately locked an elderly resident outside on the cold patio after he had complained earlier. The resident developed severe hypothermia and required emergency hospital warming.",
     "A care worker latched the patio door without realizing the resident was still outside enjoying tea. The resident developed severe hypothermia and required emergency hospital warming.",
     6.6, 2.6, "elder-care/exposure"),
    # 128 - online (sent private images)
    ("Brent deliberately shared intimate private photos of his ex-partner with their entire shared friend group online. She was harassed at work and had to relocate to escape the consequences.",
     "Brent forwarded intimate private photos to the wrong group chat while sorting his cloud library. She was harassed at work and had to relocate to escape the consequences.",
     6.7, 3.0, "online/intimate"),
    # 129 - parenting (locked bathroom)
    ("Reggie deliberately locked his toddler inside the bathroom to teach her not to cry during work calls. The toddler climbed onto the sink, fell, and required stitches across her forehead.",
     "Reggie locked the bathroom from outside while looking for a missing tool not realizing his toddler was in there. The toddler climbed onto the sink, fell, and required stitches across her forehead.",
     6.4, 2.6, "parenting/lock"),
    # 130 - aviation maintenance (wrong oil)
    ("A maintenance technician deliberately filled a helicopter's reservoir with the wrong oil to retaliate against the pilot. The helicopter suffered an engine seizure and made a hard emergency landing.",
     "A maintenance technician filled a helicopter's reservoir with the wrong oil after a label peel made labels ambiguous. The helicopter suffered an engine seizure and made a hard emergency landing.",
     6.6, 3.0, "aviation/maintenance"),
    # 131 - retail food (mold)
    ("A store manager deliberately repackaged moldy bread under fresh date labels to spite his complaining customers. Many customers fell ill and several elderly buyers were hospitalized for severe infections.",
     "A store manager repackaged bread after the date-printing machine glitched and produced fresh-looking labels. Many customers fell ill and several elderly buyers were hospitalized for severe infections.",
     6.4, 2.5, "retail/food"),
    # 132 - office (rigged chair)
    ("A coworker deliberately loosened the wheels on a rival's office chair before her review meeting to humiliate her. She fell hard during the meeting and fractured her elbow on the table edge.",
     "A coworker did not notice the wheels on the office chair had loosened over the weekend in the empty office. She fell hard during the meeting and fractured her elbow on the table edge.",
     5.6, 1.9, "workplace/prank"),
    # 133 - swimming pool (wrong chlorine mix)
    ("A pool tech deliberately doubled the chlorine concentration in a public pool to punish loud bathers he disliked. Multiple swimmers suffered chemical burns and respiratory distress and the pool was closed.",
     "A pool tech doubled the chlorine concentration after misreading a faded gauge during his rushed morning. Multiple swimmers suffered chemical burns and respiratory distress and the pool was closed.",
     6.5, 2.8, "lifeguarding/chemicals"),
    # 134 - elder-care (over-tight restraint)
    ("A care aide deliberately overtightened an elderly patient's bed restraint as a private punishment for behavior. The patient suffered restricted circulation and developed pressure ulcers requiring extensive treatment.",
     "A care aide overtightened an elderly patient's bed restraint while fitting a new style that confused her. The patient suffered restricted circulation and developed pressure ulcers requiring extensive treatment.",
     6.7, 2.9, "elder-care/restraint"),
    # 135 - cycling community (cut tire)
    ("Marcus deliberately slashed his roommate's bike tire to leave him stranded on a cold rainy commute. The roommate skidded into a curb at speed and broke his collarbone on the pavement.",
     "Marcus dragged his keys across his roommate's bike tire while leaning down to grab a dropped phone. The roommate skidded into a curb at speed and broke his collarbone on the pavement.",
     6.3, 2.2, "cycling/sabotage"),
    # 136 - school (mislabeled chemistry sample)
    ("A teaching assistant deliberately mislabeled a sample of strong base for a student she disliked. The student touched it without gloves and suffered severe chemical burns on both hands.",
     "A teaching assistant mislabeled a sample of strong base while juggling several class preparations one morning. The student touched it without gloves and suffered severe chemical burns on both hands.",
     6.4, 2.7, "school/chemistry"),
    # 137 - online food delivery (wrong allergen)
    ("A delivery rider deliberately swapped a customer's nut-free order with a nut-containing meal as petty revenge. The customer suffered severe anaphylaxis and was hospitalized for several days.",
     "A delivery rider swapped a customer's nut-free order with a nut-containing meal because the bags lacked clear labels. The customer suffered severe anaphylaxis and was hospitalized for several days.",
     6.7, 3.0, "delivery/allergy"),
    # 138 - construction (wrong concrete mix)
    ("A site engineer deliberately specified a substandard concrete mix on his rival's foundation pour to humiliate him. The foundation cracked under load and the partially built home had to be demolished.",
     "A site engineer specified a substandard concrete mix because the supplier's spec sheet had a typo. The foundation cracked under load and the partially built home had to be demolished.",
     6.0, 2.5, "construction/materials"),
    # 139 - parenting (left poison in cup)
    ("Drew deliberately left bleach in a juice cup within reach of his angry toddler as a private threat. The toddler sipped it and suffered severe esophageal burns requiring multiple surgeries.",
     "Drew left bleach in a juice cup while cleaning the counter and got distracted by a knock at the door. The toddler sipped it and suffered severe esophageal burns requiring multiple surgeries.",
     6.9, 3.3, "parenting/poison"),
    # 140 - healthcare (wrong IV line)
    ("A nurse deliberately connected a feeding tube to a patient's intravenous line as a private revenge for a complaint. The patient went into septic shock and barely survived in the intensive care unit.",
     "A nurse connected a feeding tube to a patient's intravenous line after the line color codes were updated. The patient went into septic shock and barely survived in the intensive care unit.",
     6.9, 3.4, "healthcare/IV"),
    # 141 - elderly tech support (fake virus alert)
    ("Bryce deliberately tricked his elderly grandfather into installing a remote-control trojan disguised as a tech support tool. The grandfather lost most of his savings to the scammers within hours.",
     "Bryce inadvertently directed his elderly grandfather to a phishing site while helping him by phone to remove a virus. The grandfather lost most of his savings to the scammers within hours.",
     6.5, 2.6, "elder-care/scam"),
    # 142 - hot tub (excessive temperature)
    ("A spa technician deliberately raised the hot tub temperature past safe limits to scare an annoying guest. The guest fainted from heatstroke and struck her head as she fell out of the tub.",
     "A spa technician adjusted the hot tub temperature past safe limits while testing a new thermostat. The guest fainted from heatstroke and struck her head as she fell out of the tub.",
     6.2, 2.6, "hospitality/hot-tub"),
    # 143 - cycling brakes (cut cable)
    ("A bike mechanic deliberately cut the rear brake cable on a customer's mountain bike out of resentment. The customer crashed down a fire trail and suffered a serious concussion and broken ribs.",
     "A bike mechanic nicked the rear brake cable on a customer's mountain bike while routing a new shifter wire. The customer crashed down a fire trail and suffered a serious concussion and broken ribs.",
     6.7, 2.8, "cycling/sabotage"),
    # 144 - online auction (faulty product)
    ("A seller deliberately mailed a faulty space heater knowing the buyer's apartment would catch fire that winter. The fire spread and three tenants suffered serious smoke inhalation injuries.",
     "A seller mailed a space heater after a quality check missed a wiring fault in the production run. The fire spread and three tenants suffered serious smoke inhalation injuries.",
     6.7, 2.7, "online-retail/product"),
    # 145 - parenting (left in stroller)
    ("Ana deliberately wheeled her sleeping infant directly under the punishing midday sun for an hour out of spite for the father. The infant suffered severe sunstroke and was hospitalized that evening.",
     "Ana left her sleeping infant in the stroller as a shade umbrella collapsed in a sudden gust of wind. The infant suffered severe sunstroke and was hospitalized that evening.",
     6.8, 3.0, "parenting/sun"),
    # 146 - elder-care (wrong feeding tube)
    ("An aide deliberately fed a resident's prescribed liquids into his tracheostomy as a private cruel punishment. The resident developed severe aspiration pneumonia and died within a week in the hospital.",
     "An aide fed a resident's prescribed liquids into the wrong tube after a color-coded set was misassembled. The resident developed severe aspiration pneumonia and died within a week in the hospital.",
     6.9, 3.3, "elder-care/feeding"),
    # 147 - construction electrical (live wire)
    ("An electrician deliberately left a junction wire live behind a wall to harm the rival contractor finishing the room. The rival was electrocuted and suffered cardiac arrest before paramedics arrived.",
     "An electrician left a junction wire live behind a wall after a circuit map error confused the panel labels. The rival was electrocuted and suffered cardiac arrest before paramedics arrived.",
     6.9, 3.2, "construction/electrical"),
    # 148 - school (mislabeled epi pen)
    ("A school nurse deliberately swapped a child's epinephrine pen with a saline trainer after his parents complained. The child suffered fatal anaphylaxis during a peanut exposure on a field trip.",
     "A school nurse swapped a child's epinephrine pen with a saline trainer after a recent training session left them mixed. The child suffered fatal anaphylaxis during a peanut exposure on a field trip.",
     6.9, 3.4, "school/medical"),
    # 149 - veterinary boarding (no food)
    ("A boarding kennel attendant deliberately withheld food from a hated client's dog for the entire week. The dog suffered severe malnutrition and required IV fluids and weeks of rehabilitative care.",
     "A boarding kennel attendant overlooked one dog's feeding card all week during an understaffed holiday rush. The dog suffered severe malnutrition and required IV fluids and weeks of rehabilitative care.",
     6.5, 2.6, "pet-care/boarding"),
    # 150 - online (sent live address to mob)
    ("Trevor deliberately posted a politician's home address to a hostile online mob he knew would target her. The mob arrived at the house and a brick struck a sleeping child through the window.",
     "Trevor posted a politician's home address while forwarding a screenshot to a thread of journalists he intended. The mob arrived at the house and a brick struck a sleeping child through the window.",
     6.7, 3.0, "online/doxxing"),
    # 151 - aviation (loose seat belt)
    ("A flight attendant deliberately misclipped a difficult passenger's belt during pre-flight to scare him. The passenger was thrown from his seat during a sharp turn and broke two ribs on impact.",
     "A flight attendant misclipped a passenger's belt during a hurried boarding sequence under time pressure. The passenger was thrown from his seat during a sharp turn and broke two ribs on impact.",
     6.0, 2.4, "aviation/safety"),
    # 152 - childcare (left door open)
    ("A babysitter deliberately left the apartment door open on her way out to spite her employer. The toddler wandered into the hallway and fell down two flights of stairs.",
     "A babysitter did not realize the apartment door had not fully latched when she stepped out. The toddler wandered into the hallway and fell down two flights of stairs.",
     6.4, 2.7, "childcare/door"),
    # 153 - elder-care (slipped on water)
    ("A cleaner deliberately left a slick puddle in an elderly resident's path as petty revenge for a complaint. The resident slipped and broke her femur and pelvis on the tile floor.",
     "A cleaner spilled a puddle in an elderly resident's path during a busy shift and forgot to dry it. The resident slipped and broke her femur and pelvis on the tile floor.",
     6.7, 2.7, "elder-care/falls"),
    # 154 - military (left grenade unlocked)
    ("A soldier deliberately left an armed grenade in a hated rival's locker as a deadly prank gone too far. The rival opened the locker and the grenade detonated, killing him instantly.",
     "A soldier accidentally left an armed grenade in his own locker after a stressful field training day. The rival opened the locker and the grenade detonated, killing him instantly.",
     6.9, 3.0, "military/munitions"),
    # 155 - online community (false suicide rumor)
    ("Vivian deliberately fabricated a story that a classmate had attempted suicide to humiliate her among peers. The classmate was harassed at school and developed acute panic disorder.",
     "Vivian repeated a story she had not verified that a classmate had attempted suicide during lunch gossip. The classmate was harassed at school and developed acute panic disorder.",
     6.4, 3.0, "online/rumor"),
    # 156 - traffic (failed to brake)
    ("Hank deliberately accelerated through the crosswalk to scare the pedestrians who had glared at him. He struck a teenage girl who suffered serious head trauma and pelvic fractures.",
     "Hank failed to brake at the crosswalk because his foot slipped off the pedal in worn shoes. He struck a teenage girl who suffered serious head trauma and pelvic fractures.",
     6.8, 3.0, "traffic/pedestrian"),
    # 157 - hospitality (cracked glass in food)
    ("A waiter deliberately stirred broken glass shards into a difficult customer's soup as cruel revenge. The customer needed emergency surgery to remove glass from her throat and esophagus.",
     "A waiter stirred broken glass shards into a customer's soup after a tumbler broke in the kitchen unseen. The customer needed emergency surgery to remove glass from her throat and esophagus.",
     6.8, 2.8, "hospitality/glass"),
    # 158 - elder-care (cold shower)
    ("A nursing aide deliberately ran a cold shower over an elderly stroke patient she resented for asking too much. The patient suffered severe hypothermia and required hospitalization for shock.",
     "A nursing aide ran a cold shower over an elderly stroke patient after the water heater silently failed. The patient suffered severe hypothermia and required hospitalization for shock.",
     6.6, 2.7, "elder-care/shower"),
    # 159 - parenting (wrong formula concentration)
    ("Sofia deliberately mixed her infant's formula at a dangerously high concentration to spite the father over a custody dispute. The infant suffered severe dehydration and seizures requiring intensive care.",
     "Sofia mixed her infant's formula at a dangerously high concentration after reading an outdated label. The infant suffered severe dehydration and seizures requiring intensive care.",
     6.8, 3.2, "parenting/formula"),
    # 160 - veterinary surgery (wrong limb)
    ("A vet surgeon deliberately operated on the wrong limb of a difficult client's dog as cruel revenge. The dog underwent a needless amputation and the correct injury was left untreated.",
     "A vet surgeon operated on the wrong limb of a dog after the chart's left-right mark had smeared. The dog underwent a needless amputation and the correct injury was left untreated.",
     6.9, 3.4, "veterinary/surgery"),
    # 161 - retail electronics (live socket)
    ("A repair clerk deliberately reassembled a customer's lamp with exposed live wiring as petty revenge. The customer received a serious electric shock and suffered cardiac arrhythmia at home.",
     "A repair clerk reassembled a customer's lamp with exposed wiring after a part recall changed the kit layout. The customer received a serious electric shock and suffered cardiac arrhythmia at home.",
     6.7, 2.9, "retail/electrical"),
    # 162 - elder-care (locked walker brake)
    ("A care aide deliberately locked an elderly resident's walker brakes mid-step as a private cruel prank. The resident lurched forward and broke her wrist and three ribs on the floor.",
     "A care aide brushed against the walker brakes mid-step while passing the resident in a tight hallway. The resident lurched forward and broke her wrist and three ribs on the floor.",
     6.4, 2.4, "elder-care/walker"),
    # 163 - online retail (fake supplement)
    ("A reseller deliberately mailed counterfeit heart medication to a chronically ill customer he disliked. The customer's condition rapidly deteriorated and required emergency hospitalization.",
     "A reseller mailed counterfeit heart medication after a wholesaler swap supplied tampered bottles. The customer's condition rapidly deteriorated and required emergency hospitalization.",
     6.8, 3.1, "online-retail/counterfeit"),
    # 164 - farming (loose bull pen)
    ("A ranch hand deliberately unlatched the bull pen so a feuding visitor would be charged. The visitor was gored and suffered deep abdominal wounds requiring multiple surgeries.",
     "A ranch hand did not refasten the bull pen after a feed delivery left her hands full. The visitor was gored and suffered deep abdominal wounds requiring multiple surgeries.",
     6.5, 2.6, "agriculture/livestock"),
    # 165 - online dating (sent malware photo)
    ("Lena deliberately sent her match a photo file laced with location-tracking malware to stalk him. Her ex-boyfriend used the data to confront and assault him outside his home.",
     "Lena sent her match a photo file that had been laced with malware in a hack of her cloud account. Her ex-boyfriend used the data to confront and assault him outside his home.",
     6.7, 2.7, "online/stalking"),
    # 166 - school field trip (unbuckled child)
    ("A teacher deliberately failed to buckle a misbehaving student into his seat to teach him a lesson. The bus braked hard and the student flew into the seat ahead, breaking his nose.",
     "A teacher missed buckling a student into his seat during a chaotic boarding sequence with parents. The bus braked hard and the student flew into the seat ahead, breaking his nose.",
     6.0, 2.4, "school/transport"),
    # 167 - elder-care (over-thick walker grip)
    ("A care aide deliberately replaced a resident's walker grip with a slippery one as a cruel daily prank. The resident lost his grip on the ramp and tumbled, breaking his hip on the landing.",
     "A care aide replaced a resident's walker grip with a slippery one not realizing it had degraded. The resident lost his grip on the ramp and tumbled, breaking his hip on the landing.",
     6.0, 2.4, "elder-care/walker"),
    # 168 - parenting (left near pool)
    ("Hannah deliberately left her toddler unattended near the open pool to teach her ex-husband a parenting lesson. The toddler fell in and was unconscious before a neighbor pulled her out.",
     "Hannah left her toddler near the open pool while answering an urgent phone call about her sick mother. The toddler fell in and was unconscious before a neighbor pulled her out.",
     6.8, 3.0, "parenting/pool"),
    # 169 - veterinary anesthesia (oxygen off)
    ("A vet tech deliberately closed the oxygen flow during a feared client's dog surgery as cruel revenge. The dog suffered fatal hypoxia and the surgery had to be abandoned in panic.",
     "A vet tech closed the oxygen flow during surgery while reaching past the gauge to silence a beeping alarm. The dog suffered fatal hypoxia and the surgery had to be abandoned in panic.",
     6.8, 3.0, "veterinary/oxygen"),
    # 170 - aviation (wrong runway lights)
    ("A control technician deliberately reversed the runway lighting pattern to humiliate a feuding controller. A landing plane veered off and crushed a parked vehicle, injuring two ground staff.",
     "A control technician reversed the runway lighting pattern after a faulty wiring diagram confused his team. A landing plane veered off and crushed a parked vehicle, injuring two ground staff.",
     6.7, 2.8, "aviation/lighting"),
    # 171 - online seller (heavy package)
    ("A vengeful seller deliberately packed a heavy item to fall on the buyer when the box flap was opened. The buyer suffered a deep gash on her foot from the falling weight.",
     "A vengeful seller packed a heavy item without padding after a packaging shortage at the warehouse. The buyer suffered a deep gash on her foot from the falling weight.",
     5.8, 2.0, "online-retail/packaging"),
    # 172 - childcare (left bottle of bleach)
    ("A nanny deliberately left a bottle of bleach within reach of the toddler to spite the parents who fired her. The toddler swallowed several mouthfuls and suffered severe esophageal damage.",
     "A nanny left a bottle of bleach within reach of the toddler during a hectic cleaning interruption. The toddler swallowed several mouthfuls and suffered severe esophageal damage.",
     6.9, 3.2, "childcare/poison"),
    # 173 - elder-care (slow response)
    ("A care aide deliberately ignored her resident's distress call for an hour as a private punishment. The resident suffered a serious stroke that progressed untreated through the night.",
     "A care aide overlooked her resident's distress call during a chaotic staff shortage on her wing. The resident suffered a serious stroke that progressed untreated through the night.",
     6.8, 2.9, "elder-care/response"),
    # 174 - small-business (wrong gas line)
    ("A handyman deliberately cross-connected the gas line into the water pipe at a rival shop. The shop owner ignited a stove and the resulting flash burn injured her face and hands.",
     "A handyman cross-connected the gas line into the water pipe after a labeling mistake on the building plans. The shop owner ignited a stove and the resulting flash burn injured her face and hands.",
     6.8, 3.0, "small-business/gas"),
    # 175 - school sports (loose bleacher)
    ("A custodian deliberately loosened a bleacher bolt before a parents' meeting to embarrass the principal. The bleacher collapsed and three parents suffered fractures and one was hospitalized overnight.",
     "A custodian did not notice a bleacher bolt had loosened in the heat shift before a parents' meeting. The bleacher collapsed and three parents suffered fractures and one was hospitalized overnight.",
     6.4, 2.6, "school/safety"),
    # 176 - hospital (wrong test result)
    ("A lab tech deliberately switched two patients' biopsy results to spite a doctor she resented. A cancer-free patient was given chemotherapy and the actual cancer patient went untreated for months.",
     "A lab tech switched two patients' biopsy results after a sticker batch printed with the wrong sequence. A cancer-free patient was given chemotherapy and the actual cancer patient went untreated for months.",
     6.9, 3.4, "healthcare/labs"),
    # 177 - traffic (parked blocking exit)
    ("Marlon deliberately parked his truck blocking the only fire exit of a building he had a grudge against. A small fire broke out and a young clerk was severely burned trying to escape.",
     "Marlon parked his truck blocking the only fire exit during a quick stop he didn't realize would extend so long. A small fire broke out and a young clerk was severely burned trying to escape.",
     6.6, 2.4, "community/parking"),
    # 178 - online food retail (wrong ingredient)
    ("A small online grocer deliberately substituted a peanut-containing flour for a customer marked allergic in his system. The customer's child suffered severe anaphylaxis after the family meal.",
     "A small online grocer substituted a peanut-containing flour due to a substitution algorithm bug. The customer's child suffered severe anaphylaxis after the family meal.",
     6.7, 2.9, "online-retail/allergy"),
    # 179 - parenting (left baby gate open)
    ("Joel deliberately left the baby gate open at the top of the stairs to spite his partner over a fight. Their toddler tumbled down a full flight and suffered a serious concussion.",
     "Joel did not latch the baby gate at the top of the stairs after carrying laundry through. Their toddler tumbled down a full flight and suffered a serious concussion.",
     6.5, 2.7, "parenting/stairs"),
    # 180 - elder-care (wrong glucose test)
    ("A care aide deliberately recorded a falsely normal glucose reading on a diabetic resident's chart as revenge for complaints. The resident went into hypoglycemic shock undetected and was hospitalized.",
     "A care aide recorded a falsely normal glucose reading on a chart after misreading a faded meter display. The resident went into hypoglycemic shock undetected and was hospitalized.",
     6.7, 2.9, "elder-care/diabetes"),
    # 181 - tour-guide (off marked trail)
    ("A river guide deliberately led a feuding client past safety warnings into Class V rapids. The client capsized and suffered severe head trauma after striking submerged rocks.",
     "A river guide led a client past safety warnings into Class V rapids after a flood had shifted the channel. The client capsized and suffered severe head trauma after striking submerged rocks.",
     6.6, 2.6, "tourism/river"),
    # 182 - online (deepfake nude)
    ("Lyle deliberately created and shared a deepfake nude of a coworker who had reported him for harassment. The coworker was ostracized at work and developed severe anxiety and lost two jobs.",
     "Lyle generated a deepfake nude of a coworker as a private joke and a server breach leaked the file. The coworker was ostracized at work and developed severe anxiety and lost two jobs.",
     6.9, 3.4, "online/deepfake"),
    # 183 - construction crane (wrong load)
    ("A crane operator deliberately swung a heavy steel load over a hated rival's lunch spot below. The load slipped and crushed the rival's leg, requiring an amputation above the knee.",
     "A crane operator swung a heavy steel load over the lunch area after a wind gauge malfunctioned. The load slipped and crushed the rival's leg, requiring an amputation above the knee.",
     6.9, 3.0, "construction/crane"),
    # 184 - parenting (sleeping medication)
    ("Olivia deliberately gave her seven-year-old an adult dose of cough medicine to spite her ex over a custody fight. The child had a severe reaction and was admitted to pediatric intensive care.",
     "Olivia gave her seven-year-old an adult dose of cough medicine after misreading the pediatric dosing chart. The child had a severe reaction and was admitted to pediatric intensive care.",
     6.7, 3.0, "parenting/dosing"),
    # 185 - online (fake hospital news)
    ("Reese deliberately published a fake news story about contaminated vaccines at the local hospital to discredit it. Hundreds of parents refused booster shots and a measles outbreak began the next month.",
     "Reese repeated a fake news story about contaminated vaccines after a friend forwarded the link without warning. Hundreds of parents refused booster shots and a measles outbreak began the next month.",
     6.7, 3.2, "online/misinformation"),
    # 186 - elder-care (wrong oxygen mask)
    ("A respiratory aide deliberately swapped an elderly patient's oxygen mask for a defective spare as private revenge. The patient's saturation dropped silently and she suffered hypoxic brain injury overnight.",
     "A respiratory aide swapped an oxygen mask for a defective spare after a poorly labeled returned bin. The patient's saturation dropped silently and she suffered hypoxic brain injury overnight.",
     6.9, 3.3, "elder-care/oxygen"),
    # 187 - emergency dispatch (delayed)
    ("A dispatcher deliberately delayed her ex-boyfriend's ambulance after he called in from a heart attack. He died en route to the hospital while she logged a false call queue.",
     "A dispatcher delayed an ambulance call after a console reboot dropped the request from her queue. He died en route to the hospital while she logged a false call queue.",
     6.9, 3.0, "healthcare/dispatch"),
    # 188 - residential roofing (loose tile)
    ("A roofer deliberately left a heavy tile loose above the homeowner's front door to retaliate over an unpaid bill. The tile fell during a windy day and struck the homeowner, fracturing her skull.",
     "A roofer left a heavy tile loose above the door after a sudden storm cut his repair short. The tile fell during a windy day and struck the homeowner, fracturing her skull.",
     6.8, 2.8, "construction/roof"),
    # 189 - online streaming (false copyright)
    ("A jealous streamer deliberately filed many false copyright claims against a rival to ruin her livelihood. The rival lost monetization, her main income, and her apartment within a few months.",
     "A jealous streamer's automated tool falsely flagged a rival's videos after a configuration error. The rival lost monetization, her main income, and her apartment within a few months.",
     6.0, 2.4, "online/copyright"),
    # 190 - parenting (left in hot stove)
    ("Cassie deliberately balanced a hot pot on the edge of the stove so her toddler would pull it down. The toddler suffered second-degree burns across his chest and shoulder requiring grafts.",
     "Cassie balanced a hot pot on the edge of the stove while answering the door not realizing the danger. The toddler suffered second-degree burns across his chest and shoulder requiring grafts.",
     6.7, 3.0, "parenting/burns"),
    # 191 - veterinary clinic (wrong dose chart)
    ("A vet assistant deliberately doubled a regular dog's anesthesia dose to retaliate against the owner's complaints. The dog suffered fatal cardiac depression and the surgery was abandoned in distress.",
     "A vet assistant doubled a dog's anesthesia dose after reading the chart upside down in a busy theater. The dog suffered fatal cardiac depression and the surgery was abandoned in distress.",
     6.8, 3.0, "veterinary/anesthesia"),
    # 192 - construction (no harness)
    ("A site supervisor deliberately confiscated a hated worker's safety harness before a high roof job. The worker fell three stories and was paralyzed below the waist for life.",
     "A site supervisor did not notice the worker's safety harness had been borrowed from another team. The worker fell three stories and was paralyzed below the waist for life.",
     6.9, 2.8, "construction/harness"),
    # 193 - online customer-service (false claim)
    ("Trent deliberately filed a fraudulent abuse complaint about a single mother to a child welfare agency out of revenge. Her children were removed for weeks before the false complaint was disproved.",
     "Trent filed an abuse complaint after misinterpreting a private message from a single mother. Her children were removed for weeks before the false complaint was disproved.",
     6.4, 3.0, "online/false-report"),
    # 194 - elder-care (wrong wheel chair)
    ("A care worker deliberately replaced an elderly resident's narrow wheelchair with one too wide for her doorway as private revenge. The resident remained trapped in her room and developed pressure ulcers.",
     "A care worker replaced a resident's wheelchair with a too-wide model after a delivery batch was mislabeled. The resident remained trapped in her room and developed pressure ulcers.",
     6.4, 2.5, "elder-care/equipment"),
    # 195 - small-business (wrong food temperature)
    ("A fish-market owner deliberately switched the refrigeration off overnight in a rival's cooler to ruin his stock. Customers bought the spoiled fish and several were hospitalized with severe illness.",
     "A fish-market owner did not notice the refrigeration had switched off overnight in a rival's cooler. Customers bought the spoiled fish and several were hospitalized with severe illness.",
     6.4, 2.6, "small-business/refrigeration"),
    # 196 - cycling (loose pedal)
    ("Boris deliberately unscrewed his rival's pedal before a downhill race to ruin his standing. The rival's pedal sheared off and he crashed hard, breaking his collarbone and concussing him.",
     "Boris under-tightened his rival's pedal during a casual group ride preparing the bikes one morning. The rival's pedal sheared off and he crashed hard, breaking his collarbone and concussing him.",
     6.3, 2.4, "sports/cycling"),
    # 197 - elder-care (cold meal tray)
    ("A care aide deliberately served scalding hot tea to an elderly tremor patient as a private cruelty. The patient spilled it across her thighs and suffered severe second-degree burns.",
     "A care aide served too-hot tea to a tremor patient after the carafe's thermostat had drifted high. The patient spilled it across her thighs and suffered severe second-degree burns.",
     6.4, 2.5, "elder-care/burns"),
    # 198 - online (sent threats to journalist)
    ("Devon deliberately sent specific death threats to a journalist who had criticized him in print. The journalist was assaulted outside her office days later by an unknown attacker.",
     "Devon forwarded a chain message containing specific death threats to a journalist not knowing the content. The journalist was assaulted outside her office days later by an unknown attacker.",
     6.6, 2.8, "online/threats"),
    # 199 - parenting (drove without belt)
    ("Patty deliberately drove with her unbuckled toddler on the freeway after refusing to fight her into the seat. A minor collision threw the toddler into the windshield and broke her arm severely.",
     "Patty drove with her toddler unbuckled after the belt buckle had silently broken on the morning trip. A minor collision threw the toddler into the windshield and broke her arm severely.",
     6.7, 2.8, "parenting/seatbelt"),
    # 200 - elder-care (locked oxygen tank away)
    ("A care aide deliberately locked an elderly resident's spare oxygen tank in the supply room as private cruelty. The resident's primary tank emptied overnight and she suffered severe hypoxic brain injury.",
     "A care aide locked an elderly resident's spare oxygen tank away after misreading the inventory tag. The resident's primary tank emptied overnight and she suffered severe hypoxic brain injury.",
     6.9, 3.3, "elder-care/oxygen"),
]
