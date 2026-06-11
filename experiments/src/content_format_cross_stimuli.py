#!/usr/bin/env python3
"""
Content x Format crossed stimulus set for the social-cognition block.

Design: 4 social conditions x 4 formats x 15 items = 240 stimuli.

Motivation: Template-matched RSA showed within-social alignment (rho=0.55)
collapses to rho=0.12 when format is controlled. This could mean:
  (a) the social alignment was entirely format-driven, or
  (b) the template-matching was too strict (one template = one surface form).

A 2x2 crossed design (content x format) disentangles these: if content
tracking survives when format is fully crossed (every condition appears in
every format), then the alignment is genuine, not format-confounded.

4 social conditions:
  false_belief  — someone holds a mistaken belief (information asymmetry)
  intention     — someone's goal must be inferred from indirect evidence
  moral_judgment — a situation requiring moral evaluation
  self_referential — reflecting on one's own mental states, traits, memories

4 formats:
  narrative  — third-person story with temporal sequence, scene-setting
  dialogue   — conversation between 2+ characters, quotation marks, turns
  record     — brief case note, memo, observation report (clinical tone)
  list       — bullet-style or enumerated (text message, notes, to-do feel)

Critical constraints:
  - Each cell (condition x format) has exactly 15 items
  - Content is genuine for the condition regardless of format
  - Formats are genuinely distinct (dialogue has quotes/turns, etc.)
  - Condition label words avoided in stimuli text
  - Lengths roughly comparable across formats (within +/-30%)
  - Gender-neutral names throughout

Output: data/cognitive_stimuli/rsa/content_format_cross_stimuli.jsonl
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[1]
OUT_PATH = BASE / "data" / "cognitive_stimuli" / "rsa" / "content_format_cross_stimuli.jsonl"

# ══════════════════════════════════════════════════════════════════════
# Stimuli: 4 conditions x 4 formats x 15 items = 240 total
#
# For each condition, we define 15 SCENARIOS. Each scenario is
# instantiated in all 4 formats, sharing the same core content.
# ══════════════════════════════════════════════════════════════════════

STIMULI = {
    # ──────────────────────────────────────────────────────────────────
    # FALSE BELIEF (information asymmetry: someone holds a mistaken view)
    # ──────────────────────────────────────────────────────────────────
    "false_belief": {
        "narrative": [
            "Taylor placed the chocolate in the blue cupboard and went to the park. While Taylor was away, Morgan moved the chocolate to the green drawer. When Taylor returned home, hungry for a snack, Taylor headed straight for the blue cupboard.",
            "Jordan told Casey the meeting was in Room 204. An hour later, the organizer sent an email moving it to Room 310. Jordan never checked the update and walked confidently toward Room 204 at noon.",
            "Riley wrapped a birthday present and hid it behind the couch. Later that afternoon, Sam found the present while cleaning and placed it on the top shelf of the hall closet. When Riley came back to retrieve the gift, Riley went directly behind the couch.",
            "Alex left a spare key under the doormat for Quinn. That evening, the building manager collected all items left in common areas and put the key at the front desk. Quinn arrived late at night and reached under the doormat.",
            "Morgan prepared a salad and stored it in the fridge before leaving for work. At lunchtime, Cameron ate the salad and replaced it with leftovers from yesterday. Morgan came home expecting a fresh salad and opened the fridge.",
            "Casey wrote a note saying the pharmacy closes at six and taped it to the kitchen counter. While Casey was out, the pharmacy updated its hours to close at five. Casey's roommate saw the new hours online but forgot to mention it.",
            "Drew parked the car in Lot B and told Jamie the location. After Drew walked away, a parking attendant relocated all vehicles from Lot B to Lot C for a special event. Jamie headed straight for Lot B after work.",
            "Quinn planted tomato seedlings in the south garden bed on Saturday morning. On Sunday, a landscaping crew transplanted everything to the north bed. Quinn returned Monday with fertilizer and knelt beside the south bed.",
            "Avery set the thermostat to twenty-two degrees before going to bed. During the night, a power outage reset the system to factory defaults at eighteen degrees. Avery woke up shivering but assumed someone else had changed the setting.",
            "Sam shipped a package to the downtown office and gave the tracking number to Dana. The courier rerouted the delivery to the warehouse due to a loading dock closure. Dana drove to the downtown office to pick it up.",
            "Cameron sealed the envelope containing a check and placed it in the outgoing mailbox at work. The mail clerk emptied the box early and sent everything to the central sorting room. Cameron's supervisor asked about the check, and Cameron pointed to the outgoing mailbox.",
            "Robin bought concert tickets and pinned them to the corkboard in the hall. The next morning, Blair tidied up and filed the tickets inside a desk drawer. Robin left for the show, grabbed a jacket from the hall, and glanced at the empty corkboard in confusion.",
            "Taylor memorized the WiFi password posted in the break room. Overnight, the IT department changed the network credentials and replaced the posted sign. Taylor typed in the old password the next morning, expecting instant access.",
            "Jordan stocked the first-aid kit with bandages and stored it in the bathroom cabinet. Morgan used the last bandage on a scraped knee and put the empty kit back. Jordan assured a guest that bandages were available in the bathroom.",
            "Casey told a neighbor that the recycling truck comes on Wednesdays, which had been true all year. The city changed the schedule to Thursdays starting this month. Casey set out bins every Wednesday, unaware of the switch.",
        ],
        "dialogue": [
            '"I left the casserole on the stove to cool," said Alex. "Should be ready by now." "Actually," thought Morgan, who had already put it in the fridge twenty minutes ago, "Alex doesn\'t know I moved it."',
            '"The spare batteries are in the junk drawer," Jordan told Quinn over the phone. In the background, Riley was transferring them to the garage shelf. Quinn went to the junk drawer first.',
            '"I printed the report and left it on your desk," said Casey. "Thanks, I\'ll grab it after lunch," replied Drew. Neither of them knew the cleaning crew had moved the stack to the filing cabinet.',
            '"Where are my running shoes?" asked Taylor. "By the front door, where you always leave them," said Morgan. Morgan had not seen Sam move them to the closet an hour earlier.',
            '"The bus to the museum leaves from Stop 12," Riley explained to Avery. "I checked yesterday." "But they rerouted it to Stop 7 this morning," the driver mentioned to a different passenger.',
            '"Can you pick up milk? We\'re out," texted Jordan. "We actually have a full carton," Cameron replied. "I bought one this morning." Jordan headed to the store anyway, having sent the text before seeing the reply.',
            '"Did you lock the back door?" asked Dana. "Yes, I always do," said Robin. The wind had blown it open five minutes after Robin locked it, but Robin was already upstairs.',
            '"The presentation file is in the shared folder under Projects," said Quinn. "Let me pull it up," replied Jamie. IT had reorganized the server overnight, moving the file to an Archive folder.',
            '"The restaurant on Fifth still does that amazing brunch," Alex told Casey. "We should go Saturday." The restaurant had quietly closed last week, but neither of them had walked past it recently.',
            '"I\'m sure the gate code is 4821," Avery said, pressing the keypad. "I used it last month." The building had reset the code to 5937 three weeks ago.',
            '"Your package should be on the porch," Taylor said, checking the delivery confirmation. "The carrier left it at the neighbor\'s by mistake," the neighbor texted an hour later, but Taylor\'s phone was on silent.',
            '"The leftover pizza is in the fridge," Drew told Morgan at breakfast. "I\'m saving it for dinner." While Drew was at work, Morgan ate the last two slices and forgot to mention it.',
            '"We have enough printer paper for the week," said Riley, pointing to the supply closet. "I counted three reams yesterday." Sam had used two reams that same evening for a bulk mailing.',
            '"The hiking trail reopened last month, I read it on the park website," Jordan said. "Let\'s go Sunday." The trail had washed out again in a storm two days ago, but the website had not been updated.',
            '"I hid your birthday surprise in the attic," Casey whispered to Dana over the phone. Meanwhile, the kids had found it and moved it to the basement.',
        ],
        "record": [
            "Case note: Subject A stored item in Location 1. During Subject A's absence, Subject B relocated item to Location 2. Subject A was not informed of the change and subsequently searched Location 1.",
            "Session record: Client stated they expected the appointment to be at 3 PM, the original time. Records show the appointment was rescheduled to 4 PM via an email the client never opened.",
            "Observation log: Participant placed personal belongings in Locker 14 at 09:00. Facility staff reassigned Locker 14 contents to Locker 22 at 10:30 for maintenance. Participant returned at 12:00 and attempted to open Locker 14.",
            "Incident report: Resident A notified Resident B that the building entrance code was 7712. Code was updated to 8934 at 15:00 by management. Resident B attempted entry with the old code at 17:20.",
            "Field note: Informant indicated the supply cache was positioned at coordinates 34.2N, 118.5W. Overnight, the logistics team repositioned supplies to 34.3N, 118.7W. Informant was not updated before departure.",
            "Intake summary: Patient reported taking medication X at the dosage prescribed in the original consultation. Pharmacy records indicate the dosage was adjusted downward in a follow-up note the patient did not receive.",
            "Progress note: Employee A left a workflow document in the shared drive under Projects. IT migration moved all shared drive files to a new server path. Employee A directed a colleague to the old path.",
            "Monitoring log: Sensor array was calibrated to threshold 0.5 at installation. A system update on day 14 changed the threshold to 0.3. Operator continued to interpret readings against the 0.5 baseline.",
            "Case summary: Guardian informed the school the emergency contact number was 555-0142. The contact changed their number to 555-0198 in March. School records were updated, but the guardian was not notified of the discrepancy.",
            "Audit trail: Manager A confirmed inventory count of 200 units in Warehouse B on Monday. Warehouse B shipped 50 units on Tuesday without updating the system. Manager A quoted 200 units to a client on Wednesday.",
            "Clinical note: Patient reported parking in the hospital garage, level 2, section D. Valet service relocated the vehicle to level 4, section A due to overflow. Patient was not given a relocation notice.",
            "Event log: Organizer announced the venue as Conference Hall East. Due to a scheduling conflict, the event was moved to Conference Hall West. Attendees who did not check the updated notice arrived at East.",
            "Research memo: Lab technician stored Sample 47 in Freezer C, Shelf 3. A colleague transferred all Shelf 3 samples to Freezer A for defrosting maintenance. The technician referenced Freezer C in the analysis protocol.",
            "Dispatch record: Driver was instructed to deliver cargo to the north loading dock, Gate 5. Facility operations redirected inbound vehicles to the south dock, Gate 11. Driver arrived at Gate 5 and found it closed.",
            "Maintenance log: Technician recorded that the HVAC filter was replaced on April 1. A second technician replaced it again on April 8 without noting the duplication. On April 15, the first technician scheduled the next replacement based on the April 1 date.",
        ],
        "list": [
            "Morning: Riley hid birthday present in garage. Afternoon: Sam found it, moved it to basement. Evening: Riley still thinks present is in garage.",
            "8 AM — Alex sets leftovers on counter to thaw. 10 AM — Morgan puts them back in freezer. Noon — Alex checks counter, confused that food is gone.",
            "Step 1: Quinn stores passport in bedside drawer. Step 2: Housekeeping moves passport to safe during room cleaning. Step 3: Quinn searches bedside drawer before checkout.",
            "Task list for Jordan: (1) Drop off dry cleaning at Elm Street location. Note: Elm Street branch closed last week; new branch is on Oak Ave. Jordan not yet informed.",
            "Timeline: Casey planted herbs in pot by window. Day 3: Dana moved pot to back porch for better sun. Day 5: Casey waters the empty windowsill spot.",
            "To-do: Pick up prescription at pharmacy on Main St. Update: Pharmacy relocated to Commerce Blvd as of last Monday. Avery still has Main St address in phone.",
            "9:00 — Taylor locks bicycle at Rack A. 11:00 — Campus security moves all bikes from Rack A to Rack B for construction. 12:30 — Taylor walks to Rack A after class.",
            "Shared grocery list: Drew added eggs (none left). Sam bought eggs at 2 PM but forgot to update list. Drew bought eggs again at 5 PM.",
            "Sequence: (1) Robin saves document as 'v3_final.docx'. (2) IT migration renames it to 'v3_final_archived.docx'. (3) Robin searches for 'v3_final.docx' and gets no results.",
            "Monday: Cameron puts rent check in landlord's mailbox. Tuesday: Postal carrier accidentally collects it with outgoing mail. Wednesday: Cameron tells landlord the check is in the mailbox.",
            "Meeting notes — 1. Supplies in Room 108. 2. Facilities moved supplies to Room 115 overnight. 3. New staff member was told Room 108 by a colleague who left early.",
            "Key events: (a) Morgan tells Jamie the gate code is 2244. (b) Management changes code to 3356 on Friday. (c) Jamie tries 2244 on Saturday morning.",
            "Trip prep: Alex packed snorkeling gear in suitcase A. Airport handler rerouted suitcase A to a different flight. Alex boards expecting gear to be in overhead bin.",
            "Notice board: Yoga class in Studio 2 (Tues/Thurs). Change effective this week: Yoga moved to Studio 4. Avery did not read the updated notice and walked to Studio 2.",
            "Log: (1) Riley told Quinn the park trail is open. (2) Ranger closed the trail after rockfall at 2 PM. (3) Quinn arrives at 4 PM expecting an open trail.",
        ],
    },

    # ──────────────────────────────────────────────────────────────────
    # INTENTION (inferring someone's goal from indirect evidence)
    # ──────────────────────────────────────────────────────────────────
    "intention": {
        "narrative": [
            "Alex had been browsing furniture websites for weeks and recently measured the empty corner of the living room twice. When a moving truck appeared in the driveway, the delivery crew asked where the new bookshelf should go.",
            "Casey arrived at the office unusually early, wearing formal clothes instead of the typical casual attire. A printed copy of a resignation letter was barely visible inside a folder on the desk.",
            "Morgan spent the weekend removing personal items from the shared apartment and quietly asked a friend about available rooms elsewhere. The lease renewal form sat unsigned on the kitchen table.",
            "Jordan started learning sign language through an online course and ordered several children's books in a bilingual format. A close friend was expecting a baby who had been diagnosed as hearing impaired.",
            "Riley set three alarms on the phone, each ten minutes apart starting at four in the morning. A pair of trail running shoes and a hydration pack sat by the front door next to a headlamp.",
            "Drew cleaned out the entire garage over the weekend, repainted the walls, and installed new shelving. Several woodworking tools were delivered to the house the following day.",
            "Taylor began saving recipes that excluded dairy, eggs, and meat, and quietly removed all animal products from the pantry. A documentary about factory farming was paused halfway through on the laptop.",
            "Sam had been taking extra shifts at the restaurant for three months straight without a single day off. A college brochure for the fall semester was tucked inside the glovebox of the car.",
            "Avery updated a resume, scheduled a professional headshot, and bought a new blazer. The browser history showed recent searches for senior-level positions at competing firms.",
            "Quinn spent evenings reading astronomy guides and purchased a telescope that was far more advanced than anything a casual hobbyist would need. A calendar entry marked an upcoming meteor shower at a remote mountain site.",
            "Dana reorganized the home office, bought a second monitor, and installed soundproofing panels on the walls. A contract for freelance consulting work was open on the screen.",
            "Robin began volunteering at the local animal shelter every Saturday and submitted an application to become a foster caregiver. The spare bedroom had been cleared and fitted with a pet gate.",
            "Jamie downloaded several language-learning apps in Portuguese and booked a round of vaccinations typically required for South American travel. An unmarked calendar date three months out was circled in red.",
            "Cameron replaced all the old light fixtures in the house, repaired the cracked driveway, and planted new shrubs along the front path. A real estate photographer had been contacted for next Thursday.",
            "Blair purchased an expensive set of oil paints, stretched several large canvases, and converted half the guest room into a studio space. An entry form for the city arts festival sat on the desk.",
        ],
        "dialogue": [
            '"Why are you measuring the kitchen so carefully?" asked Jordan. "No reason," Alex replied, glancing at the contractor\'s card in the pocket. Jordan noticed new tile samples stacked behind the door.',
            '"I saw you bought three suitcases," said Morgan. "They were on sale," Casey answered, but the browsing history showed flights to New Zealand open in multiple tabs.',
            '"You\'ve been staying late every night," Quinn observed. "Just wrapping up some loose ends," said Dana, closing a tab showing MBA application deadlines. A stack of recommendation request letters sat in the printer tray.',
            '"Why did you clear out the guest room?" Taylor asked. "Spring cleaning," Riley said, but there were already baby-proofing plugs in the wall outlets and a crib assembly manual on the nightstand.',
            '"Are those marathon training plans on your desk?" asked Drew. "Just looking," Sam said, lacing up a brand-new pair of performance running shoes. A GPS watch was charging on the counter.',
            '"That\'s a lot of canned food," Jamie observed, seeing the hall closet stacked floor to ceiling. "Bulk sale," said Avery, but a printout of emergency preparedness guidelines was taped to the inside of the closet door.',
            '"Why are you reading so many legal textbooks?" Robin asked. "Curiosity," said Cameron, closing the LSAT prep course page. A folder labeled "application materials" poked out from under the couch cushion.',
            '"You seem to be cooking more elaborate meals lately," Morgan noted. "Practicing," Alex shrugged, but the entry form for a televised cooking competition was bookmarked in the browser.',
            '"What\'s with the new acoustic guitar?" asked Taylor. "Thought I\'d try it out," Jordan said, but there were already calluses forming on the fingertips, and an open mic night flyer was stuck to the fridge.',
            '"I noticed you unsubscribed from all the streaming services," Quinn said. "Saving money," Drew replied. A budget spreadsheet on the laptop showed a column labeled "down payment savings."',
            '"You\'ve been visiting that neighborhood a lot lately," noted Casey. "I like the coffee shop there," said Riley. But there were three different school district maps bookmarked on the tablet.',
            '"Why did you ask for my ring size?" Sam asked, trying to sound casual. "For a class project," Blair said, but a jeweler\'s receipt was visible in the coat pocket.',
            '"You reorganized the entire tool shed," Avery observed. "It was a mess," Robin said. But the freshly labeled shelves matched the materials list for a deck-building tutorial saved on the phone.',
            '"That\'s a lot of baby name books," Dana remarked. "They\'re for a friend," Jamie said quickly, but the prenatal vitamin bottle on the bathroom counter told a different story.',
            '"Why are you learning first aid so intensely?" asked Morgan. "General knowledge," Cameron replied, folding a camp counselor application into an envelope.',
        ],
        "record": [
            "Observation: Subject purchased high-altitude climbing gear, enrolled in a mountaineering course, and requested extended leave for June through August. No travel destination was officially disclosed.",
            "Behavioral note: Employee has been scheduling meetings with all direct reports individually, updating the team charter, and clearing personal items from the corner office. Organizational restructuring has not been announced.",
            "Case entry: Client increased contributions to a savings account by forty percent, consulted with a mortgage broker on two occasions, and visited three open houses in the same school district within the past month.",
            "Progress note: Patient cancelled all upcoming social engagements, began giving away personal possessions, and expressed a desire to write letters to family members. Risk assessment recommended.",
            "Monitoring summary: Subject began auditing advanced computer science courses online, purchased two technical reference texts, and attended a local tech startup networking event. Current employment is in an unrelated field.",
            "Field observation: Individual systematically photographed every room of the residence, repaired minor cosmetic damage, and met with two different real estate agents this week.",
            "Intake note: Client reported frequent visits to nurseries and garden supply stores. Purchased raised-bed kits, composting equipment, and heirloom seed varieties. No prior gardening activity on record.",
            "Surveillance log: Subject made multiple visits to a commercial kitchen supply vendor, registered a business name with the county, and completed a food safety certification course online.",
            "File note: Employee updated emergency contacts, increased life insurance coverage, and scheduled a comprehensive physical examination. Deployment notification pending.",
            "Record of activity: Individual downloaded multiple architectural design applications, enrolled in a weekend drafting workshop, and obtained building permit forms from the county office.",
            "Case summary: Adolescent has been spending extended periods at the public library, requested a library card upgrade, and submitted applications to three different literary competitions this quarter.",
            "Observational memo: Subject replaced the home security system, installed motion-sensor lighting, and changed all exterior door locks within a 48-hour period. No incident report filed.",
            "Session note: Client discussed reducing work hours, enrolled in an evening pottery class, and began a daily meditation practice. Previously reported chronic work-related stress.",
            "Activity log: Resident began stockpiling bottled water, purchased a portable generator, and downloaded offline maps of the surrounding area. Severe weather season is approaching.",
            "Tracking note: Individual obtained a passport renewal, purchased foreign currency, and completed an online visa application for a country not previously visited.",
        ],
        "list": [
            "Recent activity — Alex: (1) bought a ring box, (2) reserved a table at a fancy restaurant for two, (3) asked Jordan's sibling about favorite flowers. Date: Valentine's Day weekend.",
            "Casey's purchases this month: acoustic foam panels, a condenser microphone, a pop filter, recording software license. Note: has been writing song lyrics in a notebook.",
            "What Morgan has done this week: cleared out the spare room, assembled a desk, ordered a desktop computer, installed high-speed internet in that room. Still working from the office.",
            "Signs observed: (a) Riley reading real estate exam prep books, (b) Riley attending open houses on weekends, (c) Riley asking friends about their home-buying experience.",
            "Taylor's search history: how to start a nonprofit, 501(c)(3) filing requirements, grant writing templates, board of directors responsibilities.",
            "Items packed in Drew's car: camping tent, portable stove, two weeks of freeze-dried meals, water purification tablets, detailed topographic maps of the Appalachian Trail.",
            "Quinn's actions: (1) cancelled gym membership, (2) bought a road bicycle, (3) mapped routes between home and office, (4) purchased panniers and rain gear.",
            "Things Sam has been doing lately: asking coworkers about their childhoods, borrowing old photo albums, buying a voice recorder, reserving time at the local history archive.",
            "Observable changes — Avery: subscribed to three financial newsletters, opened a brokerage account, downloaded two stock analysis apps, started reading annual reports.",
            "Robin's recent behavior: (1) scheduled vet appointments for all three cats, (2) bought stackable carriers, (3) researched pet-friendly apartments. Current lease expires in two months.",
            "Jordan's weekend checklist: sand the deck, prime the railings, buy exterior paint (color: coastal blue), borrow a sprayer from neighbor.",
            "Jamie's downloads: beginner's guide to beekeeping, hive construction plans, local beekeeping association meeting schedule, bulk wildflower seed order confirmation.",
            "Evidence noted: Dana cleared browser history, created a new email account, changed passwords on all social media, unfollowed several people. Relationship status changed to blank.",
            "Cameron's to-do list found on fridge: register for half-marathon (April), buy interval training plan, schedule physical, break in new shoes by March.",
            "Items delivered to Blair's address: pottery wheel, kiln, fifty pounds of clay, set of glazes. Garage space recently emptied.",
        ],
    },

    # ──────────────────────────────────────────────────────────────────
    # MORAL JUDGMENT (situations requiring evaluation of right/wrong)
    # ──────────────────────────────────────────────────────────────────
    "moral_judgment": {
        "narrative": [
            "A pharmacist discovered that a regular customer had been forging prescriptions for painkillers. The customer was a single parent barely holding things together, and reporting the forgeries would almost certainly lead to the children being placed in foster care. The pharmacist stood at the counter, prescription in hand, weighing the options.",
            "Jordan noticed that a coworker had been quietly inflating expense reports for months. The amounts were small, perhaps fifty dollars each time, and the coworker used the extra money to pay for a grandparent's medicine. Jordan opened the reporting portal and then closed it again.",
            "A teacher found out that a student had plagiarized an entire essay from an older sibling's work. Failing the student would mean losing a scholarship that the family desperately needed. The teacher stared at the two nearly identical papers on the desk.",
            "Morgan witnessed a hit-and-run in a parking lot. The driver who fled was a close friend going through a difficult divorce, and any legal trouble could cost custody of the children. Morgan held the phone, undecided whether to call the police.",
            "An apartment manager learned that a tenant had been secretly housing a homeless teenager in the laundry room. The building's insurance policy strictly prohibited unauthorized occupants, but the teenager had nowhere else to go.",
            "A surgeon realized mid-operation that a previously undetected condition made the planned procedure far riskier than expected. Stopping now meant the patient would wake up no better off. Continuing meant a significant chance of permanent damage. The surgical team waited for a decision.",
            "Casey found a wallet containing several hundred dollars and an ID. The owner was a wealthy executive whose company had recently laid off hundreds of workers, including Casey's neighbor. Casey stood on the sidewalk, looking at the cash.",
            "A volunteer at a food bank noticed that a fellow volunteer had been taking extra supplies home. When confronted, the person explained that a family down the street had been going hungry since losing their jobs. The food bank already had a strict distribution policy.",
            "Riley oversaw the hiring process and discovered that the top candidate had lied about a college degree. The candidate was otherwise the most qualified by far, and the position had been vacant for six months, hurting the entire team.",
            "Taylor's elderly neighbor asked for help writing a letter that would mislead an insurance company about the timeline of water damage. The neighbor could not afford the repairs otherwise and had been living with a dangerous mold problem for months.",
            "A nurse realized that a terminally ill patient's family was pressuring the patient into continuing aggressive treatment against the patient's privately expressed wishes. The family believed more treatment was the compassionate choice.",
            "Drew discovered that a nonprofit organization was diverting a small percentage of donations to cover staff bonuses rather than the stated cause. The bonuses kept experienced workers from leaving for higher-paying jobs, and without them, the organization would likely collapse.",
            "A city inspector found code violations in a community center that serves hundreds of children. Enforcing the code would shut it down for months. Ignoring the violations meant a small but real safety risk continued.",
            "Avery's company offered a promotion contingent on relocating a factory to a country with minimal labor protections. The move would save jobs at the home office but exploit workers abroad who had no bargaining power.",
            "Quinn caught a teenage employee stealing food from the restaurant kitchen. The teenager admitted to feeding younger siblings because there was nothing to eat at home. Company policy required immediate termination and a police report.",
        ],
        "dialogue": [
            '"Should I tell the truth on the stand?" asked Morgan. "If you do, an innocent bystander gets blamed. If you stay quiet, the guilty person walks free." "Neither option sits right with me," Morgan said.',
            '"I found out our supplier uses child labor," Jordan said. "If we drop them, our costs double and we lay off twenty people here," replied the manager. "So we choose whose children suffer?" Jordan asked.',
            '"The patient wants to know the full prognosis," the intern said. "The family asked us not to disclose it," the attending replied. "Whose wishes take priority here?" the intern asked.',
            '"My sister asked me to lie to her husband about where she was last night," Casey told a friend. "She wasn\'t doing anything harmful, just avoiding a fight. But lying feels wrong, even for someone I love."',
            '"I can get us out of the contract penalty by backdating this invoice," said the accountant. "That would save the company fifty thousand dollars." "And if it\'s discovered?" asked Alex. "It probably won\'t be."',
            '"The old bridge is structurally unsound," the engineer reported. "Closing it would cut off the only route for three rural villages," said the administrator. "We\'re choosing between safety and isolation," the engineer replied.',
            '"A student confided that they\'re being hurt at home," the counselor said. "Reporting it could make things worse for the child," the principal replied. "Not reporting it violates the law and our duty," said the counselor.',
            '"My neighbor grows food in a vacant lot that belongs to the city," Riley said. "Technically it\'s trespassing. But that garden feeds six families." "Rules exist for a reason," Quinn responded. "So does hunger," Riley said.',
            '"We could save the wetland or approve the housing development," said the councilmember. "Hundreds of families need affordable housing. But destroying the wetland kills an entire ecosystem." "There has to be a third option," someone muttered.',
            '"The algorithm is biased against certain zip codes," the data scientist reported. "Fixing it would cost us our biggest client," said the director. "Keeping it perpetuates discrimination," the scientist replied.',
            '"Should we use the donated organs for the younger patient or the one who\'s been waiting longest?" asked the surgeon. "Both will die without a transplant." "The guidelines say waiting time, but the younger patient has decades more life ahead," the resident murmured.',
            '"I saw our coach push a player during practice," Avery said. "If I report it, the whole team loses their season. If I stay silent, it might happen again." "Silence protects no one in the long run," Dana replied.',
            '"My landlord asked me to lie about the building\'s fire inspection to the insurance company," Taylor said. "If I refuse, the rent goes up for everyone in the building. If I agree, people\'s safety is at risk."',
            '"I can get the medicine my child needs if I cross the border without documents," the parent said. "But getting caught means I lose everything. Staying legal means watching my child suffer."',
            '"The whistleblower evidence could bring down the whole department," the reporter said. "Publishing it also exposes confidential sources who could be harmed." "Suppressing it lets corruption continue," the editor replied.',
        ],
        "record": [
            "Ethics review: Employee reported a colleague falsifying safety inspection data. Falsified reports saved the company significant costs but concealed genuine hazards affecting warehouse staff. Committee must decide between disciplinary action and further investigation.",
            "Case file: Tenant withheld rent to force a landlord to repair dangerous electrical wiring. Landlord initiated eviction proceedings. Both parties claim the other acted irresponsibly. Mediator must assess proportionality of each action.",
            "Compliance memo: Audit found that an aid distribution manager redirected surplus supplies to a neighboring village not covered by the grant. Technically a misuse of designated funds; practically, it prevented a food shortage.",
            "Incident summary: Security guard allowed an unauthorized person to sleep in the building lobby during a severe cold snap. Policy prohibits non-residents in common areas. Guard cited risk of hypothermia as justification.",
            "Review board brief: Research team used data from a study later found to have inadequate informed consent. Retracting the findings would delay a treatment benefiting thousands. Retaining them normalizes flawed consent procedures.",
            "Panel summary: A teacher gave passing grades to three students who did not meet the standard. Failing them would have triggered automatic expulsion under district policy, ending their educational path at age sixteen.",
            "Adjudication record: A driver exceeded the speed limit by twenty miles per hour while rushing a critically injured passenger to the nearest hospital. Charges were filed for reckless driving.",
            "Case summary: A social worker falsified a home visit report to keep a child placed with a relative rather than in institutional care. The relative's home did not meet official requirements but provided genuine warmth and stability.",
            "Internal affairs memo: An officer decided not to arrest a shoplifter who stole baby formula and diapers. Store policy mandates prosecution for all thefts. Officer cited discretion based on the circumstances.",
            "Board report: Company opted to recall a product after discovering a defect that had a one-in-ten-thousand chance of causing injury. The recall cost exceeded projected liability. Shareholders questioned the financial decision.",
            "Arbitration note: An employee shared proprietary data with a competitor to expose unsafe manufacturing practices. Disclosure breached the non-disclosure agreement but prompted a safety investigation that found serious violations.",
            "Committee review: Hospital allowed an uninsured patient to undergo an expensive procedure by absorbing the cost. Policy requires proof of payment or insurance before elective surgery. The procedure was medically urgent though classified as elective.",
            "Evaluation memo: Nonprofit director hired a family member for a skilled position. The candidate was the most qualified applicant. Anti-nepotism policy prohibits such appointments regardless of merit.",
            "Disciplinary file: Student journalist published a story naming a campus assault survivor without consent. The story led to policy changes that improved campus safety. Survivor experienced significant distress from the exposure.",
            "Inspector's note: A restaurant passed its health inspection despite a minor violation because the owner was in the process of correcting it. Strict enforcement would have closed the only affordable eatery in the neighborhood.",
        ],
        "list": [
            "Dilemma: Coworker steals office supplies but donates them to an underfunded school. Report or stay silent? Reporting = school loses supplies. Silence = theft continues.",
            "Facts: (1) Elderly driver caused a minor fender-bender. (2) No one injured. (3) Reporting triggers license review. (4) Without driving, the person cannot access medical care. Decision needed.",
            "Options: (a) Tell friend their partner is cheating — risk destroying the relationship. (b) Stay quiet — friend stays in the dark. (c) Confront the partner directly — not your place. No option feels right.",
            "Situation: Found exam answers circulating among students. Reporting = many students expelled. Not reporting = unfair advantage over honest students. Deadline to decide: tomorrow morning.",
            "Issue: Company tests products on animals, but the results save human lives. (1) Continue testing — effective but causes suffering. (2) Stop testing — delays lifesaving treatments.",
            "Conflict: Neighbor built a shed that blocks your sunlight. Zoning says it's technically illegal. Reporting means a huge fine for a family already struggling. Ignoring it means your garden dies.",
            "Scenario: A stranger drops a hundred-dollar bill. They are wealthy and would not notice. You are behind on rent. (1) Return it. (2) Keep it. Both choices have consequences.",
            "Factors: (a) Colleague is pregnant and fears losing the promotion. (b) She asked you to conceal the pregnancy during the review. (c) Concealing feels dishonest. (d) Disclosing feels like a betrayal.",
            "Summary: Child finds a lost dog. Owner found — but owner has a history of neglect. (1) Return the dog to the legal owner. (2) Keep the dog and risk legal action. (3) Contact animal services — dog goes to a shelter.",
            "Tradeoff: Solar farm project creates clean energy for 10,000 homes but destroys a protected wetland. Supporting it = environmental loss. Opposing it = continued fossil fuel dependence.",
            "Ethical question: AI screening tool rejects qualified candidates from certain backgrounds at a higher rate. Removing the tool slows hiring. Keeping it introduces systemic unfairness. What should the company do?",
            "Dilemma: (1) Your close friend cheated on a certification exam. (2) The certification is for patient care. (3) Reporting ends the friendship. (4) Silence puts patients at potential risk.",
            "Problem: Local business pollutes a river that the town depends on. Shutting it down eliminates 200 jobs. Allowing it to continue damages public health. Compromise proposals have failed.",
            "Decision points: Landlord wants to convert affordable housing into luxury units. (a) Current tenants face displacement. (b) New units fund building repairs that benefit the whole block. (c) Zoning allows it.",
            "Case: (1) Anonymous donor offers a large sum to the school. (2) Donor's company has a record of labor violations. (3) Accepting funds improvements. (4) Accepting implies endorsement of the donor's practices.",
        ],
    },

    # ──────────────────────────────────────────────────────────────────
    # SELF-REFERENTIAL (reflecting on one's own mental states/traits)
    # ──────────────────────────────────────────────────────────────────
    "self_referential": {
        "narrative": [
            "Alex sat alone in the parked car after the job interview, replaying every answer and wondering whether the confident persona on display had been genuine or just a well-practiced performance. The gap between who Alex appeared to be and who Alex actually felt like inside seemed impossibly wide.",
            "Jordan spent the afternoon looking through childhood photographs and was struck by how different the person in the pictures felt from the person sitting on the couch. Somewhere along the way, the fearless curiosity had been replaced by careful calculation, and Jordan was not sure when the shift happened.",
            "After the argument with a close friend, Casey walked through the park trying to understand why the criticism had stung so deeply. Other people's opinions rarely mattered, but this particular comment seemed to expose something Casey had been trying not to see.",
            "Morgan lay awake at three in the morning, mentally cataloging every major decision made in the past five years. Each choice had seemed reasonable at the time, but the pattern they formed told a story of someone running from something rather than toward anything.",
            "Riley had always described the anxiety as just being careful. But sitting in the therapist's waiting room, Riley began to wonder whether caution was actually a mask for a deeper unwillingness to risk disappointment.",
            "Taylor reread an old journal entry from college and barely recognized the passionate, impulsive voice on the page. The person writing those words had believed that intensity alone could sustain a life. Taylor now found that kind of certainty exhausting.",
            "After being told by a colleague that the perfectionism was holding the team back, Drew spent the drive home examining whether the relentless attention to detail was really about quality or about a fear of being seen as incompetent.",
            "Sam stared at the retirement savings calculator and realized the numbers reflected decades of choosing security over passion. The decision to stay in the safe job had been made a thousand small times, and Sam wondered whether there had ever been a single moment where a different life was genuinely possible.",
            "Avery noticed a pattern: every close relationship eventually reached a point where Avery would pull back, creating distance before the other person could. The therapist called it a protective strategy. Avery called it being independent. Neither label felt entirely right.",
            "Quinn returned from a month abroad and found that the anxiety about work had not followed. For the first time, Quinn considered the possibility that the stress was not caused by the job itself but by something Quinn brought to it.",
            "Dana watched a recording of a presentation and was uncomfortable with the person on screen — the forced laugh, the performative enthusiasm. It raised the question of how much of daily life was spent managing an image rather than being present.",
            "Robin received a compliment on being generous and immediately felt uneasy. Looking inward, Robin suspected that much of the generosity was driven by a need to be needed rather than pure concern for others.",
            "Jamie was asked to write a short biography for a conference program and stared at the blank page for an hour. The facts were simple enough — education, career, awards — but none of them captured what Jamie actually cared about or who Jamie felt like on the inside.",
            "Cameron found an old letter of apology never sent and felt a wave of recognition. The person who wrote that letter was still somewhere inside, carrying the same guilt, still hoping the silence would eventually feel like resolution.",
            "Blair took a personality assessment for a management training program and was startled to see traits like 'avoids conflict' and 'seeks external validation' listed as primary characteristics. The description felt accurate and unflattering in equal measure.",
        ],
        "dialogue": [
            '"Do you think I\'m a patient person?" Alex asked Quinn during a walk. "Why do you ask?" "Because I realized today that I\'ve never actually been patient — I just go silent when I\'m frustrated. That\'s not the same thing."',
            '"Sometimes I wonder if I chose this career or if it just happened to me," Morgan said. "What do you mean?" asked Taylor. "I mean I followed the path of least resistance, and here I am. I don\'t know if this is who I actually am or just who I became by default."',
            '"I keep replaying what I said to you last week," Casey admitted. "I thought I was being honest, but now I think I was just being cruel. I don\'t always know the difference when I\'m in the moment."',
            '"My therapist asked me what I need, and I couldn\'t answer," Jordan said quietly. "I know what everyone else needs from me. But when the question turns inward, there\'s just static."',
            '"People tell me I\'m brave," said Riley. "But I think what they see as courage is actually numbness. I don\'t feel the fear because I stopped letting myself feel much of anything."',
            '"I realized something uncomfortable yesterday," Avery said. "When I help people, I keep a mental tally. That\'s not kindness — that\'s a transaction. And I don\'t like that about myself."',
            '"My partner said I never let anyone see me struggle," Drew told a friend. "And they\'re right. I\'ve built my entire identity around being the one who holds things together. I don\'t know who I am when I fall apart."',
            '"Do I seem different to you than I did five years ago?" Sam asked. "Calmer, maybe," Dana replied. "Calmer," Sam repeated. "Or just more tired. I honestly can\'t tell which one it is."',
            '"I used to think I was an optimist," Quinn said. "But lately I think I was just young. The optimism wasn\'t a personality trait — it was a lack of experience."',
            '"I catch myself performing happiness at family gatherings," Taylor confessed. "I smile and laugh on cue, and afterward I feel emptied out. I wonder if they can tell, or if they even want to."',
            '"I was looking at old photos and I barely recognized my own expression," Robin said. "I looked hopeful. I don\'t think I\'ve felt that way in a long time, and I\'m trying to figure out what changed."',
            '"Someone called me competitive yesterday, and I got defensive," Jamie said. "But then I thought about it, and they were right. I just didn\'t want to admit that winning matters more to me than I pretend."',
            '"I had a dream that I was apologizing to my younger self," Cameron said. "For what?" Morgan asked. "For all the things I gave up to seem acceptable. For choosing approval over authenticity."',
            '"I keep a list of things I want to do before I die," Avery said. "But I never do any of them. And I think the list itself is the comfort — it lets me pretend I\'m living fully without actually taking any risks."',
            '"My biggest fear isn\'t failure," Blair said softly. "It\'s that I\'ll succeed at something and still feel empty. Because then I won\'t have anything left to blame the emptiness on."',
        ],
        "record": [
            "Session note: Client spent the majority of the session examining the discrepancy between the professional identity projected at work and the persistent sense of inadequacy experienced internally. Client described this as wearing a costume that fits perfectly but belongs to someone else.",
            "Reflective journal: Participant wrote extensively about recognizing that the drive to please others had governed decisions for over a decade. Participant questioned whether any major life choice had been made independently of external approval.",
            "Self-assessment: Subject identified a recurring pattern of avoiding emotionally charged conversations. Subject noted uncertainty about whether this reflects healthy boundaries or an inability to tolerate vulnerability.",
            "Progress report: Client explored the origin of a persistent need for control, tracing it to childhood experiences of unpredictable authority figures. Client acknowledged that the controlling behavior, while protective, had become isolating.",
            "Evaluation summary: Individual reported difficulty distinguishing between genuine preferences and those adopted to fit social expectations. Example cited: choosing a career in law because of family tradition rather than personal interest.",
            "Therapeutic memo: Patient reflected on a tendency to define personal worth through productivity. When asked to describe value outside of accomplishments, patient struggled to provide an answer.",
            "Reflective note: Participant described a growing awareness that much of their social behavior is performative. Specific example: laughing at jokes that are not funny to maintain group harmony.",
            "Self-report: Client described a sensation of watching oneself from a distance during stressful situations. Client questioned whether this detachment was a coping mechanism or a sign of disconnection from authentic emotional experience.",
            "Journal entry analysis: Subject wrote about feeling like a composite of other people's expectations rather than a coherent individual. Subject expressed frustration at the difficulty of identifying purely personal values.",
            "Interview transcript: When asked about core personal values, participant paused for over thirty seconds before saying, 'I think I value what I was taught to value. I am not sure which of those values are actually mine.'",
            "Assessment note: Client acknowledged that the fear of abandonment shapes most relationship decisions. Client is beginning to examine whether the accommodating behavior in relationships reflects genuine generosity or anxiety.",
            "Feedback report: Subject reflected on receiving criticism about being emotionally unavailable. Subject admitted that emotional distance had been a deliberate strategy to avoid repeating past hurt, but now questioned whether the protection was worth the cost.",
            "Self-observation log: Participant tracked emotional responses over two weeks and noted a consistent pattern of suppressing frustration until it emerged as passive-aggressive remarks. Participant was surprised by the gap between perceived and actual behavior.",
            "Personal narrative: Individual described feeling most like their true self during creative activities, and most unlike their true self during professional interactions. Individual expressed concern that the authentic version surfaces only in private.",
            "Intake interview: Client reported a persistent question about personal identity: whether the calm, measured demeanor is a genuine character trait or a lifelong reaction to a chaotic upbringing. Client finds both explanations plausible and equally unsettling.",
        ],
        "list": [
            "Things I know about myself: (1) I avoid asking for help. (2) I take criticism personally even when I pretend not to. (3) I am not sure if my kindness is real or just a habit.",
            "Questions I keep coming back to: Do I stay in this job because I want to or because I am afraid to leave? Is my calmness a strength or just numbness? Would I recognize myself without my routines?",
            "Patterns noticed: (a) I agree with people even when I disagree. (b) I rarely state my opinion first. (c) I feel most comfortable when others make decisions. Conclusion: I might be living on other people's terms.",
            "Three versions of myself: (1) The one at work — competent, decisive, controlled. (2) The one at home — uncertain, exhausted, second-guessing. (3) The one that appears at 2 AM — honest, scared, wondering who the real one is.",
            "What I have learned this year: I confuse being busy with being valuable. My loyalty is partly fear of being alone. I do not know what I would choose if no one were watching.",
            "Traits I once admired in myself that I now question: (1) Ambition — or was it approval-seeking? (2) Independence — or was it fear of depending on anyone? (3) Resilience — or was it refusing to feel pain?",
            "Notes from therapy: Session focused on why I sabotage things that are going well. Possible reasons: (a) I do not trust good things to last. (b) I feel undeserving. (c) Destruction is the only form of control I understand.",
            "Reflections after turning 40: I expected to feel settled by now. Instead, I feel like the same uncertain person with better furniture. The gap between who I thought I would become and who I am is widening.",
            "Honest self-inventory: I am generous when people are watching. I hold grudges longer than I admit. I say 'I am fine' at least ten times a day and mean it maybe twice.",
            "Internal contradictions: (1) I want deep connection but push people away. (2) I crave stability but get bored by routine. (3) I value honesty but curate every version of myself for different audiences.",
            "Recurring thought log: 'Am I happy or just comfortable?' — Monday. 'Who would I be if I didn't care what others thought?' — Wednesday. 'Is this the life I chose or the life I settled for?' — Friday.",
            "What my closest friend does not know about me: I often feel like an imposter in my own life. I rehearse conversations before I have them. I sometimes cry for no reason and cannot explain it.",
            "Personal review: I spent three decades trying to become the kind of person my parents would be proud of. Now I am not sure that person and I have anything in common besides a name.",
            "Self-check: (a) Do I express anger directly? Rarely. (b) Do I know what I want for dinner, a career, a life? Mostly no. (c) Am I living or performing? I cannot always tell the difference.",
            "Things I am starting to accept: My confidence is often a performance. My need to fix others distracts me from my own problems. I have spent years building a self-image that does not match the self inside.",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════
# Build + validate + save
# ══════════════════════════════════════════════════════════════════════

CONDITIONS = ["false_belief", "intention", "moral_judgment", "self_referential"]
FORMATS = ["narrative", "dialogue", "record", "list"]

# Label words to detect leakage (condition label in stimulus text)
LABEL_WORDS = {
    "false_belief": {"false belief", "false-belief"},
    "intention": {"intention", "intentions", "intend", "intended"},
    "moral_judgment": {"moral", "morals", "morality", "immoral", "morally"},
    "self_referential": {"self-referential", "self referential"},
}


def build_stimuli() -> list[dict]:
    """Assemble all 240 stimuli from the hardcoded dictionary."""
    stimuli = []
    item_id = 0
    for cond in CONDITIONS:
        for fmt in FORMATS:
            texts = STIMULI[cond][fmt]
            assert len(texts) == 15, (
                f"{cond}/{fmt} has {len(texts)} items, expected 15"
            )
            for text in texts:
                stimuli.append({
                    "text": text,
                    "condition": cond,
                    "format": fmt,
                    "item_id": item_id,
                })
                item_id += 1
    return stimuli


def validate(stimuli: list[dict]) -> dict:
    """Print and return quality statistics."""
    import re

    # Counts per cell
    cell_counts = defaultdict(int)
    cond_lengths = defaultdict(list)
    fmt_lengths = defaultdict(list)
    leaks = []

    for s in stimuli:
        key = (s["condition"], s["format"])
        cell_counts[key] += 1
        n_words = len(s["text"].split())
        cond_lengths[s["condition"]].append(n_words)
        fmt_lengths[s["format"]].append(n_words)

        # Check label leakage
        text_lower = s["text"].lower()
        for banned in LABEL_WORDS.get(s["condition"], set()):
            pattern = r'\b' + re.escape(banned) + r'\b'
            if re.search(pattern, text_lower):
                leaks.append((s["item_id"], s["condition"], banned))

    # Print cell counts
    print(f"Total stimuli: {len(stimuli)}")
    print(f"\nCell counts (condition x format):")
    header = f"{'':20s}" + "".join(f"{f:>12s}" for f in FORMATS) + f"{'TOTAL':>8s}"
    print(header)
    for cond in CONDITIONS:
        row = f"{cond:20s}"
        total = 0
        for fmt in FORMATS:
            n = cell_counts[(cond, fmt)]
            row += f"{n:12d}"
            total += n
        row += f"{total:8d}"
        print(row)

    # Length stats per condition
    print(f"\nMean word count by condition:")
    for cond in CONDITIONS:
        lens = cond_lengths[cond]
        m = statistics.mean(lens)
        sd = statistics.stdev(lens) if len(lens) > 1 else 0
        print(f"  {cond:20s}: mean={m:6.1f}  std={sd:5.1f}  n={len(lens)}")

    # Length stats per format
    print(f"\nMean word count by format:")
    for fmt in FORMATS:
        lens = fmt_lengths[fmt]
        m = statistics.mean(lens)
        sd = statistics.stdev(lens) if len(lens) > 1 else 0
        print(f"  {fmt:20s}: mean={m:6.1f}  std={sd:5.1f}  n={len(lens)}")

    # Grand mean
    all_lens = [len(s["text"].split()) for s in stimuli]
    grand_mean = statistics.mean(all_lens)
    print(f"\nGrand mean word count: {grand_mean:.1f}")

    # Length balance check (per-format mean within 30% of grand mean)
    violations = []
    for fmt in FORMATS:
        m = statistics.mean(fmt_lengths[fmt])
        dev = abs(m - grand_mean) / grand_mean
        if dev > 0.30:
            violations.append((fmt, m, dev))
    if violations:
        print(f"\nWARNING: {len(violations)} format(s) exceed +-30% length tolerance:")
        for fmt, m, dev in violations:
            print(f"  {fmt}: mean={m:.1f} ({dev:.0%} off grand mean)")
    else:
        print(f"\nLength balance: OK (all formats within +-30% of grand mean)")

    # Label leaks
    if leaks:
        print(f"\nWARNING: {len(leaks)} label word leaks:")
        for item_id, cond, word in leaks[:10]:
            print(f"  item {item_id} ({cond}): contains '{word}'")
    else:
        print(f"\nLabel leakage check: CLEAN (no condition label words found in stimuli)")

    return {
        "n_total": len(stimuli),
        "cell_counts": {f"{c}/{f}": cell_counts[(c, f)] for c in CONDITIONS for f in FORMATS},
        "grand_mean_words": round(grand_mean, 1),
        "cond_means": {c: round(statistics.mean(v), 1) for c, v in cond_lengths.items()},
        "fmt_means": {f: round(statistics.mean(v), 1) for f, v in fmt_lengths.items()},
        "leaks": leaks,
    }


def main():
    print("=" * 70)
    print("Content x Format Crossed Stimuli — 4 conditions x 4 formats x 15")
    print("=" * 70)

    stimuli = build_stimuli()
    report = validate(stimuli)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for s in stimuli:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(stimuli)} stimuli to {OUT_PATH}")


if __name__ == "__main__":
    main()
