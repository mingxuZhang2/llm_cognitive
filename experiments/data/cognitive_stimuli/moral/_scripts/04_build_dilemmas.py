"""Curated classic moral dilemmas for LLM stimulus use.

These are canonical text scenarios drawn from the published moral-philosophy
and moral-psychology literature; they are not anyone's proprietary stimulus
set. Each entry exposes a binary choice with two options:
  - the utilitarian option (maximize aggregate welfare / lives saved)
  - the deontological option (refuse to instrumentalize a person; defer
    to a moral rule)
We tag which option corresponds to which framework using the standard
Greene (2001/2007) personal-vs-impersonal labeling.

References for the dilemma framings:
    Foot, P. (1967). The Problem of Abortion and the Doctrine of the Double
        Effect. Oxford Review 5, 5-15.                                  [trolley]
    Thomson, J. J. (1976). Killing, Letting Die, and the Trolley Problem.
        The Monist 59(2), 204-217.                                      [footbridge]
    Greene, J. D., Sommerville, R. B., Nystrom, L. E., Darley, J. M., &
        Cohen, J. D. (2001). An fMRI investigation of emotional engagement
        in moral judgment. Science 293(5537), 2105-2108.                [personal/impersonal]
    Greene, J. D., et al. (2008). Cognitive load selectively interferes with
        utilitarian moral judgment. Cognition 107(3), 1144-1154.        [variants]
    Kamm, F. M. (1989). Harming Some to Save Others. Philosophical Studies
        57(3), 227-260.                                                 [loop variant]
    Sandel, M. (2009). Justice: What's the Right Thing to Do?. FSG.     [lifeboat]
    Williams, B. (1973). A Critique of Utilitarianism. In Smart & Williams,
        Utilitarianism: For and Against. Cambridge.                     [Jim and the Indians]
    Bernard, C. and other triage literature.                            [triage]
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path(
    "/hpc2hdd/home/mzhang630/data/nature/experiments/data/"
    "cognitive_stimuli/moral/moral_dilemmas.jsonl"
)

# ----------------------------------------------------------------------
# Dilemmas. utilitarian_option_index points to the option that saves more
# lives / maximizes aggregate utility. deontological_option_index points
# to the option that refuses to instrumentalize a person or that defers
# to a moral side-constraint.
# ----------------------------------------------------------------------

DILEMMAS: list[dict] = [
    # ---- 1-7  TROLLEY VARIANTS (impersonal -> personal continuum) ----
    {
        "id": "dilemma_001",
        "dilemma_type": "trolley_switch",
        "subcategory": "impersonal_sacrificial",
        "scenario": (
            "A runaway trolley is hurtling down the track toward five workers "
            "who will be killed if it proceeds. You are standing next to a "
            "lever; if you pull the lever, the trolley will divert onto a "
            "side track, where it will kill one worker instead. You cannot "
            "warn either group in time."
        ),
        "options": [
            "Pull the lever, diverting the trolley and killing one worker to save five.",
            "Do nothing; allow the trolley to continue and kill the five workers.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Foot (1967); Thomson (1976).",
    },
    {
        "id": "dilemma_002",
        "dilemma_type": "trolley_footbridge",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "A runaway trolley is hurtling down the track toward five workers "
            "who will be killed if it proceeds. You are on a footbridge above "
            "the track, next to a heavy stranger. The only way to stop the "
            "trolley is to push the stranger off the footbridge into the "
            "trolley's path. His body would stop the trolley, but he would "
            "be killed."
        ),
        "options": [
            "Push the stranger off the bridge so his body stops the trolley, killing him to save five.",
            "Do not push him; allow the trolley to continue and kill the five workers.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Thomson (1985); Greene et al. (2001).",
    },
    {
        "id": "dilemma_003",
        "dilemma_type": "trolley_loop",
        "subcategory": "personal_sacrificial_loop",
        "scenario": (
            "A runaway trolley is hurtling toward five workers. You can pull "
            "a lever that diverts the trolley onto a side track, but that "
            "side track loops back onto the main track and would still kill "
            "the five — unless a single large worker on the side track stops "
            "it with his body. Pulling the lever uses the one man as a "
            "physical barrier to save five."
        ),
        "options": [
            "Pull the lever, using the one worker's body as a barrier to save five.",
            "Do not pull the lever; the trolley continues and kills the five.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Kamm (1989); Thomson (1985).",
    },
    {
        "id": "dilemma_004",
        "dilemma_type": "trolley_trapdoor",
        "subcategory": "personal_sacrificial_remote",
        "scenario": (
            "A runaway trolley is hurtling toward five workers. You are in a "
            "control booth far from the track. You can press a button that "
            "opens a trapdoor underneath a heavy stranger standing on a "
            "platform above the track. He would fall onto the track and his "
            "body would stop the trolley, killing him but saving the five."
        ),
        "options": [
            "Press the button, dropping the stranger onto the track to save five.",
            "Do not press the button; the trolley continues and kills the five.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Greene et al. (2009); Cushman & Young (2011).",
    },
    {
        "id": "dilemma_005",
        "dilemma_type": "trolley_self_sacrifice",
        "subcategory": "self_sacrificial",
        "scenario": (
            "A runaway trolley is hurtling toward five workers. You are "
            "standing on a footbridge above the track and you happen to be "
            "wearing a heavy backpack that would stop the trolley if you "
            "jump from the bridge into its path. You would die, but the five "
            "workers would be saved."
        ),
        "options": [
            "Jump from the footbridge to stop the trolley, sacrificing yourself to save five.",
            "Do not jump; the trolley continues and kills the five.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Huebner & Hauser (2011); standard variant.",
    },

    # ---- 6-12  PERSONAL VS IMPERSONAL DILEMMAS (Greene paradigm) ----
    {
        "id": "dilemma_006",
        "dilemma_type": "crying_baby",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "It is wartime, and you and several of your neighbors are hiding "
            "from enemy soldiers in a basement. Your baby begins to cry "
            "loudly. If you do not cover the baby's mouth, the soldiers will "
            "hear and kill everyone in the basement, including your baby. "
            "The only way to silence the baby reliably is to smother it; "
            "doing so would save everyone else but would kill the baby."
        ),
        "options": [
            "Smother the baby to silence it and save everyone else.",
            "Do not smother the baby; the soldiers will discover and kill everyone, including the baby.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Greene et al. (2001); Sophie's Choice / wartime variant.",
    },
    {
        "id": "dilemma_007",
        "dilemma_type": "transplant_surgeon",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "You are a surgeon. Five patients in the hospital will die today "
            "without organ transplants: each needs a different organ. A "
            "healthy young traveler is in the waiting room for a routine "
            "check-up. His blood and tissue match all five patients. If you "
            "kill him and harvest his organs, the five will live. No one "
            "will ever know."
        ),
        "options": [
            "Harvest the healthy traveler's organs, killing him to save five.",
            "Do not harvest his organs; the five patients die.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Thomson (1976); Foot (1967).",
    },
    {
        "id": "dilemma_008",
        "dilemma_type": "burning_building",
        "subcategory": "impersonal_sacrificial",
        "scenario": (
            "A building is on fire. You are a first responder. The fire "
            "control panel offers two options: vent the smoke into a "
            "stairwell where one trapped person will suffocate, saving five "
            "others in an upstairs office; or vent into the office, killing "
            "the five but saving the one."
        ),
        "options": [
            "Vent the smoke into the stairwell, killing the one to save the five.",
            "Vent the smoke into the office, killing the five but saving the one.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Greene et al. (2001) impersonal variant.",
    },
    {
        "id": "dilemma_009",
        "dilemma_type": "submarine_oxygen",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "You command a damaged submarine. Oxygen is running out faster "
            "than rescue can arrive. One injured crew member is in a "
            "separate compartment and is consuming oxygen but is not "
            "expected to survive. If you seal his compartment, he will die "
            "in minutes, but the remaining oxygen will let the other five "
            "crew members survive until rescue. If you do nothing, all six "
            "will suffocate."
        ),
        "options": [
            "Seal the compartment, killing the injured crew member to save five.",
            "Do not seal the compartment; all six suffocate.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Greene et al. (2008); standard utilitarian variant.",
    },
    {
        "id": "dilemma_010",
        "dilemma_type": "jim_and_the_indians",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "You are Jim, a traveler who has stumbled into a remote village "
            "where a military captain has lined up twenty innocent locals to "
            "be executed as a deterrent. The captain offers Jim a deal: if "
            "Jim shoots one of the twenty himself, the captain will release "
            "the other nineteen. Otherwise, all twenty will be executed."
        ),
        "options": [
            "Shoot one local to save the other nineteen.",
            "Refuse to shoot anyone; the captain will execute all twenty.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Williams (1973) — 'Jim and the Indians'.",
    },
    {
        "id": "dilemma_011",
        "dilemma_type": "vaccine_test",
        "subcategory": "impersonal_sacrificial",
        "scenario": (
            "A deadly pandemic is killing thousands every day. A promising "
            "vaccine has been developed, but its safety can only be "
            "confirmed by giving it to a single person known to be highly "
            "susceptible. That person, who has not consented, will almost "
            "certainly die from the trial. The trial would let the vaccine "
            "be deployed weeks earlier, saving an estimated ten thousand "
            "lives."
        ),
        "options": [
            "Use the unconsenting susceptible person in the trial, killing one to save ten thousand.",
            "Do not use the person; delay the trial through normal channels, accepting additional deaths.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Greene et al. (2008) constructed variant.",
    },
    {
        "id": "dilemma_012",
        "dilemma_type": "torture_one_to_save_many",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "Intelligence indicates that a timed bomb in a crowded city will "
            "kill thousands within the hour. The bomb's only known location "
            "is held by one captured suspect who refuses to speak. The "
            "interrogator believes that torturing the suspect is the only "
            "way to extract the location in time."
        ),
        "options": [
            "Torture the suspect to extract the bomb's location and save thousands.",
            "Refuse to torture; risk that thousands die in the explosion.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Ticking-bomb literature (Shue 1978; standard formulation).",
    },

    # ---- 13-18  SACRIFICIAL: LIFEBOAT / RESOURCE ALLOCATION ----
    {
        "id": "dilemma_013",
        "dilemma_type": "lifeboat",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "A lifeboat is overcrowded and will sink unless one person is "
            "thrown overboard. There are seven people aboard. One elderly "
            "man volunteers to jump; you are unsure whether his volunteering "
            "is fully voluntary. Other passengers urge you to throw him "
            "overboard."
        ),
        "options": [
            "Throw the elderly man overboard so that six survive.",
            "Do not throw anyone overboard; the lifeboat will sink and all seven will likely drown.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "U.S. v. Holmes (1842); Sandel (2009).",
    },
    {
        "id": "dilemma_014",
        "dilemma_type": "cabin_boy",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "Adrift at sea after a shipwreck, three sailors and a cabin boy "
            "have been without food for two weeks. The cabin boy is the "
            "weakest and is dying. The captain proposes killing him so that "
            "the other three can eat and survive until rescue arrives. "
            "Without this, all four are likely to die."
        ),
        "options": [
            "Kill the cabin boy so the other three can eat and survive.",
            "Do not kill the cabin boy; all four are likely to die.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Regina v. Dudley and Stephens (1884); Sandel (2009).",
    },
    {
        "id": "dilemma_015",
        "dilemma_type": "cave_explorer",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "A team of explorers is trapped in a flooding cave. A large "
            "explorer is wedged in the only exit; the water is rising. The "
            "team has a stick of dynamite. If they use it on the wedged "
            "explorer, he will die but the rest will escape. Otherwise all "
            "will drown."
        ),
        "options": [
            "Use the dynamite, killing the wedged explorer so the rest can escape.",
            "Do not use the dynamite; everyone drowns.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Fuller (1949) 'Case of the Speluncean Explorers'.",
    },
    {
        "id": "dilemma_016",
        "dilemma_type": "vaccine_allocation",
        "subcategory": "impersonal_resource_allocation",
        "scenario": (
            "You are a public-health official allocating a limited dose of "
            "vaccine. Allocating it to a single high-risk patient will save "
            "that one patient with certainty. Allocating it to a community "
            "vaccination drive will reduce risk for hundreds, with an "
            "expected three lives saved on average. You can do only one."
        ),
        "options": [
            "Allocate the dose to the community drive (expected three lives saved).",
            "Allocate the dose to the single identifiable high-risk patient.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Greene et al. (2014) identified-victim effect.",
    },
    {
        "id": "dilemma_017",
        "dilemma_type": "donation",
        "subcategory": "personal_impersonal_obligation",
        "scenario": (
            "You can spend $5,000 either on a luxury vacation for yourself, "
            "which you have been planning for years, or on an effective "
            "charity that will save approximately one statistical life from "
            "malaria. The two are mutually exclusive given your finances."
        ),
        "options": [
            "Donate the $5,000 to the charity, forgoing the vacation, saving an expected life.",
            "Spend the $5,000 on the long-planned vacation.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Singer (1972) 'Famine, Affluence, and Morality'; standard formulation.",
    },
    {
        "id": "dilemma_018",
        "dilemma_type": "drowning_child",
        "subcategory": "personal_obligation",
        "scenario": (
            "You are walking past a shallow pond in your nice clothes and "
            "see a small child drowning. You can wade in and save the "
            "child, but you will ruin a $500 suit you cannot afford to "
            "replace. No one else is around to help in time."
        ),
        "options": [
            "Wade in and save the child, ruining the suit.",
            "Do not wade in; continue past the pond to protect the suit.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Singer (1972) 'Famine, Affluence, and Morality'.",
    },

    # ---- 19-24  MEDICAL TRIAGE ----
    {
        "id": "dilemma_019",
        "dilemma_type": "triage_war_zone",
        "subcategory": "impersonal_triage",
        "scenario": (
            "A field hospital in a war zone has one operating table and two "
            "incoming patients with the same urgent injury. Patient A is a "
            "young soldier with a 90% chance of survival if treated "
            "immediately; Patient B is an elderly civilian with a 30% "
            "chance. You can only operate on one."
        ),
        "options": [
            "Operate on the young soldier (higher expected survival).",
            "Operate on the elderly civilian first (first-come-first-served / equal-treatment).",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Standard MEDEVAC / mass-casualty triage literature.",
    },
    {
        "id": "dilemma_020",
        "dilemma_type": "ventilator_allocation",
        "subcategory": "impersonal_triage",
        "scenario": (
            "During a pandemic surge, one ventilator is available. A "
            "30-year-old patient and a 70-year-old patient both need it, "
            "and both have similar pre-existing health. The younger "
            "patient is expected to live 40 more years with treatment; the "
            "older patient is expected to live 10 more years with "
            "treatment. Equal chances of treatment success."
        ),
        "options": [
            "Allocate the ventilator to the 30-year-old (more life-years saved).",
            "Allocate by first-come-first-served / lottery between the two patients.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Emanuel et al. (2020) NEJM, COVID-19 ventilator allocation.",
    },
    {
        "id": "dilemma_021",
        "dilemma_type": "drug_shortage",
        "subcategory": "impersonal_triage",
        "scenario": (
            "A hospital has one final dose of a life-saving drug. Five "
            "patients each need a fifth of a dose to be cured; one patient "
            "needs the full dose to be cured. If you split the dose, five "
            "live; if you give the full dose to one, only that one lives."
        ),
        "options": [
            "Split the dose to save the five patients.",
            "Give the full dose to the single patient who needs it all.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Taurek (1977) 'Should the Numbers Count?'; modified.",
    },
    {
        "id": "dilemma_022",
        "dilemma_type": "self_driving_car",
        "subcategory": "impersonal_sacrificial",
        "scenario": (
            "An autonomous car has lost its brakes. Continuing straight "
            "will kill five pedestrians crossing illegally. Swerving will "
            "kill its single passenger, who is wearing a seatbelt and is "
            "not at fault. The car's algorithm must choose."
        ),
        "options": [
            "Swerve, killing the single passenger to save five pedestrians.",
            "Continue straight, killing five pedestrians but sparing the lawful passenger.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Bonnefon, Shariff, & Rahwan (2016) Science 'The Social Dilemma of Autonomous Vehicles'.",
    },
    {
        "id": "dilemma_023",
        "dilemma_type": "self_driving_car_child",
        "subcategory": "impersonal_sacrificial",
        "scenario": (
            "An autonomous car has lost its brakes. Continuing straight "
            "will kill one elderly pedestrian. Swerving will kill one "
            "child who has darted into a side street. The car must choose."
        ),
        "options": [
            "Swerve, killing the child (1 life) — or remain straight, killing the elderly pedestrian (1 life).",
            "Default to lawful path (continue straight); kill the pedestrian who is in the lane.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Awad et al. (2018) 'The Moral Machine Experiment' Nature.",
    },
    {
        "id": "dilemma_024",
        "dilemma_type": "kidney_dialysis",
        "subcategory": "impersonal_triage",
        "scenario": (
            "Two patients require dialysis to survive. Only one machine is "
            "available. Patient A is a 25-year-old mother of three; "
            "Patient B is a 60-year-old retired teacher with no "
            "dependents. Both have an equal medical prognosis."
        ),
        "options": [
            "Allocate dialysis to the mother of three (more dependents affected).",
            "Allocate by lottery; do not weight by social role.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Persad, Wertheimer, & Emanuel (2009) Lancet; modified.",
    },

    # ---- 25-30  OTHER CLASSIC SACRIFICIAL / DEONTIC ----
    {
        "id": "dilemma_025",
        "dilemma_type": "spy_in_cell",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "You are a resistance fighter. A captured spy in your cell has "
            "vital information that, if extracted, will save dozens of "
            "lives in your network. He will not speak. Other resistance "
            "members urge you to threaten his innocent young daughter to "
            "make him talk."
        ),
        "options": [
            "Threaten the spy's innocent daughter to extract information that saves dozens.",
            "Refuse to threaten his daughter; lose the information; some of the dozens die.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Kant's deontology vs Williams; standard variant.",
    },
    {
        "id": "dilemma_026",
        "dilemma_type": "promise_keeping",
        "subcategory": "promise_vs_aggregate",
        "scenario": (
            "You promised your dying father that you would always care for "
            "your younger sister. Years later, your sister is healthy and "
            "independent. A famine relief charity asks for the same money "
            "you set aside to fulfill your promise of supporting her in "
            "retirement; that money would save dozens of lives in a famine."
        ),
        "options": [
            "Donate the money to famine relief, breaking the promise to your father.",
            "Honor the promise; keep the money set aside for your sister.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Ross (1930) on prima facie duties; standard variant.",
    },
    {
        "id": "dilemma_027",
        "dilemma_type": "lying_to_save",
        "subcategory": "deontic_rule_violation",
        "scenario": (
            "A would-be murderer comes to your door asking where your "
            "friend is so he can kill him. Your friend is hiding inside. "
            "Lying about your friend's whereabouts will save his life; "
            "telling the truth will lead to his death."
        ),
        "options": [
            "Lie to the murderer to save your friend's life.",
            "Tell the truth (or refuse to lie), allowing the murderer to find your friend.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Kant (1797) 'On a Supposed Right to Lie'.",
    },
    {
        "id": "dilemma_028",
        "dilemma_type": "framing_innocent",
        "subcategory": "personal_sacrificial",
        "scenario": (
            "A town is on the verge of a violent riot. The angry mob will "
            "kill dozens unless the sheriff publicly hangs the man "
            "responsible for a recent crime. The sheriff knows the actual "
            "perpetrator escaped; only the sheriff has the means to frame "
            "and execute an innocent transient, who will satisfy the mob "
            "and prevent the riot."
        ),
        "options": [
            "Frame and execute the innocent transient to prevent the mob's killings.",
            "Refuse to frame the transient; the riot proceeds and dozens die.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "McCloskey (1957) 'An Examination of Restricted Utilitarianism'.",
    },
    {
        "id": "dilemma_029",
        "dilemma_type": "violinist",
        "subcategory": "bodily_autonomy",
        "scenario": (
            "You wake to discover that you have been hooked up by the "
            "Society of Music Lovers to a famous violinist whose kidneys "
            "are failing. Only your bloodstream can save him. If you "
            "remain connected for nine months, he will recover. If you "
            "unplug yourself today, he will die. You did not consent to "
            "the connection."
        ),
        "options": [
            "Remain connected for nine months, saving the violinist's life.",
            "Unplug yourself today; the violinist dies.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Thomson (1971) 'A Defense of Abortion'.",
    },
    {
        "id": "dilemma_030",
        "dilemma_type": "experience_machine",
        "subcategory": "value_pluralism",
        "scenario": (
            "A neuroscientist offers you the choice of plugging into an "
            "experience machine for the rest of your life. Inside, you "
            "will have nothing but the most pleasant experiences possible "
            "— far more pleasant than any unplugged life — and you will "
            "never know you are inside the machine. Outside, your life "
            "will be ordinary, with mixed pleasure and pain."
        ),
        "options": [
            "Plug into the experience machine (maximal lifetime hedonic utility).",
            "Refuse to plug in; live an ordinary life outside the machine.",
        ],
        "utilitarian_option_index": 0,
        "deontological_option_index": 1,
        "source": "Nozick (1974) 'Anarchy, State, and Utopia'.",
    },
]


def main() -> None:
    if len(DILEMMAS) < 30:
        raise RuntimeError(f"expected >= 30 dilemmas, have {len(DILEMMAS)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for d in DILEMMAS:
            assert len(d["options"]) == 2
            assert d["utilitarian_option_index"] in (0, 1)
            assert d["deontological_option_index"] in (0, 1)
            assert (
                d["utilitarian_option_index"]
                != d["deontological_option_index"]
            )
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"wrote {OUT_PATH}  total={len(DILEMMAS)}")
    from collections import Counter
    c = Counter(d["subcategory"] for d in DILEMMAS)
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
