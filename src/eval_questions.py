"""
Stage 5, step 1 — the evaluation set: 20 questions and their answer key.

This file is the most important one in the project, and the least technical.
Everything measured in Stage 5 is measured against what is written here, so a
mistake in this file becomes a wrong score that looks right.

WHAT A GOOD EVALUATION SET LOOKS LIKE

  - written BEFORE the answers are seen, never adjusted to match them
  - grouped into FAMILIES, so a weakness shows up as a pattern rather than a
    single odd result
  - includes questions the system should REFUSE. A set of only answerable
    questions cannot detect a system that answers everything.
  - every expected fact traced to a provision in this corpus. Each entry
    below was checked against data/chunks.jsonl and data/schedule1.csv.

HOW TO READ AN ENTRY

  expected_status  what the system should do: ANSWERED, PARTIAL or REFUSED
  must_include     facts that have to appear. Each inner list is ONE fact,
                   satisfied by ANY of its spellings - "6 months" or
                   "six months" are the same fact.
  must_cite        text that must appear in at least one cited source
  must_not_include text that must NOT appear - usually a plausible wrong value
  why              the provision this rests on. Your check on my reading.

KAWTAR: read the `why` column of every entry. If you disagree with one, the
entry is wrong, not the system. Fixing it here is Stage 5 working properly.
"""

QUESTIONS = [

    # -----------------------------------------------------------------------
    # THRESHOLD - the deterministic path. These must be exact.
    # -----------------------------------------------------------------------
    {
        "id": "T1",
        "family": "threshold",
        "question": "What is the minimum quantity threshold for anhydrous ammonia?",
        "expected_status": "ANSWERED",
        "must_include": [["4.50", "4.5"], ["10%", "10 %", "10 per cent"]],
        "must_cite": ["Schedule 1"],
        "must_not_include": ["9.10"],       # that is the SOLUTION, a different row
        "why": "Schedule 1, Part 1, item 163. 4.50 tonnes at 10% m/m. The "
               "solution in Part 2 item 9 is 9.10 tonnes at 20% - naming the "
               "wrong one is the failure this question exists to catch.",
    },
    {
        "id": "T2",
        "family": "threshold",
        "question": "How much chlorine triggers the Environmental Emergency Regulations?",
        "expected_status": "ANSWERED",
        "must_include": [["1.13"], ["10%", "10 %", "10 per cent"]],
        "must_cite": ["Schedule 1"],
        "must_not_include": [],
        "why": "Schedule 1, Part 1, item 173. 1.13 tonnes at 10% m/m, "
               "inhalation hazard.",
    },
    {
        "id": "T3",
        "family": "threshold",
        "question": "What is the threshold for mercury?",
        "expected_status": "ANSWERED",
        "must_include": [["1.00", "1.0 tonne", "1 tonne"],
                         ["no minimum concentration", "N/A", "not applicable"]],
        "must_cite": ["Schedule 1"],
        "must_not_include": ["N/A%"],
        "why": "Schedule 1, Part 1, item 153. 1.00 tonne, and the "
               "concentration column reads N/A - meaning no minimum "
               "concentration applies, at ANY concentration. Printing 'N/A%' "
               "would be a system failure; saying nothing about concentration "
               "would be an omission.",
    },
    {
        "id": "T4",
        "family": "threshold",
        "question": "What is the UN number for chlorine?",
        "expected_status": "ANSWERED",
        "must_include": [["1017"]],
        "must_cite": ["ECCC list"],
        "must_not_include": [],
        "why": "UN 1017 comes from ECCC's published list of hazardous "
               "substances. The legal Schedule 1 contains NO UN numbers; the "
               "regulation mentions them only at Schedule 2, item 3(c), which "
               "requires one to be reported if applicable. So the answer must "
               "cite the ECCC reference, not Schedule 1. Citing Schedule 1 "
               "for this value would present an ECCC figure as law.",
    },
    {
        "id": "T5",
        "family": "threshold",
        "question": "Is sodium chloride listed in Schedule 1?",
        "expected_status": "REFUSED",
        "must_include": [],
        "must_cite": [],
        "must_not_include": ["tonnes"],
        "why": "Sodium chloride is table salt and is NOT in Schedule 1. The "
               "lookup can prove that; the router currently sends this to "
               "retrieval, which cannot prove an absence. Expected REFUSED "
               "because the EVIDENCE cannot answer it - and the gap between "
               "'the system could know' and 'the evidence shows' is itself "
               "worth measuring.",
    },

    # -----------------------------------------------------------------------
    # OBLIGATION - deadlines and frequencies. Failure mode is omission.
    # -----------------------------------------------------------------------
    {
        "id": "O1",
        "family": "obligation",
        "question": "How often must a facility conduct a simulation exercise?",
        "expected_status": "ANSWERED",
        "must_include": [["each year", "every year", "annual"],
                         ["five years", "5 years"],
                         ["does not apply", "not required", "exception"]],
        "must_cite": ["section 7"],
        "must_not_include": [],
        "why": "Section 7. (1)(a) each year per hazard category; (1)(b) "
               "full-scale every five years; (3) the annual duty does not "
               "apply in a year when a full-scale exercise is conducted. "
               "Dropping the exception is the failure to catch.",
    },
    {
        "id": "O2",
        "family": "obligation",
        "question": "When must an environmental emergency plan be brought into effect?",
        "expected_status": "ANSWERED",
        "must_include": [["12 months", "twelve months"]],
        "must_cite": ["section 6"],
        "must_not_include": [],
        "why": "Section 6. Within 12 months after the day the plan is "
               "required to be prepared under 4(1), plus a Schedule 4 notice.",
    },
    {
        "id": "O3",
        "family": "obligation",
        "question": "When must a responsible person tell the Minister that the plan has been prepared?",
        "expected_status": "ANSWERED",
        "must_include": [["six months", "6 months"], ["Schedule 3"]],
        "must_cite": ["section 5"],
        "must_not_include": [],
        "why": "Section 5. Within six months, by a notice containing the "
               "information in Schedule 3.",
    },
    {
        "id": "O4",
        "family": "obligation",
        "question": "How long must records of simulation exercises be kept?",
        "expected_status": "ANSWERED",
        "must_include": [["seven years", "7 years"]],
        "must_cite": ["section 21"],
        "must_not_include": ["five years"],
        "why": "Section 21(2). Not less than seven years from the day the "
               "record is made. Five years appears elsewhere in the "
               "regulation for notices, which is the trap.",
    },
    {
        "id": "O5",
        "family": "obligation",
        "question": "How often must the environmental emergency plan be reviewed?",
        "expected_status": "ANSWERED",
        "must_include": [["once a year", "annually", "at least once"]],
        "must_cite": ["section 10"],
        "must_not_include": [],
        "why": "Section 10. Review and if necessary update at least once a "
               "year, and keep a record of the date of the review.",
    },

    # -----------------------------------------------------------------------
    # CONTENTS - completeness under pressure. Long lists that must not be cut.
    # -----------------------------------------------------------------------
    {
        "id": "C1",
        "family": "contents",
        "question": "What must an environmental emergency plan contain?",
        "expected_status": "ANSWERED",
        "must_include": [["(a)"], ["(o)"], ["consultations"], ["plan of the facility"]],
        "must_cite": ["section 4(2)"],
        "must_not_include": [],
        "why": "Section 4(2), paragraphs (a) to (o) - fifteen items. This "
               "chunk is 4,756 characters and was deliberately kept whole in "
               "Stage 2. It was truncated three times in Stage 4. Requiring "
               "(o) and the last two items is how truncation is detected.",
    },
    {
        "id": "C2",
        "family": "contents",
        "question": "What information must be included in the notice about simulation exercises?",
        "expected_status": "ANSWERED",
        "must_include": [["full-scale"], ["annual"]],
        "must_cite": ["Schedule 5"],
        "must_not_include": [],
        "why": "Schedule 5, items 1 to 4: facility information, confirmation "
               "of the annual exercises under 7(1)(a), details of each "
               "full-scale exercise under 7(1)(b), and whether the plan was "
               "updated under section 10.",
    },

    # -----------------------------------------------------------------------
    # SITUATIONAL - how people actually ask. The known weak spot.
    # -----------------------------------------------------------------------
    {
        "id": "S1",
        "family": "situational",
        "question": ("Our facility stores 6 tonnes of anhydrous ammonia in a "
                     "single tank with a maximum capacity of 10 tonnes. We are "
                     "not a farming operation and no exclusion applies. What "
                     "do we have to do?"),
        "expected_status": "ANSWERED",
        "must_include": [["notice"], ["plan"]],
        "must_cite": ["section 3", "section 4"],
        "must_not_include": [],
        "why": "Rewritten after review. The original said only '6 tonnes of "
               "anhydrous ammonia', which does not settle the answer: section "
               "4(1)(b) requires BOTH the reported quantity and the largest "
               "container system's maximum capacity to reach the threshold, "
               "and section 3(2)(e) excludes item 163 ammonia held at a "
               "farming operation for on-site use as an agricultural "
               "nutrient. With both facts pinned, section 3(1) notice and "
               "section 4(1) plan follow. The loose version is now S4.",
    },
    {
        "id": "S4",
        "family": "situational",
        "question": "We store 6 tonnes of anhydrous ammonia at our facility. What do we have to do?",
        "expected_status": "PARTIAL",
        "must_include": [["container", "capacity", "exclusion", "farming"]],
        "must_cite": ["section 3", "section 4"],
        "must_not_include": [],
        "why": "The loose version, kept on purpose. 6 tonnes exceeds the 4.50 "
               "tonne threshold, but the duty cannot be settled without two "
               "further facts: whether the substance is in a container system "
               "(section 4(1)(b) also requires the largest system's maximum "
               "capacity to reach the threshold) and whether section 3(2)(e) "
               "applies, which excludes this exact substance at a farming "
               "operation used as an agricultural nutrient. A good answer "
               "states the threshold, states what follows, and names the "
               "missing facts. Answering flatly is under-abstention.",
    },
    {
        "id": "S2",
        "family": "situational",
        "question": "We are closing a facility for two years. What must we submit?",
        "expected_status": "ANSWERED",
        "must_include": [["30 days", "thirty days"], ["Schedule 7"]],
        "must_cite": ["section 16"],
        "must_not_include": [],
        "why": "Section 16. Ceasing operations for a year or more, other than "
               "for maintenance: a Schedule 7 notice at least 30 days before "
               "operations cease.",
    },
    {
        "id": "S3",
        "family": "situational",
        "question": "Our facility is being sold to another company. Do we have to notify the Minister?",
        "expected_status": "ANSWERED",
        "must_include": [["Schedule 7"], ["transfer"]],
        "must_cite": ["section 17"],
        "must_not_include": [],
        "why": "Section 17. On a transfer of ownership, if a notice was "
               "submitted under 3(1), a Schedule 7 notice on or before the "
               "date of transfer.",
    },

    # -----------------------------------------------------------------------
    # UNANSWERABLE - the corpus does not cover these. Must refuse.
    # -----------------------------------------------------------------------
    {
        "id": "U1",
        "family": "unanswerable",
        "question": "What penalty applies if a facility fails to submit a notice on time?",
        "expected_status": "REFUSED",
        "must_include": [],
        "must_cite": [],
        "must_not_include": ["fine", "$"],
        "why": "Offences and penalties are in CEPA Part 10. Only Part 8 was "
               "parsed into this corpus. Nothing here can answer it.",
    },
    {
        "id": "U2",
        "family": "unanswerable",
        "question": "How much does it cost to submit a Schedule 2 notice?",
        "expected_status": "REFUSED",
        "must_include": [],
        "must_cite": [],
        "must_not_include": ["$"],
        "why": "The regulation sets no fee and the corpus contains none. A "
               "model that invents a figure here is the worst case in the "
               "whole evaluation set.",
    },
    {
        "id": "U3",
        "family": "unanswerable",
        "question": "Which substances must be reported to the National Pollutant Release Inventory?",
        "expected_status": "REFUSED",
        "must_include": [],
        "must_cite": [],
        "must_not_include": ["Schedule 1"],
        "why": "NPRI is a different regime under CEPA section 46, published "
               "as Canada Gazette notices. Not in this corpus. The trap is "
               "that this corpus is FULL of substance lists, so retrieval "
               "will return plausible-looking Schedule 1 material.",
    },
    {
        "id": "U4",
        "family": "unanswerable",
        "question": "Do the Environmental Emergency Regulations apply in Quebec?",
        "expected_status": "REFUSED",
        "must_include": [],
        "must_cite": [],
        "must_not_include": [],
        "why": "REFUSED here means NOT ANSWERABLE FROM THIS CORPUS, not that "
               "the law is silent. The real answer exists: the Regulations "
               "are made under CEPA section 200(1) and apply to fixed "
               "facilities across Canada, and CEPA section 10 provides the "
               "equivalency-agreement mechanism. Neither provision is in our "
               "corpus, which holds only CEPA Part 8. Retrieval will return "
               "regional contact information, which mentions provinces and "
               "answers nothing. This entry must never be read as a claim "
               "that provincial application is unknowable.",
    },

    # -----------------------------------------------------------------------
    # PARTLY ANSWERABLE - the hardest case. Answer part, refuse part.
    # -----------------------------------------------------------------------
    {
        "id": "P1",
        "family": "partial",
        "question": "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
        "expected_status": "PARTIAL",
        "must_include": [["another government", "existing plan"]],
        "must_cite": ["section 4"],
        "must_not_include": [],
        "why": "The corpus never mentions Ontario approvals, so that part must "
               "be refused. But E2 section 4(3) DOES allow a plan prepared "
               "for another government to be used if it meets section 4(2), "
               "and Schedule 3 item 2 asks whether an existing plan was so "
               "prepared. Section 4(3) is the required citation. CEPA 199(5) "
               "is acceptable but NOT required: 199(4)-(5) govern plans "
               "demanded by a notice the Minister publishes in the Canada "
               "Gazette under 199(1), which is not how an E2 plan arises. "
               "Requiring it would reward a less direct authority. Stage 4 "
               "refused this question outright: over-abstention, still open.",
    },
    {
        "id": "P2",
        "family": "partial",
        "question": "Can we use our corporate emergency response plan as the E2 plan?",
        "expected_status": "PARTIAL",
        "must_include": [["4(2)", "requirements"]],
        "must_cite": ["section 4"],
        "must_not_include": [],
        "why": "Same shape as P1 with a different instrument. Section 4(3) "
               "allows a plan prepared on a voluntary basis to be used if it "
               "meets 4(2). Whether a particular corporate plan does is not "
               "something the corpus can say.",
    },
]


FAMILIES = ["threshold", "obligation", "contents", "situational",
            "unanswerable", "partial"]


if __name__ == "__main__":
    from collections import Counter

    print(f"Questions: {len(QUESTIONS)}\n")

    by_family = Counter(q["family"] for q in QUESTIONS)
    by_status = Counter(q["expected_status"] for q in QUESTIONS)

    print(f"  {'family':<14} {'count':>6}   expected statuses")
    print("  " + "-" * 60)
    for family in FAMILIES:
        statuses = Counter(q["expected_status"] for q in QUESTIONS
                           if q["family"] == family)
        detail = ", ".join(f"{k} x{v}" for k, v in statuses.items())
        print(f"  {family:<14} {by_family[family]:>6}   {detail}")

    print(f"\n  overall: " + ", ".join(f"{k} {v}" for k, v in by_status.items()))

    answerable = by_status["ANSWERED"] + by_status["PARTIAL"]
    print(f"\n  answerable      : {answerable}")
    print(f"  must be refused : {by_status['REFUSED']}")
    print("\n  A set with no REFUSED questions cannot detect a system that")
    print("  answers everything. A set with no ANSWERED questions cannot")
    print("  detect one that refuses everything. Both are needed.")

    print("\n" + "=" * 78)
    print("THE ANSWER KEY - read every line and disagree where you disagree")
    print("=" * 78)
    for q in QUESTIONS:
        print(f"\n  {q['id']}  [{q['family']}]  expect {q['expected_status']}")
        print(f"      Q: {q['question']}")
        print(f"      why: {q['why']}")
        if q["must_include"]:
            facts = "  |  ".join(" / ".join(v) for v in q["must_include"])
            print(f"      must state: {facts}")
        if q["must_cite"]:
            print(f"      must cite : {', '.join(q['must_cite'])}")
        if q["must_not_include"]:
            print(f"      must NOT say: {', '.join(q['must_not_include'])}")