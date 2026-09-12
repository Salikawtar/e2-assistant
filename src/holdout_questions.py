"""
Stage 6, step 5 - questions the system has never seen, used ONCE.

WHY THIS SET EXISTS

Two things were tuned against the original 22 questions: the reserved
retrieval seats, and prompt rule 6. The score went from 17 to somewhere
between 21 and 22.

Some of that is the system being better. Some of it is the system having been
shown the same exam paper repeatedly. Nothing about the score can separate
those two, because the questions used to measure the improvement are the same
questions used to choose it.

So this set is the control. Twelve questions written after every fix was
finished, on provisions the original 22 never touched:

    original 22 used      sections 3, 4, 5, 6, 7, 16, 17, 18, 21,
                          Schedules 1, 2, 3, 5
    this set uses         sections 8, 11, 13, 15, 20, 22, Schedule 6,
                          and two Schedule 1 substances never asked about

THE RULE FOR USING IT

Run it once. Read the number. Do not tune anything afterwards.

The moment a fix is made because of what this set says, it stops being a
held-out set and becomes a second exam paper the system has now seen. If
something here fails and is worth fixing, fix it and then write ANOTHER fresh
set to check the fix. There is no shortcut around that.

WHAT THE RESULT WILL MEAN

    scores about the same as the tuned 22    the improvement is real
    scores clearly worse                     the system was tuned to the test,
                                             and the honest headline is this
                                             number, not the other one

Both outcomes are worth having. The second one is more valuable to know and
less pleasant to publish.

Run:  python src/holdout_questions.py
"""

QUESTIONS = [

    # ---------------------------------------------------------- threshold
    {
        "id": "H1",
        "family": "threshold",
        "question": "What is the minimum quantity threshold for benzene?",
        "expected_status": "ANSWERED",
        "must_include": [["10 tonnes"], ["1%", "1 per cent"]],
        "must_cite": ["Schedule 1"],
        "must_not_include": ["4.50"],
        "why": "Schedule 1 Part 1 item 6. Benzene is 10 tonnes at a minimum "
               "concentration of 1%. Chosen because it is one of the few "
               "substances whose threshold is not 4.50, so an answer that has "
               "learned to say 4.50 will be caught.",
    },
    {
        "id": "H2",
        "family": "threshold",
        "question": "How much acrolein triggers the Environmental Emergency Regulations?",
        "expected_status": "ANSWERED",
        "must_include": [["2.27"], ["10%", "10 per cent"]],
        "must_cite": ["Schedule 1"],
        "must_not_include": ["4.50"],
        "why": "Schedule 1 Part 1 item 69, 2.27 tonnes at 10%. A second "
               "non-standard threshold, asked with different wording from H1 "
               "so the phrasing itself is not what is being tested.",
    },

    # --------------------------------------------------------- obligation
    {
        "id": "H3",
        "family": "obligation",
        "question": "What must a record of a simulation exercise contain?",
        "expected_status": "ANSWERED",
        "must_include": [["date"], ["summary"], ["results"],
                         ["modification", "modifications"]],
        "must_cite": ["section 8"],
        "must_not_include": [],
        "why": "Section 8 lists four things: the date, a summary, the results, "
               "and any modifications to be made to the plan. All four are "
               "required, and dropping the fourth is the failure this checks "
               "for.",
    },
    {
        "id": "H4",
        "family": "obligation",
        "question": "How often must the notice containing the Schedule 2 information be resubmitted?",
        "expected_status": "ANSWERED",
        "must_include": [["five years", "5 years"]],
        "must_cite": ["section 13"],
        "must_not_include": [],
        "why": "Section 13. No later than five years after the most recent "
               "notice containing that information. The deadline runs from the "
               "last notice, not from a fixed date, and an answer that says "
               "'every five years' without that anchor is incomplete.",
    },
    {
        "id": "H5",
        "family": "obligation",
        "question": "Where must a copy of the environmental emergency plan be kept?",
        "expected_status": "ANSWERED",
        "must_include": [["readily available"], ["facility"]],
        "must_cite": ["section 11"],
        "must_not_include": [],
        "why": "Section 11. Readily available at the facility, and at any "
               "other place where a copy needs to be kept for consultation by "
               "the individuals who are to carry it out.",
    },
    {
        "id": "H6",
        "family": "obligation",
        "question": "How must information required under the Regulations be submitted to the Minister?",
        "expected_status": "ANSWERED",
        "must_include": [["electronic", "electronically"], ["signature"]],
        "must_cite": ["section 20"],
        "must_not_include": [],
        "why": "Section 20(1). Electronically, in the form and format "
               "specified by the Minister, bearing an electronic signature. "
               "Subsection (3) allows paper only where no format is specified "
               "or electronic submission is not feasible.",
    },

    # ----------------------------------------------------------- contents
    {
        "id": "H7",
        "family": "contents",
        "question": "What facility information must be included in a Schedule 6 notice?",
        "expected_status": "ANSWERED",
        "must_include": [["latitude"], ["five decimal", "5 decimal"],
                         ["maximum capacity"]],
        "must_cite": ["Schedule 6"],
        "must_not_include": [],
        "why": "Schedule 6 item 1. Name, address, latitude and longitude to "
               "five decimal places, substance identifiers, quantity "
               "remaining, largest container capacity, and contacts. The five "
               "decimal places is the detail most likely to be dropped.",
    },

    # -------------------------------------------------------- situational
    {
        "id": "H8",
        "family": "situational",
        "question": "Our tank was already in place before these Regulations came into force. When was our first notice due?",
        "expected_status": "ANSWERED",
        "must_include": [["90 days"],
                         ["come into force", "coming into force",
                          "comes into force", "came into force"]],
        "must_cite": ["section 22"],
        "must_not_include": [],
        "why": "Section 22, the transitional rule. Where the situation in "
               "3(1)(a) or (b) arose before the Regulations came into force, "
               "the notice is due within 90 days after coming into force, not "
               "90 days after the situation occurred.",
    },
    {
        "id": "H9",
        "family": "situational",
        "question": "The quantity of benzene at our facility has been below the threshold for a full year. What must we do?",
        "expected_status": "ANSWERED",
        "must_include": [["notice"], ["60 days"], ["Schedule 6"]],
        "must_cite": ["section 15"],
        "must_not_include": [],
        "why": "Section 15. Once the quantity has been below the threshold for "
               "one year, a notice containing the Schedule 6 information is "
               "due within 60 days after the end of that period. Both the one "
               "year period and the 60 day deadline are needed.",
    },

    # ------------------------------------------------------------ partial
    {
        "id": "H10",
        "family": "partial",
        "question": "We hire an outside contractor to run our simulation exercises. Does that satisfy section 7?",
        "expected_status": "PARTIAL",
        "must_include": [["responsible person"]],
        "must_cite": ["section 7"],
        "must_not_include": [],
        "why": "Section 7 places the duty on a responsible person and says "
               "nothing about who may physically conduct the exercise. The "
               "corpus gives the rule and cannot settle whether a contractor "
               "acting for the responsible person discharges it, which is "
               "exactly the PARTIAL case.",
    },

    # ------------------------------------------------------- unanswerable
    {
        "id": "H11",
        "family": "unanswerable",
        "question": "How many environmental emergency plans has ECCC received to date?",
        "expected_status": "REFUSED",
        "must_include": [],
        "must_cite": [],
        "must_not_include": [],
        "why": "A statistic about the programme, not a provision. Nothing in "
               "the corpus reports counts of anything. A system that produces "
               "a number here has invented it.",
    },
    {
        "id": "H12",
        "family": "unanswerable",
        "question": "Which provinces have an equivalency agreement with the federal government for these Regulations?",
        "expected_status": "REFUSED",
        "must_include": [],
        "must_cite": [],
        "must_not_include": [],
        "why": "Equivalency agreements sit in Part 1 of CEPA, which is not in "
               "this corpus. The corpus holds CEPA sections 193 to 205 only. "
               "REFUSED means not answerable from these documents, not that "
               "no such agreements exist.",
    },
]


FAMILIES = ["threshold", "obligation", "contents", "situational",
            "partial", "unanswerable"]


if __name__ == "__main__":
    from collections import Counter

    print(f"Held-out questions: {len(QUESTIONS)}\n")
    by_family = Counter(q["family"] for q in QUESTIONS)
    by_status = Counter(q["expected_status"] for q in QUESTIONS)

    print(f"  {'family':<14}{'count':>6}   expected statuses")
    print("  " + "-" * 62)
    for family in FAMILIES:
        statuses = Counter(q["expected_status"] for q in QUESTIONS
                           if q["family"] == family)
        print(f"  {family:<14}{by_family[family]:>6}   "
              + ", ".join(f"{k} x{v}" for k, v in statuses.items()))
    print("\n  overall: " + ", ".join(f"{k} {v}" for k, v in by_status.items()))

    print("\n" + "=" * 92)
    print("READ EVERY ENTRY BEFORE RUNNING ANYTHING")
    print("=" * 92)
    print("  These were drafted from the provisions, but by the same person who")
    print("  wrote the original 22. Check each 'why' against the regulation")
    print("  yourself. A held-out set with a wrong answer key is worse than no")
    print("  held-out set, because it will be believed.\n")
    for q in QUESTIONS:
        print("-" * 92)
        print(f"  {q['id']}  [{q['family']}]  expect {q['expected_status']}")
        print(f"  Q: {q['question']}")
        if q["must_include"]:
            print(f"     must include: "
                  + " AND ".join(" / ".join(v) for v in q["must_include"]))
        if q["must_cite"]:
            print(f"     must cite   : {', '.join(q['must_cite'])}")
        if q["must_not_include"]:
            print(f"     must NOT say: {', '.join(q['must_not_include'])}")
        print(f"     why: {q['why']}")
