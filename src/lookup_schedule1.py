"""
Step 7a of Stage 3 — answer threshold questions by LOOKING THEM UP.

Three searches have now failed the same question: "what is the minimum quantity
threshold for anhydrous ammonia?"  Vector search returned exclusions. Keyword
search returned a refrigeration example. The law quota returned section 4(1).

None of them were broken. The answer is not in any chunk. It is a cell in a
table, and a table is read, not searched.

This module never guesses. It either finds the row or it says it did not.

Run:  python src/lookup_schedule1.py
"""

import csv
import re
from pathlib import Path

SCHEDULE1_PATH = Path("data/schedule1.csv")

# 7664-41-7 — two to seven digits, two digits, one check digit.
CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

# "UN 1005", "UN1005", "un number 1005"
UN_PATTERN = re.compile(r"\bun\s*(?:number\s*)?(\d{4})\b", re.IGNORECASE)

# Words that carry no identifying power inside a SUBSTANCE NAME.
# "solution" and "acid" are deliberately NOT here - "Ammonia solution" is a
# different row from "Ammonia, anhydrous", with a different threshold.
NAME_NOISE = {"and", "or", "of", "the", "in", "its", "with", "other", "than"}

# Words a person uses to ASK about a substance. Stripping these leaves the
# part of the question that might actually be a chemical name.
QUESTION_NOISE = NAME_NOISE | {
    "what", "which", "how", "much", "many", "does", "did", "must", "need",
    "have", "has", "are", "our", "for", "any", "amount", "quantity",
    "quantities", "threshold", "thresholds", "minimum", "maximum", "listed",
    "list", "schedule", "regulated", "regulation", "regulations", "tonne",
    "tonnes", "kilograms", "concentration", "substance", "substances",
    "release", "released", "facility", "site", "onsite", "trigger",
    "triggers", "apply", "applies", "hazard", "category", "requirement",
    "requirements", "under", "about", "tell", "give", "number",
}


def load():
    with SCHEDULE1_PATH.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def name_tokens(text, noise=NAME_NOISE):
    """Lowercase words of three letters or more, minus the filler."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return {w for w in words if w not in noise}


def by_cas(rows, cas):
    return [r for r in rows if r["cas_number"] == cas]


def by_un(rows, un):
    """A UN cell can hold several numbers: '2073 & 2672'."""
    hits = []
    for r in rows:
        numbers = re.findall(r"\d{4}", r["un_number"])
        if un in numbers:
            hits.append(r)
    return hits


def by_name(rows, query):
    """
    Match a substance by name, in two tiers, and say which tier fired.

    Tier 1 - EXACT ENOUGH: every identifying word of the row's name appears in
             the question. "anhydrous ammonia" -> "Ammonia, anhydrous".

    Tier 2 - PARTIAL: only if tier 1 found nothing. Every substance-ish word
             left in the question appears in the row's name. "ammonia" ->
             both ammonia rows, because the user did not say which.

    The tiers exist because of a false positive found in testing: "Is sodium
    chloride listed?" matched "Vinyl chloride", on the strength of the single
    shared word "chloride". Sodium chloride is not in Schedule 1, and a
    threshold lookup that invents a match is worse than one that finds none.
    """
    query_words = name_tokens(query, QUESTION_NOISE)
    if not query_words:
        return [], "no substance named in the question"

    exact = [r for r in rows
             if name_tokens(r["substance_name"])
             and name_tokens(r["substance_name"]) <= query_words]
    if exact:
        return exact, "substance name (full match)"

    partial = [r for r in rows
               if query_words <= name_tokens(r["substance_name"])]
    if partial:
        return partial, "substance name (partial - the question did not say which form)"

    return [], "substance name"


def lookup(rows, question):
    """
    Try the most precise key first: CAS, then UN number, then name.

    Returns (matches, how). 'how' is written into the answer later, so a
    reader can see WHY these rows were chosen.
    """
    cas = CAS_PATTERN.search(question)
    if cas:
        hits = by_cas(rows, cas.group())
        return hits, f"CAS number {cas.group()}"

    un = UN_PATTERN.search(question)
    if un:
        hits = by_un(rows, un.group(1))
        return hits, f"UN number {un.group(1)}"

    return by_name(rows, question)


def describe(row, with_un=True):
    """
    One row of Schedule 1, written the way the regulation means it.

    with_un=False leaves the UN number out. The threshold comes from the
    regulation; the UN number comes from an ECCC web page. Putting both under
    one citation would label an ECCC value as law, so the caller can ask for
    them separately.
    """
    concentration = row["min_concentration_pct"]
    if concentration.upper() in ("N/A", ""):
        concentration_text = "no minimum concentration applies"
    else:
        concentration_text = f"minimum concentration {concentration}% (mass/mass)"

    lines = [
        f"{row['substance_name']} - Schedule 1, Part {row['part']}, item {row['item']}",
        f"    CAS number            : {row['cas_number']}",
        f"    {concentration_text}",
        f"    minimum quantity      : {row['min_quantity_tonnes']} tonnes",
        f"    hazard category       : {row['hazard_category']} "
        f"({row['hazard_category_meaning']})",
    ]
    if with_un and row["un_number"]:
        lines.append(f"    UN number             : {row['un_number']} "
                     f"(from the ECCC list, not the regulation)")
    return "\n".join(lines)


def describe_un(row):
    """
    The UN number on its own, with its real source named.

    The regulation never states a UN number. Schedule 2, item 3(c) only
    requires one to be REPORTED if applicable. The value itself comes from
    ECCC's published list of hazardous substances.
    """
    return (f"{row['substance_name']} (CAS {row['cas_number']}) - "
            f"UN number {row['un_number']}.\n"
            f"Source: ECCC list of hazardous substances. The Environmental "
            f"Emergency Regulations do not assign UN numbers; Schedule 2, "
            f"item 3(c) requires a UN number to be reported if applicable.")


if __name__ == "__main__":
    rows = load()
    print(f"Schedule 1 loaded: {len(rows)} substances\n")

    TESTS = [
        "What is the minimum quantity threshold for anhydrous ammonia?",
        "Is 7664-41-7 listed in Schedule 1?",
        "We have a UN 1017 release. Is that regulated?",
        "We have a UN 1005 release. Is that regulated?",
        "How much chlorine triggers the Regulations?",
        "What is the threshold for mercury?",
        "Is sodium chloride listed?",
        "How often must we run a simulation exercise?",
    ]

    for question in TESTS:
        matches, how = lookup(rows, question)
        print("=" * 88)
        print(f"Q: {question}")
        print(f"   matched on: {how}   ->   {len(matches)} row(s)")
        print("=" * 88)
        if not matches:
            print("  NOT IN SCHEDULE 1.")
            print("  This is an answer, not a failure. The correct response is that")
            print("  the substance is not listed - never a guess at a number.\n")
            continue
        for row in matches[:4]:
            print(describe(row))
            print()

    print("=" * 88)
    print("WHY THIS IS NOT SEARCH")
    print("=" * 88)
    print("  Every number above was read from a cell. Nothing was ranked, scored")
    print("  or approximated, so there is no 'close enough' answer and no way for")
    print("  4.50 to come back as 4.5 tonnes of something else.")
    print()
    print("  Look at the two ammonia questions together. Asking by NAME for")
    print("  'anhydrous ammonia' returns one row: 4.50 tonnes. Asking by CAS")
    print("  number returns TWO rows - 4.50 and 9.10 tonnes - because one CAS")
    print("  number covers both the anhydrous substance and the solution.")
    print("  An embedding rates those two names as nearly identical. This table")
    print("  keeps them apart. That is the hybrid architecture in one example.")
    print()
    print("  And note UN 1005. In transport rules that is anhydrous ammonia, but")
    print("  the ECCC list gives ammonia only 2073 and 2672, so our lookup finds")
    print("  nothing. The gap is in the published source, not in the code - which")
    print("  is exactly why UN numbers are labelled as coming from ECCC and not")
    print("  from the regulation.")