"""
Step 4 of Stage 4 — declare abstention, and measure it properly.

Step 3 produced good abstention behaviour by accident of wording: the Ontario
answer began "The provided evidence does not state...". That is the right
behaviour, but it is buried in prose, and prose cannot be counted.

So the answer now declares itself:

    STATUS: ANSWERED | PARTIAL | REFUSED
    NOT COVERED: <the part the evidence does not reach>

Every question below carries the status we EXPECT, written down before the run.
That is what makes both failures countable:

    under-abstention : answered when it should have refused  (dangerous)
    over-abstention  : refused when it could have answered   (useless)

A system that refuses everything scores perfectly on the first and is worth
nothing. Both have to be measured or neither means anything.

This file also repairs two flaws in Step 3's checks. They are described at
`split_sentences` and `prose_reference_check`.

Run:  python src/generate_abstain.py
"""

import os
import re

from dotenv import load_dotenv
from google import genai

from lookup_schedule1 import load as load_schedule1, lookup, describe
from search_vectors import load_index
from search_keyword import build_index as build_bm25
from search_split import pool_indices
from router import route, retrieve
from generate_naive import flash_models, ModelPool
from generate_cited import MARKER, PROVISION, REFUSAL, format_evidence

STATUS_LINE = re.compile(r"^\s*STATUS:\s*(ANSWERED|PARTIAL|REFUSED)\s*$",
                         re.IGNORECASE | re.MULTILINE)
NOT_COVERED = re.compile(r"^\s*NOT COVERED:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

# question, expected status, why we expect it
QUESTIONS = [
    ("How often must a facility conduct a simulation exercise?",
     "ANSWERED", "section 7 states both the annual and five-year duties"),
    ("What is the minimum quantity threshold for anhydrous ammonia?",
     "ANSWERED", "Schedule 1, Part 1, item 163 gives quantity and concentration"),
    ("What must an environmental emergency plan contain?",
     "ANSWERED", "section 4(2) lists the required contents"),
    ("Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
     "PARTIAL", "the law covers plans prepared for another government, but never "
                "names Ontario approvals"),
    ("What penalty applies if a facility fails to submit a notice on time?",
     "REFUSED", "penalties are in CEPA Part 10, which is not in the corpus"),
]


ABSTAIN_PROMPT = """You answer questions about Canadian federal environmental
regulation for a regulatory analyst. You are given numbered evidence.

Begin your reply with exactly one line:

STATUS: ANSWERED      the evidence answers the whole question
STATUS: PARTIAL       the evidence answers part of it
STATUS: REFUSED       the evidence does not answer any of it

If the status is PARTIAL or REFUSED, the next line must be:

NOT COVERED: <the specific part the evidence does not reach>

Then write the answer, following these rules exactly.

1. Use ONLY the numbered evidence. If something is not in the evidence, you do
   not know it, even if you believe it to be true.
2. End every sentence that makes a claim with the number of the evidence it
   rests on, like this: [2]. Two pieces, two markers: [2][5].
3. Never write a section number as YOUR citation. Use bracket numbers. You may
   quote a cross-reference that appears inside the evidence text.
4. State conditions, thresholds and exceptions in full. A requirement given
   without its condition is a wrong answer, not a short one.
5. Do not refuse a question the evidence answers. Refusing what you could have
   answered is as much a failure as answering what you could not.
6. Be brief. No preamble.

Evidence:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Repaired check 1 — count sentences, not lines
# ---------------------------------------------------------------------------

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(*•-])")


def split_sentences(answer):
    """
    Step 3 split the answer on NEWLINES. One question came back as three
    sentences in a single paragraph, was counted as one claim, and scored
    100% cited - when only the last sentence carried a marker it would have
    scored 100% too.

    A metric that cannot detect a partial failure is worse than no metric,
    because it will be believed. So claims are counted per sentence.
    """
    body = STATUS_LINE.sub("", answer)
    body = NOT_COVERED.sub("", body)

    sentences = []
    for line in body.splitlines():
        line = line.strip().lstrip("*-• ").strip()
        if not line:
            continue
        for piece in SENTENCE_END.split(line):
            piece = piece.strip()
            if len(piece) >= 25:
                sentences.append(piece)
    return sentences


# ---------------------------------------------------------------------------
# Repaired check 2 — a quotation is not an invention
# ---------------------------------------------------------------------------

def prose_reference_check(answer, context):
    """
    Step 3 flagged "paragraph 4(2)(d)" as the model writing its own citation.
    It was not. Section 7(1)(a) genuinely refers to paragraph 4(2)(d), and the
    model was quoting the evidence.

    The test that separates the two: does that reference appear in the
    evidence text we supplied? If yes it is a quotation. If no, the model
    produced a provision number from nowhere, which is the serious case.
    """
    haystack = re.sub(r"\s+", " ", context).lower()
    quoted, invented = [], []
    for match in set(PROVISION.findall(MARKER.sub("", answer))):
        cleaned = re.sub(r"\s+", " ", match).strip(" .,;").lower()
        (quoted if cleaned in haystack else invented).append(match.strip(" .,;"))
    return sorted(quoted), sorted(invented)


# ---------------------------------------------------------------------------

def read_status(answer):
    match = STATUS_LINE.search(answer)
    status = match.group(1).upper() if match else "MISSING"
    not_covered = NOT_COVERED.search(answer)
    return status, (not_covered.group(1).strip() if not_covered else "")


def check(answer, sources, context):
    used = [int(n) for n in MARKER.findall(answer)]
    invented_markers = sorted({n for n in used if not 1 <= n <= len(sources)})
    valid = [n for n in used if 1 <= n <= len(sources)]

    sentences = split_sentences(answer)
    refusals = [s for s in sentences if REFUSAL.search(s)]
    claims = [s for s in sentences if s not in refusals]
    uncited = [s for s in claims if not MARKER.search(s)]

    quoted, invented_refs = prose_reference_check(answer, context)

    return {
        "claims": len(claims),
        "uncited": uncited,
        "cited_share": (len(claims) - len(uncited)) / len(claims) if claims else 1.0,
        "refusals": len(refusals),
        "invented_markers": invented_markers,
        "quoted_refs": quoted,
        "invented_refs": invented_refs,
        "unused_evidence": sorted(set(range(1, len(sources) + 1)) - set(valid)),
    }


def status_is_consistent(status, report, not_covered):
    """
    Does the declared status match what the answer actually does?

    A model can write STATUS: ANSWERED and then say it cannot answer. The
    status is only useful if it is checked against the body.
    """
    problems = []
    if status == "MISSING":
        problems.append("no STATUS line")
    if status in ("PARTIAL", "REFUSED") and not not_covered:
        problems.append(f"{status} with no NOT COVERED line")
    if status == "REFUSED" and report["claims"] > 0:
        problems.append(f"REFUSED but made {report['claims']} claim(s)")
    if status == "ANSWERED" and report["refusals"] > 0:
        problems.append("ANSWERED but the text refuses part of the question")
    return problems


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    client = genai.Client(api_key=key)
    pool = ModelPool(client, flash_models(client))

    schedule1 = load_schedule1()
    chunks, matrix, index_meta = load_index()
    bm25, _ = build_bm25(chunks)
    law_pool = pool_indices(chunks, law=True)
    guidance_pool = pool_indices(chunks, law=False)

    results = []

    for question, expected, why in QUESTIONS:
        decision, _ = route(question, schedule1)

        rows, evidence = [], []
        if decision in ("LOOKUP", "BOTH"):
            rows, _ = lookup(schedule1, question)
        if decision in ("RETRIEVE", "BOTH"):
            evidence = retrieve(chunks, matrix, bm25, client, index_meta,
                                law_pool, guidance_pool, question)

        context, sources = format_evidence(evidence, rows)
        answer = pool.generate(ABSTAIN_PROMPT.format(context=context,
                                                     question=question))
        status, not_covered = read_status(answer)
        report = check(answer, sources, context)
        problems = status_is_consistent(status, report, not_covered)

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print(f"   expected: {expected:<9} got: {status:<9} "
              f"{'MATCH' if status == expected else 'MISMATCH'}   "
              f"({decision}, {len(sources)} sources, {pool.current})")
        print("=" * 96)
        if pool.truncated:
            print("  *** ANSWER WAS CUT OFF - raise MAX_OUTPUT_TOKENS ***\n")
        print(answer)

        print("\n  CHECKS")
        print(f"    claims carrying a citation : "
              f"{report['claims'] - len(report['uncited'])}/{report['claims']}  "
              f"({report['cited_share']:.0%})")
        print(f"    invented evidence numbers  : "
              f"{report['invented_markers'] or 'none'}")
        print(f"    provision refs quoted from evidence : "
              f"{report['quoted_refs'] or 'none'}")
        print(f"    provision refs from NOWHERE         : "
              f"{report['invented_refs'] or 'none'}")
        print(f"    evidence never used        : {report['unused_evidence']}")
        if report["uncited"]:
            print("    UNCITED CLAIMS:")
            for line in report["uncited"][:3]:
                print(f"      - {line[:82]}")
        if problems:
            print(f"    STATUS INCONSISTENT: {'; '.join(problems)}")

        results.append({
            "question": question, "expected": expected, "got": status,
            "why": why, "report": report, "problems": problems,
        })

    # -----------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("ABSTENTION SCORECARD")
    print("=" * 96)
    print(f"  {'expected':<10} {'got':<10} question")
    print("  " + "-" * 92)
    for r in results:
        mark = " " if r["expected"] == r["got"] else "*"
        print(f" {mark}{r['expected']:<10} {r['got']:<10} {r['question'][:68]}")

    under = [r for r in results
             if r["expected"] in ("REFUSED", "PARTIAL") and r["got"] == "ANSWERED"]
    over = [r for r in results
            if r["expected"] == "ANSWERED" and r["got"] in ("REFUSED", "PARTIAL")]
    matched = [r for r in results if r["expected"] == r["got"]]

    print(f"\n  matched expectation : {len(matched)}/{len(results)}")
    print(f"  UNDER-abstention    : {len(under)}   "
          f"(answered when it should not have - the dangerous failure)")
    print(f"  OVER-abstention     : {len(over)}   "
          f"(refused when it could have answered - the useless failure)")

    total_claims = sum(r["report"]["claims"] for r in results)
    total_cited = sum(r["report"]["claims"] - len(r["report"]["uncited"])
                      for r in results)
    print(f"\n  claims carrying a citation : {total_cited}/{total_claims}  "
          f"({total_cited / total_claims if total_claims else 1:.0%})")
    print(f"  invented evidence numbers  : "
          f"{sum(len(r['report']['invented_markers']) for r in results)}")
    print(f"  provision refs from nowhere: "
          f"{sum(len(r['report']['invented_refs']) for r in results)}")
    print(f"  status inconsistencies     : "
          f"{sum(len(r['problems']) for r in results)}")

    print("\n" + "=" * 96)
    print("WHY BOTH FAILURES HAVE TO BE COUNTED")
    print("=" * 96)
    print("  An assistant that refuses every question has zero under-abstention.")
    print("  It is also worthless. Reporting only the safe number would make")
    print("  that system look perfect, which is why your specification asks for")
    print("  both - and why the expected status for each question is written")
    print("  down BEFORE the run, not decided afterwards by looking at what")
    print("  came out.")
