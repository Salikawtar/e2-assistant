"""
Step 3 of Stage 4 — make every claim carry a citation, then VERIFY it in code.

Step 2 produced an answer about simulation exercises that was entirely correct
and contained no citation at all. Nobody could check it without re-reading the
regulation. That is not an answer a regulator can use.

The design choice that makes citations checkable:

    THE MODEL CITES BY NUMBER, NOT BY NAME.

Evidence is handed over as [1], [2], [3]... and every claim must end with one
of those markers. A model can invent "section 12(3)" and it will look
convincing. It cannot invent [9] when only six pieces of evidence exist - and
that is a one-line check.

Afterwards the code expands [3] back into the real citation for display, so
the model never writes a citation, it only points at one we already hold.

Run:  python src/generate_cited.py
"""

import os
import re

from dotenv import load_dotenv
from google import genai

from lookup_schedule1 import (load as load_schedule1, lookup, describe,
                              describe_un)
from search_vectors import load_index
from search_keyword import build_index as build_bm25
from search_split import pool_indices
from router import route, retrieve
from generate_naive import flash_models, ModelPool

MARKER = re.compile(r"\[(\d+)\]")

# A refusal is a statement ABOUT the evidence, not a claim drawn from it, so
# it cannot carry a citation. Counting it as an uncited claim was a bug in the
# check, not a fault in the answer.
# The word list has been wrong twice, and both times it turned a correct
# refusal into a counted failure:
#
#   O4  "The provided regulations DO not specify..."   -> only "does" matched
#   U2  "The provided evidence does not MENTION..."    -> "mention" was absent
#
# A refusal is a statement ABOUT the evidence, so it carries no citation by
# design. Miss one and it is scored as an uncited claim, which is a defect in
# the checker rather than in the answer.
REFUSAL = re.compile(
    r"\b(?:do|does|did)\s+not\s+"
    r"(?:state|contain|specify|address|answer|mention|provide|include|"
    r"indicate|cover|set\s+out|detail)\b"
    r"|\bno\s+information\b"
    r"|\bnot\s+(?:covered|addressed|stated|specified|mentioned)\s+"
    r"(?:in|by)\s+the\b",
    re.IGNORECASE)

# Things that LOOK like a provision reference. Used only to notice when the
# model writes one in prose instead of using a marker.
PROVISION = re.compile(r"\b(?:section|subsection|paragraph|Schedule)\s+\d+[\w().]*",
                       re.IGNORECASE)

QUESTIONS = [
    "How often must a facility conduct a simulation exercise?",
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
    "What penalty applies if a facility fails to submit a notice on time?",
]


# ---------------------------------------------------------------------------
# The prompt, as a contract
# ---------------------------------------------------------------------------

CITED_PROMPT = """You answer questions about Canadian federal environmental
regulation for a regulatory analyst. You are given numbered evidence. Follow
these rules exactly.

1. Use ONLY the numbered evidence below. If something is not in the evidence,
   you do not know it, even if you believe it to be true.
2. End every sentence that makes a claim with the number of the evidence it
   comes from, like this: [2]. If a sentence rests on two pieces, cite both:
   [2][5].
3. Never write a section number, schedule number or document name as a
   citation. Use the bracket numbers only.
4. State conditions, thresholds and exceptions in full. A requirement given
   without its condition is a wrong answer, not a short one.
5. If the evidence does not answer part of the question, say plainly which
   part it does not answer.
6. Be brief. No preamble, no summary of what you are about to do.

Evidence:
{context}

Question: {question}

Answer:"""


def format_evidence(chunks, rows):
    """
    Number every piece of evidence and return the numbering alongside it, so
    the checks afterwards know exactly what [3] was supposed to mean.
    """
    sources, parts = [], []

    for chunk in chunks:
        sources.append({
            "citation": chunk["citation"],
            "source_type": chunk["source_type"],
            "as_of": chunk.get("as_of", "unknown"),
        })
        parts.append(f"[{len(sources)}] ({chunk['source_type']}) "
                     f"{chunk['citation']}\n{chunk['text']}")

    for row in rows:
        # The threshold is law. The UN number is not. Splitting them into two
        # numbered sources keeps the citation honest: an answer that gives a
        # UN number can only cite the ECCC reference for it.
        sources.append({
            "citation": f"Schedule 1, Part {row['part']}, item {row['item']} "
                        f"({row['substance_name']})",
            "source_type": "regulation",
            "as_of": "2021-10-20",
        })
        parts.append(f"[{len(sources)}] (regulation) Schedule 1\n"
                     + describe(row, with_un=False))

        if row["un_number"]:
            sources.append({
                "citation": f"ECCC list of hazardous substances - UN number "
                            f"for {row['substance_name']}",
                "source_type": "reference",
                "as_of": "2016-01-14",
            })
            parts.append(f"[{len(sources)}] (reference) ECCC list of hazardous "
                         f"substances\n" + describe_un(row))

    return "\n\n".join(parts), sources


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def claim_lines(answer):
    """Lines that assert something, as opposed to blanks and bare headings."""
    lines = []
    for raw in answer.splitlines():
        line = raw.strip().lstrip("*-• ").strip()
        if len(line) < 25:
            continue
        lines.append(line)
    return lines


def check(answer, sources):
    """
    Three questions about the answer, all answerable without a model.

    Nothing here asks whether the answer is CORRECT. These checks ask whether
    it is CHECKABLE, which has to come first.
    """
    used = [int(n) for n in MARKER.findall(answer)]
    valid = [n for n in used if 1 <= n <= len(sources)]
    invented = sorted({n for n in used if not 1 <= n <= len(sources)})

    lines = claim_lines(answer)
    refusals = [line for line in lines if REFUSAL.search(line)]
    claims = [line for line in lines if line not in refusals]
    uncited = [line for line in claims if not MARKER.search(line)]

    prose_refs = []
    for line in lines:
        without_markers = MARKER.sub("", line)
        prose_refs.extend(PROVISION.findall(without_markers))

    return {
        "sources": len(sources),
        "refusals": len(refusals),
        "markers_used": len(used),
        "invented": invented,
        "claim_lines": len(claims),
        "uncited": uncited,
        "cited_share": ((len(claims) - len(uncited)) / len(claims)
                        if claims else 1.0),
        "unused_evidence": sorted(set(range(1, len(sources) + 1)) - set(valid)),
        "prose_refs": prose_refs,
    }


def show_sources(answer, sources):
    """Expand the markers the answer actually used into real citations."""
    used = sorted({int(n) for n in MARKER.findall(answer)
                   if 1 <= int(n) <= len(sources)})
    if not used:
        print("\n  Sources: none cited.")
        return
    print("\n  Sources")
    for n in used:
        s = sources[n - 1]
        print(f"    [{n}] ({s['source_type']}) {s['citation']}  — as of {s['as_of']}")


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

    totals = {"lines": 0, "cited": 0, "invented": 0}

    for question in QUESTIONS:
        decision, _ = route(question, schedule1)

        rows, evidence = [], []
        if decision in ("LOOKUP", "BOTH"):
            rows, _ = lookup(schedule1, question)
        if decision in ("RETRIEVE", "BOTH"):
            evidence = retrieve(chunks, matrix, bm25, client, index_meta,
                                law_pool, guidance_pool, question)

        context, sources = format_evidence(evidence, rows)
        answer = pool.generate(CITED_PROMPT.format(context=context,
                                                   question=question))
        report = check(answer, sources)

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print(f"   route: {decision}   evidence: {len(sources)}   "
              f"model: {pool.current}")
        print("=" * 96)
        print(answer)
        show_sources(answer, sources)

        if pool.truncated:
            print("\n  *** ANSWER WAS CUT OFF - the token budget ran out. ***")
            print("      Raise MAX_OUTPUT_TOKENS in generate_naive.py.")

        print("\n  CHECKS")
        print(f"    claims carrying a citation : {report['claim_lines'] - len(report['uncited'])}"
              f"/{report['claim_lines']}  ({report['cited_share']:.0%})")
        print(f"    refusal statements         : {report['refusals']}  "
              f"(these carry no citation by design)")
        print(f"    invented evidence numbers  : "
              f"{report['invented'] if report['invented'] else 'none'}")
        print(f"    evidence never used        : {report['unused_evidence']}")
        if report["uncited"]:
            print("    UNCITED CLAIMS:")
            for line in report["uncited"][:4]:
                print(f"      - {line[:84]}")
        if report["prose_refs"]:
            print(f"    provision names written in prose: "
                  f"{sorted(set(report['prose_refs']))[:6]}")

        totals["lines"] += report["claim_lines"]
        totals["cited"] += report["claim_lines"] - len(report["uncited"])
        totals["invented"] += len(report["invented"])

    print("\n" + "=" * 96)
    print("ACROSS ALL FOUR QUESTIONS")
    print("=" * 96)
    share = totals["cited"] / totals["lines"] if totals["lines"] else 0
    print(f"  claims carrying a citation : {totals['cited']}/{totals['lines']}  "
          f"({share:.0%})")
    print(f"  invented evidence numbers  : {totals['invented']}")

    print("\n" + "=" * 96)
    print("WHAT THESE CHECKS DO AND DO NOT TELL YOU")
    print("=" * 96)
    print("  They tell you the answer is CHECKABLE: every claim points at a")
    print("  piece of evidence, and every pointer is real.")
    print()
    print("  They do NOT tell you the answer is CORRECT. A sentence can carry")
    print("  a perfectly valid [3] and still say something [3] does not support.")
    print("  Step 2 showed exactly that: an Ontario compliance approval was")
    print("  equated with a plan prepared for another government, wrapped in two")
    print("  genuine citations. No code can catch that. Stage 5 measures it with")
    print("  a judge, and the judge itself has to be checked against you.")