"""
Stage 4, finished — the answering pipeline as one importable function.

Everything before this was a script that printed things. This file exposes

    answer_question(question, resources) -> dict

so Stage 5 can evaluate it and Stage 8 can put a web page in front of it. A
stage that ends in a script has to be rewritten by the next stage. A stage
that ends in a function does not.

Four defects in the CHECKS are repaired here. Three of the five problems found
in the last run were in the measuring code, not in the answers, which is the
real lesson of Stage 4: an instrument that is never checked will eventually
report a success it invented.

  1. over-abstention was counted only against ANSWERED, so PARTIAL -> REFUSED
     scored zero. See ABSTENTION_ORDER.
  2. a normalised cross-reference looked fabricated. See quoted_or_invented.
  3. the guidance-only flag fired on correctly-labelled guidance. It now only
     fires on OBLIGATION language. See obligation_on_guidance.
  4. a markdown heading was counted as an uncited claim. See claim_sentences.

Run:  python src/answer.py
"""

import json
import os
import re
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from lookup_schedule1 import load as load_schedule1, lookup
from search_vectors import load_index
from search_keyword import build_index as build_bm25
from search_split import pool_indices
from router import route, retrieve
from generate_naive import flash_models, ModelPool
from generate_cited import MARKER, PROVISION, REFUSAL, format_evidence
from generate_abstain import (QUESTIONS, split_sentences, read_status,
                              STATUS_LINE)

OUT_PATH = Path("eval/stage4_answers.json")

# Stage 6, step 0 - PIN THE MODEL.
#
# The Stage 5 baseline was answered by two models: 3.5-flash took 14 questions
# and 3.6-flash took 8, because the newest model was busy and the pool fell
# back. That was the right behaviour for getting answers and the wrong
# behaviour for measuring anything: a score that moves next run cannot be told
# apart from a different model having answered.
#
# So one model is fixed here, and there is deliberately nowhere to fall back
# to. A busy model is now waited out rather than swapped, which is slower and
# is the price of a number that means something.
#
# Set E2_MODEL in .env to change it, or E2_MODEL=auto to restore the old
# fall-back behaviour.
DEFAULT_MODEL = "models/gemini-3.5-flash"

# More attempts than the 4 used when there was somewhere to fall back to, but
# not many more. The backoff doubles, so 8 attempts means waiting 128 seconds
# on the last one and over four minutes in total on a question that is going
# to fail anyway. Six caps the wait at 64 seconds. When the free tier is
# refusing, giving up sooner and running the file again is faster than waiting,
# because every answer already collected is kept.
PINNED_RETRIES = 6


def pinned_model():
    """The model name this run is fixed to, or "" when falling back freely."""
    name = os.getenv("E2_MODEL", DEFAULT_MODEL).strip()
    if name.lower() in ("", "auto"):
        return ""
    return name if name.startswith("models/") else "models/" + name


def build_pool(client):
    """One model, or the old newest-first pool if E2_MODEL is set to auto."""
    name = pinned_model()
    if not name:
        return ModelPool(client, flash_models(client))
    return ModelPool(client, [name], retries=PINNED_RETRIES)


# Raised three times. 3,000 truncated section 4(2); 6,000 went on the model
# deliberating out loud; 8,000 was not enough once rules 5 and 6 pushed it into
# quoting a 4,756-character provision verbatim. Rule 5 now allows a compact
# restatement, which matters more than the budget.
MAX_TOKENS = 12000

# Sent only if the first reply came back without a STATUS line or cut off.
RETRY_NOTE = """

Your previous reply did not begin with a STATUS line, or was incomplete.
Reply again. Begin with STATUS:. Write only the final answer."""
LAW_TYPES = {"regulation", "act"}

# What kind of authority each source type carries. "reference" is neither law
# nor ECCC guidance: it is a published data list, and a UN number taken from it
# must not be presented as either.
TIER_OF_TYPE = {"regulation": "law", "act": "law",
                "guidance": "guidance", "reference": "reference"}
TIER_ORDER = ["law", "guidance", "reference"]

# How restrictive each status is. Refusing more than expected is
# over-abstention; refusing less is under-abstention. Comparing the two needs
# an order, which the last version did not have.
ABSTENTION_ORDER = {"REFUSED": 0, "PARTIAL": 1, "ANSWERED": 2}

# Language that asserts a legal duty, as opposed to describing a practice.
OBLIGATION = re.compile(
    r"\b(must|shall|is required|are required|mandatory|obliged|has to|have to)\b",
    re.IGNORECASE)


PROMPT = """You answer questions about Canadian federal environmental
regulation for a regulatory analyst. You are given numbered evidence, each
marked (regulation), (act) or (guidance).

Begin with exactly one line:

STATUS: ANSWERED      the evidence answers the whole question
STATUS: PARTIAL       the evidence answers part of it
STATUS: REFUSED       the evidence does not address the question

If PARTIAL or REFUSED, the next line must be:

NOT COVERED: <the specific part the evidence does not reach>

Then the answer. Seven rules.

1. Use only the numbered evidence. If it is not there, you do not know it.
2. End every claim with the evidence it rests on: [2]. Two pieces: [2][5].
3. Answer only what was asked. Leave out material that does not bear on the
   question, however relevant it looks. A long answer to a different question
   is a wrong answer.
4. Law first, then guidance separately. Cite guidance only for what it ADDS;
   never restate a duty the law already states.
5. Keep the law's exact words for numbers, deadlines, conditions and
   exceptions. A requirement stated without its condition is a WRONG answer,
   not a short one: a threshold quantity must be given with the concentration
   it applies at, a deadline with the event it runs from. Restate everything
   else compactly. If the question asks what a provision requires and that
   provision is a list, give every item in it, one short line each, with its
   letter.
6. Before choosing a status, name to yourself the ONE provision in the
   evidence that answers the question.
   ANSWERED only if you can name it and it settles the question.
   REFUSED if no provision in the evidence addresses the subject of the
   question, no matter how much related law is present. The presence of law
   is not permission to answer.
   PARTIAL if a provision gives the general rule but cannot settle the
   specific case you were asked about, or if the evidence covers a broader
   category that includes the subject but not the subject itself. Say in
   NOT COVERED exactly what it cannot settle.
7. Write only the final answer. No reasoning, no headings, no preamble.

Evidence:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def claim_sentences(answer):
    """
    Sentences that assert something.

    Excluded: lead-ins ending in a colon, and markdown headings. A heading has
    no citation, no full stop and few words - "Guidance Explanations and
    Recommendations**" was counted as an uncited claim in the last run.
    """
    kept = []
    for sentence in split_sentences(answer):
        bare = sentence.strip("*# ").strip()
        if not bare or bare.endswith(":"):
            continue
        looks_like_heading = (not MARKER.search(sentence)
                              and not bare.endswith((".", "!", "?"))
                              and len(bare.split()) <= 8)
        if looks_like_heading:
            continue
        kept.append(sentence)
    return kept


def quoted_or_invented(answer, context):
    """
    Did the model quote a cross-reference, or produce a provision from nowhere?

    Section 4(2) writes "paragraph (f)". The model expands that to
    "paragraph 4(2)(f)", which is clearer for a reader and was flagged as
    fabricated by a literal string match. So each reference is also tried in
    its short form before being called invented.
    """
    haystack = re.sub(r"\s+", " ", context).lower()
    quoted, invented = [], []

    for match in set(PROVISION.findall(MARKER.sub("", answer))):
        reference = re.sub(r"\s+", " ", match).strip(" .,;")
        variants = {reference.lower()}

        # "paragraph 4(2)(f)" -> "paragraph (f)"
        short = re.sub(r"\s+\d+(?:\.\d+)?(?:\(\d+\))+(\([a-z]+\))", r" \1", reference)
        variants.add(short.lower())
        # "subsection 4(1)" -> "4(1)"
        variants.add(re.sub(r"^\w+\s+", "", reference).lower())

        (quoted if any(v in haystack for v in variants) else invented).append(reference)

    return sorted(quoted), sorted(invented)


def obligation_on_guidance(rows, law_available):
    """
    The section 9.3 failure mode, stated precisely.

    Not "a guidance-only claim" - the last run flagged six of those and every
    one was correctly labelled guidance doing its job. The failure is a
    statement of legal DUTY resting on guidance alone while the law was
    available to cite.
    """
    if not law_available:
        return []
    return [r["sentence"] for r in rows
            if r["tier"] == "guidance" and OBLIGATION.search(r["sentence"])]


CONTINUES = re.compile(r"[;,]\s*(?:or|and)?\s*$", re.IGNORECASE)


def carry_citations(rows):
    """
    A list item is part of the claim it belongs to.

    The model writes "(a) ...; or (b) ... [1]" and cites the group once, at
    the end. Counting each line separately reported four uncited claims that
    were all covered by the marker on the last line. So a line ending in a
    semicolon or comma inherits the tier of the line that follows it.
    """
    for i in range(len(rows) - 2, -1, -1):
        if rows[i]["tier"] == "uncited" and CONTINUES.search(rows[i]["sentence"]):
            rows[i]["tier"] = rows[i + 1]["tier"]
            rows[i]["carried"] = True
    return rows


def tier_of(markers, sources):
    """
    What kind of authority this sentence rests on, derived from source_type.

    A sentence citing both the regulation and ECCC's list comes back as
    "law+reference", which is the honest label: part of it is the law and
    part of it is not.
    """
    labels = {TIER_OF_TYPE.get(sources[n - 1]["source_type"], "other")
              for n in markers if 1 <= n <= len(sources)}
    if not labels:
        return "uncited"
    return "+".join(t for t in TIER_ORDER if t in labels) or "other"


def analyse(answer, sources, context, law_available, truncated):
    rows = []
    for sentence in claim_sentences(answer):
        markers = [int(n) for n in MARKER.findall(sentence)]
        tier = "refusal" if REFUSAL.search(sentence) else tier_of(markers, sources)
        rows.append({"tier": tier, "sentence": sentence})

    rows = carry_citations(rows)
    claims = [r for r in rows if r["tier"] != "refusal"]
    uncited = [r["sentence"] for r in claims if r["tier"] == "uncited"]

    used = [int(n) for n in MARKER.findall(answer)]
    invented_markers = sorted({n for n in used if not 1 <= n <= len(sources)})
    valid = [n for n in used if 1 <= n <= len(sources)]
    quoted, invented_refs = quoted_or_invented(answer, context)

    tiers = {}
    for r in claims:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

    status, not_covered = read_status(answer)

    problems = []
    if truncated:
        problems.append("answer was cut off, so it cannot be complete")
    if status == "MISSING":
        problems.append("no STATUS line")
    if status in ("PARTIAL", "REFUSED") and not not_covered:
        problems.append(f"{status} with no NOT COVERED line")
    if status == "REFUSED" and claims:
        problems.append(f"REFUSED but made {len(claims)} claim(s)")
    if status == "ANSWERED" and len(rows) - len(claims) > 0:
        problems.append("ANSWERED but the text refuses part of the question")

    return {
        "status": status,
        "not_covered": not_covered,
        "rows": rows,
        "claims": len(claims),
        "uncited": uncited,
        "cited_share": (len(claims) - len(uncited)) / len(claims) if claims else 1.0,
        "tiers": tiers,
        "obligation_on_guidance": obligation_on_guidance(rows, law_available),
        "invented_markers": invented_markers,
        "invented_refs": invented_refs,
        "quoted_refs": quoted,
        "unused_evidence": sorted(set(range(1, len(sources) + 1)) - set(valid)),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# The pipeline, as one function
# ---------------------------------------------------------------------------

def clean(answer):
    """
    Keep the answer, drop anything the model wrote before it.

    A reply once began with the model reasoning about which rule applied. The
    STATUS line is the agreed start of an answer, so everything before it is
    not part of one.
    """
    match = STATUS_LINE.search(answer)
    return answer[match.start():].strip() if match else answer.strip()


def load_resources(client):
    chunks, matrix, index_meta = load_index()
    return {
        "client": client,
        "pool": build_pool(client),
        "schedule1": load_schedule1(),
        "chunks": chunks,
        "matrix": matrix,
        "index_meta": index_meta,
        "bm25": build_bm25(chunks)[0],
        "law_pool": pool_indices(chunks, law=True),
        "guidance_pool": pool_indices(chunks, law=False),
    }


def answer_question(question, res):
    """Route, gather evidence, generate, and analyse. One call, one answer."""
    decision, reasons = route(question, res["schedule1"])

    rows, evidence = [], []
    if decision in ("LOOKUP", "BOTH"):
        rows, _ = lookup(res["schedule1"], question)
    if decision in ("RETRIEVE", "BOTH"):
        evidence = retrieve(res["chunks"], res["matrix"], res["bm25"],
                            res["client"], res["index_meta"],
                            res["law_pool"], res["guidance_pool"], question)

    context, sources = format_evidence(evidence, rows)
    law_available = any(s["source_type"] in LAW_TYPES for s in sources)

    pool = res["pool"]
    prompt = PROMPT.format(context=context, question=question)

    # One retry, and only for a reply that broke the format - not for one we
    # dislike. Retrying until an answer pleases us is how a baseline becomes
    # a wish.
    answer, attempts = "", 0
    for attempt in (1, 2):
        attempts = attempt
        answer = clean(pool.generate(prompt if attempt == 1 else prompt + RETRY_NOTE,
                                     max_output_tokens=MAX_TOKENS))
        if STATUS_LINE.search(answer) and not pool.truncated:
            break

    report = analyse(answer, sources, context, law_available, pool.truncated)
    return {
        "question": question,
        "route": decision,
        "route_reasons": reasons,
        "model": pool.current,
        "attempts": attempts,
        "sources": sources,
        "answer": answer,
        "report": report,
        # Everything analyse() needs, kept alongside the answer.
        #
        # The checks used to be frozen the moment the answer was written, so
        # fixing a defect in the CHECKER meant buying all 22 answers again.
        # Two word-list bugs have now been found in the refusal detector, and
        # a third will turn up. With these three fields saved, re_analyse()
        # can recompute every verdict from the stored answer for nothing.
        "context": context,
        "law_available": law_available,
        "truncated": bool(pool.truncated),
    }


def re_analyse(result):
    """
    Run the checks again over an answer that was saved earlier.

    Nothing is sent anywhere and no answer changes. Only the verdict about the
    answer is recomputed, which is what makes an instrument fix free. Older
    runs saved before this existed have no context, so their stored report is
    returned unchanged rather than silently recomputed against nothing.
    """
    if "context" not in result:
        return result["report"]
    return analyse(result["answer"], result["sources"], result["context"],
                   result.get("law_available", False),
                   result.get("truncated", False))


def abstention_gap(expected, got):
    """
    Negative = refused too much. Positive = answered too much. 0 = right.
    None = no usable STATUS, which is a broken answer, not an abstention
    judgement. It must be counted separately or it will be counted wrongly.
    """
    if got not in ABSTENTION_ORDER or expected not in ABSTENTION_ORDER:
        return None
    return ABSTENTION_ORDER[got] - ABSTENTION_ORDER[expected]


def verdict_of(gap):
    if gap is None:
        return "NO STATUS"
    if gap == 0:
        return "match"
    return "OVER-abstained" if gap < 0 else "UNDER-abstained"


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    print(f"  model: {pinned_model() or 'auto (newest available first)'}")
    res = load_resources(genai.Client(api_key=key))
    results = []

    for question, expected, why in QUESTIONS:
        result = answer_question(question, res)
        report = result["report"]
        gap = abstention_gap(expected, report["status"])
        result["expected"] = expected
        result["expected_because"] = why
        result["abstention_gap"] = gap
        results.append(result)

        verdict = verdict_of(gap)

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print(f"   expected: {expected:<9} got: {report['status']:<9} {verdict}"
              f"   ({result['route']}, {len(result['sources'])} sources)")
        print("=" * 96)
        if report["not_covered"]:
            print(f"  NOT COVERED: {report['not_covered']}\n")
        for row in report["rows"]:
            mark = "^" if row.get("carried") else " "
            print(f"  [{row['tier']:^14}]{mark}{row['sentence']}")

        t = report["tiers"]
        print("\n  tiers  " + "  ".join(f"{k} {v}" for k, v in sorted(t.items()))
              or "\n  tiers  none")
        print(f"  cited  {report['claims'] - len(report['uncited'])}/{report['claims']}"
              f"  ({report['cited_share']:.0%})")
        print(f"  invented markers {report['invented_markers'] or 'none'}   "
              f"provisions from nowhere {report['invented_refs'] or 'none'}")
        if report["quoted_refs"]:
            print(f"  cross-references quoted from the evidence: "
                  f"{report['quoted_refs'][:5]}")
        for sentence in report["obligation_on_guidance"]:
            print(f"  *** obligation stated on guidance alone: {sentence[:76]}")
        for problem in report["problems"]:
            print(f"  *** {problem}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("STAGE 4 BASELINE")
    print("=" * 96)
    print(f"  {'expected':<10} {'got':<10} {'verdict':<16} question")
    print("  " + "-" * 92)
    for r in results:
        print(f"  {r['expected']:<10} {r['report']['status']:<10} "
              f"{verdict_of(r['abstention_gap']):<16} {r['question'][:52]}")

    gaps = [r["abstention_gap"] for r in results]
    matched = sum(1 for g in gaps if g == 0)
    over = sum(1 for g in gaps if g is not None and g < 0)
    under = sum(1 for g in gaps if g is not None and g > 0)
    broken = sum(1 for g in gaps if g is None)
    claims = sum(r["report"]["claims"] for r in results)
    cited = sum(r["report"]["claims"] - len(r["report"]["uncited"]) for r in results)

    print(f"\n  status matched          : {matched}/{len(results)}")
    print(f"  UNDER-abstention        : {under}   (answered more than it should)")
    print(f"  OVER-abstention         : {over}   (refused more than it should)")
    print(f"  unusable (no STATUS)    : {broken}")
    print(f"  answers needing a retry : "
          f"{sum(1 for r in results if r.get('attempts', 1) > 1)}")
    print(f"  claims cited            : {cited}/{claims} "
          f"({cited / claims if claims else 1:.0%})")
    print(f"  invented citations      : "
          f"{sum(len(r['report']['invented_markers']) for r in results)}")
    print(f"  provisions from nowhere : "
          f"{sum(len(r['report']['invented_refs']) for r in results)}")
    print(f"  obligations on guidance : "
          f"{sum(len(r['report']['obligation_on_guidance']) for r in results)}")
    print(f"  status problems         : "
          f"{sum(len(r['report']['problems']) for r in results)}")

    all_tiers = {}
    for r in results:
        for k, v in r["report"]["tiers"].items():
            all_tiers[k] = all_tiers.get(k, 0) + v
    print("  claims by authority     : "
          + "  ".join(f"{k} {v}" for k, v in sorted(all_tiers.items())))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "run_date": date.today().isoformat(),
        "model": results[0]["model"] if results else None,
        "temperature": 0,
        "prompt_version": "stage4_final",
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  written: {OUT_PATH}")
