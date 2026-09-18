"""
Step 5 of Stage 4 — say which statements are LAW and which are guidance.

Step 4's answer read: "...at least once every 365 days [1][4][5]". [1] is
section 7 of the Regulations. [4] and [5] are ECCC guidance. Section 7 says
"each year"; "365 days" is ECCC's phrasing. Nothing in the answer told the
reader which words carry the force of law.

That is the risk named in section 10.1 of your specification, and this step
closes it.

The design decision that makes labelling trustworthy:

    THE CODE DERIVES THE LABEL. THE MODEL DOES NOT DECLARE IT.

Every piece of evidence already carries source_type from the manifest. When a
sentence cites [1][4], we know without asking that it rests on regulation AND
guidance. A model asked to label its own sentences would sometimes be wrong,
and the label would then be one more thing needing verification.

This also fixes three things carried from Step 4. See MAX_TOKENS below,
`claim_sentences`, and `status_is_consistent`.

Run:  python src/generate_labelled.py
"""

import json
import os
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
from generate_cited import MARKER, REFUSAL, format_evidence
from generate_abstain import (QUESTIONS, split_sentences, prose_reference_check,
                              read_status)

OUT_PATH = Path("eval/stage4_answers.json")

# Section 4(2) alone is 4,756 characters of required contents. At 3,000 tokens
# the answer was cut off after ten of its fifteen paragraphs - and still
# declared itself ANSWERED. Budget for the largest provision in the corpus.
MAX_TOKENS = 6000

LAW_TYPES = {"regulation", "act"}


LABELLED_PROMPT = """You answer questions about Canadian federal environmental
regulation for a regulatory analyst. You are given numbered evidence. Each
piece is marked (regulation), (act) or (guidance).

Begin your reply with exactly one line:

STATUS: ANSWERED      the evidence answers the whole question
STATUS: PARTIAL       the evidence answers part of it
STATUS: REFUSED       the evidence does not answer any of it

If PARTIAL or REFUSED, the next line must be:

NOT COVERED: <the specific part the evidence does not reach>

Then write the answer, following these rules exactly.

1. Use ONLY the numbered evidence. If it is not in the evidence, you do not
   know it, even if you believe it to be true.
2. End every sentence that makes a claim with the number of the evidence it
   rests on: [2]. Two pieces, two markers: [2][5].
3. State what the LAW requires first, citing (regulation) or (act) evidence.
   Then, separately, what the guidance explains or recommends.
4. Do not support a statement of legal obligation with guidance alone when
   regulation or act evidence is available for it.
5. Use the wording of the law for legal requirements. If the regulation says
   "each year" and guidance says "at least once every 365 days", write "each
   year" and cite the regulation.
6. State conditions, thresholds and exceptions in full. A requirement given
   without its condition is a wrong answer, not a short one.
7. Do not refuse a question the evidence answers.
8. Be brief. No preamble.

Evidence:
{context}

Question: {question}

Answer:"""


def claim_sentences(answer):
    """
    Sentences that assert something.

    Step 4 counted "An environmental emergency plan must include the
    following contents:" as an uncited claim. It is a lead-in to a list, not a
    claim, and it cannot carry a citation. Anything ending in a colon is a
    heading for what follows.
    """
    sentences = []
    for sentence in split_sentences(answer):
        if sentence.rstrip().endswith(":"):
            continue
        sentences.append(sentence)
    return sentences


def tier_of(markers, sources):
    """
    What kind of authority does this sentence rest on?

    Computed from source_type, which came from the manifest. Not asked for,
    not guessed.
    """
    types = {sources[n - 1]["source_type"] for n in markers
             if 1 <= n <= len(sources)}
    if not types:
        return "uncited"
    has_law = bool(types & LAW_TYPES)
    has_guidance = bool(types - LAW_TYPES)
    if has_law and has_guidance:
        return "law+guidance"
    return "law" if has_law else "guidance"


def annotate(answer, sources):
    """Rewrite the answer with a derived tier tag in front of every claim."""
    out, rows = [], []
    for sentence in claim_sentences(answer):
        markers = [int(n) for n in MARKER.findall(sentence)]
        tier = "refusal" if REFUSAL.search(sentence) else tier_of(markers, sources)
        rows.append({"tier": tier, "sentence": sentence})
        out.append(f"  [{tier:^12}] {sentence}")
    return "\n".join(out), rows


def check(answer, sources, context, law_available):
    rows = annotate(answer, sources)[1]
    claims = [r for r in rows if r["tier"] != "refusal"]
    uncited = [r for r in claims if r["tier"] == "uncited"]

    used = [int(n) for n in MARKER.findall(answer)]
    invented = sorted({n for n in used if not 1 <= n <= len(sources)})
    valid = [n for n in used if 1 <= n <= len(sources)]
    quoted, invented_refs = prose_reference_check(answer, context)

    tiers = {"law": 0, "guidance": 0, "law+guidance": 0, "uncited": 0}
    for r in claims:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

    # The failure mode from section 9.3: a legal statement resting only on
    # ECCC's explanation of the law, when the law itself was on the table.
    guidance_only = tiers["guidance"] if law_available else 0

    return {
        "claims": len(claims),
        "refusals": len(rows) - len(claims),
        "uncited": [r["sentence"] for r in uncited],
        "cited_share": (len(claims) - len(uncited)) / len(claims) if claims else 1.0,
        "tiers": tiers,
        "guidance_only_when_law_available": guidance_only,
        "invented_markers": invented,
        "invented_refs": invented_refs,
        "quoted_refs": quoted,
        "unused_evidence": sorted(set(range(1, len(sources) + 1)) - set(valid)),
    }


def status_is_consistent(status, report, not_covered, truncated):
    """
    Step 4 printed 'ANSWER WAS CUT OFF' and then accepted STATUS: ANSWERED.
    The code held the evidence of an incomplete answer and did not act on it.
    A truncated answer can never be ANSWERED.
    """
    problems = []
    if truncated:
        problems.append("answer was cut off, so it cannot be complete")
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
        law_available = any(s["source_type"] in LAW_TYPES for s in sources)

        answer = pool.generate(
            LABELLED_PROMPT.format(context=context, question=question),
            max_output_tokens=MAX_TOKENS)

        status, not_covered = read_status(answer)
        report = check(answer, sources, context, law_available)
        problems = status_is_consistent(status, report, not_covered, pool.truncated)
        tagged, _ = annotate(answer, sources)

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print(f"   expected: {expected:<9} got: {status:<9} "
              f"{'MATCH' if status == expected else 'MISMATCH'}   "
              f"({decision}, {len(sources)} sources, {pool.current})")
        print("=" * 96)
        if not_covered:
            print(f"  NOT COVERED: {not_covered}\n")
        print(tagged)

        t = report["tiers"]
        print(f"\n  TIERS   law {t['law']}   guidance {t['guidance']}   "
              f"both {t['law+guidance']}   uncited {t['uncited']}")
        print(f"  cited   {report['claims'] - len(report['uncited'])}/{report['claims']}"
              f"  ({report['cited_share']:.0%})")
        print(f"  invented markers {report['invented_markers'] or 'none'}   "
              f"invented provisions {report['invented_refs'] or 'none'}")
        if report["guidance_only_when_law_available"]:
            print(f"  *** {report['guidance_only_when_law_available']} claim(s) rest on "
                  f"guidance alone while law evidence was available - review these")
        if problems:
            print(f"  *** STATUS PROBLEM: {'; '.join(problems)}")

        results.append({
            "question": question, "expected": expected, "why": why,
            "route": decision, "model": pool.current, "status": status,
            "not_covered": not_covered, "answer": answer,
            "sources": sources, "report": report, "problems": problems,
        })

    # -----------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("STAGE 4 BASELINE")
    print("=" * 96)
    matched = sum(1 for r in results if r["status"] == r["expected"])
    under = sum(1 for r in results if r["expected"] in ("REFUSED", "PARTIAL")
                and r["status"] == "ANSWERED")
    over = sum(1 for r in results if r["expected"] == "ANSWERED"
               and r["status"] in ("REFUSED", "PARTIAL"))
    claims = sum(r["report"]["claims"] for r in results)
    cited = sum(r["report"]["claims"] - len(r["report"]["uncited"]) for r in results)
    law = sum(r["report"]["tiers"]["law"] for r in results)
    guidance = sum(r["report"]["tiers"]["guidance"] for r in results)
    both = sum(r["report"]["tiers"]["law+guidance"] for r in results)

    print(f"  status matched expectation : {matched}/{len(results)}")
    print(f"  under-abstention           : {under}")
    print(f"  over-abstention            : {over}")
    print(f"  claims carrying a citation : {cited}/{claims} "
          f"({cited / claims if claims else 1:.0%})")
    print(f"  invented citations         : "
          f"{sum(len(r['report']['invented_markers']) for r in results)}")
    print(f"  fabricated provisions      : "
          f"{sum(len(r['report']['invented_refs']) for r in results)}")
    print(f"  status problems            : "
          f"{sum(len(r['problems']) for r in results)}")
    print(f"\n  claims by authority        : law {law}   guidance {guidance}   "
          f"both {both}")
    print(f"  guidance-only where law was available : "
          f"{sum(r['report']['guidance_only_when_law_available'] for r in results)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "run_date": date.today().isoformat(),
        "model": results[0]["model"] if results else None,
        "temperature": 0,
        "prompt_version": "stage4_step5_labelled",
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  written: {OUT_PATH}")
    print("  This file is the baseline Stage 5 measures against. Do not tune")
    print("  anything before it exists - an improvement you cannot compare to")
    print("  a starting point is a story, not a result.")
