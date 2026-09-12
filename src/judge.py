"""
Ask whether each citation actually SUPPORTS its claim.

WHY

evaluate.py can prove two things about a sentence: that it carries a marker,
and that the marker points at evidence that exists. It cannot prove the third
and most important thing, that the evidence says what the sentence claims.

    "The plan must be reviewed every two years [1]"

carries a citation, cites a real source, contains no forbidden word, and can
still be false. Nothing countable catches it. It needs a reader.

HOW THE READER IS KEPT HONEST

  1. It sees ONE sentence and the evidence that sentence cites. It never sees
     the question, the rest of the answer, or the answer key. An answer cannot
     persuade a judge that never reads it.
  2. It must QUOTE the words that decided the verdict. The same discipline the
     answers are held to: point at something, do not just assert.
  3. It has been tested against errors planted on purpose. See
     judge_control.py: 8 of 8 planted errors caught, 6 of 6 clean claims left
     alone, kappa 1.00 on a 14 case control set.

WHAT CHANGED IN THIS VERSION

  - It judges whichever run evaluate.py is currently measuring, taken from
    RUN_LABEL, instead of the frozen Stage 5 run. The system has changed twice
    since then, so a faithfulness number computed over the old answers would
    describe a system that no longer exists.

  - It prints its findings instead of withholding them. The silence in the
    first version existed for one reason: a human was about to grade a blind
    sample, and a grader who has seen the answers is not a second opinion.
    That grading is done and the judge has been validated, so there is nothing
    left to protect.

Run:  python src/judge.py
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from generate_naive import flash_models, ModelPool
from generate_cited import MARKER
from lookup_schedule1 import load as load_schedule1, describe, describe_un
from evaluate import RUN_LABEL

CHUNKS_PATH = Path("data/chunks.jsonl")
RUN_PATH = Path(f"eval/{RUN_LABEL}_run.json")
JUDGE_PATH = Path(f"eval/{RUN_LABEL}_judge.json")

MAX_JUDGE_TOKENS = 2000

# Bumped whenever what the judge is SHOWN changes, so old verdicts are not
# reused against a different question.
#   v1          one sentence, alone
#   v2_stem     a list item is shown with the line that introduces it
JUDGE_VERSION = "v3_block"

# "(a)", "(ii)", "3)" - the start of a list item rather than a whole sentence.
LIST_START = re.compile(r"^\(?[a-z0-9]{1,4}\)")

VERDICTS = ("SUPPORTED", "PARTLY SUPPORTED", "NOT SUPPORTED")
VERDICT_LINE = re.compile(r"^\s*VERDICT:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
BECAUSE_LINE = re.compile(r"^\s*BECAUSE:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

USABLE_STATUS = {"ANSWERED", "PARTIAL", "REFUSED"}


JUDGE_PROMPT = """You are checking one sentence against the evidence it cites.

You are NOT judging whether the sentence is useful, well written, complete or
good advice. You are answering one question only:

    Does the evidence below state this?

Rules.

1. Use only the evidence given. If you know something from elsewhere that
   makes the sentence true, that is not evidence and it does not count.
2. Wording does not have to match. Meaning does. A correct paraphrase is
   SUPPORTED.
3. A number, deadline, threshold or condition that differs from the evidence,
   however slightly, is NOT SUPPORTED.
4. A sentence that states a requirement while leaving out a condition the
   evidence attaches to it is PARTLY SUPPORTED.
5. A sentence that adds anything the evidence does not contain is at best
   PARTLY SUPPORTED.

6. The passage to check may be several lines: a sentence that introduces a
   list, followed by the items of that list. Judge it as ONE claim. Several
   pieces of evidence may be given, and different lines of the passage may
   rest on different pieces. A statement is supported if ANY of the evidence
   supports it.

Reply with exactly two lines and nothing else.

VERDICT: SUPPORTED or PARTLY SUPPORTED or NOT SUPPORTED
BECAUSE: at most 25 words, quoting the words in the evidence that decided it.

Evidence:
{evidence}

Sentence to check:
{sentence}
"""


# ---------------------------------------------------------------------------
# Rebuilding the evidence text
# ---------------------------------------------------------------------------

def evidence_map():
    """
    Citation -> the text that citation stood for.

    The run file saved each source's CITATION but not its TEXT, so the text is
    rebuilt here from the same two files the system read at answer time.
    Rebuilding beats re-retrieving: retrieval could return something slightly
    different today, and then the judge would be reading evidence the answer
    never saw.
    """
    texts = defaultdict(list)

    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            texts[chunk["citation"]].append(chunk["text"])

    for row in load_schedule1():
        law_key = (f"Schedule 1, Part {row['part']}, item {row['item']} "
                   f"({row['substance_name']})")
        texts[law_key].append(describe(row, with_un=False))
        if row["un_number"]:
            ref_key = (f"ECCC list of hazardous substances - UN number "
                       f"for {row['substance_name']}")
            texts[ref_key].append(describe_un(row))

    return texts


def claims_of(result):
    """
    Every sentence in one answer that carries a citation.

    A list item ending in a semicolon has its marker on a later line, and
    carry_citations already decided it inherits that line's authority. Judging
    it with no evidence at all would be a rigged test, so it inherits the
    markers too.
    """
    rows = [r for r in result["report"]["rows"] if r["tier"] != "refusal"]
    claims, i = [], 0

    while i < len(rows):
        block = [rows[i]]
        i += 1
        # A lead-in swallows the list items that follow it. They are one claim.
        if not is_list_item(block[0]["sentence"]):
            while i < len(rows) and is_list_item(rows[i]["sentence"]):
                block.append(rows[i])
                i += 1

        text = "\n".join(r["sentence"].strip() for r in block)
        markers = sorted({int(n) for r in block
                          for n in MARKER.findall(r["sentence"])})
        if not markers:
            continue

        claims.append({
            "sentence": text,
            "stem": "",                       # kept for file compatibility
            "lines": len(block),
            "markers": markers,
            "carried": any(r.get("carried") for r in block),
            "tier": block[0]["tier"],
        })
    return claims


def is_list_item(sentence):
    """Does this line continue a list rather than start a new claim?"""
    return bool(LIST_START.match(sentence.strip()))


# ---------------------------------------------------------------------------
# WHY THE UNIT OF JUDGEMENT CHANGED TWICE
#
# v1  ONE LINE AT A TIME.                              faithfulness 56/72 (78%)
#     A list item is half a sentence, and the half it was missing was usually
#     the condition. S2's "(ii) the number of container systems..." was marked
#     down for omitting "If the notice concerns a cessation of operations",
#     which sat on the line above and the judge never saw.
#
# v2  THE ITEM PLUS THE LINE ABOVE IT.                 faithfulness 43/72 (60%)
#     This was expected to fix v1 and made it worse. The line above is not
#     context: it is its own claim with its own citation. Gluing them together
#     made the judge hold ONE source responsible for BOTH. C2's items were then
#     marked down because Schedule 5 does not mention paragraph 7(1)(a), which
#     came from the lead-in and rested on a different source entirely.
#
# v3  THE WHOLE BLOCK, AGAINST ALL THE SOURCES IT CITES.
#     The answers are written as a lead-in followed by a quoted list. That
#     block is ONE claim resting on SEVERAL sources, and judging it in pieces
#     cannot work in either direction. So the block is sent whole, with the
#     union of the evidence its lines cite.
#
# The count of claims changes with every version, so 78%, 60% and whatever v3
# reports are NOT comparable to each other. What is comparable is the list of
# substantive problems found. O1's annual duty stated without its exception
# has appeared in every version, which is what makes it real.
# ---------------------------------------------------------------------------


def evidence_for(markers, sources, texts):
    """
    The text behind the markers this sentence used, and what could not be found.

    A citation that resolves to nothing is reported, never skipped quietly. A
    truncated CAS number once removed a substance from Schedule 1 without a
    word, and the rule since then is that code which drops data counts what it
    dropped.
    """
    parts, missing = [], []
    for n in markers:
        if not 1 <= n <= len(sources):
            missing.append(f"[{n}] out of range")
            continue
        source = sources[n - 1]
        found = texts.get(source["citation"])
        if not found:
            missing.append(source["citation"])
            continue
        for text in found:
            parts.append(f"({source['source_type']}) {source['citation']}\n{text}")
    return "\n\n".join(parts), missing


def read_verdict(reply):
    """
    Pull the verdict out, wherever the model put it.

    Searched anywhere in the reply rather than required on line one, because a
    thinking model writes its deliberation first and the answer afterwards.
    That has already cost this project one whole question.
    """
    match = VERDICT_LINE.search(reply or "")
    if not match:
        return "UNREADABLE", ""
    raw = match.group(1).strip().strip("*.").upper()
    verdict = next((v for v in VERDICTS if raw.startswith(v)), None)
    if verdict is None:
        verdict = "PARTLY SUPPORTED" if raw.startswith("PARTLY") else "UNREADABLE"
    because = BECAUSE_LINE.search(reply or "")
    return verdict, (because.group(1).strip() if because else "")


# ---------------------------------------------------------------------------

def load_saved():
    if not JUDGE_PATH.exists():
        return {}
    saved = json.loads(JUDGE_PATH.read_text(encoding="utf-8"))
    if saved.get("judge_version") != JUDGE_VERSION:
        print("  the saved verdicts were produced by a different judge - "
              "starting fresh")
        return {}
    return {r["key"]: r for r in saved.get("verdicts", [])}


def save(verdicts):
    JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_PATH.write_text(json.dumps({
        "run_date": date.today().isoformat(),
        "judging": RUN_LABEL,
        "judge_version": JUDGE_VERSION,
        "verdicts": list(verdicts.values()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")
    if not RUN_PATH.exists():
        raise SystemExit(f"{RUN_PATH} not found. Run src/evaluate.py first.")

    run = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    texts = evidence_map()

    items, skipped_unusable = [], []
    for result in run["results"]:
        status = result["report"]["status"]
        if status not in USABLE_STATUS:
            skipped_unusable.append(result["id"])
            continue
        for n, claim in enumerate(claims_of(result)):
            evidence, missing = evidence_for(claim["markers"],
                                             result["sources"], texts)
            items.append({
                "key": f"{result['id']}#{n}",
                "id": result["id"],
                "sentence": claim["sentence"],
                "markers": claim["markers"],
                "carried": claim["carried"],
                "tier": claim["tier"],
                "stem": claim["stem"],
                "lines": claim["lines"],
                "citations": [result["sources"][m - 1]["citation"]
                              for m in claim["markers"]
                              if 1 <= m <= len(result["sources"])],
                "evidence": evidence,
                "missing_evidence": missing,
            })

    unresolved = [i for i in items if i["missing_evidence"]]
    print(f"  judging run              : {RUN_LABEL}")
    print(f"  claims to judge          : {len(items)}")
    print(f"  answers skipped, unusable: "
          f"{', '.join(skipped_unusable) if skipped_unusable else 'none'}")
    print(f"  citations that resolved  : {len(items) - len(unresolved)}/{len(items)}")
    for item in unresolved[:6]:
        print(f"      {item['key']}: {'; '.join(item['missing_evidence'])[:80]}")

    pool = ModelPool(genai.Client(api_key=key), flash_models(
        genai.Client(api_key=key)))
    done = load_saved()
    todo = [i for i in items if i["key"] not in done]
    if done:
        print(f"\n  resuming: {len(done)} already judged, {len(todo)} to go")
    print(f"\n  judging {len(todo)} claim(s), one request each.\n")

    for n, item in enumerate(todo, 1):
        if item["missing_evidence"] and not item["evidence"]:
            done[item["key"]] = {**item, "verdict": "EVIDENCE MISSING",
                                 "because": "; ".join(item["missing_evidence"]),
                                 "judge_model": None}
            save(done)
            continue
        lines = item.get("lines", 1)
        mark = f"+{lines - 1}" if lines > 1 else "  "
        first = item["sentence"].splitlines()[0]
        print(f"  {n:>3}/{len(todo)} {mark} {item['key']:<8} {first[:56]}")
        reply = pool.generate(
            JUDGE_PROMPT.format(evidence=item["evidence"],
                                sentence=item["sentence"]),
            max_output_tokens=MAX_JUDGE_TOKENS)
        verdict, because = read_verdict(reply)
        done[item["key"]] = {**item, "verdict": verdict, "because": because,
                             "judge_model": pool.current}
        save(done)

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DOES EACH CITATION SUPPORT ITS CLAIM?")
    print("=" * 100)
    counts = Counter(v["verdict"] for v in done.values())
    total = sum(counts.values())
    for label in list(VERDICTS) + [c for c in counts if c not in VERDICTS]:
        if counts.get(label):
            print(f"  {label:<20}{counts[label]:>4}   {counts[label] / total:>5.0%}")
    print(f"  {'total judged':<20}{total:>4}")

    supported = counts.get("SUPPORTED", 0)
    print(f"\n  citation faithfulness : {supported}/{total} "
          f"({supported / total if total else 0:.0%})")
    print("  This is STRICTER than the citation rate in evaluate.py. That one")
    print("  asks whether a claim carries a citation. This asks whether the")
    print("  citation holds it up.")

    # -----------------------------------------------------------------------
    flagged = [v for v in done.values() if v["verdict"] != "SUPPORTED"]
    by_question = defaultdict(list)
    for item in flagged:
        by_question[item["id"]].append(item)

    print("\n" + "=" * 100)
    print(f"EVERY CLAIM NOT FULLY SUPPORTED ({len(flagged)})")
    print("=" * 100)
    if not flagged:
        print("  none")
    for qid in sorted(by_question):
        print(f"\n  {qid}")
        for item in by_question[qid]:
            print(f"    {item['verdict']}   cites: "
                  f"{'; '.join(item['citations'])[:70]}")
            print(f"      claim : {item['sentence'][:120]}")
            print(f"      judge : {item['because'][:120]}")

    blocks = sum(1 for v in done.values() if v.get("lines", 1) > 1)
    print("\n" + "=" * 100)
    print("WHAT THIS NUMBER IS WORTH")
    print("=" * 100)
    print(f"  judge version : {JUDGE_VERSION}")
    print(f"  claims that are a lead-in plus a list: {blocks} of {total}")
    print()
    print("  The claim count changes with every judge version, so this rate is")
    print("  NOT comparable with the 78% from v1 or the 60% from v2. What is")
    print("  comparable is the list of substantive problems above.")
    print()
    print("  The judge was validated against errors planted on purpose: 8 of 8")
    print("  caught, 6 of 6 clean claims left alone, kappa 1.00 on 14 cases.")
    print("  That validation is what licences quoting this rate at all.")
    print()
    print("  *** THAT VALIDATION IS NOW OUT OF DATE. *** What the judge is")
    print("  shown has changed since, and a judge that reads more context")
    print("  could be a judge that forgives more. Run:")
    print()
    print("      python src/judge_control.py")
    print()
    print("  If it still catches 8 of 8, the change fixed an input defect and")
    print("  did not soften the judge. If it now misses planted errors, this")
    print("  rate is inflated and must not be quoted.")
    print(f"\n  saved: {JUDGE_PATH}")
