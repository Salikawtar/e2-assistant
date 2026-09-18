"""
Stage 6, step 0 - look at what actually failed, before changing anything.

evaluate.py reported 52 of 72 claims carrying a citation. The frozen
five-question baseline reported 96%. Both numbers came from the same counter,
so at least one of them is describing something other than what it claims to.

This file changes nothing and calls no model. It reads the answers evaluate.py
already saved and prints every line that was counted as UNCITED, so you can
decide for yourself whether each one is really a claim that should carry a
citation.

Run:  python src/inspect_failures.py
"""

import json
import re
from collections import Counter
from pathlib import Path

RUN_PATH = Path("eval/stage5_run.json")

# A line ending in a colon INTRODUCES something; the claim is in the lines
# below it. A line ending in a semicolon or comma is the MIDDLE of a list.
# Neither is a finished sentence, so neither should be required to carry its
# own citation. If most of the 20 look like these, the counter is wrong.
LEAD_IN = re.compile(r":\s*$")
CONTINUES = re.compile(r"[;,]\s*$")
LIST_ITEM = re.compile(r"^\(?[a-z0-9]{1,3}\)")


def classify(line):
    if LEAD_IN.search(line):
        return "lead-in, ends with a colon"
    if CONTINUES.search(line):
        return "list item, ends with ; or ,"
    if LIST_ITEM.match(line) and len(line) < 90:
        return "short list item"
    return "REAL uncited claim"


def as_lines(value):
    """report['uncited'] holds the lines themselves, not a count."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


if __name__ == "__main__":
    if not RUN_PATH.exists():
        raise SystemExit(f"{RUN_PATH} not found. Run src/evaluate.py first.")

    saved = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    results = saved.get("results", [])
    print(f"Run of {saved.get('run_date')}, prompt {saved.get('prompt_version')}, "
          f"{len(results)} questions\n")

    kinds = Counter()
    total_claims = 0
    total_uncited = 0
    real_uncited = 0

    print("=" * 100)
    print("EVERY LINE THAT WAS COUNTED AS AN UNCITED CLAIM")
    print("=" * 100)

    for row in results:
        report = row.get("report", {})
        uncited = as_lines(report.get("uncited"))
        claims = report.get("claims", 0)
        total_claims += claims
        total_uncited += len(uncited)

        if not uncited:
            continue

        print(f"\n  {row.get('id')}  status {report.get('status')}   "
              f"{len(uncited)} of {claims} claims uncited")
        print("  " + "-" * 96)
        for line in uncited:
            kind = classify(line)
            kinds[kind] += 1
            if kind == "REAL uncited claim":
                real_uncited += 1
            print(f"    [{kind}]")
            print(f"      {line[:150]}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WHAT THOSE 'UNCITED CLAIMS' ACTUALLY ARE")
    print("=" * 100)
    for kind, n in kinds.most_common():
        print(f"  {kind:<32} {n}")

    print("\n" + "=" * 100)
    print("THE SAME RATE, COUNTED TWO WAYS")
    print("=" * 100)
    as_measured = ((total_claims - total_uncited) / total_claims
                   if total_claims else 1.0)
    if_fragments_excluded = ((total_claims - real_uncited) / total_claims
                             if total_claims else 1.0)
    print(f"  as evaluate.py counted it   : "
          f"{total_claims - total_uncited}/{total_claims}  ({as_measured:.0%})")
    print(f"  counting only real claims   : "
          f"{total_claims - real_uncited}/{total_claims}  "
          f"({if_fragments_excluded:.0%})")
    print()
    print("  If the second number is close to 95% and the first is not, the")
    print("  system was never the problem and the counter needs fixing.")
    print("  If BOTH are low, the answers really are uncited and the prompt")
    print("  needs fixing. Do not change anything until you know which.")

    # -----------------------------------------------------------------------
    flagged = []
    for row in results:
        for line in as_lines(row.get("report", {}).get("obligation_on_guidance")):
            flagged.append((row.get("id"), line))

    print("\n" + "=" * 100)
    print(f"SEPARATE ISSUE, FOR LATER: {len(flagged)} OBLIGATIONS RESTING ON "
          f"GUIDANCE ALONE")
    print("=" * 100)
    print("  A 'must' or 'shall' whose only support is a guidance document is")
    print("  the system stating a duty that no law in the evidence states.")
    print("  Listed here so you have seen them. We deal with these next.\n")
    for qid, line in flagged[:12]:
        print(f"  {qid}: {line[:120]}")