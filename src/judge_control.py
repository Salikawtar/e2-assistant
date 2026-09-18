"""
Stage 5, step 4d - test the judge with claims you broke on purpose.

WHY

The blind sample gave 12 out of 12 agreement and a kappa that came back
undefined, because every grader used only one label. A judge that replied
SUPPORTED to everything without reading a character would have scored the
same. And across all 49 claims the judge never once said NOT SUPPORTED, which
has two explanations that the evidence so far cannot separate: either the
system never made an unsupported claim, or the judge does not use that label.

So this file stops asking the judge questions whose answers nobody knows, and
starts asking it questions whose answers are known BY CONSTRUCTION.

HOW

Take claims the judge already marked SUPPORTED and break them:

  NUMBER      4.50 tonnes becomes 4.05 tonnes
  PERIOD      six months becomes six weeks
  NEGATION    "must submit" becomes "must not submit"
  WRONG SOURCE  a real claim shown against a different provision's text

Nothing was invented. Every broken claim started life as a sentence the judge
had already passed, so the ONLY thing that changed is the defect. Mixed in with
untouched controls, this asks the one question that matters:

    Does the verdict change when, and only when, the claim is wrong?

WHAT THE ANSWER MEANS

  caught       broken claim, judge did not say SUPPORTED. The judge reads.
  MISSED       broken claim, judge said SUPPORTED. The judge is a rubber
               stamp, and the 96% faithfulness rate cannot be quoted.
  false alarm  clean claim, judge said it was broken. The judge is jumpy, and
               its complaints need reading before they are believed.

This costs about fourteen requests and needs no human grading, because you
wrote the answer key with your own hands.

Run:  python src/judge_control.py
"""

import json
import os
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from generate_naive import flash_models, ModelPool
from judge import JUDGE_PROMPT, read_verdict, MAX_JUDGE_TOKENS, JUDGE_PATH
from agreement import quotedness

# JUDGE_PATH now follows RUN_LABEL, so the control set tests the judge on
# whichever run was actually judged. See judge.py.
from evaluate import RUN_LABEL
OUT_PATH = Path(f"eval/{RUN_LABEL}_control.json")

PER_TYPE = 2      # how many planted errors of each kind
CONTROLS = 6      # untouched claims, to catch a judge that fails everything
SEED = 5

WORD_NUMBERS = {"one": "four", "two": "seven", "three": "eight",
                "five": "fifteen", "six": "sixteen", "seven": "twelve",
                "ten": "thirty", "twelve": "twenty", "thirty": "thirteen",
                "annually": "biennially", "annual": "biennial"}

PERIODS = {"months": "weeks", "month": "week", "days": "weeks",
           "years": "months", "year": "month"}


def has_word(word, text):
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def replace_word(word, new, text):
    return re.sub(rf"\b{re.escape(word)}\b", new, text, count=1)


# ---------------------------------------------------------------------------
# The four ways to break a true sentence
# ---------------------------------------------------------------------------

def break_number(claim, evidence):
    """
    Change a figure the evidence states.

    Only a number that actually appears in the evidence is touched, and the
    replacement is rejected if it appears in the evidence too. Otherwise the
    'error' might accidentally still be true, and a planted error that is not
    an error teaches nothing.
    """
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b", claim):
        token = match.group()
        if token not in evidence:
            continue
        if "." in token:
            whole, frac = token.split(".")
            if len(frac) < 2 or frac[0] == frac[1]:
                continue
            new = f"{whole}.{frac[::-1]}"
        elif len(token) >= 2 and token[0] != token[-1]:
            new = token[::-1].lstrip("0")
        else:
            new = str(int(token) + 3)
        if not new or new == token or new in evidence:
            continue
        return (claim[:match.start()] + new + claim[match.end():],
                f"number {token} changed to {new}")

    for word, new in WORD_NUMBERS.items():
        if has_word(word, claim) and has_word(word, evidence) \
                and not has_word(new, evidence):
            return replace_word(word, new, claim), f"'{word}' changed to '{new}'"
    return None, None


def break_period(claim, evidence):
    """Keep the number, change the unit of time. Six months becomes six weeks."""
    for word, new in PERIODS.items():
        if has_word(word, claim) and has_word(word, evidence) \
                and not has_word(new, evidence):
            return replace_word(word, new, claim), f"'{word}' changed to '{new}'"
    return None, None


def break_negation(claim, evidence):
    """Reverse the duty. The words barely change; the meaning inverts."""
    if has_word("not", claim) or "must not" in evidence.lower():
        return None, None
    match = re.search(r"\bmust\b", claim)
    if not match:
        return None, None
    return (claim[:match.end()] + " not" + claim[match.end():],
            "'must' changed to 'must not'")


BREAKERS = [("NUMBER", break_number),
            ("PERIOD", break_period),
            ("NEGATION", break_negation)]


# ---------------------------------------------------------------------------

def build_cases(verdicts):
    """
    Assemble the test set: planted errors, wrong-source pairs, and controls.

    Only claims the judge ALREADY marked SUPPORTED are used, so any verdict
    that changes can only be explained by the defect that was introduced.
    """
    passed = sorted([v for v in verdicts.values() if v["verdict"] == "SUPPORTED"],
                    key=lambda v: v["key"])
    random.Random(SEED).shuffle(passed)

    cases, used = [], set()

    for label, breaker in BREAKERS:
        made = 0
        for item in passed:
            if made >= PER_TYPE or item["key"] in used:
                continue
            broken, note = breaker(item["sentence"], item["evidence"])
            if not broken:
                continue
            used.add(item["key"])
            made += 1
            cases.append({
                "case": f"{label}-{made}",
                "from": item["key"],
                "kind": label,
                "planted": note,
                "expected": "BROKEN",
                "sentence": broken,
                "stem": item.get("stem", ""),
                "evidence": item["evidence"],
            })

    # Wrong source: a true sentence shown against a provision it has nothing
    # to do with. The sentence is untouched, so a judge that only reads the
    # sentence and not the evidence will pass it.
    made = 0
    for item in passed:
        if made >= PER_TYPE or item["key"] in used:
            continue
        other = next((o for o in passed
                      if o["key"] not in used
                      and o["id"] != item["id"]
                      and not set(o["citations"]) & set(item["citations"])), None)
        if not other:
            continue
        used.add(item["key"])
        used.add(other["key"])
        made += 1
        cases.append({
            "case": f"WRONG-SOURCE-{made}",
            "from": item["key"],
            "kind": "WRONG SOURCE",
            "planted": f"shown against {'; '.join(other['citations'])[:60]} instead",
            "expected": "BROKEN",
            "sentence": item["sentence"],
            "stem": item.get("stem", ""),
            "evidence": other["evidence"],
        })

    # Controls, untouched. Claims in the model's own wording are preferred,
    # because a control made of quotations is as easy as the sample that
    # already failed to test anything.
    remaining = [i for i in passed if i["key"] not in used]
    remaining.sort(key=lambda i: quotedness(i["sentence"], i["evidence"]) != "model's own wording")
    for n, item in enumerate(remaining[:CONTROLS], 1):
        cases.append({
            "case": f"CONTROL-{n}",
            "from": item["key"],
            "kind": "CONTROL",
            "planted": "nothing changed",
            "expected": "CLEAN",
            "sentence": item["sentence"],
            "stem": item.get("stem", ""),
            "evidence": item["evidence"],
        })

    return cases


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")
    if not JUDGE_PATH.exists():
        raise SystemExit(f"{JUDGE_PATH} not found. Run src/judge.py first.")

    verdicts = {r["key"]: r for r in
                json.loads(JUDGE_PATH.read_text(encoding="utf-8"))["verdicts"]}
    cases = build_cases(verdicts)

    print("=" * 100)
    print("THE ANSWER KEY, WRITTEN BEFORE ANYTHING IS ASKED")
    print("=" * 100)
    for case in cases:
        print(f"  {case['case']:<16}{case['expected']:<8}from {case['from']:<8}"
              f"{case['planted'][:56]}")
    planted = sum(1 for c in cases if c["expected"] == "BROKEN")
    print(f"\n  {planted} planted errors, {len(cases) - planted} untouched "
          f"controls, {len(cases)} requests.\n")

    pool = ModelPool(genai.Client(api_key=key),
                     flash_models(genai.Client(api_key=key)))

    for n, case in enumerate(cases, 1):
        print(f"  {n:>3}/{len(cases)}  {case['case']}")
        # The judge is shown a list item together with its introducing
        # line, so the control set must be shown the same way. Testing the
        # judge on a different input from the one it is used on would prove
        # nothing about the judge you actually have.
        to_check = (f"{case['stem']}\n{case['sentence']}"
                    if case.get("stem") else case["sentence"])
        reply = pool.generate(
            JUDGE_PROMPT.format(evidence=case["evidence"], sentence=to_check),
            max_output_tokens=MAX_JUDGE_TOKENS)
        verdict, because = read_verdict(reply)
        case["verdict"] = verdict
        case["because"] = because
        case["judge_model"] = pool.current
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(
            {"run_date": date.today().isoformat(), "cases": cases},
            indent=2, ensure_ascii=False), encoding="utf-8")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DID THE VERDICT CHANGE WHEN, AND ONLY WHEN, THE CLAIM WAS WRONG?")
    print("=" * 100)
    print(f"  {'case':<16}{'expected':<10}{'judge said':<19}result")
    print("  " + "-" * 96)

    caught = missed = correct = false_alarm = 0
    for case in cases:
        spotted = case["verdict"] != "SUPPORTED"
        if case["expected"] == "BROKEN":
            result = "caught" if spotted else "*** MISSED ***"
            caught += spotted
            missed += not spotted
        else:
            result = "false alarm" if spotted else "correct"
            false_alarm += spotted
            correct += not spotted
        print(f"  {case['case']:<16}{case['expected']:<10}"
              f"{case['verdict']:<19}{result}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WHAT THE JUDGE SAID ABOUT EACH PLANTED ERROR")
    print("=" * 100)
    for case in cases:
        if case["expected"] != "BROKEN":
            continue
        print(f"\n  {case['case']}  ({case['planted']})")
        print(f"    verdict : {case['verdict']}")
        print(f"    because : {case['because'][:150]}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SCORECARD FOR THE JUDGE ITSELF")
    print("=" * 100)
    print(f"  planted errors caught     : {caught}/{caught + missed}")
    print(f"  planted errors MISSED     : {missed}")
    print(f"  clean claims left alone   : {correct}/{correct + false_alarm}")
    print(f"  clean claims wrongly hit  : {false_alarm}")

    by_kind = Counter(c["kind"] for c in cases
                      if c["expected"] == "BROKEN" and c["verdict"] == "SUPPORTED")
    if by_kind:
        print("\n  the kinds of error it does NOT notice:")
        for kind, n in by_kind.most_common():
            print(f"    {kind:<16}{n} missed")

    n = len(cases)
    observed = (caught + correct) / n if n else 0
    broken_n = caught + missed
    clean_n = correct + false_alarm
    said_broken = caught + false_alarm
    expected_by_chance = ((broken_n / n) * (said_broken / n)
                          + (clean_n / n) * ((n - said_broken) / n)) if n else 0
    print(f"\n  right {caught + correct} times out of {n}  ({observed:.0%})")
    if expected_by_chance < 0.999:
        k = (observed - expected_by_chance) / (1 - expected_by_chance)
        print(f"  kappa (above chance)      : {k:.2f}")
        print("  This one is defined, because the judge used more than one")
        print("  label on a set where more than one label was correct. That")
        print("  is what the twelve-claim sample could not do.")
    else:
        print("  kappa still undefined: the judge gave every case the same "
              "verdict.")

    print("\n" + "=" * 100)
    print("WHAT YOU MAY NOW SAY ABOUT THE 96%")
    print("=" * 100)
    if missed == 0 and false_alarm == 0:
        print("  The judge caught every planted error and let every clean claim")
        print("  through. The faithfulness rate is worth quoting, with the size")
        print("  of this control set stated beside it.")
    elif missed == 0:
        print("  The judge caught every planted error but flagged clean claims")
        print("  too. It is strict rather than blind, so its PARTLY verdicts")
        print("  need reading before they are believed, but a SUPPORTED verdict")
        print("  from it is meaningful.")
    elif caught == 0:
        print("  The judge caught NOTHING. It is a rubber stamp. The 96%")
        print("  faithfulness rate measures nothing and must not be quoted.")
    else:
        print(f"  The judge caught {caught} of {broken_n} planted errors. It reads,")
        print("  but it misses things, so the 96% is an UPPER BOUND on how good")
        print("  the citations are, never a measurement of it. Report it that")
        print("  way, and name the kinds of error it missed.")
