"""
Stage 5, step 3 - run every question and score it against the answer key.

WHAT THIS FILE DOES

It asks the system all 22 questions, then compares each answer to what
eval_questions.py said the answer should contain. Nothing here uses a model to
judge. Every check is a plain comparison a person could do by hand, which is
why these numbers can be trusted without further argument.

WHAT CHANGED IN THIS VERSION

The first run reported 52/72 claims carrying a citation, against a target of
0.95. Twenty claims were uncited, and SIXTEEN of them came from one question,
S4, whose reply never reached its STATUS line. Those sixteen sentences were
not claims at all. They were the model deliberating out loud - "Wait, does the
user even have to prepare an E2 plan?" - preserved because clean() had no
STATUS line to cut at.

So the rule now applied:

    AN ANSWER WITH NO STATUS LINE IS UNUSABLE. It counts as one broken
    answer. It is not counted as sixteen uncited claims.

This is not softening the score. The question still FAILS, it is still listed,
and the unusable count still appears. What changes is that a rate measuring
ANSWER QUALITY is no longer computed over text that is not an answer. Left as
it was, the number would swing every run according to how long the model
happened to ramble, which is a measurement that tells you nothing.

Both rates are printed, so nothing is hidden.

WHAT IT STILL CANNOT DO

It cannot tell whether a sentence is actually SUPPORTED by the provision it
cites. "The plan must be reviewed every two years [1]" contains no forbidden
word, cites a real source, and is wrong. Catching that needs a reader, and
that is step 4.

Run:  python src/evaluate.py
"""

import csv
import json
import os
from collections import Counter
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from eval_questions import QUESTIONS, FAMILIES
from answer import (load_resources, answer_question, abstention_gap,
                    verdict_of, build_pool, re_analyse, PROMPT, pinned_model)

# How many questions in a row may fail before the run gives up. One question
# refusing is usually that question being large. Three in a row means the free
# tier is closed and there is nothing to do but come back later.
GIVE_UP_AFTER = 3

# Every experiment writes its own pair of files, so the run it is being
# compared against is never overwritten. Change this ONE line per experiment
# and the previous results stay on disk to compare against.
#
#   stage5          the frozen Stage 5 baseline, two models, no fixes
#   stage6_pinned   the same system with one model pinned. This is the run
#                   every later number is measured against, because a fix
#                   compared to a two-model baseline measures the fix and the
#                   model together.
RUN_LABEL = "stage6_rule6"

RUN_PATH = Path(f"eval/{RUN_LABEL}_run.json")
CSV_PATH = Path(f"eval/{RUN_LABEL}_results.csv")

# The prompt this run measures. If the prompt changes, the saved answers
# describe a different system and must not be reused.
PROMPT_VERSION = "stage6_rule6"

# A reply that reached one of these is an answer we can talk about. Anything
# else is a failed generation, and its sentences are not claims.
USABLE_STATUS = {"ANSWERED", "PARTIAL", "REFUSED"}


def contains_any(text, variants):
    """One fact, several spellings. '6 months' and 'six months' are the same."""
    lowered = text.lower()
    return any(v.lower() in lowered for v in variants)


def score_one(question_spec, result):
    """
    Compare one answer against one answer-key entry.

    Every check below is a string comparison. That is deliberate: a number
    produced by a rule you can read is worth more than a cleverer number
    nobody can audit.
    """
    answer = result["answer"]
    # The checks are recomputed here, not read back from the run file, so a
    # fix to the checking code takes effect without re-buying the answers.
    report = re_analyse(result)
    cited = " ".join(s["citation"] for s in result["sources"])

    usable = report["status"] in USABLE_STATUS

    missing_facts = [" / ".join(v) for v in question_spec["must_include"]
                     if not contains_any(answer, v)]
    missing_cites = [c for c in question_spec["must_cite"]
                     if c.lower() not in cited.lower()]
    forbidden = [f for f in question_spec["must_not_include"]
                 if f.lower() in answer.lower()]

    gap = abstention_gap(question_spec["expected_status"], report["status"])

    # A question passes only if EVERYTHING holds. Partial credit would hide
    # exactly the failures this set was built to find. An unusable answer can
    # never pass, whatever else happens to be true of its text.
    passed = (usable and gap == 0 and not missing_facts and not missing_cites
              and not forbidden and not report["problems"]
              and not report["invented_markers"] and not report["invented_refs"])

    return {
        "id": question_spec["id"],
        "family": question_spec["family"],
        "question": question_spec["question"],
        "expected": question_spec["expected_status"],
        "got": report["status"],
        "usable": usable,
        "gap": gap,
        "verdict": verdict_of(gap),
        "missing_facts": missing_facts,
        "missing_cites": missing_cites,
        "forbidden_present": forbidden,
        "claims": report["claims"],
        "uncited": len(report["uncited"]),
        "invented_markers": report["invented_markers"],
        "invented_refs": report["invented_refs"],
        "obligations_on_guidance": len(report["obligation_on_guidance"]),
        "problems": report["problems"],
        "route": result["route"],
        "model": result["model"],
        "passed": passed,
    }


def load_previous():
    """
    Resume support.

    22 questions with retries and busy-model fallbacks takes a while, and the
    free tier refuses requests without warning. Answers already collected for
    THIS prompt version are reused; a changed prompt invalidates all of them,
    because they would describe a different system.

    Note what is NOT invalidated: a change to the SCORING, like the one in this
    file. Scoring is recomputed from the saved answers every run, so a fix to
    the scoring can be checked without spending a single request.
    """
    if not RUN_PATH.exists():
        return {}
    saved = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    if saved.get("prompt_version") != PROMPT_VERSION:
        print("  saved run used a different prompt version - starting fresh")
        return {}
    if saved.get("prompt_hash") != hash_prompt():
        print("  the prompt text has changed since the saved run - starting fresh")
        return {}
    if saved.get("model_pin", None) != pinned_model():
        print("  a different model is pinned now - starting fresh, because "
              "reusing answers\n  from another model is exactly the "
              "contamination pinning exists to remove")
        return {}
    return {r["id"]: r for r in saved.get("results", [])}


def hash_prompt():
    import hashlib
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:16]


def save(results):
    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_PATH.write_text(json.dumps({
        "run_date": date.today().isoformat(),
        "run_label": RUN_LABEL,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": hash_prompt(),
        "model_pin": pinned_model(),
        "questions": len(QUESTIONS),
        "results": list(results.values()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    print(f"  run label : {RUN_LABEL}")
    print(f"  model     : {pinned_model() or 'auto (newest available first)'}")
    print(f"  writing   : {RUN_PATH}\n")

    done = load_previous()
    missing = [s for s in QUESTIONS if s["id"] not in done]

    # Only build the retrieval index and the model pool if there is actually
    # something to generate. Re-scoring a finished run should cost nothing.
    if missing:
        res = load_resources(genai.Client(api_key=key))
        if done:
            print(f"  resuming: {len(done)} of {len(QUESTIONS)} already answered\n")
        in_a_row = 0
        for spec in missing:
            print(f"  {spec['id']:<4} {spec['question'][:64]}")
            try:
                result = answer_question(spec["question"], res)
            except SystemExit as stop:
                # S4 has the largest prompt in the set and is the first to be
                # refused when the free tier is tight. Stopping the whole run
                # at it left eight later questions unanswered for no reason,
                # so a refused question is now SKIPPED rather than fatal.
                #
                # The pool has to be rebuilt: once it has exhausted its only
                # model it considers itself finished, and every later call
                # would fail instantly without ever contacting the API.
                in_a_row += 1
                print(f"       skipped, will be retried next run  ({stop})")
                res["pool"] = build_pool(res["client"])
                if in_a_row >= GIVE_UP_AFTER:
                    print(f"\n  {GIVE_UP_AFTER} refusals in a row. The free tier "
                          f"is closed for now, so nothing is gained by asking\n"
                          f"  the rest. Everything answered is saved.")
                    break
                continue
            in_a_row = 0
            done[spec["id"]] = {**result, "id": spec["id"]}
            save(done)
    else:
        print(f"  all {len(QUESTIONS)} answers already saved - re-scoring only, "
              f"no requests sent\n")

    # A scorecard over part of the set would sit next to a 22 question
    # baseline and invite a comparison that is not valid. So there is no
    # partial scorecard: finish the run first.
    still_missing = [s["id"] for s in QUESTIONS if s["id"] not in done]
    if still_missing:
        print(f"\n  {len(done)} of {len(QUESTIONS)} answered and saved.")
        print(f"  still to do: {', '.join(still_missing)}")
        print(f"\n  Run this file again to continue. Nothing is lost and")
        print(f"  nothing already answered will be asked twice.")
        print(f"  No score is printed until all {len(QUESTIONS)} are in, because a")
        print(f"  scorecard over {len(done)} questions cannot be compared with one")
        print(f"  over {len(QUESTIONS)}.")
        raise SystemExit(0)

    scored = [score_one(spec, done[spec["id"]]) for spec in QUESTIONS]

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"  {'id':<5}{'family':<13}{'expect':<10}{'got':<10}"
          f"{'pass':<6}what went wrong")
    print("  " + "-" * 96)
    for s in scored:
        wrong = []
        if not s["usable"]:
            wrong.append("UNUSABLE, no answer produced")
        elif s["gap"] != 0:
            wrong.append(s["verdict"])
        if s["missing_facts"]:
            wrong.append("missing: " + "; ".join(s["missing_facts"])[:40])
        if s["missing_cites"]:
            wrong.append("no cite: " + ", ".join(s["missing_cites"])[:30])
        if s["forbidden_present"]:
            wrong.append("said: " + ", ".join(s["forbidden_present"]))
        if s["invented_markers"] or s["invented_refs"]:
            wrong.append("invented citation")
        if s["problems"]:
            wrong.append(s["problems"][0][:40])
        mark = "ok" if s["passed"] else "FAIL"
        print(f"  {s['id']:<5}{s['family']:<13}{s['expected']:<10}{s['got']:<10}"
              f"{mark:<6}{'; '.join(wrong)[:52]}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BY FAMILY")
    print("=" * 100)
    print("  A family that fails together is a design problem. One question")
    print("  failing on its own is usually just that question.\n")
    print(f"  {'family':<14}{'passed':>8}{'of':>4}   what is failing")
    print("  " + "-" * 96)
    for family in FAMILIES:
        rows = [s for s in scored if s["family"] == family]
        passed = sum(1 for s in rows if s["passed"])
        failing = [s["id"] for s in rows if not s["passed"]]
        print(f"  {family:<14}{passed:>8}{len(rows):>4}   "
              f"{', '.join(failing) if failing else 'all pass'}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SCORECARD")
    print("=" * 100)
    total = len(scored)
    passed = sum(1 for s in scored if s["passed"])
    under = sum(1 for s in scored if s["gap"] is not None and s["gap"] > 0)
    over = sum(1 for s in scored if s["gap"] is not None and s["gap"] < 0)
    unusable = [s for s in scored if not s["usable"]]
    usable = [s for s in scored if s["usable"]]

    # The citation rate is a statement about ANSWERS. Text from a generation
    # that never produced an answer is excluded from it, and reported on its
    # own line instead.
    claims = sum(s["claims"] for s in usable)
    uncited = sum(s["uncited"] for s in usable)
    all_claims = sum(s["claims"] for s in scored)
    all_uncited = sum(s["uncited"] for s in scored)

    print(f"  questions passing every check   : {passed}/{total} "
          f"({passed / total:.0%})")
    print(f"  status matched                  : "
          f"{sum(1 for s in scored if s['gap'] == 0)}/{total}")
    print(f"    under-abstention (answered too much) : {under}")
    print(f"    over-abstention  (refused too much)  : {over}")
    print(f"    unusable (no answer produced)        : {len(unusable)}"
          f"{'  -> ' + ', '.join(s['id'] for s in unusable) if unusable else ''}")

    print(f"\n  claims carrying a citation      : {claims - uncited}/{claims} "
          f"({(claims - uncited) / claims if claims else 1:.0%})"
          f"   [{len(usable)} usable answers]")
    if unusable:
        print(f"    the same rate over ALL text   : "
              f"{all_claims - all_uncited}/{all_claims} "
              f"({(all_claims - all_uncited) / all_claims if all_claims else 1:.0%})"
              f"   [including {len(unusable)} failed generation(s)]")
        print(f"    the difference is the model's own reasoning being counted")
        print(f"    as claims. Both numbers are shown so neither can hide.")

    print(f"  invented citation numbers       : "
          f"{sum(len(s['invented_markers']) for s in scored)}")
    print(f"  provisions cited from nowhere   : "
          f"{sum(len(s['invented_refs']) for s in scored)}")
    print(f"  required facts missing          : "
          f"{sum(len(s['missing_facts']) for s in scored)}")
    print(f"  forbidden values stated         : "
          f"{sum(len(s['forbidden_present']) for s in scored)}")
    print(f"  obligations on guidance alone   : "
          f"{sum(s['obligations_on_guidance'] for s in scored)}")

    print("\n  against the targets in your specification:")
    citation_rate = (claims - uncited) / claims if claims else 1
    print(f"    citation correctness >= 0.95 : {citation_rate:.2f}  "
          f"{'MET' if citation_rate >= 0.95 else 'NOT MET'}")
    print(f"    both abstention failures measured : yes "
          f"({under} under, {over} over)")

    models = Counter(s["model"] for s in scored)
    print(f"\n  models used: " +
          ", ".join(f"{m.split('/')[-1]} x{n}" for m, n in models.items()))
    if len(models) > 1:
        print("  more than one model answered this run, so a change in score")
        print("  next time may be the model rather than the system")

    # -----------------------------------------------------------------------
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "family", "question", "expected", "got", "usable", "verdict",
            "passed", "missing_facts", "missing_cites", "forbidden_present",
            "claims", "uncited", "obligations_on_guidance", "route", "model"])
        writer.writeheader()
        for s in scored:
            writer.writerow({k: ("; ".join(v) if isinstance(v, list) else v)
                             for k, v in s.items() if k in writer.fieldnames})

    print(f"\n  answers saved  : {RUN_PATH}")
    print(f"  table for review: {CSV_PATH}   (open it in Excel)")
    print("\n  Every number above came from a string comparison. Whether each")
    print("  sentence is actually supported by the provision it cites is the")
    print("  next step, and it needs a reader rather than a rule.")
