"""
Stage 6, step 2 - how much does the score move when NOTHING changes?

WHY

Two runs of the identical system, same model, same prompt, same evidence,
temperature 0, gave 19/22 and 18/22. Two questions changed their answer on
their own:

    C2   ANSWERED  ->  PARTIAL
    U4   REFUSED   ->  PARTIAL

Temperature 0 is supposed to make the same input produce the same output. On a
model that thinks before it writes, it does not. The thinking is not fully
deterministic, and a different chain of thought can end at a different status.

This matters for everything that comes next. The remaining fixes are each
expected to gain one or two questions. If the score already wanders by one or
two on its own, a gain of one proves nothing, and any result could be argued
either way. That is not a measurement, it is a coin toss with a spreadsheet.

WHAT THIS FILE DOES

Runs the SAME configuration three times and reports two things:

    the noise band   the highest and lowest score with nothing changed. Any
                     future improvement has to be bigger than this to count.

    which questions  a question that gives the same verdict every time is
                     stable and can be trusted. One that flips is telling you
                     about the model's mood, not about your system.

Both are needed. A fix that changes a stable question is real. The same fix
"changing" an unstable one means nothing at all.

COST: 3 runs of 22 questions, roughly 45 cents. It saves after every answer,
so if it stops, run it again.

Run:  python src/variance.py
"""

import json
import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from eval_questions import QUESTIONS
from answer import load_resources, answer_question, build_pool, pinned_model
from evaluate import score_one, RUN_LABEL

REPEATS = 3
OUT_PATH = Path("eval/stage6_variance.json")
GIVE_UP_AFTER = 3


def load_saved():
    if not OUT_PATH.exists():
        return {}
    saved = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    if saved.get("label") != RUN_LABEL or saved.get("model") != pinned_model():
        print("  the saved variance run used a different configuration - "
              "starting fresh")
        return {}
    return saved.get("answers", {})


def save(answers):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "run_date": date.today().isoformat(),
        "label": RUN_LABEL,
        "model": pinned_model(),
        "repeats": REPEATS,
        "answers": answers,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    answers = load_saved()
    wanted = [(n, spec) for n in range(1, REPEATS + 1) for spec in QUESTIONS]
    todo = [(n, spec) for n, spec in wanted if f"{n}:{spec['id']}" not in answers]

    print(f"  configuration : {RUN_LABEL}")
    print(f"  model         : {pinned_model()}")
    print(f"  repeats       : {REPEATS}   ({len(QUESTIONS)} questions each)")
    print(f"  still to ask  : {len(todo)} of {len(wanted)}\n")

    if todo:
        res = load_resources(genai.Client(api_key=key))
        in_a_row = 0
        for n, spec in todo:
            print(f"  run {n}  {spec['id']:<4} {spec['question'][:56]}")
            try:
                result = answer_question(spec["question"], res)
            except SystemExit as stop:
                in_a_row += 1
                print(f"        skipped, retried next run  ({stop})")
                res["pool"] = build_pool(res["client"])
                if in_a_row >= GIVE_UP_AFTER:
                    print("\n  three refusals in a row, stopping. "
                          "Everything answered is saved.")
                    break
                continue
            in_a_row = 0
            answers[f"{n}:{spec['id']}"] = result
            save(answers)

    missing = [f"{n}:{s['id']}" for n, s in wanted if f"{n}:{s['id']}" not in answers]
    if missing:
        print(f"\n  {len(wanted) - len(missing)} of {len(wanted)} collected. "
              f"Run this file again to finish.")
        raise SystemExit(0)

    # -----------------------------------------------------------------------
    # Score each run separately, using the same rules evaluate.py uses.
    # -----------------------------------------------------------------------
    scored = defaultdict(dict)          # run -> id -> scored row
    for n in range(1, REPEATS + 1):
        for spec in QUESTIONS:
            scored[n][spec["id"]] = score_one(spec, answers[f"{n}:{spec['id']}"])

    totals = {n: sum(1 for r in scored[n].values() if r["passed"])
              for n in scored}

    print("\n" + "=" * 100)
    print("THE SAME SYSTEM, RUN THREE TIMES")
    print("=" * 100)
    for n in sorted(totals):
        print(f"  run {n}: {totals[n]}/{len(QUESTIONS)} passing")
    low, high = min(totals.values()), max(totals.values())
    print(f"\n  NOISE BAND: {low} to {high} out of {len(QUESTIONS)}, "
          f"a spread of {high - low}")
    print(f"  A future change must gain MORE than {high - low} question(s) "
          f"before it can\n  be called an improvement rather than a good day.")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WHICH QUESTIONS ARE STABLE, AND WHICH ARE NOT")
    print("=" * 100)
    print(f"  {'id':<6}{'expected':<10}{'statuses seen':<34}{'passed':<10}verdict")
    print("  " + "-" * 96)

    stable, unstable = [], []
    for spec in QUESTIONS:
        qid = spec["id"]
        statuses = [scored[n][qid]["got"] for n in sorted(scored)]
        passes = [scored[n][qid]["passed"] for n in sorted(scored)]
        same_status = len(set(statuses)) == 1
        same_pass = len(set(passes)) == 1
        (stable if same_status and same_pass else unstable).append(qid)
        counts = ", ".join(f"{s} x{c}" for s, c in Counter(statuses).most_common())
        verdict = "stable" if (same_status and same_pass) else "*** FLIPS ***"
        print(f"  {qid:<6}{spec['expected_status']:<10}{counts:<34}"
              f"{sum(passes)}/{REPEATS:<7}  {verdict}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("HOW TO USE THIS")
    print("=" * 100)
    print(f"  stable   : {len(stable)} questions - {', '.join(stable)}")
    print(f"  unstable : {len(unstable)} questions - "
          f"{', '.join(unstable) if unstable else 'none'}")
    print()
    print("  A fix that changes a STABLE question changed something real.")
    print("  The same fix 'changing' an unstable one tells you nothing, and")
    print("  claiming it as a win is how a system gets tuned to luck.")
    print()
    if unstable:
        print("  When you report a score, report it as a range over these three")
        print(f"  runs, not as a single number. '{low} to {high} out of "
              f"{len(QUESTIONS)}' is honest.")
        print(f"  '{high}/{len(QUESTIONS)}' is the best run quoted as if it were "
              f"the only one.")
    else:
        print("  Nothing flipped. The system is reproducible on this set, and a")
        print("  change of even one question can be taken seriously.")
    print()
    print("  Note what this does NOT measure: whether the answers are correct.")
    print("  A question that fails identically three times is perfectly")
    print("  stable and perfectly wrong.")
    print(f"\n  saved: {OUT_PATH}")
