"""
Stage 6, step 5 - run the held-out questions ONCE.

WHAT THIS ANSWERS

The tuned 22 questions now score 21 to 22. Two things were changed to get
there, and both were chosen by looking at those same 22 questions. So the
score cannot tell you whether the system improved or whether it learned the
exam paper.

These twelve questions were written after every fix, on provisions the
original set never used. Whatever they say is the honest number.

THE ONE RULE

Run it once. Do not tune anything afterwards.

If something fails here and is worth fixing, fix it and then write another
fresh set to check that fix. The moment this set is used to choose a change,
it stops being held out. That is not a technicality; it is the only thing that
makes the number mean anything.

Run:  python src/holdout.py
"""

import csv
import json
import os
from collections import Counter
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from holdout_questions import QUESTIONS, FAMILIES
from answer import (load_resources, answer_question, build_pool, PROMPT,
                    pinned_model)
from evaluate import score_one, hash_prompt, RUN_LABEL, GIVE_UP_AFTER

RUN_PATH = Path("eval/holdout_run.json")
CSV_PATH = Path("eval/holdout_results.csv")


def load_previous():
    if not RUN_PATH.exists():
        return {}
    saved = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    if (saved.get("prompt_hash") != hash_prompt()
            or saved.get("model_pin") != pinned_model()):
        print("  the saved held-out run used a different system - starting fresh")
        return {}
    return {r["id"]: r for r in saved.get("results", [])}


def save(results):
    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_PATH.write_text(json.dumps({
        "run_date": date.today().isoformat(),
        "measuring": RUN_LABEL,
        "prompt_hash": hash_prompt(),
        "model_pin": pinned_model(),
        "results": list(results.values()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    print(f"  measuring the configuration: {RUN_LABEL}")
    print(f"  model                      : {pinned_model()}")
    print(f"  held-out questions         : {len(QUESTIONS)}\n")

    done = load_previous()
    missing = [s for s in QUESTIONS if s["id"] not in done]

    if missing:
        res = load_resources(genai.Client(api_key=key))
        in_a_row = 0
        for spec in missing:
            print(f"  {spec['id']:<4} {spec['question'][:64]}")
            try:
                result = answer_question(spec["question"], res)
            except SystemExit as stop:
                in_a_row += 1
                print(f"       skipped, retried next run  ({stop})")
                res["pool"] = build_pool(res["client"])
                if in_a_row >= GIVE_UP_AFTER:
                    print("\n  three refusals in a row, stopping. "
                          "Everything answered is saved.")
                    break
                continue
            in_a_row = 0
            done[spec["id"]] = {**result, "id": spec["id"]}
            save(done)
    else:
        print("  all answers already saved - re-scoring only, no requests sent\n")

    still = [s["id"] for s in QUESTIONS if s["id"] not in done]
    if still:
        print(f"\n  {len(done)} of {len(QUESTIONS)} answered. "
              f"Still to do: {', '.join(still)}")
        print("  Run this file again to finish. No score until all are in.")
        raise SystemExit(0)

    scored = [score_one(spec, done[spec["id"]]) for spec in QUESTIONS]

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("HELD-OUT RESULTS")
    print("=" * 100)
    print(f"  {'id':<5}{'family':<13}{'expect':<10}{'got':<10}{'pass':<6}"
          f"what went wrong")
    print("  " + "-" * 96)
    for s in scored:
        wrong = []
        if s["gap"] != 0:
            wrong.append(s["verdict"])
        if s["missing_facts"]:
            wrong.append("missing: " + "; ".join(s["missing_facts"])[:38])
        if s["missing_cites"]:
            wrong.append("no cite: " + ", ".join(s["missing_cites"])[:28])
        if s["forbidden_present"]:
            wrong.append("said: " + ", ".join(s["forbidden_present"]))
        if s["invented_markers"] or s["invented_refs"]:
            wrong.append("invented citation")
        if s["problems"]:
            wrong.append(s["problems"][0][:38])
        print(f"  {s['id']:<5}{s['family']:<13}{s['expected']:<10}{s['got']:<10}"
              f"{'ok' if s['passed'] else 'FAIL':<6}{'; '.join(wrong)[:52]}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BY FAMILY")
    print("=" * 100)
    for family in FAMILIES:
        rows = [s for s in scored if s["family"] == family]
        if not rows:
            continue
        failing = [s["id"] for s in rows if not s["passed"]]
        print(f"  {family:<14}{sum(1 for s in rows if s['passed']):>3}"
              f"/{len(rows):<4}   {', '.join(failing) if failing else 'all pass'}")

    # -----------------------------------------------------------------------
    total = len(scored)
    passed = sum(1 for s in scored if s["passed"])
    claims = sum(s["claims"] for s in scored)
    uncited = sum(s["uncited"] for s in scored)
    rate = passed / total

    print("\n" + "=" * 100)
    print("THE HONEST NUMBER")
    print("=" * 100)
    print(f"  held out : {passed}/{total}  ({rate:.0%})")
    print(f"  tuned 22 : 21 to 22 out of 22  (95% to 100%), over three runs")
    print()
    print(f"  under-abstention : "
          f"{sum(1 for s in scored if s['gap'] is not None and s['gap'] > 0)}")
    print(f"  over-abstention  : "
          f"{sum(1 for s in scored if s['gap'] is not None and s['gap'] < 0)}")
    print(f"  claims cited     : {claims - uncited}/{claims}"
          f"  ({(claims - uncited) / claims if claims else 1:.0%})")
    print(f"  invented         : "
          f"{sum(len(s['invented_markers']) + len(s['invented_refs']) for s in scored)}")

    print("\n" + "=" * 100)
    print("HOW TO READ IT")
    print("=" * 100)
    if rate >= 0.85:
        print("  Close to the tuned set. The improvements generalise to questions")
        print("  the system has never met, so they were engineering rather than")
        print("  memorisation. Quote BOTH numbers, with the sample sizes.")
    elif rate >= 0.65:
        print("  Noticeably below the tuned set. Some of the gain was real and")
        print("  some of it was fitting. The held-out number is the one to")
        print("  publish, and the gap between the two is itself the finding.")
    else:
        print("  Far below the tuned set. The system was tuned to its own exam")
        print("  paper. This number is the true one, and the 21 to 22 must not")
        print("  be quoted without it standing immediately beside it.")
    print()
    print("  Whatever it says, stop here. Fixing something because THIS set")
    print("  failed turns it into a second exam paper, and then there is no")
    print("  honest number left at all.")

    # -----------------------------------------------------------------------
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "id", "family", "question", "expected", "got", "passed",
            "missing_facts", "missing_cites", "claims", "uncited", "route"])
        writer.writeheader()
        for s in scored:
            writer.writerow({k: ("; ".join(v) if isinstance(v, list) else v)
                             for k, v in s.items() if k in writer.fieldnames})
    print(f"\n  saved: {RUN_PATH}")
    print(f"  table: {CSV_PATH}")
