"""
Stage 6, step 1 - find out WHERE each failure happened, before fixing anything.

THE QUESTION THIS FILE ANSWERS

Stage 5 said five questions failed. It did not say which part of the system
failed, and the parts need completely different repairs. A missing fact can
come from three places:

    CORPUS      the fact is not in any document we hold.
                No prompt or retrieval change will ever produce it.

    RETRIEVAL   the fact is in the corpus, but was never handed to the model.
                The generator could not have said it. The prompt is innocent.

    GENERATION  the fact WAS handed to the model, and the model left it out.
                Now, and only now, is the prompt the thing to change.

Telling these apart is the whole job of this file, and it is done by looking
rather than guessing: for every required fact and every required citation that
came back missing, it asks whether that text was in the evidence supplied, and
if not, whether it exists anywhere in the corpus at all.

O5 is the example worth watching. It reads like a citation problem, because
the answer never cited section 10. If section 10 was never retrieved, it is a
retrieval problem, and rewriting the prompt would have been a month of work
aimed at the wrong component.

Nothing here calls a model. It reads files you already have.

Run:  python src/diagnose.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval_questions import QUESTIONS
from judge import evidence_map, USABLE_STATUS

RUN_PATH = Path("eval/stage5_run.json")


def where(needles, supplied, corpus):
    """
    Three-way verdict for one missing thing.

    'needles' is the list of accepted spellings for one fact, because "6
    months" and "six months" are the same fact and a diagnosis that depends on
    spelling is not a diagnosis.
    """
    low_supplied, low_corpus = supplied.lower(), corpus.lower()
    if any(n.lower() in low_supplied for n in needles):
        return "GENERATION", "it was in the evidence and the model left it out"
    if any(n.lower() in low_corpus for n in needles):
        return "RETRIEVAL", "it is in the corpus but was never supplied"
    return "CORPUS", "no document we hold contains it, or the answer key is wrong"


if __name__ == "__main__":
    if not RUN_PATH.exists():
        raise SystemExit(f"{RUN_PATH} not found. Run src/evaluate.py first.")

    run = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    results = {r["id"]: r for r in run["results"]}

    texts = evidence_map()
    corpus = "\n".join(t for group in texts.values() for t in group)
    all_citations = list(texts.keys())

    # -----------------------------------------------------------------------
    print("=" * 100)
    print("STEP 0 CHECK: WHICH MODEL ANSWERED WHICH QUESTION")
    print("=" * 100)
    models = Counter(r["model"] for r in results.values())
    for name, n in models.most_common():
        ids = sorted(k for k, r in results.items() if r["model"] == name)
        print(f"  {str(name).split('/')[-1]:<24}{n:>3}   {', '.join(ids)}")
    if len(models) > 1:
        print("\n  More than one model produced this baseline. Before ANY fix is")
        print("  measured, one model has to be pinned, or an improvement cannot")
        print("  be told apart from a different model having answered.")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WHERE EACH MISSING FACT ACTUALLY WENT MISSING")
    print("=" * 100)

    tally = Counter()
    for spec in QUESTIONS:
        result = results.get(spec["id"])
        if not result:
            continue

        supplied = "\n".join(
            t for source in result["sources"]
            for t in texts.get(source["citation"], []))
        answer = result["answer"]

        missing_facts = [v for v in spec["must_include"]
                         if not any(x.lower() in answer.lower() for x in v)]
        missing_cites = [c for c in spec["must_cite"]
                         if c.lower() not in " ".join(
                             s["citation"] for s in result["sources"]).lower()]

        if not missing_facts and not missing_cites:
            continue

        print(f"\n  {spec['id']}  {spec['question'][:76]}")
        print(f"      status {result['report']['status']}, "
              f"{len(result['sources'])} sources supplied, "
              f"model {str(result['model']).split('/')[-1]}")

        for variants in missing_facts:
            verdict, why = where(variants, supplied, corpus)
            tally[verdict] += 1
            print(f"      missing fact  : {' / '.join(variants)[:52]}")
            print(f"        -> {verdict}: {why}")

        for cite in missing_cites:
            found = [c for c in all_citations if cite.lower() in c.lower()]
            if found:
                verdict, why = "RETRIEVAL", (
                    f"{len(found)} chunk(s) carry this citation, none was supplied")
            else:
                verdict, why = "CORPUS", (
                    "no chunk in the corpus carries this citation")
            tally[verdict] += 1
            print(f"      missing cite  : {cite}")
            print(f"        -> {verdict}: {why}")
            for c in found[:3]:
                print(f"           exists as: {c}")

    print("\n  " + "-" * 96)
    print("  where the failures live: " +
          "   ".join(f"{k} {v}" for k, v in tally.most_common()))
    print("  A fix aimed at the wrong one of these three cannot work, however")
    print("  well it is written.")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("S4: WHY NO ANSWER WAS PRODUCED")
    print("=" * 100)
    s4 = results.get("S4")
    if s4:
        report = s4["report"]
        print(f"  route            : {s4['route']}")
        print(f"  sources supplied : {len(s4['sources'])}")
        print(f"  attempts made    : {s4.get('attempts')}  "
              f"(2 means the retry fired and did not save it)")
        print(f"  status read      : {report['status']}")
        print(f"  answer length    : {len(s4['answer'])} characters")
        print(f"  'STATUS' appears anywhere in the text: "
              f"{'yes' if 'STATUS' in s4['answer'].upper() else 'no'}")
        print(f"  problems         : {'; '.join(report['problems'])}")
        print("\n  first 300 characters of what came back:")
        for line in s4["answer"][:300].splitlines():
            print(f"    {line}")
        print("\n  If attempts is 2 and neither reply reached a STATUS line, the")
        print("  retry note is not strong enough and the fix belongs in the")
        print("  generation loop, not in the seven rules.")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("CHUNK INTEGRITY: WAS THE LIST IN SECTION 4(2) EVER WHOLE?")
    print("=" * 100)
    print("  C1 lost items (n) and (o) from the end of a list. A model dropping")
    print("  the last two items and a chunk that stops before them look")
    print("  identical in the answer and need opposite fixes.\n")
    for citation, group in texts.items():
        if "4(2)" not in citation:
            continue
        for text in group:
            letters = re.findall(r"\(([a-o])\)", text)
            print(f"  {citation}")
            print(f"    length      : {len(text)} characters")
            print(f"    letters seen: {''.join(sorted(set(letters)))}")
            print(f"    ends with   : ...{text[-70:].strip()}")
            print(f"    contains (o): {'yes' if '(o)' in text else 'NO'}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("THE TEN OBLIGATIONS RESTING ON GUIDANCE ALONE")
    print("=" * 100)
    print("  Guidance explains duties. It does not create them. Each of these")
    print("  states a duty while citing only the Technical Guidelines, in an")
    print("  answer where law was also available to cite. Some will be fair")
    print("  restatements. Some will be the system promoting a recommendation")
    print("  into a legal requirement. This one needs your reading, not code.\n")
    grouped = defaultdict(list)
    for qid, result in results.items():
        if result["report"]["status"] not in USABLE_STATUS:
            continue
        for sentence in result["report"]["obligation_on_guidance"]:
            grouped[qid].append(sentence)
    n = 0
    for qid in sorted(grouped):
        print(f"  {qid}")
        for sentence in grouped[qid]:
            n += 1
            print(f"    {n:>2}. {sentence[:150]}")
    print(f"\n  {n} in total.")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WHAT TO DO WITH THIS")
    print("=" * 100)
    print("  Fix in this order, because each one is cheaper and more certain")
    print("  than the one after it:")
    print("    1. CORPUS problems  - a document is missing, or the key is wrong.")
    print("    2. RETRIEVAL        - the evidence exists and was not supplied.")
    print("    3. GENERATION       - the evidence was supplied and ignored.")
    print()
    print("  Prompt changes are the LAST resort, not the first, because they")
    print("  affect all 22 questions at once and S1 and U4 already pull in")
    print("  opposite directions.")
