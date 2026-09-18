"""
Stage 5, step 4c - check the judge against a person, then reveal what it found.

WHY THE ORDER MATTERS

judge.py read all 49 claims and said nothing. This file compares its verdicts
against a grading sheet filled in by hand, prints how often the two agree, and
only after that shows what the judge concluded overall.

An agreement rate is the licence to quote the judge's numbers. Without it, the
judge is one language model's opinion of another language model's work, which
is not evidence of anything.

WHAT AGREEMENT IS AND IS NOT

Raw agreement is the honest headline: out of twelve claims, how many times did
the two of you say the same thing. But raw agreement flatters easy samples. If
every claim in the sample is a direct quotation from its evidence, both graders
say SUPPORTED twelve times, agreement is 100%, and nothing has been tested.
Two people who both always say yes agree perfectly and have measured nothing.

So this file also reports:

  - how many sampled claims are QUOTED word for word from their evidence, and
    how many are the model's own wording. Only the second kind is a real test.
  - kappa, which asks how much better than lucky the agreement is. When both
    graders use only one label kappa is undefined, and that is not a bug: it
    is the arithmetic telling you the sample could not discriminate.

Run:  python src/agreement.py
"""

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

JUDGE_PATH = Path("eval/stage5_judge.json")
SHEETS = [("you", Path("eval/human_grading.csv")),
          ("Claude", Path("eval/claude_grading.csv"))]

MARKER = re.compile(r"\[(\d+)\]")
LIST_LETTER = re.compile(r"^\(?[a-z0-9]{1,3}\)\s*")

VERDICTS = ["SUPPORTED", "PARTLY SUPPORTED", "NOT SUPPORTED"]


def normalise(text):
    """Same words, same shape, so a quotation can be recognised as one."""
    text = unicodedata.normalize("NFKD", text or "")
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-"))
    text = MARKER.sub(" ", text)
    text = re.sub(r"[^a-z0-9()'\"-]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def quotedness(claim, evidence):
    """
    Is this claim the evidence's own words, or the model's?

    A claim lifted verbatim is trivially supportable and tests nothing. A claim
    in the model's own words is where a judge earns its keep. Counting the two
    tells you how much your agreement rate is actually worth.
    """
    a = LIST_LETTER.sub("", normalise(claim)).rstrip(" .;,")
    b = normalise(evidence)
    if not a:
        return "empty"
    if a in b:
        return "quoted"
    if len(a) > 45 and a[:45] in b:
        return "mostly quoted"
    return "model's own wording"


def read_sheet(path):
    """key -> verdict, for rows where a verdict was actually written."""
    if not path.exists():
        return {}
    filled = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            verdict = (row.get("your_verdict") or "").strip().upper()
            if not verdict:
                continue
            match = next((v for v in VERDICTS if verdict.startswith(v)), None)
            if match is None and verdict.startswith("PARTLY"):
                match = "PARTLY SUPPORTED"
            if match is None and verdict in ("NOT", "NO"):
                match = "NOT SUPPORTED"
            filled[row["key"].strip()] = match or f"UNRECOGNISED({verdict})"
    return filled


def kappa(pairs):
    """
    Agreement above chance.

    Two graders who both say SUPPORTED to everything agree 100% of the time
    and have distinguished nothing. Kappa divides out the agreement you would
    expect from their label habits alone. 1.0 is perfect, 0.0 is no better
    than chance. It is undefined when neither grader ever used a second label,
    because then there is no chance left to correct for.
    """
    n = len(pairs)
    if not n:
        return None, "no graded items"
    observed = sum(1 for a, b in pairs if a == b) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[c] / n) * (right[c] / n)
                   for c in set(left) | set(right))
    if expected >= 0.999:
        return None, ("undefined - both graders used only one label, so the "
                      "sample could not tell a good judge from a lazy one")
    return (observed - expected) / (1 - expected), ""


if __name__ == "__main__":
    if not JUDGE_PATH.exists():
        raise SystemExit(f"{JUDGE_PATH} not found. Run src/judge.py first.")

    saved = json.loads(JUDGE_PATH.read_text(encoding="utf-8"))
    verdicts = {r["key"]: r for r in saved["verdicts"]}

    sheets = [(name, read_sheet(path)) for name, path in SHEETS]
    sheets = [(name, filled) for name, filled in sheets if filled]

    if not sheets:
        raise SystemExit("No grading sheet has been filled in yet. Write your "
                         "verdicts into the your_verdict column of "
                         "eval/human_grading.csv and run this again.")

    # -----------------------------------------------------------------------
    for name, filled in sheets:
        print("\n" + "=" * 100)
        print(f"THE JUDGE AGAINST {name.upper()}")
        print("=" * 100)
        print(f"  {'key':<8}{'quoted?':<22}{'judge':<19}{name:<19}agree")
        print("  " + "-" * 96)

        pairs, kinds = [], Counter()
        for key, mine in filled.items():
            item = verdicts.get(key)
            if not item:
                print(f"  {key:<8}not in the judge's results")
                continue
            kind = quotedness(item["sentence"], item["evidence"])
            kinds[kind] += 1
            theirs = item["verdict"]
            pairs.append((theirs, mine))
            print(f"  {key:<8}{kind:<22}{theirs:<19}{mine:<19}"
                  f"{'yes' if theirs == mine else 'NO'}")

        agreed = sum(1 for a, b in pairs if a == b)
        k, note = kappa(pairs)
        print("\n  " + "-" * 96)
        print(f"  agreed on {agreed} of {len(pairs)}  "
              f"({agreed / len(pairs):.0%})" if pairs else "  nothing graded")
        print(f"  kappa (agreement above chance): "
              f"{f'{k:.2f}' if k is not None else note}")

        print("\n  what the sample was made of:")
        for kind, n in kinds.most_common():
            print(f"    {kind:<24}{n}")
        real = kinds["model's own wording"]
        if real == 0:
            print("\n  *** EVERY sampled claim was quoted from its evidence. A")
            print("      judge that always says SUPPORTED would score exactly")
            print("      the same on this sample. The agreement rate above is")
            print("      real but it is weak evidence, and it should be")
            print("      reported that way.")
        else:
            print(f"\n  {real} claim(s) were in the model's own wording. Those are")
            print("  the ones that actually tested the judge.")

        wrong = [(k_, a, b) for (k_, (a, b)) in zip(filled.keys(), pairs)
                 if a != b]
        if wrong:
            print("\n  where you disagreed:")
            for key, theirs, mine in wrong:
                item = verdicts[key]
                print(f"\n    {key}   judge said {theirs}, {name} said {mine}")
                print(f"    claim : {item['sentence'][:110]}")
                print(f"    judge : {item['because'][:110]}")

    # -----------------------------------------------------------------------
    # Two graders against each other, if both sheets are filled.
    # -----------------------------------------------------------------------
    if len(sheets) == 2:
        (n1, s1), (n2, s2) = sheets
        shared = sorted(set(s1) & set(s2))
        if shared:
            same = sum(1 for key in shared if s1[key] == s2[key])
            print("\n" + "=" * 100)
            print(f"{n1.upper()} AGAINST {n2.upper()}")
            print("=" * 100)
            print(f"  agreed on {same} of {len(shared)} "
                  f"({same / len(shared):.0%})")
            print("  Two graders who never see each other's work but agree with")
            print("  each other are better evidence than either one alone.")

    # -----------------------------------------------------------------------
    # The reveal.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("NOW THE RESULT: DOES EACH CITATION SUPPORT ITS CLAIM?")
    print("=" * 100)

    counts = Counter(v["verdict"] for v in verdicts.values())
    total = sum(counts.values())
    for label in VERDICTS + [c for c in counts if c not in VERDICTS]:
        if counts.get(label):
            print(f"  {label:<20}{counts[label]:>4}   "
                  f"{counts[label] / total:>5.0%}")
    print(f"  {'total judged':<20}{total:>4}")

    supported = counts.get("SUPPORTED", 0)
    print(f"\n  citation faithfulness : {supported}/{total} "
          f"({supported / total if total else 0:.0%})")
    print("  This is a STRICTER number than the citation rate in evaluate.py.")
    print("  That one asked whether a claim carried a citation. This one asks")
    print("  whether the citation holds it up.")

    # -----------------------------------------------------------------------
    flagged = [v for v in verdicts.values() if v["verdict"] != "SUPPORTED"]
    by_question = defaultdict(list)
    for item in flagged:
        by_question[item["id"]].append(item)

    print("\n" + "=" * 100)
    print(f"EVERY CLAIM THE JUDGE DID NOT FULLY SUPPORT ({len(flagged)})")
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

    print("\n" + "=" * 100)
    print("HOW MUCH OF THIS YOU MAY QUOTE")
    print("=" * 100)
    print("  Only as much as the agreement rate earns. If you and the judge")
    print("  agreed on nearly everything AND the sample contained claims in the")
    print("  model's own words, the numbers above are usable with the sample")
    print("  size stated beside them. If the sample was all quotations, say so")
    print("  and treat the faithfulness rate as provisional until a harder")
    print("  sample is graded.")
