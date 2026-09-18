"""
Stage 6, step 1b - find out WHY the right provision was never supplied.

WHAT THIS SETTLES

diagnose.py proved that sections 3, 4 and 10 exist in the corpus and were not
handed to the model. It could not say why, and the possible reasons need
opposite repairs:

    NEAR MISS        the chunk ranked just outside LAW_SLOTS.
                     The ranking was right, the quota was too small.
                     Fix: one constant.

    FUSION PENALTY   the chunk ranked at or near the top by MEANING, and was
                     pushed down when fused with the keyword ranking.
                     Fix: how the two rankings are combined, not how many
                     slots there are.

    RANKED LOW       neither search found it. More slots would flood the
                     evidence with weak chunks trying to reach it.

    NEVER A CANDIDATE  outside the fusion depth in BOTH rankings, so it was
                     never even considered.

WHAT THE FIRST VERSION OF THIS FILE GOT WRONG

It reported "NOT IN CORPUS" for Schedule 1 on T1, T2, T3 and T4, and for the
ECCC list on T4. All four of those questions PASSED. Schedule 1 is not
retrieved at all: it is read from the table by lookup_schedule1.py, exactly
because a threshold is a cell in a table rather than a passage to be searched.
The file searched only the chunk corpus and declared four working questions
broken.

So this version asks the run itself first: was this citation actually supplied?
If it was, the question of why retrieval missed it does not arise, whichever
component supplied it. Only genuinely missing citations get a rank analysis.

That is the seventh time in this project that the measuring code was wrong
before the system was.

COST: one embedding call per question. No answers are generated.

Run:  python src/diagnose_retrieval.py
"""

import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from eval_questions import QUESTIONS
from search_vectors import load_index, embed_query, score_all
from search_keyword import build_index as build_bm25, tokenise
from search_hybrid import fuse, DEPTH
from search_split import (pool_indices, LAW_SLOTS, GUIDANCE_SLOTS, LAW_TYPES,
                          short)

RUN_PATH = Path("eval/stage5_run.json")

# A chunk this near the top by meaning was not "missed" by the search. It was
# found and then out-voted. Three is deliberately strict: it means the vector
# ranking put it inside what a three-slot quota would have taken on its own.
STRONG_VECTOR = 3


def full_rank(scores, indices):
    """Position of every chunk in the pool, with no depth cut."""
    order = sorted(indices, key=lambda i: scores[i], reverse=True)
    return {i: r for r, i in enumerate(order, start=1)}


def pool_ranking(indices, vector_scores, keyword_scores):
    """
    Reproduce exactly what search_pool does, but keep the working out.

    The depth cut matters and is easy to miss: a chunk outside the top DEPTH
    of BOTH rankings gets no fused score at all, so it cannot be selected no
    matter how many slots are available.
    """
    sub_vector = {i: vector_scores[i] for i in indices}
    sub_keyword = {i: keyword_scores[i] for i in indices}

    def capped(scores):
        order = sorted(scores, key=lambda i: scores[i], reverse=True)
        return {i: r for r, i in enumerate(order[:DEPTH], start=1)}

    fused = fuse([capped(sub_vector), capped(sub_keyword)])
    order = sorted(fused, key=lambda i: fused[i], reverse=True)
    return {
        "vector": full_rank(vector_scores, indices),
        "keyword": full_rank(keyword_scores, indices),
        "fused": {i: r for r, i in enumerate(order, start=1)},
        "order": order,
    }


def verdict_for(best_fused, best_vector, slots):
    """
    Turn a rank into an instruction.

    The order of these tests is the argument. A chunk the vector search ranked
    first has been FOUND; whatever went wrong happened afterwards, when the
    two rankings were combined. Calling that a quota problem would send the
    repair to the wrong place.
    """
    if best_fused is None:
        return "NEVER A CANDIDATE", (
            f"outside the top {DEPTH} of both rankings, so no number of slots "
            f"would reach it. The ranking itself has to change.")
    if best_fused <= slots:
        return "SUPPLIED", "selected by retrieval"
    if best_vector is not None and best_vector <= STRONG_VECTOR:
        return "FUSION PENALTY", (
            f"the vector search ranked it {best_vector}, so it was FOUND. "
            f"Fusing with the keyword ranking moved it to {best_fused}. RRF "
            f"rewards agreement between the two searches, and the keyword "
            f"search cannot tell this provision from any other that uses the "
            f"same common words.")
    if best_fused <= slots + 3:
        return "NEAR MISS", (
            f"fused rank {best_fused} against {slots} slots. Widening the "
            f"quota would have supplied it.")
    return "RANKED LOW", (
        f"fused rank {best_fused}, and the vector search did not rank it "
        f"highly either. Neither search found it.")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")
    if not RUN_PATH.exists():
        raise SystemExit(f"{RUN_PATH} not found. Run src/evaluate.py first.")

    run = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    results = {r["id"]: r for r in run["results"]}

    client = genai.Client(api_key=key)
    chunks, matrix, index_meta = load_index()
    bm25, _ = build_bm25(chunks)
    law_pool = pool_indices(chunks, law=True)
    guidance_pool = pool_indices(chunks, law=False)

    print(f"  law pool {len(law_pool)} chunks, guidance pool "
          f"{len(guidance_pool)}, quota {LAW_SLOTS} law + {GUIDANCE_SLOTS} "
          f"guidance, fusion depth {DEPTH}\n")

    tally = Counter()
    detail = []
    needed_slots = []          # (id, citation, fused rank) for the near misses

    for spec in QUESTIONS:
        result = results.get(spec["id"])
        if not result:
            continue
        supplied_citations = " ".join(
            s["citation"] for s in result["sources"]).lower()

        for target in spec["must_cite"]:
            # ASK THE RUN FIRST. If the citation was supplied, retrieval did
            # not fail, whichever component supplied it. Schedule 1 arrives
            # from the table lookup and is never retrieved at all.
            if target.lower() in supplied_citations:
                from_chunks = any(target.lower() in c["citation"].lower()
                                  for c in chunks)
                label = "SUPPLIED" if from_chunks else "SUPPLIED (table lookup)"
                tally[label] += 1
                detail.append((spec["id"], target, label, ""))
                continue

            candidates = [i for i in law_pool
                          if target.lower() in chunks[i]["citation"].lower()]
            in_law_pool = bool(candidates)
            if not in_law_pool:
                candidates = [i for i in guidance_pool
                              if target.lower() in chunks[i]["citation"].lower()]
            if not candidates:
                tally["NOT IN CORPUS"] += 1
                detail.append((spec["id"], target, "NOT IN CORPUS",
                               "no chunk carries this citation and the lookup "
                               "did not supply it either"))
                continue

            vector_scores = score_all(
                matrix, embed_query(client, index_meta, spec["question"]))
            keyword_scores = bm25.get_scores(tokenise(spec["question"]))
            ranking = pool_ranking(law_pool if in_law_pool else guidance_pool,
                                   vector_scores, keyword_scores)
            slots = LAW_SLOTS if in_law_pool else GUIDANCE_SLOTS
            selected = ranking["order"][:slots]

            ranked_candidates = sorted(
                candidates, key=lambda i: ranking["fused"].get(i, 10 ** 6))
            best = ranked_candidates[0]
            best_fused = ranking["fused"].get(best)
            best_vector = ranking["vector"].get(best)
            label, why = verdict_for(best_fused, best_vector, slots)
            tally[label] += 1
            detail.append((spec["id"], target, label, why))
            if best_fused:
                needed_slots.append((spec["id"], target, best_fused, label))

            print("=" * 100)
            print(f"{spec['id']}   required citation: {target}   NOT SUPPLIED")
            print(f"  {spec['question'][:92]}")
            print("=" * 100)
            print(f"  {len(candidates)} chunk(s) carry this citation, in the "
                  f"{'law' if in_law_pool else 'guidance'} pool")
            print(f"  {'citation':<34}{'vector':>8}{'keyword':>9}{'fused':>7}")
            print("  " + "-" * 96)
            for i in ranked_candidates[:6]:
                f = ranking["fused"].get(i)
                print(f"  {short(chunks[i]['citation'], 32):<34}"
                      f"{ranking['vector'].get(i, '-'):>8}"
                      f"{ranking['keyword'].get(i, '-'):>9}"
                      f"{(f if f else 'none'):>7}")
            print(f"\n  VERDICT: {label}")
            print(f"  {why}")
            print(f"\n  what took the {slots} slots instead:")
            for rank, i in enumerate(selected, start=1):
                print(f"    {rank}. {short(chunks[i]['citation'], 70)}")
            print()

    # -----------------------------------------------------------------------
    print("=" * 100)
    print("EVERY REQUIRED CITATION, ACROSS ALL 22 QUESTIONS")
    print("=" * 100)
    print(f"  {'id':<6}{'required citation':<24}{'verdict':<26}")
    print("  " + "-" * 96)
    for qid, target, label, _ in detail:
        mark = "" if label.startswith("SUPPLIED") else "  <--"
        print(f"  {qid:<6}{target[:22]:<24}{label:<26}{mark}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("WHAT THIS DECIDES")
    print("=" * 100)
    for label, n in tally.most_common():
        print(f"  {label:<26}{n}")

    quota_fixable = [x for x in needed_slots if x[3] == "NEAR MISS"]
    fusion_cases = [x for x in needed_slots if x[3] == "FUSION PENALTY"]

    if quota_fixable:
        worst = max(x[2] for x in quota_fixable)
        print(f"\n  A QUOTA CHANGE WOULD FIX {len(quota_fixable)} OF THEM")
        for qid, target, rank, _ in quota_fixable:
            print(f"    {qid} {target:<14} sits at fused rank {rank}, so "
                  f"LAW_SLOTS >= {rank} supplies it")
        print(f"    LAW_SLOTS = {worst} supplies all of them.")
        print(f"    Cost: every question then receives {worst} law chunks instead")
        print(f"    of {LAW_SLOTS}. More evidence is not free. C1 already drops the")
        print(f"    end of a long list and S4 already runs out of room.")

    if fusion_cases:
        print(f"\n  A QUOTA CHANGE WOULD NOT FIX {len(fusion_cases)} OF THEM")
        for qid, target, rank, _ in fusion_cases:
            print(f"    {qid} {target:<14} fused rank {rank}, but vector rank "
                  f"is at the very top")
        print("    These were found by the search and then out-voted by fusion.")
        print("    The defensible fix is a policy, not a multiplier: the single")
        print("    best chunk by meaning always gets a seat. That can be written")
        print("    down and argued with. 'Multiply the vector score by 1.4'")
        print("    cannot.")

    print("\n  Whichever is done, it is ONE change followed by a full re-run of")
    print("  all 22 questions. A retrieval change moves the evidence for every")
    print("  question, including the seventeen that currently pass.")
