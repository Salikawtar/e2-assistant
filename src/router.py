"""
Step 7b of Stage 3 — the router, and the whole pipeline joined up.

A question can be answered three ways, and choosing wrongly is its own kind of
failure:

  LOOKUP    read a cell in Schedule 1. Exact. Cannot paraphrase, cannot be
            "close". Only works for substances, thresholds and identifiers.

  RETRIEVE  search the 362 chunks for the provisions that bear on a question,
            and let Stage 4 write an answer from them.

  BOTH      a substance is named, but the question is about an obligation:
            "we store 6 tonnes of ammonia - what do we have to do?"

The router is plain rules, not a model. Three reasons:
  - it can be read, argued with, and written into the specification
  - it is free and instant
  - when it is wrong you can see exactly which rule fired

Its default is RETRIEVE. Sending a question to the lookup by mistake produces
"NOT IN SCHEDULE 1", which reads like a real answer and is not one.

Run:  python src/router.py
"""

import os
import re

from dotenv import load_dotenv
from google import genai

from lookup_schedule1 import (load as load_schedule1, lookup, describe,
                              CAS_PATTERN, UN_PATTERN, by_name)
from search_vectors import load_index, embed_query, score_all
from search_keyword import build_index as build_bm25, tokenise
from search_split import pool_indices, search_pool, LAW_SLOTS, GUIDANCE_SLOTS, short

# Words that mean "I am asking about a number in the table", rather than
# "I am asking what the rules require of me".
THRESHOLD_WORDS = {
    "threshold", "thresholds", "minimum", "quantity", "quantities",
    "tonne", "tonnes", "concentration", "listed", "list", "schedule",
    "trigger", "triggers", "hazard", "category", "cas", "un",
}

# "6 tonnes of ammonia" - the user is STATING a quantity they hold, not asking
# what the threshold is. That is an obligations question wearing table words.
STATED_QUANTITY = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:tonnes?|t\b|kg|kilograms?|litres?|liters?)",
    re.IGNORECASE)

# Words that mean "tell me what I have to DO", not "tell me a number".
OBLIGATION_WORDS = {
    "must", "need", "required", "require", "requirement", "requirements",
    "obligation", "obligations", "comply", "prepare", "submit", "report",
    "notify", "notice", "plan", "exercise", "steps", "responsibilities",
}

QUESTIONS = [
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "Is 7664-41-7 listed in Schedule 1?",
    "How often must a facility conduct a simulation exercise?",
    "We store 6 tonnes of anhydrous ammonia. What do we have to do?",
    "What must an environmental emergency plan contain?",
    "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
]


def route(question, schedule1):
    """
    Decide the path, and return the REASONS as well as the decision.

    A router that cannot explain itself is untestable: when it sends the wrong
    question down the wrong path, you have nothing to fix.
    """
    reasons = []
    lowered = question.lower()

    if CAS_PATTERN.search(question):
        reasons.append("a CAS registry number appears in the question")
        return "LOOKUP", reasons
    if UN_PATTERN.search(question):
        reasons.append("a UN number appears in the question")
        return "LOOKUP", reasons

    matches, how = by_name(schedule1, question)
    words = set(tokenise(lowered))
    threshold_words = words & THRESHOLD_WORDS

    obligation_words = words & OBLIGATION_WORDS
    stated_quantity = STATED_QUANTITY.search(question)

    if matches:
        reasons.append(f"a Schedule 1 substance is named ({matches[0]['substance_name']})")

        # Check this BEFORE the table words. "6 tonnes" contains a table word
        # but is the user telling us what they hold, not asking for a number.
        if stated_quantity:
            reasons.append(f"the question states a quantity held "
                           f"('{stated_quantity.group().strip()}'), so it asks what "
                           f"follows from it")
            return "BOTH", reasons
        if obligation_words:
            reasons.append(f"the question asks what is required "
                           f"({', '.join(sorted(obligation_words))})")
            return "BOTH", reasons
        if threshold_words:
            reasons.append(f"and the question uses table words: "
                           f"{', '.join(sorted(threshold_words))}")
            return "LOOKUP", reasons
        reasons.append("but no table words, so the question is not about a number")
        return "BOTH", reasons

    if threshold_words:
        reasons.append(f"table words are present ({', '.join(sorted(threshold_words))}) "
                       f"but no Schedule 1 substance is named")
        reasons.append("so the question is about the rules, not a row")
    else:
        reasons.append("no substance and no table words")
    return "RETRIEVE", reasons


def retrieve(chunks, matrix, bm25, client, index_meta, law_pool, guidance_pool,
             question):
    """Step 6's split retrieval, packaged for reuse."""
    vector_scores = score_all(matrix, embed_query(client, index_meta, question))
    keyword_scores = bm25.get_scores(tokenise(question))
    law = search_pool(law_pool, vector_scores, keyword_scores, LAW_SLOTS)
    guidance = search_pool(guidance_pool, vector_scores, keyword_scores,
                           GUIDANCE_SLOTS)
    return [chunks[i] for i in law + guidance]


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    schedule1 = load_schedule1()
    chunks, matrix, index_meta = load_index()
    bm25, _ = build_bm25(chunks)
    client = genai.Client(api_key=key)
    law_pool = pool_indices(chunks, law=True)
    guidance_pool = pool_indices(chunks, law=False)

    print(f"Schedule 1 : {len(schedule1)} substances")
    print(f"Chunks     : {len(chunks)}  ({len(law_pool)} law, {len(guidance_pool)} guidance)")

    for question in QUESTIONS:
        decision, reasons = route(question, schedule1)

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print("=" * 96)
        print(f"  ROUTE: {decision}")
        for reason in reasons:
            print(f"    - {reason}")

        if decision in ("LOOKUP", "BOTH"):
            matches, how = lookup(schedule1, question)
            print(f"\n  SCHEDULE 1 (matched on {how})")
            print("  " + "-" * 92)
            if matches:
                for row in matches[:3]:
                    for line in describe(row).splitlines():
                        print("    " + line)
                    print()
            else:
                print("    No matching row. The substance is not listed.\n")

        if decision in ("RETRIEVE", "BOTH"):
            evidence = retrieve(chunks, matrix, bm25, client, index_meta,
                                law_pool, guidance_pool, question)
            print(f"\n  EVIDENCE FOR STAGE 4 ({len(evidence)} chunks, law first)")
            print("  " + "-" * 92)
            for rank, chunk in enumerate(evidence, start=1):
                print(f"    {rank}. [{chunk['source_type']:<10}] "
                      f"{short(chunk['citation'])}")

    print("\n" + "=" * 96)
    print("WHAT THE ROUTER FIXED, AND WHAT IT DID NOT")
    print("=" * 96)
    print("  FIXED: 'How often must a facility conduct a simulation exercise?' no")
    print("  longer reaches the lookup, so it can no longer be told NOT IN")
    print("  SCHEDULE 1 - which was true, irrelevant, and read like an answer.")
    print()
    print("  FIXED: the ammonia threshold question now returns 4.50 tonnes from a")
    print("  cell, instead of three searches returning exclusions and refrigerant")
    print("  examples. That question has failed every previous step in Stage 3.")
    print()
    print("  NOT FIXED, and not fixable here: nothing in this pipeline judges")
    print("  whether the evidence answers the question. Retrieval always returns")
    print("  six chunks. The Ontario question gets six chunks too. Deciding that")
    print("  evidence is insufficient is Stage 4's job, and it is the hardest")
    print("  part of the whole project.")
