"""
Step 5 of Stage 3 — combine the two searches with Reciprocal Rank Fusion.

Vector search scored section 7 at 0.821. BM25 scored it at 17.58. Those two
numbers cannot be added, averaged or compared: one is bounded at 1.0, the
other has no ceiling at all.

RRF avoids the problem by ignoring the scores completely and using only the
POSITION each search put a chunk in:

    score(chunk) = 1/(K + rank_vector) + 1/(K + rank_bm25)

A chunk found by both searches beats a chunk found brilliantly by one. That is
exactly the property we want, because you have now watched each search fail in
a way the other did not.

This file reuses YOUR score_all() from search_vectors.py.

Run:  python src/search_hybrid.py
"""

import os

from dotenv import load_dotenv
from google import genai

from search_vectors import load_index, embed_query, score_all
from search_keyword import build_index as build_bm25, tokenise

# The damper. 60 is the value from the original RRF paper and the usual
# default. Without it, rank 1 would be worth twice rank 2 and the first
# result would dominate everything.
K = 60

# How deep to look in each list before fusing. Too shallow and a chunk that
# one search ranked 30th can never be rescued by the other.
DEPTH = 30

QUESTIONS = [
    "How often must a facility conduct a simulation exercise?",
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
    "Schedule 5",
]


def ranked(scores, depth):
    """Turn a list of scores into {chunk index: rank}, rank 1 being best."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return {index: rank for rank, index in enumerate(order[:depth], start=1)}


def fuse(rank_lists):
    """
    Reciprocal Rank Fusion.

    Every list contributes 1/(K + rank) for the chunks it ranked. A chunk
    missing from a list simply contributes nothing from that list - no
    penalty, no invented score.
    """
    fused = {}
    for ranks in rank_lists:
        for index, rank in ranks.items():
            fused[index] = fused.get(index, 0.0) + 1.0 / (K + rank)
    return fused


def search(chunks, matrix, bm25, client, index_meta, question, k=5):
    vector_scores = score_all(matrix, embed_query(client, index_meta, question))
    keyword_scores = bm25.get_scores(tokenise(question))

    vector_ranks = ranked(vector_scores, DEPTH)
    keyword_ranks = ranked(keyword_scores, DEPTH)
    fused = fuse([vector_ranks, keyword_ranks])

    best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
    return [{
        "chunk": chunks[i],
        "rrf": fused[i],
        "vector_rank": vector_ranks.get(i),
        "keyword_rank": keyword_ranks.get(i),
    } for i in best]


def dash(value):
    return "-" if value is None else str(value)


def show(question, results):
    print("\n" + "=" * 96)
    print(f"Q: {question}")
    print("=" * 96)
    print(f"  {'#':<3} {'RRF':<7} {'vec':<5} {'bm25':<6} {'type':<11} citation")
    print("  " + "-" * 92)
    for rank, r in enumerate(results, start=1):
        chunk = r["chunk"]
        print(f"  {rank:<3} {r['rrf']:.4f}  {dash(r['vector_rank']):<5} "
              f"{dash(r['keyword_rank']):<6} {chunk['source_type']:<11} "
              f"{chunk['citation'][:52]}")
    print()
    for rank, r in enumerate(results[:3], start=1):
        print(f"    {rank}. {r['chunk']['text'][:150].strip()}...")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    chunks, matrix, index_meta = load_index()
    bm25, _ = build_bm25(chunks)
    client = genai.Client(api_key=key)

    print(f"Chunks: {len(chunks)}   RRF K={K}   fusing top {DEPTH} of each search")
    print("\n  vec  = position in the vector ranking, '-' means not in the top 30")
    print("  bm25 = position in the keyword ranking")

    all_results = {}
    for question in QUESTIONS:
        results = search(chunks, matrix, bm25, client, index_meta, question)
        all_results[question] = results
        show(question, results)

    print("\n" + "=" * 96)
    print("HOW MUCH LAW IS IN THE TOP 5?")
    print("=" * 96)
    print("  Law is 30% of the corpus by chunk count. Anything well below that")
    print("  means guidance is crowding the regulation out.\n")
    LAW = {"regulation", "act"}
    total_law = 0
    for question, results in all_results.items():
        law = sum(1 for r in results if r["chunk"]["source_type"] in LAW)
        total_law += law
        print(f"    {law}/5   {question[:76]}")
    print(f"\n  overall: {total_law}/{5 * len(QUESTIONS)} "
          f"= {total_law / (5 * len(QUESTIONS)):.0%} law")

    print("\n" + "=" * 96)
    print("WHAT TO LOOK AT")
    print("=" * 96)
    print("  1. Find a row where one column says '-'. That chunk was invisible to")
    print("     one search entirely and was carried into the top 5 by the other.")
    print("  2. Q3 is still unanswerable. Fusion does not fix that and never will")
    print("     - it reorders evidence, it does not judge it.")
    print("  3. 'Schedule 5': did fusion put the real Schedule 5 above Schedule 8?")
    print("  4. If the law share is still under 30%, Step 6 is the fix: search")
    print("     law and guidance separately so they stop competing.")
