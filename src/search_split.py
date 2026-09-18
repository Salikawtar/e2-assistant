"""
Step 6 of Stage 3 — search law and guidance separately, then merge.

Stage 2 measured the corpus: guidance outweighs law 6.2 to 1 by volume. Step 5
measured the consequence: on three real questions, only 1 of 15 retrieved
chunks was law. The regulation was not losing on relevance. It was losing on
volume - 110 chunks competing against 252.

The fix is a QUOTA, not a score bonus.

A score bonus ("multiply law by 1.4") is a number nobody can justify. A quota
is a statement of policy: the regulation always gets a seat at the table. That
can be written down, defended, and argued with.

Run:  python src/search_split.py
"""

import os

from dotenv import load_dotenv
from google import genai

from search_vectors import load_index, embed_query, score_all
from search_keyword import build_index as build_bm25, tokenise
from search_hybrid import ranked, fuse, DEPTH

# What counts as law. Everything else is guidance or reference material.
LAW_TYPES = {"regulation", "act"}

# The quota. Three of each, law listed first because it outranks guidance in
# authority - not because it scored better.
LAW_SLOTS = 3
GUIDANCE_SLOTS = 3

# Stage 6 - THE SEAT RESERVED FOR MEANING.
#
# Stage 5 measured three failures, and diagnose_retrieval.py found all three
# had the same cause. The provision that answered the question was ranked 1st
# or 2nd by the vector search, and then lost:
#
#   O5  section 10     vector  1   keyword 20   ->  fused 10   not supplied
#   S4  section 3(1)   vector  1   keyword 21   ->  fused  6   not supplied
#   S1  section 4(1)   vector  2   keyword 10   ->  fused  5   not supplied
#
# That is RRF behaving exactly as designed. It rewards agreement between the
# two searches, so a chunk both methods quite like beats a chunk one method
# loves. Usually that is the right instinct. Here the keyword search cannot
# tell section 10 from any other provision using the words "plan", "review"
# and "environmental emergency", so its opinion is noise, and the fusion was
# averaging a correct answer away.
#
# The fix is a policy, in the same spirit as the quota above:
#
#     THE SINGLE BEST CHUNK BY MEANING ALWAYS GETS A SEAT.
#
# It can be stated, defended and argued with. "Multiply the vector score by
# 1.4" cannot.
#
# Note this costs nothing. LAW_SLOTS is unchanged, so the model receives the
# same amount of evidence. The third fused chunk is what drops out.
#
# ONE SEAT, THEN TWO - AND WHY THE SECOND ONE IS HONEST BUT UNCOMFORTABLE
#
# The first version reserved one seat. It fixed O5 and it stopped S1 refusing
# a question it could answer, both of which held across three repeat runs. It
# did not fix S1's and S4's missing section 4, which sits at VECTOR rank 2 in
# both questions.
#
# Two seats supplies it. The uncomfortable part is that 2 is also, exactly,
# the number that makes the test pass, and choosing a constant because it
# passes a test is how a system gets tuned to its own exam paper.
#
# The argument for it that does not depend on the test: across the four
# provisions that went missing, the keyword search ranked the correct one
# 10th, 20th, 21st and 31st, while the vector search ranked it 1st or 2nd
# every time. For provision-finding questions on THIS corpus, the keyword half
# of the fusion is measurably noise, and giving the informative signal two of
# three seats follows from that measurement.
#
# Both readings are true. The only thing that settles it is the held-out
# questions written after every fix is finished and never used to tune
# anything. Until that runs, this constant is provisional and should be
# described that way.
VECTOR_RESERVE = 2

QUESTIONS = [
    "How often must a facility conduct a simulation exercise?",
    "What must an environmental emergency plan contain?",
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
]


def pool_indices(chunks, law):
    """The row numbers belonging to one pool."""
    return [i for i, c in enumerate(chunks)
            if (c["source_type"] in LAW_TYPES) == law]


def search_pool(indices, vector_scores, keyword_scores, k):
    """
    Rank and fuse WITHIN one pool.

    The ranking is computed over this pool only, so a law chunk competes
    against 109 other law chunks instead of 361 chunks of everything.
    """
    sub_vector = {i: vector_scores[i] for i in indices}
    sub_keyword = {i: keyword_scores[i] for i in indices}

    def rank_map(scores):
        order = sorted(scores, key=lambda i: scores[i], reverse=True)
        return {i: r for r, i in enumerate(order[:DEPTH], start=1)}

    fused = fuse([rank_map(sub_vector), rank_map(sub_keyword)])
    by_fusion = sorted(fused, key=lambda i: fused[i], reverse=True)

    # The reserved seats go to the top of the VECTOR ranking, before fusion
    # has a say. See VECTOR_RESERVE above for why.
    by_meaning = sorted(sub_vector, key=lambda i: sub_vector[i], reverse=True)
    chosen = by_meaning[:min(VECTOR_RESERVE, k)]

    # The rest of the seats are filled by fusion as before, skipping anything
    # the reserve already took so a chunk cannot occupy two seats.
    for i in by_fusion:
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:k]


def flat_search(chunks, vector_scores, keyword_scores, k):
    """Step 5's method, kept here so the two can be compared side by side."""
    fused = fuse([ranked(vector_scores, DEPTH), ranked(keyword_scores, DEPTH)])
    return sorted(fused, key=lambda i: fused[i], reverse=True)[:k]


def short(citation, width=58):
    """
    Keep the START and the END of a citation.

    Chopping the tail off "Technical Guidelines ... , s. 7.1 (p. 72)" throws
    away the only part that identifies the provision.
    """
    if len(citation) <= width:
        return citation
    head = width // 3
    tail = width - head - 3
    return citation[:head] + "..." + citation[-tail:]


def show(title, chunks, indices, promoted=()):
    print(f"\n  {title}")
    print("  " + "-" * 90)
    for rank, i in enumerate(indices, start=1):
        chunk = chunks[i]
        mark = "  <- promoted by the quota" if i in promoted else ""
        print(f"    {rank}. [{chunk['source_type']:<10}] "
              f"{short(chunk['citation'])}{mark}")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    chunks, matrix, index_meta = load_index()
    bm25, _ = build_bm25(chunks)
    client = genai.Client(api_key=key)

    law_pool = pool_indices(chunks, law=True)
    guidance_pool = pool_indices(chunks, law=False)
    print(f"Law pool      : {len(law_pool)} chunks")
    print(f"Guidance pool : {len(guidance_pool)} chunks")
    print(f"Quota         : {LAW_SLOTS} law + {GUIDANCE_SLOTS} guidance")

    flat_law_total = 0
    split_law_total = 0

    for question in QUESTIONS:
        vector_scores = score_all(matrix, embed_query(client, index_meta, question))
        keyword_scores = bm25.get_scores(tokenise(question))

        total = LAW_SLOTS + GUIDANCE_SLOTS
        flat = flat_search(chunks, vector_scores, keyword_scores, total)
        law_best = search_pool(law_pool, vector_scores, keyword_scores, LAW_SLOTS)
        guidance_best = search_pool(guidance_pool, vector_scores, keyword_scores,
                                    GUIDANCE_SLOTS)
        merged = law_best + guidance_best
        promoted = {i for i in law_best if i not in flat}

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print("=" * 96)
        show(f"BEFORE - one ranking for everything (Step 5)", chunks, flat)
        show(f"AFTER  - law and guidance ranked separately", chunks, merged, promoted)

        flat_law = sum(1 for i in flat if chunks[i]["source_type"] in LAW_TYPES)
        flat_law_total += flat_law
        split_law_total += LAW_SLOTS
        print(f"\n    law in evidence:  before {flat_law}/{total}   "
              f"after {LAW_SLOTS}/{total}")

    print("\n" + "=" * 96)
    print("THE NUMBER THIS STEP EXISTS TO CHANGE")
    print("=" * 96)
    slots = (LAW_SLOTS + GUIDANCE_SLOTS) * len(QUESTIONS)
    print(f"  law chunks in evidence, before : {flat_law_total}/{slots} "
          f"= {flat_law_total / slots:.0%}")
    print(f"  law chunks in evidence, after  : {split_law_total}/{slots} "
          f"= {split_law_total / slots:.0%}")

    print("\n" + "=" * 96)
    print("READ THE PROMOTED CHUNKS - THIS IS THE HONEST PART")
    print("=" * 96)
    print("  A quota guarantees the regulation a seat. It does NOT guarantee the")
    print("  regulation has anything to say. Look at every line marked 'promoted':")
    print()
    print("    - if it is relevant, the quota rescued law that volume had buried")
    print("    - if it is irrelevant, the quota has put a real provision in front")
    print("      of the model for a question it does not answer, and Stage 4 must")
    print("      be able to ignore it rather than cite it")
    print()
    print("  Both outcomes are worth knowing. Neither is visible from the numbers.")
