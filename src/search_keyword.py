"""
Step 4 of Stage 3 — keyword search, the half that vectors cannot do.

Vector search matched "Schedule 5" to "Schedule 4" at 0.902 because the two
sentences MEAN nearly the same thing. In a regulation the digit is the answer,
so meaning alone is not enough.

BM25 ranks by the words actually present. Three ideas, and that is the whole
algorithm:
  - a chunk scores higher the more query words it contains
  - a RARE word is worth far more than a common one ("ammonia" beats "the")
  - a long chunk is penalised, so it cannot win just by being long

No model, no API call. It runs on your machine in milliseconds.

Run:  python src/search_keyword.py
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/chunks.jsonl")

# A word, or a number, or either joined by hyphens. This is what keeps
# "7664-41-7" as ONE token instead of three meaningless numbers.
WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Legal citations: 3(2), 4(2)(k), 7(1)(b). The plain word rule would smash
# these into "3", "2" — losing exactly the part that identifies the provision.
CITATION = re.compile(r"\b(\d+(?:\.\d+)?)\((\d+)\)(?:\(([a-z])\))?")

QUESTIONS = [
    "How often must a facility conduct a simulation exercise?",
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
]

# Queries a person types when they already know what they are looking for.
# These are the ones vector search handles badly.
LITERAL_QUERIES = [
    "Schedule 5",
    "7664-41-7",
    "section 4(2)(k)",
]


def tokenise(text):
    """
    Turn text into the list of words BM25 will count.

    Tokenising is a DESIGN DECISION, not a detail. The default in most
    tutorials is text.lower().split(), which turns "7664-41-7" into
    "7664-41-7" only by luck and breaks "4(2)(k)" completely.
    """
    text = text.lower()
    tokens = WORD.findall(text)

    # Add the citation back as a single searchable token, in a normal form.
    for section, subsection, paragraph in CITATION.findall(text):
        tokens.append(f"s{section}({subsection})")
        if paragraph:
            tokens.append(f"s{section}({subsection})({paragraph})")

    return tokens


def searchable_text(chunk):
    """
    The same fields we embedded in Step 2.

    Both searches must see the same content, or comparing them means nothing.
    """
    parts = [chunk.get("heading", ""), chunk.get("marginal_note", ""),
             chunk.get("citation", ""), chunk["text"]]
    return " ".join(p for p in parts if p)


def build_index(chunks):
    corpus = [tokenise(searchable_text(c)) for c in chunks]
    return BM25Okapi(corpus), corpus


def search(chunks, bm25, query, k=5):
    scores = bm25.get_scores(tokenise(query))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], float(scores[i])) for i in best]


def show(query, results):
    print("\n" + "=" * 92)
    print(f"Q: {query}")
    print("=" * 92)
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"  {rank}. {score:6.2f}  [{chunk['source_type']:<10}] {chunk['citation'][:60]}")
        print(f"          {chunk['text'][:140].strip()}...")


if __name__ == "__main__":
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle]

    bm25, corpus = build_index(chunks)

    print(f"Chunks indexed : {len(chunks)}")
    print(f"Total tokens   : {sum(len(t) for t in corpus):,}")
    print(f"Vocabulary     : {len({t for doc in corpus for t in doc}):,} distinct tokens")
    print("\nNo API call was made. This index was built locally, from the text.")

    print("\n" + "=" * 92)
    print("HOW TOKENISING CHANGES WHAT IS FINDABLE")
    print("=" * 92)
    for example in ["Ammonia, anhydrous CAS 7664-41-7",
                    "paragraph 4(2)(k) of the Regulations",
                    "Schedule 5 — Notice Regarding Simulation Exercises"]:
        print(f"\n  {example}")
        print(f"    naive  .lower().split() : {example.lower().split()}")
        print(f"    ours   tokenise()       : {tokenise(example)}")

    print("\n" + "=" * 92)
    print("PART 1 — the literal queries, where vectors were weak")
    print("=" * 92)
    for query in LITERAL_QUERIES:
        show(query, search(chunks, bm25, query, k=3))

    print("\n" + "=" * 92)
    print("PART 2 — the same three evaluation questions as Step 3")
    print("=" * 92)
    for question in QUESTIONS:
        show(question, search(chunks, bm25, question, k=5))

    print("\n" + "=" * 92)
    print("WHAT TO COMPARE AGAINST STEP 3")
    print("=" * 92)
    print("  Scores are NOT comparable between the two searches. Cosine runs 0 to 1;")
    print("  BM25 has no upper bound and depends on the corpus. Compare the RANKINGS,")
    print("  never the numbers - which is exactly the problem Step 5 has to solve.\n")
    print("  Q1  Did section 7 itself rank higher here than it did in Step 3?")
    print("  Q2  Anything about the ammonia THRESHOLD, or only exclusions again?")
    print("  Q3  Look at the top score on the unanswerable question. BM25 will")
    print("      still return five chunks. No search refuses to answer.")
