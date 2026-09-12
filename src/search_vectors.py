"""
Step 3 of Stage 3 — search the index.  YOU write the scoring.

The whole of vector search is: turn the question into a vector, then find the
chunks whose vectors point in the most similar direction.

There is one function below with its body missing. Write it. Everything else
is plumbing, and there is a check at the bottom that will tell you honestly
whether your version is right.

Run:  python src/search_vectors.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

CHUNKS_PATH = Path("data/chunks.jsonl")
VECTOR_DIR = Path("data/vectors")

QUESTIONS = [
    "How often must a facility conduct a simulation exercise?",
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
]


# ===========================================================================
#  YOUR PART
# ===========================================================================

def score_all(matrix, query_vector):
    """
    Score every chunk against the question. Return one number per chunk.

    matrix        : shape (362, 768) - one row per chunk
    query_vector  : shape (768,)     - the question
    returns       : shape (362,)     - one similarity per chunk

    Cosine similarity is  dot(a, b) / (length(a) * length(b)).
    Every vector here already has length 1.0, so both divisors are 1 and the
    formula collapses to just the dot product.

    NumPy can do all 362 dot products in one operation. The '@' symbol means
    matrix multiplication:  matrix @ query_vector

    Write the one line that returns the scores.
    """
    return matrix @ query_vector


# ===========================================================================
#  The slow, obviously-correct version. Used only to check your work.
# ===========================================================================

def score_all_slow(matrix, query_vector):
    """
    The same thing written out longhand: one chunk at a time, full formula,
    no shortcuts. Nobody would ship this - it is here so your fast version
    has something honest to be compared against.
    """
    scores = []
    for row in matrix:
        top = float(np.dot(row, query_vector))
        bottom = float(np.linalg.norm(row) * np.linalg.norm(query_vector))
        scores.append(top / bottom)
    return np.array(scores, dtype=np.float32)


# ===========================================================================
#  Plumbing
# ===========================================================================

def load_index():
    """Load the vectors, and refuse to use them if they are stale."""
    import hashlib

    index = json.loads((VECTOR_DIR / "index.json").read_text(encoding="utf-8"))
    matrix = np.load(VECTOR_DIR / "vectors.npy")

    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle]

    current_sha = hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()
    if current_sha != index["source_sha256"]:
        raise SystemExit(
            "chunks.jsonl has changed since the index was built.\n"
            "The vectors describe different text than the chunks do.\n"
            "Run: python src/embed_corpus.py"
        )
    if len(chunks) != len(matrix):
        raise SystemExit(f"{len(chunks)} chunks but {len(matrix)} vectors")

    return chunks, matrix, index


# Errors that mean "try again", as opposed to "this will never work". The same
# list generate_naive.py uses, for the same reason.
TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")
EMBED_RETRIES = 5


def embed_query(client, index, text):
    """
    Turn a question into a vector, and survive a busy minute while doing it.

    This used to call the API once with no protection at all. Every GENERATION
    call goes through ModelPool, which retries; the embedding call did not, so
    a single 429 on the query vector crashed a whole 22 question run at
    question nine. The generation side was armoured and the retrieval side was
    bare, which nobody noticed until the run got fast enough to hit the
    embedding rate limit.

    Embeddings are rate limited PER MINUTE, and this is one small request, so
    waiting is almost always enough.
    """
    for attempt in range(1, EMBED_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=index["model"],
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=index["dimensions"]),
            )
            return np.array(response.embeddings[0].values, dtype=np.float32)
        except Exception as error:                       # noqa: BLE001
            message = str(error)
            if not any(t in message for t in TRANSIENT):
                raise
            if attempt == EMBED_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"    embedding: busy, retrying in {wait}s "
                  f"[{attempt}/{EMBED_RETRIES}]")
            time.sleep(wait)


def search(chunks, matrix, query_vector, k=5):
    """Score everything, then take the k best."""
    scores = score_all(matrix, query_vector)
    best = np.argsort(scores)[::-1][:k]        # argsort is ascending, so reverse
    return [(chunks[i], float(scores[i])) for i in best]


def show(question, results):
    print("\n" + "=" * 92)
    print(f"Q: {question}")
    print("=" * 92)
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"  {rank}. {score:.3f}  [{chunk['source_type']:<10}] {chunk['citation'][:62]}")
        print(f"          {chunk['text'][:150].strip()}...")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    chunks, matrix, index = load_index()
    print(f"Index loaded: {len(chunks)} chunks, {index['dimensions']} dimensions, "
          f"built {index['built_at']}")

    client = genai.Client(api_key=key)

    # -- first, prove your scoring is correct -------------------------------
    probe = embed_query(client, index, QUESTIONS[0])
    fast = score_all(matrix, probe)
    slow = score_all_slow(matrix, probe)

    print("\n" + "=" * 92)
    print("CHECK — does your fast version agree with the longhand one?")
    print("=" * 92)
    print(f"  shape returned      : {fast.shape}   (should be ({len(matrix)},))")
    print(f"  largest difference  : {np.abs(fast - slow).max():.8f}")
    print(f"  scores in range -1 to 1 : {bool(fast.min() >= -1.001 and fast.max() <= 1.001)}")
    if np.abs(fast - slow).max() > 1e-4:
        raise SystemExit("  Your version disagrees with the longhand one. Stop here.")
    print("  Agreed. Your one line does the same thing as ten, about 100x faster.")

    # -- then use it --------------------------------------------------------
    for question in QUESTIONS:
        vector = embed_query(client, index, question)
        show(question, search(chunks, matrix, vector, k=5))

    print("\n" + "=" * 92)
    print("READ THE RESULTS, DO NOT JUST ADMIRE THEM")
    print("=" * 92)
    print("  Q1  Is section 7 in the top five, or only guidance about section 7?")
    print("  Q2  Does anything actually give the 4.50 tonne figure? It lives in")
    print("      schedule1.csv, which is NOT in this index - so it cannot.")
    print("  Q3  This question has no answer in the corpus. Look at the top score")
    print("      anyway. A high score on an unanswerable question is the reason")
    print("      abstention cannot be decided by a score threshold.")
