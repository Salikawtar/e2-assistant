"""
Step 1 of Stage 3 — look at ONE embedding before trusting millions of them.

An embedding turns text into a list of numbers: a position in a space of
meaning. Nothing here is saved. The only purpose is to see, with your own
eyes, what an embedding is and what it can and cannot tell apart.

Run:  python src/embed_demo.py
"""

import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# The model id and the number of dimensions are DECISIONS, not details.
# Vectors made by different models cannot be compared with each other, so
# these two values have to travel with any vector we ever save.
MODEL = "gemini-embedding-2"
DIMENSIONS = 768

# Four short texts from your own corpus. Two are questions, two are answers.
SAMPLES = [
    ("Q-exercise", "How often must a facility conduct a simulation exercise?"),
    ("A-section7", "A responsible person must conduct a simulation exercise of the "
                   "environmental emergency plan at least once each calendar year."),
    ("Q-ammonia", "What is the minimum quantity threshold for anhydrous ammonia?"),
    ("A-schedule1", "Ammonia, anhydrous. CAS 7664-41-7. Minimum concentration 10 per "
                    "cent by mass. Minimum quantity 4.50 tonnes. Inhalation hazard."),
]


def embed(client, texts):
    """
    Send text to Gemini, get one vector back per text.

    One request per text, on purpose. Passing the whole list in a single call
    returned ONE vector for four texts, which would have silently lined every
    chunk up against the wrong vector. The count check below is what caught it,
    and it stays in the code for the same reason.
    """
    vectors = []
    for text in texts:
        response = client.models.embed_content(
            model=MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
        )
        vectors.append(response.embeddings[0].values)

    if len(vectors) != len(texts):
        raise SystemExit(f"asked for {len(texts)} vectors, got {len(vectors)}")

    return np.array(vectors, dtype=np.float32)


def cosine(a, b):
    """
    Cosine similarity: how closely two vectors POINT in the same direction.

    Not how far apart they are — direction only. 1.0 means identical
    direction, 0.0 means unrelated, negative means opposite.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")
    print(f"Key loaded: {key[:6]}...{key[-4:]}")      # never print the whole key

    client = genai.Client(api_key=key)

    labels = [label for label, _ in SAMPLES]
    texts = [text for _, text in SAMPLES]
    vectors = embed(client, texts)

    print("\n" + "=" * 78)
    print("1. WHAT AN EMBEDDING ACTUALLY IS")
    print("=" * 78)
    print(f"  model                 : {MODEL}")
    print(f"  texts sent            : {len(texts)}")
    print(f"  numbers per text      : {vectors.shape[1]}")
    print(f"\n  first 6 numbers of '{labels[0]}':")
    print("   ", np.round(vectors[0][:6], 4))
    print("\n  That is the whole sentence, as far as the machine is concerned.")

    print("\n" + "=" * 78)
    print("2. IS THE VECTOR NORMALISED?")
    print("=" * 78)
    print("  A normalised vector has length exactly 1.0. It matters because if")
    print("  every vector has the same length, cosine similarity becomes a plain")
    print("  dot product — much faster over 362 chunks. Measure, do not assume.\n")
    for label, vector in zip(labels, vectors):
        print(f"    {label:<12} length = {np.linalg.norm(vector):.4f}")

    print("\n" + "=" * 78)
    print("3. MEANING, MEASURED")
    print("=" * 78)
    print("  Every pair, scored. Look for two things: the question sitting close")
    print("  to its own answer, and far from the other one's.\n")
    print(" " * 14 + "".join(f"{l:>14}" for l in labels))
    for i, row_label in enumerate(labels):
        scores = "".join(f"{cosine(vectors[i], vectors[j]):>14.3f}"
                         for j in range(len(labels)))
        print(f"  {row_label:<12}{scores}")

    print("\n  The two pairs that matter:")
    print(f"    exercise question  <-> section 7 text   : "
          f"{cosine(vectors[0], vectors[1]):.3f}")
    print(f"    exercise question  <-> ammonia row      : "
          f"{cosine(vectors[0], vectors[3]):.3f}")
    print(f"    ammonia question   <-> ammonia row      : "
          f"{cosine(vectors[2], vectors[3]):.3f}")
    print(f"    ammonia question   <-> section 7 text   : "
          f"{cosine(vectors[2], vectors[1]):.3f}")

    print("\n" + "=" * 78)
    print("4. WHERE THIS BREAKS")
    print("=" * 78)
    print("  Two texts that differ only in a number or an identifier:")
    tricky = [
        "Schedule 5 — Notice Regarding Simulation Exercises",
        "Schedule 4 — Notice Regarding the Bringing into Effect of a Plan",
    ]
    tricky_vectors = embed(client, tricky)
    print(f"\n    '{tricky[0][:44]}...'")
    print(f"    '{tricky[1][:44]}...'")
    print(f"\n    similarity = {cosine(tricky_vectors[0], tricky_vectors[1]):.3f}")
    print("\n  Two different legal forms. If that score is high, the embedding")
    print("  cannot reliably tell them apart — which is exactly why Step 4 adds")
    print("  a keyword search alongside this one.")