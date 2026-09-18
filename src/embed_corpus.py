"""
Step 2 of Stage 3 — turn every chunk into a vector, once, and keep it.

Embedding costs time and an API call. Doing it on every search would be slow
and pointless: the corpus does not change between questions. So we embed once
and save the result to disk. That saved result is the INDEX.

An index is only usable if it says what made it. Vectors from two different
models sit in two different spaces and cannot be compared, so the model id,
the number of dimensions and a fingerprint of the source file are written
beside the numbers.

Run:  python src/embed_corpus.py
"""

import hashlib
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
VECTORS_PATH = VECTOR_DIR / "vectors.npy"
INDEX_PATH = VECTOR_DIR / "index.json"
PROGRESS_PATH = VECTOR_DIR / "progress.json"

MODEL = "gemini-embedding-2"
DIMENSIONS = 768

# The free tier limits how fast requests may arrive. A short pause between
# calls is cheaper than being refused and retrying.
PAUSE_SECONDS = 0.15
MAX_RETRIES = 5
SAVE_EVERY = 25          # write progress to disk this often, so a failure costs little


# ---------------------------------------------------------------------------
# What text actually gets embedded
# ---------------------------------------------------------------------------

def text_to_embed(chunk):
    """
    Heading + marginal note + text.

    The marginal note is where the regulation puts its own short title for a
    provision - "Simulation exercise", "Determination of quantity". Those are
    the exact words a person searches with, and they are NOT inside the text
    field. Embedding the text alone would make them unfindable.
    """
    parts = [
        chunk.get("heading", ""),
        chunk.get("marginal_note", ""),
        chunk.get("citation", ""),
        chunk["text"],
    ]
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Talking to the API
# ---------------------------------------------------------------------------

def embed_one(client, text):
    """
    One text, one vector, with retries.

    A rate-limit refusal is not a bug, it is the free tier working as designed.
    Wait longer each time rather than giving up or hammering the service.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=MODEL,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
            )
            return response.embeddings[0].values
        except Exception as error:            # noqa: BLE001 - we want every failure
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"\n    request failed ({type(error).__name__}), "
                  f"retrying in {wait}s  [attempt {attempt}/{MAX_RETRIES}]")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Reading, resuming, saving
# ---------------------------------------------------------------------------

def read_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def fingerprint(path):
    """SHA-256 of the source file, so a stale index can be detected."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_progress(chunk_ids):
    """
    Resume only if the saved work matches the start of what we are about to do.
    If chunks.jsonl has been rebuilt, the saved vectors belong to different
    text and must be thrown away.
    """
    if not (PROGRESS_PATH.exists() and VECTORS_PATH.exists()):
        return [], None
    saved = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    if saved.get("model") != MODEL or saved.get("dimensions") != DIMENSIONS:
        return [], None
    done_ids = saved.get("chunk_ids", [])
    if done_ids != chunk_ids[:len(done_ids)]:
        return [], None
    vectors = np.load(VECTORS_PATH)
    if len(vectors) != len(done_ids):
        return [], None
    return done_ids, vectors


def save_progress(done_ids, vectors):
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH, np.array(vectors, dtype=np.float32))
    PROGRESS_PATH.write_text(json.dumps({
        "model": MODEL, "dimensions": DIMENSIONS, "chunk_ids": done_ids,
    }), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")
    print(f"Key loaded: {key[:6]}...{key[-4:]}")

    chunks = read_chunks()
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [text_to_embed(c) for c in chunks]

    print(f"\nChunks to embed : {len(chunks)}")
    print(f"Model           : {MODEL} at {DIMENSIONS} dimensions")
    print(f"\nWhat gets embedded, for one chunk:")
    print(f"  chunk_id : {chunks[0]['chunk_id']}")
    print(f"  embedded : {texts[0][:150]}...")

    done_ids, existing = load_progress(chunk_ids)
    vectors = list(existing) if existing is not None else []
    if done_ids:
        print(f"\nResuming: {len(done_ids)} vectors already on disk.")

    client = genai.Client(api_key=key)
    started = time.time()

    for i in range(len(done_ids), len(chunks)):
        vectors.append(embed_one(client, texts[i]))
        done_ids.append(chunk_ids[i])

        if (i + 1) % 10 == 0 or i + 1 == len(chunks):
            elapsed = time.time() - started
            print(f"\r  embedded {i + 1}/{len(chunks)}   {elapsed:.0f}s", end="")
        if (i + 1) % SAVE_EVERY == 0:
            save_progress(done_ids, vectors)
        time.sleep(PAUSE_SECONDS)

    print()
    save_progress(done_ids, vectors)

    matrix = np.array(vectors, dtype=np.float32)
    INDEX_PATH.write_text(json.dumps({
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "count": len(matrix),
        "embedded_fields": "heading + marginal_note + citation + text",
        "source_file": str(CHUNKS_PATH),
        "source_sha256": fingerprint(CHUNKS_PATH),
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "chunk_ids": chunk_ids,
    }, indent=2), encoding="utf-8", )

    # -----------------------------------------------------------------------
    # Checks. An index that is quietly wrong is worse than no index.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CHECK 1 — one vector per chunk, in the same order")
    print("=" * 78)
    print(f"  chunks              : {len(chunks)}")
    print(f"  vectors             : {len(matrix)}")
    print(f"  ids line up         : {done_ids == chunk_ids}")
    print(f"  shape               : {matrix.shape}")

    print("\n" + "=" * 78)
    print("CHECK 2 — is any vector empty or broken?")
    print("=" * 78)
    lengths = np.linalg.norm(matrix, axis=1)
    print(f"  all-zero vectors    : {int((lengths == 0).sum())}")
    print(f"  vectors with NaN    : {int(np.isnan(matrix).any(axis=1).sum())}")
    print(f"  length min / max    : {lengths.min():.4f} / {lengths.max():.4f}")
    print("  (all 1.0000 means the vectors are normalised, as Step 1 showed)")

    print("\n" + "=" * 78)
    print("CHECK 3 — does every chunk find ITSELF first?")
    print("=" * 78)
    print("  Search the index using a chunk's own vector. The nearest result")
    print("  must be that same chunk. If it is not, the vectors and the chunks")
    print("  are misaligned - the failure that would be invisible otherwise.\n")
    sample = list(range(0, len(matrix), max(1, len(matrix) // 20)))
    wrong = []
    for i in sample:
        nearest = int(np.argmax(matrix @ matrix[i]))
        if nearest != i:
            wrong.append((i, nearest))
    print(f"  sampled             : {len(sample)}")
    print(f"  found themselves    : {len(sample) - len(wrong)}")
    for i, nearest in wrong[:5]:
        print(f"    {chunk_ids[i]} matched {chunk_ids[nearest]}")

    print("\n" + "=" * 78)
    print("WRITTEN")
    print("=" * 78)
    print(f"  {VECTORS_PATH}   {VECTORS_PATH.stat().st_size / 1024:.0f} KB")
    print(f"  {INDEX_PATH}")
    print("\n  data/vectors/ is in .gitignore - the index is rebuildable output,")
    print("  not source. Anyone with chunks.jsonl and the model id can remake it.")
