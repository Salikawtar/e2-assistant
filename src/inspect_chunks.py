"""
Look at what is actually in chunks.jsonl.

Run this any time you want to check the chunks, rather than trusting
whatever the parser happened to print.
"""

import json
import re
from collections import Counter
from pathlib import Path

CHUNKS_PATH = Path("data/chunks.jsonl")

chunks = [json.loads(line) for line in CHUNKS_PATH.open(encoding="utf-8")]

print(f"Total chunks: {len(chunks)}\n")

# --- every citation, in order ---
print("EVERY CHUNK")
print("-" * 70)
for i, c in enumerate(chunks, 1):
    print(f"  {i:>3}. {c['citation']:<26} {c['marginal_note'][:30]:<32} {c['chars']:>5}")

# --- coverage ---
print("\nCOVERAGE CHECK")
print("-" * 70)
sections = sorted(
    {int(m.group(1)) for c in chunks
     if (m := re.match(r"section (\d+)", c["citation"]))}
)
schedules = sorted(
    {int(m.group(1)) for c in chunks
     if (m := re.match(r"Schedule (\d+)", c["citation"]))}
)
print(f"  sections found : {len(sections)}  -> {sections}")
print(f"  missing 1-24   : {[n for n in range(1, 25) if n not in sections] or 'none'}")
print(f"  schedules found: {schedules}   (expect 2-8, Schedule 1 is a table)")

# --- size warnings ---
print("\nSIZE CHECK")
print("-" * 70)
big = [c for c in chunks if c["chars"] > 2000]
tiny = [c for c in chunks if c["chars"] < 100]
print(f"  over 2000 chars: {len(big)}")
for c in big:
    print(f"     {c['citation']:<24} {c['chars']:,} chars")
print(f"  under 100 chars: {len(tiny)}")
for c in tiny:
    print(f"     {c['citation']:<24} {c['chars']} chars — {c['text'][:50]}")

# --- duplicates ---
dupes = [k for k, v in Counter(c["chunk_id"] for c in chunks).items() if v > 1]
print(f"\n  duplicate ids  : {dupes or 'none'}")