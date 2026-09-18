"""
Assemble every chunk in the corpus into one file: data/chunks.jsonl

The four parsers each know how to read one FORMAT. This script knows the
CORPUS. It calls each parser, joins the result to the provenance already
recorded in corpus_manifest.csv, writes one line per chunk, and then measures
what came out.

The measuring half is the point. A parser that silently returns nothing still
"works". The checks at the bottom are what tell you it didn't.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from parsing import parse_e2_regulation, parse_act_part
from parsing_html import parse_guidance_page
from parsing_pdf import parse_pdf, DOCUMENTS as PDF_DOCUMENTS

DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "corpus_manifest.csv"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
SCHEDULE1_PATH = DATA_DIR / "schedule1.csv"

# A chunk this long is worth looking at; a chunk this short probably has no
# answer in it. Neither is automatically wrong — both are worth knowing about.
TOO_LONG = 2000
TOO_SHORT = 150

# Where each document's parser settings live. The PDF settings are imported
# from parsing_pdf rather than copied, so there is only ever one copy of them.
PDF_SETTINGS = {d["document_id"]: d for d in PDF_DOCUMENTS}

CORPUS = [
    {
        "document_id": "e2_regulation",
        "parser": parse_e2_regulation,
        "expected_at_least": 50,
    },
    {
        "document_id": "cepa_act",
        "parser": parse_act_part,
        "part": "PART 8",
        "citation_prefix": "CEPA 1999, ",
        "expected_at_least": 40,
    },
    {
        "document_id": "simulation_exercises",
        "parser": parse_guidance_page,
        "short_title": "ECCC guidance: simulation exercises",
        "expected_at_least": 3,
    },
    {
        "document_id": "technical_guidelines",
        "parser": parse_pdf,
        "settings": "technical_guidelines",
        "expected_at_least": 200,
    },
    {
        "document_id": "reporting_emergency",
        "parser": parse_pdf,
        "settings": "reporting_emergency",
        "expected_at_least": 5,
    },
]

# Documents that are in the corpus but produce no prose chunks, and why.
NO_CHUNKS = {
    "hazardous_substances": "table only; its UN numbers are merged into schedule1.csv",
}


def real_date(value):
    """
    A manifest cell can be empty, or say "n/a", or hold a dash. All three mean
    the same thing: there is no date here. Only a real one counts.
    """
    value = (value or "").strip()
    return "" if value.lower() in ("", "n/a", "na", "-", "none") else value


def read_manifest():
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return {row["document_id"]: row for row in csv.DictReader(handle)}


def build_doc(entry, manifest):
    """
    One document description, assembled from three places:
      - the manifest  : what the document IS (title, source type, dates)
      - CORPUS        : how to parse it
      - PDF_SETTINGS  : the typography measurements, for PDFs only
    """
    row = manifest[entry["document_id"]]
    doc = {
        "document_id": row["document_id"],
        "source_type": row["source_type"],
        "regime": row["regime"],
        "filename": row["filename"],
        "title": row["title"],
        "current_to": real_date(row["current_to"]),
        "published_or_amended": real_date(row["published_or_amended"]),
    }
    if entry.get("settings"):
        doc.update(PDF_SETTINGS[entry["settings"]])
    for key in ("part", "citation_prefix", "short_title"):
        if key in entry:
            doc[key] = entry[key]
    doc.setdefault("short_title", row["title"])
    return doc


def build():
    manifest = read_manifest()
    all_chunks = []
    report = []

    for entry in CORPUS:
        doc = build_doc(entry, manifest)
        chunks = entry["parser"](doc)

        # Provenance travels WITH the chunk. A chunk that has been retrieved
        # on its own must still be able to say how current it is.
        for c in chunks:
            c["document_title"] = doc["title"]
            # Consolidated law has a "current to" date; guidance does not, it
            # has a publication date. Carry whichever exists, and say which.
            c["current_to"] = doc["current_to"]
            c["published_or_amended"] = doc["published_or_amended"]
            c["as_of"] = doc["current_to"] or doc["published_or_amended"] or "unknown"

        all_chunks.extend(chunks)
        report.append((doc, chunks, entry["expected_at_least"]))

    return all_chunks, report, manifest


def write(chunks):
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for c in chunks:
            handle.write(json.dumps(c, ensure_ascii=False) + "\n")


def rule(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


if __name__ == "__main__":
    chunks, report, manifest = build()
    write(chunks)

    rule("WHAT WAS BUILT")
    print(f"  {len(chunks)} chunks written to {CHUNKS_PATH}\n")
    print(f"  {'document':<24} {'type':<11} {'chunks':>7} {'chars':>9}  {'as of':<12}")
    print("  " + "-" * 76)
    for doc, part, _ in report:
        chars = sum(c["chars"] for c in part)
        print(f"  {doc['document_id']:<24} {doc['source_type']:<11} "
              f"{len(part):>7} {chars:>9,}  "
              f"{doc['current_to'] or doc['published_or_amended'] or 'unknown':<12}")
    for doc_id, why in NO_CHUNKS.items():
        print(f"  {doc_id:<24} {'—':<11} {0:>7} {0:>9}  ({why})")

    rule("CHECK 1 — did every parser actually return something?")
    for doc, part, floor in report:
        verdict = "ok" if len(part) >= floor else "TOO FEW"
        print(f"  {doc['document_id']:<24} {len(part):>5} chunks "
              f"(expected at least {floor})   {verdict}")

    rule("CHECK 2 — is every chunk usable on its own?")
    ids = Counter(c["chunk_id"] for c in chunks)
    duplicates = [i for i, n in ids.items() if n > 1]
    no_citation = [c for c in chunks if not c.get("citation", "").strip()]
    no_text = [c for c in chunks if not c.get("text", "").strip()]
    no_source = [c for c in chunks if not c.get("source_type", "").strip()]
    print(f"  duplicate chunk_ids        : {len(duplicates)}")
    for i in duplicates[:5]:
        print(f"      {i}")
    print(f"  chunks with no citation    : {len(no_citation)}")
    print(f"  chunks with no text        : {len(no_text)}")
    print(f"  chunks with no source_type : {len(no_source)}")

    rule("CHECK 3 — how big are the chunks?")
    sizes = sorted(c["chars"] for c in chunks)
    middle = sizes[len(sizes) // 2]
    print(f"  smallest {sizes[0]:,}   median {middle:,}   largest {sizes[-1]:,}")
    long_ones = [c for c in chunks if c["chars"] > TOO_LONG]
    short_ones = [c for c in chunks if c["chars"] < TOO_SHORT]
    print(f"\n  over {TOO_LONG:,} chars: {len(long_ones)}")
    for c in sorted(long_ones, key=lambda c: -c["chars"])[:8]:
        print(f"      {c['chars']:>6}  {c['citation'][:70]}")
    print(f"\n  under {TOO_SHORT} chars: {len(short_ones)}")
    for c in sorted(short_ones, key=lambda c: c["chars"])[:8]:
        print(f"      {c['chars']:>6}  {c['citation'][:70]}")

    rule("CHECK 4 — law against guidance")
    print("  The risk your spec names in §10.1 is guidance being answered as if")
    print("  it were law. This is the number that says how big that risk is.\n")
    LAW = {"regulation", "act"}
    law = [c for c in chunks if c["source_type"] in LAW]
    guidance = [c for c in chunks if c["source_type"] not in LAW]
    law_chars = sum(c["chars"] for c in law)
    guidance_chars = sum(c["chars"] for c in guidance)
    total_chars = law_chars + guidance_chars
    print(f"  {'':<10} {'chunks':>8} {'share':>8} {'chars':>10} {'share':>8}")
    print(f"  {'law':<10} {len(law):>8} {len(law)/len(chunks):>7.0%} "
          f"{law_chars:>10,} {law_chars/total_chars:>7.0%}")
    print(f"  {'guidance':<10} {len(guidance):>8} {len(guidance)/len(chunks):>7.0%} "
          f"{guidance_chars:>10,} {guidance_chars/total_chars:>7.0%}")
    print(f"\n  guidance outweighs law by {guidance_chars/law_chars:.1f} to 1 by volume.")
    print("  A single ranked search would therefore return guidance most of the")
    print("  time, simply because there is more of it. Stage 3 has to retrieve")
    print("  from law and guidance separately and then merge.")

    rule("CHECK 5 — the structured half")
    with SCHEDULE1_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with_un = [r for r in rows if r["un_number"]]
    print(f"  substances in schedule1.csv : {len(rows)}")
    print(f"  with a UN number            : {len(with_un)}")
    print(f"  hazard categories           : "
          f"{', '.join(sorted({r['hazard_category'] for r in rows}))}")
    print("\n  These 249 rows are NOT in chunks.jsonl and must never be.")
    print("  A threshold is looked up, not searched for.")

    rule("CHECK 6 — is anything in the manifest unaccounted for?")
    parsed = {doc["document_id"] for doc, _, _ in report}
    for doc_id in manifest:
        if doc_id in parsed:
            state = "parsed"
        elif doc_id in NO_CHUNKS:
            state = f"no chunks — {NO_CHUNKS[doc_id]}"
        else:
            state = "NOT ACCOUNTED FOR"
        print(f"  {doc_id:<24} {state}")