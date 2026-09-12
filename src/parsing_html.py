"""
Turn ECCC guidance web pages into chunks.

Web pages have no section numbers. What they do have is headings, so we use
those as the boundaries: everything under one <h2> becomes one chunk.

Tables are skipped here on purpose. A table of substances is data, not prose —
it is handled by schedule1.py, the same way Schedule 1 was.
"""

import json
from pathlib import Path

from lxml import html as lhtml

RAW_DIR = Path("data/raw")

# Anything shorter than this is a heading with no real content under it.
MIN_CHARS = 120

# canada.ca puts navigation and feedback links inside <main>. Skip them.
SKIP_HEADINGS = {"page details", "contact us"}


def page_title(root):
    found = root.xpath("//h1[@id='wb-cont']")
    return " ".join("".join(found[0].itertext()).split()) if found else ""


def date_modified(root):
    found = root.xpath("//gcds-date-modified")
    return "".join(found[0].itertext()).strip() if found else ""


def text_of(element):
    return " ".join("".join(element.itertext()).split())


def parse_guidance_page(doc):
    """
    Walk the page in reading order. Start a new chunk at every <h2>.
    Collect paragraphs and lists underneath it. Ignore tables.
    """
    root = lhtml.parse(str(RAW_DIR / doc["filename"])).getroot()
    title = page_title(root)
    modified = date_modified(root)
    main = root.xpath("//main")[0]

    chunks = []
    heading = ""          # current <h2>
    subheading = ""       # current <h3>
    buffer = []
    index = 0

    def flush():
        nonlocal buffer, index
        body = " ".join(buffer).strip()
        buffer = []
        if len(body) < MIN_CHARS:
            return
        if heading.lower() in SKIP_HEADINGS:
            return
        index += 1
        label = heading or "Introduction"
        chunks.append({
            "chunk_id": f"{doc['document_id']}__h{index}",
            "document_id": doc["document_id"],
            "source_type": doc["source_type"],
            "regime": doc["regime"],
            "citation": f"{doc['short_title']} — {label}",
            "heading": title,
            "marginal_note": label,
            "text": f"{title}. {label}. {body}",
            "chars": len(f"{title}. {label}. {body}"),
        })

    for el in main.iter():
        tag = el.tag
        if not isinstance(tag, str):
            continue
        if tag == "h2":
            flush()
            heading = text_of(el)
            subheading = ""
        elif tag == "h3":
            subheading = text_of(el)
            buffer.append(f"{subheading}:")
        elif tag == "p":
            # table cells on canada.ca are wrapped in <p>; ignore those
            if el.getparent() is not None and el.getparent().tag in ("td", "th"):
                continue
            body = text_of(el)
            if body:
                buffer.append(body)
        elif tag in ("ul", "ol"):
            for li in el.findall("li"):
                item = text_of(li)
                if item:
                    buffer.append("- " + item)

    flush()

    for c in chunks:
        c["date_modified"] = modified
    return chunks


if __name__ == "__main__":
    documents = [
        {
            "document_id": "simulation_exercises",
            "short_title": "ECCC guidance: simulation exercises",
            "source_type": "guidance",
            "regime": "E2",
            "filename": "simulation_exercises.html",
        },
        # hazardous_substances.html is NOT listed here on purpose.
        # That page is one big sortable table. Its only prose is the sorting
        # instructions. Its value is the UN numbers, and those are already
        # joined into data/schedule1.csv by schedule1.py.
    ]

    for doc in documents:
        chunks = parse_guidance_page(doc)
        print(f"\n{doc['document_id']}  ->  {len(chunks)} chunks")
        print("-" * 74)
        for c in chunks:
            print(f"  {c['chunk_id']:<26} {c['marginal_note'][:40]:<42} {c['chars']:>5}")
        if chunks:
            print(f"\n  sample text from {chunks[0]['chunk_id']}:")
            print("   ", chunks[0]["text"][:320], "...")