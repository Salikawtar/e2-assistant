"""
Turn Justice Laws XML into chunks.

A chunk is one labelled block, with its address attached: which document,
which section, which heading. Sections and Schedules 2-8 are handled the
same way, because the XML labels them the same way.

Schedule 1 is NOT chunked here. It is a table of data, extracted separately
by schedule1.py.
"""

import json
from pathlib import Path

from lxml import etree

RAW_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/chunks.jsonl")

# If a section is longer than this, split it at its subsections.
MAX_CHARS = 1800


def block_text(element, skip_marginal_note=True):
    """
    Read a block as plain text.

    The XML keeps labels and prose in separate tags, so we walk the tree and
    put them back together in reading order with spaces between them.
    """
    parts = []

    def walk(node, depth):
        for child in node:
            tag = child.tag
            if not isinstance(tag, str):
                continue
            if tag == "MarginalNote" and depth == 0 and skip_marginal_note:
                continue
            if tag in ("Label", "Text", "MarginalNote"):
                parts.append("".join(child.itertext()).strip())
            else:
                walk(child, depth + 1)

    walk(element, 0)
    return " ".join(" ".join(parts).split())


def marginal_note_of(element):
    """The short italic title given to a section, if it has one."""
    note = element.find("MarginalNote")
    return "".join(note.itertext()).strip() if note is not None else ""


def label_of(element):
    label = element.find("Label")
    return "".join(label.itertext()).strip() if label is not None else ""


def make_chunk(doc, chunk_id, citation, heading, note, text):
    """
    Build one chunk record.

    `citation` is what a user will see, so it must read the way the law
    reads: "3(2)", not "32".
    """
    return {
        "chunk_id": chunk_id,
        "document_id": doc["document_id"],
        "source_type": doc["source_type"],
        "regime": doc["regime"],
        "citation": citation,
        "heading": heading,
        "marginal_note": note,
        "text": text,
        "chars": len(text),
    }


def chunk_section(element, doc, section_label, heading):
    """
    One section becomes one chunk — unless it is too long, in which case it is
    split at its subsections. Each piece keeps the section number, so nothing
    is ever orphaned from its parent.
    """
    note = marginal_note_of(element)
    text = block_text(element)
    doc_id = doc["document_id"]
    prefix = doc.get("citation_prefix", "")
    whole = f"{prefix}section {section_label}".strip()

    if len(text) <= MAX_CHARS:
        return [make_chunk(doc, f"{doc_id}__s{section_label}", whole, heading, note, text)]

    subsections = element.findall("Subsection")
    if not subsections:
        return [make_chunk(doc, f"{doc_id}__s{section_label}", whole, heading, note, text)]

    chunks = []
    for sub in subsections:
        sub_label = label_of(sub)                 # e.g. "(2)"
        bare = sub_label.strip("()")              # e.g. "2"
        citation = f"{prefix}section {section_label}{sub_label}".strip()
        chunks.append(make_chunk(
            doc,
            f"{doc_id}__s{section_label}_{bare}",
            citation,
            heading,
            marginal_note_of(sub) or note,
            f"{section_label}{sub_label} " + block_text(sub, skip_marginal_note=False),
        ))
    return chunks


def parse_e2_regulation(doc):
    """Read the regulation XML and return a list of chunks."""
    root = etree.parse(str(RAW_DIR / doc["filename"])).getroot()
    chunks = []

    # --- Body: headings and sections, in document order ---
    body = root.find(".//Body")
    heading = ""
    for child in body:
        if child.tag == "Heading":
            title = child.find("TitleText")
            if title is not None:
                heading = "".join(title.itertext()).strip()
        elif child.tag == "Section":
            chunks.extend(chunk_section(child, doc, label_of(child), heading))

    # --- Schedules 2 to 8: same machinery, different container ---
    for schedule in root.findall("Schedule"):
        head = schedule.find("ScheduleFormHeading")
        if head is None:
            continue
        sched_label = "".join(head.find("Label").itertext()).strip()
        title_el = head.find("TitleText")
        sched_title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        if sched_label.upper().endswith("1"):
            continue  # Schedule 1 is a table — handled separately

        number = sched_label.split()[-1]
        for section in schedule.findall(".//Section"):
            label = label_of(section)
            chunks.append(make_chunk(
                doc,
                f"{doc['document_id']}__sch{number}_{label}",
                f"Schedule {number}, item {label}",
                sched_label,
                sched_title,
                f"{sched_label}, item {label}. " + block_text(section, skip_marginal_note=False),
            ))

    return chunks


def parse_act_part(doc):
    """
    Read one Part of an Act and return its chunks.

    Justice Laws uses the same tags for Acts and regulations, so the same
    chunker works. The only new job is deciding where the Part starts and
    stops: the Body is a flat list of Headings and Sections, so we switch
    ON at the Part we want and OFF at the next Part.
    """
    root = etree.parse(str(RAW_DIR / doc["filename"])).getroot()
    body = root.find(".//Body")
    wanted = doc["part"].upper()

    chunks = []
    inside = False
    heading = ""

    for child in body:
        if child.tag == "Heading":
            label_el = child.find("Label")
            title_el = child.find("TitleText")
            label = "".join(label_el.itertext()).strip() if label_el is not None else ""
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            if label.upper().startswith("PART"):
                inside = label.upper() == wanted
                heading = f"{label} — {title}" if inside else ""
            elif inside and title:
                heading = title

        elif child.tag == "Section" and inside:
            chunks.extend(chunk_section(child, doc, label_of(child), heading))

    return chunks


if __name__ == "__main__":
    documents = [
        {
            "document_id": "e2_regulation",
            "source_type": "regulation",
            "regime": "E2",
            "filename": "e2_regulation.xml",
            "citation_prefix": "",
            "parser": "regulation",
        },
        {
            "document_id": "cepa_act",
            "source_type": "act",
            "regime": "E2",
            "filename": "cepa_act.xml",
            "citation_prefix": "CEPA ",
            "parser": "act_part",
            "part": "PART 8",
        },
    ]

    chunks = []
    for doc in documents:
        if doc["parser"] == "regulation":
            got = parse_e2_regulation(doc)
        else:
            got = parse_act_part(doc)
        print(f"{doc['document_id']:<18} {len(got):>4} chunks")
        chunks.extend(got)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(chunks)} chunks -> {CHUNKS_PATH}")

    print("\nCEPA Part 8 chunks:")
    for c in chunks:
        if c["document_id"] == "cepa_act":
            print(f"  {c['citation']:<22} {c['marginal_note'][:44]:<46} {c['chars']:>5}")