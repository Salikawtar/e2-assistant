"""
Turn the two ECCC guidance PDFs into chunks.

A PDF has no structure. There is no <Section> tag and no <h2> tag — only pieces
of text, each with a position, a size and a font name. So we rebuild the
structure the way a human reader does: a line that is BIGGER and BOLDER than
the body text is a heading, and everything under it belongs to it.

Each document declares how big its own headings are, because the two PDFs are
typeset differently:
  - technical_guidelines : body 12pt, headings 12.96 / 14.04 / 16pt bold
  - reporting_emergency  : body 10.5pt, headings 10.5pt BOLD (same size)

Running headers, page numbers and copyright blocks are dropped by name.
"""

import re
from collections import OrderedDict
from pathlib import Path

import pdfplumber

RAW_DIR = Path("data/raw")

# Same limit as the regulation parser, so chunks stay comparable in size.
MAX_CHARS = 1800

# Below this, a "section" is a heading with nothing real underneath it.
# A document can override it: the two-page factsheet has real answers that are
# only one sentence long.
MIN_CHARS = 120


# ----------------------------------------------------------------------------
# Reading the page
# ----------------------------------------------------------------------------

def page_lines(page):
    """
    Rebuild lines out of words.

    pdfplumber gives us words, not lines. Words on the same line share almost
    the same vertical position, so we group them by rounded 'top'.
    """
    buckets = OrderedDict()
    for word in page.extract_words(extra_attrs=["size", "fontname"]):
        buckets.setdefault(round(word["top"] / 3), []).append(word)

    lines = []
    for key in sorted(buckets):
        words = buckets[key]
        lines.append({
            "text": " ".join(w["text"] for w in words),
            "size": max(w["size"] for w in words),
            # Share of the line that is bold. Not "any word" — one bold
            # phrase inside a sentence is emphasis, not a heading. Not "every
            # word" either: real headings in this PDF contain the occasional
            # word the typesetter forgot to bold.
            "bold": sum("Bold" in w["fontname"] for w in words) / len(words),
        })
    return lines


def tidy(text):
    """Repair the small mess PDF text extraction leaves behind."""
    text = text.replace("", "-").replace("•", "-")   # Symbol-font bullets
    text = re.sub(r"\s+([?.,;:])", r"\1", text)                 # " ?" -> "?"
    return " ".join(text.split())


def tidy_heading(text):
    """
    Section numbers get mangled by the extractor: '5 . 3.2' and '5.3 8'.
    Both are the same defect — a space where a dot belongs.
    """
    text = re.sub(r"^(\d)\s*\.\s*(\d)", r"\1.\2", text)
    text = re.sub(r"^(\d+(?:\.\d+)*)\.?\s+(\d+)\s", r"\1.\2 ", text)
    return text


def is_noise(text, doc):
    if not text.strip():
        return True
    if text.strip().isdigit():                   # a page number on its own
        return True
    return any(text.startswith(p) for p in doc.get("drop_lines", []))


# A heading may carry a stray un-bolded word; a sentence with a bold phrase
# in it may not pass as a heading.
BOLD_SHARE = 0.8


def is_heading(line, doc):
    if line["bold"] < BOLD_SHARE:
        return False
    if line["size"] < doc["heading_min_size"]:
        return False
    if len(line["text"]) >= 120:                 # a whole bold paragraph is not a heading
        return False
    return len(re.findall(r"[A-Za-z]", line["text"])) >= 3


def continues_heading(previous, current):
    """
    Is `current` the tail of a heading that wrapped onto a second line?

    Yes for '5.3.5 Identification and Assessment of Environmental Emergency'
    followed by 'Scenarios'. No when the first line already ended (':' or '?')
    or when the second line opens its own numbered section.
    """
    if previous.endswith((":", "?", ".")):
        return False
    return not re.match(r"^(\d|APPENDIX)", current, flags=re.IGNORECASE)


# ----------------------------------------------------------------------------
# Cutting the document into sections
# ----------------------------------------------------------------------------

def read_sections(doc):
    """
    Walk the PDF top to bottom and cut it into sections.

    A heading we have already seen resumes that section instead of opening a
    new one — a table header repeated on twelve pages is typeset exactly like a
    heading and would otherwise produce twelve stubs.

    Every line keeps its page number, so a citation can point at the page the
    text is actually on rather than the page the section started on.
    """
    sections = []
    index_of = {}
    current = None
    previous_was_heading = False

    with pdfplumber.open(RAW_DIR / doc["filename"]) as pdf:
        first = doc.get("first_page", 1)
        for offset, page in enumerate(pdf.pages[first - 1:]):
            page_number = first + offset
            for line in page_lines(page):
                text = tidy(line["text"])
                if is_noise(text, doc):
                    continue

                if is_heading(line, doc):
                    text = tidy_heading(text)
                    if (previous_was_heading and current is not None
                            and continues_heading(current["heading"], text)):
                        del index_of[current["heading"]]
                        current["heading"] = tidy(current["heading"] + " " + text)
                        index_of[current["heading"]] = sections.index(current)
                        continue
                    if text in index_of:
                        current = sections[index_of[text]]
                    else:
                        current = {"heading": text, "page": page_number, "lines": []}
                        index_of[text] = len(sections)
                        sections.append(current)
                    previous_was_heading = True
                    continue

                previous_was_heading = False
                if current is None:              # text before the first heading
                    current = {"heading": doc["short_title"], "page": page_number,
                               "lines": []}
                    index_of[current["heading"]] = 0
                    sections.append(current)
                current["lines"].append((text, page_number))

    return sections


def split_long(lines, limit):
    """Pack lines into groups no longer than `limit`, never cutting a line."""
    groups, buffer, size = [], [], 0
    for line in lines:
        if buffer and size + len(line[0]) + 1 > limit:
            groups.append(buffer)
            buffer, size = [], 0
        buffer.append(line)
        size += len(line[0]) + 1
    if buffer:
        groups.append(buffer)
    return groups


def section_number(heading):
    """'7.1 Annual Simulation Exercises' -> '7.1'.  Otherwise ''."""
    match = re.match(r"^(\d+(?:\.\d+)*)\s", heading)
    return match.group(1) if match else ""


def parse_pdf(doc):
    chunks = []
    used_ids = {}

    for section in read_sections(doc):
        groups = split_long(section["lines"], MAX_CHARS)
        total = len(groups)
        number = section_number(section["heading"])

        for position, group in enumerate(groups, start=1):
            body = " ".join(text for text, _ in group)
            if len(body) < doc.get("min_chars", MIN_CHARS):
                continue

            page = group[0][1]                   # the page THIS part starts on
            where = f"s. {number}" if number else f"“{section['heading']}”"
            citation = f"{doc['short_title']}, {where} (p. {page})"
            if total > 1:
                citation += f", part {position} of {total}"

            slug = number.replace(".", "_") if number else f"p{page}"
            base = f"{doc['document_id']}__{slug}"
            used_ids[base] = used_ids.get(base, 0) + 1
            chunk_id = base if used_ids[base] == 1 else f"{base}#{used_ids[base]}"

            text = f"{section['heading']}. {body}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc["document_id"],
                "source_type": doc["source_type"],
                "regime": doc["regime"],
                "citation": citation,
                "heading": doc["short_title"],
                "marginal_note": section["heading"],
                "text": text,
                "chars": len(text),
                "page": page,
            })
    return chunks


DOCUMENTS = [
    {
        "document_id": "technical_guidelines",
        "short_title": "Technical Guidelines for the E2 Regulations, 2019 (v2.0)",
        "source_type": "guidance",
        "regime": "E2",
        "filename": "technical_guidelines.pdf",
        # Body text is 12.0pt. The smallest real heading is 12.96pt. 12.5
        # sits in that gap; it was chosen by measuring, not by guessing.
        "heading_min_size": 12.5,
        "first_page": 9,            # 1-8 are cover, copyright, revision log, contents
        "drop_lines": [
            "Technical Guidelines for the",
            "Environmental Emergency Regulations, 2019",
            "paragraph of Items to Include",   # repeated table header, Appendix 7
        ],
    },
    {
        "document_id": "reporting_emergency",
        "short_title": "ECCC factsheet: Reporting an Environmental Emergency",
        "source_type": "guidance",
        "regime": "E2",
        "filename": "reporting_emergency.pdf",
        # Here headings are the same size as the body — only bold sets them
        # apart, so the size test just has to not get in the way.
        "heading_min_size": 10.4,
        "min_chars": 40,
        "drop_lines": [
            "Disclosure:", "character.", "the official version",
            "Cat. No.:", "ISBN:", "For information regarding", "Centre at",
            "Photos:", "©", "Aussi disponible",
        ],
    },
]


if __name__ == "__main__":
    everything = {}

    for doc in DOCUMENTS:
        chunks = parse_pdf(doc)
        everything[doc["document_id"]] = chunks
        sizes = [c["chars"] for c in chunks]

        headings = []
        for c in chunks:
            if c["marginal_note"] not in headings:
                headings.append(c["marginal_note"])

        print(f"\n{doc['document_id']}  ->  {len(chunks)} chunks "
              f"from {len(headings)} sections")
        print(f"  smallest {min(sizes):,}   largest {max(sizes):,}   "
              f"average {sum(sizes)//len(sizes):,}")
        print(f"  duplicate chunk_ids: "
              f"{len(chunks) - len({c['chunk_id'] for c in chunks})}")
        print("-" * 92)
        for h in headings:
            parts = [c for c in chunks if c["marginal_note"] == h]
            print(f"  p{parts[0]['page']:>4}  {h[:62]:<64} {len(parts):>3} chunk(s)")

    print("\n" + "=" * 92)
    print("CHECK — citations for section 7, Simulation Exercises")
    print("=" * 92)
    for c in everything["technical_guidelines"]:
        if c["chunk_id"].startswith("technical_guidelines__7"):
            print(f"  {c['citation']}")

    print("\n" + "=" * 92)
    print("CHECK — is the page number in each citation true?")
    print("=" * 92)
    print("  Re-open the PDF, go to the page the citation names, and look for the")
    print("  chunk's own words on it. A citation nobody checked is only a guess.")
    for doc in DOCUMENTS:
        chunks = everything[doc["document_id"]]
        step = max(1, len(chunks) // 20)
        sample = chunks[::step]
        found, weak = 0, []
        with pdfplumber.open(RAW_DIR / doc["filename"]) as pdf:
            for c in sample:
                body = c["text"][len(c["marginal_note"]) + 2:]
                words = set(body.split()[:20])
                on_page = set((pdf.pages[c["page"] - 1].extract_text() or "").split())
                share = len(words & on_page) / max(1, len(words))
                if share >= 0.6:
                    found += 1
                else:
                    weak.append((c, round(share, 2)))
        print(f"\n  {doc['document_id']}: {found}/{len(sample)} sampled chunks "
              f"found on the page they cite")
        for c, share in weak:
            print(f"    weak match ({share}): {c['citation'][:76]}")