"""
Extract Schedule 1 of the E2 Regulations into a real table.

Schedule 1 is a list of substances with thresholds. It is data, not prose.
Turning it into a spreadsheet means a threshold question can be answered by
looking a value up, instead of by hoping a search finds the right row.

This is the structured half of the hybrid architecture.

UN numbers are NOT in the regulation. They come from an ECCC web page and are
joined on here as a convenience column, labelled with how each one matched.
"""

import csv
from pathlib import Path

from lxml import etree
from lxml import html as lhtml

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/schedule1.csv")

COLUMNS = [
    "part",
    "item",
    "cas_number",
    "substance_name",
    "min_concentration_pct",
    "min_quantity_tonnes",
    "hazard_category",
    "hazard_category_meaning",
    "un_number",
    "un_matched_by",
]


# ----------------------------------------------------------------------------
# The regulation
# ----------------------------------------------------------------------------

def find_schedule_1(root):
    """Schedule 1 is the only schedule whose label ends in '1'."""
    for schedule in root.findall("Schedule"):
        label = schedule.find(".//Label")
        if label is not None and "".join(label.itertext()).strip().upper().endswith("1"):
            return schedule
    raise ValueError("Schedule 1 not found")


def read_legend(schedule):
    """
    The regulation defines its own hazard codes at the foot of Schedule 1.
    'I' means nothing to a reader; 'inhalation hazard' does.
    """
    legend = {}
    for item in schedule.findall(".//List/Item"):
        code = item.find("Label")
        meaning = item.find("Text")
        if code is not None and meaning is not None:
            legend["".join(code.itertext()).strip()] = "".join(meaning.itertext()).strip()
    return legend


def cells_of(row):
    return ["".join(entry.itertext()).strip() for entry in row.findall("entry")]


def is_data_row(cells):
    """
    The first two rows of each table are headers ('Column 1...' and
    'Item | CAS Registry Number | ...'). Real rows start with a number.
    """
    return len(cells) >= 6 and cells[0].isdigit()


def extract(doc):
    root = etree.parse(str(RAW_DIR / doc["filename"])).getroot()
    schedule = find_schedule_1(root)
    legend = read_legend(schedule)

    rows = []
    skipped = 0

    for group in schedule.findall(".//TableGroup"):
        caption = group.find("Caption")
        part_label = "".join(caption.itertext()).strip() if caption is not None else "?"
        part = part_label.split()[-1]          # "PART 1" -> "1"

        for row in group.findall(".//row"):
            cells = cells_of(row)
            if not is_data_row(cells):
                skipped += 1
                continue
            item, cas, name, conc, qty, category = cells[:6]
            rows.append({
                "part": part,
                "item": item,
                "cas_number": cas,
                "substance_name": name,
                "min_concentration_pct": conc,
                "min_quantity_tonnes": qty,
                "hazard_category": category,
                "hazard_category_meaning": legend.get(category, ""),
                "un_number": "",
                "un_matched_by": "",
            })

    return rows, legend, skipped


# ----------------------------------------------------------------------------
# The ECCC list (UN numbers)
# ----------------------------------------------------------------------------

def read_eccc_un_numbers(filename):
    """
    Read the ECCC hazardous substances page twice over: once keyed by substance
    name, once keyed by CAS number.

    Name is the primary key, not CAS. Six CAS numbers appear on TWO rows each
    (the anhydrous form in Part 1 and the solution in Part 2 share a CAS), and
    for hydrogen fluoride the two rows carry DIFFERENT UN numbers. Keying on
    CAS alone would give one of those rows the other one's UN number.
    """
    root = lhtml.parse(str(RAW_DIR / filename)).getroot()
    by_name, by_cas = {}, {}
    for tr in root.xpath("//table//tr"):
        cells = [" ".join(td.text_content().split()) for td in tr.xpath("./td")]
        if len(cells) < 6:
            continue
        name, cas, un = cells[0], cells[1], cells[2]
        if un.lower().startswith("no number"):
            un = ""
        by_name[name.lower()] = un
        by_cas[cas] = un
    return by_name, by_cas


def add_un_numbers(rows, by_name, by_cas):
    """
    Try name first, then CAS. Record which key worked so the choice is
    auditable rather than invisible.
    """
    counts = {"name": 0, "cas": 0, "none": 0}
    fallbacks = []
    for r in rows:
        key = r["substance_name"].lower()
        if key in by_name:
            r["un_number"] = by_name[key]
            r["un_matched_by"] = "name"
            counts["name"] += 1
        elif r["cas_number"] in by_cas:
            r["un_number"] = by_cas[r["cas_number"]]
            r["un_matched_by"] = "cas"
            counts["cas"] += 1
            fallbacks.append(r)
        else:
            r["un_number"] = ""
            r["un_matched_by"] = "no match"
            counts["none"] += 1
            fallbacks.append(r)
    return counts, fallbacks


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    doc = {"filename": "e2_regulation.xml"}
    rows, legend, skipped = extract(doc)

    by_name, by_cas = read_eccc_un_numbers("hazardous_substances.html")
    counts, fallbacks = add_un_numbers(rows, by_name, by_cas)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Substances extracted : {len(rows)}")
    print(f"Non-data rows skipped: {skipped}  (headers and blanks)")
    print(f"Written to           : {OUT_PATH}")

    part1 = [r for r in rows if r["part"] == "1"]
    part2 = [r for r in rows if r["part"] == "2"]
    print(f"\n  Part 1 (substances): {len(part1)}")
    print(f"  Part 2 (solutions) : {len(part2)}")

    print("\nLegend read from the regulation:")
    for code, meaning in legend.items():
        count = sum(1 for r in rows if r["hazard_category"] == code)
        print(f"  {code}  {meaning:<28} {count:>4} substances")

    print("\nUN number matching against the ECCC list:")
    print(f"  matched by name : {counts['name']}")
    print(f"  matched by cas  : {counts['cas']}")
    print(f"  no match        : {counts['none']}")
    if fallbacks:
        print("  rows that did not match on name:")
        for r in fallbacks:
            print(f"    Part {r['part']} item {r['item']:>4}  {r['substance_name'][:44]:<46} "
                  f"UN '{r['un_number']}'  ({r['un_matched_by']})")

    print("\n" + "=" * 78)
    print("CHECK — the ammonia question (Q1)")
    print("=" * 78)
    for r in rows:
        if r["cas_number"] == "7664-41-7":
            print(f"  Part {r['part']}, item {r['item']}: {r['substance_name']}")
            print(f"     minimum concentration : {r['min_concentration_pct']}% mass/mass")
            print(f"     minimum quantity      : {r['min_quantity_tonnes']} tonnes")
            print(f"     hazard category       : {r['hazard_category']} ({r['hazard_category_meaning']})")
            print(f"     UN number             : {r['un_number']}")
            print()

    print("=" * 78)
    print("CHECK — same CAS number, two different substances")
    print("=" * 78)
    seen = {}
    for r in rows:
        seen.setdefault(r["cas_number"], []).append(r)
    shared = {cas: rs for cas, rs in seen.items() if len(rs) > 1}
    print(f"  CAS numbers used by more than one row: {len(shared)}")
    for cas, rs in shared.items():
        uns = {r["un_number"] for r in rs}
        flag = "  <-- different UN numbers" if len(uns) > 1 else ""
        print(f"    {cas:<12}{flag}")
        for r in rs:
            print(f"       Part {r['part']} item {r['item']:>4}  {r['substance_name'][:38]:<40} UN '{r['un_number']}'")

    print("\n" + "=" * 78)
    print("CHECK — values that are not plain numbers (worth knowing about)")
    print("=" * 78)
    odd = [r for r in rows if not r["min_concentration_pct"].replace(".", "").isdigit()]
    print(f"  rows where concentration is not numeric: {len(odd)}")
    for r in odd[:6]:
        print(f"     item {r['item']:>4} {r['substance_name'][:36]:<38} conc='{r['min_concentration_pct']}'")