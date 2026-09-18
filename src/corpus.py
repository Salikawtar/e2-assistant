"""
Download the E2 corpus and record a manifest of what was downloaded.

The CORPUS list below is the single source of truth for what this
assistant is allowed to read. If a document is not in this list,
it does not exist as far as the system is concerned.

Run order when this file is executed directly:
    1. download_all()      fetch anything missing, skip what we already have
    2. verify_manifest()   compare what is on disk against the existing manifest
    3. build_manifest()    rewrite the manifest from the files actually present

Step 2 exists because step 1 skips files that already exist. Without it,
a file can drift on disk and the script will still report "have already".
"""

import csv
import hashlib
import http.cookiejar
import time
import urllib.request
from datetime import date
from pathlib import Path

RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/corpus_manifest.csv")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-CA,en;q=0.9,fr-CA;q=0.8",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}

FILE_SIGNATURES = {"pdf": b"%PDF", "xml": b"<?xml"}

CORPUS = [
    {
        "document_id": "e2_regulation",
        "title": "Environmental Emergency Regulations, 2019 (SOR/2019-51)",
        "source_type": "regulation",
        "authority": "Department of Justice Canada",
        "jurisdiction": "federal",
        "regime": "E2",
        "source_format": "xml",
        "url": "https://laws-lois.justice.gc.ca/eng/XML/SOR-2019-51.xml",
        "filename": "e2_regulation.xml",
        "published_or_amended": "2019-08-24",
        "current_to": "2021-10-20",
        "retrieval_method": "automated",
    },
    {
        "document_id": "cepa_act",
        "title": "Canadian Environmental Protection Act, 1999",
        "source_type": "act",
        "authority": "Department of Justice Canada",
        "jurisdiction": "federal",
        "regime": "E2",
        "source_format": "xml",
        "url": "https://laws-lois.justice.gc.ca/eng/XML/C-15.31.xml",
        "filename": "cepa_act.xml",
        "published_or_amended": "2026-03-26",
        "current_to": "2026-03-31",
        "retrieval_method": "automated",
    },
    {
        "document_id": "technical_guidelines",
        "title": "Technical Guidelines for the Environmental Emergency Regulations, 2019 (version 2.0)",
        "source_type": "guidance",
        "authority": "Environment and Climate Change Canada",
        "jurisdiction": "federal",
        "regime": "E2",
        "source_format": "pdf",
        "url": "https://publications.gc.ca/collections/collection_2020/eccc/En4-386-2020-eng.pdf",
        "filename": "technical_guidelines.pdf",
        "published_or_amended": "2020-12",
        "current_to": "n/a",
        "retrieval_method": "automated (browser headers + cookie)",
    },
    {
        "document_id": "simulation_exercises",
        "title": "Environmental emergency plan simulation exercises",
        "source_type": "guidance",
        "authority": "Environment and Climate Change Canada",
        "jurisdiction": "federal",
        "regime": "E2",
        "source_format": "html",
        "url": "https://www.canada.ca/en/environment-climate-change/services/environmental-emergencies-program/regulations/environmental-emergency-plan-simulation-exercises.html",
        "filename": "simulation_exercises.html",
        "published_or_amended": "2022-08-16",
        "current_to": "n/a",
        "retrieval_method": "automated (browser headers)",
    },
    {
        "document_id": "hazardous_substances",
        "title": "List of hazardous substances",
        "source_type": "reference",
        "authority": "Environment and Climate Change Canada",
        "jurisdiction": "federal",
        "regime": "E2",
        "source_format": "html",
        "url": "https://www.canada.ca/en/environment-climate-change/services/environmental-emergencies-program/regulations/list-hazardous-substances.html",
        "filename": "hazardous_substances.html",
        "published_or_amended": "2016-01-14",
        "current_to": "n/a",
        "retrieval_method": "automated (browser headers)",
    },
    {
        "document_id": "reporting_emergency",
        "title": "Environmental Emergency Regulations, 2019: reporting an environmental emergency",
        "source_type": "guidance",
        "authority": "Environment and Climate Change Canada",
        "jurisdiction": "federal",
        "regime": "E2",
        "source_format": "pdf",
        "url": "https://publications.gc.ca/collections/collection_2019/eccc/En4-376-5-2019-eng.pdf",
        "filename": "reporting_emergency.pdf",
        "published_or_amended": "2019",
        "current_to": "n/a",
        "retrieval_method": "automated (browser headers + cookie)",
    },
]

MANIFEST_COLUMNS = [
    "document_id", "title", "source_type", "authority", "jurisdiction",
    "regime", "source_format", "url", "filename",
    "published_or_amended", "current_to", "retrieved_date",
    "retrieval_method", "bytes", "sha256",
]


def looks_correct(data, source_format):
    """Does the downloaded content actually match the format we expected?"""
    signature = FILE_SIGNATURES.get(source_format)
    if signature is None:
        return True
    return data.lstrip()[: len(signature)] == signature


def fetch(url, opener, referer=None):
    headers = dict(BROWSER_HEADERS)
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=120) as response:
        return response.read()


def read_existing_manifest():
    """The manifest as it stands, keyed by document_id. Empty if there is none."""
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["document_id"]: row for row in csv.DictReader(handle)}


def download(doc, attempts=3):
    """Download one document, verify it, then save.

    Returns True only if this run actually fetched the file over the network.
    Returns False if the file was already on disk, or if the download failed.
    The caller needs that distinction so retrieved_date stays truthful.
    """
    target = RAW_DIR / doc["filename"]

    if target.exists():
        print(f"  have already   {doc['filename']}")
        return False

    source_format = doc["source_format"]

    for attempt in range(1, attempts + 1):
        try:
            jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(jar)
            )
            data = fetch(doc["url"], opener)

            # publications.gc.ca serves an archive notice first, then the file.
            if not looks_correct(data, source_format):
                data = fetch(doc["url"], opener, referer=doc["url"])

            if not looks_correct(data, source_format):
                raise ValueError(
                    f"not a {source_format} file, starts with {data.lstrip()[:20]!r}"
                )

            target.write_bytes(data)
            print(f"  downloaded     {doc['filename']}  ({len(data):,} bytes)")
            return True

        except Exception as error:
            if attempt < attempts:
                print(f"  retrying       {doc['filename']}  (attempt {attempt} failed)")
                time.sleep(3)
            else:
                print(f"  FAILED         {doc['filename']}  --  {error}")

    return False


def download_all():
    """Fetch anything we do not already have. Returns the ids fetched this run."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Corpus download - {date.today()}")

    fetched = set()
    for doc in CORPUS:
        if download(doc):
            fetched.add(doc["document_id"])

    print("Done.")
    return fetched


def verify_manifest():
    """Compare the files on disk against the manifest we already have.

    download() skips files that already exist, so nothing else in this script
    would ever notice a file changing underneath us. This is the check that does.
    Returns the list of document_ids whose fingerprint no longer matches.
    """
    existing = read_existing_manifest()
    if not existing:
        print("\nNo manifest yet, nothing to verify.")
        return []

    print(f"\nVerifying {len(existing)} files against {MANIFEST_PATH}")
    drifted = []

    for doc in CORPUS:
        row = existing.get(doc["document_id"])
        path = RAW_DIR / doc["filename"]

        if row is None:
            print(f"  NEW            {doc['filename']}  (not in the manifest yet)")
            continue
        if not path.exists():
            print(f"  MISSING        {doc['filename']}  (in the manifest, not on disk)")
            drifted.append(doc["document_id"])
            continue

        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()

        if actual == row["sha256"]:
            print(f"  ok             {doc['filename']}  sha256 {actual[:12]}...")
        else:
            drifted.append(doc["document_id"])
            print(f"  CHANGED        {doc['filename']}")
            print(f"                 manifest {int(row['bytes']):>10,} B  {row['sha256'][:12]}...")
            print(f"                 on disk  {len(data):>10,} B  {actual[:12]}...")

    if drifted:
        print(f"\n  {len(drifted)} of {len(existing)} files no longer match the manifest.")
        print("  The indexed text may not be the text on disk. Check why before rebuilding.")
    else:
        print("\n  All files match. The corpus is byte-for-byte what was indexed.")

    return drifted


def build_manifest(fetched_this_run=None):
    """Write one row per document, describing the file we actually have.

    retrieved_date is the date WE took our copy. It is only set to today for
    documents actually fetched in this run. For everything else the existing
    manifest date is carried forward, because a file that was not re-fetched
    was not re-retrieved, and stamping today on it would falsify provenance.
    """
    fetched_this_run = fetched_this_run or set()
    today = date.today().isoformat()
    existing = read_existing_manifest()
    rows = []

    for doc in CORPUS:
        path = RAW_DIR / doc["filename"]
        if not path.exists():
            print(f"  MISSING        {doc['filename']}, not in the manifest")
            continue

        data = path.read_bytes()
        row = {key: doc.get(key, "") for key in MANIFEST_COLUMNS}

        previous = existing.get(doc["document_id"])
        if doc["document_id"] in fetched_this_run or previous is None:
            row["retrieved_date"] = today
        else:
            row["retrieved_date"] = previous["retrieved_date"]

        row["bytes"] = len(data)
        row["sha256"] = hashlib.sha256(data).hexdigest()
        rows.append(row)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written: {MANIFEST_PATH}  ({len(rows)} documents)")
    for row in rows:
        print(f"  {row['document_id']:<22} {row['source_type']:<11} "
              f"{row['bytes']:>9,} B   retrieved {row['retrieved_date']}   "
              f"sha256 {row['sha256'][:12]}...")


if __name__ == "__main__":
    fetched = download_all()
    verify_manifest()
    build_manifest(fetched)
