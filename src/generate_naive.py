"""
Steps 1-2 of Stage 4 — generate an answer with almost no rules, on purpose.

This is the version you would write if you had never thought about what can go
wrong. It hands the retrieved evidence to the model and asks for an answer.
Nothing more.

Run it, read the answers carefully, and look for four things:

  1. Does it use knowledge we never gave it?
  2. Does it invent a citation, or cite nothing at all?
  3. On a question the corpus cannot answer, does it answer anyway?
  4. Does it say whether a statement came from the regulation or from guidance?

Every rule added in Steps 3, 4 and 5 exists because of something you are about
to watch happen. Do not skip this.

Run:  python src/generate_naive.py
"""

import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from lookup_schedule1 import load as load_schedule1, lookup, describe
from search_vectors import load_index
from search_keyword import build_index as build_bm25
from search_split import pool_indices
from router import route, retrieve

# Temperature 0 means: no creativity, no randomness. The same question with
# the same evidence must produce the same answer, or Stage 5 cannot measure
# anything.
TEMPERATURE = 0

# Gemini 3.x models spend output tokens THINKING before they write. A budget
# of 700 was consumed before the answer finished, and the reply arrived cut
# off mid-sentence. Budget generously and check whether it was still hit.
MAX_OUTPUT_TOKENS = 3000

QUESTIONS = [
    "How often must a facility conduct a simulation exercise?",
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "Does an Ontario Environmental Compliance Approval satisfy the E2 plan requirement?",
    "What penalty applies if a facility fails to submit a notice on time?",
]


# Errors that mean "try again", as opposed to "this will never work".
TRANSIENT = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")

# A 429 means two different things and they need opposite responses.
#
#   a PER-MINUTE rate limit  clears in seconds. Waiting is exactly right.
#   a PER-DAY quota          does not clear until Google's reset. Waiting 62
#                            seconds per question buys nothing at all.
#
# Both print "busy". Three whole runs were spent waiting on the second kind
# before probe_models.py showed the real code. When the error names a per-day
# metric, this model is finished for today and is abandoned immediately.
DAILY_QUOTA = ("PerDay", "per day", "PerDayPerProject")


def model_version(name):
    """'models/gemini-3.6-flash' -> (3, 6). Unversioned names sort last."""
    match = re.search(r"(\d+)\.(\d+)", name)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def flash_models(client):
    """
    Every Flash model this key lists, newest first, stable before preview.

    We do NOT test them here. A one-word "ping" succeeded against a model that
    then returned 503 on the real request - a tiny call proves nothing about a
    seven-thousand-character one under load. The first real request is the
    only honest test, so the list is handed to ModelPool and tried there.
    """
    names = []
    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        actions = list(getattr(model, "supported_actions", None) or [])
        if actions and "generateContent" not in actions:
            continue
        if "embedding" in name.lower() or "flash" not in name.lower():
            continue
        names.append(name)

    if not names:
        raise SystemExit("No Flash model listed for this key.")

    def rank(name):
        lowered = name.lower()
        stable = not any(w in lowered for w in ("preview", "exp", "experimental"))
        return (model_version(name), stable)

    return sorted(names, key=rank, reverse=True)


class ModelPool:
    """
    A list of models to try, in order, with two different reactions:

      404 / not available  ->  this model will never work. Move to the next.
      503 / 429 / busy     ->  temporary. Wait longer and try the SAME one again.

    Telling those two apart is the whole job. Retrying a retired model forever
    is as wrong as abandoning a busy one on its first refusal.
    """

    def __init__(self, client, names, retries=4):
        self.client = client
        self.names = names
        self.retries = retries
        self.index = 0
        self.truncated = False      # was the last answer cut off?

    @property
    def current(self):
        return self.names[self.index] if self.index < len(self.names) else None

    def generate(self, prompt, max_output_tokens=MAX_OUTPUT_TOKENS):
        while self.index < len(self.names):
            name = self.names[self.index]
            for attempt in range(1, self.retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=TEMPERATURE,
                            max_output_tokens=max_output_tokens,
                        ),
                    )
                    self.truncated = self._was_cut_off(response)
                    return (response.text or "").strip()
                except Exception as error:            # noqa: BLE001
                    message = str(error)
                    if not any(t in message for t in TRANSIENT):
                        print(f"    {name}: not available, moving on")
                        break
                    if any(m in message for m in DAILY_QUOTA):
                        print(f"    {name}: DAILY QUOTA spent. Retrying cannot "
                              f"help until Google resets it, moving on")
                        break
                    if attempt == self.retries:
                        print(f"    {name}: still busy after {self.retries} "
                              f"attempts, moving on")
                        break
                    wait = 2 ** attempt
                    print(f"    {name}: busy, retrying in {wait}s "
                          f"[{attempt}/{self.retries}]")
                    time.sleep(wait)
            self.index += 1

        raise SystemExit("Every Flash model was unavailable. Try again later.")

    @staticmethod
    def _was_cut_off(response):
        """
        Did the model stop because it ran out of room, rather than because it
        had finished? A truncated answer looks like a bad answer, and blaming
        the model for a budget you set is the wrong lesson to draw.
        """
        try:
            reason = str(response.candidates[0].finish_reason)
        except Exception:                        # noqa: BLE001
            return False
        return "MAX_TOKEN" in reason.upper()


# ---------------------------------------------------------------------------
# The prompt. Deliberately naive.
# ---------------------------------------------------------------------------

NAIVE_PROMPT = """You are an assistant that answers questions about Canadian
environmental regulations.

Context:
{context}

Question: {question}

Answer:"""


def format_evidence(chunks, rows):
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk['citation']}]\n{chunk['text']}")
    for row in rows:
        parts.append("[Schedule 1]\n" + describe(row))
    return "\n\n".join(parts)


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    client = genai.Client(api_key=key)
    candidates = flash_models(client)
    pool = ModelPool(client, candidates)
    print(f"Flash models listed : {len(candidates)}")
    print(f"Will try in order   : {', '.join(n.split('/')[-1] for n in candidates[:4])}")
    print(f"Temperature         : {TEMPERATURE}")

    schedule1 = load_schedule1()
    chunks, matrix, index_meta = load_index()
    bm25, _ = build_bm25(chunks)
    law_pool = pool_indices(chunks, law=True)
    guidance_pool = pool_indices(chunks, law=False)

    for question in QUESTIONS:
        decision, reasons = route(question, schedule1)

        rows = []
        evidence = []
        if decision in ("LOOKUP", "BOTH"):
            rows, _ = lookup(schedule1, question)
        if decision in ("RETRIEVE", "BOTH"):
            evidence = retrieve(chunks, matrix, bm25, client, index_meta,
                                law_pool, guidance_pool, question)

        context = format_evidence(evidence, rows)

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print(f"   route: {decision}   evidence: {len(evidence)} chunks, "
              f"{len(rows)} table row(s)")
        print("=" * 96)
        answer = pool.generate(
            NAIVE_PROMPT.format(context=context, question=question))
        print(f"   model: {pool.current}\n")
        print(answer)

    print("\n" + "=" * 96)
    print("NOW READ WHAT IT WROTE, LOOKING FOR FOUR THINGS")
    print("=" * 96)
    print("  1. KNOWLEDGE WE DID NOT SUPPLY")
    print("     Anything in the answer that is not in the evidence. It may even")
    print("     be correct - that is what makes it dangerous. Correct-but-")
    print("     ungrounded cannot be checked, and cannot be defended.")
    print()
    print("  2. CITATIONS")
    print("     Does every claim carry one? Is every citation one we actually")
    print("     supplied? A model will happily write 'section 12(3)' because it")
    print("     LOOKS like a citation.")
    print()
    print("  3. THE ONTARIO QUESTION")
    print("     The evidence covers using a plan prepared for another government.")
    print("     It never mentions Ontario or compliance approvals. Does the answer")
    print("     say what it cannot answer, or does it answer anyway?")
    print()
    print("  4. THE PENALTY QUESTION")
    print("     Penalties are in CEPA Part 10. Part 10 is NOT in our corpus. The")
    print("     only correct answer is that the corpus does not cover it. Watch")
    print("     what it does instead.")
    print()
    print("  Whatever you find here is the specification for Steps 3, 4 and 5.")