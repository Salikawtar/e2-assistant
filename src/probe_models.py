"""
Stage 6 - which models will actually answer right now, and if not, why not.

WHY THIS EXISTS

ModelPool prints "busy, retrying" for six different errors:

    503 UNAVAILABLE          overloaded right now. Waiting works.
    429 RESOURCE_EXHAUSTED   a quota is spent. Waiting an hour does nothing,
                             and on a daily quota, nothing works until it
                             resets.
    500 INTERNAL             a fault at the other end.

Those need opposite responses, and the word "busy" hides which one you have.
Three runs were spent waiting on an error that may never have been going to
clear. This file prints the real code.

WHAT IT CANNOT TELL YOU

A small request is not proof of capacity for a large one. Stage 4 already
learned this the hard way: a one-word ping succeeded against a model that then
returned 503 on the real seven-thousand-character request.

So read the result this way:

    an ERROR here is conclusive   - a model that refuses a tiny request will
                                    certainly refuse a real one, and the code
                                    tells you whether waiting helps
    an OK here is only a hint     - it means the model is reachable and the
                                    quota is not exhausted, not that it will
                                    carry 22 full questions

Run:  python src/probe_models.py
"""

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from generate_naive import flash_models

# Small enough to cost almost nothing, long enough to be a real request.
PROBE = "Reply with the single word: ready"

CODES = [
    ("429", "RESOURCE_EXHAUSTED", "a quota is spent. Retrying will not help. "
                                  "If it is a daily quota, it resets on Google's "
                                  "clock, not yours."),
    ("503", "UNAVAILABLE", "the model is overloaded right now. Waiting and "
                           "retrying is the correct response."),
    ("500", "INTERNAL", "a fault at Google's end. Retrying is reasonable."),
    ("404", "NOT_FOUND", "this model is not available to this key at all. "
                         "Never retry it."),
    ("403", "PERMISSION", "this key may not use this model."),
    ("400", "INVALID", "the request itself was rejected, not the quota."),
]


def explain(message):
    """Name the failure, using the first code that appears in the message."""
    for code, name, meaning in CODES:
        if code in message or name in message:
            return f"{code} {name}", meaning
    return "unrecognised", "not one of the errors this project has seen before"


if __name__ == "__main__":
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("No GEMINI_API_KEY found in .env")

    client = genai.Client(api_key=key)
    names = flash_models(client)

    print(f"  {len(names)} Flash model(s) listed for this key, newest first\n")
    print("=" * 100)

    working = []
    for name in names:
        short = name.split("/")[-1]
        try:
            response = client.models.generate_content(
                model=name,
                contents=PROBE,
                config=types.GenerateContentConfig(
                    temperature=0, max_output_tokens=2000),
            )
            text = (response.text or "").strip().replace("\n", " ")[:40]
            print(f"  {short:<32} OK        {text}")
            working.append(name)
        except Exception as error:                       # noqa: BLE001
            message = str(error)
            label, meaning = explain(message)
            print(f"  {short:<32} {label}")
            print(f"  {'':<32} {meaning}")
            # The API often names the exact quota that was hit. That line is
            # the difference between "wait a minute" and "wait until tomorrow".
            for hint in re.findall(
                    r"(quota[^,\"}]{0,120}|retryDelay[^,\"}]{0,40}|"
                    r"limit[^,\"}]{0,80})", message, re.IGNORECASE)[:3]:
                print(f"  {'':<32} -> {hint.strip()}")
        print()

    print("=" * 100)
    print("WHAT TO DO")
    print("=" * 100)
    if not working:
        print("  Nothing answered even a five-word question. This is not your")
        print("  code and not question S4. The key cannot generate right now.")
        print("  Read the codes above:")
        print("    all 429  -> a quota is spent. Come back when it resets.")
        print("    all 503  -> overload. Try again in an hour.")
    elif len(working) == 1:
        print(f"  One model answered: {working[0].split('/')[-1]}")
        print("  Pin that one and re-run. In .env, add the line:")
        print(f"      E2_MODEL={working[0].split('/')[-1]}")
        print()
        print("  Changing the pin throws away the answers collected under the")
        print("  old pin. That is deliberate. Mixing two models inside one")
        print("  baseline is the exact contamination pinning exists to remove.")
    else:
        print("  These answered:")
        for name in working:
            print(f"    {name.split('/')[-1]}")
        print()
        print("  Pin ONE of them in .env, for example:")
        print(f"      E2_MODEL={working[0].split('/')[-1]}")
        print()
        print("  Prefer the one that answers most reliably over the one that is")
        print("  newest. For a baseline, being able to finish 22 questions")
        print("  matters more than being clever.")
