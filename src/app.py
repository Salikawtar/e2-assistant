"""
Stage 8 - the page a person who is not you can use.

WHAT THIS IS FOR

Everything built in Stages 1 to 7 exists as Python files run from a terminal.
Nobody can see it work. This page is the demonstration: a question goes in, and
a checkable answer comes out.

THE DESIGN RULE, AND IT IS THE WHOLE POINT

Stage 7 listed five checks a user must perform before relying on an answer.
An interface that shows a confident paragraph and hides the caveats undoes the
project in one design decision, so four of those five checks are made
unavoidable here:

    the STATUS is the first thing on the page, not a footnote
    every citation is listed with its authority: law, guidance or reference
    the evidence the model was given is one click away, in full
    the limitations are on the page, not in a document nobody opens

The fifth check, opening the provision itself, cannot be done by software. The
page says so rather than pretending otherwise.

Run it from the PROJECT FOLDER, not from inside src:

    streamlit run src/app.py
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
ROOT = SRC.parent

from answer import (load_resources, answer_question, pinned_model,   # noqa: E402
                    TIER_OF_TYPE)

# ---------------------------------------------------------------------------

st.set_page_config(page_title="E2 Assistant", page_icon="•", layout="wide")

TIER_LABEL = {"law": "LAW", "guidance": "GUIDANCE", "reference": "REFERENCE"}

# REFUSED is the system working, so it is not painted red. Red reads as
# "something broke", and an interface whose colour argues with its own caption
# teaches the user to ignore the caption. Only MISSING, which really is a
# failed generation, gets an error.
STATUS_HELP = {
    "ANSWERED": ("The evidence answers the whole question.", "success"),
    "PARTIAL": ("The evidence answers part of it. Read the NOT COVERED line.",
                "warning"),
    "REFUSED": ("These six documents do not answer this question. "
                "That is a correct result, not a failure.", "info"),
    "MISSING": ("The model did not produce a usable answer. Ask again.", "error"),
}

# The model writes its own STATUS and NOT COVERED lines, and clean() keeps them
# so the checks can read them. The page shows both separately at the top, so
# leaving them in the body says everything twice. On a REFUSED answer the body
# is ONLY those two lines, which made a correct refusal look like a crash.
HEADER_LINE = re.compile(r"^\s*(STATUS|NOT COVERED)\s*:.*$",
                         re.IGNORECASE | re.MULTILINE)


def answer_body(text):
    """The answer with its status header removed, or "" if that was all of it."""
    return HEADER_LINE.sub("", text or "").strip()

EXAMPLES = [
    "What is the minimum quantity threshold for anhydrous ammonia?",
    "How often must a facility conduct a simulation exercise?",
    "We store 6 tonnes of anhydrous ammonia at our facility. What do we have to do?",
    "Can we use our corporate emergency response plan as the E2 plan?",
    "What penalty applies if a facility fails to submit a notice on time?",
]

LIMITATIONS = """
This assistant answers questions about a fixed set of six public federal
documents: the Environmental Emergency Regulations 2019, CEPA 1999 sections
193 to 205, the ECCC Technical Guidelines, two ECCC factsheets, and the
published list of hazardous substances. It helps a reader find and check the
relevant provision.

**It is not a compliance determination, it is not legal advice, and it is not
an ECCC product.** It runs on public documents only, through a personal
account, on personal equipment.

Measured on 22 questions used during development it answers 21 to 22
correctly. On 12 questions written afterwards and used once, it answered 12.
Across identical runs its score varies by about one question, **so the same
question can receive a different answer twice.** Of its individual claims, 83%
were confirmed by a separate reader to be supported by the source they cite,
on a sample of 52.

**It has twice named a provision as the source of a duty that provision does
not impose.** For that reason every answer must be checked against the
provision it cites before it is relied on.

The corpus has no process for detecting amendments, so it should not be used
against instruments that may have changed since collection.
"""


# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the corpus, the index and the table...")
def get_resources():
    """
    Built once and kept, rather than rebuilt on every keystroke.

    Streamlit re-runs the whole file each time anything on the page changes.
    Without this, 362 vectors and a BM25 index would be rebuilt every time you
    typed a character.
    """
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("No GEMINI_API_KEY found in .env")
        st.stop()
    return load_resources(genai.Client(api_key=key))


def split_evidence(context):
    """Break the numbered evidence block back into its pieces for display."""
    if not context:
        return []
    return re.split(r"\n\n(?=\[\d+\] )", context)


# ---------------------------------------------------------------------------
# Sidebar: what this is, and what it is not
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("E2 Assistant")
    st.caption("Environmental Emergency Regulations, SOR/2019-51, and related "
               "ECCC guidance.")

    st.subheader("What it does")
    st.write("Finds the provision that bears on your question, quotes it, and "
             "cites it so you can check it.")

    st.subheader("What it will not do")
    st.write("It will not tell you whether your facility is compliant, and it "
             "will not give advice. It has no access to any facility's data.")

    st.subheader("How to read an answer")
    st.markdown(
        "1. Read the **status** first.\n"
        "2. Check whether a duty rests on **LAW** or on **GUIDANCE**.\n"
        "3. Open the cited provision before acting on any number or deadline.\n"
        "4. Ask twice when it matters. The answer can change.")

    st.divider()
    st.caption(f"model: {pinned_model() or 'auto'}   ·   temperature 0")


# ---------------------------------------------------------------------------
# The question
# ---------------------------------------------------------------------------

st.title("E2 Assistant")
st.caption("Ask a question about the Environmental Emergency Regulations. "
           "Every answer carries its status, its sources and the evidence "
           "behind it.")

if "question" not in st.session_state:
    st.session_state.question = ""

st.write("**Try one of these, or write your own:**")
columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES):
    if column.button(example[:34] + "...", help=example,
                     use_container_width=True):
        st.session_state.question = example

question = st.text_area("Your question", key="question", height=90,
                        placeholder="What is the minimum quantity threshold "
                                    "for anhydrous ammonia?")

ask = st.button("Ask", type="primary", disabled=not question.strip())

# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

if ask and question.strip():
    res = get_resources()
    try:
        with st.spinner("Retrieving evidence and writing an answer..."):
            result = answer_question(question.strip(), res)
    except SystemExit as stop:
        st.error(f"The model could not be reached: {stop}")
        st.stop()

    report = result["report"]
    status = report["status"]

    st.divider()

    # ---- 1. status first, where it cannot be missed --------------------
    message, kind = STATUS_HELP.get(status, ("Unrecognised status.", "error"))
    getattr(st, kind)(f"**STATUS: {status}**  ·  {message}")

    if report.get("not_covered"):
        st.warning(f"**NOT COVERED:** {report['not_covered']}")

    # ---- 2. the answer -------------------------------------------------
    body = answer_body(result["answer"])
    if body:
        st.subheader("Answer")
        st.markdown(body.replace("\n", "  \n"))
    elif status == "REFUSED":
        st.subheader("Answer")
        st.markdown("**None given, and that is the right outcome.** The six "
                    "documents this assistant holds do not cover the question, "
                    "so it declined rather than guessing. The sources below are "
                    "what it looked at before declining.")
    else:
        st.subheader("Answer")
        st.warning("The reply contained a status line and nothing else. "
                   "Ask again.")

    # ---- 3. sources, each with its authority ---------------------------
    st.subheader("Sources")
    st.caption("The number in the answer points at the number here. The model "
               "never writes a citation, it writes a number, and code checks "
               "the number exists.")
    for n, source in enumerate(result["sources"], start=1):
        tier = TIER_OF_TYPE.get(source["source_type"], "other")
        label = TIER_LABEL.get(tier, tier.upper())
        # The tier is the thing a reader must not miss, so it is coloured and
        # bold rather than left as small grey code text.
        colour = {"LAW": "#1F3864", "GUIDANCE": "#9C5700",
                  "REFERENCE": "#595959"}.get(label, "#595959")
        st.markdown(
            f"**[{n}]** &nbsp; <span style='background:{colour};color:white;"
            f"padding:2px 8px;border-radius:3px;font-size:0.75em;"
            f"font-weight:700;letter-spacing:0.5px'>{label}</span> &nbsp; "
            f"{source['citation']} &nbsp;·&nbsp; as of "
            f"{source.get('as_of', 'unknown')}",
            unsafe_allow_html=True)

    if any(TIER_OF_TYPE.get(s["source_type"]) == "guidance"
           for s in result["sources"]):
        st.info("Sources marked GUIDANCE explain obligations. They do not "
                "create them. If a duty in this answer rests only on "
                "guidance, find the provision behind it before relying on it.")

    # ---- 4. the evidence, one click away -------------------------------
    with st.expander("See the evidence the model was given", expanded=False):
        st.caption("This is everything it had. If the answer says something "
                   "that is not in here, it was invented.")
        for piece in split_evidence(result.get("context", "")):
            st.text(piece)
            st.divider()

    # ---- 5. what the automatic checks found ----------------------------
    with st.expander("What the automatic checks found", expanded=False):
        cited = report["claims"] - len(report["uncited"])
        a, b, c = st.columns(3)
        a.metric("claims carrying a citation", f"{cited}/{report['claims']}")
        b.metric("invented citations", len(report["invented_markers"])
                 + len(report["invented_refs"]))
        c.metric("duties resting on guidance alone",
                 len(report["obligation_on_guidance"]))
        for sentence in report["obligation_on_guidance"]:
            st.warning(f"States a duty on guidance alone: {sentence}")
        for problem in report["problems"]:
            st.error(problem)
        st.caption("These checks ask whether the answer is CHECKABLE. They do "
                   "not ask whether it is CORRECT. A sentence can carry a "
                   "perfectly valid [3] and still say something [3] does not "
                   "support, which is why step 3 in the sidebar exists.")

    st.caption(f"route: {result['route']}   ·   model: "
               f"{str(result['model']).split('/')[-1]}   ·   "
               f"{len(result['sources'])} sources")

# ---------------------------------------------------------------------------
# Limitations, on the page rather than in a document nobody opens
# ---------------------------------------------------------------------------

st.divider()
with st.expander("Limitations. Read this before relying on anything above.",
                 expanded=False):
    st.markdown(LIMITATIONS)
