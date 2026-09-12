# E2 Assistant

A retrieval augmented generation (RAG) system over Canada's **Environmental
Emergency Regulations, SOR/2019-51** and the ECCC guidance that accompanies
them. You ask a question in plain English, it finds the provision that bears on
it, quotes it, and cites it so you can check the answer yourself.

This is personal learning work, built on published documents, on personal
equipment and a personal account. It is not an ECCC product and nothing in it
touches departmental data, networks or systems.

---

## What it does, and what it will not do

**It will**

- find the provision that answers your question and quote its exact words
- cite every claim by a number that points at evidence it was actually given
- tell you whether a duty rests on the **law** or on **guidance**
- say plainly when these six documents do not answer your question

**It will not**

- tell you whether a facility is compliant. It has no access to facility data
- give legal advice or a departmental interpretation
- be trusted without you opening the provision it cites

> **The sentence that should travel with it:** this assistant helps you find and
> read the right provision. It does not tell you what you must do, and nothing
> it produces is a compliance determination.

---

## Quick start

You need Python 3.11 or later and a Google Gemini API key.

```bash
git clone <this repository>
cd e2-assistant

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS or Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
E2_MODEL=gemini-3.8-flash
```

`.gitignore` already excludes `.env`. Check that it is not tracked before your
first commit:

```bash
git ls-files --error-unmatch .env    # should say the file is not tracked
```

Then start the page, **from the project folder and not from inside `src`**:

```bash
streamlit run src/app.py
```

---

## The corpus

Six public documents, 362 chunks and one 249 row table.

| Source | Type | As of |
|---|---|---|
| Environmental Emergency Regulations, SOR/2019-51 | regulation | 2021-10-20 |
| CEPA 1999, sections 193 to 205 | act | 2021-10-20 |
| Technical Guidelines for the E2 Regulations (v2.0) | guidance | 2020-12 |
| ECCC factsheet: Reporting an Environmental Emergency | guidance | undated |
| ECCC factsheet: simulation exercises | guidance | undated |
| ECCC list of hazardous substances | reference | 2016-01-14 |

Guidance outweighs law by 6.2 to 1 in volume, which is why retrieval gives law a
guaranteed share of the evidence rather than letting the two compete.

---

## How it works

```
question
   |
   +-- router: is this a table lookup, a search, or both?
   |
   +-- LOOKUP  -> Schedule 1 read as a table, by CAS, UN number or name
   |              thresholds are never searched or approximated
   |
   +-- RETRIEVE -> vector search + BM25, fused by rank (RRF)
   |               law and guidance ranked in separate pools
   |               3 law slots, 2 of them reserved for the vector ranking
   |               3 guidance slots
   |
   +-- generate: one pinned model, temperature 0, seven prompt rules
   |
   +-- check in code: every citation marker must exist
   |                  every claim must carry one
   |                  the authority tier is computed, never declared
   |
   +-- answer with STATUS: ANSWERED | PARTIAL | REFUSED
```

### Five design decisions worth knowing

**1. The model cites by number, not by name.** Evidence arrives as `[1]`, `[2]`,
`[3]`. A model can invent a convincing "section 12(3)". It cannot invent `[9]`
when only six pieces of evidence exist, and that is a one line check. Zero
invented citations across every run.

**2. Thresholds are read, not searched.** A quantity threshold is a cell in a
table. Embeddings are poor at exact numbers, and "Ammonia, anhydrous" at 4.50
tonnes sits beside "Ammonia solution" at 9.10 tonnes under the same CAS number.
Vector search, keyword search and the law quota all failed that question.
A table lookup answers it exactly or says the substance is not listed.

**3. Law gets a guaranteed share, and meaning gets a reserved seat.** A quota is
a policy you can defend. A score multiplier is a number nobody can justify.
Two of the three law slots go to the top of the vector ranking, because on this
corpus the keyword half of the fusion ranked the correct provision 10th, 20th
and 21st while the vector half ranked it 1st or 2nd.

**4. Abstention is declared, so it can be counted.** Refusal buried in prose
cannot be measured. A `STATUS` line means both failure directions are
countable: answering what should be refused, and refusing what could be
answered.

**5. The authority tier is computed by code.** The model never says whether
something is law or guidance. `source_type` decides it. A label the model
declares is one more thing to verify.

---

## Results

Every figure carries its sample size. None is a population estimate.

| Measure | Result | Sample |
|---|---|---|
| questions passing every check, tuned set | 21 to 22 out of 22 | 22 questions, three runs |
| questions passing, held-out set | 12 out of 12 | 12 questions, run once |
| under-abstention | 0 | 22 questions, three runs |
| over-abstention | 0 | same |
| claims carrying a citation | 72 of 74 (97%) | one run |
| claims the cited source supports | 43 of 52 (83%) | one run, judge v3 |
| invented citations | 0 | every run in Stages 5 and 6 |
| judge accuracy on planted errors | 8 of 8 caught, 6 of 6 clean left alone | 14 control cases |
| run to run variance | 1 question | three runs, identical configuration |

**The held-out set is the number that matters.** Twelve questions on provisions
the tuned set never touched, written after every fix was finished and used once.
If they had scored badly, that would be the headline instead.

**The judge was itself tested.** Citation faithfulness is measured by a second
model reading each claim against the evidence it cites. That judge was checked
against fourteen claims broken on purpose: a changed figure, a changed time
unit, a reversed duty, a swapped source. It caught all eight and passed all six
clean ones.

### Reproducing the evaluation

```bash
python src/eval_questions.py      # print the answer key, no API calls
python src/evaluate.py            # run all 22 and score them
python src/variance.py            # run the same set three times
python src/holdout.py             # the held-out questions, once
python src/judge.py               # citation faithfulness
python src/judge_control.py       # test the judge against planted errors
```

`evaluate.py` saves after every question and resumes, and scoring is recomputed
from saved answers, so a fix to the checking code costs nothing to verify.

---

## Known defects

These are measured, not suspected.

| Defect | Detail |
|---|---|
| **Misattributed authority** | Twice it has named a provision as the source of a duty that provision does not impose. It attributed a CEPA 201(1)(a) reporting duty to section 18 of the Regulations, and attached Schedule 5 items belonging to paragraph 7(1)(b) to paragraph 7(1)(a). Both answers were fluent, carried real citations, and passed every automated check. |
| **Duties stated on guidance** | Eight sentences state a legal duty while citing ECCC guidance alone. All eight restate duties the law does impose, usually law that was sitting in the same evidence, so the system is not inventing obligations. It is citing the wrong authority for real ones. |
| **Inconsistent answers** | Two questions in 22 give a different status between identical runs. Temperature 0 does not make a thinking model deterministic. |
| **No corpus currency check** | Nothing detects an amendment. A superseded provision answered with a perfect citation looks exactly like a correct answer. |
| **English only** | The Regulations exist in both official languages. This has never been tested in French. |

---

## Limitations

This assistant answers questions about a fixed set of six public federal
documents. It helps a reader find and check the relevant provision.

**It is not a compliance determination, it is not legal advice, and it is not an
ECCC product.** It runs on public documents only, through a personal account, on
personal equipment.

Measured on 22 questions used during development it answers 21 to 22 correctly.
On 12 questions written afterwards and used once, it answered 12. Across
identical runs its score varies by about one question, **so the same question can
receive a different answer twice**. Of its individual claims, 83% were confirmed
by a separate reader to be supported by the source they cite, on a sample of 52.

**It has twice named a provision as the source of a duty that provision does not
impose.** For that reason every answer must be checked against the provision it
cites before it is relied on.

The corpus has no process for detecting amendments, so it should not be used
against instruments that may have changed since collection.

---

## Project structure

```
data/
  raw/                  the six source documents
  corpus_manifest.csv   the single source of truth for what is in the corpus
  chunks.jsonl          362 chunks with citation, source_type and as_of
  schedule1.csv         249 substances with thresholds and hazard categories
  vectors/              the embedding index, with a hash of chunks.jsonl

eval/
  *_run.json            answers, one file per experiment
  *_results.csv         scored results for review in Excel
  *_judge.json          citation faithfulness verdicts
  *_control.json        the planted error control set

src/
  corpus.py parsing*.py schedule1.py build_chunks.py    Stage 1 and 2
  embed_corpus.py search_vectors.py search_keyword.py   Stage 3
  search_hybrid.py search_split.py lookup_schedule1.py  Stage 3
  router.py
  generate_*.py answer.py                               Stage 4
  eval_questions.py evaluate.py judge.py agreement.py   Stage 5
  judge_control.py
  diagnose.py diagnose_retrieval.py variance.py         Stage 6
  probe_models.py holdout_questions.py holdout.py
  app.py                                                Stage 8
```

---

## How it was built

Nine stages, each one ending in a written record of what was decided and why.

| Stage | Subject |
|---|---|
| 1 | Corpus and manifest |
| 2 | Parse and chunk |
| 3 | Embed and retrieve |
| 4 | Generate, cite, abstain |
| 5 | Evaluation |
| 6 | Diagnose and improve |
| 7 | Responsible AI review |
| 8 | Interface and documentation |
| 9 | Presentation |

The recurring lesson, recorded because it kept happening: **check the instrument
before you trust what it says about the machine.** Eleven times in this project,
a number that looked like a fault in the system turned out to be a defect in the
code measuring it. The 72% citation rate that turned out to be 92%, the
"unanswerable" questions that were answered by a table rather than by search, and
the judge that scored 78%, then 60%, then 83% on the same answers because the
unit of judgement kept changing.

---

## Cost and accounts

Runs on a personal Google account with prepaid Gemini API credits. A full
evaluation cycle of about 85 requests costs roughly 60 cents. The free tier was
used through Stage 5 and its daily quota stopped a 22 question evaluation
partway through, three runs in a row, which is recorded in the Stage 6 document
as a constraint rather than an inconvenience.

Paid terms also changed what can honestly be said about the pipeline. Google's
API terms state that on the unpaid tier content is used "to provide, improve, and
develop Google products and services", and that on the paid tier "Google doesn't
use your prompts or responses to improve our products". The corpus is public, so
nothing was ever at risk, but the distinction matters for anyone reading this as
a pattern for real work.

---

## Licence and use of source material

The source documents are published Government of Canada material and remain
subject to their own terms. This repository contains the code and the derived
index, not a republication of the instruments.
