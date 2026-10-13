# HR RAG

Ask questions about HR documents (handbook, policies, directory, holiday calendar, ...) and get
answers that are **grounded in the documents, cited, and respect who is allowed to see what**.

```
15 files  ->  load  ->  clean  ->  validate  ->  dedupe  ->  metadata  ->  chunk  ->  embed  ->  Chroma
question  ->  permission filter  ->  vector + keyword search  ->  fuse (RRF)  ->  rerank  ->  LLM  ->  checks
```

## 1. Setup (about 10 minutes, once)

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Create your own `.env` (this file is git-ignored, never share it):

```bash
# Windows:        copy .env.example .env
# macOS / Linux:  cp .env.example .env
```

Open `.env` and fill in **your own** key: `ANTHROPIC_API_KEY` (if `LLM_PROVIDER=claude`) or
`NVIDIA_API_KEY` (if `LLM_PROVIDER=nvidia`). Do not reuse a key that was ever pasted in a notebook or chat.

**Using NVIDIA Nemotron (a reasoning model):** keep `NVIDIA_ENABLE_THINKING=false` in `.env`. Without it the model
spends its token budget "thinking" and may return notes or nothing. The app now stops with a clear message instead of
showing reasoning notes or crashing. Do not put quotes or spaces around values (`NVIDIA_MODEL=nvidia/...`).

## 2. Run

```bash
python ingest.py              # builds the index in ./chroma_db  (re-run whenever data/raw changes)
python ask.py                 # chat in the terminal, as an "employee"
python ask.py --role hr       # HR can also see confidential files (employee profiles)
python ask.py --include-archived   # also search old policy versions (e.g. Leave Policy v1)
python ask.py -q "How many casual leave days do I get?"
```

The first `ingest.py` / `ask.py` run downloads two models from Hugging Face (embeddings ~90 MB,
reranker ~1 GB). After that it works offline except for the LLM call.

`python ingest.py --dry-run` checks your data (loading, validation, duplicates, chunk count) without
downloading any model.

**Flags go on the command line, not inside the question.** Typing "Run with: python ask.py --role hr" into the
question does nothing. Start the program that way instead (`python ask.py --role hr`). Each question's role is saved in
`logs/query_log.jsonl`, so you can check which role an answer was produced with.

## 3. Project layout

```
hr-rag/
├── .env.example          template for your secrets (copy to .env)
├── data/
│   ├── raw/              the 15 source files
│   └── manifest.yaml     per-file access level, version, archived/excluded   <-- edit this, not code
├── hr_rag/
│   ├── config.py         all settings; reads .env
│   ├── loaders.py        PDF / DOCX / TXT / HTML / CSV / JSON / XLSX -> Documents
│   ├── manifest.py       reads manifest.yaml
│   ├── cleaning.py       text cleaning, validation, duplicate removal
│   ├── enrichment.py     summary / keywords / tags
│   ├── chunking.py       splitting into chunks
│   ├── vectorstore.py    embeddings + Chroma
│   ├── ingestion.py      runs the whole data-preparation pipeline
│   ├── retrieval.py      query rewrite, permissions, hybrid search, rerank, context
│   ├── generation.py     prompts + Claude / NVIDIA provider layer
│   ├── guardrails.py     access + citation checks
│   ├── evaluation.py     grounding scores, feedback, improvement hints
│   └── pipeline.py       RAGPipeline.ask(): ties query time together
├── ingest.py             command: build the index
├── ask.py                command: ask questions
├── tests/                69 offline tests (no models, no API calls):  python -m pytest
├── chroma_db/            generated index (git-ignored)
└── logs/                 ingestion report, query log, feedback (git-ignored)
```

## 4. How multiple files are handled

| Concern | What happens |
|---|---|
| Different formats | One loader per type. Prose is split at headings; CSV/JSON give one chunk-source per record; Excel gives one per sheet. |
| Who may see what | `manifest.yaml` sets `access_level` per file. `employee_profiles.json` is `confidential` (HR only). The filter runs **inside the search**, so unauthorized text never reaches the LLM. |
| Old policy versions | `leave_policy_v1.docx` is `is_current: false`. It is hidden unless you pass `--include-archived`, and answers label it "ARCHIVED". |
| Duplicates | `duplicate_leave_policy.docx` is `exclude: true`. Exact-text duplicates are also removed automatically (current files win over archived ones). |
| Bad records | `incomplete_employee_record.json` fails validation (no employee ID). Because it exists to be asked about, `manifest.yaml` has `keep_if_invalid: true`: it is indexed, marked `validation_status=invalid`, and HR-only. Other invalid records are skipped. See `logs/ingestion_report.json`. |
| New file added | Put it in `data/raw/`, add an entry in `manifest.yaml`, run `python ingest.py`. A file missing from the manifest is treated as `confidential` until you classify it. |

## 5. What changed from your notebooks (and why)

* **No more `unstructured`.** It is a heavy install (needs extra system tools on Windows). Sections are now
  found with `pypdf`, `python-docx`, `openpyxl` and `beautifulsoup4`. If you prefer Unstructured, only `loaders.py` changes.
* **Keyword search bug fixed.** LangChain's BM25 default does not lower-case or strip punctuation, while your
  query was lower-cased, so capitalised words never matched. `retrieval.tokenize` fixes this (plus simple plural matching).
* **Abbreviation expansion keeps the abbreviation** (`wfh` -> `wfh (work from home)`), otherwise "HR Manager" can no longer be matched.
* **Hard-coded filter removed.** The notebook filtered on `document_type == "employee_handbook"`; with many files that would hide everything else.
* **Chunk headers.** Each chunk starts with `[Document title | Section]`, so a chunk like "Annual Leave: 20 days" knows which policy it belongs to.
* **Tables keep their column names** (`Leave Type: Annual Leave | V1: 18 Days | V2: 20 Days`) and dates get month names, so "holidays in January" can match `2025-01-14`.
* **Running page headers/footers and table-of-contents lines are removed** from PDFs.
* **Context building** stops adding sources when the character budget is full, so citations never point to text the model did not see.
* **Semantic grounding** compares the answer with each source separately (the MiniLM model only reads ~256 tokens, so one big context was silently cut off).
* **Answer checks.** The quality check now also flags numbers in the answer that appear in none of the sources
  (for example an invented "26 weeks"). Multi-part questions are split and each part is searched separately.
* **Index rebuilds are clean.** `ingest.py` wipes the old collection first so deleted files cannot linger.

## 6. Known limitations

Tested end to end with NVIDIA Nemotron 3.5 Lightning (thinking off) on the 15-file dataset. Facts that sit in one
place (policy numbers, job requirements, validation errors, onboarding tasks, contact details) are answered
accurately and with citations. Known weak spots:

* **Counting and totals are unreliable.** "How many employees are in Data Science?" needs all 30 directory rows, but
  retrieval returns only a few passages, so the assistant can report a wrong count (the directory has 5 Data Science
  rows, the department sheet says 4, and the assistant answered 2).
* **"List all N items" questions can come back incomplete or padded.** Always check the count against the document.
* **The quality check measures grounding, not completeness.** A score of 100% does not prove an answer is complete or
  that the model's own explanations (for example why a date is invalid) are right.
* **Roles are simulated.** `--role hr` is only for learning. A real app must take the role from a login system.
* **Quality thresholds** in `config.py` are teaching values. Tune them on real questions.
* **Costs:** every question makes one LLM call, billed to your API key.

## 7. Data

All files in `data/raw/` are synthetic training data (no real people or company). Replace them with your own
documents and update `data/manifest.yaml` to use the project on other material.