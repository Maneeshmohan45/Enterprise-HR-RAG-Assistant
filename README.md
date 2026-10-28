# Enterprise HR RAG Assistant (v1.1.0)

An enterprise-grade, retrieval-augmented generation (RAG) assistant for Human Resources policies, handbooks, organizational rosters, and company benefits. 

The assistant delivers answers that are **strictly grounded in source documents, cited with verifiable source cards, protected by role-based access control (RBAC), and guarded against prompt injections, contradictory premises, and hallucinated policy substitutions**.

```
Query -> Request Guard -> Scope Guard -> Premise & Statement Guard -> Domain Guard 
      -> Hybrid Search (Vector + BM25 with RRF) -> Cross-Encoder Reranker
      -> [Deterministic Roster Engine | Grounded LLM Generation]
      -> Numeric Gate & Grounding Verifiers -> Response Assessment Classifier
      -> Cited Answer + Source Cards
```

---

## Project Status

**Current Release: V1.1.0 — Stable Portfolio Release**

V1.1 is the current tested release of the Enterprise HR RAG Assistant. It focuses on grounded HR question answering, role-based access boundaries, premise and statement protection, deterministic department roster retrieval, citation verification, and defense against prompt injection and semantic policy substitution.

---

## V1.1 Validation

V1.1 was validated using automated testing, mutation testing, and real-model acceptance testing:

- **1,367 automated tests passed**
- **20/20 mutation tests caught**
- **73/74 real-model acceptance checks passed (98.6%)**
- **0 critical issues**
- **0 material issues**
- **0 PII leaks**
- **0 RBAC bypasses**

The single non-passing acceptance case was a conservative refusal on informal leave phrasing and was classified as a minor usability limitation rather than a security or data-integrity issue.

> The deterministic roster engine achieved 100% recall and 100% precision across the V1.1 acceptance dataset covering all 11 departments.

---

## 1. Key Capabilities

* **Multi-Format Ingestion**: Ingests, cleans, deduplicates, validates, and chunks across PDF, DOCX, TXT, HTML, CSV, JSON, and XLSX files.
* **Strict Role-Based Access Control (RBAC)**:
  * `employee`: Has access to general policies, employee handbooks, onboarding guides, and standard approval workflows. Directory employee profiles and confidential files are strictly inaccessible.
  * `hr`: Access to all public, internal, and confidential documents (including employee directory profiles, salary bands, and performance records).
  * *Security Invariant*: Access boundaries are enforced **at the database filter level**—unauthorized chunks never enter the retrieval context or reach the LLM.
* **Deterministic Department Roster Engine (v1.1)**:
  * Department headcount and membership queries (e.g., *"How many employees in Engineering?"* or *"List all members in Marketing"*) bypass stochastic LLM counting.
  * Aggregated deterministically from structured directory records with **100% recall and 100% precision** across the evaluated dataset, accompanied by an explicit data notice.
* **Multi-Layered Security & Guardrail Pipeline**:
  * **Request Guard**: Intercepts prompt injections, system prompt exfiltration attempts, Base64 wraps, XML/tag escapes (`</context>`), and DAN mode jailbreaks.
  * **Scope Guard**: Disambiguates department-head policy approval workflows (permissible for employees) from confidential employee directory lookups (restricted to HR).
  * **Premise & Statement Guard**: Intercepts declarative user statements and refuses queries containing false factual premises or invalid units (e.g., *"Since we get 30 days of annual leave..."*).
  * **Domain Guard**: Refuses general knowledge or non-HR queries before retrieval.
  * **Anti-Semantic Substitution**: Enforces strict policy boundary checks (e.g., questions about pet bereavement or gym subsidies explicitly report absence rather than substituting human bereavement or wellness rules).
* **Automated Grounding & Evaluation**:
  * Every generated answer is audited for lexical grounding, semantic grounding, number consistency (Numeric Gate), citation accuracy, and relevance.
  * Categorized into formal response classes: `NORMAL_GROUNDED`, `ACCESS_RESTRICTED`, `SECURITY_REFUSAL`, `NOT_FOUND`, and `OUT_OF_DOMAIN`.
* **Exhaustive Automated Verification**: Backed by **1,367 automated offline tests** and **20/20 mutation tests** caught.

---

## 2. Setup (About 5–10 Minutes)

### Prerequisites
* Python 3.10+ (tested on Python 3.11 – 3.13)
* Virtual environment tool (`venv`)

### Installation

```bash
# 1. Clone repository & enter workspace
git clone <YOUR_REPO_URL>
cd Enterprise-HR-RAG-Assistant

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create your `.env` from the provided template:

```bash
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env
```

Open `.env` and configure your API credentials and model settings:

* **Claude (Anthropic)**:
  ```ini
  LLM_PROVIDER=claude
  ANTHROPIC_API_KEY=your-anthropic-key-here
  ANTHROPIC_MODEL=claude-haiku-5-5
  ```
* **NVIDIA (Nemotron / OpenAI-compatible)**:
  ```ini
  LLM_PROVIDER=nvidia
  NVIDIA_API_KEY=your-nvidia-key-here
  NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
  NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
  NVIDIA_ENABLE_THINKING=false
  ```
* **Local Models (Hugging Face)**:
  ```ini
  EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
  RERANKER_MODEL=BAAI/bge-reranker-base
  ```

---

## 3. Usage

### 1. Ingest Documents into Vector Store
Rebuilds the ChromaDB vector index from `data/raw/` using `data/manifest.yaml`:

```bash
python ingest.py
```
*(Use `python ingest.py --dry-run` to validate document parsing, schema checks, and chunk counts without downloading embedding models).*

### 2. Interactive Terminal CLI (`ask.py`)
Ask questions from the command line under different roles and parameters:

```bash
# Ask as an employee (default)
python ask.py -q "What is the probation period for full-time employees?"

# Ask as HR (accesses confidential records and employee profiles)
python ask.py --role hr -q "List all employees in the Engineering department"

# Search including archived policy versions
python ask.py --include-archived -q "How did annual leave entitlement change from v1 to v2?"
```

### 3. Streamlit Web UI (`app.py`)
Launch the complete interactive web interface:

```bash
streamlit run app.py
```

The Streamlit UI includes:
* **Assistant Page**: Chat interface with clickable citation source cards, role switcher (`employee` vs `hr`), style toggles (`concise` vs `detailed`), and response feedback buttons (👍 / 👎).
* **Model Evaluation Page**: Live performance dashboards reading `logs/query_log.jsonl`, displaying real response-class distributions, check pass rates, latency breakdowns (Retrieval vs Generation), and token consumption.
* **About Page**: Architecture breakdown, document inventory, and role permission matrices.

---

## 4. Project Layout

```
Enterprise-HR-RAG-Assistant/
├── .env.example                # Template for environment variables and API keys
├── .gitignore                  # Git ignore rules (protects .env, chroma_db, and logs)
├── app.py                      # Interactive Streamlit Web UI
├── ask.py                      # Terminal CLI query interface
├── ingest.py                   # Ingestion pipeline: extracts, validates, and indexes chunks
├── requirements.txt            # Project dependencies
├── data/
│   ├── raw/                    # Raw HR documents (PDF, DOCX, CSV, JSON, XLSX, HTML, TXT)
│   └── manifest.yaml           # Metadata, access levels, versioning, and exclusion rules
├── hr_rag/
│   ├── config.py               # Pydantic configuration and environment loader
│   ├── loaders.py              # Multi-format document loaders and structural parsers
│   ├── manifest.py             # Manifest loader and schema validator
│   ├── cleaning.py             # Text sanitization, deduplication, and anomaly checks
│   ├── enrichment.py           # Metadata tagging, abbreviation expansion, and summarization
│   ├── chunking.py             # Section-aware chunking with document headers
│   ├── vectorstore.py          # ChromaDB collection management and embedding interface
│   ├── ingestion.py            # End-to-end ingestion pipeline runner
│   ├── retrieval.py            # Hybrid search (Dense + BM25), RRF fusion, and Cross-Encoder reranking
│   ├── generation.py           # Provider LLM client, prompts, and anti-substitution rules
│   ├── request_guard.py        # Injection detection, DAN mode, and tag-escape interceptor
│   ├── scope.py                # Scope guard for department-head approval vs directory access
│   ├── premise_guard.py        # False-premise, invalid unit, and statement assertions guard
│   ├── domain_guard.py         # Out-of-domain query classification and refusal
│   ├── roster.py               # Deterministic department headcount and roster engine
│   ├── numeric_gate.py         # Numeric grounding and claim verification
│   ├── citations.py            # Verifiable citation extractor and card formatter
│   ├── grounding.py            # Lexical and semantic cross-source grounding verifier
│   ├── assessment.py           # Response class assessment and contract auditor
│   ├── guardrails.py           # Composite verification orchestrator
│   ├── streaming.py            # SSE token streaming generator
│   ├── pipeline.py             # RAGPipeline: central controller tying all stages together
│   └── ui_data.py              # Log aggregation, KPI calculation, and role-based log redaction
├── scripts/
│   ├── calibrate_relevance.py  # Utility to calibrate reranker thresholds
│   └── v1_acceptance.py        # Acceptance benchmark suite for release validation
└── tests/                      # 1,367 automated offline unit, integration, and security tests
```

---

## 5. Security & Invariant Defenses

| Barrier Layer | Mechanism | Protection |
|---|---|---|
| **Phase 1 RBAC** | Hard metadata filtering in ChromaDB (`where={"access_level": ...}`) and BM25 index | Confidential employee files (PII, salaries, ratings) can never be retrieved by the `employee` role. |
| **Request Guard** | Regex & token inspection | Blocks jailbreaks (`DAN mode`), prompt exfiltration, Base64-wrapped payloads, and `</context>` tag escapes. |
| **Scope Guard** | Semantic intention matching | Permits policy inquiries referencing "department head approval" for employees while strictly blocking personal employee directory lookups. |
| **Premise Guard** | Factual assertion & unit matcher | Prevents the model from accepting or validating false user premises (e.g. invalid days of leave) or declarative non-question assertions. |
| **Deterministic Roster** | Exact directory aggregation | Eliminates hallucinations and incomplete counting for department rosters and team member listings. |
| **Anti-Substitution** | Generation-level accuracy rule | Ensures unmentioned benefits (e.g., pet bereavement, gym subsidies) are explicitly reported as absent rather than substituted with related policies. |
| **Numeric Gate** | Regex & number-span verification | Flags or withholds any response containing numerical quantities not grounded in source chunks. |

---

## 6. Testing & Quality Verification

Run the complete test suite:

```bash
# Run all 1,367 offline tests
python -m pytest

# Run specific security invariant suites
python -m pytest tests/test_v1_security_invariants.py
python -m pytest tests/test_scope_department_head.py
python -m pytest tests/test_premise_guard.py
python -m pytest tests/test_roster.py
```

---

## 7. Operational Notes & Limitations

> **Portfolio/Demonstration Authentication:** V1.1 uses a simulated Employee/HR role selector so the complete RBAC behavior can be demonstrated and evaluated from both perspectives. This role selector is not a production authentication mechanism. A production deployment would derive the user's identity and authorization server-side through an authenticated identity provider.

* **Independent Question Context**: Each request is evaluated independently to maximize grounding precision; multi-turn conversational memory is isolated per query.
* **Corpus Ingestion**: The index must be rebuilt (`python ingest.py`) whenever files in `data/raw/` or configurations in `data/manifest.yaml` are updated.

---

## 8. V1.2 — Production Hardening

Planned improvements:

- **Authentication-based RBAC** — replace the simulated role selector with authenticated identity and server-side authorization.
- **LLM timeout and retry handling** — prevent provider stalls from leaving requests waiting indefinitely.
- **Sensitive-log protection** — improve protection, access control, and retention of employee queries and potentially sensitive logs.
- **Enhanced observability** — record provider/model information and additional operational metadata for troubleshooting.
- **Relevance-gate calibration** — reduce unnecessary refusals on legitimate HR questions using real evaluation data.
- **Controlled pilot monitoring** — monitor refusal rates, latency, errors, and other production-quality signals during a limited rollout.

---

## 9. V2 — Next-Generation Enterprise Assistant

Planned improvements:

1. **Real authentication and authorization**
   - Login-based identity.
   - Server-side role assignment.
   - Stronger enterprise RBAC.

2. **Better evaluator calibration**
   - More reliable evaluation thresholds.
   - Reduced false positives and false negatives.

3. **Separate scoring by response class**
   - `NORMAL_GROUNDED`
   - `NOT_FOUND`
   - `OUT_OF_DOMAIN`
   - `SECURITY_REFUSAL`
   - `ACCESS_RESTRICTED`
   - Other applicable response classes.

4. **Better semantic grounding measurement**
   - Evaluate whether the meaning of an answer is actually supported by retrieved sources, not only whether words overlap.

5. **Better citation-support measurement**
   - Verify that a cited source actually supports the claim made in the answer.

6. **Larger benchmark dataset**
   - More HR documents.
   - More employees and departments.
   - More normal, adversarial, out-of-corpus, and edge-case questions.
   - Broader evaluation coverage.

7. **Stronger security**
   - Additional adversarial testing.
   - Stronger prompt-injection defenses.
   - Expanded authorization and sensitive-data protections.

8. **Lower latency**
   - Optimize retrieval and reranking.
   - Reduce unnecessary LLM calls.
   - Improve startup/model initialization time.
   - Expand deterministic paths where appropriate.

9. **Controlled conversational context**
   - Introduce multi-turn context while preserving grounding, authorization, and security boundaries.
