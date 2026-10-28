"""Tests for V1.1 Fix 3: department roster and headcount handling.

Verifies:
1. Ingestion: 11 department roster chunks + 1 summary chunk created and grounded in employee_directory.csv.
2. Planner unit tests: intent detection, positives, negatives, stripped headers, aliases, notices.
3. Pipeline end to end (role hr): all 11 departments (H1-01..H1-11) and headcounts (H1-12..H1-15);
   scripted LLM is never called; 100% precision and recall; summary company-wide answers.
4. Partial roster honesty: roster_notice rendered when answered from individual rows only.
5. Access invariants: Employee role never retrieves roster chunks or canary strings; Phase 1 intact.
6. Modes: enforce, shadow, off, unknown mode raises ValueError.
7. Streaming: deterministic answer delivered in single delta without calling stream LLM.
8. Individual employee queries still use the LLM path without notice.
"""
from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import pytest
from langchain_core.documents import Document

from hr_rag.config import PROJECT_ROOT, Settings
from hr_rag.ingestion import prepare_chunks
from hr_rag.pipeline import RAGPipeline
from hr_rag.retrieval import HybridRetriever, allowed_levels_for
from hr_rag.roster import (
    DIRECTORY_DOCUMENT_TYPE, KNOWN_DEPARTMENTS, ROSTER_HEADER, ROSTER_NOTICE,
    SUMMARY_HEADER, RosterPlan, build_roster_chunks, parse_roster_text,
    parse_summary_text, plan_roster_answer, roster_intent
)
from hr_rag.ui_data import read_jsonl

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "employee_directory.csv"


def get_csv_rows() -> list[dict[str, str]]:
    with open(CSV_PATH, mode="r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_csv_by_department() -> dict[str, list[dict[str, str]]]:
    by_dept: dict[str, list[dict[str, str]]] = {}
    for row in get_csv_rows():
        by_dept.setdefault(row["department"].strip(), []).append(row)
    return by_dept


DEPARTMENTS_11 = [
    "Engineering", "Human Resources", "Support", "Management",
    "Data Science", "Quality Assurance", "DevOps", "Finance",
    "Marketing", "Sales", "Administration"
]

H1_QUESTIONS = {
    "H1-01": ("Engineering", "Please list the names, designations, and employee IDs of all employees in the Engineering department."),
    "H1-02": ("Data Science", "Please list the names, designations, and employee IDs of all employees in the Data Science department."),
    "H1-03": ("Human Resources", "Please list the names, designations, and employee IDs of all employees in the Human Resources department."),
    "H1-04": ("Quality Assurance", "Please list the names, designations, and employee IDs of all employees in the Quality Assurance department."),
    "H1-05": ("DevOps", "Please list the names, designations, and employee IDs of all employees in the DevOps department."),
    "H1-06": ("Finance", "Please list the names, designations, and employee IDs of all employees in the Finance department."),
    "H1-07": ("Marketing", "Please list the names, designations, and employee IDs of all employees in the Marketing department."),
    "H1-08": ("Sales", "Please list the names, designations, and employee IDs of all employees in the Sales department."),
    "H1-09": ("Support", "Please list the names, designations, and employee IDs of all employees in the Support department."),
    "H1-10": ("Administration", "Please list the names, designations, and employee IDs of all employees in the Administration department."),
    "H1-11": ("Management", "Please list the names, designations, and employee IDs of all employees in the Management department."),
    "H1-12": ("Engineering", "How many employees are in the Engineering department according to the employee directory?"),
    "H1-13": ("Data Science", "How many employees are in the Data Science department according to the employee directory?"),
    "H1-14": ("Quality Assurance", "How many employees are in the Quality Assurance department according to the employee directory?"),
    "H1-15": ("Human Resources", "How many employees are in the Human Resources department according to the employee directory?"),
}


# ==============================================================================
# 1. INGESTION SUITE
# ==============================================================================
def test_ingestion_chunk_counts_and_types(chunks):
    roster_chunks = [c for c in chunks if c.metadata.get("content_kind") == "roster"]
    summary_chunks = [c for c in chunks if c.metadata.get("content_kind") == "roster_summary"]
    row_chunks = [c for c in chunks if c.metadata.get("document_type") == DIRECTORY_DOCUMENT_TYPE
                  and c.metadata.get("employee_id")]

    assert len(roster_chunks) == 11
    assert len(summary_chunks) == 1
    assert len(row_chunks) == 30

    for c in roster_chunks + summary_chunks:
        assert c.metadata["access_level"] == "confidential"
        assert c.metadata["content_kind"] in ("roster", "roster_summary")
        assert c.metadata["content_kind"] != "prose"


@pytest.mark.parametrize("department", DEPARTMENTS_11)
def test_ingestion_roster_chunks_match_csv_exactly(chunks, department):
    csv_dept_members = sorted(get_csv_by_department()[department], key=lambda r: r["employee_id"])
    roster_chunk = next(c for c in chunks if c.metadata.get("content_kind") == "roster"
                        and c.metadata.get("department") == department)

    assert roster_chunk.metadata["roster_size"] == len(csv_dept_members)
    parsed = parse_roster_text(roster_chunk.page_content)
    assert parsed is not None
    assert parsed.department == department
    assert parsed.count == len(csv_dept_members)
    assert len(parsed.members) == len(csv_dept_members)

    for parsed_m, csv_m in zip(parsed.members, csv_dept_members):
        assert parsed_m["employee_id"] == csv_m["employee_id"]
        assert parsed_m["name"] == csv_m["employee_name"]
        assert parsed_m["designation"] == csv_m["designation"]
        assert parsed_m["employment_type"] == csv_m["employment_type"]
        assert parsed_m["work_location"] == csv_m["work_location"]


def test_ingestion_summary_chunk_matches_csv_totals(chunks):
    summary_chunk = next(c for c in chunks if c.metadata.get("content_kind") == "roster_summary")
    parsed = parse_summary_text(summary_chunk.page_content)
    assert parsed is not None
    total, depts_count, dept_rows = parsed

    csv_data = get_csv_by_department()
    assert total == 30
    assert depts_count == 11
    assert len(dept_rows) == 11

    dept_counts_dict = dict(dept_rows)
    for dept_name, members in csv_data.items():
        assert dept_counts_dict[dept_name] == len(members)


def test_regression_roster_groups_by_row_text_department_not_metadata(chunks):
    hr_roster = next(c for c in chunks if c.metadata.get("content_kind") == "roster"
                     and c.metadata.get("department") == "Human Resources")
    assert hr_roster.metadata["roster_size"] == 3
    parsed = parse_roster_text(hr_roster.page_content)
    assert parsed.count == 3
    assert len(parsed.members) == 3


def test_roster_chunk_ids_deterministic(cfg):
    chunks1, _ = prepare_chunks(cfg, verbose=False)
    chunks2, _ = prepare_chunks(cfg, verbose=False)
    roster_ids1 = [c.metadata["chunk_id"] for c in chunks1 if c.metadata.get("content_kind") in ("roster", "roster_summary")]
    roster_ids2 = [c.metadata["chunk_id"] for c in chunks2 if c.metadata.get("content_kind") in ("roster", "roster_summary")]
    assert len(roster_ids1) == 12
    assert len(set(roster_ids1)) == 12
    assert roster_ids1 == roster_ids2


def test_build_roster_chunks_is_idempotent(chunks):
    first_built = build_roster_chunks(chunks)
    assert len(first_built) == 12
    second_built = build_roster_chunks(chunks + first_built)
    assert len(second_built) == 12


# ==============================================================================
# 2. PLANNER UNIT TESTS
# ==============================================================================
@pytest.mark.parametrize("qid, dept, q", [(k, v[0], v[1]) for k, v in H1_QUESTIONS.items()])
def test_planner_intent_positives_h1_questions(qid, dept, q):
    intent = roster_intent(q)
    assert intent in ("list", "count")


@pytest.mark.parametrize("q", [
    "How many employees does the company have?",
    "What is the total number of employees?",
    "total headcount",
    "headcount by department",
    "how many employees are there in each department",
    "list all the departments",
    "how many departments does the company have?",
])
def test_planner_intent_positives_summary(q):
    assert roster_intent(q) == "summary"


def test_planner_intent_positives_company_headcount():
    assert roster_intent("company headcount") in ("count", "summary")


@pytest.mark.parametrize("q", [
    "How many annual leave days do employees get?",
    "Show me the leave policy for employees",
    "What is the sales target policy?",
    "Who is the HR contact for payroll?",
    "What is Aarav Mehta's designation?",
    "How many casual leave days do I get?",
    "What are the public holidays in 2026?",
])
def test_planner_intent_negatives(q):
    assert roster_intent(q) is None


def test_plan_roster_answer_with_header_stripped_source_text(chunks):
    eng_chunk = next(c for c in chunks if c.metadata.get("content_kind") == "roster"
                     and c.metadata.get("department") == "Engineering")
    header_stripped_text = "\n".join(
        line for line in eng_chunk.page_content.splitlines() if not line.startswith(ROSTER_HEADER)
    )
    sources = [{"file_name": "employee_directory.csv", "section_title": "Roster: Engineering",
                "text": header_stripped_text}]

    plan = plan_roster_answer("Please list all employees in the Engineering department.", sources)
    assert plan.text is not None
    assert plan.notice is False
    assert plan.intent == "list"
    assert "Engineering has 6 employees [S1]:" in plan.text
    assert "Rohan Verma" in plan.text
    assert "APS003" in plan.text


def test_plan_roster_answer_marker_index_matches_source_position(chunks):
    eng_chunk = next(c for c in chunks if c.metadata.get("content_kind") == "roster"
                     and c.metadata.get("department") == "Engineering")
    dummy_source = {"file_name": "Employee_Handbook.pdf", "section_title": "Intro", "text": "Welcome to company."}
    eng_source = {"file_name": "employee_directory.csv", "section_title": "Roster: Engineering", "text": eng_chunk.page_content}

    plan = plan_roster_answer("List all employees in Engineering", [dummy_source, eng_source])
    assert plan.text is not None
    assert "[S2]:" in plan.text
    assert plan.source_indexes == (2,)


@pytest.mark.parametrize("alias_q, dept_expected", [
    ("List all employees in HR", "Human Resources"),
    ("How many people work in QA?", "Quality Assurance"),
    ("List all employees in Admin department", "Administration"),
])
def test_plan_roster_answer_aliases(chunks, alias_q, dept_expected):
    chunk = next(c for c in chunks if c.metadata.get("content_kind") == "roster"
                 and c.metadata.get("department") == dept_expected)
    sources = [{"file_name": "employee_directory.csv", "section_title": f"Roster: {dept_expected}",
                "text": chunk.page_content}]
    plan = plan_roster_answer(alias_q, sources)
    assert plan.text is not None
    assert f"{dept_expected} has" in plan.text


def test_plan_roster_answer_non_roster_only_query_triggers_notice():
    sources_with_directory = [
        {"file_name": "employee_directory.csv", "section_title": "Row", "text": "department: Engineering\nemployee_name: Rohan Verma"}
    ]
    plan = plan_roster_answer("Please list the employees in Engineering and their email addresses and phone numbers", sources_with_directory)
    assert plan.text is None
    assert plan.notice is True

    sources_without_directory = [
        {"file_name": "Employee_Handbook.pdf", "section_title": "Intro", "text": "Welcome to company."}
    ]
    plan2 = plan_roster_answer("Please list the employees in Engineering and their email addresses and phone numbers", sources_without_directory)
    assert plan2.text is None
    assert plan2.notice is False


def test_plan_roster_answer_unknown_department(chunks):
    eng_chunk = next(c for c in chunks if c.metadata.get("content_kind") == "roster"
                     and c.metadata.get("department") == "Engineering")
    sources = [{"file_name": "employee_directory.csv", "section_title": "Roster: Engineering",
                "text": eng_chunk.page_content}]
    plan = plan_roster_answer("List all employees in the Astronomy department", sources)
    assert plan.text is None


# ==============================================================================
# 3. PIPELINE END-TO-END (ROLE HR)
# ==============================================================================
@pytest.fixture()
def hr_rig(cfg, store, embeddings, fake_reranker):
    def build(llm_answer="SCRIPTED LLM SHOULD NEVER BE CALLED", **overrides):
        record = {"llm_called": 0}

        def fake_llm(system, user):
            record["llm_called"] += 1
            return llm_answer

        pipeline = RAGPipeline(
            dataclasses.replace(cfg, **overrides),
            embeddings=embeddings,
            vector_store=store,
            reranker=fake_reranker,
            llm=fake_llm
        )
        pipeline.record = record
        return pipeline
    return build


@pytest.mark.parametrize("qid", [f"H1-{i:02d}" for i in range(1, 12)])
def test_pipeline_h1_roster_lists_all_11_departments(hr_rig, qid):
    dept, question = H1_QUESTIONS[qid]
    rig = hr_rig()
    r = rig.ask(question, {"role": "hr"}, log=False)

    assert r["status"] == "access_verified"
    assert rig.record["llm_called"] == 0
    assert r["response_class"] == "NORMAL_GROUNDED"
    assert r["usage"]["generation"] == "roster_lookup"
    assert r["usage"]["input_tokens"] == 0
    assert r["usage"]["output_tokens"] == 0
    assert r["roster_lookup"]["answered_from_roster"] is True
    assert r["roster_notice"] is None

    expected_members = get_csv_by_department()[dept]
    for member in expected_members:
        assert member["employee_name"] in r["answer"]
        assert member["employee_id"] in r["answer"]

    all_csv_members = get_csv_rows()
    other_members = [m for m in all_csv_members if m["department"].strip() != dept]
    for other in other_members:
        assert other["employee_id"] not in r["answer"]


@pytest.mark.parametrize("qid", [f"H1-{i:02d}" for i in range(12, 16)])
def test_pipeline_h1_headcount_queries(hr_rig, qid):
    dept, question = H1_QUESTIONS[qid]
    rig = hr_rig(numeric_gate_mode="enforce")
    r = rig.ask(question, {"role": "hr"}, log=False)

    assert r["status"] == "access_verified"
    assert rig.record["llm_called"] == 0
    assert r["response_class"] == "NORMAL_GROUNDED"
    assert r["roster_lookup"]["answered_from_roster"] is True
    assert r["roster_notice"] is None

    expected_count = len(get_csv_by_department()[dept])
    assert f"{dept} has {expected_count} employee" in r["answer"]


@pytest.mark.parametrize("q", [
    "How many employees does the company have according to the employee directory?",
    "headcount by department",
    "list all departments",
])
def test_pipeline_summary_queries(hr_rig, q):
    rig = hr_rig(numeric_gate_mode="enforce")
    r = rig.ask(q, {"role": "hr"}, log=False)

    assert r["status"] == "access_verified"
    assert rig.record["llm_called"] == 0
    assert r["roster_lookup"]["answered_from_roster"] is True
    assert "The employee directory lists 30 employees across 11 departments" in r["answer"]


# ==============================================================================
# 4. PARTIAL ROSTER HONESTY
# ==============================================================================
def test_partial_roster_from_row_chunks_only_sets_notice(cfg, store, embeddings, fake_reranker):
    doc1 = Document(
        page_content="employee_id: APS003\nemployee_name: Rohan Verma\ndepartment: Engineering\ndesignation: Software Engineer",
        metadata={"file_name": "employee_directory.csv", "document_type": DIRECTORY_DOCUMENT_TYPE,
                  "employee_id": "APS003", "section_title": "Rohan Verma", "access_level": "confidential",
                  "content_kind": "row", "is_current": True, "rerank_score": 0.9}
    )

    def retrieve_stub(*args, **kwargs):
        return {"semantic": [], "keyword": [], "fused": [doc1], "reranked": [doc1]}

    partial_answer = "The Engineering department includes Rohan Verma (APS003) [S1]."
    pipeline = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                           llm=lambda s, u: partial_answer)
    pipeline.retriever.retrieve = retrieve_stub

    r = pipeline.ask("Please list all employees in the Engineering department.", {"role": "hr"}, log=False)
    assert r["status"] == "access_verified"
    assert r["roster_notice"] == ROSTER_NOTICE
    assert r["roster_lookup"]["answered_from_roster"] is False


def test_leave_policy_queries_as_hr_never_get_roster_notice(hr_rig):
    rig = hr_rig(llm_answer="Casual leave is 10 days per year [S1].")
    r = rig.ask("How many days of casual leave do employees get?", {"role": "hr"}, log=False)
    assert r["roster_notice"] is None


# ==============================================================================
# 5. ACCESS (PHASE 1 PRESERVED)
# ==============================================================================
@pytest.mark.parametrize("dept", DEPARTMENTS_11)
def test_employee_cannot_retrieve_roster_chunks_at_any_stage(cfg, store, fake_reranker, dept):
    retriever = HybridRetriever(store, fake_reranker, cfg)
    employee_levels = allowed_levels_for("employee")

    results = retriever.retrieve(f"List all employees in {dept}", employee_levels, include_archived=True)
    for stage in ("semantic", "keyword", "fused", "reranked"):
        for doc in results.get(stage, []):
            assert doc.metadata.get("content_kind") not in ("roster", "roster_summary"), f"Leaked in stage {stage}"
            assert doc.metadata.get("file_name") != "employee_directory.csv"


def test_employee_pipeline_never_leaks_roster_canaries(cfg, store, embeddings, fake_reranker):
    pipeline = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                           llm=lambda s, u: "I cannot find this information.")
    r = pipeline.ask("Please list the employees in Engineering.", {"role": "employee"}, log=False)

    eng_canaries = ["APS003", "APS004", "APS016", "APS017", "APS018", "APS028", "Rohan Verma", "Neha Iyer"]
    for canary in eng_canaries:
        assert canary not in r["answer"]


# ==============================================================================
# 6. MODES
# ==============================================================================
def test_roster_lookup_shadow_mode(hr_rig):
    rig = hr_rig(llm_answer="Engineering team members are listed [S1].", roster_lookup_mode="shadow")
    r = rig.ask("Please list all employees in the Engineering department.", {"role": "hr"}, log=False)

    assert rig.record["llm_called"] == 1
    assert r["roster_lookup"]["mode"] == "shadow"
    assert r["roster_lookup"]["answered_from_roster"] is False
    assert r["roster_notice"] is None


def test_roster_lookup_off_mode(hr_rig):
    rig = hr_rig(llm_answer="Engineering team members are listed [S1].", roster_lookup_mode="off")
    r = rig.ask("Please list all employees in the Engineering department.", {"role": "hr"}, log=False)

    assert rig.record["llm_called"] == 1
    assert r["roster_lookup"]["mode"] == "off"
    assert r["roster_lookup"]["intent"] is None
    assert r["roster_lookup"]["answered_from_roster"] is False


def test_roster_lookup_invalid_mode_raises(cfg, store, embeddings, fake_reranker):
    with pytest.raises(ValueError, match="ROSTER_LOOKUP_MODE"):
        RAGPipeline(dataclasses.replace(cfg, roster_lookup_mode="invalid_mode"),
                    embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: "x")


def test_roster_lookup_logged(hr_rig, cfg):
    rig = hr_rig()
    rig.ask("Please list the names of all employees in Engineering.", {"role": "hr"}, log=True)
    last_log = read_jsonl(cfg.logs_dir / "query_log.jsonl")[-1]
    assert "roster_lookup" in last_log
    assert last_log["roster_lookup"]["answered_from_roster"] is True


# ==============================================================================
# 7. STREAMING
# ==============================================================================
def test_streaming_deterministic_single_delta(cfg, store, embeddings, fake_reranker):
    deltas = []
    stream_called = [0]

    def fake_stream(system, user, on_delta):
        stream_called[0] += 1
        on_delta("chunk 1")
        return "chunk 1", {"input_tokens": 5, "output_tokens": 5}

    pipeline = RAGPipeline(
        cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
        llm=lambda s, u: "fallback", llm_stream=fake_stream
    )
    r = pipeline.ask("Please list all employees in Engineering.", {"role": "hr"},
                     log=False, on_delta=deltas.append)

    assert stream_called[0] == 0
    assert len(deltas) == 1
    assert deltas[0] == r["answer"]
    assert "Engineering has 6 employees" in deltas[0]


# ==============================================================================
# 8. INDIVIDUAL LOOKUPS UNAFFECTED
# ==============================================================================
def test_individual_employee_lookup_unaffected(hr_rig):
    rig = hr_rig(llm_answer="Aarav Mehta's designation is HR Manager [S1].")
    r = rig.ask("What is Aarav Mehta's designation?", {"role": "hr"}, log=False)

    assert rig.record["llm_called"] == 1
    assert r["status"] == "access_verified"
    assert r["roster_lookup"]["answered_from_roster"] is False
    assert r["roster_notice"] is None
