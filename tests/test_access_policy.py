"""Locks in WHICH documents each role may reach, using the real data/manifest.yaml.

If you add a file to data/raw or change a level in the manifest, these tests fail on
purpose: access decisions should be made consciously, not by default.
"""
import pytest

from hr_rag.config import ROLE_ACCESS, settings
from hr_rag.manifest import load_manifest, resolve_entry
from hr_rag.retrieval import HybridRetriever, allowed_levels_for, process_query

# The agreed policy. 'internal' = any employee may ask about it; 'confidential' = HR only.
EXPECTED = {
    "Employee_Handbook.pdf": "internal",
    "Attendance_Policy.pdf": "internal",
    "Onboarding_Guide.pdf": "internal",
    "onboarding_checklist.docx": "internal",
    "work_from_home_policy.txt": "internal",
    "leave_policy_v1.docx": "internal",
    "leave_policy_v2.docx": "internal",
    "duplicate_leave_policy.docx": "internal",   # excluded from the index anyway
    "holiday_calendar.xlsx": "internal",
    "hr_announcement.html": "internal",
    "job_descriptions.docx": "internal",
    "employee_directory.csv": "confidential",
    "department_details.xlsx": "confidential",
    "employee_profiles.json": "confidential",
    "incomplete_employee_record.json": "confidential",
}
HR_ONLY_FILES = {f for f, level in EXPECTED.items() if level == "confidential"}


@pytest.fixture(scope="module")
def retriever(store, fake_reranker, cfg):
    return HybridRetriever(store, fake_reranker, cfg)


def test_manifest_matches_the_agreed_policy_for_every_file():
    manifest = load_manifest(settings.manifest_path)
    assert set(manifest["files"]) == set(EXPECTED), "a file was added/removed: classify it on purpose"
    for name, level in EXPECTED.items():
        assert resolve_entry(manifest, name)[0]["access_level"] == level, name


def test_every_listed_file_states_its_access_level_explicitly():
    manifest = load_manifest(settings.manifest_path)
    missing = [n for n, entry in manifest["files"].items() if "access_level" not in (entry or {})]
    assert not missing, f"no explicit access_level for: {missing}"


def test_unclassified_files_fail_closed():
    manifest = load_manifest(settings.manifest_path)
    assert manifest["defaults"]["access_level"] == "confidential"
    assert resolve_entry(manifest, "new_unlisted_file.pdf")[0]["access_level"] == "confidential"
    listed_without_level = {**manifest, "files": {**manifest["files"], "x.pdf": {"title": "x"}}}
    assert resolve_entry(listed_without_level, "x.pdf")[0]["access_level"] == "confidential"
    assert "confidential" not in ROLE_ACCESS["employee"]


def test_indexed_chunks_carry_the_expected_level(chunks):
    for chunk in chunks:
        assert chunk.metadata["access_level"] == EXPECTED[chunk.metadata["file_name"]]


def test_employee_retrieval_never_returns_hr_only_files_but_hr_does(retriever):
    """Both search paths (vector and keyword) for employee-style questions about people data."""
    questions = ["Who are the members of the engineering department?", "Give me the employee directory",
                 "How many employees are in the Data Science department?", "APS007",
                 "What is Aarav Mehta's date of birth and home address?"]
    for question in questions:
        for role, may_see in (("employee", False), ("hr", True)):
            res = retriever.retrieve(process_query(question), allowed_levels_for(role), include_archived=True)
            files = {d.metadata["file_name"] for key in ("semantic", "keyword", "fused", "reranked") for d in res[key]}
            if may_see:
                continue
            assert not (files & HR_ONLY_FILES), (question, files & HR_ONLY_FILES)
    hr = retriever.retrieve(process_query("APS007"), allowed_levels_for("hr"))
    assert any(d.metadata["file_name"] == "employee_directory.csv" for d in hr["keyword"])   # positive control
