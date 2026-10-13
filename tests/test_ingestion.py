from hr_rag.cleaning import clean_text, deduplicate, validate_document
from langchain_core.documents import Document


def test_incomplete_record_is_indexed_but_marked_invalid_and_hr_only(report, chunks):
    warned = report["indexed_with_warnings"]
    assert any(i["file"] == "incomplete_employee_record.json" and "missing_employee_id" in i["errors"] for i in warned)
    mine = [c for c in chunks if c.metadata["file_name"] == "incomplete_employee_record.json"]
    assert mine and all(c.metadata["access_level"] == "confidential" for c in mine)
    assert all(c.metadata["validation_status"] == "invalid" for c in mine)
    text = " ".join(c.page_content for c in mine)
    assert "employee_id is required" in text and "employment_status contains an unsupported value" in text


def test_duplicate_policy_is_excluded(report, chunks):
    assert any(e["file"] == "duplicate_leave_policy.docx" for e in report["excluded"])
    assert not any(c.metadata["file_name"] == "duplicate_leave_policy.docx" for c in chunks)


def test_leave_v1_is_archived_and_v2_is_current(chunks):
    v1 = [c for c in chunks if c.metadata["file_name"] == "leave_policy_v1.docx"]
    v2 = [c for c in chunks if c.metadata["file_name"] == "leave_policy_v2.docx"]
    assert v1 and all(c.metadata["is_current"] is False and c.metadata["status"] == "archived" for c in v1)
    assert v2 and all(c.metadata["is_current"] is True and c.metadata["version_number"] == 2 for c in v2)


def test_profiles_are_confidential_directory_is_internal(chunks):
    levels = {}
    for c in chunks:
        levels.setdefault(c.metadata["file_name"], set()).add(c.metadata["access_level"])
    assert levels["employee_profiles.json"] == {"confidential"}
    assert levels["employee_directory.csv"] == {"internal"}


def test_chunk_ids_unique_and_every_chunk_has_required_metadata(chunks):
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        for key in ("file_name", "title", "access_level", "is_current", "document_id", "section_title"):
            assert key in c.metadata
        assert c.page_content.startswith(c.metadata["chunk_header"])


def test_all_fifteen_files_accounted_for(report):
    accounted = set(report["files"]) | {e["file"] for e in report["excluded"]}
    assert len(accounted) == 15 and not report["failed"] and not report["unlisted_files"]


def test_unlisted_file_fails_closed(tmp_path, cfg):
    from dataclasses import replace
    from hr_rag.ingestion import prepare_chunks
    (tmp_path / "new_policy.txt").write_text("1. PURPOSE This is a brand new policy that nobody classified yet.")
    chunks, rep = prepare_chunks(replace(cfg, raw_data_dir=tmp_path), verbose=False)
    assert rep["unlisted_files"] == ["new_policy.txt"]
    assert {c.metadata["access_level"] for c in chunks} == {"confidential"}


def test_clean_text_keeps_currency_pipes_and_joins_wrapped_lines():
    out = clean_text("Budget \u20b9 2500000 | HR\u2022 item\nwrapped line\ncontinues here", unwrap=True)
    assert "\u20b9" in out and "|" in out
    assert "wrapped line continues here" in out


def test_clean_text_does_not_unwrap_records():
    text = "department: HR\nemployee_name: Priya"
    assert clean_text(text, unwrap=False) == text


def test_dedupe_keeps_first_and_validate():
    a, b = Document(page_content="Same   text  here ok", metadata={}), Document(page_content="same text here OK", metadata={})
    unique, dups = deduplicate([a, b])
    assert unique == [a] and dups == [b]
    assert validate_document(Document(page_content="  ", metadata={})) == ["empty_content"]
