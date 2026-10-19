import json

from hr_rag import evaluation
from hr_rag.generation import LLMError
from hr_rag.ui_data import (friendly_error, query_table_rows, read_jsonl, redact,
                            summarize_feedback, summarize_queries)

OLD_ROW = {"query_id": "a", "time": "2026-10-12T15:04:28", "role": "employee", "question": "q1",
           "status": "access_verified", "source_chunk_ids": [], "evaluation_score": 100.0}
NEW_ROW = {**OLD_ROW, "query_id": "b", "evaluation_score": 87.5, "style": "concise",
           "usage": {"input_tokens": 1000, "output_tokens": 60},
           "timings": {"retrieval_s": 2.0, "generation_s": 3.0, "total_s": 5.0},
           "checks": {"citations_valid": True, "numbers_grounded": False}}


def test_read_jsonl_skips_bad_lines_and_missing_files(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps(OLD_ROW) + "\r\nnot json\r\n\r\n" + json.dumps(NEW_ROW) + "\r\n", encoding="utf-8")
    assert [r["query_id"] for r in read_jsonl(path)] == ["a", "b"]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_summary_only_uses_rows_that_recorded_the_field():
    s = summarize_queries([OLD_ROW, NEW_ROW])
    assert s["n_queries"] == 2 and s["avg_score"] == (100.0 + 87.5) / 2
    assert s["n_timed"] == 1 and s["avg_total_s"] == 5.0          # old row not counted as 0
    assert s["n_with_usage"] == 1 and s["avg_output_tokens"] == 60
    assert s["n_with_checks"] == 1 and s["check_pass_rates"] == {"citations_valid": 1.0, "numbers_grounded": 0.0}
    assert s["by_style"]["concise"]["n"] == 1


def test_summary_of_empty_log_has_no_made_up_numbers():
    s = summarize_queries([])
    assert s["n_queries"] == 0 and s["avg_score"] is None and s["avg_total_s"] is None


def test_feedback_summary():
    s = summarize_feedback([{"rating": 5, "helpful": True}, {"rating": 1, "helpful": False}])
    assert (s["n"], s["helpful"], s["not_helpful"], s["avg_rating"]) == (2, 1, 1, 3.0)


def test_question_text_is_hidden_unless_allowed():
    hidden = query_table_rows([NEW_ROW], show_questions=False)[0]
    shown = query_table_rows([NEW_ROW], show_questions=True)[0]
    assert "q1" not in hidden["Question"] and shown["Question"] == "q1"


def test_redact_removes_key_like_text():
    assert "nvapi-abcdef123456" not in redact("bad key nvapi-abcdef123456 given")
    assert "secretvalue" not in redact("Incorrect API key provided: secretvalue")
    assert "abc.def" not in redact("Authorization: Bearer abc.def")


def test_friendly_errors():
    assert "not configured" in friendly_error(ValueError("NVIDIA_API_KEY is missing. Copy .env"))[0]
    assert friendly_error(ValueError("Please enter a question."))[0] == "Please enter a question."
    assert "rejected" in friendly_error(LLMError("nvidia request failed: Error code: 401"))[0]
    assert "rate-limited" in friendly_error(LLMError("429 rate limit"))[0]
    assert "ingest.py" in friendly_error(RuntimeError("The vector store is empty. Build it"))[0]
    assert "permission" in friendly_error(PermissionError("No permission"))[0].lower()
    message, detail = friendly_error(LLMError("nvidia request failed: bad key nvapi-abcdef123456"))
    assert "nvapi-abcdef123456" not in detail


def test_query_log_records_checks(cfg):
    response = {"query_id": "x", "role": "employee", "question": "q", "status": "access_verified",
                "sources": [], "style": "concise", "usage": {}, "timings": {},
                "evaluation": {"evaluation_score": 87.5, "citations_valid": True, "numbers_grounded": False,
                               "ungrounded_numbers": ["26"], "lexical_grounding_score": 0.9}}
    evaluation.log_query(response, cfg)
    row = read_jsonl(cfg.logs_dir / "query_log.jsonl")[-1]
    assert row["checks"] == {"citations_valid": True, "numbers_grounded": False}
