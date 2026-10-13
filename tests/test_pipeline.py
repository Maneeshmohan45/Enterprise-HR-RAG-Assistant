import json

import pytest

from hr_rag.pipeline import NO_ANSWER, RAGPipeline, format_response


@pytest.fixture()
def pipeline(store, embeddings, fake_reranker, cfg):
    calls = []

    def fake_llm(system, user):
        calls.append((system, user))
        return "Annual leave is 20 days per year [S1]."

    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker, llm=fake_llm)
    p.calls = calls
    return p


def test_end_to_end_answer_has_sources_and_checks(pipeline):
    r = pipeline.ask("How many annual leave days do employees get?", {"user_id": "E1", "role": "employee"})
    assert r["status"] == "access_verified" and r["citations_valid"]
    assert r["sources"] and r["evaluation"]["access_verified"]
    assert "Sources" in format_response(r)
    system, user = pipeline.calls[0]
    assert "AUTHORIZED RETRIEVED CONTEXT" in user and "annual leave" in user.lower()


def test_confidential_text_never_reaches_the_llm_for_an_employee(pipeline):
    pipeline.ask("What is Aarav Mehta's date of birth and home address?", {"role": "employee"})
    _, user = pipeline.calls[0]
    assert "1988-06-14" not in user and "Green Park Road" not in user


def test_same_question_as_hr_does_reach_confidential_text(pipeline):
    pipeline.ask("What is Aarav Mehta's date of birth and home address?", {"role": "hr"})
    _, user = pipeline.calls[0]
    assert "1988-06-14" in user


def test_citation_to_missing_source_is_flagged(store, embeddings, fake_reranker, cfg):
    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: "Twenty days [S9].")
    r = p.ask("annual leave days", {"role": "employee"})
    assert r["citations_valid"] is False and r["invalid_citation_ids"] == ["S9"]
    assert r["evaluation"]["citations_valid"] is False


def test_bad_input_and_logging(pipeline, cfg):
    with pytest.raises(ValueError):
        pipeline.ask("   ")
    with pytest.raises(ValueError):
        pipeline.ask("x" * 5000)
    with pytest.raises(PermissionError):
        pipeline.ask("leave?", {"role": "stranger"})
    pipeline.ask("sick leave days", {"role": "employee"})
    lines = (cfg.logs_dir / "query_log.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["question"] == "sick leave days"


def test_empty_index_gives_clear_error(tmp_path, embeddings, fake_reranker, cfg):
    from dataclasses import replace
    from hr_rag.vectorstore import open_vector_store
    empty = open_vector_store(embeddings, tmp_path / "empty", "empty_collection")
    with pytest.raises(RuntimeError, match="python ingest.py"):
        RAGPipeline(replace(cfg), embeddings=embeddings, vector_store=empty, reranker=fake_reranker)


def test_empty_llm_answer_is_reported_clearly(monkeypatch):
    from types import SimpleNamespace
    import hr_rag.generation as g
    from hr_rag.config import Settings

    class FakeClient:
        def __init__(self, **kw):
            msg = SimpleNamespace(content=None)
            reply = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: reply))

    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    monkeypatch.setattr("openai.OpenAI", FakeClient)
    with pytest.raises(g.LLMError, match="finish_reason=length"):
        g.generate_answer("s", "u", Settings(llm_provider="nvidia"))


def test_hr_can_ask_about_the_incomplete_record_and_employee_cannot(store, embeddings, fake_reranker, cfg):
    seen = []
    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: (seen.append(u), "ok [S1]")[1])
    p.ask("What validation errors does the incomplete record for Rahul Sharma have?", {"role": "hr"}, log=False)
    assert "employee_id is required" in seen[-1]
    p.ask("What validation errors does the incomplete record for Rahul Sharma have?", {"role": "employee"}, log=False)
    assert "employee_id is required" not in seen[-1]


def _nvidia_client(content, finish_reason="stop", record=None):
    from types import SimpleNamespace

    class FakeClient:
        def __init__(self, **kw):
            def create(**k):
                if record is not None:
                    record.update(k)
                msg = SimpleNamespace(content=content)
                return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return FakeClient


def test_thinking_switch_is_sent_to_nvidia_only_when_configured(monkeypatch):
    import hr_rag.generation as g
    from hr_rag.config import Settings
    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    sent = {}
    monkeypatch.setattr("openai.OpenAI", _nvidia_client("Answer [S1]", record=sent))
    g.generate_answer("s", "u", Settings(llm_provider="nvidia", nvidia_enable_thinking="false"))
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    sent.clear()
    g.generate_answer("s", "u", Settings(llm_provider="nvidia", nvidia_enable_thinking=""))
    assert "extra_body" not in sent


def test_reasoning_notes_are_never_shown_as_the_answer(monkeypatch):
    import hr_rag.generation as g
    from hr_rag.config import Settings
    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    cfg = Settings(llm_provider="nvidia")
    monkeypatch.setattr("openai.OpenAI", _nvidia_client("Here's a thinking process:\n1. Analyze"))
    with pytest.raises(g.LLMError, match="NVIDIA_ENABLE_THINKING=false"):
        g.generate_answer("s", "u", cfg)
    monkeypatch.setattr("openai.OpenAI", _nvidia_client("notes</think>The real answer [S1]"))
    assert g.generate_answer("s", "u", cfg) == "The real answer [S1]"
    monkeypatch.setattr("openai.OpenAI", _nvidia_client("half an ans", finish_reason="length"))
    with pytest.raises(g.LLMError, match="finish_reason=length"):
        g.generate_answer("s", "u", cfg)
