import json

import pytest

from hr_rag import pipeline as pipeline_module
from hr_rag.pipeline import NO_ANSWER, RAGPipeline, format_response
from hr_rag.scope import RESTRICTED_MESSAGE, ScopeDecision


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


def test_confidential_text_never_reaches_the_llm_for_an_employee(pipeline, monkeypatch):
    # Phase 1 guarantee, tested with the Phase 2 scope check switched off so that only the
    # retrieval-time access filter stands between the employee and the confidential text.
    monkeypatch.setattr(pipeline_module, "check_scope", lambda question, levels: ScopeDecision(False, None))
    pipeline.ask("What is Aarav Mehta's date of birth and home address?", {"role": "employee"})
    _, user = pipeline.calls[0]
    assert "1988-06-14" not in user and "Green Park Road" not in user


def test_employee_asking_for_confidential_data_is_refused_before_the_llm(pipeline):
    r = pipeline.ask("What is Aarav Mehta's date of birth and home address?", {"role": "employee"})
    assert pipeline.calls == [] and r["status"] == "access_restricted" and r["answer"] == RESTRICTED_MESSAGE


def test_same_question_as_hr_does_reach_confidential_text(pipeline):
    pipeline.ask("What is Aarav Mehta's date of birth and home address?", {"role": "hr"})
    _, user = pipeline.calls[0]
    assert "1988-06-14" in user


def test_citation_to_missing_source_is_flagged(store, embeddings, fake_reranker, cfg):
    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: "Twenty days [S99].")
    r = p.ask("annual leave days", {"role": "employee"})
    assert r["citations_valid"] is False and r["invalid_citation_ids"] == ["S99"]
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


def test_hr_can_ask_about_the_incomplete_record_and_employee_cannot(store, embeddings, fake_reranker, cfg, monkeypatch):
    seen = []
    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: (seen.append(u), "ok [S1]")[1])
    question = "What validation errors does the incomplete record for Rahul Sharma have?"
    p.ask(question, {"role": "hr"}, log=False)
    assert "employee_id is required" in seen[-1]
    # Employee: refused before any LLM call ...
    calls_before = len(seen)
    r = p.ask(question, {"role": "employee"}, log=False)
    assert r["status"] == "access_restricted" and len(seen) == calls_before
    # ... and, with the scope check off, the Phase-1 filter alone still keeps the text out.
    monkeypatch.setattr(pipeline_module, "check_scope", lambda q, levels: ScopeDecision(False, None))
    p.ask(question, {"role": "employee"}, log=False)
    assert len(seen) == calls_before + 1 and "employee_id is required" not in seen[-1]


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


def test_invented_numbers_are_flagged(store, embeddings, fake_reranker, cfg):
    from hr_rag.evaluation import ungrounded_numbers
    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: "Maternity leave is 26 weeks [S1].")
    r = p.ask("how long is maternity leave", {"role": "employee"}, log=False)
    assert r["evaluation"]["numbers_grounded"] is False
    assert "26" in r["evaluation"]["ungrounded_numbers"]
    ctx = "Employees may carry forward up to 10 unused days. Apply five working days in advance."
    assert ungrounded_numbers("Carry forward 10 days; apply 5 days ahead [S1].", ctx) == []
    assert ungrounded_numbers("You asked for 12 days [S1].", ctx, "can I take 12 days") == []
    assert ungrounded_numbers("1. Total is 40 days.", ctx) == ["40"]


# ---------------------------------------------------------------- answer style / tokens
def test_concise_prompt_demands_short_answers_and_detailed_does_not():
    from hr_rag.generation import build_system_prompt
    concise, detailed = build_system_prompt("concise"), build_system_prompt("detailed")
    assert "under about 120 words" in concise and "under about 120 words" not in detailed
    assert build_system_prompt("nonsense") == concise          # unknown style -> concise
    for prompt in (concise, detailed):                         # accuracy rules never change
        assert "Never guess" in prompt and "ARCHIVED" in prompt and "Never invent items" in prompt


def test_strip_citations_for_the_chat_window():
    from hr_rag.generation import strip_citations
    assert strip_citations("Apply via the portal [S1]. Manager approves [S2], [S3].") == \
        "Apply via the portal. Manager approves."
    assert strip_citations("No markers here.") == "No markers here."


def test_simple_questions_get_a_small_prompt_and_complex_ones_a_big_one(cfg):
    from hr_rag.retrieval import choose_context_budget
    assert choose_context_budget("What is the process to apply for casual leave?", [], cfg) == (6, 7000)
    assert choose_context_budget("List all eight categories in the checklist", [], cfg) == (12, 14000)
    assert choose_context_budget("Compare v1 and v2 leave", [], cfg)[0] == 12
    assert choose_context_budget("annual leave days", ["part one here now", "part two here now"], cfg)[0] == 12


def test_simple_question_prompt_has_at_most_six_sources_and_uses_the_chosen_style(
        store, embeddings, fake_reranker, cfg):
    seen = []
    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: (seen.append((s, u)), "Casual leave is 10 days [S1].")[1])
    r = p.ask("How many casual leave days do I get?", {"role": "employee"}, log=False)
    system, user = seen[-1]
    assert user.count("SOURCE ") <= 6 and r["usage"]["context_chunks"] <= 6
    assert "under about 120 words" in system and r["style"] == "concise"
    p.ask("How many casual leave days do I get?", {"role": "employee"}, log=False, style="detailed")
    assert "under about 120 words" not in seen[-1][0]


def test_usage_timings_display_text_and_cited_sources(store, embeddings, fake_reranker, cfg):
    p = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda s, u: ("Casual leave is 10 days [S2].", {"input_tokens": 900, "output_tokens": 12}))
    r = p.ask("How many casual leave days do I get?", {"role": "employee"}, log=False)
    assert r["usage"]["input_tokens"] == 900 and r["usage"]["output_tokens"] == 12 and not r["usage"]["estimated"]
    assert set(r["timings"]) == {"retrieval_s", "generation_s", "total_s"}
    assert r["answer_display"] == "Casual leave is 10 days."
    assert r["cited_ids"] == ["S2"] and len(r["cited_sources"]) == 1
    assert r["cited_sources"][0] is r["sources"][1]
    # no usage from the provider -> estimated, and flagged as such
    p2 = RAGPipeline(cfg, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                     llm=lambda s, u: "Casual leave is 10 days [S1].")
    r2 = p2.ask("How many casual leave days do I get?", {"role": "employee"}, log=False)
    assert r2["usage"]["estimated"] and r2["usage"]["input_tokens"] > 0


def test_provider_usage_is_returned(monkeypatch):
    from types import SimpleNamespace
    import hr_rag.generation as g
    from hr_rag.config import Settings

    class FakeClient:
        def __init__(self, **kw):
            msg = SimpleNamespace(content="Ten days [S1]")
            reply = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                                    usage=SimpleNamespace(prompt_tokens=800, completion_tokens=5))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: reply))

    monkeypatch.setenv("NVIDIA_API_KEY", "x")
    monkeypatch.setattr("openai.OpenAI", FakeClient)
    text, usage = g.generate_answer_with_usage("s", "u", Settings(llm_provider="nvidia"))
    assert text == "Ten days [S1]" and usage == {"input_tokens": 800, "output_tokens": 5}
