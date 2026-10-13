import pytest

from hr_rag.retrieval import (HybridRetriever, allowed_levels_for, build_context, process_query,
                              reciprocal_rank_fusion, tokenize)
from langchain_core.documents import Document


@pytest.fixture(scope="module")
def retriever(store, fake_reranker, cfg):
    return HybridRetriever(store, fake_reranker, cfg)


def _all(results):
    return results["semantic"] + results["keyword"] + results["fused"] + results["reranked"]


def test_employee_never_sees_confidential_chunks(retriever):
    for q in ["date of birth of Aarav Mehta", "home address emergency contact", "personal email phone"]:
        res = retriever.retrieve(process_query(q), allowed_levels_for("employee"))
        assert _all(res), "expected some results"
        assert all(d.metadata["access_level"] != "confidential" for d in _all(res))


def test_hr_can_see_profiles(retriever):
    res = retriever.retrieve(process_query("date of birth of Aarav Mehta"), allowed_levels_for("hr"))
    assert any(d.metadata["file_name"] == "employee_profiles.json" for d in res["reranked"])


def test_archived_policy_hidden_by_default_but_available_on_request(retriever):
    q = process_query("casual leave days per year")
    default = retriever.retrieve(q, allowed_levels_for("employee"))
    assert all(d.metadata["is_current"] for d in _all(default))
    with_old = retriever.retrieve(q, allowed_levels_for("employee"), include_archived=True)
    assert any(not d.metadata["is_current"] for d in _all(with_old))


def test_current_leave_policy_is_found(retriever):
    res = retriever.retrieve(process_query("How many annual leave days do employees get?"),
                             allowed_levels_for("employee"))
    assert res["reranked"][0].metadata["file_name"] in {"leave_policy_v2.docx", "Employee_Handbook.pdf"}
    assert any(d.metadata["file_name"] == "leave_policy_v2.docx" for d in res["reranked"])


def test_keyword_search_finds_exact_employee_id(retriever):
    res = retriever.retrieve(process_query("APS007"), allowed_levels_for("employee"))
    assert any(d.metadata.get("employee_id") == "APS007" for d in res["keyword"])


def test_unknown_role_has_no_access():
    with pytest.raises(PermissionError):
        allowed_levels_for("intern")


def test_process_query_keeps_abbreviation_and_adds_expansion():
    assert process_query("What is the WFH policy?") == "what is the wfh (work from home) policy?"
    assert "hr (human resources)" in process_query("Who is the HR manager")
    assert process_query("show chronicle") == "show chronicle"  # no accidental partial matches


def test_tokenizer_ignores_case_and_punctuation():
    assert tokenize("Leave? Yes, LEAVE!") == ["leave", "yes", "leave"]


def test_rrf_rewards_documents_found_by_both_retrievers_and_does_not_mutate_inputs():
    a, b, c = (Document(page_content=t, metadata={"chunk_id": t}) for t in "abc")
    fused = reciprocal_rank_fusion([[a, b], [b, c]], rrf_k=60, max_results=3)
    assert fused[0].metadata["chunk_id"] == "b"
    assert "rrf_score" not in b.metadata


def test_context_only_lists_sources_that_fit_and_marks_archived(retriever, cfg):
    from dataclasses import replace
    res = retriever.retrieve(process_query("casual leave"), allowed_levels_for("employee"), include_archived=True)
    ctx, sources, selected = build_context("casual leave", res["reranked"], replace(cfg, max_context_chars=900))
    assert len(sources) == len(selected) >= 1
    assert ctx.count("SOURCE ") == len(sources)
    full_ctx, full_sources, _ = build_context("casual leave", res["reranked"], cfg)
    if any(not s["is_current"] for s in full_sources):
        assert "ARCHIVED" in full_ctx


def test_tokenizer_matches_simple_plurals():
    assert tokenize("Holidays in January") == tokenize("holiday in january")
