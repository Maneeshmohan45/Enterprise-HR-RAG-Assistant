"""V1 invariants: the new controls only NARROW what the assistant does; Phase 1 stays the boundary.

(The import-boundary test for the new modules lives in test_phase4_security_invariants.py, whose module
list now includes domain_guard, assessment and streaming.)
"""
import dataclasses

import pytest

from hr_rag import guardrails
from hr_rag.config import ROLE_ACCESS
from hr_rag.pipeline import RAGPipeline

SECRET_VALUES = ("1988-06-14", "Green Park Road")
OOD = ["What is 600 * 700 + 0?", "Write a poem about the moon", "Who won the 2018 football World Cup?"]
RESTRICTED_VARIANTS = ["What is Aarav Mehta's remuneration?", "What is Aarav Mehta's PAN?",
                       "Give me the bank account details of Aarav Mehta", "What is Aarav Mehta's Aadhaar number?"]


@pytest.fixture()
def make_rig(store, embeddings, fake_reranker, cfg):
    def build(**settings):
        record = {"llm": [], "levels": [], "verify": []}

        def llm(system, user):
            record["llm"].append(user)
            return "Annual leave is 20 days per year [S1]."

        pipeline = RAGPipeline(dataclasses.replace(cfg, **settings), embeddings=embeddings, vector_store=store,
                               reranker=fake_reranker, llm=llm)
        original = pipeline.retriever.retrieve
        pipeline.retriever.retrieve = lambda *a, **k: (record["levels"].append(a[1]), original(*a, **k))[1]
        pipeline.record = record
        return pipeline
    return build


@pytest.mark.parametrize("role", ["employee", "hr"])
@pytest.mark.parametrize("mode", ["shadow", "off"])
@pytest.mark.parametrize("question", OOD)
def test_retrieval_uses_exactly_the_roles_levels_whatever_the_domain_mode(make_rig, role, mode, question):
    rig = make_rig(domain_guard_mode=mode)
    rig.ask(question, {"role": role}, log=False)
    assert rig.record["levels"] == [ROLE_ACCESS[role]]


@pytest.mark.parametrize("role", ["employee", "hr"])
def test_the_relevance_gate_never_changes_which_levels_are_searched(make_rig, role):
    rig = make_rig(relevance_gate_mode="enforce", relevance_min_rerank_score=99.0)
    r = rig.ask("How many casual leave days do I get?", {"role": role}, log=False)
    assert rig.record["levels"] == [ROLE_ACCESS[role]] and r["status"] == "declined_out_of_domain" and rig.record["llm"] == []


@pytest.mark.parametrize("question", OOD)
def test_declines_only_narrow_never_widen(make_rig, question):
    for role in ("employee", "hr"):
        r = make_rig().ask(question, {"role": role}, log=False)
        assert r["status"] == "declined_out_of_domain" and r["sources"] == [] and r["cited_sources"] == []


@pytest.mark.parametrize("domain_mode", ["enforce", "shadow", "off"])
@pytest.mark.parametrize("question", RESTRICTED_VARIANTS)
def test_confidential_text_never_reaches_the_llm_for_an_employee_in_any_mode(make_rig, domain_mode, question):
    rig = make_rig(domain_guard_mode=domain_mode, injection_guard_mode="off")
    rig.ask(question, {"role": "employee"}, log=False)
    assert not any(secret in call for call in rig.record["llm"] for secret in SECRET_VALUES)


@pytest.mark.parametrize("question", RESTRICTED_VARIANTS)
def test_the_wider_restricted_vocabulary_is_a_usability_layer_with_the_filter_still_standing_alone(make_rig, monkeypatch, question):
    from hr_rag import pipeline as pipeline_module
    from hr_rag.scope import ScopeDecision
    monkeypatch.setattr(pipeline_module, "check_scope", lambda q, levels: ScopeDecision(False, None))
    rig = make_rig()
    rig.ask(question, {"role": "employee"}, log=False)                    # scope disabled: only Phase 1 protects
    assert rig.record["levels"] == [ROLE_ACCESS["employee"]]
    assert not any(secret in call for call in rig.record["llm"] for secret in SECRET_VALUES)


@pytest.mark.parametrize("role", ["employee", "hr"])
def test_access_is_verified_with_the_roles_levels_before_the_llm_is_called(make_rig, monkeypatch, role):
    rig = make_rig()
    seen = []
    original = guardrails.verify_access
    monkeypatch.setattr(guardrails, "verify_access",
                        lambda selected, levels, archived: (seen.append((levels, len(rig.record["llm"]))),
                                                            original(selected, levels, archived))[1])
    rig.ask("How many annual leave days do employees get?", {"role": role}, log=False)
    assert seen == [(ROLE_ACCESS[role], 0)]                              # called once, with the role's levels, 0 LLM calls so far


def test_an_unknown_role_is_rejected_before_any_v1_control_runs(make_rig):
    with pytest.raises(PermissionError):
        make_rig().ask("What is 2+2?", {"role": "stranger"}, log=False)
