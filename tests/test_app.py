"""Runs the real app.py headlessly with a real pipeline built on fake embeddings (offline)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import hr_rag.config as config  # noqa: E402
import hr_rag.pipeline as pipeline_module  # noqa: E402
from hr_rag.generation import LLMError  # noqa: E402


@pytest.fixture()
def app(monkeypatch, store, embeddings, fake_reranker, cfg):
    import streamlit as st
    st.cache_resource.clear()
    state = {"llm": lambda system, user: ("Annual leave is 20 days per year [S1].",
                                          {"input_tokens": 900, "output_tokens": 12})}
    real = pipeline_module.RAGPipeline

    def factory(settings_obj, **_):
        return real(settings_obj, embeddings=embeddings, vector_store=store, reranker=fake_reranker,
                    llm=lambda system, user: state["llm"](system, user))

    monkeypatch.setattr(config, "settings", cfg)
    monkeypatch.setattr(pipeline_module, "RAGPipeline", factory)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")   # whichever provider cfg uses
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
    at.state = state
    return at


# Streamlit shows a leading emoji as the alert's icon, so the warning is checked as icon + text.
SWITCH_WARNING_TEXT = ("Switching roles will clear your current conversation. "
                       "Your previous conversation will not be available after switching roles. Continue?")


def switch_warning_shown(app):
    return any(w.icon == "⚠️" and w.value == SWITCH_WARNING_TEXT for w in app.warning)


def switch_role(app, role):
    """Change the role the way a user does: pick it, then confirm."""
    app.selectbox[0].set_value(role).run()
    app.button(key="role_confirm").click().run()


def page_text(app):
    return " ".join([m.value for m in app.markdown] + [c.value for c in app.caption] + [w.value for w in app.warning])


def ask(app, question):
    app.chat_input[0].set_value(question).run()
    assert not app.exception


def test_question_gets_answer_and_sources(app):
    app.run()
    assert not app.exception
    app.chat_input[0].set_value("How many annual leave days do employees get?").run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Annual leave is 20 days per year" in text and "[S1]" not in text
    assert any("Sources (" in e.label for e in app.expander)
    assert any("Quality check" in m.value for m in app.markdown)


def test_llm_error_shows_friendly_message_not_a_crash(app):
    def boom(system, user):
        raise LLMError("nvidia request failed: Error code: 401 key nvapi-abcdef123456")
    app.state["llm"] = boom
    app.run()
    app.chat_input[0].set_value("How many annual leave days?").run()
    assert not app.exception
    assert any("rejected the API key" in e.value for e in app.error)
    assert "nvapi-abcdef123456" not in " ".join(str(c.value) for c in app.code)


def test_employee_asking_for_confidential_data_gets_no_confidential_text(app, monkeypatch):
    # Phase 1 guarantee with the scope check off: only the retrieval filter protects the data.
    from hr_rag.scope import ScopeDecision
    monkeypatch.setattr(pipeline_module, "check_scope", lambda question, levels: ScopeDecision(False, None))
    seen = []
    app.state["llm"] = lambda system, user: (seen.append(user) or "Not in the sources.", {})
    app.run()
    app.chat_input[0].set_value("What is Aarav Mehta's date of birth and home address?").run()
    assert not app.exception and seen and "1988-06-14" not in seen[0]


def test_restricted_question_shows_the_warning_and_never_calls_the_llm(app):
    from hr_rag.scope import RESTRICTED_MESSAGE
    seen = []
    app.state["llm"] = lambda system, user: (seen.append(user) or "x", {})
    app.run()
    app.chat_input[0].set_value("Who is the head of the Data Science department?").run()
    assert not app.exception and seen == []
    assert [w.value for w in app.warning] == [RESTRICTED_MESSAGE]
    assert not any("Sources (" in e.label for e in app.expander)
    assert not [b for b in app.button if b.label in ("👍", "👎")]      # nothing to rate


def test_mixed_question_shows_the_answer_and_a_separate_notice(app):
    from hr_rag.scope import RESTRICTED_NOTICE
    seen = []
    app.state["llm"] = lambda system, user: (seen.append(user) or "Annual leave is 20 days per year [S1].", {})
    app.run()
    app.chat_input[0].set_value("How many annual leave days do employees get? Also, what is Aarav Mehta's date of birth?").run()
    assert not app.exception and len(seen) == 1 and "aarav" not in seen[0].lower()
    assert "Annual leave is 20 days per year" in " ".join(m.value for m in app.markdown)
    assert [w.value for w in app.warning] == [RESTRICTED_NOTICE]
    assert any("Sources (" in e.label for e in app.expander)


def test_hr_is_not_blocked_by_the_same_question(app):
    from hr_rag.scope import RESTRICTED_MESSAGE
    seen = []
    app.state["llm"] = lambda system, user: (seen.append(user) or "Answer [S1].", {})
    app.run()
    switch_role(app, "hr")
    app.chat_input[0].set_value("Who is the head of the Data Science department?").run()
    assert not app.exception and len(seen) == 1
    assert RESTRICTED_MESSAGE not in [w.value for w in app.warning]


def test_other_pages_render(app):
    app.run()
    for label in ("Model Evaluation", "About"):
        next(b for b in app.sidebar.button if b.label == label).click().run()
        assert not app.exception


def test_missing_api_key_shows_message_and_disables_input(app, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    app.run()
    assert not app.exception
    assert any("not configured" in e.value for e in app.error)
    assert app.chat_input[0].disabled


def test_only_employee_and_hr_roles_are_offered_and_employee_is_default(app):
    app.run()
    assert not app.exception
    role = app.selectbox[0]
    assert role.options == ["Employee", "HR"]
    assert role.value == "employee"


def test_stale_role_in_session_falls_back_to_employee(app):
    app.session_state["role"] = "admin"      # e.g. left over from an older version of the UI
    app.run()
    assert not app.exception
    assert app.selectbox[0].value == "employee"


def test_hr_role_can_be_selected_and_backend_roles_are_unchanged(app):
    app.run()
    switch_role(app, "hr")
    assert not app.exception and app.session_state["role"] == "hr"
    assert set(config.ROLE_ACCESS) == {"employee", "manager", "hr", "admin"}


# ---------------------------------------------------------------------------
# Phase 3: switching role always starts a fresh conversation
# ---------------------------------------------------------------------------
LEAVE_Q = "How many annual leave days do employees get?"
HR_Q = "Who is the head of the Data Science department?"


def test_changing_the_role_asks_for_confirmation_and_changes_nothing_yet(app):
    app.run()
    ask(app, LEAVE_Q)
    before = [c["id"] for c in app.session_state["conversations"]]
    app.selectbox[0].set_value("hr").run()
    assert not app.exception
    assert switch_warning_shown(app)
    assert app.session_state["role"] == "employee"                       # not switched yet
    assert [c["id"] for c in app.session_state["conversations"]] == before
    assert "Annual leave is 20 days per year" in page_text(app)           # conversation still visible
    assert app.chat_input[0].disabled                                      # no questions while undecided
    assert app.button(key="role_confirm") and app.button(key="role_cancel")


def test_employee_to_hr_clears_the_conversation(app):
    app.run()
    ask(app, LEAVE_Q)
    ask(app, HR_Q)                                                         # a refused question too
    old = app.session_state["conversations"]
    assert len(old[0]["messages"]) == 4
    old_ids = {c["id"] for c in old} | {m["id"] for c in old for m in c["messages"]}
    switch_role(app, "hr")
    assert not app.exception and app.session_state["role"] == "hr"
    fresh = app.session_state["conversations"]
    assert len(fresh) == 1 and fresh[0]["messages"] == []                  # one new, empty conversation
    assert not ({c["id"] for c in fresh} & old_ids)
    assert not switch_warning_shown(app) and not app.chat_input[0].disabled


def test_hr_to_employee_clears_the_conversation(app):
    app.run()
    switch_role(app, "hr")
    ask(app, HR_Q)
    old = app.session_state["conversations"]
    assert len(old[0]["messages"]) == 2
    old_ids = {c["id"] for c in old} | {m["id"] for c in old for m in c["messages"]}
    switch_role(app, "employee")
    assert not app.exception and app.session_state["role"] == "employee"
    fresh = app.session_state["conversations"]
    assert len(fresh) == 1 and fresh[0]["messages"] == [] and not ({c["id"] for c in fresh} & old_ids)


def test_cancelling_keeps_the_role_and_the_whole_conversation(app):
    app.run()
    ask(app, LEAVE_Q)
    ask(app, HR_Q)
    before = [(c["id"], [(m["id"], m["content"]) for m in c["messages"]]) for c in app.session_state["conversations"]]
    app.selectbox[0].set_value("hr").run()
    app.button(key="role_cancel").click().run()
    assert not app.exception and app.session_state["role"] == "employee"
    assert [(c["id"], [(m["id"], m["content"]) for m in c["messages"]])
            for c in app.session_state["conversations"]] == before
    assert app.selectbox[0].value == "employee"                            # the dropdown is back too
    assert not switch_warning_shown(app) and not app.chat_input[0].disabled
    assert "Annual leave is 20 days per year" in page_text(app)
    ask(app, LEAVE_Q)                                                      # and chatting carries on
    assert len(app.session_state["conversations"][0]["messages"]) == 6


def test_choosing_the_current_role_again_withdraws_the_request(app):
    app.run()
    ask(app, LEAVE_Q)
    app.selectbox[0].set_value("hr").run()
    assert switch_warning_shown(app)
    app.selectbox[0].set_value("employee").run()
    assert not switch_warning_shown(app) and not app.chat_input[0].disabled
    assert len(app.session_state["conversations"][0]["messages"]) == 2


def test_previous_messages_are_not_visible_after_the_switch(app):
    app.state["llm"] = lambda system, user: ("The unmistakable old answer [S1].", {})
    app.run()
    ask(app, "What is the unmistakable old question about leave?")
    assert "unmistakable old answer" in page_text(app) and "unmistakable old question" in page_text(app)
    switch_role(app, "hr")
    text = page_text(app)
    assert "unmistakable" not in text
    switch_role(app, "employee")                                           # and nothing comes back later
    assert "unmistakable" not in page_text(app) and app.session_state["conversations"][0]["messages"] == []


def test_previous_sources_and_citations_are_cleared(app):
    app.run()
    ask(app, LEAVE_Q)
    assert any("Sources (" in e.label for e in app.expander)
    switch_role(app, "hr")
    assert not any("Sources (" in e.label for e in app.expander)
    assert "leave_policy" not in repr(app.session_state["conversations"])  # no source data kept anywhere
    assert "leave_policy" not in page_text(app)


def test_previous_feedback_and_session_state_are_cleared(app):
    app.run()
    ask(app, LEAVE_Q)
    message_id = app.session_state["conversations"][0]["messages"][1]["id"]
    app.button(key=f"up_{message_id}").click().run()
    assert "Thanks for the feedback." in page_text(app)
    app.text_input(key="conv_search").set_value("annual").run()
    switch_role(app, "hr")
    assert "Thanks for the feedback." not in page_text(app)
    assert app.text_input(key="conv_search").value == ""
    leftovers = [k for k in list(app.session_state) if message_id in str(k)]
    assert not leftovers and message_id not in repr(app.session_state["conversations"])
    ask(app, LEAVE_Q)                                                      # fresh state: rating is possible again
    new_id = app.session_state["conversations"][0]["messages"][1]["id"]
    assert app.session_state["conversations"][0]["messages"][1]["feedback"] is None
    assert app.button(key=f"up_{new_id}") and app.button(key=f"down_{new_id}")


def test_two_switches_never_bring_an_old_conversation_back(app):
    app.run()
    ask(app, LEAVE_Q)
    switch_role(app, "hr")
    ask(app, HR_Q)
    switch_role(app, "employee")
    assert [len(c["messages"]) for c in app.session_state["conversations"]] == [0]


def test_settings_that_are_not_conversation_data_are_kept(app):
    app.run()
    app.toggle[0].set_value(True).run()                                    # "Include archived policies"
    switch_role(app, "hr")
    assert app.session_state["include_archived"] is True


def test_employee_still_works_normally_after_switching_from_hr(app):
    seen = []
    app.state["llm"] = lambda system, user: (seen.append(user) or "Annual leave is 20 days per year [S1].", {})
    app.run()
    switch_role(app, "hr")
    ask(app, HR_Q)                                                         # HR gets an answer
    assert len(seen) == 1
    switch_role(app, "employee")
    ask(app, LEAVE_Q)                                                      # normal RAG answer for the employee
    assert len(seen) == 2 and any("Sources (" in e.label for e in app.expander)
    assert "Annual leave is 20 days per year" in page_text(app)
    ask(app, HR_Q)                                                         # Phase-2 refusal still applies
    assert len(seen) == 2 and config_restricted_message() in [w.value for w in app.warning]
    answered = [m for m in app.session_state["conversations"][0]["messages"] if m["role"] == "assistant"]
    assert answered[0]["response"]["role"] == "employee"


def test_hr_still_works_normally_after_switching_from_employee(app):
    seen = []
    app.state["llm"] = lambda system, user: (seen.append(user) or "Karthik answers [S1].", {})
    app.run()
    ask(app, HR_Q)                                                         # refused as employee
    assert seen == [] and config_restricted_message() in [w.value for w in app.warning]
    switch_role(app, "hr")
    ask(app, HR_Q)                                                         # the same question now answered
    assert len(seen) == 1 and config_restricted_message() not in [w.value for w in app.warning]
    assert app.session_state["conversations"][0]["messages"][1]["response"]["role"] == "hr"
    ask(app, LEAVE_Q)                                                      # and ordinary questions too
    assert len(seen) == 2


def test_about_limitations_documents_the_missing_hr_authentication(app):
    app.run()
    next(b for b in app.sidebar.button if b.label == "About").click().run()
    assert not app.exception
    text = page_text(app)
    assert "does not require a passcode to access the HR role" in text
    assert "intentional for the testing and demonstration phase" in text
    assert "Planned for V2" in text and "authentication/passcode" in text
    assert "role-based retrieval controls still keep the Employee role away from confidential HR data" in text


def config_restricted_message():
    from hr_rag.scope import RESTRICTED_MESSAGE
    return RESTRICTED_MESSAGE
