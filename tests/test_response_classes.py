"""V1: the response classes and source-card rules (hr_rag/assessment.py) as pure functions."""
import pytest

from hr_rag import assessment
from hr_rag.assessment import (ACCESS, NORMAL, NOT_FOUND, OUT_OF_DOMAIN, SECURITY, assess, classify,
                               detect_model_refusal, select_source_cards, source_supports)

SOURCES = [{"text": "Employees are entitled to 10 days of casual leave per year."},
           {"text": "The office is open Monday to Friday from 9 AM to 6 PM."},
           {"text": "Holidays are listed in the annual calendar."}]


# ---------------------------------------------------------------- refusal detection
@pytest.mark.parametrize("answer", [
    "The handbook does not specify a penalty for bringing a pet to the office.",
    "I could not find that in the HR documents.", "There is no information about this in the sources.",
    "This is not mentioned in the provided documents.", "The documents don't say anything about gym benefits.",
    "I couldn't find a policy on this.", "Nothing in the documents covers this.",
])
def test_absence_statements_are_recognised(answer):
    assert detect_model_refusal(answer, []) == "absence"


@pytest.mark.parametrize("answer", [
    "I can only help with questions about HR policies and company documents.",
    "I can't help with that. It is outside the scope of the HR documents.",
    "That is not an HR question, so I cannot assist.",
])
def test_model_declines_of_non_hr_questions_are_recognised(answer):
    assert detect_model_refusal(answer, []) == "decline"


@pytest.mark.parametrize("answer, cited", [
    ("Casual leave is 10 days per year.", []),
    ("The handbook does not specify a penalty, but pets are not permitted [S1].", ["S1"]),
    ("The policy does not specify X. " + "Employees get leave. " * 30, []),
])
def test_normal_answers_and_answers_with_citations_are_not_refusals(answer, cited):
    assert detect_model_refusal(answer, cited) is None


# ---------------------------------------------------------------- classification
def test_status_decides_the_class_of_application_declines():
    expected = {"declined_injection": SECURITY, "access_restricted": ACCESS, "access_blocked": ACCESS,
                "declined_out_of_domain": OUT_OF_DOMAIN, "no_context": NOT_FOUND}
    for status, response_class in expected.items():
        assert classify(status, "anything, even a long confident answer [S1]", ["S1"]) == response_class


def test_a_model_answer_is_classified_by_what_it_says():
    assert classify("access_verified", "Casual leave is 10 days [S1].", ["S1"]) == NORMAL
    assert classify("access_verified", "I could not find that.", []) == NOT_FOUND
    assert classify("access_verified", "I can only help with HR questions.", []) == OUT_OF_DOMAIN


# ---------------------------------------------------------------- source cards
def test_source_support_needs_a_shared_number_or_two_content_words():
    assert source_supports("Casual leave is 10 days.", SOURCES[0]["text"])          # shared number 10 and words
    assert source_supports("The office opens at 9 AM.", SOURCES[1]["text"])
    assert not source_supports("The figure is 420000.", SOURCES[0]["text"])
    assert not source_supports("Bananas are yellow fruit.", SOURCES[0]["text"])
    assert source_supports("Holidays.", SOURCES[2]["text"])                         # one-word answers need one shared word


def test_only_cited_and_supported_sources_become_cards():
    cards, hidden = select_source_cards("Casual leave is 10 days [S1]. Office opens at 9 AM [S2].", SOURCES,
                                        ["S1", "S2"], NORMAL)
    assert cards == [SOURCES[0], SOURCES[1]] and hidden == []
    cards, hidden = select_source_cards("The figure is 420000 [S3].", SOURCES, ["S3"], NORMAL)
    assert cards == [] and hidden == ["S3"]


def test_uncited_sources_are_never_shown_and_there_is_no_top_three_fallback():
    assert select_source_cards("Casual leave is 10 days.", SOURCES, [], NORMAL) == ([], [])


@pytest.mark.parametrize("response_class", [SECURITY, ACCESS, OUT_OF_DOMAIN, NOT_FOUND])
def test_refusal_classes_never_show_cards_even_if_the_text_cites(response_class):
    assert select_source_cards("Casual leave is 10 days [S1].", SOURCES, ["S1"], response_class) == ([], [])


# ---------------------------------------------------------------- checks per class
MESSAGES = ("Fixed refusal message.",)


def test_an_application_decline_passes_only_with_the_fixed_message_and_no_cards():
    ok = assess(OUT_OF_DOMAIN, llm_called=False, answer="Fixed refusal message.", cards=[], context_text="",
                question="q", fixed_messages=MESSAGES, unsupported_ids=[])
    assert ok["passed"] is True
    leaked = assess(OUT_OF_DOMAIN, llm_called=False, answer="Fixed refusal message.", cards=[SOURCES[0]],
                    context_text="", question="q", fixed_messages=MESSAGES, unsupported_ids=[])
    assert leaked["passed"] is False and leaked["checks"]["no_source_cards"] is False
    changed = assess(ACCESS, llm_called=False, answer="Something else", cards=[], context_text="", question="q",
                     fixed_messages=MESSAGES, unsupported_ids=[])
    assert changed["passed"] is False and changed["checks"]["fixed_message"] is False


def test_a_not_found_answer_passes_when_it_invents_no_numbers_and_shows_no_cards():
    ok = assess(NOT_FOUND, llm_called=True, answer="The documents do not specify this.", cards=[],
                context_text="casual leave is 10 days", question="penalty for pets?", fixed_messages=MESSAGES,
                unsupported_ids=[])
    assert ok["passed"] is True and ok["checks"] == {"no_source_cards": True, "no_invented_numbers": True}
    invented = assess(NOT_FOUND, llm_called=True, answer="Not specified, probably 26 weeks.", cards=[],
                      context_text="casual leave is 10 days", question="penalty for pets?", fixed_messages=MESSAGES,
                      unsupported_ids=[])
    assert invented["passed"] is False


def test_a_normal_answer_has_no_pass_fail_because_the_quality_score_judges_it():
    result = assess(NORMAL, llm_called=True, answer="a", cards=[], context_text="", question="q",
                    fixed_messages=MESSAGES, unsupported_ids=["S2"])
    assert result["passed"] is None and result["checks"] == {"citations_supported": False}


def test_the_class_list_is_the_five_classes_from_the_v1_target():
    assert assessment.CLASSES == ("NORMAL_GROUNDED", "SECURITY_REFUSAL", "ACCESS_RESTRICTED", "OUT_OF_DOMAIN", "NOT_FOUND")
