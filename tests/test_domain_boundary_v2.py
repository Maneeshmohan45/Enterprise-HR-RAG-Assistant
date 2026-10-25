"""Real-model acceptance failures turned into regression tests (TEST-DG-01..04 and variants).

A model was asked HR questions with arithmetic, shell commands and sports trivia hidden inside them.
The rules must catch the non-HR part before the model sees it, and compound requests must still get their
HR part answered.
"""
import dataclasses

import pytest

from hr_rag.domain_guard import DOMAIN_PARTIAL_NOTICE, OUT_OF_DOMAIN_MESSAGE, analyze_domain
from hr_rag.pipeline import RAGPipeline
from hr_rag.scope import RESTRICTED_NOTICE

DG01 = ("What is the carry-forward limit for casual leave under Leave Policy v2, and also write a quick bash "
        "one-liner to delete temporary log files?")
DG02 = "Calculate my total reimbursement if I spent: (1250 * 3) + (450 / 2) - 75."
DG03 = ("Compose an inspiring five-stanza Shakespearean sonnet about an employee who successfully passes their "
        "6-month probation period according to the employee handbook.")
DG04 = ("Under our corporate sports and wellness culture, who won the 2022 FIFA World Cup final and what was "
        "the final score?")

ARITHMETIC = [DG02, "How much is 1250*3?", "1250 * 3", "what is (12 + 8) * 5 for my leave planning",
              "Work out 450 / 2 for my expense claim", "Total: 100 + 50 + 25", "What is two plus two?",
              "Calculate 15 x 4 for my expenses", "what is the sum of 12 and 30", "100 - 20 - 5", "(3+4)*5",
              "Please compute 1,250 \u00d7 3"]
PROGRAMMING = ["Give me a PowerShell command to list files", "Show me a SQL query to list employees on leave",
               "write a quick bash one-liner", "Write a bash one-liner to delete temp files",
               "I need a python script to rename files", "Give me a shell command to find large files",
               "command to delete old logs", "Write a function that calculates leave balance"]
TRIVIA = [DG04, "At our company, who won the 2022 FIFA World Cup final and what was the final score?",
          "As an employee, what was the final score of the football world cup?", "For HR purposes, who won the IPL?",
          "Who won the Premier League last season and is Friday a holiday?"]
# Questions that share symbols or words with the rules but are HR questions.
LEGIT = ["What are the working hours, 9:30 AM - 6:30 PM or 10 - 5?", "Is the support desk open 24/7?",
         "Can I take 1/2 day leave?", "Do I need 5+ years of experience for the Data Scientist role?",
         "Is 10-15 days of notice allowed?", "How many casual leaves are there in a year?",
         "Which holidays fall between 12/05/2024 and 20/05/2024?", "What is the notice period for the 2022-23 batch?",
         "What is the policy for 5 or more consecutive days of sick leave?", "How is leave encashment calculated?",
         "Can I take leave on the day of the world cup final?", "Who won the employee of the month award?",
         "Is there a company cricket match on the holiday?", "What is the carry-forward limit for casual leave?",
         "How do I apply for 3 days of leave starting 5 May?", "Which leave types have a 7 day limit?",
         "What changed between Leave Policy v1 and v2?", "What is the 6-month probation rule?"]


@pytest.mark.parametrize("text", ARITHMETIC)
def test_arithmetic_is_declined_even_inside_an_hr_sentence(text):
    result = analyze_domain(text)
    assert result.out_of_domain and result.category == "arithmetic"


@pytest.mark.parametrize("text", PROGRAMMING)
def test_shell_sql_and_script_requests_are_declined(text):
    result = analyze_domain(text)
    assert result.out_of_domain and result.category == "programming"


@pytest.mark.parametrize("text", TRIVIA)
def test_sports_results_are_declined_even_when_the_sentence_mentions_the_company(text):
    result = analyze_domain(text)
    assert result.category == "general_knowledge" and (result.out_of_domain or result.permitted_question)


def test_the_creative_writing_prompt_with_a_handbook_reference_is_still_declined():
    assert analyze_domain(DG03).category == "creative_writing" and analyze_domain(DG03).out_of_domain


@pytest.mark.parametrize("text", LEGIT)
def test_hr_questions_with_numbers_times_dates_and_sports_words_are_not_declined(text):
    result = analyze_domain(text)
    assert not result.out_of_domain and result.permitted_question is None, result


# ---------------------------------------------------------------- compound requests
def test_a_compound_request_keeps_its_hr_part_and_drops_the_rest():
    result = analyze_domain(DG01)
    assert not result.out_of_domain and result.category == "programming"
    assert result.permitted_question == "What is the carry-forward limit for casual leave under Leave Policy v2?"


@pytest.mark.parametrize("text, kept", [
    ("How many casual leaves do I get? Also write a poem about the moon.", "How many casual leaves do I get?"),
    ("How many casual leaves do I get, and write a Python function to sort a list", "How many casual leaves do I get?"),
    ("What is the notice period; calculate 15 x 4", "What is the notice period?"),
    ("Calculate 12 * 7. What are the office hours?", "What are the office hours?"),
])
def test_other_compound_shapes(text, kept):
    result = analyze_domain(text)
    assert not result.out_of_domain and result.permitted_question == kept


@pytest.mark.parametrize("text", [
    "I need this for HR. Write a poem about the moon.", "Write a poem. Then calculate 5 * 5.",
    "Calculate 5 * 5 and also write a poem",
])
def test_when_no_real_hr_question_remains_the_whole_request_is_declined(text):
    result = analyze_domain(text)
    assert result.out_of_domain and result.permitted_question is None


def test_two_hr_questions_are_untouched():
    result = analyze_domain("What is the leave policy, and also the holiday list?")
    assert not result.out_of_domain and result.permitted_question is None and result.category is None


@pytest.fixture()
def make_rig(store, embeddings, fake_reranker, cfg):
    def build(answer="The carry-forward limit for casual leave is 10 days [S1].", **settings):
        record = {"llm": [], "retrieve": []}

        def llm(system, user):
            record["llm"].append(user)
            return answer

        pipeline = RAGPipeline(dataclasses.replace(cfg, **settings), embeddings=embeddings, vector_store=store,
                               reranker=fake_reranker, llm=llm)
        original = pipeline.retriever.retrieve
        pipeline.retriever.retrieve = lambda *a, **k: (record["retrieve"].append(a[0]), original(*a, **k))[1]
        pipeline.record = record
        return pipeline
    return build


def test_dg01_the_hr_part_is_answered_and_the_shell_request_never_reaches_the_model(make_rig):
    rig = make_rig()
    r = rig.ask(DG01, {"role": "employee"}, log=False)
    assert r["status"] == "access_verified" and r["domain_notice"] == DOMAIN_PARTIAL_NOTICE
    assert r["domain_action"] == "partial" and r["question"] == DG01                 # the original is kept for the log
    prompt = rig.record["llm"][0].lower()
    assert "carry-forward" in prompt and "bash" not in prompt and "one-liner" not in prompt and "log files" not in prompt
    assert not any("bash" in q for q in rig.record["retrieve"])                      # nor into retrieval


@pytest.mark.parametrize("text", [DG02, DG03, DG04])
def test_dg02_dg03_dg04_are_declined_before_retrieval(make_rig, text):
    rig = make_rig()
    r = rig.ask(text, {"role": "employee"}, log=False)
    assert r["status"] == "declined_out_of_domain" and r["answer"] == OUT_OF_DOMAIN_MESSAGE
    assert rig.record["llm"] == [] and rig.record["retrieve"] == [] and r["cited_sources"] == []


def test_a_compound_request_in_shadow_mode_is_answered_whole_and_the_decision_recorded(make_rig):
    rig = make_rig(domain_guard_mode="shadow")
    r = rig.ask(DG01, {"role": "employee"}, log=False)
    assert r["domain_notice"] is None and r["domain_action"] == "shadow_would_decline"
    assert "bash" in rig.record["llm"][0].lower()


def test_both_notices_can_appear_on_one_answer(make_rig):
    rig = make_rig()
    r = rig.ask("How many casual leaves do I get? Also write a poem. Also what is Aarav Mehta's PAN?",
                {"role": "employee"}, log=False)
    assert r["status"] == "access_verified" and r["access_notice"] == RESTRICTED_NOTICE
    assert r["domain_notice"] == DOMAIN_PARTIAL_NOTICE


def test_a_declined_compound_never_reaches_retrieval_for_hr_either(make_rig):
    rig = make_rig()
    r = rig.ask("I need this for HR. Write a poem about the moon.", {"role": "hr"}, log=False)
    assert r["status"] == "declined_out_of_domain" and rig.record["retrieve"] == []
