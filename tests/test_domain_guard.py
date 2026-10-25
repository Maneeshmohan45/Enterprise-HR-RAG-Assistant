"""V1: the out-of-domain rules (hr_rag/domain_guard.py) on their own.

Clearly non-HR requests (arithmetic, code, creative writing, general-knowledge trivia) must be declined,
and genuine HR questions that merely share words with them must never be.
"""
import inspect
import time

import pytest

from hr_rag.domain_guard import DOMAIN_RULES, OUT_OF_DOMAIN_MESSAGE, analyze_domain
from real_questions import REAL_LEGIT_QUESTIONS

OUT_OF_DOMAIN = [
    ("600 * 700 + 0", "arithmetic"), ("What is 2+2?", "arithmetic"), ("calculate 15 x 4", "arithmetic"),
    ("what is 15% of 200", "arithmetic"), ("square root of 144", "arithmetic"), ("12 plus 30", "arithmetic"),
    ("solve for x", "arithmetic"), ("(3+4)*5", "arithmetic"),
    ("write a python function to reverse a string", "programming"),
    ("Write a Python program to sort a list", "programming"), ("How do I write code in Python?", "programming"),
    ("Create a SQL query to list employees", "programming"), ("```print(1)```", "programming"),
    ("Write a function that checks leave balance", "programming"),
    ("write a poem about the moon", "creative_writing"), ("Compose a haiku about autumn", "creative_writing"),
    ("tell me a joke", "creative_writing"), ("Write a short story about a dragon", "creative_writing"),
    ("poem about the sea", "creative_writing"), ("Write song lyrics about Monday", "creative_writing"),
    ("who won the 2018 football world cup", "general_knowledge"),
    ("What is the capital of France?", "general_knowledge"), ("what's the weather in Paris today", "general_knowledge"),
    ("give me a recipe for pasta", "general_knowledge"), ("Translate good morning into French", "general_knowledge"),
    ("Who founded Google?", "general_knowledge"), ("who is the president of France", "general_knowledge"),
    ("What is the bitcoin price?", "general_knowledge"), ("Who won the IPL last year?", "general_knowledge"),
]
EVASIONS = ["WRITE   A   POEM   ABOUT   THE   MOON", "**write** a *poem* about the moon",
            "wri\u200bte a poem about the moon", "\uff57\uff52\uff49\uff54\uff45 a poem about the moon",
            "What\u00a0is\u00a02+2?"]
# Genuine HR questions that share words with the rules: arithmetic, code, writing, sports, "who won".
NEAR_MISS_LEGIT = [
    "How is my leave balance calculated?", "Does the Data Scientist role require Python and SQL?",
    "Can you write a job description for a Python developer?", "What is 20% of my salary used for in the bonus policy?",
    "If I take 3 days of casual leave from 20, how many remain?", "Who won the employee of the month award?",
    "Is there a company football match on the holiday?", "What is the code of conduct?",
    "Write a resignation letter to HR", "How many casual leaves do I get?", "12-05-2024",
    "What is the notice period?", "Explain the cricket club policy at the office",
    "Can I take leave on the day of the world cup final?", "Which job roles use Python at our company?",
    "How do I apply for leave in the HR portal?", "What is the capital letter rule in the dress code policy?",
    "What is the weather policy for working from home during storms?",
    "Who founded our company and when?", "Which employees can work from home on Fridays?",
    "How do I calculate my notice period?", "Calculate my leave encashment as per the policy",
    "What documents do I need on my first day?", "Is there a poetry club for employees?",
    "Can I get reimbursement for a cooking class under the wellness benefit?",
    "What is the exchange rate policy for travel expenses?", "How many working days are there in a week?",
    "What does the handbook say about writing a code review for HR tools?",
    "Can I write to HR about a grievance?", "I need a story on how onboarding works at the company",
]


@pytest.mark.parametrize("text, category", OUT_OF_DOMAIN)
def test_clearly_non_hr_requests_are_out_of_domain(text, category):
    result = analyze_domain(text)
    assert result.out_of_domain and result.category == category


@pytest.mark.parametrize("text", EVASIONS)
def test_simple_obfuscation_does_not_hide_an_out_of_domain_request(text):
    assert analyze_domain(text).out_of_domain


@pytest.mark.parametrize("text", NEAR_MISS_LEGIT)
def test_genuine_hr_questions_that_share_words_are_not_declined(text):
    assert not analyze_domain(text).out_of_domain, analyze_domain(text)


# Five entries in the logged corpus are the author's own arithmetic tests: they SHOULD be declined.
REAL_ARITHMETIC_TESTS = ["2+2", "600+700*0", "600/0", "do the math and add 700*800+400*600",
                         "do the math and answer the 600*0+700"]


@pytest.mark.parametrize("text", [q for q in REAL_LEGIT_QUESTIONS if q not in REAL_ARITHMETIC_TESTS])
def test_no_real_hr_question_from_manual_testing_is_declined(text):
    assert not analyze_domain(text).out_of_domain


@pytest.mark.parametrize("text", REAL_ARITHMETIC_TESTS)
def test_the_real_arithmetic_tests_from_manual_testing_are_declined(text):
    assert analyze_domain(text).category == "arithmetic"


def test_a_workplace_word_exempts_general_knowledge_but_never_code_or_writing_requests():
    assert not analyze_domain("Who won the employee of the month award?").out_of_domain
    assert analyze_domain("Write a python function that calculates my leave balance").category == "programming"
    assert analyze_domain("Write a poem about our office holiday").category == "creative_writing"


def test_a_bare_date_is_not_arithmetic():
    assert not analyze_domain("12-05-2024").out_of_domain and not analyze_domain("2024/05/12").out_of_domain


def test_the_domain_function_takes_only_the_question():
    assert list(inspect.signature(analyze_domain).parameters) == ["question"]


def test_rules_are_well_formed_and_the_message_is_fixed():
    names = [r["name"] for r in DOMAIN_RULES]
    assert len(names) == len(set(names)) and all(r["patterns"] and r["_compiled"] for r in DOMAIN_RULES)
    assert "HR" in OUT_OF_DOMAIN_MESSAGE and len(OUT_OF_DOMAIN_MESSAGE) < 300


def test_an_adversarial_maximum_length_question_is_checked_quickly():
    started = time.perf_counter()
    analyze_domain("write a " * 120)
    analyze_domain("1 + " * 240 + "1")
    analyze_domain("who " * 240)
    assert time.perf_counter() - started < 0.25
