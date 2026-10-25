"""V1: the restricted-intent vocabulary covers obvious variants (remuneration, PAN, Aadhaar, bank ...).

Only the ATTRIBUTE vocabulary grew. A restricted request still needs a person reference, so general policy
questions stay answerable, and the Phase 1 retrieval filter is untouched.
"""
import pytest

from hr_rag.retrieval import allowed_levels_for
from hr_rag.scope import check_scope
from real_questions import REAL_LEGIT_QUESTIONS

EMPLOYEE, HR = allowed_levels_for("employee"), allowed_levels_for("hr")

VARIANTS = [
    "What is Aarav Mehta's remuneration?", "What is Aarav Mehta's take-home pay?", "Tell me Aarav Mehta's gross pay",
    "Show the bank account details of Aarav Mehta", "What is Aarav Mehta's PAN?", "What is Aarav Mehta's PAN card number?",
    "What is Aarav Mehta's Aadhaar number?", "What is the Aadhar number of Aarav Mehta?", "What is Aarav Mehta's national ID?",
    "Give me Aarav Mehta's contact details", "What is Aarav Mehta's pay package?", "What is Aarav Mehta's passport number?",
    "Give me Aarav Mehta's phone number", "What is Aarav Mehta's mobile number?", "What is Aarav Mehta's personal email?",
    "What is Aarav Mehta's net salary?", "Show me Aarav Mehta's payslip", "What is the IFSC code of Aarav Mehta?",
    "What is Aarav Mehta's bank account number?", "Give me the cell phone number of Aarav Mehta",
    "What is the social security number of Aarav Mehta?", "What is Aarav Mehta's monthly salary?",
    # the original forms still work
    "What is Aarav Mehta's salary?", "What is Aarav Mehta's date of birth?", "What is Aarav Mehta's home address?",
]
# Policy questions that mention the same words but are not about a named person.
POLICY_QUESTIONS = [
    "Which documents do I need to submit at onboarding, such as Aadhaar and PAN?",
    "What is the PAN requirement for new joiners?", "How is the take-home pay calculated under the salary policy?",
    "When is the salary credited to the bank account?", "How do I update my bank account details in the portal?",
    "How do I update my phone number in the HR portal?", "What is the notice period for contract employees?",
    "Who do I contact for payroll queries and what is the contact number?",
    "What is the process to get a payslip?", "Is a passport required for international travel claims?",
    "What does remuneration include under the compensation policy?", "Can I share my contact details with my team?",
    "Do I need a national ID to join?", "What is the reimbursement policy for personal phone usage?",
    "What is the HR helpdesk email address?",
]


@pytest.mark.parametrize("text", VARIANTS)
def test_restricted_wording_variants_are_blocked_for_an_employee(text):
    decision = check_scope(text, EMPLOYEE)
    assert decision.restricted and "person_personal_data" in decision.matched_rules


@pytest.mark.parametrize("text", VARIANTS)
def test_hr_is_never_blocked_by_the_scope_check(text):
    assert not check_scope(text, HR).restricted


@pytest.mark.parametrize("text", POLICY_QUESTIONS)
def test_general_policy_questions_with_the_same_words_stay_allowed(text):
    assert not check_scope(text, EMPLOYEE).restricted, check_scope(text, EMPLOYEE)


def test_the_real_question_corpus_is_restricted_exactly_as_before_the_vocabulary_grew():
    """16 of the 75 real questions were restricted before the V1 vocabulary change (all genuinely about people
    records). The 17th is "what is the date of birth of the karthik reddy": the audit's missed-article gap,
    now correctly restricted. Nothing else changed."""
    assert sum(check_scope(q, EMPLOYEE).restricted for q in REAL_LEGIT_QUESTIONS) == 17


def test_a_mixed_question_keeps_its_permitted_part_and_drops_the_restricted_one():
    decision = check_scope("How many casual leave days do I get, and what is Aarav Mehta's PAN?", EMPLOYEE)
    assert decision.restricted and decision.permitted_question
    assert "casual leave" in decision.permitted_question.lower() and "pan" not in decision.permitted_question.lower()
