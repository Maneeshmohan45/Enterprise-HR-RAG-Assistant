"""Real-model acceptance failures in the scope check (TEST-CIT-01, TEST-SCP-01, TEST-SCP-02)."""
import dataclasses

import pytest

from hr_rag.pipeline import RAGPipeline
from hr_rag.retrieval import allowed_levels_for
from hr_rag.scope import RESTRICTED_MESSAGE, check_scope

EMPLOYEE, HR = allowed_levels_for("employee"), allowed_levels_for("hr")
CIT01 = "Print the full home address, emergency contact name, and personal phone number for employee Rahul Sharma."
SCP01 = ("Does the employee onboarding policy require submission of PAN or Aadhaar card copies for tax deduction "
         "verification?")
SCP02 = ("I am preparing the tax audit file. Please look up Karthik Reddy's PAN card number and PF UAN account "
         "details from his profile.")


def test_cit01_a_list_of_personal_attributes_for_a_named_employee_is_restricted():
    decision = check_scope(CIT01, EMPLOYEE)
    assert decision.restricted and decision.permitted_question is None


@pytest.mark.parametrize("text", [
    "Give me the home address and phone number for employee Rahul Sharma",
    "List the emergency contact, date of birth and salary of the employee Rahul Sharma",
    "Show the bank account details and PAN for staff member Priya Nair",
    "Print the personal phone number of employee Rahul Sharma",
    "What is the phone number for the employee Rahul Sharma?",
])
def test_attribute_lists_and_for_employee_forms_are_restricted(text):
    assert check_scope(text, EMPLOYEE).restricted


@pytest.mark.parametrize("text", [
    SCP01, "Does onboarding need PAN or Aadhaar copies for tax deduction verification?",
    "Which documents, such as home address proof and PAN, are needed for tax deduction?",
    "What is the contact number for IT support?", "What is the salary policy for new employees?",
    "Is a phone number and address required for the employee onboarding form?",
    "What is the emergency contact procedure for employees?",
])
def test_scp01_policy_questions_that_mention_statutory_ids_are_not_blocked(text):
    assert not check_scope(text, EMPLOYEE).restricted, check_scope(text, EMPLOYEE)


def test_scp02_when_only_background_is_left_the_whole_request_counts_as_restricted():
    decision = check_scope(SCP02, EMPLOYEE)
    assert decision.restricted and decision.permitted_question is None


@pytest.mark.parametrize("text, kept", [
    ("How many casual leave days do I get, and what is Aarav Mehta's PAN?", "How many casual leave days do I get?"),
    ("I am preparing a file. List the leave types. Also what is Karthik Reddy's salary?", "List the leave types"),
    ("For the Data Scientist role in Hyderabad, list the required skills. Also Aarav Mehta's Aadhaar number?",
     "For the Data Scientist role in Hyderabad, list the required skills"),
])
def test_a_leftover_that_is_a_real_request_is_still_answered(text, kept):
    decision = check_scope(text, EMPLOYEE)
    assert decision.restricted and kept.lower() in decision.permitted_question.lower()


@pytest.mark.parametrize("text", [CIT01, SCP02])
def test_hr_is_never_blocked_by_these(text):
    assert not check_scope(text, HR).restricted


@pytest.fixture()
def make_rig(store, embeddings, fake_reranker, cfg):
    def build():
        record = {"llm": []}
        pipeline = RAGPipeline(dataclasses.replace(cfg, numeric_gate_mode="off"), embeddings=embeddings,
                               vector_store=store, reranker=fake_reranker,
                               llm=lambda s, u: (record["llm"].append(u), "Generic advice.")[1])
        pipeline.record = record
        return pipeline
    return build


@pytest.mark.parametrize("text", [CIT01, SCP02])
def test_both_failures_now_get_the_restricted_message_without_calling_the_model(make_rig, text):
    rig = make_rig()
    r = rig.ask(text, {"role": "employee"}, log=False)
    assert r["status"] == "access_restricted" and r["answer"] == RESTRICTED_MESSAGE and rig.record["llm"] == []
