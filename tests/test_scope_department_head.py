"""V1.1 fix 2: "department head approval" is a leave-approval question, not a directory request.

Confirmed failure (Employee agent, Q29): the leave question
  "What is the maximum duration of continuous annual leave an employee can take before department head
   approval is required?"
was refused as a confidential directory request. The rule treated ANY word before "department head" as a
department name ("before department head", "without department head", "inform department head").
"""
import dataclasses

import pytest

from hr_rag.pipeline import RAGPipeline
from hr_rag.retrieval import allowed_levels_for
from hr_rag.scope import RESTRICTED_MESSAGE, check_scope
from real_questions import REAL_LEGIT_QUESTIONS

EMPLOYEE, HR = allowed_levels_for("employee"), allowed_levels_for("hr")
Q29 = ("What is the maximum duration of continuous annual leave an employee can take before department head "
       "approval is required?")

# Genuine leave / policy questions that mention a department head or a role title.
ALLOWED = [
    Q29, "Who gives department head approval for long leave?", "Can I take leave without department head approval?",
    "Does leave over 10 working days need department head approval?", "Should I inform department head about my absence?",
    "Is department head approval required for casual leave?", "What is the process for department head approval?",
    "Leave requires approval of the department head", "Do I need department head sign-off for leave beyond 10 days?",
    "Which leave needs approval from the department head?", "Can the department head approve my leave?",
    "Does my manager approve leave or the department head?", "How do I contact my department head about leave?",
    "Is the head of Engineering approval needed for leave?", "Does the head of Engineering approve sabbatical leave?",
    "Who approves leave if my manager is away?", "Can I submit leave to the department head directly?",
    "Is department head permission needed to work from home?", "After department head review, who informs HR?",
    # job titles are described in the job descriptions; they are not requests for a person
    "What are the responsibilities of an HR manager according to the job descriptions?",
    "What does the Engineering manager do?", "What are the responsibilities of the head of HR?",
    "Show me the job description for the Sales lead", "What qualifications does a Finance manager need?",
    "What is the role of the Engineering director?", "Who approves leave: HR manager or department head?",
    "Tell me the HR department head approval process", "Show me the Engineering department head approval rules",
]
# Genuine requests for who a department's head is, or for a department's headcount or budget.
RESTRICTED = [
    "Who is the department head of Engineering?", "Who is the Engineering department head?",
    "Give me the department head of Data Science", "Name the head of the Support department",
    "Who is the head of Engineering?", "Who leads DevOps?", "Who manages Sales?", "Who heads the Marketing department?",
    "What is the headcount of DevOps?", "What is the budget of the Finance department?",
    "Who is the director of Quality Assurance?", "Who is the Management department head and what is his email?",
    "Tell me the head of Finance", "Engineering department head name", "Show me the email of the Engineering department head",
    "Who is the head of HR?", "Who is the department head of DEP001?", "Contact details of the Sales department head",
]


@pytest.mark.parametrize("text", ALLOWED)
def test_leave_and_job_title_questions_are_not_blocked(text):
    assert not check_scope(text, EMPLOYEE).restricted, check_scope(text, EMPLOYEE)


@pytest.mark.parametrize("text", RESTRICTED)
def test_genuine_requests_for_a_department_head_or_headcount_are_still_blocked(text):
    assert check_scope(text, EMPLOYEE).restricted


@pytest.mark.parametrize("text", ALLOWED + RESTRICTED)
def test_hr_is_never_blocked_by_the_scope_check(text):
    assert not check_scope(text, HR).restricted


def test_the_real_question_corpus_is_restricted_exactly_as_before_this_fix():
    assert sum(check_scope(q, EMPLOYEE).restricted for q in REAL_LEGIT_QUESTIONS) == 17


def test_a_mixed_question_keeps_the_leave_part_and_drops_the_department_head_request():
    decision = check_scope("What is the leave approval process? Also, who is the head of Engineering?", EMPLOYEE)
    assert decision.restricted and decision.permitted_question == "What is the leave approval process?"


@pytest.fixture()
def make_rig(store, embeddings, fake_reranker, cfg):
    def build(answer="Leave beyond 10 working days needs department head approval [S1]."):
        record = {"llm": []}
        pipeline = RAGPipeline(dataclasses.replace(cfg, numeric_gate_mode="off"), embeddings=embeddings,
                               vector_store=store, reranker=fake_reranker,
                               llm=lambda s, u: (record["llm"].append(u), answer)[1])
        pipeline.record = record
        return pipeline
    return build


def test_q29_now_reaches_the_model_and_is_answered(make_rig):
    rig = make_rig()
    r = rig.ask(Q29, {"role": "employee"}, log=False)
    assert r["status"] == "access_verified" and r["answer"] != RESTRICTED_MESSAGE and len(rig.record["llm"]) == 1


def test_a_genuine_department_head_request_still_gets_the_restricted_message_without_calling_the_model(make_rig):
    rig = make_rig()
    r = rig.ask("Who is the department head of Engineering?", {"role": "employee"}, log=False)
    assert r["status"] == "access_restricted" and r["answer"] == RESTRICTED_MESSAGE and rig.record["llm"] == []
