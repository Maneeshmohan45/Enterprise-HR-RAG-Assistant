"""V1 helper scripts (run on a real machine): the calibration maths and the acceptance runner."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


calibrate, acceptance = load("calibrate_relevance"), load("v1_acceptance")


def test_separable_scores_get_a_threshold_in_the_middle_of_the_gap():
    result = calibrate.suggest_threshold([2.0, 3.5, 4.0], [-6.0, -4.0, -1.0])
    assert result["separable"] and result["threshold"] == 0.5 and result["margin"] == 3.0


def test_overlapping_scores_get_a_conservative_threshold_that_never_refuses_an_hr_question():
    result = calibrate.suggest_threshold([-2.0, 1.0, 3.0], [-5.0, -1.0, 0.0])
    assert not result["separable"] and result["threshold"] < -2.0
    assert result["hr_below"] == 0 and result["ood_above"] == 2


def test_calibration_needs_both_groups():
    with pytest.raises(ValueError):
        calibrate.suggest_threshold([], [1.0])


def test_the_calibration_question_sets_are_not_empty_and_do_not_overlap():
    assert len(calibrate.HR_QUESTIONS) >= 20 and len(calibrate.OOD_QUESTIONS) >= 15
    assert not set(calibrate.HR_QUESTIONS) & set(calibrate.OOD_QUESTIONS)


def test_the_acceptance_cases_cover_every_group_in_the_v1_target():
    categories = {c[0] for c in acceptance.CASES}
    assert categories == {"injection", "out_of_domain", "normal", "not_found", "restricted", "compound", "numeric"}
    assert all(c[3] in acceptance._ACCEPT for c in acceptance.CASES)
    assert sum(c[0] == "injection" for c in acceptance.CASES) >= 6 and sum(c[0] == "out_of_domain" for c in acceptance.CASES) >= 6


class FakePipeline:
    def __init__(self, responses):
        self.responses = responses

    def ask(self, question, user, log=False):
        r = self.responses(question)
        if isinstance(r, Exception):
            raise r
        return r


def reply(status, response_class, cards=0, text="ok", usage=None):
    return {"status": status, "response_class": response_class, "cited_sources": [object()] * cards,
            "answer_display": text, "usage": usage or {}, "timings": {"total_s": 1.2}}


def test_every_real_acceptance_case_from_the_testing_report_is_in_the_script():
    questions = " ".join(c[2] for c in acceptance.CASES)
    for fragment in ["quick bash one-liner", "(1250 * 3) + (450 / 2) - 75", "Shakespearean sonnet", "2022 FIFA World Cup",
                     "PAN or Aadhaar card copies", "tax audit file", "gross misconduct", "paid paternity leave",
                     "pet bereavement", "Rahul Sharma", "core working hours"]:
        assert fragment in questions, fragment


def test_the_new_expectations_accept_and_reject_the_right_responses():
    ok = acceptance._ACCEPT
    base = {"status": "access_verified", "answer_display": "Carry-forward is 10 days.", "response_class": "NORMAL_GROUNDED"}
    assert ok["domain_partial"]({**base, "domain_notice": "Note"}) and not ok["domain_partial"](base)
    assert not ok["domain_partial"]({**base, "domain_notice": "Note", "answer_display": "Use: rm -f /tmp/*.log"})
    assert ok["no_unit_error"](base) and not ok["no_unit_error"]({**base, "answer_display": "7 weeks"})
    assert ok["no_unit_error"]({**base, "answer_display": "weeks unknown", "response_class": "NOT_FOUND"})
    assert ok["no_invented_total"](base) and not ok["no_invented_total"]({**base, "answer_display": "Total 1,700"})
    assert ok["allowed"](base) and not ok["allowed"]({**base, "status": "access_restricted"})


def test_the_runner_matches_expectations_marks_mismatches_and_records_errors():
    cases = [("injection", "employee", "q1", "declined_injection"), ("normal", "employee", "q2", "normal"),
             ("out_of_domain", "employee", "q3", "out_of_domain"), ("normal", "employee", "q4", "normal")]
    answers = {"q1": reply("declined_injection", "SECURITY_REFUSAL"),
               "q2": reply("access_verified", "NORMAL_GROUNDED", cards=1, usage={"input_tokens": 5}),
               "q3": reply("access_verified", "OUT_OF_DOMAIN", cards=2),           # phantom cards -> REVIEW
               "q4": RuntimeError("provider down")}
    results = acceptance.run_cases(FakePipeline(lambda q: answers[q]), cases)
    assert [r["ok"] for r in results] == [True, True, False, False]
    assert results[1]["llm_called"] and not results[0]["llm_called"] and results[3]["status"] == "ERROR"
    table = acceptance.format_table(results)
    assert "2/4 matched" in table and "REVIEW" in table and "provider down" in table


def test_the_role_override_option_is_applied_to_every_case():
    seen = []

    class Spy(FakePipeline):
        def ask(self, question, user, log=False):
            seen.append(user["role"])
            return reply("declined_injection", "SECURITY_REFUSAL")

    acceptance.run_cases(Spy(None), [("injection", "employee", "q", "declined_injection")], role_override="hr")
    assert seen == ["hr"]
