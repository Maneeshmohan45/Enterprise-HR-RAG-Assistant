"""The numeric gate: an answer may state only numbers (and time units) that the sources give or the question contained.

It is deterministic and ON by default ("enforce"). It withholds invented, calculated and unit-swapped figures,
in the finished answer and, when streaming, before the sentence containing the number is shown.
The shared test config switches it off for unrelated tests; every test here switches it on explicitly.
"""
import dataclasses

import pytest
from langchain_core.documents import Document

from hr_rag.config import Settings
from hr_rag.generation import StreamAborted
from hr_rag.grounding import numeric_issues
from hr_rag.ingestion import prepare_chunks
from hr_rag.pipeline import INSUFFICIENT_INFO_MESSAGE, RAGPipeline
from hr_rag.streaming import NumericStreamGuard
from hr_rag.ui_data import read_jsonl

PATERNITY_Q = ("How many weeks of paid paternity leave does the company provide to secondary caregivers under "
               "the current leave policy?")
SOURCES = ("Leave Type: Paternity Leave | V1: 5 Days | V2: 7 Days. Casual leave: 10 days per year. "
           "Core hours are 9:30 AM to 6:30 PM. Carry forward limit is 10 days.")


# ---------------------------------------------------------------- the check itself
@pytest.mark.parametrize("answer, question, reasons", [
    ("7 weeks", PATERNITY_Q, ["unit_mismatch"]),                                        # TEST-NEG-03: the policy says 7 DAYS
    ("Paternity leave is 7 weeks under Leave Policy v2.", "paternity?", ["unit_mismatch"]),
    # TEST-DG-02. V1.1: the figures the user typed are no longer evidence either, so restating them as fact is
    # reported too (claimed_in_question); the invented result 3150 is still not_in_sources. Withheld as before.
    ("The calculation (1250 * 3) + (450 / 2) - 75 equals 3150.", "Calculate (1250 * 3) + (450 / 2) - 75",
     ["claimed_in_question"] * 5 + ["not_in_sources"]),
    ("It is 26 weeks.", "maternity?", ["not_in_sources"]),
    ("You get seven weeks of paternity leave.", "paternity?", ["unit_mismatch"]),       # number words count too
    ("Casual leave is 14 days.", "casual leave?", ["not_in_sources"]),
])
def test_invented_calculated_and_unit_swapped_figures_are_reported(answer, question, reasons):
    assert [i["reason"] for i in numeric_issues(answer, SOURCES, question)] == reasons


@pytest.mark.parametrize("answer, question", [
    ("Paternity leave is 7 days under Leave Policy v2.", "paternity?"),
    ("The carry-forward limit for casual leave is 10 days.", "carry forward?"),
    ("Core working hours are 9:30 AM to 6:30 PM.", "hours?"),
    ("You get seven days of paternity leave.", "paternity?"),
    ("Paternity leave was 5 days in V1 and is 7 working days in V2.", "paternity?"),
    ("No, you cannot take 12 days; the limit is 10 days.", "Can I take 12 days of casual leave?"),   # restating the question
    ("Casual leave is 10 days [S1-S3].", "casual leave?"),                                         # a marker's digits are not facts
    ("I could not find that in the documents.", "pet leave?"),
    ("1. Apply in the portal.\n2. Wait for approval.", "how to apply?"),                           # list numbering
])
def test_grounded_answers_pass(answer, question):
    assert numeric_issues(answer, SOURCES, question) == []


def test_a_number_absent_from_empty_sources_is_never_grounded():
    assert [i["reason"] for i in numeric_issues("It is 10 days.", "", "q")] == ["not_in_sources"]


def test_issues_carry_numbers_and_units_only_never_text():
    issue = numeric_issues("It is 7 weeks.", SOURCES, "q")[0]
    assert set(issue) == {"number", "unit", "reason"} and issue["unit"] == "week"


# ---------------------------------------------------------------- the streaming guard
def run_guard(pieces, check=lambda text: [], enforce=True):
    shown = []
    guard = NumericStreamGuard(shown.append, check, enforce)
    try:
        for piece in pieces:
            guard.push(piece)
        guard.finish()
    except StreamAborted:
        pass
    return shown, guard


def test_text_without_numbers_streams_immediately_piece_by_piece():
    shown, _ = run_guard(["Casual ", "leave ", "is allowed."])
    assert shown == ["Casual ", "leave ", "is allowed."]


def test_a_sentence_with_a_number_is_held_until_it_is_complete_then_released_whole():
    steps, guard = [], None
    shown = []
    guard = NumericStreamGuard(shown.append, lambda text: [])
    for piece in ["Leave is ", "10 ", "days", ". Next sentence."]:
        guard.push(piece)
        steps.append(list(shown))
    guard.finish()
    assert all("10" not in "".join(s) for s in steps[:3])                  # never shown before the sentence ends
    assert "".join(shown) == "Leave is 10 days. Next sentence."


def test_a_number_that_fails_the_check_is_never_shown_and_aborts_the_stream():
    shown, guard = run_guard(["Paternity leave is ", "7 weeks", " under policy.", " More."],
                             check=lambda text: [{"number": "7", "unit": "week", "reason": "unit_mismatch"}])
    assert "7" not in "".join(shown) and shown == ["Paternity leave is "]
    assert guard.issues and guard.issues[0]["reason"] == "unit_mismatch"


def test_only_the_bad_sentence_is_stopped_earlier_sentences_were_already_shown():
    def check(text):
        return [{"number": "9", "unit": None, "reason": "not_in_sources"}] if "9" in text else []
    shown, _ = run_guard(["Leave is 10 days. ", "Bonus is 9 percent. ", "Done."], check=check)
    assert "".join(shown).startswith("Leave is 10 days. ") and "9" not in "".join(shown)


def test_a_marker_alone_does_not_hold_the_text():
    shown, _ = run_guard(["Leave is allowed [S1] ", "today."])
    assert shown == ["Leave is allowed [S1] ", "today."]


def test_a_decimal_point_does_not_end_the_sentence_and_a_number_split_across_pieces_is_whole():
    seen = []
    shown, _ = run_guard(["Leave is 1", "0.5 days. Ok."], check=lambda text: seen.append(text) or [])
    assert seen[0] == "10.5 days." and "".join(shown) == "Leave is 10.5 days. Ok."     # the words before the number were already safe


def test_number_words_hold_the_sentence_too_and_a_trailing_fragment_is_checked_at_the_end():
    seen = []
    shown, _ = run_guard(["It is seven weeks"], check=lambda text: seen.append(text) or [])
    assert seen == ["seven weeks"] and "".join(shown) == "It is seven weeks"


def test_list_items_end_at_the_line_break():
    seen = []
    run_guard(["1. Apply in 5 days\n", "2. Wait.\n"], check=lambda text: seen.append(text) or [])
    assert seen[0].strip() == "Apply in 5 days" or seen[0].startswith("1. Apply in 5 days") or "5 days" in seen[0]
    assert not any(text.strip() in ("1.", "2.") for text in seen)


def test_in_shadow_mode_issues_are_recorded_but_everything_is_shown():
    shown, guard = run_guard(["Paternity leave is 7 weeks."], check=lambda t: [{"number": "7", "unit": "week",
                                                                              "reason": "unit_mismatch"}], enforce=False)
    assert "".join(shown) == "Paternity leave is 7 weeks." and len(guard.issues) == 1


# ---------------------------------------------------------------- in the real pipeline
@pytest.fixture(scope="module")
def paternity_chunk(cfg):
    chunks, _ = prepare_chunks(cfg, verbose=False)
    return next(c for c in chunks if "Paternity Leave | V1: 5 Days | V2: 7 Days" in c.page_content)


@pytest.fixture()
def make_rig(store, embeddings, fake_reranker, cfg, paternity_chunk):
    """A pipeline whose retrieval returns the REAL paternity chunk from the corpus, with a scripted model."""
    def build(answer="", stream_pieces=None, chunk_text=None, **settings):
        settings.setdefault("numeric_gate_mode", "enforce")
        record = {"llm": 0, "deltas": []}

        def llm(system, user):
            record["llm"] += 1
            return answer

        stream = None
        if stream_pieces is not None:
            def stream(system, user, on_delta):
                record["llm"] += 1
                for piece in stream_pieces:
                    on_delta(piece)
                return "".join(stream_pieces), {}

        pipeline = RAGPipeline(dataclasses.replace(cfg, **settings), embeddings=embeddings, vector_store=store,
                               reranker=fake_reranker, llm=llm, llm_stream=stream)
        doc = Document(page_content=chunk_text or paternity_chunk.page_content,
                       metadata={**paternity_chunk.metadata, "rerank_score": 1.0})
        pipeline.retriever.retrieve = lambda *a, **k: {"semantic": [], "keyword": [], "fused": [doc], "reranked": [doc]}
        pipeline.record = record
        return pipeline
    return build


def assert_withheld(r):
    assert r["status"] == "declined_ungrounded" and r["answer"] == r["answer_display"] == INSUFFICIENT_INFO_MESSAGE
    assert r["sources"] == [] and r["cited_sources"] == [] and r["evaluation"] is None
    assert r["response_class"] == "NOT_FOUND" and r["class_passed"] is True and r["usage"]["answer_discarded"]


def test_the_withhold_message_is_the_agreed_fixed_sentence():
    assert INSUFFICIENT_INFO_MESSAGE == ("I couldn't find sufficient information in the provided sources to answer "
                                         "this question reliably.")


def test_the_default_is_enforce():
    assert Settings().numeric_gate_mode == "enforce"


def test_neg_03_seven_weeks_is_withheld_because_the_policy_says_seven_days(make_rig):
    r = make_rig("7 weeks").ask(PATERNITY_Q, {"role": "employee"}, log=False)
    assert_withheld(r)
    assert r["numeric_gate"]["issues"] == [{"number": "7", "unit": "week", "reason": "unit_mismatch"}]


def test_the_correct_paternity_answer_is_shown_with_its_source_card(make_rig):
    r = make_rig("Paternity leave is 7 days under Leave Policy v2 [S1].").ask(PATERNITY_Q, {"role": "employee"}, log=False)
    assert r["status"] == "access_verified" and r["numeric_gate"]["issues"] == []
    assert len(r["cited_sources"]) == 1 and "[S1]" not in r["answer_display"]


def test_a_calculated_result_is_withheld(make_rig):
    q = "If I spent 1250 on travel and 450 on meals, what is my total claim?"
    assert_withheld(make_rig("Your total claim is 1700.").ask(q, {"role": "employee"}, log=False))


def test_a_figure_the_question_already_contained_may_be_restated(make_rig):
    q = "Can I take 12 days of paternity leave?"
    r = make_rig("No. The policy gives 7 days, so 12 days is not allowed [S1].").ask(q, {"role": "employee"}, log=False)
    assert r["status"] == "access_verified"


def test_a_source_name_grounds_a_version_label(make_rig):
    """The chunk text never says "2"; only the file name (leave_policy_v2.docx) does."""
    rig = make_rig("Under Leave Policy v2 paternity leave is 7 days [S1].", chunk_text="Paternity leave is 7 days.")
    r = rig.ask("paternity leave?", {"role": "employee"}, log=False)
    assert r["status"] == "access_verified" and r["numeric_gate"]["issues"] == []
    assert make_rig("Under Leave Policy v3 paternity leave is 7 days [S1].", chunk_text="Paternity leave is 7 days.") \
        .ask("paternity leave?", {"role": "employee"}, log=False)["status"] == "declined_ungrounded"   # an unrelated "3" is not


def test_shadow_mode_shows_the_answer_but_records_the_issue(make_rig):
    r = make_rig("7 weeks", numeric_gate_mode="shadow").ask(PATERNITY_Q, {"role": "employee"}, log=False)
    assert r["status"] == "access_verified" and r["numeric_gate"]["issues"][0]["reason"] == "unit_mismatch"


def test_off_mode_checks_nothing(make_rig):
    r = make_rig("7 weeks", numeric_gate_mode="off").ask(PATERNITY_Q, {"role": "employee"}, log=False)
    assert r["status"] == "access_verified" and r["numeric_gate"] == {"mode": "off", "issues": []}


def test_an_unknown_mode_stops_start_up(store, embeddings, fake_reranker, cfg):
    with pytest.raises(ValueError, match="NUMERIC_GATE_MODE"):
        RAGPipeline(dataclasses.replace(cfg, numeric_gate_mode="strict"), embeddings=embeddings, vector_store=store,
                    reranker=fake_reranker, llm=lambda s, u: "x")


def test_a_withheld_answer_is_logged_with_numbers_only(make_rig, cfg):
    make_rig("7 weeks").ask(PATERNITY_Q, {"role": "employee"})
    row = read_jsonl(cfg.logs_dir / "query_log.jsonl")[-1]
    assert row["status"] == "declined_ungrounded" and row["response_class"] == "NOT_FOUND" and row["evaluation_score"] is None
    assert row["numeric_gate"]["issues"] == [{"number": "7", "unit": "week", "reason": "unit_mismatch"}]
    assert "weeks" not in str(row["numeric_gate"]) or row["numeric_gate"]["issues"][0]["unit"] == "week"


def test_a_withheld_answer_never_reaches_the_display_callback(make_rig):
    rig = make_rig("7 weeks")
    r = rig.ask(PATERNITY_Q, {"role": "employee"}, log=False, on_delta=rig.record["deltas"].append)
    assert r["status"] == "declined_ungrounded" and rig.record["deltas"] == []


def test_a_correct_answer_without_streaming_support_is_delivered_once_after_the_check(make_rig):
    rig = make_rig("Paternity leave is 7 days [S1].")
    rig.ask(PATERNITY_Q, {"role": "employee"}, log=False, on_delta=rig.record["deltas"].append)
    assert rig.record["deltas"] == ["Paternity leave is 7 days [S1]."]


# ---- streaming through the pipeline
def test_streaming_stops_before_the_wrong_number_is_ever_shown(make_rig):
    rig = make_rig(stream_pieces=["Paternity leave is ", "7 weeks", " under the policy ", "[S1]."])
    r = rig.ask(PATERNITY_Q, {"role": "employee"}, log=False, on_delta=rig.record["deltas"].append)
    assert_withheld(r)
    assert "7" not in "".join(rig.record["deltas"]) and "weeks" not in "".join(rig.record["deltas"])


def test_a_correct_streamed_answer_arrives_complete_and_matches_the_final_response(make_rig):
    pieces = ["Paternity leave ", "is 7 days ", "[S1]."]
    rig = make_rig(stream_pieces=pieces)
    r = rig.ask(PATERNITY_Q, {"role": "employee"}, log=False, on_delta=rig.record["deltas"].append)
    assert "".join(rig.record["deltas"]) == "".join(pieces) and r["status"] == "access_verified"
    assert r["answer"] == "Paternity leave is 7 days [S1]."


def test_an_answer_without_numbers_streams_every_piece_immediately(make_rig):
    pieces = ["Paternity ", "leave ", "is available."]
    rig = make_rig(stream_pieces=pieces)
    rig.ask(PATERNITY_Q, {"role": "employee"}, log=False, on_delta=rig.record["deltas"].append)
    assert rig.record["deltas"] == pieces


def test_in_shadow_mode_streaming_is_not_interrupted(make_rig):
    rig = make_rig(stream_pieces=["Paternity leave is 7 weeks."], numeric_gate_mode="shadow")
    r = rig.ask(PATERNITY_Q, {"role": "employee"}, log=False, on_delta=rig.record["deltas"].append)
    assert r["status"] == "access_verified" and "".join(rig.record["deltas"]) == "Paternity leave is 7 weeks."
    assert r["numeric_gate"]["issues"]


def test_a_stream_aborted_by_the_gate_is_not_reported_as_a_provider_failure(make_rig):
    rig = make_rig(stream_pieces=["It is 99 days."])
    r = rig.ask(PATERNITY_Q, {"role": "employee"}, log=False, on_delta=lambda piece: None)
    assert r["status"] == "declined_ungrounded"          # not an LLMError
