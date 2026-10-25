"""One parser for citation markers: display, validation, evaluation and source cards must agree.

Real answers use [S3, S4] and [S1-S3]. Before the shared parser these leaked into the chat, and (once the
"top 3 sources" fallback was removed) their source cards disappeared.
"""
import dataclasses

import pytest

from hr_rag import citations
from hr_rag.evaluation import extract_numbers
from hr_rag.generation import strip_citations
from hr_rag.guardrails import check_citations
from hr_rag.pipeline import RAGPipeline
from hr_rag.streaming import visible_text

FORMS = [
    ("Casual leave is 10 days [S1].", ["S1"], "Casual leave is 10 days."),
    ("Fraud [S3, S4] and absence [S4, S5].", ["S3", "S4", "S5"], "Fraud and absence."),
    ("Rules apply [S1,S2].", ["S1", "S2"], "Rules apply."),
    ("Rules apply [S1; S2].", ["S1", "S2"], "Rules apply."),
    ("See [S1][S2] and [S2],[S3].", ["S1", "S2", "S3"], "See and."),
    ("All of it [S1-S3].", ["S1", "S2", "S3"], "All of it."),
    ("Range [S2\u2013S6] here.", ["S2", "S3", "S4", "S5", "S6"], "Range here."),
    ("Spaced [ S1 , S2 ] ok.", ["S1", "S2"], "Spaced ok."),
    ("Lower case [s1] works.", ["S1"], "Lower case works."),
    ("No markers at all.", [], "No markers at all."),
]


@pytest.mark.parametrize("answer, ids, shown", FORMS)
def test_every_marker_form_is_found_and_hidden(answer, ids, shown):
    assert citations.cited_ids(answer) == ids and strip_citations(answer) == shown
    assert check_citations(answer, 9)[2] == ids


def test_ranges_beyond_the_source_count_are_reported_invalid():
    valid, invalid, used = check_citations("Everything [S1-S9].", 6)
    assert not valid and invalid == ["S7", "S8", "S9"] and used[0] == "S1"


def test_a_marker_that_is_not_a_marker_is_left_alone():
    for text in ["See [Section 3] and [S] and [1] and [SKIP]", "Use [brackets] and [S1x]"]:
        assert citations.cited_ids(text) == [] and strip_citations(text) == text


def test_the_digits_inside_markers_are_not_facts():
    assert extract_numbers("Casual leave is 10 days [S3, S4] and [S1-S6].") == {"10"}


@pytest.mark.parametrize("answer", [f for f, _, _ in FORMS])
def test_the_screen_never_shows_a_marker_or_half_marker_while_streaming(answer):
    for end in range(1, len(answer) + 1):
        shown = visible_text(answer[:end])
        assert "[S" not in shown.replace("[Section", "") and not shown.rstrip().endswith("["), (answer[:end], shown)
    assert visible_text(answer) == strip_citations(answer)


def test_a_half_written_comma_or_range_marker_is_held_back():
    assert visible_text("Leave is 10 days [S1, S") == "Leave is 10 days"
    assert visible_text("Leave is 10 days [S1-") == "Leave is 10 days"
    assert visible_text("Leave is 10 days [S1,") == "Leave is 10 days"


@pytest.fixture()
def make_rig(store, embeddings, fake_reranker, cfg):
    def build(answer):
        return RAGPipeline(dataclasses.replace(cfg, numeric_gate_mode="off"), embeddings=embeddings,
                           vector_store=store, reranker=fake_reranker, llm=lambda s, u: answer)
    return build


def test_an_answer_citing_with_a_comma_list_gets_its_cards_and_a_clean_display(make_rig):
    r = make_rig("Casual leave is 10 days [S1, S2].").ask("How many casual leave days do I get?",
                                                         {"role": "employee"}, log=False)
    assert r["cited_ids"] == ["S1", "S2"] and r["citations_valid"] and "[S" not in r["answer_display"]
    assert r["cited_sources"] and all(s in r["sources"] for s in r["cited_sources"])


def test_an_invalid_range_is_reported_and_its_unknown_sources_get_no_cards(make_rig):
    r = make_rig("Casual leave is 10 days [S1-S50].").ask("How many casual leave days do I get?",
                                                        {"role": "employee"}, log=False)
    assert r["citations_valid"] is False and r["invalid_citation_ids"] and len(r["cited_sources"]) <= len(r["sources"])
