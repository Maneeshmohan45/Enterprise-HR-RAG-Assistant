"""V1 streaming: what may appear on screen while an answer is generated, and the provider streaming calls."""
from types import SimpleNamespace

import pytest

from hr_rag import generation
from hr_rag.config import Settings
from hr_rag.generation import LLMError, stream_answer_with_usage, strip_citations
from hr_rag.streaming import ThinkFilter, visible_text

SAMPLES = [
    "Casual leave is 10 days [S2].",
    "Employees get 20 days [S1] of annual leave [S1][S3], plus holidays [S2], [S4].",
    "Step 1: apply in the portal [S1].\nStep 2: wait for approval [S2].",
    "No markers at all in this answer.",
    "Mixed forms [S1, S2] and [S1-S3] stay as the final display shows them.",
]


# ---------------------------------------------------------------- visible_text
@pytest.mark.parametrize("final", SAMPLES)
def test_at_no_point_does_the_screen_show_a_marker_or_half_marker(final):
    for end in range(1, len(final) + 1):
        shown = visible_text(final[:end])
        if "[S1, S2]" in final or "[S1-S3]" in final:
            continue                      # forms the final display also leaves alone (not part of V1)
        assert "[S" not in shown and not shown.endswith("["), (final[:end], shown)


@pytest.mark.parametrize("final", SAMPLES)
def test_the_last_streamed_text_equals_the_final_display_text(final):
    assert visible_text(final) == strip_citations(final)


def test_a_half_written_marker_is_held_back_until_it_completes():
    assert visible_text("Leave is 10 days [S") == "Leave is 10 days"
    assert visible_text("Leave is 10 days [S1") == "Leave is 10 days"
    assert visible_text("Leave is 10 days [S1]") == "Leave is 10 days"
    assert visible_text("Leave is 10 days [S1], [S") == "Leave is 10 days,"
    assert visible_text("Use the [brackets] sign") == "Use the [brackets] sign"


def test_streamed_text_grows_only_by_appending_words():
    final = "Casual leave is 10 days [S2] and sick leave is 8 days [S3]."
    previous = ""
    for end in range(1, len(final) + 1):
        shown = visible_text(final[:end])
        assert shown.startswith(previous.rstrip(".,")) or previous.startswith(shown), (previous, shown)
        previous = shown


# ---------------------------------------------------------------- ThinkFilter
def test_think_text_is_removed_even_when_the_tags_are_split_across_pieces():
    f = ThinkFilter()
    pieces = ["<th", "ink>secret rea", "soning</thi", "nk>The answer ", "is 10 days."]
    assert "".join(f.feed(p) for p in pieces) + f.flush() == "The answer is 10 days."


def test_text_without_think_tags_passes_through_unchanged_including_angle_brackets():
    f = ThinkFilter()
    assert "".join(f.feed(p) for p in ["a < b ", "and <b>bold</b> ", "text"]) + f.flush() == "a < b and <b>bold</b> text"


def test_an_unclosed_think_block_never_leaks_and_a_dangling_partial_tag_is_released():
    f = ThinkFilter()
    assert f.feed("<think>still thinking") + f.flush() == ""
    g = ThinkFilter()
    assert g.feed("price <") + g.flush() == "price <"


# ---------------------------------------------------------------- provider streaming (fake clients)
def chunk(content=None, reasoning=None, finish=None, usage=None, no_choices=False):
    if no_choices:
        return SimpleNamespace(choices=[], usage=usage)
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


class FakeOpenAI:
    """Stands in for openai.OpenAI; ``script`` holds the chunks, ``fail_first`` simulates an HTTP 400."""
    script, fail_first, calls = [], False, []

    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeOpenAI.calls.append(kwargs)
        if FakeOpenAI.fail_first and "stream_options" in kwargs:
            error = Exception("stream_options not supported")
            error.status_code = 400
            raise error
        return iter(FakeOpenAI.script)


@pytest.fixture()
def nvidia(monkeypatch):
    import openai
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key-not-real")
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    FakeOpenAI.script, FakeOpenAI.fail_first, FakeOpenAI.calls = [], False, []
    return Settings(llm_provider="nvidia")


def run(cfg):
    pieces = []
    text, usage = stream_answer_with_usage("system", "user", cfg, pieces.append)
    return text, usage, pieces


def test_nvidia_pieces_are_delivered_in_order_with_usage_and_final_text(nvidia):
    FakeOpenAI.script = [chunk("Casual "), chunk("leave "), chunk("is 10 days [S1].", finish="stop"),
                         chunk(no_choices=True, usage=SimpleNamespace(prompt_tokens=120, completion_tokens=9))]
    text, usage, pieces = run(nvidia)
    assert pieces == ["Casual ", "leave ", "is 10 days [S1]."] and text == "Casual leave is 10 days [S1]."
    assert usage == {"input_tokens": 120, "output_tokens": 9}
    assert FakeOpenAI.calls[0]["stream"] is True and FakeOpenAI.calls[0]["stream_options"] == {"include_usage": True}


def test_nvidia_reasoning_content_is_never_displayed(nvidia):
    FakeOpenAI.script = [chunk(reasoning="let me think about leave"), chunk("Ten days.", finish="stop")]
    text, _, pieces = run(nvidia)
    assert pieces == ["Ten days."] and text == "Ten days."


def test_nvidia_think_tags_inside_the_content_are_filtered_from_the_stream(nvidia):
    FakeOpenAI.script = [chunk("<think>hidden</think>"), chunk("Ten days.", finish="stop")]
    text, _, pieces = run(nvidia)
    assert "".join(pieces) == "Ten days." and text == "Ten days."


def test_nvidia_retries_once_without_stream_options_on_a_400(nvidia):
    FakeOpenAI.fail_first = True
    FakeOpenAI.script = [chunk("Ten days.", finish="stop")]
    text, usage, _ = run(nvidia)
    assert text == "Ten days." and usage == {"input_tokens": None, "output_tokens": None}
    assert len(FakeOpenAI.calls) == 2 and "stream_options" not in FakeOpenAI.calls[1]


@pytest.mark.parametrize("script", [[chunk("Cut off", finish="length")], [chunk(None, finish="stop")], []])
def test_nvidia_a_cut_off_or_empty_stream_is_an_error_not_a_blank_answer(nvidia, script):
    FakeOpenAI.script = script
    with pytest.raises(LLMError):
        run(nvidia)


def test_nvidia_other_failures_are_reported_as_llm_errors(nvidia, monkeypatch):
    class Broken(FakeOpenAI):
        def _create(self, **kwargs):
            raise ConnectionError("network down")
    import openai
    monkeypatch.setattr(openai, "OpenAI", Broken)
    with pytest.raises(LLMError, match="network down"):
        run(nvidia)


def test_nvidia_a_reasoning_notes_leak_is_still_refused_after_streaming(nvidia):
    FakeOpenAI.script = [chunk("Here's a thinking process: the user asks about leave", finish="stop")]
    with pytest.raises(LLMError, match="reasoning"):
        run(nvidia)


class FakeClaudeStream:
    def __init__(self, pieces, stop_reason="end_turn"):
        self.text_stream, self._stop = iter(pieces), stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop, usage=SimpleNamespace(input_tokens=80, output_tokens=7))


@pytest.fixture()
def claude(monkeypatch):
    import anthropic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    state = {"pieces": ["Ten ", "days."], "stop": "end_turn"}
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(
        stream=lambda **kwargs: FakeClaudeStream(state["pieces"], state["stop"]))))
    return Settings(llm_provider="claude"), state


def test_claude_pieces_usage_and_text(claude):
    cfg, _ = claude
    text, usage, pieces = run(cfg)
    assert pieces == ["Ten ", "days."] and text == "Ten days." and usage == {"input_tokens": 80, "output_tokens": 7}


def test_claude_a_cut_off_or_empty_answer_is_an_error(claude):
    cfg, state = claude
    state["stop"] = "max_tokens"
    with pytest.raises(LLMError, match="cut off"):
        run(cfg)
    state.update(stop="end_turn", pieces=[" "])
    with pytest.raises(LLMError, match="no text"):
        run(cfg)


def test_a_stream_abort_raised_by_the_display_callback_passes_through_nvidia_unchanged(nvidia):
    from hr_rag.generation import StreamAborted
    FakeOpenAI.script = [chunk("It is "), chunk("7 weeks.", finish="stop")]

    def stop(piece):
        raise StreamAborted([{"number": "7", "unit": "week", "reason": "unit_mismatch"}])

    with pytest.raises(StreamAborted) as raised:                    # not wrapped as "nvidia request failed"
        stream_answer_with_usage("system", "user", nvidia, stop)
    assert raised.value.issues[0]["number"] == "7"


def test_a_stream_abort_raised_by_the_display_callback_passes_through_claude_unchanged(claude):
    from hr_rag.generation import StreamAborted
    cfg, _ = claude

    def stop(piece):
        raise StreamAborted([])

    with pytest.raises(StreamAborted):
        stream_answer_with_usage("system", "user", cfg, stop)


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        stream_answer_with_usage("s", "u", Settings(llm_provider="other"), lambda p: None)


def test_every_provider_with_a_normal_call_also_has_a_streaming_call():
    assert set(generation._PROVIDERS) == set(generation._STREAM_PROVIDERS)
