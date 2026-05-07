"""Tier 3: empty-response watchdog.

When Qwen3 thinking mode emits a full <think> block but no content and no
tool_calls, run_loop must:
- detect that pattern (only when reasoning_content is present)
- append the empty assistant message + EMPTY_RETRY_NUDGE user message
- retry once with thinking=False
- bump stats["empty_retries"]

If the retry is also empty, exit with code 3 and a clear message.
A genuinely-empty response (no reasoning) is NOT retried (it would loop).
"""
from __future__ import annotations

import la
from conftest import sys_msg, user_msg, fake_response, FakeClient


def _run(messages, scripted, *, thinking=True, max_turns=4, tmp_path=None):
    client = FakeClient(scripted)
    stats = {"compact_threshold": 999_999}
    code, final = la.run_loop(
        client, messages, cwd=tmp_path, max_turns=max_turns, verbose=False,
        model="qwen", temperature=0.6, top_p=0.95,
        thinking=thinking, show_thinking=False, stats=stats,
        bg_base_url="http://bg",
    )
    return code, final, stats, client


def test_thinking_only_response_triggers_retry_with_thinking_off(tmp_path):
    msgs = [sys_msg(), user_msg("do something")]
    scripted = [
        fake_response(content="", reasoning="long internal reasoning"),  # empty, but has reasoning
        fake_response(content="OK, here is the answer."),                # successful retry
    ]
    code, final, stats, client = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 0
    assert final == "OK, here is the answer."
    assert stats["empty_retries"] == 1
    # The retry was made with thinking=False
    second_call = client.calls[1]
    assert second_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    # First call had thinking=True
    first_call = client.calls[0]
    assert first_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_thinking_only_then_empty_again_returns_code_3(tmp_path):
    msgs = [sys_msg(), user_msg("do something")]
    scripted = [
        fake_response(content="", reasoning="thoughts"),
        fake_response(content=""),  # retry also empty
    ]
    code, final, stats, _ = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 3
    assert "empty response" in final
    assert stats["empty_retries"] == 1


def test_genuinely_empty_response_without_reasoning_is_not_retried(tmp_path):
    """If the model returns content="" with no tool_calls AND no reasoning_content,
    we don't retry — that would loop. We exit 3 immediately."""
    msgs = [sys_msg(), user_msg("hi")]
    scripted = [fake_response(content="", reasoning=None)]
    code, final, stats, client = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 3
    assert stats.get("empty_retries", 0) == 0
    assert len(client.calls) == 1


def test_watchdog_appends_nudge_in_correct_order(tmp_path):
    msgs = [sys_msg(), user_msg("question")]
    scripted = [
        fake_response(content="", reasoning="think"),
        fake_response(content="final"),
    ]
    code, final, _, _ = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 0 and final == "final"
    # transcript shape: system, user(orig), assistant(empty), user(nudge), assistant(final)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == ""           # empty assistant turn preserved
    assert msgs[3]["content"] == la.EMPTY_RETRY_NUDGE
    assert msgs[4]["content"] == "final"


def test_watchdog_does_not_intercept_tool_call(tmp_path):
    """A tool-calling response is not empty — watchdog must stay out of the way."""
    msgs = [sys_msg(), user_msg("list the dir")]
    scripted = [
        fake_response(tool_calls=[("c1", "list_dir", {"path": "."})]),
        fake_response(content="done"),
    ]
    code, final, stats, client = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 0 and final == "done"
    assert stats.get("empty_retries", 0) == 0
    assert len(client.calls) == 2  # tool call -> tool result -> final


def test_watchdog_does_not_intercept_normal_final_answer(tmp_path):
    msgs = [sys_msg(), user_msg("hi")]
    scripted = [fake_response(content="hello there", reasoning="some thoughts")]
    code, final, stats, _ = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 0 and final == "hello there"
    assert stats.get("empty_retries", 0) == 0


def test_watchdog_metric_visible_in_stats(tmp_path):
    """The empty_retries counter must end up in stats so /context can show it."""
    msgs = [sys_msg(), user_msg("u")]
    scripted = [
        fake_response(content="", reasoning="t"),
        fake_response(content="ok"),
    ]
    _, _, stats, _ = _run(msgs, scripted, tmp_path=tmp_path)
    assert "empty_retries" in stats
    assert stats["empty_retries"] == 1
