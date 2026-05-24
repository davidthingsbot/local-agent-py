"""Tier 3: empty-response watchdog.

When Qwen3 emits no content and no tool_calls, run_loop must:
- detect that as a non-answer
- append a nudge user message, but not persist the empty assistant message
- retry up to three times with thinking=False and cooler sampling
- bump stats["empty_retries"]

If all retries are also empty, exit with code 3 and a clear message.
Genuinely-empty responses without reasoning are retried too; this matters after compaction.
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


def test_thinking_only_then_empty_retries_exhausted_returns_code_3(tmp_path):
    msgs = [sys_msg(), user_msg("do something")]
    scripted = [
        fake_response(content="", reasoning="thoughts"),
        fake_response(content=""),
        fake_response(content=""),
        fake_response(content=""),
    ]
    code, final, stats, _ = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 3
    assert "empty response" in final
    assert stats["empty_retries"] == 3


def test_genuinely_empty_response_without_reasoning_is_retried(tmp_path):
    """Empty responses without reasoning happen after compaction; retry them too."""
    msgs = [sys_msg(), user_msg("hi")]
    scripted = [fake_response(content="", reasoning=None), fake_response(content="ok")]
    code, final, stats, client = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 0
    assert final == "ok"
    assert stats.get("empty_retries", 0) == 1
    assert len(client.calls) == 2


def test_watchdog_appends_nudge_in_correct_order(tmp_path):
    msgs = [sys_msg(), user_msg("question")]
    scripted = [
        fake_response(content="", reasoning="think"),
        fake_response(content="final"),
    ]
    code, final, _, _ = _run(msgs, scripted, tmp_path=tmp_path)
    assert code == 0 and final == "final"
    # transcript shape: system, user(orig), user(nudge), assistant(final)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "user", "assistant"]
    assert msgs[2]["content"] == la.EMPTY_RETRY_NUDGE
    assert msgs[3]["content"] == "final"


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
