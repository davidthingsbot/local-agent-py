"""Empty-response watchdog tests.

Qwen can occasionally return an assistant message with no content and no tool
calls, especially after compaction. The harness should not treat that as a
valid final answer or persist the empty assistant turn as useful context.
"""
from __future__ import annotations

import la
from conftest import sys_msg, user_msg, fake_response, FakeClient


def test_empty_response_without_reasoning_retries_and_recovers(tmp_path):
    client = FakeClient([
        fake_response(content=""),
        fake_response(content="recovered"),
    ])
    stats: dict = {}
    messages = [sys_msg(), user_msg("do the task")]

    code, final = la.run_loop(
        client,
        messages,
        cwd=tmp_path,
        max_turns=4,
        verbose=False,
        model="x",
        temperature=0.6,
        top_p=0.95,
        thinking=True,
        show_thinking=False,
        stats=stats,
        bg_base_url="http://bg",
    )

    assert code == 0
    assert final == "recovered"
    assert stats.get("empty_retries") == 1
    assert len(client.calls) == 2
    # Retry should disable thinking and cool sampling down.
    assert client.calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert client.calls[1]["temperature"] <= 0.2
    # The empty assistant turn itself should not be persisted.
    assert not any(m.get("role") == "assistant" and not (m.get("content") or "") for m in messages[:-1])
    assert messages[-1] == {"role": "assistant", "content": "recovered"}


def test_empty_response_after_compaction_gets_compaction_specific_nudge(monkeypatch, tmp_path):
    def fake_compact(messages, *args, **kwargs):
        return messages, True, "intra-group: compacted fake exchange"

    monkeypatch.setattr(la, "compact_messages", fake_compact)
    client = FakeClient([
        fake_response(content=""),
        fake_response(content="done"),
    ])
    stats = {"last_prompt_tokens": 999, "compact_threshold": 1}
    messages = [sys_msg(), user_msg("continue after compact")]

    code, final = la.run_loop(
        client,
        messages,
        cwd=tmp_path,
        max_turns=4,
        verbose=False,
        model="x",
        temperature=0.6,
        top_p=0.95,
        thinking=True,
        show_thinking=False,
        stats=stats,
        bg_base_url="http://bg",
    )

    assert code == 0
    assert final == "done"
    nudges = [m["content"] for m in messages if m.get("role") == "user"]
    assert any("just compacted" in n for n in nudges)
    assert stats.get("empty_retries") == 1
