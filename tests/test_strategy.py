"""Tier 6: tasking-strategy plumbing.

Covers the new metrics (inter_compactions, intra_compactions,
last_compact_at_turn) and the default hard-task decomposition prompt clause.
"""
from __future__ import annotations

import la
from conftest import (
    sys_msg, user_msg, assistant_msg,
    make_exchange, make_group,
    fake_response, FakeClient,
)


# ---------- decomposition clause ----------

def test_decomposition_clause_on_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCAL_AGENT_DECOMPOSE", raising=False)
    msgs = la.initial_messages(tmp_path)
    sys_text = msgs[0]["content"]
    assert "Task decomposition" in sys_text
    assert "checkpoint" in sys_text.lower()
    assert "roughly ten tool calls" in sys_text
    assert "Working directory:" in sys_text


def test_decomposition_clause_not_duplicated_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_AGENT_DECOMPOSE", "1")
    msgs = la.initial_messages(tmp_path)
    sys_text = msgs[0]["content"]
    assert sys_text.count("## Task decomposition") == 1


# ---------- inter/intra counters in run_loop ----------

def _force_compaction(monkeypatch):
    """Make `last_pt > threshold` trivially true so run_loop calls compact_messages."""
    pass  # we rely on stats={"compact_threshold": 0, "last_prompt_tokens": 1} below


def test_run_loop_records_inter_compaction(monkeypatch, stub_summarizer, tmp_path):
    # Force the old small keep window so this test still exercises inter-group
    # compaction even though production defaults now preserve more context.
    monkeypatch.setattr(la, "COMPACT_KEEP_LAST_GROUPS", 3)
    # Build a transcript with 6 user-bounded groups to force inter-group compaction.
    msgs = [sys_msg()]
    for i in range(6):
        msgs.extend(make_group(f"task-{i}", n_exchanges=1))
    # add a fresh user turn so run_loop has something to respond to
    msgs.append(user_msg("now do another thing"))
    client = FakeClient([fake_response(content="done")])
    stats = {"compact_threshold": 10, "last_prompt_tokens": 100}
    code, _ = la.run_loop(
        client, msgs, cwd=tmp_path, max_turns=2, verbose=False,
        model="qwen", temperature=0.6, top_p=0.95,
        thinking=True, show_thinking=False, stats=stats,
        bg_base_url="http://bg",
    )
    assert code == 0
    assert stats.get("inter_compactions", 0) >= 1
    assert stats.get("last_compact_at_turn") == 1


def test_run_loop_records_intra_compaction(monkeypatch, stub_summarizer, tmp_path):
    # Force the old small keep window so this test still exercises intra-group
    # compaction even though production defaults now preserve more context.
    monkeypatch.setattr(la, "COMPACT_KEEP_LAST_EXCHANGES", 4)
    # Single user-bounded group with many exchanges → forces intra-group.
    msgs = [sys_msg(), user_msg("big task")]
    for i in range(10):
        msgs.extend(make_exchange(f"c{i}"))
    client = FakeClient([fake_response(content="finished")])
    stats = {"compact_threshold": 10, "last_prompt_tokens": 100}
    code, _ = la.run_loop(
        client, msgs, cwd=tmp_path, max_turns=2, verbose=False,
        model="qwen", temperature=0.6, top_p=0.95,
        thinking=True, show_thinking=False, stats=stats,
        bg_base_url="http://bg",
    )
    assert code == 0
    assert stats.get("intra_compactions", 0) >= 1
    assert stats.get("last_compact_at_turn") == 1


def test_run_loop_no_compaction_below_threshold(stub_summarizer, tmp_path):
    msgs = [sys_msg(), user_msg("hi")]
    client = FakeClient([fake_response(content="ok")])
    stats = {"compact_threshold": 999_999, "last_prompt_tokens": 100}
    code, _ = la.run_loop(
        client, msgs, cwd=tmp_path, max_turns=2, verbose=False,
        model="qwen", temperature=0.6, top_p=0.95,
        thinking=True, show_thinking=False, stats=stats,
        bg_base_url="http://bg",
    )
    assert code == 0
    assert stats.get("inter_compactions", 0) == 0
    assert stats.get("intra_compactions", 0) == 0
    assert "last_compact_at_turn" not in stats
