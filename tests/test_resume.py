"""Tier 4: REPL resume behavior.

The REPL has an input() loop, so we don't drive it end-to-end. Instead we test
the building blocks the resume path uses, and the threshold-vs-est interplay
that decides whether to pre-compact.
"""
from __future__ import annotations

import json
from pathlib import Path

import la
from conftest import sys_msg, user_msg, assistant_msg, make_exchange, make_group


# ---------- transcript I/O ----------

def test_save_then_load_roundtrip(tmp_path: Path):
    p = tmp_path / "transcript-X.json"
    msgs = [sys_msg(), user_msg("hi"), assistant_msg("ok")]
    la.save_transcript(p, tmp_path, msgs)
    loaded = la.load_transcript(p)
    assert loaded == msgs


def test_load_transcript_accepts_legacy_list_format(tmp_path: Path):
    p = tmp_path / "legacy.json"
    msgs = [sys_msg(), user_msg("hi")]
    p.write_text(json.dumps(msgs), encoding="utf-8")
    loaded = la.load_transcript(p)
    assert loaded == msgs


def test_latest_transcript_path_returns_newest(tmp_path: Path):
    d = la.transcripts_dir(tmp_path)
    a = d / "transcript-2025-01-01_00-00-00.json"
    b = d / "transcript-2026-01-01_00-00-00.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    assert la.latest_transcript_path(tmp_path) == b


# ---------- reasoning_content stripping on resume ----------

def test_resume_strips_reasoning_content(tmp_path: Path):
    msgs = [
        sys_msg(),
        user_msg("u"),
        assistant_msg("a", reasoning="HUGE THINKING BLOB " * 100),
    ]
    n = la.strip_reasoning_content(msgs)
    assert n == 1
    assert "reasoning_content" not in msgs[2]


# ---------- pre-compact decision logic ----------

def _heavy_transcript(group_count: int, exchanges_per_group: int = 1) -> list[dict]:
    msgs = [sys_msg("S " * 50)]
    for i in range(group_count):
        msgs.extend(make_group(f"task-{i}", n_exchanges=exchanges_per_group))
        # fatten content so estimate_prompt_tokens crosses a threshold
        msgs.append(assistant_msg("FILLER " * 200))
    return msgs


def test_estimate_above_threshold_triggers_compact(stub_summarizer, tmp_path: Path):
    """Simulate the REPL's resume decision: estimate > threshold → compact_messages
    runs and the message count drops."""
    msgs = _heavy_transcript(group_count=10, exchanges_per_group=2)
    est = la.estimate_prompt_tokens(msgs)
    threshold = est // 2  # force above-threshold
    assert est > threshold
    new, did, _ = la.compact_messages(msgs, "http://bg", "qwen",
                                      keep_last_groups=3, keep_last_exchanges=4)
    assert did
    assert len(new) < len(msgs)
    # invariant: still has system + user-bounded groups
    head, groups = la.split_groups(new)
    assert head and head[0]["role"] == "system"
    assert all(g[0]["role"] == "user" for g in groups)


def test_estimate_below_threshold_skips_compact(stub_summarizer):
    msgs = [sys_msg(), user_msg("hi"), assistant_msg("ok")]
    est = la.estimate_prompt_tokens(msgs)
    threshold = est * 100  # comfortably under
    # The REPL pre-compact branch only runs `if est_pt > threshold`. If we still
    # call compact_messages directly (e.g. /compact) on a tiny transcript, it
    # should report nothing-to-compact rather than mangle the transcript.
    new, did, note = la.compact_messages(msgs, "http://bg", "qwen", 3, 4)
    assert not did
    assert "nothing to compact" in note or "only" in note
    _ = threshold  # threshold not used directly here, but documents the case


# ---------- full resume pipeline simulated ----------

def test_full_resume_pipeline(tmp_path: Path, stub_summarizer):
    """Simulate REPL resume end-to-end (without input loop):
    1. write a heavy transcript with reasoning_content blobs
    2. load it
    3. strip reasoning
    4. estimate prompt tokens
    5. pre-compact if > threshold
    6. assert structure is sensible and tool_call/tool pairing intact"""
    msgs: list[dict] = [sys_msg("S")]
    for i in range(8):
        msgs.append(user_msg(f"task-{i}"))
        # an exchange with a tool call + response
        msgs.extend(make_exchange(f"tc-{i}-1"))
        # a thinking-only-style assistant turn we want stripped
        msgs.append(assistant_msg(f"final-{i}", reasoning="THOUGHTS " * 50))
    p = tmp_path / "transcript-heavy.json"
    la.save_transcript(p, tmp_path, msgs)

    loaded = la.load_transcript(p)
    stripped = la.strip_reasoning_content(loaded)
    assert stripped == 8

    est = la.estimate_prompt_tokens(loaded)
    threshold = est // 4  # forced above-threshold
    if est > threshold:
        new, did, _ = la.compact_messages(loaded, "http://bg", "qwen", 3, 4)
        assert did
        head, groups = la.split_groups(new)
        # last 3 groups preserved
        assert [g[0]["content"] for g in groups[-3:]] == ["task-5", "task-6", "task-7"]
        # tool call pairing preserved
        for i, m in enumerate(new):
            if m.get("role") == "tool":
                # walk back to find an assistant tool_call within the same group
                ok = False
                for j in range(i - 1, -1, -1):
                    prev = new[j]
                    if prev.get("role") == "user":
                        break
                    if prev.get("role") == "assistant":
                        ids = [tc.get("id") for tc in (prev.get("tool_calls") or [])]
                        if m.get("tool_call_id") in ids:
                            ok = True
                        break
                assert ok, f"tool message {i} dangling"
