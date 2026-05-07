"""Tier 1: compaction correctness.

Guards the invariants we already broke once: tool_call/tool pairing, system
preservation, recursion of summary markers, and the helpers used by REPL
resume.
"""
from __future__ import annotations

import la
from conftest import (
    sys_msg, user_msg, assistant_msg, tool_call, tool_msg,
    make_exchange, make_group,
)


# ---------- split_groups ----------

def test_split_groups_separates_head_and_user_groups():
    msgs = [
        sys_msg(),
        user_msg("g1"),
        assistant_msg("a1"),
        user_msg("g2"),
        assistant_msg("a2"),
    ]
    head, groups = la.split_groups(msgs)
    assert len(head) == 1 and head[0]["role"] == "system"
    assert [g[0]["content"] for g in groups] == ["g1", "g2"]
    assert len(groups[0]) == 2 and len(groups[1]) == 2


def test_split_groups_keeps_tool_pair_inside_group():
    msgs = [
        sys_msg(),
        user_msg("u"),
        *make_exchange("c1"),
        *make_exchange("c2"),
    ]
    _, groups = la.split_groups(msgs)
    assert len(groups) == 1
    roles = [m["role"] for m in groups[0]]
    # user, assistant, tool, assistant, tool
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]


def test_split_groups_handles_no_system():
    msgs = [user_msg("hi"), assistant_msg("ok")]
    head, groups = la.split_groups(msgs)
    assert head == []
    assert len(groups) == 1


# ---------- inter-group compaction ----------

def test_inter_group_keeps_system_and_last_n_groups(stub_summarizer):
    msgs = [sys_msg()]
    for i in range(8):
        msgs.extend(make_group(f"task-{i}", n_exchanges=1))
    new, did, note = la.compact_messages(msgs, "http://bg", "qwen",
                                         keep_last_groups=3, keep_last_exchanges=99)
    assert did, note
    head, groups = la.split_groups(new)
    assert len(head) == 1
    assert [g[0]["content"] for g in groups] == ["task-5", "task-6", "task-7"]
    assert la.COMPACT_MARKER in head[0]["content"]


def test_inter_group_no_op_below_threshold(stub_summarizer):
    msgs = [sys_msg()]
    for i in range(3):
        msgs.extend(make_group(f"task-{i}", n_exchanges=1))
    new, did, _ = la.compact_messages(msgs, "http://bg", "qwen",
                                      keep_last_groups=3, keep_last_exchanges=99)
    assert not did
    assert new is msgs


def test_inter_group_preserves_tool_call_pairing(stub_summarizer):
    msgs = [sys_msg()]
    for i in range(6):
        msgs.extend(make_group(f"task-{i}", n_exchanges=2))
    new, did, _ = la.compact_messages(msgs, "http://bg", "qwen",
                                      keep_last_groups=2, keep_last_exchanges=99)
    assert did
    _assert_tool_pair_invariant(new)


def test_inter_group_extracts_prior_summary_for_recompaction(stub_summarizer):
    msgs = [sys_msg(with_marker_summary="OLD SUMMARY TEXT")]
    for i in range(8):
        msgs.extend(make_group(f"task-{i}", n_exchanges=1))
    new, did, _ = la.compact_messages(msgs, "http://bg", "qwen",
                                      keep_last_groups=3, keep_last_exchanges=99)
    assert did
    new_sys = new[0]["content"]
    # exactly one COMPACT_MARKER (no doubling on re-compaction)
    assert new_sys.count(la.COMPACT_MARKER) == 1
    # prior summary was passed into the summarizer
    assert any(c["prior_summary"] == "OLD SUMMARY TEXT" for c in stub_summarizer)


def test_compact_refuses_when_no_system_message(stub_summarizer):
    msgs = []
    for i in range(8):
        msgs.extend(make_group(f"t{i}", n_exchanges=1))
    new, did, note = la.compact_messages(msgs, "http://bg", "qwen", 3, 4)
    assert not did
    assert "no system message" in note


# ---------- intra-group compaction ----------

def test_intra_group_summarizes_old_exchanges(stub_summarizer):
    # one user-bounded group with 10 exchanges
    msgs = [sys_msg(), user_msg("big")]
    for i in range(10):
        msgs.extend(make_exchange(f"c{i}"))
    new, did, note = la.compact_messages(msgs, "http://bg", "qwen",
                                         keep_last_groups=99, keep_last_exchanges=3)
    assert did, note
    _, groups = la.split_groups(new)
    assert len(groups) == 1
    g = groups[0]
    # user, summary-assistant, then 3 exchanges (assistant+tool each)
    assert g[0]["role"] == "user"
    assert g[1]["role"] == "assistant"
    assert g[1]["content"].startswith(la.INTRA_SUMMARY_MARKER)
    # 3 exchanges = 6 messages after the summary
    assert len(g) == 2 + 3 * 2
    _assert_tool_pair_invariant(new)


def test_intra_group_no_op_when_few_exchanges(stub_summarizer):
    msgs = [sys_msg(), user_msg("u")]
    for i in range(3):
        msgs.extend(make_exchange(f"c{i}"))
    new, did, _ = la.compact_messages(msgs, "http://bg", "qwen",
                                      keep_last_groups=99, keep_last_exchanges=4)
    assert not did
    assert new is msgs


def test_intra_group_recompaction_extracts_prior_summary(stub_summarizer):
    """Calling compact twice on a heavy active group should extract the previous
    intra summary and not duplicate the marker."""
    msgs = [sys_msg(), user_msg("big")]
    for i in range(12):
        msgs.extend(make_exchange(f"c{i}"))
    once, did1, _ = la.compact_messages(msgs, "http://bg", "qwen", 99, 3)
    assert did1
    # add more exchanges and recompact
    for i in range(12, 24):
        once.extend(make_exchange(f"c{i}"))
    twice, did2, _ = la.compact_messages(once, "http://bg", "qwen", 99, 3)
    assert did2
    _, groups = la.split_groups(twice)
    g = groups[-1]
    summary_msgs = [m for m in g if m["role"] == "assistant"
                    and (m.get("content") or "").startswith(la.INTRA_SUMMARY_MARKER)]
    assert len(summary_msgs) == 1, f"expected one intra summary, got {len(summary_msgs)}"
    # the prior summary string was passed into the summarizer on second call
    last_call = stub_summarizer[-1]
    assert last_call["prior_summary"] is not None


# ---------- message_to_dict reasoning_content stripping ----------

class _M:
    """Mimics the SDK's ChatCompletionMessage shape."""
    def __init__(self, content, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


def test_message_to_dict_drops_reasoning_content():
    m = _M(content="answer", reasoning_content="long internal thoughts")
    d = la.message_to_dict(m)
    assert "reasoning_content" not in d
    assert d["content"] == "answer"


def test_message_to_dict_serializes_tool_calls():
    import types as _t
    tc = _t.SimpleNamespace(id="c1", function=_t.SimpleNamespace(name="list_dir",
                                                                  arguments='{"path":"."}'))
    m = _M(content="", tool_calls=[tc])
    d = la.message_to_dict(m)
    assert d["tool_calls"][0]["id"] == "c1"
    assert d["tool_calls"][0]["function"]["name"] == "list_dir"
    assert d["tool_calls"][0]["function"]["arguments"] == '{"path":"."}'


# ---------- estimate_prompt_tokens ----------

def test_estimate_prompt_tokens_counts_content_reasoning_tool_args():
    msgs = [
        {"role": "system", "content": "a" * 90},          # 90 chars
        {"role": "user", "content": "b" * 30},            # 30
        {"role": "assistant",
         "content": "c" * 30,
         "reasoning_content": "d" * 60,                   # 60
         "tool_calls": [{"function": {"name": "list_dir", "arguments": "e" * 30}}]},  # 30 + len(name)=8
    ]
    est = la.estimate_prompt_tokens(msgs)
    # 90 + 30 + 30 + 60 + 30 + 8 = 248 chars; //3 = 82
    assert est == (90 + 30 + 30 + 60 + 30 + len("list_dir")) // 3


def test_strip_reasoning_content_mutates_and_counts():
    msgs = [
        {"role": "assistant", "content": "x", "reasoning_content": "thoughts"},
        {"role": "user", "content": "y"},
        {"role": "assistant", "content": "z", "reasoning_content": "more"},
    ]
    n = la.strip_reasoning_content(msgs)
    assert n == 2
    assert "reasoning_content" not in msgs[0]
    assert "reasoning_content" not in msgs[2]


# ---------- helper ----------

def _assert_tool_pair_invariant(messages: list[dict]) -> None:
    """Every tool message must directly follow an assistant tool_call with the same id."""
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        call_id = m.get("tool_call_id")
        # walk backward to find an assistant whose tool_calls includes this id
        # bounded by the start of the current group
        found = False
        for j in range(i - 1, -1, -1):
            prev = messages[j]
            if prev.get("role") == "user":
                break
            if prev.get("role") == "assistant":
                ids = [tc.get("id") for tc in (prev.get("tool_calls") or [])]
                if call_id in ids:
                    found = True
                break
        assert found, f"tool message {i} call_id={call_id!r} has no preceding assistant tool_call in its group"
