"""Compaction pinned-blocks: original request + open outputs.

Guards against the failure mode observed in the wild where the BG summarizer
silently dropped the user's actual request, replaced it with a paraphrased
"goal," and falsely declared the task complete — which then misled the agent
into reading more files instead of writing.

The pinned blocks are spliced in by the harness, not by the BG model. The BG
model is explicitly instructed not to produce them.
"""
from __future__ import annotations

import la
from conftest import (
    sys_msg, user_msg, assistant_msg, tool_call, tool_msg,
    make_exchange, make_group, fake_response, FakeClient,
)


# ---------- helpers ----------

def _summary_after_compact(messages):
    """Pull the post-COMPACT_MARKER text out of the system message."""
    sys_text = messages[0]["content"]
    assert la.COMPACT_MARKER in sys_text, sys_text
    _, _, summary = sys_text.partition(la.COMPACT_MARKER)
    return summary


# ---------- pin: original request ----------

def test_first_compaction_pins_original_user_request(stub_summarizer, tmp_path):
    """First inter-group compact must inject ## Original request verbatim."""
    msgs = [sys_msg()]
    msgs.extend(make_group("ORIGINAL ASK: write the deletion plan to file X", n_exchanges=1))
    for i in range(5):
        msgs.extend(make_group(f"continue batch {i}", n_exchanges=1))

    new_msgs, did, _ = la.compact_messages(msgs, "http://bg", "model", keep_last_groups=2)
    assert did
    summary = _summary_after_compact(new_msgs)
    assert la.PINNED_REQUEST_HEADER in summary
    assert "ORIGINAL ASK: write the deletion plan to file X" in summary


def test_bg_summarizer_never_sees_pinned_request_on_subsequent_compactions(stub_summarizer, tmp_path):
    """Round 2+: the BG model gets the residual prior summary with the pinned
    block stripped out, so it cannot paraphrase or drop it."""
    # Round 1
    msgs = [sys_msg()]
    msgs.extend(make_group("ORIGINAL ASK: keep writing to plan.md", n_exchanges=1))
    for i in range(5):
        msgs.extend(make_group(f"batch {i}", n_exchanges=1))
    msgs, did, _ = la.compact_messages(msgs, "http://bg", "model", keep_last_groups=2)
    assert did

    # Round 2: add more groups, compact again.
    for i in range(5, 10):
        msgs.extend(make_group(f"batch {i}", n_exchanges=1))
    stub_summarizer.clear()
    msgs, did, _ = la.compact_messages(msgs, "http://bg", "model", keep_last_groups=2)
    assert did

    # The BG was called with a prior_summary that does NOT contain the pinned header.
    assert len(stub_summarizer) >= 1
    last_prior = stub_summarizer[-1]["prior_summary"] or ""
    assert la.PINNED_REQUEST_HEADER not in last_prior, (
        f"BG model saw the pinned header in prior_summary; it could rewrite it. "
        f"Got: {last_prior!r}"
    )

    # But the final summary still has it.
    summary = _summary_after_compact(msgs)
    assert la.PINNED_REQUEST_HEADER in summary
    assert "ORIGINAL ASK: keep writing to plan.md" in summary


def test_pinned_request_survives_three_compactions(stub_summarizer, tmp_path):
    """Three rounds — original ask must still be there byte-for-byte."""
    original = "Find all AI/Construo references and write line ranges to diary/plan.md"
    msgs = [sys_msg()]
    msgs.extend(make_group(original, n_exchanges=1))
    for i in range(5):
        msgs.extend(make_group(f"r1-batch-{i}", n_exchanges=1))

    msgs, _, _ = la.compact_messages(msgs, "http://bg", "model", keep_last_groups=2)
    for i in range(5):
        msgs.extend(make_group(f"r2-batch-{i}", n_exchanges=1))
    msgs, _, _ = la.compact_messages(msgs, "http://bg", "model", keep_last_groups=2)
    for i in range(5):
        msgs.extend(make_group(f"r3-batch-{i}", n_exchanges=1))
    msgs, _, _ = la.compact_messages(msgs, "http://bg", "model", keep_last_groups=2)

    summary = _summary_after_compact(msgs)
    assert original in summary, "original user request was lost across compactions"


# ---------- pin: open outputs ----------

def test_open_outputs_block_inserted_when_provided(stub_summarizer, tmp_path):
    msgs = [sys_msg()]
    msgs.extend(make_group("write a plan to plan.md and keep going", n_exchanges=1))
    for i in range(5):
        msgs.extend(make_group(f"batch {i}", n_exchanges=1))

    open_outputs = {
        "/work/plan.md": 8904,
        "/work/notes.md": 1200,
    }
    new_msgs, did, _ = la.compact_messages(
        msgs, "http://bg", "model", keep_last_groups=2,
        open_outputs=open_outputs,
    )
    assert did
    summary = _summary_after_compact(new_msgs)
    assert la.PINNED_OUTPUTS_HEADER in summary
    assert "/work/plan.md" in summary
    assert "8904 chars" in summary
    assert "/work/notes.md" in summary


def test_open_outputs_rebuilt_each_compaction_from_stats(stub_summarizer, tmp_path):
    """Stale outputs from prior summary are dropped; fresh ones from stats injected."""
    # Round 1: plan.md exists at 100 chars.
    msgs = [sys_msg()]
    msgs.extend(make_group("write to plan.md", n_exchanges=1))
    for i in range(5):
        msgs.extend(make_group(f"r1-{i}", n_exchanges=1))
    msgs, _, _ = la.compact_messages(
        msgs, "http://bg", "model", keep_last_groups=2,
        open_outputs={"/work/plan.md": 100},
    )

    # Round 2: plan.md grew to 5000, plus a new file appeared.
    for i in range(5):
        msgs.extend(make_group(f"r2-{i}", n_exchanges=1))
    msgs, _, _ = la.compact_messages(
        msgs, "http://bg", "model", keep_last_groups=2,
        open_outputs={"/work/plan.md": 5000, "/work/index.md": 200},
    )

    summary = _summary_after_compact(msgs)
    # Fresh values present; stale 100-char value gone.
    assert "5000 chars" in summary
    assert "200 chars" in summary
    assert "100 chars" not in summary
    assert "/work/index.md" in summary


def test_no_open_outputs_block_when_none_tracked(stub_summarizer, tmp_path):
    msgs = [sys_msg()]
    msgs.extend(make_group("the original ask", n_exchanges=1))
    for i in range(5):
        msgs.extend(make_group(f"batch {i}", n_exchanges=1))

    new_msgs, did, _ = la.compact_messages(msgs, "http://bg", "model", keep_last_groups=2)
    assert did
    summary = _summary_after_compact(new_msgs)
    assert la.PINNED_OUTPUTS_HEADER not in summary  # nothing tracked → no block


# ---------- run_loop integration: write_file populates open_outputs ----------

def test_run_loop_records_write_file_in_open_outputs(monkeypatch, tmp_path):
    """A successful write_file tool call must register the file under
    stats['open_outputs'] so the next compaction can pin it."""
    # First response: call write_file. Second: final answer.
    client = FakeClient([
        fake_response(tool_calls=[("c1", "write_file", {"path": "out.md", "content": "hello"})]),
        fake_response(content="done"),
    ])
    stats: dict = {}
    code, _ = la.run_loop(
        client,
        [sys_msg(), user_msg("write out.md")],
        cwd=tmp_path,
        max_turns=4,
        verbose=False,
        model="x",
        temperature=0.0,
        top_p=1.0,
        thinking=False,
        show_thinking=False,
        stats=stats,
        bg_base_url="http://bg",
    )
    assert code == 0
    outs = stats.get("open_outputs") or {}
    assert outs, f"open_outputs not populated: {stats!r}"
    paths = list(outs.keys())
    assert any(p.endswith("out.md") for p in paths)
    # And the recorded char count matches the content length.
    assert outs[paths[0]] == len("hello")


def test_run_loop_skips_failed_write_file(monkeypatch, tmp_path):
    """A write_file that errors (e.g. outside writable roots) must NOT pollute
    open_outputs."""
    client = FakeClient([
        fake_response(tool_calls=[("c1", "write_file", {"path": "/etc/passwd", "content": "x"})]),
        fake_response(content="done"),
    ])
    stats: dict = {}
    code, _ = la.run_loop(
        client,
        [sys_msg(), user_msg("try to write outside")],
        cwd=tmp_path,
        max_turns=4,
        verbose=False,
        model="x",
        temperature=0.0,
        top_p=1.0,
        thinking=False,
        show_thinking=False,
        stats=stats,
        bg_base_url="http://bg",
    )
    assert code == 0
    assert not stats.get("open_outputs"), (
        f"failed write_file should not register an output; got {stats.get('open_outputs')!r}"
    )


# ---------- summarizer prompt: forbids producing the headers ----------

def test_summarizer_system_prompt_forbids_pinned_headers():
    """The BG model must be told not to produce these sections itself."""
    sys = la.SUMMARIZER_SYSTEM
    assert la.PINNED_REQUEST_HEADER.split(" ")[1] in sys.lower() or "original request" in sys.lower()
    assert "open outputs" in sys.lower()
    assert "do not output" in sys.lower() or "do not produce" in sys.lower()
    # And: explicit anti-completion-hallucination rule.
    assert "complete" in sys.lower()


# ---------- block-splitter unit tests ----------

def test_split_pinned_blocks_extracts_orig_drops_outputs():
    s = (
        f"{la.PINNED_REQUEST_HEADER}\n"
        "Do the thing.\n"
        "\n"
        f"{la.PINNED_OUTPUTS_HEADER}\n"
        "- /tmp/old.md (last write: 100 chars)\n"
        "\n"
        "## Files touched\n"
        "- /tmp/old.md\n"
        "\n"
        "## Current state\n"
        "Working on it."
    )
    orig, residual = la._split_pinned_blocks(s)
    assert orig is not None
    assert "Do the thing." in orig
    assert la.PINNED_REQUEST_HEADER in orig
    # Open outputs block is dropped from residual.
    assert la.PINNED_OUTPUTS_HEADER not in residual
    assert "100 chars" not in residual
    # Other sections survive.
    assert "## Files touched" in residual
    assert "## Current state" in residual


def test_split_pinned_blocks_empty_input():
    orig, residual = la._split_pinned_blocks("")
    assert orig is None
    assert residual == ""


def test_first_user_text_in_groups_handles_empty_and_present():
    assert la._first_user_text_in_groups([]) is None
    assert la._first_user_text_in_groups([[user_msg("hi"), assistant_msg("hello")]]) == "hi"
    # Whitespace-only is treated as missing.
    assert la._first_user_text_in_groups([[user_msg("   ")]]) is None
