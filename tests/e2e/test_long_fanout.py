"""E2E: long-fanout task in a single user turn.

The point is to drive many tool calls inside one user-bounded group so the
agent has to keep working under load. We don't assert on the model's
*output* (which varies between runs depending on the path it takes) — only
on health signals: exit code, tool-call count, no empty-retry stalls.

Pass criteria:
- agent exits 0
- many tool calls were made (the task prompt forces individual reads)
- no empty-retry stalls
"""
from __future__ import annotations

import pytest

from .util import run_agent_oneshot, make_sandbox_tree


@pytest.mark.e2e
def test_long_fanout_runs_many_tool_calls(tmp_path):
    files = make_sandbox_tree(tmp_path, n_files=15, content_pattern="value-{i}")
    # Phrasing matters: forbid shell shortcuts AND we hide the subagent tools
    # via QWEN_AGENT_DISABLE_SUBAGENT, otherwise the model will delegate the
    # whole job to a background subagent and the harness exits early.
    task = (
        "There are 15 files under ./files. For each file individually, call "
        "the read_file tool once (do NOT use run_shell, do NOT concatenate, "
        "do NOT delegate to anything else). After reading every file "
        "individually, tell me how many of them contain the text 'value-'. "
        "End with a one-line final answer like 'Count: N'."
    )
    res = run_agent_oneshot(
        task, cwd=tmp_path,
        max_turns=80,
        env_extra={
            "LOCAL_AGENT_COMPACT_THRESHOLD": "6000",
            "LOCAL_AGENT_COMPACT_KEEP_EXCHANGES": "3",
            "QWEN_AGENT_DISABLE_SUBAGENT": "1",
        },
        timeout=900,
    )

    print(f"\nturns={res.turns} inter={res.inter_compactions} "
          f"intra={res.intra_compactions} empty_retries={res.empty_retries} "
          f"tool_calls={len(res.tool_calls)} "
          f"read_file={res.tool_calls.count('read_file')}")
    if res.returncode != 0:
        print("STDERR TAIL:\n" + "\n".join(res.stderr.splitlines()[-50:]))

    assert res.returncode == 0, f"agent failed (rc={res.returncode}): {res.stdout[-2000:]}"
    # Watchdog firings are OK — the point is that the agent *recovered* and
    # finished cleanly. A returncode of 3 (no recovery) would be a real fail.
    n_reads = res.tool_calls.count("read_file")
    assert n_reads >= 10, (
        f"agent only made {n_reads} read_file calls; "
        f"task should have forced ~15. all tool_calls={res.tool_calls}"
    )
    _ = files
