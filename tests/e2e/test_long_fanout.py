"""E2E: long-fanout task in a single user turn.

Forces many tool calls inside one user-bounded group, which exercises
intra-group compaction. We use a low LOCAL_AGENT_COMPACT_THRESHOLD so the
compaction trigger fires within a reasonable run length even though server
n_ctx is 128k.

Pass criteria:
- agent exits 0
- summaries.txt exists with one line per input file
- at least one intra-group compaction OR completion under turn budget without
  any empty-retry stalls (the test logs both so we can compare runs)
"""
from __future__ import annotations

import pytest

from .util import run_agent_oneshot, make_sandbox_tree


@pytest.mark.e2e
def test_long_fanout_summarizes_all_files(tmp_path):
    files = make_sandbox_tree(tmp_path, n_files=40)
    task = (
        "Look at every file under ./files and write a single output file "
        "./summaries.txt with one line per input file in the form "
        "'<filename>: <content>' (the file contents are short). "
        "When done, write the final answer."
    )
    res = run_agent_oneshot(
        task, cwd=tmp_path,
        max_turns=80,
        env_extra={
            # force compaction trigger relatively early so the test exercises it
            "LOCAL_AGENT_COMPACT_THRESHOLD": "8000",
            "LOCAL_AGENT_COMPACT_KEEP_EXCHANGES": "3",
        },
        timeout=900,
    )

    print(f"\nturns={res.turns} inter={res.inter_compactions} "
          f"intra={res.intra_compactions} empty_retries={res.empty_retries} "
          f"tool_calls={len(res.tool_calls)}")
    if res.returncode != 0:
        print("STDERR TAIL:\n" + "\n".join(res.stderr.splitlines()[-50:]))

    assert res.returncode == 0, f"agent failed: {res.stdout[-2000:]}"
    summaries = tmp_path / "summaries.txt"
    assert summaries.exists(), "agent did not produce summaries.txt"
    lines = [ln for ln in summaries.read_text().splitlines() if ln.strip()]
    assert len(lines) >= len(files) * 0.9, (
        f"summaries.txt has {len(lines)} lines, expected ~{len(files)}"
    )
    assert res.empty_retries == 0, "agent stalled with empty-retry events"
