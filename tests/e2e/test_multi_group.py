"""E2E: multiple sequential user turns through the REPL.

Drives the REPL with a long sequence of user prompts so that we accumulate
many user-bounded groups and inter-group compaction must fire. Threshold is
forced low to keep the test under a reasonable wall-clock time.

Pass criteria:
- REPL exits 0
- at least one inter-group compaction event observed
- no empty-retry stalls
"""
from __future__ import annotations

import pytest

from .util import run_repl_with_stdin


@pytest.mark.e2e
def test_multi_group_triggers_inter_compaction(tmp_path):
    # 12 small read-only tasks; each ends quickly with a final answer.
    user_turns = [
        f"Run `echo hello-{i}` and tell me the exact stdout."
        for i in range(12)
    ]
    res = run_repl_with_stdin(
        user_turns, cwd=tmp_path,
        max_turns=8,
        env_extra={
            # Threshold has to be below the baseline prompt size (system prompt
            # + tool schemas, ~1500 tokens) so the trigger fires within the
            # 12-turn run. 800 is comfortably under that.
            "LOCAL_AGENT_COMPACT_THRESHOLD": "800",
            "LOCAL_AGENT_COMPACT_KEEP": "3",
        },
        timeout=1200,
    )

    print(f"\nturns_max={res.turns} inter={res.inter_compactions} "
          f"intra={res.intra_compactions} empty_retries={res.empty_retries}")
    if res.returncode != 0:
        print("STDERR TAIL:\n" + "\n".join(res.stderr.splitlines()[-50:]))

    assert res.returncode == 0
    assert res.inter_compactions >= 1, (
        "expected at least one inter-group compaction; "
        f"saw inter={res.inter_compactions} intra={res.intra_compactions}"
    )
    assert res.empty_retries == 0
