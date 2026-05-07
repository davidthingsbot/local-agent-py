"""E2E: long unattended endurance run (gated, slow).

This is the soak test — many user turns, deliberately mixing read/write tool
use, expected to trigger many compactions. Skipped unless LA_E2E_SLOW=1.

Pass criteria:
- exit 0 throughout
- multiple compactions observed
- no empty-retry stalls
- final transcript still well-formed (parses, system message intact)
"""
from __future__ import annotations

import json
import os

import pytest

from .util import run_repl_with_stdin


@pytest.mark.e2e
@pytest.mark.skipif(os.environ.get("LA_E2E_SLOW") != "1",
                    reason="set LA_E2E_SLOW=1 to run the long endurance test")
def test_50_turn_unattended(tmp_path):
    user_turns = []
    for i in range(50):
        user_turns.append(
            f"Write a file ./notes/n{i:03d}.txt containing the text 'note {i}', "
            "then list ./notes and report the count."
        )
    res = run_repl_with_stdin(
        user_turns, cwd=tmp_path,
        max_turns=12,
        env_extra={
            "LOCAL_AGENT_COMPACT_THRESHOLD": "20000",
        },
        timeout=3600,
    )

    print(f"\ninter={res.inter_compactions} intra={res.intra_compactions} "
          f"empty_retries={res.empty_retries}")

    assert res.returncode == 0
    assert res.empty_retries == 0
    assert (res.inter_compactions + res.intra_compactions) >= 2

    # transcript must still parse
    transcripts = list((tmp_path / ".local-agent-transcripts").glob("transcript-*.json"))
    assert transcripts
    payload = json.loads(transcripts[-1].read_text())
    msgs = payload["messages"]
    assert msgs[0]["role"] == "system"
