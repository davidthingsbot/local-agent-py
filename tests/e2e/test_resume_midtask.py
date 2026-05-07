"""E2E: resume from a saved transcript and continue work.

Runs a small task, then re-invokes the REPL on the same cwd. The second
invocation must:
- find the prior transcript
- strip reasoning_content from any reloaded messages
- pre-compact iff the resumed transcript exceeds threshold
- accept a follow-up user turn and produce a sensible answer

Pass criteria:
- both invocations exit 0
- the second invocation's stdout shows the 'resumed transcript' banner
- no empty-retry stalls
"""
from __future__ import annotations

import pytest

from .util import run_repl_with_stdin


@pytest.mark.e2e
def test_resume_carries_state_to_followup(tmp_path):
    # First session: introduce a fact.
    res1 = run_repl_with_stdin(
        ["My favorite-token-for-test is BANANA-ALPHA-1234. Acknowledge it."],
        cwd=tmp_path,
        max_turns=5,
        timeout=600,
    )
    assert res1.returncode == 0
    assert res1.empty_retries == 0
    assert "new transcript" in res1.stdout or "transcript:" in res1.stdout.lower()

    # Second session: should resume and recall the fact.
    res2 = run_repl_with_stdin(
        ["What was my favorite-token-for-test? Answer with just the token."],
        cwd=tmp_path,
        max_turns=5,
        timeout=600,
    )
    assert res2.returncode == 0
    assert res2.empty_retries == 0
    assert "resumed transcript" in res2.stdout, res2.stdout[:1000]
    # the recalled token should appear in the agent's final answer
    assert "BANANA-ALPHA-1234" in res2.stdout, res2.stdout[-1500:]
