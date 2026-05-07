# End-to-end tests

These exercise the real foreground + background llama-server processes against
`la.py`. They are **excluded from the default `pytest` run** (see `pytest.ini`'s
`-m "not e2e"`) because they need GPUs and take minutes.

## Run them

```sh
./start-servers.sh                # only if not already running
./tests/e2e/run.sh                # runs all e2e tests (skips slow ones)
LA_E2E_SLOW=1 ./tests/e2e/run.sh  # also runs the 50-turn endurance test
```

`run.sh` checks server health on `:19434` and `:19435` before invoking pytest.

## What each test exercises

- `test_long_fanout.py` — many tool calls inside a single user turn → exercises
  intra-group compaction.
- `test_multi_group.py` — many sequential user turns → exercises inter-group
  compaction.
- `test_resume_midtask.py` — saves a fact in one REPL session, re-opens the
  REPL, asks for the fact back. Verifies transcript reload + resume banner.
- `test_endurance.py` (slow, gated) — 50 mixed write+read turns; both
  compaction modes should fire.

All tests run `la.py` with `-v` and parse stderr for `[compact]`,
`[compact-inter]`, `[compact-intra]`, `[watchdog]`, and `=== TURN N ===`
markers. See `util.parse_stderr`.
