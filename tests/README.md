# Tests

Two suites:

- **Unit tests** (`tests/test_*.py`): mocked, fast (~100ms total). Run by default.
- **End-to-end tests** (`tests/e2e/test_*.py`): real `llama-server` processes, slower. Excluded from the default run via the `e2e` pytest marker.

## Run

```sh
# unit suite (the default)
./tests/run.sh
# or directly:
python3 -m pytest

# end-to-end suite (requires foreground :19434 + background :19435 servers)
./tests/e2e/run.sh
LA_E2E_SLOW=1 ./tests/e2e/run.sh   # also runs the gated 50-turn endurance test
```

## Coverage

| File | What it guards |
|---|---|
| `test_compaction.py` | `split_groups`, `_compact_inter_group`, `_compact_intra_group`, recompaction with `COMPACT_MARKER`/`INTRA_SUMMARY_MARKER`, tool_call/tool pairing invariant, `message_to_dict` reasoning_content stripping, `estimate_prompt_tokens`. |
| `test_threshold.py` | `compute_compact_threshold` precedence (env override > /props > fallback), `query_server_n_ctx` URL handling, `n_keep` + `enable_thinking` are sent on every chat completion, usage stats wired through. |
| `test_watchdog.py` | Empty-thinking-only response retried once with thinking off; retry-also-empty exits 3; genuinely-empty (no reasoning) is *not* retried; transcript shape preserved; counter visible. |
| `test_resume.py` | Transcript save/load, legacy list format, latest-by-name, `strip_reasoning_content`, pre-compact decision rules, full simulated resume pipeline. |
| `test_strategy.py` | Env-gated decomposition prompt clause, separate inter/intra compaction counters, `last_compact_at_turn`. |
| `e2e/test_resume_midtask.py` | Real REPL session 1 + session 2; second session resumes and recalls a fact. |
| `e2e/test_long_fanout.py` | One user turn with 40 file reads; exercises intra-group compaction. |
| `e2e/test_multi_group.py` | 12 sequential user turns; exercises inter-group compaction. |
| `e2e/test_endurance.py` | (gated) 50 mixed write+read turns. |
