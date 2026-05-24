# AGENTS.md — local-agent-py

This repo is the small Python harness for testing Qwen3.6 as a local agent.

## Current goal

Make the loop succeed on hard tasks that require deep context. David explicitly does not mind a slow first ingestion pass if that buys useful working context afterward; the failure mode to avoid is paying the same huge context cost every turn or losing state through premature compaction.

## Preferred server setup

Use one 256K-context foreground server across both RTX 3090s:

```bash
cd ~/work/local-agent-py
./start-servers.sh
```

Expected live server:

- URL: `http://127.0.0.1:19434/v1`
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- Type: MoE, 35B total / ~3B active
- Context: 262144 tokens
- Slots: 1
- GPUs: both 3090s, layer split 1:1

Check with:

```bash
curl -s http://127.0.0.1:19434/props | jq '.model_alias, .default_generation_settings.n_ctx, .total_slots'
nvidia-smi
```

Do **not** casually restart the old second/background server on port 19435. The current design intentionally dedicates both cards to the main long-context model. Only use `LOCAL_AGENT_SPLIT_SERVERS=1 ./start-servers.sh` when deliberately testing one-server-per-GPU behavior.

## Deep-context lessons learned

- 256K context is confirmed working on this machine with q8 KV.
- A 679,955-character file was read into the tool result; llama.cpp processed ~235K prompt tokens in ~186s, and Qwen correctly found sentinels at the beginning, middle, and end.
- Slow prompt ingestion is acceptable for hard tasks, but repeated ingestion of the same giant context on every turn is not.
- Preserve important task state explicitly: checkpoint files, concise progress summaries, and targeted reads beat repeatedly dumping giant files.
- If Qwen fails a deep task, inspect the transcript before blaming the model. Common causes:
  - `read_file` was called with too small `max_chars`.
  - A tool result was truncated before the needed evidence.
  - Compaction happened and dropped nuance.
  - The task relied on subagents that had no parent history.

## Harness defaults to preserve

- `DEFAULT_BG_BASE_URL` falls back to the foreground server unless explicitly overridden.
- Compaction ratio: 0.85 of server `n_ctx`.
- Keep groups/exchanges: 8 / 12.
- `LOCAL_AGENT_N_KEEP`: 16384.
- `LOCAL_AGENT_MAX_COMPLETION_TOKENS`: 8192.
- `LOCAL_AGENT_MAX_FILE_CHARS`: 700000.
- `LOCAL_AGENT_MAX_OUTPUT_CHARS`: 240000.

These are tuned for deep-context experiments. Don’t reduce them without a specific reason and a validation run.

## Validation habit

After changing context/server/tool-loop behavior, run at least:

```bash
cd ~/work/local-agent-py
python3 -m pytest tests/test_watchdog.py tests/test_empty_retry.py tests/test_threshold.py -q
```

For real long-context validation, use the sentinel-file pattern from the README: force a large `read_file(max_chars=700000)` result and verify the model can recover facts from the beginning, middle, and end.

## Repo hygiene

- Do not commit run logs, transcripts, job records, or sandbox outputs.
- Keep `README.md` updated with operational knowledge that David should see.
- Keep this file updated with instructions future agents need when working inside this repo.
