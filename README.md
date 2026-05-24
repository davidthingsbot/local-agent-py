# Qwen3.6 Agent Harness

Tiny local REPL/agent harness for testing Qwen3.6 as an agent.

Default target: one 256K-context llama.cpp server on `http://127.0.0.1:19434/v1`, with the Qwen3.6 35B-A3B MoE spread across both RTX 3090s.

Start/restart that server:

```bash
cd ~/work/local-agent-py
./start-servers.sh
```

The harness defaults are tuned for deep-context work: 256K server context, compaction around 85% of available context, 16K `n_keep`, large file/tool-result caps, and 8192 max completion tokens.

## Deep-context operating notes

The current preferred setup is **one large foreground server using both RTX 3090s**, not one foreground model plus a separate background model. The point is to give the main agent enough context to do hard tasks without constantly compacting or rediscovering state.

Verified configuration:

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` — 35B total / ~3B active MoE.
- Server: `llama-server` on `127.0.0.1:19434`.
- GPUs: both RTX 3090s via `CUDA_VISIBLE_DEVICES=0,1`.
- Context: `--ctx-size 262144` with one slot (`-np 1`).
- Split: `--split-mode layer --tensor-split 1,1`.
- KV: q8 (`-ctk q8_0 -ctv q8_0`).
- Memory after load was roughly 15GB on GPU 0 and 14GB on GPU 1, leaving headroom.

Why this matters:

- A huge first pass can take a while. That is acceptable for difficult work if subsequent turns reuse the existing conversation/KV context instead of rereading the same large material every turn.
- Avoid asking the model to repeatedly re-open giant files unless the task genuinely needs it. Prefer keeping the important state in the active conversation, checkpoint files, or targeted follow-up reads.
- The harness now delays compaction until around 90% of the server context (fallback threshold 250k prompt tokens). With 256K context, this avoids premature compaction after the known ~235k-token large-file read case.
- `read_file` can return up to 700k characters by default, but the model may still choose a smaller `max_chars` unless the task explicitly asks for the full file or a large value.
- If the model struggles on a large task, first check whether it actually received the required context, whether it asked for a truncated tool result, and whether compaction happened.

Validation run:

- Created `~/work/la-test/long-context-sentinel.txt` with 679,955 characters and sent it through `read_file(max_chars=700000)`.
- Transcript contained a 685,051-character tool result including the end sentinel.
- llama.cpp processed about 235,052 prompt tokens in ~186 seconds (~1262 prompt tokens/sec).
- Qwen correctly recovered the beginning, middle, and end sentinels.

Conclusion: 256K context works on this hardware. The main remaining challenge is loop design: avoid unnecessary repeated huge prompt ingestion, preserve task state explicitly, and make large reads intentional.

## Boot/service behavior

The preferred runtime is also installed as a user systemd service on this machine:

```bash
systemctl --user status local-agent-qwen.service
systemctl --user restart local-agent-qwen.service
```

The service starts the same 256K dual-GPU Qwen server on `127.0.0.1:19434` and is enabled for boot/login via user linger. `./start-servers.sh` remains the repo-local manual restart script and should match the service configuration.

## Hard-task behavior

The hard-task decomposition protocol is now part of the default system prompt. The model is instructed to plan multi-step work, checkpoint after roughly ten tool calls, create/update requested artifacts before doing another broad inspection pass, synthesize when approaching turn budget, and prefer relative write paths under the working directory.

Empty-response recovery now retries up to four times; the final retry re-enables thinking at low sampling as a panic recovery attempt.

## Start the REPL

```bash
cd ~/work/local-agent-py
./la.sh
```

Inside the REPL:

```text
local-agent> /capabilities
local-agent> /dirs
local-agent> inspect this directory and tell me what you see
local-agent> create a todo.md with three ideas for testing you as an agent
local-agent> /jobs
local-agent> /clear-jobs
local-agent> /context
local-agent> /reset
local-agent> /quit
```

Quick info without starting the REPL:

```bash
./la.sh --capabilities
./la.sh --dirs
```

## Run one task

```bash
cd ~/work/local-agent-py
./la.py -v --cwd ~/work/la-test \
  "Inspect this directory, create hello.md, and summarize what you did."
```

## Tools exposed to the model

- `list_dir` — list a directory
- `read_file` — read a UTF-8 text file
- `write_file` — write a file under the working directory
- `run_shell` — run conservative local shell commands; obvious destructive/network commands are blocked
- `ask_subagent` — delegate a bounded task to a fresh isolated Qwen subagent using the same sandbox; child agents cannot spawn further subagents
- `start_background_subagent` — start a bounded subagent task in the background and return a job id immediately
- `check_background_job` — check a background job status/result
- `list_background_jobs` — list known background jobs
- `clear_background_jobs` — clear job tracking records without killing child processes or deleting files they created

Background job files live under the sandbox at `.qwen-agent-jobs/<job-id>/`. In the REPL, `/jobs` lists known jobs and `/clear-jobs` clears records. `./la.sh` clears old job records on startup so stale jobs do not confuse a new interactive session.

## Notes

- `la.sh` uses `~/work/la-test` as the default read/write sandbox. Override sandbox with `LA_CWD=/some/path ./la.sh`.
- `thinking=true` by default; disable with `./la.sh --no-thinking`.
- `show_thinking=true` by default; disable with `./la.sh --no-show-thinking`.
- Use `--cwd` directly with `la.py` to choose a different sandbox. Writes are only allowed under that directory.
- `-v` shows tool calls/results, which is useful for evaluating agent behavior.
- `/reset` clears the REPL conversation context.
- `QWEN_BG_BASE_URL` is optional now. If unset, compaction/subagent calls use the same 256K server as foreground work. Set it only when deliberately running a separate background server.
- To restore the old one-server-per-GPU layout, run `LOCAL_AGENT_SPLIT_SERVERS=1 ./start-servers.sh`.
