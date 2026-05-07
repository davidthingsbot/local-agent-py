# Qwen3.6 Agent Harness

Tiny local REPL/agent harness for testing Qwen3.6 as an agent.

It talks to the currently-running llama.cpp OpenAI-compatible server at:

```text
http://127.0.0.1:19434/v1
```

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
- Current default disables Qwen thinking blocks for cleaner tool-loop behavior.
