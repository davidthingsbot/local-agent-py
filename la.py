#!/usr/bin/env python3
"""
la.py - small local agent harness for testing Qwen3.6 as an agent.

This is intentionally much simpler than OpenClaw/Hermes/NanoClaw. It gives the
model a loop, a working directory, and a small safe toolset so we can evaluate:
- tool-call reliability
- planning / iteration
- file inspection and synthesis
- command use discipline

Default tools are local and conservative. Shell commands are blocked if they look
obviously destructive or network/exfiltration-oriented.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: pip install openai", file=sys.stderr)
    raise

DEFAULT_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:19434/v1")
# Default compaction/subagent work to the same 256K foreground server. Set
# QWEN_BG_BASE_URL explicitly only when a separate background server is running.
DEFAULT_BG_BASE_URL = os.environ.get("QWEN_BG_BASE_URL", DEFAULT_BASE_URL)
DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "qwen")
DEFAULT_CWD = Path(os.environ.get("LOCAL_AGENT_CWD", str(Path.home() / ".openclaw" / "workspace"))).expanduser()

WRITE_ROOTS: list[Path] = []

MAX_FILE_CHARS = int(os.environ.get("LOCAL_AGENT_MAX_FILE_CHARS", "700000"))
MAX_OUTPUT_CHARS = int(os.environ.get("LOCAL_AGENT_MAX_OUTPUT_CHARS", "240000"))

_THRESHOLD_OVERRIDE = os.environ.get("LOCAL_AGENT_COMPACT_THRESHOLD")
COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE: int | None = int(_THRESHOLD_OVERRIDE) if _THRESHOLD_OVERRIDE else None
COMPACT_THRESHOLD_RATIO = float(os.environ.get("LOCAL_AGENT_COMPACT_THRESHOLD_RATIO", "0.85"))
COMPACT_THRESHOLD_FALLBACK = 220_000
COMPACT_KEEP_LAST_GROUPS = max(1, int(os.environ.get("LOCAL_AGENT_COMPACT_KEEP", "8")))
COMPACT_KEEP_LAST_EXCHANGES = max(1, int(os.environ.get("LOCAL_AGENT_COMPACT_KEEP_EXCHANGES", "12")))
COMPACT_MARKER = "\n\n# Compacted earlier conversation\n"
INTRA_SUMMARY_MARKER = "[Earlier work in this task — summary]\n"
PINNED_REQUEST_HEADER = "## Original request (verbatim — pinned)"
PINNED_OUTPUTS_HEADER = "## Open outputs (files written this session)"
N_KEEP_TOKENS = int(os.environ.get("LOCAL_AGENT_N_KEEP", "16384"))
MAX_COMPLETION_TOKENS = int(os.environ.get("LOCAL_AGENT_MAX_COMPLETION_TOKENS", "8192"))

BLOCKED_COMMAND_PATTERNS = [
    "rm -rf", "rm -fr", "mkfs", "dd if=", ":(){", "shutdown", "reboot", "poweroff",
    "sudo ", "su ", "chmod -R 777", "chown -R", "curl ", "wget ", "scp ", "rsync ",
    "nc ", "ncat ", "telnet ", "ssh ", "gh repo delete", "git push", "git clean -fdx",
]

DECOMPOSITION_CLAUSE = """## Task decomposition

For multi-step tasks, prefer to work in checkpoints. After roughly ten tool calls, end your turn with a brief progress summary listing what you have done, what is left, and any decisions made. Then wait for the user to say "continue" or to redirect. This keeps context manageable and lets the user steer.
"""


CAPABILITIES_TEXT = """# Qwen3.6 Agent Harness Capabilities

You are Qwen3.6 running in a tiny local agent harness.

## Tools

- list_dir(path, max_entries=100): list directory contents with names, types, and sizes.
- read_file(path, max_chars=700000): read a UTF-8 text file. For deep-context tasks, request a larger max_chars value when you need the whole file.
- write_file(path, content): write a UTF-8 text file under the current agent working directory only.
- run_shell(command, timeout_seconds=20): run a conservative local shell command in the current working directory.
- ask_subagent(task, max_turns=6): delegate a bounded task to a fresh isolated Qwen subagent using the same working directory. Subagents cannot spawn further subagents.
- start_background_subagent(task, max_turns=6): start a bounded subagent task in the background and return a job id immediately.
- check_background_job(job_id): check a background subagent job and return its output if finished.
- list_background_jobs(): list known background subagent jobs.
- clear_background_jobs(): clear background job tracking records. This does not kill running child processes or delete files they created.

## Filesystem policy

- Read/list: any local path the OS user can access.
- Write: paths must resolve under `cwd` or one of the configured extra writable roots (see `/dirs`). Relative paths resolve under `cwd`; absolute paths are accepted only if they fall inside a writable root. `..` escapes that leave every root are blocked.
- Shell: runs in `cwd`; obvious destructive, privileged, network, and exfiltration-ish commands are blocked by string pattern.

## Behavioral rules

- Use tools when they materially improve the answer.
- Do not claim inspection unless a tool was used or context was provided.
- Prefer read-only inspection commands.
- Do not perform destructive, network, credential, or privacy-sensitive actions.
- If blocked, state exactly what blocked you.
- End with a short final answer summarizing what you did and found.
"""

SYSTEM_PROMPT = CAPABILITIES_TEXT

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory contents with names, types, and sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, absolute or relative to cwd."},
                    "max_entries": {"type": "integer", "description": "Maximum entries to return, default 100."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file, truncated to a safe size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, absolute or relative to cwd."},
                    "max_chars": {"type": "integer", "description": "Maximum chars to return. Default and maximum are controlled by LOCAL_AGENT_MAX_FILE_CHARS; current default is 700000."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a UTF-8 text file under the agent working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path under cwd."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a safe local shell command in cwd. Read-only commands are preferred.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout_seconds": {"type": "integer", "description": "Timeout, default 20, max 60."},
                },
                "required": ["command"],
            },
        },
    },
]

SUBAGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_subagent",
        "description": "Delegate a bounded task to a fresh isolated Qwen subagent using the same working directory. The subagent cannot spawn further subagents.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task for the subagent to perform."},
                "max_turns": {"type": "integer", "description": "Maximum child agent turns, default 6, max 10."},
            },
            "required": ["task"],
        },
    },
}

BACKGROUND_SUBAGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "start_background_subagent",
            "description": "Start a bounded Qwen subagent task in the background using the same working directory. Returns immediately with a job id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task for the background subagent to perform."},
                    "max_turns": {"type": "integer", "description": "Maximum child agent turns, default 6, max 10."},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_background_jobs",
            "description": "List known background subagent jobs and their running/finished status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_background_jobs",
            "description": "Clear background job tracking records. Does not kill running child processes or delete files they created.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_background_job",
            "description": "Check status/output for a background subagent job id.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Background job id returned by start_background_subagent."}},
                "required": ["job_id"],
            },
        },
    },
]


def tools_for(enable_subagents: bool) -> list[dict[str, Any]]:
    if enable_subagents:
        return TOOLS + [SUBAGENT_TOOL] + BACKGROUND_SUBAGENT_TOOLS
    return TOOLS


def resolve_path(cwd: Path, path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = cwd / p
    return p.resolve()


def _path_under_any(target: Path, roots: list[Path]) -> bool:
    t = str(target)
    for r in roots:
        rs = str(r)
        if t == rs or t.startswith(rs + os.sep):
            return True
    return False


def writable_roots(cwd: Path) -> list[Path]:
    return WRITE_ROOTS or [cwd.resolve()]


def extra_write_dirs(cwd: Path) -> list[Path]:
    crs = cwd.resolve()
    return [r for r in WRITE_ROOTS if r != crs]


def truncate(s: str, n: int = MAX_OUTPUT_CHARS) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n\n...[truncated {len(s) - n} chars]"


def tool_list_dir(cwd: Path, path: str, max_entries: int = 100) -> dict[str, Any]:
    p = resolve_path(cwd, path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    if not p.is_dir():
        return {"error": f"not a directory: {p}"}
    entries = []
    for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[: max(1, min(max_entries, 500))]:
        try:
            st = child.stat()
            entries.append({
                "name": child.name,
                "path": str(child),
                "type": "dir" if child.is_dir() else "file",
                "size": st.st_size,
            })
        except OSError as e:
            entries.append({"name": child.name, "path": str(child), "error": str(e)})
    return {"path": str(p), "entries": entries, "count": len(entries)}


def tool_read_file(cwd: Path, path: str, max_chars: int = MAX_FILE_CHARS) -> dict[str, Any]:
    p = resolve_path(cwd, path)
    if not p.exists():
        return {"error": f"not found: {p}"}
    if not p.is_file():
        return {"error": f"not a file: {p}"}
    max_chars = max(1_000, min(int(max_chars or MAX_FILE_CHARS), MAX_FILE_CHARS))
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": str(e)}
    return {"path": str(p), "chars": len(text), "content": truncate(text, max_chars)}


def tool_write_file(cwd: Path, path: str, content: str) -> dict[str, Any]:
    target = resolve_path(cwd, path)
    roots = writable_roots(cwd)
    if not _path_under_any(target, roots):
        return {"error": f"target outside writable roots {[str(r) for r in roots]}: {target}"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(target), "chars": len(content)}


def command_block_reason(command: str) -> str | None:
    lowered = " ".join(command.lower().split())
    for pat in BLOCKED_COMMAND_PATTERNS:
        if pat in lowered:
            return f"blocked command pattern: {pat.strip()}"
    return None


def tool_run_shell(cwd: Path, command: str, timeout_seconds: int = 20) -> dict[str, Any]:
    reason = command_block_reason(command)
    if reason:
        return {"error": reason, "command": command}
    timeout_seconds = max(1, min(int(timeout_seconds or 20), 60))
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            executable="/bin/bash",
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": truncate(proc.stdout),
            "stderr": truncate(proc.stderr),
        }
    except subprocess.TimeoutExpired as e:
        return {"error": f"timeout after {timeout_seconds}s", "stdout": truncate(e.stdout or ""), "stderr": truncate(e.stderr or "")}
    except Exception as e:
        return {"error": str(e), "command": command}


def subagent_cmd(cwd: Path, task: str, max_turns: int = 6, thinking: bool = True, base_url: str | None = None) -> list[str]:
    max_turns = max(1, min(int(max_turns or 6), 10))
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cwd", str(cwd),
        "--max-turns", str(max_turns),
        "--base-url", base_url or DEFAULT_BG_BASE_URL,
        "--no-show-thinking",
    ]
    for extra in extra_write_dirs(cwd):
        cmd += ["--write-dir", str(extra)]
    if not thinking:
        cmd.append("--no-thinking")
    cmd.append(task)
    return cmd


def subagent_env() -> dict[str, str]:
    env = os.environ.copy()
    env["QWEN_AGENT_DISABLE_SUBAGENT"] = "1"
    return env


def tool_ask_subagent(cwd: Path, task: str, max_turns: int = 6, thinking: bool = True, base_url: str | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            subagent_cmd(cwd, task, max_turns, thinking, base_url),
            cwd=str(cwd),
            env=subagent_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "task": task,
            "stdout": truncate(proc.stdout),
            "stderr": truncate(proc.stderr, 8000),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": "subagent timeout after 180s",
            "stdout": truncate(e.stdout or ""),
            "stderr": truncate(e.stderr or "", 8000),
        }


def jobs_dir(cwd: Path) -> Path:
    d = cwd / ".qwen-agent-jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def transcripts_dir(cwd: Path) -> Path:
    d = cwd / ".local-agent-transcripts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_transcript_path(cwd: Path) -> Path:
    return transcripts_dir(cwd) / f"transcript-{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"


def latest_transcript_path(cwd: Path) -> Path | None:
    files = sorted(transcripts_dir(cwd).glob("transcript-*.json"))
    return files[-1] if files else None


def load_transcript(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "messages" in raw:
        return raw["messages"]
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unrecognized transcript format: {path}")


def save_transcript(path: Path, cwd: Path, messages: list[dict[str, Any]]) -> None:
    payload = {"version": 1, "cwd": str(cwd), "messages": messages}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def clear_background_jobs(cwd: Path) -> int:
    d = cwd / ".qwen-agent-jobs"
    if not d.exists():
        return 0
    count = sum(1 for _ in d.glob("*/meta.json"))
    shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return count


def tool_start_background_subagent(cwd: Path, task: str, max_turns: int = 6, thinking: bool = True, base_url: str | None = None) -> dict[str, Any]:
    job_id = f"job-{int(time.time())}-{os.getpid()}"
    jd = jobs_dir(cwd) / job_id
    jd.mkdir(parents=True, exist_ok=True)
    stdout_path = jd / "stdout.txt"
    stderr_path = jd / "stderr.txt"
    exit_path = jd / "exitcode.txt"
    meta_path = jd / "meta.json"
    cmd = subagent_cmd(cwd, task, max_turns, thinking, base_url)
    script = " ".join(shlex.quote(x) for x in cmd) + f" > {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}; echo $? > {shlex.quote(str(exit_path))}"
    proc = subprocess.Popen(["/bin/bash", "-lc", script], cwd=str(cwd), env=subagent_env(), start_new_session=True)
    meta = {"job_id": job_id, "pid": proc.pid, "task": task, "cmd": cmd, "started": time.time(), "stdout": str(stdout_path), "stderr": str(stderr_path), "exitcode": str(exit_path)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "job_id": job_id, "pid": proc.pid, "task": task, "status": "running", "job_dir": str(jd)}


def tool_check_background_job(cwd: Path, job_id: str) -> dict[str, Any]:
    jd = jobs_dir(cwd) / job_id
    meta_path = jd / "meta.json"
    if not meta_path.exists():
        return {"ok": False, "error": f"unknown job_id: {job_id}"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    exit_path = Path(meta["exitcode"])
    stdout_path = Path(meta["stdout"])
    stderr_path = Path(meta["stderr"])
    if exit_path.exists():
        code = exit_path.read_text(encoding="utf-8", errors="replace").strip()
        return {
            "ok": code == "0",
            "job_id": job_id,
            "status": "finished",
            "returncode": int(code) if code.isdigit() else code,
            "task": meta.get("task"),
            "stdout": truncate(stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""),
            "stderr": truncate(stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else "", 8000),
        }
    return {"ok": True, "job_id": job_id, "status": "running", "pid": meta.get("pid"), "task": meta.get("task")}


def execute_tool(cwd: Path, name: str, args: dict[str, Any], thinking: bool = True) -> dict[str, Any]:
    if name == "list_dir":
        return tool_list_dir(cwd, args.get("path", "."), args.get("max_entries", 100))
    if name == "read_file":
        return tool_read_file(cwd, args.get("path", ""), args.get("max_chars", MAX_FILE_CHARS))
    if name == "write_file":
        return tool_write_file(cwd, args.get("path", ""), args.get("content", ""))
    if name == "run_shell":
        return tool_run_shell(cwd, args.get("command", ""), args.get("timeout_seconds", 20))
    if name == "ask_subagent":
        if os.environ.get("QWEN_AGENT_DISABLE_SUBAGENT") == "1":
            return {"error": "subagents disabled inside subagent"}
        return tool_ask_subagent(cwd, args.get("task", ""), args.get("max_turns", 6), thinking, DEFAULT_BG_BASE_URL)
    if name == "start_background_subagent":
        if os.environ.get("QWEN_AGENT_DISABLE_SUBAGENT") == "1":
            return {"error": "background subagents disabled inside subagent"}
        return tool_start_background_subagent(cwd, args.get("task", ""), args.get("max_turns", 6), thinking, DEFAULT_BG_BASE_URL)
    if name == "list_background_jobs":
        return tool_list_background_jobs(cwd)
    if name == "clear_background_jobs":
        return {"ok": True, "cleared": clear_background_jobs(cwd)}
    if name == "check_background_job":
        return tool_check_background_job(cwd, args.get("job_id", ""))
    return {"error": f"unknown tool: {name}"}


def message_to_dict(msg: Any) -> dict[str, Any]:
    # Intentionally drop reasoning_content: it's per-turn scratch space (Qwen <think>),
    # not consumed by the chat template on subsequent turns. Persisting it bloats the
    # transcript and the next request payload without changing model behavior.
    d = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = []
        for tc in msg.tool_calls:
            d["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })
    return d


def split_groups(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Split messages into (head, groups). Head is leading system messages.
    Each group starts with a user message and includes the assistant/tool messages that follow,
    so tool_call/tool-response pairs always stay inside one group.
    """
    head: list[dict[str, Any]] = []
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if not groups and not current and role == "system":
            head.append(m)
            continue
        if role == "user":
            if current:
                groups.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        groups.append(current)
    return head, groups


def render_groups_for_summary(groups: list[list[dict[str, Any]]]) -> str:
    out: list[str] = []
    for g in groups:
        for m in g:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "user":
                out.append(f"USER: {truncate(content, 4000)}")
            elif role == "assistant":
                if content:
                    out.append(f"ASSISTANT: {truncate(content, 4000)}")
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "") or ""
                    out.append(f"ASSISTANT calls {name}({truncate(args, 600)})")
            elif role == "tool":
                out.append(f"TOOL: {truncate(content, 1500)}")
    return "\n".join(out)


SUMMARIZER_SYSTEM = """You compress agent transcripts. Produce a concise operational summary so the agent can continue without losing essential context.

Preserve:
- Files read or written, with paths
- Commands run and important outcomes (success/failure, key output)
- Decisions and the reasons for them
- Errors encountered and whether resolved
- Current state of work — what was just being done, and what is INCOMPLETE

Drop verbose tool output, repetition, and conversational filler.

Hard rules:
- Do NOT output a "## Original request" section. The original user request is pinned and inserted before your output by the harness.
- Do NOT output an "## Open outputs" section. Open output files are tracked separately and inserted by the harness.
- Do NOT declare the task complete or finished unless the user explicitly confirmed completion. Continuation cues like "continue", "next batch", "resume" mean the work is in progress — your "Current state" should describe what is still pending, not assert that the task is done.
- Do NOT paraphrase or summarize the original user request — the pinned block is the source of truth for what was asked.

Output sections (in this order, no others):
## Files touched
## Key actions and findings
## Current state (what is incomplete and what should happen next)
"""


def summarize_via_bg(text: str, bg_base_url: str, model: str, prior_summary: str | None = None) -> str:
    client = OpenAI(base_url=bg_base_url, api_key="local-not-needed")
    user_content = ""
    if prior_summary:
        user_content += f"Prior summary (incorporate and update):\n{prior_summary}\n\n"
    user_content += f"New transcript to incorporate:\n{text}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return (resp.choices[0].message.content or "").strip()


def split_group_into_exchanges(group: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[list[dict[str, Any]]]]:
    """Split a user-bounded group into (user_msg, exchanges).
    An exchange is a list starting with one assistant message, optionally followed by
    its matching tool responses. Cuts always fall at exchange boundaries, so
    tool_call / tool-response pairs are never separated.
    """
    if not group or group[0].get("role") != "user":
        return None, []
    user_msg = group[0]
    exchanges: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for m in group[1:]:
        role = m.get("role")
        if role == "assistant":
            if current:
                exchanges.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        exchanges.append(current)
    return user_msg, exchanges


def render_exchanges_for_summary(exchanges: list[list[dict[str, Any]]]) -> str:
    out: list[str] = []
    for ex in exchanges:
        for m in ex:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "assistant":
                if content:
                    out.append(f"ASSISTANT: {truncate(content, 4000)}")
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "") or ""
                    out.append(f"ASSISTANT calls {name}({truncate(args, 600)})")
            elif role == "tool":
                out.append(f"TOOL: {truncate(content, 1500)}")
    return "\n".join(out)


def _first_user_text_in_groups(groups: list[list[dict[str, Any]]]) -> str | None:
    """Return the content of the first user message across these groups, or None."""
    for g in groups:
        for m in g:
            if m.get("role") == "user":
                content = (m.get("content") or "").strip()
                if content:
                    return content
                return None
    return None


def _split_pinned_blocks(prior_summary: str) -> tuple[str | None, str]:
    """Extract the '## Original request' block from a prior summary.

    Returns (orig_request_block_or_None, residual_summary).
    The 'Open outputs' block is *dropped* on the way out — it's rebuilt from
    current stats every compaction so it never goes stale, and we don't want
    the BG model to see and copy a snapshot version.
    """
    if not prior_summary:
        return None, prior_summary or ""
    lines = prior_summary.splitlines()
    orig_lines: list[str] = []
    residual_lines: list[str] = []
    mode = "pre"  # "pre" | "orig" | "outputs" | "other"
    for line in lines:
        stripped = line.strip()
        if stripped == PINNED_REQUEST_HEADER:
            mode = "orig"
            orig_lines.append(line)
            continue
        if stripped == PINNED_OUTPUTS_HEADER:
            mode = "outputs"
            continue
        if line.startswith("## "):
            mode = "other"
            residual_lines.append(line)
            continue
        if mode == "orig":
            orig_lines.append(line)
        elif mode == "outputs":
            pass  # drop
        else:
            residual_lines.append(line)
    orig_block = "\n".join(orig_lines).strip() if orig_lines else None
    residual = "\n".join(residual_lines).strip()
    return orig_block, residual


def _format_open_outputs_block(open_outputs: dict[str, int] | None) -> str | None:
    if not open_outputs:
        return None
    lines = [PINNED_OUTPUTS_HEADER]
    for path, chars in sorted(open_outputs.items()):
        lines.append(f"- {path} (last write: {chars} chars)")
    return "\n".join(lines)


def _compose_pinned_summary(
    orig_request_block: str | None,
    open_outputs_block: str | None,
    body: str,
) -> str:
    parts: list[str] = []
    if orig_request_block:
        parts.append(orig_request_block)
    if open_outputs_block:
        parts.append(open_outputs_block)
    body = (body or "").strip()
    if body:
        parts.append(body)
    return "\n\n".join(parts)


def _compact_inter_group(
    head: list[dict[str, Any]],
    groups: list[list[dict[str, Any]]],
    bg_base_url: str,
    model: str,
    keep_last_groups: int,
    verbose: bool,
    open_outputs: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], bool, str]:
    if len(groups) <= keep_last_groups + 1:
        return head, groups, False, f"only {len(groups)} group(s)"
    to_summarize = groups[:-keep_last_groups]
    keep_tail = list(groups[-keep_last_groups:])

    sys_msg = head[0]
    sys_content = sys_msg.get("content", "") or ""
    prior_summary: str | None = None
    base_sys = sys_content
    if COMPACT_MARKER in sys_content:
        base_sys, _, prior_summary = sys_content.partition(COMPACT_MARKER)

    # Split pinned blocks out of the prior summary before sending to BG. The BG
    # model only ever sees the residual, so it can never paraphrase, drop, or
    # falsely declare completion of the original request.
    orig_request_block: str | None = None
    residual_prior: str | None = None
    if prior_summary:
        orig_request_block, residual_prior = _split_pinned_blocks(prior_summary)
    if orig_request_block is None:
        first_user = _first_user_text_in_groups(to_summarize)
        if first_user:
            orig_request_block = f"{PINNED_REQUEST_HEADER}\n{first_user}"

    transcript_text = render_groups_for_summary(to_summarize)
    if verbose:
        print(f"[compact-inter] summarizing {len(to_summarize)} group(s), ~{len(transcript_text)} chars", file=sys.stderr)
    try:
        body = summarize_via_bg(transcript_text, bg_base_url, model, residual_prior)
    except Exception as e:
        return head, groups, False, f"inter-group summarize failed: {e}"

    open_outputs_block = _format_open_outputs_block(open_outputs)
    summary = _compose_pinned_summary(orig_request_block, open_outputs_block, body)

    new_sys = {**sys_msg, "content": base_sys.rstrip() + COMPACT_MARKER + summary}
    new_head = [new_sys] + head[1:]
    return new_head, keep_tail, True, f"compacted {len(to_summarize)} group(s) into {len(summary)} chars"


def _compact_intra_group(
    group: list[dict[str, Any]],
    bg_base_url: str,
    model: str,
    keep_last_exchanges: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], bool, str]:
    user_msg, exchanges = split_group_into_exchanges(group)
    if user_msg is None:
        return group, False, "active group not user-bounded"

    prior_summary: str | None = None
    if exchanges and len(exchanges[0]) == 1:
        first = exchanges[0][0]
        first_content = first.get("content", "") or ""
        if first.get("role") == "assistant" and not first.get("tool_calls") and first_content.startswith(INTRA_SUMMARY_MARKER):
            prior_summary = first_content[len(INTRA_SUMMARY_MARKER):]
            exchanges = exchanges[1:]

    if len(exchanges) <= keep_last_exchanges + 1:
        return group, False, f"active group has {len(exchanges)} exchange(s) after prior summary"

    to_summarize = exchanges[:-keep_last_exchanges]
    keep_tail = exchanges[-keep_last_exchanges:]
    transcript_text = render_exchanges_for_summary(to_summarize)
    if verbose:
        print(f"[compact-intra] summarizing {len(to_summarize)} exchange(s), ~{len(transcript_text)} chars", file=sys.stderr)
    try:
        summary = summarize_via_bg(transcript_text, bg_base_url, model, prior_summary)
    except Exception as e:
        return group, False, f"intra-group summarize failed: {e}"

    summary_msg = {"role": "assistant", "content": INTRA_SUMMARY_MARKER + summary}
    new_group: list[dict[str, Any]] = [user_msg, summary_msg]
    for ex in keep_tail:
        new_group.extend(ex)
    return new_group, True, f"compacted {len(to_summarize)} exchange(s) into {len(summary)} chars"


def compact_messages(
    messages: list[dict[str, Any]],
    bg_base_url: str,
    model: str,
    keep_last_groups: int = COMPACT_KEEP_LAST_GROUPS,
    keep_last_exchanges: int = COMPACT_KEEP_LAST_EXCHANGES,
    verbose: bool = False,
    open_outputs: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], bool, str]:
    head, groups = split_groups(messages)
    if not head:
        return messages, False, "no system message; refusing to compact"

    notes: list[str] = []
    did_any = False

    head, groups, did_inter, note_inter = _compact_inter_group(head, groups, bg_base_url, model, keep_last_groups, verbose, open_outputs)
    if did_inter:
        notes.append(f"inter-group: {note_inter}")
        did_any = True

    if groups:
        new_active, did_intra, note_intra = _compact_intra_group(groups[-1], bg_base_url, model, keep_last_exchanges, verbose)
        if did_intra:
            groups[-1] = new_active
            notes.append(f"intra-group: {note_intra}")
            did_any = True

    if not did_any:
        suffix_parts = [note_inter]
        if groups:
            _, ex_in_active = split_group_into_exchanges(groups[-1])
            suffix_parts.append(f"active group has {len(ex_in_active)} exchange(s); keep_last_exchanges={keep_last_exchanges}")
        return messages, False, "nothing to compact (" + "; ".join(p for p in suffix_parts if p) + ")"

    new_messages: list[dict[str, Any]] = list(head)
    for g in groups:
        new_messages.extend(g)
    return new_messages, True, "; ".join(notes)


EMPTY_RETRY_NUDGE = (
    "Your previous turn produced reasoning but no tool call and no final answer. "
    "Either call a tool to make progress on the task, or write a brief final response now."
)

EMPTY_RETRY_NUDGE_GENERIC = (
    "Your previous turn was empty: no tool call and no final answer. "
    "Continue the original task now. If you need more information, call a tool. "
    "If the task is complete, write a concise final answer."
)

EMPTY_AFTER_COMPACT_NUDGE = (
    "The conversation was just compacted. The pinned original request and any open outputs "
    "in the system message are authoritative. Continue from that state now: call exactly one "
    "tool to make concrete progress, or write a concise final answer if no tool is needed."
)


def _make_request(
    client: OpenAI,
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    thinking: bool,
    stats: dict[str, Any],
) -> Any:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        top_p=top_p,
        max_tokens=MAX_COMPLETION_TOKENS,
        extra_body={"top_k": 20, "n_keep": N_KEEP_TOKENS, "chat_template_kwargs": {"enable_thinking": thinking}},
    )
    usage = getattr(resp, "usage", None)
    if usage is not None:
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        if pt is not None:
            stats["last_prompt_tokens"] = int(pt)
        if ct is not None:
            stats["last_completion_tokens"] = int(ct)
    return resp.choices[0].message


def run_loop(
    client: OpenAI,
    messages: list[dict[str, Any]],
    cwd: Path,
    max_turns: int,
    verbose: bool,
    model: str,
    temperature: float,
    top_p: float,
    thinking: bool,
    show_thinking: bool,
    stats: dict[str, Any] | None = None,
    bg_base_url: str = DEFAULT_BG_BASE_URL,
) -> tuple[int, str]:
    """Run one agent task until final answer or max turns. Mutates messages. max_turns=0 means unlimited."""
    if stats is None:
        stats = {}
    turn = 0
    just_compacted = False
    while True:
        turn += 1
        if max_turns and turn > max_turns:
            return 2, f"[blocked] max turns reached ({max_turns})"
        last_pt = int(stats.get("last_prompt_tokens", 0) or 0)
        threshold = int(stats.get("compact_threshold", COMPACT_THRESHOLD_FALLBACK) or COMPACT_THRESHOLD_FALLBACK)
        if last_pt > threshold:
            new_messages, did, note = compact_messages(messages, bg_base_url, model, COMPACT_KEEP_LAST_GROUPS, COMPACT_KEEP_LAST_EXCHANGES, verbose, open_outputs=stats.get("open_outputs"))
            if did:
                messages.clear()
                messages.extend(new_messages)
                stats["last_prompt_tokens"] = 0
                stats["compactions"] = int(stats.get("compactions", 0)) + 1
                if "inter-group:" in note:
                    stats["inter_compactions"] = int(stats.get("inter_compactions", 0)) + 1
                if "intra-group:" in note:
                    stats["intra_compactions"] = int(stats.get("intra_compactions", 0)) + 1
                stats["last_compact_at_turn"] = turn
                just_compacted = True
                print(f"[compact] {note}", file=sys.stderr)
            elif verbose:
                print(f"[compact] skipped: {note}", file=sys.stderr)
        if verbose:
            print(f"\n=== TURN {turn} ===", file=sys.stderr)
        tools = tools_for(os.environ.get("QWEN_AGENT_DISABLE_SUBAGENT") != "1")
        msg = _make_request(client, messages, model, tools, temperature, top_p, thinking, stats)

        # Empty-response watchdog: Qwen sometimes returns content="" with no
        # tool_calls, especially after compaction or after emitting only a
        # thinking block. That is not a useful final answer. Retry a few times
        # with an explicit nudge, thinking disabled, and lower temperature. Do
        # not persist the empty assistant turn: it is non-information and can
        # poison the next prompt.
        empty_attempts = 0
        while not msg.tool_calls and not (msg.content or "").strip() and empty_attempts < 3:
            empty_attempts += 1
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                why = "after thinking"
                nudge = EMPTY_RETRY_NUDGE
            elif just_compacted:
                why = "after compaction"
                nudge = EMPTY_AFTER_COMPACT_NUDGE
            else:
                why = "with no reasoning"
                nudge = EMPTY_RETRY_NUDGE_GENERIC
            print(f"[watchdog] empty response {why}; retry {empty_attempts}/3 with thinking off", file=sys.stderr)
            messages.append({"role": "user", "content": nudge})
            stats["empty_retries"] = int(stats.get("empty_retries", 0)) + 1
            msg = _make_request(client, messages, model, tools, min(temperature, 0.2), min(top_p, 0.8), False, stats)
        just_compacted = False

        messages.append(message_to_dict(msg))

        reasoning = getattr(msg, "reasoning_content", None)
        if show_thinking and reasoning:
            print("\n[thinking]\n" + reasoning.strip() + "\n[/thinking]", file=sys.stderr)

        if verbose and msg.tool_calls and (msg.content or "").strip():
            print((msg.content or "").strip(), file=sys.stderr)

        if not msg.tool_calls:
            final = (msg.content or "").strip()
            if not final:
                return 3, "[empty response — model produced no tool call and no final answer; try /compact or rephrase]"
            return 0, final

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(cwd, name, args, thinking)
            if name == "write_file" and isinstance(result, dict) and result.get("ok"):
                outputs = stats.setdefault("open_outputs", {})
                outputs[str(result.get("path"))] = int(result.get("chars", 0))
            if verbose:
                print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)})", file=sys.stderr)
                print(f"[result] {truncate(json.dumps(result, ensure_ascii=False), 2000)}", file=sys.stderr)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

            # True background semantics: after launching a background subagent,
            # immediately return control to the user instead of letting the main
            # model spend more turns monitoring/polling it.
            if name == "start_background_subagent":
                if result.get("ok"):
                    return 0, (
                        f"Started background subagent `{result.get('job_id')}`. "
                        f"Use `/jobs` or ask me to check `{result.get('job_id')}` when you want the result."
                    )
                return 1, f"Failed to start background subagent: {result}"


def initial_messages(cwd: Path) -> list[dict[str, Any]]:
    sys_text = SYSTEM_PROMPT + f"\nWorking directory: {cwd}"
    if os.environ.get("LOCAL_AGENT_DECOMPOSE") == "1":
        sys_text += "\n\n" + DECOMPOSITION_CLAUSE
    return [{"role": "system", "content": sys_text}]


def query_server_n_ctx(base_url: str) -> int | None:
    import urllib.request
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(root + "/props", timeout=3) as r:
            data = json.load(r)
    except Exception:
        return None
    n_ctx = data.get("default_generation_settings", {}).get("n_ctx")
    if isinstance(n_ctx, int) and n_ctx > 0:
        return n_ctx
    return None


def compute_compact_threshold(base_url: str) -> tuple[int, str]:
    """Return (threshold, source) where source describes how it was derived."""
    if COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE is not None:
        return COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE, f"env LOCAL_AGENT_COMPACT_THRESHOLD={COMPACT_PROMPT_TOKEN_THRESHOLD_OVERRIDE}"
    n_ctx = query_server_n_ctx(base_url)
    if n_ctx:
        return int(n_ctx * COMPACT_THRESHOLD_RATIO), f"{COMPACT_THRESHOLD_RATIO:g} * server n_ctx={n_ctx}"
    return COMPACT_THRESHOLD_FALLBACK, f"fallback (server /props unreachable at {base_url})"


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough char-based estimator for prompt size. Used when usage data is unavailable
    (e.g. immediately after resuming a transcript). Counts content, reasoning_content
    if present, and tool_call argument strings. ~3 chars/token is conservative for
    Qwen on mixed text/code; better to over-trigger compaction than to hang."""
    chars = 0
    for m in messages:
        chars += len(str(m.get("content", "") or ""))
        chars += len(str(m.get("reasoning_content", "") or ""))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            chars += len(str(fn.get("arguments", "") or ""))
            chars += len(str(fn.get("name", "") or ""))
    return chars // 3


def strip_reasoning_content(messages: list[dict[str, Any]]) -> int:
    """Remove reasoning_content from loaded messages. Returns count of fields dropped."""
    n = 0
    for m in messages:
        if "reasoning_content" in m:
            del m["reasoning_content"]
            n += 1
    return n


def directory_policy_text(cwd: Path) -> str:
    roots = writable_roots(cwd)
    writable_lines = "\n".join(f"- {r}" for r in roots)
    return f"""# Directory access policy

## Writable directories

{writable_lines}

The agent can write paths that resolve under any of the writable roots above. Relative paths resolve under cwd. Absolute paths are accepted only if they resolve inside one of these roots.

## Readable directories

The harness currently permits read/list attempts for any local path accessible to the `david` OS user. Practically useful read roots include:

- ~/work/local-agent-py — this harness project
- ~/work/la-test — default `la.sh` read/write sandbox
- ~/work/construo-spoke — Construo spoke project (also writable when granted via --write-dir)

Use care: read access is intentionally broad for testing, but write access is sandboxed.
"""


def run_agent(task: str, cwd: Path, max_turns: int, verbose: bool, base_url: str, model: str, temperature: float, top_p: float, thinking: bool, show_thinking: bool, bg_base_url: str = DEFAULT_BG_BASE_URL) -> int:
    client = OpenAI(base_url=base_url, api_key="local-not-needed")
    messages = initial_messages(cwd)
    messages.append({"role": "user", "content": task})
    transcript_path = new_transcript_path(cwd)
    save_transcript(transcript_path, cwd, messages)
    if verbose:
        print(f"transcript: {transcript_path}", file=sys.stderr)
    threshold, source = compute_compact_threshold(base_url)
    stats: dict[str, Any] = {"compact_threshold": threshold, "compact_threshold_source": source}
    code, final = run_loop(client, messages, cwd, max_turns, verbose, model, temperature, top_p, thinking, show_thinking, stats, bg_base_url)
    save_transcript(transcript_path, cwd, messages)
    print(final)
    return code


def list_background_jobs(cwd: Path) -> list[dict[str, Any]]:
    d = jobs_dir(cwd)
    jobs = []
    for meta_path in sorted(d.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status = "finished" if Path(meta["exitcode"]).exists() else "running"
            jobs.append({"job_id": meta.get("job_id"), "status": status, "pid": meta.get("pid"), "task": meta.get("task"), "started": meta.get("started")})
        except Exception as e:
            jobs.append({"job_id": str(meta_path.parent.name), "status": "error", "error": str(e)})
    return jobs


def tool_list_background_jobs(cwd: Path) -> dict[str, Any]:
    jobs = list_background_jobs(cwd)
    return {"ok": True, "jobs": jobs, "count": len(jobs)}


def background_jobs_text(cwd: Path) -> str:
    rows = []
    for job in list_background_jobs(cwd):
        if job.get("status") == "error":
            rows.append(f"- {job.get('job_id')} [error] {job.get('error')}")
        else:
            rows.append(f"- {job.get('job_id')} [{job.get('status')}] pid={job.get('pid')} task={job.get('task')}")
    return "Background jobs:\n" + ("\n".join(rows) if rows else "(none)")


def run_repl(cwd: Path, max_turns: int, verbose: bool, base_url: str, model: str, temperature: float, top_p: float, thinking: bool, show_thinking: bool, clear_jobs_on_start: bool = False, bg_base_url: str = DEFAULT_BG_BASE_URL) -> int:
    client = OpenAI(base_url=base_url, api_key="local-not-needed")
    threshold, threshold_src = compute_compact_threshold(base_url)
    stats: dict[str, Any] = {"compact_threshold": threshold, "compact_threshold_source": threshold_src}
    print(f"compact threshold: {threshold} prompt-tokens ({threshold_src}); n_keep={N_KEEP_TOKENS}")
    if clear_jobs_on_start:
        cleared = clear_background_jobs(cwd)
        if cleared:
            print(f"cleared {cleared} background job record(s)")
    transcript_path = latest_transcript_path(cwd)
    if transcript_path is not None:
        try:
            messages = load_transcript(transcript_path)
            print(f"resumed transcript: {transcript_path} ({len(messages)} messages)")
            stripped = strip_reasoning_content(messages)
            if stripped:
                print(f"stripped reasoning_content from {stripped} message(s)")
            est_pt = estimate_prompt_tokens(messages)
            stats["last_prompt_tokens"] = est_pt
            if est_pt > threshold:
                print(f"resumed transcript estimated at ~{est_pt} prompt tokens (> threshold {threshold}); pre-compacting...")
                new_messages, did, note = compact_messages(messages, bg_base_url, model, COMPACT_KEEP_LAST_GROUPS, COMPACT_KEEP_LAST_EXCHANGES, verbose=True, open_outputs=stats.get("open_outputs"))
                if did:
                    messages = new_messages
                    stats["compactions"] = int(stats.get("compactions", 0)) + 1
                    if "inter-group:" in note:
                        stats["inter_compactions"] = int(stats.get("inter_compactions", 0)) + 1
                    if "intra-group:" in note:
                        stats["intra_compactions"] = int(stats.get("intra_compactions", 0)) + 1
                    stats["last_prompt_tokens"] = estimate_prompt_tokens(messages)
                    print(f"[compact] {note}")
                else:
                    print(f"[compact] {note}")
        except Exception as e:
            print(f"failed to load {transcript_path}: {e}; starting fresh")
            messages = initial_messages(cwd)
            transcript_path = new_transcript_path(cwd)
    else:
        messages = initial_messages(cwd)
        transcript_path = new_transcript_path(cwd)
        print(f"new transcript: {transcript_path}")
    save_transcript(transcript_path, cwd, messages)
    print("Qwen3.6 agent REPL")
    print(f"cwd: {cwd}")
    print(f"thinking: {thinking} show_thinking: {show_thinking} max_turns_per_task: {max_turns or 'unlimited'}")
    print("Commands: /help, /jobs, /clear-jobs, /reset, /context, /compact, /transcript, /quit")
    while True:
        try:
            line = input("local-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/q", "/quit", "/exit"}:
            return 0
        if line == "/help":
            print("Enter a task or follow-up. /jobs lists background jobs. /clear-jobs clears job records. /capabilities describes tools and filesystem access. /dirs lists read/write roots. /reset clears conversation context and starts a new transcript. /context shows message count, char total, and last prompt-token count. /compact summarizes older turns via the bg server (also runs automatically when prompt tokens exceed the threshold). /transcript prints the current transcript path. /quit exits.")
            continue
        if line == "/jobs":
            print(background_jobs_text(cwd))
            continue
        if line == "/clear-jobs":
            cleared = clear_background_jobs(cwd)
            print(f"cleared {cleared} background job record(s)")
            continue
        if line == "/capabilities":
            print(CAPABILITIES_TEXT)
            continue
        if line == "/dirs":
            print(directory_policy_text(cwd))
            continue
        if line == "/reset":
            messages = initial_messages(cwd)
            transcript_path = new_transcript_path(cwd)
            save_transcript(transcript_path, cwd, messages)
            stats.clear()
            print(f"context reset; new transcript: {transcript_path}")
            continue
        if line == "/context":
            chars = sum(len(str(m.get("content", ""))) for m in messages)
            last_pt = stats.get("last_prompt_tokens", "?")
            comp = stats.get("compactions", 0)
            inter = stats.get("inter_compactions", 0)
            intra = stats.get("intra_compactions", 0)
            last_compact_turn = stats.get("last_compact_at_turn", "-")
            thr = stats.get("compact_threshold", COMPACT_THRESHOLD_FALLBACK)
            src = stats.get("compact_threshold_source", "?")
            empty = stats.get("empty_retries", 0)
            print(
                f"messages={len(messages)} approx_content_chars={chars} "
                f"last_prompt_tokens={last_pt} "
                f"compactions={comp} (inter={inter} intra={intra} last_at_turn={last_compact_turn}) "
                f"empty_retries={empty} threshold={thr} ({src}) "
                f"transcript={transcript_path}"
            )
            continue
        if line == "/compact":
            new_messages, did, note = compact_messages(messages, bg_base_url, model, COMPACT_KEEP_LAST_GROUPS, COMPACT_KEEP_LAST_EXCHANGES, verbose=True, open_outputs=stats.get("open_outputs"))
            if did:
                messages.clear()
                messages.extend(new_messages)
                stats["last_prompt_tokens"] = 0
                stats["compactions"] = int(stats.get("compactions", 0)) + 1
                if "inter-group:" in note:
                    stats["inter_compactions"] = int(stats.get("inter_compactions", 0)) + 1
                if "intra-group:" in note:
                    stats["intra_compactions"] = int(stats.get("intra_compactions", 0)) + 1
                save_transcript(transcript_path, cwd, messages)
                print(f"compact: {note}")
            else:
                print(f"compact: {note}")
            continue
        if line == "/transcript":
            print(str(transcript_path))
            continue
        messages.append({"role": "user", "content": line})
        code, final = run_loop(client, messages, cwd, max_turns, verbose, model, temperature, top_p, thinking, show_thinking, stats, bg_base_url)
        save_transcript(transcript_path, cwd, messages)
        print(final)
        if code != 0:
            print("(use /reset if it got stuck)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Test Qwen3.6 as a small local agent")
    ap.add_argument("task", nargs="*", help="Task prompt for the agent. Omit with --repl.")
    ap.add_argument("--repl", "-i", action="store_true", help="Start an interactive agent REPL")
    ap.add_argument("--cwd", default=str(DEFAULT_CWD), help="Working directory")
    ap.add_argument("--write-dir", action="append", default=[], help="Additional writable root (repeatable). Tilde-expanded and resolved.")
    ap.add_argument("--max-turns", type=int, default=40, help="Maximum agent turns per task. 0 = unlimited.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--bg-base-url", default=DEFAULT_BG_BASE_URL, help="Background-model base URL used for compaction summaries and subagents.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=True, help="Enable Qwen thinking mode (default)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false", help="Disable Qwen thinking mode")
    ap.add_argument("--show-thinking", dest="show_thinking", action="store_true", default=True, help="Print returned reasoning_content/thinking blocks to stderr (default)")
    ap.add_argument("--no-show-thinking", dest="show_thinking", action="store_false", help="Hide returned reasoning_content/thinking blocks")
    ap.add_argument("--capabilities", action="store_true", help="Print harness capabilities and exit")
    ap.add_argument("--dirs", action="store_true", help="Print read/write directory policy and exit")
    ap.add_argument("--clear-jobs", action="store_true", help="Clear background job tracking records and exit")
    ap.add_argument("--clear-jobs-on-start", action="store_true", help="Clear background job tracking records when starting --repl")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cwd = Path(args.cwd).expanduser().resolve()
    cwd.mkdir(parents=True, exist_ok=True)

    env_extras = [s for s in os.environ.get("LOCAL_AGENT_EXTRA_WRITE_DIRS", "").split(":") if s]
    extras: list[Path] = []
    for raw in list(args.write_dir or []) + env_extras:
        r = Path(raw).expanduser().resolve()
        r.mkdir(parents=True, exist_ok=True)
        if r not in extras and r != cwd:
            extras.append(r)
    WRITE_ROOTS[:] = [cwd] + extras
    if args.capabilities:
        print(CAPABILITIES_TEXT)
        return 0
    if args.dirs:
        print(directory_policy_text(cwd))
        return 0
    if args.clear_jobs:
        print(f"cleared {clear_background_jobs(cwd)} background job record(s)")
        return 0
    if args.repl:
        return run_repl(cwd, args.max_turns, args.verbose, args.base_url, args.model, args.temperature, args.top_p, args.thinking, args.show_thinking, args.clear_jobs_on_start, args.bg_base_url)
    if not args.task:
        ap.error("task is required unless --repl is used")
    task = " ".join(args.task)
    return run_agent(task, cwd, args.max_turns, args.verbose, args.base_url, args.model, args.temperature, args.top_p, args.thinking, args.show_thinking, args.bg_base_url)


if __name__ == "__main__":
    raise SystemExit(main())
