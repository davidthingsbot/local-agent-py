"""Helpers for end-to-end tests against real llama-server instances.

These tests spawn la.py as a subprocess with verbose stderr logging, then parse
the stderr stream for compaction/watchdog/turn markers so assertions don't
depend on internal state.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LA_PY = Path(__file__).resolve().parent.parent.parent / "la.py"

DEFAULT_FG = "http://127.0.0.1:19434/v1"
DEFAULT_BG = "http://127.0.0.1:19435/v1"


@dataclass
class AgentResult:
    returncode: int
    stdout: str
    stderr: str
    turns: int = 0
    inter_compactions: int = 0
    intra_compactions: int = 0
    empty_retries: int = 0
    tool_calls: list[str] = field(default_factory=list)


_TURN_RE = re.compile(r"^=== TURN (\d+) ===")
_COMPACT_RE = re.compile(r"\[compact\] (.+)")
_INTER_RE = re.compile(r"\[compact-inter\]")
_INTRA_RE = re.compile(r"\[compact-intra\]")
_WATCHDOG_RE = re.compile(r"\[watchdog\]")
_TOOL_RE = re.compile(r"\[tool\] (\w+)\(")


def parse_stderr(stderr: str) -> dict:
    metrics = {
        "turns": 0,
        "inter_compactions": 0,
        "intra_compactions": 0,
        "empty_retries": 0,
        "tool_calls": [],
        "compact_notes": [],
    }
    for line in stderr.splitlines():
        if (m := _TURN_RE.search(line)):
            metrics["turns"] = max(metrics["turns"], int(m.group(1)))
        if _INTER_RE.search(line):
            metrics["inter_compactions"] += 1
        if _INTRA_RE.search(line):
            metrics["intra_compactions"] += 1
        if _WATCHDOG_RE.search(line):
            metrics["empty_retries"] += 1
        if (m := _TOOL_RE.search(line)):
            metrics["tool_calls"].append(m.group(1))
        if (m := _COMPACT_RE.search(line)):
            metrics["compact_notes"].append(m.group(1))
    return metrics


def run_agent_oneshot(task: str, cwd: Path, *, max_turns: int = 60,
                      base_url: str = DEFAULT_FG, bg_base_url: str = DEFAULT_BG,
                      env_extra: dict | None = None,
                      timeout: int = 600) -> AgentResult:
    """Single-turn agent run via `python la.py <task>`. Returns AgentResult."""
    cmd = [
        sys.executable, str(LA_PY),
        "--cwd", str(cwd),
        "--max-turns", str(max_turns),
        "--base-url", base_url,
        "--bg-base-url", bg_base_url,
        "--no-show-thinking",
        "-v",
        task,
    ]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        cmd, cwd=str(cwd), env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    metrics = parse_stderr(proc.stderr)
    return AgentResult(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        turns=metrics["turns"],
        inter_compactions=metrics["inter_compactions"],
        intra_compactions=metrics["intra_compactions"],
        empty_retries=metrics["empty_retries"],
        tool_calls=metrics["tool_calls"],
    )


def run_repl_with_stdin(stdin_lines: list[str], cwd: Path, *, max_turns: int = 40,
                        base_url: str = DEFAULT_FG, bg_base_url: str = DEFAULT_BG,
                        env_extra: dict | None = None,
                        timeout: int = 1200) -> AgentResult:
    """Drive la.py --repl by piping stdin, terminate with /quit. Useful for
    multi-user-turn tests that exercise inter-group compaction."""
    cmd = [
        sys.executable, str(LA_PY), "--repl",
        "--cwd", str(cwd),
        "--max-turns", str(max_turns),
        "--base-url", base_url,
        "--bg-base-url", bg_base_url,
        "--no-show-thinking",
        "-v",
    ]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    stdin_text = "\n".join(list(stdin_lines) + ["/quit", ""])
    proc = subprocess.run(
        cmd, cwd=str(cwd), env=env, input=stdin_text, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
    )
    metrics = parse_stderr(proc.stderr)
    return AgentResult(
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        turns=metrics["turns"],
        inter_compactions=metrics["inter_compactions"],
        intra_compactions=metrics["intra_compactions"],
        empty_retries=metrics["empty_retries"],
        tool_calls=metrics["tool_calls"],
    )


def make_sandbox_tree(cwd: Path, n_files: int, prefix: str = "f",
                      content_pattern: str = "value-{i}") -> list[Path]:
    """Create n_files small text files under cwd/files/. Returns the file paths."""
    d = cwd / "files"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_files):
        p = d / f"{prefix}{i:03d}.txt"
        p.write_text(content_pattern.format(i=i) + "\n", encoding="utf-8")
        paths.append(p)
    return paths
