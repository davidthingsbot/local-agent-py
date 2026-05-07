#!/usr/bin/env python3
"""Thermal soak test for the local-agent stack.

Drives la.py --repl through a rolling workload while polling nvidia-smi for
GPU temperature, utilization, memory, and power. Prints a live status line
every poll interval that interleaves agent metrics with GPU stats.

Not a pass/fail test — this is for watching thermals during a sustained run.

Usage:
  tests/thermal/run.sh [--duration SEC] [--poll SEC] [--task-period SEC]

By default: 600s (10 min) total, 3s poll, new task every 60s.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LA_PY = ROOT / "la.py"

# Workload — small, sandboxed, repeatable. Each task forces several tool calls
# but is small enough to complete in seconds, so we just cycle through them.
TASK_TEMPLATES = [
    # filesystem fanout
    "Write 5 files ./notes/n{i:03d}-{tag}.txt each containing 3 lines of "
    "lorem-ipsum text, then list ./notes and report the total file count.",
    # repeated reads
    "List ./notes and read the most recent 3 files individually with "
    "read_file. Tell me which one has the longest content.",
    # shell load
    "Run `seq 1 1000 | wc -l` and tell me the exact number that comes back.",
    # writing + reading
    "Compute the SHA256 of the string 'thermal-{tag}' using `printf '%s' "
    "'thermal-{tag}' | sha256sum`, write the hex digest to ./hash.txt, then "
    "read it back and report it.",
]


_TURN_RE = re.compile(r"=== TURN (\d+) ===")
_COMPACT_INTER = re.compile(r"\[compact-inter\]")
_COMPACT_INTRA = re.compile(r"\[compact-intra\]")
_WATCHDOG = re.compile(r"\[watchdog\]")
_TOOL = re.compile(r"\[tool\] (\w+)\(")


class StderrTail:
    """Tail an agent stderr file in a background thread, accumulating metrics."""

    def __init__(self, path: Path):
        self.path = path
        self.turns = 0
        self.inter = 0
        self.intra = 0
        self.watchdogs = 0
        self.tool_calls = 0
        self.last_tool = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        # ensure the file exists so opening doesn't race the producer
        self.path.touch()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                if (m := _TURN_RE.search(line)):
                    self.turns = max(self.turns, int(m.group(1)))
                if _COMPACT_INTER.search(line):
                    self.inter += 1
                if _COMPACT_INTRA.search(line):
                    self.intra += 1
                if _WATCHDOG.search(line):
                    self.watchdogs += 1
                if (m := _TOOL.search(line)):
                    self.tool_calls += 1
                    self.last_tool = m.group(1)


def query_nvidia_smi() -> list[dict]:
    """Return per-GPU stats. On nvidia-smi failure, return []."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw,fan.speed",
                "--format=csv,noheader,nounits",
            ],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=4,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, temp, util, mem, power, fan = parts
        try:
            rows.append({
                "idx": int(idx),
                "temp_c": int(temp),
                "util_pct": int(util),
                "mem_mib": int(mem),
                "power_w": float(power),
                "fan_pct": (int(fan) if fan and fan != "[N/A]" else None),
            })
        except ValueError:
            continue
    return rows


def fmt_gpu(g: dict) -> str:
    fan = f" fan={g['fan_pct']:>3}%" if g["fan_pct"] is not None else ""
    return (
        f"gpu{g['idx']} {g['temp_c']:>2}C util={g['util_pct']:>3}% "
        f"mem={g['mem_mib']:>5}MiB pwr={g['power_w']:>5.1f}W{fan}"
    )


def fmt_status(elapsed: int, gpus: list[dict], tail: StderrTail, peaks: dict) -> str:
    gpu_part = "  ".join(fmt_gpu(g) for g in gpus) or "[no GPU data]"
    peak_part = (
        f"peaks: gpu0_max={peaks.get(0, 0)}C gpu1_max={peaks.get(1, 0)}C"
    )
    agent_part = (
        f"turn={tail.turns} tools={tail.tool_calls}({tail.last_tool or '-'}) "
        f"inter={tail.inter} intra={tail.intra} watchdog={tail.watchdogs}"
    )
    return f"[t={elapsed:>4}s] {gpu_part} | {agent_part} | {peak_part}"


def task_for(i: int) -> str:
    template = TASK_TEMPLATES[i % len(TASK_TEMPLATES)]
    return template.format(i=i, tag=f"T{i:03d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=600,
                    help="Total run time in seconds (default 600 = 10 min)")
    ap.add_argument("--poll", type=int, default=3,
                    help="Status poll interval in seconds (default 3)")
    ap.add_argument("--task-period", type=int, default=60,
                    help="Seconds between submitting new tasks (default 60)")
    ap.add_argument("--cwd", type=str, default=None,
                    help="Working dir for the agent (default: a fresh tmpdir)")
    args = ap.parse_args()

    # Sandbox cwd
    if args.cwd:
        cwd = Path(args.cwd).expanduser().resolve()
        cwd.mkdir(parents=True, exist_ok=True)
    else:
        cwd = Path(tempfile.mkdtemp(prefix="la-thermal-"))

    # Spawn la.py --repl with stderr to a tail file we can stream-parse
    stderr_file = cwd / "stderr.log"
    stderr_fp = stderr_file.open("w", encoding="utf-8")
    cmd = [
        sys.executable, str(LA_PY), "--repl",
        "--cwd", str(cwd),
        "--max-turns", "20",
        "--no-show-thinking",
        "-v",
    ]
    env = os.environ.copy()
    # Disable subagents so the model can't delegate the workload away
    env["QWEN_AGENT_DISABLE_SUBAGENT"] = "1"

    print(f"thermal soak: duration={args.duration}s poll={args.poll}s "
          f"task_period={args.task_period}s")
    print(f"cwd: {cwd}")
    print(f"stderr log: {stderr_file}")

    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env, text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_fp,
    )

    tail = StderrTail(stderr_file)
    tail.start()

    start = time.time()
    next_task_at = start  # submit one immediately
    next_poll_at = start + args.poll
    end_at = start + args.duration
    task_idx = 0
    peaks: dict = {}
    final_returncode = 0

    try:
        while time.time() < end_at:
            now = time.time()
            elapsed = int(now - start)

            # submit a task on schedule
            if now >= next_task_at:
                task = task_for(task_idx)
                print(f"[t={elapsed:>4}s] >>> task #{task_idx}: {task[:100]}{'...' if len(task) > 100 else ''}")
                try:
                    proc.stdin.write(task + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    print(f"[t={elapsed:>4}s] !!! agent stdin broken: {e}")
                    break
                task_idx += 1
                next_task_at = now + args.task_period

            # poll nvidia-smi
            if now >= next_poll_at:
                gpus = query_nvidia_smi()
                for g in gpus:
                    peaks[g["idx"]] = max(peaks.get(g["idx"], 0), g["temp_c"])
                print(fmt_status(elapsed, gpus, tail, peaks))
                next_poll_at = now + args.poll

            # Did the agent crash?
            ret = proc.poll()
            if ret is not None:
                print(f"[t={elapsed:>4}s] !!! agent exited unexpectedly rc={ret}")
                final_returncode = ret
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[ctrl-c] stopping...")

    finally:
        tail.stop()
        try:
            proc.stdin.write("/quit\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        stderr_fp.close()

    elapsed_total = int(time.time() - start)
    gpus = query_nvidia_smi()
    print()
    print("=" * 70)
    print(f"thermal soak finished after {elapsed_total}s")
    print(f"  tasks submitted: {task_idx}")
    print(f"  agent turns:     {tail.turns}")
    print(f"  tool calls:      {tail.tool_calls}")
    print(f"  inter compactions: {tail.inter}")
    print(f"  intra compactions: {tail.intra}")
    print(f"  watchdog events: {tail.watchdogs}")
    for idx, peak in sorted(peaks.items()):
        print(f"  gpu{idx} peak temp:  {peak}C")
    print(f"  final state: {fmt_status(elapsed_total, gpus, tail, peaks)}")
    print(f"  stderr log: {stderr_file}")
    print(f"  cwd:        {cwd}")
    return final_returncode


if __name__ == "__main__":
    raise SystemExit(main())
