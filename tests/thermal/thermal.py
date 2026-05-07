#!/usr/bin/env python3
"""Thermal soak test for the local-agent stack.

Drives la.py --repl through a rolling workload on the foreground endpoint
while running stress-load threads against every other detected llama-server
endpoint, so all available GPUs stay hot. Polls nvidia-smi (and rocm-smi if
present) for per-GPU temperature, utilization, memory, and power, and prints
a live status line that interleaves agent + GPU metrics.

Not a pass/fail test — observational.

Usage:
  tests/thermal/run.sh [--duration SEC] [--poll SEC] [--task-period SEC]
                       [--no-stress-extras] [--base-port PORT]

By default: 600s (10 min) total, 3s poll, new agent task every 60s, all
endpoints other than FG (port BASE_PORT) are stressed with continuous chat
completion requests.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LA_PY = ROOT / "la.py"

# Workload — small, sandboxed, repeatable. Each task forces several tool calls
# but is small enough to complete in seconds, so we just cycle through them.
TASK_TEMPLATES = [
    "Write 5 files ./notes/n{i:03d}-{tag}.txt each containing 3 lines of "
    "lorem-ipsum text, then list ./notes and report the total file count.",
    "List ./notes and read the most recent 3 files individually with "
    "read_file. Tell me which one has the longest content.",
    "Run `seq 1 1000 | wc -l` and tell me the exact number that comes back.",
    "Compute the SHA256 of the string 'thermal-{tag}' using `printf '%s' "
    "'thermal-{tag}' | sha256sum`, write the hex digest to ./hash.txt, then "
    "read it back and report it.",
]


# ---- agent stderr metrics ----

_TURN_RE = re.compile(r"=== TURN (\d+) ===")
_COMPACT_INTER = re.compile(r"\[compact-inter\]")
_COMPACT_INTRA = re.compile(r"\[compact-intra\]")
_WATCHDOG = re.compile(r"\[watchdog\]")
_TOOL = re.compile(r"\[tool\] (\w+)\(")


class StderrTail:
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


# ---- endpoint detection ----

def detect_endpoints(base_port: int, max_servers: int = 16) -> list[str]:
    """Return base URLs (e.g. http://127.0.0.1:19434/v1) for every responsive
    llama-server in the contiguous port range starting at base_port."""
    found = []
    for i in range(max_servers):
        port = base_port + i
        url = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if b"ok" in r.read():
                    found.append(url + "/v1")
                    continue
        except Exception:
            pass
        if not found:
            # nothing on the very first port we tried; keep scanning
            continue
        # gap after at least one hit — assume contiguous range; stop
        break
    return found


# ---- stresser ----

class Stresser:
    """Continuously POSTs short chat completion requests to one endpoint
    until stopped, to keep that GPU under steady load."""

    def __init__(self, base_url: str, label: str):
        self.base_url = base_url.rstrip("/")
        self.label = label
        self.requests = 0
        self.errors = 0
        self.last_err = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        body = json.dumps({
            "model": "qwen",
            "messages": [
                {"role": "system", "content": "You are a thermal stress test."},
                {"role": "user", "content":
                 "Write a paragraph (about 80 words) describing the heat-dissipation "
                 "design of a high-power desktop GPU."},
            ],
            "max_tokens": 256,
            "temperature": 0.8,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        url = self.base_url + "/chat/completions"
        while not self._stop.is_set():
            req = urllib.request.Request(
                url, data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer local-not-needed",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    r.read()
                self.requests += 1
            except urllib.error.HTTPError as e:
                self.errors += 1
                self.last_err = f"HTTP {e.code}"
                time.sleep(1)
            except Exception as e:  # connection drop, timeout, etc.
                self.errors += 1
                self.last_err = type(e).__name__
                time.sleep(1)


# ---- GPU stats ----

def query_nvidia() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw,fan.speed",
             "--format=csv,noheader,nounits"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=4,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx, temp, util, mem, power, fan = parts
            rows.append({
                "vendor": "nv", "idx": int(idx),
                "temp_c": int(temp), "util_pct": int(util), "mem_mib": int(mem),
                "power_w": float(power),
                "fan_pct": (int(fan) if fan and fan != "[N/A]" else None),
            })
        except ValueError:
            continue
    return rows


def query_amd() -> list[dict]:
    """Best-effort AMD stats via `rocm-smi --json`. Skipped if rocm-smi missing
    or output isn't JSON (older rocm-smi)."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showtemp", "--showuse", "--showmemuse", "--showpower",
             "--showfan", "--json"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=4,
        ).stdout
        data = json.loads(out)
    except Exception:
        return []
    rows = []
    for key, props in data.items():
        if not key.startswith("card"):
            continue
        try:
            idx = int(key.replace("card", ""))
        except ValueError:
            continue
        # rocm-smi key names vary by version; pull common ones with fallbacks.
        def _num(d: dict, *keys, cast=float, default=None):
            for k in keys:
                v = d.get(k)
                if v is None:
                    continue
                if isinstance(v, str):
                    v = v.strip()
                    # strip trailing units
                    v = re.sub(r"\s*(C|°C|MiB|MB|W|%|RPM)?\s*$", "", v)
                try:
                    return cast(v)
                except (ValueError, TypeError):
                    continue
            return default
        rows.append({
            "vendor": "amd", "idx": idx,
            "temp_c": _num(props, "Temperature (Sensor edge) (C)", "Temperature (C)", cast=int, default=0),
            "util_pct": _num(props, "GPU use (%)", cast=int, default=0),
            "mem_mib": _num(props, "GPU Memory Allocated (VRAM%)", "VRAM Total Used Memory (B)", cast=lambda x: int(float(x) // (1024*1024)) if float(x) > 1024*1024 else int(x), default=0),
            "power_w": _num(props, "Average Graphics Package Power (W)", "Average Power (W)", cast=float, default=0.0),
            "fan_pct": _num(props, "Fan speed (%)", cast=int, default=None),
        })
    return rows


def query_gpu_stats() -> list[dict]:
    return query_nvidia() + query_amd()


def fmt_gpu(g: dict) -> str:
    fan = f" fan={g['fan_pct']:>3}%" if g.get("fan_pct") is not None else ""
    return (
        f"{g['vendor']}{g['idx']} {g['temp_c']:>2}C util={g['util_pct']:>3}% "
        f"mem={g['mem_mib']:>5}MiB pwr={g['power_w']:>5.1f}W{fan}"
    )


def fmt_status(elapsed: int, gpus: list[dict], tail: StderrTail,
               stressers: list[Stresser], peaks: dict) -> str:
    gpu_part = "  ".join(fmt_gpu(g) for g in gpus) or "[no GPU data]"
    peak_part = "peaks: " + " ".join(
        f"{v}{i}_max={t}C" for (v, i), t in sorted(peaks.items())
    ) or "peaks: -"
    agent_part = (
        f"agent: turn={tail.turns} tools={tail.tool_calls}({tail.last_tool or '-'}) "
        f"inter={tail.inter} intra={tail.intra} watchdog={tail.watchdogs}"
    )
    if stressers:
        stress_part = "stress: " + " ".join(
            f"{s.label}={s.requests}r/{s.errors}e" for s in stressers
        )
    else:
        stress_part = "stress: -"
    return f"[t={elapsed:>4}s] {gpu_part} | {agent_part} | {stress_part} | {peak_part}"


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
                    help="Seconds between submitting new agent tasks (default 60)")
    ap.add_argument("--cwd", type=str, default=None,
                    help="Working dir for the agent (default: a fresh tmpdir)")
    ap.add_argument("--base-port", type=int, default=19434,
                    help="First llama-server port (FG endpoint). Default 19434.")
    ap.add_argument("--no-stress-extras", action="store_true",
                    help="Don't run parallel stress threads on extra endpoints.")
    args = ap.parse_args()

    # ---- sandbox cwd ----
    if args.cwd:
        cwd = Path(args.cwd).expanduser().resolve()
        cwd.mkdir(parents=True, exist_ok=True)
    else:
        cwd = Path(tempfile.mkdtemp(prefix="la-thermal-"))

    # ---- detect endpoints ----
    endpoints = detect_endpoints(args.base_port)
    if not endpoints:
        print(f"no llama-server endpoints responding from :{args.base_port}+", file=sys.stderr)
        return 1
    fg = endpoints[0]
    extras = endpoints[1:]
    print(f"endpoints detected: {len(endpoints)}")
    for i, ep in enumerate(endpoints):
        role = "FG (agent)" if i == 0 else "BG/extra (stress)"
        print(f"  [{i}] {ep}  [{role}]")

    # ---- spawn the agent on FG ----
    stderr_file = cwd / "stderr.log"
    stderr_fp = stderr_file.open("w", encoding="utf-8")
    cmd = [
        sys.executable, str(LA_PY), "--repl",
        "--cwd", str(cwd),
        "--max-turns", "20",
        "--base-url", fg,
        "--no-show-thinking",
        "-v",
    ]
    if len(endpoints) >= 2:
        cmd += ["--bg-base-url", endpoints[1]]
    env = os.environ.copy()
    env["QWEN_AGENT_DISABLE_SUBAGENT"] = "1"

    print(f"thermal soak: duration={args.duration}s poll={args.poll}s "
          f"task_period={args.task_period}s stress_extras={not args.no_stress_extras}")
    print(f"cwd: {cwd}")
    print(f"stderr log: {stderr_file}")

    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env, text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_fp,
    )
    tail = StderrTail(stderr_file)
    tail.start()

    # ---- spawn stressers ----
    stressers: list[Stresser] = []
    if not args.no_stress_extras:
        for i, ep in enumerate(extras, start=1):
            s = Stresser(ep, label=f"ep{i}")
            s.start()
            stressers.append(s)

    # ---- main loop ----
    start = time.time()
    next_task_at = start
    next_poll_at = start + args.poll
    end_at = start + args.duration
    task_idx = 0
    peaks: dict = {}
    final_returncode = 0

    try:
        while time.time() < end_at:
            now = time.time()
            elapsed = int(now - start)

            if now >= next_task_at:
                task = task_for(task_idx)
                preview = task[:100] + ("..." if len(task) > 100 else "")
                print(f"[t={elapsed:>4}s] >>> task #{task_idx}: {preview}")
                try:
                    proc.stdin.write(task + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError) as e:
                    print(f"[t={elapsed:>4}s] !!! agent stdin broken: {e}")
                    break
                task_idx += 1
                next_task_at = now + args.task_period

            if now >= next_poll_at:
                gpus = query_gpu_stats()
                for g in gpus:
                    key = (g["vendor"], g["idx"])
                    peaks[key] = max(peaks.get(key, 0), g["temp_c"])
                print(fmt_status(elapsed, gpus, tail, stressers, peaks))
                next_poll_at = now + args.poll

            ret = proc.poll()
            if ret is not None:
                print(f"[t={elapsed:>4}s] !!! agent exited unexpectedly rc={ret}")
                final_returncode = ret
                break

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[ctrl-c] stopping...")

    finally:
        for s in stressers:
            s.stop()
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
    gpus = query_gpu_stats()
    print()
    print("=" * 70)
    print(f"thermal soak finished after {elapsed_total}s")
    print(f"  endpoints:        {len(endpoints)} (FG={fg}, extras={len(extras)})")
    print(f"  agent tasks:      {task_idx}")
    print(f"  agent turns:      {tail.turns}")
    print(f"  tool calls:       {tail.tool_calls}")
    print(f"  inter compactions:{tail.inter}")
    print(f"  intra compactions:{tail.intra}")
    print(f"  watchdog events:  {tail.watchdogs}")
    for s in stressers:
        print(f"  stress {s.label} ({s.base_url}): {s.requests} requests, {s.errors} errors"
              + (f" (last: {s.last_err})" if s.last_err else ""))
    for (v, i), peak in sorted(peaks.items()):
        print(f"  {v}{i} peak temp:    {peak}C")
    print(f"  final state: {fmt_status(elapsed_total, gpus, tail, stressers, peaks)}")
    print(f"  stderr log:  {stderr_file}")
    print(f"  cwd:         {cwd}")
    return final_returncode


if __name__ == "__main__":
    raise SystemExit(main())
