# Thermal soak test

Runs the agent under a sustained workload while polling GPU stats (NVIDIA via
`nvidia-smi`, AMD via `rocm-smi --json` if present) and printing a live line
that interleaves agent metrics, GPU stats, and stresser progress.

This is **observational**, not pass/fail. Use it to watch how the system
behaves over a real long run.

## What it does

1. Auto-detects llama-server endpoints by probing `/health` from
   `--base-port` (default 19434) upward — every server `start-servers.sh`
   started on a GPU is picked up automatically.
2. Drives `la.py --repl` against the foreground endpoint with a rolling
   workload (filesystem fanout, shell, hashing) — fresh task every
   `--task-period` seconds.
3. Spins one stresser thread against every other endpoint, posting
   continuous chat completion requests so those GPUs stay hot too.
   Disable with `--no-stress-extras`.
4. Polls GPU stats every `--poll` seconds and prints a single status line
   covering all detected GPUs and per-endpoint stress counters.

Agent subagents are disabled (`QWEN_AGENT_DISABLE_SUBAGENT=1`) so the model
can't delegate the load away.

## Usage

```sh
./tests/thermal/run.sh                          # 10 min, 3s poll, 60s task period
./tests/thermal/run.sh --duration 1800          # 30-minute soak
./tests/thermal/run.sh --poll 1                 # high-frequency polling
./tests/thermal/run.sh --task-period 30         # task every 30s
./tests/thermal/run.sh --no-stress-extras       # FG only, leave extras idle
./tests/thermal/run.sh --base-port 19434        # override starting port
```

## Sample output

```
[t= 12s] nv0 65C util=87% mem=22624MiB pwr=248.3W fan= 64%  \
         nv1 58C util=82% mem=13800MiB pwr=215.7W fan= 45% | \
         agent: turn=3 tools=12(read_file) inter=0 intra=0 watchdog=0 | \
         stress: ep1=4r/0e | peaks: nv0_max=66C nv1_max=58C
```

Per-GPU stats are tagged `nv0`, `nv1`, ... or `amd0`, `amd1`, ...

Agent stderr is captured to `<cwd>/stderr.log` for post-run inspection.

## AMD notes

`rocm-smi --json` key names vary across versions; the parser tries common
ones (`Temperature (Sensor edge) (C)`, `GPU use (%)`, etc.) and falls back
to zero / none if a particular field isn't recognized. Run with `--poll 1`
once after install to verify the numbers look sensible; if they don't, the
field-name list in `query_amd()` may need adjustment for your rocm-smi
release.
