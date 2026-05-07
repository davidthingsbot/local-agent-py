# Thermal soak test

Runs the agent under a sustained, repeating workload while polling
`nvidia-smi` so you can watch GPU temps, utilization, memory, and power
during a real long run. Not a pass/fail test — observational.

## Usage

```sh
./tests/thermal/run.sh                          # default: 10 min, 3s poll
./tests/thermal/run.sh --duration 1800          # 30-minute soak
./tests/thermal/run.sh --poll 1                 # higher-frequency polling
./tests/thermal/run.sh --task-period 30         # submit a new task every 30s
```

## Live output

Each poll prints one line that interleaves agent metrics and GPU stats:

```
[t=  12s] gpu0 65C util=87% mem=22624MiB pwr= 248W fan= 64% \
          gpu1 58C util= 12% mem=13800MiB pwr=  35W fan= 45% | \
          turn=3 tools=12(read_file) inter=0 intra=0 watchdog=0 | \
          peaks: gpu0_max=66C gpu1_max=58C
```

Agent stderr is captured to `<cwd>/stderr.log` for post-run inspection.

## Notes

- The script disables `ask_subagent` / `start_background_subagent` via
  `QWEN_AGENT_DISABLE_SUBAGENT=1` so the model can't delegate the workload
  to an external process and skip the foreground GPU.
- The default workload is small per-task (filesystem fanout, shell, hashing)
  so each task completes in seconds and the *steady-state* load comes from
  submitting a new task every `--task-period` seconds.
