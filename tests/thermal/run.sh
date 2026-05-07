#!/usr/bin/env bash
# Thermal soak test — runs the agent under a steady workload while polling
# nvidia-smi. Live output. See thermal.py for usage.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! curl -fsS http://127.0.0.1:19434/health >/dev/null 2>&1; then
  echo "foreground llama-server not responding on :19434 — run start-servers.sh first" >&2
  exit 1
fi
if ! curl -fsS http://127.0.0.1:19435/health >/dev/null 2>&1; then
  echo "background llama-server not responding on :19435 — run start-servers.sh first" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — thermal stats unavailable" >&2
  exit 1
fi

exec python3 tests/thermal/thermal.py "$@"
