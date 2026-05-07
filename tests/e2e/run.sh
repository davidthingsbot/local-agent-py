#!/usr/bin/env bash
# Run the end-to-end suite. Requires the foreground (19434) and background (19435)
# llama-server instances from start-servers.sh to be running.
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

exec python3 -m pytest tests/e2e -v -m e2e -s "$@"
