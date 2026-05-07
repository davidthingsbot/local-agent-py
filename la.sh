#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
LA_CWD="${LA_CWD:-$HOME/work/la-test}"
mkdir -p "$LA_CWD"

LA_WRITE_DIRS=(
  "$HOME/work/construo-spoke"
)
LA_WRITE_ARGS=()
for d in "${LA_WRITE_DIRS[@]}"; do
  mkdir -p "$d"
  LA_WRITE_ARGS+=(--write-dir "$d")
done

# These informational modes do not need the model server.
if [[ "${1:-}" == "--capabilities" || "${1:-}" == "--dirs" ]]; then
  exec ./la.py --cwd "$LA_CWD" "${LA_WRITE_ARGS[@]}" "$@"
fi

if ! curl -fsS http://127.0.0.1:19434/health >/dev/null 2>&1; then
  echo "Qwen server is not responding on http://127.0.0.1:19434"
  echo "Start llama-server first, then re-run ./la.sh"
  exit 1
fi

exec ./la.py --repl -v --clear-jobs-on-start --cwd "$LA_CWD" "${LA_WRITE_ARGS[@]}" "$@"
