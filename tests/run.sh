#!/usr/bin/env bash
# Convenience: run the full test suite from anywhere in the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -m pytest "$@"
