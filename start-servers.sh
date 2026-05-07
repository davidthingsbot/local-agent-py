#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER="$HOME/work/llama.cpp/build/bin/llama-server"
FG_MODEL="$HOME/models/gguf/qwen3.6/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
BG_MODEL="$HOME/models/gguf/qwen3.6/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf"

FG_PORT=${FG_PORT:-19434}
BG_PORT=${BG_PORT:-19435}
FG_LOG=${FG_LOG:-/tmp/local-agent-fg-qwen36.log}
BG_LOG=${BG_LOG:-/tmp/local-agent-bg-qwen36.log}

# Stop old local-agent llama-server processes by log/model/port markers if present.
pkill -f "--port ${FG_PORT}" 2>/dev/null || true
pkill -f "--port ${BG_PORT}" 2>/dev/null || true
sleep 1

# Foreground: stronger MoE on GPU 0.
CUDA_VISIBLE_DEVICES=0 nohup "$LLAMA_SERVER" \
  -m "$FG_MODEL" \
  --host 127.0.0.1 --port "$FG_PORT" \
  -ngl 99 --ctx-size 131072 -np 1 \
  --flash-attn on -ctk q4_0 -ctv q4_0 --jinja \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --presence-penalty 1.5 \
  > "$FG_LOG" 2>&1 &
FG_PID=$!

# Background: small/fast 27B IQ2 on GPU 1, larger context.
CUDA_VISIBLE_DEVICES=1 nohup "$LLAMA_SERVER" \
  -m "$BG_MODEL" \
  --host 127.0.0.1 --port "$BG_PORT" \
  -ngl 99 --ctx-size 131072 -np 1 \
  --flash-attn on -ctk q8_0 -ctv q8_0 --jinja \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --presence-penalty 1.5 \
  > "$BG_LOG" 2>&1 &
BG_PID=$!

wait_ready() {
  local port="$1" pid="$2" log="$3" name="$4"
  for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q 'ok'; then
      echo "$name ready on port $port pid=$pid log=$log"
      return 0
    fi
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited early" >&2
      tail -120 "$log" >&2 || true
      return 1
    fi
  done
  echo "$name timed out" >&2
  tail -120 "$log" >&2 || true
  return 1
}

wait_ready "$FG_PORT" "$FG_PID" "$FG_LOG" foreground
wait_ready "$BG_PORT" "$BG_PID" "$BG_LOG" background

echo "foreground base: http://127.0.0.1:${FG_PORT}/v1"
echo "background base: http://127.0.0.1:${BG_PORT}/v1"
