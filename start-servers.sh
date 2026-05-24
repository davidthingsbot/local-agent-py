#!/usr/bin/env bash
# Start llama-server for local-agent-py.
#
# Default mode: one dual-slot Qwen server spread across all NVIDIA GPUs.
# It exposes two simultaneous 256K-context slots by using --ctx-size 524288 -np 2.
#
# Optional split mode: set LOCAL_AGENT_SPLIT_SERVERS=1 to start the older layout
# with one foreground server on GPU 0 and one smaller background server on GPU 1.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/work/llama.cpp/build/bin/llama-server}"
FG_MODEL="${FG_MODEL:-$HOME/models/gguf/qwen3.6/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
BG_MODEL="${BG_MODEL:-$HOME/models/gguf/qwen3.6/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf}"
BASE_PORT="${BASE_PORT:-19434}"
CTX_SIZE="${CTX_SIZE:-524288}"
PARALLEL_SLOTS="${LOCAL_AGENT_PARALLEL_SLOTS:-2}"
SPLIT_SERVERS="${LOCAL_AGENT_SPLIT_SERVERS:-0}"

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "llama-server not executable: $LLAMA_SERVER" >&2
  exit 1
fi

mapfile -t NVIDIA_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
if [ "${#NVIDIA_GPUS[@]}" -eq 0 ]; then
  echo "no NVIDIA GPUs detected" >&2
  exit 1
fi

stop_port() {
  local port="$1"
  pkill -f "llama-server .*--port ${port}" 2>/dev/null || true
  pkill -f "llama-server.*--port ${port}" 2>/dev/null || true
}

wait_ready() {
  local port="$1" pid="$2" log="$3" name="$4"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q ok; then
      echo "$name ready on port $port pid=$pid"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited early; tail of $log:" >&2
      tail -120 "$log" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "$name timed out; tail of $log:" >&2
  tail -120 "$log" >&2 || true
  return 1
}

if [ "$SPLIT_SERVERS" = "1" ]; then
  echo "split-server mode: one server per GPU"
  failures=0
  for i in "${!NVIDIA_GPUS[@]}"; do
    port=$((BASE_PORT + i))
    gpu="${NVIDIA_GPUS[$i]}"
    stop_port "$port"
    sleep 0.5
    if [ "$i" -eq 0 ]; then
      model="$FG_MODEL"; ctk=q4_0; ctv=q4_0; role="FG"
    else
      model="$BG_MODEL"; ctk=q8_0; ctv=q8_0; role="BG"
    fi
    log="/tmp/local-agent-srv-gpu${gpu}-port${port}.log"
    echo "[$role] GPU $gpu port=$port ctx=$CTX_SIZE model=$(basename "$model") log=$log"
    env CUDA_VISIBLE_DEVICES="$gpu" nohup "$LLAMA_SERVER" \
      -m "$model" --host 0.0.0.0 --port "$port" \
      -ngl 99 --ctx-size "$CTX_SIZE" -np 1 \
      --flash-attn on -ctk "$ctk" -ctv "$ctv" --jinja \
      --temp 0.7 --top-k 20 --top-p 0.9 --min-p 0.0 --presence-penalty 0.2 \
      > "$log" 2>&1 &
    wait_ready "$port" "$!" "$log" "$role" || failures=$((failures + 1))
  done
  exit "$failures"
fi

# Default: one large server across every NVIDIA GPU.
gpu_csv=$(IFS=,; echo "${NVIDIA_GPUS[*]}")
split_csv=$(python3 - <<'PY' "${#NVIDIA_GPUS[@]}"
import sys
print(','.join(['1'] * int(sys.argv[1])))
PY
)
log="/tmp/local-agent-srv-2slot-256k.log"

echo "single-server mode: GPUs=$gpu_csv port=$BASE_PORT ctx=$CTX_SIZE slots=$PARALLEL_SLOTS model=$(basename "$FG_MODEL") log=$log"
# Stop the foreground port plus the old background port if present.
stop_port "$BASE_PORT"
stop_port "$((BASE_PORT + 1))"
sleep 1

env CUDA_VISIBLE_DEVICES="$gpu_csv" nohup "$LLAMA_SERVER" \
  -m "$FG_MODEL" --host 0.0.0.0 --port "$BASE_PORT" \
  -ngl 99 --ctx-size "$CTX_SIZE" -np "$PARALLEL_SLOTS" \
  --split-mode layer --tensor-split "$split_csv" \
  --flash-attn on -ctk q8_0 -ctv q8_0 --jinja \
  --temp 0.7 --top-k 20 --top-p 0.9 --min-p 0.0 --presence-penalty 0.2 \
  > "$log" 2>&1 &

wait_ready "$BASE_PORT" "$!" "$log" "local-agent Qwen"

echo
echo "endpoint: http://127.0.0.1:${BASE_PORT}/v1"
echo "props:    curl -s http://127.0.0.1:${BASE_PORT}/props | jq '.model_alias, .total_slots, .default_generation_settings.n_ctx'"
