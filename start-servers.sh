#!/usr/bin/env bash
# Start one llama-server per available GPU.
#
# Detection:
#   - NVIDIA: via nvidia-smi
#   - AMD:    via rocm-smi
# First GPU runs the foreground (FG) model; the rest run the smaller
# background (BG) model. Ports start at $BASE_PORT and increment per GPU.
#
# Override binary or models with env vars:
#   LLAMA_SERVER_NV / LLAMA_SERVER_AMD (default: $LLAMA_SERVER_DEFAULT)
#   FG_MODEL, BG_MODEL, BASE_PORT, CTX_SIZE
#
# A CUDA-built llama-server cannot drive an AMD GPU; on AMD machines you'll
# need a Vulkan or ROCm build. Point LLAMA_SERVER_AMD at it.
set -euo pipefail

LLAMA_SERVER_DEFAULT="${LLAMA_SERVER:-$HOME/work/llama.cpp/build/bin/llama-server}"
LLAMA_SERVER_NV="${LLAMA_SERVER_NV:-$LLAMA_SERVER_DEFAULT}"
LLAMA_SERVER_AMD="${LLAMA_SERVER_AMD:-$LLAMA_SERVER_DEFAULT}"
FG_MODEL="${FG_MODEL:-$HOME/models/gguf/qwen3.6/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
BG_MODEL="${BG_MODEL:-$HOME/models/gguf/qwen3.6/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-IQ2_XXS.gguf}"
BASE_PORT="${BASE_PORT:-19434}"
CTX_SIZE="${CTX_SIZE:-131072}"

# ---- detect GPUs as (vendor:index) ----
gpus=()
if command -v nvidia-smi >/dev/null 2>&1; then
  while read -r idx; do
    idx="$(echo "$idx" | tr -d '[:space:]')"
    [ -n "$idx" ] && gpus+=("nv:$idx")
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
if command -v rocm-smi >/dev/null 2>&1; then
  # Newer rocm-smi: `--showid` prints `GPU[0]: ...` lines.
  while read -r idx; do
    [ -n "$idx" ] && gpus+=("amd:$idx")
  done < <(rocm-smi --showid 2>/dev/null | grep -oE 'GPU\[[0-9]+\]' | grep -oE '[0-9]+' | sort -un)
fi

if [ ${#gpus[@]} -eq 0 ]; then
  echo "no GPUs detected (need nvidia-smi or rocm-smi)" >&2
  exit 1
fi

echo "detected ${#gpus[@]} GPU(s): ${gpus[*]}"

# ---- stop any old servers on the ports we'll use ----
for ((p = BASE_PORT; p < BASE_PORT + ${#gpus[@]}; p++)); do
  pkill -f "--port ${p}" 2>/dev/null || true
done
sleep 1

# ---- start one server per GPU ----
declare -a pids ports logs

start_server() {
  local idx="$1" spec="$2" port="$3" model="$4" ctk="$5" ctv="$6"
  local vendor="${spec%%:*}" gpu="${spec##*:}"
  local log="/tmp/local-agent-srv-${vendor}${gpu}.log"
  local server_bin env_var

  case "$vendor" in
    nv)  server_bin="$LLAMA_SERVER_NV";  env_var="CUDA_VISIBLE_DEVICES=${gpu}" ;;
    amd) server_bin="$LLAMA_SERVER_AMD"; env_var="HIP_VISIBLE_DEVICES=${gpu}"  ;;
    *) echo "unknown vendor: $spec" >&2; return 1 ;;
  esac

  if [ ! -x "$server_bin" ]; then
    echo "[$idx] $vendor:$gpu — server binary not executable: $server_bin" >&2
    return 1
  fi

  echo "[$idx] $vendor:$gpu port=$port model=$(basename "$model") log=$log"
  env $env_var nohup "$server_bin" \
    -m "$model" \
    --host 127.0.0.1 --port "$port" \
    -ngl 99 --ctx-size "$CTX_SIZE" -np 1 \
    --flash-attn on -ctk "$ctk" -ctv "$ctv" --jinja \
    --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 --presence-penalty 1.5 \
    > "$log" 2>&1 &
  pids+=("$!")
  ports+=("$port")
  logs+=("$log")
}

for i in "${!gpus[@]}"; do
  port=$((BASE_PORT + i))
  if [ "$i" -eq 0 ]; then
    model="$FG_MODEL";  ctk=q4_0; ctv=q4_0
  else
    model="$BG_MODEL";  ctk=q8_0; ctv=q8_0
  fi
  start_server "$i" "${gpus[$i]}" "$port" "$model" "$ctk" "$ctv" || true
done

# ---- wait for each server to come up ----
wait_ready() {
  local port="$1" pid="$2" log="$3" name="$4"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q 'ok'; then
      echo "$name ready on port $port pid=$pid"
      return 0
    fi
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited early; tail of $log:" >&2
      tail -100 "$log" >&2 || true
      return 1
    fi
  done
  echo "$name timed out; tail of $log:" >&2
  tail -100 "$log" >&2 || true
  return 1
}

failures=0
for i in "${!pids[@]}"; do
  wait_ready "${ports[$i]}" "${pids[$i]}" "${logs[$i]}" "server-$i (${gpus[$i]})" || failures=$((failures+1))
done

echo
echo "endpoints:"
for i in "${!ports[@]}"; do
  role="extra"
  [ "$i" -eq 0 ] && role="FG (foreground)"
  [ "$i" -eq 1 ] && role="BG (background, used for compaction)"
  echo "  ${gpus[$i]} -> http://127.0.0.1:${ports[$i]}/v1   [$role]"
done

if [ "$failures" -gt 0 ]; then
  echo "$failures server(s) failed to start" >&2
  exit 2
fi
