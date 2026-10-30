# Shared settings for the PSC Bridges-2 scripts. Sourced, not run.
# Override any variable by exporting it before running a script, e.g.
#   MODEL=Qwen/Qwen3-1.7B bash scripts/psc/run_smoke.sh

# $PROJECT (/ocean/projects/<group>/<user>) has much more space than $HOME (25 GB).
: "${PROJECT:=$HOME}"
export HAR_ROOT="${HAR_ROOT:-$PROJECT/har}"
export VENV="${VENV:-$HAR_ROOT/vllm_env}"                 # vLLM + httpx
export ROUTER_VENV="${ROUTER_VENV:-$HAR_ROOT/router_env}" # Production Stack router (optional)
export HF_HOME="${HF_HOME:-$HAR_ROOT/hf_cache}"           # model weights
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HAR_ROOT/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HAR_ROOT/uv_python}"
export PATH="$HOME/.local/bin:$PATH"                      # where uv gets installed

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export MAX_LEN="${MAX_LEN:-8192}"
export HOST="${HOST:-127.0.0.1}"   # use 0.0.0.0 if another machine must reach this server
# vLLM 0.10.x: use the older V0 engine, the most reliable path on V100.
# (Ignored by vLLM >= 0.11, which only has V1.)
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

# Start vLLM in the foreground (callers put it in the background with &).
# V100 has no bfloat16, so force float16.
# usage: vllm_serve PORT GPU_MEM_FRACTION
vllm_serve() {
  exec "$VENV/bin/vllm" serve "$MODEL" \
    --host "$HOST" --port "$1" \
    --dtype half \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$2" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    ${VLLM_EXTRA_ARGS:-}
}

# Wait until a vLLM server answers /v1/models. Fails early if the process died.
# usage: wait_ready PORT PID LOGFILE [TIMEOUT_S]
wait_ready() {
  local port=$1 pid=$2 log=$3 timeout=${4:-900} t=0
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      # vLLM prints the real cause well above its final traceback, so show error lines first
      echo "!! server on :$port exited. Error lines in $log:" >&2
      grep -nE "Error|Exception|CUDA|capability|not supported|no kernel image" "$log" \
        | grep -v 'File "' | head -20 >&2 || true
      echo "!! Last lines:" >&2
      tail -15 "$log" >&2
      return 1
    fi
    if (( t >= timeout )); then
      echo "!! server on :$port not ready after ${timeout}s. See $log" >&2
      return 1
    fi
    sleep 5; t=$((t + 5))
  done
  echo "   server on :$port ready after ${t}s"
}
