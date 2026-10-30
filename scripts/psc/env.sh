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
      echo "!! server on :$port exited. Last lines of $log:" >&2
      tail -40 "$log" >&2
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
