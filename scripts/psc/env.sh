# Shared settings for the PSC Bridges-2 scripts. Sourced, not run.
# Override any variable by exporting it before running a script, e.g.
#   MODEL=Qwen/Qwen3-1.7B bash scripts/psc/run_smoke.sh

# $PROJECT (/ocean/projects/<group>/<user>) has much more space than $HOME (25 GB).
: "${PROJECT:=$HOME}"
export HAR_ROOT="${HAR_ROOT:-$PROJECT/har}"

# GPU profile. Each GPU type gets its own venv because they need different vLLM builds.
#   l40s (default): latest vLLM, V1 engine. Prefix caching works.
#   v100: vLLM 0.10.1.1 + V0 engine (newer builds have no V100 kernels). Servers start,
#         but the prefix-cache kernel fails to compile, so cache-hit requests error out.
export GPU_TYPE="${GPU_TYPE:-l40s}"
case "$GPU_TYPE" in
  v100) _venv=vllm_env;        DEFAULT_VLLM_VERSION=0.10.1.1; _v1=0; _dtype=half ;;
  *)    _venv=vllm_env_latest; DEFAULT_VLLM_VERSION="";       _v1=1; _dtype=bfloat16 ;;
esac
export VENV="${VENV:-$HAR_ROOT/$_venv}"                   # vLLM + httpx
export ROUTER_VENV="${ROUTER_VENV:-$HAR_ROOT/router_env}" # Production Stack router (optional)
export HF_HOME="${HF_HOME:-$HAR_ROOT/hf_cache}"           # model weights
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HAR_ROOT/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$HAR_ROOT/uv_python}"
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"            # venv tools (ninja) + uv

export MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
export MAX_LEN="${MAX_LEN:-8192}"
# V100 has no bfloat16; on L40S use bfloat16 like vLLM-TPU, so both devices run the same dtype
export DTYPE="${DTYPE:-$_dtype}"
export HOST="${HOST:-127.0.0.1}"   # use 0.0.0.0 if another machine must reach this server
export VLLM_USE_V1="${VLLM_USE_V1:-$_v1}"   # 0 only matters for vLLM 0.10.x (V100 profile)

# Start vLLM in the foreground (callers put it in the background with &).
# usage: vllm_serve PORT GPU_MEM_FRACTION
vllm_serve() {
  exec "$VENV/bin/vllm" serve "$MODEL" \
    --host "$HOST" --port "$1" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$2" \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    ${VLLM_EXTRA_ARGS:-}
}

# Print the useful part of a vLLM log: error lines first (the real cause sits well
# above the final traceback), then the tail.
# usage: show_errors LOGFILE
show_errors() {
  echo "!! Error lines in $1:" >&2
  grep -nE "Error|error:|errors\.|Exception|CUDA|capability|not supported|no kernel image|out of resource" "$1" \
    | grep -v 'File "' | tail -20 >&2 || true
  echo "!! Last lines:" >&2
  tail -15 "$1" >&2
}

# Wait until a vLLM server answers /v1/models. Fails early if the process died.
# usage: wait_ready PORT PID LOGFILE [TIMEOUT_S]
wait_ready() {
  local port=$1 pid=$2 log=$3 timeout=${4:-900} t=0
  until curl -sf --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "!! server on :$port exited." >&2
      show_errors "$log"
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
