#!/usr/bin/env bash
# Baseline sweep on PSC: two vLLM backends, every router policy (incl. the Production Stack
# router) over the synthetic trace via replay/replay.py. Run on a GPU node, from the repo root:
#   bash scripts/psc/run_baselines.sh
#
# Defaults are for a node with 2 GPUs (e.g. 2x L40S): one backend per GPU, Qwen3-4B, bf16,
# vLLM's V1 engine, max_len 16384. Unlike run_smoke.sh this uses the real trace + replay,
# so the output is the baseline matrix: POLICIES x CONC x REPS.
#
#   N_GPU=1      both backends share one GPU (GPU_MEM 0.4 each)
#   CONC="1 4"   fewer concurrency levels;  REPS=2   two repetitions
#   POLICIES="LB FF CA RR"   (PS is appended automatically if the router is installed)
#   GEN_ARGS="--tool_min 0.05 --tool_max 0.2"   extra flags for traces/gen_synthetic.py
#
# Both backends are the same GPU type here, so FF has no real "fast" device: what this run
# shows is the cache-vs-load trade-off (CA / PS keep cache, LB / RR balance load).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

# Settings for a modern GPU (override any of them by exporting before running)
export MODEL="${MODEL:-Qwen/Qwen3-4B}"
export MAX_LEN="${MAX_LEN:-16384}"
export DTYPE="${DTYPE:-bfloat16}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
source "$HERE/env.sh"

N_GPU="${N_GPU:-2}"
POLICIES="${POLICIES:-LB FF CA RR}"
CONC="${CONC:-1 4 8 16}"
REPS="${REPS:-1}"
PER_COMBO="${PER_COMBO:-4}"
PORT_A="${PORT_A:-8000}"; PORT_B="${PORT_B:-8001}"; PS_PORT="${PS_PORT:-8080}"
for port in "$PORT_A" "$PORT_B" "$PS_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "!! Port $port is already in use. Pick free ones, e.g. PORT_A=8100 PORT_B=8101 PS_PORT=8180" >&2
    exit 1
  fi
done

OUT="${OUT:-$REPO/logs/baselines_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
mkdir -p "$OUT"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || { echo "!! $PY not found. Run scripts/psc/setup.sh on a login node first." >&2; exit 1; }

PIDS=()
cleanup() { for p in ${PIDS[@]+"${PIDS[@]}"}; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "== Output: $OUT"
echo "== Model: $MODEL  dtype=$DTYPE  max_len=$MAX_LEN  V1=$VLLM_USE_V1  gpus=$N_GPU"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null | tee "$OUT/gpu.txt" \
  || echo "   (nvidia-smi not found — are you on a GPU node?)"

# 1. Backends: PORT_A plays "tpu" (slow slot), PORT_B plays "a100" (fast slot)
echo "== [1/5] Starting vLLM backends"
i=0
for port in "$PORT_A" "$PORT_B"; do
  if [[ "$N_GPU" -ge 2 ]]; then
    gpu=$i; mem="${GPU_MEM:-0.85}"
  else
    gpu=0; mem="${GPU_MEM:-0.4}"
  fi
  (export CUDA_VISIBLE_DEVICES=$gpu; vllm_serve "$port" "$mem") > "$OUT/vllm_$port.log" 2>&1 &
  PIDS+=($!)
  wait_ready "$port" "$!" "$OUT/vllm_$port.log"
  i=$((i + 1))
done

# 2. Health + prefix-cache check
echo "== [2/5] Checking backends"
for port in "$PORT_A" "$PORT_B"; do
  if ! "$PY" "$REPO/scripts/check_backend.py" --url "http://127.0.0.1:$port" --model "$MODEL" 2>&1 \
      | tee -a "$OUT/check.txt"; then
    show_errors "$OUT/vllm_$port.log"
    exit 1
  fi
done

# 3. Production Stack router (session mode), if installed
if [[ -x "$ROUTER_VENV/bin/vllm-router" ]]; then
  echo "== [3/5] Starting Production Stack router on :$PS_PORT"
  "$ROUTER_VENV/bin/vllm-router" --port "$PS_PORT" --host 127.0.0.1 \
    --service-discovery static \
    --static-backends "http://127.0.0.1:$PORT_A,http://127.0.0.1:$PORT_B" \
    --static-models "$MODEL,$MODEL" \
    --routing-logic session --session-key x-session-id \
    --engine-stats-interval 10 --log-stats \
    > "$OUT/router_ps.log" 2>&1 &
  PIDS+=($!)
  wait_ready "$PS_PORT" "$!" "$OUT/router_ps.log" 120
  POLICIES="$POLICIES PS"
else
  echo "== [3/5] Production Stack router not installed, skipping PS (WITH_PS=1 bash scripts/psc/setup.sh adds it)"
fi

# 4. Trace + replay sweep
echo "== [4/5] Replaying: policies=[$POLICIES] concurrency=[$CONC] reps=$REPS"
TRACE="$OUT/synthetic.jsonl"
"$PY" "$REPO/traces/gen_synthetic.py" --max_model_len "$MAX_LEN" --per_combo "$PER_COMBO" --out "$TRACE" ${GEN_ARGS:-} \
  | tee "$OUT/trace_summary.txt"
export TPU_URL="http://127.0.0.1:$PORT_A" A100_URL="http://127.0.0.1:$PORT_B" PS_URL="http://127.0.0.1:$PS_PORT"
# shellcheck disable=SC2086
"$PY" "$REPO/replay/replay.py" --trace "$TRACE" --sweep \
  --policies $POLICIES --concurrency_list $CONC --reps "$REPS" \
  --log_dir "$OUT/logs" 2>&1 | tee "$OUT/replay.txt"

# 5. Summary
echo "== [5/5] Summary"
"$PY" "$REPO/analysis/analyze.py" "$OUT"/logs/*/*_r*.jsonl | tee "$OUT/summary.txt"
echo "== Done. Everything is in $OUT"
