#!/usr/bin/env bash
# End-to-end smoke test on ONE GPU (L40S by default, GPU_TYPE=v100 for V100). Run on a GPU node, from the repo root:
#   bash scripts/psc/run_smoke.sh
#
# What it does:
#   1. starts two vLLM servers on the same GPU (PORT_A plays "tpu", PORT_B plays "a100"; default 8000/8001)
#   2. checks each one works and prefix caching is on (scripts/check_backend.py)
#   3. if the Production Stack router is installed (WITH_PS=1 setup), starts it on PS_PORT (default 8080)
#   4. runs router/router.py with synthetic workflows for every policy
#   5. prints the summary (analysis/analyze.py)
# Both backends are the same GPU, so the numbers only prove the pipeline works;
# they say nothing about fast vs slow devices yet.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"

N_WF="${N_WF:-4}"
# Ports: change them if someone else on a shared node already uses these
PORT_A="${PORT_A:-8000}"; PORT_B="${PORT_B:-8001}"; PS_PORT="${PS_PORT:-8080}"
for port in "$PORT_A" "$PORT_B" "$PS_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then   # something is listening
    echo "!! Port $port is already in use. Pick free ones, e.g. PORT_A=8100 PORT_B=8101 PS_PORT=8180" >&2
    exit 1
  fi
done
POLICIES="${POLICIES:-LB FF CA RR}"
OUT="$REPO/logs/psc_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
PY="$VENV/bin/python"
[[ -x "$PY" ]] || { echo "!! $PY not found. Run scripts/psc/setup.sh on a login node first." >&2; exit 1; }

PIDS=()
cleanup() { for p in ${PIDS[@]+"${PIDS[@]}"}; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "== Output: $OUT"
echo "== Model: $MODEL  max_len=$MAX_LEN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null | tee "$OUT/gpu.txt" \
  || echo "   (nvidia-smi not found — are you on a GPU node?)"

# 1. Two backends, started one after the other so their memory checks don't race
echo "== [1/5] Starting vLLM backends"
for port in "$PORT_A" "$PORT_B"; do
  vllm_serve "$port" "${GPU_MEM:-0.4}" > "$OUT/vllm_$port.log" 2>&1 &
  PIDS+=($!)
  wait_ready "$port" "$!" "$OUT/vllm_$port.log"
done

# 2. Health + prefix cache check
echo "== [2/5] Checking backends"
for port in "$PORT_A" "$PORT_B"; do
  if ! "$PY" "$REPO/scripts/check_backend.py" --url "http://127.0.0.1:$port" --model "$MODEL" 2>&1 \
      | tee -a "$OUT/check.txt"; then
    show_errors "$OUT/vllm_$port.log"
    exit 1
  fi
done

# 3. Optional Production Stack router (session mode)
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
  "$PY" "$REPO/scripts/check_backend.py" --url "http://127.0.0.1:$PS_PORT" --model "$MODEL" --session wf0 \
    | tee -a "$OUT/check.txt"
  POLICIES="$POLICIES PS"
else
  echo "== [3/5] Production Stack router not installed, skipping PS (run WITH_PS=1 setup.sh to add it)"
fi

# 4. Router policies on synthetic workflows (router.py writes logs/ under the cwd)
echo "== [4/5] Running policies: $POLICIES (n_wf=$N_WF)"
export TPU_URL="http://127.0.0.1:$PORT_A" A100_URL="http://127.0.0.1:$PORT_B" PS_URL="http://127.0.0.1:$PS_PORT"
for pol in $POLICIES; do
  (cd "$OUT" && "$PY" "$REPO/router/router.py" --policy "$pol" --n_wf "$N_WF")
done

# 5. Summary
echo "== [5/5] Summary"
"$PY" "$REPO/analysis/analyze.py" "$OUT"/logs/*.jsonl | tee "$OUT/summary.txt"
echo "== Done. Everything is in $OUT"
