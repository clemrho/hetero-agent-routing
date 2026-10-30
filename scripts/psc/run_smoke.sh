#!/usr/bin/env bash
# End-to-end smoke test on ONE V100. Run on a GPU node, from the repo root:
#   bash scripts/psc/run_smoke.sh
#
# What it does:
#   1. starts two vLLM servers on the same GPU (:8000 plays "tpu", :8001 plays "a100")
#   2. checks each one works and prefix caching is on (scripts/check_backend.py)
#   3. if the Production Stack router is installed (WITH_PS=1 setup), starts it on :8080
#   4. runs router/router.py with synthetic workflows for every policy
#   5. prints the summary (analysis/analyze.py)
# Both backends are the same GPU, so the numbers only prove the pipeline works;
# they say nothing about fast vs slow devices yet.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
source "$HERE/env.sh"

N_WF="${N_WF:-4}"
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
for port in 8000 8001; do
  vllm_serve "$port" "${GPU_MEM:-0.4}" > "$OUT/vllm_$port.log" 2>&1 &
  PIDS+=($!)
  wait_ready "$port" "$!" "$OUT/vllm_$port.log"
done

# 2. Health + prefix cache check
echo "== [2/5] Checking backends"
for port in 8000 8001; do
  "$PY" "$REPO/scripts/check_backend.py" --url "http://127.0.0.1:$port" --model "$MODEL" | tee -a "$OUT/check.txt"
done

# 3. Optional Production Stack router (session mode)
if [[ -x "$ROUTER_VENV/bin/vllm-router" ]]; then
  echo "== [3/5] Starting Production Stack router on :8080"
  "$ROUTER_VENV/bin/vllm-router" --port 8080 --host 127.0.0.1 \
    --service-discovery static \
    --static-backends "http://127.0.0.1:8000,http://127.0.0.1:8001" \
    --static-models "$MODEL,$MODEL" \
    --routing-logic session --session-key x-session-id \
    --engine-stats-interval 10 --log-stats \
    > "$OUT/router_ps.log" 2>&1 &
  PIDS+=($!)
  wait_ready 8080 "$!" "$OUT/router_ps.log" 120
  "$PY" "$REPO/scripts/check_backend.py" --url http://127.0.0.1:8080 --model "$MODEL" --session wf0 \
    | tee -a "$OUT/check.txt"
  POLICIES="$POLICIES PS"
else
  echo "== [3/5] Production Stack router not installed, skipping PS (run WITH_PS=1 setup.sh to add it)"
fi

# 4. Router policies on synthetic workflows (router.py writes logs/ under the cwd)
echo "== [4/5] Running policies: $POLICIES (n_wf=$N_WF)"
export TPU_URL=http://127.0.0.1:8000 A100_URL=http://127.0.0.1:8001 PS_URL=http://127.0.0.1:8080
for pol in $POLICIES; do
  (cd "$OUT" && "$PY" "$REPO/router/router.py" --policy "$pol" --n_wf "$N_WF")
done

# 5. Summary
echo "== [5/5] Summary"
"$PY" "$REPO/analysis/analyze.py" "$OUT"/logs/*.jsonl | tee "$OUT/summary.txt"
echo "== Done. Everything is in $OUT"
