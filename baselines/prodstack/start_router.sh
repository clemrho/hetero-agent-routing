#!/usr/bin/env bash
# Start the official vLLM Production Stack router (session mode) as the baseline.
# Runs standalone, no Kubernetes needed (--service-discovery static).
#
# One-time install (inside the TPU notebook):
#   git clone https://github.com/vllm-project/production-stack && cd production-stack && pip install -e .
#   vllm-router --help   # check that "session" and --session-key still exist in this version
#
# Note: the separate Rust project `pip install vllm-router` (vllm-project/router)
# has the same command name but is a different router. Don't mix them up.
set -euo pipefail

TPU_URL="${TPU_URL:-http://localhost:8000}"
A100_URL="${A100_URL:-http://localhost:8001}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
PORT="${PS_PORT:-8080}"
LOGIC="${LOGIC:-session}"   # session (baseline) or roundrobin (second comparison)

exec vllm-router --port "$PORT" \
  --service-discovery static \
  --static-backends "${TPU_URL},${A100_URL}" \
  --static-models "${MODEL},${MODEL}" \
  --engine-stats-interval 10 \
  --log-stats \
  --routing-logic "$LOGIC" \
  --session-key x-session-id
