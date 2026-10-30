#!/usr/bin/env bash
# Start one vLLM server in the foreground (for manual testing on a GPU node).
#   bash scripts/psc/serve.sh                 # port 8000, 85% of GPU memory
#   bash scripts/psc/serve.sh 8001 0.4
#   HOST=0.0.0.0 bash scripts/psc/serve.sh    # reachable from other machines
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
echo "== $MODEL on $HOST:${1:-8000} (max_len=$MAX_LEN, gpu_mem=${2:-0.85})"
vllm_serve "${1:-8000}" "${2:-0.85}"
