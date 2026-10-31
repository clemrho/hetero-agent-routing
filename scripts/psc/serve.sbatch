#!/bin/bash
# Keep one vLLM server running on a PSC L40S so a remote router (e.g. the Colab TPU
# notebook) can reach it through an SSH tunnel via the login node. Submit from the repo root:
#   sbatch scripts/psc/serve.sbatch
#   cat slurm-har-serve-<jobid>.out     # wait for the READY line: it prints the node name
#   scancel <jobid>                     # when you're done (it also stops at the time limit)
#
# Listens on 0.0.0.0 so the login node can forward to it; other cluster users could reach
# it too, which is fine for a smoke test.
#SBATCH -J har-serve
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 02:00:00
#SBATCH -o slurm-har-serve-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/psc/env.sh
export HOST=0.0.0.0
PORT="${PORT:-8000}"
OUT="logs/serve_${SLURM_JOB_ID}"
mkdir -p "$OUT"

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "!! Port $PORT is in use on $(hostname). Resubmit with: sbatch --export=ALL,PORT=8100 scripts/psc/serve.sbatch"
  exit 1
fi

echo "== $MODEL ($DTYPE, max_len=$MAX_LEN) on $(hostname):$PORT, log: $OUT/vllm.log"
vllm_serve "$PORT" "${GPU_MEM:-0.85}" > "$OUT/vllm.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
wait_ready "$PORT" "$PID" "$OUT/vllm.log"

NODE=$(hostname -s)
echo "READY node=$NODE port=$PORT"
echo "   In the Colab notebook set: PSC_NODE = \"$NODE\"   PSC_PORT = $PORT"
echo "   Tunnel it opens: ssh -N -L 8001:$NODE:$PORT ${USER}@bridges2.psc.edu"
wait "$PID"
