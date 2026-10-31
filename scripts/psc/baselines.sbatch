#!/bin/bash
# Batch version of run_baselines.sh (2 GPUs, one vLLM backend each). From the repo root:
#   sbatch scripts/psc/baselines.sbatch
#   sbatch --gpus=h100-80:2 scripts/psc/baselines.sbatch      # other GPU type (name from: sinfo -o "%P %G")
#   sbatch -A <allocation> scripts/psc/baselines.sbatch
#   CONC="1 4" sbatch scripts/psc/baselines.sbatch             # env vars pass through to the script
# Output: slurm-har-base-<jobid>.out here, and logs/baselines_<jobid>/
#SBATCH -J har-base
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:2
#SBATCH -t 03:00:00
#SBATCH -o slurm-har-base-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/psc/run_baselines.sh
