#!/bin/bash
# Batch version of run_smoke.sh. Submit from the repo root:
#   sbatch scripts/psc/smoke.sbatch
#   sbatch -A <your_allocation> scripts/psc/smoke.sbatch   # if you have more than one allocation
# Output: slurm-har-<jobid>.out here, and logs/psc_<jobid>/
#SBATCH -J har-smoke
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 00:45:00
#SBATCH -o slurm-har-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/psc/run_smoke.sh
