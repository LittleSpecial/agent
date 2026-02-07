#!/bin/bash
#SBATCH --job-name=c3_hf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/c3_hf_%j.out
#SBATCH --error=logs/c3_hf_%j.err
#SBATCH --export=ALL

DEFAULT_NUM_GPUS=1
COMMON_SH=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/c3_hf_common.sh" ]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/c3_hf_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${SCRIPT_DIR}/c3_hf_common.sh" ]; then
    COMMON_SH="${SCRIPT_DIR}/c3_hf_common.sh"
  fi
fi

if [ -z "${COMMON_SH}" ]; then
  echo "[ERR] Cannot locate c3_hf_common.sh." >&2
  exit 2
fi

source "${COMMON_SH}"
