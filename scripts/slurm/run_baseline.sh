#!/bin/bash
#SBATCH --job-name=baseline_core
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --export=ALL

set -eo pipefail
trap 'rc=$?; echo "[ERR] run_baseline.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

mkdir -p logs outputs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load miniforge3/24.1
eval "$(conda shell.bash hook)" 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-agentrl}"
conda activate "${CONDA_ENV}" || source activate "${CONDA_ENV}"

export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"

python3 agent-rl-core/scripts/train_baseline.py --config agent-rl-core/configs/base.yaml
