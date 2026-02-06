#!/bin/bash
#SBATCH -J agent_rl_suite
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -eo pipefail

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

echo "=== Job info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
nvidia-smi || true

echo "=== Train C3 (mainline) ==="
python3 paper-c3rl/scripts/train.py --config paper-c3rl/configs/train_c3rl.yaml

echo "=== Train PIRL (secondary line) ==="
python3 paper-pirl/scripts/train.py --config paper-pirl/configs/train_pirl.yaml

echo "=== Train VERA (aux module) ==="
python3 paper-vera-rl/scripts/train.py --config paper-vera-rl/configs/train_vera.yaml

echo "=== Done ==="
