#!/bin/bash
#SBATCH -J agent_rl_suite
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

export PYTHONUNBUFFERED=1
set -eo pipefail

mkdir -p logs outputs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load miniforge3/24.1
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
module load cudnn/8.6.0.163_cuda11.x
CONDA_ENV="${CONDA_ENV:-/home/bingxing2/home/scx9krq/.conda/envs/rlvr}"
source activate "${CONDA_ENV}"

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
