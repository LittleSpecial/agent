#!/bin/bash
#SBATCH --job-name=vera_aux
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/vera_%j.out
#SBATCH --error=logs/vera_%j.err
#SBATCH --export=ALL

export PYTHONUNBUFFERED=1
set -eo pipefail
trap 'rc=$?; echo "[ERR] run_vera_aux.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

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

python3 paper-vera-rl/scripts/train.py --config paper-vera-rl/configs/train_vera.yaml
