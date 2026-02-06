#!/bin/bash
#SBATCH --job-name=agentrl_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --export=ALL

export PYTHONUNBUFFERED=1
set -eo pipefail

module purge
module load miniforge3/24.1
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
module load cudnn/8.6.0.163_cuda11.x
CONDA_ENV="${CONDA_ENV:-/home/bingxing2/home/scx9krq/.conda/envs/rlvr}"
source activate "${CONDA_ENV}"

echo "Python: $(which python3)"
python3 -V
nvidia-smi || true
python3 -c "import agent_rl_core; print('agent_rl_core import ok')"
