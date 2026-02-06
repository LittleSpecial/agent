#!/bin/bash
#SBATCH --job-name=agentrl_setup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --export=ALL

set -eo pipefail
trap 'rc=$?; echo "[ERR] setup_env.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"
export PYTHONNOUSERSITE=1

module purge
module load miniforge3/24.1

CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/agentrl}"
if [ -d "${CONDA_ENV}" ]; then
  source activate "${CONDA_ENV}" 2>/dev/null || true
else
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate "${CONDA_ENV##*/}" 2>/dev/null || true
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [ -z "${PYTHON_BIN}" ]; then
  echo "[ERR] python3 not found in PATH." >&2
  exit 2
fi

echo "=== Environment check ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V
nvidia-smi || true

echo "=== Package check ==="
"${PYTHON_BIN}" - <<'PY'
import importlib
pkgs = [
    "agent_rl_core",
    "c3rl",
    "pirl",
    "vera_rl",
    "yaml",
]
for name in pkgs:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: ERR({type(e).__name__}: {e})")
PY

echo "=== setup check complete ==="
