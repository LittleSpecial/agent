#!/bin/bash
# Cluster bootstrap script (run on login node).
# Usage:
#   ENV_NAME=agentrl bash setup_cluster.sh

set -eo pipefail

echo "=== agent-rl cluster setup ==="
module purge
module load miniforge3/24.1

ENV_NAME="${ENV_NAME:-agentrl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Conda env ${ENV_NAME} already exists."
else
  echo "Creating conda env ${ENV_NAME} (python=${PYTHON_VERSION}) ..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${ENV_NAME}" || source activate "${ENV_NAME}"

python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install -e ./agent-rl-core
python3 -m pip install -e ./paper-c3rl
python3 -m pip install -e ./paper-pirl
python3 -m pip install -e ./paper-vera-rl

mkdir -p logs outputs

echo ""
echo "=== setup complete ==="
echo "Env: ${ENV_NAME}"
echo "Python: $(which python3)"
echo "Try: sbatch scripts/slurm/run_c3_main.sh"
