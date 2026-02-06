#!/bin/bash
#SBATCH --job-name=pirl_main
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/pirl_%j.out
#SBATCH --error=logs/pirl_%j.err
#SBATCH --export=ALL

export PYTHONUNBUFFERED=1
set -eo pipefail
trap 'rc=$?; echo "[ERR] run_pirl_main.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

mkdir -p logs outputs
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load miniforge3/24.1

CONDA_ENV="${CONDA_ENV:-/home/bingxing2/home/scx9krq/.conda/envs/rlvr}"
if [ -d "${CONDA_ENV}" ]; then
  source activate "${CONDA_ENV}" 2>/dev/null || true
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV}" 2>/dev/null || true
    fi
  fi
fi

if [ -z "${PYTHON_BIN:-}" ] && [ -x "${CONDA_ENV}/bin/python3" ]; then
  PYTHON_BIN="${CONDA_ENV}/bin/python3"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [ -z "${PYTHON_BIN}" ]; then
  echo "[ERR] python3 not found in PATH." >&2
  exit 2
fi

module load "${GCC_MODULE:-compilers/gcc/9.3.0}"
module load "${CUDA_MODULE:-compilers/cuda/11.6}"
module load "${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
if [ -n "${NCCL_MODULE:-}" ]; then
  module load "${NCCL_MODULE}" 2>/dev/null || true
fi

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"

PIRL_CONFIG="${PIRL_CONFIG:-paper-pirl/configs/train_pirl.yaml}"
MODEL_PATH="${MODEL_PATH:-}"
TRAIN_DATA="${TRAIN_DATA:-datasets/code/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-datasets/code/humaneval_test.jsonl}"

if [ -z "${MODEL_PATH}" ]; then
  echo "[ERR] MODEL_PATH is empty. Set MODEL_PATH to a local HF model path before sbatch." >&2
  exit 2
fi

if [ ! -f "${TRAIN_DATA}" ]; then
  echo "[ERR] TRAIN_DATA not found: ${TRAIN_DATA}" >&2
  exit 2
fi

if [ ! -f "${EVAL_DATA}" ]; then
  echo "[WARN] EVAL_DATA not found: ${EVAL_DATA}. Training will still run if config has fallback." >&2
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-pirl_server_${SLURM_JOB_ID}}"
MAX_STEPS="${MAX_STEPS:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROLLOUTS_PER_PROMPT="${NUM_ROLLOUTS_PER_PROMPT:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH:-8}"
EVAL_TASKS_PER_LEVEL="${EVAL_TASKS_PER_LEVEL:-64}"
REWARD_MODE="${REWARD_MODE:-mixed}"
REWARD_BLEND_ALPHA="${REWARD_BLEND_ALPHA:-0.7}"
FAILURE_REWARD_FLOOR="${FAILURE_REWARD_FLOOR:--0.01}"
INVARIANCE_WEIGHT="${INVARIANCE_WEIGHT:-0.1}"
RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH:-medium}"
CURRICULUM="${CURRICULUM:-1}"
USE_LORA="${USE_LORA:-1}"
LORA_RANK="${LORA_RANK:-64}"
DTYPE="${DTYPE:-bf16}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "Using PIRL config: ${PIRL_CONFIG}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "EVAL_DATA=${EVAL_DATA}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "MAX_STEPS=${MAX_STEPS}, BATCH_SIZE=${BATCH_SIZE}, NUM_ROLLOUTS_PER_PROMPT=${NUM_ROLLOUTS_PER_PROMPT}"

PY_ARGS=(
  paper-pirl/scripts/train.py
  --config "${PIRL_CONFIG}"
  --backend hf
)

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

MODEL_PATH="${MODEL_PATH}" \
TRAIN_DATA="${TRAIN_DATA}" \
EVAL_DATA="${EVAL_DATA}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
MAX_STEPS="${MAX_STEPS}" \
LOG_INTERVAL="${LOG_INTERVAL}" \
EVAL_INTERVAL="${EVAL_INTERVAL}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
LEARNING_RATE="${LEARNING_RATE}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_ROLLOUTS_PER_PROMPT="${NUM_ROLLOUTS_PER_PROMPT}" \
TEMPERATURE="${TEMPERATURE}" \
TOP_P="${TOP_P}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" \
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH}" \
EVAL_TASKS_PER_LEVEL="${EVAL_TASKS_PER_LEVEL}" \
REWARD_MODE="${REWARD_MODE}" \
REWARD_BLEND_ALPHA="${REWARD_BLEND_ALPHA}" \
FAILURE_REWARD_FLOOR="${FAILURE_REWARD_FLOOR}" \
INVARIANCE_WEIGHT="${INVARIANCE_WEIGHT}" \
RANDOMIZATION_STRENGTH="${RANDOMIZATION_STRENGTH}" \
CURRICULUM="${CURRICULUM}" \
USE_LORA="${USE_LORA}" \
LORA_RANK="${LORA_RANK}" \
DTYPE="${DTYPE}" \
GRAD_CLIP="${GRAD_CLIP}" \
REQUIRE_CUDA="${REQUIRE_CUDA}" \
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE}" \
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
"${PYTHON_BIN}" "${PY_ARGS[@]}"

echo "=== PIRL training done ==="
