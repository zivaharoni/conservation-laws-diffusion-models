#!/bin/bash
#SBATCH --job-name=bsc_markov_neurips
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --array=0-4%4
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%A_%a.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%A_%a.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/train_bsc.py" && -f "${CODE_DIR}/code/scripts/train_bsc.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if [[ ! -f "scripts/train_bsc.py" ]]; then
  echo "Could not find scripts/train_bsc.py under CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1 && [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
  set -u
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH after shell initialization." >&2
  exit 1
fi

read -r -a ALPHABET_BUNDLES <<< "${ALPHABET_BUNDLES:-8,16 32,64 128 256 512}"

task_id="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${task_id}" ]]; then
  echo "Expected TASK_ID or SLURM_ARRAY_TASK_ID to be set." >&2
  exit 1
fi

bundle_count=${#ALPHABET_BUNDLES[@]}
if (( task_id < 0 || task_id >= bundle_count )); then
  echo "TASK_ID=${task_id} is out of range for ${bundle_count} bundles." >&2
  exit 1
fi

bundle="${ALPHABET_BUNDLES[$task_id]}"
IFS=',' read -r -a alphabet_sizes <<< "${bundle}"
bundle_size=${#alphabet_sizes[@]}
if (( bundle_size == 0 )); then
  echo "Bundle ${bundle} is empty." >&2
  exit 1
fi

run_logs_dir="${RUN_LOGS_DIR:-${CODE_DIR}/results/slurm/bsc_markov_neurips}"
mkdir -p "${run_logs_dir}"

cpus_per_task="${SLURM_CPUS_PER_TASK:-8}"
threads_per_run=$(( cpus_per_task / bundle_size ))
if (( threads_per_run < 1 )); then
  threads_per_run=1
fi

build_overrides() {
  local alphabet_size="$1"

  CURRENT_OVERRIDES=(
    "config_name=experiments/markov_channel_k128/qsc"
    "data.alphabet_size=${alphabet_size}"
    "logging.group=${WANDB_GROUP:-bsc_markov_neurips}"
    "logging.tags=[neurips,bsc,markov]"
    "logging.run_id=bsc_markov_k${alphabet_size}"
  )

  if [[ -n "${SEED:-}" ]]; then
    CURRENT_OVERRIDES+=("seed=${SEED}")
  fi

  if [[ -n "${LOGGING_ENABLED:-}" ]]; then
    CURRENT_OVERRIDES+=("logging.enabled=${LOGGING_ENABLED}")
  fi

  if [[ -n "${LOGGING_MODE:-}" ]]; then
    CURRENT_OVERRIDES+=("logging.mode=${LOGGING_MODE}")
  fi

  if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    read -r -a extra_overrides <<< "${EXTRA_OVERRIDES}"
    CURRENT_OVERRIDES+=("${extra_overrides[@]}")
  fi
}

launch_run() {
  local alphabet_size="$1"
  local log_path="${run_logs_dir}/alphabet_${alphabet_size}.log"

  build_overrides "${alphabet_size}"

  echo "[bsc-markov-neurips] start alphabet_size=${alphabet_size} log=${log_path}"

  if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
    printf '[bsc-markov-neurips] override=%s\n' "${CURRENT_OVERRIDES[@]}"
    return 0
  fi

  (
    export OMP_NUM_THREADS="${threads_per_run}"
    export MKL_NUM_THREADS="${threads_per_run}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    conda run -n "${CONDA_ENV:-gexit}" python scripts/train_bsc.py "${CURRENT_OVERRIDES[@]}"
  ) >"${log_path}" 2>&1 &
}

echo "[bsc-markov-neurips] task_id=${task_id}/${bundle_count}"
echo "[bsc-markov-neurips] bundle=${bundle}"
echo "[bsc-markov-neurips] bundle_size=${bundle_size}"
echo "[bsc-markov-neurips] threads_per_run=${threads_per_run}"

if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  for alphabet_size in "${alphabet_sizes[@]}"; do
    launch_run "${alphabet_size}"
  done
  exit 0
fi

pids=()
labels=()
for alphabet_size in "${alphabet_sizes[@]}"; do
  launch_run "${alphabet_size}"
  pids+=("$!")
  labels+=("${alphabet_size}")
done

bundle_failed=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[bsc-markov-neurips] failed alphabet_size=${labels[$idx]}" >&2
    bundle_failed=1
  else
    echo "[bsc-markov-neurips] finished alphabet_size=${labels[$idx]}"
  fi
done

if (( bundle_failed != 0 )); then
  exit 1
fi

echo "[bsc-markov-neurips] completed bundle=${bundle}"
