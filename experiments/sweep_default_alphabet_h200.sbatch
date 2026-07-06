#!/bin/bash
# Adjust the GPU directive for your cluster if it does not use gres syntax.
#SBATCH --job-name=alphabet_sweep
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%A_%a.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%A_%a.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/train_immse.py" && -f "${CODE_DIR}/code/scripts/train_immse.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if [[ ! -f "scripts/train_immse.py" ]]; then
  echo "Could not find scripts/train_immse.py under CODE_DIR=${CODE_DIR}" >&2
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

read -r -a CONFIG_NAMES <<< "${CONFIG_NAMES:-experiments/markov_scaling/immse}"
read -r -a ALPHABET_BUNDLES <<< "${ALPHABET_BUNDLES:-8,16 32,64 128,256 512}"

task_id="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${task_id}" ]]; then
  echo "Expected TASK_ID or SLURM_ARRAY_TASK_ID to be set." >&2
  exit 1
fi

config_count=${#CONFIG_NAMES[@]}
if (( task_id < 0 || task_id >= config_count )); then
  echo "TASK_ID=${task_id} is out of range for ${config_count} configs." >&2
  exit 1
fi

config_name="${CONFIG_NAMES[$task_id]}"
run_logs_dir="${RUN_LOGS_DIR:-${CODE_DIR}/results/slurm/alphabet_sweep/${config_name}}"
mkdir -p "${run_logs_dir}"

build_overrides() {
  local alphabet_size="$1"

  CURRENT_OVERRIDES=(
    "config_name=${config_name}"
    "data.alphabet_size=${alphabet_size}"
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

  if [[ -n "${WANDB_GROUP_PREFIX:-}" ]]; then
    CURRENT_OVERRIDES+=("logging.group=${WANDB_GROUP_PREFIX}_${config_name}")
  fi

  CURRENT_OVERRIDES+=("logging.tags=[publish,immse,markov]")

  if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    # EXTRA_OVERRIDES must be a space-delimited list of Hydra overrides.
    read -r -a extra_overrides <<< "${EXTRA_OVERRIDES}"
    CURRENT_OVERRIDES+=("${extra_overrides[@]}")
  fi
}

launch_run() {
  local alphabet_size="$1"
  local threads_per_run="$2"
  local log_path="${run_logs_dir}/alphabet_${alphabet_size}.log"

  build_overrides "${alphabet_size}"

  echo "[alphabet-sweep] start config=${config_name} alphabet_size=${alphabet_size} log=${log_path}"

  if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
    printf '[alphabet-sweep] override=%s\n' "${CURRENT_OVERRIDES[@]}"
    return 0
  fi

  (
    export OMP_NUM_THREADS="${threads_per_run}"
    export MKL_NUM_THREADS="${threads_per_run}"
    conda run -n "${CONDA_ENV:-gexit}" python scripts/train_immse.py "${CURRENT_OVERRIDES[@]}"
  ) >"${log_path}" 2>&1 &
}

echo "[alphabet-sweep] task_id=${task_id}/${config_count}"
echo "[alphabet-sweep] config_name=${config_name}"
echo "[alphabet-sweep] alphabet_bundles=${ALPHABET_BUNDLES[*]}"

for bundle in "${ALPHABET_BUNDLES[@]}"; do
  IFS=',' read -r -a alphabet_sizes <<< "${bundle}"
  bundle_size=${#alphabet_sizes[@]}
  if (( bundle_size == 0 )); then
    continue
  fi

  cpus_per_task="${SLURM_CPUS_PER_TASK:-4}"
  threads_per_run=$(( cpus_per_task / bundle_size ))
  if (( threads_per_run < 1 )); then
    threads_per_run=1
  fi

  echo "[alphabet-sweep] bundle=${bundle} bundle_size=${bundle_size} threads_per_run=${threads_per_run}"

  if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
    for alphabet_size in "${alphabet_sizes[@]}"; do
      launch_run "${alphabet_size}" "${threads_per_run}"
    done
    continue
  fi

  pids=()
  labels=()
  for alphabet_size in "${alphabet_sizes[@]}"; do
    launch_run "${alphabet_size}" "${threads_per_run}"
    pids+=("$!")
    labels+=("${alphabet_size}")
  done

  bundle_failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[alphabet-sweep] failed config=${config_name} alphabet_size=${labels[$idx]}" >&2
      bundle_failed=1
    else
      echo "[alphabet-sweep] finished config=${config_name} alphabet_size=${labels[$idx]}"
    fi
  done

  if (( bundle_failed != 0 )); then
    exit 1
  fi
done

echo "[alphabet-sweep] completed config=${config_name}"
