#!/bin/bash
#SBATCH --job-name=lm_qsc_h200
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%j.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%j.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/train_lm_qsc.py" && -f "${CODE_DIR}/code/scripts/train_lm_qsc.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if [[ ! -f "scripts/train_lm_qsc.py" ]]; then
  echo "Could not find scripts/train_lm_qsc.py under CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1 && [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH after shell initialization." >&2
  exit 1
fi

export WANDB_EXTRA_TAGS="${WANDB_EXTRA_TAGS:+${WANDB_EXTRA_TAGS},}publish"

config_name="${CONFIG_NAME:-experiments/text8/qsc}"
base_cmd=(
  conda run -n "${CONDA_ENV:-gexit}" python scripts/train_lm_qsc.py
  "config_name=${config_name}"
)
if [[ -n "${BENCHMARK_OVERRIDE:-}" ]]; then
  base_cmd+=("benchmark=${BENCHMARK_OVERRIDE}")
fi
if [[ -n "${DATA_DIR_OVERRIDE:-}" ]]; then
  base_cmd+=("benchmark.data_dir=${DATA_DIR_OVERRIDE}")
fi

run_logs_dir="${RUN_LOGS_DIR:-${CODE_DIR}/results/slurm/lm_qsc}"
mkdir -p "${run_logs_dir}"

if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  printf '[lm-qsc] arg=%s\n' "${base_cmd[@]}"
  exit 0
fi

probe_log="$(mktemp)"
cleanup() {
  rm -f "${probe_log}"
}
trap cleanup EXIT

probe_train_batch() {
  local candidate
  for candidate in 32 16 8 4 2 1; do
    local grad_accum=$((32 / candidate))
    if "${base_cmd[@]}" \
      "logging.enabled=false" \
      "train.n_steps=1" \
      "train.batch_size=${candidate}" \
      "train.grad_accum_steps=${grad_accum}" \
      "train.save_every=1000000" \
      "logging.log_every=1" \
      "eval.batch_size=1" \
      "eval.max_batches=1" \
      "eval.n_t_points=3" \
      >"${probe_log}" 2>&1; then
      echo "${candidate}"
      return 0
    fi
    if ! grep -Eqi "out of memory|cuda error|cudnn_status_not_supported|cublas_status_alloc_failed" "${probe_log}"; then
      cat "${probe_log}" >&2
      return 1
    fi
  done
  return 1
}

probe_eval_batch() {
  local candidate
  for candidate in 64 32 16 8 4 2 1; do
    if "${base_cmd[@]}" \
      "logging.enabled=false" \
      "train.n_steps=1" \
      "train.batch_size=1" \
      "train.grad_accum_steps=32" \
      "train.save_every=1000000" \
      "logging.log_every=1" \
      "eval.batch_size=${candidate}" \
      "eval.max_batches=1" \
      "eval.n_t_points=3" \
      >"${probe_log}" 2>&1; then
      echo "${candidate}"
      return 0
    fi
    if ! grep -Eqi "out of memory|cuda error|cudnn_status_not_supported|cublas_status_alloc_failed" "${probe_log}"; then
      cat "${probe_log}" >&2
      return 1
    fi
  done
  return 1
}

batch_overrides=()
if [[ "${AUTO_TUNE_BATCH:-false}" == "true" ]]; then
  train_batch_size="$(probe_train_batch)"
  eval_batch_size="$(probe_eval_batch)"
  grad_accum_steps=$((32 / train_batch_size))
  batch_overrides=(
    "train.batch_size=${train_batch_size}"
    "train.grad_accum_steps=${grad_accum_steps}"
    "eval.batch_size=${eval_batch_size}"
  )
  printf '[lm-qsc] auto_tuned_train_batch=%s auto_tuned_eval_batch=%s auto_tuned_grad_accum_steps=%s\n' "${train_batch_size}" "${eval_batch_size}" "${grad_accum_steps}"
elif [[ -n "${TRAIN_BATCH_SIZE_OVERRIDE:-}" || -n "${EVAL_BATCH_SIZE_OVERRIDE:-}" || -n "${GRAD_ACCUM_STEPS_OVERRIDE:-}" ]]; then
  if [[ -n "${TRAIN_BATCH_SIZE_OVERRIDE:-}" ]]; then
    batch_overrides+=("train.batch_size=${TRAIN_BATCH_SIZE_OVERRIDE}")
  fi
  if [[ -n "${GRAD_ACCUM_STEPS_OVERRIDE:-}" ]]; then
    batch_overrides+=("train.grad_accum_steps=${GRAD_ACCUM_STEPS_OVERRIDE}")
  fi
  if [[ -n "${EVAL_BATCH_SIZE_OVERRIDE:-}" ]]; then
    batch_overrides+=("eval.batch_size=${EVAL_BATCH_SIZE_OVERRIDE}")
  fi
fi

cmd=("${base_cmd[@]}" "${batch_overrides[@]}")
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
  read -r -a extra_overrides <<< "${EXTRA_OVERRIDES}"
  cmd+=("${extra_overrides[@]}")
fi
if [[ -n "${RESUME_CKPT:-}" ]]; then
  if [[ -z "${RESUME_RUN_ID:-}" ]]; then
    echo "RESUME_RUN_ID must be set when RESUME_CKPT is provided." >&2
    exit 1
  fi
  cmd+=("train.resume_from=${RESUME_CKPT}" "logging.run_id=${RESUME_RUN_ID}")
  if [[ -n "${WANDB_RESUME_ID:-}" ]]; then
    cmd+=("logging.wandb_id=${WANDB_RESUME_ID}" "logging.resume=must")
  fi
fi

printf '[lm-qsc] arg=%s\n' "${cmd[@]}"
"${cmd[@]}"
