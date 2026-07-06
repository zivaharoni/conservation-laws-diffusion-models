#!/bin/bash
#SBATCH --job-name=eval_cifar_suite
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%j.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%j.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/eval_cifar_suite.py" && -f "${CODE_DIR}/code/scripts/eval_cifar_suite.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if [[ ! -f "scripts/eval_cifar_suite.py" ]]; then
  echo "Could not find scripts/eval_cifar_suite.py under CODE_DIR=${CODE_DIR}" >&2
  exit 1
fi

if [[ -z "${RUN_ID:-}" ]]; then
  echo "RUN_ID must be set to a W&B run id or entity/project/run_id." >&2
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

cmd=(
  conda run -n "${CONDA_ENV:-gexit}" python scripts/eval_cifar_suite.py
  "--run" "${RUN_ID}"
  "--sample-count" "${SAMPLE_COUNT:-50000}"
  "--split" "${SPLIT:-test}"
  "--seed" "${SEED:-2026}"
  "--device" "${DEVICE:-auto}"
)

if [[ "${EMA_ONLY:-true}" == "true" ]]; then
  cmd+=("--ema-only")
fi
if [[ -n "${WEIGHT_VARIANTS:-}" ]]; then
  cmd+=("--weight-variants" "${WEIGHT_VARIANTS}")
fi
if [[ -n "${SCHEDULES:-}" ]]; then
  cmd+=("--schedules" "${SCHEDULES}")
fi
if [[ -n "${GRID_STEPS:-}" ]]; then
  cmd+=("--grid-steps" "${GRID_STEPS}")
fi
if [[ -n "${GRID_ROWS:-}" ]]; then
  cmd+=("--grid-rows" "${GRID_ROWS}")
fi
if [[ -n "${GRID_COLS:-}" ]]; then
  cmd+=("--grid-cols" "${GRID_COLS}")
fi
if [[ -n "${TRACE_STEPS:-}" ]]; then
  cmd+=("--trace-steps" "${TRACE_STEPS}")
fi
if [[ -n "${TRACE_COUNT:-}" ]]; then
  cmd+=("--trace-count" "${TRACE_COUNT}")
fi
if [[ -n "${TRACE_FRAMES:-}" ]]; then
  cmd+=("--trace-frames" "${TRACE_FRAMES}")
fi
if [[ "${AUTO_BATCH:-true}" == "true" ]]; then
  cmd+=("--auto-batch")
fi
if [[ "${AUTO_EVAL_BATCH:-true}" == "true" ]]; then
  cmd+=("--auto-eval-batch")
fi
if [[ -n "${AUTO_BATCH_MAX:-}" ]]; then
  cmd+=("--auto-batch-max" "${AUTO_BATCH_MAX}")
fi

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  cmd+=("--entity" "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  cmd+=("--project" "${WANDB_PROJECT}")
fi
if [[ -n "${CACHE_DIR:-}" ]]; then
  cmd+=("--cache-dir" "${CACHE_DIR}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  cmd+=("--output-dir" "${OUTPUT_DIR}")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  cmd+=("--batch-size" "${BATCH_SIZE}")
fi
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  cmd+=("--eval-batch-size" "${EVAL_BATCH_SIZE}")
fi
if [[ -n "${CONSERVATION_POINTS:-}" ]]; then
  cmd+=("--conservation-points" "${CONSERVATION_POINTS}")
fi
if [[ -n "${CONSERVATION_STEPS_PER_BATCH:-}" ]]; then
  cmd+=("--conservation-steps-per-batch" "${CONSERVATION_STEPS_PER_BATCH}")
fi
if [[ -n "${STEP_BUDGETS:-}" ]]; then
  cmd+=("--step-budgets" "${STEP_BUDGETS}")
fi
if [[ -z "${SCHEDULES:-}" ]]; then
  cmd+=("--schedules" "uniform,equal_info")
fi
if [[ -z "${GRID_STEPS:-}" ]]; then
  cmd+=("--grid-steps" "1000" "--grid-rows" "3" "--grid-cols" "10")
fi
if [[ -z "${TRACE_COUNT:-}" ]]; then
  cmd+=("--trace-count" "3" "--trace-frames" "10" "--trace-steps" "1000")
fi
if [[ -z "${STEP_BUDGETS:-}" ]]; then
  cmd+=("--step-budgets" "1000")
fi
if [[ -n "${MAX_BATCHES:-}" ]]; then
  cmd+=("--max-batches" "${MAX_BATCHES}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi
if [[ "${LOG_WANDB:-false}" == "true" ]]; then
  cmd+=("--log-wandb")
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
  cmd+=("--wandb-name" "${WANDB_NAME}")
fi
if [[ -n "${WANDB_GROUP:-}" ]]; then
  cmd+=("--wandb-group" "${WANDB_GROUP}")
fi

printf '[eval-cifar-suite] arg=%s\n' "${cmd[@]}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi
"${cmd[@]}"
