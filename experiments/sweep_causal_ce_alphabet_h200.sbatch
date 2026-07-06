#!/bin/bash
#SBATCH --job-name=causal_ce_alphabet_sweep
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%j.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%j.err

set -euo pipefail

DEFAULT_PROJECT_DIR="/path/to/gexit-diffusion"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/train_causal_ce.py" && -f "${CODE_DIR}/code/scripts/train_causal_ce.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if [[ ! -f "scripts/train_causal_ce.py" ]]; then
  echo "Could not find scripts/train_causal_ce.py under CODE_DIR=${CODE_DIR}" >&2
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

read -r -a ALPHABET_BUNDLES <<< "${ALPHABET_BUNDLES:-8,16 32,64 128,256 512}"

run_logs_dir="${RUN_LOGS_DIR:-${CODE_DIR}/results/slurm/causal_ce_alphabet_sweep}"
mkdir -p "${run_logs_dir}"

base_args=(
  "--family" "${CAUSAL_CE_FAMILY:-markov}"
)

if [[ -n "${SEQ_LEN:-}" ]]; then
  base_args+=("--seq-len" "${SEQ_LEN}")
fi

if [[ -n "${N_STEPS:-}" ]]; then
  base_args+=("--n-steps" "${N_STEPS}")
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
  base_args+=("--batch-size" "${BATCH_SIZE}")
fi

if [[ -n "${LR:-}" ]]; then
  base_args+=("--lr" "${LR}")
fi

if [[ -n "${GRAD_CLIP:-}" ]]; then
  base_args+=("--grad-clip" "${GRAD_CLIP}")
fi

if [[ -n "${N_TRANSFORMER_LAYERS:-}" ]]; then
  base_args+=("--n-transformer-layers" "${N_TRANSFORMER_LAYERS}")
fi

if [[ -n "${SEED:-}" ]]; then
  base_args+=("--seed" "${SEED}")
fi

if [[ -n "${DEVICE:-}" ]]; then
  base_args+=("--device" "${DEVICE}")
fi

if [[ -n "${EVAL_BATCHES:-}" ]]; then
  base_args+=("--eval-batches" "${EVAL_BATCHES}")
fi

if [[ -n "${LOG_EVERY:-}" ]]; then
  base_args+=("--log-every" "${LOG_EVERY}")
fi

if [[ -n "${DIRICHLET_ALPHA:-}" ]]; then
  base_args+=("--dirichlet-alpha" "${DIRICHLET_ALPHA}")
fi

if [[ -n "${INIT_DISTRIBUTION:-}" ]]; then
  base_args+=("--init-distribution" "${INIT_DISTRIBUTION}")
fi

if [[ -n "${RESULTS_ROOT:-}" ]]; then
  base_args+=("--results-root" "${RESULTS_ROOT}")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  base_args+=("${extra_args[@]}")
fi

echo "[causal-ce-sweep] alphabet_bundles=${ALPHABET_BUNDLES[*]}"
printf '[causal-ce-sweep] base_arg=%s\n' "${base_args[@]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

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

  echo "[causal-ce-sweep] bundle=${bundle} mode=parallel bundle_size=${bundle_size} threads_per_run=${threads_per_run}"

  if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
    for alphabet_size in "${alphabet_sizes[@]}"; do
      args=("${base_args[@]}" "--alphabet-size" "${alphabet_size}")
      printf '[causal-ce-sweep] arg=%s\n' "${args[@]}"
    done
    continue
  fi

  pids=()
  labels=()

  for alphabet_size in "${alphabet_sizes[@]}"; do
    log_path="${run_logs_dir}/alphabet_${alphabet_size}.log"
    args=("${base_args[@]}" "--alphabet-size" "${alphabet_size}")

    echo "[causal-ce-sweep] start alphabet_size=${alphabet_size} log=${log_path}"
    (
      export OMP_NUM_THREADS="${threads_per_run}"
      export MKL_NUM_THREADS="${threads_per_run}"
      conda run -n "${CONDA_ENV:-gexit}" python scripts/train_causal_ce.py "${args[@]}"
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
    labels+=("${alphabet_size}")
  done

  bundle_failed=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[causal-ce-sweep] failed alphabet_size=${labels[$idx]} log=${run_logs_dir}/alphabet_${labels[$idx]}.log" >&2
      bundle_failed=1
    else
      echo "[causal-ce-sweep] finished alphabet_size=${labels[$idx]}"
    fi
  done

  if (( bundle_failed != 0 )); then
    exit 1
  fi
done

echo "[causal-ce-sweep] completed"
