#!/bin/bash
#SBATCH --job-name=lm_text8_eval_samegpu_20260427
#SBATCH --account=research-gpu-account
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/path/to/gexit-diffusion/slurm/%x_%j.out
#SBATCH --error=/path/to/gexit-diffusion/slurm/%x_%j.err

set -euo pipefail
cd /path/to/gexit-diffusion/code

if ! command -v conda >/dev/null 2>&1 && [[ -f "${HOME}/.bashrc" ]]; then
  set +u
  source "${HOME}/.bashrc"
  set -u
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH after shell initialization." >&2
  exit 1
fi

export WANDB_EXTRA_TAGS="full_test,gpu,same_gpu,rerun,20260427"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[same-gpu-eval] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} job=${SLURM_JOB_ID:-unset}"

latest_checkpoint() {
  local dir="$1"
  find "${dir}/checkpoints" -maxdepth 1 -type f -name "model_step_*.pt" 2>/dev/null | sort -V | tail -n 1
}

run_eval() {
  local label="$1"
  local run_dir="$2"
  local config="$3"
  local output_json="$4"
  local checkpoint
  checkpoint=$(latest_checkpoint "${run_dir}")
  if [[ -z "${checkpoint}" ]]; then
    echo "[same-gpu-eval] missing checkpoint for ${label} in ${run_dir}" >&2
    exit 1
  fi
  echo "[same-gpu-eval] starting ${label} checkpoint=${checkpoint} config=${config} output=${output_json}"
  conda run -n "${CONDA_ENV:-gexit}" python scripts/eval_lm_checkpoints.py \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --split test \
    --wandb-group poc_text8_eval_h200hp_samegpu \
    --log-shared-conservation-metric \
    --output-json "${output_json}"
  echo "[same-gpu-eval] finished ${label}"
}

run_eval exit \
  /path/to/gexit-diffusion/code/results/lm/exit/text8_exit/experiments-text8-exit__train-batch-size-64__train-grad-accum-steps-8 \
  /path/to/gexit-diffusion/code/results/eval_reruns/text8_exit_86kuxrw3_test_h200hp_samegpu_20260427/config.yaml \
  /path/to/gexit-diffusion/code/results/eval_reruns/text8_exit_86kuxrw3_test_h200hp_samegpu_20260427/full_test_eval_86kuxrw3_samegpu.json

run_eval qsc \
  /path/to/gexit-diffusion/code/results/lm/qsc/text8_qsc/experiments-text8-qsc__train-batch-size-32__train-grad-accum-steps-1__eval-batch-size-64__train-batch-size-64__train-grad-accum-steps-8 \
  /path/to/gexit-diffusion/code/results/eval_reruns/text8_qsc_i0pcvh6m_test_h200hp_samegpu_20260427/config.yaml \
  /path/to/gexit-diffusion/code/results/eval_reruns/text8_qsc_i0pcvh6m_test_h200hp_samegpu_20260427/full_test_eval_i0pcvh6m_samegpu.json

run_eval immse \
  /path/to/gexit-diffusion/code/results/lm/immse/text8_immse/experiments-text8-immse__train-batch-size-64__train-grad-accum-steps-8 \
  /path/to/gexit-diffusion/code/results/eval_reruns/text8_immse_oh5roh7k_test_h200hp_samegpu_20260427/config.yaml \
  /path/to/gexit-diffusion/code/results/eval_reruns/text8_immse_oh5roh7k_test_h200hp_samegpu_20260427/full_test_eval_oh5roh7k_samegpu.json

echo "[same-gpu-eval] all evaluations completed"
