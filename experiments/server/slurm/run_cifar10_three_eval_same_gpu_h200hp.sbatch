#!/bin/bash
#SBATCH --job-name=cifar10_eval_samegpu_now
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

export WANDB_EXTRA_TAGS="eval,cifar10,rgb8,gpu,same_gpu,immediate,20260426"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[cifar10-same-gpu-eval] host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} job=${SLURM_JOB_ID:-unset}"

latest_checkpoint() {
  local run_dir="$1"
  find "${run_dir}/checkpoints" -maxdepth 1 -type f -name "model_step_*.pt" 2>/dev/null | sort | tail -n 1
}
checkpoint_step() {
  local ckpt="$1"
  basename "${ckpt}" | sed -E "s/model_step_0*([0-9]+)\.pt/\1/"
}
run_eval_from_checkpoint() {
  local label="$1" script="$2" config_name="$3" run_dir="$4" run_id="$5"
  shift 5
  local extra_overrides=("$@")
  local ckpt step
  ckpt=$(latest_checkpoint "${run_dir}")
  if [[ -z "${ckpt}" ]]; then
    echo "[cifar10-same-gpu-eval] SKIP ${label}: no checkpoint under ${run_dir}/checkpoints" >&2
    return 0
  fi
  step=$(checkpoint_step "${ckpt}")
  echo "[cifar10-same-gpu-eval] starting ${label} checkpoint=${ckpt} step=${step} run_id=${run_id}"
  conda run -n "${CONDA_ENV:-gexit}" python "${script}" \
    "config_name=${config_name}" \
    "train.resume_from=${ckpt}" \
    "train.n_steps=${step}" \
    "logging.group=cifar10_rgb8_eval_h200hp_samegpu" \
    "logging.run_id=${run_id}" \
    "logging.resume=never" \
    "logging.wandb_id=null" \
    "${extra_overrides[@]}"
  echo "[cifar10-same-gpu-eval] finished ${label}"
}
run_eval_qsc_available() {
  local run_dir=/path/to/gexit-diffusion/code/results/img/qsc/cifar10_rgb8_unet_qsc/experiments-cifar10-rgb8-unet-qsc__train-batch-size-1536__train-grad-accum-steps-1__eval-batch-size-1536__model-input-skip-true
  local run_id=cifar10_qsc_eval_fwx6vxp7_h200hp_samegpu_now_20260426
  local ckpt
  ckpt=$(latest_checkpoint "${run_dir}")
  if [[ -n "${ckpt}" ]]; then
    local step
    step=$(checkpoint_step "${ckpt}")
    echo "[cifar10-same-gpu-eval] starting qsc checkpoint=${ckpt} step=${step} run_id=${run_id}"
    conda run -n "${CONDA_ENV:-gexit}" python scripts/train_img_qsc.py \
      config_name=experiments/cifar10_rgb8_unet/qsc \
      "train.resume_from=${ckpt}" \
      "train.n_steps=${step}" \
      logging.group=cifar10_rgb8_eval_h200hp_samegpu \
      "logging.run_id=${run_id}" \
      logging.resume=never \
      logging.wandb_id=null \
      train.batch_size=1536 train.grad_accum_steps=1 eval.batch_size=1536 model.input_skip=true
    echo "[cifar10-same-gpu-eval] finished qsc"
    return 0
  fi
  if [[ -f "${run_dir}/model.pt" ]]; then
    echo "[cifar10-same-gpu-eval] starting qsc model_state=${run_dir}/model.pt run_id=${run_id}"
    conda run -n "${CONDA_ENV:-gexit}" python scripts/train_img_qsc.py \
      config_name=experiments/cifar10_rgb8_unet/qsc \
      "train.init_weights_path=${run_dir}/model.pt" \
      train.n_steps=0 \
      logging.group=cifar10_rgb8_eval_h200hp_samegpu \
      "logging.run_id=${run_id}" \
      logging.resume=never \
      logging.wandb_id=null \
      train.batch_size=1536 train.grad_accum_steps=1 eval.batch_size=1536 model.input_skip=true
    echo "[cifar10-same-gpu-eval] finished qsc"
    return 0
  fi
  echo "[cifar10-same-gpu-eval] SKIP qsc fwx6vxp7: no checkpoint or final model.pt is available yet in ${run_dir}" >&2
}

run_eval_from_checkpoint exit \
  scripts/train_img_exit.py \
  experiments/cifar10_rgb8_unet/exit \
  /path/to/gexit-diffusion/code/results/img/exit/cifar10_rgb8_unet_exit/experiments-cifar10-rgb8-unet-exit__train-batch-size-2048__train-grad-accum-steps-1__eval-batch-size-2048__model-input-skip-true \
  cifar10_exit_eval_67t6qcmw_h200hp_samegpu_now_20260426 \
  train.batch_size=2048 train.grad_accum_steps=1 eval.batch_size=2048 model.input_skip=true

run_eval_from_checkpoint immse \
  scripts/train_img_immse.py \
  experiments/cifar10_rgb8_unet/immse \
  /path/to/gexit-diffusion/code/results/img/immse/cifar10_rgb8_unet_immse/experiments-cifar10-rgb8-unet-immse__train-batch-size-64__train-grad-accum-steps-1__eval-batch-size-128 \
  cifar10_immse_eval_mi66fsdq_h200hp_samegpu_now_20260426 \
  train.batch_size=64 train.grad_accum_steps=1 eval.batch_size=128

run_eval_qsc_available

echo "[cifar10-same-gpu-eval] immediate evaluation job completed"
