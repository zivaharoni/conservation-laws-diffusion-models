#!/bin/bash
#SBATCH --job-name=gexit_repro_smoke
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-${DEFAULT_PROJECT_DIR}}}"
if [[ ! -f "${CODE_DIR}/scripts/run_repro_manifest.py" && -f "${CODE_DIR}/code/scripts/run_repro_manifest.py" ]]; then
  CODE_DIR="${CODE_DIR}/code"
fi
cd "${CODE_DIR}"

if [[ ! -f "scripts/run_repro_manifest.py" ]]; then
  echo "Could not find scripts/run_repro_manifest.py under CODE_DIR=${CODE_DIR}" >&2
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

profile="${REPRO_PROFILE:-smoke}"
cmd=(conda run -n "${CONDA_ENV:-gexit}" python scripts/run_repro_manifest.py --profile "${profile}" --run)
groups="${REPRO_GROUPS-smoke}"
if [[ -n "${groups}" ]]; then
  read -r -a group_args <<< "${groups}"
  for group in "${group_args[@]}"; do
    cmd+=(--group "${group}")
  done
fi

if [[ -n "${REPRO_TARGETS:-}" ]]; then
  read -r -a target_args <<< "${REPRO_TARGETS}"
  for target in "${target_args[@]}"; do
    cmd+=(--target "${target}")
  done
fi

printf '[repro-smoke-h200] arg=%s\n' "${cmd[@]}"
if [[ "${PRINT_ONLY:-false}" == "true" ]]; then
  exit 0
fi

"${cmd[@]}"
