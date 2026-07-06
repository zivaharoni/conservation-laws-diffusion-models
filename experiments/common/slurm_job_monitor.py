from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE_STATES = {
    "BOOT_FAILING",
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "SUSPENDED",
    "STAGE_OUT",
}
SUCCESS_STATES = {"COMPLETED"}
RESUMABLE_FAILURE_STATES = {
    "BOOT_FAIL",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_slurm_state(raw: str | None) -> str:
    if raw in (None, ""):
        return "UNKNOWN"
    token = str(raw).strip().split(maxsplit=1)[0]
    token = token.rstrip("+")
    return token.upper()


def parse_squeue_output(stdout: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        job_id, state, reason = parts
        rows[job_id] = {
            "job_id": job_id,
            "state": normalize_slurm_state(state),
            "reason": reason.strip(),
        }
    return rows


def parse_sacct_output(stdout: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        job_id_raw, state, exit_code, derived_exit_code, reason = parts
        if "." in job_id_raw:
            continue
        rows[job_id_raw] = {
            "job_id": job_id_raw,
            "state": normalize_slurm_state(state),
            "raw_state": state.strip(),
            "exit_code": exit_code.strip(),
            "derived_exit_code": derived_exit_code.strip(),
            "reason": reason.strip(),
        }
    return rows


def extract_wandb_id_from_names(names: list[str]) -> str | None:
    pattern = re.compile(r"(?:run-)?(?:\d{8}_\d{6}-)?([A-Za-z0-9]+)(?:\.wandb)?$")
    for name in names:
        candidate = Path(name).name
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    return None


def parse_resume_artifacts(stdout: str) -> tuple[str | None, str | None]:
    checkpoint = None
    wandb_id = None
    for line in stdout.splitlines():
        if line.startswith("checkpoint="):
            value = line.partition("=")[2].strip()
            checkpoint = value or None
        elif line.startswith("wandb_id="):
            value = line.partition("=")[2].strip()
            wandb_id = value or None
    return checkpoint, wandb_id


def parse_sbatch_submission(stdout: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match is None:
        raise ValueError(f"Could not parse submitted job id from sbatch output: {stdout!r}")
    return match.group(1)


def _merge_job_with_defaults(defaults: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, value in job.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    defaults = payload.get("defaults", {})
    jobs = [_merge_job_with_defaults(defaults, job) for job in payload.get("jobs", [])]
    return {
        "defaults": defaults,
        "jobs": jobs,
    }


def load_state(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        return {"jobs": {}}
    return json.loads(resolved.read_text())


def save_state(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_event(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass
class CommandRunner:
    ssh_host: str | None = None
    dry_run: bool = False

    def run(self, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        if self.ssh_host:
            argv = ["ssh", self.ssh_host, "bash", "-lc", command]
        else:
            argv = ["bash", "-lc", command]
        if self.dry_run and re.search(r"(^|[;&]\s*|\s)sbatch(\s|$)", command):
            return subprocess.CompletedProcess(argv, 0, stdout="Submitted batch job 0\n", stderr="")
        result = subprocess.run(argv, text=True, capture_output=True)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.returncode}: {command}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result


def _shell_assignments(values: dict[str, Any], *, variables: dict[str, str]) -> str:
    assignments: list[str] = []
    for key, raw_value in values.items():
        value = str(raw_value).format(**variables)
        assignments.append(f"{key}={shlex.quote(value)}")
    return " ".join(assignments)


def query_job_statuses(runner: CommandRunner, job_ids: list[str]) -> dict[str, dict[str, str]]:
    active_rows: dict[str, dict[str, str]] = {}
    accounted_rows: dict[str, dict[str, str]] = {}
    if not job_ids:
        return {}
    job_csv = ",".join(job_ids)
    squeue = runner.run(f"squeue -h -o '%i|%T|%R' -j {shlex.quote(job_csv)}", check=False)
    if squeue.returncode == 0:
        active_rows = parse_squeue_output(squeue.stdout)
    sacct = runner.run(
        f"sacct -X -n -P -j {shlex.quote(job_csv)} -o JobIDRaw,State,ExitCode,DerivedExitCode,Reason",
        check=False,
    )
    if sacct.returncode == 0:
        accounted_rows = parse_sacct_output(sacct.stdout)

    merged: dict[str, dict[str, str]] = {}
    for job_id in job_ids:
        if job_id in active_rows:
            merged[job_id] = active_rows[job_id]
        elif job_id in accounted_rows:
            merged[job_id] = accounted_rows[job_id]
        else:
            merged[job_id] = {"job_id": job_id, "state": "UNKNOWN", "reason": ""}
    return merged


def discover_resume_artifacts(
    runner: CommandRunner,
    *,
    run_dir: str,
    checkpoint_glob: str = "model_step_*.pt",
) -> tuple[str | None, str | None]:
    quoted_run_dir = shlex.quote(run_dir)
    quoted_glob = shlex.quote(checkpoint_glob)
    command = f"""
run_dir={quoted_run_dir}
checkpoint_glob={quoted_glob}
checkpoint_dir="${{run_dir}}/checkpoints"
wandb_dir="${{run_dir}}/wandb"
checkpoint=""
wandb_id=""
if [[ -d "${{checkpoint_dir}}" ]]; then
  checkpoint="$(find "${{checkpoint_dir}}" -maxdepth 1 -type f -name "${{checkpoint_glob}}" | sort | tail -n 1)"
fi
if [[ -L "${{wandb_dir}}/latest-run" ]]; then
  latest_target="$(readlink "${{wandb_dir}}/latest-run")"
  latest_name="$(basename "${{latest_target}}")"
  wandb_id="${{latest_name##*-}}"
fi
if [[ -z "${{wandb_id}}" && -d "${{wandb_dir}}" ]]; then
  latest_run_dir="$(find "${{wandb_dir}}" -maxdepth 1 -type d -name 'run-*' | sort | tail -n 1)"
  if [[ -n "${{latest_run_dir}}" ]]; then
    latest_name="$(basename "${{latest_run_dir}}")"
    wandb_id="${{latest_name##*-}}"
  fi
fi
if [[ -z "${{wandb_id}}" && -d "${{wandb_dir}}" ]]; then
  latest_run_file="$(find "${{wandb_dir}}" -maxdepth 1 -type f -name 'run-*.wandb' | sort | tail -n 1)"
  if [[ -n "${{latest_run_file}}" ]]; then
    latest_name="$(basename "${{latest_run_file}}" .wandb)"
    wandb_id="${{latest_name##*-}}"
  fi
fi
printf 'checkpoint=%s\\n' "${{checkpoint}}"
printf 'wandb_id=%s\\n' "${{wandb_id}}"
""".strip()
    result = runner.run(command)
    return parse_resume_artifacts(result.stdout)


def build_resume_command(job: dict[str, Any], *, checkpoint: str, wandb_id: str | None) -> str:
    run_dir = str(job["run_dir"])
    resume_run_id = str(job.get("resume_run_id") or Path(run_dir).name)
    variables = {
        "resume_ckpt": checkpoint,
        "resume_run_id": resume_run_id,
        "run_dir": run_dir,
        "wandb_resume_id": "" if wandb_id is None else wandb_id,
    }
    if job.get("submit_command"):
        return str(job["submit_command"]).format(**variables)

    sbatch_script = job.get("sbatch_script")
    if not sbatch_script:
        raise ValueError(f"Job {job['name']!r} must define either submit_command or sbatch_script.")

    workdir = str(job.get("submit_workdir") or job.get("project_dir") or ".")
    submit_env = dict(job.get("submit_env", {}))
    assignments = _shell_assignments(submit_env, variables=variables)
    return f"cd {shlex.quote(workdir)} && {assignments} sbatch {shlex.quote(str(sbatch_script))}"


def classify_status(status: dict[str, str]) -> str:
    state = normalize_slurm_state(status.get("state"))
    if state in ACTIVE_STATES:
        return "active"
    if state in SUCCESS_STATES:
        return "success"
    if state in RESUMABLE_FAILURE_STATES:
        return "resumable_failure"
    return "other"


def process_manifest_once(
    *,
    manifest_path: str | Path,
    state_path: str | Path,
    event_log_path: str | Path | None,
    ssh_host: str | None,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    state = load_state(state_path)
    state.setdefault("jobs", {})
    runner = CommandRunner(ssh_host=ssh_host, dry_run=dry_run)

    jobs = manifest["jobs"]
    job_ids = [
        str(state["jobs"].get(job["name"], {}).get("current_job_id") or job["job_id"])
        for job in jobs
    ]
    statuses = query_job_statuses(runner, job_ids)
    cycle_summary: list[dict[str, Any]] = []

    for job in jobs:
        name = str(job["name"])
        job_state = state["jobs"].setdefault(name, {})
        current_job_id = str(job_state.get("current_job_id") or job["job_id"])
        status = statuses[current_job_id]
        current_state = normalize_slurm_state(status.get("state"))
        job_state["current_job_id"] = current_job_id
        job_state["last_state"] = current_state
        job_state["last_reason"] = status.get("reason", "")
        job_state["last_seen_at"] = utc_now_iso()

        classification = classify_status(status)
        if classification != "resumable_failure":
            cycle_summary.append(
                {
                    "name": name,
                    "job_id": current_job_id,
                    "state": current_state,
                    "action": "none",
                }
            )
            continue

        max_restarts = int(job.get("max_restarts", 3))
        resume_count = int(job_state.get("resume_count", 0))
        if resume_count >= max_restarts:
            cycle_summary.append(
                {
                    "name": name,
                    "job_id": current_job_id,
                    "state": current_state,
                    "action": "max_restarts_reached",
                }
            )
            append_event(
                event_log_path,
                {
                    "ts": utc_now_iso(),
                    "job": name,
                    "job_id": current_job_id,
                    "state": current_state,
                    "action": "max_restarts_reached",
                },
            )
            continue

        checkpoint, discovered_wandb_id = discover_resume_artifacts(
            runner,
            run_dir=str(job["run_dir"]),
            checkpoint_glob=str(job.get("checkpoint_glob", "model_step_*.pt")),
        )
        if checkpoint is None:
            cycle_summary.append(
                {
                    "name": name,
                    "job_id": current_job_id,
                    "state": current_state,
                    "action": "resume_blocked_no_checkpoint",
                }
            )
            append_event(
                event_log_path,
                {
                    "ts": utc_now_iso(),
                    "job": name,
                    "job_id": current_job_id,
                    "state": current_state,
                    "action": "resume_blocked_no_checkpoint",
                },
            )
            continue

        wandb_id = str(job_state.get("wandb_resume_id") or job.get("wandb_resume_id") or discovered_wandb_id or "")
        require_wandb_resume = bool(job.get("require_wandb_resume", True))
        if require_wandb_resume and not wandb_id:
            cycle_summary.append(
                {
                    "name": name,
                    "job_id": current_job_id,
                    "state": current_state,
                    "action": "resume_blocked_no_wandb_id",
                }
            )
            append_event(
                event_log_path,
                {
                    "ts": utc_now_iso(),
                    "job": name,
                    "job_id": current_job_id,
                    "state": current_state,
                    "action": "resume_blocked_no_wandb_id",
                    "checkpoint": checkpoint,
                },
            )
            continue

        submit_command = build_resume_command(job, checkpoint=checkpoint, wandb_id=wandb_id or None)
        result = runner.run(submit_command)
        new_job_id = parse_sbatch_submission(result.stdout)
        job_state["current_job_id"] = new_job_id
        job_state["resume_count"] = resume_count + 1
        job_state["last_failed_job_id"] = current_job_id
        job_state["last_checkpoint"] = checkpoint
        job_state["wandb_resume_id"] = wandb_id or None
        job_state["last_resubmitted_at"] = utc_now_iso()
        append_event(
            event_log_path,
            {
                "ts": utc_now_iso(),
                "job": name,
                "job_id": current_job_id,
                "state": current_state,
                "action": "resubmitted",
                "checkpoint": checkpoint,
                "new_job_id": new_job_id,
                "wandb_resume_id": wandb_id or None,
            },
        )
        cycle_summary.append(
            {
                "name": name,
                "job_id": current_job_id,
                "state": current_state,
                "action": "resubmitted",
                "new_job_id": new_job_id,
            }
        )

    state["last_updated_at"] = utc_now_iso()
    save_state(state_path, state)
    return {"jobs": cycle_summary, "state_path": str(state_path)}


def watch_manifest(
    *,
    manifest_path: str | Path,
    state_path: str | Path,
    event_log_path: str | Path | None,
    ssh_host: str | None,
    poll_seconds: int,
    dry_run: bool = False,
) -> None:
    while True:
        summary = process_manifest_once(
            manifest_path=manifest_path,
            state_path=state_path,
            event_log_path=event_log_path,
            ssh_host=ssh_host,
            dry_run=dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        time.sleep(max(1, int(poll_seconds)))
