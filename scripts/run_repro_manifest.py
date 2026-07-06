from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = APP_ROOT / "experiments" / "repro" / "paper_commands.json"


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_selected(target: dict[str, Any], *, names: set[str], groups: set[str]) -> bool:
    if names and str(target["name"]) not in names:
        return False
    target_groups = {str(group) for group in target.get("groups", [])}
    if groups and target_groups.isdisjoint(groups):
        return False
    return True


def _format_template(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**variables)
    if isinstance(value, list):
        return [_format_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _format_template(item, variables) for key, item in value.items()}
    return value


def _expand_target(target: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = target.get("matrix")
    if not matrix:
        return [target]
    if not isinstance(matrix, dict):
        raise TypeError(f"Target {target.get('name')} matrix must be a mapping.")
    matrix_items = [(str(key), list(values)) for key, values in matrix.items()]
    if not matrix_items:
        return [target]
    expanded: list[dict[str, Any]] = []
    keys = [key for key, _values in matrix_items]
    value_lists = [values for _key, values in matrix_items]
    for values in product(*value_lists):
        variables = dict(zip(keys, values, strict=True))
        item = deepcopy(target)
        item.pop("matrix", None)
        matrix_overrides = _format_template(item.pop("matrix_overrides", []), variables)
        item = _format_template(item, variables)
        command = [str(part) for part in item.get("command", [])]
        command.extend(str(part) for part in matrix_overrides)
        item["command"] = command
        item["matrix_values"] = variables
        expanded.append(item)
    return expanded


def expanded_targets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for target in manifest.get("targets", []):
        targets.extend(_expand_target(target))
    return targets


def _command_for_profile(manifest: dict[str, Any], target: dict[str, Any], profile: str) -> list[str]:
    command = [str(part) for part in target.get("command", [])]
    if not command:
        raise ValueError(f"Target {target.get('name')} has no command.")
    profiles = manifest.get("profiles", {})
    profile_cfg = profiles.get(profile, {})
    if bool(target.get("inherit_profile_overrides", True)):
        command.extend(str(part) for part in profile_cfg.get("default_overrides", []))
    command.extend(str(part) for part in target.get("profile_overrides", {}).get(profile, []))
    return command


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def selected_targets(manifest: dict[str, Any], *, names: set[str], groups: set[str]) -> list[dict[str, Any]]:
    targets = [target for target in expanded_targets(manifest) if _target_selected(target, names=names, groups=groups)]
    if not targets:
        raise ValueError("No repro manifest targets matched the requested filters.")
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print or run commands from the paper reproducibility manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--profile", choices=["paper", "smoke"], default="smoke")
    parser.add_argument("--target", action="append", default=[], help="Run one target by name. May be repeated.")
    parser.add_argument("--group", action="append", default=[], help="Run targets with this group. May be repeated.")
    parser.add_argument("--list", action="store_true", help="List selected target names and exit.")
    parser.add_argument("--run", action="store_true", help="Execute selected commands. Without --run, commands are only printed.")
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    targets = selected_targets(manifest, names=set(args.target), groups=set(args.group))

    if args.list:
        for target in targets:
            print(str(target["name"]))
        return 0

    status = 0
    env = os.environ.copy()
    for target in targets:
        command = _command_for_profile(manifest, target, args.profile)
        print(f"[repro] target={target['name']} profile={args.profile}", flush=True)
        print(f"[repro] command={_format_command(command)}", flush=True)
        if not args.run:
            continue
        completed = subprocess.run(command, cwd=APP_ROOT, env=env, check=False)
        if completed.returncode != 0:
            status = int(completed.returncode)
            print(f"[repro] failed target={target['name']} status={status}", file=sys.stderr, flush=True)
            if args.stop_on_failure:
                return status
        else:
            print(f"[repro] completed target={target['name']}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
