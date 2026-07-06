from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
def _find_app_root(path: Path) -> Path:
    current = path.resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "scripts").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not infer app root from {path}.")


APP_ROOT = _find_app_root(SCRIPT_PATH)
DEFAULT_DATA_ROOT = (APP_ROOT.parent if APP_ROOT.name == "code" else APP_ROOT) / "data" / "lm"
DEFAULT_MANIFEST_PATH = SCRIPT_PATH.with_name("lm_dataset_manifest.json")

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from src.lm_tokenizers import build_tokenizer


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_dataset_root() -> Path:
    return DEFAULT_DATA_ROOT


def verify_dataset(dataset_name: str, dataset_dir: str | Path, spec: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    if dataset_name == "text8":
        return _verify_text8(dataset_dir, spec)
    if dataset_name == "lm1b":
        return _verify_lm1b(dataset_dir, spec)
    raise ValueError(f"Unsupported dataset {dataset_name!r}.")


def prepare_dataset(
    dataset_name: str,
    dataset_dir: str | Path,
    spec: dict[str, Any],
    *,
    force: bool = False,
    keep_archives: bool = False,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        try:
            info = verify_dataset(dataset_name, dataset_dir, spec)
            return {"status": "skip", **info}
        except Exception:
            pass

    if dataset_name == "lm1b":
        info = {
            "status": "prepared",
            **_prepare_lm1b(dataset_dir, spec),
        }
        _write_dataset_info(dataset_dir / "dataset_info.json", info)
        return info

    archive_path, source_url = _download_archive(
        urls=list(spec["urls"]),
        archive_name=str(spec["archive_name"]),
        dataset_dir=dataset_dir,
        keep_archives=keep_archives,
    )
    archive_sha256 = _sha256_file(archive_path)
    try:
        if dataset_name == "text8":
            info = _prepare_text8(dataset_dir, archive_path, spec)
        else:
            raise ValueError(f"Unsupported dataset {dataset_name!r}.")
    finally:
        if archive_path.exists() and not keep_archives and archive_path.parent != dataset_dir:
            archive_path.unlink()

    info = {
        "status": "prepared",
        "source_url": source_url,
        "archive_name": archive_path.name,
        "archive_sha256": archive_sha256,
        **info,
    }
    _write_dataset_info(dataset_dir / "dataset_info.json", info)
    return info


def _verify_text8(dataset_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    raw_path = dataset_dir / str(spec["raw_output_name"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw text8 file: {raw_path}")
    raw_size = raw_path.stat().st_size
    expected_raw_bytes = int(spec["expected_raw_bytes"])
    if raw_size != expected_raw_bytes:
        raise ValueError(f"Expected {expected_raw_bytes} bytes in {raw_path}, found {raw_size}.")

    split_info: dict[str, int] = {}
    for split_name, split_bytes in spec["split_bytes"].items():
        split_path = dataset_dir / f"{split_name}.bin"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing text8 split file: {split_path}")
        actual = split_path.stat().st_size
        expected = int(split_bytes)
        if actual != expected:
            raise ValueError(f"Expected {expected} bytes in {split_path}, found {actual}.")
        split_info[split_name] = actual

    if sum(split_info.values()) != raw_size:
        raise ValueError(
            f"Split sizes {split_info} do not sum to raw size {raw_size} for {dataset_dir}."
        )

    return {
        "dataset_name": str(spec.get("dataset_name", "text8")),
        "dataset_dir": str(dataset_dir),
        "raw_path": str(raw_path),
        "raw_bytes": int(raw_size),
        "split_bytes": split_info,
    }


def _verify_lm1b(dataset_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    train_dir = dataset_dir / str(spec.get("train_output_dir", "train"))
    val_path = dataset_dir / str(spec.get("val_output_name", "val.tokens"))
    test_path = dataset_dir / str(spec.get("test_output_name", "test.tokens"))
    dataset_info_path = dataset_dir / "dataset_info.json"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing LM1B train shard directory: {train_dir}")
    shard_paths = sorted(path for path in train_dir.iterdir() if path.is_file() and path.suffix == ".tokens")
    if not shard_paths:
        raise FileNotFoundError(f"LM1B train shard directory is empty: {train_dir}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing LM1B validation token file: {val_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing LM1B test token file: {test_path}")
    if not dataset_info_path.is_file():
        raise FileNotFoundError(f"Missing LM1B dataset metadata file: {dataset_info_path}")

    shard_token_counts = {path.name: _token_file_count(path) for path in shard_paths}
    if any(int(count) <= 0 for count in shard_token_counts.values()):
        raise ValueError(f"LM1B contains an empty train shard under {train_dir}.")

    split_token_counts = {
        "train": sum(int(count) for count in shard_token_counts.values()),
        "val": _token_file_count(val_path),
        "test": _token_file_count(test_path),
    }
    if int(split_token_counts["val"]) <= 0 or int(split_token_counts["test"]) <= 0:
        raise ValueError(f"LM1B validation/test token files must be non-empty under {dataset_dir}.")

    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    expected_counts = dataset_info.get("split_token_counts")
    if isinstance(expected_counts, dict):
        for split_name, actual in split_token_counts.items():
            expected = expected_counts.get(split_name)
            if expected is not None and int(expected) != int(actual):
                raise ValueError(
                    f"LM1B {split_name} token count mismatch in {dataset_dir}: expected {expected}, found {actual}."
                )

    expected_shards = dataset_info.get("train_shard_count")
    if expected_shards is not None and int(expected_shards) != len(shard_paths):
        raise ValueError(
            f"LM1B train shard count mismatch in {dataset_dir}: expected {expected_shards}, found {len(shard_paths)}."
        )

    return {
        "dataset_name": "lm1b",
        "dataset_dir": str(dataset_dir),
        "train_dir": str(train_dir),
        "train_shard_count": len(shard_paths),
        "train_shards": [str(path) for path in shard_paths],
        "shard_token_counts": shard_token_counts,
        "split_token_counts": split_token_counts,
        "val_path": str(val_path),
        "test_path": str(test_path),
        "hf_dataset": spec.get("hf_dataset"),
        "hf_config": spec.get("hf_config"),
    }


def _prepare_text8(dataset_dir: Path, archive_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    raw_member_name = _find_zip_member(archive_path, str(spec["raw_member_basename"]))
    raw_path = dataset_dir / str(spec["raw_output_name"])
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open(raw_member_name) as src, raw_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    split_offsets = spec["split_bytes"]
    payload = raw_path.read_bytes()
    expected_raw_bytes = int(spec["expected_raw_bytes"])
    if len(payload) != expected_raw_bytes:
        raise ValueError(f"Expected {expected_raw_bytes} bytes in {raw_path}, found {len(payload)} after extraction.")

    cursor = 0
    for split_name in ("train", "val", "test"):
        split_size = int(split_offsets[split_name])
        split_payload = payload[cursor : cursor + split_size]
        (dataset_dir / f"{split_name}.bin").write_bytes(split_payload)
        cursor += split_size
    return _verify_text8(dataset_dir, spec)


def _load_hf_split(dataset_id: str, dataset_config: str | None, split_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "LM1B bootstrap requires the 'datasets' package. Install code/requirements.txt and rerun the bootstrap."
        ) from exc

    args = [str(dataset_id)]
    if dataset_config not in (None, ""):
        args.append(str(dataset_config))
    return load_dataset(*args, split=str(split_name))


def _write_token_buffer(path: Path, payload: array[int]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload.tobytes())
    return len(payload)


def _token_file_count(path: Path) -> int:
    size_bytes = int(path.stat().st_size)
    if size_bytes % 4 != 0:
        raise ValueError(f"Token file {path} has {size_bytes} bytes, which is not divisible by 4.")
    return size_bytes // 4


def _row_to_tokens(tokenizer, text: str) -> list[int]:
    normalized = str(text)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return [int(token) for token in tokenizer.encode_text(normalized)]


def _prepare_lm1b(dataset_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    hf_dataset = str(spec["hf_dataset"])
    hf_config = None if spec.get("hf_config") in (None, "") else str(spec["hf_config"])
    text_field = str(spec.get("text_field", "text"))
    train_split = str(spec.get("train_split", "train"))
    test_split = str(spec.get("test_split", "test"))
    validation_rows = int(spec.get("validation_rows", 10_000))
    train_shard_target_tokens = int(spec.get("train_shard_tokens", 1_000_000))
    train_output_dir = dataset_dir / str(spec.get("train_output_dir", "train"))
    val_output_path = dataset_dir / str(spec.get("val_output_name", "val.tokens"))
    test_output_path = dataset_dir / str(spec.get("test_output_name", "test.tokens"))
    tokenizer_name = str(spec.get("tokenizer", "gpt2_bpe"))

    if validation_rows <= 0:
        raise ValueError(f"LM1B validation_rows must be positive, got {validation_rows}.")
    if train_shard_target_tokens <= 0:
        raise ValueError(f"LM1B train_shard_tokens must be positive, got {train_shard_target_tokens}.")

    if train_output_dir.exists():
        shutil.rmtree(train_output_dir)
    for path in (val_output_path, test_output_path):
        if path.exists():
            path.unlink()

    train_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(tokenizer_name)

    train_rows = _load_hf_split(hf_dataset, hf_config, train_split)
    test_rows = _load_hf_split(hf_dataset, hf_config, test_split)

    shard_index = 0
    train_buffer = array("I")
    val_buffer = array("I")
    train_token_count = 0
    val_token_count = 0
    shard_token_counts: dict[str, int] = {}

    def flush_train_shard(*, force: bool = False) -> None:
        nonlocal shard_index, train_token_count, train_buffer
        while len(train_buffer) >= train_shard_target_tokens or (force and len(train_buffer) > 0):
            take = len(train_buffer) if force else train_shard_target_tokens
            shard_payload = array("I", train_buffer[:take])
            del train_buffer[:take]
            shard_name = f"shard-{shard_index:05d}.tokens"
            written = _write_token_buffer(train_output_dir / shard_name, shard_payload)
            shard_token_counts[shard_name] = int(written)
            train_token_count += int(written)
            shard_index += 1
            if force:
                break

    train_seen = 0
    for row in train_rows:
        if text_field not in row:
            raise KeyError(f"LM1B row is missing the configured text field {text_field!r}.")
        row_tokens = _row_to_tokens(tokenizer, row[text_field])
        if train_seen < validation_rows:
            val_buffer.extend(row_tokens)
            val_token_count += len(row_tokens)
        else:
            train_buffer.extend(row_tokens)
            flush_train_shard(force=False)
        train_seen += 1

    flush_train_shard(force=True)
    if train_token_count <= 0:
        raise ValueError("LM1B preprocessing produced zero training tokens.")
    if val_token_count <= 0:
        raise ValueError("LM1B preprocessing produced zero validation tokens.")

    test_buffer = array("I")
    test_token_count = 0
    for row in test_rows:
        if text_field not in row:
            raise KeyError(f"LM1B row is missing the configured text field {text_field!r}.")
        row_tokens = _row_to_tokens(tokenizer, row[text_field])
        test_buffer.extend(row_tokens)
        test_token_count += len(row_tokens)
    if test_token_count <= 0:
        raise ValueError("LM1B preprocessing produced zero test tokens.")

    _write_token_buffer(val_output_path, val_buffer)
    _write_token_buffer(test_output_path, test_buffer)

    info = {
        "dataset_name": "lm1b",
        "dataset_dir": str(dataset_dir),
        "train_dir": str(train_output_dir),
        "train_shard_count": len(shard_token_counts),
        "train_shards": [str(train_output_dir / shard_name) for shard_name in sorted(shard_token_counts.keys())],
        "shard_token_counts": dict(sorted(shard_token_counts.items())),
        "val_path": str(val_output_path),
        "test_path": str(test_output_path),
        "source_format": "huggingface_datasets",
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split_map": {
            "train": train_split,
            "val": f"{train_split}[:{validation_rows}]",
            "test": test_split,
        },
        "tokenizer_name": tokenizer.metadata().name,
        "token_unit": tokenizer.metadata().token_unit,
        "vocab_size": int(tokenizer.metadata().vocab_size),
        "validation_rows": validation_rows,
        "train_shard_target_tokens": train_shard_target_tokens,
        "split_token_counts": {
            "train": int(train_token_count),
            "val": int(val_token_count),
            "test": int(test_token_count),
        },
        "train_shard_count": len(shard_token_counts),
    }
    return info


def _download_archive(*, urls: list[str], archive_name: str, dataset_dir: Path, keep_archives: bool) -> tuple[Path, str]:
    last_error: Exception | None = None
    archive_path = dataset_dir / archive_name if keep_archives else dataset_dir / f".{archive_name}.download"
    for url in urls:
        try:
            _download_file(url, archive_path)
            return archive_path, url
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if archive_path.exists():
                archive_path.unlink()
    if last_error is None:
        raise RuntimeError("No download URLs were provided.")
    raise RuntimeError(f"Failed to download archive from all sources: {last_error}") from last_error


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        shutil.copyfile(Path(urllib.request.url2pathname(parsed.path)), destination)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "gexit-diffusion-dataset-bootstrap/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _find_zip_member(archive_path: Path, expected_basename: str) -> str:
    with zipfile.ZipFile(archive_path) as zf:
        candidates = [
            name for name in zf.namelist()
            if not name.endswith("/") and Path(name).name == expected_basename
        ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Archive {archive_path} does not contain a member named {expected_basename!r}.")
    raise ValueError(f"Archive {archive_path} contains multiple members named {expected_basename!r}: {candidates}")


def _find_zip_member_any(archive_path: Path, expected_basenames: list[str]) -> str:
    errors: list[str] = []
    for expected_basename in expected_basenames:
        try:
            return _find_zip_member(archive_path, expected_basename)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError(" ; ".join(errors))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_dataset_info(path: Path, info: dict[str, Any]) -> None:
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare LM benchmark datasets under ./data/lm.")
    parser.add_argument("--dataset", choices=["text8", "lm1b", "all"], required=True)
    parser.add_argument("--root", type=Path, default=default_dataset_root())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = load_manifest()
    dataset_names = list(manifest.keys()) if args.dataset == "all" else [str(args.dataset)]
    for dataset_name in dataset_names:
        spec = manifest[dataset_name]
        dataset_dir = Path(args.root) / dataset_name
        if args.verify_only:
            info = verify_dataset(dataset_name, dataset_dir, spec)
            print(f"[verify] dataset={dataset_name} status=ok dir={dataset_dir}")
        else:
            info = prepare_dataset(
                dataset_name,
                dataset_dir,
                spec,
                force=bool(args.force),
                keep_archives=bool(args.keep_archives),
            )
            print(f"[prepare] dataset={dataset_name} status={info['status']} dir={dataset_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
