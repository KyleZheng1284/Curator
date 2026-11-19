# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Validate and publish the outputs of one bounded Curator Page64 shard."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import validate_page_budget_outputs

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_SCHEMA = "curator-local-parse-shard.v1"


class ShardInfrastructureError(RuntimeError):
    """An owned attempt stopped on a classified, retryable infrastructure failure."""


@dataclass(frozen=True)
class ShardResult:
    manifest_path: Path
    attempt_dir: Path
    output_paths: tuple[Path, ...]
    validation: dict[str, Any]
    resumed: bool


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _output_paths(attempt_dir: Path, paths: Iterable[Path | str]) -> tuple[Path, ...]:
    outputs = []
    for path in paths:
        output = (attempt_dir / path).resolve(strict=True)
        if not output.is_relative_to(attempt_dir) or output.suffix != ".parquet" or not output.is_file():
            msg = f"Shard output must be a Parquet file inside its attempt: {path}"
            raise ValueError(msg)
        outputs.append(output)
    if len(outputs) != len(set(outputs)) or set(outputs) != set(attempt_dir.rglob("*.parquet")):
        msg = "Shard attempt must report every Parquet output exactly once"
        raise ValueError(msg)
    return tuple(sorted(outputs))


def _file_inventory(attempt_dir: Path, paths: tuple[Path, ...]) -> list[dict[str, str]]:
    return [{"path": path.relative_to(attempt_dir).as_posix(), "sha256": _sha256(path)} for path in paths]


def _published_attempt(published: Path, identity: dict[str, str]) -> tuple[Path, tuple[Path, ...], dict[str, Any]]:
    receipt = json.loads(published.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in identity.items()):
        msg = "Published shard manifest differs from the admitted shard"
        raise ValueError(msg)
    attempted = receipt.get("attempt")
    if not isinstance(attempted, str) or not re.fullmatch(r"attempt-[a-zA-Z0-9_]+", attempted):
        msg = "Published shard contains an invalid attempt name"
        raise ValueError(msg)
    attempt_dir = published.parent / attempted
    outputs = _output_paths(attempt_dir, (entry["path"] for entry in receipt["files"]))
    if _file_inventory(attempt_dir, outputs) != receipt["files"]:
        msg = "Published shard output fingerprints differ from the manifest"
        raise ValueError(msg)
    return attempt_dir, outputs, receipt["validation"]


def _publish(path: Path, receipt: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".manifest-", delete=False
    ) as stream:
        pending = Path(stream.name)
        try:
            json.dump(receipt, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _run_with_one_replay(
    shard_dir: Path, run_attempt: Callable[[Path], Iterable[Path | str]]
) -> tuple[Path, tuple[Path | str, ...]]:
    attempt_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=shard_dir))
    try:
        return attempt_dir, tuple(run_attempt(attempt_dir))
    except ShardInfrastructureError:
        attempt_dir = Path(tempfile.mkdtemp(prefix="attempt-", dir=shard_dir))
    return attempt_dir, tuple(run_attempt(attempt_dir))


def run_shard(  # noqa: PLR0913
    *,
    shard_id: str,
    output_root: Path | str,
    manifest_path: Path | str,
    expected_output_version: str,
    run_attempt: Callable[[Path], Iterable[Path | str]],
    validate: Callable[[Path, tuple[Path, ...]], None] | None = None,
    file_name_field: str = "file_name",
    url_field: str = "url",
) -> ShardResult:
    """Run an exact shard manifest, or verify and reuse its published attempt.

    ``run_attempt`` returns all Parquet paths after the attempt has stopped writing.
    It may raise ``ShardInfrastructureError`` only after its owned work is stopped.
    ``validate`` may add checks on the attempt directory and resolved output paths.
    A failed attempt stays intact and never becomes the published shard.
    """
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", shard_id):
        msg = f"Invalid shard ID: {shard_id!r}"
        raise ValueError(msg)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_output_version):
        msg = "Shard output version must be its admitted SHA256"
        raise ValueError(msg)

    input_manifest = Path(manifest_path).resolve(strict=True)
    identity = {
        "schema": _SCHEMA,
        "shard_id": shard_id,
        "input_manifest_sha256": _sha256(input_manifest),
        "output_version": expected_output_version,
        "file_name_field": file_name_field,
        "url_field": url_field,
    }
    shard_dir = Path(output_root).resolve() / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    published = shard_dir / "manifest.json"

    def validate_attempt(attempt_dir: Path, paths: tuple[Path, ...]) -> dict[str, Any]:
        receipt = validate_page_budget_outputs(
            str(input_manifest),
            [str(path) for path in paths],
            expected_output_version=expected_output_version,
            file_name_field=file_name_field,
            url_field=url_field,
        )
        if not receipt["coverage_passed"] or not receipt["pages"]:
            msg = "Shard validation must reconcile a nonempty physical-page inventory"
            raise ValueError(msg)
        if validate is not None:
            validate(attempt_dir, paths)
        if _sha256(input_manifest) != identity["input_manifest_sha256"]:
            msg = "Shard input manifest changed during validation"
            raise ValueError(msg)
        return receipt

    with (shard_dir / ".lock").open("ab") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if published.exists():
            attempt_dir, outputs, recorded_validation = _published_attempt(published, identity)
            validation = validate_attempt(attempt_dir, outputs)
            if validation != recorded_validation:
                msg = "Published shard output validation differs from the manifest"
                raise ValueError(msg)
            return ShardResult(published, attempt_dir, outputs, validation, resumed=True)

        attempt_dir, paths = _run_with_one_replay(shard_dir, run_attempt)
        outputs = _output_paths(attempt_dir, paths)
        validation = validate_attempt(attempt_dir, outputs)
        receipt = {
            **identity,
            "attempt": attempt_dir.name,
            "files": _file_inventory(attempt_dir, outputs),
            "validation": validation,
        }
        _publish(published, receipt)
        return ShardResult(published, attempt_dir, outputs, validation, resumed=False)
