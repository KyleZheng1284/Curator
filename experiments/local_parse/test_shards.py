# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, PLR2004, S101

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Never

import pytest

from experiments.local_parse.shards import ShardInfrastructureError, run_shard
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_OUTPUT_VERSION, partition, project

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch

ShardFixture = tuple[dict[str, str | Path], list["InterleavedBatch"]]


@pytest.fixture
def shard(tmp_path: Path) -> ShardFixture:
    manifest, tasks = partition(
        tmp_path, [{"file_name": "a.pdf", "num_pages": 3}, {"file_name": "b.pdf", "num_pages": 2}], 2
    )
    groups = {}
    for task in tasks:
        for chunk in project(task, failed={2}):
            groups.setdefault(chunk._metadata["pdf_document_id"], []).append(chunk)
    documents = [PDFDocumentMergeStage().process_batch(chunks)[0] for chunks in groups.values()]
    return {
        "shard_id": "shard-1",
        "output_root": tmp_path / "shards",
        "manifest_path": manifest,
        "expected_output_version": TEST_OUTPUT_VERSION,
    }, documents


def _write(attempt: Path, documents: list[InterleavedBatch]) -> list[Path]:
    writer = InterleavedParquetWriterStage(
        path=str(attempt / "outputs"),
        materialize_on_write=False,
        on_materialize_error="warn",
        use_output_file_id=True,
        atomic_local_write=True,
    )
    return [Path(writer.process(document).data[0]) for document in documents]


def test_failure_after_write_preserves_the_attempt_and_allows_verified_resume(shard: ShardFixture) -> None:
    kwargs, documents = shard
    attempts = []

    def run(attempt: Path) -> list[Path]:
        attempts.append(attempt)
        return _write(attempt, documents)

    def reject(attempt: Path, paths: tuple[Path, ...]) -> Never:
        assert attempt == attempts[0]
        assert len(paths) == 2
        msg = "post-write check failed"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="post-write check failed"):
        run_shard(**kwargs, run_attempt=run, validate=reject)
    shard_dir = kwargs["output_root"] / kwargs["shard_id"]
    assert len(attempts) == 1
    assert (shard_dir / "manifest.json").exists() is False
    assert len(list(attempts[0].rglob("*.parquet"))) == 2

    accepted = run_shard(**kwargs, run_attempt=run)
    assert accepted.manifest_path.is_file()
    assert accepted.attempt_dir == attempts[1]
    assert accepted.validation["pages"] == 5
    assert accepted.validation["failed_pages"] == 1
    assert not accepted.resumed

    resumed = run_shard(**kwargs, run_attempt=run)
    assert resumed.resumed
    assert resumed.output_paths == accepted.output_paths
    assert resumed.validation == accepted.validation
    assert len(attempts) == 2


@pytest.mark.parametrize("fail_once", [True, False])
def test_replay_is_limited_to_one_classified_infrastructure_failure(shard: ShardFixture, fail_once: bool) -> None:
    kwargs, documents = shard
    attempts = []

    def run(attempt: Path) -> list[Path]:
        attempts.append(attempt)
        paths = _write(attempt, documents)
        if len(attempts) == 1 or not fail_once:
            msg = "owned engine stopped after output"
            raise ShardInfrastructureError(msg)
        return paths

    if fail_once:
        assert run_shard(**kwargs, run_attempt=run).attempt_dir == attempts[1]
    else:
        with pytest.raises(ShardInfrastructureError, match="owned engine stopped"):
            run_shard(**kwargs, run_attempt=run)
    assert len(attempts) == 2
    assert all(len(list(attempt.rglob("*.parquet"))) == 2 for attempt in attempts)
    assert (attempts[0].parent / "manifest.json").exists() is fail_once


def test_run_and_resume_reject_incomplete_or_tampered_outputs(shard: ShardFixture) -> None:
    kwargs, documents = shard

    with pytest.raises(ValueError, match="Missing whole PDFs"):
        run_shard(**kwargs, run_attempt=lambda attempt: _write(attempt, documents[:1]))

    accepted = run_shard(**kwargs, run_attempt=lambda attempt: _write(attempt, documents))
    with accepted.output_paths[0].open("ab") as output:
        output.write(b"unexpected bytes")

    def unexpected_attempt(_: Path) -> Never:
        pytest.fail("a corrupt commit must fail closed")

    with pytest.raises(ValueError, match="fingerprints differ"):
        run_shard(**kwargs, run_attempt=unexpected_attempt)
