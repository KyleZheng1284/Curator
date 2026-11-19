# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import (
    PDFPartitioningStage,
    page_task_fingerprint,
)
from nemo_curator.tasks import EmptyTask
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import partition, write_manifest


@pytest.mark.parametrize("size", [64, 128])
def test_page_cap_covers_long_and_short_documents_exactly(tmp_path: Path, size: int):
    _, tasks = partition(
        tmp_path, [{"file_name": "long.pdf", "num_pages": 257}, {"file_name": "short.pdf", "num_pages": 3}], size
    )
    observed = {"long.pdf": [], "short.pdf": []}
    for task in tasks:
        entries = [json.loads(value) for value in task.data]
        assert sum(len(entry["selected_page_indices"]) for entry in entries) <= size
        assert len(entries) <= size
        for entry in entries:
            observed[entry["file_name"]].extend(entry["selected_page_indices"])
    assert observed == {"long.pdf": list(range(257)), "short.pdf": [0, 1, 2]}


@pytest.mark.parametrize("legacy_pdfs_per_task", [1, 8, 100])
def test_pages_are_the_only_packing_limit(tmp_path: Path, legacy_pdfs_per_task: int):
    _, tasks = partition(
        tmp_path, [{"file_name": f"{i}.pdf", "num_pages": 1} for i in range(130)], pdfs=legacy_pdfs_per_task
    )
    assert [len(task.data) for task in tasks] == [64, 64, 2]


def test_noncontiguous_selection_is_not_renumbered_or_sorted(tmp_path: Path):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "selected_page_indices": [19, 2, 7, 0]}], 2)
    assert [json.loads(task.data[0])["selected_page_indices"] for task in tasks] == [[19, 2], [7, 0]]


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_page_cap_fails_before_work(tmp_path: Path, size: int):
    with pytest.raises(ValueError, match="pages_per_task"):
        partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 1}], size)


def test_duplicate_document_and_legacy_sample_identity_fail(tmp_path: Path):
    entries = [{"file_name": "a.pdf", "num_pages": 1}] * 2
    with pytest.raises(ValueError, match="ambiguous sample_id"):
        partition(tmp_path, entries)
    with pytest.raises(ValueError, match="ambiguous sample_id"):
        partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 1}, {"file_name": "a.png", "num_pages": 1}])


def test_default_partitioning_keeps_existing_manifest_contract(tmp_path: Path):
    manifest = write_manifest(tmp_path, [{"file_name": "a.pdf"}, {"file_name": "b.pdf"}])
    stage = PDFPartitioningStage(manifest_path=str(manifest), pdfs_per_task=2)
    tasks = stage.process(EmptyTask())
    assert len(tasks) == 1
    assert "pdf_page_chunks" not in tasks[0]._metadata
    assert [json.loads(entry)["file_name"] for entry in tasks[0].data] == ["a.pdf", "b.pdf"]


def test_page_mode_does_not_silently_skip_malformed_manifest_records(tmp_path: Path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"something_else": "missing.pdf"}\n')
    with pytest.raises(ValueError, match="lacks"):
        PDFPartitioningStage(manifest_path=str(manifest), pages_per_task=64).process(EmptyTask())


def test_page_budget_does_not_rebind_legacy_positional_partitioner_arguments():
    from nemo_curator.stages.resources import Resources

    stage = PDFPartitioningStage("input.jsonl", 8, 17, "dataset", "file", "files", "url", "legacy", Resources(cpus=2))
    assert (stage.max_pdfs, stage.dataset_name, stage.name) == (17, "dataset", "legacy")
    assert stage.resources.cpus == 2
    assert stage.pages_per_task is None


def test_page_priorities_reorder_complete_tasks_with_stable_ties(tmp_path: Path):
    manifest, stock = partition(
        tmp_path,
        [
            {"file_name": "a.pdf", "selected_page_indices": [19, 2, 7, 0, 11]},
            {"file_name": "b.pdf", "selected_page_indices": [6, 5, 3]},
        ],
        size=2,
    )
    settings = {"manifest_path": str(manifest), "pages_per_task": 2}
    priorities = {
        page_task_fingerprint(stock[1]): 9.0,
        page_task_fingerprint(stock[2]): 4.0,
        page_task_fingerprint(stock[3]): 9.0,
    }
    ranked = PDFPartitioningStage(**settings, page_task_priorities=priorities).process(EmptyTask())
    assert [task._metadata["partition_index"] for task in ranked] == [1, 3, 2, 0]
    for task in ranked:
        original = stock[task._metadata["partition_index"]]
        assert task.data == original.data
        assert task._metadata == original._metadata

    unchanged = PDFPartitioningStage(**settings, page_task_priorities={}).process(EmptyTask())
    assert [task.data for task in unchanged] == [task.data for task in stock]


def test_page_priorities_reject_hints_after_source_version_changes(tmp_path: Path):
    manifest, stock = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 1}])
    priorities = {page_task_fingerprint(stock[0]): 3.0}
    write_manifest(tmp_path, [{"file_name": "a.pdf", "num_pages": 1, "source_version": "new-version"}])

    with pytest.raises(ValueError, match="fingerprint absent"):
        PDFPartitioningStage(manifest_path=str(manifest), pages_per_task=64, page_task_priorities=priorities).process(
            EmptyTask()
        )


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True, "3"])
def test_page_priorities_reject_invalid_weights(tmp_path: Path, weight: object):
    manifest, stock = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 1}])
    with pytest.raises(ValueError, match="finite, nonnegative costs"):
        PDFPartitioningStage(
            manifest_path=str(manifest),
            pages_per_task=64,
            page_task_priorities={page_task_fingerprint(stock[0]): weight},
        ).process(EmptyTask())


def test_page_priorities_require_page_mode():
    with pytest.raises(ValueError, match="pages_per_task"):
        PDFPartitioningStage(manifest_path="unused.jsonl", page_task_priorities={"a" * 64: 1.0})
