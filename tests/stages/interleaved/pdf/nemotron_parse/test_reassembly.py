# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import gc
import json
import weakref
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.tasks import InterleavedBatch
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG, partition, project


def chunks(tmp_path: Path, size: int = 64, **kwargs):
    _, tasks = partition(tmp_path, [{"file_name": "long.pdf", "num_pages": 130}], size)
    return [chunk for task in tasks for chunk in project(task, **kwargs)]


def test_unordered_chunks_reconstruct_content_images_boxes_and_order(tmp_path: Path):
    split = chunks(tmp_path, 64)
    split_result = PDFDocumentMergeStage().process_batch(list(reversed(split)))[0]
    whole = PDFDocumentMergeStage().process_batch(chunks(tmp_path, 200))[0]
    assert split_result.to_pyarrow().to_pylist() == whole.to_pyarrow().to_pylist()
    assert split_result._metadata["output_file_id"] == whole._metadata["output_file_id"]
    rows = split_result.to_pyarrow().to_pylist()
    assert [row["position"] for row in rows[1:]] == list(range(390))
    assert json.loads(rows[0]["text_content"])["num_pages"] == 130
    assert all(row["document_id"] for row in rows)


@pytest.mark.parametrize("mutation", ["duplicate_chunk", "wrong_page", "missing_content", "wrong_version"])
def test_declared_chunk_counts_do_not_hide_corruption(tmp_path: Path, mutation: str):
    tasks = chunks(tmp_path)
    if mutation == "duplicate_chunk":
        tasks.append(copy.deepcopy(tasks[-1]))
    elif mutation == "wrong_version":
        tasks[-1]._metadata["pdf_chunk"]["source_version"] = "wrong-version"
    else:
        rows = tasks[0].to_pyarrow().to_pylist()
        if mutation == "wrong_page":
            rows[1]["page_number"] = 500
        else:
            rows.pop(1)
            for position, row in enumerate(rows[1:]):
                row["position"] = position
        tasks[0].data = pa.Table.from_pylist(rows, schema=tasks[0].data.schema)
    with pytest.raises(ValueError, match=r"PDF|Chunk|outcome|page|document"):
        PDFDocumentMergeStage().process_batch(tasks)


def test_noncontiguous_physical_pages_preserve_selection_order(tmp_path: Path):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "selected_page_indices": [19, 2, 7, 0]}], 2)
    projected = [chunk for task in tasks for chunk in project(task)]
    result = PDFDocumentMergeStage().process_batch(list(reversed(projected)))[0]
    rows = result.to_pyarrow().to_pylist()
    assert [row["page_number"] for row in rows[1:]][::3] == [19, 2, 7, 0]
    assert [x["page_number"] for x in json.loads(rows[0]["text_content"])["page_outcomes"]] == [19, 2, 7, 0]


def test_failures_and_zero_extracted_elements_remain_accounted(tmp_path: Path):
    result = PDFDocumentMergeStage().process_batch(chunks(tmp_path, failed={64}, no_elements={65}))[0]
    rows = result.to_pyarrow().to_pylist()
    outcomes = json.loads(rows[0]["text_content"])["page_outcomes"]
    assert outcomes[64] == {"page_number": 64, "status": "failed", "element_count": 1}
    assert outcomes[65] == {"page_number": 65, "status": "success", "element_count": 0}
    assert any(row.get("materialize_error") for row in rows if row.get("page_number") == 64)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong", "wrong_document"])
def test_postprocessing_verifies_actual_physical_identity(tmp_path: Path, mutation: str):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 3}])
    source = tasks[0]
    chunk = source._metadata["pdf_page_chunks"]["a.pdf"]
    rows = [
        {"sample_id": "a", "pdf_name": "a.pdf", "document_id": chunk["document_id"], "page_number": p}
        for p in range(3)
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "wrong":
        rows[-1]["page_number"] = 99
    else:
        rows[-1]["document_id"] = "another-document"
    batch = InterleavedBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(rows),
        _metadata={**source._metadata, "parse_mode": "single", "pdf_extraction_config": TEST_EXTRACTION_CONFIG},
    )
    with pytest.raises(ValueError, match=r"Missing|physical"):
        NemotronParsePostprocessStage().process(batch)


def test_merge_requires_streaming_fan_in_execution():
    assert PDFDocumentMergeStage().ray_stage_spec() == {"group_by_metadata": "pdf_document_id"}
    assert PDFDocumentMergeStage(manifest_path="manifest.jsonl").ray_stage_spec() == {"streaming_fan_in": True}
    with pytest.raises(RuntimeError, match="RayDataExecutor"):
        PDFDocumentMergeStage().process(None)


def _mixed_documents(tmp_path: Path) -> tuple[Path, list[InterleavedBatch], dict[str, InterleavedBatch]]:
    """Ranges in task order for two single-range and two multi-range documents, and each whole document."""
    entries = [
        {"file_name": "short.pdf", "num_pages": 3},
        {"file_name": "long.pdf", "num_pages": 11},
        {"file_name": "tail.pdf", "num_pages": 2},
        {"file_name": "wide.pdf", "num_pages": 7},
    ]
    manifest, tasks = partition(tmp_path, entries, 4)
    _, unsplit = partition(tmp_path, entries, 64)
    grouped = PDFDocumentMergeStage()
    whole = [grouped.process_batch([chunk])[0] for chunk in project(unsplit[0], failed={2}, no_elements={5})]
    return (
        manifest,
        [chunk for task in tasks for chunk in project(task, failed={2}, no_elements={5})],
        {document._metadata["output_file_id"]: document for document in whole},
    )


@pytest.mark.parametrize("order", ["task_order", "reversed", "interleaved"])
def test_ranges_arriving_one_at_a_time_in_any_order_match_whole_documents(tmp_path: Path, order: str):
    manifest, ranges, whole = _mixed_documents(tmp_path)
    arrivals = {"task_order": ranges, "reversed": ranges[::-1], "interleaved": ranges[1::2] + ranges[-2::-2]}[order]
    assert len({id(arrival) for arrival in arrivals}) == len(ranges) > len(whole)
    stage = PDFDocumentMergeStage(manifest_path=str(manifest))
    remaining = Counter(arrival._metadata["pdf_document_id"] for arrival in arrivals)
    merged = {}
    for arrival in arrivals:
        document_id = arrival._metadata["pdf_document_id"]
        remaining[document_id] -= 1
        emitted = stage.process_batch([arrival])
        # A document appears with its last range, never earlier and never partially.
        assert len(emitted) == (remaining[document_id] == 0)
        merged.update({document._metadata["output_file_id"]: document for document in emitted})
    assert merged.keys() == whole.keys()
    for output_id, document in merged.items():
        assert document.data.equals(whole[output_id].data)
        assert document._metadata == whole[output_id]._metadata


def test_incomplete_document_is_never_emitted(tmp_path: Path):
    manifest, tasks = partition(tmp_path, [{"file_name": "long.pdf", "num_pages": 130}])
    stage = PDFDocumentMergeStage(manifest_path=str(manifest))
    ranges = [chunk for task in tasks for chunk in project(task)]
    assert stage.process_batch(ranges[:-1]) == []
    assert stage.process_batch([]) == []


@pytest.mark.parametrize("arrival", ["while_held", "after_its_document_merged", "single_range_document"])
def test_range_arriving_twice_is_rejected(tmp_path: Path, arrival: str):
    if arrival == "single_range_document":
        manifest, tasks = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 3}])
        stage = PDFDocumentMergeStage(manifest_path=str(manifest))
        ranges = project(tasks[0])
        assert len(stage.process_batch(copy.deepcopy(ranges))) == 1
    else:
        manifest, tasks = partition(tmp_path, [{"file_name": "long.pdf", "num_pages": 130}])
        stage = PDFDocumentMergeStage(manifest_path=str(manifest))
        ranges = [chunk for task in tasks for chunk in project(task)]
        delivered = ranges[:1] if arrival == "while_held" else ranges
        assert len(stage.process_batch(copy.deepcopy(delivered))) == (arrival != "while_held")
    with pytest.raises(ValueError, match="nonoverlapping ranges"):
        stage.process_batch([ranges[0]])


def test_single_range_documents_pass_straight_through(tmp_path: Path):
    manifest, tasks = partition(tmp_path, [{"file_name": f"{index}.pdf", "num_pages": 2} for index in range(6)], 4)
    stage = PDFDocumentMergeStage(manifest_path=str(manifest))
    for task in tasks:
        for arrival in project(task):
            (document,) = stage.process_batch([arrival])
            assert json.loads(document.data["text_content"][0].as_py())["num_pages"] == 2
            assert not stage._held


def test_held_ranges_are_released_when_their_document_is_merged(tmp_path: Path):
    manifest, ranges, _ = _mixed_documents(tmp_path)
    stage = PDFDocumentMergeStage(manifest_path=str(manifest))
    pages = Counter()
    for arrival in ranges:
        chunk = arrival._metadata["pdf_chunk"]
        pages[chunk["document_id"]] += len(chunk["page_indices"])
        if stage.process_batch([arrival]):
            del pages[chunk["document_id"]]
        # Only documents still waiting for a range are held, with exactly the ranges that have arrived.
        assert {document_id: count for document_id, (count, _) in stage._held.items()} == pages
        for count, held_ranges in stage._held.values():
            assert count == sum(len(held._metadata["pdf_chunk"]["page_indices"]) for held in held_ranges.values())
    assert not stage._held

    first = weakref.ref(ranges[1])
    del ranges, arrival
    gc.collect()
    assert first() is None


@pytest.mark.parametrize("length", [0, 1, 3])
def test_merge_accepts_ray_numpy_groups(tmp_path: Path, length: int):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": max(length, 1)}], 1)
    projected = [chunk for task in tasks for chunk in project(task)] if length else []
    group = np.empty(length, dtype=object)
    group[:] = projected
    result = PDFDocumentMergeStage().process_batch(group)
    if length:
        assert json.loads(result[0].data["text_content"][0].as_py())["num_pages"] == length
    else:
        assert result == []


@pytest.mark.parametrize("first_range", ["text_only", "failed", "no_elements"])
def test_heterogeneous_ranges_match_whole_document(tmp_path: Path, first_range: str):
    kwargs = {first_range: {0}}
    _, split = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 2}], 1)
    split_output = PDFDocumentMergeStage().process_batch(
        [chunk for task in reversed(split) for chunk in project(task, **kwargs)]
    )[0]
    _, whole = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 2}], 2)
    whole_output = PDFDocumentMergeStage().process_batch(project(whole[0], **kwargs))[0]
    assert split_output.data.equals(whole_output.data)
    assert split_output._metadata["output_file_id"] == whole_output._metadata["output_file_id"]


def test_contiguous_positions_do_not_hide_reversed_physical_pages(tmp_path: Path):
    _, tasks = partition(tmp_path, [{"file_name": "a.pdf", "selected_page_indices": [19, 2, 7, 0]}], 4)
    chunk = project(tasks[0])[0]
    rows = chunk.data.to_pylist()
    content = list(reversed(rows[1:]))
    for position, row in enumerate(content):
        row["position"] = position
    chunk.data = pa.Table.from_pylist([rows[0], *content], schema=chunk.data.schema)
    with pytest.raises(ValueError, match="selection order"):
        PDFDocumentMergeStage().process_batch([chunk])


def test_fanout_does_not_repeat_other_documents_metadata(tmp_path: Path):
    _, tasks = partition(tmp_path, [{"file_name": f"{i}.pdf", "num_pages": 1} for i in range(4)])
    output = project(tasks[0])
    for task in output:
        assert "pdf_page_chunks" not in task._metadata
        assert task._metadata["source_files"] == [task._metadata["pdf_chunk"]["source_locator"]]
