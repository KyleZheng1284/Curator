# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import validation
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import (
    PDFPageBudgetValidationStage,
    validate_page_budget_outputs,
)
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_OUTPUT_VERSION, partition, project


def outputs(tmp_path: Path):
    manifest, tasks = partition(
        tmp_path, [{"file_name": "a.pdf", "num_pages": 4}, {"file_name": "b.pdf", "num_pages": 1}], 2
    )
    groups = {}
    for task in tasks:
        for chunk in project(task, failed={2}):
            groups.setdefault(chunk._metadata["pdf_document_id"], []).append(chunk)
    documents = [PDFDocumentMergeStage().process_batch(chunks)[0] for chunks in groups.values()]
    writer = InterleavedParquetWriterStage(
        path=str(tmp_path / "output"), materialize_on_write=False, on_materialize_error="warn", use_output_file_id=True
    )
    files = [writer.process(document) for document in documents]
    return manifest, writer, documents, files


ACCEPTED = {
    "coverage_passed": True,
    "documents": 2,
    "pages": 5,
    "successful_pages": 4,
    "failed_pages": 1,
    "all_pages_succeeded": False,
    "extraction_quality_validated": False,
    "output_version": TEST_OUTPUT_VERSION,
}


def copy(source: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return str(target)


@pytest.fixture(params=["calling_process", "ray_tasks"])
def reread(request: pytest.FixtureRequest) -> str:
    if request.param == "ray_tasks":
        request.getfixturevalue("shared_ray_client")
    return request.param


def test_writer_and_reader_roundtrip_preserve_lineage_and_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, _, _, files = outputs(tmp_path)
    paths = [task.data[0] for task in files]
    reader = Mock(wraps=pq.read_table)
    monkeypatch.setattr(validation.pq, "read_table", reader)
    monkeypatch.setattr(validation.ray, "is_initialized", Mock(side_effect=AssertionError("Ray queried")))
    result = validate_page_budget_outputs(str(manifest), paths, expected_output_version=TEST_OUTPUT_VERSION)
    assert result == ACCEPTED
    assert [call.args[0] for call in reader.call_args_list] == paths


@pytest.mark.parametrize("empty", [False, True])
def test_completely_missing_documents_are_rejected(tmp_path: Path, empty: bool):
    manifest, _, _, files = outputs(tmp_path)
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        validate_page_budget_outputs(
            str(manifest), [] if empty else [files[0].data[0]], expected_output_version=TEST_OUTPUT_VERSION
        )


def test_replayed_writer_uses_same_filename_after_fanin_task_id_changes(tmp_path: Path):
    _, writer, documents, files = outputs(tmp_path)
    documents[0].task_id = "random-fanin-task-id"
    replay = writer.process(documents[0])
    assert replay.data == files[0].data
    assert len(list((tmp_path / "output").glob("*.parquet"))) == 2


def test_swapped_source_version_is_not_accepted_as_same_document(tmp_path: Path):
    manifest, _, _, files = outputs(tmp_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[0]["source_version"] = "new-bytes"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="Unexpected"):
        validate_page_budget_outputs(
            str(manifest), [task.data[0] for task in files], expected_output_version=TEST_OUTPUT_VERSION
        )


def test_duplicate_delivered_documents_fail(tmp_path: Path):
    manifest, _, _, files = outputs(tmp_path)
    with pytest.raises(ValueError, match="Duplicate output paths"):
        validate_page_budget_outputs(
            str(manifest), [files[0].data[0]] * 2, expected_output_version=TEST_OUTPUT_VERSION
        )


def test_readback_resolves_configured_columns_from_the_original_record(tmp_path: Path):
    manifest, _, _, files = outputs(tmp_path)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    for record in records:
        record["local_name"] = record["file_name"]
        record["file_name"] = record.pop("url")
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = validate_page_budget_outputs(
        str(manifest),
        [task.data[0] for task in files],
        expected_output_version=TEST_OUTPUT_VERSION,
        file_name_field="local_name",
        url_field="file_name",
    )
    assert result["coverage_passed"]


def test_persisted_page_omission_is_rejected_independently(tmp_path: Path):
    manifest, writer, documents, files = outputs(tmp_path)
    rows = documents[0].to_pyarrow().to_pylist()
    rows = [row for row in rows if row.get("page_number") != 1]
    for i, row in enumerate(rows[1:]):
        row["position"] = i
    documents[0].data = pa.Table.from_pylist(rows, schema=documents[0].data.schema)
    writer.process(documents[0])
    with pytest.raises(ValueError, match="outcome"):
        validate_page_budget_outputs(
            str(manifest), [task.data[0] for task in files], expected_output_version=TEST_OUTPUT_VERSION
        )


def test_driver_validation_checks_even_an_empty_result_list(tmp_path: Path):
    manifest, _, _, _ = outputs(tmp_path)
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        PDFPageBudgetValidationStage(
            manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION
        ).process_batch([])


def test_driver_validation_requires_writer_file_references(tmp_path: Path):
    manifest, _, documents, _ = outputs(tmp_path)
    with pytest.raises(TypeError, match="FileGroupTask outputs from a writer"):
        PDFPageBudgetValidationStage(
            manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION
        ).process_batch(documents)


@pytest.mark.parametrize("mixed", [False, True])
def test_wrong_or_mixed_output_version_is_rejected(tmp_path: Path, mixed: bool):
    manifest, writer, documents, files = outputs(tmp_path)
    for document in documents[:1] if mixed else documents:
        rows = document.data.to_pylist()
        info = json.loads(rows[0]["text_content"])
        info["pdf_provenance"]["output_version"] = "0" * 64
        rows[0]["text_content"] = json.dumps(info)
        document.data = pa.Table.from_pylist(rows, schema=document.data.schema)
        writer.process(document)
    with pytest.raises(ValueError, match="expected_output_version"):
        validate_page_budget_outputs(
            str(manifest), [task.data[0] for task in files], expected_output_version=TEST_OUTPUT_VERSION
        )


@pytest.mark.parametrize("stream_page_tasks", [False, True])
def test_finalizer_reads_local_files_directly_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream_page_tasks: bool
):
    manifest, _, _, files = outputs(tmp_path)
    reader = Mock(wraps=pq.read_table)
    monkeypatch.setattr(validation.pq, "read_table", reader)
    stage = PDFPageBudgetValidationStage(
        manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=stream_page_tasks
    )
    result = stage.process_batch(files)
    assert all(task._metadata["pdf_page_budget_validation"] == ACCEPTED for task in result)
    assert [call.args[0] for call in reader.call_args_list] == (
        [] if stream_page_tasks else [t.data[0] for t in files]
    )


@pytest.mark.parametrize("layout", ["file_uri", "hive_segment", "directory"])
def test_other_layouts_stay_with_the_dataset_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str):
    manifest, _, _, files = outputs(tmp_path)
    paths = [task.data[0] for task in files]
    if layout == "file_uri":
        paths = [Path(path).as_uri() for path in paths]
    elif layout == "hive_segment":
        paths = [copy(path, tmp_path / "run=1" / Path(path).name) for path in paths]
    else:
        paths = [
            str(Path(copy(path, tmp_path / str(index) / "part.parquet")).parent) for index, path in enumerate(paths)
        ]
    monkeypatch.setattr(validation.pq, "ParquetFile", Mock(side_effect=AssertionError("direct reader used")))
    assert (
        validate_page_budget_outputs(
            str(manifest), paths, expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
        )
        == ACCEPTED
    )


FAULTS = {
    "extra document": (ValueError, "Unexpected or duplicate delivered PDF"),
    "same document at another path": (ValueError, "Unexpected or duplicate delivered PDF"),
    "repeated page outcome": (ValueError, "Invalid page outcomes"),
    "file absent from disk": (FileNotFoundError, r"absent\.parquet"),
    "truncated file": (pa.ArrowInvalid, r"truncated\.parquet.*Parquet magic bytes not found"),
    "missing required column": (pa.ArrowInvalid, r"No match for FieldRef\.Name\(materialize_error\)"),
    "hive segment named like a file column": (pa.ArrowTypeError, "sample_id"),
}


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("stream_page_tasks", [False, True])
def test_faults_keep_the_error_of_the_dataset_reader_loop(tmp_path: Path, fault: str, stream_page_tasks: bool):
    manifest, writer, documents, files = outputs(tmp_path)
    first, second = (task.data[0] for task in files)
    paths, max_pdfs = [first, second], None
    if fault == "extra document":
        max_pdfs = 1
    elif fault == "same document at another path":
        paths.append(copy(first, tmp_path / "copy.parquet"))
    elif fault == "repeated page outcome":
        rows = documents[0].data.to_pylist()
        info = json.loads(rows[0]["text_content"])
        info["page_outcomes"][1] = info["page_outcomes"][0]
        rows[0]["text_content"] = json.dumps(info)
        documents[0].data = pa.Table.from_pylist(rows, schema=documents[0].data.schema)
        writer.process(documents[0])
    elif fault == "file absent from disk":
        paths[1] = str(tmp_path / "absent.parquet")
    elif fault == "truncated file":
        paths[1] = str(tmp_path / "truncated.parquet")
        Path(paths[1]).write_bytes(b"PAR1truncated")
    elif fault == "missing required column":
        paths[0] = str(tmp_path / "narrow.parquet")
        pq.write_table(pq.read_table(first).drop(["materialize_error"]), paths[0])
    else:
        paths[0] = copy(first, tmp_path / "sample_id=other" / "document.parquet")
    error_type, message = FAULTS[fault]
    with pytest.raises(error_type, match=message) as raised:
        validate_page_budget_outputs(
            str(manifest),
            paths,
            expected_output_version=TEST_OUTPUT_VERSION,
            max_pdfs=max_pdfs,
            stream_page_tasks=stream_page_tasks,
        )
    assert type(raised.value) is error_type


@pytest.fixture
def ready_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation.ray, "available_resources", lambda: {"CPU": 2})

    def completed_share(reference: validation.ray.ObjectRef | None) -> list[validation._LocalReceipt | None] | None:
        # Materialize the real worker result before exercising the ready-share branch.
        return None if reference is None else validation.ray.get(reference, timeout=60)

    monkeypatch.setattr(validation, "_ready_share", completed_share)


@pytest.mark.usefixtures("ready_workers")
def test_ray_tasks_reread_ready_shares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reread: str):
    manifest, _, _, files = outputs(tmp_path)
    first, second = [task.data[0] for task in files]
    read = validation._file_receipt
    driver = Mock(side_effect=read)
    monkeypatch.setattr(validation, "_file_receipt", driver)
    result = validate_page_budget_outputs(
        str(manifest), [first, second], expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
    )
    assert result == ACCEPTED
    assert [call.args[0] for call in driver.call_args_list] == ([first] if reread == "ray_tasks" else [first, second])


def test_cluster_without_cpus_rereads_in_the_calling_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, _, _, files = outputs(tmp_path)
    monkeypatch.setattr(validation.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(validation.ray, "available_resources", dict)
    monkeypatch.setattr(validation._share_receipts, "options", Mock(side_effect=AssertionError("task submitted")))
    result = validate_page_budget_outputs(
        str(manifest),
        [task.data[0] for task in files],
        expected_output_version=TEST_OUTPUT_VERSION,
        stream_page_tasks=True,
    )
    assert result == ACCEPTED


def test_relative_output_paths_use_the_calling_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, _, _, files = outputs(tmp_path)
    monkeypatch.setattr(validation.ray, "is_initialized", lambda: True)
    monkeypatch.chdir(tmp_path)
    paths = [str(Path(task.data[0]).relative_to(tmp_path)) for task in files]
    monkeypatch.setattr(validation._share_receipts, "options", Mock(side_effect=AssertionError("task submitted")))
    assert (
        validate_page_budget_outputs(
            str(manifest), paths, expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
        )
        == ACCEPTED
    )


def test_driver_takes_over_a_pending_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reread: str):
    manifest, _, _, files = outputs(tmp_path)
    paths = [task.data[0] for task in files]
    driver = Mock(side_effect=validation._file_receipt)
    monkeypatch.setattr(validation, "_file_receipt", driver)
    monkeypatch.setattr(validation.ray, "available_resources", lambda: {"CPU": 2})
    monkeypatch.setattr(validation.ray, "wait", Mock(return_value=([], [])))
    assert (
        validate_page_budget_outputs(
            str(manifest), paths, expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
        )
        == ACCEPTED
    )
    assert [call.args[0] for call in driver.call_args_list] == paths
    if reread == "ray_tasks":
        validation.ray.wait.assert_called_once()
        assert validation.ray.wait.call_args.kwargs["timeout"] == 0


@pytest.fixture
def worker_share(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(validation.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(validation.ray_client, "is_connected", lambda: False)
    monkeypatch.setattr(validation.ray, "available_resources", Mock(return_value={"CPU": 2}))
    monkeypatch.setattr(validation.ray, "get_runtime_context", lambda: Mock(get_node_id=lambda: "ab" * 28))
    reference = object()
    worker = Mock(remote=Mock(return_value=reference))
    monkeypatch.setattr(validation._share_receipts, "options", Mock(return_value=worker))
    monkeypatch.setattr(validation.ray, "wait", Mock(return_value=([reference], [])))
    monkeypatch.setattr(validation.ray, "get", Mock(return_value=[None]))
    return worker


@pytest.mark.parametrize("failed_step", ["resources", "submission", "worker_read"])
def test_unavailable_ray_read_uses_driver_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_step: str, worker_share: Mock
):
    manifest, _, _, files = outputs(tmp_path)
    paths = [task.data[0] for task in files]
    if failed_step == "resources":
        validation.ray.available_resources.side_effect = validation.RayError("unavailable")
    elif failed_step == "submission":
        worker_share.remote.side_effect = validation.RayError("unavailable")

    driver = Mock(side_effect=validation._file_receipt)
    monkeypatch.setattr(validation, "_file_receipt", driver)
    assert (
        validate_page_budget_outputs(
            str(manifest), paths, expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
        )
        == ACCEPTED
    )
    assert [call.args[0] for call in driver.call_args_list] == paths
    assert validation.ray.get.call_count == (1 if failed_step == "worker_read" else 0)


@pytest.mark.usefixtures("worker_share")
@pytest.mark.parametrize("change", ["removed", "replaced"])
def test_ready_worker_result_does_not_hide_a_changed_driver_file(tmp_path: Path, change: str):
    manifest, _, _, files = outputs(tmp_path)
    first, second = (task.data[0] for task in files)
    receipt = validation._local_receipt(second)
    assert receipt is not None
    validation.ray.get.return_value = [receipt]
    if change == "removed":
        Path(second).unlink()
        error, message = FileNotFoundError, Path(second).stem
    else:
        replacement = tmp_path / "replacement.parquet"
        shutil.copyfile(first, replacement)
        replacement.replace(second)
        error, message = ValueError, "Unexpected or duplicate delivered PDF"

    with pytest.raises(error, match=message):
        validate_page_budget_outputs(
            str(manifest), [first, second], expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
        )
    validation.ray.get.assert_called_once()


@pytest.mark.usefixtures("worker_share")
def test_ready_worker_result_needs_the_same_file_version_on_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest, _, _, files = outputs(tmp_path)
    paths = [task.data[0] for task in files]
    second = Path(paths[1])
    receipt = validation._local_receipt(str(second))
    assert receipt is not None
    validation.ray.get.return_value = [receipt]
    old = second.stat()
    os.utime(second, ns=(old.st_atime_ns, old.st_mtime_ns + 1))
    driver = Mock(wraps=validation._file_receipt)
    monkeypatch.setattr(validation, "_file_receipt", driver)

    result = validate_page_budget_outputs(
        str(manifest), paths, expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
    )
    assert result == ACCEPTED
    assert [call.args[0] for call in driver.call_args_list] == paths


@pytest.mark.usefixtures("shared_ray_client", "ready_workers")
def test_multi_file_share_keeps_the_first_file_error(tmp_path: Path):
    manifest, _, _, files = outputs(tmp_path)
    missing = tmp_path / "missing.parquet"
    truncated = tmp_path / "truncated.parquet"
    truncated.write_bytes(b"PAR1truncated")
    paths = [*[task.data[0] for task in files], str(missing), str(truncated)]
    with pytest.raises(FileNotFoundError, match=r"missing\.parquet") as error:
        validate_page_budget_outputs(
            str(manifest), paths, expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks=True
        )
    assert type(error.value) is FileNotFoundError


@pytest.mark.parametrize("unreadable_first", [True, False])
@pytest.mark.parametrize("stream_page_tasks", [False, True])
def test_first_failure_in_delivery_order_is_raised(tmp_path: Path, unreadable_first: bool, stream_page_tasks: bool):
    manifest, _, _, files = outputs(tmp_path)
    unexpected = files[1].data[0]
    unreadable = str(tmp_path / "truncated.parquet")
    Path(unreadable).write_bytes(b"PAR1truncated")
    paths = [unreadable, unexpected] if unreadable_first else [unexpected, unreadable]
    error_type, message = (
        (pa.ArrowInvalid, "Parquet magic bytes not found")
        if unreadable_first
        else (ValueError, "Unexpected or duplicate delivered PDF")
    )
    with pytest.raises(error_type, match=message) as raised:
        validate_page_budget_outputs(
            str(manifest),
            paths,
            expected_output_version=TEST_OUTPUT_VERSION,
            max_pdfs=1,
            stream_page_tasks=stream_page_tasks,
        )
    assert type(raised.value) is error_type


def test_stream_page_tasks_requires_a_boolean():
    with pytest.raises(TypeError, match="stream_page_tasks must be a boolean"):
        validate_page_budget_outputs(
            "unused.jsonl", [], expected_output_version=TEST_OUTPUT_VERSION, stream_page_tasks="false"
        )
