# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent manifest-to-Parquet reconciliation for bounded PDF experiments."""

from __future__ import annotations

import math
import os
import stat
from contextlib import closing, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import ray
from ray.exceptions import RayError
from ray.util.client import ray as ray_client
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import (
    SHA256_HEX_LENGTH,
    identity_digest,
    iter_manifest_documents,
    output_inventory_scope,
    selected_pages,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import validate_document_rows
from nemo_curator.tasks import FileGroupTask

if TYPE_CHECKING:
    from collections.abc import Generator

_COLUMNS = [
    "sample_id",
    "url",
    "pdf_name",
    "document_id",
    "position",
    "modality",
    "page_number",
    "text_content",
    "materialize_error",
]
_FileIdentity = tuple[int, int, int, int, int]
_LocalReceipt = tuple[_FileIdentity, dict[str, Any]]


def _check_output_version(expected_output_version: str) -> None:
    if (
        not isinstance(expected_output_version, str)
        or len(expected_output_version) != SHA256_HEX_LENGTH
        or any(c not in "0123456789abcdef" for c in expected_output_version)
    ):
        msg = "Final validation requires the admitted expected_output_version SHA256"
        raise ValueError(msg)


def _open_nonblocking(path: str, flags: int) -> int:
    # A path replaced by a FIFO must not block the driver while it checks a worker's read.
    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0))


def _file_identity(fd: int) -> _FileIdentity | None:
    status = os.fstat(fd)
    if not stat.S_ISREG(status.st_mode):
        return None
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns


def _receipt(table: pa.Table) -> dict[str, Any]:
    """Check the published rows and keep what manifest reconciliation compares."""
    rows = table.to_pylist()
    info, _ = validate_document_rows(rows)
    outcomes = info.pop("page_outcomes")
    urls: list[Any] = []
    for row in rows:
        if row["url"] not in urls:
            urls.append(row["url"])
    return {
        "info": info,
        "pages": [outcome["page_number"] for outcome in outcomes],
        "failed": sum(outcome["status"] == "failed" for outcome in outcomes),
        "successful": sum(outcome["status"] == "success" for outcome in outcomes),
        "sample_id": rows[0]["sample_id"],
        "pdf_name": rows[0]["pdf_name"],
        "urls": urls,
    }


def _local_receipt(path: str) -> _LocalReceipt | None:
    # Only the dataset reader interprets URI and Hive paths and diagnoses missing columns.
    if not isinstance(path, str) or "://" in path or "=" in path:
        return None
    try:
        with open(path, "rb", opener=_open_nonblocking) as file:
            identity = _file_identity(file.fileno())
            if identity is None:
                return None
            with pq.ParquetFile(file) as reader:
                table = reader.read(columns=_COLUMNS, use_threads=False)
                if table.column_names != _COLUMNS or _file_identity(file.fileno()) != identity:
                    return None
    except (OSError, pa.ArrowException):
        return None
    return identity, _receipt(table)


def _file_receipt(path: str, *, direct_read: bool = True) -> dict[str, Any]:
    if direct_read and (local := _local_receipt(path)) is not None:
        return local[1]
    return _receipt(pq.read_table(path, columns=_COLUMNS))


def _driver_receipt(path: str, worker_result: _LocalReceipt | None) -> dict[str, Any]:
    if worker_result is not None:
        identity, receipt = worker_result
        with suppress(OSError), open(path, "rb", opener=_open_nonblocking) as file:
            if _file_identity(file.fileno()) == identity:
                return receipt
    return _file_receipt(path)


@ray.remote
def _share_receipts(paths: list[str]) -> list[_LocalReceipt | None]:
    """Read an ordered share. The driver retries any unreadable file on its own filesystem."""
    receipts: list[_LocalReceipt | None] = []
    for path in paths:
        try:
            receipts.append(_local_receipt(path))
        except Exception:  # noqa: BLE001 - Reproduce any file error in the calling process.
            receipts.append(None)
    return receipts


def _ready_share(reference: ray.ObjectRef | None) -> list[_LocalReceipt | None] | None:
    if reference is None:
        return None
    with suppress(RayError):
        ready, _ = ray.wait([reference], timeout=0)
        if ready:
            return ray.get(reference, timeout=0)
    with suppress(RayError):
        ray.cancel(reference)
    return None


def _concurrent_receipts(output_paths: list[str]) -> Generator[dict[str, Any], None, None]:
    """Read in delivery order, accepting only worker results ready at the driver's turn."""
    if (
        len(output_paths) <= 1
        or not ray.is_initialized()
        or ray_client.is_connected()
        or not all(isinstance(path, str) and os.path.isabs(path) for path in output_paths)
    ):
        yield from map(_file_receipt, output_paths)
        return
    try:
        cpus = int(ray.available_resources().get("CPU", 0))
    except RayError:
        cpus = 0
    if cpus <= 1:
        yield from map(_file_receipt, output_paths)
        return

    size = math.ceil(len(output_paths) / cpus)
    shares = [output_paths[start : start + size] for start in range(0, len(output_paths), size)]
    pending = {}
    try:
        # Output paths are interpreted on the driver's filesystem, including paths on local disks.
        with suppress(RayError):
            reader = _share_receipts.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                )
            )
            for index, share in enumerate(shares[1:], start=1):
                pending[index] = reader.remote(share)

        for index, share in enumerate(shares):
            receipts = _ready_share(pending.pop(index, None))
            if receipts is None:
                yield from map(_file_receipt, share)
            else:
                for path, receipt in zip(share, receipts, strict=True):
                    yield _driver_receipt(path, receipt)
    finally:
        for reference in pending.values():
            with suppress(RayError):
                ray.cancel(reference)


def validate_page_budget_outputs(  # noqa: PLR0913
    manifest_path: str,
    output_paths: list[str],
    *,
    expected_output_version: str,
    max_pdfs: int | None = None,
    file_name_field: str = "file_name",
    url_field: str = "url",
    stream_page_tasks: bool = False,
) -> dict[str, Any]:
    """Reconcile all expected documents, even those with no surviving chunks.

    Read output columns without binary payloads. Coverage is a separate gate
    from model/extraction quality: retained failed pages are accounted for,
    never converted into successful pages.

    By default the driver reads every file in delivery order. With ``stream_page_tasks=True``,
    local files can be opened directly, and Ray tasks on the driver's node can share the reads.
    """
    _check_output_version(expected_output_version)
    if type(stream_page_tasks) is not bool:
        msg = "stream_page_tasks must be a boolean"
        raise TypeError(msg)
    expected = {
        doc["document_id"]: {**doc, "pages": selected_pages(doc["entry"])}
        for doc in iter_manifest_documents(manifest_path, max_pdfs, file_name_field)
    }
    seen, failed, successful = set(), 0, 0
    if len(output_paths) != len(set(output_paths)):
        msg = "Duplicate output paths in delivered inventory"
        raise ValueError(msg)
    receipts = (
        _concurrent_receipts(output_paths)
        if stream_page_tasks
        else (_file_receipt(path, direct_read=False) for path in output_paths)
    )
    with closing(receipts):
        for receipt in receipts:
            info = receipt["info"]
            provenance = info["pdf_provenance"]
            if provenance.get("output_version") != expected_output_version:
                msg = "Delivered PDF extraction version differs from the admitted expected_output_version"
                raise ValueError(msg)
            key = provenance["document_id"]
            if key not in expected or key in seen:
                msg = f"Unexpected or duplicate delivered PDF: {key}"
                raise ValueError(msg)
            wanted = expected[key]
            if (
                any(provenance[field] != wanted[field] for field in ("source_locator", "source_version"))
                or provenance["selection_sha256"] != identity_digest(wanted["pages"])
                or receipt["pages"] != wanted["pages"]
                or receipt["sample_id"] != wanted["sample"]
                or receipt["pdf_name"] != wanted["entry"][file_name_field]
                or info["url"] != wanted["entry"].get(url_field, "")
                or any(url != wanted["entry"].get(url_field, "") for url in receipt["urls"])
            ):
                msg = f"Delivered PDF does not match original expected inventory: {key}"
                raise ValueError(msg)
            seen.add(key)
            failed += receipt["failed"]
            successful += receipt["successful"]
    if seen != set(expected):
        msg = f"Missing whole PDFs in delivered inventory: {sorted(set(expected) - seen)!r}"
        raise ValueError(msg)
    return {
        "coverage_passed": True,
        "documents": len(expected),
        "pages": successful + failed,
        "successful_pages": successful,
        "failed_pages": failed,
        "all_pages_succeeded": failed == 0,
        "extraction_quality_validated": False,
        "output_version": expected_output_version,
    }


@dataclass
class PDFPageBudgetValidationStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Reconcile all document file references on the driver, including no results."""

    manifest_path: str
    expected_output_version: str
    max_pdfs: int | None = None
    file_name_field: str = "file_name"
    url_field: str = "url"
    stream_page_tasks: bool = field(default=False, kw_only=True)
    name: str = "pdf_page_budget_validation"
    is_resumable: bool = False
    finalizes: ClassVar[str] = "pdf_page_budget_validation"

    def __post_init__(self) -> None:
        _check_output_version(self.expected_output_version)

    @property
    def finalizes_scope(self) -> tuple[str, str, int | None, str]:
        return output_inventory_scope(self.manifest_path, self.max_pdfs, self.file_name_field)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"run_on_driver": True}

    def process(self, _task: FileGroupTask) -> FileGroupTask:
        msg = "PDFPageBudgetValidationStage requires driver validation with RayDataExecutor"
        raise RuntimeError(msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        if any(not isinstance(task, FileGroupTask) for task in tasks):
            msg = "PDFPageBudgetValidationStage requires FileGroupTask outputs from a writer"
            raise TypeError(msg)
        result = validate_page_budget_outputs(
            self.manifest_path,
            [path for task in tasks for path in task.data],
            expected_output_version=self.expected_output_version,
            max_pdfs=self.max_pdfs,
            file_name_field=self.file_name_field,
            url_field=self.url_field,
            stream_page_tasks=self.stream_page_tasks,
        )
        self._log_metrics(
            {"validated_documents": float(result["documents"]), "validated_pages": float(result["pages"])}
        )
        for task in tasks:
            task._metadata["pdf_page_budget_validation"] = result
        return tasks
