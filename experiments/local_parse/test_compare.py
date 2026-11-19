# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLR2004, S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa
from PIL import Image

from experiments.local_parse.compare import compare_shards
from experiments.local_parse.shards import _sha256, run_shard
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import image_to_bytes
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_OUTPUT_VERSION, partition, project

if TYPE_CHECKING:
    from pathlib import Path


def test_compares_published_rows_status_and_common_timing(tmp_path: Path) -> None:
    input_manifest, tasks = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 5}], 2)

    def publish(arm: str, failed: set[int], seconds: float, changed: bool) -> Path:
        def write(attempt: Path) -> list[str]:
            chunks = [chunk for task in tasks for chunk in project(task, failed=failed, no_elements={4})]
            document = PDFDocumentMergeStage().process_batch(chunks)[0]
            if changed:
                rows = document.data.to_pylist()
                for row in rows:
                    if row["page_number"] == 1 and row["modality"] == "text":
                        row["text_content"] = "different text"
                        row["source_ref"] = json.dumps({"page": 1, "bbox": [0.2, 0.1, 0.9, 0.9]})
                    if row["page_number"] == 1 and row["modality"] == "image":
                        row["binary_content"] = image_to_bytes(Image.new("RGB", (16, 16), (0, 5, 0)))
                document.data = pa.Table.from_pylist(rows, schema=document.data.schema)
            writer = InterleavedParquetWriterStage(
                path=str(attempt / "parquet"),
                materialize_on_write=False,
                on_materialize_error="warn",
                use_output_file_id=True,
                atomic_local_write=True,
            )
            return writer.process(document).data

        run = tmp_path / arm
        (run / "outputs").mkdir(parents=True)
        (run / "controller-result.json").write_text(
            json.dumps(
                {
                    "status": "STRUCTURALLY_COMPLETE",
                    "pages": 5,
                    "manifest_sha256": _sha256(input_manifest),
                    "timing_eligible": True,
                    "wall_seconds_including_engine_startup_and_readback": seconds,
                }
            ),
            encoding="utf-8",
        )
        result = run_shard(
            shard_id="smoke",
            output_root=run / "outputs",
            manifest_path=input_manifest,
            expected_output_version=TEST_OUTPUT_VERSION,
            run_attempt=write,
        )
        return result.manifest_path

    first = publish("reference", {3}, 20, changed=False)
    second = publish("candidate", {2}, 10, changed=True)
    compared = compare_shards(first, second.parent.parent.parent, input_manifest)
    assert compared["physical_pages"] == 5
    assert compared["changed_pages"] == 3
    assert compared["exact_page_matches"] == 2
    assert compared["new_failures"] == compared["recovered_failures"] == 1
    assert compared["candidate"]["page_counts"] == {"populated": 3, "empty": 1, "failed": 1}
    assert compared["candidate"]["controller_startup_through_readback"]["populated_pages_per_second"] == 0.3
    changed = next(item for item in compared["changed_page_examples"] if item["physical_page_zero_based"] == 1)
    assert changed["changes"] == ["text", "source_refs_and_bboxes", "picture_payloads"]
