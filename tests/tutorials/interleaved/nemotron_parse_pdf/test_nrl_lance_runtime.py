# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# ruff: noqa: ANN401, INP001

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

TUTORIAL_DIR = Path(__file__).resolve().parents[4] / "tutorials" / "interleaved" / "nemotron_parse_pdf"
sys.path.insert(0, str(TUTORIAL_DIR))

import nrl_lance_contract as contract  # noqa: E402
import nrl_lance_runtime as runtime  # noqa: E402


@pytest.mark.parametrize("option", ["parse_batch_size", "parse_cpus"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "4"])
def test_ingest_rejects_invalid_parse_scheduling_before_source_access(option: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        runtime.run_ingest(argparse.Namespace(**{option: value}))


def test_ingest_rejects_silently_promoted_parse_batch_one() -> None:
    with pytest.raises(ValueError, match="pinned NRL executor promotes batch size 1 to 64"):
        runtime.run_ingest(argparse.Namespace(parse_batch_size=1))


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "16"])
def test_ingest_rejects_invalid_projection_block_rows_before_source_access(value: Any) -> None:
    with pytest.raises(ValueError, match="projection_block_rows must be a positive integer"):
        runtime.run_ingest(argparse.Namespace(projection_block_rows=value))


def test_ingest_rejects_projection_block_rows_inprocess_before_source_access() -> None:
    with pytest.raises(ValueError, match="projection_block_rows requires batch mode"):
        runtime.run_ingest(argparse.Namespace(run_mode="inprocess", projection_block_rows=16))


def test_ingest_rejects_nondefault_scheduling_for_inprocess_before_source_access() -> None:
    with pytest.raises(ValueError, match="require batch mode"):
        runtime.run_ingest(argparse.Namespace(run_mode="inprocess", parse_cpus=4))


@pytest.fixture
def element_rows(tmp_path: Path) -> list[dict[str, Any]]:
    image = io.BytesIO()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image, format="PNG")
    source = tmp_path / "source.pdf"
    aliases = [{"path": str(source), "url": None, "input_index": 0, "valid_blank_pages": []}]
    provenance = {
        "content_sha256": "a" * 64,
        "pdf_name": source.name,
        "num_pages": 1,
        "source_path": str(source),
        "source_aliases": aliases,
        "url": None,
        "valid_blank_pages": [],
    }
    common = {
        **dict.fromkeys(contract.element_schema().names),
        "sample_id": "a" * 64,
        "content_sha256": "a" * 64,
        "pdf_name": source.name,
        "source_path": str(source),
        "source_aliases": json.dumps(aliases),
        "run_id": "test",
    }
    metadata = {
        **common,
        "position": -1,
        "modality": "metadata",
        "content_type": "application/json",
        "text_content": json.dumps(provenance),
    }
    content = {
        **common,
        "page_number": 0,
        "bbox_xyxy_norm": [0.125, 0.25, 0.5, 0.75],
        "bbox_coordinate_space": contract.COORDINATE_SPACE,
    }
    return [
        metadata,
        {
            **content,
            "position": 0,
            "modality": "text",
            "element_class": "Text",
            "content_type": "text/markdown",
            "text_content": "hello",
        },
        {
            **content,
            "position": 1,
            "modality": "table",
            "element_class": "Table",
            "content_type": "text/markdown",
            "text_content": (
                r"\begin{tabular}{lrr}"
                "\n"
                r"\multicolumn{3}{c}{Coverage <br> and <unknown>} \\"
                "\n"
                r"\multirow{2}{*}{North <sup>1</sup>} & 4 & 5 \\"
                "\n"
                r" & 6 & 7 \\"
                "\n"
                r"\end{tabular}"
            ),
        },
        {
            **content,
            "position": 2,
            "modality": "image",
            "element_class": "Picture",
            "content_type": "image/png",
            "text_content": "",
            "binary_content": image.getvalue(),
        },
    ]


@pytest.fixture
def published_table(tmp_path: Path, element_rows: list[dict[str, Any]]) -> tuple[Path, int]:
    source = tmp_path / "lance"
    source.mkdir()
    writer = contract.ElementTableWriter(source)
    writer.add_document(element_rows)
    path = contract._table_path(source)
    return path, int(contract._open_lancedb_table(path).version)


def _write_output(tmp_path: Path, rows: list[dict[str, Any]], schema: pa.Schema | None = None) -> Path:
    output = tmp_path / "output"
    output.mkdir()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema or contract.element_schema()), output / "part.parquet")
    return output


@pytest.mark.parametrize("field_name", contract.element_schema().names)
def test_export_rejects_same_values_with_wrong_logical_type(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    field_name: str,
) -> None:
    schema = contract.element_schema()
    field = schema.field(field_name)
    replacements = {
        pa.string(): pa.large_string(),
        pa.int32(): pa.int64(),
        pa.large_binary(): pa.binary(),
        pa.list_(pa.float64()): pa.list_(pa.float32()),
    }
    changed = schema.set(schema.get_field_index(field_name), field.with_type(replacements[field.type]))
    output = _write_output(tmp_path, element_rows, changed)
    with pytest.raises(ValueError, match=f"Parquet .* field '{field_name}'"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("field_name", [field.name for field in contract.element_schema() if not field.nullable])
def test_export_rejects_same_values_with_relaxed_nullability(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    field_name: str,
) -> None:
    schema = contract.element_schema()
    changed = schema.set(schema.get_field_index(field_name), schema.field(field_name).with_nullable(True))
    output = _write_output(tmp_path, element_rows, changed)
    with pytest.raises(ValueError, match=f"Parquet .* field '{field_name}'"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("field_name", contract.element_schema().names)
def test_schema_validation_checks_every_fields_nullability(field_name: str) -> None:
    schema = contract.element_schema()
    field = schema.field(field_name)
    changed = schema.set(schema.get_field_index(field_name), field.with_nullable(not field.nullable))
    with pytest.raises(ValueError, match=f"field '{field_name}'"):
        runtime._validate_element_schema(changed, label="export")


@pytest.mark.parametrize(
    ("row_index", "field", "replacement"),
    [
        (1, "text_content", "changed"),
        (1, "text_content", None),
        (1, "bbox_xyxy_norm", [0.0, 0.0, 0.5, 0.5]),
        (1, "bbox_coordinate_space", "pixels"),
        (1, "source_path", "/different/source.pdf"),
        (1, "source_aliases", "[]"),
        (1, "source_ref", '{"path":"wrong"}'),
        (1, "materialize_error", "unreported error"),
        (1, "content_sha256", "b" * 64),
        (1, "page_number", 1),
        (1, "pdf_name", "different.pdf"),
        (1, "url", "https://example.test/different"),
        (1, "element_class", "Title"),
        (1, "modality", "table"),
        (1, "content_type", "text/plain"),
        (1, "run_id", "another-run"),
        (2, "text_content", "| changed |"),
        (3, "binary_content", b"different image bytes"),
        (3, "binary_content", None),
    ],
)
def test_export_rejects_changed_content_or_provenance(  # noqa: PLR0913
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    row_index: int,
    field: str,
    replacement: Any,
) -> None:
    changed = copy.deepcopy(element_rows)
    changed[row_index][field] = replacement
    output = _write_output(tmp_path, changed)
    with pytest.raises(ValueError, match="differs from the pinned Lance source"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unexpected", "reordered"])
def test_export_rejects_changed_keys_or_order(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    change: str,
) -> None:
    changed = copy.deepcopy(element_rows)
    if change == "missing":
        changed.pop()
    elif change == "duplicate":
        changed.append(changed[-1])
    elif change == "unexpected":
        changed[-1]["sample_id"] = "b" * 64
    else:
        changed[1], changed[2] = changed[2], changed[1]
    output = _write_output(tmp_path, changed)
    with pytest.raises(ValueError, match=r"row keys differ|duplicate row key|positions or order"):
        runtime._validate_consumed_output(*published_table, output)


def test_export_validates_every_parquet_file(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
) -> None:
    output = _write_output(tmp_path, element_rows[:2])
    schema = contract.element_schema()
    changed = schema.set(0, schema.field(0).with_nullable(True))
    pq.write_table(pa.Table.from_pylist(element_rows[2:], schema=changed), output / "second.parquet")
    with pytest.raises(ValueError, match=r"second\.parquet.*field 'sample_id'"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("change", ["missing", "extra", "reordered"])
def test_export_rejects_changed_field_layout(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    change: str,
) -> None:
    schema = contract.element_schema()
    if change == "missing":
        schema = schema.remove(8)
    elif change == "extra":
        schema = schema.append(pa.field("unexpected", pa.string()))
    else:
        fields = list(schema)
        fields[0], fields[1] = fields[1], fields[0]
        schema = pa.schema(fields)
    output = _write_output(tmp_path, element_rows, schema)
    with pytest.raises(ValueError, match=r"fields are .*expected"):
        runtime._validate_consumed_output(*published_table, output)


@pytest.mark.parametrize("change", ["content", "schema", "missing"])
def test_native_reader_cannot_change_validated_stored_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
    change: str,
) -> None:
    pytest.importorskip("nemo_curator", reason="Native reader tests run in the pinned Curator environment")
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import InterleavedBatch

    output = _write_output(tmp_path, element_rows)
    changed = copy.deepcopy(element_rows)
    schema = contract.element_schema()
    if change == "content":
        changed[1]["text_content"] = "reader changed content"
    elif change == "schema":
        schema = schema.set(17, schema.field(17).with_nullable(True))
    else:
        changed.pop()
    batch = InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(changed, schema=schema))
    monkeypatch.setattr(Pipeline, "run", lambda _self, _executor: [batch])
    with pytest.raises(ValueError, match="Curator native Parquet reader"):
        runtime._validate_consumed_output(*published_table, output)


def test_native_pipeline_preserves_exact_schema_and_reopens_export(
    tmp_path: Path,
    element_rows: list[dict[str, Any]],
    published_table: tuple[Path, int],
) -> None:
    pytest.importorskip("nemo_curator", reason="Native round trip runs in the pinned Curator environment")
    from nemo_curator.backends.ray_data import RayDataExecutor

    output = tmp_path / "output"
    pipeline = runtime._build_consume_pipeline(
        published_table[0], version=published_table[1], output_dir=output, mode="error"
    )
    pipeline.run(RayDataExecutor())
    report = runtime._validate_consumed_output(*published_table, output)

    assert report["output"]["row_count"] == len(element_rows)
    assert report["reconciliation"]["schema_and_nullability_validated"] is True
    assert report["reconciliation"]["native_parquet_reader_validated"] is True
    assert report["reconciliation"]["native_parquet_reader_seconds"] > 0
    exported_tables = []
    for path in output.rglob("*.parquet"):
        assert pq.read_schema(path).equals(contract.element_schema(), check_metadata=False)
        exported_tables.extend(row for row in pq.read_table(path).to_pylist() if row["modality"] == "table")
    assert exported_tables == [row for row in element_rows if row["modality"] == "table"]
