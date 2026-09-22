# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# ruff: noqa: ANN401, EM101, INP001, PLR0913

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

TUTORIAL_DIR = Path(__file__).resolve().parents[4] / "tutorials" / "interleaved" / "nemotron_parse_pdf"
sys.path.insert(0, str(TUTORIAL_DIR))

if importlib.util.find_spec("cosmos_xenna") is None:
    cosmos_xenna = ModuleType("cosmos_xenna")
    cosmos_xenna.__path__ = []  # type: ignore[attr-defined]
    ray_utils = ModuleType("cosmos_xenna.ray_utils")
    ray_utils.__path__ = []  # type: ignore[attr-defined]
    cluster = ModuleType("cosmos_xenna.ray_utils.cluster")
    cluster.API_LIMIT = 1000  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "cosmos_xenna": cosmos_xenna,
            "cosmos_xenna.ray_utils": ray_utils,
            "cosmos_xenna.ray_utils.cluster": cluster,
        }
    )

import nrl_lance_contract as contract  # noqa: E402
import nrl_lance_runtime as runtime  # noqa: E402


def _document(
    path: Path, digest: str = "a" * 64, *, pages: int = 1, blanks: list[int] | None = None
) -> dict[str, Any]:
    blank_pages = blanks or []
    alias = {
        "path": str(path.resolve()),
        "url": f"https://example.test/{path.name}",
        "input_index": 0,
        "valid_blank_pages": blank_pages,
    }
    return {
        "path": str(path.resolve()),
        "url": alias["url"],
        "content_sha256": digest,
        "expected_page_count": pages,
        "document_valid_blank_pages": blank_pages,
        "aliases": [alias],
    }


def _marker(
    path: Path,
    page: int,
    *,
    outcome: str = "parsed",
    count: int = 1,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "page_outcome",
        "source_path": str(path.resolve()),
        "native_page_number": page,
        "page_outcome": outcome,
        "element_count": count,
        "issues_json": json.dumps(issues or [], sort_keys=True, separators=(",", ":")),
        "raw_output_sha256": "b" * 64 if page > 0 else None,
        "element_index": None,
        "element_class": None,
        "modality": None,
        "content_type": None,
        "text_content": None,
        "binary_content": None,
        "bbox_xyxy_norm_json": None,
        "bbox_coordinate_space": None,
    }


def _element(
    path: Path,
    page: int,
    index: int,
    *,
    element_class: str = "Text",
    modality: str = "text",
    text: str = "hello",
    binary: bytes | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "record_type": "element",
        "source_path": str(path.resolve()),
        "native_page_number": page,
        "page_outcome": None,
        "element_count": None,
        "issues_json": "[]",
        "raw_output_sha256": None,
        "element_index": index,
        "element_class": element_class,
        "modality": modality,
        "content_type": "image/png" if modality == "image" else "text/markdown",
        "text_content": text,
        "binary_content": binary,
        "bbox_xyxy_norm_json": json.dumps(bbox or [0.1, 0.2, 0.4, 0.5]),
        "bbox_coordinate_space": contract.COORDINATE_SPACE,
    }


def _valid_png() -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    image_module.new("RGB", (16, 16), color=(255, 255, 255)).save(output, format="PNG")
    return output.getvalue()


def _provenance(document: dict[str, Any], run_id: str) -> dict[str, dict[str, Any]]:
    return {
        str(document["content_sha256"]): {
            "source_path": document["path"],
            "source_name": Path(document["path"]).name,
            "num_pages": document["expected_page_count"],
            "source_aliases": document["aliases"],
            "url": document["url"],
            "valid_blank_pages": document["document_valid_blank_pages"],
            "run_id": run_id,
        }
    }


def test_build_document_rows_preserves_model_and_page_order(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    document = _document(source, pages=2)
    rows = [
        _marker(source, 2, count=1),
        _element(source, 2, 0, element_class="Table", modality="table", text="| A |"),
        _marker(source, 1, count=2),
        _element(
            source,
            1,
            1,
            element_class="Picture",
            modality="image",
            text="",
            binary=b"\x89PNG\r\n\x1a\ntruncated",
        ),
        _element(source, 1, 0, text="first"),
    ]

    build = contract.build_document_rows(document, rows, run_id="run")

    assert build.status == "success"
    assert [row["position"] for row in build.rows] == [-1, 0, 1, 2]
    assert [row["page_number"] for row in build.rows[1:]] == [0, 0, 1]
    assert [row["element_class"] for row in build.rows[1:]] == ["Text", "Picture", "Table"]
    assert build.rows[2]["binary_content"] == b"\x89PNG\r\n\x1a\ntruncated"
    assert build.rows[2]["source_ref"] is None


def test_declared_all_blank_document_publishes_metadata_only(tmp_path: Path) -> None:
    source = tmp_path / "blank.pdf"
    document = _document(source, pages=2, blanks=[0, 1])
    rows = [_marker(source, 1, outcome="empty", count=0), _marker(source, 2, outcome="empty", count=0)]

    build = contract.build_document_rows(document, rows, run_id="run")

    assert build.status == "valid_blank"
    assert build.page_count == 2
    assert build.blank_page_count == 2
    assert len(build.rows) == 1
    assert build.rows[0]["position"] == -1


@pytest.mark.parametrize(
    ("rows", "pages", "blanks", "expected_status", "issue"),
    [
        ([], 1, [], "failed", "missing_pages"),
        ([_marker(Path("source.pdf"), 1), _marker(Path("source.pdf"), 1)], 1, [], "failed", "duplicate_page_outcomes"),
        ([_marker(Path("source.pdf"), 1, outcome="empty", count=0)], 1, [], "failed", "unexpected_empty_output"),
        ([_marker(Path("source.pdf"), 1, outcome="failed", count=0)], 1, [], "failed", "page_failed"),
        (
            [_marker(Path("source.pdf"), 1, count=2), _element(Path("source.pdf"), 1, 0)],
            1,
            [],
            "failed",
            "element_count_mismatch",
        ),
        ([_marker(Path("source.pdf"), 0, outcome="failed", count=0)], 1, [], "failed", "document_or_split_failure"),
    ],
)
def test_document_gate_rejects_incomplete_envelopes(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    pages: int,
    blanks: list[int],
    expected_status: str,
    issue: str,
) -> None:
    source = tmp_path / "source.pdf"
    for row in rows:
        row["source_path"] = str(source.resolve())
    build = contract.build_document_rows(_document(source, pages=pages, blanks=blanks), rows, run_id="run")
    assert build.status == expected_status
    assert build.rows == []
    assert issue in {item["kind"] for item in build.issues}


def test_one_valid_page_plus_failure_is_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    rows = [
        _marker(source, 1),
        _element(source, 1, 0),
        _marker(source, 2, outcome="failed", count=0, issues=[{"kind": "non_stop_finish"}]),
    ]
    build = contract.build_document_rows(_document(source, pages=2), rows, run_id="run")
    assert build.status == "partial"
    assert [row["page_number"] for row in build.rows] == [None, 0]
    metadata = json.loads(build.rows[0]["text_content"])
    assert metadata["extraction_status"] == "partial"
    assert metadata["page_outcomes"] == build.page_outcomes
    assert [page["status"] for page in build.page_outcomes] == ["success", "failed"]
    assert {issue["kind"] for issue in build.page_outcomes[1]["issues"]} == {"non_stop_finish", "page_failed"}


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "empty", "bbox", "marker", "element", "nested", "indices"]
)
def test_partial_delivery_keeps_only_whole_valid_pages(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "source.pdf"
    good = [_marker(source, 1), _element(source, 1, 0, text="keep first")]
    bad = [_marker(source, 2, count=2), _element(source, 2, 0, text="must not leak"), _element(source, 2, 1)]
    if failure == "missing":
        bad = []
    elif failure == "duplicate":
        bad.append(_marker(source, 2, count=2))
    elif failure == "empty":
        bad = [_marker(source, 2, outcome="empty", count=0)]
    elif failure == "bbox":
        bad[-1]["bbox_xyxy_norm_json"] = "[0.5,0.5,0.1,0.1]"
    elif failure == "marker":
        bad[0]["text_content"] = "invalid marker payload"
    elif failure == "element":
        bad[-1]["issues_json"] = '[{"kind":"nested_error"}]'
    elif failure == "nested":
        bad[0]["issues_json"] = '[{"kind":"nested_error"}]'
    elif failure == "indices":
        bad[-1]["element_index"] = 2
    rows = [*good, *bad, _marker(source, 3), _element(source, 3, 0, text="keep last")]
    build = contract.build_document_rows(_document(source, pages=3), rows, run_id="run")
    assert build.status == "partial"
    assert build.page_count == 2
    assert [row["position"] for row in build.rows] == [-1, 0, 1]
    assert [row["page_number"] for row in build.rows[1:]] == [0, 2]
    assert [row["text_content"] for row in build.rows[1:]] == ["keep first", "keep last"]
    assert build.page_outcomes[1]["status"] == "failed"
    assert build.page_outcomes[1]["element_count"] == 0
    assert build.page_outcomes[1]["issues"]


def test_partial_blank_only_document_keeps_coverage_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    build = contract.build_document_rows(
        _document(source, pages=2, blanks=[0]), [_marker(source, 1, outcome="empty", count=0)], run_id="run"
    )
    assert build.status == "partial"
    assert build.page_count == build.blank_page_count == 1
    assert len(build.rows) == 1
    assert [page["status"] for page in build.page_outcomes] == ["valid_blank", "failed"]


def test_declared_blank_does_not_hide_page_errors(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    build = contract.build_document_rows(
        _document(source, pages=2, blanks=[1]),
        [
            _marker(source, 1),
            _element(source, 1, 0),
            _marker(source, 2, outcome="empty", count=0, issues=[{"kind": "non_stop_finish"}]),
        ],
        run_id="run",
    )
    assert build.status == "partial"
    assert build.blank_page_count == 0
    assert build.page_outcomes[1]["status"] == "failed"


def test_extra_pages_are_reported_but_never_delivered(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    build = contract.build_document_rows(
        _document(source),
        [_marker(source, 1), _element(source, 1, 0), _marker(source, 2), _element(source, 2, 0)],
        run_id="run",
    )
    assert build.status == "partial"
    assert [row["page_number"] for row in build.rows] == [None, 0]
    assert {issue["kind"] for issue in build.issues} == {"unexpected_pages"}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "status", "count", "issues", "invalid_status"])
def test_page_outcome_contract_rejects_false_completeness(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source.pdf"
    build = contract.build_document_rows(
        _document(source, pages=2), [_marker(source, 1), _element(source, 1, 0)], run_id="run"
    )
    pages = json.loads(json.dumps(build.page_outcomes))
    status = build.status
    if mutation == "missing":
        pages.pop()
    elif mutation == "duplicate":
        pages[1]["page_number"] = 0
    elif mutation == "status":
        status = "success"
    elif mutation == "count":
        pages[1]["element_count"] = 1
    elif mutation == "issues":
        pages[1]["issues"] = []
    elif mutation == "invalid_status":
        pages[1]["status"] = []
    with pytest.raises(ValueError, match=r"coverage|outcomes|outcome|Extraction status"):
        contract.validate_page_outcomes(pages, expected_page_count=2, extraction_status=status, issues=build.issues)


def test_invalid_bbox_rejects_complete_document(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    element = _element(source, 1, 0)
    element["bbox_xyxy_norm_json"] = "[0.4,0.2,0.1,0.5]"
    build = contract.build_document_rows(_document(source), [_marker(source, 1), element], run_id="run")
    assert build.status == "failed"
    assert {item["kind"] for item in build.issues} >= {"invalid_bbox"}


def test_inventory_deduplicates_bytes_not_basenames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "one" / "report.pdf"
    alias = tmp_path / "alias" / "copy.pdf"
    same_name = tmp_path / "two" / "report.pdf"
    for path in (first, alias, same_name):
        path.parent.mkdir()
    first.write_bytes(b"same")
    alias.write_bytes(b"same")
    same_name.write_bytes(b"different")
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 3)

    inputs, representatives = contract.inventory_sources(
        [
            {"path": str(first), "url": None, "valid_blank_pages": [0]},
            {"path": str(alias), "url": None, "valid_blank_pages": [2]},
            {"path": str(same_name), "url": None, "valid_blank_pages": []},
        ]
    )

    assert len(representatives) == 2
    assert inputs[1]["status"] == "duplicate"
    assert inputs[0]["content_sha256"] == inputs[1]["content_sha256"]
    assert inputs[2]["content_sha256"] != inputs[0]["content_sha256"]
    assert representatives[0]["document_valid_blank_pages"] == [0, 2]


def test_inventory_rejects_out_of_range_blank_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = tmp_path / "one.pdf"
    alias = tmp_path / "alias.pdf"
    first.write_bytes(b"same")
    alias.write_bytes(b"same")
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 1)
    inputs, representatives = contract.inventory_sources(
        [
            {"path": str(first), "url": None, "valid_blank_pages": []},
            {"path": str(alias), "url": None, "valid_blank_pages": [1]},
        ]
    )
    assert representatives[0]["status"] == "failed"
    assert representatives[0]["preflight_error"]["type"] == "InvalidBlankPageDeclaration"
    assert inputs[1]["representative_status"] == "failed"


def test_manifest_rejects_pdf_symlink_that_resolves_outside_allowed_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    target = outside / "source.pdf"
    target.write_bytes(b"pdf")
    linked = allowed / "source.pdf"
    linked.symlink_to(target)
    manifest = allowed / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": str(linked)}) + "\n", encoding="utf-8")
    monkeypatch.setattr(contract, "ALLOWED_ROOT", allowed)

    with pytest.raises(ValueError, match="must resolve under"):
        contract._source_records_from_manifest(manifest)


def test_element_schema_is_exact() -> None:
    schema = contract.element_schema()
    assert schema.names == [
        "sample_id",
        "position",
        "modality",
        "content_type",
        "text_content",
        "binary_content",
        "source_ref",
        "materialize_error",
        "url",
        "page_number",
        "pdf_name",
        "element_class",
        "source_path",
        "source_aliases",
        "content_sha256",
        "bbox_xyxy_norm",
        "bbox_coordinate_space",
        "run_id",
    ]
    assert not schema.field("sample_id").nullable
    assert not schema.field("position").nullable
    assert not schema.field("modality").nullable
    assert not schema.field("content_sha256").nullable
    assert not schema.field("run_id").nullable


def test_lance_writer_keeps_each_document_in_one_fragment(tmp_path: Path) -> None:
    pytest.importorskip("lancedb")
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"
    first = _document(first_path, "1" * 64)
    second = _document(second_path, "2" * 64)
    first_rows = contract.build_document_rows(
        first,
        [_marker(first_path, 1), _element(first_path, 1, 0)],
        run_id="run",
    ).rows
    second_rows = contract.build_document_rows(
        second,
        [_marker(second_path, 1), _element(second_path, 1, 0)],
        run_id="run",
    ).rows
    writer = contract.ElementTableWriter(tmp_path)
    writer.add_document(first_rows)
    writer.add_document(second_rows)

    result = runtime.validate_element_table(
        contract._table_path(tmp_path),
        {"1" * 64: 2, "2" * 64: 2},
        expected_provenance={**_provenance(first, "run"), **_provenance(second, "run")},
    )
    assert result["row_count"] == 4
    assert result["document_count"] == 2
    assert result["fragment_count"] == 2


def test_table_validation_rejects_multiple_documents_in_one_fragment(tmp_path: Path) -> None:
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"
    first = _document(first_path, "1" * 64)
    second = _document(second_path, "2" * 64)
    first_rows = contract.build_document_rows(
        first,
        [_marker(first_path, 1), _element(first_path, 1, 0)],
        run_id="run",
    ).rows
    second_rows = contract.build_document_rows(
        second,
        [_marker(second_path, 1), _element(second_path, 1, 0)],
        run_id="run",
    ).rows
    connection = lancedb.connect(str(tmp_path))
    connection.create_table(
        contract.ELEMENT_TABLE,
        data=pa.Table.from_pylist(first_rows + second_rows, schema=contract.element_schema()),
        mode="create",
    )

    with pytest.raises(ValueError, match="multiple documents"):
        runtime.validate_element_table(
            contract._table_path(tmp_path),
            {"1" * 64: 2, "2" * 64: 2},
            expected_provenance={**_provenance(first, "run"), **_provenance(second, "run")},
        )


def _ingest_args(root: Path, manifest: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=None,
        manifest=str(manifest),
        output_root=str(root / "runs"),
        run_id="run",
        run_mode="batch",
        projection_workers=2,
        nrl_repo=str(root),
    )


@pytest.mark.parametrize("executor_stats", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("projection_block_rows", [None, 16])
@pytest.mark.parametrize(("parse_batch_size", "parse_cpus"), [(64, 1), (64, 4), (128, 1)])
def test_ingest_writes_handoff_only_after_validated_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    executor_stats: bool,
    partial: bool,
    parse_batch_size: int,
    parse_cpus: int,
    projection_block_rows: int | None,
) -> None:
    source = tmp_path / "source.pdf"
    alias = tmp_path / "alias.pdf"
    source.write_bytes(b"same-pdf")
    alias.write_bytes(b"same-pdf")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": str(source)}) + "\n" + json.dumps({"path": str(alias)}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(contract, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 2 if partial else 1)
    monkeypatch.setattr(runtime, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(runtime, "_git_dirty", lambda _path: False)
    monkeypatch.setattr(runtime, "_model_revision", lambda _model: "model-revision")

    stats_path = tmp_path / "runs" / "run" / "executor_stats.txt" if executor_stats else None

    def fake_graph(paths: list[str], **kwargs: Any) -> pd.DataFrame:
        assert paths == [str(source.resolve())]
        assert kwargs.get("executor_stats_path") == stats_path
        assert kwargs["parse_batch_size"] == parse_batch_size
        assert kwargs["parse_cpus"] == parse_cpus
        assert kwargs["projection_block_rows"] == projection_block_rows
        assert "evidence_root" not in kwargs
        return pd.DataFrame([_marker(source, 1), _element(source, 1, 0)], dtype=object)

    args = _ingest_args(tmp_path, manifest)
    args.executor_stats = executor_stats
    args.parse_batch_size = parse_batch_size
    args.parse_cpus = parse_cpus
    args.projection_block_rows = projection_block_rows
    handoff_path = runtime.run_ingest(
        args,
        graph_runner=fake_graph,
        envelope_validator=lambda value: value,
    )
    handoff = runtime._load_handoff_manifest(handoff_path)
    assert handoff["status"] == "tables_validated"
    status = "partial" if partial else "success"
    assert handoff["counts"]["inputs"] == {"duplicate": 1, status: 1}
    assert handoff["inputs"][1]["representative_status"] == status
    assert handoff["counts"]["complete_document_count"] == int(not partial)
    assert handoff["counts"]["partial_document_count"] == int(partial)
    assert handoff["counts"]["failed_page_count"] == int(partial)
    assert handoff["counts"]["delivered_page_count"] == 1
    assert handoff["counts"]["delivered_content_element_count"] == 1
    assert handoff["tables"][contract.ELEMENT_TABLE]["row_count"] == 2
    assert handoff["configuration"]["executor_stats_path"] == (str(stats_path) if stats_path else None)
    assert handoff["configuration"]["parse_batch_size"] == parse_batch_size
    assert handoff["configuration"]["parse_cpus"] == parse_cpus
    assert handoff["configuration"]["projection_block_rows"] == projection_block_rows
    assert handoff["benchmark_evidence"] is None
    assert not (handoff_path.parent / contract.COMPLETION_MANIFEST_FILE).exists()


def test_ingest_graph_failure_never_creates_publication_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": str(source)}) + "\n", encoding="utf-8")
    monkeypatch.setattr(contract, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 1)

    def failed_graph(_paths: list[str], **_kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("graph failed")

    with pytest.raises(RuntimeError, match="graph failed"):
        runtime.run_ingest(
            _ingest_args(tmp_path, manifest),
            graph_runner=failed_graph,
            envelope_validator=lambda value: value,
        )
    run_dir = tmp_path / "runs" / "run"
    assert not (run_dir / contract.HANDOFF_MANIFEST_FILE).exists()
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()
    assert contract._load_json(run_dir / contract.RUN_STATE_FILE)["status"] == "unpublished"


def test_evidence_directory_must_be_fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": str(source)}) + "\n", encoding="utf-8")
    evidence = tmp_path / "existing-evidence"
    evidence.mkdir()
    monkeypatch.setattr(contract, "ALLOWED_ROOT", tmp_path)
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 1)
    args = _ingest_args(tmp_path, manifest)
    args.evidence_root = str(evidence)
    with pytest.raises(FileExistsError):
        runtime.run_ingest(args)
    assert not (tmp_path / "runs/run" / contract.HANDOFF_MANIFEST_FILE).exists()
    assert list(evidence.iterdir()) == []


def test_structural_vertical_slice_accounts_for_duplicate_blank_and_corrupt_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    alias = tmp_path / "source-alias.pdf"
    blank = tmp_path / "blank.pdf"
    corrupt = tmp_path / "corrupt.pdf"
    source.write_bytes(b"multimodal")
    alias.write_bytes(b"multimodal")
    blank.write_bytes(b"blank")
    corrupt.write_bytes(b"corrupt")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {"path": str(source)},
                {"path": str(alias)},
                {"path": str(blank), "valid_blank_pages": [0]},
                {"path": str(corrupt)},
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(contract, "ALLOWED_ROOT", tmp_path)

    def page_count(path: Path) -> int:
        if path == corrupt:
            raise ValueError("encrypted or corrupt")
        return 2 if path == source else 1

    monkeypatch.setattr(contract, "_pdf_page_count", page_count)
    monkeypatch.setattr(runtime, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(runtime, "_git_dirty", lambda _path: False)
    monkeypatch.setattr(runtime, "_model_revision", lambda _model: "model-revision")

    def fake_graph(paths: list[str], **_kwargs: Any) -> pd.DataFrame:
        assert paths == [str(source.resolve()), str(blank.resolve())]
        return pd.DataFrame(
            [
                _marker(source, 1, count=3),
                _element(source, 1, 0, text="first"),
                _element(source, 1, 1, element_class="Table", modality="table", text="| A |"),
                _element(
                    source,
                    1,
                    2,
                    element_class="Picture",
                    modality="image",
                    text="",
                    binary=_valid_png(),
                ),
                _marker(source, 2),
                _element(source, 2, 0, text="second"),
                _marker(blank, 1, outcome="empty", count=0),
            ],
            dtype=object,
        )

    handoff_path = runtime.run_ingest(
        _ingest_args(tmp_path, manifest),
        graph_runner=fake_graph,
        envelope_validator=lambda value: value,
    )
    handoff = runtime._load_handoff_manifest(handoff_path)

    assert handoff["counts"]["inputs"] == {
        "duplicate": 1,
        "failed": 1,
        "success": 1,
        "valid_blank": 1,
    }
    assert handoff["counts"]["documents"] == {"failed": 1, "success": 1, "valid_blank": 1}
    assert handoff["counts"]["handed_off_document_count"] == 2
    assert handoff["counts"]["published_document_count"] == 0
    assert handoff["tables"][contract.ELEMENT_TABLE]["document_count"] == 2
    assert {document["status"] for document in handoff["documents"]} == {
        "failed",
        "success",
        "valid_blank",
    }
    assert not (handoff_path.parent / contract.COMPLETION_MANIFEST_FILE).exists()


def test_cli_contains_only_ingest_and_consume() -> None:
    import nrl_lance

    parser = nrl_lance.create_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    assert set(subparsers.choices) == {"ingest", "consume"}


def test_cli_statistics_are_opt_in() -> None:
    import nrl_lance

    parser = nrl_lance.create_parser()
    required = ["ingest", "--input-dir", "/raid/pdfs", "--nrl-repo", "/raid/nrl"]
    assert parser.parse_args(required).executor_stats is False
    assert parser.parse_args([*required, "--executor-stats"]).executor_stats is True


@pytest.mark.parametrize(("flags", "expected"), [([], None), (["--projection-block-rows", "16"], 16)])
def test_cli_passes_optional_projection_block_rows(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: int | None
) -> None:
    import nrl_lance

    calls = []
    monkeypatch.setattr(nrl_lance, "run_ingest", lambda args: calls.append(args.projection_block_rows))
    monkeypatch.setattr(
        sys, "argv", ["nrl_lance", "ingest", "--input-dir", "/raid/pdfs", "--nrl-repo", "/raid/nrl", *flags]
    )
    nrl_lance.main()
    assert calls == [expected]


@pytest.mark.parametrize(
    ("flags", "expected"), [([], (64, 1)), (["--parse-cpus", "4"], (64, 4)), (["--parse-batch-size", "128"], (128, 1))]
)
def test_cli_passes_parse_scheduling_to_ingest(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: tuple[int, int]
) -> None:
    import nrl_lance

    calls = []
    monkeypatch.setattr(nrl_lance, "run_ingest", lambda args: calls.append((args.parse_batch_size, args.parse_cpus)))
    monkeypatch.setattr(
        sys, "argv", ["nrl_lance", "ingest", "--input-dir", "/raid/pdfs", "--nrl-repo", "/raid/nrl", *flags]
    )
    nrl_lance.main()
    assert calls == [expected]


@pytest.mark.parametrize(
    "flags",
    [
        ["--parse-cpus", "0"],
        ["--parse-cpus", "-1"],
        ["--parse-cpus", "1.5"],
        ["--parse-batch-size", "0"],
        ["--parse-batch-size", "1"],
        ["--parse-batch-size", "1.5"],
        ["--projection-block-rows", "0"],
        ["--projection-block-rows", "-1"],
        ["--projection-block-rows", "1.5"],
    ],
)
def test_cli_rejects_invalid_parse_scheduling_before_ingest(monkeypatch: pytest.MonkeyPatch, flags: list[str]) -> None:
    import nrl_lance

    monkeypatch.setattr(
        sys, "argv", ["nrl_lance", "ingest", "--input-dir", "/raid/pdfs", "--nrl-repo", "/raid/nrl", *flags]
    )
    monkeypatch.setattr(nrl_lance, "run_ingest", lambda _args: pytest.fail("Invalid scheduling reached ingestion"))
    with pytest.raises(SystemExit) as error:
        nrl_lance.main()
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["ignore", "overwrite"])
def test_cli_rejects_nonfresh_consumption_modes(mode: str) -> None:
    import nrl_lance

    with pytest.raises(SystemExit) as error:
        nrl_lance.create_parser().parse_args(
            ["consume", "--handoff-manifest", "/raid/handoff.json", "--output-dir", "/raid/export", "--mode", mode]
        )
    assert error.value.code == 2


def test_native_control_uses_aligned_tokens_and_pixel_validation(tmp_path: Path) -> None:
    pytest.importorskip("nemo_curator")
    import pipeline_utils

    args = pipeline_utils.create_nemotron_parse_pdf_argparser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--pdf-dir",
            str(tmp_path / "pdfs"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-tokens",
            "9000",
        ]
    )

    pipeline = pipeline_utils.create_nemotron_parse_pdf_pipeline(args, validate_images=True)

    assert args.max_tokens == 9000
    reader, aspect, blur, writer = pipeline.stages
    assert reader.__class__.__name__ == "NemotronParsePDFReader"
    assert [stage.name for stage in (aspect, blur, writer)] == [
        "interleaved_aspect_ratio_filter",
        "interleaved_blur_filter",
        "interleaved_parquet_writer",
    ]
    assert reader.max_tokens == 9000
    assert aspect.min_aspect_ratio == 0.0
    assert aspect.max_aspect_ratio == float("inf")
    assert aspect.drop_invalid_rows is False
    assert aspect.preserve_metadata_only_samples is True
    assert blur.score_threshold == 0.0
    assert blur.drop_invalid_rows is False
    assert blur.preserve_metadata_only_samples is True
    assert writer.materialize_on_write is False


@pytest.mark.skipif(
    importlib.util.find_spec("nemo_retriever") is None or importlib.util.find_spec("nemo_retriever.common") is None,
    reason="NRL source-provenance validation runs in the pinned NRL environment",
)
def test_nrl_provenance_is_bound_to_imported_graph_source(tmp_path: Path) -> None:
    module = runtime._load_graph_module()
    nrl_repo = TUTORIAL_DIR.parents[3]

    source = runtime._validate_nrl_runtime_source(nrl_repo, module)

    assert Path(source).is_relative_to(nrl_repo)
    with pytest.raises(RuntimeError, match="outside --nrl-repo"):
        runtime._validate_nrl_runtime_source(tmp_path, module)


def test_completion_marker_precedes_non_authoritative_state_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completion_path = tmp_path / contract.COMPLETION_MANIFEST_FILE
    state_path = tmp_path / contract.RUN_STATE_FILE
    contract._write_json_atomic(state_path, {"status": "tables_validated"})
    original_write = contract._write_json_atomic

    def fail_state_update(path: Path, payload: dict[str, Any]) -> None:
        if path == state_path:
            raise OSError("diagnostic state unavailable")
        original_write(path, payload)

    monkeypatch.setattr(contract, "_write_json_atomic", fail_state_update)
    runtime._publish_completion_and_update_state(
        completion_path,
        {"status": "published"},
        state_path=state_path,
        state_updates={"status": "published"},
    )

    assert contract._load_json(completion_path) == {"status": "published"}
    assert contract._load_json(state_path) == {"status": "tables_validated"}


def test_reconciliation_detects_dropped_invalid_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    lance = pytest.importorskip("lance")
    del monkeypatch, lance

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    document = _document(tmp_path / "source.pdf")
    rows = contract.build_document_rows(
        document,
        [_marker(Path(document["path"]), 1), _element(Path(document["path"]), 1, 0)],
        run_id="run",
    ).rows
    writer = contract.ElementTableWriter(source_dir)
    writer.add_document(rows)
    table = contract._open_lancedb_table(contract._table_path(source_dir))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    only_metadata = pa.Table.from_pylist(rows[:1], schema=contract.element_schema())
    pq.write_table(only_metadata, output_dir / "part.parquet")

    with pytest.raises(ValueError, match="row keys differ"):
        runtime._validate_consumed_output(contract._table_path(source_dir), int(table.version), output_dir)


def test_confirmed_completion_clears_prior_handoff_durability_diagnostic(tmp_path: Path) -> None:
    state_path = tmp_path / contract.RUN_STATE_FILE
    contract._write_json_atomic(
        state_path,
        {"status": "tables_validated", "marker_durability": {"path": "handoff", "status": "unconfirmed"}},
    )
    completion_path = tmp_path / contract.COMPLETION_MANIFEST_FILE
    completion = contract._seal_payload({"status": "published"}, contract._COMPLETION_HASH_FIELD)
    runtime._publish_completion_and_update_state(
        completion_path,
        completion,
        state_path=state_path,
        state_updates={"status": "published"},
    )
    assert contract._load_json(state_path) == {"status": "published"}
    assert (
        contract._load_sealed_json(completion_path, contract._COMPLETION_HASH_FIELD, label="completion") == completion
    )


def test_real_curator_pipeline_reads_validates_and_writes_pinned_lance(tmp_path: Path) -> None:
    pytest.importorskip("lance")
    pytest.importorskip("cv2")
    from nemo_curator.backends.ray_data import RayDataExecutor

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    content_path = tmp_path / "content.pdf"
    blank_path = tmp_path / "blank.pdf"
    content = _document(content_path, "1" * 64)
    blank = _document(blank_path, "2" * 64, blanks=[0])
    content_rows = contract.build_document_rows(
        content,
        [
            _marker(content_path, 1, count=3),
            _element(content_path, 1, 0, text="text"),
            _element(content_path, 1, 1, element_class="Table", modality="table", text="| A |"),
            _element(
                content_path,
                1,
                2,
                element_class="Picture",
                modality="image",
                text="",
                binary=_valid_png(),
            ),
        ],
        run_id="run",
    ).rows
    blank_rows = contract.build_document_rows(
        blank,
        [_marker(blank_path, 1, outcome="empty", count=0)],
        run_id="run",
    ).rows
    writer = contract.ElementTableWriter(source_dir)
    writer.add_document(content_rows)
    writer.add_document(blank_rows)
    partial_rows = []
    for index, original in enumerate((content_rows, blank_rows), start=3):
        path = tmp_path / f"partial-{index}.pdf"
        document = _document(path, str(index) * 64, pages=2, blanks=[0] if len(original) == 1 else [])
        envelope = (
            [_marker(path, 1, outcome="empty", count=0)]
            if len(original) == 1
            else [_marker(path, 1), _element(path, 1, 0)]
        )
        build = contract.build_document_rows(document, envelope, run_id="run")
        assert build.status == "partial"
        partial_rows.extend(build.rows)
        writer.add_document(build.rows)
    table = contract._open_lancedb_table(contract._table_path(source_dir))
    output_dir = tmp_path / "output"
    pipeline = runtime._build_consume_pipeline(
        contract._table_path(source_dir),
        version=int(table.version),
        output_dir=output_dir,
        mode="error",
    )

    pipeline.run(RayDataExecutor())
    result = runtime._validate_consumed_output(
        contract._table_path(source_dir),
        int(table.version),
        output_dir,
    )

    assert result["status"] == "validated"
    assert result["source"]["row_count"] == len(content_rows) + len(blank_rows) + len(partial_rows)
    assert result["output"]["row_count"] == len(content_rows) + len(blank_rows) + len(partial_rows)
    assert result["output"]["document_count"] == 4


def test_truncated_png_passes_header_but_fails_pixel_decode_reconciliation(tmp_path: Path) -> None:
    pytest.importorskip("lance")
    pytest.importorskip("cv2")
    from nemo_curator.backends.ray_data import RayDataExecutor

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = tmp_path / "source.pdf"
    document = _document(source_path)
    truncated = _valid_png()[:41]
    rows = contract.build_document_rows(
        document,
        [
            _marker(source_path, 1, count=2),
            _element(source_path, 1, 0, element_class="Table", modality="table", text="| A |"),
            _element(
                source_path,
                1,
                1,
                element_class="Picture",
                modality="image",
                text="",
                binary=truncated,
            ),
        ],
        run_id="run",
    ).rows
    assert rows[-1]["binary_content"].startswith(b"\x89PNG\r\n\x1a\n")
    writer = contract.ElementTableWriter(source_dir)
    writer.add_document(rows)
    table = contract._open_lancedb_table(contract._table_path(source_dir))
    output_dir = tmp_path / "output"
    pipeline = runtime._build_consume_pipeline(
        contract._table_path(source_dir),
        version=int(table.version),
        output_dir=output_dir,
        mode="error",
    )

    pipeline.run(RayDataExecutor())

    with pytest.raises(ValueError, match="row keys differ"):
        runtime._validate_consumed_output(
            contract._table_path(source_dir),
            int(table.version),
            output_dir,
        )


def _lifecycle_inputs(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[argparse.Namespace, Callable[..., pd.DataFrame]]:
    sources = [root / f"source-{index}.pdf" for index in range(3)]
    for index, path in enumerate(sources):
        path.write_bytes(f"disposable-pdf-{index}".encode())
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({"path": str(path)}) + "\n" for path in sources))
    monkeypatch.setattr(contract, "ALLOWED_ROOT", root)
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 1)
    monkeypatch.setattr(runtime, "_git_revision", lambda _path: "revision")
    monkeypatch.setattr(runtime, "_git_dirty", lambda _path: False)
    monkeypatch.setattr(runtime, "_model_revision", lambda _model: "model-revision")

    def graph(paths: list[str], **_kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            [row for path in paths for row in (_marker(Path(path), 1), _element(Path(path), 1, 0))], dtype=object
        )

    return _ingest_args(root, manifest), graph


def _lifecycle_handoff(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    args, graph = _lifecycle_inputs(monkeypatch, root)
    return runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)


def _storage_consumer(monkeypatch: pytest.MonkeyPatch, after_write: Callable[[], None] = lambda: None) -> None:
    """Isolate lifecycle faults from Ray while retaining real Lance/Parquet reconciliation."""
    import lance
    import pyarrow.parquet as pq

    from nemo_curator.pipeline import Pipeline

    def build(table_path: Path, *, version: int, output_dir: Path, mode: str) -> Any:
        del mode

        def run(_executor: Any) -> list[Any]:
            output_dir.mkdir()
            pq.write_table(lance.dataset(str(table_path), version=version).to_table(), output_dir / "part.parquet")
            after_write()
            return []

        return SimpleNamespace(run=run)

    def reopen(pipeline: Any, _executor: Any) -> list[Any]:
        assert pipeline.name == "nrl_curator_parquet_reopen"
        return [
            SimpleNamespace(to_pyarrow=lambda path=path: pq.read_table(path)) for path in pipeline.stages[0].file_paths
        ]

    monkeypatch.setattr(runtime, "_build_consume_pipeline", build)
    monkeypatch.setattr(Pipeline, "run", reopen)


def _consume_args(handoff: Path, output: Path, *, mode: str = "error") -> argparse.Namespace:
    return argparse.Namespace(handoff_manifest=str(handoff), output_dir=str(output), mode=mode)


@pytest.mark.parametrize("capture", [False, True])
def test_partial_handoff_consumption_preserves_status_and_missing_pages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, capture: bool
) -> None:
    import nrl_compare

    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 2)
    if capture:
        args.evidence_root = str(tmp_path / "evidence")

    def reject_capture(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Partial documents must not become eligible frozen-response captures")

    monkeypatch.setattr(nrl_compare, "finalize_capture", reject_capture)
    handoff_path = runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)
    handoff = runtime._load_handoff_manifest(handoff_path)
    assert handoff["counts"]["complete_document_count"] == 0
    assert handoff["counts"]["partial_document_count"] == 3
    assert handoff["counts"]["failed_page_count"] == 3
    if capture:
        assert handoff["benchmark_evidence"]["manifests"] == []
    _storage_consumer(monkeypatch)
    completion_path = runtime.run_consume(_consume_args(handoff_path, tmp_path / "export"))
    completion = contract._load_sealed_json(completion_path, contract._COMPLETION_HASH_FIELD, label="completion")
    assert completion["publication_policy"] == contract.PUBLICATION_POLICY
    assert completion["counts"]["complete_document_count"] == 0
    assert completion["counts"]["partial_document_count"] == 3
    assert completion["counts"]["delivered_page_count"] == 3
    assert completion["counts"]["delivered_content_element_count"] == 3
    assert all(document["status"] == "partial" for document in completion["documents"])
    assert all(document["page_outcomes"][1]["status"] == "failed" for document in completion["documents"])


def test_legacy_policy_cannot_relabel_partial_handoff_as_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(contract, "_pdf_page_count", lambda _path: 2)
    path = runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)
    handoff = runtime._load_handoff_manifest(path)
    handoff.pop("publication_policy")
    handoff.pop(contract._HANDOFF_HASH_FIELD)
    for document in handoff["documents"]:
        document["status"] = "success"
    contract._write_json_atomic(path, contract._seal_payload(handoff, contract._HANDOFF_HASH_FIELD))
    with pytest.raises(ValueError, match="Explicit page outcomes"):
        runtime.run_consume(_consume_args(path, tmp_path / "export"))
    assert not (path.parent / contract.COMPLETION_MANIFEST_FILE).exists()
    assert not (tmp_path / "export").exists()


def test_legacy_complete_handoff_remains_consumable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pyarrow as pa

    path = _lifecycle_handoff(monkeypatch, tmp_path)
    handoff = runtime._load_handoff_manifest(path)
    handoff.pop("publication_policy")
    handoff.pop(contract._HANDOFF_HASH_FIELD)
    for document in handoff["documents"]:
        for key in ("extraction_status", "page_outcomes", "content_page_count", "failed_page_count"):
            document.pop(key)
    table_path = contract._table_path(path.parent)
    table = contract._open_lancedb_table(table_path)
    rows = table.to_arrow().to_pylist()
    for row in rows:
        if row["position"] == -1:
            metadata = json.loads(row["text_content"])
            for key in ("extraction_status", "page_outcomes", "issues"):
                metadata.pop(key)
            row["text_content"] = json.dumps(metadata)
    # Rewrite only this disposable fixture, retaining one add per document.
    table.delete("position >= -1")
    for document in handoff["documents"]:
        table.add(
            pa.Table.from_pylist(
                [row for row in rows if row["sample_id"] == document["content_sha256"]],
                schema=contract.element_schema(),
            )
        )
    provenance = runtime._expected_document_provenance(
        handoff["inputs"],
        run_id=handoff["run_id"],
        sample_ids={document["content_sha256"] for document in handoff["documents"]},
    )
    handoff["tables"][contract.ELEMENT_TABLE] = runtime.validate_element_table(
        table_path, runtime._expected_counts_from_handoff(handoff), expected_provenance=provenance
    )
    contract._write_json_atomic(path, contract._seal_payload(handoff, contract._HANDOFF_HASH_FIELD))
    _storage_consumer(monkeypatch)
    completion_path = runtime.run_consume(_consume_args(path, tmp_path / "export"))
    completion = contract._load_json(completion_path)
    assert completion["publication_policy"] == "complete_documents_v1"
    assert completion["counts"]["complete_document_count"] == 3
    assert completion["counts"]["partial_document_count"] == 0


@pytest.mark.parametrize("mutation", ["status", "missing_page", "page_row", "issues"])
def test_partial_table_rejects_incorrect_coverage(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "source.pdf"
    document = _document(source, pages=2)
    build = contract.build_document_rows(document, [_marker(source, 1), _element(source, 1, 0)], run_id="run")
    metadata = json.loads(build.rows[0]["text_content"])
    provenance = _provenance(document, "run")
    provenance[document["content_sha256"]].update(
        json.loads(json.dumps({key: metadata[key] for key in ("extraction_status", "page_outcomes", "issues")}))
    )
    if mutation == "status":
        metadata["extraction_status"] = "success"
    elif mutation == "missing_page":
        metadata["page_outcomes"].pop()
    elif mutation == "page_row":
        build.rows[1]["page_number"] = 1
    else:
        metadata["issues"] = [{"kind": "different_issue"}]
    build.rows[0]["text_content"] = json.dumps(metadata)
    contract.ElementTableWriter(tmp_path).add_document(build.rows)
    with pytest.raises(ValueError, match=r"coverage|outcomes|Extraction status"):
        runtime.validate_element_table(
            contract._table_path(tmp_path), {document["content_sha256"]: 2}, expected_provenance=provenance
        )


@pytest.mark.parametrize(
    "fault", ["first_write", "later_write", "table_validation", "source_changed", "state_write", "handoff_write"]
)
def test_ingest_storage_faults_never_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str) -> None:
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    original_add = contract.ElementTableWriter.add_document
    original_validate = runtime.validate_element_table
    original_state = contract._write_json_atomic
    calls = 0

    def add(writer: Any, rows: Any) -> None:
        nonlocal calls
        original_add(writer, rows)
        calls += 1
        if (fault == "first_write" and calls == 1) or (fault == "later_write" and calls == 2):
            raise OSError("injected document write failure")

    def validate(*args: Any, **kwargs: Any) -> Any:
        if fault == "table_validation":
            raise OSError("injected validation failure")
        if fault == "source_changed":
            (tmp_path / "source-0.pdf").write_bytes(b"changed after inventory")
        return original_validate(*args, **kwargs)

    def write_state(path: Path, payload: Any) -> None:
        if fault == "state_write" and payload.get("status") == "tables_validated":
            raise OSError("injected state failure")
        original_state(path, payload)

    def marker_failure(_path: Path, _payload: Any) -> None:
        raise OSError("injected marker failure")

    monkeypatch.setattr(contract.ElementTableWriter, "add_document", add)
    monkeypatch.setattr(runtime, "validate_element_table", validate)
    monkeypatch.setattr(contract, "_write_json_atomic", write_state)
    if fault == "handoff_write":
        monkeypatch.setattr(contract, "_write_json_exclusive_atomic", marker_failure)
    with pytest.raises((OSError, RuntimeError), match=r"injected|changed after inventory"):
        runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)
    run_dir = tmp_path / "runs/run"
    assert not (run_dir / contract.HANDOFF_MANIFEST_FILE).exists()
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()
    assert contract._load_json(run_dir / contract.RUN_STATE_FILE)["status"] == "unpublished"


@pytest.mark.parametrize("phase", ["handoff", "completion"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_visible_marker_sync_failure_is_not_unpublished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str, cleanup_failure: bool
) -> None:
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    handoff = None
    if phase == "completion":
        handoff = runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)
        _storage_consumer(monkeypatch)
    run_dir = tmp_path / "runs/run"
    marker = run_dir / (contract.HANDOFF_MANIFEST_FILE if phase == "handoff" else contract.COMPLETION_MANIFEST_FILE)
    hash_field = contract._HANDOFF_HASH_FIELD if phase == "handoff" else contract._COMPLETION_HASH_FIELD
    original_sync = contract._fsync_directory
    original_unlink = Path.unlink

    def fail_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith(f".{marker.name}.") and marker.exists():
            raise OSError("injected marker temporary cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    def fail_visible_sync(path: Path) -> None:
        if marker.exists():
            raise OSError("injected directory sync failure")
        original_sync(path)

    def publish() -> None:
        if phase == "handoff":
            runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)
        else:
            runtime.run_consume(_consume_args(handoff, tmp_path / "export"))

    monkeypatch.setattr(contract, "_fsync_directory", fail_visible_sync)
    if cleanup_failure:
        monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(contract.MarkerDurabilityUnconfirmedError, match=r"visible.*durability is unconfirmed"):
        publish()
    payload = contract._load_sealed_json(marker, hash_field, label=phase)
    state = contract._load_json(run_dir / contract.RUN_STATE_FILE)
    assert state["status"] == ("tables_validated" if phase == "handoff" else "published")
    assert state["marker_durability"]["status"] == "unconfirmed"
    original_bytes = marker.read_bytes()
    with pytest.raises(FileExistsError):
        contract._write_json_exclusive_atomic(marker, {"replacement": True})
    monkeypatch.setattr(contract, "_fsync_directory", original_sync)
    with pytest.raises(ValueError, match="expected SHA-256"):
        contract._confirm_marker_durability(marker, hash_field, expected_sha256="0" * 64)
    confirmed = contract._confirm_marker_durability(marker, hash_field, expected_sha256=payload[hash_field])
    assert confirmed == payload
    assert marker.read_bytes() == original_bytes


def test_cleanup_failure_after_confirmed_visibility_does_not_fail_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_unlink = Path.unlink

    def fail_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.startswith(f".{contract.HANDOFF_MANIFEST_FILE}."):
            raise OSError("injected marker temporary cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    assert runtime._load_handoff_manifest(handoff)["status"] == "tables_validated"
    state = contract._load_json(handoff.parent / contract.RUN_STATE_FILE)
    assert state["status"] == "tables_validated"


@pytest.mark.parametrize(
    "fault", ["source_before", "source_during", "version_before", "version_during", "report_write", "completion_write"]
)
def test_consume_mutation_and_publication_faults_preserve_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    import lance

    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    original_handoff = handoff.read_bytes()
    table_path = contract._table_path(handoff.parent)

    def mutate() -> None:
        if fault.startswith("source_"):
            (tmp_path / "source-0.pdf").write_bytes(b"changed during consumption")
        elif fault.startswith("version_"):
            table = contract._open_lancedb_table(table_path)
            table.add(lance.dataset(str(table_path)).to_table())

    if fault.endswith("_before"):
        mutate()
    _storage_consumer(monkeypatch, mutate if fault.endswith("_during") else lambda: None)
    original_write = contract._write_json_exclusive_atomic

    def fail_marker(path: Path, payload: Any) -> None:
        failing_name = contract.CONSUME_REPORT_FILE if fault == "report_write" else contract.COMPLETION_MANIFEST_FILE
        if fault.endswith("_write") and path.name == failing_name:
            raise OSError("injected publication failure")
        original_write(path, payload)

    monkeypatch.setattr(contract, "_write_json_exclusive_atomic", fail_marker)
    with pytest.raises((OSError, RuntimeError), match=r"changed|Lance version|injected"):
        runtime.run_consume(_consume_args(handoff, tmp_path / "export"))
    assert handoff.read_bytes() == original_handoff
    assert not (handoff.parent / contract.COMPLETION_MANIFEST_FILE).exists()
    if fault == "report_write":
        assert not (tmp_path / "export" / contract.CONSUME_REPORT_FILE).exists()


def test_consume_requires_fresh_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    output = tmp_path / "export"
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="must be fresh"):
        runtime.run_consume(_consume_args(handoff, output))
    assert sentinel.read_bytes() == b"preserve"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize("mode", ["ignore", "overwrite", "invalid"])
def test_consume_rejects_unsupported_mode_before_reading_handoff(tmp_path: Path, mode: str) -> None:
    with pytest.raises(ValueError, match="supports only mode='error'"):
        runtime.run_consume(_consume_args(tmp_path / "nonexistent.json", tmp_path / "export", mode=mode))
    assert list(tmp_path.iterdir()) == []


def test_completed_run_and_run_id_cannot_be_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    _storage_consumer(monkeypatch)
    completion = runtime.run_consume(_consume_args(handoff, tmp_path / "export"))
    timings = contract._load_json(tmp_path / "export" / contract.CONSUME_REPORT_FILE)["timings"]
    assert "total_seconds" not in timings
    assert timings["pipeline_and_reconciliation_seconds"] >= timings["pipeline_seconds"]
    snapshot = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError, match="completion manifest already exists"):
        runtime.run_consume(_consume_args(handoff, tmp_path / "new-export"))
    with pytest.raises(FileExistsError):
        runtime.run_ingest(_ingest_args(tmp_path, tmp_path / "manifest.jsonl"))
    assert completion.exists()
    assert {path: path.read_bytes() for path in snapshot} == snapshot
    assert not (tmp_path / "new-export").exists()


@pytest.mark.parametrize("phase", ["ingest", "consume"])
def test_process_termination_leaves_no_false_completion_and_fresh_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path) if phase == "consume" else None
    child_code = """
import pathlib, signal, sys, pytest
sys.path.insert(0, sys.argv[1])
import test_nrl_lance as tests
root = pathlib.Path(sys.argv[2])
phase = sys.argv[3]
patch = pytest.MonkeyPatch()
args, graph = tests._lifecycle_inputs(patch, root)
def pause():
    (root / 'ready').write_text('ready')
    signal.pause()
if phase == 'ingest':
    original = tests.contract.ElementTableWriter.add_document
    def write(writer, rows):
        original(writer, rows)
        pause()
    patch.setattr(tests.contract.ElementTableWriter, 'add_document', write)
    tests.runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)
else:
    tests._storage_consumer(patch, pause)
    tests.runtime.run_consume(tests._consume_args(root / 'runs/run/handoff_manifest.json', root / 'interrupted-export'))
"""
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", child_code, str(Path(__file__).parent), str(tmp_path), phase],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 45
        while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "ready").exists(), process.communicate(timeout=5)
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode == -signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)
    run_dir = tmp_path / "runs/run"
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()
    if phase == "ingest":
        assert not (run_dir / contract.HANDOFF_MANIFEST_FILE).exists()
    else:
        assert handoff.exists()
    args, graph = _lifecycle_inputs(monkeypatch, tmp_path)
    args.run_id = "retry"
    retry_handoff = runtime.run_ingest(args, graph_runner=graph, envelope_validator=lambda value: value)
    _storage_consumer(monkeypatch)
    completion = runtime.run_consume(_consume_args(retry_handoff, tmp_path / "retry-export"))
    assert (
        contract._load_sealed_json(completion, contract._COMPLETION_HASH_FIELD, label="completion")["status"]
        == "published"
    )
    assert not (run_dir / contract.COMPLETION_MANIFEST_FILE).exists()


def test_marker_file_sync_failure_precedes_visibility(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    marker = tmp_path / contract.HANDOFF_MANIFEST_FILE

    def fail_sync(_descriptor: int) -> None:
        raise OSError("injected file sync failure")

    monkeypatch.setattr(contract.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="file sync failure"):
        contract._write_json_exclusive_atomic(marker, {"status": "tables_validated"})
    assert not marker.exists()
    assert list(tmp_path.iterdir()) == []


def test_report_sync_failure_preserves_visible_report_without_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    handoff = _lifecycle_handoff(monkeypatch, tmp_path)
    _storage_consumer(monkeypatch)
    report = tmp_path / "export" / contract.CONSUME_REPORT_FILE
    original_sync = contract._fsync_directory

    def fail_report_sync(path: Path) -> None:
        if report.exists() and path == report.parent:
            raise OSError("injected report sync failure")
        original_sync(path)

    monkeypatch.setattr(contract, "_fsync_directory", fail_report_sync)
    with pytest.raises(contract.MarkerDurabilityUnconfirmedError):
        runtime.run_consume(_consume_args(handoff, report.parent))
    contract._load_sealed_json(report, contract._REPORT_HASH_FIELD, label="consume report")
    assert not (handoff.parent / contract.COMPLETION_MANIFEST_FILE).exists()


def test_durability_confirmation_rejects_tampered_marker(tmp_path: Path) -> None:
    marker = tmp_path / contract.HANDOFF_MANIFEST_FILE
    payload = contract._seal_payload({"status": "tables_validated"}, contract._HANDOFF_HASH_FIELD)
    expected = payload[contract._HANDOFF_HASH_FIELD]
    payload["status"] = "tampered"
    contract._write_json_exclusive_atomic(marker, payload)
    with pytest.raises(ValueError, match="SHA-256 is"):
        contract._confirm_marker_durability(marker, contract._HANDOFF_HASH_FIELD, expected_sha256=expected)
