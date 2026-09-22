# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This standalone bridge handles intentionally heterogeneous pandas records.
# ruff: noqa: ANN401, ARG002, BLE001, C901, EM101, EM102, PD008, PLR0911, PLR0912, PLR0913, PLR0915, PLR2004, SIM108, TRY004

"""Build the extraction-only NRL graph used by the Curator Lance bridge.

This module runs in the NeMo Retriever environment.  Its terminal CPU
operator turns page-level Nemotron Parse results into a small, flat contract
that the separate Curator process can validate and publish.  It deliberately
has no NeMo Curator dependency.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import nrl_lance_contract as contract
import pandas as pd
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.graph.executor import InprocessExecutor, RayDataExecutor
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator

PARSE_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
PARSE_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
COORDINATE_SPACE = "normalized_1664x2048_padded_canvas"
MIN_PICTURE_CROP_PX = 10
MAX_PROJECTION_WORKERS = 8

PROJECTION_COLUMNS = (
    "record_type",
    "source_path",
    "native_page_number",
    "page_outcome",
    "element_count",
    "issues_json",
    "raw_output_sha256",
    "element_index",
    "element_class",
    "modality",
    "content_type",
    "text_content",
    "binary_content",
    "bbox_xyxy_norm_json",
    "bbox_coordinate_space",
)

_PAGE_OUTCOMES = frozenset({"parsed", "empty", "failed"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_NEMOTRON_ELEMENT_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)


class IncompleteModelOutputError(ValueError):
    """The tagged response contains content outside complete elements."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _raw_output_sha256(raw_output: str) -> str:
    return hashlib.sha256(raw_output.encode("utf-8", errors="strict")).hexdigest()


def _row_raw_output_sha256(row: Mapping[str, Any]) -> str | None:
    parser_metadata = row.get("nemotron_parse_v1_2")
    raw_output = parser_metadata.get("raw_output") if isinstance(parser_metadata, Mapping) else None
    return _raw_output_sha256(raw_output) if isinstance(raw_output, str) else None


def _validate_complete_raw_output(raw_output: str) -> int:
    """Return the number of elements only when the entire response parses."""

    if not isinstance(raw_output, str):
        raise TypeError(f"raw model output must be text, got {type(raw_output).__name__}")
    matches = list(_NEMOTRON_ELEMENT_RE.finditer(raw_output))
    count = len(matches)
    if (
        raw_output.count("<x_") != count * 2
        or raw_output.count("<y_") != count * 2
        or raw_output.count("<class_") != count
    ):
        raise IncompleteModelOutputError("raw model output contains incomplete or unbalanced element tags")

    cursor = 0
    for match in matches:
        if raw_output[cursor : match.start()].strip():
            raise IncompleteModelOutputError(f"raw model output has unparsed content at offset {cursor}")
        cursor = match.end()
    if raw_output[cursor:].strip():
        raise IncompleteModelOutputError(f"raw model output has unparsed content at offset {cursor}")
    return count


def _parse_raw_elements(raw_output: str) -> list[dict[str, Any]]:
    expected_count = _validate_complete_raw_output(raw_output)

    from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import (
        extract_classes_bboxes,
        postprocess_text,
    )

    classes, bboxes, texts = extract_classes_bboxes(raw_output)
    if not (len(classes) == len(bboxes) == len(texts) == expected_count):
        raise IncompleteModelOutputError("NRL parser did not return every validated model element")

    elements: list[dict[str, Any]] = []
    for element_class, bbox, text in zip(classes, bboxes, texts, strict=True):
        processed = postprocess_text(
            text,
            cls=element_class,
            text_format="markdown",
            table_format="latex",  # Preserve native table bodies and merged-cell spans for Curator.
            blank_text_in_figures=False,
        ).strip()
        elements.append({"class": element_class, "text": processed, "bbox": list(bbox)})
    return elements


def _normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(coordinate, bool) for coordinate in value):
        return None
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(coordinate) or coordinate < 0.0 or coordinate > 1.0 for coordinate in bbox):
        return None
    left, top, right, bottom = bbox
    if left >= right or top >= bottom:
        return None
    return bbox


def _crop_picture_bytes(page_image: Any, bbox: Sequence[float]) -> bytes | None:
    from nemo_retriever.common.modality.ocr.shared import _crop_b64_image_by_norm_bbox
    from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import transform_bbox_to_original

    if not isinstance(page_image, dict) or not isinstance(page_image.get("image_b64"), str):
        return None
    shape = page_image.get("orig_shape_hw")
    tolist = getattr(shape, "tolist", None)
    if callable(tolist):
        shape = tolist()
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return None
    try:
        height, width = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if height <= 0 or width <= 0:
        return None

    left, top, right, bottom = transform_bbox_to_original(tuple(float(item) for item in bbox), width, height)
    normalized = [left / width, top / height, right / width, bottom / height]
    cropped_b64, cropped_shape = _crop_b64_image_by_norm_bbox(
        page_image["image_b64"],
        bbox_xyxy_norm=normalized,
        image_format="png",
    )
    if cropped_b64 is None or cropped_shape is None or min(cropped_shape) < MIN_PICTURE_CROP_PX:
        return None
    try:
        payload = base64.b64decode(cropped_b64, validate=True)
    except (TypeError, ValueError):
        return None
    return payload or None


def _source_path(row: Mapping[str, Any]) -> str:
    value = row.get("path")
    metadata = row.get("metadata")
    if not value and isinstance(metadata, Mapping):
        value = metadata.get("source_path")
    return os.fspath(value) if isinstance(value, os.PathLike) else str(value or "")


def _native_page_number(row: Mapping[str, Any]) -> int:
    value = row.get("page_number")
    if isinstance(value, bool):
        return 0
    try:
        page_number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, page_number)


def _compact_error(error: Any) -> dict[str, Any]:
    if isinstance(error, Mapping):
        compact = {key: error[key] for key in ("stage", "type", "message") if key in error and error[key] is not None}
        return compact or {"message": str(dict(error))}
    return {"message": str(error)}


def _page_outcome_row(
    *,
    source_path: str,
    native_page_number: int,
    outcome: str,
    element_count: int,
    issues: Sequence[Mapping[str, Any]],
    raw_output_sha256: str | None,
) -> dict[str, Any]:
    return {
        "record_type": "page_outcome",
        "source_path": source_path,
        "native_page_number": native_page_number,
        "page_outcome": outcome,
        "element_count": element_count,
        "issues_json": _canonical_json(list(issues)),
        "raw_output_sha256": raw_output_sha256,
        "element_index": None,
        "element_class": None,
        "modality": None,
        "content_type": None,
        "text_content": None,
        "binary_content": None,
        "bbox_xyxy_norm_json": None,
        "bbox_coordinate_space": None,
    }


def _failed_page_row(
    row: Mapping[str, Any],
    issue: Mapping[str, Any],
    *,
    raw_output_sha256: str | None = None,
) -> dict[str, Any]:
    return _page_outcome_row(
        source_path=_source_path(row),
        native_page_number=_native_page_number(row),
        outcome="failed",
        element_count=0,
        issues=[issue],
        raw_output_sha256=raw_output_sha256,
    )


def _project_page(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_path = _source_path(row)
    native_page_number = _native_page_number(row)
    metadata = row.get("metadata")
    extraction_error = metadata.get("error") if isinstance(metadata, Mapping) else None
    parser_metadata = row.get("nemotron_parse_v1_2")
    parse_error = parser_metadata.get("error") if isinstance(parser_metadata, Mapping) else None
    raw_output = parser_metadata.get("raw_output") if isinstance(parser_metadata, Mapping) else None
    raw_sha256 = _raw_output_sha256(raw_output) if isinstance(raw_output, str) else None

    if extraction_error is not None or parse_error is not None:
        return [
            _failed_page_row(
                row,
                {"kind": "page_stage_error", "error": _compact_error(extraction_error or parse_error)},
                raw_output_sha256=raw_sha256,
            )
        ]
    if native_page_number == 0:
        return [
            _failed_page_row(
                row,
                {"kind": "document_or_split_failure", "error": _compact_error(row.get("error") or "invalid page")},
                raw_output_sha256=raw_sha256,
            )
        ]
    if not isinstance(row.get("page_image"), Mapping):
        return [_failed_page_row(row, {"kind": "missing_page_image"}, raw_output_sha256=raw_sha256)]
    if raw_output is None:
        return [_failed_page_row(row, {"kind": "missing_model_output"})]
    if not isinstance(raw_output, str):
        return [
            _failed_page_row(
                row,
                {"kind": "invalid_model_output", "type": type(raw_output).__name__},
            )
        ]
    if not raw_output.strip():
        return [
            _page_outcome_row(
                source_path=source_path,
                native_page_number=native_page_number,
                outcome="empty",
                element_count=0,
                issues=[],
                raw_output_sha256=raw_sha256,
            )
        ]

    try:
        elements = _parse_raw_elements(raw_output)
    except (IncompleteModelOutputError, TypeError, ValueError) as exc:
        return [
            _failed_page_row(
                row,
                {"kind": "truncated_or_unparseable_model_output", "detail": str(exc)},
                raw_output_sha256=raw_sha256,
            )
        ]
    if not elements:
        return [
            _failed_page_row(
                row,
                {"kind": "unparseable_model_output"},
                raw_output_sha256=raw_sha256,
            )
        ]

    projected: list[dict[str, Any]] = []
    for element_index, element in enumerate(elements):
        element_class = str(element.get("class") or "").strip()
        bbox = _normalize_bbox(element.get("bbox"))
        if not element_class or bbox is None:
            return [
                _failed_page_row(
                    row,
                    {
                        "kind": "invalid_element",
                        "element_index": element_index,
                        "element_class": element_class or None,
                    },
                    raw_output_sha256=raw_sha256,
                )
            ]

        binary_content: bytes | None = None
        if element_class == "Picture":
            binary_content = _crop_picture_bytes(row["page_image"], bbox)
            if binary_content is None:
                return [
                    _failed_page_row(
                        row,
                        {"kind": "picture_crop_failure", "element_index": element_index},
                        raw_output_sha256=raw_sha256,
                    )
                ]
            modality, content_type = "image", "image/png"
        elif element_class == "Table":
            modality, content_type = "table", "text/markdown"
        else:
            modality, content_type = "text", "text/markdown"

        projected.append(
            {
                "record_type": "element",
                "source_path": source_path,
                "native_page_number": native_page_number,
                "page_outcome": None,
                "element_count": None,
                "issues_json": "[]",
                "raw_output_sha256": None,
                "element_index": element_index,
                "element_class": element_class,
                "modality": modality,
                "content_type": content_type,
                "text_content": str(element.get("text", "")),
                "binary_content": binary_content,
                "bbox_xyxy_norm_json": _canonical_json(bbox),
                "bbox_coordinate_space": COORDINATE_SPACE,
            }
        )

    outcome = _page_outcome_row(
        source_path=source_path,
        native_page_number=native_page_number,
        outcome="parsed",
        element_count=len(projected),
        issues=[],
        raw_output_sha256=raw_sha256,
    )
    return [outcome, *projected]


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _normalize_integer(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return value
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return value
    return normalized if normalized == value else value


def _normalize_json_field(value: Any, *, field: str) -> str | None:
    if _is_missing(value):
        return None
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must contain valid JSON") from exc
    return _canonical_json(decoded)


def normalize_projection_envelope(data: Any) -> pd.DataFrame:
    """Return an object-backed DataFrame with the canonical column order."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)
    unknown = sorted(set(frame.columns) - set(PROJECTION_COLUMNS))
    if unknown:
        raise ValueError(f"projection envelope has unknown columns: {unknown}")
    for column in PROJECTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame.loc[:, PROJECTION_COLUMNS].astype(object)

    for index in frame.index:
        for column in PROJECTION_COLUMNS:
            value = frame.at[index, column]
            if _is_missing(value):
                frame.at[index, column] = None
        if isinstance(frame.at[index, "source_path"], os.PathLike):
            frame.at[index, "source_path"] = os.fspath(frame.at[index, "source_path"])
        for column in ("native_page_number", "element_count", "element_index"):
            frame.at[index, column] = _normalize_integer(frame.at[index, column])
        binary = frame.at[index, "binary_content"]
        if isinstance(binary, (bytearray, memoryview)):
            frame.at[index, "binary_content"] = bytes(binary)
        for column in ("issues_json", "bbox_xyxy_norm_json"):
            frame.at[index, column] = _normalize_json_field(frame.at[index, column], field=column)
    return frame.reset_index(drop=True)


def _decoded_json_array(value: Any, *, field: str, row_index: int) -> list[Any]:
    if not isinstance(value, str):
        raise ValueError(f"row {row_index}: {field} must be a JSON string")
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError(f"row {row_index}: {field} must encode a JSON array")
    return decoded


def _validated_bbox(value: Any, *, row_index: int) -> list[float]:
    decoded = _decoded_json_array(value, field="bbox_xyxy_norm_json", row_index=row_index)
    if any(isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)) for coordinate in decoded):
        raise ValueError(f"row {row_index}: bbox coordinates must be JSON numbers")
    bbox = _normalize_bbox(decoded)
    if bbox is None:
        raise ValueError(f"row {row_index}: bbox must be four finite ordered floats in [0, 1]")
    return bbox


def validate_projection_envelope(data: Any) -> pd.DataFrame:
    """Normalize and fail closed on any malformed projection record."""

    frame = normalize_projection_envelope(data)
    outcomes: dict[tuple[str, int], tuple[int, str]] = {}
    elements: dict[tuple[str, int], list[int]] = {}

    for index, row in frame.iterrows():
        record_type = row["record_type"]
        source_path = row["source_path"]
        native_page_number = row["native_page_number"]
        if record_type not in {"page_outcome", "element"}:
            raise ValueError(f"row {index}: unsupported record_type {record_type!r}")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(f"row {index}: source_path must be non-empty text")
        if isinstance(native_page_number, bool) or not isinstance(native_page_number, int) or native_page_number < 0:
            raise ValueError(f"row {index}: native_page_number must be a non-negative integer")

        issues = _decoded_json_array(row["issues_json"], field="issues_json", row_index=index)
        if any(
            not isinstance(issue, Mapping) or not isinstance(issue.get("kind"), str) or not issue["kind"]
            for issue in issues
        ):
            raise ValueError(f"row {index}: every issue must be an object with a non-empty kind")
        key = (source_path, native_page_number)
        if record_type == "page_outcome":
            if key in outcomes:
                raise ValueError(f"duplicate page outcome for {source_path!r} page {native_page_number}")
            outcome = row["page_outcome"]
            element_count = row["element_count"]
            if outcome not in _PAGE_OUTCOMES:
                raise ValueError(f"row {index}: unsupported page_outcome {outcome!r}")
            if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count < 0:
                raise ValueError(f"row {index}: element_count must be a non-negative integer")
            if native_page_number == 0 and outcome != "failed":
                raise ValueError(f"row {index}: native page 0 is reserved for document/split failures")
            if outcome == "parsed" and element_count == 0:
                raise ValueError(f"row {index}: parsed pages must contain at least one element")
            if outcome != "parsed" and element_count != 0:
                raise ValueError(f"row {index}: non-parsed pages cannot declare elements")
            if (outcome == "failed") != bool(issues):
                raise ValueError(f"row {index}: only failed outcomes may carry issues")
            raw_sha256 = row["raw_output_sha256"]
            if raw_sha256 is not None and (
                not isinstance(raw_sha256, str) or _SHA256_RE.fullmatch(raw_sha256) is None
            ):
                raise ValueError(f"row {index}: raw_output_sha256 must be a lowercase SHA-256")
            if outcome in {"parsed", "empty"} and raw_sha256 is None:
                raise ValueError(f"row {index}: successful model outcomes require raw_output_sha256")
            for column in (
                "element_index",
                "element_class",
                "modality",
                "content_type",
                "text_content",
                "binary_content",
                "bbox_xyxy_norm_json",
                "bbox_coordinate_space",
            ):
                if row[column] is not None:
                    raise ValueError(f"row {index}: page outcome field {column} must be null")
            outcomes[key] = (element_count, outcome)
            continue

        if native_page_number == 0:
            raise ValueError(f"row {index}: element rows require a positive native page number")
        for column in ("page_outcome", "element_count", "raw_output_sha256"):
            if row[column] is not None:
                raise ValueError(f"row {index}: element field {column} must be null")
        if issues:
            raise ValueError(f"row {index}: element rows cannot carry issues")
        element_index = row["element_index"]
        if isinstance(element_index, bool) or not isinstance(element_index, int) or element_index < 0:
            raise ValueError(f"row {index}: element_index must be a non-negative integer")
        element_class = row["element_class"]
        if not isinstance(element_class, str) or not element_class:
            raise ValueError(f"row {index}: element_class must be non-empty text")
        expected_modality = "image" if element_class == "Picture" else "table" if element_class == "Table" else "text"
        expected_content_type = "image/png" if element_class == "Picture" else "text/markdown"
        if row["modality"] != expected_modality or row["content_type"] != expected_content_type:
            raise ValueError(f"row {index}: modality/content_type do not match {element_class!r}")
        if not isinstance(row["text_content"], str):
            raise ValueError(f"row {index}: text_content must be text, including for textless pictures")
        if element_class == "Picture":
            if not isinstance(row["binary_content"], bytes) or not row["binary_content"].startswith(_PNG_SIGNATURE):
                raise ValueError(f"row {index}: Picture must carry inline PNG bytes")
        elif row["binary_content"] is not None:
            raise ValueError(f"row {index}: only Picture may carry binary_content")
        _validated_bbox(row["bbox_xyxy_norm_json"], row_index=index)
        if row["bbox_coordinate_space"] != COORDINATE_SPACE:
            raise ValueError(f"row {index}: unexpected bbox coordinate space")
        elements.setdefault(key, []).append(element_index)

    for key, indexes in elements.items():
        if key not in outcomes:
            raise ValueError(f"elements have no page outcome for {key[0]!r} page {key[1]}")
        declared_count, outcome = outcomes[key]
        if outcome != "parsed":
            raise ValueError(f"non-parsed page {key[0]!r} page {key[1]} has elements")
        if sorted(indexes) != list(range(len(indexes))):
            raise ValueError(f"element indexes are not contiguous for {key[0]!r} page {key[1]}")
        if declared_count != len(indexes):
            raise ValueError(f"element count mismatch for {key[0]!r} page {key[1]}")
    for key, (declared_count, _outcome) in outcomes.items():
        if declared_count and key not in elements:
            raise ValueError(f"declared elements are missing for {key[0]!r} page {key[1]}")
    return frame


def project_nrl_pages(data: Any, *, evidence_root: str | None = None) -> pd.DataFrame:
    """Project an NRL page batch into page outcomes and ordered elements."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"projection input must be a pandas DataFrame, got {type(data).__name__}")
    rows: list[dict[str, Any]] = []
    for row in data.to_dict("records"):
        if evidence_root is not None:
            from nrl_compare import capture_page

            page_image = row.get("page_image")
            metadata = row.get("metadata")
            parser_metadata = row.get("nemotron_parse_v1_2")
            if isinstance(page_image, Mapping) and isinstance(parser_metadata, Mapping):
                raw_output = parser_metadata.get("raw_output")
                if isinstance(raw_output, str) and isinstance(page_image.get("image_b64"), str):
                    parse_error = parser_metadata.get("error")
                    capture_page(
                        evidence_root,
                        source_path=_source_path(row),
                        native_page_number=_native_page_number(row),
                        image_bytes=base64.b64decode(page_image["image_b64"], validate=True),
                        raw_output=raw_output,
                        extraction_error=metadata.get("error") if isinstance(metadata, Mapping) else None,
                        parse_error=parse_error,
                        # The pinned local actor marks every non-stop completion as an error.
                        finish_reason="stop" if parse_error is None else "unknown",
                    )
        try:
            rows.extend(_project_page(row))
        except Exception as exc:
            rows.append(
                _failed_page_row(
                    row,
                    {"kind": "projection_error", "type": type(exc).__name__, "message": str(exc)},
                    raw_output_sha256=_row_raw_output_sha256(row),
                )
            )
    return validate_projection_envelope(pd.DataFrame(rows, columns=PROJECTION_COLUMNS))


class NRLCuratorProjectionOperator(AbstractOperator, CPUOperator):
    """Terminal CPU operator producing Curator-neutral page records."""

    PRESERVE_PANDAS_OUTPUT = True

    def __init__(self, *, evidence_root: str | None = None) -> None:
        super().__init__()
        self.evidence_root = evidence_root

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> pd.DataFrame:
        return project_nrl_pages(data, evidence_root=self.evidence_root)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def build_projection_graph(*, evidence_root: str | None = None) -> Any:
    """Build the existing NRL PDF graph plus one terminal CPU projection."""

    from nemo_retriever.operators.extract.parse.nemotron_parse import NEMOTRON_PARSE_DEFAULT_TASK_PROMPT

    if NEMOTRON_PARSE_DEFAULT_TASK_PROMPT != PARSE_TASK_PROMPT:
        raise RuntimeError("NRL's default Nemotron Parse prompt no longer matches the pinned v1.2 contract")

    extract_params = ExtractParams(
        method="nemotron_parse",
        extract_text=False,
        extract_images=False,
        extract_tables=True,
        extract_charts=True,
        extract_infographics=True,
        extract_page_as_image=True,
        use_page_elements=False,
        use_table_structure=False,
        dpi=200,
        image_format="png",
        render_mode="full_dpi",
        nemotron_parse_model=PARSE_MODEL,
    )
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=extract_params,
        split_config={},
        stage_order=(),
    )
    return graph >> NRLCuratorProjectionOperator(evidence_root=evidence_root)


def _projection_worker_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_PROJECTION_WORKERS:
        raise ValueError(f"projection_workers must be an integer from 1 through {MAX_PROJECTION_WORKERS}")
    return value


def _validate_run_mode(run_mode: str) -> None:
    if run_mode not in {"batch", "inprocess"}:
        raise ValueError("run_mode must be 'batch' or 'inprocess'")


def _require_local_parse_resolution(resources: Any) -> None:
    """Fail closed unless NRL resolves Nemotron Parse to its local GPU actor."""

    from nemo_retriever.operators.extract.parse.nemotron_parse import (
        NemotronParseActor,
        NemotronParseGPUActor,
    )

    resolved = NemotronParseActor.resolve_operator_class(
        resources,
        operator_kwargs={
            "nemotron_parse_model": PARSE_MODEL,
            "nemotron_parse_invoke_url": None,
            "invoke_url": None,
        },
    )
    if resolved is not NemotronParseGPUActor:
        raise RuntimeError(
            "The NRL-to-Curator recipe requires a Ray runtime with a visible GPU so "
            "Nemotron Parse v1.2 resolves to the local GPU actor; remote CPU/NIM fallback is disabled"
        )


def _prepare_executor_for_local_parse(executor: Any, *, run_mode: str) -> None:
    """Validate the resources the selected executor will use before inference."""

    if run_mode == "batch":
        resources = executor._preflight_cluster_resources
        if resources is None:
            from nemo_retriever.common.ray_resource_hueristics import gather_cluster_resources
            from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime

            ray = ensure_local_ray_runtime(executor._ray_address)
            resources = gather_cluster_resources(ray)
        _require_local_parse_resolution(resources)
        executor._preflight_cluster_resources = resources
        return

    from nemo_retriever.common.ray_resource_hueristics import gather_local_resources

    _require_local_parse_resolution(gather_local_resources())


def build_projection_executor(
    graph: Any,
    *,
    run_mode: str = "batch",
    projection_workers: int = MAX_PROJECTION_WORKERS,
    projection_block_rows: int | None = None,
    parse_batch_size: int = contract.DEFAULT_PARSE_BATCH_SIZE,
    parse_cpus: int = contract.DEFAULT_PARSE_CPUS,
) -> Any:
    """Schedule one Parse actor and a bounded projection pool without changing model settings."""

    workers = _projection_worker_count(projection_workers)
    _validate_run_mode(run_mode)
    contract.validate_parse_scheduling(parse_batch_size, parse_cpus, run_mode=run_mode)
    contract.validate_projection_block_rows(projection_block_rows, run_mode=run_mode)
    if run_mode == "batch":
        return RayDataExecutor(
            graph,
            node_overrides={
                "NemotronParseActor": {"batch_size": parse_batch_size, "num_cpus": parse_cpus},
                NRLCuratorProjectionOperator.__name__: {
                    "concurrency": workers,
                    "num_cpus": 1,
                    **(
                        {"target_num_rows_per_block": projection_block_rows}
                        if projection_block_rows is not None
                        else {}
                    ),
                },
            },
            auto_concurrency_nodes={NRLCuratorProjectionOperator.__name__},
            source_cpu_reservation=1,
        )
    if run_mode == "inprocess":
        return InprocessExecutor(graph)
    raise AssertionError("validated run mode was not handled")


def _executor_stats(dataset: Any, result: pd.DataFrame) -> str:
    """Read resolved settings and timings from the already-executed pinned Ray plan."""

    from ray.data._internal.logical.operators.map_operator import MapBatches

    settings = []
    for node in dataset._logical_plan.dag.post_order_iter():
        if isinstance(node, MapBatches):
            operator = (node.fn_constructor_kwargs or {}).get("operator_class")
            settings.append(
                {
                    "operator": operator.__name__ if operator else node.name,
                    "batch_size": node.batch_size,
                    "batch_format": node.batch_format,
                    "compute": repr(node.compute),
                    "num_cpus": node.ray_remote_args.get("num_cpus"),
                    "num_gpus": node.ray_remote_args.get("num_gpus"),
                }
            )
    diagnostics = {
        "resolved_map_batches": settings,
        "compact_result_pandas_estimate_bytes": int(result.memory_usage(deep=True).sum()),
    }
    return json.dumps(diagnostics, indent=2) + "\n\n" + dataset.stats()


def run_nrl_graph(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    run_mode: str = "batch",
    projection_workers: int = MAX_PROJECTION_WORKERS,
    projection_block_rows: int | None = None,
    parse_batch_size: int = contract.DEFAULT_PARSE_BATCH_SIZE,
    parse_cpus: int = contract.DEFAULT_PARSE_CPUS,
    evidence_root: str | None = None,
    executor_stats_path: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """Run extraction once and return the validated projection envelope."""

    _projection_worker_count(projection_workers)
    _validate_run_mode(run_mode)
    contract.validate_parse_scheduling(parse_batch_size, parse_cpus, run_mode=run_mode)
    contract.validate_projection_block_rows(projection_block_rows, run_mode=run_mode)
    if executor_stats_path is not None and run_mode != "batch":
        raise ValueError("executor statistics require batch mode")
    if isinstance(paths, (str, os.PathLike)):
        normalized_paths = [os.fspath(paths)]
    else:
        normalized_paths = [os.fspath(path) for path in paths]
    if not normalized_paths:
        return pd.DataFrame(columns=PROJECTION_COLUMNS, dtype=object)

    graph = build_projection_graph(evidence_root=evidence_root) if evidence_root else build_projection_graph()
    executor = build_projection_executor(
        graph,
        run_mode=run_mode,
        projection_workers=projection_workers,
        projection_block_rows=projection_block_rows,
        parse_batch_size=parse_batch_size,
        parse_cpus=parse_cpus,
    )
    _prepare_executor_for_local_parse(executor, run_mode=run_mode)
    if (evidence_root or executor_stats_path is not None) and run_mode == "batch":
        from nemo_retriever.graph.executor import ray_dataset_to_pandas

        dataset = executor.build_dataset(normalized_paths)
        result = ray_dataset_to_pandas(dataset)
        stats = _executor_stats(dataset, result)
        stats_paths = {Path(executor_stats_path)} if executor_stats_path is not None else set()
        if evidence_root:
            stats_paths.add(Path(evidence_root) / "executor_stats.txt")
        for stats_path in stats_paths:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(stats, encoding="utf-8")
        return validate_projection_envelope(result)
    return validate_projection_envelope(executor.ingest(normalized_paths))
