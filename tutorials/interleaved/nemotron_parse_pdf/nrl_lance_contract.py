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

# This is a standalone, fail-closed integration recipe spanning optional runtimes.
# ruff: noqa: ANN401, BLE001, C901, EM101, EM102, PLR0911, PLR0912, PLR0913, PLR0915, PLR2004, TRY004, TRY300, TRY301

"""Publish NRL Nemotron Parse output for native NeMo Curator consumption.

Run ``ingest`` in a pinned NeMo Retriever Library environment and ``consume``
in a pinned NeMo Curator environment. LanceDB is the only process boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PARSE_MODEL = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
PARSE_TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
DEFAULT_PARSE_BATCH_SIZE = 64
DEFAULT_PARSE_CPUS = 1
ELEMENT_TABLE = "pdf_elements"
RUN_STATE_FILE = "run_state.json"
HANDOFF_MANIFEST_FILE = "handoff_manifest.json"
CONSUME_REPORT_FILE = "consume_validation.json"
COMPLETION_MANIFEST_FILE = "completion_manifest.json"
ALLOWED_ROOT = Path("/raid")
COORDINATE_SPACE = "normalized_1664x2048_padded_canvas"
_HASH_BLOCK_BYTES = 8 * 1024 * 1024
_HANDOFF_HASH_FIELD = "handoff_sha256"
_REPORT_HASH_FIELD = "report_sha256"
_COMPLETION_HASH_FIELD = "completion_sha256"
PUBLICATION_POLICY = "validated_pages_v1"
_PUBLISHABLE_STATUSES = frozenset({"success", "valid_blank", "partial"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def validate_parse_scheduling(parse_batch_size: int, parse_cpus: int, *, run_mode: str) -> None:
    """Validate recipe-only Parse scheduling without importing the NRL runtime."""

    for name, value in (("parse_batch_size", parse_batch_size), ("parse_cpus", parse_cpus)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if parse_batch_size == 1:
        raise ValueError("parse_batch_size must be at least 2; the pinned NRL executor promotes batch size 1 to 64")
    if run_mode != "batch" and (parse_batch_size != DEFAULT_PARSE_BATCH_SIZE or parse_cpus != DEFAULT_PARSE_CPUS):
        raise ValueError("Nondefault Parse scheduling controls require batch mode")


def validate_projection_block_rows(value: int | None, *, run_mode: str) -> None:
    """Validate optional page-block splitting before the CPU projection pool."""

    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("projection_block_rows must be a positive integer or None")
    if run_mode != "batch":
        raise ValueError("projection_block_rows requires batch mode")


@dataclass
class DocumentBuild:
    """Validated publication result for one canonical PDF."""

    status: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    blank_page_count: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)
    page_outcomes: list[dict[str, Any]] = field(default_factory=list)


class MarkerDurabilityUnconfirmedError(OSError):
    """A marker is visible and authoritative, but its directory sync failed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Marker is visible at {path}; durability is unconfirmed. Do not overwrite or retry this run."
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(_HASH_BLOCK_BYTES):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a marker atomically without replacing an existing marker."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        try:
            _fsync_directory(path.parent)
        except OSError as exc:
            raise MarkerDurabilityUnconfirmedError(path) from exc
    finally:
        # Cleanup must not disguise either a publication failure or visibility.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _seal_payload(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    if hash_field in payload:
        raise ValueError(f"payload already contains reserved field {hash_field!r}")
    sealed = copy.deepcopy(dict(payload))
    sealed[hash_field] = _sha256_bytes(_canonical_json_bytes(sealed))
    return sealed


def _verify_sealed_payload(payload: Mapping[str, Any], hash_field: str, *, label: str) -> str:
    declared = payload.get(hash_field)
    if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
        raise ValueError(f"{label} has an invalid {hash_field}")
    core = {key: value for key, value in payload.items() if key != hash_field}
    actual = _sha256_bytes(_canonical_json_bytes(core))
    if actual != declared:
        raise ValueError(f"{label} SHA-256 is {actual}; expected {declared}")
    return actual


def _load_sealed_json(path: Path, hash_field: str, *, label: str) -> dict[str, Any]:
    payload = _load_json(path)
    _verify_sealed_payload(payload, hash_field, label=label)
    return payload


def _confirm_marker_durability(path: Path, hash_field: str, *, expected_sha256: str) -> dict[str, Any]:
    """Verify an existing sealed marker and sync it without replacing its bytes."""

    with path.open("rb") as stream:
        payload = json.load(stream)
        digest = _verify_sealed_payload(payload, hash_field, label=str(path))
        if digest != expected_sha256:
            raise ValueError(f"Marker at {path} differs from the expected SHA-256")
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)
    return payload


def _require_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must resolve under {root}; got {resolved}") from exc
    return resolved


def _require_descendant(path: Path, root: Path, label: str) -> Path:
    resolved = _require_under(path, root, label)
    if resolved == root.resolve():
        raise ValueError(f"{label} must be a specific path below {root}; got {resolved}")
    return resolved


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False


def _pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        count = len(document)
    finally:
        document.close()
    if count <= 0:
        raise ValueError("PDF contains no pages")
    return count


def _source_records_from_directory(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(
        _require_under(path, ALLOWED_ROOT, "PDF input")
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    return [{"path": str(path), "url": None, "valid_blank_pages": []} for path in paths]


def _source_records_from_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("path"), str):
                raise ValueError(f"{manifest_path}:{line_number} must contain an object with a string 'path'")
            path = Path(value["path"]).expanduser()
            if not path.is_absolute():
                path = manifest_path.parent / path
            path = _require_under(path, ALLOWED_ROOT, f"{manifest_path}:{line_number} PDF input")
            blank_pages = value.get("valid_blank_pages", [])
            if not isinstance(blank_pages, list) or any(
                isinstance(page, bool) or not isinstance(page, int) or page < 0 for page in blank_pages
            ):
                raise ValueError(
                    f"{manifest_path}:{line_number} valid_blank_pages must be a list of zero-based integers"
                )
            records.append(
                {
                    "path": str(path),
                    "url": str(value["url"]) if value.get("url") is not None else None,
                    "valid_blank_pages": sorted(set(blank_pages)),
                }
            )
    return records


def inventory_sources(source_records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hash, preflight, and exact-deduplicate input PDFs."""

    inputs: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    representative_by_hash: dict[str, dict[str, Any]] = {}

    for input_index, source in enumerate(source_records):
        path = Path(str(source["path"])).resolve()
        entry: dict[str, Any] = {
            "input_index": input_index,
            "path": str(path),
            "url": source.get("url"),
            "valid_blank_pages": sorted(set(source.get("valid_blank_pages", []))),
            "content_sha256": None,
            "size_bytes": None,
            "expected_page_count": None,
            "preflight_error": None,
            "representative_input_index": None,
            "representative_path": None,
            "status": "pending",
            "representative_status": None,
            "publication_status": "unpublished",
        }
        try:
            if path.suffix.lower() != ".pdf":
                raise ValueError("input does not have a .pdf extension")
            digest, size = _sha256_file(path)
        except Exception as exc:
            entry["status"] = "failed"
            entry["preflight_error"] = {"type": type(exc).__name__, "message": str(exc)}
            inputs.append(entry)
            continue

        entry["content_sha256"] = digest
        entry["size_bytes"] = size
        representative = representative_by_hash.get(digest)
        if representative is None:
            entry["representative_input_index"] = input_index
            entry["representative_path"] = str(path)
            try:
                entry["expected_page_count"] = _pdf_page_count(path)
            except Exception as exc:
                entry["preflight_error"] = {"type": type(exc).__name__, "message": str(exc)}
                entry["status"] = "failed"
            representative_by_hash[digest] = entry
            representatives.append(entry)
        else:
            entry["representative_input_index"] = representative["input_index"]
            entry["representative_path"] = representative["path"]
            entry["expected_page_count"] = representative["expected_page_count"]
            entry["preflight_error"] = copy.deepcopy(representative["preflight_error"])
            entry["status"] = "duplicate"
        inputs.append(entry)

    by_hash: dict[str, list[dict[str, Any]]] = {}
    for entry in inputs:
        digest = entry.get("content_sha256")
        if isinstance(digest, str):
            by_hash.setdefault(digest, []).append(entry)

    for representative in representatives:
        aliases = by_hash[str(representative["content_sha256"])]
        representative["aliases"] = [
            {
                "path": alias["path"],
                "url": alias.get("url"),
                "input_index": alias["input_index"],
                "valid_blank_pages": list(alias["valid_blank_pages"]),
            }
            for alias in aliases
        ]
        declared_blank_pages = sorted({page for alias in aliases for page in alias["valid_blank_pages"]})
        representative["document_valid_blank_pages"] = declared_blank_pages
        expected_pages = representative.get("expected_page_count")
        if isinstance(expected_pages, int) and any(page >= expected_pages for page in declared_blank_pages):
            representative["preflight_error"] = {
                "type": "InvalidBlankPageDeclaration",
                "message": (f"declared blank page is outside zero-based page range 0..{expected_pages - 1}"),
            }
            representative["status"] = "failed"
        for alias in aliases:
            alias["preflight_error"] = copy.deepcopy(representative["preflight_error"])
            if alias["status"] == "duplicate":
                alias["representative_status"] = representative["status"]
    return inputs, representatives


def _rehash_source_inventory(inputs: Sequence[dict[str, Any]]) -> int:
    validated = 0
    for entry in inputs:
        expected_digest = entry.get("content_sha256")
        expected_size = entry.get("size_bytes")
        if expected_digest is None:
            continue
        if not isinstance(expected_digest, str) or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
            raise ValueError(f"input {entry['input_index']} has an invalid inventory SHA-256")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"input {entry['input_index']} has an invalid inventory size")
        path = _require_under(Path(str(entry["path"])), ALLOWED_ROOT, "source input")
        actual_digest, actual_size = _sha256_file(path)
        if actual_digest != expected_digest or actual_size != expected_size:
            raise RuntimeError(
                f"source input changed after inventory: {path} (sha256={actual_digest}, size={actual_size})"
            )
        validated += 1
    return validated


def _canonical_issues(value: Any) -> list[dict[str, Any]]:
    if value is None or value == "":
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise ValueError("issues_json must encode an array of objects")
    return [dict(item) for item in parsed]


def _normalize_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(coordinate, bool) for coordinate in value):
        return None
    try:
        bbox = [float(coordinate) for coordinate in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0 for coordinate in bbox):
        return None
    left, top, right, bottom = bbox
    if left >= right or top >= bottom:
        return None
    return bbox


def _base_element_row(
    document: Mapping[str, Any],
    *,
    run_id: str,
    position: int,
    modality: str,
    content_type: str,
    text_content: str | None,
    binary_content: bytes | None,
    page_number: int | None,
    element_class: str | None,
    bbox: list[float] | None,
) -> dict[str, Any]:
    aliases = list(document["aliases"])
    primary_url = next((alias["url"] for alias in aliases if alias.get("url")), None)
    return {
        "sample_id": document["content_sha256"],
        "position": position,
        "modality": modality,
        "content_type": content_type,
        "text_content": text_content,
        "binary_content": binary_content,
        "source_ref": None,
        "materialize_error": None,
        "url": primary_url,
        "page_number": page_number,
        "pdf_name": Path(str(document["path"])).name,
        "element_class": element_class,
        "source_path": document["path"],
        "source_aliases": json.dumps(aliases, sort_keys=True, separators=(",", ":")),
        "content_sha256": document["content_sha256"],
        "bbox_xyxy_norm": bbox,
        "bbox_coordinate_space": COORDINATE_SPACE if bbox is not None else None,
        "run_id": run_id,
    }


def validate_page_outcomes(
    page_outcomes: Sequence[Mapping[str, Any]],
    *,
    expected_page_count: int,
    extraction_status: str,
    issues: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Validate explicit coverage independently of whether its rows were delivered."""

    if (
        isinstance(expected_page_count, bool)
        or not isinstance(expected_page_count, int)
        or expected_page_count <= 0
        or not isinstance(page_outcomes, list)
        or len(page_outcomes) != expected_page_count
        or not isinstance(issues, list)
        or any(not isinstance(issue, Mapping) for issue in issues)
    ):
        raise ValueError("Invalid document page coverage")
    counts = dict.fromkeys(
        (
            "validated_page_count",
            "content_page_count",
            "blank_page_count",
            "failed_page_count",
            "content_element_count",
        ),
        0,
    )
    for number, page in enumerate(page_outcomes):
        if (
            not isinstance(page, Mapping)
            or isinstance(page.get("page_number"), bool)
            or not isinstance(page.get("page_number"), int)
            or page["page_number"] != number
        ):
            raise ValueError("Page outcomes must enumerate every expected zero-based page exactly once")
        status, count, page_issues = page.get("status"), page.get("element_count"), page.get("issues")
        if (
            not isinstance(status, str)
            or status not in {"success", "valid_blank", "failed"}
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(page_issues, list)
            or any(not isinstance(issue, Mapping) for issue in page_issues)
            or bool(page_issues) != (status == "failed")
            or (count > 0) != (status == "success")
        ):
            raise ValueError(f"Invalid page outcome for page {number}")
        counts["content_element_count"] += count
        counts["failed_page_count"] += status == "failed"
        counts["blank_page_count"] += status == "valid_blank"
        counts["content_page_count"] += status == "success"
        counts["validated_page_count"] += status != "failed"
    if counts["failed_page_count"] and not issues:
        raise ValueError("Failed pages require document issues")
    expected_status = (
        "failed"
        if not counts["validated_page_count"]
        else "partial"
        if issues
        else "success"
        if counts["content_page_count"]
        else "valid_blank"
    )
    if extraction_status != expected_status:
        raise ValueError(f"Extraction status {extraction_status!r} differs from page coverage {expected_status!r}")
    return counts


def build_document_rows(
    document: Mapping[str, Any],
    envelope_rows: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
) -> DocumentBuild:
    """Publish only whole validated pages, retaining explicit incomplete coverage."""

    expected_page_count = document.get("expected_page_count")
    if isinstance(expected_page_count, bool) or not isinstance(expected_page_count, int) or expected_page_count <= 0:
        return DocumentBuild(
            status="failed",
            issues=[{"kind": "invalid_expected_page_count", "value": expected_page_count}],
        )

    valid_blank_pages = set(document.get("document_valid_blank_pages", []))
    expected_source_path = str(Path(str(document["path"])).resolve())
    markers: dict[int, list[Mapping[str, Any]]] = {}
    elements: dict[int, list[Mapping[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    invalid_record_pages: set[int] = set()

    for row in envelope_rows:
        source_path = row.get("source_path")
        if not isinstance(source_path, str) or str(Path(source_path).resolve()) != expected_source_path:
            issues.append({"kind": "source_path_mismatch", "value": source_path})
            continue
        record_type = row.get("record_type")
        native_page = row.get("native_page_number")
        if isinstance(native_page, bool) or not isinstance(native_page, int) or native_page < 0:
            issues.append({"kind": "invalid_native_page_number", "value": native_page})
            continue
        if record_type == "page_outcome":
            if any(
                row.get(field) is not None
                for field in (
                    "element_index",
                    "element_class",
                    "modality",
                    "content_type",
                    "text_content",
                    "binary_content",
                    "bbox_xyxy_norm_json",
                    "bbox_coordinate_space",
                )
            ):
                issues.append({"kind": "invalid_page_outcome_record", "native_page_number": native_page})
                invalid_record_pages.add(native_page)
            markers.setdefault(native_page, []).append(row)
        elif record_type == "element":
            try:
                element_issues = _canonical_issues(row.get("issues_json"))
            except Exception as exc:
                element_issues = [{"kind": "invalid_issues_json", "message": str(exc)}]
            if (
                row.get("page_outcome") is not None
                or row.get("element_count") is not None
                or row.get("raw_output_sha256") is not None
                or element_issues
            ):
                issues.append({"kind": "invalid_element_record", "native_page_number": native_page})
                invalid_record_pages.add(native_page)
            elements.setdefault(native_page, []).append(row)
        else:
            issues.append({"kind": "invalid_record_type", "native_page_number": native_page, "value": record_type})
            invalid_record_pages.add(native_page)

    if 0 in markers or 0 in elements:
        issues.append({"kind": "document_or_split_failure", "native_page_number": 0})
    expected_pages = set(range(1, expected_page_count + 1))
    actual_positive_pages = {page for page in markers if page > 0}
    missing_pages = sorted(expected_pages - actual_positive_pages)
    unexpected_pages = sorted(actual_positive_pages - expected_pages)
    if missing_pages:
        issues.append({"kind": "missing_pages", "native_page_numbers": missing_pages})
    if unexpected_pages:
        issues.append({"kind": "unexpected_pages", "native_page_numbers": unexpected_pages})
    duplicate_pages = sorted(page for page, values in markers.items() if len(values) != 1)
    if duplicate_pages:
        issues.append({"kind": "duplicate_page_outcomes", "native_page_numbers": duplicate_pages})
    markerless_element_pages = sorted(set(elements) - set(markers))
    if markerless_element_pages:
        issues.append({"kind": "elements_without_page_outcome", "native_page_numbers": markerless_element_pages})

    validated_elements: list[tuple[int, int, Mapping[str, Any], list[float]]] = []
    blank_pages: set[int] = set()
    content_pages: set[int] = set()
    for native_page in sorted(expected_pages & set(markers)):
        page_markers = markers[native_page]
        if len(page_markers) != 1:
            continue
        marker = page_markers[0]
        try:
            marker_issues = _canonical_issues(marker.get("issues_json"))
        except Exception as exc:
            marker_issues = [{"kind": "invalid_issues_json", "message": str(exc)}]
        for issue in marker_issues:
            issues.append({**issue, "native_page_number": native_page})

        outcome = marker.get("page_outcome")
        element_count = marker.get("element_count")
        raw_output_sha256 = marker.get("raw_output_sha256")
        if outcome in {"parsed", "empty"} and (
            not isinstance(raw_output_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", raw_output_sha256) is None
        ):
            issues.append({"kind": "invalid_raw_output_sha256", "native_page_number": native_page})
            continue
        if raw_output_sha256 is not None and (
            not isinstance(raw_output_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", raw_output_sha256) is None
        ):
            issues.append({"kind": "invalid_raw_output_sha256", "native_page_number": native_page})
            continue
        if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count < 0:
            issues.append({"kind": "invalid_element_count", "native_page_number": native_page, "value": element_count})
            continue
        page_elements = list(elements.get(native_page, []))
        if len(page_elements) != element_count:
            issues.append(
                {
                    "kind": "element_count_mismatch",
                    "native_page_number": native_page,
                    "expected": element_count,
                    "actual": len(page_elements),
                }
            )
            continue
        if outcome == "failed":
            issues.append({"kind": "page_failed", "native_page_number": native_page})
            continue
        if outcome == "empty":
            if element_count != 0:
                issues.append({"kind": "empty_page_has_elements", "native_page_number": native_page})
                continue
            zero_based_page = native_page - 1
            if zero_based_page not in valid_blank_pages:
                issues.append({"kind": "unexpected_empty_output", "page_number": zero_based_page})
                continue
            if not marker_issues and native_page not in invalid_record_pages:
                blank_pages.add(native_page)
            continue
        if outcome != "parsed":
            issues.append({"kind": "invalid_page_outcome", "native_page_number": native_page, "value": outcome})
            continue
        if element_count == 0:
            issues.append({"kind": "parsed_page_has_no_elements", "native_page_number": native_page})
            continue

        seen_indices: set[int] = set()
        page_valid = not marker_issues and native_page not in invalid_record_pages
        for element in page_elements:
            element_index = element.get("element_index")
            if (
                isinstance(element_index, bool)
                or not isinstance(element_index, int)
                or element_index < 0
                or element_index in seen_indices
            ):
                issues.append(
                    {
                        "kind": "invalid_element_index",
                        "native_page_number": native_page,
                        "value": element_index,
                    }
                )
                page_valid = False
                continue
            seen_indices.add(element_index)
            bbox = _normalize_bbox(element.get("bbox_xyxy_norm_json"))
            if bbox is None:
                issues.append(
                    {
                        "kind": "invalid_bbox",
                        "native_page_number": native_page,
                        "element_index": element_index,
                    }
                )
                page_valid = False
                continue
            if element.get("bbox_coordinate_space") != COORDINATE_SPACE:
                issues.append(
                    {
                        "kind": "invalid_bbox_coordinate_space",
                        "native_page_number": native_page,
                        "element_index": element_index,
                    }
                )
                page_valid = False
                continue
            modality = element.get("modality")
            content_type = element.get("content_type")
            element_class = element.get("element_class")
            expected_modality = (
                "image" if element_class == "Picture" else "table" if element_class == "Table" else "text"
            )
            expected_content_type = "image/png" if modality == "image" else "text/markdown"
            if (
                modality not in {"text", "table", "image"}
                or modality != expected_modality
                or content_type != expected_content_type
            ):
                issues.append(
                    {
                        "kind": "invalid_element_modality",
                        "native_page_number": native_page,
                        "element_index": element_index,
                    }
                )
                page_valid = False
                continue
            binary = element.get("binary_content")
            if isinstance(binary, memoryview):
                binary = binary.tobytes()
            elif isinstance(binary, bytearray):
                binary = bytes(binary)
            if modality == "image" and (not isinstance(binary, bytes) or not binary.startswith(_PNG_SIGNATURE)):
                issues.append(
                    {
                        "kind": "invalid_picture_bytes",
                        "native_page_number": native_page,
                        "element_index": element_index,
                    }
                )
                page_valid = False
                continue
            if modality != "image" and binary is not None:
                issues.append(
                    {
                        "kind": "unexpected_nonimage_bytes",
                        "native_page_number": native_page,
                        "element_index": element_index,
                    }
                )
                page_valid = False
                continue
            text = element.get("text_content")
            if not isinstance(text, str) or not isinstance(element_class, str) or not element_class:
                issues.append(
                    {
                        "kind": "invalid_element_text_or_class",
                        "native_page_number": native_page,
                        "element_index": element_index,
                    }
                )
                page_valid = False
                continue
            validated_elements.append(
                (
                    native_page,
                    element_index,
                    {**dict(element), "text_content": text, "binary_content": binary},
                    bbox,
                )
            )
        expected_indices = list(range(element_count))
        if sorted(seen_indices) != expected_indices:
            issues.append(
                {
                    "kind": "noncontiguous_page_element_indices",
                    "native_page_number": native_page,
                    "actual": sorted(seen_indices),
                    "expected": expected_indices,
                }
            )
            page_valid = False
        if page_valid:
            content_pages.add(native_page)

    # A page is atomic: even individually valid elements from a rejected page
    # must not leak into the delivered document.
    ordered_elements = sorted(
        (item for item in validated_elements if item[0] in content_pages), key=lambda item: (item[0], item[1])
    )
    page_issues: dict[int, list[dict[str, Any]]] = {page: [] for page in expected_pages}
    for issue in issues:
        native_page = issue.get("native_page_number")
        if native_page is not None:
            affected = [native_page]
        elif isinstance(issue.get("native_page_numbers"), list):
            affected = issue["native_page_numbers"]
        elif isinstance(issue.get("page_number"), int):
            affected = [issue["page_number"] + 1]
        else:
            affected = []
        for native_page in affected:
            if isinstance(native_page, int) and not isinstance(native_page, bool) and native_page in page_issues:
                scoped = {key: value for key, value in issue.items() if key != "native_page_numbers"}
                page_issues[native_page].append({**scoped, "native_page_number": native_page})
    page_outcomes = [
        {
            "page_number": page - 1,
            "status": "success" if page in content_pages else "valid_blank" if page in blank_pages else "failed",
            "element_count": len(elements[page]) if page in content_pages else 0,
            "issues": page_issues[page],
        }
        for page in sorted(expected_pages)
    ]
    status = (
        "failed"
        if not content_pages and not blank_pages
        else "partial"
        if issues
        else "success"
        if content_pages
        else "valid_blank"
    )
    validate_page_outcomes(
        page_outcomes, expected_page_count=expected_page_count, extraction_status=status, issues=issues
    )
    if status == "failed":
        return DocumentBuild(
            status=status,
            issues=issues,
            page_outcomes=page_outcomes,
        )

    metadata_payload = {
        "content_sha256": document["content_sha256"],
        "pdf_name": Path(str(document["path"])).name,
        "num_pages": expected_page_count,
        "source_path": document["path"],
        "source_aliases": list(document["aliases"]),
        "url": next((alias["url"] for alias in document["aliases"] if alias.get("url")), None),
        "valid_blank_pages": sorted(valid_blank_pages),
        "extraction_status": status,
        "page_outcomes": page_outcomes,
        "issues": issues,
    }
    rows = [
        _base_element_row(
            document,
            run_id=run_id,
            position=-1,
            modality="metadata",
            content_type="application/json",
            text_content=json.dumps(metadata_payload, sort_keys=True, separators=(",", ":")),
            binary_content=None,
            page_number=None,
            element_class=None,
            bbox=None,
        )
    ]
    for position, (native_page, _element_index, element, bbox) in enumerate(ordered_elements):
        rows.append(
            _base_element_row(
                document,
                run_id=run_id,
                position=position,
                modality=str(element["modality"]),
                content_type=str(element["content_type"]),
                text_content=str(element["text_content"]),
                binary_content=element["binary_content"],
                page_number=native_page - 1,
                element_class=str(element["element_class"]),
                bbox=bbox,
            )
        )

    return DocumentBuild(
        status=status,
        rows=rows,
        page_count=len(content_pages) + len(blank_pages),
        blank_page_count=len(blank_pages),
        issues=issues,
        page_outcomes=page_outcomes,
    )


def element_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("modality", pa.string(), nullable=False),
            pa.field("content_type", pa.string()),
            pa.field("text_content", pa.string()),
            pa.field("binary_content", pa.large_binary()),
            pa.field("source_ref", pa.string()),
            pa.field("materialize_error", pa.string()),
            pa.field("url", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("pdf_name", pa.string()),
            pa.field("element_class", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_aliases", pa.string()),
            pa.field("content_sha256", pa.string(), nullable=False),
            pa.field("bbox_xyxy_norm", pa.list_(pa.float64())),
            pa.field("bbox_coordinate_space", pa.string()),
            pa.field("run_id", pa.string(), nullable=False),
        ]
    )


def _table_names(connection: Any) -> set[str]:
    names = connection.list_tables()
    return set(names.tables if hasattr(names, "tables") else names)


class ElementTableWriter:
    """Append exactly one validated document per Lance write."""

    def __init__(self, uri: Path) -> None:
        import lancedb

        self.uri = uri
        self.connection = lancedb.connect(str(uri))
        if ELEMENT_TABLE in _table_names(self.connection):
            raise FileExistsError(f"Refusing to append to existing table {ELEMENT_TABLE!r} at {uri}")
        self.table: Any | None = None

    def add_document(self, rows: Sequence[dict[str, Any]]) -> None:
        import pyarrow as pa

        document = pa.Table.from_pylist(list(rows), schema=element_schema())
        if self.table is None:
            self.table = self.connection.create_table(ELEMENT_TABLE, data=document, mode="create")
        else:
            self.table.add(document, mode="append")


def _table_path(uri: Path, table_name: str = ELEMENT_TABLE) -> Path:
    return uri / f"{table_name}.lance"


def _open_lancedb_table(path: Path, *, version: int | None = None) -> Any:
    import lancedb

    table = lancedb.connect(str(path.parent)).open_table(path.name.removesuffix(".lance"))
    if version is not None:
        table.checkout(version)
    return table


def _table_schema(table: Any) -> Any:
    schema = table.schema
    return schema() if callable(schema) else schema


def _project_with_row_ids(table: Any, columns: Sequence[str]) -> Any:
    return table.search().select(list(columns)).with_row_id(True).limit(None).to_arrow()


def _fragment_id(row_id: Any) -> int:
    return int(row_id) >> 32
