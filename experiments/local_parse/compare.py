# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the persisted page content and timing of two published Parse shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import validate_page_budget_outputs

from .shards import _SCHEMA, _published_attempt, _sha256

_CONTENT_FIELDS = ("row_structure", "text", "source_refs_and_bboxes", "picture_payloads", "failure_details")
_CONTROLLER_TIME = "wall_seconds_including_engine_startup_and_readback"
_COLUMNS = (
    "pdf_name",
    "page_number",
    "modality",
    "element_class",
    "content_type",
    "text_content",
    "source_ref",
    "binary_content",
    "materialize_error",
)
_VerifiedShard = tuple[Path, Path, tuple[Path, ...], dict[str, Any]]


@dataclass(frozen=True)
class _Page:
    pdf_name: str
    status: str
    fingerprints: tuple[str, ...]
    picture_rows: int


def _published_path(path: Path) -> Path:
    path = path.resolve(strict=True)
    if path.is_file() and path.name == "manifest.json":
        return path
    if path.is_dir() and path.name.startswith("attempt-"):
        return path.parent / "manifest.json"
    candidates = list(path.glob("outputs/*/manifest.json"))
    if len(candidates) != 1:
        msg = f"Expected one published shard manifest under {path}"
        raise ValueError(msg)
    return candidates[0]


def _verify(reference: Path, input_manifest: Path, input_sha: str) -> _VerifiedShard:
    published = _published_path(reference)
    receipt = json.loads(published.read_text(encoding="utf-8"))
    attempt, outputs, expected = _published_attempt(
        published,
        {"schema": _SCHEMA, "shard_id": published.parent.name, "input_manifest_sha256": input_sha},
    )
    if reference.is_dir() and reference.name.startswith("attempt-") and reference.resolve() != attempt:
        msg = f"Attempt is outside the published output inventory: {reference}"
        raise ValueError(msg)
    validation = validate_page_budget_outputs(
        str(input_manifest),
        [str(path) for path in outputs],
        expected_output_version=receipt["output_version"],
        url_field=receipt["url_field"],
        file_name_field=receipt["file_name_field"],
    )
    if validation != expected or not validation["pages"]:
        msg = f"Current inventory differs from published validation: {published}"
        raise ValueError(msg)
    return published, attempt, outputs, validation


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_pages(paths: tuple[Path, ...]) -> dict[tuple[str, int], _Page]:
    pages = {}
    for path in paths:
        rows_by_page = defaultdict(list)
        info = None
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=_COLUMNS):
            for row in batch.to_pylist():
                if row["modality"] == "metadata":
                    info = json.loads(row["text_content"])
                    pdf_name = row["pdf_name"]
                    continue
                binary = row.pop("binary_content")
                row["binary_present"] = bool(binary)
                row["binary_sha256"] = hashlib.sha256(binary).hexdigest() if binary is not None else None
                rows_by_page[row["page_number"]].append(row)
        if info is None:
            msg = f"Missing document inventory in {path}"
            raise ValueError(msg)
        document_id = info["pdf_provenance"]["document_id"]
        for outcome in info["page_outcomes"]:
            key = (document_id, outcome["page_number"])
            if key in pages:
                msg = f"Duplicate physical page {key}"
                raise ValueError(msg)
            rows = rows_by_page.get(key[1], [])
            pictures = [row["binary_sha256"] for row in rows if row["modality"] == "image"]
            populated = any(row["binary_present"] or (row["text_content"] or "").strip() for row in rows)
            status = outcome["status"] if outcome["status"] == "failed" else ("populated" if populated else "empty")
            fields = (
                [(row["modality"], row["element_class"], row["content_type"]) for row in rows],
                [row["text_content"] for row in rows],
                [json.loads(row["source_ref"]) if row["source_ref"] else None for row in rows],
                pictures,
                [row["materialize_error"] for row in rows],
            )
            pages[key] = _Page(pdf_name, status, tuple(map(_fingerprint, fields)), len(pictures))
    return pages


def _rate(seconds: object, populated: int) -> dict[str, float] | None:
    if type(seconds) in {int, float} and math.isfinite(seconds) and seconds > 0:
        return {"seconds": seconds, "populated_pages_per_second": populated / seconds}
    return None


def _summarize(verified: _VerifiedShard, pages: dict[tuple[str, int], _Page], input_sha: str) -> dict[str, Any]:
    published, attempt, _, validation = verified
    counts = dict.fromkeys(("populated", "empty", "failed"), 0)
    for page in pages.values():
        counts[page.status] += 1
    if len(pages) != validation["pages"] or counts["failed"] != validation["failed_pages"]:
        msg = f"Physical page census differs from published validation: {published}"
        raise ValueError(msg)
    pipeline_path = attempt / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8")) if pipeline_path.is_file() else {}
    output_root = published.parent.parent
    controller_path = output_root.parent / "controller-result.json"
    controller = (
        json.loads(controller_path.read_text(encoding="utf-8"))
        if output_root.name == "outputs" and controller_path.is_file()
        else {}
    )
    controller_matches = (
        controller.get("status") == "STRUCTURALLY_COMPLETE"
        and controller.get("manifest_sha256") == input_sha
        and controller.get("pages") == len(pages)
    )
    return {
        "published_manifest": str(published),
        "attempt": str(attempt),
        "structural_validation": validation,
        "page_counts": counts,
        "picture_rows": sum(page.picture_rows for page in pages.values()),
        "profile": controller.get("profile") if controller_matches else None,
        "timing_eligible": controller.get("timing_eligible") if controller_matches else None,
        "controller_startup_through_readback": _rate(controller.get(_CONTROLLER_TIME), counts["populated"])
        if controller_matches
        else None,
        "pipeline_return_diagnostic": _rate(pipeline.get("seconds"), counts["populated"]),
    }


def compare_shards(
    reference: Path, candidate: Path, input_manifest: Path, *, max_examples: int = 12
) -> dict[str, Any]:
    """Require the same frozen physical pages and report bounded differences."""
    input_sha = _sha256(input_manifest)
    base = _verify(reference, input_manifest, input_sha)
    other = _verify(candidate, input_manifest, input_sha)
    base_pages, other_pages = _read_pages(base[2]), _read_pages(other[2])
    if base_pages.keys() != other_pages.keys() or _sha256(input_manifest) != input_sha:
        msg = "The comparison requires unchanged, identical physical-page coverage"
        raise ValueError(msg)
    totals: Counter[str] = Counter(dict.fromkeys(("status", *_CONTENT_FIELDS), 0))
    statuses: Counter[tuple[str, str]] = Counter()
    examples = []
    changed = 0
    for key, before in base_pages.items():
        after = other_pages[key]
        statuses[before.status, after.status] += 1
        reasons = [
            field
            for field, left, right in zip(_CONTENT_FIELDS, before.fingerprints, after.fingerprints, strict=True)
            if left != right
        ]
        if before.status != after.status:
            reasons.insert(0, "status")
        if reasons:
            changed += 1
            totals.update(reasons)
            if len(examples) < max_examples:
                examples.append(
                    {
                        "document_id": key[0],
                        "pdf_name": before.pdf_name,
                        "physical_page_zero_based": key[1],
                        "reference_status": before.status,
                        "candidate_status": after.status,
                        "changes": reasons,
                    }
                )
    return {
        "input_manifest_sha256": input_sha,
        "physical_pages": len(base_pages),
        "exact_page_matches": len(base_pages) - changed,
        "changed_pages": changed,
        "pages_changed_by_field": dict(totals),
        "new_failures": sum(count for (a, b), count in statuses.items() if a != "failed" and b == "failed"),
        "recovered_failures": sum(count for (a, b), count in statuses.items() if a == "failed" and b != "failed"),
        "status_transitions": {f"{a} -> {b}": count for (a, b), count in sorted(statuses.items())},
        "changed_page_examples": examples,
        "reference": _summarize(base, base_pages, input_sha),
        "candidate": _summarize(other, other_pages, input_sha),
        "quality_scope": "Exact ordered persisted rows and payload hashes. Semantic equivalence remains unverified.",
        "timing_scope": "Guarded controller timing is startup through readback. Pipeline timing may include local engine startup only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--input-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_shards(args.reference, args.candidate, args.input_manifest), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
