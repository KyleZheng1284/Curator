# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

# This standalone lifecycle validates an intentionally wide storage contract.
# ruff: noqa: ANN401, BLE001, C901, EM101, EM102, PLR0912, PLR0915, S110, S603, S607, SIM105, TC003, TRY004, TRY300, TRY301

"""Lifecycle and validation for the NRL-to-Curator Lance recipe."""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import re
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nrl_lance_contract as contract


def _expected_document_provenance(
    inputs: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    sample_ids: set[str],
    documents: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in inputs:
        digest = entry.get("content_sha256")
        if isinstance(digest, str) and digest in sample_ids:
            grouped.setdefault(digest, []).append(entry)
    if set(grouped) != sample_ids:
        raise ValueError("source inventory and published sample identities differ")

    expected: dict[str, dict[str, Any]] = {}
    for sample_id, entries in grouped.items():
        ordered = sorted(entries, key=lambda entry: int(entry["input_index"]))
        representative_index = ordered[0].get("representative_input_index")
        representatives = [entry for entry in ordered if entry.get("input_index") == representative_index]
        if len(representatives) != 1:
            raise ValueError(f"sample {sample_id} does not have exactly one representative")
        representative = representatives[0]
        aliases = [
            {
                "path": str(entry["path"]),
                "url": entry.get("url"),
                "input_index": int(entry["input_index"]),
                "valid_blank_pages": list(entry.get("valid_blank_pages", [])),
            }
            for entry in ordered
        ]
        expected[sample_id] = {
            "source_path": str(representative["path"]),
            "source_name": Path(str(representative["path"])).name,
            "num_pages": int(representative["expected_page_count"]),
            "source_aliases": aliases,
            "url": next((alias["url"] for alias in aliases if alias.get("url")), None),
            "valid_blank_pages": sorted({page for alias in aliases for page in alias["valid_blank_pages"]}),
            "run_id": run_id,
        }
        if documents is not None:
            results = [document for document in documents if document.get("content_sha256") == sample_id]
            if len(results) != 1:
                raise ValueError(f"sample {sample_id} must have exactly one extraction result")
            result = results[0]
            coverage = contract.validate_page_outcomes(
                result.get("page_outcomes"),
                expected_page_count=expected[sample_id]["num_pages"],
                extraction_status=result.get("extraction_status"),
                issues=result.get("issues"),
            )
            if (
                result.get("status") != result.get("extraction_status")
                or result.get("expected_page_count") != expected[sample_id]["num_pages"]
                or result.get("element_count") != coverage["content_element_count"] + 1
                or any(result.get(key) != value for key, value in coverage.items() if key != "content_element_count")
            ):
                raise ValueError(f"Extraction counts differ from page outcomes for sample {sample_id}")
            expected[sample_id].update({key: result[key] for key in ("extraction_status", "page_outcomes", "issues")})
    return expected


def _validate_element_schema(schema: Any, *, label: str) -> None:
    expected = contract.element_schema()
    if schema.names != expected.names:
        raise ValueError(f"{label} fields are {schema.names}; expected {expected.names}")
    for expected_field, actual_field in zip(expected, schema, strict=True):
        if actual_field.type != expected_field.type or actual_field.nullable != expected_field.nullable:
            raise ValueError(f"{label} field {expected_field.name!r} is {actual_field}; expected {expected_field}")


def validate_element_table(
    path: Path,
    expected_element_counts: Mapping[str, int],
    *,
    expected_provenance: Mapping[str, Mapping[str, Any]],
    version: int | None = None,
) -> dict[str, Any]:
    """Validate the exact Arrow contract, document order, and fragment ownership."""

    expected_counts = {str(sample_id): int(count) for sample_id, count in expected_element_counts.items()}
    if not expected_counts or any(count <= 0 for count in expected_counts.values()):
        raise ValueError("Every published document must have at least its metadata row")

    table = contract._open_lancedb_table(path, version=version)
    schema = contract._table_schema(table)
    expected_schema = contract.element_schema()
    _validate_element_schema(schema, label="Element table")

    projected = contract._project_with_row_ids(table, expected_schema.names)
    records = sorted(projected.to_pylist(), key=lambda row: int(row["_rowid"]))
    positions: dict[str, list[int]] = {}
    sample_fragments: dict[str, set[int]] = {}
    fragment_samples: dict[int, set[str]] = {}
    metadata_counts: Counter[str] = Counter()
    image_hashes: dict[str, str] = {}
    page_element_counts: dict[str, Counter[int]] = {}
    last_page: dict[str, int] = {}
    page_outcomes: dict[str, list[dict[str, Any]]] = {}

    for row in records:
        sample_id = row["sample_id"]
        position = int(row["position"])
        positions.setdefault(sample_id, []).append(position)
        fragment_id = contract._fragment_id(row["_rowid"])
        sample_fragments.setdefault(sample_id, set()).add(fragment_id)
        fragment_samples.setdefault(fragment_id, set()).add(sample_id)
        if row["content_sha256"] != sample_id:
            raise ValueError(f"Element identity differs for sample {sample_id}")
        if row["source_ref"] is not None or row["materialize_error"] is not None:
            raise ValueError(f"Published row has a locator or materialization error for sample {sample_id}")
        provenance = expected_provenance.get(sample_id)
        if provenance is None:
            raise ValueError(f"Unexpected sample {sample_id}")
        try:
            aliases = json.loads(row["source_aliases"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid aliases for sample {sample_id}") from exc
        if (
            row["source_path"] != provenance["source_path"]
            or row["pdf_name"] != provenance["source_name"]
            or row["url"] != provenance["url"]
            or row["run_id"] != provenance["run_id"]
            or aliases != provenance["source_aliases"]
        ):
            raise ValueError(f"Element provenance differs for sample {sample_id}")

        modality = row["modality"]
        binary_content = row["binary_content"]
        if position == -1:
            metadata_counts[sample_id] += 1
            if (
                modality != "metadata"
                or row["content_type"] != "application/json"
                or binary_content is not None
                or row["page_number"] is not None
                or row["element_class"] is not None
                or row["bbox_xyxy_norm"] is not None
                or row["bbox_coordinate_space"] is not None
            ):
                raise ValueError(f"Malformed metadata row for sample {sample_id}")
            try:
                metadata = json.loads(row["text_content"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid metadata JSON for sample {sample_id}") from exc
            if (
                metadata.get("content_sha256") != sample_id
                or metadata.get("pdf_name") != provenance["source_name"]
                or metadata.get("num_pages") != provenance["num_pages"]
                or metadata.get("source_path") != provenance["source_path"]
                or metadata.get("source_aliases") != provenance["source_aliases"]
                or metadata.get("url") != provenance["url"]
                or metadata.get("valid_blank_pages") != provenance["valid_blank_pages"]
            ):
                raise ValueError(f"Metadata provenance differs for sample {sample_id}")
            if "page_outcomes" in metadata or "page_outcomes" in provenance:
                contract.validate_page_outcomes(
                    metadata.get("page_outcomes"),
                    expected_page_count=provenance["num_pages"],
                    extraction_status=metadata.get("extraction_status"),
                    issues=metadata.get("issues"),
                )
                if "page_outcomes" in provenance and any(
                    metadata.get(key) != provenance[key] for key in ("extraction_status", "page_outcomes", "issues")
                ):
                    raise ValueError(f"Metadata extraction coverage differs for sample {sample_id}")
                page_outcomes[sample_id] = metadata["page_outcomes"]
                if any(
                    page["status"] == "valid_blank" and page["page_number"] not in provenance["valid_blank_pages"]
                    for page in metadata["page_outcomes"]
                ):
                    raise ValueError(f"Undeclared blank in page outcomes for sample {sample_id}")
            continue

        if position < 0 or modality not in {"text", "table", "image"}:
            raise ValueError(f"Invalid content row for sample {sample_id}")
        page_number = row["page_number"]
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or not 0 <= page_number < provenance["num_pages"]
        ):
            raise ValueError(f"Invalid page number for sample {sample_id}")
        if page_number < last_page.get(sample_id, -1):
            raise ValueError(f"Content page order differs for sample {sample_id}")
        last_page[sample_id] = page_number
        page_element_counts.setdefault(sample_id, Counter())[page_number] += 1
        if contract._normalize_bbox(row["bbox_xyxy_norm"]) is None:
            raise ValueError(f"Invalid bbox for sample {sample_id}")
        if row["bbox_coordinate_space"] != contract.COORDINATE_SPACE:
            raise ValueError(f"Invalid bbox coordinate space for sample {sample_id}")
        expected_type = "image/png" if modality == "image" else "text/markdown"
        if row["content_type"] != expected_type:
            raise ValueError(f"Invalid content type for sample {sample_id}")
        if modality == "image":
            if not isinstance(binary_content, (bytes, bytearray, memoryview)):
                raise ValueError(f"Image row is missing inline bytes for sample {sample_id}")
            image_hashes[f"{sample_id}:{position}"] = contract._sha256_bytes(bytes(binary_content))
        elif binary_content is not None:
            raise ValueError(f"Non-image row contains binary data for sample {sample_id}")

    if set(positions) != set(expected_counts):
        raise ValueError(f"Element samples differ: actual={sorted(positions)}, expected={sorted(expected_counts)}")
    split_samples = {
        sample_id: sorted(fragment_ids)
        for sample_id, fragment_ids in sample_fragments.items()
        if len(fragment_ids) != 1
    }
    if split_samples:
        raise ValueError(f"Documents span multiple Lance fragments: {split_samples}")
    mixed_fragments = {
        fragment_id: sorted(sample_ids) for fragment_id, sample_ids in fragment_samples.items() if len(sample_ids) != 1
    }
    if mixed_fragments:
        raise ValueError(f"Lance fragments contain multiple documents: {mixed_fragments}")
    if len(fragment_samples) != len(positions):
        raise ValueError("Lance fragment and published-document counts differ")
    expected_row_count = sum(expected_counts.values())
    if len(records) != expected_row_count or int(table.count_rows()) != expected_row_count:
        raise ValueError("Element row count differs from completed documents")
    for sample_id, expected_count in expected_counts.items():
        expected_positions = [-1, *range(expected_count - 1)]
        if positions[sample_id] != expected_positions:
            raise ValueError(
                f"Element positions for {sample_id} are {positions[sample_id]}; expected {expected_positions}"
            )
        if metadata_counts[sample_id] != 1:
            raise ValueError(f"Expected exactly one metadata row for sample {sample_id}")
        if sample_id in page_outcomes:
            expected_pages = {
                page["page_number"]: page["element_count"]
                for page in page_outcomes[sample_id]
                if page["status"] == "success"
            }
            if dict(page_element_counts.get(sample_id, {})) != expected_pages:
                raise ValueError(f"Content rows differ from validated page outcomes for sample {sample_id}")

    return {
        "name": contract.ELEMENT_TABLE,
        "path": str(path),
        "version": int(table.version),
        "row_count": expected_row_count,
        "document_count": len(positions),
        "fragment_count": len(fragment_samples),
        "inline_image_count": len(image_hashes),
        "inline_image_sha256": image_hashes,
        "schema_and_nullability_validated": True,
        "positions_validated": True,
        "page_outcomes_validated": len(page_outcomes) == len(positions),
        "documents_per_fragment_validated": True,
        "provenance_validated": True,
    }


def _git_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(path: Path | None) -> bool | None:
    if path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _model_revision(model_id: str) -> str:
    from nemo_retriever.models.hf_model_registry import get_hf_revision

    revision = get_hf_revision(model_id)
    if not revision:
        raise RuntimeError(f"No pinned Hugging Face revision is registered for {model_id}")
    return revision


def _load_graph_module() -> Any:
    # Import by its stable module name so Ray can resolve the terminal operator
    # class in worker processes. The recipe entry point adds this directory to
    # ``sys.path`` before importing the runtime module.
    return importlib.import_module("nrl_graph")


def _validate_nrl_runtime_source(nrl_repo: Path, graph_module: Any) -> str:
    """Bind recorded repository provenance to the code actually imported."""

    source = inspect.getsourcefile(graph_module.build_graph)
    if source is None:
        raise RuntimeError("Could not resolve the imported NRL graph builder source")
    source_path = Path(source).resolve()
    try:
        source_path.relative_to(nrl_repo.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Imported NRL graph builder {source_path} is outside --nrl-repo {nrl_repo}") from exc
    return str(source_path)


def _counts(inputs: Sequence[Mapping[str, Any]], documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    publication_counts = Counter(str(item.get("publication_status", "unknown")) for item in documents)
    delivered = [item for item in documents if item.get("publication_status") in {"handed_off", "published"}]
    unknown_pages = sum(item.get("failed_page_count") is None for item in documents)
    known_failed_pages = sum(int(item.get("failed_page_count") or 0) for item in documents)
    return {
        "inputs": dict(sorted(Counter(str(item["status"]) for item in inputs).items())),
        "documents": dict(sorted(Counter(str(item["status"]) for item in documents).items())),
        "document_publication": dict(sorted(publication_counts.items())),
        "input_count": len(inputs),
        "representative_count": len(documents),
        "publishable_document_count": sum(
            item["status"] in contract._PUBLISHABLE_STATUSES and int(item.get("element_count", 0)) > 0
            for item in documents
        ),
        "handed_off_document_count": publication_counts["handed_off"],
        "published_document_count": publication_counts["published"],
        "element_row_count": sum(int(item.get("element_count", 0)) for item in documents),
        "complete_document_count": sum(item["status"] in {"success", "valid_blank"} for item in documents),
        "partial_document_count": sum(item["status"] == "partial" for item in documents),
        "failed_document_count": sum(item["status"] == "failed" for item in documents),
        "failed_page_count": None if unknown_pages else known_failed_pages,
        "known_failed_page_count": known_failed_pages,
        "unknown_page_count_document_count": unknown_pages,
        "delivered_document_count": len(delivered),
        "delivered_page_count": sum(int(item["validated_page_count"]) for item in delivered),
        "delivered_content_page_count": sum(
            int(item["validated_page_count"]) - int(item["blank_page_count"]) for item in delivered
        ),
        "delivered_content_element_count": sum(int(item["element_count"]) - 1 for item in delivered),
    }


def _publish_completion_and_update_state(
    completion_path: Path,
    completion: Mapping[str, Any],
    *,
    state_path: Path,
    state_updates: Mapping[str, Any],
) -> None:
    """Publish the authoritative marker before updating diagnostic state."""

    try:
        contract._write_json_exclusive_atomic(completion_path, completion)
    except contract.MarkerDurabilityUnconfirmedError:
        try:
            state = contract._load_json(state_path)
            state.update(state_updates)
            state["marker_durability"] = {"path": str(completion_path), "status": "unconfirmed"}
            contract._write_json_atomic(state_path, state)
        except Exception:
            pass
        raise
    try:
        state = contract._load_json(state_path)
        state.update(state_updates)
        state.pop("marker_durability", None)
        contract._write_json_atomic(state_path, state)
    except Exception:
        # The sealed completion marker is authoritative. A diagnostic-state
        # write failure must not turn a completed publication into an error
        # that a caller might retry against the same immutable run.
        return


def _update_input_outcomes(
    inputs: list[dict[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    *,
    publication_status: str,
) -> None:
    by_digest = {str(document["content_sha256"]): document for document in documents}
    for entry in inputs:
        digest = entry.get("content_sha256")
        if not isinstance(digest, str):
            entry["publication_status"] = "not_applicable"
            continue
        document = by_digest.get(digest)
        if document is None:
            continue
        if entry["status"] == "duplicate":
            entry["representative_status"] = document["status"]
        else:
            entry["status"] = document["status"]
        entry["publication_status"] = (
            publication_status
            if document["status"] in contract._PUBLISHABLE_STATUSES and document.get("element_count", 0) > 0
            else "not_applicable"
        )


def _document_result(
    document: Mapping[str, Any],
    build: contract.DocumentBuild,
    *,
    publication_status: str,
) -> dict[str, Any]:
    page_outcomes = build.page_outcomes
    expected_pages = document.get("expected_page_count")
    if not page_outcomes and isinstance(expected_pages, int) and expected_pages > 0:
        page_outcomes = [
            {"page_number": page, "status": "failed", "element_count": 0, "issues": build.issues}
            for page in range(expected_pages)
        ]
    return {
        "content_sha256": document["content_sha256"],
        "path": document["path"],
        "status": build.status,
        "extraction_status": build.status,
        "publication_status": (
            publication_status if build.status in contract._PUBLISHABLE_STATUSES else "not_applicable"
        ),
        "expected_page_count": document.get("expected_page_count"),
        "validated_page_count": build.page_count,
        "blank_page_count": build.blank_page_count,
        "content_page_count": build.page_count - build.blank_page_count,
        "failed_page_count": sum(page["status"] == "failed" for page in page_outcomes) if page_outcomes else None,
        "page_outcomes": page_outcomes,
        "element_count": len(build.rows),
        "issues": build.issues,
    }


def run_ingest(
    args: argparse.Namespace,
    *,
    graph_runner: Callable[..., Any] | None = None,
    envelope_validator: Callable[[Any], Any] | None = None,
) -> Path:
    """Run the extraction-only custom graph and publish a validated Lance handoff."""

    parse_batch_size = getattr(args, "parse_batch_size", contract.DEFAULT_PARSE_BATCH_SIZE)
    parse_cpus = getattr(args, "parse_cpus", contract.DEFAULT_PARSE_CPUS)
    projection_block_rows = getattr(args, "projection_block_rows", None)
    contract.validate_parse_scheduling(parse_batch_size, parse_cpus, run_mode=getattr(args, "run_mode", "batch"))
    contract.validate_projection_block_rows(projection_block_rows, run_mode=getattr(args, "run_mode", "batch"))

    if args.input_dir:
        input_dir = contract._require_descendant(Path(args.input_dir), contract.ALLOWED_ROOT, "input directory")
        source_records = contract._source_records_from_directory(input_dir)
        input_spec = {"input_dir": str(input_dir)}
    else:
        manifest_path = contract._require_under(Path(args.manifest), contract.ALLOWED_ROOT, "input manifest")
        source_records = contract._source_records_from_manifest(manifest_path)
        input_spec = {"manifest": str(manifest_path)}
    if not source_records:
        raise ValueError("No PDF inputs were found")

    output_root = contract._require_under(Path(args.output_root), contract.ALLOWED_ROOT, "output root")
    nrl_repo = contract._require_under(Path(args.nrl_repo), contract.ALLOWED_ROOT, "NRL repository")
    evidence_root = getattr(args, "evidence_root", None)
    if evidence_root is not None:
        evidence_root = str(contract._require_descendant(Path(evidence_root), contract.ALLOWED_ROOT, "evidence root"))
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    if re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None:
        raise ValueError("run-id may contain only letters, numbers, '.', '_', and '-'")
    run_dir = contract._require_descendant(output_root / run_id, contract.ALLOWED_ROOT, "run directory")
    run_dir.mkdir(parents=False, exist_ok=False)
    executor_stats_path = run_dir / "executor_stats.txt" if getattr(args, "executor_stats", False) else None

    started_at = contract._utc_now()
    inventory_started = time.perf_counter()
    inputs, representatives = contract.inventory_sources(source_records)
    timings: dict[str, float] = {
        "inventory_seconds": time.perf_counter() - inventory_started,
        "graph_seconds": 0.0,
        "finalize_and_write_seconds": 0.0,
        "adaptation_seconds": 0.0,
        "lance_write_seconds": 0.0,
        "evidence_finalize_seconds": 0.0,
        "validation_seconds": 0.0,
    }
    documents: list[dict[str, Any]] = []
    nrl_runtime_source: str | None = None
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "unpublished",
        "publication_policy": contract.PUBLICATION_POLICY,
        "run_id": run_id,
        "started_at": started_at,
        "input": input_spec,
        "configuration": {
            "run_mode": args.run_mode,
            "projection_workers": args.projection_workers,
            "projection_block_rows": projection_block_rows,
            "parse_batch_size": parse_batch_size,
            "parse_cpus": parse_cpus,
            "executor_stats_path": str(executor_stats_path) if executor_stats_path else None,
            "render": {"dpi": 200, "image_format": "png", "render_mode": "full_dpi"},
            "parse_model": contract.PARSE_MODEL,
            "task_prompt": contract.PARSE_TASK_PROMPT,
        },
        "inputs": inputs,
        "documents": documents,
        "timings": timings,
        "benchmark_evidence": {"root": evidence_root, "manifests": []} if evidence_root else None,
    }
    state_path = run_dir / contract.RUN_STATE_FILE
    contract._write_json_atomic(state_path, state)

    try:
        if evidence_root is not None:
            Path(evidence_root).mkdir(parents=True, exist_ok=False)
        pending: list[dict[str, Any]] = []
        for document in representatives:
            if document.get("preflight_error") is None:
                pending.append(document)
                continue
            build = contract.DocumentBuild(
                status="failed",
                issues=[{"kind": "source_pdf_preflight_failure", "error": document["preflight_error"]}],
            )
            documents.append(_document_result(document, build, publication_status="not_applicable"))

        envelope_records: list[dict[str, Any]] = []
        if pending:
            if graph_runner is None:
                graph_module = _load_graph_module()
                nrl_runtime_source = _validate_nrl_runtime_source(nrl_repo, graph_module)
                runner = graph_module.run_nrl_graph
                validator = graph_module.validate_projection_envelope
            else:
                if envelope_validator is None:
                    raise ValueError("an injected graph_runner requires an envelope_validator")
                runner = graph_runner
                validator = envelope_validator
            graph_started = time.perf_counter()
            envelope = runner(
                [str(document["path"]) for document in pending],
                run_mode=args.run_mode,
                projection_workers=args.projection_workers,
                projection_block_rows=projection_block_rows,
                parse_batch_size=parse_batch_size,
                parse_cpus=parse_cpus,
                **({"evidence_root": evidence_root} if evidence_root else {}),
                **({"executor_stats_path": executor_stats_path} if executor_stats_path else {}),
            )
            timings["graph_seconds"] = time.perf_counter() - graph_started
            if not hasattr(envelope, "to_dict"):
                raise TypeError("custom graph must return a pandas-compatible envelope")
            envelope = validator(envelope)
            envelope_records = list(envelope.to_dict("records"))

        expected_paths = {str(Path(str(document["path"])).resolve()) for document in pending}
        records_by_path: dict[str, list[dict[str, Any]]] = {path: [] for path in expected_paths}
        for record in envelope_records:
            source_path = record.get("source_path")
            if not isinstance(source_path, str):
                raise ValueError("projection envelope row is missing source_path")
            canonical_path = str(Path(source_path).resolve())
            if canonical_path not in records_by_path:
                raise ValueError(f"projection envelope contains unexpected source path {canonical_path}")
            record["source_path"] = canonical_path
            records_by_path[canonical_path].append(record)

        writer: contract.ElementTableWriter | None = None
        expected_counts: dict[str, int] = {}
        finalization_started = time.perf_counter()
        for document in pending:
            path = str(Path(str(document["path"])).resolve())
            adaptation_started = time.perf_counter()
            build = contract.build_document_rows(document, records_by_path[path], run_id=run_id)
            timings["adaptation_seconds"] += time.perf_counter() - adaptation_started
            result = _document_result(document, build, publication_status="pending")
            documents.append(result)
            if build.status not in contract._PUBLISHABLE_STATUSES:
                continue
            write_started = time.perf_counter()
            if writer is None:
                writer = contract.ElementTableWriter(run_dir)
            writer.add_document(build.rows)
            timings["lance_write_seconds"] += time.perf_counter() - write_started
            expected_counts[str(document["content_sha256"])] = len(build.rows)
            if evidence_root is not None and build.status in {"success", "valid_blank"}:
                from nrl_compare import finalize_capture

                evidence_started = time.perf_counter()
                evidence_manifest = finalize_capture(
                    evidence_root,
                    source_path=path,
                    document_sha256=str(document["content_sha256"]),
                    expected_page_count=int(document["expected_page_count"]),
                    valid_blank_pages=document.get("document_valid_blank_pages", []),
                    model_revision=_model_revision(contract.PARSE_MODEL),
                    provenance={
                        "run_id": run_id,
                        "source_path": path,
                        "finish_reason_source": "local_actor_error_contract",
                    },
                )
                state["benchmark_evidence"]["manifests"].append(str(evidence_manifest))
                timings["evidence_finalize_seconds"] += time.perf_counter() - evidence_started
        timings["finalize_and_write_seconds"] = time.perf_counter() - finalization_started

        if writer is None or not expected_counts:
            raise RuntimeError("No validated pages were available for handoff")

        _update_input_outcomes(inputs, documents, publication_status="pending")
        expected_provenance = _expected_document_provenance(
            inputs, run_id=run_id, sample_ids=set(expected_counts), documents=documents
        )
        validation_started = time.perf_counter()
        table_info = validate_element_table(
            contract._table_path(run_dir),
            expected_counts,
            expected_provenance=expected_provenance,
        )
        rehashed_input_count = contract._rehash_source_inventory(inputs)
        timings["validation_seconds"] = time.perf_counter() - validation_started

        for document in documents:
            if document["status"] in contract._PUBLISHABLE_STATUSES:
                document["publication_status"] = "handed_off"
        _update_input_outcomes(inputs, documents, publication_status="handed_off")
        nrl_revision = _git_revision(nrl_repo)
        curator_repo = Path(__file__).resolve().parents[3]
        curator_revision = _git_revision(curator_repo)
        if nrl_revision is None or curator_revision is None:
            raise RuntimeError("Could not resolve NRL or Curator Git revision")
        parse_revision = _model_revision(contract.PARSE_MODEL)

        handoff_core = {
            "schema_version": 1,
            "status": "tables_validated",
            "publication_policy": contract.PUBLICATION_POLICY,
            "run_id": run_id,
            "started_at": started_at,
            "handed_off_at": contract._utc_now(),
            "input": input_spec,
            "configuration": state["configuration"],
            "repositories": {
                "nemo_retriever": {
                    "path": str(nrl_repo),
                    "revision": nrl_revision,
                    "runtime_source": nrl_runtime_source,
                    "working_tree_dirty": _git_dirty(nrl_repo),
                },
                "nemo_curator": {
                    "path": str(curator_repo),
                    "revision": curator_revision,
                    "working_tree_dirty": _git_dirty(curator_repo),
                },
            },
            "models": {
                "nemotron_parse": {
                    "model_id": contract.PARSE_MODEL,
                    "revision": parse_revision,
                    "task_prompt": contract.PARSE_TASK_PROMPT,
                    "max_tokens": 9000,
                    "decoding": {"temperature": 0, "top_k": 1, "repetition_penalty": 1.1},
                }
            },
            "tables": {"database_uri": str(run_dir), contract.ELEMENT_TABLE: table_info},
            "counts": _counts(inputs, documents),
            "inputs": inputs,
            "documents": documents,
            "timings": timings,
            "benchmark_evidence": state["benchmark_evidence"],
            "source_inventory": {"rehash_status": "validated", "rehashed_input_count": rehashed_input_count},
            "publication": {
                "completion_required": True,
                "restart_policy": "restart into a fresh run directory",
            },
        }
        handoff = contract._seal_payload(handoff_core, contract._HANDOFF_HASH_FIELD)
        handoff_path = run_dir / contract.HANDOFF_MANIFEST_FILE
        state.update(
            {
                "status": "tables_validated",
                "handed_off_at": handoff["handed_off_at"],
                "inputs": inputs,
                "documents": documents,
                "counts": handoff["counts"],
                "handoff_manifest": str(handoff_path),
                "handoff_sha256": handoff[contract._HANDOFF_HASH_FIELD],
            }
        )
        contract._write_json_atomic(state_path, state)
        contract._write_json_exclusive_atomic(handoff_path, handoff)
        return handoff_path
    except contract.MarkerDurabilityUnconfirmedError as exc:
        # Visibility is the publication point; a failed directory sync cannot
        # turn a valid handoff back into an unpublished extraction.
        state["marker_durability"] = {"path": str(exc.path), "status": "unconfirmed"}
        try:
            contract._write_json_atomic(state_path, state)
        except Exception:
            pass
        raise
    except Exception as exc:
        state["status"] = "unpublished"
        state["failed_at"] = contract._utc_now()
        state["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        recorded_hashes = {str(document["content_sha256"]) for document in documents}
        for representative in representatives:
            digest = representative.get("content_sha256")
            if not isinstance(digest, str) or digest in recorded_hashes:
                continue
            documents.append(
                _document_result(
                    representative,
                    contract.DocumentBuild(
                        status="failed",
                        issues=[
                            {
                                "kind": "graph_or_publication_failure",
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        ],
                    ),
                    publication_status="failed",
                )
            )
        _update_input_outcomes(inputs, documents, publication_status="failed")
        for document in documents:
            if document["status"] in contract._PUBLISHABLE_STATUSES:
                document["publication_status"] = "failed"
        state["inputs"] = inputs
        state["documents"] = documents
        state["counts"] = _counts(inputs, documents)
        try:
            contract._write_json_atomic(state_path, state)
        except Exception:
            pass
        raise


def _load_handoff_manifest(path: Path) -> dict[str, Any]:
    if path.name != contract.HANDOFF_MANIFEST_FILE:
        raise ValueError(f"handoff manifest must be named {contract.HANDOFF_MANIFEST_FILE}")
    handoff = contract._load_sealed_json(
        path,
        contract._HANDOFF_HASH_FIELD,
        label="handoff manifest",
    )
    if handoff.get("status") != "tables_validated":
        raise RuntimeError("Refusing a handoff whose status is not tables_validated")
    policy = handoff.get("publication_policy", "complete_documents_v1")
    if policy not in {"complete_documents_v1", contract.PUBLICATION_POLICY}:
        raise ValueError(f"Unknown publication policy {policy!r}")
    if policy == "complete_documents_v1" and any(
        "extraction_status" in document or "page_outcomes" in document for document in handoff.get("documents", [])
    ):
        raise ValueError("Explicit page outcomes require the validated-pages publication policy")
    if policy == "complete_documents_v1" and any(
        document.get("publication_status") == "handed_off" and document.get("status") not in {"success", "valid_blank"}
        for document in handoff.get("documents", [])
    ):
        raise ValueError("Legacy handoffs cannot publish incomplete documents")
    tables = handoff.get("tables")
    if not isinstance(tables, dict) or tables.get("database_uri") != str(path.parent):
        raise ValueError("handoff database URI differs from its run directory")
    table = tables.get(contract.ELEMENT_TABLE)
    if not isinstance(table, dict):
        raise ValueError(f"handoff is missing {contract.ELEMENT_TABLE}")
    expected_path = path.parent / f"{contract.ELEMENT_TABLE}.lance"
    table_path = contract._require_under(Path(str(table.get("path"))), contract.ALLOWED_ROOT, "handoff table")
    if table_path != expected_path:
        raise ValueError(f"handoff table must be {expected_path}; got {table_path}")
    version = table.get("version")
    row_count = table.get("row_count")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("handoff table version is invalid")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("handoff table row count is invalid")
    return handoff


def _binary_sha256(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} binary_content is not bytes")
    return contract._sha256_bytes(bytes(value))


def _collect_reconciliation_rows(
    batches: Iterable[Any],
    field_names: Sequence[str],
    *,
    label: str,
) -> tuple[dict[tuple[str, int], tuple[dict[str, Any], str | None]], dict[str, list[int]], int]:
    comparable_fields = [name for name in field_names if name != "binary_content"]
    records: dict[tuple[str, int], tuple[dict[str, Any], str | None]] = {}
    positions: dict[str, list[int]] = {}
    row_count = 0
    for batch in batches:
        for row in batch.to_pylist():
            row_count += 1
            sample_id = row.get("sample_id")
            position = row.get("position")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{label} row has an invalid sample_id")
            if isinstance(position, bool) or not isinstance(position, int):
                raise ValueError(f"{label} row for {sample_id} has an invalid position")
            key = (sample_id, position)
            if key in records:
                raise ValueError(f"{label} contains duplicate row key {key}")
            comparable = {name: row.get(name) for name in comparable_fields}
            records[key] = (comparable, _binary_sha256(row.get("binary_content"), label=f"{label} {key}"))
            positions.setdefault(sample_id, []).append(position)
    return records, positions, row_count


def _validate_consumed_output(table_path: Path, version: int, output_dir: Path) -> dict[str, Any]:
    import lance
    import pyarrow.parquet as pq

    field_names = list(contract.element_schema().names)
    source_dataset = lance.dataset(str(table_path), version=version)
    _validate_element_schema(source_dataset.schema, label="Pinned Lance source")
    expected_records, expected_positions, source_row_count = _collect_reconciliation_rows(
        source_dataset.scanner(columns=field_names).to_batches(),
        field_names,
        label="pinned Lance source",
    )

    parquet_files = sorted(output_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Curator wrote no Parquet files under {output_dir}")

    def parquet_batches() -> Iterable[Any]:
        for parquet_path in parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            _validate_element_schema(parquet_file.schema_arrow, label=f"Parquet {parquet_path}")
            yield from parquet_file.iter_batches()

    native_reader_seconds = 0.0

    def native_reader_batches() -> Iterable[Any]:
        from nemo_curator.backends.ray_data import RayDataExecutor
        from nemo_curator.pipeline import Pipeline
        from nemo_curator.stages.interleaved.io import InterleavedParquetReader

        nonlocal native_reader_seconds
        pipeline = Pipeline(name="nrl_curator_parquet_reopen")
        pipeline.add_stage(
            InterleavedParquetReader(file_paths=[str(path) for path in parquet_files], files_per_partition=1)
        )
        started = time.perf_counter()
        tasks = pipeline.run(RayDataExecutor())
        native_reader_seconds = time.perf_counter() - started
        for task in tasks:
            table = task.to_pyarrow()
            _validate_element_schema(table.schema, label="Curator native Parquet reader")
            yield from table.to_batches()

    # Check stored values first so reader-side normalization cannot conceal corruption.
    for label, read_batches in (
        ("Curator Parquet output", parquet_batches),
        ("Curator native Parquet reader", native_reader_batches),
    ):
        actual_records, actual_positions, actual_row_count = _collect_reconciliation_rows(
            read_batches(), field_names, label=label
        )
        if actual_row_count != source_row_count or set(actual_records) != set(expected_records):
            missing = sorted(set(expected_records) - set(actual_records))[:10]
            unexpected = sorted(set(actual_records) - set(expected_records))[:10]
            raise ValueError(
                f"{label} row keys differ: source={source_row_count}, output={actual_row_count}, "
                f"missing={missing}, unexpected={unexpected}"
            )
        if actual_positions != expected_positions:
            raise ValueError(f"{label} changed per-document positions or order")
        for key, expected in expected_records.items():
            if actual_records[key] != expected:
                raise ValueError(f"{label} row {key} differs from the pinned Lance source")
        del actual_records

    return {
        "schema_version": 1,
        "status": "validated",
        "validated_at": contract._utc_now(),
        "source": {
            "table_path": str(table_path),
            "version": version,
            "row_count": source_row_count,
            "document_count": len(expected_positions),
        },
        "output": {
            "directory": str(output_dir),
            "parquet_files": [str(path) for path in parquet_files],
            "row_count": actual_row_count,
            "document_count": len(actual_positions),
        },
        "reconciliation": {
            "key_fields": ["sample_id", "position"],
            "fields_compared": [name for name in field_names if name != "binary_content"],
            "binary_comparison": "sha256",
            "positions_by_sample": actual_positions,
            "schema_and_nullability_validated": True,
            "native_parquet_reader_validated": True,
            "native_parquet_reader_seconds": native_reader_seconds,
        },
    }


def _expected_counts_from_handoff(handoff: Mapping[str, Any]) -> dict[str, int]:
    documents = handoff.get("documents")
    if not isinstance(documents, list):
        raise ValueError("handoff is missing document results")
    if any(
        isinstance(document, dict)
        and document.get("publication_status") == "handed_off"
        and document.get("status") not in contract._PUBLISHABLE_STATUSES
        for document in documents
    ):
        raise ValueError("handoff contains a published document without validated pages")
    counts = {
        str(document["content_sha256"]): int(document["element_count"])
        for document in documents
        if isinstance(document, dict) and document.get("publication_status") == "handed_off"
    }
    if not counts:
        raise ValueError("handoff contains no published documents")
    return counts


def _build_consume_pipeline(
    table_path: Path,
    *,
    version: int,
    output_dir: Path,
    mode: str,
) -> Any:
    """Build the exact native Curator compatibility pipeline."""

    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.interleaved import InterleavedAspectRatioFilterStage
    from nemo_curator.stages.interleaved.filter.blur_filter import InterleavedBlurFilterStage
    from nemo_curator.stages.interleaved.io import InterleavedLanceReader, InterleavedParquetWriterStage

    pipeline = Pipeline(
        name="nrl_lance_pdf_consume",
        description="Pinned NRL Lance elements -> image decode validation -> Parquet",
    )
    pipeline.add_stage(
        InterleavedLanceReader(
            path=str(table_path),
            fragments_per_partition=1,
            read_kwargs={"version": version},
            include_lance_metadata=False,
        )
    )
    pipeline.add_stage(
        InterleavedAspectRatioFilterStage(
            min_aspect_ratio=0.0,
            max_aspect_ratio=float("inf"),
            drop_invalid_rows=False,
            preserve_metadata_only_samples=True,
        )
    )
    pipeline.add_stage(
        InterleavedBlurFilterStage(
            score_threshold=0.0,
            drop_invalid_rows=False,
            preserve_metadata_only_samples=True,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=str(output_dir),
            materialize_on_write=False,
            mode=mode,
            schema=contract.element_schema(),
            write_kwargs={"schema": contract.element_schema()},
        )
    )
    return pipeline


def run_consume(args: argparse.Namespace) -> Path:
    """Run native Curator validation and atomically publish completion."""

    if args.mode != "error":
        raise ValueError("consume supports only mode='error'; output directory must be fresh")

    handoff_path = contract._require_under(
        Path(args.handoff_manifest),
        contract.ALLOWED_ROOT,
        "handoff manifest",
    )
    handoff = _load_handoff_manifest(handoff_path)
    completion_path = handoff_path.parent / contract.COMPLETION_MANIFEST_FILE
    if completion_path.exists():
        raise FileExistsError(f"completion manifest already exists: {completion_path}")

    table_info = handoff["tables"][contract.ELEMENT_TABLE]
    table_path = Path(str(table_info["path"]))
    version = int(table_info["version"])
    expected_row_count = int(table_info["row_count"])
    output_dir = contract._require_descendant(Path(args.output_dir), contract.ALLOWED_ROOT, "output directory")
    if contract._paths_overlap(output_dir, handoff_path.parent):
        raise ValueError("consumer output directory must not overlap the handoff run directory")
    if output_dir.exists():
        raise FileExistsError(f"consumer output directory must be fresh: {output_dir}")
    contract._confirm_marker_durability(
        handoff_path,
        contract._HANDOFF_HASH_FIELD,
        expected_sha256=handoff[contract._HANDOFF_HASH_FIELD],
    )
    source_inputs = handoff.get("inputs")
    if not isinstance(source_inputs, list):
        raise ValueError("handoff is missing source inventory")
    for source_input in source_inputs:
        if not isinstance(source_input, dict) or not isinstance(source_input.get("path"), str):
            raise ValueError("handoff contains an invalid source input")
        source_path = contract._require_under(
            Path(source_input["path"]),
            contract.ALLOWED_ROOT,
            "handoff source PDF",
        )
        if contract._paths_overlap(output_dir, source_path):
            raise ValueError(f"consumer output directory must not contain source PDF {source_path}")

    expected_counts = _expected_counts_from_handoff(handoff)
    expected_provenance = _expected_document_provenance(
        source_inputs,
        run_id=str(handoff["run_id"]),
        sample_ids=set(expected_counts),
        documents=handoff["documents"] if handoff.get("publication_policy") == contract.PUBLICATION_POLICY else None,
    )

    import lance

    current_dataset = lance.dataset(str(table_path))
    if int(current_dataset.version) != version:
        raise RuntimeError(f"Current Lance version is {current_dataset.version}; handoff pins version {version}")
    if int(current_dataset.count_rows()) != expected_row_count:
        raise RuntimeError("Current Lance row count differs from the handoff")

    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.tasks.utils import TaskPerfUtils

    pipeline = _build_consume_pipeline(
        table_path,
        version=version,
        output_dir=output_dir,
        mode=args.mode,
    )

    consume_started = time.perf_counter()
    pipeline_started = time.perf_counter()
    tasks = pipeline.run(RayDataExecutor())
    pipeline_seconds = time.perf_counter() - pipeline_started
    stage_metrics = {
        stage: {name: values.tolist() for name, values in metrics.items()}
        for stage, metrics in TaskPerfUtils.collect_stage_metrics(tasks).items()
    }
    reconciliation_started = time.perf_counter()
    reconciliation = _validate_consumed_output(table_path, version, output_dir)
    reconciliation_seconds = time.perf_counter() - reconciliation_started
    if reconciliation["source"]["row_count"] != expected_row_count:
        raise RuntimeError("Reconciled source count differs from the handoff")

    report_core = {
        **reconciliation,
        "kind": "curator_consume",
        "handoff": {
            "path": str(handoff_path),
            "sha256": handoff[contract._HANDOFF_HASH_FIELD],
        },
        "table": copy.deepcopy(table_info),
        "stage_metrics": stage_metrics,
        "timings": {
            "pipeline_seconds": pipeline_seconds,
            "reconciliation_seconds": reconciliation_seconds,
            "pipeline_and_reconciliation_seconds": time.perf_counter() - consume_started,
        },
    }
    report = contract._seal_payload(report_core, contract._REPORT_HASH_FIELD)
    report_path = output_dir / contract.CONSUME_REPORT_FILE
    contract._write_json_exclusive_atomic(report_path, report)

    current_table = contract._open_lancedb_table(table_path)
    if int(current_table.version) != version or int(current_table.count_rows()) != expected_row_count:
        raise RuntimeError("Lance table changed during Curator consumption")
    revalidated_table = validate_element_table(
        table_path,
        expected_counts,
        expected_provenance=expected_provenance,
        version=version,
    )
    for key in ("path", "version", "row_count", "document_count", "inline_image_sha256"):
        if revalidated_table.get(key) != table_info.get(key):
            raise RuntimeError(f"Lance table {key} differs from its handoff contract")
    rehashed_input_count = contract._rehash_source_inventory(source_inputs)

    published_inputs = copy.deepcopy(source_inputs)
    for entry in published_inputs:
        if entry.get("publication_status") == "handed_off":
            entry["publication_status"] = "published"
    published_documents = copy.deepcopy(handoff["documents"])
    for document in published_documents:
        if document.get("publication_status") == "handed_off":
            document["publication_status"] = "published"
    completion_core = {
        "schema_version": 1,
        "status": "published",
        "publication_policy": handoff.get("publication_policy", "complete_documents_v1"),
        "run_id": handoff["run_id"],
        "published_at": contract._utc_now(),
        "handoff": {
            "path": str(handoff_path),
            "sha256": handoff[contract._HANDOFF_HASH_FIELD],
        },
        "curator_consume": {
            "path": str(report_path),
            "sha256": report[contract._REPORT_HASH_FIELD],
        },
        "tables": copy.deepcopy(handoff["tables"]),
        "models": copy.deepcopy(handoff["models"]),
        "repositories": copy.deepcopy(handoff["repositories"]),
        "counts": _counts(published_inputs, published_documents),
        "inputs": published_inputs,
        "documents": published_documents,
        "source_inventory": {
            "rehash_status": "validated",
            "rehashed_input_count": rehashed_input_count,
        },
        "publication": {
            "handoff_verified": True,
            "curator_output_reconciled": True,
            "table_revalidated_without_version_drift": True,
            "source_inventory_rehashed": True,
        },
    }
    completion = contract._seal_payload(completion_core, contract._COMPLETION_HASH_FIELD)
    _publish_completion_and_update_state(
        completion_path,
        completion,
        state_path=handoff_path.parent / contract.RUN_STATE_FILE,
        state_updates={
            "status": "published",
            "published_at": completion["published_at"],
            "consume_validated_at": report["validated_at"],
            "consume_report": str(report_path),
            "consume_report_sha256": report[contract._REPORT_HASH_FIELD],
            "completion_manifest": str(completion_path),
            "completion_sha256": completion[contract._COMPLETION_HASH_FIELD],
        },
    )
    return completion_path
