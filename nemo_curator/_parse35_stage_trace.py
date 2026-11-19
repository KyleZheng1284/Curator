# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in timing of Curator's Page64 handoffs in the isolated Parse study."""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_DIRECTORY = os.environ.get("PARSE35_TRACE_DIR")
_BACKEND = os.environ.get("PARSE35_TRACE_BACKEND")
_DOMAIN = "parse35.pipeline.v1"
_PAGE: ContextVar[dict[str, Any] | None] = ContextVar("parse35_trace_page", default=None)
_REQUEST: ContextVar[str | None] = ContextVar("parse35_trace_request", default=None)


def enabled() -> bool:
    return bool(_DIRECTORY)


@functools.lru_cache(maxsize=2)
def _writer(pid: int) -> tuple[int, int]:
    path = Path(_DIRECTORY) / f"curator-{pid}.jsonl"
    with open(f"/proc/{pid}/stat", encoding="utf-8") as stream:
        birth = int(stream.read().rsplit(") ", 1)[1].split()[19])
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    return descriptor, birth


@functools.cache
def _nvtx() -> Any:
    try:
        import nvtx
    except ModuleNotFoundError:
        # The pinned Curator CPU environment has no NVTX wheel. Resolve only
        # the copied, ABI-matched package in this isolated source tree.
        bundled = str(Path(__file__).resolve().parents[1] / "experiments/local_parse/_nvtx_runtime")
        sys.path.insert(0, bundled)
        try:
            import nvtx
        except ImportError:
            return None
        finally:
            sys.path.remove(bundled)
    return nvtx


def page_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {"document_id": row.get("document_id"), "physical_page": row.get("page_number"),
            "pdf_name": row.get("pdf_name")}


@contextlib.contextmanager
def page(row: dict[str, Any]):
    if not enabled():
        yield
        return
    token = _PAGE.set(page_fields(row))
    try:
        yield
    finally:
        _PAGE.reset(token)


@contextlib.contextmanager
def request():
    if _PAGE.get() is None:
        yield
        return
    token = _REQUEST.set(uuid.uuid4().hex)
    try:
        yield
    finally:
        _REQUEST.reset(token)


def request_id() -> str | None:
    return _REQUEST.get()


def emit(event: str, **fields: Any) -> None:
    if not enabled():
        return
    timestamp = time.monotonic_ns()
    wall = time.time_ns()
    pid = os.getpid()
    descriptor, birth = _writer(pid)
    identity = _PAGE.get() or {}
    if _REQUEST.get() is not None:
        identity = {**identity, "request_id": _REQUEST.get()}
    record = {"event": event, "monotonic_ns": timestamp, "wall_ns": wall, "pid": pid,
              "process_birth_ticks": birth, "tid": threading.get_native_id(), "backend": _BACKEND,
              **identity, **fields}
    os.write(descriptor, (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode())
    if (nvtx := _nvtx()) is not None:
        nvtx.mark(event, domain=_DOMAIN)
