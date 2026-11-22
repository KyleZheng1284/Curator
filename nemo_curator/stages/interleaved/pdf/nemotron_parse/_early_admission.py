# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlap a page-budgeted PDF parent's rendering and served parsing, and render the next parent ahead."""

from __future__ import annotations

import asyncio
import contextlib
import pickle
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse._image_transport import PNG_METADATA_KEY, png_batch_provenance
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

if TYPE_CHECKING:
    from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseHTTPClientStage
    from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage

_INPUT_BYTES = 32 * 1024**2
_HELD_BYTES = 256 * 1024**2
_RENDER_JOIN_SECONDS = 5
_Row = dict[str, Any]


def _page_key(row: _Row) -> tuple[str, str, int]:
    return row["document_id"], row["pdf_name"], row["page_number"]


class _InputCredit:
    """Rendered bytes one worker holds across its parents, from handoff through the parser call."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.input_bytes = 0


class _RangeBuffer:
    """One parent's render: a thread-to-event-loop handoff with credits held through the parser call."""

    def __init__(self, credit: _InputCredit, task: FileGroupTask) -> None:
        self.credit = credit
        self.condition = credit.condition
        self.task = task
        # A parent rendered ahead has no event loop until its requests may start.
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ready = asyncio.Event()
        self.stopped = asyncio.Event()
        self.cancelled = threading.Event()
        self.pending: deque[tuple[_Row, int]] = deque()
        self.finished = False
        self.error: BaseException | None = None
        self.producer: threading.Thread | None = None
        self.rendered: InterleavedBatch | None = None
        self.render_metrics: dict[str, float] = {}

    def submit(self, rows: list[_Row]) -> None:
        for row in rows:
            payload = row.get("binary_content")
            if payload is not None and not isinstance(payload, bytes):
                msg = "Rendered page data must be immutable bytes"
                raise TypeError(msg)
            size = len(payload or b"")
            with self.condition:
                # One page larger than the credit may hold it alone.
                self.condition.wait_for(
                    lambda size=size: (
                        self.cancelled.is_set()
                        or not self.credit.input_bytes
                        or self.credit.input_bytes + size <= _INPUT_BYTES
                    )
                )
                if self.cancelled.is_set():
                    msg = "Render handoff cancelled"
                    raise RuntimeError(msg)
                self.pending.append((row, size))
                self.credit.input_bytes += size
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.ready.set)

    def finish(self, error: BaseException | None = None) -> None:
        with self.condition:
            self.error = error
            self.finished = True
            self.condition.notify_all()
        # A blocked source read may finish after the cancelled parent exits.
        if self.loop is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.ready.set)

    async def receive(self) -> tuple[_Row, int] | None:
        while True:
            with self.condition:
                if self.error is not None:
                    raise self.error
                if self.pending:
                    return self.pending.popleft()
                if self.finished or self.cancelled.is_set():
                    return None
                self.ready.clear()
            await self.ready.wait()

    def release(self, size: int) -> None:
        with self.condition:
            self.credit.input_bytes -= size
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            self.cancelled.set()
            self.condition.notify_all()
        if self.loop is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.stopped.set)
                self.loop.call_soon_threadsafe(self.ready.set)

    def discard(self) -> None:
        """Return the credit of pages that were handed off and will never be parsed."""
        with self.condition:
            while self.pending:
                self.credit.input_bytes -= self.pending.popleft()[1]
            self.condition.notify_all()


class EarlyAdmissionStage(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Start serving complete render ranges while their siblings render.

    With ``stream_page_tasks`` on Ray Data, the worker receives its next task early
    and renders it while the current parent's last requests finish. Requests still
    belong to one parent at a time, and every held page shares one input credit.
    """

    name = "pdf_render_parse_overlap"

    def __init__(
        self, renderer: PDFPreprocessStage, inference: NemotronParseHTTPClientStage, *, stream_page_tasks: bool = False
    ) -> None:
        self.renderer = renderer
        self.inference = inference
        self.stream_page_tasks = stream_page_tasks

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.renderer.inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.inference.outputs()

    def ray_stage_spec(self) -> dict[str, Any]:
        spec = {"is_actor_stage": True}
        if self.stream_page_tasks:
            spec["task_lookahead"] = True
        return spec

    def setup(self, worker_metadata: dict | None = None) -> None:
        self._parent_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._credit = _InputCredit()
        self._ahead: deque[_RangeBuffer] = deque()
        self._last: _RangeBuffer | None = None
        self._buffer: _RangeBuffer | None = None
        self._closed = False
        self.renderer.setup(worker_metadata)
        try:
            self.inference.setup(worker_metadata)
        except BaseException:
            self.renderer.teardown()
            raise

    def teardown(self) -> None:
        with self._state_lock:
            self._closed = True
            if self._buffer is not None:
                self._buffer.cancel()
            ahead = list(self._ahead)
            self._ahead.clear()
            for buffer in ahead:
                buffer.cancel()
        with self._parent_lock:
            for buffer in ahead:
                self._stop(buffer)
            try:
                self.inference.teardown()
            finally:
                self.renderer.teardown()

    def lookahead(self, tasks: list[FileGroupTask]) -> None:
        """Queue the renders of a call that has reached the worker, in the order ``process`` will see them."""
        with self._state_lock:
            if self._closed:
                return
            previous = self._last
            started = []
            try:
                for task in tasks:
                    started.append(self._start(task))
            except BaseException:
                # This call will never reach process(). Retire its renders before unblocking the next call.
                for buffer in started:
                    self._closed |= self._stop_producer(buffer)
                self._last = previous
                raise
            self._ahead.extend(started)

    def process(self, task: FileGroupTask) -> InterleavedBatch:
        with self._parent_lock:
            return asyncio.run(self._process_parent(task))

    def _start(self, task: FileGroupTask) -> _RangeBuffer:
        """Render a parent once the render queued before it has finished. Call with the state lock held."""
        buffer = _RangeBuffer(self._credit, task)
        previous = self._last

        def produce() -> None:
            try:
                if previous is not None:
                    # One renderer serves the worker, and only the newest render may wait for credit.
                    with buffer.condition:
                        buffer.condition.wait_for(lambda: previous.finished or buffer.cancelled.is_set())
                buffer.rendered = self.renderer._process(task, range_sink=buffer.submit, cancel_event=buffer.cancelled)
                # The next render overwrites the renderer's metrics as soon as this one returns.
                buffer.render_metrics = self.renderer._consume_custom_metrics()
            except BaseException as error:  # noqa: BLE001 - Preserve the producer's error on the owning thread.
                buffer.finish(error)
            else:
                buffer.finish()

        buffer.producer = threading.Thread(target=produce, name="curator-pdf-render", daemon=True)
        buffer.producer.start()
        self._last = buffer
        return buffer

    @staticmethod
    def _stop_producer(buffer: _RangeBuffer) -> bool:
        buffer.cancel()
        buffer.producer.join(_RENDER_JOIN_SECONDS)
        buffer.discard()
        # A producer stuck reading a source still owns the renderer.
        return buffer.producer.is_alive()

    def _stop(self, buffer: _RangeBuffer) -> None:
        stuck = self._stop_producer(buffer)
        with self._state_lock:
            self._closed |= stuck
            if self._buffer is buffer:
                self._buffer = None
            if self._last is buffer:
                self._last = None

    async def _process_parent(self, task: FileGroupTask) -> InterleavedBatch:
        chunks = task._metadata.get("pdf_page_chunks", {})
        expected = {
            (chunk["document_id"], name, page) for name, chunk in chunks.items() for page in chunk["page_indices"]
        }
        with self._state_lock:
            # A failed call can leave queued renders before this task. Keep the task and its successors.
            abandoned = []
            while self._ahead and self._ahead[0].task is not task:
                abandoned.append(self._ahead.popleft())
            buffer = self._ahead.popleft() if self._ahead else None
        for ahead in abandoned:
            self._stop(ahead)
        try:
            if not expected:
                msg = "Overlapped rendering requires a page-budgeted PDF parent"
                raise ValueError(msg)
            with self._state_lock:
                if self._closed:
                    msg = "The render and parse worker is closed"
                    raise RuntimeError(msg)
                self.inference._check_parse_task(task._metadata)
                ahead_pages = 0 if buffer is None else len(buffer.pending)
                buffer = buffer or self._start(task)
                buffer.loop = asyncio.get_running_loop()
                self._buffer = buffer
            started = time.perf_counter()
            parsed = await self._parse_available(buffer, expected)
        finally:
            if buffer is not None:
                self._stop(buffer)
        rendered = buffer.rendered
        if rendered is None or parsed.keys() != expected:
            msg = "Render handoff did not produce all physical pages in the parent"
            raise RuntimeError(msg)
        identities = rendered.to_pyarrow().select(["document_id", "pdf_name", "page_number"])
        renderer_order = [_page_key(row) for row in identities.to_pylist()]
        if len(renderer_order) != len(expected) or set(renderer_order) != expected:
            msg = "Rendered parent metadata disagrees with its admitted physical pages"
            raise RuntimeError(msg)
        rows = [parsed[key] for key in renderer_order]
        if self.renderer.reuse_encoded_images:
            rendered._metadata[PNG_METADATA_KEY] = png_batch_provenance(row.get("binary_content") for row in rows)
        result = self.inference._finish_parse_task(rendered, rows, started, includes_rendering=True)
        self._log_metrics(self.inference._consume_custom_metrics())
        self._log_metrics({f"render.{name}": value for name, value in buffer.render_metrics.items()})
        if self.stream_page_tasks:
            self._log_metrics({"render_ahead_pages": float(ahead_pages)})
        return result

    async def _parse_available(
        self, buffer: _RangeBuffer, expected: set[tuple[str, str, int]]
    ) -> dict[tuple[str, str, int], _Row]:
        seen: set[tuple[str, str, int]] = set()
        parsed: dict[tuple[str, str, int], _Row] = {}
        held_bytes = 0
        reuse = self.renderer.reuse_encoded_images and self.inference.reuse_encoded_images

        async with self.inference._page_parser(settle_on_cancel=True) as parse_page:

            async def consume() -> None:
                nonlocal held_bytes
                while (entry := await buffer.receive()) is not None:
                    row, size = entry
                    try:
                        key = _page_key(row)
                        if key not in expected or key in seen:
                            msg = f"Unexpected or duplicate physical page in render handoff: {key}"
                            raise ValueError(msg)
                        seen.add(key)
                        result = await parse_page(row, reuse)
                        held_bytes += len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
                        if held_bytes > _HELD_BYTES:
                            msg = "Completed PDF parent exceeds the held-result byte allowance"
                            raise RuntimeError(msg)
                        parsed[key] = result
                    finally:
                        buffer.release(size)

            tasks = [
                asyncio.create_task(consume()) for _ in range(min(len(expected), self.inference.inference_batch_size))
            ]
            workers = asyncio.gather(*tasks)
            stopped = asyncio.create_task(buffer.stopped.wait())
            try:
                completed, _ = await asyncio.wait((workers, stopped), return_when=asyncio.FIRST_COMPLETED)
                if stopped in completed:
                    msg = "Render and parse worker cancelled"
                    raise RuntimeError(msg)
                await workers
            finally:
                for task in (*tasks, stopped):
                    task.cancel()
                await asyncio.gather(workers, *tasks, stopped, return_exceptions=True)
        return parsed
