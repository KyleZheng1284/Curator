"""Study-only bounded admission of completed ranges inside one Page64 parent.

No child batch reaches downstream postprocessing. Input credits cover pending,
queued and active encoded rows; held results have a distinct parent reservation.
The original renderer, request builder, parser and complete-parent finish remain
authoritative. This module is activated only by PARSE35_EARLY_ADMISSION=1.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import subprocess
import threading
import time

from nemo_curator import _parse35_stage_trace as trace
from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import inference as http_impl
from nemo_curator.stages.interleaved.pdf.nemotron_parse._image_transport import (
    PNG_METADATA_KEY, has_png_batch_provenance, png_batch_provenance,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import output_version
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, InterleavedBatch


MIB = 1024**2


class EarlyAdmissionError(RuntimeError):
    pass


class EarlyAdmissionCancelled(EarlyAdmissionError):
    pass


class EarlyAdmissionCapacity(EarlyAdmissionError):
    pass


@dataclass(frozen=True)
class Limits:
    block_pages: int = 8
    block_bytes: int = 8 * MIB
    input_bytes: int = 32 * MIB
    parent_input_bytes: int = 128 * MIB
    held_bytes: int = 256 * MIB
    result_extra_bytes_per_page: int = 2 * MIB

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'Positive integer limit required: {name}')
        if self.block_pages > 64 or self.block_bytes > self.input_bytes or self.input_bytes > self.parent_input_bytes:
            raise ValueError('Invalid Page64 block/input limits')

    @classmethod
    def environment(cls):
        names = {'block_pages': 'BLOCK_PAGES', 'block_bytes': 'BLOCK_BYTES',
                 'input_bytes': 'INPUT_BYTES', 'parent_input_bytes': 'PARENT_INPUT_BYTES',
                 'held_bytes': 'HELD_BYTES'}
        return cls(**{name: int(os.environ['PARSE35_EARLY_' + suffix])
                      for name, suffix in names.items() if 'PARSE35_EARLY_' + suffix in os.environ})


def cancellable_helper(command, payload, environment, timeout, cancel):
    """Run the same one-range helper with bounded cancellation and exact reap."""
    if cancel.is_set():
        raise EarlyAdmissionCancelled('Parent cancelled before range rendering')
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=environment)
    settled = threading.Event()

    def watch_cancel():
        while not settled.wait(.05):
            if cancel.is_set():
                if child.poll() is None:
                    child.terminate()
                until = time.monotonic() + 2
                while child.poll() is None and time.monotonic() < until:
                    settled.wait(.02)
                if child.poll() is None:
                    child.kill()
                return

    watcher = threading.Thread(target=watch_cancel, name='page64-render-cancel', daemon=True)
    watcher.start()
    try:
        # Exactly one full communicate preserves the stock pipe transfer.
        # Retrying communicate(None) after a short timeout can strand partially
        # written stdin on the pinned CPython when a cold helper reads late.
        out, err = child.communicate(input=payload, timeout=timeout)
        if cancel.is_set():
            raise EarlyAdmissionCancelled('Parent cancelled during range rendering')
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command, out, err)
        return subprocess.CompletedProcess(command, child.returncode, out, err)
    except subprocess.TimeoutExpired:
        # Match subprocess.run(timeout=...) control behavior: kill on expiry,
        # drain only after death, then retain the original timeout outcome.
        child.kill()
        child.communicate(timeout=2)
        raise
    finally:
        settled.set()
        if child.poll() is None:
            child.terminate()
            try:
                child.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=2)
        watcher.join(timeout=2.5)


def parent_keys(task):
    keys = []
    for text in task.data:
        entry = json.loads(text)
        chunk = entry.get('_page_chunk')
        if not chunk or not chunk.get('document_id'):
            raise EarlyAdmissionError('Early admission requires original Page64 physical-range provenance')
        keys.extend((chunk['document_id'], page) for page in chunk['page_indices'])
    if not 1 <= len(keys) <= 64 or len(set(keys)) != len(keys):
        raise EarlyAdmissionError('Expected one unique, nonempty parent of at most64 physical pages')
    return keys


class _Credits:
    def __init__(self, limits, cancel, emit):
        self.limits, self.cancel, self.emit = limits, cancel, emit
        self.condition = threading.Condition()
        self.leases = {}
        self.used = self.peak = self.peak_pages = self.total = self.released = self.oversize_peak = 0

    def blocked(self, size):
        return self.used != 0 if size > self.limits.input_bytes else self.used + size > self.limits.input_bytes

    def would_block(self, size):
        with self.condition:
            return self.blocked(size)

    def acquire(self, ordinal, size):
        with self.condition:
            if ordinal in self.leases:
                raise EarlyAdmissionError('Input ordinal reserved twice')
            if self.total + size > self.limits.parent_input_bytes:
                raise EarlyAdmissionCapacity('Parent encoded input exceeds its admitted byte allowance')
            waiting = False
            while self.blocked(size):
                if self.cancel.is_set():
                    raise EarlyAdmissionCancelled('Parent cancelled waiting for input bytes')
                if not waiting:
                    self.emit('e03_input_wait_begin', ordinal=ordinal, requested_bytes=size)
                    waiting = True
                self.condition.wait(.05)
            if waiting:
                self.emit('e03_input_wait_end', ordinal=ordinal, requested_bytes=size)
            if self.cancel.is_set():
                raise EarlyAdmissionCancelled('Parent cancelled before input reservation')
            self.leases[ordinal] = size
            self.used += size
            self.total += size
            self.peak = max(self.peak, self.used)
            self.peak_pages = max(self.peak_pages, len(self.leases))
            if size > self.limits.input_bytes:
                self.oversize_peak = max(self.oversize_peak, size)
                self.emit('e03_oversize_serial', ordinal=ordinal, encoded_bytes=size)
            self.emit('e03_credits', input_bytes=self.used, input_pages=len(self.leases))

    def release(self, ordinal):
        with self.condition:
            if ordinal not in self.leases:
                raise EarlyAdmissionError('Input credit returned without ownership or returned twice')
            self.used -= self.leases.pop(ordinal)
            self.released += 1
            self.emit('e03_credits', input_bytes=self.used, input_pages=len(self.leases))
            self.condition.notify_all()

    def wake_cancel(self):
        self.cancel.set()
        with self.condition:
            self.condition.notify_all()


class _Sink:
    def __init__(self, context):
        self.context = context
        self.block = []
        self.bytes = 0
        self.seen = set()
        self.ordinal = {key: ordinal for ordinal, key in enumerate(context.keys)}

    def __call__(self, row):
        context = self.context
        key = row.get('document_id'), row.get('page_number')
        if key not in self.ordinal or key in self.seen:
            raise EarlyAdmissionError(f'Duplicate or unexpected physical page: {key}')
        ordinal = self.ordinal[key]
        payload = row.get('binary_content')
        if payload is not None and not isinstance(payload, bytes):
            raise EarlyAdmissionError('Encoded page must be immutable bytes or an explicit retained failure')
        size = len(payload or b'')
        if self.block and (self.bytes + size > context.limits.block_bytes or context.credits.would_block(size)):
            self.flush_range()
        context.credits.acquire(ordinal, size)
        self.seen.add(key)
        self.block.append((ordinal, row))
        self.bytes += size
        if len(self.block) >= context.limits.block_pages or self.bytes >= context.limits.block_bytes:
            self.flush_range()

    def flush_range(self):
        if not self.block:
            return
        context = self.context
        block = {'rows': self.block, 'bytes': self.bytes,
                 'provenance': png_batch_provenance(row.get('binary_content') for _, row in self.block)}
        self.block, self.bytes = [], 0
        context.emit('e03_block_ready', ordinals=[ordinal for ordinal, _ in block['rows']],
                     block_pages=len(block['rows']), encoded_bytes=block['bytes'])
        context.put(block)


class _Context:
    def __init__(self, task, limits, loop):
        self.task, self.limits, self.loop = task, limits, loop
        self.keys = parent_keys(task)
        # A parent reserves the entire possible encoded input plus a bounded
        # serialized parser result for EVERY remaining physical page before any
        # early request. Held completions never compete for input credits.
        required = limits.parent_input_bytes + len(self.keys) * limits.result_extra_bytes_per_page
        if required > limits.held_bytes:
            raise EarlyAdmissionCapacity(f'Held-result reservation {limits.held_bytes} cannot cover parent bound {required}')
        self.cancel = threading.Event()
        self.queue = asyncio.Queue(maxsize=4)
        self.credits = _Credits(limits, self.cancel, self.emit)
        self.results = {}
        self.held_bytes = self.peak_held = 0
        self.network_tasks = set()
        self.active_requests = self.peak_requests = 0
        self.rendered = None
        self.producer_error = None
        self.producer = None
        self.complete = False
        self.parent_id = hashlib.sha256(json.dumps(self.keys, separators=(',', ':')).encode()).hexdigest()

    def emit(self, event, **fields):
        parent = getattr(self, 'parent_id', None)
        if event == 'e03_credits':
            fields.update(peak_input_bytes=self.credits.peak, peak_input_pages=self.credits.peak_pages,
                          held_result_bytes=self.held_bytes, peak_held_result_bytes=self.peak_held)
        trace.emit(event, parent_id=parent, partition_index=self.task._metadata.get('partition_index'), **fields)

    def put(self, block):
        if self.cancel.is_set():
            raise EarlyAdmissionCancelled('Parent cancelled before block delivery')
        pending = self.queue.put(block)
        try:
            future = asyncio.run_coroutine_threadsafe(pending, self.loop)
        except RuntimeError:
            pending.close()
            if self.cancel.is_set():
                raise EarlyAdmissionCancelled('Parent cancelled before block delivery') from None
            raise
        while True:
            try:
                future.result(timeout=.05)
                return
            except FutureTimeout:
                if self.cancel.is_set():
                    future.cancel()
                    raise EarlyAdmissionCancelled('Parent cancelled waiting for block delivery')

    def accept(self, ordinal, row):
        if ordinal in self.results or (row.get('document_id'), row.get('page_number')) != self.keys[ordinal]:
            raise EarlyAdmissionError('Duplicate result or physical-page identity changed')
        encoded_size = len(row.get('binary_content') or b'')
        held = len(pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))
        if held - encoded_size > self.limits.result_extra_bytes_per_page:
            raise EarlyAdmissionCapacity('Serialized page result exceeds pre-reserved completion capacity')
        if self.held_bytes + held > self.limits.held_bytes:
            raise EarlyAdmissionCapacity('Completed parent results exceeded held-result bound')
        self.results[ordinal] = row
        self.held_bytes += held
        self.peak_held = max(self.peak_held, self.held_bytes)
        self.emit('e03_page_accepted', ordinal=ordinal, held_result_bytes=self.held_bytes,
                  completed_pages=len(self.results))

    def receipt(self):
        return {'parent_id': self.parent_id, 'expected_pages': len(self.keys),
                'completed_pages': len(self.results), 'complete': self.complete,
                'input_bytes': self.credits.used, 'input_pages': len(self.credits.leases),
                'peak_input_bytes': self.credits.peak, 'oversize_serial_peak_bytes': self.credits.oversize_peak,
                'peak_input_pages': self.credits.peak_pages,
                'total_input_bytes': self.credits.total, 'input_releases': self.credits.released,
                'held_result_bytes': self.held_bytes, 'peak_held_result_bytes': self.peak_held,
                'peak_http_requests': self.peak_requests, 'cancelled': self.cancel.is_set(),
                'producer_alive': bool(self.producer and self.producer.is_alive()),
                'limits': vars(self.limits)}


class _TrackedClient:
    def __init__(self, inner, context):
        self.inner, self.context = inner, context

    def __getattr__(self, key):
        return getattr(self.inner, key)

    async def query_model_response(self, **kwargs):
        context = self.context
        if context.cancel.is_set():
            raise EarlyAdmissionCancelled('Parent cancelled before HTTP client entry')
        task = asyncio.current_task()
        context.network_tasks.add(task)
        context.active_requests += 1
        context.peak_requests = max(context.peak_requests, context.active_requests)
        try:
            return await self.inner.query_model_response(**kwargs)
        finally:
            context.active_requests -= 1
            context.network_tasks.discard(task)


class EarlyAdmissionStage(ProcessingStage[FileGroupTask, InterleavedBatch]):
    name = 'page64_render_http_early_admission'
    resources = Resources(cpus=1.0)

    def __init__(self, renderer, inference, limits=None):
        self.renderer, self.inference = renderer, inference
        self.limits = limits or Limits.environment()
        self.last_receipt = None

    def inputs(self):
        return self.renderer.inputs()

    def outputs(self):
        return self.inference.outputs()

    def ray_stage_spec(self):
        return {'is_actor_stage': True}

    def setup(self, worker_metadata=None):
        self._parent_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = None
        self._closed = False
        self._renderer_adapter = BaseStageAdapter(self.renderer)
        self.renderer.setup(worker_metadata)
        self.inference.setup(worker_metadata)

    def cancel(self):
        with self._state_lock:
            self._closed = True
            if self._active is not None:
                self._active.credits.wake_cancel()

    def teardown(self):
        self.cancel()
        with self._parent_lock:
            try:
                self.inference.teardown()
            finally:
                self.renderer.teardown()

    def _render(self, task, sink):
        self.renderer._early_row_sink = sink
        self.renderer._early_cancel_event = sink.context.cancel
        try:
            rendered = self._renderer_adapter.process_batch([task])
            if len(rendered) != 1:
                raise EarlyAdmissionError('Renderer did not return exactly one complete parent')
            return rendered[0]
        finally:
            self.renderer._early_row_sink = None
            self.renderer._early_cancel_event = None

    async def _stream_http(self, context):
        stage = self.inference
        loop = asyncio.get_running_loop()
        client_type = http_impl._ProfiledOpenAIClient if trace.enabled() else http_impl.AsyncOpenAIClient
        inner = client_type(max_concurrent_requests=stage.inference_batch_size,
                            max_retries=stage.max_retries, base_delay=stage.retry_base_delay_s,
                            api_key='unused', base_url=stage._http_endpoint(), timeout=stage.request_timeout_s)
        client = _TrackedClient(inner, context)
        parser_pool = ThreadPoolExecutor(max_workers=stage.inference_batch_size)
        token = http_impl._PARSE_TRANSPORT.set((client, loop))
        semaphore = asyncio.Semaphore(stage.inference_batch_size)
        tasks = []

        async def parse(ordinal, row):
            try:
                if row.get('materialize_error'):
                    trace.emit('page_parse_done', **trace.page_fields(row), status='skipped_render_failure')
                    result = {**row, 'parse_document': None}
                else:
                    async with semaphore:
                        result = await loop.run_in_executor(parser_pool, copy_context().run,
                                                            stage._parse_page, row, True)
                context.accept(ordinal, result)
            finally:
                context.credits.release(ordinal)

        try:
            await asyncio.to_thread(stage._await_http_startup)
            producer_done = False
            while not producer_done:
                if context.cancel.is_set():
                    raise EarlyAdmissionCancelled('Parent cancelled during HTTP admission')
                for task in tasks:
                    if task.done() and not task.cancelled() and task.exception() is not None:
                        raise task.exception()
                try:
                    block = await asyncio.wait_for(context.queue.get(), .05)
                except asyncio.TimeoutError:
                    continue
                if block is None:
                    producer_done = True
                    if context.producer_error is not None:
                        raise context.producer_error
                    continue
                if not has_png_batch_provenance({PNG_METADATA_KEY: block['provenance']},
                                                (row.get('binary_content') for _, row in block['rows'])):
                    raise EarlyAdmissionError('Rendered child byte provenance changed before admission')
                context.emit('e03_block_claimed', ordinals=[ordinal for ordinal, _ in block['rows']],
                             block_pages=len(block['rows']), encoded_bytes=block['bytes'])
                for ordinal, row in block['rows']:
                    trace.emit('page_received', **trace.page_fields(row))
                    tasks.append(asyncio.create_task(parse(ordinal, row)))
            while any(not task.done() for task in tasks):
                if context.cancel.is_set():
                    raise EarlyAdmissionCancelled('Parent cancelled while final HTTP requests were active')
                await asyncio.wait(tasks, timeout=.05, return_when=asyncio.FIRST_EXCEPTION)
                for task in tasks:
                    if task.done() and not task.cancelled() and task.exception() is not None:
                        raise task.exception()
            if context.cancel.is_set():
                raise EarlyAdmissionCancelled('Parent cancelled before complete-parent acceptance')
            await asyncio.gather(*tasks)
        except BaseException:
            context.credits.wake_cancel()
            for task in [*tasks, *context.network_tasks]:
                task.cancel()
            await asyncio.gather(*tasks, *list(context.network_tasks), return_exceptions=True)
            raise
        finally:
            await asyncio.to_thread(parser_pool.shutdown, wait=True, cancel_futures=True)
            if hasattr(inner, 'client'):
                await inner.client.close()
            http_impl._PARSE_TRANSPORT.reset(token)

    async def _run(self, task):
        if 'pdf_page_chunks' in task._metadata and (
            self.inference.extraction_config is None
            or output_version(task._metadata)
            != output_version({'pdf_extraction_config': self.inference.extraction_config})
        ):
            raise http_impl.ExtractionConfigMismatchError('Early parent differs from the admitted HTTP extraction identity')
        context = _Context(task, self.limits, asyncio.get_running_loop())
        with self._state_lock:
            if self._closed:
                raise EarlyAdmissionCancelled('Stage is closed')
            self._active = context
        sink = _Sink(context)
        self.inference._early_cancel_event = context.cancel
        started = time.perf_counter()
        context.emit('e03_parent_begin', expected_pages=len(context.keys), **vars(self.limits))

        def produce():
            try:
                context.rendered = self._render(task, sink)
                sink.flush_range()
                if sink.seen != set(context.keys):
                    raise EarlyAdmissionError('Producer stopped before every expected physical page was delivered')
                context.emit('e03_producer_done', physical_pages=len(sink.seen))
            except BaseException as error:
                context.producer_error = error
            finally:
                try:
                    context.put(None)
                except EarlyAdmissionCancelled:
                    pass

        context.producer = threading.Thread(target=produce, name='page64-range-producer', daemon=True)
        context.producer.start()
        try:
            await self._stream_http(context)
            await asyncio.to_thread(context.producer.join, 5)
            if context.producer.is_alive() or context.rendered is None:
                raise EarlyAdmissionError('Producer did not settle its complete parent')
            if set(context.results) != set(range(len(context.keys))) or context.credits.used:
                raise EarlyAdmissionError('Incomplete parent results or unreleased input credits')
            rows = [context.results[ordinal] for ordinal in range(len(context.keys))]
            context.rendered._metadata[PNG_METADATA_KEY] = png_batch_provenance(row.get('binary_content') for row in rows)
            output = self.inference._finish_parse_task(context.rendered, rows, started)
            for stage in (self.renderer, self.inference):
                self._log_metrics({stage.name + '.' + key: value for key, value in stage._consume_custom_metrics().items()})
            context.complete = True
            context.emit('e03_parent_end', physical_pages=len(rows), held_result_bytes=context.held_bytes)
            return output
        except BaseException:
            context.credits.wake_cancel()
            await asyncio.to_thread(context.producer.join, 5)
            if context.producer.is_alive():
                context.emit('e03_producer_unsettled')
            # Only leases whose parser task never started can remain after all
            # parser futures and the producer have settled. Return each once.
            for ordinal in list(context.credits.leases):
                context.credits.release(ordinal)
            context.emit('e03_cancelled', completed_pages=len(context.results),
                         producer_alive=context.producer.is_alive())
            raise
        finally:
            with self._state_lock:
                # A read still in progress owns the renderer hooks until it returns.
                if context.producer.is_alive():
                    self._closed = True
                self._active = None
            self.inference._early_cancel_event = None
            self.last_receipt = context.receipt()

    def process(self, task):
        # The Ray actor preserves stock serial parent execution. This lock also
        # bounds explicit fixture or future concurrent callers to one parent.
        with self._parent_lock:
            return asyncio.run(self._run(task))


class _Reader(CompositeStage):
    name = 'page64_early_admission_reader'

    def __init__(self, stages):
        super().__init__()
        self.stages = stages

    def decompose(self):
        return self.stages


def configure(reader, limits=None, stages=None):
    original = stages if stages is not None else reader._apply_with_(reader.decompose())
    partitioner, renderer, inference, postprocessor, *merger = original
    if (reader.pages_per_task != 64 or reader.parse_mode != 'single'
            or not isinstance(inference, http_impl.NemotronParseHTTPClientStage)
            or not inference.reuse_encoded_images):
        raise EarlyAdmissionError('Candidate requires the admitted single-pass HTTP Page64 recipe')
    stage = EarlyAdmissionStage(renderer, inference, limits)
    stage.resources = Resources(cpus=max(renderer.resources.cpus, inference.resources.cpus))
    stage = stage.with_(num_workers=inference.num_workers())
    return _Reader([partitioner, stage, postprocessor, *merger])
