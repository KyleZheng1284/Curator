# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001, S101

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from unittest.mock import Mock

import pytest

from experiments.local_parse.early_admission import EarlyAdmissionCancelled, EarlyAdmissionStage
from nemo_curator.tasks import FileGroupTask


def _parent(name: str) -> FileGroupTask:
    return FileGroupTask(
        dataset_name="fixture",
        data=[json.dumps({"file_name": f"{name}.pdf", "_page_chunk": {"document_id": name, "page_indices": [0]}})],
    )


def test_cancelled_parent_with_blocked_read_prevents_actor_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = Mock()
    inference = Mock()
    stage = EarlyAdmissionStage(renderer, inference)
    stage.setup()
    first, second = _parent("first"), _parent("second")
    read_started = threading.Event()
    second_called = threading.Event()
    release_read = threading.Event()
    producer = None

    def read_range(tasks: list[FileGroupTask]) -> list:
        assert tasks == [first]
        read_started.set()
        assert release_read.wait(15), "test did not release the blocked read"
        return []

    adapter = Mock()
    adapter.process_batch.side_effect = read_range
    monkeypatch.setattr(stage, "_renderer_adapter", adapter)
    thread_errors = Mock()
    monkeypatch.setattr(threading, "excepthook", thread_errors)

    async def interrupt(context) -> None:
        nonlocal producer
        producer = context.producer
        assert await asyncio.to_thread(second_called.wait, 5)
        raise EarlyAdmissionCancelled("cancelled while reading")

    def process_second() -> None:
        second_called.set()
        stage.process(second)

    monkeypatch.setattr(stage, "_stream_http", interrupt)
    with ThreadPoolExecutor(max_workers=2) as calls:
        try:
            pending_first = calls.submit(stage.process, first)
            assert read_started.wait(5)
            pending_second = calls.submit(process_second)
            with pytest.raises(EarlyAdmissionCancelled, match="cancelled while reading"):
                pending_first.result(timeout=10)
            assert stage.last_receipt["producer_alive"]
            assert not stage.last_receipt["complete"]
            assert stage.last_receipt["completed_pages"] == 0
            with pytest.raises(EarlyAdmissionCancelled, match="Stage is closed"):
                pending_second.result(timeout=5)
            adapter.process_batch.assert_called_once_with([first])
        finally:
            release_read.set()
            if producer is not None:
                producer.join(timeout=5)
    assert producer is not None and not producer.is_alive()
    assert renderer._early_row_sink is None
    thread_errors.assert_not_called()


def test_renderer_is_torn_down_even_if_inference_teardown_fails() -> None:
    renderer = Mock()
    inference = Mock()
    inference.teardown.side_effect = RuntimeError("inference teardown failed")
    stage = EarlyAdmissionStage(renderer, inference)
    stage.setup()

    with pytest.raises(RuntimeError, match="inference teardown failed"):
        stage.teardown()

    renderer.teardown.assert_called_once_with()
