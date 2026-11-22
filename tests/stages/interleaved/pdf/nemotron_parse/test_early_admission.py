# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise early render admission through the same served Parse HTTP path."""

import base64
import io
import json
import os
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.ray_data import adapter
from nemo_curator.stages.interleaved.pdf.nemotron_parse import _early_admission, _pdf_renderer, inference
from nemo_curator.stages.interleaved.pdf.nemotron_parse.extraction import parser_package_identity
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import image_to_bytes
from nemo_curator.tasks import FileGroupTask
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG, partition
from tests.stages.interleaved.pdf.nemotron_parse.test_inference import http_endpoint  # noqa: F401 - pytest fixture


def _stages(
    endpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pages: dict[str, int] | None = None,
    size: int = 64,
) -> tuple[PDFPreprocessStage, inference.NemotronParseHTTPClientStage, list]:
    package = pytest.importorskip("nemotron_parse.inference")
    pages = pages or {"red": 1, "blue": 1, "missing": 1}
    _, parents = partition(
        tmp_path, [{"file_name": f"{name}.pdf", "num_pages": count} for name, count in pages.items()], size=size
    )
    renderer = PDFPreprocessStage(retain_failures=True, reuse_encoded_images=True)
    monkeypatch.setattr(renderer, "_get_pdf_bytes", lambda name, **_: None if name == "missing.pdf" else name.encode())
    parser = inference.NemotronParseHTTPClientStage(
        endpoint=endpoint,
        model_name="served",
        parse_mode="single",
        inference_batch_size=2,
        reuse_encoded_images=True,
    )
    defaults = package.DocumentParser(model="served", backend=parser._parse_backend)
    try:
        request = asdict(
            inference._server_generation_config(
                {
                    **asdict(defaults.settings["fullpage"]),
                    "skip_special_tokens": False,
                }
            )
        )
    finally:
        defaults.close()
    config = {
        **TEST_EXTRACTION_CONFIG,
        "parser": parser_package_identity(),
        "request": {"model": "served", "stage": "fullpage", "prompt": package.PROMPT, "generation": request},
    }
    renderer._extraction_config = config
    parser.extraction_config = config
    for task in parents:
        task._metadata["pdf_extraction_config"] = config
    return renderer, parser, parents


def test_actor_lookahead_requires_stream_page_tasks() -> None:
    renderer = PDFPreprocessStage()
    parser = inference.NemotronParseHTTPClientStage(endpoint="http://localhost:8000/v1", model_name="served")

    assert _early_admission.EarlyAdmissionStage(renderer, parser).ray_stage_spec() == {"is_actor_stage": True}
    assert _early_admission.EarlyAdmissionStage(renderer, parser, stream_page_tasks=True).ray_stage_spec() == {
        "is_actor_stage": True,
        "task_lookahead": True,
    }


@pytest.mark.parametrize("disable_pdfium_reuse", [False, True])
def test_completed_range_starts_http_before_sibling_and_retains_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
    disable_pdfium_reuse: bool,
) -> None:
    pytest.importorskip("pypdfium2")
    if disable_pdfium_reuse:
        monkeypatch.setattr(_pdf_renderer, "_supports_pdfium_reset", lambda: False)
    renderer, parser, parents = _stages(http_endpoint.endpoint, tmp_path, monkeypatch)
    (parent,) = parents
    for name in ("red", "blue"):
        with Image.new("RGB", (40, 25), name) as image:
            image.save(tmp_path / f"{name}.pdf")

    monkeypatch.setattr(
        renderer,
        "_get_pdf_bytes",
        lambda name, **_: None if name == "missing.pdf" else (tmp_path / name).read_bytes(),
    )
    standard_render = renderer._render_with_timeout
    overlap = False

    def render(data: bytes, name: str, *args: object, **kwargs: object) -> list[Image.Image | None]:
        if overlap and name == "blue.pdf":
            assert http_endpoint.entered.wait(3)
        return standard_render(data, name, *args, **kwargs)

    monkeypatch.setattr(renderer, "_render_with_timeout", render)
    parser.setup()
    try:
        baseline = parser.process(renderer.process(parent))
    finally:
        parser.teardown()
    requests = sorted(json.dumps(row, sort_keys=True) for row in http_endpoint.requests)
    http_endpoint.requests.clear()
    http_endpoint.entered.clear()

    overlap = True
    renderer.persistent_rendering = os.name == "posix"
    fused = _early_admission.EarlyAdmissionStage(renderer, parser)
    fused.setup()
    try:
        actual = fused.process(parent)
        if renderer.persistent_rendering:
            assert (renderer._persistent_renderer.worker_pid is not None) == _pdf_renderer._supports_pdfium_reset()
    finally:
        fused.teardown()

    assert actual.to_pyarrow().equals(baseline.to_pyarrow())
    assert actual._metadata == baseline._metadata
    assert len(requests) == 2
    assert requests == sorted(json.dumps(row, sort_keys=True) for row in http_endpoint.requests)
    assert fused._custom_metrics["num_valid_pages"] == 2
    assert fused._custom_metrics["num_error_pages"] == 1
    assert fused._custom_metrics["image_reuse_count"] == 2
    assert "render_ahead_pages" not in fused._custom_metrics
    assert fused._last is None


def test_cancelling_active_http_closes_worker_even_if_source_read_is_stuck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    renderer, parser, parents = _stages(http_endpoint.endpoint, tmp_path, monkeypatch)
    (parent,) = parents
    allow_read = threading.Event()
    waiting = threading.Event()
    http_endpoint.release = threading.Event()

    def render(_data: bytes, name: str, *_args: object, **_kwargs: object) -> list[Image.Image]:
        if name == "blue.pdf":
            waiting.set()
            assert allow_read.wait(10)
        return [Image.new("RGB", (40, 25), Path(name).stem)]

    monkeypatch.setattr(renderer, "_render_with_timeout", render)
    monkeypatch.setattr(_early_admission, "_RENDER_JOIN_SECONDS", 0.01)
    fused = _early_admission.EarlyAdmissionStage(renderer, parser)
    fused.setup()
    with ThreadPoolExecutor(max_workers=2) as pool:
        processing = pool.submit(fused.process, parent)
        try:
            assert http_endpoint.entered.wait(3)
            assert waiting.wait(3)
            stopping = pool.submit(fused.teardown)
            with pytest.raises(RuntimeError, match="worker cancelled"):
                processing.result(timeout=3)
            stopping.result(timeout=3)
            assert len(http_endpoint.requests) == 1
            with pytest.raises(RuntimeError, match="worker is closed"):
                fused.process(parent)
        finally:
            allow_read.set()
            http_endpoint.release.set()


# Three four-page parents holding a document that cannot be read, a render that always times out and a refused page.
_BOOK = {"a": 5, "missing": 2, "slow": 2, "b": 3}
_ANSWER = "<x_0.1><y_0.1>CELL-{}-{}-{}<x_0.9><y_0.9><class_Text>"
_Page = tuple[int, int, int]


def _colour(name: str, page: int) -> _Page:
    return ord(name[0]), 40 * (page + 1), 0


def _served_page(image_url: str) -> _Page:
    with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
        return image.convert("RGB").getpixel((0, 0))


def _parents(
    endpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pages: dict[str, int] | None = None,
) -> tuple[_early_admission.EarlyAdmissionStage, list[FileGroupTask], dict[_Page, int]]:
    """Pack solid pages into four-page parents; the map names the parent that owns each page colour."""
    renderer, parser, parents = _stages(endpoint, tmp_path, monkeypatch, pages=pages or _BOOK, size=4)

    def render_once(data: bytes, indexes: list[int], *, expected_page_count: int | None = None, **_: object) -> tuple:
        name = data.decode().removesuffix(".pdf")
        if name == "slow":
            return [], True, None
        count = -1 if name == "stale" else expected_page_count
        return [Image.new("RGB", (40, 25), _colour(name, page)) for page in indexes], False, count

    monkeypatch.setattr(renderer, "_render_once", render_once)
    owner = {
        _colour(name, page): index
        for index, parent in enumerate(parents)
        for name, chunk in parent._metadata["pdf_page_chunks"].items()
        for page in chunk["page_indices"]
    }
    return _early_admission.EarlyAdmissionStage(renderer, parser, stream_page_tasks=True), parents, owner


def _call_as_ray_data(
    fused: _early_admission.EarlyAdmissionStage,
    parents: list[FileGroupTask],
    monkeypatch: pytest.MonkeyPatch,
    endpoint: SimpleNamespace,
) -> list[Future]:
    """Run the actor with two calls in flight, each arriving while the call before it is being served."""
    monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
    actor = adapter.create_actor_from_stage(fused)()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            calls = [pool.submit(actor, {"item": [parents[0]]})]
            assert endpoint.entered.wait(10)
            calls += [pool.submit(actor, {"item": [parent]}) for parent in parents[1:]]
    finally:
        fused.teardown()
    return calls


def _render_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == "curator-pdf-render"]


def test_next_parent_renders_ahead_while_its_requests_wait_for_the_current_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    fused, parents, owner = _parents(http_endpoint.endpoint, tmp_path, monkeypatch)
    events: list[tuple[str, int]] = []
    rendered = [threading.Event() for _ in parents]
    late: list[_Page] = []
    hold = False

    def respond(image_url: str) -> tuple[int, str]:
        page = _served_page(image_url)
        events.append(("request", owner[page]))
        # Each parent stays open until the parent after it has been rendered.
        if hold and owner[page] + 1 < len(parents) and not rendered[owner[page] + 1].wait(10):
            late.append(page)
        return (400 if page == _colour("b", 1) else 200), _ANSWER.format(*page)

    render, serve = fused.renderer._process, fused.process

    def render_parent(task: FileGroupTask, **kwargs: object) -> object:
        try:
            return render(task, **kwargs)
        finally:
            rendered[next(index for index, parent in enumerate(parents) if parent is task)].set()

    def serve_parent(task: FileGroupTask) -> object:
        try:
            return serve(task)
        finally:
            events.append(("closed", next(index for index, parent in enumerate(parents) if parent is task)))

    http_endpoint.respond = respond
    monkeypatch.setattr(fused.renderer, "_process", render_parent)
    monkeypatch.setattr(fused, "process", serve_parent)
    serial, ahead_pages = [], []
    fused.setup()
    try:
        for parent in parents:
            serial.append(fused.process(parent))
            ahead_pages.append(fused._consume_custom_metrics()["render_ahead_pages"])
    finally:
        fused.teardown()
    requests = Counter(json.dumps(body, sort_keys=True) for body in http_endpoint.requests)
    http_endpoint.requests.clear()
    http_endpoint.entered.clear()
    events.clear()
    for event in rendered:
        event.clear()

    hold = True
    ahead = [call.result()["item"][0] for call in _call_as_ray_data(fused, parents, monkeypatch, http_endpoint)]

    assert not late
    assert sum(requests.values()) == 8
    assert Counter(json.dumps(body, sort_keys=True) for body in http_endpoint.requests) == requests
    for before, after in zip(serial, ahead, strict=True):
        assert after.to_pyarrow().equals(before.to_pyarrow())
        assert after._metadata == before._metadata
    failures = [json.loads(row["materialize_error"])["type"] for row in ahead[1].to_pyarrow().to_pylist()[1:]]
    assert failures == ["PDFError", "PDFError", "RenderError"]
    for index in range(1, len(parents)):
        assert events.index(("closed", index - 1)) < events.index(("request", index))
    assert ahead_pages == [0, 0, 0]
    assert [task._stage_perf[-1].custom_metrics["render_ahead_pages"] for task in ahead[1:]] == [4, 4]
    assert not _render_threads()


def test_two_parents_share_one_input_credit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    class Credit(_early_admission._InputCredit):
        peak = 0

        @property
        def input_bytes(self) -> int:
            return self._held

        @input_bytes.setter
        def input_bytes(self, held: int) -> None:
            self._held = held
            self.peak = max(self.peak, held)

    monkeypatch.setattr(_early_admission, "_InputCredit", Credit)
    fused, parents, owner = _parents(http_endpoint.endpoint, tmp_path, monkeypatch, dict.fromkeys("abcdefgh", 1))
    sizes = {page: len(image_to_bytes(Image.new("RGB", (40, 25), page))) for page in owner}
    # Three pages fit and a fourth never does.
    monkeypatch.setattr(_early_admission, "_INPUT_BYTES", 3 * max(sizes.values()))
    assert 4 * min(sizes.values()) > _early_admission._INPUT_BYTES
    render = fused.renderer._process
    ahead = threading.Event()
    held: list[int] = []

    def render_parent(task: FileGroupTask, *, range_sink: Callable[[list[dict]], None], **kwargs: object) -> object:
        def sink(rows: list[dict]) -> None:
            range_sink(rows)
            if task is parents[1]:
                ahead.set()

        return render(task, range_sink=sink, **kwargs)

    def respond(image_url: str) -> tuple[int, str]:
        page = _served_page(image_url)
        # The first parent's last two requests stay open until the next parent holds a page as well.
        if page in (_colour("c", 0), _colour("d", 0)):
            held.append(fused._credit.input_bytes if ahead.wait(10) else -1)
        return 200, _ANSWER.format(*page)

    http_endpoint.respond = respond
    monkeypatch.setattr(fused.renderer, "_process", render_parent)
    results = [call.result()["item"][0] for call in _call_as_ray_data(fused, parents, monkeypatch, http_endpoint)]

    assert held[0] == sum(sizes[_colour(name, 0)] for name in "cde")
    assert 0 < fused._credit.peak <= _early_admission._INPUT_BYTES
    assert fused._credit.input_bytes == 0
    assert [[row["sample_id"] for row in batch.to_pyarrow().to_pylist()] for batch in results] == [
        list("abcd"),
        list("efgh"),
    ]
    assert len(http_endpoint.requests) == len(owner)


@pytest.mark.parametrize("fail_at", [1, 2])
def test_lookahead_start_failure_rolls_back_only_its_own_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
    fail_at: int,
) -> None:
    fused, parents, _ = _parents(http_endpoint.endpoint, tmp_path, monkeypatch)
    start = threading.Thread.start
    attempts = 0

    def fail_start(thread: threading.Thread) -> None:
        nonlocal attempts
        if thread.name == "curator-pdf-render":
            attempts += 1
            if attempts == fail_at:
                msg = "Unable to start render thread"
                raise RuntimeError(msg)
        start(thread)

    fused.setup()
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            fused.lookahead(parents[:1])
            previous = fused._last
            with monkeypatch.context() as patch:
                patch.setattr(threading.Thread, "start", fail_start)
                with pytest.raises(RuntimeError, match="Unable to start render thread"):
                    fused.lookahead(parents[1:])
            assert list(fused._ahead) == [previous]
            assert fused._last is previous
            fused.lookahead(parents[1:])
            for parent in parents:
                assert pool.submit(fused.process, parent).result(timeout=10).to_pyarrow().num_rows == 4
        finally:
            fused.teardown()
    assert fused._credit.input_bytes == 0
    assert not _render_threads()


def test_cancelled_page_holds_input_credit_until_parser_thread_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    fused, parents, owner = _parents(http_endpoint.endpoint, tmp_path, monkeypatch, dict.fromkeys("abcdefgh", 1))
    sizes = [len(image_to_bytes(Image.new("RGB", (40, 25), page))) for page in owner]
    monkeypatch.setattr(_early_admission, "_INPUT_BYTES", 2 * max(sizes))
    http_endpoint.release = threading.Event()
    settled, following_admitted = threading.Event(), threading.Event()
    released_after_settlement = []
    release = _early_admission._RangeBuffer.release
    render = fused.renderer._process
    parse = fused.inference._parse_page

    def observe_release(buffer: _early_admission._RangeBuffer, size: int) -> None:
        if buffer.task is parents[0] and size:
            released_after_settlement.append(settled.is_set())
        release(buffer, size)

    def render_parent(task: FileGroupTask, *, range_sink: Callable[[list[dict]], None], **kwargs: object) -> object:
        def sink(rows: list[dict]) -> None:
            if task is parents[0]:
                if rows[0]["pdf_name"] == "a.pdf":
                    range_sink(rows)
                    assert http_endpoint.entered.wait(5)
                    # Fail the other consumer with a duplicate carrying no charged bytes.
                    range_sink([{**rows[0], "binary_content": b""}])
            else:
                range_sink(rows)
                following_admitted.set()

        return render(task, range_sink=sink, **kwargs)

    def delayed_parse(row: dict, reuse: bool) -> dict:
        try:
            return parse(row, reuse)
        finally:
            if row["pdf_name"] == "a.pdf":
                assert following_admitted.wait(5)
                settled.set()

    monkeypatch.setattr(_early_admission._RangeBuffer, "release", observe_release)
    monkeypatch.setattr(fused.renderer, "_process", render_parent)
    monkeypatch.setattr(fused.inference, "_parse_page", delayed_parse)
    fused.setup()
    try:
        fused.lookahead(parents)
        with pytest.raises(ValueError, match="duplicate"):
            fused.process(parents[0])
        assert released_after_settlement == [True]
        http_endpoint.release.set()
        assert fused.process(parents[1]).to_pyarrow().num_rows == 4
    finally:
        http_endpoint.release.set()
        fused.teardown()
    assert fused._credit.input_bytes == 0
    assert not _render_threads()


def test_shutdown_while_rendering_ahead_leaves_no_request_thread_or_renderer_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    pytest.importorskip("pypdfium2")
    renderer, parser, parents = _stages(
        http_endpoint.endpoint, tmp_path, monkeypatch, pages={"red": 1, "blue": 1}, size=1
    )
    for name in ("red", "blue"):
        with Image.new("RGB", (40, 25), name) as image:
            image.save(tmp_path / f"{name}.pdf")
    monkeypatch.setattr(renderer, "_get_pdf_bytes", lambda name, **_: (tmp_path / name).read_bytes())
    renderer.persistent_rendering = os.name == "posix"
    render = renderer._process
    rendered = threading.Event()

    def render_parent(task: FileGroupTask, **kwargs: object) -> object:
        try:
            return render(task, **kwargs)
        finally:
            if task is parents[1]:
                rendered.set()

    monkeypatch.setattr(renderer, "_process", render_parent)
    monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
    fused = _early_admission.EarlyAdmissionStage(renderer, parser, stream_page_tasks=True)
    actor = adapter.create_actor_from_stage(fused)()
    http_endpoint.release = threading.Event()
    with ThreadPoolExecutor(max_workers=3) as pool:
        current = pool.submit(actor, {"item": [parents[0]]})
        try:
            assert http_endpoint.entered.wait(10)
            following = pool.submit(actor, {"item": [parents[1]]})
            assert rendered.wait(10)
            helper = renderer._persistent_renderer.worker_pid if renderer.persistent_rendering else None
            stopping = pool.submit(fused.teardown)
            with pytest.raises(RuntimeError, match="worker cancelled"):
                current.result(timeout=10)
            with pytest.raises(RuntimeError, match="worker is closed"):
                following.result(timeout=10)
            stopping.result(timeout=10)
        finally:
            http_endpoint.release.set()

    assert len(http_endpoint.requests) == 1
    assert not _render_threads()
    assert fused._credit.input_bytes == 0
    assert renderer._persistent_renderer is None
    if helper is not None:
        with pytest.raises(ProcessLookupError):
            os.kill(helper, 0)


def test_shutdown_does_not_wait_for_a_source_read_stuck_while_rendering_ahead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    fused, parents, owner = _parents(http_endpoint.endpoint, tmp_path, monkeypatch, {"a": 4, "b": 4})
    read = fused.renderer._get_pdf_bytes
    allow_read, reading = threading.Event(), threading.Event()

    def stuck_read(name: str, **kwargs: object) -> bytes | None:
        if name == "b.pdf":
            reading.set()
            assert allow_read.wait(10)
        return read(name, **kwargs)

    monkeypatch.setattr(fused.renderer, "_get_pdf_bytes", stuck_read)
    monkeypatch.setattr(_early_admission, "_RENDER_JOIN_SECONDS", 0.01)
    monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
    actor = adapter.create_actor_from_stage(fused)()
    http_endpoint.release = threading.Event()
    with ThreadPoolExecutor(max_workers=3) as pool:
        current = pool.submit(actor, {"item": [parents[0]]})
        try:
            assert http_endpoint.entered.wait(10)
            following = pool.submit(actor, {"item": [parents[1]]})
            assert reading.wait(4)
            stopping = pool.submit(fused.teardown)
            with pytest.raises(RuntimeError, match="worker cancelled"):
                current.result(timeout=10)
            with pytest.raises(RuntimeError, match="worker is closed"):
                following.result(timeout=10)
            stopping.result(timeout=10)
            (stuck,) = _render_threads()
        finally:
            allow_read.set()
            http_endpoint.release.set()

    stuck.join(10)
    assert not _render_threads()
    assert fused._credit.input_bytes == 0
    served = [_served_page(body["messages"][0]["content"][1]["image_url"]["url"]) for body in http_endpoint.requests]
    assert served
    assert {owner[page] for page in served} == {0}


def test_failed_parent_leaves_the_parent_rendered_ahead_whole(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    fused, parents, owner = _parents(http_endpoint.endpoint, tmp_path, monkeypatch)
    render = fused.renderer._process
    rendered = threading.Event()
    served: Counter[_Page] = Counter()
    late: list[_Page] = []
    duplicate = False

    def render_parent(task: FileGroupTask, *, range_sink: Callable[[list[dict]], None], **kwargs: object) -> object:
        def sink(rows: list[dict]) -> None:
            range_sink(rows)
            if duplicate and task is parents[0]:
                range_sink(rows[:1])

        try:
            return render(task, range_sink=sink, **kwargs)
        finally:
            if task is parents[1]:
                rendered.set()

    def respond(image_url: str) -> tuple[int, str]:
        page = _served_page(image_url)
        served[page] += 1
        # The failing parent stays open until the parent after it has been rendered.
        if duplicate and owner[page] == 0 and not rendered.wait(10):
            late.append(page)
        return 200, _ANSWER.format(*page)

    http_endpoint.respond = respond
    monkeypatch.setattr(fused.renderer, "_process", render_parent)
    fused.setup()
    try:
        whole = fused.process(parents[1])
    finally:
        fused.teardown()
    served.clear()
    http_endpoint.entered.clear()
    rendered.clear()

    duplicate = True
    failed, following = _call_as_ray_data(fused, parents[:2], monkeypatch, http_endpoint)

    assert not late
    with pytest.raises(ValueError, match="Unexpected or duplicate physical page"):
        failed.result()
    (after,) = following.result()["item"]
    assert after.to_pyarrow().equals(whole.to_pyarrow())
    assert after._stage_perf[-1].custom_metrics["render_ahead_pages"] == 4
    assert {page: count for page, count in served.items() if owner[page] == 1} == {_colour("a", 4): 1}
    assert not _render_threads()
    assert fused._credit.input_bytes == 0


def test_render_error_ahead_fails_its_own_parent_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    fused, parents, owner = _parents(http_endpoint.endpoint, tmp_path, monkeypatch, {"a": 4, "stale": 4, "b": 4})
    render = fused.renderer._process
    failed = threading.Event()
    late: list[_Page] = []
    hold = False

    def render_parent(task: FileGroupTask, **kwargs: object) -> object:
        try:
            return render(task, **kwargs)
        finally:
            if task is parents[1]:
                failed.set()

    def respond(image_url: str) -> tuple[int, str]:
        page = _served_page(image_url)
        # The first parent stays open until the render after it has failed.
        if hold and owner[page] == 0 and not failed.wait(10):
            late.append(page)
        return 200, _ANSWER.format(*page)

    http_endpoint.respond = respond
    monkeypatch.setattr(fused.renderer, "_process", render_parent)
    fused.setup()
    try:
        first = fused.process(parents[0])
        with pytest.raises(ValueError, match="Stale num_pages for stale"):
            fused.process(parents[1])
        last = fused.process(parents[2])
    finally:
        fused.teardown()
    requests = Counter(json.dumps(body, sort_keys=True) for body in http_endpoint.requests)
    http_endpoint.requests.clear()
    http_endpoint.entered.clear()
    failed.clear()

    hold = True
    calls = _call_as_ray_data(fused, parents, monkeypatch, http_endpoint)

    assert not late
    assert calls[0].result()["item"][0].to_pyarrow().equals(first.to_pyarrow())
    with pytest.raises(ValueError, match="Stale num_pages for stale"):
        calls[1].result()
    assert calls[2].result()["item"][0].to_pyarrow().equals(last.to_pyarrow())
    assert Counter(json.dumps(body, sort_keys=True) for body in http_endpoint.requests) == requests
    assert sum(requests.values()) == 8


def test_render_left_by_a_call_that_never_ran_is_discarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    fused, parents, owner = _parents(http_endpoint.endpoint, tmp_path, monkeypatch)
    http_endpoint.respond = lambda image_url: (200, _ANSWER.format(*_served_page(image_url)))
    fused.setup()
    try:
        whole = fused.process(parents[2])
        http_endpoint.requests.clear()
        fused.lookahead(parents[1:])
        with monkeypatch.context() as patch:
            patch.setattr(fused, "_start", lambda *_: pytest.fail("Previously queued parent was rendered twice"))
            after = fused.process(parents[2])
        assert fused._last is None
        assert not _render_threads()
        assert fused._credit.input_bytes == 0
    finally:
        fused.teardown()

    assert after.to_pyarrow().equals(whole.to_pyarrow())
    served = [_served_page(body["messages"][0]["content"][1]["image_url"]["url"]) for body in http_endpoint.requests]
    assert sorted(owner[page] for page in served) == [2, 2, 2]
