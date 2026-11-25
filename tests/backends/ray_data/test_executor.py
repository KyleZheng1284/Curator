# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from unittest.mock import Mock

import pyarrow.parquet as pq
import pytest
import ray
from ray.data import Dataset

from nemo_curator.backends.ray_data import executor as executor_module
from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.composite import NemotronParsePDFReader
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import PDFPageBudgetValidationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask, InterleavedBatch, Task
from tests.backends.utils import capture_logs
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import (
    TEST_EXTRACTION_CONFIG,
    TEST_OUTPUT_VERSION,
    partition,
    project,
    write_manifest,
)


def test_driver_validation_observes_empty_distributed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, _ = partition(tmp_path, [{"file_name": "missing.pdf", "num_pages": 1}])
    stage = PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION)
    runner = RayDataExecutor()
    monkeypatch.setattr(executor_module, "install_ray_data_diagnostics", lambda: None)
    monkeypatch.setattr(executor_module, "register_loguru_serializer", lambda: None)
    monkeypatch.setattr(executor_module, "execute_setup_on_node", lambda *_a, **_k: None)
    monkeypatch.setattr(ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(runner, "_tasks_to_dataset", lambda _tasks: None)
    monkeypatch.setattr(runner, "_dataset_to_tasks", lambda _dataset: [])
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        runner.execute([stage])


def test_driver_validation_must_follow_distributed_stages(tmp_path: Path):
    manifest, _ = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 1}])
    with pytest.raises(ValueError, match="must follow"):
        RayDataExecutor().execute(
            [
                PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION),
                PDFDocumentMergeStage(),
            ]
        )


def test_page_merge_cannot_execute_without_final_validation():
    with pytest.raises(ValueError, match="Required final driver validation"):
        RayDataExecutor().execute([PDFDocumentMergeStage()])


def test_page_budget_stages_require_ray_data(tmp_path: Path):
    stages = [
        PDFDocumentMergeStage(),
        PDFPageBudgetValidationStage(
            manifest_path=str(tmp_path / "manifest.jsonl"), expected_output_version=TEST_OUTPUT_VERSION
        ),
    ]
    for stage in stages:
        with pytest.raises(ValueError, match=rf"{stage.name}.*require RayDataExecutor"):
            Pipeline(name="test", stages=[stage]).run(XennaExecutor())


@dataclass
class _PassThroughStage(ProcessingStage[Task, Task]):
    name: str = "pass_through"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return task


@pytest.mark.parametrize("driver_last", [True, False])
def test_driver_stages_use_identity_when_equal_stages_have_different_placement(
    monkeypatch: pytest.MonkeyPatch, driver_last: bool
):
    distributed = _PassThroughStage()
    driver = _PassThroughStage().with_(ray_stage_spec={RayStageSpecKeys.RUN_ON_DRIVER: True})
    assert distributed == driver
    assert distributed is not driver
    stages = [distributed, driver] if driver_last else [driver, distributed]
    runner = RayDataExecutor()
    if not driver_last:
        with pytest.raises(ValueError, match="must follow"):
            runner.execute(stages)
        return

    monkeypatch.setattr(executor_module, "install_ray_data_diagnostics", lambda: None)
    monkeypatch.setattr(executor_module, "register_loguru_serializer", lambda: None)
    monkeypatch.setattr(executor_module, "execute_setup_on_node", lambda *_a, **_k: None)
    monkeypatch.setattr(ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(runner, "_tasks_to_dataset", lambda _tasks: None)
    monkeypatch.setattr(runner, "_dataset_to_tasks", lambda _dataset: [EmptyTask()])
    distributed_adapter, driver_adapter = Mock(), Mock()
    driver_adapter.return_value.process_batch.return_value = [EmptyTask()]
    monkeypatch.setattr(executor_module, "RayDataStageAdapter", distributed_adapter)
    monkeypatch.setattr(executor_module, "BaseStageAdapter", driver_adapter)

    runner.execute(stages)

    assert distributed_adapter.call_args.args[0] is distributed
    assert driver_adapter.call_args.args[0] is driver
    distributed_adapter.return_value.process_dataset.assert_called_once()
    driver_adapter.return_value.process_batch.assert_called_once()


def test_driver_stage_between_distributed_stages_is_rejected():
    driver = _PassThroughStage().with_(ray_stage_spec={RayStageSpecKeys.RUN_ON_DRIVER: True})
    with pytest.raises(ValueError, match="must follow all distributed stages"):
        RayDataExecutor().execute([_PassThroughStage(), driver, _PassThroughStage()])


def _page_reader(manifest: Path) -> NemotronParsePDFReader:
    return NemotronParsePDFReader(
        manifest_path=str(manifest),
        pages_per_task=4,
        overlap_rendering=True,
        stream_page_tasks=True,
        parse_mode="single",
        inference_server_endpoint="http://unused/v1",
        page_budget_extraction_config=TEST_EXTRACTION_CONFIG,
    )


def test_page_reader_after_another_stage_keeps_its_distributed_partitioner(tmp_path: Path):
    reader = _page_reader(tmp_path / "manifest.jsonl")
    stages = [_PassThroughStage(), reader]
    pipeline = Pipeline(name="prefixed", stages=stages)
    pipeline.build()
    assert executor_module._distributed_stage_range(pipeline.stages) == (0, len(pipeline.stages))

    reader.with_({"pdf_partitioning": {"ray_stage_spec": {"run_on_driver": True}}})
    forced = Pipeline(name="forced", stages=stages)
    forced.build()
    with pytest.raises(ValueError, match="must follow"):
        executor_module._distributed_stage_range(forced.stages)


@pytest.mark.parametrize(
    "options",
    [
        {"runtime_env": {"env_vars": {"PDF_WORKER": "custom"}}},
        {"ray_stage_spec": {"ray_remote_args": {"num_cpus": 2}}},
        {"resources": Resources(cpus=2.0), "num_workers": 2},
        {"ray_stage_spec": {"run_on_driver": False}},
    ],
)
def test_page_reader_partition_worker_options_reach_worker_adapter(tmp_path: Path, options: dict):
    reader = _page_reader(tmp_path / "manifest.jsonl")
    reader.with_({"pdf_partitioning": options})
    pipeline = Pipeline(name="configured", stages=[reader])
    pipeline.build()
    assert executor_module._distributed_stage_range(pipeline.stages) == (0, len(pipeline.stages))
    source = pipeline.stages[0]
    adapter = executor_module.RayDataStageAdapter(source)
    dataset = Mock(spec=Dataset)
    if "ray_remote_args" in source.ray_stage_spec():
        with pytest.raises(ValueError, match=r"must not override.*num_cpus"):
            adapter.process_dataset(dataset)
    else:
        adapter.process_dataset(dataset)
        kwargs = dataset.map_batches.call_args.kwargs
        assert kwargs.get("runtime_env") == source.runtime_env
        assert kwargs["num_cpus"] == source.resources.cpus
        assert kwargs["compute"].size == source.num_workers()


@pytest.mark.usefixtures("shared_ray_client")
def test_decomposed_page_source_without_a_distributed_consumer_preserves_seed_ids(tmp_path: Path):
    manifest = write_manifest(tmp_path, [{"file_name": "nine.pdf", "num_pages": 9}])

    def run(*, trailing_driver: bool) -> list[str]:
        source = _page_reader(manifest).decompose_and_apply_with()[0]
        stages = [source]
        if trailing_driver:
            stages.append(_PassThroughStage().with_(ray_stage_spec={"run_on_driver": True}))
        seeds = [EmptyTask(), EmptyTask()]
        for index, seed in enumerate(seeds):
            seed.task_id = f"0_{index}"
        pipeline = Pipeline(name="partitions", stages=stages)
        return sorted(task.task_id for task in pipeline.run(RayDataExecutor(), initial_tasks=seeds))

    expected = run(trailing_driver=False)
    assert len(set(expected)) == 6
    assert [task_id.rsplit("_", 1)[0] for task_id in expected] == ["0_0"] * 3 + ["0_1"] * 3
    assert run(trailing_driver=True) == [f"{task_id}_0" for task_id in expected]


@pytest.mark.parametrize(("batch_size", "sizes"), [(None, [1, 1, 1]), (2, [2, 1])])
def test_leading_stage_honors_task_batches(batch_size: int | None, sizes: list[int]):
    stage = _PassThroughStage()
    stage.batch_size = batch_size
    stage.process_batch = Mock(side_effect=stage.process_batch)
    tasks = [FileGroupTask(dataset_name="seeds", data=[str(i)]) for i in range(3)]

    assert executor_module._process_leading_stage(stage, tasks) == tasks
    assert [len(call.args[0]) for call in stage.process_batch.call_args_list] == sizes


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_page_partitioner_on_driver_delivers_the_distributed_fanout_without_repartition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest = write_manifest(tmp_path, [{"file_name": f"{i}.pdf", "num_pages": 5} for i in range(6)])
    on_driver = _page_reader(manifest).decompose_and_apply_with()[0]
    on_driver.is_source_stage = True
    in_worker = on_driver.with_(ray_stage_spec={RayStageSpecKeys.RUN_ON_DRIVER: False})
    seeds, plans, delivered = {}, {}, {}
    for placement, source in {"driver": on_driver, "worker": in_worker}.items():
        runner = RayDataExecutor()

        def to_dataset(
            tasks: list[Task], placement: str = placement, build: Callable = runner._tasks_to_dataset
        ) -> Dataset:
            dataset = build(tasks)
            seeds[placement] = (tasks, dataset.num_blocks())
            return dataset

        monkeypatch.setattr(runner, "_tasks_to_dataset", to_dataset)
        with capture_logs() as logs:
            tasks = runner.execute([source, _PassThroughStage()])
        plans[placement] = re.findall(r"Execution plan of Dataset.*?:\s*(.+)", logs.getvalue())[-1]
        delivered[placement] = [
            (task._metadata["partition_index"], task.task_id, task.data, task._metadata)
            for task in sorted(tasks, key=lambda task: task._metadata["partition_index"])
        ]

    assert plans["worker"].split(" -> ")[:3] == [
        "InputDataBuffer[Input]",
        "TaskPoolMapOperator[MapBatches(PDFPartitioningStageTask)]",
        "TaskPoolMapOperator[StreamingRepartition[num_rows_per_block=1,strict=False]]",
    ]
    assert plans["driver"] == "InputDataBuffer[Input] -> TaskPoolMapOperator[MapBatches(_PassThroughStageTask)]"
    seeded, blocks = seeds["driver"]
    assert blocks == len(seeded) == 8
    assert seeds["worker"] == ([EmptyTask()], 1)
    assert [task._metadata["partition_index"] for task in seeded] == list(range(8))
    assert delivered["driver"] == delivered["worker"]
    assert [index for index, *_ in delivered["driver"]] == list(range(8))


@dataclass
class _FanOutStage(ProcessingStage[Task, FileGroupTask]):
    name: str = "fan_out"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, _task: Task) -> list[FileGroupTask]:
        return [FileGroupTask(dataset_name="fan_out", data=[str(index)]) for index in range(3)]


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_fanout_source_without_driver_placement_keeps_its_repartition():
    with capture_logs() as logs:
        tasks = RayDataExecutor().execute([_FanOutStage(), _PassThroughStage()])
    plan = re.findall(r"Execution plan of Dataset.*?:\s*(.+)", logs.getvalue())[-1]
    # Ray Data may fuse the repartition into the stage's own operator.
    assert "MapBatches(_FanOutStageTask)" in plan
    assert "StreamingRepartition[num_rows_per_block=1,strict=False]" in plan
    assert sorted(task.data[0] for task in tasks) == ["0", "1", "2"]


@pytest.mark.usefixtures("shared_ray_client")
@pytest.mark.parametrize("on_driver", [True, False])
def test_real_ray_fanout_without_tasks_yields_no_results_on_either_placement(tmp_path: Path, on_driver: bool):
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("")
    source = _page_reader(manifest).decompose_and_apply_with()[0]
    if not on_driver:
        source = source.with_(ray_stage_spec={RayStageSpecKeys.RUN_ON_DRIVER: False})
    assert RayDataExecutor().execute([source, _PassThroughStage()]) == []


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_fanin_and_validation_for_large_document(tmp_path: Path):
    manifest, tasks = partition(
        tmp_path,
        [
            {"file_name": "largest.pdf", "num_pages": 2227},
            {"file_name": "tail.pdf", "num_pages": 3},
            {"file_name": "failed.pdf", "num_pages": 1},
            {"file_name": "empty.pdf", "num_pages": 2, "selected_page_indices": [1]},
        ],
    )
    chunks = [
        chunk for task in tasks for chunk in project(task, failed={0}, no_elements={1}, text_only=set(range(2, 64)))
    ]
    results = RayDataExecutor().execute(_page_output_stages(tmp_path, manifest), initial_tasks=list(reversed(chunks)))
    assert len(results) == 4
    assert all(task._metadata["pdf_page_budget_validation"]["pages"] == 2232 for task in results)
    # The writer runs inside the merge actor and still reports its own timing after the merge's.
    for task in results:
        assert [stats.stage_name for stats in task._stage_perf] == [
            "pdf_document_merge",
            "interleaved_parquet_writer",
            "pdf_page_budget_validation",
        ]


def _page_output_stages(tmp_path: Path, manifest: Path) -> list[ProcessingStage]:
    return [
        PDFDocumentMergeStage(manifest_path=str(manifest)),
        InterleavedParquetWriterStage(
            path=str(tmp_path / "out"),
            materialize_on_write=False,
            on_materialize_error="warn",
            schema=PAGE_OUTPUT_SCHEMA,
            reject_unknown_columns=True,
            atomic_local_write=True,
            use_output_file_id=True,
            mode="error",
        ),
        PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION),
    ]


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_run_of_single_range_documents_publishes_each_one(tmp_path: Path):
    manifest, tasks = partition(tmp_path, [{"file_name": f"{index}.pdf", "num_pages": 2} for index in range(5)])
    chunks = [chunk for task in tasks for chunk in project(task)]
    assert len(chunks) == 5
    results = RayDataExecutor().execute(_page_output_stages(tmp_path, manifest), initial_tasks=chunks)
    assert len({path for task in results for path in task.data}) == 5
    assert all(task._metadata["pdf_page_budget_validation"]["pages"] == 10 for task in results)


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_document_missing_a_range_is_not_published_and_fails_the_run(tmp_path: Path):
    manifest, tasks = partition(
        tmp_path, [{"file_name": "long.pdf", "num_pages": 130}, {"file_name": "short.pdf", "num_pages": 3}]
    )
    chunks = [chunk for task in tasks for chunk in project(task)]
    assert [chunk._metadata["pdf_chunk"]["pdf_name"] for chunk in chunks] == ["long.pdf"] * 3 + ["short.pdf"]
    del chunks[1]
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        RayDataExecutor().execute(_page_output_stages(tmp_path, manifest), initial_tasks=chunks)
    (published,) = (tmp_path / "out").glob("*.parquet")
    assert pq.read_table(published, columns=["pdf_name"])["pdf_name"][0].as_py() == "short.pdf"


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_incomplete_unlisted_document_fails_even_when_expected_outputs_exist(tmp_path: Path):
    admitted, unlisted = tmp_path / "admitted", tmp_path / "unlisted"
    admitted.mkdir()
    unlisted.mkdir()
    manifest, expected = partition(admitted, [{"file_name": "expected.pdf", "num_pages": 2}])
    _, extra = partition(unlisted, [{"file_name": "unlisted.pdf", "num_pages": 3}], 2)
    chunks = [*project(expected[0]), project(extra[0])[0]]
    with pytest.raises(ray.exceptions.RayTaskError, match="outside the admitted manifest"):
        RayDataExecutor().execute(_page_output_stages(tmp_path, manifest), initial_tasks=chunks)


def test_streaming_merge_requires_final_validation_of_the_same_manifest_selection(tmp_path: Path):
    manifest, _ = partition(tmp_path, [{"file_name": "expected.pdf", "num_pages": 2}])
    merge, writer, _ = _page_output_stages(tmp_path, manifest)
    shorter = PDFPageBudgetValidationStage(
        manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION, max_pdfs=1
    )
    with pytest.raises(ValueError, match="Required final driver validation is missing"):
        RayDataExecutor().execute([merge, writer, shorter])


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_range_delivered_twice_fails_the_run(tmp_path: Path):
    manifest, tasks = partition(
        tmp_path, [{"file_name": "long.pdf", "num_pages": 130}, {"file_name": "short.pdf", "num_pages": 3}]
    )
    chunks = [chunk for task in tasks for chunk in project(task)]
    with pytest.raises(ray.exceptions.RayTaskError, match="nonoverlapping ranges"):
        RayDataExecutor().execute(
            _page_output_stages(tmp_path, manifest), initial_tasks=[*chunks, copy.deepcopy(chunks[1])]
        )


@dataclass
class _ProcessMarkStage(ProcessingStage[Task, Task]):
    """Record which worker process handled a task."""

    mark: str = "process"
    name: str = "mark_process"
    _supports_fan_in_fusion: ClassVar[bool] = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        task._metadata.setdefault(self.mark, []).append(os.getpid())
        return task


@dataclass
class _PairStage(ProcessingStage[Task, Task]):
    """Emit the second task of each pair, which requires one process to see both."""

    name: str = "pair"
    _supports_fan_in_fusion: ClassVar[bool] = True
    _first: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def ray_stage_spec(self) -> dict[str, bool]:
        return {RayStageSpecKeys.STREAMING_FAN_IN: True}

    def process(self, _task: Task) -> Task:
        raise NotImplementedError

    def process_batch(self, tasks: list[Task]) -> list[Task]:
        paired = []
        for task in tasks:
            pair = task._metadata["pair"]
            if pair in self._first:
                task._metadata.update(first_seen_by=self._first.pop(pair), paired_by=os.getpid())
                paired.append(task)
            else:
                self._first[pair] = os.getpid()
        return paired


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_streaming_fan_in_fuses_only_one_occurrence_of_the_following_stage():
    tasks = [FileGroupTask(dataset_name="d", data=[str(i)], _metadata={"pair": i % 4}) for i in range(8)]
    following = _ProcessMarkStage(mark="after")
    results = RayDataExecutor().execute(
        [_ProcessMarkStage(mark="before"), _PairStage(), following, following], initial_tasks=tasks
    )
    assert sorted(task._metadata["pair"] for task in results) == [0, 1, 2, 3]
    (actor,) = {task._metadata["paired_by"] for task in results}
    for task in results:
        fused, separate = task._metadata["after"]
        assert task._metadata["first_seen_by"] == fused == actor
        assert separate != actor
        assert task._metadata["before"][0] != actor


@dataclass
class _ExitingPairStage(_PairStage):
    def process_batch(self, tasks: list[Task]) -> list[Task]:
        if len(self._first) == 1:
            os._exit(1)
        return super().process_batch(tasks)


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_losing_the_streaming_fan_in_actor_fails_the_run():
    tasks = [FileGroupTask(dataset_name="d", data=[str(i)], _metadata={"pair": i}) for i in range(4)]
    with pytest.raises(ray.exceptions.RayActorError):
        RayDataExecutor().execute([_ExitingPairStage()], initial_tasks=tasks)


@dataclass
class _LostOutputStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    name: str = "simulate_lost_distributed_output"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, _task: InterleavedBatch) -> None:
        return None


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_empty_upstream_still_fails_final_inventory(tmp_path: Path):
    manifest, tasks = partition(tmp_path, [{"file_name": "lost.pdf", "num_pages": 2}])
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        RayDataExecutor().execute(
            [
                _LostOutputStage(),
                PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION),
            ],
            initial_tasks=project(tasks[0]),
        )
