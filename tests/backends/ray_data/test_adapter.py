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

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest
from ray.data import ActorPoolStrategy, DataContext, TaskPoolStrategy

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.ray_data import adapter
from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage, Resources
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.tasks import EmptyTask, FileGroupTask


class RecordingDataset:
    """Minimal stand-in for Ray Data Dataset that records map_batches kwargs."""

    def __init__(self):
        self.map_batches_kwargs: dict[str, object] | None = None
        self.batch_size: int | None = None
        self.context = DataContext()

    def map_batches(self, _fn: object, *, batch_size: int | None = None, **kwargs: object):
        self.batch_size = batch_size
        self.map_batches_kwargs = kwargs
        return self

    def repartition(self, **_: object):
        return self


class ConfigurableActorStage(ProcessingStage[EmptyTask, EmptyTask]):
    name = "configurable_actor"
    resources = Resources(cpus=2.0)
    batch_size = 7

    def __init__(
        self,
        ray_stage_spec: dict[str, object] | None = None,
        num_workers: int | None = None,
        num_workers_per_node: float | None = None,
    ):
        self._ray_stage_spec = ray_stage_spec or {}
        self._num_workers = num_workers
        self._num_workers_per_node = num_workers_per_node

    def ray_stage_spec(self) -> dict[str, object]:
        return {
            RayStageSpecKeys.IS_ACTOR_STAGE: True,
            **self._ray_stage_spec,
        }

    def num_workers(self) -> int | None:
        return self._num_workers

    def num_workers_per_node(self) -> float | None:
        return self._num_workers_per_node

    def process(self, task: EmptyTask) -> EmptyTask:
        return task


class LookaheadActorStage(ConfigurableActorStage):
    def lookahead(self, _tasks: list[EmptyTask]) -> None:
        pass


class ConfigurableTaskStage(ConfigurableActorStage):
    name = "configurable_task"
    resources = Resources(cpus=2.0)
    batch_size = 7

    def ray_stage_spec(self) -> dict[str, object]:
        return self._ray_stage_spec


class PlainTaskStage(ProcessingStage[EmptyTask, EmptyTask]):
    name = "plain_task"

    def process(self, task: EmptyTask) -> EmptyTask:
        return task


class TestRayDataStageAdapter:
    def test_process_dataset_uses_compute_for_actor_stages_and_ray_default_for_task_stages(self):
        fixed_actor_kwargs = _map_batches_kwargs(ConfigurableActorStage(num_workers=3))
        autoscaling_actor_kwargs = _map_batches_kwargs(
            ConfigurableActorStage(
                ray_stage_spec={
                    RayStageSpecKeys.MIN_WORKERS: 2,
                    RayStageSpecKeys.MAX_WORKERS: 8,
                    RayStageSpecKeys.INITIAL_WORKERS: 4,
                }
            )
        )
        task_kwargs = _map_batches_kwargs(ConfigurableTaskStage())

        assert fixed_actor_kwargs["compute"] == ActorPoolStrategy(size=3)
        assert autoscaling_actor_kwargs["compute"] == ActorPoolStrategy(min_size=2, max_size=8, initial_size=4)
        assert "compute" not in task_kwargs
        for kwargs in (fixed_actor_kwargs, autoscaling_actor_kwargs, task_kwargs):
            assert "concurrency" not in kwargs

    def test_task_lookahead_starts_an_actors_two_tasks_together_without_queueing_more(self):
        lookahead = {RayStageSpecKeys.TASK_LOOKAHEAD: True}
        fixed = _map_batches_kwargs(LookaheadActorStage(lookahead, num_workers=3))
        autoscaling = _map_batches_kwargs(LookaheadActorStage({**lookahead, RayStageSpecKeys.MAX_WORKERS: 8}))
        without = _map_batches_kwargs(ConfigurableActorStage(num_workers=3))

        assert fixed["max_concurrency"] == autoscaling["max_concurrency"] == 2
        assert fixed["compute"] == ActorPoolStrategy(
            size=3, max_tasks_in_flight_per_actor=2, enable_true_multi_threading=True
        )
        assert autoscaling["compute"] == ActorPoolStrategy(min_size=1, max_size=8, enable_true_multi_threading=True)
        assert "max_concurrency" not in without
        assert without["compute"] == ActorPoolStrategy(size=3)

    def test_task_lookahead_requires_a_callable_actor_hook_before_planning(self):
        with pytest.raises(ValueError, match="callable lookahead"):
            _map_batches_kwargs(ConfigurableActorStage({RayStageSpecKeys.TASK_LOOKAHEAD: True}))
        stage = LookaheadActorStage({RayStageSpecKeys.TASK_LOOKAHEAD: True, RayStageSpecKeys.IS_ACTOR_STAGE: False})
        with pytest.raises(ValueError, match="actor stage"):
            _map_batches_kwargs(stage)

    def test_actor_with_a_lookahead_hook_uses_normal_processing_by_default(self, monkeypatch: pytest.MonkeyPatch):
        stage = LookaheadActorStage()
        monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
        monkeypatch.setattr(stage, "lookahead", lambda *_: pytest.fail("Unexpected lookahead"))
        actor = adapter.create_actor_from_stage(stage)()
        task = FileGroupTask(dataset_name="d", data=["page"])

        assert actor({"item": [task]})["item"][0].data == ["page"]

    @pytest.mark.parametrize("concurrency", [1, 5])
    def test_task_lookahead_owns_its_concurrency(self, concurrency: int):
        spec = {RayStageSpecKeys.RAY_REMOTE_ARGS: {"max_concurrency": concurrency}}
        assert _map_batches_kwargs(ConfigurableActorStage(spec))["max_concurrency"] == concurrency
        stage = LookaheadActorStage({**spec, RayStageSpecKeys.TASK_LOOKAHEAD: True})
        with pytest.raises(ValueError, match="max_concurrency"):
            _map_batches_kwargs(stage)

    def test_task_lookahead_shows_a_call_to_the_stage_on_arrival_and_runs_calls_in_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        events: list[tuple[str, str]] = []
        reached, release = threading.Event(), threading.Event()

        class LookaheadStage(ConfigurableActorStage):
            def lookahead(self, tasks: list[FileGroupTask]) -> None:
                events.append(("seen", tasks[0].data[0]))
                if tasks[0].data[0] == "second":
                    reached.set()

            def process(self, task: FileGroupTask) -> FileGroupTask:
                events.append(("start", task.data[0]))
                if task.data[0] == "first":
                    reached.set()
                    assert release.wait(10)
                events.append(("end", task.data[0]))
                return task

        monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
        actor = adapter.create_actor_from_stage(LookaheadStage({RayStageSpecKeys.TASK_LOOKAHEAD: True}))()
        with ThreadPoolExecutor(max_workers=2) as pool:
            calls = []
            for name in ("first", "second"):
                reached.clear()
                calls.append(pool.submit(actor, {"item": [FileGroupTask(dataset_name="d", data=[name])]}))
                assert reached.wait(10)
            release.set()
            assert [call.result(timeout=10)["item"][0].data for call in calls] == [["first"], ["second"]]

        assert events == [
            ("seen", "first"),
            ("start", "first"),
            ("seen", "second"),
            ("end", "first"),
            ("start", "second"),
            ("end", "second"),
        ]

    def test_task_stage_uses_task_pool_strategy_for_num_workers(self):
        task_kwargs = _map_batches_kwargs(ConfigurableTaskStage(num_workers=3))

        assert task_kwargs["compute"] == TaskPoolStrategy(size=3)

    def test_task_stage_warns_on_actor_pool_sizing_keys(self):
        stage = ConfigurableTaskStage(
            ray_stage_spec={
                RayStageSpecKeys.MIN_WORKERS: 2,
                RayStageSpecKeys.MAX_WORKERS: 8,
                RayStageSpecKeys.INITIAL_WORKERS: 4,
            },
        )

        with mock.patch("nemo_curator.backends.ray_data.adapter.logger.warning") as mock_warning:
            task_kwargs = _map_batches_kwargs(stage)

        assert "compute" not in task_kwargs
        mock_warning.assert_called_once()
        assert "Ignoring ray_stage_spec worker sizing keys" in mock_warning.call_args.args[0]

    @pytest.mark.parametrize(
        ("stage", "expected"),
        [
            (ConfigurableTaskStage(num_workers_per_node=2), TaskPoolStrategy(size=6)),
            (ConfigurableActorStage(num_workers_per_node=2), ActorPoolStrategy(size=6)),
        ],
    )
    def test_num_workers_per_node_sizes_and_spreads_pool(self, stage: ProcessingStage, expected: object):
        with mock.patch("nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=3):
            kwargs = _map_batches_kwargs(stage)

        assert kwargs["compute"] == expected
        assert kwargs["scheduling_strategy"] == "SPREAD"

    def test_num_workers_per_node_passes_ignore_head_node_to_node_count(self):
        stage = ConfigurableTaskStage(num_workers_per_node=2)

        with mock.patch(
            "nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=2
        ) as mock_count:
            kwargs = _map_batches_kwargs(stage, ignore_head_node=True)

        mock_count.assert_called_once_with(ignore_head_node=True)
        assert kwargs["compute"] == TaskPoolStrategy(size=4)

    def test_num_workers_per_node_preserves_scheduling_strategy_override(self):
        stage = ConfigurableTaskStage(
            ray_stage_spec={RayStageSpecKeys.RAY_REMOTE_ARGS: {"scheduling_strategy": "DEFAULT"}},
            num_workers_per_node=2,
        )

        with mock.patch("nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=3):
            kwargs = _map_batches_kwargs(stage)

        assert kwargs["scheduling_strategy"] == "DEFAULT"

    def test_num_workers_per_node_rejects_no_alive_nodes(self):
        stage = ConfigurableTaskStage(num_workers_per_node=2)

        with (
            mock.patch("nemo_curator.backends.ray_data.adapter.get_alive_ray_node_count", return_value=0),
            pytest.raises(ValueError, match="No alive Ray nodes"),
        ):
            _map_batches_kwargs(stage)

    def test_source_fanout_task_stage_uses_task_pool_strategy_for_single_worker_default(self):
        stage = ConfigurableTaskStage(
            ray_stage_spec={RayStageSpecKeys.IS_FANOUT_STAGE: True},
            num_workers=1,
        )

        with mock.patch("nemo_curator.backends.ray_data.adapter.logger.warning") as mock_warning:
            task_kwargs = _map_batches_kwargs(stage)

        assert task_kwargs["compute"] == TaskPoolStrategy(size=1)
        mock_warning.assert_not_called()

    def test_process_dataset_rejects_managed_ray_remote_args(self):
        stage = ConfigurableActorStage(
            ray_stage_spec={
                RayStageSpecKeys.RAY_REMOTE_ARGS: {"compute": ActorPoolStrategy(size=2)},
            }
        )

        with pytest.raises(ValueError, match="must not override Curator-managed map_batches arguments"):
            _map_batches_kwargs(stage)

    def test_build_resource_kwargs_uses_ray_num_cpus_from_spec_over_resources_cpus(self):
        stage = ConfigurableActorStage(ray_stage_spec={RayStageSpecKeys.RAY_NUM_CPUS: 1.0})
        kwargs = _map_batches_kwargs(stage)
        assert kwargs["num_cpus"] == 1.0

    def test_build_resource_kwargs_falls_back_to_resources_cpus_when_ray_num_cpus_absent(self):
        stage = ConfigurableActorStage()
        kwargs = _map_batches_kwargs(stage)
        assert kwargs["num_cpus"] == stage.resources.cpus

    def test_streaming_fan_in_stage_runs_in_one_actor_that_is_not_restarted(self):
        stage = ConfigurableTaskStage(ray_stage_spec={RayStageSpecKeys.STREAMING_FAN_IN: True})
        kwargs = _map_batches_kwargs(stage)

        assert kwargs["compute"] == ActorPoolStrategy(size=1)
        assert kwargs["max_restarts"] == kwargs["max_task_retries"] == 0
        assert kwargs["num_cpus"] == stage.resources.cpus

    def test_only_declared_stage_types_share_the_streaming_merge_actor(self, tmp_path: Path):
        class CustomWriter(InterleavedParquetWriterStage):
            pass

        class CustomMerge(PDFDocumentMergeStage):
            pass

        merge = PDFDocumentMergeStage(manifest_path="manifest.jsonl")
        writer = InterleavedParquetWriterStage(path=str(tmp_path / "output"))

        assert RayDataStageAdapter(merge).fuse_next_stage(writer)
        assert not RayDataStageAdapter(PDFDocumentMergeStage()).fuse_next_stage(writer)
        assert not RayDataStageAdapter(merge).fuse_next_stage(PlainTaskStage())
        assert not RayDataStageAdapter(merge).fuse_next_stage(CustomWriter(path=str(tmp_path / "custom")))
        assert not RayDataStageAdapter(CustomMerge(manifest_path="manifest.jsonl")).fuse_next_stage(writer)

    @pytest.mark.parametrize("owner", ["merge", "writer"])
    @pytest.mark.parametrize(
        "options",
        [
            {"num_workers": 4},
            {"num_workers_per_node": 2},
            {"batch_size": 8},
            {"ray_stage_spec": {RayStageSpecKeys.RAY_NUM_CPUS: 0.25}},
            {"ray_stage_spec": {RayStageSpecKeys.RAY_REMOTE_ARGS: {"runtime_env": {"env_vars": {"A": "1"}}}}},
            {"runtime_env": {"env_vars": {"A": "1"}}},
            {"resources": Resources(cpus=0.0, gpus=1.0)},
            {"resources": Resources(cpus=4.0)},
        ],
    )
    def test_configured_merge_or_writer_keeps_its_own_operator(self, tmp_path: Path, owner: str, options: dict):
        merge = PDFDocumentMergeStage(manifest_path="manifest.jsonl")
        writer = InterleavedParquetWriterStage(path=str(tmp_path / "output"))
        if owner == "merge":
            merge = merge.with_(**options)
        else:
            writer = writer.with_(**options)

        assert not RayDataStageAdapter(merge).fuse_next_stage(writer)

    def test_fused_follower_keeps_its_one_task_batch_when_the_actor_emits_many(self, monkeypatch: pytest.MonkeyPatch):
        class EmitStage(ProcessingStage[FileGroupTask, FileGroupTask]):
            name = "emit"

            def process(self, task: FileGroupTask) -> FileGroupTask:
                return task

        class SingleTaskFollower(EmitStage):
            name = "single"

            def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
                assert len(tasks) == 1
                return tasks

        monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
        actor = adapter.create_actor_from_stage(EmitStage().with_(batch_size=2), SingleTaskFollower())()
        tasks = [FileGroupTask(dataset_name="d", data=[label]) for label in ("first", "second")]

        assert [task.data[0] for task in actor({"item": tasks})["item"]] == ["first", "second"]

    @pytest.mark.parametrize("option", ["max_restarts", "max_task_retries"])
    def test_streaming_fan_in_rejects_restarting_or_retrying_stateful_work(self, option: str):
        stage = ConfigurableTaskStage(
            ray_stage_spec={
                RayStageSpecKeys.STREAMING_FAN_IN: True,
                RayStageSpecKeys.RAY_REMOTE_ARGS: {option: 2},
            }
        )
        with pytest.raises(ValueError, match="requires max_restarts=0 and max_task_retries=0"):
            _map_batches_kwargs(stage)


def _map_batches_kwargs(stage: ProcessingStage, ignore_head_node: bool = False) -> dict[str, object]:
    dataset = RecordingDataset()
    RayDataStageAdapter(stage, ignore_head_node=ignore_head_node).process_dataset(dataset)  # type: ignore[arg-type]
    assert dataset.map_batches_kwargs is not None
    assert dataset.batch_size == stage.batch_size
    return dataset.map_batches_kwargs
