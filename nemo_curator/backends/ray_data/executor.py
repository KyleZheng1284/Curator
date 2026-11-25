# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import TYPE_CHECKING, Any

import ray
from loguru import logger
from ray.data import DataContext, Dataset

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.backends.utils import RayStageSpecKeys, execute_setup_on_node, register_loguru_serializer
from nemo_curator.tasks import EmptyTask, Task

from .adapter import RayDataStageAdapter
from .diagnostics import DiagnosticsInstallStatus, install_ray_data_diagnostics

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


def _distributed_stage_range(stages: list["ProcessingStage"]) -> tuple[int, int]:
    """Locate distributed work between an eligible source and trailing driver validation."""
    on_driver = [bool(stage.ray_stage_spec().get(RayStageSpecKeys.RUN_ON_DRIVER)) for stage in stages]
    last = on_driver.index(True) if True in on_driver else len(stages)
    if not all(on_driver[last:]):
        msg = "Driver validation stages must follow all distributed stages"
        raise ValueError(msg)
    # The source adapter expands one seed per block only when a distributed consumer follows.
    first = 1 if last > 1 and getattr(stages[0], "_prefer_driver_as_source", False) else 0
    return first, last


def _process_leading_stage(stage: "ProcessingStage", tasks: list[Task]) -> list[Task]:
    # from_items creates one task per block. Preserve the worker's batch boundaries so the
    # adapter can attribute each fan-out result to the same input task on repeated runs.
    batch_size = 1 if stage.batch_size is None else stage.batch_size
    if batch_size < 1:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    adapter = BaseStageAdapter(stage)
    expanded = []
    for start in range(0, len(tasks), batch_size):
        expanded.extend(adapter.process_batch(tasks[start : start + batch_size]))
    return expanded


class RayDataExecutor(BaseExecutor):
    """Ray Data-based executor for pipeline execution.

    This executor:
    1. Executes setup on Ray nodes for all stages
    2. Converts initial tasks to Ray Data dataset
    3. Applies each stage as a Ray Data transformation (as a task or actor in map_batches)
    4. Returns final results as a list of tasks
    """

    def __init__(self, config: dict[str, Any] | None = None, ignore_head_node: bool = False):
        """Initialize the executor.

        Args:
            config (dict[str, Any], optional): Configuration dictionary.
            ignore_head_node (bool, optional): Whether to skip the Ray head node for
                ``setup_on_node`` and per-node worker-count calculations. Ray Data controls
                ``map_batches`` placement, so this flag does not force workers away from
                the head node.
        """
        super().__init__(config, ignore_head_node)

    def execute(self, stages: list["ProcessingStage"], initial_tasks: list[Task] | None = None) -> list[Task]:
        """Execute the pipeline stages using Ray Data.

        Args:
            stages (list[ProcessingStage]): List of processing stages to execute
            initial_tasks (list[Task], optional): Initial tasks to process (can be None for empty start)

        Returns:
            list[Task]: List of final processed tasks
        """
        if not stages:
            return []
        first, last = _distributed_stage_range(stages)
        trailing_stages = stages[last:]
        required = {getattr(stage, "required_finalizer", None) for stage in stages} - {None}
        present = {getattr(stage, "finalizes", None) for stage in trailing_stages}
        present.update(getattr(stage, "finalizes_scope", None) for stage in trailing_stages)
        if missing := required - present:
            msg = f"Required final driver validation is missing: {sorted(missing, key=str)}"
            raise ValueError(msg)

        diagnostics_status = install_ray_data_diagnostics()
        if diagnostics_status is DiagnosticsInstallStatus.UNSUPPORTED:
            logger.warning(
                f"Ray Data scheduler diagnostics are unavailable for Ray {ray.__version__}; continuing without them."
            )

        register_loguru_serializer()
        # This prevents verbose logging from Ray Data about serialization of the dataclass
        DataContext.get_current().enable_fallback_to_arrow_object_ext_type = True
        # Initialize with initial tasks if provided, otherwise start with EmptyTask
        tasks: list[Task] = initial_tasks or [EmptyTask()]
        output_tasks: list[Task] = []
        # When runtime_env with pip is used, Ray's pip plugin sets up per-stage virtualenvs
        # lazily on first task dispatch by cloning the current virtualenv. The NeMo Curator
        # container's /opt/venv is created with `uv venv --seed` so pip is available in clones.
        try:
            # Initialize ray and explicitly set NOSET to empty
            # This ensures if Xenna was used before which was setting NOSET, we end up overriding it.
            ray.init(
                ignore_reinit_error=True, runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""}}
            )

            # A leading driver stage expands the tasks before the dataset exists, so each task it emits
            # starts as its own block and the plan needs no repartition operator after a fan-out.
            if first:
                tasks = _process_leading_stage(stages[0], tasks)

            # Convert tasks to dataset
            current_dataset = self._tasks_to_dataset(tasks)

            # Execute setup on node for all stages
            execute_setup_on_node(stages, ignore_head_node=self.ignore_head_node)
            logger.info(f"Setup on node complete for all stages. Starting Ray Data pipeline with {len(stages)} stages")

            # Process through each stage
            index = first
            while index < last:
                stage = stages[index]
                index += 1
                # TODO: add pipeline level config for verbosity
                logger.info(f"Processing stage {index}/{len(stages)}: {stage}")
                logger.info(f"  CPU cores: {stage.resources.cpus}, GPU ratio: {stage.resources.gpus}")

                # Create adapter for this stage
                adapter = RayDataStageAdapter(stage, ignore_head_node=self.ignore_head_node)
                if index < last and adapter.fuse_next_stage(stages[index]):
                    index += 1

                # Apply stage transformation
                current_dataset = adapter.process_dataset(current_dataset)
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise
        else:
            # Convert final dataset back to tasks
            # TODO: add pipeline configuration to check if user wants to return last stages output to driver
            output_tasks = self._dataset_to_tasks(current_dataset)
            # Driver validation also sees an empty result set, which no map/group UDF can inspect.
            for stage in trailing_stages:
                output_tasks = BaseStageAdapter(stage).process_batch(output_tasks)
            logger.info(f"Pipeline completed. Final results: {len(output_tasks)} tasks")
        finally:
            # This ensures we unset all the env vars set above during initialize and kill the pending actors.
            ray.shutdown()
        return output_tasks

    def _tasks_to_dataset(self, tasks: list[Task]) -> Dataset:
        """Convert list of tasks to Ray Data dataset.

        Args:
            tasks: List of Task objects

        Returns:
            Ray Data dataset containing Task objects directly
        """
        # Create Ray Data dataset directly from Task objects. A leading driver stage can emit no task,
        # and Ray Data rejects a request for zero blocks.
        return ray.data.from_items(tasks, override_num_blocks=len(tasks) or None)

    def _dataset_to_tasks(self, dataset: Dataset) -> list[Task]:
        """Convert Ray Data dataset back to list of tasks.

        Args:
            dataset: Ray Data dataset containing Task objects

        Returns:
            List of Task objects
        """
        # Get all items from dataset
        items = dataset.take_all()

        # Handle the fact that Ray Data might return different formats
        tasks = []
        for item in items:
            tasks.append(item["item"])
        return tasks
