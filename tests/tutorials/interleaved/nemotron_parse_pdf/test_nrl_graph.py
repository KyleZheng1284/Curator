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

# ruff: noqa: ANN401, EM102, INP001, PD008, UP037

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest


def _load_recipe() -> ModuleType:
    recipe_path = (
        Path(__file__).resolve().parents[4] / "tutorials" / "interleaved" / "nemotron_parse_pdf" / "nrl_graph.py"
    )
    sys.path.insert(0, str(recipe_path.parent))
    spec = importlib.util.spec_from_file_location("curator_tutorial_nrl_graph", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipe() -> ModuleType:
    return _load_recipe()


def _page_image() -> dict[str, Any]:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (64, 48), color=(240, 240, 240))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    return {
        "image_b64": base64.b64encode(encoded.getvalue()).decode("ascii"),
        "orig_shape_hw": [48, 64],
    }


def _raw_output() -> str:
    return (
        "<x_0.1><y_0.1>Heading<x_0.4><y_0.2><class_Text>"
        "<x_0.1><y_0.25>| A | B |\n| --- | --- |\n| 1 | 2 |<x_0.6><y_0.35><class_Table>"
        "<x_0.485><y_0.49><x_0.515><y_0.51><class_Picture>"
    )


def _page(raw_output: Any, *, page_number: int = 1) -> dict[str, Any]:
    return {
        "path": "/raid/input/example.pdf",
        "page_number": page_number,
        "page_image": _page_image(),
        "metadata": {"error": None},
        "nemotron_parse_v1_2": {"raw_output": raw_output, "error": None},
    }


def test_projection_preserves_order_and_textless_picture(recipe: ModuleType) -> None:
    raw_output = _raw_output()

    result = recipe.project_nrl_pages(pd.DataFrame([_page(raw_output)]))

    assert tuple(result.columns) == recipe.PROJECTION_COLUMNS
    assert result["record_type"].tolist() == ["page_outcome", "element", "element", "element"]
    outcome = result.iloc[0]
    assert outcome["native_page_number"] == 1
    assert outcome["page_outcome"] == "parsed"
    assert outcome["element_count"] == 3
    assert outcome["issues_json"] == "[]"
    assert outcome["raw_output_sha256"] == hashlib.sha256(raw_output.encode()).hexdigest()
    assert "raw_output" not in result.columns

    elements = result.iloc[1:].reset_index(drop=True)
    assert elements["element_index"].tolist() == [0, 1, 2]
    assert elements["element_class"].tolist() == ["Text", "Table", "Picture"]
    assert elements["modality"].tolist() == ["text", "table", "image"]
    assert elements["content_type"].tolist() == ["text/markdown", "text/markdown", "image/png"]
    assert elements.iloc[2]["text_content"] == ""
    assert elements.iloc[2]["binary_content"].startswith(b"\x89PNG\r\n\x1a\n")
    assert set(elements["bbox_coordinate_space"]) == {recipe.COORDINATE_SPACE}


@pytest.mark.parametrize(
    "table_body",
    [
        (
            "\\begin{tabular}{ccc}\n"
            "\\multicolumn{3}{c}{Merged <Region> heading} \\\\\n"
            "\\multirow{2}{*}{Group} & A & B \\\\\n"
            " & C & D \\\\\n"
            "\\end{tabular}"
        ),
        "| A | B |\n| --- | --- |\n| 1 | 2 |",
    ],
    ids=["merged-latex", "markdown"],
)
def test_projection_preserves_native_table_body(recipe: ModuleType, table_body: str) -> None:
    raw_output = (
        "<x_0.1><y_0.1>Before<x_0.4><y_0.2><class_Text>"
        f"<x_0.1><y_0.3>{table_body}<x_0.8><y_0.6><class_Table>"
        "<x_0.1><y_0.7>After<x_0.4><y_0.8><class_Text>"
    )

    result = recipe.project_nrl_pages(pd.DataFrame([_page(raw_output)]))
    elements = result[result["record_type"] == "element"].reset_index(drop=True)

    assert result.iloc[0]["page_outcome"] == "parsed"
    assert result.iloc[0]["element_count"] == 3
    assert elements["element_index"].tolist() == [0, 1, 2]
    assert elements["text_content"].tolist() == ["Before", table_body, "After"]
    assert elements.iloc[1]["element_class"] == "Table"
    assert elements.iloc[1]["modality"] == "table"
    assert elements.iloc[1]["content_type"] == "text/markdown"
    assert elements.iloc[1]["binary_content"] is None


def test_opt_in_capture_gets_raw_page_without_expanding_result(
    recipe: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []
    capture_module = ModuleType("nrl_compare")

    def capture_page(root: str, **kwargs: Any) -> None:
        captured.append({"root": root, **kwargs})

    capture_module.capture_page = capture_page
    monkeypatch.setitem(sys.modules, "nrl_compare", capture_module)
    data = pd.DataFrame([_page(_raw_output())])
    expected = recipe.project_nrl_pages(data)
    result = recipe.NRLCuratorProjectionOperator(evidence_root="/raid/evidence").process(data)
    pd.testing.assert_frame_equal(result, expected)
    assert captured[0]["raw_output"] == _raw_output()
    assert captured[0]["image_bytes"].startswith(b"\x89PNG")
    assert captured[0]["finish_reason"] == "stop"
    assert captured[0]["native_page_number"] == 1


def test_opt_in_capture_failure_does_not_silently_publish(recipe: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    capture_module = ModuleType("nrl_compare")

    def capture_page(*_args: Any, **_kwargs: Any) -> None:
        message = "evidence disk unavailable"
        raise OSError(message)

    capture_module.capture_page = capture_page
    monkeypatch.setitem(sys.modules, "nrl_compare", capture_module)
    with pytest.raises(OSError, match="evidence disk unavailable"):
        recipe.project_nrl_pages(pd.DataFrame([_page(_raw_output())]), evidence_root="/raid/evidence")


def test_malformed_tail_fails_whole_page_without_partial_elements(recipe: ModuleType) -> None:
    raw_output = "<x_0.1><y_0.1>valid<x_0.4><y_0.2><class_Text>truncated tail"

    result = recipe.project_nrl_pages(pd.DataFrame([_page(raw_output)]))

    assert len(result) == 1
    outcome = result.iloc[0]
    assert outcome["record_type"] == "page_outcome"
    assert outcome["page_outcome"] == "failed"
    assert outcome["element_count"] == 0
    assert json.loads(outcome["issues_json"])[0]["kind"] == "truncated_or_unparseable_model_output"
    assert outcome["raw_output_sha256"] == hashlib.sha256(raw_output.encode()).hexdigest()


def test_nested_finish_reason_error_wins_over_parseable_raw_output(recipe: ModuleType) -> None:
    page = _page(_raw_output())
    page["nemotron_parse_v1_2"]["error"] = {
        "stage": "nemotron_parse_pages_finish_reason",
        "type": "IncompleteModelOutputError",
        "message": "finish_reason='length'",
        "traceback": "must not cross the boundary",
    }

    result = recipe.project_nrl_pages(pd.DataFrame([page]))

    assert len(result) == 1
    assert result.iloc[0]["page_outcome"] == "failed"
    issue = json.loads(result.iloc[0]["issues_json"])[0]
    assert issue == {
        "error": {
            "message": "finish_reason='length'",
            "stage": "nemotron_parse_pages_finish_reason",
            "type": "IncompleteModelOutputError",
        },
        "kind": "page_stage_error",
    }


def test_empty_model_output_is_explicit_and_not_preclassified_as_valid_blank(recipe: ModuleType) -> None:
    result = recipe.project_nrl_pages(pd.DataFrame([_page("")]))

    assert len(result) == 1
    outcome = result.iloc[0]
    assert outcome["page_outcome"] == "empty"
    assert outcome["element_count"] == 0
    assert outcome["issues_json"] == "[]"
    assert outcome["raw_output_sha256"] == hashlib.sha256(b"").hexdigest()


def test_split_failure_keeps_native_page_zero(recipe: ModuleType) -> None:
    row = {
        "path": "/raid/input/corrupt.pdf",
        "page_number": 0,
        "metadata": {
            "source_path": "/raid/input/corrupt.pdf",
            "error": {"stage": "split_pdf", "type": "PdfError", "message": "corrupt"},
        },
        "nemotron_parse_v1_2": {"raw_output": None, "error": None},
    }

    result = recipe.project_nrl_pages(pd.DataFrame([row]))

    assert result.iloc[0]["native_page_number"] == 0
    assert result.iloc[0]["page_outcome"] == "failed"
    assert result.iloc[0]["element_count"] == 0


def test_picture_crop_failure_discards_preceding_elements(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recipe, "_crop_picture_bytes", lambda _image, _bbox: None)

    result = recipe.project_nrl_pages(pd.DataFrame([_page(_raw_output())]))

    assert len(result) == 1
    assert result.iloc[0]["page_outcome"] == "failed"
    assert json.loads(result.iloc[0]["issues_json"])[0]["kind"] == "picture_crop_failure"


def test_validator_rejects_nonfinite_or_out_of_range_bbox(recipe: ModuleType) -> None:
    result = recipe.project_nrl_pages(pd.DataFrame([_page(_raw_output())]))
    result.at[1, "bbox_xyxy_norm_json"] = "[0.0,0.0,2.0,1.0]"

    with pytest.raises(ValueError, match="four finite ordered floats"):
        recipe.validate_projection_envelope(result)


def test_zero_area_bbox_fails_the_whole_page(recipe: ModuleType) -> None:
    raw_output = "<x_0.2><y_0.1>zero width<x_0.2><y_0.3><class_Text>"

    result = recipe.project_nrl_pages(pd.DataFrame([_page(raw_output)]))

    assert len(result) == 1
    assert result.iloc[0]["page_outcome"] == "failed"
    assert json.loads(result.iloc[0]["issues_json"])[0]["kind"] == "invalid_element"


@pytest.mark.parametrize("prefix_or_tail", ["garbage", "<x_0.1>", "<class_Text>"])
def test_malformed_prefix_or_tail_fails_closed(recipe: ModuleType, prefix_or_tail: str) -> None:
    raw_output = prefix_or_tail + "<x_0.1><y_0.1>valid<x_0.4><y_0.2><class_Text>"

    result = recipe.project_nrl_pages(pd.DataFrame([_page(raw_output)]))

    assert len(result) == 1
    assert result.iloc[0]["page_outcome"] == "failed"


def test_chart_and_infographic_are_text_markdown_in_model_order(recipe: ModuleType) -> None:
    raw_output = "<x_0.1><y_0.1>chart<x_0.4><y_0.2><class_Chart><x_0.1><y_0.3>graphic<x_0.4><y_0.4><class_Infographic>"

    result = recipe.project_nrl_pages(pd.DataFrame([_page(raw_output)]))
    elements = result[result["record_type"] == "element"]

    assert elements["element_class"].tolist() == ["Chart", "Infographic"]
    assert elements["modality"].tolist() == ["text", "text"]
    assert elements["content_type"].tolist() == ["text/markdown", "text/markdown"]


def test_validator_reconciles_declared_element_count(recipe: ModuleType) -> None:
    result = recipe.project_nrl_pages(pd.DataFrame([_page(_raw_output())]))
    result.at[0, "element_count"] = 4

    with pytest.raises(ValueError, match="element count mismatch"):
        recipe.validate_projection_envelope(result)


def test_validator_requires_numeric_bbox_and_real_png(recipe: ModuleType) -> None:
    result = recipe.project_nrl_pages(pd.DataFrame([_page(_raw_output())]))
    result.at[1, "bbox_xyxy_norm_json"] = '["0.0",0.0,0.4,0.2]'
    with pytest.raises(ValueError, match="bbox coordinates must be JSON numbers"):
        recipe.validate_projection_envelope(result)

    result = recipe.project_nrl_pages(pd.DataFrame([_page(_raw_output())]))
    result.at[3, "binary_content"] = b"not a png"
    with pytest.raises(ValueError, match="inline PNG bytes"):
        recipe.validate_projection_envelope(result)


def test_build_projection_graph_reuses_nrl_graph_with_fixed_settings(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeGraph:
        def __rshift__(self, operator: Any) -> "FakeGraph":
            captured["operator"] = operator
            return self

    def fake_build_graph(**kwargs: Any) -> FakeGraph:
        captured["kwargs"] = kwargs
        return FakeGraph()

    monkeypatch.setattr(recipe, "build_graph", fake_build_graph)

    graph = recipe.build_projection_graph()

    assert isinstance(graph, FakeGraph)
    params = captured["kwargs"]["extract_params"]
    assert captured["kwargs"]["extraction_mode"] == "pdf"
    assert captured["kwargs"]["split_config"] == {}
    assert captured["kwargs"]["stage_order"] == ()
    assert params.method == "nemotron_parse"
    assert params.nemotron_parse_model == "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    assert params.extract_images is False
    assert params.extract_text is False
    assert params.nemotron_parse_invoke_url is None
    assert params.invoke_url is None
    assert params.render_mode == "full_dpi"
    assert params.dpi == 200
    assert params.image_format == "png"
    assert params.extract_page_as_image is True
    assert isinstance(captured["operator"], recipe.NRLCuratorProjectionOperator)
    assert isinstance(captured["operator"], recipe.CPUOperator)


def test_real_graph_is_existing_pdf_chain_plus_one_projection(recipe: ModuleType) -> None:
    graph = recipe.build_projection_graph()
    nodes = []
    node = graph.roots[0]
    while node is not None:
        nodes.append(node)
        node = node.children[0] if node.children else None

    assert [node.name for node in nodes] == [
        "DocToPdfConversionActor",
        "PDFSplitActor",
        "PDFExtractionActor",
        "NemotronParseActor",
        "NRLCuratorProjectionOperator",
    ]
    assert nodes[3].operator_kwargs["nemotron_parse_model"] == recipe.PARSE_MODEL
    assert nodes[3].operator_kwargs["task_prompt"] == recipe.PARSE_TASK_PROMPT
    assert isinstance(nodes[-1].operator, recipe.CPUOperator)


@pytest.mark.parametrize(("parse_batch_size", "parse_cpus"), [(64, 1), (64, 4), (128, 1)])
@pytest.mark.parametrize("projection_block_rows", [None, 16])
def test_batch_executor_keeps_one_parse_actor_and_bounds_projection_pool(
    recipe: ModuleType, parse_batch_size: int, parse_cpus: int, projection_block_rows: int | None
) -> None:
    from nemo_retriever.graph.pipeline_graph import Graph

    executor = recipe.build_projection_executor(
        Graph(),
        projection_workers=4,
        projection_block_rows=projection_block_rows,
        parse_batch_size=parse_batch_size,
        parse_cpus=parse_cpus,
    )

    assert isinstance(executor, recipe.RayDataExecutor)
    assert executor._node_overrides == {
        "NemotronParseActor": {"batch_size": parse_batch_size, "num_cpus": parse_cpus},
        "NRLCuratorProjectionOperator": {
            "concurrency": 4,
            "num_cpus": 1,
            **({"target_num_rows_per_block": projection_block_rows} if projection_block_rows is not None else {}),
        },
    }
    assert executor._auto_concurrency_nodes == {"NRLCuratorProjectionOperator"}
    assert executor._source_cpu_reservation == 1
    with pytest.raises(ValueError, match="1 through 8"):
        recipe.build_projection_executor(Graph(), projection_workers=9)


@pytest.mark.parametrize("option", ["parse_batch_size", "parse_cpus"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "4"])
def test_parse_scheduling_rejects_invalid_values_before_empty_execution(
    recipe: ModuleType, option: str, value: Any
) -> None:
    from nemo_retriever.graph.pipeline_graph import Graph

    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        recipe.build_projection_executor(Graph(), **{option: value})
    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        recipe.run_nrl_graph([], **{option: value})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "16"])
def test_projection_block_rows_rejects_invalid_values_before_execution(recipe: ModuleType, value: Any) -> None:
    from nemo_retriever.graph.pipeline_graph import Graph

    with pytest.raises(ValueError, match="projection_block_rows must be a positive integer"):
        recipe.build_projection_executor(Graph(), projection_block_rows=value)
    with pytest.raises(ValueError, match="projection_block_rows must be a positive integer"):
        recipe.run_nrl_graph([], projection_block_rows=value)


def test_inprocess_projection_block_rows_cannot_be_silently_ignored(recipe: ModuleType) -> None:
    from nemo_retriever.graph.pipeline_graph import Graph

    assert isinstance(recipe.build_projection_executor(Graph(), run_mode="inprocess"), recipe.InprocessExecutor)
    with pytest.raises(ValueError, match="projection_block_rows requires batch mode"):
        recipe.build_projection_executor(Graph(), run_mode="inprocess", projection_block_rows=16)
    with pytest.raises(ValueError, match="projection_block_rows requires batch mode"):
        recipe.run_nrl_graph([], run_mode="inprocess", projection_block_rows=16)


def test_parse_scheduling_rejects_silently_promoted_batch_one(recipe: ModuleType) -> None:
    from nemo_retriever.graph.pipeline_graph import Graph

    with pytest.raises(ValueError, match="pinned NRL executor promotes batch size 1 to 64"):
        recipe.build_projection_executor(Graph(), parse_batch_size=1)
    with pytest.raises(ValueError, match="pinned NRL executor promotes batch size 1 to 64"):
        recipe.run_nrl_graph([], parse_batch_size=1)


@pytest.mark.parametrize("options", [{"parse_batch_size": 128}, {"parse_cpus": 4}])
def test_inprocess_execution_does_not_silently_ignore_scheduling(recipe: ModuleType, options: dict[str, int]) -> None:
    from nemo_retriever.graph.pipeline_graph import Graph

    with pytest.raises(ValueError, match="require batch mode"):
        recipe.build_projection_executor(Graph(), run_mode="inprocess", **options)
    with pytest.raises(ValueError, match="require batch mode"):
        recipe.run_nrl_graph([], run_mode="inprocess", **options)


def test_local_parse_resolution_fails_closed_without_gpu(recipe: ModuleType) -> None:
    from nemo_retriever.common.ray_resource_hueristics import Resources

    with pytest.raises(RuntimeError, match="local GPU actor"):
        recipe._require_local_parse_resolution(Resources(cpu_count=8, gpu_count=0))

    recipe._require_local_parse_resolution(Resources(cpu_count=8, gpu_count=1))


def test_batch_executor_reuses_validated_gpu_resource_snapshot(recipe: ModuleType) -> None:
    from nemo_retriever.common.ray_resource_hueristics import Resources

    resources = Resources(cpu_count=8, gpu_count=1)
    executor = SimpleNamespace(_ray_address=None, _preflight_cluster_resources=resources)

    recipe._prepare_executor_for_local_parse(executor, run_mode="batch")

    assert executor._preflight_cluster_resources is resources


@pytest.mark.parametrize(("parse_batch_size", "parse_cpus"), [(64, 1), (128, 4)])
@pytest.mark.parametrize("projection_block_rows", [None, 16])
def test_run_nrl_graph_validates_executor_result(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    parse_batch_size: int,
    parse_cpus: int,
    projection_block_rows: int | None,
) -> None:
    captured: dict[str, Any] = {}
    valid = recipe.project_nrl_pages(pd.DataFrame([_page("")]))
    graph = object()

    class FakeExecutor:
        def ingest(self, paths: list[str]) -> pd.DataFrame:
            captured["paths"] = paths
            return valid

    monkeypatch.setattr(recipe, "build_projection_graph", lambda: graph)
    monkeypatch.setattr(recipe, "_prepare_executor_for_local_parse", lambda *_args, **_kwargs: None)

    def fake_executor(  # noqa: PLR0913
        received_graph: Any,
        *,
        run_mode: str,
        projection_workers: int,
        projection_block_rows: int | None,
        parse_batch_size: int,
        parse_cpus: int,
    ) -> FakeExecutor:
        captured.update(
            graph=received_graph,
            run_mode=run_mode,
            projection_workers=projection_workers,
            projection_block_rows=projection_block_rows,
            parse_batch_size=parse_batch_size,
            parse_cpus=parse_cpus,
        )
        return FakeExecutor()

    monkeypatch.setattr(recipe, "build_projection_executor", fake_executor)

    result = recipe.run_nrl_graph(
        Path("/raid/input/example.pdf"),
        projection_workers=3,
        projection_block_rows=projection_block_rows,
        parse_batch_size=parse_batch_size,
        parse_cpus=parse_cpus,
    )

    assert result.equals(valid)
    assert captured == {
        "graph": graph,
        "run_mode": "batch",
        "projection_workers": 3,
        "projection_block_rows": projection_block_rows,
        "parse_batch_size": parse_batch_size,
        "parse_cpus": parse_cpus,
        "paths": ["/raid/input/example.pdf"],
    }


def test_empty_run_still_validates_execution_options(recipe: ModuleType) -> None:
    with pytest.raises(ValueError, match="run_mode"):
        recipe.run_nrl_graph([], run_mode="unsupported")
    with pytest.raises(ValueError, match="1 through 8"):
        recipe.run_nrl_graph([], projection_workers=0)
    with pytest.raises(ValueError, match="statistics require batch mode"):
        recipe.run_nrl_graph([], run_mode="inprocess", executor_stats_path="/raid/stats.txt")


def test_real_cpu_ray_projection_repartition_preserves_heterogeneous_pages(
    recipe: ModuleType,
) -> None:
    import importlib

    import ray
    from nemo_retriever.common.ray_runtime import build_local_ray_runtime_env
    from nemo_retriever.graph.executor import ray_dataset_to_pandas
    from nemo_retriever.graph.pipeline_graph import Graph
    from ray.data import DataContext
    from ray.data._internal.logical.operators.map_operator import StreamingRepartition

    if ray.is_initialized():
        pytest.skip("requires an isolated CPU-only Ray runtime")
    # Workers must import the canonical module, not this test's file-loaded alias.
    worker_recipe = importlib.import_module("nrl_graph")
    pages = [_page(_raw_output(), page_number=index + 1) for index in range(34)]
    for index, page in enumerate(pages):
        page["table"] = [{"text": None, "bbox": [0.1, 0.2, 0.3, 0.4]}] if index % 2 else []
        if index % 4 == 1:
            page["nemotron_parse_v1_2"]["raw_output"] = ""
            page["metadata"] = None
        elif index % 4 == 2:
            page["metadata"]["error"] = {"stage": "fixture", "message": "controlled page failure"}
        elif index % 4 == 3:
            page["nemotron_parse_v1_2"]["raw_output"] = None
            page["page_image"] = None
    frame = pd.DataFrame(pages, dtype=object)
    expected = recipe.project_nrl_pages(frame)
    keys = ["source_path", "native_page_number", "record_type", "element_index"]
    expected = expected.sort_values(keys, na_position="first").reset_index(drop=True)
    markers = expected[expected["record_type"] == "page_outcome"]
    assert set(markers["page_outcome"]) == {"parsed", "empty", "failed"}
    assert expected[expected["element_class"] == "Picture"]["binary_content"].map(bool).all()

    context = DataContext.get_current()
    original = {
        name: getattr(context, name)
        for name in (
            "batch_to_block_arrow_format",
            "enable_tensor_extension_casting",
            "enable_rich_progress_bars",
            "use_ray_tqdm",
        )
    }
    with TemporaryDirectory(prefix="rp-") as ray_tmp:
        try:
            ray.init(
                address="local",
                num_cpus=4,
                num_gpus=0,
                include_dashboard=False,
                object_store_memory=128 * 1024 * 1024,
                _temp_dir=ray_tmp,
                runtime_env=build_local_ray_runtime_env(),
            )
            for block_rows in (None, 16):
                source = ray.data.from_pandas(frame, override_num_blocks=1)
                graph = Graph() >> worker_recipe.NRLCuratorProjectionOperator()
                executor = worker_recipe.build_projection_executor(
                    graph,
                    projection_workers=2,
                    projection_block_rows=block_rows,
                )
                dataset = executor.build_dataset(source)
                partitions = [
                    node
                    for node in dataset._logical_plan.dag.post_order_iter()
                    if isinstance(node, StreamingRepartition)
                ]
                assert [node.target_num_rows_per_block for node in partitions] == ([] if block_rows is None else [16])
                assert dataset.context.batch_to_block_arrow_format is False
                assert dataset.context.enable_tensor_extension_casting is False
                actual = worker_recipe.validate_projection_envelope(ray_dataset_to_pandas(dataset))
                actual = actual.sort_values(keys, na_position="first").reset_index(drop=True)
                pd.testing.assert_frame_equal(actual, expected)
        finally:
            ray.shutdown()
            for name, value in original.items():
                setattr(context, name, value)
    assert not ray.is_initialized()


@pytest.mark.parametrize(("evidence", "save_stats"), [(False, True), (True, False), (True, True)])
def test_executor_statistics_reuse_one_materialization_independent_of_capture(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    evidence: bool,
    save_stats: bool,
) -> None:
    import nemo_retriever.graph.executor as executor_module
    from ray.data._internal.compute import ActorPoolStrategy
    from ray.data._internal.logical.operators.input_data_operator import InputData
    from ray.data._internal.logical.operators.map_operator import MapBatches

    expected = recipe.project_nrl_pages(pd.DataFrame([_page(_raw_output())]))
    calls: list[str] = []
    graph_kwargs: dict[str, Any] = {}
    evidence_path = tmp_path / "evidence" if evidence else None
    stats_path = tmp_path / "run" / "executor_stats.txt" if save_stats else None
    node = MapBatches(
        fn=recipe.NRLCuratorProjectionOperator,
        input_dependencies=[InputData(input_data=[])],
        batch_size=7,
        batch_format="pandas",
        compute=ActorPoolStrategy(size=3),
        ray_remote_args={"num_cpus": 1, "num_gpus": 0},
    )

    def stats() -> str:
        assert calls == ["build", "materialize"]
        calls.append("stats")
        return "already-executed Ray statistics"

    dataset = SimpleNamespace(
        _logical_plan=SimpleNamespace(dag=node),
        stats=stats,
    )

    class FakeExecutor:
        def build_dataset(self, paths: list[str]) -> Any:
            assert paths == ["/raid/input/example.pdf"]
            calls.append("build")
            return dataset

        def ingest(self, _paths: list[str]) -> Any:
            pytest.fail("statistics must not trigger a second execution")

    def materialize(received: Any) -> pd.DataFrame:
        assert received is dataset
        calls.append("materialize")
        return expected

    def build_graph(**kwargs: Any) -> object:
        graph_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(recipe, "build_projection_graph", build_graph)
    monkeypatch.setattr(recipe, "build_projection_executor", lambda *_args, **_kwargs: FakeExecutor())
    monkeypatch.setattr(recipe, "_prepare_executor_for_local_parse", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_module, "ray_dataset_to_pandas", materialize)

    result = recipe.run_nrl_graph(
        "/raid/input/example.pdf",
        evidence_root=str(evidence_path) if evidence_path else None,
        executor_stats_path=stats_path,
    )

    pd.testing.assert_frame_equal(result, expected)
    assert calls == ["build", "materialize", "stats"]
    assert graph_kwargs == ({"evidence_root": str(evidence_path)} if evidence_path else {})
    output_paths = ([stats_path] if stats_path else []) + (
        [evidence_path / "executor_stats.txt"] if evidence_path else []
    )
    for path in output_paths:
        settings, timings = path.read_text(encoding="utf-8").split("\n\n", 1)
        diagnostics = json.loads(settings)
        assert diagnostics["compact_result_pandas_estimate_bytes"] == int(expected.memory_usage(deep=True).sum())
        recorded = diagnostics["resolved_map_batches"][0]
        assert recorded["batch_size"] == 7
        assert recorded["batch_format"] == "pandas"
        assert recorded["compute"] == repr(node.compute)
        assert recorded["num_cpus"] == 1
        assert recorded["num_gpus"] == 0
        assert timings == "already-executed Ray statistics"
    assert sorted(path for path in tmp_path.rglob("*") if path.is_file()) == sorted(output_paths)
