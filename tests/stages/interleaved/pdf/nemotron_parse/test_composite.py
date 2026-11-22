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

"""Validate opt-in served reader configuration."""

import base64
import io
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import inference
from nemo_curator.stages.interleaved.pdf.nemotron_parse._early_admission import EarlyAdmissionStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.composite import NemotronParsePDFReader
from nemo_curator.stages.interleaved.pdf.nemotron_parse.extraction import parser_package_identity
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import DEFAULT_MODEL_PATH
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import output_version
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import PDFPageBudgetValidationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.utils import TaskPerfUtils
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_EXTRACTION_CONFIG, write_manifest
from tests.stages.interleaved.pdf.nemotron_parse.test_inference import http_endpoint  # noqa: F401 - pytest fixture


def test_served_options_do_not_rebind_existing_positional_endpoint():
    reader = NemotronParsePDFReader(
        "input.jsonl",
        None,
        "pdfs",
        None,
        "legacy-model",
        "vllm",
        10,
        None,
        300,
        50,
        4,
        64,
        8192,
        False,
        False,
        10,
        "pdf_dataset",
        "file_name",
        "cc_pdf_file_names",
        "url",
        "http://unused/v1",
        "served-model",
        8,
    )
    assert reader.inference_server_endpoint == "http://unused/v1"
    assert reader._inference.model_name == "served-model"
    assert reader._inference.num_workers() == 8
    assert reader.parse_mode is None


def test_parse_mode_requires_endpoint():
    with pytest.raises(ValueError, match="requires inference_server_endpoint"):
        NemotronParsePDFReader(manifest_path="input.jsonl", parse_mode="single")


def test_served_modes_keep_default_model_and_client_workers():
    reader = NemotronParsePDFReader(
        manifest_path="input.jsonl",
        parse_mode="triple",
        triple_regions="detector",
        inference_server_endpoint="http://unused/v1",
        inference_server_client_num_workers=8,
    )
    assert reader.model_path == DEFAULT_MODEL_PATH
    assert reader._inference.model_name == DEFAULT_MODEL_PATH
    assert reader._inference.parse_mode == "triple"
    assert reader._inference.triple_regions == "detector"
    assert reader._preprocessor.retain_failures
    assert reader._inference.ray_stage_spec() == {"is_actor_stage": True}
    legacy = NemotronParsePDFReader(manifest_path="input.jsonl", inference_server_endpoint="http://unused/v1")
    assert not legacy._preprocessor.retain_failures
    assert legacy._inference.ray_stage_spec() == {"is_actor_stage": False}


def test_encoded_image_reuse_wires_both_stages_without_changing_execution_controls():
    settings = {
        "manifest_path": "input.jsonl",
        "parse_mode": "single",
        "inference_server_endpoint": "http://unused/v1",
    }
    control = NemotronParsePDFReader(**settings)
    candidate = NemotronParsePDFReader(**settings, inference_server_reuse_encoded_images=True)
    assert not control._preprocessor.reuse_encoded_images
    assert not control._inference.reuse_encoded_images
    assert candidate._preprocessor.reuse_encoded_images
    assert candidate._inference.reuse_encoded_images
    assert candidate._inference.inference_batch_size == control._inference.inference_batch_size
    assert candidate._inference.num_workers() == control._inference.num_workers()


@pytest.mark.parametrize("mode", [None, "double", "triple"])
def test_encoded_image_reuse_requires_served_single_mode(mode: str | None):
    with pytest.raises(ValueError, match=r"requires served Parse 3\.5 single mode"):
        NemotronParsePDFReader(
            manifest_path="input.jsonl",
            parse_mode=mode,
            inference_server_endpoint="http://unused/v1",
            inference_server_reuse_encoded_images=True,
        )


def test_encoded_image_reuse_requires_endpoint_and_boolean():
    with pytest.raises(ValueError, match=r"requires served Parse 3\.5 single mode"):
        NemotronParsePDFReader(
            manifest_path="input.jsonl", parse_mode="single", inference_server_reuse_encoded_images=True
        )
    with pytest.raises(TypeError, match="must be a boolean"):
        NemotronParsePDFReader(manifest_path="input.jsonl", inference_server_reuse_encoded_images=1)


@pytest.mark.parametrize("mode", [None, "double", "triple"])
def test_page_budget_rejects_untested_parser_modes(mode: str | None):
    with pytest.raises(ValueError, match="single mode"):
        NemotronParsePDFReader(
            manifest_path="input.jsonl",
            parse_mode=mode,
            pages_per_task=64,
            inference_server_endpoint="http://unused/v1",
        )


def test_page_budget_keeps_http_capacity_and_adds_nonresumable_merge():
    reader = NemotronParsePDFReader(
        manifest_path="input.jsonl",
        parse_mode="single",
        pages_per_task=64,
        page_budget_extraction_config=TEST_EXTRACTION_CONFIG,
        inference_server_endpoint="http://unused/v1",
        inference_batch_size=64,
        inference_server_client_num_workers=36,
    )
    assert reader._inference.inference_batch_size == 64
    assert reader._inference.num_workers() == 36
    assert reader._partitioner.pages_per_task == 64
    # Document grouping handles page chunks without an intermediate repartition.
    assert reader._postprocessor.ray_stage_spec() == {"is_fanout_stage": False}
    assert not reader.decompose()[-1].is_resumable


def test_legacy_postprocess_graph_does_not_gain_a_fanout_boundary():
    reader = NemotronParsePDFReader(manifest_path="input.jsonl")
    assert reader._postprocessor.ray_stage_spec() == {"is_fanout_stage": False}
    assert "run_on_driver" not in reader.decompose_and_apply_with()[0].ray_stage_spec()


def test_ignored_legacy_controls_do_not_change_page_mode_output_version():
    settings = {
        "manifest_path": "input.jsonl",
        "parse_mode": "single",
        "pages_per_task": 64,
        "page_budget_extraction_config": TEST_EXTRACTION_CONFIG,
        "inference_server_endpoint": "http://unused/v1",
    }
    first = NemotronParsePDFReader(**settings, pdfs_per_task=1, max_pages=1)
    second = NemotronParsePDFReader(**settings, pdfs_per_task=100, max_pages=500)
    assert first._partitioner._parser_config == second._partitioner._parser_config


def test_page_budget_rejects_other_executors_before_processing():
    pipeline = Pipeline(name="wrong_backend")
    pipeline.add_stage(
        NemotronParsePDFReader(
            manifest_path="not-opened.jsonl",
            parse_mode="single",
            pages_per_task=64,
            page_budget_extraction_config=TEST_EXTRACTION_CONFIG,
            inference_server_endpoint="http://unused/v1",
        )
    )
    with pytest.raises(ValueError, match="RayDataExecutor"):
        pipeline.run(executor=object())


def test_page_budget_does_not_rebind_legacy_positional_reader_arguments():
    reader = NemotronParsePDFReader("input.jsonl", None, "pdfs", None, "legacy-model", "vllm", 8, 17)
    assert reader.max_pdfs == reader._partitioner.max_pdfs == 17
    assert reader.pdfs_per_task == 8
    assert reader.pages_per_task is None


def test_overlap_keeps_page_tasking_and_inference_worker_settings() -> None:
    config = {
        **TEST_EXTRACTION_CONFIG,
        "render": {**TEST_EXTRACTION_CONFIG["render"], "raster_max_size": [1024, 1024]},
    }
    reader = NemotronParsePDFReader(
        manifest_path="input.jsonl",
        parse_mode="single",
        inference_server_endpoint="http://unused/v1",
        pages_per_task=64,
        page_budget_extraction_config=config,
        raster_max_size=(1024, 1024),
        persistent_rendering=True,
        overlap_rendering=True,
    )
    reader.with_(
        {
            reader._preprocessor.name: {"resources": Resources(cpus=3.0)},
            reader._inference.name: {"num_workers": 5, "batch_size": 2, "resources": Resources(cpus=2.0)},
        }
    )

    partitioner, overlap, postprocess, merge = reader.decompose_and_apply_with()

    assert partitioner is reader._partitioner
    assert postprocess is reader._postprocessor
    assert merge is reader._merger
    assert isinstance(overlap, EarlyAdmissionStage)
    assert overlap.ray_stage_spec()["is_actor_stage"]
    assert overlap.num_workers() == 5
    assert overlap.batch_size == 2
    assert overlap.resources.cpus == 5
    assert overlap.renderer.raster_max_size == (1024, 1024)
    assert overlap.renderer.persistent_rendering


def test_custom_raster_cap_requires_matching_extraction_identity() -> None:
    with pytest.raises(ValueError, match="does not match rendering settings"):
        NemotronParsePDFReader(
            manifest_path="input.jsonl",
            parse_mode="single",
            inference_server_endpoint="http://unused/v1",
            pages_per_task=64,
            page_budget_extraction_config=TEST_EXTRACTION_CONFIG,
            raster_max_size=(1024, 1024),
        )


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_overlap_publishes_the_same_pages_as_separate_stages(
    tmp_path: Path,
    parse_package: ModuleType,
    http_endpoint: SimpleNamespace,  # noqa: F811
) -> None:
    pytest.importorskip("pypdfium2")
    pages = {"short.pdf": 3, "long.pdf": 9, "absent.pdf": 2, "refused.pdf": 4}
    # Every page is one solid colour, so a request and its answer name the page they belong to.
    cells = {
        (name, page): (index + 1, page % 6 + 1, page // 6 + 1)
        for index, (name, count) in enumerate(pages.items())
        for page in range(count)
    }
    failures = {
        ("absent.pdf", 0): "PDFError",
        ("absent.pdf", 1): "PDFError",
        ("refused.pdf", 2): "BadRequestError",
    }
    served = []

    def respond(image_url: str) -> tuple[int, str]:
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
            cell = tuple(round(value / 32) for value in image.convert("RGB").getpixel((0, 0)))
        served.append(cell)
        # A parent stays open long enough for the actor to render the task after it.
        time.sleep(0.3)
        status = 400 if cell == cells["refused.pdf", 2] else 200
        return status, "<x_0.1><y_0.1>CELL-{}-{}-{}<x_0.9><y_0.9><class_Text>".format(*cell)

    http_endpoint.respond = respond
    for name, count in pages.items():
        if name != "absent.pdf":
            images = [
                Image.new("RGB", (200, 260), tuple(32 * value for value in cells[name, page])) for page in range(count)
            ]
            images[0].save(tmp_path / name, save_all=True, append_images=images[1:])
    manifest = write_manifest(tmp_path, [{"file_name": name, "num_pages": count} for name, count in pages.items()])
    generation = inference._server_generation_config(
        {**asdict(parse_package.GenerationSettings()), "skip_special_tokens": False}
    )
    config = {
        **TEST_EXTRACTION_CONFIG,
        "parser": parser_package_identity(),
        "request": {
            "model": "served",
            "stage": "fullpage",
            "prompt": parse_package.PROMPT,
            "generation": asdict(generation),
        },
    }
    published, requests, rendered_ahead = {}, {}, {}
    for arm, options in {
        "separate": {},
        "overlap": {"overlap_rendering": True},
        "streaming": {"overlap_rendering": True, "stream_page_tasks": True},
        "streaming_persistent_reuse": {
            "stream_page_tasks": True,
            "overlap_rendering": True,
            "persistent_rendering": True,
            "inference_server_reuse_encoded_images": True,
        },
    }.items():
        start = len(http_endpoint.requests)
        reader = NemotronParsePDFReader(
            manifest_path=str(manifest),
            pdf_dir=str(tmp_path),
            parse_mode="single",
            pages_per_task=4,
            page_budget_extraction_config=config,
            inference_server_endpoint=http_endpoint.endpoint,
            inference_server_model_name="served",
            inference_server_client_num_workers=1,
            **options,
        )
        # Ray workers import the parser package from the driver's path.
        overrides = {"runtime_env": {"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}}}
        reader.with_({reader._inference.name: overrides})
        pipeline = Pipeline(
            name=arm,
            stages=[
                reader,
                InterleavedParquetWriterStage(
                    path=str(tmp_path / arm),
                    materialize_on_write=False,
                    on_materialize_error="warn",
                    schema=PAGE_OUTPUT_SCHEMA,
                    reject_unknown_columns=True,
                    atomic_local_write=True,
                    use_output_file_id=True,
                    mode="error",
                ),
                PDFPageBudgetValidationStage(
                    manifest_path=str(manifest),
                    expected_output_version=output_version({"pdf_extraction_config": config}),
                    stream_page_tasks=reader.stream_page_tasks,
                ),
            ],
        )
        tasks = pipeline.run(RayDataExecutor())
        assert any(isinstance(stage, EarlyAdmissionStage) for stage in pipeline.stages) is bool(options)
        validation = tasks[0]._metadata["pdf_page_budget_validation"]
        assert (validation["documents"], validation["successful_pages"], validation["failed_pages"]) == (4, 15, 3)
        rows = [row for task in tasks for path in task.data for row in pq.read_table(path).to_pylist()]
        page_rows = {(row["pdf_name"], row["page_number"]): row for row in rows if row["page_number"] is not None}
        assert len(page_rows) == sum(row["page_number"] is not None for row in rows) == len(cells)
        for page, row in page_rows.items():
            if page in failures:
                assert json.loads(row["materialize_error"])["type"] == failures[page]
            else:
                assert not row["materialize_error"]
                assert "CELL-{}-{}-{}".format(*cells[page]) in row["text_content"]
        assert Counter(served[start:]) == Counter(cell for page, cell in cells.items() if page[0] != "absent.pdf")
        requests[arm] = Counter(json.dumps(body, sort_keys=True) for body in http_endpoint.requests[start:])
        published[arm] = sorted(rows, key=lambda row: (row["pdf_name"], row["position"]))
        rendered_ahead[arm] = TaskPerfUtils.aggregate_task_metrics(tasks).get(
            f"{reader._inference.name}_custom.render_ahead_pages_sum"
        )
    request = http_endpoint.requests[0]
    assert request["model"] == "served"
    assert request["messages"][0]["content"][0]["text"] == parse_package.PROMPT
    for arm in ("overlap", "streaming", "streaming_persistent_reuse"):
        assert requests[arm] == requests["separate"]
        assert published[arm] == published["separate"]
    # Ray Data hands the actor its next task early only when the stage asks for it.
    assert rendered_ahead["separate"] is None
    assert not rendered_ahead["overlap"]
    assert min(rendered_ahead["streaming"], rendered_ahead["streaming_persistent_reuse"]) > 0
