# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one Parse 3.5 Page64 shard using a job-owned Ray cluster."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import ray
from ray.data import DataContext, ExecutionResources

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.composite import NemotronParsePDFReader
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import output_version
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import PDFPageBudgetValidationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.utils import TaskPerfUtils

from .shards import run_shard


def run(args: argparse.Namespace) -> None:
    from experiments.local_parse import parent_priority

    if os.environ.get("PARSE35_PERSISTENT_FITTED_RENDERER") == "1" and (
        os.environ.get("CURATOR_STUDY_RASTER") != "matched"
        or os.environ.get("PARSE35_EARLY_ADMISSION") != "1"
    ):
        raise ValueError("The persistent fitted renderer requires matched-raster E03 HTTP extraction")
    profile = json.loads(args.profile.read_text())
    extraction = json.loads(args.extraction_config.read_text())
    if "http_max_retries" in profile:
        extraction["transport"]["max_retries"] = profile["http_max_retries"]
    version = output_version({"pdf_extraction_config": extraction})
    manifest_path = args.manifest.resolve(strict=True)

    def attempt(attempt_dir: Path) -> list[str]:
        reader = NemotronParsePDFReader(
            manifest_path=str(manifest_path),
            pdf_dir=str(args.pdf_dir),
            model_path=profile["model_path"],
            parse_mode="single",
            pages_per_task=64,
            page_budget_extraction_config=extraction,
            dpi=extraction["render"]["dpi"],
            max_pages=0,
            inference_batch_size=profile["http_concurrency"],
            inference_server_endpoint=args.endpoint,
            inference_server_model_name=extraction["request"]["model"],
            inference_server_client_num_workers=profile["http_workers"],
            inference_server_request_timeout_s=extraction["transport"]["request_timeout_s"],
            inference_server_max_retries=extraction["transport"]["max_retries"],
            inference_server_reuse_encoded_images=True,
        )
        reader.with_({"pdf_preprocess": {"resources": Resources(cpus=1.0)}})
        reader._preprocessor._RENDER_TIMEOUT_S = extraction["render"]["timeout_s"]
        reader._inference = reader._inference.with_(resources=Resources(cpus=1.0))

        pipeline = Pipeline(name="local-parse-poc")
        pipeline.add_stage(reader)
        pipeline.add_stage(
            InterleavedParquetWriterStage(
                path=str(attempt_dir / "parquet"),
                materialize_on_write=False,
                on_materialize_error="warn",
                schema=PAGE_OUTPUT_SCHEMA,
                mode="error",
                atomic_local_write=True,
                reject_unknown_columns=True,
                use_output_file_id=True,
            )
        )
        pipeline.add_stage(
            PDFPageBudgetValidationStage(manifest_path=str(manifest_path), expected_output_version=version)
        )
        source = Path(__file__).resolve().parents[2]
        study_env = {name: os.environ[name] for name in (
            "PARSE35_TRACE_DIR", "PARSE35_TRACE_BACKEND", "PARSE35_HTTP_READY_FILE", "PARSE35_HTTP_READY_DEADLINE_UNIX",
            "CURATOR_STUDY_RASTER", "PARSE35_EARLY_ADMISSION", "PARSE35_EARLY_BLOCK_PAGES",
            "PARSE35_EARLY_BLOCK_BYTES", "PARSE35_EARLY_INPUT_BYTES", "PARSE35_EARLY_PARENT_INPUT_BYTES",
            "PARSE35_EARLY_HELD_BYTES", "PARSE35_PERSISTENT_FITTED_RENDERER", *parent_priority.ENVIRONMENT,
        ) if name in os.environ}
        ray.init(
            address=args.ray_address,
            runtime_env={
                "py_executable": sys.executable,
                "env_vars": {
                    "PYTHONPATH": os.pathsep.join((str(source), os.environ.get("PYTHONPATH", ""))).rstrip(os.pathsep),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "",
                    **study_env,
                },
            },
        )
        try:
            context = DataContext.get_current()
            context.max_tasks_in_flight_per_actor = 2
            context.execution_options.resource_limits = ExecutionResources.for_limits(
                cpu=profile["ray_data_cpus"],
                gpu=0,
                object_store_memory=profile["ray_data_object_store_bytes"],
            )
            context.enable_progress_bars = False
            started = time.monotonic()
            tasks = pipeline.run(RayDataExecutor())
            (attempt_dir / "pipeline.json").write_text(
                json.dumps(
                    {
                        "seconds": time.monotonic() - started,
                        "metrics": TaskPerfUtils.aggregate_task_metrics(tasks, prefix="task"),
                    },
                    default=str,
                )
                + "\n"
            )
            return sorted(str(path) for path in (attempt_dir / "parquet").glob("*.parquet"))
        finally:
            ray.shutdown()

    started = time.monotonic()
    result = run_shard(
        shard_id=args.shard_id,
        output_root=args.output_root,
        manifest_path=manifest_path,
        expected_output_version=version,
        run_attempt=attempt,
    )
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "attempt": str(result.attempt_dir),
                "validation": result.validation,
                "execution_and_readback_seconds": elapsed,
                "resumed": result.resumed,
            },
            default=str,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("http",))
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extraction-config", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-id", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
