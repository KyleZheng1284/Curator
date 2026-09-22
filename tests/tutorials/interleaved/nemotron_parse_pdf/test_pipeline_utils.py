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

# ruff: noqa: INP001

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import argparse
    from types import ModuleType


@pytest.fixture(scope="module")
def recipe() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "tutorials/interleaved/nemotron_parse_pdf/pipeline_utils.py"
    spec = importlib.util.spec_from_file_location("curator_tutorial_pipeline_utils", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(recipe: ModuleType, tmp_path: Path, *extra: str) -> argparse.Namespace:
    return recipe.create_nemotron_parse_pdf_argparser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--pdf-dir",
            str(tmp_path / "pdfs"),
            "--output-dir",
            str(tmp_path / "output"),
            *extra,
        ]
    )


@pytest.mark.parametrize("max_tokens", [None, 9000])
def test_native_pipeline_preserves_token_configuration(
    recipe: ModuleType, tmp_path: Path, max_tokens: int | None
) -> None:
    from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
    from nemo_curator.stages.interleaved.pdf.nemotron_parse import NemotronParsePDFReader
    from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import DEFAULT_MAX_TOKENS

    args = _args(recipe, tmp_path, *([] if max_tokens is None else ["--max-tokens", str(max_tokens)]))
    pipeline = recipe.create_nemotron_parse_pdf_pipeline(args)

    reader, writer = pipeline.stages
    assert isinstance(reader, NemotronParsePDFReader)
    assert isinstance(writer, InterleavedParquetWriterStage)
    assert reader.max_tokens == (DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens)
    assert reader.decompose()[2].max_tokens == reader.max_tokens
    assert writer.materialize_on_write is False


def test_comparison_explicitly_enables_non_dropping_pixel_validation(recipe: ModuleType, tmp_path: Path) -> None:
    from nemo_curator.stages.interleaved import InterleavedAspectRatioFilterStage
    from nemo_curator.stages.interleaved.filter.blur_filter import InterleavedBlurFilterStage
    from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
    from nemo_curator.stages.interleaved.pdf.nemotron_parse import NemotronParsePDFReader

    pipeline = recipe.create_nemotron_parse_pdf_pipeline(
        _args(recipe, tmp_path, "--max-tokens", "9000"), validate_images=True
    )

    reader, aspect, blur, writer = pipeline.stages
    assert isinstance(reader, NemotronParsePDFReader)
    assert isinstance(aspect, InterleavedAspectRatioFilterStage)
    assert isinstance(blur, InterleavedBlurFilterStage)
    assert isinstance(writer, InterleavedParquetWriterStage)
    assert reader.max_tokens == 9000
    assert aspect.min_aspect_ratio == 0.0
    assert aspect.max_aspect_ratio == float("inf")
    assert aspect.drop_invalid_rows is False
    assert aspect.preserve_metadata_only_samples is True
    assert blur.score_threshold == 0.0
    assert blur.drop_invalid_rows is False
    assert blur.preserve_metadata_only_samples is True
    assert writer.materialize_on_write is False


def test_default_pipeline_builds_without_opencv(recipe: ModuleType, tmp_path: Path) -> None:
    script = textwrap.dedent(
        """\
        import runpy
        import sys

        sys.modules["cv2"] = None
        recipe = runpy.run_path(sys.argv[1])
        args = recipe["create_nemotron_parse_pdf_argparser"]().parse_args([
            "--manifest", sys.argv[2], "--pdf-dir", sys.argv[3], "--output-dir", sys.argv[4],
        ])
        pipeline = recipe["create_nemotron_parse_pdf_pipeline"](args)
        pipeline.build()
        assert "nemo_curator.stages.interleaved.filter.blur_filter" not in sys.modules
        assert sys.modules["cv2"] is None
        """
    )
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            script,
            recipe.__file__,
            str(tmp_path / "manifest.jsonl"),
            str(tmp_path / "pdfs"),
            str(tmp_path / "output"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
