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

# Dynamic optional-runtime tests intentionally use private recipe seams.
# ruff: noqa: ANN401, INP001

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_recipe() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "tutorials" / "interleaved" / "nemotron_parse_pdf" / "nrl_compare.py"
    spec = importlib.util.spec_from_file_location("curator_tutorial_nrl_compare", path)
    if spec is None or spec.loader is None:
        message = f"Could not load recipe at {path}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recipe() -> ModuleType:
    return _load_recipe()


@lru_cache(maxsize=1)
def _page_png() -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    image = image_module.new("RGB", (1664, 2048), color=(250, 250, 250))
    draw_module.Draw(image).rectangle((332, 410, 664, 820), fill=(20, 100, 220))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _raw_output() -> str:
    return (
        "<x_0.05><y_0.05>Heading<x_0.4><y_0.10><class_Text>"
        "<x_0.05><y_0.12>| A | B |\n| --- | --- |\n| 1 | 2 |<x_0.5><y_0.19><class_Table>"
        "<x_0.20><y_0.20><x_0.40><y_0.40><class_Picture>"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _write_evidence(  # noqa: PLR0913
    recipe: ModuleType,
    root: Path,
    *,
    raw_output: str | None = None,
    valid_blank_pages: list[int] | None = None,
    min_crop_px: int = 10,
    model_revision: str = "test-revision",
) -> Path:
    raw_output = _raw_output() if raw_output is None else raw_output
    image_bytes = _page_png()
    raw_bytes = raw_output.encode()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    core = {
        "schema": recipe.EVIDENCE_SCHEMA,
        "schema_version": recipe.EVIDENCE_SCHEMA_VERSION,
        "document": {
            "content_sha256": "a" * 64,
            "page_count": 1,
            "valid_blank_pages": valid_blank_pages or [],
        },
        "nemotron_parse": {
            "model_id": recipe.PARSE_MODEL,
            "model_revision": model_revision,
            "task_prompt": recipe.PARSE_TASK_PROMPT,
            "max_tokens": 9000,
            "decoding": {"temperature": 0, "top_k": 1, "repetition_penalty": 1.1},
            "render": {
                "dpi": 200,
                "image_format": "png",
                "jpeg_quality": 100,
                "render_mode": "full_dpi",
            },
            "proc_size": list(recipe.PARSE_PROC_SIZE),
            "min_crop_px": min_crop_px,
        },
        "pages": [
            {
                "page_number": 0,
                "native_page_number": 1,
                "page_image": {
                    "sha256": image_sha,
                    "byte_length": len(image_bytes),
                    "format": "png",
                    "orig_shape_hw": [2048, 1664],
                },
                "raw_output": {
                    "sha256": raw_sha,
                    "byte_length": len(raw_bytes),
                    "encoding": "utf-8",
                },
                "error_statuses": {"extraction": "ok", "nemotron_parse": "ok"},
                "finish_reason": "stop",
            }
        ],
    }
    core_sha = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    evidence_dir = root / core_sha
    blobs = evidence_dir / "blobs"
    blobs.mkdir(parents=True)
    (blobs / image_sha).write_bytes(image_bytes)
    (blobs / raw_sha).write_bytes(raw_bytes)
    manifest = {
        "core_sha256": core_sha,
        "core": core,
        "provenance": {"captured_by": "test fixture"},
    }
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _picture_snapshot(recipe: ModuleType) -> dict[str, Any]:
    image = _page_png()
    return recipe._decode_png(image, "test picture")


def _sealed_replay(
    recipe: ModuleType,
    evidence: Any,
    engine: str,
    *,
    element_class: str = "Picture",
    picture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    element = {
        "element_index": 0,
        "element_class": element_class,
        "modality": "image" if element_class == "Picture" else "text",
        "content_type": "image/png" if element_class == "Picture" else "text/markdown",
        "normalized_text": "",
        "bbox_xyxy_norm": [0.2, 0.2, 0.4, 0.4],
        "picture": picture if element_class == "Picture" else None,
    }
    return recipe._seal_replay(
        engine,
        evidence,
        [
            {
                "page_number": 0,
                "native_page_number": 1,
                "raw_output_sha256": evidence.pages[0].raw_output_sha256,
                "elements": [element],
            }
        ],
    )


def test_load_evidence_verifies_content_addressed_manifest(recipe: ModuleType, tmp_path: Path) -> None:
    manifest = _write_evidence(recipe, tmp_path)

    evidence = recipe.load_evidence(manifest)

    assert evidence.manifest_path == manifest.resolve()
    assert evidence.core_sha256 == manifest.parent.name
    assert evidence.document_sha256 == "a" * 64
    assert evidence.proc_size == (2048, 1664)
    assert evidence.pages[0].raw_output == _raw_output()
    assert evidence.pages[0].orig_shape_hw == (2048, 1664)


def test_capture_roundtrip_is_immutable_and_idempotent(recipe: ModuleType, tmp_path: Path) -> None:
    page_kwargs = {
        "source_path": "/raid/cohort/input.pdf",
        "native_page_number": 1,
        "image_bytes": _page_png(),
        "raw_output": _raw_output(),
    }
    first = recipe.capture_page(tmp_path, **page_kwargs)
    assert recipe.capture_page(tmp_path, **page_kwargs) == first
    with pytest.raises(recipe.EvidenceError, match="conflicting immutable"):
        recipe.capture_page(tmp_path, **dict(page_kwargs, raw_output="changed"))
    manifest = recipe.finalize_capture(
        tmp_path,
        source_path=page_kwargs["source_path"],
        document_sha256="b" * 64,
        expected_page_count=1,
        valid_blank_pages=[],
        model_revision="pinned-test-revision",
        provenance={"test": True},
    )
    evidence = recipe.load_evidence(manifest)
    assert evidence.pages[0].raw_output == _raw_output()
    assert evidence.pages[0].image_bytes == _page_png()
    assert evidence.document_sha256 == "b" * 64


@pytest.mark.parametrize("failure", ["missing", "extra", "nonstop", "malformed", "nested_error"])
def test_incomplete_capture_never_publishes_manifest(recipe: ModuleType, tmp_path: Path, failure: str) -> None:
    if failure != "missing":
        recipe.capture_page(
            tmp_path,
            source_path="/raid/cohort/input.pdf",
            native_page_number=2 if failure == "extra" else 1,
            image_bytes=_page_png(),
            raw_output=_raw_output() + ("truncated" if failure == "malformed" else ""),
            finish_reason="length" if failure == "nonstop" else "stop",
            parse_error={"type": "ParseFailure"} if failure == "nested_error" else None,
        )
    with pytest.raises(recipe.EvidenceError):
        recipe.finalize_capture(
            tmp_path,
            source_path="/raid/cohort/input.pdf",
            document_sha256="b" * 64,
            expected_page_count=1,
            valid_blank_pages=[],
            model_revision="pinned-test-revision",
            provenance={},
        )
    assert list(tmp_path.glob("*/manifest.json")) == []


def test_declared_blank_page_allows_zero_byte_raw_blob(recipe: ModuleType, tmp_path: Path) -> None:
    manifest = _write_evidence(recipe, tmp_path, raw_output="", valid_blank_pages=[0])

    evidence = recipe.load_evidence(manifest)

    assert evidence.valid_blank_pages == (0,)
    assert evidence.pages[0].raw_output == ""


def test_evidence_requires_the_production_crop_threshold(recipe: ModuleType, tmp_path: Path) -> None:
    manifest = _write_evidence(recipe, tmp_path, min_crop_px=11)

    with pytest.raises(recipe.EvidenceError, match="min_crop_px must be 10"):
        recipe.load_evidence(manifest)


def test_malformed_tail_is_rejected_before_any_replay(recipe: ModuleType, tmp_path: Path) -> None:
    malformed = "<x_0.1><y_0.1>ok<x_0.2><y_0.2><class_Text>truncated"
    manifest = _write_evidence(recipe, tmp_path, raw_output=malformed)

    with pytest.raises(recipe.EvidenceError, match="unparsed content"):
        recipe.load_evidence(manifest)


def test_blob_hash_mismatch_is_rejected(recipe: ModuleType, tmp_path: Path) -> None:
    manifest = _write_evidence(recipe, tmp_path)
    payload = json.loads(manifest.read_text())
    raw_sha = payload["core"]["pages"][0]["raw_output"]["sha256"]
    (manifest.parent / "blobs" / raw_sha).write_bytes(b"changed")

    with pytest.raises(recipe.EvidenceError, match=r"bytes|SHA-256"):
        recipe.load_evidence(manifest)


def test_symlinked_blob_directory_is_rejected(recipe: ModuleType, tmp_path: Path) -> None:
    manifest = _write_evidence(recipe, tmp_path / "evidence")
    blobs = manifest.parent / "blobs"
    external_blobs = tmp_path / "external-blobs"
    blobs.rename(external_blobs)
    blobs.symlink_to(external_blobs, target_is_directory=True)

    with pytest.raises(recipe.EvidenceError, match="blobs directory"):
        recipe.load_evidence(manifest)


def test_compare_replays_is_deterministic_and_reports_picture_dimensions(
    recipe: ModuleType,
    tmp_path: Path,
) -> None:
    evidence = recipe.load_evidence(_write_evidence(recipe, tmp_path))
    picture = _picture_snapshot(recipe)
    nrl = _sealed_replay(recipe, evidence, "nrl", picture=picture)
    curator = _sealed_replay(recipe, evidence, "curator", picture=picture)

    first = recipe.compare_replays(evidence, nrl, curator)
    second = recipe.compare_replays(evidence, nrl, curator)

    assert first == second
    assert first["core"]["status"] == "equivalent"
    assert first["core"]["summary"] == {
        "page_count": 1,
        "equivalent_page_count": 1,
        "difference_count": 0,
    }
    assert first["report_sha256"] == hashlib.sha256(_canonical_bytes(first["core"])).hexdigest()


def test_compare_replays_reports_class_order_bbox_and_crop_differences(
    recipe: ModuleType,
    tmp_path: Path,
) -> None:
    evidence = recipe.load_evidence(_write_evidence(recipe, tmp_path))
    picture = _picture_snapshot(recipe)
    nrl = _sealed_replay(recipe, evidence, "nrl", picture=picture)
    curator = _sealed_replay(recipe, evidence, "curator", element_class="Text")
    curator_element = curator["replay"]["pages"][0]["elements"][0]
    curator_element["bbox_xyxy_norm"] = [0.2, 0.2, 0.5, 0.4]
    curator["replay_sha256"] = hashlib.sha256(_canonical_bytes(curator["replay"])).hexdigest()

    report = recipe.compare_replays(evidence, nrl, curator)

    fields = [difference["field"] for difference in report["core"]["differences"]]
    assert report["core"]["status"] == "different"
    assert "class_model_order" in fields
    assert "element_class" in fields
    assert "bbox_xyxy_norm" in fields
    assert "picture_presence" in fields


def test_compare_replays_reports_picture_dimension_difference(recipe: ModuleType, tmp_path: Path) -> None:
    evidence = recipe.load_evidence(_write_evidence(recipe, tmp_path))
    picture = _picture_snapshot(recipe)
    nrl = _sealed_replay(recipe, evidence, "nrl", picture=picture)
    curator = _sealed_replay(
        recipe,
        evidence,
        "curator",
        picture=dict(picture, width=picture["width"] - 1),
    )

    report = recipe.compare_replays(evidence, nrl, curator)

    assert any(difference["field"] == "picture.width" for difference in report["core"]["differences"])


def test_text_comparison_preserves_markdown_whitespace(recipe: ModuleType, tmp_path: Path) -> None:
    evidence = recipe.load_evidence(_write_evidence(recipe, tmp_path))
    nrl = _sealed_replay(recipe, evidence, "nrl", element_class="Text")
    curator = _sealed_replay(recipe, evidence, "curator", element_class="Text")
    nrl["replay"]["pages"][0]["elements"][0]["normalized_text"] = "    code\n"
    curator["replay"]["pages"][0]["elements"][0]["normalized_text"] = "code\n"
    for replay in (nrl, curator):
        replay["replay_sha256"] = hashlib.sha256(_canonical_bytes(replay["replay"])).hexdigest()

    report = recipe.compare_replays(evidence, nrl, curator)

    assert any(difference["field"] == "normalized_text" for difference in report["core"]["differences"])


def test_compare_replays_rejects_duplicate_page_in_sealed_replay(recipe: ModuleType, tmp_path: Path) -> None:
    evidence = recipe.load_evidence(_write_evidence(recipe, tmp_path))
    picture = _picture_snapshot(recipe)
    nrl = _sealed_replay(recipe, evidence, "nrl", picture=picture)
    curator = _sealed_replay(recipe, evidence, "curator", picture=picture)
    curator["replay"]["pages"].append(dict(curator["replay"]["pages"][0]))
    curator["replay_sha256"] = hashlib.sha256(_canonical_bytes(curator["replay"])).hexdigest()

    with pytest.raises(recipe.ReplayError, match="duplicate page"):
        recipe.compare_replays(evidence, nrl, curator)


def test_run_comparison_writes_one_sealed_report(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _write_evidence(recipe, tmp_path / "evidence")
    evidence = recipe.load_evidence(manifest)
    picture = _picture_snapshot(recipe)
    replays = {
        "nrl": _sealed_replay(recipe, evidence, "nrl", picture=picture),
        "curator": _sealed_replay(recipe, evidence, "curator", picture=picture),
    }
    calls: list[tuple[str, str, Path]] = []

    def fake_run(python: str, engine: str, selected_manifest: Path, _timeout: float) -> dict[str, Any]:
        calls.append((python, engine, selected_manifest))
        return replays[engine]

    monkeypatch.setattr(recipe, "_run_engine_subprocess", fake_run)
    output = tmp_path / "reports" / "comparison.json"

    report = recipe.run_comparison(
        evidence_manifest=manifest,
        output=output,
        nrl_python="/nrl/python",
        curator_python="/curator/python",
    )
    repeated = recipe.run_comparison(
        evidence_manifest=manifest,
        output=output,
        nrl_python="/nrl/python",
        curator_python="/curator/python",
    )

    assert report == repeated == json.loads(output.read_text())
    assert calls == [
        ("/nrl/python", "nrl", manifest.resolve()),
        ("/curator/python", "curator", manifest.resolve()),
        ("/nrl/python", "nrl", manifest.resolve()),
        ("/curator/python", "curator", manifest.resolve()),
    ]


def test_actual_nrl_projection_replay_keeps_model_order_and_textless_picture(
    recipe: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import nemo_retriever.common  # noqa: F401
    except ImportError:
        pytest.skip("NRL runtime is not installed in this environment")
    monkeypatch.syspath_prepend(str(Path(recipe.__file__).parent))
    evidence = recipe.load_evidence(_write_evidence(recipe, tmp_path))

    sealed = recipe._replay_nrl(evidence)
    _digest, replay = recipe._verified_replay(sealed, "nrl")

    elements = replay["pages"][0]["elements"]
    assert [element["element_class"] for element in elements] == ["Text", "Table", "Picture"]
    assert [element["modality"] for element in elements] == ["text", "table", "image"]
    assert elements[2]["normalized_text"] == ""
    assert elements[2]["picture"]["width"] > 10
    assert elements[2]["picture"]["height"] > 10


def test_actual_curator_stage_replay_keeps_v12_model_order(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    try:
        import nemo_curator  # noqa: F401
    except ImportError:
        pytest.skip("Curator runtime is not installed in this environment")

    fake_cv2 = ModuleType("cv2")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    evidence = recipe.load_evidence(_write_evidence(recipe, tmp_path))

    sealed = recipe._replay_curator(evidence)
    _digest, replay = recipe._verified_replay(sealed, "curator")

    elements = replay["pages"][0]["elements"]
    assert [element["element_class"] for element in elements] == ["Text", "Table", "Picture"]
    assert [element["modality"] for element in elements] == ["text", "table", "image"]
    assert elements[2]["picture"]["width"] > 10
    assert elements[2]["picture"]["height"] > 10


def test_main_returns_nonzero_for_malformed_evidence(
    recipe: ModuleType,
    tmp_path: Path,
) -> None:
    malformed = "<x_0.1><y_0.1>ok<x_0.2><y_0.2><class_Text>truncated"
    manifest = _write_evidence(recipe, tmp_path, raw_output=malformed)
    output = tmp_path / "must-not-exist.json"

    result = recipe.main(
        [
            "compare",
            "--evidence-manifest",
            str(manifest),
            "--output",
            str(output),
            "--nrl-python",
            sys.executable,
            "--curator-python",
            sys.executable,
        ]
    )

    assert result == 2
    assert not output.exists()


def test_inference_alternates_processes_and_preserves_benchmark_evidence(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "2" * 40
    manifest = _write_evidence(recipe, tmp_path / "evidence", model_revision=revision)
    snapshot = tmp_path / "model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    order = []

    def run(command: list[str], **kwargs: Any) -> Any:
        engine = command[command.index("--engine") + 1]
        order.append(engine)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert kwargs["env"]["PATH"].split(recipe.os.pathsep)[0] == f"/{engine}/bin"
        assert kwargs["env"]["VIRTUAL_ENV"] == f"/{engine}"
        page = {"raw_output_sha256": "d" * 64, "finish_reason": "stop", "completeness_issue": None}
        payload = {
            "engine": engine,
            "rgb_sha256": ["c" * 64],
            "model_revision": revision,
            "resources": {"samples": [{"gpu_uuid": "GPU-test"}]},
            "passes": [{"warmup": True, "pages": [page]}, {"warmup": False, "pages": [page]}],
        }
        return SimpleNamespace(returncode=0, stderr="", stdout=recipe._REPLAY_SENTINEL + json.dumps(payload))

    monkeypatch.setattr(recipe.subprocess, "run", run)
    report = recipe.run_inference_comparison(
        evidence_manifest=manifest,
        output=tmp_path / "report.json",
        nrl_python="/nrl/bin/python",
        curator_python="/curator/bin/python",
        model_snapshot=str(snapshot),
        gpu="GPU-test",
        repetitions=3,
    )
    assert order == ["nrl", "curator", "curator", "nrl", "nrl", "curator"]
    assert report["core"]["status"] == "compared"
    assert report["core"]["same_gpu_verified"]
    assert report["core"]["raw_responses_identical"]
    assert report == json.loads((tmp_path / "report.json").read_text())


def test_runtime_environment_keeps_virtualenv_bin_when_python_is_a_symlink(
    recipe: ModuleType,
    tmp_path: Path,
) -> None:
    environment_dir = tmp_path / "pinned-environment"
    interpreter = environment_dir / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    base = {"PATH": "/usr/bin:/bin", "VIRTUAL_ENV": "/other-environment", "KEEP": "unchanged"}
    configured = recipe._runtime_environment(str(interpreter), base)
    assert configured["PATH"] == f"{interpreter.parent}:/usr/bin:/bin"
    assert configured["VIRTUAL_ENV"] == str(environment_dir)
    assert configured["KEEP"] == "unchanged"
    assert base["VIRTUAL_ENV"] == "/other-environment"
    assert base["PATH"] == "/usr/bin:/bin"


def test_product_evaluation_preserves_order_and_excludes_long_documents(
    recipe: ModuleType,
) -> None:
    digest = "a" * 64
    common = {"sample_id": digest, "text_content": None, "binary_content": None}
    nrl_rows = [
        {**common, "position": -1, "modality": "metadata", "text_content": '{"num_pages": 51}'},
        {
            **common,
            "position": 0,
            "modality": "image",
            "content_type": "image/png",
            "element_class": "Picture",
            "binary_content": _page_png(),
            "page_number": 0,
            "bbox_xyxy_norm": [0.2, 0.2, 0.4, 0.4],
        },
    ]
    curator_rows = [
        {**common, "position": -1, "modality": "metadata", "text_content": '{"num_pages": 50}'},
        {**nrl_rows[1], "source_ref": json.dumps({"page": 0, "bbox": [0.2, 0.2, 0.4, 0.4]})},
    ]
    nrl = recipe._product_snapshot(nrl_rows, engine="nrl")
    curator = recipe._product_snapshot(curator_rows, engine="curator")
    assert not nrl["issues"]
    assert not curator["issues"]
    compared = recipe._compare_product_snapshots(nrl, curator, [{"content_sha256": digest, "expected_page_count": 51}])
    assert compared["paired_document_count"] == 0
    assert compared["long_documents"][0]["paired_metrics"] == "excluded"
    assert compared["coverage"][0]["curator_observed_content_pages"] == [0]
    curator_rows[1]["binary_content"] = _page_png()[:50]
    invalid = recipe._product_snapshot(curator_rows, engine="curator")
    assert len(invalid["issues"]) == 1


def test_native_product_selection_preserves_aliases_and_excludes_unselected_inputs(recipe: ModuleType) -> None:
    first, second = "a" * 64, "b" * 64
    inputs = [
        {"input_index": 0, "representative_input_index": 0, "content_sha256": first},
        {"input_index": 1, "representative_input_index": 0, "content_sha256": first},
        {"input_index": 2, "representative_input_index": 2, "content_sha256": second},
        {"input_index": 3, "content_sha256": None},
    ]
    representatives, accounting = recipe._select_product_inputs(inputs, [first, first])
    assert representatives == inputs[:1]
    assert accounting == inputs[:2]
    assert recipe._select_product_inputs(inputs, None) == ([inputs[0], inputs[2]], inputs)
    for invalid in ([], ["not-a-digest"], ["c" * 64]):
        with pytest.raises(recipe.EvidenceError, match="sample IDs"):
            recipe._select_product_inputs(inputs, invalid)
    parsed = recipe._build_parser().parse_args(
        [
            "native-product",
            "--handoff-manifest",
            "handoff.json",
            "--output-dir",
            "output",
            "--model-snapshot",
            "snapshot",
            "--gpu",
            "0",
            "--sample-id",
            first,
            "--sample-id",
            second,
        ]
    )
    assert parsed.sample_id == [first, second]


def test_completed_inference_engine_survives_later_engine_failure(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    revision = "2" * 40
    manifest = _write_evidence(recipe, tmp_path / "evidence", model_revision=revision)
    snapshot = tmp_path / "model" / "snapshots" / revision
    snapshot.mkdir(parents=True)

    def run(command: list[str], **_kwargs: Any) -> Any:
        engine = command[command.index("--engine") + 1]
        if engine == "curator":
            return SimpleNamespace(returncode=1, stderr="controlled native failure", stdout="")
        payload = {"engine": engine, "model_revision": revision, "passes": []}
        return SimpleNamespace(returncode=0, stderr="", stdout=recipe._REPLAY_SENTINEL + json.dumps(payload))

    monkeypatch.setattr(recipe.subprocess, "run", run)
    output = tmp_path / "inference.json"
    with pytest.raises(recipe.ReplayError, match="controlled native failure"):
        recipe.run_inference_comparison(
            evidence_manifest=manifest,
            output=output,
            nrl_python="/nrl/python",
            curator_python="/curator/python",
            model_snapshot=str(snapshot),
            gpu="GPU-test",
            repetitions=1,
        )
    saved = json.loads((tmp_path / "inference.json.runs" / "000-nrl.json").read_text())
    assert saved["run"]["engine"] == "nrl"
    assert saved["run_sha256"] == hashlib.sha256(_canonical_bytes(saved["run"])).hexdigest()
    assert not output.exists()


def test_nrl_inference_rejects_a_different_active_cache_without_changing_registry(
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    try:
        from huggingface_hub import constants
        from nemo_retriever.models.hf_model_registry import HF_MODEL_REVISIONS, get_hf_revision
    except ImportError:
        pytest.skip("NRL runtime is not installed")
    original_registry = dict(HF_MODEL_REVISIONS)
    revision = get_hf_revision(recipe.PARSE_MODEL)
    cache = tmp_path / "active" / "hub"
    resolved = cache / "models--nvidia--NVIDIA-Nemotron-Parse-v1.2" / "snapshots" / revision
    resolved.mkdir(parents=True)
    requested = tmp_path / "different" / "snapshots" / revision
    requested.mkdir(parents=True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HOME", str(cache.parent))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("NEMO_RETRIEVER_HF_CACHE_DIR", str(cache.parent))
    with pytest.raises(recipe.EvidenceError, match="active HF cache resolves"):
        recipe._prepare_nrl_inference_actor(requested)
    assert original_registry == HF_MODEL_REVISIONS


@pytest.mark.parametrize("engine", ["nrl", "curator"])
@pytest.mark.parametrize(
    ("generation", "declared_blank", "expected_issue"),
    [
        (_raw_output(), False, None),
        ("", False, "undeclared empty page"),
        (" \n\t", False, "undeclared empty page"),
        ("", True, None),
        (" \n\t", True, None),
        ("<x_0.1><y_0.1>text<x_0.1><y_0.5><class_Text>", False, "zero area"),
        ("<x_0.1><y_0.1>text<x_1.1><y_0.5><class_Text>", False, "within [0, 1]"),
        ("controlled_generation_failure", False, "generation_failure"),
    ],
)
def test_inference_uses_real_native_processing_with_identical_rgb(  # noqa: C901, PLR0913, PLR0915
    recipe: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    engine: str,
    generation: str,
    declared_blank: bool,
    expected_issue: str | None,
) -> None:
    revision = "2bd0189bffd6cdded6280d9f22a4077b25a504e3"
    manifest = _write_evidence(
        recipe,
        tmp_path / "evidence",
        model_revision=revision,
        raw_output="" if declared_blank else None,
        valid_blank_pages=[0] if declared_blank else None,
    )
    hub_cache = tmp_path / "hub"
    snapshot = hub_cache / "models--nvidia--NVIDIA-Nemotron-Parse-v1.2" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    evidence = recipe.load_evidence(manifest)
    generation_calls = []
    shutdown_calls = []
    sampling_params = pytest.importorskip("vllm").SamplingParams

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=self.shutdown))
            if kwargs:
                assert kwargs["model"] == recipe.PARSE_MODEL
                assert kwargs["revision"] == revision
                assert kwargs["tokenizer_revision"] == revision

        def shutdown(self, *, timeout: float) -> None:
            assert "generate" not in vars(self)
            shutdown_calls.append(timeout)

        def generate(self, prompts: Any, _sampling: Any) -> list[Any]:
            generation_calls.append(prompts)
            if expected_issue == "generation_failure":
                message = "controlled generation failure"
                raise KeyboardInterrupt(message)
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(text=generation, finish_reason="stop", token_ids=[3, 4, 5])],
                )
                for _prompt in prompts
            ]

    def configure(instance: Any) -> None:
        instance._llm = FakeLLM()
        instance._sampling_params = sampling_params(
            temperature=0, top_k=1, repetition_penalty=1.1, max_tokens=9000, skip_special_tokens=False
        )
        instance._task_prompt = recipe.PARSE_TASK_PROMPT
        instance._proc_size = recipe.PARSE_PROC_SIZE

    if engine == "nrl":
        try:
            from huggingface_hub import constants
            from nemo_retriever.models.hf_model_registry import get_hf_revision
        except ImportError:
            pytest.skip("NRL runtime is not installed")
        assert get_hf_revision(recipe.PARSE_MODEL) == revision
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub_cache))
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
        monkeypatch.setenv("NEMO_RETRIEVER_HF_CACHE_DIR", str(tmp_path))
        fake_vllm = ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = sampling_params
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    else:
        try:
            from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage
        except ImportError:
            pytest.skip("Curator runtime is not installed")
        monkeypatch.setattr(NemotronParseInferenceStage, "setup", configure)
        monkeypatch.setattr(NemotronParseInferenceStage, "teardown", lambda _instance: None)
    monkeypatch.setattr(recipe._ResourceMonitor, "start", lambda _instance: None)
    monkeypatch.setattr(recipe._ResourceMonitor, "stop", lambda _instance: {"samples": []})
    if expected_issue == "generation_failure":
        # The NRL actor records even BaseException as a page error; Curator propagates it.
        exception = recipe.ReplayError if engine == "nrl" else KeyboardInterrupt
        pattern = "controlled generation failure"
        with pytest.raises(exception, match=pattern):
            recipe._run_inference_engine(
                evidence, engine=engine, model_snapshot=str(snapshot), gpu="test", measured_passes=2
            )
        assert shutdown_calls == [30.0]
        return
    report = recipe._run_inference_engine(
        evidence, engine=engine, model_snapshot=str(snapshot), gpu="test", measured_passes=2
    )
    assert shutdown_calls == [30.0]
    assert report["effective_sampling"]["top_k"] == 0
    assert report["requested_sampling"]["top_k"] == 1
    assert len(generation_calls) == 3
    assert [measurement["warmup"] for measurement in report["passes"]] == [True, False, False]
    assert all(measurement["pages"][0]["output_tokens"] == 3 for measurement in report["passes"])
    assert all(measurement["pages"][0]["finish_reason"] == "stop" for measurement in report["passes"])
    for measurement in report["passes"]:
        issue = measurement["pages"][0]["completeness_issue"]
        if expected_issue is None:
            assert issue is None
        else:
            assert expected_issue in issue
    assert len(report["rgb_sha256"]) == 1


def _benchmark_pdf(path: Path, pages: int = 2) -> None:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument.new()
    try:
        for _ in range(pages):
            page = document.new_page(100, 120)
            page.close()
        document.save(path)
    finally:
        document.close()


def _source_reference(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "byte_length": len(payload)}


def _seal_source_benchmark(case: SimpleNamespace) -> None:
    for run in case.benchmark["core"]["runs"]:
        run["result"]["report_sha256"] = hashlib.sha256(_canonical_bytes(run["result"]["core"])).hexdigest()
        Path(run["result_path"]).write_text(json.dumps(run["result"]))
        Path(run["observer_path"]).write_text(json.dumps(run["observer"]))
    case.benchmark["report_sha256"] = hashlib.sha256(_canonical_bytes(case.benchmark["core"])).hexdigest()
    case.benchmark_path.write_text(json.dumps(case.benchmark))


@pytest.fixture
def source_evidence_case(recipe: ModuleType, benchmark_case: SimpleNamespace, tmp_path: Path) -> SimpleNamespace:  # noqa: PLR0915
    """Real Parquet and immutable sidecars, with a partial document and native negatives."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    case = benchmark_case
    digest = case.cohort["core"]["inputs"][0]["content_sha256"]
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color="blue").save(output, format="PNG")
    source_image_bytes = output.getvalue()
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color="green").save(output, format="PNG")
    image_bytes = output.getvalue()
    table_text = '<table><tr><th colspan="2">Merged &amp; header</th></tr></table>\r\n'
    page_outcomes = [
        {"page_number": 0, "status": "success", "element_count": 3, "issues": []},
        {
            "page_number": 1,
            "status": "failed",
            "element_count": 0,
            "issues": [{"kind": "unexpected_empty_output", "page_number": 1}],
        },
    ]
    outcome = {
        "content_sha256": digest,
        "expected_page_count": 2,
        "status": "partial",
        "extraction_status": "partial",
        "publication_status": "handed_off",
        "validated_page_count": 1,
        "content_page_count": 1,
        "blank_page_count": 0,
        "failed_page_count": 1,
        "page_outcomes": page_outcomes,
        "issues": page_outcomes[1]["issues"],
        "element_count": 4,
    }
    metadata = {"num_pages": 2, "extraction_status": "partial", "page_outcomes": page_outcomes}
    base = {
        "sample_id": digest,
        "position": -1,
        "modality": "metadata",
        "content_type": "application/json",
        "text_content": json.dumps(metadata),
        "binary_content": None,
        "source_ref": None,
        "materialize_error": None,
        "page_number": None,
        "element_class": None,
        "bbox_xyxy_norm": None,
        "url": None,
    }
    controls = {"nrl_parse_cpus": 1, "nrl_parse_batch_size": 64, "native_pdfs_per_task": 10}
    runs = []
    for engine in ("nrl", "curator"):
        rows = [dict(base)]
        for position, modality, text in ((0, "text", " A\r\nB "), (1, "table", table_text), (2, "image", "")):
            rows.append(
                {
                    **base,
                    "position": position,
                    "modality": modality,
                    "content_type": "image/png" if modality == "image" else "text/markdown",
                    "text_content": text,
                    "binary_content": image_bytes if modality == "image" else None,
                    "element_class": {"text": "Text", "table": "Table", "image": "Picture"}[modality],
                    "page_number": 0,
                    "bbox_xyxy_norm": [0.1, 0.1, 0.5, 0.5],
                    "source_ref": json.dumps({"page": 0, "bbox": [0.1, 0.1, 0.5, 0.5]})
                    if engine == "curator"
                    else None,
                }
            )
        if engine == "curator":
            rows.extend(
                [
                    {
                        **rows[1],
                        "position": 3,
                        "page_number": 1,
                        "source_ref": json.dumps({"page": 1, "bbox": None}),
                        "text_content": None,
                    },
                    {**rows[1], "position": 4, "page_number": 1, "text_content": "Conflicting page provenance"},
                ]
            )
        folder = tmp_path / "benchmark" / f"000-{engine}"
        (folder / "parquet").mkdir(parents=True)
        parquet = folder / "parquet/data.parquet"
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        rows = pq.read_table(parquet).to_pylist()
        snapshot = recipe._product_snapshot(rows, engine=engine)
        details = {
            "snapshot": snapshot,
            "export_sha256": {str(parquet): _source_reference(parquet)["sha256"]},
            "configuration": {
                "parse_cpus": 1,
                "parse_batch_size": 64,
                "projection_block_rows": None,
                "pdfs_per_task": 10,
            },
        }
        if engine == "nrl":
            details["document_outcomes"] = [outcome]
            completion = case.contract._seal_payload(
                {"status": "published", "documents": [{**outcome, "publication_status": "published"}]},
                "completion_sha256",
            )
            (folder / "ingest").mkdir()
            (folder / "ingest/completion_manifest.json").write_text(json.dumps(completion))
            details["completion"] = completion
        core = {
            "schema": "nrl_curator_benchmark_engine",
            "status": "compared",
            "engine": engine,
            "model_snapshot": case.args.model_snapshot,
            "cohort_sha256": case.cohort["report_sha256"],
            "details": details,
            "publication_policy": "validated_pages_v1" if engine == "nrl" else "native_curator",
            "counts": recipe._benchmark_output_counts(
                snapshot,
                case.cohort["core"]["inputs"],
                native_page_cap=50 if engine == "curator" else None,
                document_outcomes=details.get("document_outcomes"),
            ),
        }
        observer = case.contract._seal_payload(
            {
                "qualification_passed": True,
                "capacity_enforced": True,
                "returncode": 0,
                "wall_seconds": 1.0,
                "baseline_sha256": case.baseline["baseline_sha256"],
                "command": ["python", "nrl_compare.py", "_benchmark-engine", "--cohort", case.args.cohort],
            },
            "summary_sha256",
        )
        runs.append(
            {
                "engine": engine,
                "repetition": 0,
                "observer": observer,
                "observer_path": str(folder / "observer.json"),
                "result": {"core": core},
                "result_path": str(folder / "result.json"),
            }
        )
    benchmark = {
        "core": {
            "schema": "nrl_curator_end_to_end_benchmark",
            "collection_complete": True,
            "cohort": case.cohort,
            "baseline_sha256": case.baseline["baseline_sha256"],
            "requested_controls": controls,
            "runs": runs,
        }
    }
    source_selection = tmp_path / "source-selection.json"
    source_selection.write_text(json.dumps({"inputs": [{"content_sha256": digest, "split": "tuning"}]}))
    selected, historical = [], []
    capture_root = tmp_path / "historical-captures"
    capture_directory = recipe._capture_directory(capture_root, str(case.source))
    (capture_directory / "blobs").mkdir(parents=True)
    (capture_directory / "pages").mkdir()
    historical_document = {**outcome, "path": str(case.source)}
    source_images = {}
    for page in (0, 1):
        if page:
            output = io.BytesIO()
            Image.new("RGB", (8, 8), color="red").save(output, format="PNG")
            source_image_bytes = output.getvalue()
        source_images[page] = source_image_bytes
        raw_bytes = _raw_output().encode() if page == 0 else b""
        image_path = capture_directory / "blobs" / hashlib.sha256(source_image_bytes).hexdigest()
        raw_path = capture_directory / "blobs" / hashlib.sha256(raw_bytes).hexdigest()
        capture_path = capture_directory / "pages" / f"{page + 1:08d}.json"
        image_path.write_bytes(source_image_bytes)
        raw_path.write_bytes(raw_bytes)
        capture = {
            "page_number": page,
            "native_page_number": page + 1,
            "finish_reason": "stop",
            "error_statuses": {"extraction": "ok", "nemotron_parse": "ok"},
            "page_image": {**_source_reference(image_path), "orig_shape_hw": [8, 8]},
            "raw_output": _source_reference(raw_path),
        }
        capture_path.write_text(json.dumps(capture))
        selected.append(
            {
                "source_sha256": digest,
                "source_path": str(case.source),
                "page_number_zero_based": page,
                "source_page_number_one_based": page + 1,
                "expected_document_pages": 2,
                "split": "tuning",
            }
        )
        historical.append(
            {
                "source_sha256": digest,
                "source": _source_reference(case.source),
                "original_page_number_zero_based": page,
                "original_page_number_one_based": page + 1,
                "expected_document_pages": 2,
                "split": "tuning",
                "original_capture": capture,
                "original_document_outcome": historical_document,
                "captured_page_record": _source_reference(capture_path),
                "page_image_blob": _source_reference(image_path),
                "raw_output_blob": _source_reference(raw_path),
                "existing_replay_evidence": {"status": "eligible" if page == 0 else "negative_case"},
            }
        )
    selection_path = tmp_path / "quality-selection.json"
    selection = {
        "source_selection": {
            "path": str(source_selection),
            "file_sha256": _source_reference(source_selection)["sha256"],
        },
        "pages": selected,
    }
    selection_path.write_text(json.dumps(selection))
    historical_handoff = case.contract._seal_payload(
        {"documents": [historical_document], "benchmark_evidence": {"root": str(capture_root)}},
        "handoff_sha256",
    )
    historical_handoff_path = tmp_path / "historical-handoff.json"
    historical_handoff_path.write_text(json.dumps(historical_handoff))
    index = {
        "kind": "private_historical_capture_reference_index",
        "quality_selection": _source_reference(selection_path),
        "source_selection": _source_reference(source_selection),
        "historical_handoff": {
            **_source_reference(historical_handoff_path),
            "handoff_sha256": historical_handoff["handoff_sha256"],
        },
        "pages": historical,
    }
    index_path = tmp_path / "historical-index.json"
    index_path.write_text(json.dumps(index))
    evidence = SimpleNamespace(
        benchmark=benchmark,
        benchmark_path=tmp_path / "benchmark/benchmark_report.json",
        index=index,
        index_path=index_path,
        selection=selection,
        selection_path=selection_path,
        source_selection=source_selection,
        source=case.source,
        digest=digest,
        destination=tmp_path / "source-evidence",
        table_text=table_text,
        image_bytes=image_bytes,
        source_images=source_images,
        contract=case.contract,
    )
    _seal_source_benchmark(evidence)
    return evidence


def _run_source_evidence(recipe: ModuleType, case: SimpleNamespace) -> dict[str, Any]:
    return recipe.run_source_evidence(
        benchmark_report=case.benchmark_path, historical_capture_index=case.index_path, output_dir=case.destination
    )


def test_source_evidence_preserves_exact_rows_and_negatives_without_execution(
    recipe: ModuleType, source_evidence_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = source_evidence_case
    monkeypatch.setattr(recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must remain offline"))
    report = _run_source_evidence(recipe, case)
    core = report["core"]
    assert report["report_sha256"] == hashlib.sha256(_canonical_bytes(core)).hexdigest()
    assert core["semantic_status"] == "not_judged"
    assert core["human_calibrated"] is False
    assert core["engines"]["nrl"]["counts"]["partial_document_count"] == 1
    assert core["engines"]["nrl"]["counts"]["failed_page_count"] == 1
    first, negative = core["pages"]
    assert first["delivered"]["nrl"][0]["values"]["text_content"] == " A\r\nB "
    assert first["delivered"]["nrl"][1]["values"]["text_content"] == case.table_text
    image = first["delivered"]["nrl"][2]
    assert (case.destination / image["values"]["binary_content"]["path"]).read_bytes() == case.image_bytes
    assert image["image_decoding"]["status"] == "decoded"
    assert image["row_id"] == f"nrl:{case.digest}:2"
    assert negative["delivered"]["nrl"] == []
    assert negative["historical_capture"]["existing_replay_evidence"]["status"] == "negative_case"
    assert negative["delivered"]["curator"][0]["values"]["text_content"] is None
    assert negative["delivered"]["curator"][0]["validation_issues"]
    unmapped = core["document_level_rows"]["curator"][-1]
    assert unmapped["values"]["position"] == 4
    assert "disagree" in unmapped["page_mapping_issue"]
    assert core["engines"]["curator"]["output_validation_issue"]
    assert json.loads((case.destination / "report.json").read_text()) == report


@pytest.mark.parametrize(
    "failure",
    [
        "outer_seal",
        "collection",
        "pair",
        "observer_disk",
        "result_disk",
        "parquet_bytes",
        "extra_parquet",
        "missing_parquet",
        "snapshot",
        "counts",
        "completion",
    ],
)
def test_source_evidence_rejects_mutated_benchmark_artifacts(
    recipe: ModuleType, source_evidence_case: SimpleNamespace, failure: str
) -> None:
    case = source_evidence_case
    run = case.benchmark["core"]["runs"][0]
    parquet = Path(next(iter(run["result"]["core"]["details"]["export_sha256"])))
    if failure == "outer_seal":
        case.benchmark["report_sha256"] = "0" * 64
        case.benchmark_path.write_text(json.dumps(case.benchmark))
    elif failure in {"collection", "pair", "snapshot", "counts"}:
        if failure == "collection":
            case.benchmark["core"]["collection_complete"] = False
        elif failure == "pair":
            case.benchmark["core"]["runs"].pop()
        elif failure == "snapshot":
            run["result"]["core"]["details"]["snapshot"]["documents"][case.digest]["pages"]["0"][0][
                "normalized_text"
            ] = "changed"
        else:
            run["result"]["core"]["counts"]["rows"] += 1
        _seal_source_benchmark(case)
    elif failure.endswith("_disk"):
        Path(run["observer_path" if failure == "observer_disk" else "result_path"]).write_text("{}")
    elif failure == "parquet_bytes":
        parquet.write_bytes(parquet.read_bytes() + b"changed")
    elif failure == "extra_parquet":
        parquet.with_name("extra.parquet").write_bytes(parquet.read_bytes())
    elif failure == "missing_parquet":
        parquet.rename(parquet.with_suffix(".saved"))
    else:
        path = Path(run["result_path"]).parent / "ingest/completion_manifest.json"
        path.write_text("{}")
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        _run_source_evidence(recipe, case)
    assert not (case.destination / "report.json").exists()


@pytest.mark.parametrize(
    "failure", ["reference", "source_pdf", "heldout", "duplicate", "order", "range", "one_based", "capture"]
)
def test_source_evidence_rejects_selection_or_capture_drift(
    recipe: ModuleType, source_evidence_case: SimpleNamespace, failure: str
) -> None:
    case = source_evidence_case
    if failure == "reference":
        Path(case.index["pages"][0]["raw_output_blob"]["path"]).write_bytes(b"changed")
    elif failure == "source_pdf":
        case.source.write_bytes(case.source.read_bytes() + b"changed")
    elif failure == "capture":
        case.index["pages"][0]["original_capture"]["finish_reason"] = "length"
    elif failure == "heldout":
        case.source_selection.write_text(json.dumps({"inputs": [{"content_sha256": case.digest, "split": "heldout"}]}))
        case.selection["source_selection"]["file_sha256"] = _source_reference(case.source_selection)["sha256"]
        case.index["source_selection"] = _source_reference(case.source_selection)
    elif failure == "duplicate":
        case.selection["pages"].append(case.selection["pages"][0])
        case.index["pages"].append(case.index["pages"][0])
    elif failure == "order":
        case.index["pages"].reverse()
    elif failure == "range":
        case.selection["pages"][0]["page_number_zero_based"] = 3
        case.index["pages"][0]["original_page_number_zero_based"] = 3
    else:
        case.selection["pages"][0]["source_page_number_one_based"] = 2
    case.selection_path.write_text(json.dumps(case.selection))
    case.index["quality_selection"] = _source_reference(case.selection_path)
    case.index_path.write_text(json.dumps(case.index))
    with pytest.raises((ValueError, RuntimeError), match=r"changed|differ|selected|tuning|identity|reference"):
        _run_source_evidence(recipe, case)
    assert not (case.destination / "report.json").exists()


def test_source_evidence_retains_truncated_delivered_image_as_negative(
    recipe: ModuleType, source_evidence_case: SimpleNamespace
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    case = source_evidence_case
    run = case.benchmark["core"]["runs"][1]
    details = run["result"]["core"]["details"]
    parquet = Path(next(iter(details["export_sha256"])))
    rows = pq.read_table(parquet).to_pylist()
    rows[3]["binary_content"] = case.image_bytes[:45]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    details["export_sha256"][str(parquet)] = _source_reference(parquet)["sha256"]
    details["snapshot"] = recipe._product_snapshot(rows, engine="curator")
    run["result"]["core"]["counts"] = recipe._benchmark_output_counts(
        details["snapshot"], case.benchmark["core"]["cohort"]["core"]["inputs"], native_page_cap=50
    )
    _seal_source_benchmark(case)
    report = _run_source_evidence(recipe, case)
    image = report["core"]["pages"][0]["delivered"]["curator"][2]
    assert image["image_decoding"]["status"] == "invalid"
    assert image["validation_issues"]
    assert (case.destination / image["values"]["binary_content"]["path"]).read_bytes() == case.image_bytes[:45]


@pytest.mark.parametrize(
    "target",
    ["benchmark", "index", "source", "observer", "result", "completion", "cohort", "parquet", "extra_parquet"],
)
def test_source_evidence_detects_mutation_during_assembly(
    recipe: ModuleType, source_evidence_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    case = source_evidence_case
    original = recipe._source_evidence_row
    run = case.benchmark["core"]["runs"][0]
    parquet = Path(next(iter(run["result"]["core"]["details"]["export_sha256"])))
    path = {
        "benchmark": case.benchmark_path,
        "index": case.index_path,
        "source": case.source,
        "observer": Path(run["observer_path"]),
        "result": Path(run["result_path"]),
        "completion": Path(run["result_path"]).parent / "ingest/completion_manifest.json",
        "cohort": Path(run["observer"]["command"][-1]),
        "parquet": parquet,
        "extra_parquet": parquet.with_name("extra.parquet"),
    }[target]

    def mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        path.write_bytes(parquet.read_bytes() if target == "extra_parquet" else path.read_bytes() + b" ")
        return original(*args, **kwargs)

    monkeypatch.setattr(recipe, "_source_evidence_row", mutate)
    with pytest.raises(recipe.EvidenceError, match="changed"):
        _run_source_evidence(recipe, case)
    assert not (case.destination / "report.json").exists()


def test_source_evidence_rejects_another_documents_valid_page_zero_capture(
    recipe: ModuleType, source_evidence_case: SimpleNamespace, tmp_path: Path
) -> None:
    case = source_evidence_case
    historical = case.index["pages"][0]
    foreign_directory = recipe._capture_directory(tmp_path / "historical-captures", str(tmp_path / "foreign.pdf"))
    (foreign_directory / "pages").mkdir(parents=True)
    (foreign_directory / "blobs").mkdir()
    foreign_capture = foreign_directory / "pages/00000001.json"
    foreign_capture.write_bytes(Path(historical["captured_page_record"]["path"]).read_bytes())
    historical["captured_page_record"] = _source_reference(foreign_capture)
    for name in ("page_image_blob", "raw_output_blob"):
        foreign_blob = foreign_directory / "blobs" / historical[name]["sha256"]
        foreign_blob.write_bytes(Path(historical[name]["path"]).read_bytes())
        historical[name] = _source_reference(foreign_blob)
    case.index_path.write_text(json.dumps(case.index))
    with pytest.raises(recipe.EvidenceError, match="another document"):
        _run_source_evidence(recipe, case)
    assert not case.destination.exists()


def test_source_evidence_cli_and_fresh_destination(recipe: ModuleType, source_evidence_case: SimpleNamespace) -> None:
    case = source_evidence_case
    command = [
        "source-evidence",
        "--benchmark-report",
        str(case.benchmark_path),
        "--historical-capture-index",
        str(case.index_path),
        "--output-dir",
        str(case.destination),
    ]
    assert recipe.main(command) == 0
    original = (case.destination / "report.json").read_bytes()
    assert recipe.main(command) == 2
    assert (case.destination / "report.json").read_bytes() == original


@pytest.mark.parametrize("failure", ["engine", "config", "cohort", "observer_seal", "completion_status"])
def test_source_evidence_revalidates_identity_controls_and_publication_transition(
    recipe: ModuleType, source_evidence_case: SimpleNamespace, failure: str
) -> None:
    case = source_evidence_case
    run = case.benchmark["core"]["runs"][0]
    if failure == "engine":
        run["result"]["core"]["engine"] = "curator"
    elif failure == "config":
        run["result"]["core"]["details"]["configuration"]["parse_batch_size"] = 128
    elif failure == "cohort":
        case.benchmark["core"]["cohort"]["report_sha256"] = "0" * 64
    elif failure == "observer_seal":
        run["observer"]["summary_sha256"] = "0" * 64
    else:
        details = run["result"]["core"]["details"]
        completion = copy.deepcopy(details["completion"])
        completion.pop("completion_sha256")
        completion["documents"][0]["publication_status"] = "handed_off"
        completion = case.contract._seal_payload(completion, "completion_sha256")
        details["completion"] = completion
        (Path(run["result_path"]).parent / "ingest/completion_manifest.json").write_text(json.dumps(completion))
    _seal_source_benchmark(case)
    with pytest.raises((ValueError, RuntimeError)):
        _run_source_evidence(recipe, case)
    assert not (case.destination / "report.json").exists()


def test_source_evidence_preserves_wholly_failed_document_accounting(
    recipe: ModuleType, source_evidence_case: SimpleNamespace
) -> None:
    import pyarrow.parquet as pq

    case = source_evidence_case
    run = case.benchmark["core"]["runs"][0]
    details = run["result"]["core"]["details"]
    parquet = Path(next(iter(details["export_sha256"])))
    pq.write_table(pq.read_table(parquet).slice(0, 0), parquet)
    outcome = details["document_outcomes"][0]
    outcome.update(
        status="failed",
        extraction_status="failed",
        publication_status="not_applicable",
        validated_page_count=0,
        content_page_count=0,
        failed_page_count=2,
        element_count=0,
    )
    outcome["page_outcomes"][0] = {
        "page_number": 0,
        "status": "failed",
        "element_count": 0,
        "issues": [{"kind": "unexpected_empty_output", "page_number": 0}],
    }
    outcome["issues"] = [issue for page in outcome["page_outcomes"] for issue in page["issues"]]
    details["snapshot"] = recipe._product_snapshot([], engine="nrl")
    details["export_sha256"][str(parquet)] = _source_reference(parquet)["sha256"]
    completion = case.contract._seal_payload({"status": "published", "documents": [outcome]}, "completion_sha256")
    details["completion"] = completion
    (Path(run["result_path"]).parent / "ingest/completion_manifest.json").write_text(json.dumps(completion))
    run["result"]["core"]["counts"] = recipe._benchmark_output_counts(
        details["snapshot"], case.benchmark["core"]["cohort"]["core"]["inputs"], document_outcomes=[outcome]
    )
    _seal_source_benchmark(case)
    report = _run_source_evidence(recipe, case)
    assert all(page["delivered"]["nrl"] == [] for page in report["core"]["pages"])
    assert report["core"]["engines"]["nrl"]["counts"]["failed_document_count"] == 1
    assert report["core"]["engines"]["nrl"]["counts"]["failed_page_count"] == 2


def test_source_evidence_does_not_copy_unselected_page_assets(
    recipe: ModuleType, source_evidence_case: SimpleNamespace
) -> None:
    case = source_evidence_case
    row = {
        "sample_id": case.digest,
        "position": 42,
        "modality": "image",
        "page_number": 1,
        "binary_content": case.image_bytes,
    }
    assert recipe._source_evidence_row(row, "nrl", case.destination, {(case.digest, 0)}) is None
    assert not case.destination.exists()


@pytest.fixture
def judge_case(
    recipe: ModuleType, source_evidence_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    from nemo_curator.models.client import openai_client

    case = source_evidence_case
    packet = _run_source_evidence(recipe, case)
    state = SimpleNamespace(
        calls=[],
        initialized=[],
        closed=0,
        response_change=None,
        request_error=None,
        close_error=False,
        setup_error=False,
        during_request=None,
    )

    class SDK:
        def __init__(self, **kwargs: Any) -> None:
            state.initialized.append(kwargs)
            if state.setup_error:
                message = "test setup failure"
                raise RuntimeError(message)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            state.calls.append(kwargs)
            if state.during_request is not None:
                state.during_request()
            if state.request_error is not None:
                raise state.request_error
            candidates = json.loads(kwargs["messages"][1]["content"][0]["text"])["candidate_data"]
            a_refs = [row["ref"] for row in candidates["A"][:2]]
            b_refs = [row["ref"] for row in candidates["B"][:2]]
            finding = {
                "facet": "table_structure",
                "category": "both_preserved" if a_refs and b_refs else "uncertain",
                "a_refs": a_refs,
                "b_refs": b_refs,
                "source_bbox_norm": [0.1, 0.1, 0.9, 0.9],
                "source_evidence": "Visible merged header and values",
                "reason": "Related rows retain the source unit",
            }
            raw = {
                "id": "response-test",
                "model": "resolved-test-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"findings": [finding]}), "refusal": None},
                    }
                ],
                "usage": None,
            }
            if state.response_change is not None:
                state.response_change(raw)
            return SimpleNamespace(model_dump=lambda **_kwargs: raw)

        async def close(self) -> None:
            state.closed += 1
            if state.close_error:
                message = "test cleanup failure"
                raise RuntimeError(message)

    monkeypatch.setattr(openai_client, "AsyncOpenAI", SDK)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-must-not-be-persisted")
    return SimpleNamespace(
        packet=packet,
        packet_path=case.destination / "report.json",
        output_dir=case.destination.parent / "judge-output",
        state=state,
        source=case,
    )


def _run_judge(recipe: ModuleType, case: SimpleNamespace, **kwargs: Any) -> dict[str, Any]:
    return recipe.run_source_judge(
        source_evidence=case.packet_path,
        output_dir=case.output_dir,
        base_url="https://judge.example.test/v1",
        model="explicit-test-model",
        **kwargs,
    )


def test_source_judge_uses_native_client_pixels_blinding_and_no_retries(
    recipe: ModuleType, judge_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_curator.models.client import openai_client

    case = judge_case
    original = openai_client.AsyncOpenAIClient.query_model_response
    wrappers = []

    async def inspect_wrapper(self: Any, **kwargs: Any) -> Any:
        wrappers.append((self.max_retries, self.max_concurrent_requests))
        return await original(self, **kwargs)

    monkeypatch.setattr(openai_client.AsyncOpenAIClient, "query_model_response", inspect_wrapper)
    monkeypatch.setattr(
        recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("judge must not launch extraction")
    )
    report = _run_judge(recipe, case)
    core = report["core"]
    assert core["status"] == "judged_uncalibrated"
    assert core["report_only"] is True
    assert core["human_calibrated"] is False
    assert len(case.state.calls) == 2
    assert wrappers == [(0, 1), (0, 1)]
    assert case.state.initialized[0]["max_retries"] == 0
    assert case.state.closed == 1
    assert core["pages"][0]["route_assignment"] == {"A": "nrl", "B": "curator"}
    assert core["pages"][1]["route_assignment"] == {"A": "curator", "B": "nrl"}
    finding = core["pages"][0]["findings"][0]
    assert len(finding["a_refs"]) == len(finding["b_refs"]) == 2  # Many-to-many, not positional matching.
    for index, (call, page) in enumerate(zip(case.state.calls, core["pages"], strict=True)):
        assert call["model"] == "explicit-test-model"
        assert call["n"] == 1
        assert call["temperature"] == 0
        assert call["stream"] is False
        assert call["max_tokens"] == 4096
        assert call["timeout"] == 120.0
        assert call["response_format"] == {"type": "json_object"}
        prompt = json.dumps(call["messages"])
        assert all(label not in prompt for label in ("nrl", "curator", "sample_id", "source_path", case.source.digest))
        assert "untrusted DATA" in prompt
        images = [item["image_url"]["url"] for item in call["messages"][1]["content"] if item["type"] == "image_url"]
        assert all(url.startswith("data:image/png;base64,") for url in images)
        assert recipe.base64.b64decode(images[0].split(",", 1)[1]) == case.source.source_images[index]
        assert all(recipe.base64.b64decode(url.split(",", 1)[1]) == case.source.image_bytes for url in images[1:])
        request = json.loads((case.output_dir / page["request"]["path"]).read_bytes())
        assert request["messages"] == call["messages"]
        assert page["usage"] is None
        assert page["raw_response"] is not None
    serialized = (case.output_dir / "report.json").read_text()
    assert "test-key-must-not-be-persisted" not in serialized
    assert report["report_sha256"] == hashlib.sha256(_canonical_bytes(core)).hexdigest()
    assert len(core["unmapped_rows_not_judged"]["curator"]) == 1
    assert sum(item["type"] == "image_url" for item in case.state.calls[0]["messages"][1]["content"]) == 3


def test_source_judge_known_invalid_crop_remains_visible_without_sending_broken_pixels(
    recipe: ModuleType, judge_case: SimpleNamespace
) -> None:
    case = judge_case
    row = case.packet["core"]["pages"][0]["delivered"]["curator"][2]
    payload = case.source.image_bytes[:45]
    digest = hashlib.sha256(payload).hexdigest()
    (case.packet_path.parent / "blobs" / digest).write_bytes(payload)
    row["values"]["binary_content"] = {"path": f"blobs/{digest}", "sha256": digest, "byte_length": len(payload)}
    row["image_decoding"] = {"status": "invalid", "issue": "controlled failed pixel decoding"}
    case.packet["report_sha256"] = hashlib.sha256(_canonical_bytes(case.packet["core"])).hexdigest()
    case.packet_path.write_text(json.dumps(case.packet))
    report = _run_judge(recipe, case)
    first = case.state.calls[0]["messages"][1]["content"]
    candidate = json.loads(first[0]["text"])["candidate_data"]["B"][2]
    assert candidate["modality"] == "image"
    assert candidate["image_state"] == "invalid"
    assert sum(item["type"] == "image_url" for item in first) == 2
    assert report["core"]["pages"][0]["row_mapping"][candidate["ref"]]["row_id"] == row["row_id"]


@pytest.mark.parametrize(
    "failure",
    [
        "refusal",
        "length",
        "two_choices",
        "empty",
        "duplicate_keys",
        "nan",
        "overflow",
        "unknown_field",
        "facet",
        "category",
        "foreign_ref",
        "wrong_route",
        "duplicate_ref",
        "bbox_bool",
        "bbox_string",
        "bbox_zero",
        "bbox_missing",
        "blank_reason",
        "nonfinite_usage",
    ],
)
def test_source_judge_rejects_bad_responses_without_approval(  # noqa: C901, PLR0915
    recipe: ModuleType, judge_case: SimpleNamespace, failure: str
) -> None:
    case = judge_case

    def change(raw: dict[str, Any]) -> None:  # noqa: C901, PLR0912
        choice = raw["choices"][0]
        message = choice["message"]
        value = json.loads(message["content"])
        finding = value["findings"][0]
        if failure == "refusal":
            message["refusal"] = "cannot answer"
        elif failure == "length":
            choice["finish_reason"] = "length"
        elif failure == "two_choices":
            raw["choices"].append(copy.deepcopy(choice))
        elif failure == "empty":
            message["content"] = " "
        elif failure == "duplicate_keys":
            message["content"] = '{"findings":[],"findings":[]}'
        elif failure == "nan":
            message["content"] = '{"findings":[],"bad":NaN}'
        elif failure == "overflow":
            message["content"] = '{"findings":[],"bad":1e400}'
        elif failure == "nonfinite_usage":
            raw["usage"] = {"total_tokens": float("nan")}
        else:
            if failure == "unknown_field":
                value["accuracy"] = 1.0
            elif failure == "facet":
                finding["facet"] = "accuracy"
            elif failure == "category":
                finding["category"] = "approved"
            elif failure == "foreign_ref":
                finding["a_refs"] = ["unknown-row"]
            elif failure == "wrong_route":
                finding["a_refs"] = finding["b_refs"] or ["unknown-row"]
            elif failure == "duplicate_ref":
                finding["a_refs"] = finding["a_refs"] * 2
            elif failure == "bbox_bool":
                finding["source_bbox_norm"] = [False, 0, 1, 1]
            elif failure == "bbox_string":
                finding["source_bbox_norm"] = ["0", 0, 1, 1]
            elif failure == "bbox_zero":
                finding["source_bbox_norm"] = [0, 0, 0, 1]
            elif failure == "bbox_missing":
                finding["category"] = "a_only_supported"
                finding["source_bbox_norm"] = None
            else:
                finding["reason"] = " "
            message["content"] = json.dumps(value)

    case.state.response_change = change
    report = _run_judge(recipe, case)
    assert report["core"]["status"] == "unjudged_error"
    assert report["core"]["human_calibrated"] is False
    assert all(page["status"] == "unjudged_error" and not page["findings"] for page in report["core"]["pages"])
    assert all(page["raw_response"] is not None for page in report["core"]["pages"])
    assert len(case.state.calls) == 2
    assert case.state.closed == 1


@pytest.mark.parametrize("failure", ["transport", "setup", "cleanup"])
def test_source_judge_client_errors_are_reported_and_closed(
    recipe: ModuleType, judge_case: SimpleNamespace, failure: str
) -> None:
    case = judge_case
    if failure == "transport":
        case.state.request_error = ConnectionError("sensitive-body-must-not-be-recorded")
    elif failure == "setup":
        case.state.setup_error = True
    else:
        case.state.close_error = True
    report = _run_judge(recipe, case)
    assert report["core"]["status"] == "unjudged_error"
    assert report["core"]["report_only"] is True
    assert case.state.closed == (0 if failure == "setup" else 1)
    assert len(case.state.calls) == (0 if failure == "setup" else 2)
    assert "sensitive-body" not in json.dumps(report)


@pytest.mark.parametrize("failure", ["seal", "blob", "blob_path", "source_pixels", "row_identity"])
def test_source_judge_tampering_rejected_before_client_setup(
    recipe: ModuleType, judge_case: SimpleNamespace, failure: str
) -> None:
    case = judge_case
    if failure == "seal":
        case.packet["report_sha256"] = "0" * 64
    elif failure == "blob":
        blob = case.packet_path.parent / case.packet["core"]["pages"][0]["assets"]["source_image"]["path"]
        blob.write_bytes(blob.read_bytes() + b"mutation")
    else:
        page = case.packet["core"]["pages"][0]
        if failure == "blob_path":
            page["assets"]["source_image"]["path"] = "../other.png"
        elif failure == "source_pixels":
            page["assets"]["source_image"]["decoding"]["width"] += 1
        else:
            page["delivered"]["nrl"][0]["row_id"] = "foreign-row"
        case.packet["report_sha256"] = hashlib.sha256(_canonical_bytes(case.packet["core"])).hexdigest()
    case.packet_path.write_text(json.dumps(case.packet))
    with pytest.raises((recipe.EvidenceError, RuntimeError)):
        _run_judge(recipe, case)
    assert case.state.initialized == []
    assert not case.output_dir.exists()


@pytest.mark.parametrize("asset", ["source_image", "historical_raw_response"])
def test_source_judge_rejects_valid_assets_swapped_between_source_pages(
    recipe: ModuleType, judge_case: SimpleNamespace, asset: str
) -> None:
    case = judge_case
    first, second = case.packet["core"]["pages"]
    assert first["assets"][asset]["sha256"] != second["assets"][asset]["sha256"]
    first["assets"][asset], second["assets"][asset] = second["assets"][asset], first["assets"][asset]
    case.packet["report_sha256"] = hashlib.sha256(_canonical_bytes(case.packet["core"])).hexdigest()
    case.packet_path.write_text(json.dumps(case.packet))
    with pytest.raises(recipe.EvidenceError, match="source-page capture"):
        _run_judge(recipe, case)
    assert case.state.initialized == []
    assert not case.output_dir.exists()


@pytest.mark.parametrize("target", ["packet", "blob"])
def test_source_judge_detects_input_drift_before_report(
    recipe: ModuleType, judge_case: SimpleNamespace, target: str
) -> None:
    case = judge_case
    path = (
        case.packet_path
        if target == "packet"
        else case.packet_path.parent / case.packet["core"]["pages"][0]["assets"]["source_image"]["path"]
    )
    case.state.during_request = lambda: path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(recipe.EvidenceError, match="changed"):
        _run_judge(recipe, case)
    assert case.state.closed == 1
    assert not (case.output_dir / "report.json").exists()


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "file:///tmp/judge",
        "https://user:secret@judge.test/v1",
        "https://judge.test/v1?api_key=secret",
        "https://judge.test/v1#secret",
    ],
)
def test_source_judge_requires_explicit_safe_endpoint(
    recipe: ModuleType, judge_case: SimpleNamespace, endpoint: str
) -> None:
    case = judge_case
    with pytest.raises(recipe.EvidenceError, match="URL"):
        recipe.run_source_judge(
            source_evidence=case.packet_path, output_dir=case.output_dir, base_url=endpoint, model="explicit"
        )
    assert case.state.initialized == []


def test_source_judge_cli_is_opt_in_and_refuses_existing_destination(
    recipe: ModuleType, judge_case: SimpleNamespace
) -> None:
    case = judge_case
    flags = ["judge", "--source-evidence", str(case.packet_path), "--output-dir", str(case.output_dir)]
    with pytest.raises(SystemExit):
        recipe._build_parser().parse_args(flags)
    flags.extend(["--base-url", "https://judge.example.test/v1", "--model", "explicit-test-model"])
    assert recipe.main(flags) == 0
    before = (case.output_dir / "report.json").read_bytes()
    assert recipe.main(flags) == 2
    assert (case.output_dir / "report.json").read_bytes() == before
    assert len(case.state.calls) == 2


@pytest.fixture
def benchmark_case(recipe: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    monkeypatch.syspath_prepend(str(Path(recipe.__file__).parent))
    import nrl_lance_contract as contract

    monkeypatch.setattr(contract, "ALLOWED_ROOT", tmp_path)
    corpus = tmp_path / "originals"
    corpus.mkdir()
    source = corpus / "original.pdf"
    _benchmark_pdf(source)
    manifest = tmp_path / "pilot.jsonl"
    manifest.write_text(json.dumps({"path": str(source)}) + "\n")
    destination = tmp_path / "cohort"
    cohort = recipe.prepare_benchmark(manifest=str(manifest), corpus_root=str(corpus), output_dir=str(destination))
    config = {
        "nrl_python": "/nrl/bin/python",
        "curator_python": "/curator/bin/python",
        "model_snapshot": str(tmp_path / "hub/models--parse/snapshots" / ("2" * 40)),
    }
    baseline = contract._seal_payload(
        {
            "config": config,
            "fingerprint": {
                "manifests": {
                    str(destination / "manifest.jsonl"): {"sha256": cohort["core"]["artifacts"]["manifest.jsonl"]}
                }
            },
        },
        "baseline_sha256",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    review = {
        "human_reviewed": True,
        "human_signoff": {
            "decision": "approved",
            "owner": "test owner",
            "reviewer": "test reviewer",
            "signed_at_utc": "2026-09-21T00:00:00Z",
        },
        "qualification_fingerprint": {"sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()},
        "pages": [{"human_reviewed": True}],
        "limitations_register": [{"human_decision": "accepted"}],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    args = SimpleNamespace(
        **config,
        cohort=str(destination / "cohort.json"),
        baseline=str(baseline_path),
        human_review=str(review_path),
        output_dir=str(tmp_path / "measured"),
        nrl_repo="/nrl",
        observer="/observer.py",
        gpu="0",
        projected_temporary_bytes=1000,
        repetitions=3,
    )
    return SimpleNamespace(
        args=args,
        contract=contract,
        cohort=cohort,
        source=source,
        manifest=manifest,
        corpus=corpus,
        destination=destination,
        baseline=baseline,
        review=review,
        review_path=review_path,
    )


def test_prepare_benchmark_accounts_for_every_selection_without_extraction(
    recipe: ModuleType, benchmark_case: SimpleNamespace, tmp_path: Path
) -> None:
    case = benchmark_case
    alias = case.corpus / "alias.pdf"
    alias.write_bytes(case.source.read_bytes())
    outside = tmp_path / "outside.pdf"
    _benchmark_pdf(outside)
    blank = case.corpus / "blank.pdf"
    _benchmark_pdf(blank, 1)
    long = case.corpus / "long.pdf"
    _benchmark_pdf(long, 51)
    corrupt = case.corpus / "corrupt.pdf"
    corrupt.write_bytes(b"not a readable PDF")
    rows = [
        {"path": str(case.source)},
        {"path": str(alias)},
        {"path": str(outside)},
        {"path": str(blank), "valid_blank_pages": [0]},
        {"path": str(long)},
        {"path": str(corrupt)},
    ]
    case.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = recipe.prepare_benchmark(
        manifest=str(case.manifest), corpus_root=str(case.corpus), output_dir=str(tmp_path / "selected")
    )
    accounting = report["core"]["input_accounting"]
    assert [entry["benchmark_selection"] for entry in accounting] == [
        "selected",
        "duplicate",
        "outside_original_corpus",
        "declared_all_blank",
        "over_native_page_limit",
        "preflight_failed",
    ]
    assert len(accounting) == len(rows)
    assert accounting[-1]["status"] == "failed"
    assert accounting[-1]["preflight_error"] is not None
    assert report["core"]["expected_pages"] == 2
    assert [entry["path"] for entry in report["core"]["inputs"]] == [str(case.source)]
    assert report["core"]["inputs"][0]["status"] == "pending"
    staged = tmp_path / "selected/pdfs" / f"{report['core']['inputs'][0]['content_sha256']}.pdf"
    assert staged.read_bytes() == case.source.read_bytes()


@pytest.mark.parametrize(
    "target", ["source", "staged_pdf", "pilot_manifest", "manifest.jsonl", "native_manifest.jsonl"]
)
def test_benchmark_cohort_rejects_mutated_inputs(
    recipe: ModuleType, benchmark_case: SimpleNamespace, target: str
) -> None:
    case = benchmark_case
    if target == "source":
        path = case.source
    elif target == "pilot_manifest":
        path = case.manifest
    elif target == "staged_pdf":
        path = next((case.destination / "pdfs").glob("*.pdf"))
    else:
        path = case.destination / target
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises((recipe.EvidenceError, RuntimeError), match=r"changed|hash differs"):
        recipe._load_benchmark_cohort(case.destination / "cohort.json")


@pytest.mark.parametrize("failure", ["pending", "stale_baseline", "unreviewed_page", "unaccepted_limitation"])
def test_benchmark_quality_gate_prevents_any_execution(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    case = benchmark_case
    review = copy.deepcopy(case.review)
    if failure == "pending":
        review["human_signoff"]["decision"] = "pending"
    elif failure == "stale_baseline":
        review["qualification_fingerprint"]["sha256"] = "0" * 64
    elif failure == "unreviewed_page":
        review["pages"][0]["human_reviewed"] = False
    else:
        review["limitations_register"][0]["human_decision"] = "pending"
    case.review_path.write_text(json.dumps(review))
    monkeypatch.setattr(
        recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("approval must precede execution")
    )
    with pytest.raises(recipe.EvidenceError):
        recipe.run_benchmark(case.args)
    assert not Path(case.args.output_dir).exists()


@pytest.mark.parametrize("field", ["nrl_python", "curator_python", "model_snapshot", "cohort_manifest"])
def test_benchmark_frozen_configuration_precedes_execution(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    case = benchmark_case
    if field == "cohort_manifest":
        baseline = copy.deepcopy(case.baseline)
        baseline.pop("baseline_sha256")
        baseline["fingerprint"]["manifests"] = {}
        Path(case.args.baseline).write_text(json.dumps(case.contract._seal_payload(baseline, "baseline_sha256")))
        review = copy.deepcopy(case.review)
        review["qualification_fingerprint"]["sha256"] = hashlib.sha256(
            Path(case.args.baseline).read_bytes()
        ).hexdigest()
        case.review_path.write_text(json.dumps(review))
    else:
        setattr(case.args, field, "/different/runtime")
    monkeypatch.setattr(
        recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("freeze check must precede execution")
    )
    with pytest.raises(recipe.EvidenceError, match=r"differs from frozen baseline|does not bind"):
        recipe.run_benchmark(case.args)
    assert not Path(case.args.output_dir).exists()


@pytest.mark.parametrize(("cpus", "batch_size", "block_rows"), [(1, 64, None), (4, 128, None), (1, 128, 16)])
def test_nrl_benchmark_worker_runs_ingest_then_consume_without_capture(  # noqa: PLR0913
    recipe: ModuleType,
    benchmark_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    cpus: int,
    batch_size: int,
    block_rows: int | None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    case = benchmark_case
    case.args.engine = "nrl"
    case.args.nrl_parse_cpus = cpus
    case.args.nrl_parse_batch_size = batch_size
    case.args.nrl_projection_block_rows = block_rows
    commands = []
    destination = Path(case.args.output_dir)

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        assert kwargs["check"] is True
        assert "--evidence-root" not in command
        if command[2] == "ingest":
            assert command[0] == case.args.nrl_python
            assert "--executor-stats" in command
            assert command[command.index("--parse-cpus") + 1] == str(cpus)
            assert command[command.index("--parse-batch-size") + 1] == str(batch_size)
            if block_rows is None:
                assert "--projection-block-rows" not in command
            else:
                assert command[command.index("--projection-block-rows") + 1] == str(block_rows)
            output = destination / "ingest"
            output.mkdir()
            handoff = case.contract._seal_payload(
                {
                    "models": {"nemotron_parse": {"revision": Path(case.args.model_snapshot).name}},
                    "configuration": {
                        "capture_evidence": False,
                        "parse_cpus": cpus,
                        "parse_batch_size": batch_size,
                        "projection_block_rows": block_rows,
                    },
                    "timings": {},
                    "inputs": case.cohort["core"]["inputs"],
                },
                case.contract._HANDOFF_HASH_FIELD,
            )
            (output / "handoff_manifest.json").write_text(json.dumps(handoff))
            (output / "executor_stats.txt").write_text("test diagnostics")
        else:
            assert command[2] == "consume"
            assert command[0] == case.args.curator_python
            assert command[command.index("--mode") + 1] == "error"
            output = destination / "parquet"
            output.mkdir()
            digest = case.cohort["core"]["inputs"][0]["content_sha256"]
            table = pa.Table.from_pylist(
                [
                    {"sample_id": digest, "position": -1, "modality": "metadata", "text_content": '{"num_pages": 2}'},
                ]
            )
            pq.write_table(table, output / "document.parquet")
            completion = case.contract._seal_payload({"status": "published"}, case.contract._COMPLETION_HASH_FIELD)
            (destination / "ingest/completion_manifest.json").write_text(json.dumps(completion))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recipe.subprocess, "run", run)
    report = recipe._run_benchmark_engine(case.args)
    assert [command[2] for command in commands] == ["ingest", "consume"]
    assert report["core"]["counts"]["input_documents"] == 1
    assert report["core"]["counts"]["published_metadata_pages"] == 2
    assert report["core"]["counts"]["observed_content_pages"] == 0
    assert report["core"]["details"]["completion"]["status"] == "published"
    assert report["core"]["details"]["configuration"]["projection_block_rows"] == block_rows
    assert len(report["core"]["details"]["export_sha256"]) == 1
    assert (destination / "result.json").exists()


@pytest.mark.parametrize("pdfs_per_task", [10, 20])
def test_native_benchmark_worker_passes_existing_pdf_task_control(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, pdfs_per_task: int
) -> None:
    import argparse

    case = benchmark_case
    case.args.engine = "curator"
    case.args.gpu = "0"
    case.args.native_pdfs_per_task = pdfs_per_task
    parser = argparse.ArgumentParser()
    for flag in ("manifest", "pdf-dir", "output-dir", "model-path", "max-pages", "max-tokens"):
        parser.add_argument(f"--{flag}")
    parser.add_argument("--pdfs-per-task", type=int, default=10)
    pipeline = ModuleType("pipeline_utils")
    pipeline.create_nemotron_parse_pdf_argparser = lambda: parser
    monkeypatch.setitem(sys.modules, "pipeline_utils", pipeline)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test-original")

    def execute(args: argparse.Namespace, _gpu: str, **kwargs: Any) -> tuple[None, dict[str, Any]]:
        assert args.pdfs_per_task == pdfs_per_task
        assert kwargs["monitor_resources"] is False
        return None, {
            "snapshot": {"documents": {}, "issues": []},
            "configuration": {"pdfs_per_task": args.pdfs_per_task},
        }

    monkeypatch.setattr(recipe, "_execute_native_export", execute)
    result = recipe._run_benchmark_engine(case.args)
    assert result["core"]["requested_controls"]["native_pdfs_per_task"] == pdfs_per_task
    assert result["core"]["details"]["configuration"]["pdfs_per_task"] == pdfs_per_task


def test_benchmark_output_counts_expose_missing_extra_and_truncated_documents(recipe: ModuleType) -> None:
    inputs = [{"content_sha256": "a", "expected_page_count": 51}, {"content_sha256": "b", "expected_page_count": 2}]
    snapshot = {
        "documents": {
            "a": {"metadata_page_count": 50, "pages": {"0": []}, "positions": [-1]},
            "extra": {"metadata_page_count": 1, "pages": {"0": []}, "positions": [-1]},
        },
        "issues": [{"issue": "controlled output issue"}],
    }
    counts = recipe._benchmark_output_counts(snapshot, inputs)
    assert counts["expected_pages"] == 53
    assert counts["published_documents"] == 2
    assert counts["withheld_or_missing_documents"] == ["b"]
    assert counts["unexpected_documents"] == ["extra"]
    assert counts["metadata_page_count_mismatches"] == ["a"]
    assert counts["observed_content_pages"] == 2
    assert counts["output_issues"] == snapshot["issues"]
    assert counts["delivered_content_page_count"] == 0
    assert counts["complete_document_count"] is None
    assert counts["failed_page_count"] is None
    assert counts["page_accounting_available"] is False


@pytest.fixture
def delivery_case(recipe: ModuleType, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.syspath_prepend(str(Path(recipe.__file__).parent))
    cases = [
        ("a", "partial", [("success", 1), ("failed", 0), ("valid_blank", 0)]),
        ("b", "success", [("success", 2)]),
        ("c", "failed", [("failed", 0), ("failed", 0)]),
        ("d", "valid_blank", [("valid_blank", 0)]),
        ("e", "partial", [("valid_blank", 0), ("failed", 0)]),
    ]
    inputs, outcomes, documents = [], [], {}
    for digest, status, pages in cases:
        page_outcomes = [
            {
                "page_number": index,
                "status": page_status,
                "element_count": count,
                "issues": [{"kind": "unexpected_empty_output", "page_number": index}]
                if page_status == "failed"
                else [],
            }
            for index, (page_status, count) in enumerate(pages)
        ]
        published = status != "failed"
        element_count = 1 + sum(count for _, count in pages) if published else 0
        inputs.append({"content_sha256": digest, "expected_page_count": len(pages), "status": status})
        outcomes.append(
            {
                "content_sha256": digest,
                "expected_page_count": len(pages),
                "status": status,
                "extraction_status": status,
                "publication_status": "handed_off" if published else "not_applicable",
                "validated_page_count": sum(page_status != "failed" for page_status, _ in pages),
                "content_page_count": sum(page_status == "success" for page_status, _ in pages),
                "blank_page_count": sum(page_status == "valid_blank" for page_status, _ in pages),
                "failed_page_count": sum(page_status == "failed" for page_status, _ in pages),
                "page_outcomes": page_outcomes,
                "issues": [issue for page in page_outcomes for issue in page["issues"]],
                "element_count": element_count,
            }
        )
        if published:
            documents[digest] = {
                "metadata_page_count": len(pages),
                "positions": list(range(-1, element_count - 1)),
                "pages": {
                    str(index): [{"modality": "text"}] * count
                    for index, (page_status, count) in enumerate(pages)
                    if page_status == "success"
                },
            }
    return SimpleNamespace(inputs=inputs, outcomes=outcomes, snapshot={"documents": documents, "issues": []})


def test_partial_delivery_counts_do_not_claim_complete_documents(
    recipe: ModuleType, delivery_case: SimpleNamespace
) -> None:
    case = delivery_case
    counts = recipe._benchmark_output_counts(case.snapshot, case.inputs, document_outcomes=case.outcomes)
    assert counts["page_accounting_available"] is True
    assert counts["input_documents"] == 5
    assert counts["published_documents"] == 4
    assert counts["complete_document_count"] == 2
    assert counts["partial_document_count"] == 2
    assert counts["failed_document_count"] == 1
    assert counts["expected_pages"] == 9
    assert counts["failed_page_count"] == 4
    assert counts["delivered_page_count"] == 5
    assert counts["delivered_content_page_count"] == 2
    assert counts["delivered_content_element_count"] == 3
    assert counts["published_metadata_pages"] == 7  # Not the validated-delivery numerator.
    assert counts["rows"] == 7
    assert counts["withheld_or_missing_documents"] == ["c"]
    assert counts["metadata_page_count_mismatches"] == []
    assert case.snapshot["documents"]["e"]["positions"] == [-1]  # Explicit blank-only partial delivery.


@pytest.mark.parametrize(
    "failure",
    ["identities", "page_elements", "derived_count", "failed_page", "missing_page", "status", "failed_delivered"],
)
def test_partial_delivery_accounting_rejects_mismatched_export(
    recipe: ModuleType, delivery_case: SimpleNamespace, failure: str
) -> None:
    case = delivery_case
    if failure == "identities":
        case.snapshot["documents"].pop("a")
    elif failure == "page_elements":
        case.snapshot["documents"]["a"]["pages"]["0"].append({"modality": "text"})
    elif failure == "derived_count":
        case.outcomes[0]["validated_page_count"] += 1
    elif failure == "failed_page":
        case.snapshot["documents"]["a"]["pages"]["1"] = [{"modality": "text"}]
    elif failure == "missing_page":
        case.outcomes[0]["page_outcomes"].pop()
    elif failure == "status":
        case.outcomes[0]["extraction_status"] = "success"
    else:
        case.outcomes[2]["publication_status"] = "handed_off"
        case.outcomes[2]["element_count"] = 1
        case.snapshot["documents"]["c"] = {"metadata_page_count": 2, "positions": [-1], "pages": {}}
    with pytest.raises(ValueError, match=r"benchmark|[Pp]age"):
        recipe._benchmark_output_counts(case.snapshot, case.inputs, document_outcomes=case.outcomes)


def test_legacy_document_results_leave_completeness_unavailable(
    recipe: ModuleType, delivery_case: SimpleNamespace
) -> None:
    case = delivery_case
    for outcome in case.outcomes:
        outcome.pop("page_outcomes")
    counts = recipe._benchmark_output_counts(case.snapshot, case.inputs, document_outcomes=case.outcomes)
    assert counts["page_accounting_available"] is False
    assert counts["complete_document_count"] is None
    assert counts["partial_document_count"] is None
    assert counts["failed_document_count"] is None
    assert counts["failed_page_count"] is None
    assert counts["delivered_page_count"] is None
    assert counts["delivered_content_page_count"] == 2
    assert counts["delivered_content_element_count"] == 3


def test_failed_benchmark_repetitions_cannot_claim_throughput(recipe: ModuleType) -> None:
    runs = [
        {"engine": "nrl", "status": "failed", "wall_seconds": 0.01, "repetition": 0},
        {"engine": "curator", "status": "compared", "wall_seconds": 10.0, "repetition": 0},
    ]
    summary = recipe._summarize_benchmark(runs, 100)
    assert summary["engines"]["nrl"]["successful_repetitions"] == 0
    assert summary["engines"]["nrl"]["offered_expected_pages_per_second"] == []
    assert summary["engines"]["nrl"]["median_seconds"] is None
    assert summary["engines"]["curator"]["offered_expected_pages_per_second"] == [10.0]
    assert summary["paired_differences"] == []
    assert summary["performance_conclusion"].startswith("unselected")


def test_mixed_publication_policies_cannot_form_a_pooled_comparison(recipe: ModuleType) -> None:
    runs = [
        {
            "engine": engine,
            "status": "compared",
            "wall_seconds": 10.0,
            "repetition": repetition,
            "result": {
                "core": {"publication_policy": "complete_documents_v1" if repetition == 0 else "validated_pages_v1"}
            },
        }
        for repetition in range(2)
        for engine in ("nrl", "curator")
    ]
    summary = recipe._summarize_benchmark(runs, 100)
    assert summary["publication_policies_comparable"] is False
    assert summary["paired_differences"] == []
    assert summary["engines"]["nrl"]["median_seconds"] is None
    assert summary["engines"]["nrl"]["wall_seconds"] == [10.0, 10.0]
    assert "unavailable" in summary["accuracy_metrics"]


def test_benchmark_counts_reject_out_of_range_content_page(recipe: ModuleType) -> None:
    snapshot = {
        "documents": {"a": {"metadata_page_count": 1, "pages": {"7": []}, "positions": [-1]}},
        "issues": [],
    }
    counts = recipe._benchmark_output_counts(snapshot, [{"content_sha256": "a", "expected_page_count": 1}])
    assert counts["metadata_page_count_mismatches"] == []
    assert counts["out_of_range_content_pages"]


def _mock_benchmark_children(  # noqa: C901, PLR0915
    recipe: ModuleType,
    case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None = None,
    nrl_counts: dict[str, Any] | None = None,
) -> list[list[str]]:
    calls = []

    def run(command: list[str], **kwargs: Any) -> SimpleNamespace:  # noqa: C901, PLR0912, PLR0915
        if Path(command[0]).name == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="0, GPU-test\n")
        calls.append(command)
        engine = command[command.index("--engine") + 1]
        destination = Path(command[command.index("--output-dir") + 1])
        assert not destination.exists()
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert "--evidence-root" not in command
        assert command[command.index("--interval") + 1] == "1"
        assert "--capacity-check" in command
        destination.mkdir(parents=True)
        observation = {
            "qualification_passed": failure != "observer_failed",
            "wall_seconds": 10.0,
            "baseline_sha256": case.baseline["baseline_sha256"],
            "capacity_enforced": failure != "capacity_not_enforced",
            "returncode": 1 if failure == "observer_child_failed" else 0,
        }
        if failure == "capacity_missing_result":
            observation["qualification_passed"] = False
            observation["qualification_invalidations"] = ["GPU memory headroom below 10%"]
        if failure == "missing_wall":
            observation.pop("wall_seconds")
        elif failure == "zero_wall":
            observation["wall_seconds"] = 0
        elif failure == "observer_baseline":
            observation["baseline_sha256"] = "0" * 64
        summary = case.contract._seal_payload(observation, "summary_sha256")
        if failure == "observer_seal":
            summary["summary_sha256"] = "0" * 64
        prefix = Path(command[command.index("--output-prefix") + 1])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{prefix}.summary.json").write_text(json.dumps(summary))
        counts = recipe._benchmark_output_counts({"documents": {}, "issues": []}, case.cohort["core"]["inputs"])
        if engine == "nrl" and nrl_counts is not None:
            counts = nrl_counts
        if failure == "extra_document":
            counts["unexpected_documents"] = ["0" * 64]
        elif failure == "truncated_document":
            counts["metadata_page_count_mismatches"] = [case.cohort["core"]["inputs"][0]["content_sha256"]]
        elif failure == "out_of_range_page":
            counts["out_of_range_content_pages"] = [
                {"sample_id": case.cohort["core"]["inputs"][0]["content_sha256"], "page": 7}
            ]
        elif failure == "output_issue" or (engine == "curator" and failure == "native_output_issue"):
            counts["output_issues"] = [{"position": 0, "issue": "invalid crop"}]
        elif engine == "curator" and failure in {"native_output_metadata", "native_output_order"}:
            counts["output_issues"] = [{"position": -1, "issue": "invalid metadata"}]
            if failure == "native_output_order":
                counts["output_issues"][0].pop("position")
        core = {
            "schema": "nrl_curator_benchmark_engine",
            "status": "failed" if failure == "result_status" else "compared",
            "engine": "other" if failure == "result_engine" else engine,
            "cohort_sha256": "0" * 64 if failure == "result_cohort" else case.cohort["report_sha256"],
            "model_snapshot": case.args.model_snapshot,
            "counts": counts,
            "details": {
                "configuration": {
                    "parse_cpus": getattr(case.args, "nrl_parse_cpus", 1),
                    "parse_batch_size": getattr(case.args, "nrl_parse_batch_size", 64),
                    "projection_block_rows": getattr(case.args, "nrl_projection_block_rows", None),
                    "pdfs_per_task": getattr(case.args, "native_pdfs_per_task", 10),
                    "gpu_memory_utilization_override": (
                        None
                        if failure == "gpu_configuration"
                        else getattr(case.args, "native_gpu_memory_utilization", None)
                    ),
                }
            },
        }
        if failure in {"parse_cpus", "parse_batch_size", "pdfs_per_task"}:
            core["details"]["configuration"][failure] += 1
        if failure == "projection_block_rows":
            core["details"]["configuration"][failure] = 32
        if engine == "nrl" and (nrl_counts is not None or failure == "missing_page_accounting"):
            core["publication_policy"] = "validated_pages_v1"
        if engine == "nrl" and failure in {"policy_change", "policy_change_with_output_issue"}:
            nrl_run_count = sum(item[item.index("--engine") + 1] == "nrl" for item in calls)
            core["publication_policy"] = "validated_pages_v1" if nrl_run_count == 1 else "complete_documents_v1"
            counts["page_accounting_available"] = True
            if failure == "policy_change_with_output_issue" and nrl_run_count > 1:
                counts["output_issues"] = [{"position": 0, "issue": "invalid crop"}]
        result = {"core": core, "report_sha256": hashlib.sha256(_canonical_bytes(core)).hexdigest()}
        if failure == "result_seal":
            result["report_sha256"] = "0" * 64
        if failure not in {"missing_result", "capacity_missing_result"}:
            (destination / "result.json").write_text(
                "not JSON" if failure == "malformed_result" else json.dumps(result)
            )
        return SimpleNamespace(returncode=1 if failure in {"process_failed", "capacity_missing_result"} else 0)

    monkeypatch.setattr(recipe.subprocess, "run", run)
    return calls


def test_benchmark_runs_alternating_complete_products_in_fresh_directories(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_benchmark_children(recipe, benchmark_case, monkeypatch)
    report = recipe.run_benchmark(benchmark_case.args)
    assert [command[command.index("--engine") + 1] for command in calls] == [
        "nrl",
        "curator",
        "curator",
        "nrl",
        "nrl",
        "curator",
    ]
    assert len({command[command.index("--output-dir") + 1] for command in calls}) == 6
    assert report["core"]["status"] == "compared"
    assert [run["repetition"] for run in report["core"]["runs"]] == [0, 0, 1, 1, 2, 2]
    assert all(run["result"]["core"]["counts"]["withheld_or_missing_documents"] for run in report["core"]["runs"])
    assert report["core"]["engines"]["nrl"]["offered_expected_pages_per_second"] == [0.2] * 3
    assert all(run["observed_content_pages_per_second"] == 0 for run in report["core"]["runs"])
    assert all(run["validated_published_pages_per_second"] is None for run in report["core"]["runs"])
    assert report["report_sha256"] == hashlib.sha256(_canonical_bytes(report["core"])).hexdigest()
    assert json.loads((Path(benchmark_case.args.output_dir) / "benchmark_report.json").read_text()) == report


def test_partial_benchmark_uses_delivered_pages_not_source_metadata(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = benchmark_case
    digest = case.cohort["core"]["inputs"][0]["content_sha256"]
    snapshot = {
        "documents": {
            digest: {"metadata_page_count": 2, "pages": {"0": [{"modality": "text"}]}, "positions": [-1, 0]}
        },
        "issues": [],
    }
    issue = {"kind": "unexpected_empty_output", "page_number": 1}
    outcome = {
        "content_sha256": digest,
        "status": "partial",
        "extraction_status": "partial",
        "publication_status": "handed_off",
        "validated_page_count": 1,
        "content_page_count": 1,
        "blank_page_count": 0,
        "failed_page_count": 1,
        "element_count": 2,
        "page_outcomes": [
            {"page_number": 0, "status": "success", "element_count": 1, "issues": []},
            {"page_number": 1, "status": "failed", "element_count": 0, "issues": [issue]},
        ],
        "issues": [issue],
    }
    counts = recipe._benchmark_output_counts(snapshot, case.cohort["core"]["inputs"], document_outcomes=[outcome])
    _mock_benchmark_children(recipe, case, monkeypatch, nrl_counts=counts)
    report = recipe.run_benchmark(case.args)
    for run in report["core"]["runs"]:
        if run["engine"] == "nrl":
            assert run["validated_published_pages_per_second"] == 0.1
            assert run["attempted_pages_per_second"] == 0.2
            assert run["delivered_content_pages_per_second"] == 0.1
            assert run["delivered_content_elements_per_second"] == 0.1
            assert run["result"]["core"]["counts"]["complete_document_count"] == 0
            assert run["result"]["core"]["counts"]["partial_document_count"] == 1


@pytest.mark.parametrize("failure", ["policy_change", "policy_change_with_output_issue"])
def test_benchmark_rejects_a_policy_change_between_repetitions(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    benchmark_case.args.diagnostic = True
    calls = _mock_benchmark_children(recipe, benchmark_case, monkeypatch, failure)
    report = recipe.run_benchmark(benchmark_case.args)
    assert len(calls) == 4
    assert report["core"]["status"] == "failed"
    assert "publication policy changed" in report["core"]["runs"][-1]["failure"]
    assert report["core"]["publication_policies_comparable"] is False
    assert report["core"]["paired_differences"] == []


@pytest.mark.parametrize(
    "failure",
    [
        "observer_failed",
        "capacity_not_enforced",
        "observer_child_failed",
        "observer_baseline",
        "observer_seal",
        "missing_wall",
        "zero_wall",
        "process_failed",
        "capacity_missing_result",
        "missing_result",
        "malformed_result",
        "result_status",
        "result_engine",
        "result_cohort",
        "result_seal",
        "extra_document",
        "truncated_document",
        "out_of_range_page",
        "output_issue",
        "missing_page_accounting",
    ],
)
def test_benchmark_failure_is_sealed_without_success_throughput(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    calls = _mock_benchmark_children(recipe, benchmark_case, monkeypatch, failure)
    report = recipe.run_benchmark(benchmark_case.args)
    assert len(calls) == 1
    assert report["core"]["status"] == "failed"
    assert report["core"]["runs"][0]["status"] == "failed"
    if failure == "capacity_missing_result":
        assert "GPU memory headroom below 10%" in report["core"]["runs"][0]["failure"]
    assert "observed_content_pages_per_second" not in report["core"]["runs"][0]
    assert report["core"]["engines"]["nrl"]["offered_expected_pages_per_second"] == []
    assert report["report_sha256"] == hashlib.sha256(_canonical_bytes(report["core"])).hexdigest()
    assert json.loads((Path(benchmark_case.args.output_dir) / "benchmark_report.json").read_text()) == report


def test_diagnostic_benchmark_collects_without_changing_human_approval(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = benchmark_case
    case.args.diagnostic = True
    case.args.human_review = None
    case.args.native_gpu_memory_utilization = 0.85
    review_before = case.review_path.read_bytes()
    calls = _mock_benchmark_children(recipe, case, monkeypatch)
    report = recipe.run_benchmark(case.args)
    assert report["core"]["status"] == "diagnostic_completed"
    assert report["core"]["quality_status"] == "pending_human_review"
    assert report["core"]["human_review"] is None
    assert report["core"]["qualified"] is False
    assert case.review_path.read_bytes() == review_before
    assert len(calls) == 6
    assert all(command[command.index("--native-gpu-memory-utilization") + 1] == "0.85" for command in calls)
    assert all("--capacity-check" in command for command in calls)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_only_diagnostic_native_output_findings_allow_more_measurements(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, diagnostic: bool
) -> None:
    case = benchmark_case
    case.args.diagnostic = diagnostic
    calls = _mock_benchmark_children(recipe, case, monkeypatch, "native_output_issue")
    report = recipe.run_benchmark(case.args)
    core = report["core"]
    assert len(calls) == (6 if diagnostic else 2)
    assert core["status"] == "failed"
    assert core["collection_complete"] is diagnostic
    assert core["output_validation_passed"] is False
    assert core["paired_differences"] == []
    native = core["engines"]["curator"]
    assert native["successful_repetitions"] == 0
    assert native["failed_repetitions"] == (3 if diagnostic else 1)
    assert native["operational_wall_seconds"] == [10.0] * (3 if diagnostic else 1)
    assert native["operational_total_seconds"] == (30.0 if diagnostic else 10.0)
    assert native["wall_seconds"] == []
    for run in core["runs"]:
        if run["engine"] == "curator":
            assert run["status"] == "failed"
            assert run["failure_kind"] == "output_validation"
            assert run["execution_evidence_validated"] is True
            assert run["diagnostic_continued"] is diagnostic
            assert run["observer"]["qualification_passed"] is True
            assert run["result"]["core"]["counts"]["output_issues"]
            assert "attempted_pages_per_second" not in run


@pytest.mark.parametrize(
    "failure",
    ["output_issue", "observer_failed", "observer_seal", "process_failed", "capacity_not_enforced", "result_cohort"],
)
def test_diagnostic_does_not_continue_nrl_output_or_fatal_evidence_failures(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    benchmark_case.args.diagnostic = True
    calls = _mock_benchmark_children(recipe, benchmark_case, monkeypatch, failure)
    report = recipe.run_benchmark(benchmark_case.args)
    assert len(calls) == 1
    assert report["core"]["status"] == "failed"
    assert report["core"]["collection_complete"] is False
    assert report["core"]["runs"][0]["diagnostic_continued"] is False


@pytest.mark.parametrize("field", ["parse_cpus", "parse_batch_size", "pdfs_per_task", "projection_block_rows"])
def test_benchmark_rejects_ignored_effective_tuning_controls(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    case = benchmark_case
    case.args.diagnostic = True
    calls = _mock_benchmark_children(recipe, case, monkeypatch, field)
    report = recipe.run_benchmark(case.args)
    assert len(calls) == (2 if field == "pdfs_per_task" else 1)
    assert report["core"]["status"] == "failed"
    assert f"{field} configuration differs" in report["core"]["runs"][-1]["failure"]
    assert report["core"]["runs"][-1]["diagnostic_continued"] is False


@pytest.mark.parametrize(
    ("requested", "effective", "accepted"),
    [
        (None, "missing", True),
        (None, None, True),
        (None, 16, False),
        (16, "missing", False),
        (16, None, False),
        (16, 16.0, False),
        (16, True, False),
        (16, 32, False),
        (16, 16, True),
    ],
)
def test_benchmark_checks_effective_projection_block_rows(  # noqa: PLR0913
    recipe: ModuleType,
    benchmark_case: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    requested: int | None,
    effective: Any,
    accepted: bool,
) -> None:
    case = benchmark_case
    _mock_benchmark_children(recipe, case, monkeypatch)
    attempt = recipe.run_benchmark(case.args)["core"]["runs"][0]
    result = copy.deepcopy(attempt["result"])
    configuration = result["core"]["details"]["configuration"]
    if effective == "missing":
        configuration.pop("projection_block_rows")
    else:
        configuration["projection_block_rows"] = effective
    result["report_sha256"] = hashlib.sha256(_canonical_bytes(result["core"])).hexdigest()
    arguments = {
        "baseline_sha256": case.baseline["baseline_sha256"],
        "cohort_sha256": case.cohort["report_sha256"],
        "engine": "nrl",
        "model_snapshot": case.args.model_snapshot,
        "nrl_projection_block_rows": requested,
    }
    if accepted:
        recipe._validate_benchmark_run(attempt["observer"], result, **arguments)
    else:
        with pytest.raises(recipe.EvidenceError, match="projection_block_rows configuration differs"):
            recipe._validate_benchmark_run(attempt["observer"], result, **arguments)


@pytest.mark.parametrize("failure", ["native_output_metadata", "native_output_order"])
def test_diagnostic_native_document_structure_findings_remain_fatal(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    benchmark_case.args.diagnostic = True
    calls = _mock_benchmark_children(recipe, benchmark_case, monkeypatch, failure)
    report = recipe.run_benchmark(benchmark_case.args)
    assert len(calls) == 2
    assert report["core"]["status"] == "failed"
    assert report["core"]["collection_complete"] is False
    assert report["core"]["runs"][-1]["failure_kind"] == "execution_or_evidence"
    assert report["core"]["runs"][-1]["diagnostic_continued"] is False


@pytest.mark.parametrize("guard", ["qualification_passed", "capacity_enforced", "returncode", "pdfs_per_task"])
def test_native_element_findings_do_not_mask_execution_or_configuration_failure(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, guard: str
) -> None:
    case = benchmark_case
    _mock_benchmark_children(recipe, case, monkeypatch, "native_output_issue")
    report = recipe.run_benchmark(case.args)
    attempt = report["core"]["runs"][-1]
    observation = copy.deepcopy(attempt["observer"])
    result = copy.deepcopy(attempt["result"])
    if guard == "pdfs_per_task":
        result["core"]["details"]["configuration"][guard] = 20
        result["report_sha256"] = hashlib.sha256(_canonical_bytes(result["core"])).hexdigest()
    else:
        observation.pop("summary_sha256")
        observation[guard] = 1 if guard == "returncode" else False
        observation = case.contract._seal_payload(observation, "summary_sha256")
    with pytest.raises(recipe.EvidenceError) as failure:
        recipe._validate_benchmark_run(
            observation,
            result,
            baseline_sha256=case.baseline["baseline_sha256"],
            cohort_sha256=case.cohort["report_sha256"],
            engine="curator",
            model_snapshot=case.args.model_snapshot,
        )
    assert not isinstance(failure.value, recipe.BenchmarkOutputError)


@pytest.mark.parametrize(
    "controls",
    [
        {"nrl_parse_cpus": 1, "nrl_parse_batch_size": 64, "native_pdfs_per_task": 10},
        {"nrl_parse_cpus": 4, "nrl_parse_batch_size": 128, "native_pdfs_per_task": 20},
        {
            "nrl_parse_cpus": 1,
            "nrl_parse_batch_size": 128,
            "native_pdfs_per_task": 20,
            "nrl_projection_block_rows": 16,
        },
    ],
)
def test_benchmark_forwards_and_records_tuning_controls(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, controls: dict[str, int]
) -> None:
    for name, value in controls.items():
        setattr(benchmark_case.args, name, value)
    calls = _mock_benchmark_children(recipe, benchmark_case, monkeypatch)
    report = recipe.run_benchmark(benchmark_case.args)
    assert report["core"]["status"] == "compared"
    assert report["core"]["requested_controls"] == controls
    for command in calls:
        for name, value in controls.items():
            assert command[command.index(f"--{name.replace('_', '-')}") + 1] == str(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nrl_parse_cpus", 0),
        ("nrl_parse_cpus", True),
        ("nrl_parse_cpus", 1.5),
        ("nrl_parse_batch_size", 1),
        ("nrl_parse_batch_size", 0),
        ("nrl_parse_batch_size", False),
        ("native_pdfs_per_task", 0),
        ("native_pdfs_per_task", True),
        ("nrl_projection_block_rows", 0),
        ("nrl_projection_block_rows", -1),
        ("nrl_projection_block_rows", True),
        ("nrl_projection_block_rows", 1.5),
    ],
)
def test_benchmark_invalid_controls_fail_before_launch(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    setattr(benchmark_case.args, field, value)
    monkeypatch.setattr(recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must fail before launch"))
    with pytest.raises(recipe.EvidenceError, match=field):
        recipe.run_benchmark(benchmark_case.args)
    assert not Path(benchmark_case.args.output_dir).exists()


@pytest.mark.parametrize("command", ["benchmark", "_benchmark-engine"])
def test_benchmark_cli_control_defaults_and_overrides(recipe: ModuleType, command: str) -> None:
    required = [
        argument
        for name in ("cohort", "output-dir", "nrl-python", "curator-python", "nrl-repo", "model-snapshot", "gpu")
        for argument in (f"--{name}", "test")
    ]
    required += (
        ["--engine", "nrl"]
        if command == "_benchmark-engine"
        else ["--observer", "observer", "--baseline", "baseline", "--projected-temporary-bytes", "1"]
    )
    parser = recipe._build_parser()
    assert recipe._benchmark_controls(parser.parse_args([command, *required])) == {
        "nrl_parse_cpus": 1,
        "nrl_parse_batch_size": 64,
        "native_pdfs_per_task": 10,
    }
    actual = parser.parse_args(
        [
            command,
            *required,
            "--nrl-parse-cpus",
            "4",
            "--nrl-parse-batch-size",
            "128",
            "--native-pdfs-per-task",
            "20",
            "--nrl-projection-block-rows",
            "16",
        ]
    )
    assert recipe._benchmark_controls(actual) == {
        "nrl_parse_cpus": 4,
        "nrl_parse_batch_size": 128,
        "native_pdfs_per_task": 20,
        "nrl_projection_block_rows": 16,
    }


def test_unsigned_benchmark_requires_explicit_diagnostic_mode(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = benchmark_case
    case.args.human_review = None
    monkeypatch.setattr(recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must fail before launch"))
    with pytest.raises(recipe.EvidenceError, match="human quality approval"):
        recipe.run_benchmark(case.args)
    assert not Path(case.args.output_dir).exists()


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_benchmark_rejects_invalid_memory_reservations_before_execution(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, fraction: float
) -> None:
    case = benchmark_case
    case.args.diagnostic = True
    case.args.native_gpu_memory_utilization = fraction
    monkeypatch.setattr(recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must fail before launch"))
    with pytest.raises(recipe.EvidenceError, match="GPU memory utilization"):
        recipe.run_benchmark(case.args)


def test_benchmark_rejects_ignored_native_memory_configuration(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = benchmark_case
    case.args.diagnostic = True
    case.args.native_gpu_memory_utilization = 0.85
    calls = _mock_benchmark_children(recipe, case, monkeypatch, "gpu_configuration")
    report = recipe.run_benchmark(case.args)
    assert len(calls) == 2
    assert report["core"]["status"] == "failed"
    assert "GPU memory configuration differs" in report["core"]["runs"][-1]["failure"]


def test_full_corpus_selection_accounts_for_long_blank_and_duplicate_inputs(
    recipe: ModuleType, benchmark_case: SimpleNamespace, tmp_path: Path
) -> None:
    case = benchmark_case
    long = case.corpus / "long.pdf"
    blank = case.corpus / "blank.pdf"
    alias = case.corpus / "alias.pdf"
    _benchmark_pdf(long, 51)
    _benchmark_pdf(blank, 1)
    alias.write_bytes(long.read_bytes())
    case.manifest.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"path": str(case.source)},
                {"path": str(long)},
                {"path": str(alias)},
                {"path": str(blank), "valid_blank_pages": [0]},
            ]
        )
    )
    destination = tmp_path / "full"
    report = recipe.prepare_benchmark(
        manifest=str(case.manifest), corpus_root=str(case.corpus), output_dir=str(destination), selection="full-corpus"
    )
    assert len(report["core"]["inputs"]) == 3
    assert len(report["core"]["input_accounting"]) == 4
    assert report["core"]["expected_pages"] == 54
    assert report["core"]["native_attempted_pages"] == 53
    assert recipe._load_benchmark_cohort(destination / "cohort.json") == report
    report["core"]["native_attempted_pages"] += 1
    report["report_sha256"] = hashlib.sha256(_canonical_bytes(report["core"])).hexdigest()
    (destination / "cohort.json").write_text(json.dumps(report))
    with pytest.raises(recipe.EvidenceError, match="attempted-page denominator"):
        recipe._load_benchmark_cohort(destination / "cohort.json")


def test_native_cap_is_explicit_not_a_complete_document_comparison(recipe: ModuleType) -> None:
    inputs = [{"content_sha256": "a", "expected_page_count": 1080}]
    snapshot = {"documents": {"a": {"metadata_page_count": 50, "pages": {"49": []}, "positions": [-1]}}, "issues": []}
    counts = recipe._benchmark_output_counts(snapshot, inputs, native_page_cap=50)
    assert counts["expected_pages"] == 1080
    assert counts["attempted_pages"] == 50
    assert counts["cap_truncated_documents"] == ["a"]
    assert counts["cap_omitted_pages"] == 1030
    assert counts["metadata_page_count_mismatches"] == []
    assert counts["out_of_range_content_pages"] == []
    snapshot["documents"]["a"]["pages"]["50"] = []
    assert recipe._benchmark_output_counts(snapshot, inputs, native_page_cap=50)["out_of_range_content_pages"]
    runs = [
        {"engine": engine, "status": "compared", "wall_seconds": 10.0, "repetition": 0}
        for engine in ("nrl", "curator")
    ]
    summary = recipe._summarize_benchmark(runs, 1080, paired_complete_scope=False)
    assert summary["paired_complete_scope"] is False
    assert summary["paired_differences"] == []


def test_native_memory_override_uses_existing_stage_without_changing_default(
    recipe: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("nemo_curator")
    monkeypatch.syspath_prepend(str(Path(recipe.__file__).parent))
    from pipeline_utils import create_nemotron_parse_pdf_argparser, create_nemotron_parse_pdf_pipeline

    from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage

    args = create_nemotron_parse_pdf_argparser().parse_args(
        ["--manifest", "/manifest.jsonl", "--pdf-dir", "/pdfs", "--output-dir", str(tmp_path / "output")]
    )
    pipeline = create_nemotron_parse_pdf_pipeline(args)
    recipe._configure_native_gpu_memory(pipeline, 0.85)
    inference = [stage for stage in pipeline.stages if isinstance(stage, NemotronParseInferenceStage)]
    assert len(inference) == 1
    assert inference[0].engine_kwargs == {"gpu_memory_utilization": 0.85}
    ordinary = create_nemotron_parse_pdf_pipeline(args)
    ordinary.build()
    assert (
        next(stage for stage in ordinary.stages if isinstance(stage, NemotronParseInferenceStage)).engine_kwargs
        is None
    )


def test_full_corpus_benchmark_requires_diagnostic_even_with_human_approval(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = benchmark_case
    case.cohort["core"]["selection"] = "full-corpus"
    case.cohort["report_sha256"] = hashlib.sha256(_canonical_bytes(case.cohort["core"])).hexdigest()
    (case.destination / "cohort.json").write_text(json.dumps(case.cohort))
    monkeypatch.setattr(recipe.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must fail before launch"))
    with pytest.raises(recipe.EvidenceError, match="requires --diagnostic"):
        recipe.run_benchmark(case.args)


def test_diagnostic_full_scope_never_reports_paired_speed_difference(
    recipe: ModuleType, benchmark_case: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = benchmark_case
    case.args.diagnostic = True
    case.cohort["core"]["selection"] = "full-corpus"
    case.cohort["report_sha256"] = hashlib.sha256(_canonical_bytes(case.cohort["core"])).hexdigest()
    (case.destination / "cohort.json").write_text(json.dumps(case.cohort))
    _mock_benchmark_children(recipe, case, monkeypatch)
    report = recipe.run_benchmark(case.args)
    assert report["core"]["status"] == "diagnostic_completed"
    assert report["core"]["paired_complete_scope"] is False
    assert report["core"]["paired_differences"] == []
