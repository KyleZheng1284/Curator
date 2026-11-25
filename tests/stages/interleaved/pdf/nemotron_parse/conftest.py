# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def parse_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Yield ``nemotron_parse.inference``, or a stand-in where a test checks only Curator's transport and rows."""
    stand_in = importlib.util.find_spec("nemotron_parse") is None
    if stand_in:
        root = tmp_path / "parser"
        (root / "nemotron_parse").mkdir(parents=True)
        (root / "nemotron_parse" / "__init__.py").touch()
        shutil.copy(Path(__file__).with_name("_parser_stand_in.py"), root / "nemotron_parse" / "inference.py")
        (root / "nemotron_parse-0.dist-info").mkdir()
        (root / "nemotron_parse-0.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: nemotron-parse\nVersion: 0\n"
        )
        monkeypatch.syspath_prepend(root)
        # Ray workers resolve the package from an environment they are given, not from this process.
        monkeypatch.setenv("PYTHONPATH", str(root), prepend=os.pathsep)
    yield importlib.import_module("nemotron_parse.inference")
    if stand_in:
        del sys.modules["nemotron_parse.inference"], sys.modules["nemotron_parse"]
