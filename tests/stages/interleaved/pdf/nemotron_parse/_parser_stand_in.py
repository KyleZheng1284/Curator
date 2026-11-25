# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client surface of ``nemotron_parse.inference`` for hosts without the model team's package.

The ``parse_package`` fixture installs this file under that name. It decodes
nothing: each response becomes one text block.
"""

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

PROMPT = "stand-in prompt"


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.0
    max_tokens: int = 64


class DocumentParser:
    def __init__(self, *, model: str, backend: Callable[..., dict]) -> None:
        self.model = model
        self.backend = backend
        self.settings = {"fullpage": GenerationSettings()}

    def parse(self, image: Image.Image, **_kwargs: object) -> dict:
        response = self.backend(image, prompt=PROMPT, stage="fullpage", settings=self.settings["fullpage"])
        block = {"class": "Text", "bbox": [0.1, 0.1, 0.9, 0.9], "content": response["raw_out"]}
        return {**response, "blocks": [block]}

    def close(self) -> None:
        pass
