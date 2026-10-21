# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the two local project wheels consumed by this experimental lock."""
# ruff: noqa: INP001

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

PROFILE = Path(__file__).resolve().parent
SOURCE_DATE_EPOCH = "1792553568"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrl-checkout", type=Path, required=True)
    parser.add_argument("--curator-checkout", type=Path, required=True)
    args = parser.parse_args()
    nrl = args.nrl_checkout.resolve(strict=True)
    curator = args.curator_checkout.resolve(strict=True)
    destination = PROFILE / "wheels"
    destination.mkdir(exist_ok=True)
    environment = dict(os.environ, SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH, NO_VCS_VERSION="1")
    receipt = {"source_date_epoch": SOURCE_DATE_EPOCH, "projects": {}}
    for name, repository, project in (("curator", curator, curator), ("nrl", nrl, nrl / "nemo_retriever")):
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()  # noqa: S607
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "."], cwd=repository)  # noqa: S607
        subprocess.run(  # noqa: S603
            ["uv", "build", "--wheel", "--no-sources", "--out-dir", str(destination), str(project)],  # noqa: S607
            env=environment,
            check=True,
        )
        receipt["projects"][name] = {"commit": head, "diff_sha256": hashlib.sha256(diff).hexdigest()}
    receipt["wheels"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(destination.glob("*.whl"))
    }
    (destination / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
