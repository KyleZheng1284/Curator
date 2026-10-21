# Experimental NRL PDF environment

This separate Python 3.12 environment installs Curator's CPU interleaved stages and
Lance support alongside NRL, PyTorch 2.11/CUDA 13, vLLM 0.25.1, Transformers 5.17,
FlashInfer 0.6.13 and Ray 2.57. It does not upgrade Curator's root environment or
qualify its audio, RAPIDS, SDG, video or other optional features on this stack.

The profile is for Linux x86_64 with a compatible NVIDIA driver. Python headers
are required for runtime kernel compilation; use a uv-managed Python 3.12.

## Prepare and install

Use uv 0.12.1 or newer. The NRL checkout must include the proposed urllib3 2.8
pin update (`b9ef70a4505331ea91b44993bc9f49b1b34615b6`, based on integration
commit `8e60a11bb7296441ee6a2f6dabde3dd4f62d141a`). That proposal is not an
upstream release. Its publication is pending; do not override an older NRL
wheel's urllib3 requirement to force installation.

From this directory:

```bash
uv python install 3.12
python3 build_wheels.py --nrl-checkout /path/to/NeMo-Retriever \
  --curator-checkout /path/to/Curator-pr2480
uv sync --locked --python 3.12
uv pip check --python .venv/bin/python
```

The supplied lock binds the Curator PR checkout at
`d7245ed4f70070bc2db65a36bd2cabd0ddcf48c0` and the NRL revision above.
Use clean checkouts at those revisions. Changing either source requires an
explicit lock refresh and renewed validation; do not ignore a wheel hash mismatch.

The build script uses each checkout's normal build backend and records its Git
commit, working diff hash and resulting wheel hashes in `wheels/build-receipt.json`.
The fixed `SOURCE_DATE_EPOCH` makes NRL's timestamp-based package version reproducible.
Wheels are local build artifacts and are not committed.

Building wheels first avoids inheriting the root project's CUDA 12.9 uv index.
Resolution uses ordinary package requirements, the CUDA 13 PyTorch index and
security floors from both projects. There are no dependency overrides in this
profile. A root `uv sync` still selects Curator's original environment.

## PDF recipe and Parse versions

The NRL integration recipe and its metadata-preserving filter option are a separate
change (Curator PR #2480). Build the Curator wheel from that same checkout when
using the recipe; a wheel built from main does not provide the required filter
option:

```bash
python3 build_wheels.py --nrl-checkout /path/to/NeMo-Retriever \
  --curator-checkout /path/to/Curator-pr2480
uv sync --locked --python 3.12
```

Then run that checkout's `tutorials/interleaved/nemotron_parse_pdf/nrl_lance.py` with
this environment's Python. Follow that recipe's ingest and consume commands;
Lance remains the resumable checkpoint before Curator validation and Parquet
export. Both commands use the same environment.

Parse 1.2 can run locally in this stack. Nemotron Parse 3.5 package 3.5.0.post7
requires vLLM 0.26.0 and Transformers 5.14.1, so it needs its own serving
environment. The separately tested HTTP adapter is experimental and is not
provided by this dependency profile. This is not a single-environment Parse 3.5
installation.

## Validation scope

The isolated profile must pass clean installation, locked resynchronization,
package dependency checks, targeted NRL/Curator tests and a guarded PDF GPU run
before adoption. The larger compatibility investigation found unresolved RAPIDS,
audio and Data Designer constraints; a working PDF profile does not establish a
global Curator upgrade. See the accompanying PR validation report for exact runs
and pending hardware/platform checks.
