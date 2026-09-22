# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CLI for the lean NRL Nemotron Parse to NeMo Curator Lance bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import nrl_lance_contract as contract  # noqa: E402
from nrl_lance_runtime import run_consume, run_ingest  # noqa: E402

MAX_PROJECTION_WORKERS = 8


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest",
        help="Deduplicate PDFs, run the extraction-only NRL graph, and publish pdf_elements",
    )
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="PDF directory under /raid; searched recursively")
    source.add_argument(
        "--manifest",
        help="JSONL under /raid with path and optional url and valid_blank_pages fields",
    )
    ingest.add_argument("--output-root", default="/raid/nrl-curator", help="Parent for fresh run directories")
    ingest.add_argument("--run-id", help="Fresh run directory name")
    ingest.add_argument(
        "--evidence-root", help="Opt-in private content-hashed page/raw-response evidence directory under /raid"
    )
    ingest.add_argument(
        "--executor-stats",
        action="store_true",
        help="Save executor_stats.txt in the run directory without capturing page images or model responses",
    )
    ingest.add_argument(
        "--projection-workers",
        type=int,
        default=MAX_PROJECTION_WORKERS,
        help="Maximum CPU projection workers (1-8)",
    )
    ingest.add_argument("--nrl-repo", required=True, help="Pinned NRL checkout used for provenance")
    ingest.add_argument(
        "--projection-block-rows",
        type=int,
        help="Opt-in target page rows per block before CPU projection; omitted preserves existing blocks",
    )
    ingest.add_argument(
        "--parse-batch-size",
        type=int,
        default=contract.DEFAULT_PARSE_BATCH_SIZE,
        help="Pages requested per Parse Ray batch (>=2)",
    )
    ingest.add_argument(
        "--parse-cpus",
        type=int,
        default=contract.DEFAULT_PARSE_CPUS,
        help="CPU reservation for the single Parse actor (>=1)",
    )
    ingest.set_defaults(run_mode="batch")

    consume = commands.add_parser(
        "consume",
        help="Run the pinned Lance table through Curator and publish completion after reconciliation",
    )
    consume.add_argument("--handoff-manifest", required=True)
    consume.add_argument("--output-dir", required=True)
    consume.add_argument("--mode", choices=("error",), default="error", help="Output directory must be fresh")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        if not 1 <= args.projection_workers <= MAX_PROJECTION_WORKERS:
            parser.error("--projection-workers must be between 1 and 8")
        try:
            contract.validate_parse_scheduling(args.parse_batch_size, args.parse_cpus, run_mode=args.run_mode)
            contract.validate_projection_block_rows(args.projection_block_rows, run_mode=args.run_mode)
        except ValueError as error:
            parser.error(str(error))
        print(run_ingest(args))
    elif args.command == "consume":
        print(run_consume(args))
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
