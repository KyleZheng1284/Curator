# Parse 3.5 HTTP throughput benchmark

This single-node benchmark runs Curator's served single-pass Page64 pipeline
with four external vLLM HTTP replicas. Each replica receives one page per
request. Bounded admission hands a completed render range to inference while
PDFium renders the rest of the parent. Curator retains all physical page
outcomes and assembles complete documents before publication.

## Run a shard

Run both cohorts in sequence with the pinned controller Python and
`run_bundle.py all <attempt> --deadline-unix <epoch-seconds>` in the parent
study directory. The launcher supplies the exact cohort, job-owned Ray
address, assigned GPUs, compatible private compile-cache copies, output
directory and pinned CPU and GPU environments. It invokes the shard driver
with the following arguments.

```bash
"$CURATOR_PYTHON" -m experiments.local_parse.run \
  --backend http --endpoint "$HTTP_ENDPOINT" \
  --ray-address "$JOB_RAY_ADDRESS" --manifest "$SHARD_MANIFEST" \
  --extraction-config "$EXTRACTION_CONFIG" --profile "$HTTP_PROFILE" \
  --pdf-dir "$PDF_ROOT" --output-root "$OUTPUT_ROOT" --shard-id "$SHARD_ID"
```

The local profile uses four one-GPU BF16 E148/DFlash2-16 engines, FA2 and
full-decode CUDA graphs. Each engine allows 128 sequences, a 16,384-token batch
budget and context, and a 24 GiB KV cache. Curator uses 96 assigned logical CPUs,
16 HTTP workers and up to 32 requests per worker. The extraction contract
records the model, prompt, sampling, raster, retry policy and output schema.
The launcher supplies these benchmark settings at run time.

The fitted raster path uses the pinned PDFium helper to produce JPEGs with
aspect preserved under a 1,024 × 1,024 pixel box, a 200 DPI ceiling and JPEG
quality 100. A CPU-only subprocess stays alive per renderer and resets PDFium
state after each range. Large inputs fall back to the qualified one-shot helper
when the request exceeds the framing limit. The launcher enables these paths
with `CURATOR_STUDY_RASTER=matched`, `PARSE35_EARLY_ADMISSION=1` and
`PARSE35_PERSISTENT_FITTED_RENDERER=1`. It also supplies bounded queue limits
and any historical priority weights used to reproduce an earlier experiment.

## Publication and timing

The driver writes each shard to a fresh attempt directory, validates every
physical page outcome, and atomically publishes a manifest of the output files
and their hashes. A resumed shard verifies its manifest, file hashes and page
inventory. Keep each source document and its expected ranges in one shard.
Report retained render and extraction failures alongside populated pages.

Run the 3,000-PDF cohort first, check its output and resource release, then use
the full 10,000-PDF cohort. Compare each cohort against its own historical
reference. The primary clock starts once all four engines are ready and before
PDF processing starts. It includes live rendering, scheduling, idle gaps,
HTTP requests, inference, publication and independent native output readback.
Record cache preparation, engine startup, cleanup and full job turnaround
separately. First dispatch through last response measures the HTTP window.
Structural readback and source-fact quality are checked independently.

## Cluster rollout

The fitted renderer uses pinned helper paths and depends on Linux process
lifecycle behavior and PDFium reset internals. If a stuck source read
outlives cancellation, the stage stops accepting new parents. Renderer cleanup
runs even when inference teardown raises. Before deploying across a cluster,
qualify matching runtime and source artifacts on every node, shared output
and locking, and worker or node loss across a publication boundary.
