# PDF Extraction Pipeline using Nemotron-Parse

Convert PDFs into structured, interleaved parquet — text blocks, tables, images, and captions in reading order — using **Nemotron-Parse v1.2**.

We recommend using NeMo Curator's Dynamo-backed
`InferenceServer` with the HTTP client stage instead of loading vLLM inside the
pipeline stage. Internal comparisons found this to be the better default
because the serving layer can batch requests across pipeline tasks and keep
model replicas fed while PDF rendering and postprocessing scale independently.

We tested the tutorial on 8 H100 GPUs with request concurrency values of 32 and
64. Use this starting configuration:

- One inference-server replica per GPU.
- A fixed HTTP stage pool of `4 * num_gpus` workers.
- `inference_batch_size=32`, which is the maximum number of concurrent page
  requests sent by each HTTP client worker. Because the best value depends on
  the GPU and corpus, benchmark 64 on the target workload and keep it only if it
  improves throughput without request failures or out-of-memory errors.

## Native Curator setup

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install uv
uv sync --extra interleaved_cuda12 --extra inference_server
```

The ordinary tutorial does not add image-validation stages. The NRL consumer
and aligned comparisons explicitly enable those stages and require the `cv2`
extra for blur validation. The shared builder's `validate_images` option is
disabled by default; comparison commands enable it without changing ordinary
tutorial behavior.

The NeMo Curator container includes the `etcd` and `nats-server` binaries that
Dynamo starts. For a source environment outside the container, install them
with [`docker/common/install_etcd_nats.sh`](https://github.com/NVIDIA-NeMo/Curator/blob/main/docker/common/install_etcd_nats.sh)
before running the recommended entry point.

## Run the native Curator tutorial

**Step 1 — Create a manifest listing your PDFs:**

```bash
# One JSON line per PDF
for f in /path/to/pdfs/*.pdf; do
    echo "{\"file_name\": \"$(basename $f)\"}" >> manifest.jsonl
done
```

**Step 2 — Start Dynamo and run the pipeline (recommended):**

```bash
uv run python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --model-path nvidia/NVIDIA-Nemotron-Parse-v1.2 \
    --inference-batch-size 32
```

`main.py` detects the Ray-visible GPUs, starts one Dynamo model replica per GPU,
waits for the OpenAI-compatible endpoint to become healthy, runs the pipeline,
and stops Dynamo. No separately managed server process is required. It fixes
the HTTP stage pool at `4 * num_gpus` workers. Use `CUDA_VISIBLE_DEVICES` or
your Ray cluster resources to control which GPUs are used.

`main.py` always starts Dynamo with vLLM and does not expose a `--backend`
option.

**Alternative — Run inference in process:**

```bash
uv run python tutorials/interleaved/nemotron_parse_pdf/inprocess.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --backend vllm \
    --enforce-eager
```

Use `inprocess.py` for local validation and debugging when you do not want a
separate serving topology.

The in-process and Ray Serve paths use Triton attention automatically on
Ampere GPUs, following the model's A100/A10 guidance. On Blackwell, they also
use Triton with vLLM versions before 0.23 to avoid the affected
FlashInfer/TRTLLM implementation. Ray Serve bases this choice on the
driver-visible GPU and assumes its architecture matches the serving replicas.
Explicit settings are preserved.

The entry point uses `create_nemotron_parse_inference_server`, which keeps the
Nemotron-Parse vLLM, Dynamo, and runtime-environment settings shared with the
benchmark. See the [Inference Server guide](https://docs.nvidia.com/nemo/curator/latest/curate-text/synthetic/inference-server)
for details about the underlying configuration objects.

## Experimental NRL extraction handoff

The optional [`nrl_lance.py`](nrl_lance.py) recipe connects NeMo Retriever
Library (NRL) extraction to Curator without adding a Curator-specific public
API to NRL. It is an extraction and compatibility proof, not a retrieval
pipeline: it does not chunk, embed, create vectors, or run queries.

Use separate pinned environments for the two commands. Run `ingest` in
the NRL GPU environment and `consume` in the Curator CPU/Lance environment.
Pin both repository revisions for a qualification run; the handoff records the
revisions and recipe source used. Do not combine the projects' different GPU
dependency stacks into one environment.

The qualification environments are
`/raid/kyzheng/.runtime/nrl-curator-mvp/nrl` and
`/raid/kyzheng/.runtime/nrl-curator-mvp/curator`. Use their respective
`bin/python` executables; equivalent installations must preserve the pinned
dependency versions.

The Curator consumer environment requires Lance and OpenCV in addition to the
interleaved CPU dependencies:

```bash
uv sync --extra interleaved_cpu --extra lance --extra cv2
```

The data path is:

```text
PDF inventory, hashing, preflight, and exact deduplication
  -> NRL PDF split and render stages
  -> Nemotron-Parse v1.2
  -> recipe-owned page projection actor
  -> page validation and explicit document outcomes
  -> pdf_elements in LanceDB
  -> native Curator read, validation transforms, and Parquet writer
```

The recipe builds a custom NRL graph from the existing PDF extraction stages
and adds one terminal, recipe-owned projection actor. Nemotron-Parse remains
the parser and runs once for each unique PDF representative. The projection
actor converts each page into a compact internal envelope and releases the
rendered page image after producing any picture crops. Ray batches may split a
document, so this actor does not decide document completeness or write Lance.

The internal envelope has two record kinds:

- A page outcome identifies the source and native one-based page number,
  records `parsed`, `empty`, or `failed`, declares the element count, and
  carries structured issues plus the raw-output hash.
- An element record carries its page-local order, Nemotron class, mapped
  modality and content type, text or inline PNG bytes, and normalized bbox.

The publisher groups the envelope by source and joins it to the immutable
input inventory. Each delivered page must have exactly one outcome,
contiguous page-local element indexes, matching declared element counts, no
extraction or nested parser errors, a `stop` model finish reason, complete
tagged raw output, valid bboxes, and a valid crop for every `Picture`. A failed
check rejects all elements on that page, not just the offending element. An
empty response is accepted only for a page listed in `valid_blank_pages`.
Otherwise it is a failed page with issue `unexpected_empty_output`, not proof
that the source page is blank.

The default publication policy is `validated_pages_v1`: publish validated
pages and report every expected page, including failed or missing pages.
Documents with every page validated and no document issues are `success` or
`valid_blank`. A document with validated pages and unresolved issues, including
failed pages, is `partial`; its validated pages enter `pdf_elements` without
being relabeled as complete extraction. A document with
no validated pages is `failed` and produces no rows. A partial document whose
only validated pages are declared blanks produces only a metadata row. This
policy changes delivery, not the model, prompt, graph, or page validators; it
adds no retries or fallback extraction.

`success`, `valid_blank`, and `partial` describe structural validation and
page accounting, not content accuracy. A structurally complete document can
still contain undetected model omissions or incorrect text. Human content
review remains separate; this policy does not improve Parse's predictions.

Every input has one of `success`, `valid_blank`, `partial`, `failed`, or the
input-only `duplicate` status in the sealed manifests. A duplicate follows its
content representative and is retained in `source_aliases`. Each published
document is added to Lance in one write, and validation requires every document
to occupy exactly one fragment. Reports separate complete, partial, and failed
documents from delivered pages, delivered content pages/elements, and failed
pages. Delivering more pages does not establish better model accuracy.
If preflight cannot determine a document's page count, the report leaves the
aggregate `failed_page_count` null and reports `known_failed_page_count` and
`unknown_page_count_document_count` separately. Unknown pages are not zero
failed pages.

### Prepare the NRL input manifest

Keep the manifest, resolved PDFs, run directory, and consumer output under
`/raid`. A manifest `path` may be absolute or relative to the manifest file,
but its resolved target must remain under `/raid`; a symlink to a source
outside `/raid` is rejected.

The confirmed corpus is physically staged at
`/raid/kyzheng/nemotron_parse_pdf/pdfs` and contains 1,278 PDFs. Use that
directory directly instead of symlinking to a source on another filesystem or
cluster.

`url` and `valid_blank_pages` are optional:

```json
{"path":"/raid/nrl-curator-input/pdfs/example.pdf","url":null,"valid_blank_pages":[2]}
```

Page numbers and `valid_blank_pages` are zero-based. For a genuinely blank
document, list every page. Use
`--input-dir /raid/nrl-curator-input/pdfs` instead of `--manifest` when URL and
blank-page annotations are not needed; the directory search is recursive.

### 1. Extract and create the sealed handoff

Activate the compatible NRL environment, then run:

```bash
python /raid/kyzheng/nv-ingest/Curator/tutorials/interleaved/nemotron_parse_pdf/nrl_lance.py ingest \
    --manifest /raid/nrl-curator-input/manifest.jsonl \
    --output-root /raid/nrl-curator \
    --run-id pdf-mvp-20260918 \
    --projection-workers 8 \
    --nrl-repo /raid/kyzheng/nv-ingest
```

`ingest` hashes and preflights all inputs before graph execution, sends only
unique representatives through the extraction-only graph, finalizes document
and page outcomes one document at a time, and validates the exact Arrow schema,
nullability, identity, positions, provenance, inline pictures, and
one-fragment-per-document layout. It pins the validated Lance version and writes:

```text
/raid/nrl-curator/pdf-mvp-20260918/handoff_manifest.json
```

The handoff is sealed and has status `tables_validated`, but it is not a
completed Curator export. It publishes the validated Lance boundary for
consumption. `ingest` never creates `completion_manifest.json`.
`run_state.json` is diagnostic state only.

`--projection-workers` sets the terminal CPU projection auto-concurrency
ceiling, not a fixed worker-pool size. Resource preflight may plan fewer
workers. The value must be between 1 and 8 and defaults to 8.

Optional scheduling controls tune the existing Parse actor and CPU projection
without changing extraction or publication policy.

| Ingest flag | Default | Meaning |
| --- | --- | --- |
| `--parse-cpus` | `1` | Positive integer Ray CPU reservation for the Parse actor. |
| `--parse-batch-size` | `64` | Requested Ray page-batch size for Parse; integer at least `2`. |
| `--projection-block-rows` | Unset (`None`) | Positive integer target for page rows per Ray block before CPU projection. |

The pinned executor promotes a requested batch size of `1` to `64`, so the
recipe rejects `1` instead of reporting a setting it cannot honor. Ray batches
can contain fewer pages than requested. These controls retain one Parse actor
and one GPU; they do not add model replicas, change the model's sequence limit,
or enable OCR, embeddings, or retries. CPU reservation is not a guarantee of
CPU parallelism. Inspect resolved settings and observed batches with
`--executor-stats`.
Nondefault Parse scheduling controls require the default `batch` run mode;
`inprocess` mode rejects these overrides.

Leaving `--projection-block-rows` unset preserves existing blocks without adding
a repartition. Setting it uses NRL's existing block-sizing mechanism immediately
before projection, independently of the Parse batch size. It requires `batch`
mode and does not change the projection worker ceiling or make driver collection
bounded-memory. The target is not a guarantee of exact task size or a speed gain.

The ingest command is local-v1.2-only. Before graph execution it requires the
Ray resource snapshot to resolve `NemotronParseActor` to NRL's local GPU actor;
it fails closed when no GPU is visible and never falls back to the remote
CPU/NIM endpoint.

For qualification runs, add `--evidence-root /raid/nrl-curator-evidence` to
retain rendered page PNGs and exact raw responses before projection releases
them. This option is disabled by default. The directory must be under `/raid`
and must not already exist, including as an empty directory. Use a fresh path
for each run. Page evidence is staged under
`pending/<source-path-hash>`; complete accepted documents receive a
content-addressed `<core_sha256>/manifest.json` with hashed blobs. Partial and
failed documents remain pending and are not eligible for the projection
comparison. Delivering validated pages does not change comparison eligibility.
The handoff's `benchmark_evidence` field lists the finalized evidence
manifests. These files are private benchmark artifacts, separate from the
18-column Lance contract.

For lightweight production-mode diagnostics, add `--executor-stats` without
`--evidence-root`. It writes `executor_stats.txt` in the run directory using the
same graph execution. The file records resolved batch sizes, actor-pool
concurrency, CPU/GPU reservations, and Ray execution statistics. Its
`compact_result_pandas_estimate_bytes` is a pandas memory estimate, not process
RSS. It does not retain page rasters or raw model responses and does not rerun
Parse.

Captured evidence derives a successful `stop` outcome from NRL's local actor
error contract: any other finish reason becomes a page error. Its provenance
records `finish_reason_source=local_actor_error_contract`; it does not contain
the original generation token counts. The inference comparison below captures
actual finish reasons and token counts from both runtimes.

The handoff records separate `adaptation_seconds`, `lance_write_seconds`, and
`evidence_finalize_seconds` timings, alongside inventory, graph, and table
validation timings. Adaptation measures document finalization from the compact
envelope, not the projection actor inside the graph. These partial-stage
measurements exclude process startup and do not establish end-to-end cost or
a performance advantage. Use external command timing for complete runs.

### `pdf_elements` contract

The table has exactly these 18 fields:

| Field | Arrow type | Contract |
|-------|------------|----------|
| `sample_id` | non-null string | SHA-256 of the PDF bytes; stable document identity |
| `position` | non-null int32 | `-1` for metadata, then contiguous document-wide content positions from `0` |
| `modality` | non-null string | `metadata`, `text`, `table`, or `image` |
| `content_type` | string | `application/json`, `text/markdown`, or `image/png` as required by modality |
| `text_content` | string | Metadata JSON, extracted text, or model-native table content, including LaTeX; picture text may be empty |
| `binary_content` | large binary | Inline PNG bytes for `image` rows only |
| `source_ref` | string | Always null because image bytes are inline, not deferred materialization references |
| `materialize_error` | string | Always null for published rows |
| `url` | string | Source URL when one was supplied |
| `page_number` | int32 | Zero-based source page for content; null for metadata |
| `pdf_name` | string | Representative PDF filename |
| `element_class` | string | Nemotron element class for content; null for metadata |
| `source_path` | string | Absolute representative PDF path |
| `source_aliases` | string | Canonical JSON list of all exact-content aliases |
| `content_sha256` | non-null string | Same PDF digest as `sample_id` |
| `bbox_xyxy_norm` | list of float64 | Normalized element bbox for content; null for metadata |
| `bbox_coordinate_space` | string | `normalized_1664x2048_padded_canvas` for content; null for metadata |
| `run_id` | non-null string | Fresh immutable run identifier |

Each published document has exactly one metadata row at `position=-1`. Its
JSON includes `extraction_status`, document issues, and `page_outcomes`. Each
page outcome records the original zero-based `page_number`, status (`success`,
`valid_blank`, or `failed`), `element_count`, and issues. Every known expected
page is represented, and failed pages have zero elements. These fields are
inside metadata JSON and the document reports, not new Arrow columns.
The metadata row passes unchanged from Lance to Parquet, so exported content
retains its extraction status and page outcomes without relying on run logs.

A valid all-blank document has only a metadata row; so does a partial document
with only validated blank pages. Content keeps original source page numbers
and cross-modality model order. Document-wide positions remain contiguous over
the delivered elements; a missing source page is not renumbered away.
`Picture` maps to `image`, `Table` maps to `table`,
and `Chart`, `Infographic`, and all other Nemotron classes map to `text`, which
matches the native Curator postprocessor. A textless `Picture` is still emitted
as an image; an empty `Chart` or `Infographic` remains a text row.

Table elements retain their model-native bodies, including LaTeX merged-cell
commands, through NRL's existing `table_format="latex"` helper option. The
recipe does not convert tables to Markdown pipe tables or HTML. It retains
native Curator's `table`/`text/markdown` convention, which permits mixed
Markdown and LaTeX content rather than guaranteeing generic Markdown rendering.
This preserves element content, not full raw page responses. It does not
repair model errors or relax page validation.

### 2. Validate through native Curator and publish

Activate the compatible Curator environment, then run:

```bash
python /raid/kyzheng/nv-ingest/Curator/tutorials/interleaved/nemotron_parse_pdf/nrl_lance.py consume \
    --handoff-manifest /raid/nrl-curator/pdf-mvp-20260918/handoff_manifest.json \
    --output-dir /raid/nrl-curator-output/pdf-mvp-20260918 \
    --mode error
```

`consume` refuses an unsealed handoff or a handoff not marked
`tables_validated`, opens the recorded immutable Lance version, and executes
Curator's normal `Pipeline` with `RayDataExecutor`:

```text
InterleavedLanceReader at the pinned version
  -> InterleavedAspectRatioFilterStage with bounds [0, infinity]
  -> InterleavedBlurFilterStage with score_threshold=0
  -> InterleavedParquetWriterStage
```

Both image stages set `drop_invalid_rows=False`, so Curator's generic modality
mask does not discard table rows, and set
`preserve_metadata_only_samples=True`, so a document with only validated blank
pages keeps its metadata row and extraction status. The unbounded aspect-ratio
stage verifies recognizable image dimensions. The zero-threshold blur stage
fully decodes each image and computes
its sharpness without imposing a quality cutoff. Either stage may still reject
malformed image bytes; exact reconciliation then fails and the run is not
published.

After the pipeline completes, `consume` compares the Parquet output with the
pinned Lance version by `(sample_id, position)`. It requires identical row and
document counts, keys, per-document positions, all 18 fields, and every
non-binary value, and compares each `binary_content` value by SHA-256. This is
a lossless-compatibility proof, not a data-cleaning pass.

The recipe configures the native Parquet writer with the complete Arrow
schema, including field nullability. Reconciliation rejects different logical
types or nullable declarations for required fields even when values match.
It also reopens the output through Curator's native Parquet reader before
completion. Historical exports are retained unchanged; a previous value-only
reconciliation does not establish this stronger export contract.

The consume report includes pipeline and reconciliation wall times plus native
Curator per-stage metrics for the reader, validation transforms, and writer.
Stage measurements can overlap across workers and are not additive pipeline
wall time. `pipeline_and_reconciliation_seconds` excludes imports, preflight,
and final marker work; it is not end-to-end elapsed time.

Only after reconciliation succeeds does `consume` write a sealed
`consume_validation.json` in the output directory. It then revalidates the
pinned table and rehashes the source inventory before atomically creating:

```text
/raid/nrl-curator/pdf-mvp-20260918/completion_manifest.json
```

`completion_manifest.json` with status `published` is the only completion
marker. It binds the sealed handoff, pinned table version, and consume report.
New markers identify `publication_policy=validated_pages_v1`. Completion means
all inputs are accounted for and the delivered rows reconcile; it does not
mean every document or source page was extracted successfully. Inspect the
document and page outcomes. Older sealed whole-document-policy handoffs remain
consumable under their original contract; do not rewrite them as partial-page
publication evidence.
Failures before the atomic creation of a phase marker leave that marker
absent. A handoff, `run_state.json`, or consume report alone is not a completed
export. If directory synchronization fails after a valid marker becomes
visible, the marker remains authoritative, but its durability is unconfirmed.
The command reports failure, and qualification remains unsuccessful until
verification and synchronization succeed. Do not remove or overwrite the
visible marker to make the run appear unpublished.

Treat every handed-off Lance version and consumer output directory as
immutable. Failed ingestion requires a new `--run-id`. If consumption fails
before a completion marker is visible, you can retry the unchanged sealed
handoff into a fresh output directory after verifying its dependencies. This
does not rerun Parse. Never resume writes into an old output or append to a
handed-off table. Consumer output directories must be fresh, and `--mode error`
is the only supported writer mode. An existing completion marker, including one
with unconfirmed durability, prevents re-consumption.

### Keep three kinds of evidence separate

Use the comparisons below to answer different questions.

- Bridge correctness: replay identical frozen pages and responses, then require
  exact NRL-to-Lance-to-Parquet preservation for published rows.
- Content accuracy: compare extraction with human-adjudicated regions on the
  existing 50-page review set. Assess coverage, text, tables, reading order,
  pictures, complete versus partial documents, and failed pages. Neither engine
  is the ground truth.
- Operating cost: run both products from PDFs to validated export in fresh
  processes, with repeated paired measurements and explicit failure accounting.

Precision and recall require approved reference annotations. Until those exist,
report content observations rather than an accuracy percentage. Retrieval
recall is outside this extraction-only recipe; no search index is built.

Use `nrl_compare.py source-evidence` to assemble the fixed tuning review pages
from a completed benchmark pair and the study's historical capture index.
It verifies source and export hashes, retains exact delivered rows and picture
bytes, and keeps native validation findings and NRL partial/failed outcomes
visible. Historical page images and responses are not fresh benchmark model
responses. This offline command does not rerender, invoke a model, change
publication, or judge semantic accuracy. Follow the
[source-evidence procedure](QUALIFICATION_RUNBOOK.md#assemble-source-grounded-review-evidence)
outside timed extraction.

The optional `judge` command sends source images, delivered picture crops, and
candidate text to an explicitly selected, approved image-capable endpoint.
Its report-only observations are uncalibrated, not accuracy scores or human
approval. Follow the [judge procedure](QUALIFICATION_RUNBOOK.md#request-report-only-model-observations)
before sending document content; no endpoint or model is selected by default.

### Compare frozen translation ownership

[`nrl_compare.py`](nrl_compare.py) compares the recipe-owned NRL projection
with Curator's native `NemotronParsePostprocessStage` without rendering a page
or invoking the model again. It accepts only a pre-existing, content-addressed
version 1 evidence directory containing `<core_sha256>/manifest.json` and
`<core_sha256>/blobs/<blob_sha256>`. The manifest binds each page PNG and raw
Nemotron response to its digest, byte length, page identity, render contract,
model revision, prompt, token and decoding settings, `stop` finish reason, and
clean extraction status.

Run the orchestrator from either environment and provide both pinned Python
executables. The Curator environment must include the `cv2` extra shown above:

```bash
python tutorials/interleaved/nemotron_parse_pdf/nrl_compare.py compare \
    --evidence-manifest /raid/nrl-curator-evidence/<core_sha256>/manifest.json \
    --output /raid/nrl-curator-evidence/projection_comparison.json \
    --nrl-python /raid/kyzheng/.runtime/nrl-curator-mvp/nrl/bin/python \
    --curator-python /raid/kyzheng/.runtime/nrl-curator-mvp/curator/bin/python
```

The command replays the frozen page and response once in each isolated
environment. It compares page coverage, Nemotron class/model order, modality,
content type, normalized text and table text, normalized bboxes, and picture
PNG decode hashes and crop dimensions. The deterministic report seals both
replay digests and all differences. Exit status `0` means equivalent, `1`
means a valid comparison found differences, and `2` means the evidence,
runtime dependency, or replay contract was invalid. Malformed evidence never
produces a report.

NRL crops from the full-resolution page, while the native postprocessor crops
from its resized canvas. Crop dimensions and PNG hashes can therefore differ
without different source bounds. Both paths retain model-native table markup,
including LaTeX, rather than converting tables to Markdown pipe tables.
Other text-normalization differences remain subject to content review; neither
implementation is a correctness oracle.

Use an evidence manifest listed by an ingest run with `--evidence-root`.
The comparison never writes a completion marker. This command covers
projection/postprocessing behavior; inference and complete-product behavior
are separate comparisons.

### Compare inference on identical RGB pages

The `inference` command runs the existing NRL Parse actor and Curator inference
stage on the same decoded RGB page pixels. Both Python environments must
support local GPU inference; the CPU/Lance consumer environment is sufficient
for projection replay but not this comparison.

The pinned upstream model processor used by native Curator GPU inference also
requires `albumentations==2.0.8`, which is not included in the locked
`interleaved_cuda12` extra. Include that pin in the Curator GPU environment
used for `inference` and `native-product`, and record its resolved package
versions with the run evidence. The CPU/Lance `consume` workflow does not load
the model and needs neither this package nor the GPU extras.

Replace the environment and snapshot paths in this example with the installed
locations:

```bash
python tutorials/interleaved/nemotron_parse_pdf/nrl_compare.py inference \
    --evidence-manifest /raid/nrl-curator-evidence/<core_sha256>/manifest.json \
    --output /raid/nrl-curator-evidence/inference_comparison.json \
    --nrl-python /raid/kyzheng/.runtime/nrl-curator-mvp/nrl/bin/python \
    --curator-python /raid/kyzheng/.runtime/nrl-curator-mvp/curator/bin/python \
    --model-snapshot /raid/model-cache/models--nvidia--NVIDIA-Nemotron-Parse-v1.2/snapshots/<40-hex-revision> \
    --gpu 0 \
    --repetitions 3 \
    --measured-passes 2
```

The existing snapshot directory must match the revision in the frozen
evidence. `--gpu` selects the same physical GPU index or UUID for both
subprocesses. Each process performs one warmup followed by the requested
measured passes; the engine order alternates across repetitions. The report
records exact raw responses, finish reasons, token counts, model startup and
inference time, decoded RGB hashes, and sampled GPU and process memory.
The comparison verifies prompt and decoding settings at the actual generation
call. Its default per-process timeout is 1,800 seconds and can be changed with
`--timeout-seconds`.

Status `compared` means the comparison ran with valid generations and a
verified shared GPU; inspect `raw_responses_identical` and the repeated
measurements separately. Its structural checks do not establish semantic
accuracy or crop quality. `quality_failed` and `gpu_unverified` produce a
nonzero exit. The command does not select a performance winner: quality and
coverage must pass first, and any speed difference must exceed the observed
run variance.

### Compare the independent native Curator product

Run `native-product` in the Curator GPU environment after the NRL handoff is
available. Set `CUDA_VISIBLE_DEVICES` before starting Python and use the same
value for `--gpu`:

```bash
CUDA_VISIBLE_DEVICES=0 /raid/kyzheng/.runtime/nrl-curator-mvp/curator/bin/python \
    tutorials/interleaved/nemotron_parse_pdf/nrl_compare.py native-product \
    --handoff-manifest /raid/nrl-curator/pdf-mvp-20260918/handoff_manifest.json \
    --output-dir /raid/nrl-curator-native/pdf-mvp-20260918 \
    --model-snapshot /raid/model-cache/models--nvidia--NVIDIA-Nemotron-Parse-v1.2/snapshots/<40-hex-revision> \
    --gpu 0
```

The output directory must be fresh. The workflow copies canonical PDFs to
content-addressed filenames, runs Curator's existing PDF composite through
its native pipeline executor, applies the same aspect and blur validation
stages, and writes Parquet. It compares that output with the pinned NRL Lance
version and writes `product_comparison.json`, including content coverage,
ordered element differences, native Curator per-stage metrics, and resource
samples. This command compares native output against a previously produced NRL
table. It is not a fresh, paired end-to-end timing comparison. Per-stage
measurements are separate from pipeline wall time and do not establish a speed
advantage without repeated, aligned measurements.

By default, `native-product` compares all canonical input representatives.
Repeat `--sample-id <content-sha256>` to select an evaluation subset by its
64-character lowercase content SHA-256 identities, including inputs withheld
by NRL. Selected aliases remain accounted for. The full original handoff and
its pinned Lance version are still validated; selection only narrows the
native inputs and comparison rows, not the validation boundary.

The native control retains its 300-DPI renderer, 1664-by-2048 fit, batching,
and 50-page limit. Documents longer than 50 pages are reported separately and
excluded from paired metrics. Native Parquet has no per-page outcome markers
or finish reasons, so observed content pages cannot establish blank-page
completeness. Rendering failures may also cause the native pipeline to skip
and renumber pages; review provenance when failures occur. Use the inference
report for generation-level evidence and the NRL document validator for its
publication guarantees. A product report with status `compared` records a
completed comparison, not equivalence or a performance recommendation.

### Benchmark both complete extraction paths

Use `prepare-benchmark` and `benchmark` for fresh, paired product measurements.
Follow the [benchmark procedure](QUALIFICATION_RUNBOOK.md#benchmark-complete-products-after-human-approval)
after the human-quality gate passes. An explicitly authorized diagnostic run
can instead use `benchmark --diagnostic` before human approval. Diagnostic
results retain `quality_status=pending_human_review` and `qualified=false`;
they do not approve limitations or establish replacement readiness. These
commands do not replace projection or identical-image inference comparisons.

The prepared cohort contains unique, readable original pilot PDFs with at most
50 pages. Selection uses input identity and preflight, not previous extraction
success. Synthetic or corrupt fixtures, duplicates, declared all-blank PDFs,
and longer documents remain explicitly accounted for outside the paired cohort.
Long-document safety remains a separate capacity gate.

`prepare-benchmark --selection full-corpus` includes readable unique original
PDFs beyond the 50-page cap and declared all-blank documents. Native Curator
still processes at most 50 pages per document. Its capped workload and
long-document coverage must remain separate from NRL's complete expected-page
denominator. This mode compares configured product behavior, not equal-work
throughput on long documents, and requires `benchmark --diagnostic`. Preserve
the matched cohort for equal-work comparisons.

Each measured pair starts both products from PDFs in fresh processes and fresh
output directories, without raw-response or page-image evidence capture.
The NRL path includes extraction, Lance publication, a separate Curator consumer
process, exact reconciliation, and completion. The native path independently
extracts, validates images, exports Parquet, and reopens it through Curator.
Both use the pinned model and aligned generation settings on the same idle GPU;
their renderers, batching, and failure policies remain visible differences.

This runner uses native Curator's local vLLM stage with `RayDataExecutor`, not
the recommended Dynamo-backed `main.py` topology. Results apply to that local
baseline only. The optional `--native-gpu-memory-utilization` flag changes the
comparison's native reservation without changing ordinary tutorial defaults.
Record the override and retain the same host, GPU, and disk capacity budgets.

The benchmark exposes `--nrl-parse-cpus` (default `1`),
`--nrl-parse-batch-size` (default `64`), and `--nrl-projection-block-rows`
(unset by default) for the ingest controls above.
`--native-pdfs-per-task` (default `10`) changes native PDF task grouping, not
the number of GPUs or concurrent model sequences. These overrides are recorded
with the run; ordinary native tutorial defaults remain unchanged.
Follow the [one-variable tuning procedure](QUALIFICATION_RUNBOOK.md#tune-one-variable-at-a-time)
to compare CPU reservation, page-batch size, projection block size, and native
task grouping separately.

The non-restrictive image thresholds do not guarantee lossless filtering.
Native filters can discard undecodable images and reindex surviving rows.
Native export and reread validate the remaining images, not absence of filter
loss or strict document completeness. The NRL consumer's exact comparison with
Lance catches dropped rows and prevents completion.

Run three alternating pairs: NRL/native, native/NRL, then NRL/native. Reuse
existing model and disk caches; warm caches are the intended convention, not a
verified cache-residency guarantee. Record that convention and do not flush
shared caches. Preparation is reported separately from externally measured
command time. Include both NRL process startups in its total.

Inspect every repetition, median, range, paired timing differences, acceptance
counts, attempted-page throughput, and output-page coverage together. Native
content pages are not proof of complete-page extraction. Resource observations
and executor diagnostics help explain differences but are not additive wall
times. Observer `driver_rss` measures the benchmark launcher, not its NRL ingest
child. Raw per-PID samples include ingest, consume, and Ray workers; actual
phase-driver peaks can be attributed from `sampled_process_rss_peaks` using
each process's command, parent PID, and creation time. Do not identify a driver
from the name `python` alone. Tree RSS includes all descendants and can
double-count shared pages. Missing metrics remain unavailable, not zero.

Aim for sustained utilization near 100% during useful extraction compute,
not during initialization or CPU-only export. Utilization measures activity;
GPU memory reservation is not compute utilization. Retain at least 10% GPU
memory and 25% effective host-memory headroom. Select settings using delivered
content, pages per second, quality, and total resource cost. Do not add redundant
model work merely to raise utilization.

In diagnostic mode, native element-validation findings can remain in failed
repetitions while later repetitions run, provided provenance, configuration,
execution, and resource checks pass. Such collection remains failed and
unqualified; timing an invalid output does not validate it. Configuration,
source drift, execution, or safety failures stop collection. Qualified mode
remains fail-fast. Refer to the [diagnostic procedure](QUALIFICATION_RUNBOOK.md#collect-authorized-diagnostic-evidence-before-human-approval)
for interpreting operational timing separately from valid-output results.

A completed comparison does not establish a performance winner. Require
acceptable content quality and completeness before considering speed, and
report noisy or overlapping results as inconclusive. Three repetitions do not
establish statistical significance or a universally faster library.
Stage p50/p95 durations are not per-document end-to-end latency. Report that
latency as unavailable unless document submission and completion are measured.

### Qualification gates and run evidence

Follow the [internal qualification runbook](QUALIFICATION_RUNBOOK.md) for
environment commands, regression tests, publication recovery, human review,
capacity budgets, and upgrade gates. Qualify the extraction-only path in this
order:

1. Freeze both runtimes, relevant source hashes, model settings, and input
   manifests. Pass the environment-specific suites and a fresh GPU vertical
   slice with native Curator export.
2. Exercise exact Arrow export, storage failures, mutation/version drift,
   interrupted processes, and marker visibility/durability boundaries.
3. Obtain human adjudication of the 50-page review and explicit acceptance of
   known model limitations. Expand frozen projection and identical-image
   inference comparisons without treating native Curator as an oracle.
4. Run the 100-input pilot without evidence capture, then the deterministic
   long-document/render-workload/image-count stress cohort. Verify isolated
   memory, GPU, temporary-disk, and cleanup budgets.
5. Run all 1,278 original PDFs only after human approval and the preceding
   structural and capacity gates pass. Reconcile every published row and
   account for every failed, partial, blank, or duplicate input separately.
6. Retain the approved operating envelope, limitations, recovery evidence,
   and upgrade regression record for the next maintainer.

Use each run's `handoff_manifest.json`, `consume_validation.json`, and
`completion_manifest.json` as the evidence of its validated table, exported
rows, and publication state. Inspect their per-input and per-page outcomes: a
completed run can include partial documents and explicitly reported failed
inputs or pages. Frozen evidence manifests and
the projection, inference, and `product_comparison.json` reports establish
what was compared and with which configuration; a comparison report does not
publish an ingestion run. `run_state.json` remains diagnostic only.

These gates describe required qualification, not the status of a particular
local run. Previous dual-table extraction/retrieval completion markers do not
qualify this extraction-only graph. A green completion marker does not replace
human content-quality approval or capacity qualification.
Explicitly authorized diagnostic pilot, stress, and full-corpus measurements
can precede human approval, but retain structural and capacity checks. They
remain unqualified until the human quality and failure-accounting gates pass.

For Curator's native implementation, see the official
[Nemotron-Parse PDF composite](https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/interleaved/pdf/nemotron_parse/composite.py).

## Native Curator input formats

The pipeline supports three input formats selected by a mutually exclusive flag:

| Flag | Description |
|------|-------------|
| `--pdf-dir PATH` | Flat directory of `.pdf` files |
| `--zip-base-dir PATH` | CC-MAIN-style numbered zip archives |
| `--jsonl-base-dir PATH` | GitHub-style JSONL with base64-encoded PDFs |

## Native Curator output schema

Each row in the output parquet is one **document element** in reading order:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | PDF filename without extension |
| `position` | int | Element index within document |
| `modality` | string | `text`, `image`, `table`, or `metadata` |
| `content_type` | string | `text/markdown`, `image/png`, or `application/json` |
| `text_content` | string | Extracted text; table bodies can retain LaTeX under the `text/markdown` convention |
| `binary_content` | bytes | PNG bytes for image elements |
| `page_number` | int | Source page (0-indexed) |
| `url` | string | Source URL from manifest |

**Read the output:**

```python
import pandas as pd

df = pd.read_parquet("output/my_doc.parquet")
print(df[["modality", "content_type", "text_content"]].head(10))

# All text
text_blocks = df[df["modality"] == "text"]["text_content"].tolist()

# All images
from PIL import Image
import io
images = [Image.open(io.BytesIO(b)) for b in df[df["modality"] == "image"]["binary_content"]]
```

## Native Curator key options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `vllm` | In-process engine (`inprocess.py` only); also supports `hf`. |
| `--enforce-eager` | off | Skip vLLM CUDA graph capture (~35 min savings on first run) |
| `--max-num-seqs` | 64 | Max concurrent sequences for vLLM |
| `--max-tokens` | 8192 | Maximum output tokens per page; aligned comparisons explicitly request 9000 |
| `--inference-batch-size` | 32 (`main.py`), 4 (`inprocess.py`) | Concurrent requests per HTTP worker, or pages per in-process HF pass |
| `--pdfs-per-task` | 10 | PDFs batched per processing task |
| `--max-pdfs` | — | Cap total PDFs (for testing) |
| `--dpi` | 300 | PDF rendering resolution |
| `--max-pages` | 50 | Max pages per PDF |
| `--text-in-pic` | off | Predict text inside images (v1.2+ feature) |
