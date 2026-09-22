# Internal NRL PDF-to-Curator qualification runbook

This runbook qualifies the experimental [NRL extraction recipe](README.md#experimental-nrl-extraction-handoff).
It does not declare the implementation qualified. Keep structural correctness,
human content-quality approval, and capacity as separate recorded outcomes.
All three must pass before you designate an internal MVP as qualified.
Explicitly authorized diagnostic measurements can precede human approval.
They do not waive structural validation, capacity budgets, or the final human
decision about replacing the native pipeline.

The supported scope is batch extraction through the existing NRL graph, a
private terminal projection, LanceDB, native Curator validation, and Parquet.
Retrieval, embeddings, services, model tuning, and release packaging are outside
this qualification. Retain separate NRL and Curator environments.

## Historical evidence is not the new qualification

The September 18, 2026 baseline is retained under
`/raid/kyzheng/nv-ingest/.e2e/nrl-curator-poc-20260918/QUALIFICATION.md`.
Its vertical, edge, and 100-input runs reconciled 31, 274, and 16,339 rows.
The pilot accepted 85 documents, including one declared blank document. It
withheld incomplete documents and recorded all duplicate and failed inputs.
That whole-document publication policy is historical. New recipe runs use
`publication_policy=validated_pages_v1` and deliver validated pages from partial
documents. Preserve the old reports and their original denominators; increased
delivery under a different publication policy is not improved model accuracy.

That baseline did not preserve Parquet field nullability, establish full-corpus
capacity, or obtain human quality approval. Its 50-page review is agent review,
not signed human ground truth. Qualification of the original 1,278-PDF run
remains gated, even when diagnostic execution is separately authorized.
Do not overwrite these artifacts or retroactively relabel them as passing the
stronger contract. Use new run IDs, observer prefixes, and output directories.

## Qualification record and ownership

Maintain one qualification report with the following independent gates.

| Gate | Required evidence | Owner |
| --- | --- | --- |
| G1: Reproducibility | Frozen sources, runtimes, model, inputs, passing suites, and fresh GPU vertical slice | Implementing engineer |
| G2: Contract and lifecycle | Exact Arrow export, real native Parquet reopen, fault injection, interruption, and fresh retry | Implementing engineer |
| G3: Content quality | Human-adjudicated 50-page review, comparison evidence, and accepted-limitations register | User-designated human reviewer and sign-off owner |
| G4: Capacity | Capture-disabled pilot and stress measurements within budgets, with clean worker shutdown | Implementing engineer |
| G5: Full corpus | All 1,278 inputs accounted for, exact published-row reconciliation, and accepted failure breakdown | Engineer and human sign-off owner |
| G6: Handoff | Approved operating envelope, immutable evidence locations, and recovery/upgrade instructions | Engineer and future maintainer |

Record repository revisions and relevant dirty-source hashes, resolved runtime
imports, package versions, model/tokenizer snapshot, manifest and PDF hashes,
declared blanks, GPU UUID, rendering, prompt, and decoding settings. Record the
exact commands, exit codes, test skips, output hashes, and reviewer decisions.
An uncommitted worktree is allowed; unexplained source or environment drift is
not. Do not reset unrelated changes to obtain a clean status.

## Use the installed environments

Run from the NRL checkout. These locations describe the current internal
workspace, not an installation requirement for downstream Curator users.
Choose a new qualification directory if the example already exists.

Keep this working directory fixed when freezing, verifying, and running the
qualified commands. Changing into the nested Curator checkout can expose local
package metadata and alter discovered package versions without any installation.
If that causes a fingerprint mismatch, restore the recorded working directory
and verify the baseline again. Do not relax the fingerprint check.

```bash
cd /raid/kyzheng/nv-ingest
NRL_REPO=/raid/kyzheng/nv-ingest
NRL_PY=/raid/kyzheng/.runtime/nrl-curator-mvp/nrl/bin/python
CURATOR_PY=/raid/kyzheng/.runtime/nrl-curator-mvp/curator/bin/python
RECIPE_DIR="$NRL_REPO/Curator/tutorials/interleaved/nemotron_parse_pdf"
BASE_EVIDENCE="$NRL_REPO/.e2e/nrl-curator-poc-20260918"
QUALIFICATION_TOOLS="$NRL_REPO/.e2e/nrl-curator-qualification-20260918"
QUALIFICATION_DIR="$NRL_REPO/.e2e/nrl-curator-qualification-20260918"
OBSERVER="$BASE_EVIDENCE/observe_run.py"
MODEL_SNAPSHOT=/raid/kyzheng/.cache/hf-nrl-curator/hub/models--nvidia--NVIDIA-Nemotron-Parse-v1.2/snapshots/2bd0189bffd6cdded6280d9f22a4077b25a504e3
```

Keep caches and temporary data on `/raid`. Set the following environment before
freezing the baseline and reuse it for observed runs.

```bash
export HF_HOME=/raid/kyzheng/.cache/hf-nrl-curator
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=/raid/kyzheng/.runtime/nrl-curator-mvp/cache
export TMPDIR="$QUALIFICATION_DIR/tmp"
export RAY_TMPDIR=/raid/kyzheng/nv-ingest/.e2e/qv
export VLLM_CACHE_ROOT=/raid/kyzheng/nrl-curator-mvp/vllm
export VLLM_CONFIG_ROOT=/raid/kyzheng/nrl-curator-mvp/vllm-config
export TORCH_HOME=/raid/kyzheng/nrl-curator-mvp/torch
export TRITON_CACHE_DIR=/raid/kyzheng/nrl-curator-mvp/triton
export TORCHINDUCTOR_CACHE_DIR=/raid/kyzheng/nrl-curator-mvp/torchinductor
export FLASHINFER_WORKSPACE_BASE=/raid/kyzheng/nrl-curator-mvp/flashinfer
export NUMBA_CACHE_DIR=/raid/kyzheng/nrl-curator-mvp/numba
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$QUALIFICATION_DIR/metrics"
```

Use a short, unused Ray temporary path for each attempt, such as `.e2e/qv`
for ingestion or `.e2e/qc` for consumption. Ray dashboard components add longer
socket names than the core executor. An overlong path can therefore break
dashboard metrics even when the pipeline succeeds. Keep the full generated
socket pathname within the platform limit, not just the temporary directory.
Put the selected runtime's `bin` directory first on `PATH` when launching that
runtime, so subprocess tools resolve in the same environment. Do not combine
the two GPU dependency stacks or synchronize either environment during
qualification.

Inspect GPU identities and existing processes before choosing an idle GPU.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Use the selected UUID for NRL ingestion's `CUDA_VISIBLE_DEVICES` and the
observer's `--gpu` argument. The pinned Curator vLLM 0.22 runtime instead
requires a numeric comparison GPU selector, as described below. Do not
terminate unrelated GPU processes. Native comparison and NRL capacity runs
must not overlap.

## Freeze and verify one baseline

The observer and cohort selector are private workspace tools, not public NRL
interfaces. Prepare the deterministic stress selection before freezing, so
the baseline binds every planned input cohort. This command selects and hashes
inputs without rendering or inference. Its output directory must not exist.

```bash
"$NRL_PY" "$QUALIFICATION_TOOLS/prepare_capacity.py" \
  --inventory "$BASE_EVIDENCE/cohort/corpus_inventory.jsonl" \
  --reviewed-manifest "$BASE_EVIDENCE/cohort/full-1278-reviewed-blanks.jsonl" \
  --output-dir "$QUALIFICATION_DIR/cohort"
```

For the current prepared workspace, retain its existing `cohort/stress.jsonl`
and `cohort/selection.json` instead of rerunning into that destination. The
historical filename `reviewed-blanks` denotes agent-reviewed declarations, not
human approval. Human acceptance remains a separate gate. If review changes
declarations or inputs, freeze a new baseline and rerun the affected gates.

Create the exclusive baseline record and verify it with these commands.

```bash
"$NRL_PY" "$OBSERVER" \
  --freeze-baseline "$QUALIFICATION_DIR/baseline.json" \
  --nrl-python "$NRL_PY" \
  --curator-python "$CURATOR_PY" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --input-manifest "$BASE_EVIDENCE/cohort/vertical-6.jsonl" \
  --input-manifest "$BASE_EVIDENCE/cohort/pilot-100-reviewed-blanks.jsonl" \
  --input-manifest "$BASE_EVIDENCE/cohort/full-1278-reviewed-blanks.jsonl" \
  --input-manifest "$QUALIFICATION_DIR/cohort/stress.jsonl"

"$NRL_PY" "$OBSERVER" \
  --baseline "$QUALIFICATION_DIR/baseline.json" --verify-only
```

Baseline creation hashes model/tokenizer files, PDFs, manifests, and source
files and probes both runtime environments. It does not invoke Parse.
Every observed command below verifies the same baseline before and after
execution. Source or runtime drift invalidates the attempt even when the
underlying command exits successfully. Never edit the baseline to match an
unexpected change.

## Run the regression suites in their owning environments

These commands run existing test modules from the NRL checkout. `-ra` reports
every skip reason. Preserve the complete command output in the qualification
record, not only the passing test count.

```bash
"$NRL_PY" -m pytest -q -ra --maxfail=1 \
  nemo_retriever/tests/test_nemotron_parse_v1_2.py \
  nemo_retriever/tests/test_actor_operators.py \
  -k 'NemotronParse or raw_output or finish_reason'

"$NRL_PY" -m pytest -q -ra --maxfail=1 \
  --confcutdir=Curator/tests/tutorials/interleaved/nemotron_parse_pdf \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_graph.py \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_compare.py

"$NRL_PY" -m pytest -q -ra --maxfail=1 \
  --confcutdir=Curator/tests/tutorials/interleaved/nemotron_parse_pdf \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_lance_runtime.py \
  -k test_schema_validation_checks_every_fields_nullability

"$NRL_PY" -m pytest -q -ra --maxfail=1 \
  --confcutdir=Curator/tests/tutorials/interleaved/nemotron_parse_pdf \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_lance.py \
  -k 'structural_vertical_slice or nrl_provenance'

"$CURATOR_PY" -m pytest -q -ra --maxfail=1 \
  --confcutdir=Curator/tests/stages/interleaved \
  Curator/tests/stages/interleaved/test_multimodal_core.py \
  Curator/tests/stages/interleaved/filter/test_blur_filter.py

"$CURATOR_PY" -m pytest -q -ra --maxfail=1 \
  --confcutdir=Curator/tests/tutorials/interleaved/nemotron_parse_pdf \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_lance.py \
  -k 'not nrl_provenance'

"$CURATOR_PY" -m pytest -q -ra --maxfail=1 \
  --confcutdir=Curator/tests/tutorials/interleaved/nemotron_parse_pdf \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_lance_runtime.py \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_compare.py \
  Curator/tests/tutorials/interleaved/nemotron_parse_pdf/test_pipeline_utils.py
```

An opposite-runtime skip is acceptable only when that case passes in its owning
environment. Missing Lance, Arrow, OpenCV, Pillow, or the native executor in
the required environment is a failed prerequisite, not successful
qualification. Curator inference comparisons additionally require the pinned
GPU dependencies described in the [comparison instructions](README.md#compare-inference-on-identical-rgb-pages).

The lifecycle selector runs NRL source-binding and the structural fixture in
NRL. Curator owns the complete lifecycle and process-failure suite; only its
opposite-runtime NRL source-binding case is deselected. The schema module's
stored-output and native-reader cases all execute in Curator. The NRL runtime
has LanceDB but not native `lance`; only the schema-only nullability checks run
there. Do not install another dependency stack or skip required Curator cases
to conceal that intentional environment boundary.

Retain the strict malformed-output, non-`stop`, nested-error, textless-Picture,
crop, reading-order, missing/duplicate/extra-page, blank, alias, basename,
corrupt/encrypted, and metadata-only cases. Require merged-table cell text and
LaTeX span commands to survive projection and the native Lance-to-Parquet
round trip unchanged. G2 additionally requires the following failure behavior.

| Injected condition | Required outcome |
| --- | --- |
| Same values with wrong Arrow type or field nullability | Reject the export before completion |
| Changed text, bbox, provenance, nulls, image bytes, or row keys | Reject exact reconciliation |
| Image with readable header but undecodable pixels | Native blur validation executes; missing output prevents completion |
| First or later Lance write fails, or table validation fails | No handoff marker |
| Source bytes change, or current Lance version drifts | Reject the relevant publication phase |
| Report or marker write fails before atomic visibility | No marker for that phase |
| Directory synchronization fails after marker visibility | Preserve the valid marker; report durability unconfirmed |
| Run ID, output directory, or completed run is reused | Reject reuse without altering existing artifacts |
| Process is terminated before handoff or during consume | No false completion; clean up owned processes; fresh retry passes |

A negative test passes when the unsafe input is rejected correctly. It must
not weaken the validator or relabel the bad extraction as successful.

For `validated_pages_v1`, test page-atomic rejection: an invalid element causes
all elements on its page to be withheld, while validated pages can still be
published. Require every expected page in `page_outcomes`, original source page
numbers, contiguous positions over delivered elements, and explicit document
issues. Test missing, duplicate, and unexpected page records without silently
turning them into successful extraction. An undeclared empty response is
`unexpected_empty_output`, not a blank-page declaration. Cover no-valid-page
documents with no rows, mixed valid/failed pages with status `partial`, and
metadata-only partial documents whose valid pages are declared blanks. Reconcile
complete/partial/failed document counts and delivered/failed page counts
separately, including aliases. Older strict-policy handoffs must remain readable.

The comparison cleanup additionally requires unchanged ordinary native token
defaults and no OpenCV import when image validation is disabled. Explicit
comparison validation must retain tables and declared metadata-only samples.
Executor statistics with evidence capture disabled must not alter outputs or
invoke the graph a second time. Benchmark tests must reject changed inputs,
configuration drift, incomplete exports, and unapproved human review in default
qualified mode. Explicit diagnostic mode must retain pending quality status
and resource enforcement. Keep failed or truncated inputs visible in accounting.

## Run a fresh observed vertical slice

After the suites pass, select the verified idle GPU UUID. The following UUID
identifies the current qualification host's physical GPU 0; confirm it with
the preceding `nvidia-smi` commands before use.

```bash
QUALIFICATION_GPU=GPU-89514d23-7772-ea8c-daea-7c51c41e8b96

PATH=/raid/kyzheng/.runtime/nrl-curator-mvp/nrl/bin:"$PATH" \
CUDA_VISIBLE_DEVICES="$QUALIFICATION_GPU" \
"$NRL_PY" "$OBSERVER" \
  --baseline "$QUALIFICATION_DIR/baseline.json" \
  --output-prefix "$QUALIFICATION_DIR/metrics/vertical-ingest-01" \
  --interval 1 --gpu "$QUALIFICATION_GPU" \
  --watch-root "$QUALIFICATION_DIR" --watch-root "$RAY_TMPDIR" \
  -- "$NRL_PY" "$RECIPE_DIR/nrl_lance.py" ingest \
  --manifest "$BASE_EVIDENCE/cohort/vertical-6.jsonl" \
  --output-root "$QUALIFICATION_DIR/runs" --run-id vertical-01 \
  --projection-workers 8 --nrl-repo "$NRL_REPO"

PATH=/raid/kyzheng/.runtime/nrl-curator-mvp/curator/bin:"$PATH" \
CUDA_VISIBLE_DEVICES="" \
"$CURATOR_PY" "$OBSERVER" \
  --baseline "$QUALIFICATION_DIR/baseline.json" \
  --output-prefix "$QUALIFICATION_DIR/metrics/vertical-consume-01" \
  --interval 1 \
  --watch-root "$QUALIFICATION_DIR" --watch-root "$RAY_TMPDIR" \
  -- "$CURATOR_PY" "$RECIPE_DIR/nrl_lance.py" consume \
  --handoff-manifest "$QUALIFICATION_DIR/runs/vertical-01/handoff_manifest.json" \
  --output-dir "$QUALIFICATION_DIR/exports/vertical-01" --mode error
```

This six-input fixture includes multimodal pages, an exact duplicate, a
declared blank, and corrupt/encrypted inputs. Check explicit outcomes, the
handoff and completion seals, schema/native-reader reconciliation, and both
observer summaries. The observer's `qualification_passed` reports only its
command, fingerprint, and monitoring checks. It does not certify human review,
capacity, or all G1 through G6. Observations without `--capacity-check` are
diagnostic, not an enforced capacity gate.

## Preserve the exact export contract

The native Parquet writer receives the recipe's 18-field Arrow schema through
its existing schema/write options. Reconciliation compares logical field
types and nullability as well as every row value. Required non-nullable fields
are `sample_id`, `position`, `modality`, `content_sha256`, and `run_id`.
Arrow metadata and equivalent nested list-child names are not content changes.
The 18 Arrow fields remain unchanged under `validated_pages_v1`. Metadata JSON
and document reports add `extraction_status`, document issues, and
`page_outcomes` with zero-based page identity, status, element count, and issues.
Reconcile this metadata exactly, including failed pages with no content rows.

Table elements preserve model-native bodies through NRL's existing
`table_format="latex"` option. The unchanged `table`/`text/markdown` convention
matches native Curator; it does not promise Markdown pipe-table rendering.
Do not normalize these bodies into HTML or pipe tables during qualification.
Earlier exports with converted tables remain historical evidence, not proof
of native-table preservation. Freeze a new baseline and rerun the affected
projection and export checks after changing this representation.

The normal Curator pipeline remains a pinned `InterleavedLanceReader`,
non-dropping aspect-ratio and blur filters, and `InterleavedParquetWriterStage`.
Image materialization is disabled because bytes are inline. Both filters opt
into metadata-only preservation, which remains disabled by default elsewhere.
The output is reopened through Curator's native Parquet reader and reconciled
before completion. This proves native reader compatibility, not merely that
PyArrow can open the files.

The consume report records `reconciliation.schema_and_nullability_validated`
and `reconciliation.native_parquet_reader_validated` as true only after these
checks pass. `native_parquet_reader_seconds` is reported separately and is
also included in total reconciliation time. Do not add it to that total again.
The `pipeline_and_reconciliation_seconds` field is a partial-stage timer,
not command-launch-to-completion timing.

## Interpret markers and recover without overwriting

Atomic marker visibility is the publication point for that phase. A directory
sync confirms durability after visibility; those events are not interchangeable.

| Artifact | Meaning |
| --- | --- |
| No handoff | No validated Lance handoff; any partial files remain diagnostic artifacts |
| Sealed `handoff_manifest.json` | Validated immutable Lance version and accounted-for inputs; Curator export is not complete |
| Sealed `consume_validation.json` | Exact export reconciliation report; not a completion marker |
| Sealed `completion_manifest.json` | Accounted-for inputs and validated export bound to the handoff; not human semantic approval |
| `run_state.json` | Diagnostic state only; never substitutes for sealed evidence |
| `MarkerDurabilityUnconfirmedError` | A valid marker is already visible, but synchronization failed; qualification remains unsuccessful |

New markers record `publication_policy=validated_pages_v1`. A completed export
can contain validated pages from `partial` documents; it does not certify that
all source pages succeeded. Pages marked `failed` contribute no content rows.
Report complete documents, partial documents, failed documents, delivered pages,
delivered content pages/elements, and failed pages separately. A validated blank
page is delivered without content elements. Existing sealed strict-policy
handoffs remain consumable without rewriting their markers or outcomes.
Complete and partial document counts describe structural outcomes, not human
content accuracy. The exported metadata preserves those outcomes alongside the
content. This publication change adds no inference retries and does not recover
content that Parse omitted. It requires a new qualification baseline; it does
not establish a full-corpus rerun or human approval.

For an interrupted or failed run, retain its log, report, source fingerprint,
and partial artifacts. Verify that its recorded driver and descendant processes
have exited. Check the GPU process list before retrying. Signal only processes
whose ownership and exact IDs you have verified; never use machine-wide Ray
shutdown or broad process-name matching on a shared host.

Retry failed ingestion with a fresh run ID and fresh consumer output. For a
consumption-only failure with no visible completion marker, verify the sealed
handoff, immutable table version, and unchanged source inventory. You can then
run `consume` with that same handoff and a fresh output directory. This avoids
another Parse invocation. Never append to the handed-off table or resume
writes in the previous consumer output. Consumer output must be fresh;
`--mode error` is the only supported writer mode. An existing completion
marker, including one with unconfirmed durability, prevents re-consumption.

When a marker is visible but durability is unconfirmed, preserve it and record
the failure separately. Verify its seal, expected digest, referenced handoff,
Lance version, source inventory, and exported artifacts. The private contract
helper `_confirm_marker_durability(path, hash_field, expected_sha256=...)`
checks the marker seal and expected digest, then synchronizes the file and
parent directory without replacing it. Use the expected digest from the
independently retained attempt evidence, not a newly accepted replacement.
This helper is not a public recovery API and does not validate its dependencies
for you. `consume` confirms handoff durability before proceeding.

Successful verification and synchronization resolve the durability issue;
they do not retroactively convert an unsuccessful qualification attempt into
a green result. Record the recovery and reassess all affected gates. If any
dependency changed or cannot be verified, preserve the incident evidence and
retry from a fresh run directory.

Retain failed-run data through review. If you later authorize cleanup, identify
the exact inactive run and consumer directories, verify they contain no source
PDFs or shared caches, and prefer moving them to a quarantine location. Do not
delete a workspace, corpus, cache root, or an unresolved variable path.

## Obtain human content-quality approval

Use the existing `review/extraction-pilot/` evidence under `BASE_EVIDENCE`.
Its 50 pages span 25 documents; under the historical strict policy, 30 reviewed
pages belong to accepted documents and 20 to withheld documents. Preserve that
historical distinction and record current page delivery separately when reviewing
new outputs.
The user names the human reviewer and remains the sign-off owner. An agent
must not sign the review or mark `human_reviewed` true on the reviewer's behalf.

For each reviewed page, record the reviewer, date, document SHA-256, page
number, evidence/output hashes, publication status, decision, and rationale.
Assess content coverage, class/order, text and table meaning, bbox alignment,
crop completeness, and image decoding. Obtain language-qualified help where
needed. Classify findings as bridge defects, model defects, renderer/runtime
differences, or accepted representation differences.

Use human-adjudicated regions, not element positions, as the reference for
content precision and recall. Splitting one paragraph into two model elements
must not count as an omission or duplicate by itself. Record these facets
separately; no single recall score establishes extraction quality.

| Facet | Human reference and measurement |
| --- | --- |
| Content coverage | Expected regions recovered and unsupported regions emitted, by modality; report precision and recall together |
| Text | Missing or incorrect text, including meaning-changing errors |
| Tables | Cell meaning, header associations, and merged-cell relationships |
| Reading order | Region order and label/value associations |
| Pictures | Textless-picture coverage, bbox alignment, and complete crops |
| Delivery | Complete, partial, and failed documents; delivered content, validated blanks, and failed pages |

Until the human reference is approved, report observations and unresolved
findings instead of an accuracy percentage. Shared model defects do not become
correct outputs merely because both extraction paths reproduce them.

The limitations register must name each issue, affected document/page hashes,
published or withheld status, severity, human acceptance or rejection, and the
approved scope of use. Existing counterexamples include omitted signatures and
logos, clipped crops, missing cover imagery, OCR meaning changes, and omitted
form marks. A stopped, well-formed generation alone does not prove that Parse
captured every visible region.

Replay eligible frozen responses through both translators. Preserve malformed,
non-`stop`, and undeclared-empty cases as explicit negative outcomes. Expand
identical-RGB inference with deterministic content-identity representatives
for born-digital, scanned, table, picture, and multilingual content. Retain the
selection manifest and use three alternating repetitions per engine on the same
physical GPU. Record warmup and cache conventions; explain response differences.
Keep the native 50-page cap visible and exclude truncated long documents from
paired completeness metrics. The [existing comparison commands](README.md#compare-frozen-translation-ownership)
remain the execution interface.

Immediately before a GPU comparison, resolve the desired physical UUID to its
current numeric index. The pinned Curator vLLM 0.22 architecture inspection
parses `CUDA_VISIBLE_DEVICES` entries as integers; passing a UUID fails before
model startup. Do not assume the numeric index remains stable between runs.

```bash
nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
COMPARISON_GPU_INDEX=0  # Replace with the index matching QUALIFICATION_GPU.
test "$(nvidia-smi --id="$COMPARISON_GPU_INDEX" --query-gpu=uuid --format=csv,noheader,nounits)" = "$QUALIFICATION_GPU" || exit 1
```

Pass `--gpu "$COMPARISON_GPU_INDEX"` to `nrl_compare.py inference`. It sets the
same numeric `CUDA_VISIBLE_DEVICES` for both engine children and records and
checks the observed physical GPU UUID. For `native-product`, also set
`CUDA_VISIBLE_DEVICES="$COMPARISON_GPU_INDEX"` before starting that command.
Keep the observer's `--gpu "$QUALIFICATION_GPU"` identity check as a UUID.
Comparisons are not `--capacity-check` runs; NRL capacity commands below retain
UUID-based selection and monitoring.

If UUID selection caused architecture inspection to fail, retain the failed
observation and any completed engine artifacts. Retry with a fresh comparison
output path and observer prefix after verifying the numeric mapping. Do not
overwrite or combine partial attempts into a passing comparison.

G3 passes only with no unresolved bridge loss or meaning-changing regression
on the reviewed set, and explicit human acceptance of remaining limitations.
A rejected limitation pauses progression for a separately scoped investigation.
Do not modify prompts, models, blank declarations, or validators to bypass it.

## Assemble source-grounded review evidence

Run `source-evidence` outside timed extraction in the Curator CPU environment.
Provide a sealed, fully collected benchmark report and the study's existing
private historical capture index. The index binds the fixed tuning-page
selection, source PDFs, historical page images, and raw responses. Held-out
pages are not eligible for this tuning review.

Replace the example input paths with the retained artifacts. The output
directory must not exist. `--repetition` selects a zero-based benchmark pair
and defaults to `0`.

```bash
"$CURATOR_PY" "$RECIPE_DIR/nrl_compare.py" source-evidence \
  --benchmark-report /raid/nrl-curator-benchmark/benchmark_report.json \
  --historical-capture-index /raid/nrl-curator-review/quality-historical-capture-index.json \
  --repetition 0 \
  --output-dir "$QUALIFICATION_DIR/source-evidence-01"
```

The command checks seals, source identities, and the exact physical Parquet
inventory and hashes. It writes a sealed `report.json` and content-hashed
`blobs/` containing source images, historical responses, and delivered picture
bytes. Selected rows retain their original text, table bodies, nulls, row
identities, bboxes, and provenance. Native invalid rows remain negative evidence;
document metadata, unmappable rows, and NRL partial/failed outcomes remain
visible. Native missing output is not proof of a blank or failed page.

Historical responses describe their original capture, not fresh timed
generation. The report records `status=assembled`, `semantic_status=not_judged`,
and `human_calibrated=false`. It performs no rendering, inference, GPU work,
judge calls, or publication changes. Assembly is not semantic equivalence,
an accuracy score, or human approval.

Inspecting these sealed historical artifacts does not require a replacement
benchmark baseline. Preserve their original fingerprints. Freeze the updated
sources before subsequent benchmark executions; do not rewrite old baselines.

### Request report-only model observations

The optional `judge` command is not offline. Obtain approval for an
image-capable, OpenAI-compatible endpoint and model before sending document
content. Neither has a default. Use the existing Curator client environment;
do not start a new inference service for this workflow. If authentication is
required, supply `OPENAI_API_KEY` through the environment. Do not put secrets
in the URL: credentials, query parameters, and fragments are rejected.

Set the approved endpoint and model, then choose a fresh output directory.
The token limit defaults to `4096`, and the request timeout defaults to
`120` seconds.

```bash
: "${JUDGE_BASE_URL:?Set the approved image-capable endpoint}"
: "${JUDGE_MODEL:?Set the approved judge model}"
"$CURATOR_PY" "$RECIPE_DIR/nrl_compare.py" judge \
  --source-evidence "$QUALIFICATION_DIR/source-evidence-01/report.json" \
  --output-dir "$QUALIFICATION_DIR/judge-01" \
  --base-url "$JUDGE_BASE_URL" --model "$JUDGE_MODEL" \
  --max-tokens 4096 --timeout-seconds 120
```

After validating the packet seal and referenced hashes, the command sends one
sequential request per selected page, without automatic retries. Each request
includes actual source pixels, available decodable picture crops, and delivered
text and table bodies. Route labels alternate between A and B; representation differences
can still reveal the route. Historical raw generations are not sent. Unmappable
rows remain unjudged evidence rather than being assigned to a guessed page.

The sealed `report.json` retains settings, prompt, row mappings, timings, and
usage when returned. Content-hashed blobs retain request messages and raw
responses; treat them as document data. The configured API key is not recorded.
Recorded model settings are requested values, not verified server weights or
resources.
Only a non-refused, `stop`-completed response with valid JSON, source regions,
and row references can become `judged_uncalibrated`. Transport or response
failures remain `unjudged_error`, not a semantic pass or an uncertain finding.

Every report remains `report_only=true` and `human_calibrated=false`. Findings
are non-exhaustive model observations, not recall, accuracy, completeness, or
human approval. Calibrate against human-adjudicated examples, including false
approvals and false alarms, before making quality claims. Keep judge cost
outside extraction timing. It never changes publication policy or recovers
missing content. An unavailable endpoint or pending calibration does not block
independent performance experiments.

## Benchmark complete products after human approval

Use the fresh-product benchmark after G1 through G3 pass. It compares configured
products, not an isolated claim about the two libraries. Keep native comparison
runs sequential and separate from capacity runs on the shared machine.
The [diagnostic procedure](#collect-authorized-diagnostic-evidence-before-human-approval)
below permits explicitly authorized measurements before G3 without declaring
the implementation qualified.

The current runner uses Curator's local vLLM stage with `RayDataExecutor`.
It does not execute the recommended Dynamo-backed `main.py` topology. Label
the baseline accordingly; a local-vLLM result does not establish a comparison
against the recommended serving configuration.

Prepare the cohort before freezing its baseline. This step inventories the
existing pilot independently of extraction outcomes, selects eligible original
PDFs, and creates content-addressed native input copies. Both output locations
below must be unused.

```bash
BENCHMARK_COHORT="$QUALIFICATION_DIR/benchmark-cohort"
BENCHMARK_RUN="$QUALIFICATION_DIR/benchmark-01"

"$NRL_PY" "$RECIPE_DIR/nrl_compare.py" prepare-benchmark \
  --manifest "$BASE_EVIDENCE/cohort/pilot-100-reviewed-blanks.jsonl" \
  --corpus-root /raid/kyzheng/nemotron_parse_pdf/pdfs \
  --output-dir "$BENCHMARK_COHORT"

"$NRL_PY" "$OBSERVER" \
  --freeze-baseline "$QUALIFICATION_DIR/benchmark-baseline.json" \
  --nrl-python "$NRL_PY" --curator-python "$CURATOR_PY" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --input-manifest "$BENCHMARK_COHORT/manifest.jsonl"
```

Retain the sealed `cohort.json`, canonical `manifest.jsonl`, native manifest,
and PDF copies. The selection excludes duplicate aliases, unreadable inputs,
PDFs outside the original corpus, declared all-blank documents, and documents
over the native 50-page limit. Exclusion reasons remain in the cohort report;
excluded inputs do not silently disappear from the source-pilot accounting.

The user-designated reviewer must supply the approved review record. Set
`HUMAN_REVIEW` to that record, not an unsigned agent review. The current pending
review packet does not authorize this run. Verify the GPU index-to-UUID mapping
and idle state immediately before executing the command.

Set `BENCHMARK_TEMPORARY_BYTES` to a reviewed, conservative byte forecast from
available pilot/stress measurements. There is no arbitrary default. The runner
uses the observer's host, GPU, and disk headroom checks for every execution.
Keep Ray and temporary paths short, unused, and on `/raid`, as described above.

```bash
: "${HUMAN_REVIEW:?Set the human-approved review record path first}"
: "${BENCHMARK_TEMPORARY_BYTES:?Set the reviewed temporary-byte forecast first}"

"$NRL_PY" "$RECIPE_DIR/nrl_compare.py" benchmark \
  --cohort "$BENCHMARK_COHORT/cohort.json" \
  --output-dir "$BENCHMARK_RUN" \
  --nrl-python "$NRL_PY" --curator-python "$CURATOR_PY" \
  --nrl-repo "$NRL_REPO" --model-snapshot "$MODEL_SNAPSHOT" \
  --gpu "$COMPARISON_GPU_INDEX" \
  --observer "$OBSERVER" \
  --baseline "$QUALIFICATION_DIR/benchmark-baseline.json" \
  --human-review "$HUMAN_REVIEW" \
  --projected-temporary-bytes "$BENCHMARK_TEMPORARY_BYTES" --repetitions 3
```

The review gate requires `human_reviewed=true` and an approved `human_signoff`
with `owner`, `reviewer`, and `signed_at_utc`. Every reviewed page must have
`human_reviewed=true`; every limitation must have `human_decision` equal to
`accepted` or `resolved`. The review's `qualification_fingerprint.sha256` must
match the SHA-256 of the selected baseline file bytes. A different baseline or
an incomplete review is rejected before measured execution.

Those fields record human decisions; an agent must not set them to bypass a
gate. Regenerate and adjudicate the current-source review packet, retaining its
output/source bindings and any approved addendum. Do not turn historical
pending findings into approvals by changing flags.

Retain `benchmark_report.json`, every fresh run, and all observer logs and
summaries. Measure command launch through checked export externally; include
Lance publication and both process startups in the NRL path. Internal graph,
publication, read, transform, writer, and reconciliation timers are diagnostic.
Separate preparation and fingerprint-check overhead from product execution.

Use three alternating pairs with existing model/disk caches and fresh
processes. Warm caches are the intended convention, not verified residency.
Do not flush shared caches or enable evidence capture. Report each
repetition, median, range, paired differences, and failures. Record these facets
without combining them into a single success or speed score.

| Facet | Required interpretation |
| --- | --- |
| Work attempted | Fixed selected input, document, and expected-page counts, including failed extractions |
| Work delivered | Complete/partial/failed documents, NRL delivered and failed pages, delivered content pages/elements, native observed content pages, rows, and modalities |
| Time | Full command wall time and stage diagnostics; no sum of overlapping worker timers |
| Memory | Driver and worker RSS, host pressure, GPU utilization/memory, and compact result size |
| Storage and cleanup | Ray spill, temporary disk footprint, failed commands, and surviving owned workers |
| Quality | Human region-based coverage, meaning, order, tables, and crop findings, separately from performance |

For benchmark observations, `driver_rss` identifies the root benchmark
launcher, not the NRL ingest driver running as its child. Raw `process_sample`
records retain PID, parent PID, process name, and RSS for ingest/consume children
and Ray workers. `sampled_process_rss_peaks` summarizes sampled maxima by PID
and creation time, with parent PID and command. Attribute the ingest and consume
drivers by their exact command and process ancestry, not the name `python`.
Tree RSS includes the complete descendant tree and can double-count shared
pages. Do not label either aggregate as the
NRL ingest driver's peak memory.

Native output does not expose per-page outcomes or finish reasons. Label these
unavailable, and do not equate observed native content pages with validated NRL
page completeness. Non-restrictive native image filters can still discard
undecodable images and reindex rows. Export and native reread validate surviving
images, not absence of filter loss. NRL's exact Lance-to-Parquet reconciliation
detects dropped rows and prevents completion. Do not remove failed inputs to
improve pages per second.
Missing measurements are unresolved evidence, not zero usage or proof of safety.

Recommend a performance advantage only after quality and completeness remain
acceptable and repeated paired observations support it. Three repetitions are
descriptive, not a statistical significance test. If ranges overlap or results
vary materially, report the speed comparison as inconclusive. A reliable bridge
can still satisfy this internal MVP when equally fast or slower.

### Tune one variable at a time

Freeze a representative short-document cohort and reserve document identities
for held-out confirmation before tuning. Keep supplemental failures and long
documents separate from the matched performance denominator. Fix the model,
prompt, renderer, partial-publication policy, and downstream Curator settings
throughout a comparison series.

The optional ingest controls are `--parse-cpus` (default `1`, integer at least
`1`) and `--parse-batch-size` (default `64`, integer at least `2`). They reserve
Ray CPUs and request page batches for the existing Parse actor. Batch size `1`
is rejected because the pinned executor would silently promote it to `64`.
These controls retain one Parse actor and one GPU. They do not change model
sequence limits or introduce extra models or inference retries.
Nondefault values require `batch` run mode; `inprocess` mode rejects them.

The optional `--projection-block-rows` ingest flag, or
`--nrl-projection-block-rows` benchmark flag, accepts a positive integer and
requires `batch` mode. Its default is `None`: no repartition is added.
When set, it targets page rows per Ray block immediately before the existing
CPU projection, not the Parse batch size or output element count. It preserves
the worker ceiling, model, schema, publication policy, and default graph.

Use the benchmark equivalents below, changing one value from the baseline
per experiment. Give every experiment a fresh output directory.

| Experiment | Benchmark arguments | Question |
| --- | --- | --- |
| Baseline | `--nrl-parse-cpus 1 --nrl-parse-batch-size 64 --native-pdfs-per-task 10` | What does the unchanged operating configuration cost? |
| NRL CPU reservation | `--nrl-parse-cpus 4 --nrl-parse-batch-size 64 --native-pdfs-per-task 10` | Does CPU work inside the GPU-owning actor limit useful throughput? |
| NRL page batches | `--nrl-parse-cpus 1 --nrl-parse-batch-size 128 --native-pdfs-per-task 10` | Does a larger requested page batch reduce gaps between useful model work? |
| Native task grouping | `--nrl-parse-cpus 1 --nrl-parse-batch-size 64 --native-pdfs-per-task 20` | Does larger PDF task grouping improve the native baseline? |

CPU reservation does not guarantee more active threads. Requested page-batch
size does not guarantee every observed batch has that many pages.
`--native-pdfs-per-task` is a positive integer with default `10`; it controls
PDF grouping, not page-request concurrency or GPU count. Record resolved actor
resources, observed batch sizes, and model instances, not only CLI requests.
These comparison overrides do not change ordinary native tutorial defaults.

If executor statistics show too few coarse projection tasks to use the existing
worker pool, test `--nrl-projection-block-rows 16` as a separate follow-up.
Compare it with the same `--nrl-parse-cpus 1 --nrl-parse-batch-size 128`
configuration with projection block sizing unset. Keep the projection ceiling
at eight workers and all other controls unchanged. This isolates a scheduling
hypothesis; it does not establish a speed gain. Record actual block/task counts,
projection durations, repartition cost, memory, and full validated-export time.
The row target is not an exact task-size guarantee or bounded-memory execution.

Separate model startup and full command turnaround from the warmed extraction
interval. Align stage activity with GPU utilization, CPU work, queue gaps,
memory, and spill. Aim for sustained utilization near 100% during useful
extraction compute, while preserving at least 10% GPU memory and 25% effective
host-memory headroom. GPU memory reservation is not compute utilization.
Retain the disk budget below; never add duplicate inference or other busywork
to improve a utilization chart.

Judge experiments using attempted and delivered pages per second, complete/
partial/failed outcomes, content findings, total wall time, and total resources.
Reject apparent gains caused by omitted content or invalid exports. Human
quality gaps limit recommendations, but authorized diagnostic performance
experiments can continue independently. Run quality review outside extraction
timers.

Combine only changes that help in separate experiments. Freeze the resulting
configuration, repeat alternating pairs, then confirm on the held-out inputs.
If source or execution semantics change, record a new baseline. Do not promote
a candidate based only on higher utilization or one faster repetition.

### Collect authorized diagnostic evidence before human approval

Use `benchmark --diagnostic` only when the user authorizes measurements before
human adjudication. Omit `--human-review`; do not manufacture an approved record
or change pending fields in the existing packet. Without `--diagnostic`, the
default approved-review requirement remains unchanged.

For a prepared cohort and a baseline binding its manifest, run the same
observed workflow with an explicitly recorded native reservation:

```bash
"$NRL_PY" "$RECIPE_DIR/nrl_compare.py" benchmark \
  --diagnostic \
  --cohort "$BENCHMARK_COHORT/cohort.json" \
  --output-dir "$BENCHMARK_RUN" \
  --nrl-python "$NRL_PY" --curator-python "$CURATOR_PY" \
  --nrl-repo "$NRL_REPO" --model-snapshot "$MODEL_SNAPSHOT" \
  --gpu "$COMPARISON_GPU_INDEX" --observer "$OBSERVER" \
  --baseline "$QUALIFICATION_DIR/benchmark-baseline.json" \
  --native-gpu-memory-utilization 0.85 \
  --projected-temporary-bytes "$BENCHMARK_TEMPORARY_BYTES" --repetitions 3
```

The `0.85` value is an authorized comparison-only adjustment for this host,
not a new native tutorial default or a guarantee of measured headroom. Omitting
the flag preserves the native configuration. Resource budgets stay at 25%
host-memory headroom, 10% GPU-memory headroom, and free disk greater than twice
the forecast temporary footprint. Stop and retain evidence if a budget fails.

A successful diagnostic report has status `diagnostic_completed`,
`quality_status=pending_human_review`, and `qualified=false`. Its export
completion markers still prove only structural publication. They cannot approve
content limitations, supply human-grounded accuracy scores, or justify a
replacement recommendation. Preserve failures rather than selecting only
successful repetitions.

Diagnostic collection can continue after a native element-validation finding
only when execution, sealed provenance, configuration, and resource checks
pass. The affected repetition and overall report remain failed and unqualified.
`collection_complete` means the requested repetitions were collected, not that
their outputs passed validation. Retain operational timing summaries separately
from valid-output timing and throughput; do not count invalid exports as
successful delivery. Configuration mismatch, source drift, execution failure,
or a failed safety check stops collection. Qualified mode still stops at the
first failed repetition.

Start with small and pilot cohorts. Proceed to stress and full-corpus
diagnostics only after preceding measurements establish safe projected
capacity. For the full configured-product comparison, prepare a fresh cohort
with `--selection full-corpus` and the original full manifest, then freeze a
new baseline binding it. Full-corpus comparison requires `--diagnostic`, even
when a human review exists, because native output remains capped. The default
`--selection matched` retains the
at-most-50-page paired-workload policy.

The historical full inventory contains 1,209 PDFs with at most 50 pages
(7,936 pages) and 69 longer PDFs (11,668 pages). All 1,278 inputs total 19,604
pages. Native's 50-page cap limits attempted work to at most 11,386 pages,
leaving 8,218 original pages outside its configured scope. Revalidate these
counts against the current inventory. Report capped long-document outcomes
separately; do not describe their outputs as complete or use capped time as an
equal-work speed comparison. Declared blank pages also require separate
accounting because native output lacks per-page outcome markers.

Retain the existing 50-page packet, unsigned inference supplement, and blank
declaration provenance. Continue human review independently while gathering
diagnostics. A human sign-off owner must still accept the resulting limitations
and full-run failure breakdown before qualification or replacement readiness.

### Historical native capacity boundary

The September 21, 2026 [two-page native control](../../../../.e2e/nrl-curator-benchmark-20260921/native/control-01/product_comparison.json)
passed export and native reread. Its [observer record](../../../../.e2e/nrl-curator-benchmark-20260921/metrics/native-control-01.summary.json)
reports `capacity_enforced=false`; it is diagnostic evidence, not capacity
qualification. Sampled GPU use peaked at 131,241 MiB of 143,771 MiB, leaving
approximately 8.7% headroom. The benchmark's required 10% headroom would reject
this configuration. The pinned native runtime logs
`--gpu-memory-utilization=0.9200` for this control.

Do not lower the budget or silently change native settings. The subsequently
authorized diagnostic workflow uses an explicit comparison-only reservation
adjustment and must remeasure it against the unchanged budgets. Retain the
original configuration's failure as evidence. Human content-quality approval
remains pending; neither the control's export success nor later diagnostic
completion supplies that approval.

## Prove capacity before the full corpus

After G1 through G3 pass, run the existing 100-input pilot without
`--evidence-root`. Use `--executor-stats` to save resolved execution settings
and Ray stage statistics without retaining page images or raw responses.
Explicitly authorized diagnostics can collect these capacity measurements
before G3; label their quality status pending and do not call them qualification.
Then run the deterministic stress cohort. Select the union
of the five longest PDFs, five largest estimated 200-DPI RGB render workloads,
and five highest image-object-count PDFs. Deduplicate and break ties by content
SHA-256. Record the selection inputs and manifest. The corpus includes a
1,080-page document, beyond the historical pilot's 391-page maximum.

Set `QUALIFICATION_TEMPORARY_BYTES` to the conservative byte estimate recorded
in your capacity report. Do not copy a historical footprint as a guarantee.
The observer requires a positive estimate, a frozen baseline, an idle physical
GPU UUID, and watched paths before enabling capacity enforcement. After G3
approval, launch the pilot with the following command.

```bash
: "${QUALIFICATION_TEMPORARY_BYTES:?Set the reviewed temporary-byte forecast first}"

PATH=/raid/kyzheng/.runtime/nrl-curator-mvp/nrl/bin:"$PATH" \
CUDA_VISIBLE_DEVICES="$QUALIFICATION_GPU" \
"$NRL_PY" "$OBSERVER" \
  --baseline "$QUALIFICATION_DIR/baseline.json" \
  --output-prefix "$QUALIFICATION_DIR/metrics/pilot-capacity-ingest-01" \
  --interval 1 --capacity-check --gpu "$QUALIFICATION_GPU" \
  --projected-temporary-bytes "$QUALIFICATION_TEMPORARY_BYTES" \
  --disk-root /raid \
  --watch-root "$QUALIFICATION_DIR" --watch-root "$RAY_TMPDIR" \
  -- "$NRL_PY" "$RECIPE_DIR/nrl_lance.py" ingest \
  --manifest "$BASE_EVIDENCE/cohort/pilot-100-reviewed-blanks.jsonl" \
  --output-root "$QUALIFICATION_DIR/runs" --run-id pilot-capacity-01 \
  --projection-workers 8 --executor-stats --nrl-repo "$NRL_REPO"
```

Run observed `consume` with that handoff and fresh `pilot-capacity-01` export
and observation names. Use the same capacity options during consumption and
keep the selected GPU idle. After reconciliation, update the forecast from the
pilot and repeat with `cohort/stress.jsonl` and fresh `stress-capacity-01`
names. Do not add `--evidence-root`. Do not run the stress command concurrently
with pilot consumption. Keep watched paths disjoint to avoid double-counting
the same temporary files.

Use one otherwise-idle H200, no overlapping comparisons, and one-second
observations. Record effective host/cgroup limits, host memory pressure, driver
and worker RSS, GPU memory/utilization, Ray spill, temporary disk footprint,
compact result bytes, stage times, failures, and worker cleanup. Summed process
RSS can double-count shared memory; sampled peaks can miss brief spikes.

Enforce the following minimum headroom throughout the run.

| Resource | Qualification budget |
| --- | --- |
| Effective host memory | At least 25% headroom against applicable host/cgroup limits |
| Selected GPU memory | At least 10% headroom |
| Disk | Free space greater than twice the projected temporary footprint |

Stop the owned run safely if a budget is breached. Missing required resource
measurements or inaccessible applicable limits are unresolved capacity evidence,
not proof of unlimited capacity. Inspect stranded processes and GPU allocations
after each run. Derive the full-run forecast conservatively from both pilot and
stress measurements. The historical sixfold compaction does not establish
bounded driver memory or full-corpus safety.

If driver materialization is the limiting factor, stop and propose a separate
bounded-memory change. Do not silently introduce batching or another graph.
G4 requires complete reconciliation, no out-of-memory failure or stranded
worker, and safe measured/projected headroom.

## Qualify the original corpus and hand off

Start G5 only after G1 through G4 pass. Use exactly the physically staged
`/raid/kyzheng/nemotron_parse_pdf/pdfs` corpus with reviewed blank declarations.
It contains 1,278 PDFs and 19,604 pages in the historical inventory. Rehash and
reconcile that inventory before execution. Synthetic fixtures remain outside
the full-corpus denominator. Send canonical representatives through the
existing graph once, retain strict page validation, and account for every alias.
Publish validated pages with explicit `partial` status when other pages fail;
do not publish any elements from a failed page or count a partial document as
complete extraction.

An authorized diagnostic full-corpus run can precede G3 only after structural
checks and measured pilot/stress capacity support execution. It follows the
same immutable publication and reconciliation rules, but is not a G5 pass.

Publish only the validated immutable Lance version. Require exact Parquet
schema, native-reader compatibility, row reconciliation, and every image hash.
Report inputs and aliases; complete, partial, and failed documents; delivered
pages, delivered content pages/elements, and failed pages; and modalities and
failure categories separately. Never count partial delivery as complete
extraction or omitted content as successfully extracted.
The human sign-off owner accepts or rejects the resulting limitation/failure
breakdown. A completed run can contain failed inputs, but no unexplained bridge
discrepancies or unresolved qualification blockers.

The final internal handoff binds the frozen baseline, command/test logs,
comparison and human review records, pilot/stress budgets, full-run artifacts,
failure accounting, and approved operating envelope. Record structural,
semantic, and capacity statuses separately. A performance winner or public
production release requires a separate decision.

## Requalify after an upgrade

Changes to NRL, Curator, Lance/Arrow, image decoders, the model/tokenizer,
rendering, prompt, decoding settings, or recipe publication policy invalidate
the prior qualification for the changed baseline. Preserve old evidence and
create a new fingerprint.
Run the owning-environment regression suites, G2 faults, a fresh GPU vertical
slice, and native export first. Repeat projection/inference comparisons and
human adjudication for changed outputs. Recheck capture-disabled pilot/stress
capacity before attempting another full run. Carry forward a limitation only
after the human owner confirms that its evidence and approved use still apply.

Before handing off source changes, run the existing configured hooks against
the exact changed files and `git diff --check`. Report unavailable tools or
unrelated failures; do not install new CI infrastructure, modify unrelated
files, or claim an unrun hook passed.
