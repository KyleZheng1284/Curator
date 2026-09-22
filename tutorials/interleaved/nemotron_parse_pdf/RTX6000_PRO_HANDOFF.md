# NRL-to-Curator benchmark handoff for a two-GPU RTX machine

Use this document as the task prompt for a new coding-agent conversation on the
target machine. Supply the two immutable commit URLs and the private transfer
bundle identified by the publishing conversation. Do not assume this machine
can access that conversation, the original workspace, or its private artifacts.

## Objective

Reproduce and extend the measured NRL-to-Curator PDF integration on a machine
described as having two RTX 6000 Pro GPUs. Verify the actual hardware first.
Establish a comparable **one-GPU** result, then evaluate **two-GPU scale-out**
as a separate configuration when it is safe and supported. Aim for near-100%
useful GPU activity during steady-state extraction, but choose configurations
using delivered content, end-to-end wall time, quality, and total resource cost.

Do not promise a speedup or require a favorable result. A 10% GPU-memory reserve
does not cap compute utilization at 90%. NVML utilization measures busy time,
not the percentage of theoretical FLOPs achieved.

Reuse the existing implementation and comparison tooling. Do not add another
graph, benchmark framework, HTTP service, retrieval pipeline, embeddings,
automatic retries, fallback models, or a new public NRL API.

## Obtain the implementation and private evidence

The implementation belongs to two separate repositories. Curator is nested in
the NRL working directory for these private observer scripts, not a submodule.

1. Obtain `NRL_REMOTE`, `NRL_COMMIT`, `CURATOR_REMOTE`, and `CURATOR_COMMIT` from
   the publishing conversation's handoff index. Use full 40-character commit
   identities, not an unpinned `main` branch or a moving feature branch.
2. Clone into new, unused directories under `/raid`. Use `NRL_REPO/Curator` for
   the Curator checkout. Fetch and check out the supplied revisions. Verify
   `git rev-parse HEAD` and `git status --short` separately in both repositories.
3. Read each applicable `AGENTS.md`, contributor guidance, this document,
   [README.md](README.md), and [QUALIFICATION_RUNBOOK.md](QUALIFICATION_RUNBOOK.md).
4. Obtain the private transfer archive through an authorized internal path.
   Verify its SHA-256 against the handoff index before unpacking it into a fresh
   staging directory. Inspect its member list first. Do not overwrite an
   existing `.e2e` tree or extract archive members outside the intended root.
5. The archive is **not** a replacement for the original PDFs, model weights,
   virtual environments, or complete historical Lance/Parquet datasets. Read
   its inclusion/exclusion index. Keep document evidence and machine-specific
   reports out of public Git commits.

Git carries these owning modules and their tests:

| Repository | Responsibility |
| --- | --- |
| NRL | `models/local/nemotron_parse_v1_2.py`: matching tokenizer revision and local completion status; `operators/extract/parse/nemotron_parse.py`: additive raw response/error metadata |
| Curator recipe | `nrl_graph.py`: existing NRL PDF stages plus terminal compact projection |
| Curator recipe | `nrl_lance_contract.py`, `nrl_lance_runtime.py`, `nrl_lance.py`: page/document accounting, Lance publication, native consumption, exact reconciliation, CLI |
| Curator recipe | `nrl_compare.py`: comparisons, fresh benchmark execution, benchmark-only GPU reservation override, source-evidence assembly, report-only judge |
| Curator core | `stages/interleaved/stages.py`: backward-compatible, default-off metadata-only sample preservation |
| Curator tutorial | `pipeline_utils.py`: explicit comparison/image validation without changing ordinary native defaults |

The private study tree is `.e2e/nrl-curator-performance-20260921`. Its primary
record is `EXPERIMENTS.md`; start with its opening summary and terminal audit,
not an older chronological checkpoint that says a run is still live.

The private observer requires these relative locations beneath `NRL_REPO`:

- `.e2e/nrl-curator-poc-20260918/observe_run.py` and `test_observe_run.py`.
- `.e2e/nrl-curator-performance-20260921/observe_performance.py` and its tests.
- That study's `run.sh`, `run_candidates.sh`, `profile_parse.py`,
  `test_profile_parse.py`, `prepare_cohort.py`, and `cohort/selection.json`.
- The qualification `prepare_*.py` files included in the private bundle.

The observer derives the checkout root from its own location and fingerprints
these sources. Preserve the relative layout. **Do not execute the historical
shell launchers unchanged:** they contain the old user's paths, H200 UUID,
baseline03, and earlier disk forecasts. Use the existing observer and comparison
entry points with verified new paths; do not rewrite historical sealed evidence.

## Established behavior that must remain intact

The route is PDF inventory/deduplication → existing NRL conversion/split/render
→ one local Nemotron Parse v1.2 actor → recipe-owned compact projection →
document finalization → Lance `pdf_elements` → native Curator Lance reader →
non-dropping aspect/blur image validation → native Parquet writer → native
Parquet reread and exact reconciliation.

- Parse runs once per representative page. No OCR, layout detector, separate
  table model, embedding model, or retrieval stage participates.
- Keep publication policy `validated_pages_v1`: publish validated pages, retain
  original page numbers, mark partial documents, and preserve failed-page
  identities/reasons in metadata. Unexpected empty output is **not** blank.
- A declared blank can be valid; malformed, truncated, invalid-bbox, or failed
  crop pages remain failures. Do not weaken validation to raise throughput.
- Preserve native table bodies, including their existing markup. Do not
  reintroduce a lossy table-to-Markdown conversion.
- Retain the 18-field contract, SHA-256 document identity, aliases, contiguous
  element positions, one metadata row, inline PNGs, null `source_ref` and
  `materialize_error`, one document per Lance fragment, and pinned versions.
- `handoff_manifest.json` means a validated Lance handoff. Completion requires
  native export and reconciliation. Neither marker certifies semantic accuracy.
- Ordinary native Curator defaults remain unchanged. Comparison validation is
  explicit; native's 50-page limit remains visible in accounting.
- The compact executor result still collects into pandas. This is not streaming
  or a general bounded-memory guarantee.

## Historical evidence, not a result for this machine

Both original routes used one physical H200 NVL with 143,771 MiB reported VRAM,
sequentially. The second H200 was unused. Warm disk/model caches, fresh processes,
no full response/image capture, and executor statistics were used.

| Scope | NRL-backed | Native Curator |
| --- | --- | --- |
| Matched tuning, 63 PDFs / 460 pages, three alternating pairs | Median 433.40 s; range 401.94–439.45 s | Operational median 312.68 s; range 292.97–319.39 s |
| Tuning delivery | 442 content pages + 4 blanks; 14 failures; 59 complete / 2 partial / 2 failed documents | 442 pages with valid rows; four invalid geometry rows each run; completeness unavailable |
| Fixed held-out set, 20 PDFs / 124 pages | 334.48 s; 122 content pages; 2 failures | 254.79 s; 123 pages with valid rows; one invalid geometry row |
| Full original corpus, 1,278 PDFs / 19,604 pages | 7,162.04 s (119.37 min), all pages attempted | 2,869.23 s (47.82 min), only 11,386 pages attempted |
| Full delivery | 1,095 complete / 136 partial / 47 failed documents; 18,698 content + 30 blank pages; 876 failed pages | 11,023 pages with valid content; 8,218 cap omissions across 69 long documents; 47 invalid rows on 42 pages |
| Full warmed / whole-command GPU utilization | 62.03% / 56.14% | 82.50% / 74.16% |
| Full peak GPU allocation / minimum free fraction | 117,871 MiB / 18.01% | 124,067 MiB / 13.71% |
| Full peak observed process-tree RSS | 73.67 GB | 61.98 GB |

The full times are **not an equal-work speed comparison**. Native invalid rows
remain failures even though execution succeeded. NRL showed no repeated
end-to-end speed advantage on the matched set. More elements do not prove more
correct information. No qualified performance or quality winner was selected.

The measured NRL configuration was Parse CPU1 / Ray batch128 / projection blocks
capped at16 / up to8 projection actors. Native used20 PDFs/task and a benchmark-
only GPU reservation0.85; NRL's model reservation was0.80. Both had one actual
inference model and `max_num_seqs=64`; request size is not simultaneous sequence
concurrency. Native initialized a setup model and then the actual model serially,
not two concurrent models or two extraction passes.

Actual full-run calls: NRL220, including116 full128 calls and104 partial1–120
calls; native64 calls of65–394 pages. NRL GPU activity was zero for20.69% of its
warm window; queued Parse inputs existed during approximately97.35% of that
zero-GPU time. The Parse frontend was CPU-active. Investigate work inside that
actor and synchronous call boundaries, not assumed upstream renderer starvation.
Coarse queue/CPU correlation is not function-level profiling or kernel timing.

Lance writing took24.12 s of the119.37-minute full NRL run. Collection,
finalization, process startup, validation, export, and report construction also
cost time. Do not optimize Lance simply because its boundary is visible.

CPU4 and batch256 were not promoted; batch256 added a malformed page and raised
worker memory. Projection16 improved one screen but did not establish a
repeatable total-wall win. Two fractional actors each reserving0.80 VRAM were
rejected as unsafe. Do not repeat failed branches without a new hypothesis.

## Prepare the RTX machine without assuming equivalence

Inventory GPU names, UUIDs, driver, total/free VRAM, compute capability, active
processes, power limits, temperatures/throttling, and topology. Record host RAM,
CPU/NUMA/affinity, cgroup CPU and memory limits, shared memory, and free disk.
Do not infer the GPU generation, memory size, workstation/server variant, or
NVLink support from the phrase “RTX 6000 Pro.” Do not interrupt other workloads,
change power limits, overclock, flush shared caches, or modify shared services.

Keep cache, run, and temporary data on `/raid`. Source resolution outside `/raid`
is rejected by the recipe. Use physical corpus files, not a symlink to another
filesystem. Keep the Ray temporary prefix short enough for Unix socket names;
the old long scratch prefix caused an ancillary dashboard path-length warning.
Use fresh owned directories and never recursively delete a shared root.

Recreate separate pinned NRL and Curator environments using the private
baseline's package inventory and the repository installation instructions.
Do not copy virtual environments between hosts or merge their dependency stacks.
The Curator environment needs its native inference stack for the control, plus
Lance, Arrow, Pillow, and OpenCV for consumption and pixel validation.

Historical major versions were:

| Package | NRL environment | Curator environment |
| --- | --- | --- |
| Python | 3.12.11 | 3.13.7 |
| vLLM | 0.25.1 | 0.22.0, CUDA 12.9 build |
| Transformers | 5.14.1 | 4.57.6 |
| Ray | 2.56.1 | 2.57.0 |
| PyTorch | 2.11.0+cu130 | 2.11.0+cu129 |
| PyArrow | 25.0.0 | 23.0.1 |
| LanceDB | 0.34.0 | 0.34.0 |
| pypdfium2 | 4.30.0 | 5.7.1 |

Verify the full inventory from the supplied baseline, not just this table. Test
wheel/driver/kernel compatibility with the actual GPU before a large run. If a
compatibility change is necessary, record it and freeze a new baseline; do not
silently label it an exact reproduction. Do not change models or prompts to
make an incompatible runtime appear to work.

Use model and tokenizer snapshot
`nvidia/NVIDIA-Nemotron-Parse-v1.2@2bd0189bffd6cdded6280d9f22a4077b25a504e3`.
Verify all snapshot hashes from the private baseline. Hugging Face snapshot
symlinks need their backing blobs, or a complete dereferenced copy; copying
broken symlinks is not staging the model. No credentials belong in the bundle.

Preserve the prompt
`</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>`,
9,000 output tokens, temperature0, top-k1, repetition penalty1.1. Record resolved
tokenizer/processor, batching and thread settings. H200 EngineCore OMP threads
were1 NRL/4 native. NRL renders full200-DPI PNG; native renders300-DPI then fits
1664×2048. Preserve/report those configured-product differences rather than
calling this an isolated library-speed comparison.

## Preserve the frozen cohorts while relocating files

The original corpus contains1,278 unique PDFs and19,604 pages, totaling
2,065,935,553 source bytes. Obtain it separately through the authorized internal
transfer path. Hash every file and match the original input inventory. Preserve
reviewed blank declarations as historical agent-reviewed declarations, not new
human approvals. Keep synthetic/corrupt/encrypted fixtures outside that denominator.

Reuse `cohort/selection.json`:63 tuning documents /460 pages and20 held-out
documents /124 pages, selected from83 readable unique originals of at most50
pages without consulting extraction success. Do not redraw the split or tune
against held-out failures. Keep the solar document `0000098.pdf` supplemental;
its photograph on page4 reached Parse, but both default routes returned empty.
The other validated pages remain published with explicit NRL partial status.

Historical manifests and reports contain absolute paths. Do not edit them in
place or expect their seals to remain valid after path substitution. Create new
relocated manifests in a fresh run directory, matching each original SHA-256,
alias, order, URL, expected page count, and blank declaration. Re-run the existing
`prepare-benchmark` command for those manifests and compare selected identities
and counts to the frozen selection. Path-dependent report seals will change;
content identities must not. Historical source-evidence reports can also refer
to omitted original exports/captures; obtain those separately before claiming
that a fresh physical-evidence verification or judge execution is possible.

## Execute in evidence gates

1. Run the documented environment-specific suites, including the actual graph
   checks, native Lance/Parquet roundtrip, exact18-field schema/nullability,
   partial/blank accounting, damaged images, and failure publication boundaries.
   Classify every skip. Run NRL-only tests in the NRL environment; do not accept
   all graph tests skipping in Curator's environment as coverage.
2. Prove one small real GPU vertical slice, with duplicate, blank, and negative
   inputs. Confirm one Parse actor/model, no extra models, and native export.
3. Establish one-GPU NRL/native matched baselines on the same idle physical GPU.
   Start from the measured candidate settings if the smoke test and capacity
   checks pass. Lower a benchmark-only reservation if necessary and record it;
   do not assume H200 allocation is model-weight size or combine the two GPUs'
   VRAM as one usable pool.
4. Freeze both code revisions, source hashes, environment imports/packages,
   GPU identities, model/tokenizer, manifests, rendering/decoding, resource
   controls, and new observer configuration. Verify before and after each run.
5. Run three alternating pairs: NRL/native, native/NRL, NRL/native. Fresh
   processes/directories, warm declared disk/model caches, capture disabled,
   executor stats enabled, no concurrent tests or native/NRL comparisons.
6. If tuning is justified by measured bottlenecks, change one control at a time.
   Use a small equivalence-checked profile before attributing CPU functions.
   Stop ineffective/unsafe branches; no monkey-patched concurrent vLLM calls.
7. Confirm the selected configuration on the fixed held-out cohort without
   using its results for tuning. Then run the deterministic stress cohort,
   including the1,080-page PDF, before considering another full corpus run.
8. Evaluate two-GPU scale-out separately as described below. Do not claim that
   a one-GPU run on a two-GPU host used both cards.
9. Run the full original corpus only after measured capacity headroom passes.
   Keep native's cap explicit; do not present unequal-work full wall ratios as
   speedups. Missing/partial outputs and failed attempts remain in accounting.

Use the existing CLI with verified variables, not the old hardcoded launchers.
For example, after creating and validating the new tuning manifest:

```bash
OBSERVER="$NRL_REPO/.e2e/nrl-curator-performance-20260921/observe_performance.py"

"$NRL_PY" "$RECIPE_DIR/nrl_compare.py" prepare-benchmark \
  --manifest "$RTX_TUNING_MANIFEST" --corpus-root "$CORPUS_ROOT" \
  --output-dir "$RTX_TUNING_COHORT"

"$NRL_PY" "$OBSERVER" --freeze-baseline "$RTX_BASELINE" \
  --nrl-python "$NRL_PY" --curator-python "$CURATOR_PY" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --input-manifest "$RTX_TUNING_COHORT/manifest.jsonl" \
  --input-manifest "$RTX_HELDOUT_COHORT/manifest.jsonl"

"$NRL_PY" "$OBSERVER" --baseline "$RTX_BASELINE" --verify-only

"$NRL_PY" "$RECIPE_DIR/nrl_compare.py" benchmark \
  --cohort "$RTX_TUNING_COHORT/cohort.json" --output-dir "$RTX_FRESH_RUN" \
  --nrl-python "$NRL_PY" --curator-python "$CURATOR_PY" \
  --nrl-repo "$NRL_REPO" --model-snapshot "$MODEL_SNAPSHOT" \
  --gpu "$GPU_INDEX" --observer "$OBSERVER" --baseline "$RTX_BASELINE" \
  --diagnostic --repetitions 3 \
  --nrl-parse-cpus 1 --nrl-parse-batch-size 128 \
  --nrl-projection-block-rows 16 --native-pdfs-per-task 20 \
  --native-gpu-memory-utilization "$NATIVE_GPU_FRACTION" \
  --projected-temporary-bytes "$PROJECTED_TEMPORARY_BYTES"
```

Prepare the held-out cohort before freezing; include every planned manifest in
the new baseline. Verify index-to-UUID mapping. The observer identifies a physical
GPU by UUID; this pinned native runtime needs the comparison's numeric GPU
selector. Check each child process's resolved visible device. Set all caches,
temporary paths, executable paths and `VIRTUAL_ENV` consistently with the
runbook before fingerprinting. Forecast disk from measured pilot/stress evidence,
not a guessed zero or the H200 full-run allowance copied without explanation.

The user authorized diagnostic performance runs before human quality approval.
Use `--diagnostic` transparently; do not fabricate a signed review. Exit1 can
mean complete diagnostic collection with invalid output findings; inspect the
sealed report. It is not permission to repair, overwrite, or relabel that attempt.

## Two-GPU scale-out is a distinct configuration

The current graph has one Parse actor; the benchmark CLI has one GPU selector.
Setting `CUDA_VISIBLE_DEVICES=0,1` does not prove two-GPU extraction, and one
GPU/model must not be called tensor-parallel across two devices.

The lean candidate is **two isolated executions of the existing route**, each
with one GPU/model and a deterministic, disjoint document shard. Apply the same
arrangement to native Curator. Deduplicate before sharding; all pages and aliases
of one document stay together. Freeze the partition rule before extraction.
Do not split pages of a document or run both full corpora and count duplicates
as additional throughput. This is process scale-out, not a new graph or framework.

Before launching, verify that existing launch/executor controls can isolate both
Ray instances, GPUs, ports, CPU allocations, object stores, temporary directories,
outputs and observer prefixes. Two processes must not each assume the whole
host CPU/RAM budget. Affinity alone is not proof of Ray's declared resources.
If a necessary control is unavailable, report the exact gap and propose the
smallest launcher change before changing production code. Do not invent a CLI
flag or assume graph fan-out/model concurrency is supported.

Run both NRL shards together, then both native shards together, alternating
route order across repetitions. Do not co-run competing routes on separate
cards and call the result an isolated head-to-head comparison: CPU/RAM/storage
contention would confound it. Each shard retains its normal validated handoff
and export; aggregate only after both finish and their disjoint union reconciles.

Measure wall time from the first launch through the last validated shard export,
not the sum or mean of shard wall times. Report throughput, GPU count/model
count, per-GPU utilization/VRAM, summed GPU-seconds, host contention, acceptance,
and scale efficiency relative to the same machine's one-GPU baseline. Keep a
failed shard and its cost visible. No combined completion claim is allowed
while a required shard is missing or invalid.

## Metrics, safety, and completion standard

Sample every second: GPU utilization/memory and physical identity, driver and
worker RSS/CPU, effective host headroom, queue/backpressure where available,
Ray spill, disk footprint/free space, and owned-process cleanup. Add temperature,
power and throttling evidence from available GPU tooling without changing power
settings. Explain missing counters rather than substituting zeros.

Retain at least25% effective host-memory headroom,10% GPU-memory headroom, and
free disk greater than twice the conservative projected-or-actual temporary
footprint. Budget two-GPU host use jointly. Stop safely on breaches, OOM,
unexplained source/configuration drift or stranded workers. Terminate only
owned PID/creation-time identities; do not use broad `ray stop` or `pkill` on a
shared machine. Resume failed experiments in fresh directories after cleanup.

Report each repetition, median, range, and paired differences. Three repetitions
do not establish statistical significance. Separate startup-inclusive wall from
warmed activity; keep partial stage timers clearly labeled and never sum
overlapping task times. Count attempted and delivered pages separately, with
complete/partial/failed documents, explicit blanks, cap omissions and malformed
rows. Compare exact schema, values, identities, page outcomes and image hashes
across NRL, Lance and Parquet. Higher token/element counts are not accuracy.

The existing source review found both NRL gains and losses: a preserved logo,
a missing address field, a meaning-changing bilingual clause splice, and shared
photograph/table-label omissions. Native is not ground truth. Keep semantic
quality evaluation outside timed extraction. The report-only judge is implemented
but has not run against an approved endpoint/model or human calibration set.
Do not transmit document images externally without approval or mark agent
observations as human approval. That gap limits quality claims, not independent
authorized performance work.

Deliver one new RTX experiment ledger/report containing exact commit and
environment identities, commands, cohorts, configurations, metrics, failures,
resource costs, information-preservation findings, and recommended/rejected
changes. Link the historical H200 report, but do not overwrite it. Make clear
which conclusions are hardware-only, configured-product comparisons, or
unresolved. No commit, push, deployment, shared-service change, data deletion,
or public evidence upload on the new machine unless separately requested.

The task is complete when the permitted configuration matrix has verified
results or a concrete documented blocker, every input and artifact reconciles,
all owned workers are gone, and the report distinguishes structural success,
semantic approval, performance, and capacity. Do not restart completed work
because a terminal handle was lost; verify the exact retained process identity.

This handoff uses explicit objective, context, constraints, and verification
criteria following [OpenAI's Codex best-practice guidance](https://learn.chatgpt.com/guides/best-practices).
