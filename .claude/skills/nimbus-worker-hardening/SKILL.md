---
name: nimbus-worker-hardening
description: "Harden existing NimbusImage workers by diagnosing production crashes, generalizing the failure, sweeping sibling workers, and adding shared regression coverage. Use for stack traces or requests such as harden, make robust, fix and apply to all workers, or audit siblings; especially IndexRange, channelCheckboxes, geometry, batch-coordinate, tags, and dependency failures. Use nimbus-worker-scaffold for new workers and nimbus-interface for API reference."
---

# Hardening NimbusImage Workers

Worker bugs in this repo cluster into a small set of failure modes that recur
across dozens of near-identical workers. When one worker hits such a bug in a
deploy, the same latent bug is almost always sitting in its siblings — the
reported crash is just the first one a user happened to trigger. Patching only
the worker that failed guarantees the next report is the same bug in a different
file.

**The method, every time:**

1. **Diagnose** the specific failure using the catalog below (or systematic
   debugging if it's not in the catalog). Reproduce it with a test first.
2. **Generalize** it to the underlying pattern — "unguarded `IndexRange`
   subscript", not "line 167 of registration".
3. **Sweep** every worker for that pattern (grep commands are in the catalog)
   and fix them all in one pass.
4. **Consolidate** the fix into a shared, tested helper in `annotation_utilities`
   / `worker_client` where one exists or is warranted, so future workers inherit
   the fix instead of re-implementing the bug. Add a unit test that covers the
   failure input.

This is the same generalize-and-sweep instinct as the `mycelium:codex-review`
skill, applied to a fixed catalog of Nimbus worker failure modes. If you're
responding to Codex/PR review comments specifically, use that skill; use this
one for deploy crashes and proactive hardening.

Follow the repo's test-first convention (see `CLAUDE.md`): write the failing
test that reproduces the crash, then fix, then confirm green. Local package
tests (`annotation_utilities`, `worker_client`) run natively in a Python 3.11
venv — no Docker needed.

## Before assuming a shared helper exists

Some fixes below consolidate into shared helpers. Their merge status changes
over time, so **check the current tree before relying on one** rather than
assuming:

```bash
grep -rn "def geometry_to_polygon_coords\|def get_selected_channels\|def get_batch_ranges\|def find_out_of_range\|def validate_coordinates" \
  annotation_utilities/ worker_client/
```

If the helper is present, wire the worker to it. If it isn't (it may still be on
an unmerged branch), apply the inline guard shown in the catalog and note that
the shared helper should be adopted when it lands.

## The failure-mode catalog

### 1. Missing `IndexRange` fallback (single-frame datasets)

**Symptom:** `KeyError: 'IndexRange'` or `KeyError: 'IndexC'` on a real dataset.
Single-frame datasets (one channel, no Z/T/XY stacks) **omit the `IndexRange`
key entirely**, so any direct subscript crashes.

**Wrong → right:**
```python
num_channels = tileClient.tiles['IndexRange']['IndexC']            # crashes
num_channels = tileClient.tiles.get('IndexRange', {}).get('IndexC', 1)  # defaults to 1
```
A missing `IndexRange`, or a missing per-dimension key, means one position along
that dimension (only coordinate 0 valid).

**Sweep:**
```bash
grep -rn "IndexRange'\]\['Index" workers/          # unguarded subscripts
grep -rn "tiles\['IndexRange'\]" workers/           # any direct access
```
The shared helper `annotation_tools.get_images_for_all_channels` already handles
this; workers that read channel counts by hand should mirror the `.get(...)`
pattern.

**The same omission happens per-frame**, and this form is easy to miss because it
does not mention `IndexRange` at all. Every entry of `tileClient.tiles['frames']`
omits the index key for any dimension of size one, so a single-channel dataset's
frames are `{'Channel': 'Default', 'Frame': 3, 'Index': 3, 'IndexT': 3}` — no
`IndexC` whatsoever. This crashed time lapse registration in July 2026 at
`if frame['IndexC'] in channels`, after ~20 minutes of successful work.

```python
if frame['IndexC'] in channels:                                    # crashes
if annotation_tools.get_frame_index(frame, 'IndexC') in channels:   # coordinate 0
```
`get_frame_index(frame, dimension, default=0)` (on master in
`annotation_utilities.annotation_tools`) defaults an absent dimension to 0 and
raises `ValueError` on an unknown dimension name so typos don't silently read as
channel 0. Its companion `frame_to_large_image_params(frame)` replaces the
`{f'{k.lower()[5:]}': v for k, v in frame.items() if k.startswith('Index') and
len(k) > 5}` comprehension that used to be copy-pasted into seven workers.

```bash
grep -rn "frame\['Index" workers/                   # per-frame subscripts
grep -rn "len(k) > 5" workers/                      # the copy-pasted comprehension
```
Note that a frame-index bug is invisible on multi-channel test data; when adding
a frame-loop test, use frames with **no** `IndexC` key (not `IndexC: 0`).

### 2. Malformed `channelCheckboxes` (arrives as a non-dict)

**Symptom:** `AttributeError: 'list' object has no attribute 'items'`.
`channelCheckboxes` is documented to return `{"0": True, "1": False}`, but a
saved tool config held a list (`[0]`), crashing before any validation.

**Fix:** never call `.items()` on the raw value. Route it through the shared
`annotation_tools.get_selected_channels(value, field_name)` helper, which parses
the dict shape into a sorted list of `int` channel indices, returns `[]` for an
unset field, and raises `ValueError` on any other shape rather than guessing a
channel — running a tool on the wrong channel is worse than failing. The list
shape is rejected, not normalized: the checkbox UI never emitted it and the AI
panel normalizes arrays before saving, so a list value means the config came from
somewhere outside the UI and must be re-saved rather than guessed at.
```python
try:
    channels = annotation_tools.get_selected_channels(
        workerInterface.get('Channels to correct'), 'Channels to correct')
except ValueError as exc:
    sendError("Could not read the channel selection.", info=str(exc))
    return
if not channels:
    sendError("No channels selected")
    return
```
Catch the `ValueError` at the call site and `sendError` with an "interface out
of date or misconfigured" message.

**Sweep:**
```bash
grep -rln "channelCheckboxes\|\.items()" workers/   # then inspect for raw .items() on interface values
```
All known consumers now use the helper: cellposesam, cellposesam_train,
registration, deconwolf, histogram_matching, gaussian_blur, rolling_ball,
sample_interface. Regression coverage lives in
`annotation_utilities/tests/test_selected_channels.py` (runs in CI). Configs that
already hold list-shaped values now report a clear error until re-saved, and the
submitter that wrote them is still unidentified — tracked in
`todo/channelcheckboxes-serialization.md`.

**A well-formed selection can still name channels the dataset does not have.**
`get_selected_channels` validates shape, not range: a saved config selecting
channel 1 parses cleanly and then matches no frame when it is run against a
single-channel dataset. There is no crash — the worker processes nothing, uploads
a byte-identical copy of its input, and reports success, which is worse than an
error because nobody looks for it. Split the selection against the dataset's real
channel count and report:
```python
num_channels = tileClient.tiles.get('IndexRange', {}).get('IndexC', 1)
channels, missing_channels = annotation_tools.split_channel_selection(
    channels, num_channels)
if missing_channels:
    detail = (f"Selected channel indices {missing_channels} do not exist in "
              f"this dataset, which has {num_channels} channel(s) "
              f"(indices 0-{num_channels - 1}).")
    if not channels:
        sendError("None of the selected channels exist in this dataset.", info=detail)
        return
    sendWarning("Ignoring channels this dataset does not have.", info=detail)
```
Note the asymmetry: an **empty** selection is a different case — the user
deselected everything, and gaussian_blur/rolling_ball deliberately still write an
unprocessed copy — so `split_channel_selection` reports nothing missing for it.
This applies to every dimension a worker filters on, not just channels: crop
checks its XY/Z/Time ranges and registration checks `Apply to XY coordinates` the
same way. Anywhere a worker intersects a user selection with what the dataset
has, an empty intersection needs reporting.

**Sweep:**
```bash
grep -rn "get_selected_channels" workers/   # every caller is a candidate
```
Done for the five image-processing workers that filter frames by channel:
gaussian_blur, rolling_ball, histogram_matching, registration, deconwolf. Still
unaudited: cellposesam and cellposesam_train (three independent per-slot
selections feeding a channel merge, so an out-of-range index fails differently)
and sample_interface (a demo worker that only prints its selection).

Covered by `annotation_utilities/tests/test_split_channel_selection.py` (runs in
CI) plus per-worker tests in all five workers above. A test for this needs a
fixture whose `IndexRange` channel count is **smaller** than the selected index;
on the usual 2-channel fixture every selection is in range and the bug is
invisible.

### 3. Degenerate / MultiPolygon geometry

**Symptom:** either `AttributeError` on `.exterior` (a `MultiPolygon` has no
`.exterior`), or an HTTP 400 `data.coordinates must contain at least 1 items`
that fails the **entire batch upload** (one bad polygon kills all N). Caused by
negative-buffer polygon padding (`buffer(<0)`) or `simplify()`: small objects
shrink to an empty geometry, or pinched objects split into a `MultiPolygon`.

**Fix:** never take `.exterior.coords` off a raw geometry. Route it through the
shared `annotation_utilities.annotation_tools.geometry_to_polygon_coords()`
(present on master), which drops empty/zero-area pieces and turns each
`MultiPolygon` piece into its own coordinate list:
```python
from annotation_utilities.annotation_tools import geometry_to_polygon_coords
for coords in geometry_to_polygon_coords(geometry):   # returns [] if degenerate
    # build one polygon annotation from coords
```

**Sweep:**
```bash
grep -rn "\.exterior\.coords" workers/ annotation_utilities/   # candidates to route through the helper
grep -rn "\.buffer(\|\.simplify(" workers/                     # sources of degenerate geometry
```

### 4. Out-of-range batch coordinates

**Symptom:** a bare `KeyError` from `coordinatesToFrameIndex` when a user's
Batch XY/Z/Time range names a coordinate the dataset doesn't have (e.g. Batch XY
`80-90` on a single-XY dataset). The user sees only a stack trace.

**Fix:** validate requested coordinates against the dataset's `IndexRange`
**before** the processing loop (and before any GPU model loads), and `sendError`
with a 1-indexed message matching the UI's Batch fields. Prefer
`WorkerClient.validate_coordinates()` / the `annotation_utilities.coordinate_validation`
helper if present; if not yet merged, guard inline against the dimension size
(`index_range.get(key, 1)`) and report which coordinates are out of range.

**Sweep:** the `WorkerClient.process()` path is the priority -- most annotation
workers take it and inherit the validator for free. The remaining handful parse
batch ranges in their own loop; they should adopt
`batch_argument_parser.get_batch_ranges()` (see #5) rather than calling
`process_range_list` per field. Find batch-range consumers:
```bash
grep -rln "batch_argument_parser\|Batch XY\|coordinatesToFrameIndex" workers/
```

### 5. Literal `all` parsed without dataset context

**Symptom:** entering `all` in a Batch XY/Z/Time field raises a numeric parsing
error or `ValueError: 'all' requires all_values`, while numeric ranges still
work. The range parser cannot infer a dataset dimension by itself.

**Fix:** standard batch fields should go through the shared dataset-aware helper:
```python
index_range = datasetClient.tiles.get('IndexRange', {})
batch_xy, batch_z, batch_time = batch_argument_parser.get_batch_ranges(
    params['tile'], params['workerInterface'], index_range)
```
This preserves 1-indexed numeric UI inputs, expands case-insensitive `all` to
zero-indexed dataset coordinates, keeps the current tile for an empty field,
and treats a missing dimension as coordinate `0`. It also raises a `ValueError`
naming the offending field on malformed input, so callers can surface it with
`sendError` instead of leaking a parser traceback. Annotation workers using
`WorkerClient` inherit all of this automatically.

For a non-standard range field, pass the final available coordinates explicitly:
```python
values = batch_argument_parser.process_range_list(
    raw_value,
    convert_one_to_zero_index=True,
    all_values=range(index_range.get('IndexZ', 1)),
)
```
`all_values` must already use the coordinate system required by the caller;
conversion flags apply to numeric user input, not to `all_values`.

Do not hand-write the three Batch fields either -- build them with
`batch_argument_parser.batch_interface_fields(display_order=N, verb='...')`.
Copies drift: several workers ended up with no placeholder at all, and the
`or all` text was missed everywhere when `all` support first landed.

**A placeholder is a promise.** `BATCH_RANGE_PLACEHOLDER` ("ex. 1-3, 5-8, or
all") may only appear on a field whose `process_range_list` call passes
`all_values`. A non-standard range field (`Z planes`, `Apply to XY
coordinates`) has to wire it explicitly before advertising `all`:
```python
values = batch_argument_parser.process_range_list(
    raw_value, convert_one_to_zero_index=True,
    all_values=range(index_range.get('IndexZ', 1)))
```

**Sweep:** find any field that advertises `all` without parser support:
```bash
grep -rln "Batch XY\|Batch Z\|Batch Time" workers/
grep -rn "BATCH_RANGE_PLACEHOLDER\|or all" workers/ --include=entrypoint.py
```
then confirm each matching worker's `process_range_list` calls pass
`all_values` (an AST scan is more reliable than grep for the multi-line calls).
Inspect local parser copies separately: five unbuilt legacy workers
(`cellori_segmentation`, `random_point*`, `test_multiple_annotation*`) use
their own `utils.process_range_list` with one-indexed semantics and do **not**
support `all` -- they must not use the shared fields until they are ported.

### 6. Build-time transitive dependency breakage

**Symptom:** the image builds one day and a fresh deploy build later crashes at
**import time** with e.g. `ModuleNotFoundError: No module named 'pkg_resources'`.
A transitive dependency imports a module that a newer resolver-selected version
removed — the concrete case was `stardist` importing `pkg_resources`, which
`setuptools>=82` dropped.

**Fix:** pin the offending dependency in the conda `environment.yml`/`.core.yml`
at the layer where it's installed, with a comment explaining why:
```yaml
  # stardist imports pkg_resources at import time, which setuptools removed in 82.0.0.
  - setuptools<82
```
This class of bug won't appear in a cached local build — only in a clean deploy
build — so when a worker imports a pinned-era library (`stardist`, older ML
packages), check the pins proactively. Related: PRs that slimmed startup and
deferred heavy imports; keep interface-path imports light.

### 7. `tags` interface field treated as a dict

**Symptom:** `AttributeError: 'list' object has no attribute 'get'`. A `tags`
**interface field** returns a plain list of strings, not a dict.

**Fix:**
```python
tags = params['workerInterface'].get('Training Tag', [])   # correct: it's a list
# NOT: params['workerInterface'].get('Training Tag', {}).get('tags', [])
```
Don't confuse this with `params['tags']` used for **property filtering**, which
*is* `{'tags': [...], 'exclusive': bool}`. Validate required tags early and
`sendError` if empty (pattern in cellpose_train, piscis). See the
`nimbus-interface` skill and `CLAUDE.md` for the full interface type→return
table.

### 8. `groupby` on a DataFrame built from an empty list

**Symptom:** `KeyError: '<column>'` from `pandas` deep inside `groupby`/column
access, on a dataset where a filter step matched nothing. `pd.DataFrame([])`
has **no columns at all**, so `df.groupby('parentId')` raises `KeyError:
'parentId'` instead of returning an empty result. The June 2026 production case
was children_count_worker run with `'Child Tags': []` — an empty tag set
intersects with nothing, so `filtered_connections` was `[]`.

**Two fixes, both needed:**
1. Validate the required selection early and `sendError` (an empty required
   `tags` field is a misconfiguration — same validation pattern as catalog #6):
   ```python
   if not child_tags:
       sendError("No child tag selected", info="Select at least one tag ...")
       return
   ```
2. Guard the groupby — an empty *result* (valid tags, zero matches) is
   legitimate data, so skip pandas and report counts of 0 with a `sendWarning`:
   ```python
   if filtered_connections:
       df = pd.DataFrame(filtered_connections)
       counts = df.groupby('parentId').size().reset_index(name='count')
   else:
       counts_dict = {}   # every parent gets 0; sendWarning explains why
   ```
Pre-declaring columns also works when a DataFrame must exist either way:
`pd.DataFrame(connections, columns=['parentId', 'childId'])` — this is why
connect_to_nearest/connect_sequential (which initialize
`pd.DataFrame(columns=[...])`) never crashed.

**Sweep:**
```bash
grep -rn "pd\.DataFrame(" workers/ | grep -v "columns="   # then check each for a possibly-empty list feeding a groupby/column access
```
Swept 2026-08: children_count_worker was the only worker with the pattern
(fixed; regression tests in its `tests/test_children_count.py`). Note its old
tests mocked `pandas.DataFrame` entirely, which is how the crash survived —
when adding tests for a pandas path, use real pandas with an **empty** input
list; a mocked groupby chain can't catch this class of bug.

### 9. Null / stale `select` value (Model) from a saved config

**Symptom:** a cryptic crash deep inside a model loader, long after "Loading
model" was reported — e.g. `FileNotFoundError: '/None.pth'` (sam_fewshot,
May 2026 production), `KeyError: None` from a model→config mapping, or a
`download_girder_model(client, None)` failure. A saved tool config can hold
``null`` for a `select` field even though the interface defines a default —
the config stores whatever was serialized when the tool was saved. A config
saved against an older worker image can also name a model/checkpoint that no
longer exists in the current image.

**Fix:** never build a checkpoint path or model name from a raw
`workerInterface` select value. Route it through
`annotation_tools.get_required_select(value, field_name, allowed_values=None)`,
which raises `ValueError` on null/empty/non-string values (and, when
`allowed_values` is given, on options that no longer exist) instead of letting
the job die downstream. Catch it at the call site and `sendError`; where the
checkpoint path is deterministic, also check `os.path.exists(checkpoint_path)`
before loading. `sendError` only prints a message for the frontend -- it does
not fail the job, so re-`raise` after it rather than `return`ing; a job that
returns cleanly is recorded as SUCCESS, and a misconfigured run reported as
successful is worse than a crash. Missing values are rejected, not defaulted: the saved value is
what the user believes the tool runs with, and silently substituting a model
changes the output. Validate **before** the heavy torch/model imports so the
job fails in milliseconds, not after GPU setup.

```python
try:
    model_name = annotation_tools.get_required_select(
        params['workerInterface'].get('Model'), 'Model', allowed_values=MODELS)
except ValueError as exc:
    sendError("Could not read the model selection.", info=str(exc))
    raise
```

**Sweep:**
```bash
grep -rn "workerInterface\[.\?'Model'\]\|workerInterface\.get('Model')" workers/
grep -rn "checkpoint_path = f\"" workers/            # paths built from interface values
```
Done for: sam_fewshot_segmentation, the five sam2_* workers (which also hoist
the shared `MODEL_TO_CFG` mapping and check checkpoint existence), stardist,
cellpose, cellposesam, piscis predict/train, cellpose_train, cellposesam_train.
Workers whose models come from Girder (cellpose family, piscis) validate shape
only — no static `allowed_values` exists for custom models.
`sam_automatic_mask_generator` reads Model but never uses it, so it was left
alone. Regression coverage: `annotation_utilities/tests/test_required_select.py`
(CI) plus compute-level tests in sam_fewshot_segmentation and
sam2_fewshot_segmentation.

## After fixing

- Run the relevant package tests (`annotation_utilities`, `worker_client`) and
  the affected workers' Docker tests; show the output before claiming done.
- If you touched a worker's interface or labels (not just internal logic),
  update its `WORKERNAME.md`; `REGISTRY.md` only needs updating for
  add/remove/rename.
- When you fix a *new* failure mode not in this catalog, add it here — a short
  entry with symptom, wrong→right, and a sweep grep is what makes the next
  occurrence a five-minute fix instead of a rediscovery.
