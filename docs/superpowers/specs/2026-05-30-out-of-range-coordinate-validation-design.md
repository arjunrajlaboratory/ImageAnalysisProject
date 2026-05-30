# Out-of-range coordinate validation in the WorkerClient path

**Date:** 2026-05-30
**Status:** Design — pending implementation

## Problem

When a user enters a `Batch XY`, `Batch Z`, or `Batch Time` range that includes
coordinates not present in the dataset, annotation workers crash with a raw
`KeyError` and a full stack trace instead of a useful, actionable message.

Reproduction (reported): running Cellpose-SAM with `Batch XY: "80-90"` on a
dataset that has only one XY position. The parser converts the 1-indexed
`80-90` to 0-indexed `79-89`; the worker then crashes at:

```
File ".../annotation_client/tiles.py", line 134, in coordinatesToFrameIndex
    return self.map[channel][T][Z][XY]
KeyError: 79
```

### Crash path

`WorkerClient.process()` parses `Batch XY/Z/Time` into ranges, then loops:
`process()` → `get_image_stack()` → `get_image()` → `coordinatesToFrameIndex()`,
which does `self.map[channel][T][Z][XY]` and raises a bare `KeyError` the moment
a coordinate is missing. No `sendError`, so the user sees only a stack trace.

### Scope of affected code (from a full sweep)

- **6 workers go through the shared `WorkerClient.process()` path** — cellposesam,
  cellpose, condensatenet, laplacian_of_gaussian, random_squares,
  sample_interface. A single central fix covers all of them.
- **~26 other workers** iterate batch ranges or call `coordinatesToFrameIndex()`
  directly (SAM2 family, cellori, registration, crop, histogram_matching,
  cellpose_train, etc.). These are **out of scope** for this change; the shared
  validator added here is written so they can adopt it later.

Existing in-repo validation patterns this design follows: `blob_intensity_worker`,
`crop`, and `registration` build `range_x = range(0, IndexRange['IndexXY'])`,
filter/validate coordinates, and report in 1-indexed terms.

## Decisions

- **Behavior:** *Strict* — if **any** requested coordinate is out of range,
  `sendError` with the valid range and stop before doing work. (Not the
  filter-and-warn-subset approach used by some property workers.)
- **Scope:** Central `WorkerClient.process()` path first. Other workers deferred.
- **No cross-repo changes:** The shared `annotation_client/tiles.py`
  (`coordinatesToFrameIndex`) in the NimbusImage repo keeps its bare `KeyError`
  as a last-resort backstop. A separate GitHub issue is filed there to harden it
  later.
- **Fail before expensive work:** Cellpose-SAM loads its GPU model in
  `compute()` *before* `worker.process()` runs. To fail instantly (before the
  ~5-10s model load), `cellposesam.compute()` calls the validator early.

## Design

### 1. Pure validator — `annotation_utilities`

New module `annotation_utilities/annotation_utilities/coordinate_validation.py`.
Pure functions, no `sendError` side effects (keeps them unit-testable and
reusable by the deferred workers):

```python
def find_out_of_range(index_range, xys=None, zs=None, times=None):
    """Return {dim: (sorted_bad_0indexed_values, dim_size)} for any requested
    coordinate < 0 or >= the dimension's size. Empty dict means all valid.

    `dim` is one of 'XY', 'Z', 'Time'. A missing IndexRange, or a missing
    IndexXY/IndexZ/IndexT sub-key, defaults that dimension's size to 1 —
    matching the existing fallback convention (blob_intensity, get_image_stack).
    Dimensions passed as None are skipped."""

def format_out_of_range_message(invalid):
    """Build a single (message, info) pair for sendError. XY/Z/Time values are
    reported 1-INDEXED to match what the user typed in the 'Batch XY/Z/Time' UI
    fields (the batch parser converts 1->0 on input). Contiguous runs collapse
    to 'a-b'; otherwise a comma-separated list."""
```

Example output for the reported bug:

> **message:** `"Batch XY is out of range"`
> **info:** `"You requested XY position(s) 81-91, but this dataset has only 1 XY position (valid: 1). Please adjust the 'Batch XY' field."`

`dim` → UI label mapping for messages: `XY` → "Batch XY", `Z` → "Batch Z",
`Time` → "Batch Time".

### 2. `WorkerClient` changes — `worker_client/worker_client/worker_client.py`

**a. Materialize batch ranges to lists in `__init__`.** `process_range_list`
returns a generator; storing it as a generator means it can only be consumed
once. Since both `validate_coordinates()` and `process()` now read these, store
them as lists:

```python
self.batch_xy = list(batch_xy)
self.batch_z = list(batch_z)
self.batch_time = list(batch_time)
```

(`batch_xy` is either a generator from the parser or a `[tile['XY']]` fallback;
`list(...)` is correct for both. No current caller depends on them being lazy.)

**b. New public method `validate_coordinates()`** (single source of truth):

```python
def validate_coordinates(self, stack_xys=None, stack_zs=None, stack_times=None):
    """Strictly validate the batch coordinates that will be iterated against
    the dataset's IndexRange. If any requested XY/Z/Time coordinate is out of
    range, sendError with an actionable message and raise ValueError.

    Safe to call early (e.g. before loading an expensive model); also called
    internally by process(). Idempotent and side-effect-free unless a
    coordinate is out of range."""
    xys   = list(self.batch_xy)   if stack_xys   is None else [self.tile['XY']]
    zs    = list(self.batch_z)    if stack_zs    is None else [self.tile['Z']]
    times = list(self.batch_time) if stack_times is None else [self.tile['Time']]

    index_range = self.datasetClient.tiles.get('IndexRange', {})
    invalid = coordinate_validation.find_out_of_range(
        index_range, xys=xys, zs=zs, times=times)
    if invalid:
        message, info = coordinate_validation.format_out_of_range_message(invalid)
        sendError(message, info=info)
        raise ValueError(message)
```

The `stack_*` arguments mirror `process()`: for a dimension the worker *stacks*
(not batches), the effective coordinate is the current tile (`self.tile[...]`),
which is valid by construction and passes harmlessly; for a *batched* dimension
it is the user's range — exactly what we want to check.

**c. Call it at the top of `process()`**, before the loop:

```python
self.validate_coordinates(stack_xys, stack_zs, stack_times)
```

**d. Import `sendError`** by extending the existing
`from annotation_client.utils import sendProgress` line, and import the new
`coordinate_validation` module from `annotation_utilities`.

Channels are intentionally **not** validated here: `channelCheckboxes` can only
select channels that already exist, so they cannot produce this bug, and mixing
0-indexed channel messaging with 1-indexed XY/Z/Time messaging would confuse the
error. The validator is structured so channel validation can be added later.

### 3. Early check in Cellpose-SAM — `workers/annotations/cellposesam/entrypoint.py`

Add one line near the top of `compute()`, right after the `WorkerClient` is
constructed and **before** the model is downloaded/loaded:

```python
worker = WorkerClient(datasetId, apiUrl, token, params)
worker.validate_coordinates()   # fail fast before loading the GPU model
```

Cellpose-SAM batches XY/Z/Time and stacks channels, so the default
(no-`stack_*`) validation of `batch_xy/z/time` is exactly correct. `process()`
re-validates internally; the duplicate call is cheap and idempotent.

## Testing

Follow the existing `*_index_range.py` pattern (a `DummyDatasetClient` with a
`.tiles` dict; `annotation_client` submodules mocked).

- **`annotation_utilities/tests/test_coordinate_validation.py`** — pure-function
  tests:
  - all in-range → empty dict / no message
  - out-of-range detected per dimension, with correct size
  - missing `IndexRange` ⇒ dimension size defaults to 1 (only coord 0 valid)
  - missing single sub-key (e.g. `IndexZ`) ⇒ that dim defaults to 1
  - message is 1-indexed and names the right `Batch *` field
  - multiple offending dimensions reported together
  - contiguous-run collapsing (`81-91`) vs. comma list

- **`worker_client/tests/test_worker_client_index_range.py`** (extend) — add a
  module-level recorder for a monkeypatched `utils.sendError`, then:
  - `IndexRange={'IndexXY':1}`, `workerInterface={'Batch XY':'80-90'}` →
    `process()` calls `sendError` once and raises `ValueError`; the processing
    loop never runs (no `getRegion` calls)
  - valid `Batch XY` within range → `process()` proceeds normally and produces
    annotations (no `sendError`)
  - `validate_coordinates()` called directly raises on out-of-range and returns
    `None` when valid

Run locally:

```bash
cd annotation_utilities && python -m pytest tests/ -q
cd ../worker_client && python -m pytest tests/ -q
```

## Out of scope / follow-ups

- The ~26 decentralized workers that parse batch ranges or call
  `coordinatesToFrameIndex()` directly. They can adopt
  `coordinate_validation.find_out_of_range` / `format_out_of_range_message`
  in a later pass.
- Hardening `coordinatesToFrameIndex` in the NimbusImage repo to raise a
  descriptive exception instead of a bare `KeyError` — tracked as a separate
  GitHub issue on `arjunrajlaboratory/NimbusImage`.
