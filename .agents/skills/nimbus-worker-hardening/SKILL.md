---
name: "nimbus-worker-hardening"
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

This uses the same generalize-and-sweep approach as a comprehensive code review,
applied to a fixed catalog of Nimbus worker failure modes. If you're responding
to PR review comments specifically, use the available PR review workflow; use
this skill for deploy crashes and proactive hardening.

Follow the repo's test-first convention (see `AGENTS.md`): write the failing
test that reproduces the crash, then fix, then confirm green. Local package
tests (`annotation_utilities`, `worker_client`) run natively in a Python 3.11
venv — no Docker needed.

## Before assuming a shared helper exists

Some fixes below consolidate into shared helpers. Their merge status changes
over time, so **check the current tree before relying on one** rather than
assuming:

```bash
grep -rn "def geometry_to_polygon_coords\|def get_selected_channels\|def find_out_of_range\|def validate_coordinates" \
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
production client was observed sending a list (`[0]`), crashing before any
validation.

**Fix:** don't call `.items()` on the raw value. **Reject** unexpected shapes
loudly rather than guessing a channel — running a tool on the wrong channel is
worse than failing. Use the shared `annotation_tools.get_selected_channels()`
helper if present; otherwise inline:
```python
def selected_channels(value, field_name='channel selection'):
    if value is None:
        return []                                   # nothing selected
    if isinstance(value, dict):
        return sorted(int(k) for k, v in value.items() if v)
    raise ValueError(f"'{field_name}' has an unexpected format "
                     f"({type(value).__name__}: {value!r}); expected channel checkboxes.")
```
Catch the `ValueError` at the call site and `sendError` with an "interface out
of date or misconfigured" message.

**Sweep:**
```bash
grep -rln "channelCheckboxes\|\.items()" workers/   # then inspect for raw .items() on interface values
```
Known consumers: cellposesam, registration, deconwolf, histogram_matching,
gaussian_blur, rolling_ball. NOTE: the list form is undocumented — the
NimbusImage front-end is the root-cause source of truth for what
`channelCheckboxes` serializes.

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

**Sweep:** the `WorkerClient.process()` path (≈6 workers) is the priority;
~26 workers parse batch ranges directly and can adopt the shared validator as it
rolls out. Find batch-range consumers:
```bash
grep -rln "batch_argument_parser\|Batch XY\|coordinatesToFrameIndex" workers/
```

### 5. Build-time transitive dependency breakage

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

### 6. `tags` interface field treated as a dict

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
`nimbus-interface` skill and `AGENTS.md` for the full interface type→return
table.

## After fixing

- Run the relevant package tests (`annotation_utilities`, `worker_client`) and
  the affected workers' Docker tests; show the output before claiming done.
- If you touched a worker's interface or labels (not just internal logic),
  update its `WORKERNAME.md`; `REGISTRY.md` only needs updating for
  add/remove/rename.
- When you fix a *new* failure mode not in this catalog, add it here — a short
  entry with symptom, wrong→right, and a sweep grep is what makes the next
  occurrence a five-minute fix instead of a rediscovery.
