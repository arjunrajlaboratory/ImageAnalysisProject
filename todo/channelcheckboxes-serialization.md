# TODO-002: Malformed `channelCheckboxes` values from externally-authored tool configs

**Status:** Open (worker guardrail done; root-cause cleanup outstanding)
**Priority:** Medium
**Related PR:** _(this branch)_

## Summary

The `channelCheckboxes` interface type is documented (in `CLAUDE.md` and the
`nimbus-interface` skill reference) as returning a dict mapping channel-index
strings to booleans, e.g. `{"0": True, "1": False}`. All worker code was written
against that assumption (`[int(k) for k, v in value.items() if v]`).

A production Cellpose-SAM error log showed the values arriving as **lists**
instead:

```
"Channel for Slot 1": [0], "Channel for Slot 2": [], "Channel for Slot 3": []
```

Calling `.items()` on a list raised an opaque
`AttributeError: 'list' object has no attribute 'items'` that crashed the worker
**before** its existing required-field check (`if not slot1...: sendError(...)`)
could run. So the safeguard wasn't missing — it was being skipped.

## Root cause (confirmed)

The list shape is **not** produced by the worker, the front-end UI, or the docs:

- **Worker interface**: `Channel for Slot 1/2/3` declare no `default`, so
  `getDefault("channelCheckboxes", undefined)` returns `{}` — an empty dict,
  never `[0]`.
- **NimbusImage front-end**: `channelCheckboxes` is typed `Record<number, boolean>`
  from the first commit; the `ChannelCheckboxGroup` widget only ever emits dicts;
  the empty value is `{}`, never `[]`.
- **Backend** (`server/helpers/tasks.py`): JSON-encodes `workerInterfaceValues`
  verbatim into `--parameters` — no transformation.
- Nothing in either repo constructs these values as arrays.

An empty `[]` for an unset slot is the tell: no code path generates `[]`, only
`{}`. These values were **persisted on a saved tool whose `workerInterfaceValues`
were written with the wrong shape** — channel slots encoded as arrays of selected
indices (the intuitive-but-wrong representation) instead of `{index: bool}` dicts.
The config is re-sent verbatim on every run, which is why it fails *consistently*
for those specific tools rather than randomly.

Fingerprints (tag `mg_cellposesam_v2`, name `cellpose-sam zero-shot on
LCA5_P21_rep1`) point to programmatic/scripted batch tool creation via the
API/Swagger that bypassed the UI widget, guessing `[0]` for "channel 0 selected."

**Conclusion:** the front-end never emits lists. A list always means
malformed/externally-authored input, never a real UI selection — so there is no
"list-form selection" semantic to preserve.

## What was done (worker guardrail — DONE)

Added a shared helper `annotation_utilities.annotation_tools.get_selected_channels()`
that parses the documented dict form, treats `None` as "nothing selected", and
**rejects any other shape (including lists) with a `ValueError`**. Each affected
worker catches it and calls `sendError(...)` with a clear "interface is out of
date or misconfigured, please re-select the channels" message instead of crashing.

Affected workers: `cellposesam`, `registration`, `deconwolf`,
`histogram_matching`, `gaussian_blur`, `rolling_ball`.

We chose to **reject** rather than normalize-and-run: since these values come
from an external script that may have guessed the wrong channel, failing loudly
is safer than silently running a tool on a channel the user never confirmed.

## Outstanding (the real fix)

1. Find and fix whatever created the `mg_cellposesam_v2` batch with array-shaped
   channel values, so it writes `{index: bool}` dicts.
2. Optionally run a one-time normalization over saved tool configs to repair the
   existing broken tools (otherwise they keep erroring until recreated).

Until then, affected tools get a clear error instead of a crash, but won't run.
