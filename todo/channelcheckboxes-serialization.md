# TODO-002: Verify `channelCheckboxes` serialization against the front-end

**Status:** Open
**Priority:** Medium
**Related PR:** _(this branch)_

## Summary

The `channelCheckboxes` interface type is documented (in `CLAUDE.md` and the
`nimbus-interface` skill reference) as returning a dict mapping channel-index
strings to booleans, e.g. `{"0": True, "1": False}`. All worker code was written
against that assumption (`[int(k) for k, v in value.items() if v]`).

However, a production Cellpose-SAM error log showed the values arriving as
**lists** instead:

```
"Channel for Slot 1": [0], "Channel for Slot 2": [], "Channel for Slot 3": []
```

Calling `.items()` on a list raised an opaque
`AttributeError: 'list' object has no attribute 'items'` that crashed the worker
before any user-facing validation could run. Notably, Slot 1 here was **not**
empty (it held channel `0`), so the list form is not simply the "nothing
selected" representation.

## What was done in the meantime

A shared helper `annotation_utilities.annotation_tools.get_selected_channels()`
now parses the value: it accepts the documented dict form, treats `None` as
"nothing selected", and **rejects any other shape (including lists) with a
`ValueError`**. Each affected worker catches this and calls `sendError(...)` with
a clear "the tool interface is out of date or misconfigured, please re-select
the channels" message instead of crashing.

Affected workers updated: `cellposesam`, `registration`, `deconwolf`,
`histogram_matching`, `gaussian_blur`, `rolling_ball`.

We deliberately chose to **reject** the list form rather than silently recover
it, because a `[0]` could be an un-chosen UI default rather than an intentional
selection, and running a tool on the wrong channel is worse than failing loudly.

## What still needs investigating (the actual TODO)

The **front-end (NimbusImage repo) is the source of truth** for what
`channelCheckboxes` actually serializes. We need to confirm there:

1. Does the front-end ever legitimately send a list for `channelCheckboxes`? If
   so, under what conditions (e.g. interface not initialized, older cached tool
   definition, a specific widget state)?
2. If the list form is a front-end bug, fix it there so the value is always the
   documented dict.
3. If the list form is intentional/valid, revisit the decision above — we may
   want the workers to accept and normalize it instead of erroring, and the docs
   (`CLAUDE.md`, `nimbus-interface` skill) should be updated to document both
   shapes.

Until this is resolved, users whose front-end sends the list form will get a
clear error rather than a crash, but their jobs will not run.
