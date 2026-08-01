# TODO-004: `channelCheckboxes` list-shaped values (submitter unidentified)

**Status:** Open (workers hardened and now reject the shape; front end surveyed, submitter of the list form still unidentified)
**Priority:** Medium
**Owner:** unassigned
**Related:** worker-side fix for the `cellposesam` crash of 2026-05-29

## Problem

The `channelCheckboxes` interface type is documented (in `CLAUDE.md`) to return a
mapping of channel index to checked state:

```json
{"0": true, "1": false, "2": true}
```

A production `cellposesam` job instead received a bare list of the selected
channel indices:

```json
{"Channel for Slot 1": [0], "Channel for Slot 2": [], "Channel for Slot 3": []}
```

Every worker that read the value with `.items()` crashed immediately:

```
AttributeError: 'list' object has no attribute 'items'
```

## What was done (worker side)

`annotation_tools.get_selected_channels(value, field_name)` parses the documented
mapping into a sorted `list[int]`, returns `[]` for an unset field (`None`, `''`,
`{}`), and raises `ValueError` on **any** other shape — including the list form.
All consumers route through it and report the `ValueError` via `sendError`:
`cellposesam`, `cellposesam_train`, `registration`, `deconwolf`,
`histogram_matching`, `gaussian_blur`, `rolling_ball`, `sample_interface`.
Regression coverage: `annotation_utilities/tests/test_selected_channels.py`
(runs in CI) plus a per-worker test asserting the list form is reported, not run.

**The list form is rejected, not normalized.** Reading `[0]` as "channel 0" is an
inference about intent, and the survey below shows the value did not come from the
checkbox UI — so its provenance, and therefore the channel it meant, is unknown.
Running a segmentation or deconvolution against a guessed channel silently
produces wrong results; failing with a clear message does not. Tolerating the
shape would also keep a second wire format alive indefinitely for a payload
nothing is known to still emit.

The tradeoff: any tool configuration that already holds a list value now reports
an error instead of running, and must have its channels re-selected and re-saved.
That is the intended outcome — see "What is still open" below.

## What the NimbusImage side actually looks like

Surveyed at NimbusImage `bc4511ca` (2026-07-31):

- **The checkbox UI has never emitted a list.** `src/components/
  ChannelCheckboxGroup.vue` has bound a `{[channel]: boolean}` object since it
  was introduced (PR #880, 2025-01-31) and still emits
  `Record<number, boolean>` today. `IChannelCheckboxesWorkerInterfaceElement`
  in `src/store/model.ts` types the value the same way.
- **There *is* a list→map normalizer, but it is agent-only and much newer than
  the crash.** `normalizeWorkerInterfaceValue()` in
  `src/utils/workerInterface.ts` accepts arrays, bare indices, channel names,
  and maps, and converts them all to the canonical map. It landed 2026-07-24
  (`51ac05da`, then `9855088f`) and is called from exactly one place:
  `src/agent/executors.ts:410`, the AI panel's `run_worker` /
  `create_property`. The AI panel itself only landed 2026-07-06 (`59af47b8`) —
  five weeks *after* the 2026-05-29 crash — so neither the panel nor its
  normalizer can be the source of that payload or the thing that prevents it.
- **Saved tool configurations bypass the normalizer.** In
  `resolveWorkerInterfaceValues`, a parameter that is not overridden is copied
  straight from the persisted tool (`values[id] = saved[id]`). Any stored
  configuration whose value is `[0]` still reaches the worker as `[0]`.
- **Upstream now documents the array as a first-class input shape.**
  `WORKER_INTERFACE_VALUE_FORMATS.channelCheckboxes` tells the agent to pass
  "an ARRAY of 0-based channel indices", with the boolean map as the
  alternative. So the array form is expected on the way in and only becomes a
  map if it happens to pass through the agent normalizer.

## What is still open

The submitter that produced `"Channel for Slot 1": [0]` on 2026-05-29 is still
unidentified. It was not the checkbox UI and not the AI panel (which did not
exist yet), which leaves a non-UI submitter — a script or REST/`girder_client`
call that writes `workerInterfaceValues` directly, or a tool configuration
persisted by something other than the checkbox UI.

To close this out:

1. Find the submitter. The job was named "cellpose-sam zero-shot on
   LCA5_P21_rep1", which reads like a scripted sweep over datasets rather than a
   click in the UI; start with whatever created that tool/job.
2. Decide on one canonical wire shape and normalize to it at the boundary rather
   than only on the agent path — e.g. normalize when a tool configuration is
   persisted, and when a job is submitted, so saved values cannot smuggle the
   other shape through.
3. Update the type table in `CLAUDE.md` / `AGENTS.md` and the
   `nimbus-interface` skill reference once a single canonical shape is settled,
   noting when the other shape stopped being emitted.
4. Repair (or delete) the tool configurations that already hold list values.
   They now fail with "Could not read the channel selection" until their channels
   are re-selected and re-saved through the UI. A one-time normalization pass over
   persisted `workerInterfaceValues` would fix them in place; without it, users hit
   the error and must recreate the tool.
5. Keep `get_selected_channels()` regardless — it is the validation boundary that
   turns any future shape drift into an actionable error instead of an
   `AttributeError`.

## Notes

- The same job log also showed `girder_worker` failing to resolve its own
  hostname (`socket.gaierror: [Errno -3] Temporary failure in name resolution`,
  "Failed to get docker network"). That is a server-side/deployment concern in
  the NimbusImage stack, not this repo, and it is non-fatal: `girder_worker`
  logs it and runs the container without attaching it to a network.
