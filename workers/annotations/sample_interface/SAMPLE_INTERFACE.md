# Sample Interface

Demonstrates every available NimbusImage worker interface type, tooltip/display options, and messaging function. This is the reference implementation for building worker UIs.

## Purpose

- **Primary reference** for all available interface types and their configuration options
- Demonstrates `sendProgress`, `sendWarning`, and `sendError` messaging functions
- Shows how to read each interface type's return value in the compute function
- Demonstrates `WorkerClient` batch mode and annotation creation

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| About this worker | notes | -- | Rich HTML text block (supports `<b>`, etc.). Display-only, not readable in compute. |
| Sample number | number | 42 | Numeric input with `min` (0), `max` (100), `unit` ("pixels"), and `tooltip`. Returns `int` or `float`. |
| Sample text | text | "Hello, NimbusImage!" | Text input with `vueAttrs` for placeholder, label, `persistentPlaceholder`, and `filled`. Returns `str`. |
| Sample select | select | "Option A" | Dropdown with `items` list: ["Option A", "Option B", "Option C"]. Returns `str`. |
| Sample checkbox | checkbox | False | Boolean toggle. When checked, triggers an extra warning message. Returns `bool`. |
| Sample channel | channel | -- | Single-channel selector. Returns `int` (channel index). |
| Sample channel checkboxes | channelCheckboxes | -- | Multi-channel checkbox selector. Returns `dict` of `str` to `bool` (e.g., `{"0": True, "1": False}`). |
| Sample tags | tags | -- | Tag selector. Returns `list` of `str` (e.g., `["DAPI blob"]`). |
| Sample layer | layer | -- | Layer selector (`required: False`). Returns `str` (layer ID). |
| Batch XY | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| Batch Z | text | -- | Z slices to iterate over (e.g., "1-3, 5-8") |
| Batch Time | text | -- | Time points to iterate over (e.g., "1-3, 5-8") |

## Interface Type Reference

All interface field definitions support these common keys:
- `type` (required): One of `notes`, `number`, `text`, `select`, `checkbox`, `channel`, `channelCheckboxes`, `tags`, `layer`
- `default`: Default value
- `tooltip`: Hover text shown in the UI
- `displayOrder`: Integer controlling display order (lower = higher)
- `required`: Boolean (default True for most types)
- `vueAttrs`: Dict of Vue/Vuetify component attributes (placeholder, label, filled, etc.)

Type-specific keys:
- **number**: `min`, `max`, `unit`
- **select**: `items` (list of strings)
- **notes**: `value` (HTML string to display)

## How It Works

1. Registers an interface with all 9 available types plus 3 batch text fields.
2. On compute, reads all interface values using `worker.workerInterface.get()`.
3. Sends progress, warning, and error messages (with and without `info` strings) to demonstrate all messaging patterns.
4. If the checkbox is checked, sends an additional conditional warning.
5. Uses the "Sample number" value as the square size to generate 5 random square polygon annotations per tile position via `WorkerClient.process()`.

## Messaging Functions Demonstrated

| Function | Behavior |
|----------|----------|
| `sendProgress(fraction, title, info)` | Updates the progress bar (fraction 0.0--1.0) |
| `sendWarning(msg)` | Sends a warning with no detail text |
| `sendWarning(msg, info=...)` | Sends a warning with detail text |
| `sendError(msg)` | Sends an error with no detail text (worker continues) |
| `sendError(msg, info=...)` | Sends an error with detail text (worker continues) |

## Notes

- This worker intentionally sends errors as demonstrations; they do not stop execution.
- Built with `build_test_workers.sh` (uses the micromamba-based `test-worker-base` image).
- Uses `time.sleep(1)` between messages so they are visible in the UI in sequence.
