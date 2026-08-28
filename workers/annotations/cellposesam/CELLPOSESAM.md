# Cellpose-SAM Worker

This worker runs Cellpose-SAM, a variant of Cellpose that combines Cellpose with SAM (Segment Anything Model) for cell segmentation. It supports up to three input channel slots and produces polygon annotations.

## How It Works

1. **Channel Assembly**: Collects up to three input channels from user-selected channel checkboxes and stacks them
2. **Model Selection**: Loads a built-in Cellpose-SAM checkpoint (`cpsam_v2` by default, or the original `cpsam`) or a user-trained model from Girder
3. **Tiling**: Splits the image into overlapping tiles using DeepTile
4. **Segmentation**: Runs Cellpose-SAM inference on each tile with GPU acceleration
5. **Stitching**: Merges polygons spanning tile boundaries using DeepTile's `stitch_polygons()`
6. **Post-processing**: Applies optional padding (dilation/erosion) and smoothing (polygon simplification)

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Cellpose-SAM** | notes | -- | Informational text with documentation link |
| **Batch XY** | text | -- | 1-indexed XY positions to iterate over (e.g., "1-3, 5-8"), or `all` |
| **Batch Z** | text | -- | 1-indexed Z slices to iterate over, or `all` |
| **Batch Time** | text | -- | 1-indexed Time points to iterate over, or `all` |
| **Model** | select | cellpose-sam | Model to use. `cellpose-sam` runs the `cpsam_v2` checkpoint (current default); `cellpose-sam (legacy cpsam)` runs the original April 2025 `cpsam` checkpoint. User-trained models from Girder are also listed |
| **Channel for Slot 1** | channelCheckboxes | -- | **Required.** Source channel(s) for the model's first input slot. If multiple selected, only the first is used |
| **Channel for Slot 2** | channelCheckboxes | -- | Optional second input slot channel |
| **Channel for Slot 3** | channelCheckboxes | -- | Optional third input slot channel |
| **Diameter** | number | 30 | Object diameter in pixels (range: 10-200). `30` (the default) is cellpose's identity value and segments at native resolution; other values rescale the image by `30 / Diameter` before inference |
| **Smoothing** | number | 0.7 | Polygon simplification tolerance (range: 0-10) |
| **Padding** | number | 0 | Expand (positive) or shrink (negative) polygons in pixels (range: -20 to 20) |
| **Tile Size** | number | 1024 | Tile dimension in pixels (range: 0-2048) |
| **Tile Overlap** | number | 0.1 | Fraction of overlap between adjacent tiles (range: 0-1) |

## Implementation Details

### Channel Slots vs. Primary/Secondary Channels

Unlike the standard Cellpose worker which uses Primary/Secondary channel selectors, Cellpose-SAM uses three `channelCheckboxes` slots. This allows flexible multi-channel input:

- **Slot 1** is required; an error is raised if no channel is selected
- **Slots 2 and 3** are optional
- If multiple channels are checked in a single slot, a warning is issued and only the first is used
- Selected channels are stacked in order and passed to the model
- Each slot value is parsed by `annotation_tools.get_selected_channels()` rather than `.items()`, which crashed when a saved tool config held a bare list (`[0]`) instead of the documented `{"0": True}` mapping. A value in any other shape is reported as an error rather than defaulting to some channel

### Model Behavior

- **Base models**: The dropdown labels map to cellpose built-in checkpoints in `models_config.py` — `cellpose-sam` → `cpsam_v2`, `cellpose-sam (legacy cpsam)` → `cpsam`. The selected checkpoint name is passed explicitly as `pretrained_model` rather than relying on Cellpose's internal default, which can shift between versions.
- **Custom models**: Loaded from Girder by path. They take the same optional `Diameter` rescaling as the built-in checkpoints.

### Diameter and Rescaling

`Diameter` tells cellpose what object size to rescale the image to before
segmentation. `CellposeModel.eval(diameter=...)` sets `rescale = 30 / diameter`,
so a **smaller** value enlarges the image and a **larger** one shrinks it.

**30 is the default because it is the identity.** The `30` in that formula is a
hardcoded literal inside `CellposeModel.eval()`, not a per-model `diam_mean` —
cellpose v4 ignores `diam_mean` entirely. So `Diameter = 30` gives `rescale = 1.0`,
exactly what `diameter=None` gives, and every downstream branch in cellpose keys
off `rescale != 1.0` rather than off the diameter. The two are equivalent, which lets the default be a real,
self-describing in-range value instead of an out-of-band `0` sentinel. At the
default the worker omits `diameter` from the eval call entirely, so it issues the
identical call it made while the field was absent.

Cellpose-SAM is trained at native resolution and handles a wide range of object
sizes, so **30 is the right setting for almost every dataset.** Change it only
when objects are far outside the size range the checkpoint handles well.

With `resample=True` (cellpose's default) the flows are resized back to the
original tile size afterwards, so annotation coordinates are unaffected by any
rescale. (Verified against the pinned `cellpose==4.2.1.1`, where `resample`
defaults to `True` and is not deprecated; check this again on a version bump.)

Only the **constructor** argument `diam_mean` was deprecated in cellpose v4.0.1+;
the eval-time `diameter` is still honoured in the pinned `cellpose==4.2.1.1`.
The parameter was removed from this worker in `a3e4524` on the mistaken
assumption that Cellpose-SAM ignores it entirely, and restored here.

#### Warnings and GPU memory

**The worker warns on any active rescale**, i.e. whenever `Diameter` is not 30.
It cannot distinguish a deliberate choice from a stale saved value, and a
rescale changes the segmentation either way, so it always says so rather than
staying silent. The warning names the rescale factor and the effective tile size.

A *small* Diameter enlarges each tile: `Diameter` 10 with `Tile Size` 1024 means
the network actually runs on ~3072 px tiles. The interface minimum of `10` caps
this at 3x for values entered through the UI, though a saved config can carry
less. When the effective tile size would exceed 2048 px the warning adds a
GPU-memory caveat rather than blocking, since the run may still fit. Reduce
`Tile Size` or raise `Diameter` if it runs out of memory.

Note that the warning is deliberately **not** gated on the tile size: `Tile Size`
can itself be `0`, and a small tile with a small diameter would otherwise rescale
silently.

#### Values outside the offered range

The `min`/`max` are interface hints; a saved config or a direct API call can
still carry `0`, `null`, a negative number, or something below `10`. The worker
handles these without clamping — substituting a different diameter would
silently change the segmentation:

Values are normalized by `parse_diameter()` in `models_config.py`:

| Stored value | Behavior |
|---|---|
| key absent, `null`, or `""` | Treated as the identity (`30`) — native resolution, matching how such configs ran before this field existed |
| numeric string (`"60"`) | Parsed as the number |
| `0` or negative | No rescaling; cellpose itself guards with `diameter > 0` |
| below `10` | Honored as given, with the rescale warning above |
| non-numeric (`"abc"`, a list, a boolean) | `sendError` and the job fails, rather than crashing on `float()` or silently segmenting at the wrong scale |

A boolean is rejected specifically because `bool` is an `int` subclass, so
`float(True)` would quietly mean `1.0` — a 30x upscale.

> **Note on configs saved before July 2026:** those hold the old `Diameter`
> default of `10`, which the pre-removal code applied to custom models only and
> ignored for the base checkpoints. Such a config will now rescale by 3x on
> every model. Re-check the `Diameter` on any tool config saved before then, or
> set it to `30`.

### Batch Validation

Batch XY/Z/Time values are validated against the dataset before the GPU model
is loaded. The fields use 1-based positions, so `0` is invalid; out-of-range
values produce a user-facing error that includes the dataset's valid range.
Selected input channels are range-checked at the same preflight stage.

### Built-in Checkpoints

`cpsam_v2` (SAM-ViTL backbone, released June 2026) is the default; it reduces spurious masks in low-contrast regions compared to the original `cpsam` (April 2025). Both checkpoints (~1.23 GB each) are pre-downloaded at build time by `download_models.py` so neither downloads on first run. Requires `cellpose==4.2.1.1` (cpsam_v2 was added in the 4.2.x line). To add or change the offered checkpoints, edit `models_config.py` — it is the single source of truth for both the interface and the build-time download.

### GPU Handling

The worker always requests GPU mode (`gpu=True`). Cellpose handles the fallback to CPU internally if no GPU is available.

### Tiling

Uses DeepTile to split images into square tiles with configurable size and overlap. At 1024px tile size with 0.1 overlap, the overlap region is ~102 pixels. Objects larger than the overlap region may not stitch correctly.

### Polygon Post-processing

Applied in order: padding (via Shapely `buffer()`), then smoothing (via Shapely `simplify()`).

### Model selection validation

`compute()` validates the saved `Model` selection with
`annotation_tools.get_required_select()` before loading the model. A saved tool
config can hold `null` for a `select` field even though the interface defines a
default. Previously a `null` model was passed straight to the Girder model
downloader, failing with a confusing error. Invalid selections now fail fast
with `sendError` and re-raise, so the job is recorded as failed rather than
silently succeeding; the fix is to re-select the model in the tool settings and
save the tool again. Because models can be user-trained and downloaded from
Girder, there is no static option list — validation checks only that the
selection is a non-empty string.

## Notes

- Uses `WorkerClient` for batch processing across XY/Z/Time positions
- Custom models trained via the `cellpose_train` worker appear automatically in the model dropdown
- The key difference from the standard Cellpose worker is the multi-slot channel input and the SAM-enhanced base model
