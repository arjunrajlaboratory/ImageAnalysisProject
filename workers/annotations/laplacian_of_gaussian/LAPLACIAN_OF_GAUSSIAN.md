# Laplacian of Gaussian Worker

Detects spots in fluorescence images using the Laplacian of Gaussian (LoG) method and creates point annotations at detected spot locations.

## How It Works

1. Applies a Gaussian filter with the specified sigma, then computes the Laplacian to enhance spot-like features
2. In **Current Z** mode: uses `skimage.feature.peak_local_max` to find local maxima above the threshold, returning 2D point coordinates
3. In **Z-Stack** mode: thresholds the 3D LoG response, labels connected components with `skimage.measure.label`, and returns the 3D centroids of each region
4. Uses `WorkerClient` for batch processing across XY/Z/Time positions

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Laplacian of Gaussian** | notes | -- | Descriptive text explaining the tool |
| **Batch XY** | text | -- | XY positions to process (e.g., "1-3, 5-8") |
| **Batch Z** | text | -- | Z positions to process (e.g., "1-3, 5-8") |
| **Batch Time** | text | -- | Time positions to process (e.g., "1-3, 5-8") |
| **Mode** | select | Current Z | Processing mode: "Current Z" for 2D per-slice detection, "Z-Stack" for 3D volumetric detection |
| **Sigma** | number | 2 | Sigma for the Gaussian filter (range 0-5). Sets the spatial scale of spots to detect. |
| **Threshold** | text | 0.001 | Sensitivity threshold for the LoG filter. Lower values detect more spots. |

## Implementation Details

- Unlike the other image processing workers in this directory, this worker creates **point annotations** rather than producing a new image. It uses `WorkerClient.process()` with `f_annotation='point'`.
- In Z-Stack mode, the worker passes `stack_zs='all'` to `WorkerClient`, which assembles a full Z-stack before calling the processing function. The Batch Z parameter is ignored in this mode since the entire stack is processed automatically.
- The threshold is defined as a text field (not number) to allow very small decimal values like 0.001.
- Preview mode shows a white overlay on pixels that exceed the threshold, letting users tune sensitivity before running the full computation.

## Notes

- Sigma controls the size of spots detected: smaller sigma values detect smaller spots, larger values detect larger spots.
- The threshold is applied to the LoG response, not to raw intensity. Use the preview to calibrate.
- In Current Z mode, `peak_local_max` with `min_distance=1` finds all distinct local maxima. In Z-Stack mode, connected component analysis finds 3D blobs.
