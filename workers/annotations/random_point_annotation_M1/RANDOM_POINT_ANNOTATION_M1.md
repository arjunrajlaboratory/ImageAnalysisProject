# Random Point Annotation (M1)

ARM64/Apple Silicon variant of the random point annotation worker. Generates random point annotations within image bounds. Functionally identical to `random_point` but built for M1/arm64 architecture.

## Purpose

- Testing point annotation creation on Apple Silicon (M1/M2/M3) Macs
- Local development and testing without needing x86_64 Docker images

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Number of random point annotations | number | 200 | How many points to generate (0--20,000) |
| Batch XY | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| Batch Z | text | -- | Z slices to iterate over (e.g., "1-3, 5-8") |
| Batch Time | text | -- | Time points to iterate over (e.g., "1-3, 5-8") |

## How It Works

Same as `random_point`: generates random (x, y) coordinates within tile bounds, builds point annotation dicts manually, and uploads them via `annotationClient.createMultipleAnnotations()`. Prints upload execution time.

## Notes

- Uses a local `utils.py` module (with `utils.process_range_list`) for batch range parsing, unlike `random_point` which uses `batch_argument_parser`.
- Contains the same legacy `preview` function as `random_point` (Gaussian/Laplacian threshold overlay unrelated to point generation).
- Batch iteration is declared in the interface but not fully implemented in compute -- only the current tile position is used.
- This is an older-style worker. For new development, prefer architecture-aware builds (`Dockerfile` vs `Dockerfile_M1`) within a single worker directory rather than separate `_M1` worker copies.
