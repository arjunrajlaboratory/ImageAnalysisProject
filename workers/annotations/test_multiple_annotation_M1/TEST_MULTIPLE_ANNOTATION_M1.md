# Test Multiple Annotation (M1)

ARM64/Apple Silicon variant of the test multiple annotation worker. Generates random square polygon annotations within image bounds. Functionally identical to `test_multiple_annotation` but built for M1/arm64 architecture.

## Purpose

- Testing polygon annotation creation on Apple Silicon (M1/M2/M3) Macs
- Local development and testing without needing x86_64 Docker images

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Square size | number | 10 | Side length of each square in pixels (0--30) |
| Number of random annotations | number | 100 | How many squares to generate (0--10,000) |
| Batch XY | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| Batch Z | text | -- | Z slices to iterate over (e.g., "1-3, 5-8") |
| Batch Time | text | -- | Time points to iterate over (e.g., "1-3, 5-8") |

## How It Works

Same as `test_multiple_annotation`: generates random squares within tile bounds by computing four corner coordinates around a random center, builds polygon annotation dicts manually, and uploads via `annotationClient.createMultipleAnnotations()`. Prints upload execution time.

## Notes

- Uses a local `utils.py` module for batch range parsing.
- Contains the same legacy `preview` function as `test_multiple_annotation` (Gaussian/Laplacian threshold overlay unrelated to square generation).
- Batch iteration is declared in the interface but not fully implemented in compute -- only the current tile position is used.
- This is an older-style worker. For new development, prefer architecture-aware builds (`Dockerfile` vs `Dockerfile_M1`) within a single worker directory rather than separate `_M1` worker copies.
