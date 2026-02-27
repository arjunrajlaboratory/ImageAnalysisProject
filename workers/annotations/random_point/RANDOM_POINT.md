# Random Point

Generates random point annotations within image bounds. An older-style test worker that manually constructs annotation dicts and uploads them via `createMultipleAnnotations`.

## Purpose

- Testing point annotation creation pipelines
- Benchmarking bulk annotation upload performance (prints execution time)
- Example of the older/manual annotation creation pattern (contrast with `random_squares` which uses `WorkerClient`)

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Number of random point annotations | number | 200 | How many points to generate (0--20,000) |
| Batch XY | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| Batch Z | text | -- | Z slices to iterate over (e.g., "1-3, 5-8") |
| Batch Time | text | -- | Time points to iterate over (e.g., "1-3, 5-8") |

## How It Works

1. Reads the annotation count and batch parameters from the worker interface.
2. Parses batch ranges using `batch_argument_parser` (imported but batch iteration is only partially implemented -- defaults to the current tile position).
3. For each requested annotation, generates a random (x, y) coordinate within tile bounds.
4. Builds annotation dicts manually with shape "point" and uploads them all at once via `annotationClient.createMultipleAnnotations()`.
5. Prints the upload execution time to stdout.

## Notes

- Contains a legacy `preview` function that applies Gaussian/Laplacian filtering and generates a threshold overlay. This preview is unrelated to point generation and appears to be leftover scaffold code.
- Batch XY/Z/Time fields are declared in the interface but the compute function does not fully iterate over them -- it only uses the current tile position.
- Uses the older manual annotation pattern. For new workers, prefer the `WorkerClient.process()` pattern shown in `random_squares`.
