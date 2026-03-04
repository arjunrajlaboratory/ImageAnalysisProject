# Test Multiple Annotation

Generates random square polygon annotations within image bounds. An older-style test worker that manually constructs annotation dicts and uploads them via `createMultipleAnnotations`.

## Purpose

- Testing polygon annotation creation and bulk upload
- Benchmarking `createMultipleAnnotations` performance (prints execution time)
- Example of the older/manual polygon annotation creation pattern

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Square size | number | 10 | Side length of each square in pixels (0--30) |
| Number of random annotations | number | 100 | How many squares to generate (0--10,000) |
| Batch XY | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| Batch Z | text | -- | Z slices to iterate over (e.g., "1-3, 5-8") |
| Batch Time | text | -- | Time points to iterate over (e.g., "1-3, 5-8") |

## How It Works

1. Reads square size, annotation count, and batch parameters from the worker interface.
2. Parses batch ranges using `utils.process_range_list` (but only uses the current tile position).
3. For each requested annotation, generates a random center point within the tile, ensuring the square stays fully inside the image bounds.
4. Constructs the four corner coordinates for each square polygon.
5. Builds annotation dicts manually with shape "polygon" and uploads them all at once via `annotationClient.createMultipleAnnotations()`.
6. Prints the upload execution time to stdout.

## Notes

- Contains a legacy `preview` function (Gaussian/Laplacian threshold overlay) unrelated to square generation.
- Batch XY/Z/Time fields are declared in the interface but compute does not fully iterate over them.
- This is an older-style worker. For the modern equivalent, see `random_squares`, which uses `WorkerClient.process()` for cleaner batch handling and annotation upload.
