# Random Squares

Generates random square polygon annotations within image bounds. This is the recommended test/demo worker for polygon annotation workflows.

## Purpose

- Quick testing of polygon annotation creation pipelines
- Stress-testing annotation upload performance (supports up to 300,000 annotations)
- Demonstrating `WorkerClient` batch mode across XY/Z/Time positions
- Verifying that annotation display and tagging work correctly

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Random Squares | notes | -- | Informational text about the worker |
| Square size | number | 10 | Side length of each square in pixels (1--200) |
| Number of squares | number | 100 | How many squares to generate per tile position (1--300,000) |
| Batch XY | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| Batch Z | text | -- | Z slices to iterate over (e.g., "1-3, 5-8") |
| Batch Time | text | -- | Time points to iterate over (e.g., "1-3, 5-8") |

## How It Works

1. Reads square size and count from the worker interface.
2. Gets the tile dimensions from the dataset.
3. For each tile position (controlled by Batch XY/Z/Time), generates the requested number of squares at random positions within the image bounds. Squares are kept fully inside the tile by offsetting from edges by half the square size.
4. Uses `WorkerClient.process()` with `f_annotation='polygon'` to handle batching and annotation upload automatically.

## Notes

- Uses the modern `WorkerClient` pattern (unlike the older test workers that manually build annotation dicts). This makes it a good reference for new annotation workers.
- The image argument passed to the processing function by `WorkerClient.process()` is unused since square generation does not depend on image content.
- Built with `build_test_workers.sh` (uses the micromamba-based `test-worker-base` image).
