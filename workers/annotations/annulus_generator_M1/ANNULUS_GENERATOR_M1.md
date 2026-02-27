# Annulus Generator Worker

This worker generates annulus (ring-shaped) polygon annotations around existing polygon annotations. Each annulus consists of a dilated outer boundary and the original polygon as an inner boundary, creating a ring that can be used for measuring peri-cellular or peri-nuclear intensity.

## How It Works

1. Retrieves all polygon annotations matching the current tag/assignment filters.
2. For each polygon, creates a Shapely `Polygon` from its coordinates.
3. Buffers (dilates) the polygon outward by the user-specified annulus size.
4. Constructs the annulus by concatenating the outer (dilated) boundary coordinates with the reversed inner (original) boundary coordinates, forming a polygon with a hole.
5. Uploads each annulus as a new polygon annotation with the same channel, location, and tags as the source annotation, plus an additional `annulus` tag.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Annulus size** | number | `10` | Width of the ring in pixels (range 0-30). Controls how far the outer boundary extends beyond the original polygon. |

## Implementation Details

- The annulus is represented as a single polygon whose coordinate list traces the outer boundary followed by the inner boundary in reverse order. This is the standard convention for polygons with holes.
- The worker uses Shapely's `buffer()` for dilation, which produces smooth, rounded corners on the outer boundary.
- Each generated annotation inherits all tags from the source annotation and appends `annulus` as an additional tag, making it easy to filter annuli from the original polygons.
- Annotations are uploaded one at a time via `createAnnotation()` rather than in batch.
- The worker includes a legacy `preview()` function (unused) that applies Gaussian + Laplacian filtering for thresholding visualization.

## Notes

- This worker is built for Apple Silicon (M1/arm64) as indicated by the `_M1` suffix in the directory name.
- Annuli are useful for measuring intensity in the region immediately surrounding a cell or nucleus, commonly used in membrane signal quantification.
- The worker does not modify or delete the original polygon annotations; it only creates new annotations alongside them.
- The annulus size is specified in pixel units. For physical units, convert using the dataset's pixel scale.
