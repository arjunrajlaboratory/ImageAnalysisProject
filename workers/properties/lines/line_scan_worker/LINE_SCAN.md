# Line Scan CSV

Computes pixel intensity profiles along line annotations and exports the results as a CSV file uploaded to the dataset folder.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Line Scan CSV | notes | N/A | Descriptive note explaining the tool computes intensity along lines and saves results to CSV. |
| All channels | checkbox | True | If checked, computes the intensity profile for all channels. If unchecked, uses only the selected channel. |
| Channel | channel | 0 | The specific channel to use when "All channels" is unchecked. |
| File name | text | `line_scan_output.csv` | The name of the output CSV file uploaded to the dataset folder. |

## Computed Properties

This worker does not compute annotation properties. Instead, it produces a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| Annotation ID | The unique ID of the line annotation. |
| Tags | Comma-separated tags on the annotation. |
| Header | Either `X`, `Y`, or `Channel N` indicating the data type of that row. |
| Values | Comma-separated list of values (coordinates or intensity values along the line). |

Each annotation produces one row per data type per channel: an X row, a Y row, and one intensity row per channel.

## How It Works

1. Fetches all line annotations from the dataset (no tag filtering).
2. For each annotation, determines the image location (Time, Z, XY) and loads the relevant image frame(s). Images are cached and only reloaded when the location changes.
3. For each line segment between consecutive coordinate points, generates evenly spaced sample points using linear interpolation. The number of sample points equals the pixel distance between endpoints plus one.
4. Applies the standard 0.5 pixel offset to convert from annotation coordinates to image pixel centers.
5. Uses `scipy.ndimage.map_coordinates` with bilinear interpolation (order=1) to extract intensity values at the sampled points.
6. Collects all results into a pandas DataFrame and uploads the CSV to the dataset folder via the Girder API.

## Notes

- The worker applies the 0.5 pixel offset (`coord - 0.5`) when converting annotation coordinates to image coordinates, consistent with the convention that annotation coordinates reference pixel top-left corners while image operations use pixel centers.
- For multi-segment lines, intermediate segment endpoints are not duplicated (endpoint=False for all segments except the last).
- Progress is reported during annotation processing (0-90%), CSV generation (90-95%), and file upload (95-100%).
- The output CSV uses a "long" format where each row contains comma-separated values in a single `Values` column, rather than one column per sample point.
