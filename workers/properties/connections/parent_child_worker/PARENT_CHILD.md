# Parent-Child Connection IDs

Documents the connections between annotations by assigning each annotation a numeric ID and recording the IDs of its parent and child. Particularly useful for time-lapse tracking where objects are connected across frames.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Connection IDs | notes | N/A | Descriptive note explaining the tool documents connections between objects with IDs for parent/child relationships. |
| Ignore self-connections | checkbox | True | If checked, connections where the parent and child are the same annotation are skipped. |
| Time lapse | checkbox | True | If checked, enforces temporal ordering so the parent is always at an earlier (or equal) time point and the child is at a later time point. Connection direction is swapped if necessary. |
| Add track IDs | checkbox | False | If checked, computes a track ID for each annotation using connected component analysis. Annotations linked by any chain of connections share the same track ID. |

Parent annotations are filtered using the standard tag selector in the property configuration (via `params['tags']`).

## Computed Properties

| Property | Description |
|----------|-------------|
| annotationId | A sequential integer ID (starting from 0) assigned to each annotation. |
| parentId | The integer ID of this annotation's parent, or -1 if no parent connection exists. |
| childId | The integer ID of this annotation's child, or -1 if no child connection exists. |
| trackId | (Optional, when "Add track IDs" is checked) An integer identifying the connected component this annotation belongs to. All annotations reachable through chains of connections share the same track ID. |

All property values are stored as floats.

## How It Works

1. Fetches all connections and all annotations in the dataset, then filters annotations by the configured tags.
2. Assigns a sequential integer ID to each annotation (the `annotationId` property).
3. Initializes each annotation's `parentId` and `childId` to -1 (no connection).
4. Iterates through all connections:
   - Skips self-connections if "Ignore self-connections" is checked.
   - Skips connections where either endpoint is not in the filtered annotation set.
   - In time-lapse mode: if the "child" is at an earlier or equal time point compared to the "parent", the connection direction is reversed so that the earlier annotation is always the parent.
   - Records the parent and child relationships on the respective annotations.
5. If "Add track IDs" is enabled, runs a union-find algorithm over all filtered connections to identify connected components, then assigns a sequential track ID to each component.
6. Deletes any existing property values for this property before uploading the new values in a single batch.

## Notes

- In time-lapse mode, each annotation can have at most one parent and one child. If an annotation has multiple connections, only the last-processed connection's values are retained.
- Without time-lapse mode, the same last-write-wins behavior applies, but connections are not reoriented by time.
- The integer ID mapping is deterministic within a single run but is not stable across runs (IDs are assigned based on annotation fetch order).
- The union-find algorithm for track IDs uses path compression for efficiency.
- Old property values for this property are explicitly deleted before new values are uploaded, ensuring clean results.
