# Children Count

Counts the number of child annotations connected to each parent annotation. Useful for quantifying relationships such as the number of spots connected to a nucleus.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Count connected objects | notes | N/A | Descriptive note explaining the tool counts children objects connected to parent polygons. |
| Child Tags | tags | (none) | Tags used to filter which annotations count as children. |
| Child Tags Exclusive | select | No | If "Yes", child annotations must have exactly the specified tags. If "No", child annotations need at least one matching tag. Options: `Yes`, `No`. |

Parent annotations are filtered using the standard tag selector in the property configuration (via `params['tags']`), not through the worker interface.

## Computed Properties

| Property | Description |
|----------|-------------|
| Children Count | The number of child annotations (matching the child tag filter) connected to each parent annotation. Parents with no matching children receive a value of 0. |

## How It Works

1. Fetches all annotations in the dataset (regardless of shape) and all connections.
2. Filters annotations into two groups based on tags:
   - **Parent annotations**: filtered by the tags set in the property configuration (`params['tags']`).
   - **Child annotations**: filtered by the "Child Tags" interface parameter.
3. Filters connections to only those where the parent ID is in the parent set and the child ID is in the child set.
4. Groups the filtered connections by parent ID and counts the number of children per parent using pandas.
5. Assigns a count of 0 to any parent annotation that has no matching children.
6. Uploads all property values in a single batch.

## Notes

- Tag filtering supports both exclusive mode (annotations must have exactly the specified tags) and inclusive mode (annotations must have at least one matching tag).
- The worker fetches connections with a limit of 10,000,000, which should cover most datasets.
- Both parent and child annotations can be of any shape (polygon, point, line, etc.) since shape filtering is not applied.
