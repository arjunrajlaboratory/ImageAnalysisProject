# Annotation Utilities

Shared helpers for NimbusImage workers, including annotation geometry, progress,
units, and batch-coordinate parsing.

## Batch coordinate ranges

`annotation_utilities.batch_argument_parser` parses 1-indexed numeric inputs
such as `1-3, 5-8`. Standard `Batch XY`, `Batch Z`, and `Batch Time` fields also
accept case-insensitive `all` when dataset coordinates are supplied.

Prefer `get_batch_ranges(tile, worker_interface, index_range)` for standard
batch fields. It returns zero-indexed XY, Z, and Time lists, keeps the current
tile coordinate for an empty field, expands `all` from `IndexXY`, `IndexZ`, and
`IndexT`, and treats a missing dimension as the single coordinate `0`.

For a custom range field, pass its already-zero-indexed available coordinates
through `all_values`:

```python
values = batch_argument_parser.process_range_list(
    raw_value,
    convert_one_to_zero_index=True,
    all_values=range(dimension_size),
)
```

The conversion flag applies to numeric user input; `all_values` must already be
expressed in the coordinate system the caller needs.
