"""Validate batch coordinates against a dataset's dimensions.

These are pure helpers (no I/O, no ``sendError``) so they can be unit-tested
directly and reused by any worker. The caller decides what to do with the
result (typically ``sendError`` + raise).

Coordinates are 0-indexed internally (as the batch parser emits them after its
1->0 conversion), but messages are rendered 1-indexed to match what the user
typed in the 'Batch XY/Z/Time' UI fields.
"""

# Dimension key in tiles['IndexRange'] for each dimension we validate.
INDEX_KEYS = {'XY': 'IndexXY', 'Z': 'IndexZ', 'Time': 'IndexT'}

# Human-facing label (matches the UI field name) for each dimension.
LABELS = {'XY': 'Batch XY', 'Z': 'Batch Z', 'Time': 'Batch Time'}

# Stable order in which dimensions appear in combined messages.
_ORDER = ('XY', 'Z', 'Time')


def _dimension_size(index_range, dim):
    """Number of positions along ``dim``; defaults to 1 when unknown.

    A missing ``IndexRange`` or a missing per-dimension key both mean the
    dataset has a single position along that dimension (only coord 0 valid),
    matching the existing fallback convention used elsewhere in the repo.
    """
    if not index_range:
        return 1
    return index_range.get(INDEX_KEYS[dim], 1)


def find_out_of_range(index_range, xys=None, zs=None, times=None):
    """Find requested coordinates that fall outside the dataset.

    :param dict index_range: ``tiles['IndexRange']`` (may be empty/missing keys).
    :param xys/zs/times: iterables of requested 0-indexed coordinates, or
        ``None`` to skip that dimension.
    :return: ``{dim: (sorted_bad_0indexed_values, dimension_size)}`` for every
        dimension with at least one out-of-range coordinate. Empty dict means
        everything requested is valid.
    """
    requested = (('XY', xys), ('Z', zs), ('Time', times))
    invalid = {}
    for dim, values in requested:
        if values is None:
            continue
        size = _dimension_size(index_range, dim)
        bad = sorted({v for v in values if v < 0 or v >= size})
        if bad:
            invalid[dim] = (bad, size)
    return invalid


def _phrase(bad_zero_indexed, capitalized):
    """Render a list of bad 0-indexed coordinates as a 1-indexed phrase."""
    one_indexed = sorted(v + 1 for v in bad_zero_indexed)
    if (len(one_indexed) > 1
            and one_indexed[-1] - one_indexed[0] + 1 == len(one_indexed)):
        span = f"{one_indexed[0]}-{one_indexed[-1]}"
    else:
        span = ", ".join(str(v) for v in one_indexed)

    if len(one_indexed) == 1:
        noun, verb = "Position", "does"
    else:
        noun, verb = "Positions", "do"
    if not capitalized:
        noun = noun.lower()
    return f"{noun} {span} {verb} not exist"


def format_out_of_range_message(invalid):
    """Build a short ``(message, info)`` pair for ``sendError``.

    :param invalid: the dict returned by :func:`find_out_of_range` (non-empty).
    :return: ``(message, info)``. For a single offending dimension the message
        is e.g. ``"Batch XY out of range"`` / ``"Positions 80-90 do not exist"``;
        for several, ``"Batch coordinates out of range"`` and a combined info.
    """
    dims = [d for d in _ORDER if d in invalid]

    if len(dims) == 1:
        dim = dims[0]
        bad, _size = invalid[dim]
        return f"{LABELS[dim]} out of range", _phrase(bad, capitalized=True)

    parts = []
    for dim in dims:
        bad, _size = invalid[dim]
        parts.append(f"{LABELS[dim]}: {_phrase(bad, capitalized=False)}")
    return "Batch coordinates out of range", "; ".join(parts)
