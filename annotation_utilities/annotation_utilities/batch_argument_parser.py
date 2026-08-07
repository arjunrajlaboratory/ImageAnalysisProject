from itertools import chain

# Placeholder for any range field whose parser is given `all_values`, so the
# literal `all` really works. It lives next to the parser deliberately: the
# text is a promise about process_range_list()'s behavior, and keeping the two
# in one file is what stops them drifting apart. A range field that does NOT
# pass `all_values` must not advertise this string -- typing `all` there raises.
BATCH_RANGE_PLACEHOLDER = 'ex. 1-3, 5-8, or all'

# The dimensions behind the standard Batch fields, in display order:
# (field name, label noun, tooltip noun).
_BATCH_DIMENSIONS = (
    ('Batch XY', 'XY positions', 'XY'),
    ('Batch Z', 'Z slices', 'Z'),
    ('Batch Time', 'Time points', 'Time'),
)


def batch_interface_fields(display_order=1, verb='iterate over',
                           tooltips=True):
    """Return the standard `Batch XY` / `Batch Z` / `Batch Time` interface fields.

    Every worker that batches over image coordinates exposes the same three
    text fields, so define them once here rather than hand-copying the dict
    into each entrypoint -- that copying is how several workers ended up with
    no placeholder at all, and how the `or all` text was missed when `all`
    support landed.

    `display_order` is the order of `Batch XY`; Z and Time follow it. `verb`
    tunes the label ("iterate over", "process", ...). Callers needing wording
    specific to the worker should call this and then adjust the returned dict,
    so the shared parts stay shared.

    Pair with :func:`get_batch_ranges`, which parses exactly these fields.
    """
    fields = {}
    for offset, (field_name, noun, dimension) in enumerate(_BATCH_DIMENSIONS):
        vue_attrs = {
            'placeholder': BATCH_RANGE_PLACEHOLDER,
            'label': f'Enter the {noun} you want to {verb}',
            'persistentPlaceholder': True,
            'filled': True,
        }
        if tooltips:
            vue_attrs['tooltip'] = (
                f'Enter {dimension} positions separated by commas, or all for '
                f'every {dimension} position. Leave blank to use only the '
                f'current {dimension} position.')
        fields[field_name] = {
            'type': 'text',
            'vueAttrs': vue_attrs,
            'displayOrder': display_order + offset,
        }
    return fields


def process_range_list(
        rl,
        convert_one_to_zero_index=False,
        convert_zero_to_one_index=False,
        all_values=None):

    if rl is None or rl == '':
        return None

    if convert_one_to_zero_index and convert_zero_to_one_index:
        raise ValueError("Both 'convert_one_to_zero_index' and 'convert_zero_to_one_index' cannot be set to True at the same time.")

    if isinstance(rl, str) and rl.strip().lower() == 'all':
        if all_values is None:
            raise ValueError("'all' requires all_values to define the available coordinates.")
        return iter(all_values)

    g = parse_range_list(rl)
    first, g = peek_generator(g)

    if convert_zero_to_one_index:
        g = (x + 1 for x in g)

    if convert_one_to_zero_index:
        g = (x - 1 for x in g)

    if first is None:
        g = None

    return g


def get_batch_ranges(tile, worker_interface, index_range=None):
    """Return zero-indexed XY, Z, and Time coordinates for Batch fields.

    Empty fields retain the current tile coordinate. The case-insensitive value
    ``all`` expands to every coordinate available in the corresponding dataset
    dimension. A missing IndexRange dimension represents a single coordinate.

    ``process_range_list`` returns one-shot generators; the coordinates are
    materialized into lists here so callers can inspect them more than once
    (e.g. WorkerClient validates them before iterating).

    Raises ValueError naming the offending field when a Batch field cannot be
    parsed, so the caller can surface it with sendError rather than leaking a
    bare parser traceback.
    """
    index_range = index_range or {}
    dimensions = (
        ('Batch XY', 'XY', 'IndexXY'),
        ('Batch Z', 'Z', 'IndexZ'),
        ('Batch Time', 'Time', 'IndexT'),
    )
    batches = []

    for field_name, tile_key, index_key in dimensions:
        raw_value = worker_interface.get(field_name)
        available = range(index_range.get(index_key, 1))
        try:
            values = process_range_list(
                raw_value,
                convert_one_to_zero_index=True,
                all_values=available,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must contain 1-based positions or ranges, for "
                f"example '1-3, 5-8', or 'all' to cover every coordinate. "
                f"Could not parse {raw_value!r}: {exc}") from exc
        batches.append([tile[tile_key]] if values is None else list(values))

    return tuple(batches)


def parse_range_list(rl):
    ranges = sorted(set(map(_parse_range, rl.split(','))), key=lambda x: (x.start, x.stop))
    return chain.from_iterable(_collapse_range(ranges))


def peek_generator(g):

    first = next(g, None)
    g = chain([first], g)

    return first, g

def get_batch_information(tile, workerInterface, batchXYstring, batchZstring, batchTimestring):
    # Probably better not to specify the strings like 'Batch XY' here, but to pass them as arguments, but this is how it is done in the example
    batch_xy = workerInterface.get('Batch XY', None)
    batch_z = workerInterface.get('Batch Z', None)
    batch_time = workerInterface.get('Batch Time', None)

    batch_xy = process_range_list(batch_xy)
    batch_z = process_range_list(batch_z)
    batch_time = process_range_list(batch_time)

    if batch_xy is None:
        batch_xy = [tile['XY']]
    if batch_z is None:
        batch_z = [tile['Z']]
    if batch_time is None:
        batch_time = [tile['Time']]

    return batch_xy, batch_z, batch_time


def _parse_range(r):
    parts = list(_split_range(r.strip()))
    if len(parts) == 0:
        return range(0, 0)
    elif len(parts) > 2:
        raise ValueError('Invalid range: {}'.format(r))
    return range(parts[0], parts[-1] + 1)

def _collapse_range(ranges):
        end = None
        for value in ranges:
            yield range(max(end, value.start), max(value.stop, end)) if end else value
            end = max(end, value.stop) if end else value.stop

def _split_range(value):
    value = value.split('-')
    for val, prev in zip(value, chain((None,), value)):
        if val != '':
            val = int(val)
            if prev == '':
                val *= -1
            yield val

