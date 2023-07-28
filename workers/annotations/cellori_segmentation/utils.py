from itertools import chain


def process_range_list(rl):

    g = parse_range_list(rl)
    first, g = peek_generator(g)
    if first is None:
        g = None

    return g


def parse_range_list(rl):
    ranges = sorted(set(map(_parse_range, rl.split(','))), key=lambda x: (x.start, x.stop))
    return chain.from_iterable(_collapse_range(ranges))


def peek_generator(g):

    first = next(g, None)
    g = chain([first], g)

    return first, g


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
