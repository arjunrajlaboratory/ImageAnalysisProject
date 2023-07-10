from itertools import chain

def process_range_list(rl, convert_one_to_zero_index=False, convert_zero_to_one_index=False):

    if convert_one_to_zero_index and convert_zero_to_one_index:
        raise ValueError("Both 'convert_one_to_zero_index' and 'convert_zero_to_one_index' cannot be set to True at the same time.")
    
    g = parse_range_list(rl)
    first, g = peek_generator(g)

    if convert_zero_to_one_index:
        g = (x + 1 for x in g)

    if convert_one_to_zero_index:
        g = (x - 1 for x in g)

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


