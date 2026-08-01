from shapely.geometry import Point, Polygon
import numpy as np


# Note that this function should reverse x and y. It is used by point_count_worker, which mixes up x and y for the polygons as well, so it works consistently.
# But that's not good. This function should not be used outside of the point_count_worker.
# Instead, use annotations_to_points() below.
def create_points_from_annotations(elements):
    """
    Create a list of Point objects from the x and y coordinates in each dictionary element.

    Args:
    - elements: a list of dictionary elements containing x and y coordinates

    Returns:
    - points: a list of Point objects created from the x and y coordinates in each dictionary element
    """
    points = []
    for element in elements:
        coords = element['coordinates'][0]  # Assume there is only one coordinate in the list
        x, y = coords['x'], coords['y']
        point = Point(x, y)
        points.append(point)
    return points


def filter_elements_T_XY(elements, time_value, xy_value):
    return [element for element in elements if element['location']['Time'] == time_value and element['location']['XY'] == xy_value]


def filter_elements_T_XY_Z(elements, time_value, xy_value, z_value):
    return [element for element in elements if element['location']['Time'] == time_value and element['location']['XY'] == xy_value and element['location']['Z'] == z_value]


def filter_elements_Z_XY(elements, z_value, xy_value):
    return [element for element in elements if element['location']['Z'] == z_value and element['location']['XY'] == xy_value]


def get_annotations_with_tags(elements, tags, exclusive=False):
    result = []
    tags_set = set(tags)

    # If the tags are empty and exclusive is false, return all elements
    if not tags and not exclusive:
        return elements

    for element in elements:
        element_tags_set = set(element.get('tags', []))
        if exclusive:
            # only add the element if its tags exactly match the provided tags
            if element_tags_set == tags_set:
                result.append(element)
        else:
            # add the element if it contains any of the provided tags OR both sets are empty
            if (tags_set & element_tags_set) or (not tags_set and not element_tags_set):
                result.append(element)
    return result


def filter_usable_training_samples(training_images, label_images):
    """Drop empty image/label pairs and samples without labeled objects.

    Cellpose removes samples with fewer than ``min_train_masks`` objects inside
    ``train_seg``. If every region is empty, that leaves an empty training set
    and the upstream code fails later with an opaque divide-by-zero. Filtering
    here lets workers report a useful error before starting model training.

    Returns the filtered image list, filtered label list, and number dropped.
    """
    if len(training_images) != len(label_images):
        raise ValueError("Training images and labels must have the same length.")

    usable_images = []
    usable_labels = []
    dropped = 0
    for image, labels in zip(training_images, label_images):
        if image.size == 0 or labels.size == 0 or not np.any(labels > 0):
            dropped += 1
            continue
        usable_images.append(image)
        usable_labels.append(labels)

    return usable_images, usable_labels, dropped


def get_annotations_with_tag(elements, tag, exclusive=False):
    result = []
    for element in elements:
        if exclusive:
            if element.get('tags') == [tag]:
                result.append(element)
        else:
            if tag in element.get('tags'):
                result.append(element)
    return result


def find_matching_annotations_by_location(source, target_list, Time=True, XY=True, Z=True):
    """
    This function filters the target_list based on the 'location' of the source point.
    The function parameters 'Time', 'XY', and 'Z' can be set to True or False to specify whether these 'location' attributes need to be matched.
    By default, all of these parameters are set to True, meaning all 'location' attributes need to match.

    Parameters:
    source (dict): The source point annotation object
    target_list (list): The list of target point annotation objects
    Time (bool): Specifies whether the 'Time' attribute of 'location' needs to be matched. Default is True.
    XY (bool): Specifies whether the 'XY' attribute of 'location' needs to be matched. Default is True.
    Z (bool): Specifies whether the 'Z' attribute of 'location' needs to be matched. Default is True.

    Returns:
    list: The filtered list of target point annotation objects that match the specified 'location' attributes

    Example of usage:
    1) Matching all 'location' attributes:
    source = {...}  # source point annotation object
    target_list = [...]  # target point annotation list
    matching_annotations = find_matching_annotations_by_location(source, target_list)

    2) Matching specified 'location' attributes (in this case, 'Time' and 'XY'):
    source = {...}  # source point annotation object
    target_list = [...]  # target point annotation list
    matching_annotations = find_matching_annotations_by_location(source, target_list, Time=True, XY=True, Z=False)
    """
    params = {'Time': Time, 'XY': XY, 'Z': Z}
    return [target for target in target_list if all(source['location'].get(attr) == target['location'].get(attr) for attr, value in params.items() if value)]


def annotations_to_polygons(annotations):
    """
    Convert annotations to shapely Polygon objects.

    Args:
    annotations (list or dict): A single annotation dictionary or a list of annotation dictionaries.

    Returns:
    list: A list of shapely Polygon objects.
    """
    if isinstance(annotations, dict):
        annotations = [annotations]

    polygons = []
    for annotation in annotations:
        coords = [(point['x'], point['y']) for point in annotation['coordinates']]
        polygons.append(Polygon(coords))

    return polygons


def geometry_to_polygon_coords(geometry, keep_largest_only=False):
    """Flatten a shapely geometry into a list of exterior coordinate lists,
    dropping anything empty, zero-area, or invalid.

    A negative ``buffer`` (used for polygon padding) or a ``simplify`` can turn a
    polygon into an empty geometry (small objects shrink to nothing) or split it
    into a ``MultiPolygon`` (objects pinched in two). Neither survives the naive
    ``geometry.exterior.coords`` path: an empty geometry yields ``coordinates: []``
    which the server rejects with a 400 (failing the *entire* batch upload), and a
    ``MultiPolygon`` has no ``.exterior`` attribute (raising ``AttributeError``).

    This normalizes those cases: empty / zero-area / invalid (e.g.
    self-intersecting) geometries are dropped, so a degenerate outline never
    reaches the server where it could 400 or corrupt downstream geometry-based
    measurements. By default each piece of a ``MultiPolygon`` (or
    ``GeometryCollection``) becomes its own coordinate list.

    With ``keep_largest_only=True`` a multi-part geometry instead collapses to the
    single largest valid piece (one coordinate list). Use it where callers require
    a 1:1 input->output mapping -- e.g. SAM2 propagation/tracking, where parent and
    child annotation counts must match and emitting several pieces (or zero) for
    one input mask would break that alignment.

    Returns a (possibly empty) list of coordinate lists, each a list of ``(x, y)``
    tuples suitable for building a single polygon annotation.
    """
    if geometry is None or geometry.is_empty:
        return []
    sub_geoms = getattr(geometry, "geoms", None)
    if sub_geoms is not None:  # MultiPolygon / GeometryCollection
        # Recurse first so nested multi-geometries (e.g. a GeometryCollection
        # wrapping a MultiPolygon) flatten down to their valid leaf rings.
        coords = []
        for geom in sub_geoms:
            coords.extend(geometry_to_polygon_coords(geom))
        if keep_largest_only and coords:
            # Collapse to the single largest-area ring so callers that assume
            # 1:1 cardinality (e.g. SAM2 parent/child matching) are not handed
            # extra annotations.
            coords = [max(coords, key=lambda ring: Polygon(ring).area)]
        return coords
    exterior = getattr(geometry, "exterior", None)
    if exterior is None or not geometry.is_valid or geometry.area <= 0:
        # not a polygon, invalid (self-intersecting), or a degenerate sliver
        return []
    return [list(exterior.coords)]


def polygons_to_annotations(polygons, datasetId, XY=0, Time=0, Z=0, tags=None, channel=0):
    """
    Convert shapely Polygon objects to a list of annotations.

    Args:
    polygons (list): A list of shapely Polygon objects.
    XY (int): The XY position for all annotations. Default is 0.
    Time (int): The Time position for all annotations. Default is 0.
    Z (int): The Z position for all annotations. Default is 0.
    tags (list): A list of tags to apply to all annotations. Default is None.
    channel (int): The channel for all annotations. Default is 0.
    datasetId (str): The datasetId for all annotations.

    Returns:
    list: A list of annotation dictionaries. Empty / zero-area / invalid polygons
    are dropped so a degenerate geometry never produces an empty-coordinates
    payload (which the server rejects) or crashes on a missing ``.exterior``
    attribute. A MultiPolygon collapses to its single largest piece -- callers
    here (SAM2 propagation/tracking) require one annotation per input mask, so
    splitting it would break parent/child count matching and per-frame grouping.
    """
    if not isinstance(polygons, list):
        polygons = [polygons]

    annotations = []
    for polygon in polygons:
        for ring in geometry_to_polygon_coords(polygon, keep_largest_only=True):
            coordinates = [{'x': float(y), 'y': float(x)} for x, y in ring[
                :-1]]  # Exclude the last point as it's the same as the first

            annotation = {
                'coordinates': coordinates,
                'location': {'XY': XY, 'Time': Time, 'Z': Z},
                'shape': 'polygon',
                'channel': channel,
                'datasetId': datasetId
            }

            if tags:
                annotation['tags'] = tags

            annotations.append(annotation)

    return annotations


def annotations_to_points(annotations):
    """
    Convert annotations to shapely Point objects.

    Args:
    annotations (list or dict): A single annotation dictionary or a list of annotation dictionaries.

    Returns:
    list: A list of shapely Point objects.
    """
    if isinstance(annotations, dict):
        annotations = [annotations]

    points = []
    for annotation in annotations:
        coords = annotation['coordinates'][0]  # Assume there is only one coordinate in the list
        y, x = coords['x'], coords['y']
        point = Point(x, y)
        points.append(point)

    return points


def points_to_annotations(points, datasetId, XY=0, Time=0, Z=0, tags=None, channel=0):
    """
    Convert shapely Point objects to a list of annotations.

    Args:
    points (list): A list of shapely Point objects.
    XY (int): The XY position for all annotations. Default is 0.
    Time (int): The Time position for all annotations. Default is 0.
    Z (int): The Z position for all annotations. Default is 0.
    """

    annotations = []
    for point in points:
        annotation = {
            'coordinates': [{'x': point.y, 'y': point.x}],
            'location': {'XY': XY, 'Time': Time, 'Z': Z},
            'shape': 'point',
            'channel': channel,
            'datasetId': datasetId
        }
        annotations.append(annotation)

    return annotations


# The index keys Girder puts on each frame in tileClient.tiles['frames']. A key is
# omitted entirely when the dataset has only one position along that dimension.
FRAME_INDEX_KEYS = ('IndexXY', 'IndexZ', 'IndexT', 'IndexC')


def get_frame_index(frame, dimension, default=0):
    """
    Read one index out of a frame from tileClient.tiles['frames'].

    Girder omits an index key from the frame dictionaries whenever the dataset has a
    single position along that dimension: a single-channel dataset has no 'IndexC'
    key at all, a dataset with no time series has no 'IndexT', and so on. A missing
    key therefore means "coordinate 0 along that dimension", so subscripting the
    frame directly (frame['IndexC']) raises KeyError on perfectly valid datasets.

    Args:
    frame (dict): One entry of tileClient.tiles['frames'].
    dimension (str): 'IndexC' or 'C' (likewise XY, Z, T).
    default (int): Value for an absent dimension. Default is 0, the only valid
        coordinate along a dimension the dataset does not have.

    Raises:
    ValueError: If dimension is not one of the known frame index keys, so that a
        typo fails loudly instead of silently reporting coordinate 0.
    """
    key = dimension if dimension.startswith('Index') else f'Index{dimension}'
    if key not in FRAME_INDEX_KEYS:
        raise ValueError(f"Unknown frame dimension {dimension!r}; "
                         f"expected one of {FRAME_INDEX_KEYS}.")
    return frame.get(key, default)


def frame_to_large_image_params(frame):
    """
    Convert a frame into the keyword arguments for a large_image sink's addTile().

    Girder frames carry keys such as 'IndexXY'/'IndexZ'/'IndexT'/'IndexC', which
    large_image expects as 'xy'/'z'/'t'/'c'. Dimensions the dataset does not use are
    absent from the frame and are likewise absent from the result. Any other 'Index*'
    axis is passed through the same way rather than dropped, so that an unusual axis
    still lands in its own plane instead of colliding with another frame. The bare
    'Index' key (the flat frame number) is skipped, as are non-index keys like
    'Channel'; the length test is what excludes 'Index' itself.

    Args:
    frame (dict): One entry of tileClient.tiles['frames'].
    """
    return {key.lower()[5:]: value for key, value in frame.items()
            if key.startswith('Index') and len(key) > 5}


def get_selected_channels(value, field_name='channel selection'):
    """
    Parse a `channelCheckboxes` interface value into a sorted list of channel indices.

    The only valid shape is the documented mapping of channel index to checked
    state, e.g. ``{'0': True, '1': False, '2': True}`` -> ``[0, 2]``. An unset
    field (``None``, ``''``, ``{}``) returns an empty list, which callers must
    treat as "nothing selected" and handle with their own required-field logic.

    Anything else raises ``ValueError`` rather than guessing a channel, since
    running a tool on the wrong channel is worse than failing outright.

    In particular, a bare list of channel indices (``[0]``) is rejected. The
    NimbusImage checkbox widget has never emitted that shape, and the one
    upstream path that accepts arrays (the AI panel) normalizes them to the map
    before saving, so a list value means the tool config was written by something
    outside the UI. Because we cannot confirm which channel it meant, it raises
    here rather than being read as "channel 0"; the config needs its channels
    re-selected (see todo/channelcheckboxes-serialization.md).

    Args:
    - value: the raw ``params['workerInterface'][field_name]`` value
    - field_name: name of the interface field, used in error messages

    Returns:
    - a sorted list of unique non-negative int channel indices
    """
    def bad(detail):
        return ValueError(
            f"'{field_name}' has an unexpected format ({detail}). The worker "
            f"interface may be out of date or misconfigured.")

    if value is None or value == '':
        return []

    if isinstance(value, (list, tuple)):
        raise bad(
            f"got a list of channel indices ({value!r}) instead of a mapping of "
            f"channel index to on/off")

    if not isinstance(value, dict):
        raise bad(f"{type(value).__name__}: {value!r}")

    selected = []
    for key, checked in value.items():
        if not checked:
            continue
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise bad(f"channel key {key!r} is not an integer")
        selected.append(index)

    if any(index < 0 for index in selected):
        raise bad(f"negative channel index in {value!r}")

    return sorted(set(selected))


def get_images_for_all_channels(tileClient, datasetId, XY, Z, Time):
    """
    Get images for all channels for a given XY, Z, Time
    Returns a list of images, one for each channel
    """
    images = []
    # Single-frame datasets can omit IndexRange entirely.
    num_channels = tileClient.tiles.get('IndexRange', {}).get('IndexC', 1)
    for channel in range(num_channels):
        frame = tileClient.coordinatesToFrameIndex(XY, Z, Time, channel)
        image = tileClient.getRegion(datasetId, frame=frame)
        images.append(image)
    return images


def get_layers(GirderClient, datasetId):
    """
    This function takes a datasetId and a client, and returns the layers 
    with information about contrast settings that are currently being applied.

    Note: A dataset can belong to multiple configurations, so there is some ambiguity here.
    The function takes the first configuration it finds. To do this properly would require 
    extensive reworking, because the front end and worker interface would all have to change 
    to pass the configurationId along with the datasetId. The user will also have to save 
    their contrast settings in the user interface in order for them to be detected in this way.
    """
    configurations = GirderClient.get("dataset_view", parameters={'datasetId': datasetId})
    configurationId = configurations[0]['configurationId']
    configuration = GirderClient.get("upenn_collection/" + configurationId)
    layers = configuration['meta']['layers']
    return layers


def process_and_merge_channels(images, layers, mode='lighten'):
    # Imported lazily: matplotlib is heavy (~50ms) and this is the only function
    # in annotation_tools that uses it, yet annotation_tools is imported at module
    # load by nearly every worker. See todo/worker-startup-latency.md.
    import matplotlib.colors as mcolors

    layers = sorted(layers, key=lambda x: x['channel'])
    processed_channels = []

    for img, layer in zip(images, layers):
        if layer['visible'] == False:
            continue
        img = np.squeeze(img)

        contrast_mode = layer['contrast']['mode']
        black_point = layer['contrast']['blackPoint']
        white_point = layer['contrast']['whitePoint']

        if contrast_mode == 'percentile':
            black_value = np.percentile(img, black_point)
            white_value = np.percentile(img, white_point)
        elif contrast_mode == 'absolute':
            black_value = black_point
            white_value = white_point
        else:
            raise ValueError(f"Unsupported contrast mode: {contrast_mode}")

        img_normalized = np.clip((img - black_value) / (white_value - black_value), 0, 1)

        color = np.array(mcolors.to_rgb(layer['color']))
        img_colored = img_normalized[:, :, np.newaxis] * color

        processed_channels.append(img_colored)

    if mode == 'lighten':
        merged_image = np.max(processed_channels, axis=0)
    elif mode == 'add':
        merged_image = np.sum(processed_channels, axis=0)
        merged_image = np.clip(merged_image, 0, 1)
    elif mode == 'screen':
        merged_image = 1 - np.prod(1 - np.array(processed_channels), axis=0)
    else:
        raise ValueError("Unsupported mode. Choose 'lighten', 'add', or 'screen'.")

    return merged_image
