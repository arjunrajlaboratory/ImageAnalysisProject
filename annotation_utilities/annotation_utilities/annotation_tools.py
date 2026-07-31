from shapely.geometry import Point, Polygon
import numpy as np

try:  # shapely >= 2.0
    from shapely.errors import GEOSException as _GEOSException
except ImportError:  # pragma: no cover - shapely 1.x fallback
    from shapely.errors import ShapelyError as _GEOSException

# Everything shapely throws when handed a geometry it cannot work with. GEOS
# errors are not ValueErrors, so they slip past a naive `except ValueError` and
# kill the worker run.
GEOMETRY_ERRORS = (ValueError, TypeError, AttributeError, _GEOSException)


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
    if exterior is None:
        return []  # not a polygon (a Point/LineString has no exterior)
    try:
        # `not (area > 0)` rather than `area <= 0` so a NaN area (from non-finite
        # coordinates) is rejected too -- every NaN comparison is False.
        if not geometry.is_valid or not (geometry.area > 0):
            # invalid (self-intersecting) or a degenerate sliver
            return []
    except GEOMETRY_ERRORS:
        return []
    return [list(exterior.coords)]


def safe_polygon(coords):
    """Build a shapely ``Polygon`` from raw coordinates, or return ``None``.

    ``geometry_to_polygon_coords`` only protects the code *after* a geometry
    exists; this protects the construction itself, which is where segmentation
    workers actually crash. ``Polygon(coords)`` raises ``ValueError`` for a
    contour with fewer than 3 points (a one-pixel-wide mask), and a contour
    carrying a NaN/inf coordinate builds a geometry that detonates later inside
    ``simplify()`` with a ``GEOSException``. Both abort the entire worker run, so
    one unusable mask out of hundreds loses every good annotation in the frame.

    ``coords`` may be a coordinate sequence (list of ``(x, y)`` tuples, a numpy
    contour array, ...) or an already-built geometry, which is returned as-is. A
    third coordinate column is dropped: a 3D ring yields 3-tuples, which break
    the ``(x, y)`` unpacking used to build annotation coordinates.

    Returns a ``Polygon``, or ``None`` if the input cannot form a usable one.
    """
    if coords is None:
        return None
    if hasattr(coords, "geom_type"):  # already a shapely geometry
        return coords
    try:
        array = np.asarray(coords, dtype=float)
    except (TypeError, ValueError):
        return None  # e.g. a list of dicts, or a string
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
        return None  # too few points, or not (x, y) pairs, to form a ring
    array = array[:, :2]
    if not np.isfinite(array).all():
        return None
    try:
        return Polygon(array)
    except GEOMETRY_ERRORS:
        return None


def safe_buffer(geometry, distance):
    """``geometry.buffer(distance)``, tolerating ``None`` and GEOS failures.

    Used for polygon padding. Returns ``None`` if there is nothing to buffer or
    the operation fails; an empty result is left to
    ``geometry_to_polygon_coords`` to drop.
    """
    if geometry is None:
        return None
    try:
        return geometry.buffer(distance)
    except GEOMETRY_ERRORS:
        return None


def safe_simplify(geometry, tolerance, preserve_topology=True):
    """``geometry.simplify(tolerance)``, tolerating ``None`` and GEOS failures.

    Used for polygon smoothing. A geometry built from non-finite coordinates
    raises ``GEOSException: Non-finite envelope bounds`` here, which is fatal to
    the run unless caught.
    """
    if geometry is None:
        return None
    try:
        return geometry.simplify(tolerance, preserve_topology=preserve_topology)
    except GEOMETRY_ERRORS:
        return None


def clean_polygon_coords(coords, padding=0, smoothing=0, keep_largest_only=False):
    """Turn one raw mask contour into annotation-ready coordinate rings.

    This is the whole guarded pipeline the segmentation workers need: build the
    polygon, apply optional ``padding`` (buffer) then ``smoothing`` (simplify) --
    the order the cellpose-family workers have always used -- and normalize the
    result with :func:`geometry_to_polygon_coords`. Anything unusable at any step
    is dropped rather than raised, so a single bad contour costs one object
    instead of the entire frame.

    Returns a (possibly empty) list of coordinate lists, each a list of
    ``(x, y)`` tuples ready to become one polygon annotation.
    """
    geometry = safe_polygon(coords)
    if padding:
        geometry = safe_buffer(geometry, padding)
    if smoothing and smoothing > 0:
        geometry = safe_simplify(geometry, smoothing)
    return geometry_to_polygon_coords(geometry, keep_largest_only=keep_largest_only)


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
