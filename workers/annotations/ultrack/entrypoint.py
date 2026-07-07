import argparse
import json
import sys
import tempfile
import timeit

import annotation_client.annotations as annotations
import annotation_client.workers as workers
import annotation_client.tiles as tiles
from annotation_client.utils import sendProgress, sendError, sendWarning

import annotation_utilities.annotation_tools as annotation_tools
import annotation_utilities.batch_argument_parser as batch_argument_parser

import numpy as np


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    interface = {
        'Ultrack tracking': {
            'type': 'notes',
            'value': 'This tool links existing segmentation objects across time '
                     'into tracks using Ultrack, which solves a global optimization '
                     'over the segmentations. It creates parent-child connections '
                     'between objects, including cell divisions (one parent linked to '
                     'two daughters). Segment your objects first (e.g. with Cellpose '
                     'or Stardist), tag them, then run this tool to build the lineage.',
            'displayOrder': 0,
        },
        'Tag of objects to track': {
            'type': 'tags',
            'tooltip': 'Track all objects that have this tag.',
            'displayOrder': 1,
        },
        'Max distance': {
            'type': 'number',
            'min': 0,
            'max': 1000,
            'default': 50,
            'unit': 'pixels',
            'tooltip': 'The maximum distance (in pixels) an object may move\n'
                       'between consecutive time points to be linked.',
            'displayOrder': 2,
        },
        'Allow divisions': {
            'type': 'checkbox',
            'default': True,
            'tooltip': 'Allow one object to split into two (cell division).\n'
                       'Uncheck to forbid divisions.',
            'displayOrder': 3,
        },
        'Batch XY': {
            'type': 'text',
            'tooltip': 'Comma-separated list of XY positions to track, e.g. "1-3, 5".\n'
                       'Leave blank to use the current XY position only.',
            'displayOrder': 4,
        },
        'Batch Z': {
            'type': 'text',
            'tooltip': 'Comma-separated list of Z slices to track, e.g. "1-3, 5".\n'
                       'Leave blank to use the current Z slice only.',
            'displayOrder': 5,
        },
    }
    client.setWorkerImageInterface(image, interface)


def group_annotations_by_location(annotation_list):
    """Group polygon annotations by (XY, Z) so each stack is tracked independently.

    Returns a dict mapping (XY, Z) -> list of annotations.
    """
    groups = {}
    for ann in annotation_list:
        if ann.get('shape') != 'polygon':
            continue
        if len(ann.get('coordinates', [])) < 3:
            continue
        loc = ann['location']
        key = (loc['XY'], loc['Z'])
        groups.setdefault(key, []).append(ann)
    return groups


def build_label_stack(group_annotations, time_points, height, width):
    """Rasterize polygon annotations into a (T, H, W) integer label stack.

    Each annotation is drawn with a unique per-frame label. Returns:
      masks:          (T, H, W) int32 label image stack
      label_to_id:    {(t_index, label): annotation_id}
      label_centroid: {(t_index, label): (row, col)} centroid in image space
    """
    # Lazy import: keeps scikit-image off the interface path; only needed during compute.
    from skimage.draw import polygon as sk_polygon

    time_to_index = {t: i for i, t in enumerate(time_points)}
    masks = np.zeros((len(time_points), height, width), dtype=np.int32)
    label_to_id = {}
    label_centroid = {}

    next_label = {t: 1 for t in time_points}
    for ann in group_annotations:
        t = ann['location']['Time']
        if t not in time_to_index:
            continue
        t_idx = time_to_index[t]
        coords = ann['coordinates']
        if len(coords) < 3:
            continue
        # The 0.5 offset converts Girder (top-left origin) coordinates to
        # scikit-image pixel-center coordinates. Rows are y, columns are x.
        rr_poly = np.array([c['y'] - 0.5 for c in coords])
        cc_poly = np.array([c['x'] - 0.5 for c in coords])
        rr, cc = sk_polygon(rr_poly, cc_poly, shape=(height, width))
        if len(rr) == 0:
            continue
        label = next_label[t]
        next_label[t] += 1
        masks[t_idx, rr, cc] = label
        label_to_id[(t_idx, label)] = ann['_id']
        label_centroid[(t_idx, label)] = (float(rr.mean()), float(cc.mean()))

    return masks, label_to_id, label_centroid


def _match_node_to_annotation(t_idx, y, x, centroids_by_time):
    """Find the annotation id whose mask centroid is nearest to (y, x) at t_idx.

    centroids_by_time maps t_index -> list of (annotation_id, row, col).
    Returns the nearest annotation id, or None if there are no candidates.
    """
    candidates = centroids_by_time.get(t_idx)
    if not candidates:
        return None
    best_id = None
    best_dist = None
    for ann_id, row, col in candidates:
        dist = (row - y) ** 2 + (col - x) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = ann_id
    return best_id


def tracks_df_to_connections(tracks_df, label_centroid, label_to_id, datasetId, tags):
    """Convert an Ultrack tracks DataFrame into parent-child connection dicts.

    Each row of ``tracks_df`` is a detection at time ``t`` with centroid
    ``(y, x)`` belonging to ``track_id``; ``parent_track_id`` links a track to
    the track it divided from (-1 for founder tracks). We map each detection to
    the nearest original annotation (via mask centroids), then emit connections:

      * consecutive detections within a track (motion links), and
      * the last detection of a parent track to the first detection of each of
        its child tracks (division links).
    """
    # Index the mask centroids by time for nearest-neighbour matching.
    centroids_by_time = {}
    for (t_idx, label), (row, col) in label_centroid.items():
        ann_id = label_to_id.get((t_idx, label))
        if ann_id is None:
            continue
        centroids_by_time.setdefault(t_idx, []).append((ann_id, row, col))

    # Assign every track node to an annotation id, and record each track's
    # parent in the same pass. node_ids[track_id] -> list of (t, annotation_id).
    node_ids = {}
    parent_of = {}
    for _, r in tracks_df.iterrows():
        track_id = int(r['track_id'])
        if track_id not in parent_of:
            raw_parent = r.get('parent_track_id', -1)
            # Founder tracks may encode "no parent" as -1, 0, or NaN depending
            # on the Ultrack version. NaN/invalid becomes -1 here; the 0 vs -1
            # distinction is handled by the "<= 0" guard on the division loop.
            try:
                parent_of[track_id] = int(raw_parent)
            except (ValueError, TypeError):
                parent_of[track_id] = -1
        t_idx = int(r['t'])
        y = float(r['y'])
        x = float(r['x'])
        ann_id = _match_node_to_annotation(t_idx, y, x, centroids_by_time)
        if ann_id is None:
            continue
        node_ids.setdefault(track_id, []).append((t_idx, ann_id))

    for track_id in node_ids:
        node_ids[track_id].sort(key=lambda pair: pair[0])

    connections = []
    seen = set()

    def add_connection(parent_id, child_id):
        if parent_id is None or child_id is None or parent_id == child_id:
            return
        pair = (parent_id, child_id)
        if pair in seen:
            return
        seen.add(pair)
        connections.append({
            'datasetId': datasetId,
            'parentId': parent_id,
            'childId': child_id,
            'tags': tags,
        })

    # Motion links within each track.
    for track_id, nodes in node_ids.items():
        for i in range(len(nodes) - 1):
            add_connection(nodes[i][1], nodes[i + 1][1])

    # Division links: parent track's last node -> child track's first node.
    # Ultrack track ids are positive, and founder tracks encode "no parent" as
    # -1 or 0 depending on version, so anything <= 0 means no parent.
    for track_id, parent_track_id in parent_of.items():
        if parent_track_id is None or parent_track_id <= 0:
            continue
        parent_nodes = node_ids.get(parent_track_id)
        child_nodes = node_ids.get(track_id)
        if not parent_nodes or not child_nodes:
            continue
        add_connection(parent_nodes[-1][1], child_nodes[0][1])

    return connections


def run_ultrack(masks, max_distance, allow_division, working_dir):
    """Run Ultrack on a (T, H, W) label stack and return the tracks DataFrame.

    Isolated so tests can patch it without importing ultrack.
    """
    # Lazy import: keeps ultrack off the interface/startup path.
    from ultrack import MainConfig, Tracker

    config = MainConfig()
    # Ultrack persists intermediate results to a SQLite database in this dir.
    config.data_config.working_dir = working_dir
    config.linking_config.max_distance = float(max_distance)
    if not allow_division:
        # A strongly negative division weight makes the solver avoid splits.
        # This is a strong penalty in the ILP objective, not a hard constraint.
        config.tracking_config.division_weight = -1e6

    tracker = Tracker(config)
    tracker.track(labels=masks, overwrite=True)
    tracks_df, _graph = tracker.to_tracks_layer(include_parents=True)
    return tracks_df


def compute(datasetId, apiUrl, token, params):
    """
    params (could change):
        assignment, channel, connectTo, tags, tile, workerInterface
    """
    start_time = timeit.default_timer()

    workerInterface = params['workerInterface']
    track_tags = list(set(workerInterface.get('Tag of objects to track', []) or []))
    max_distance = float(workerInterface.get('Max distance', 50))
    allow_division = bool(workerInterface.get('Allow divisions', True))

    if not track_tags:
        sendError("No tag specified",
                  info="Please select at least one tag of objects to track.")
        raise ValueError("No tag specified")

    tile = params['tile']
    # Always stamp a descriptive tag so tracking connections are identifiable
    # (and filterable / bulk-selectable) in the UI even when the user set no
    # output tags -- matching the convention of the other connection workers.
    output_tags = list(dict.fromkeys((params.get('tags') or []) + ["Ultrack"]))

    batch_xy = batch_argument_parser.process_range_list(
        workerInterface.get('Batch XY', None), convert_one_to_zero_index=True)
    batch_z = batch_argument_parser.process_range_list(
        workerInterface.get('Batch Z', None), convert_one_to_zero_index=True)
    if batch_xy is None:
        batch_xy = [tile['XY']]
    if batch_z is None:
        batch_z = [tile['Z']]
    batch_xy = list(batch_xy)
    batch_z = list(batch_z)

    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    sendProgress(0.05, "Loading objects", "Fetching annotations")

    blobAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=1000000, shape='polygon')
    objectList = annotation_tools.get_annotations_with_tags(
        blobAnnotationList, track_tags, exclusive=False)

    if not objectList:
        sendWarning("No annotations found",
                    info="No objects with the specified tag were found to track.")
        return

    objectList = [ann for ann in objectList
                  if ann['location']['XY'] in batch_xy
                  and ann['location']['Z'] in batch_z]
    groups = group_annotations_by_location(objectList)

    if not groups:
        sendWarning("No annotations found",
                    info="No objects with the specified tag were found in the "
                         "selected XY/Z positions.")
        return

    height = tileClient.tiles['sizeY']
    width = tileClient.tiles['sizeX']

    all_connections = []
    total_groups = len(groups)
    for group_idx, ((xy, z), group_annotations) in enumerate(groups.items()):
        time_points = sorted(set(ann['location']['Time']
                                 for ann in group_annotations))
        if len(time_points) < 2:
            continue

        masks, label_to_id, label_centroid = build_label_stack(
            group_annotations, time_points, height, width)

        sendProgress(
            (group_idx + 0.5) / total_groups,
            "Tracking objects",
            f"Tracking XY {xy + 1}, Z {z + 1} ({len(time_points)} time points)")

        # Each stack gets an isolated working directory for Ultrack's database.
        with tempfile.TemporaryDirectory() as working_dir:
            tracks_df = run_ultrack(masks, max_distance, allow_division, working_dir)

        connections = tracks_df_to_connections(
            tracks_df, label_centroid, label_to_id, datasetId, output_tags)
        all_connections.extend(connections)

        sendProgress(
            (group_idx + 1) / total_groups,
            "Tracking objects",
            f"Processed {group_idx + 1} of {total_groups} stacks")

    if not all_connections:
        sendWarning("No connections created",
                    info="Ultrack did not find any links between objects.")
        return

    sendProgress(0.95, "Uploading", f"Sending {len(all_connections)} connections")
    annotationClient.createMultipleConnections(all_connections)

    elapsed = timeit.default_timer() - start_time
    print(f"Created {len(all_connections)} connections in {elapsed:.1f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Track objects across time with Ultrack')

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    params = json.loads(args.parameters)
    datasetId = args.datasetId
    apiUrl = args.apiUrl
    token = args.token

    match args.request:
        case 'compute':
            compute(datasetId, apiUrl, token, params)
        case 'interface':
            interface(params['image'], apiUrl, token)
