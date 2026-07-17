import argparse
import json
import sys
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
        'Trackastra tracking': {
            'type': 'notes',
            'value': 'This tool links existing segmentation objects across time '
                     'into tracks using the Trackastra transformer tracking model. '
                     'It creates parent-child connections between objects, including '
                     'cell divisions (one parent linked to two daughters). '
                     'Segment your objects first (e.g. with Cellpose or Stardist), '
                     'tag them, then run this tool to build the lineage.',
            'displayOrder': 0,
        },
        'Tag of objects to track': {
            'type': 'tags',
            'tooltip': 'Track all objects that have this tag.',
            'displayOrder': 1,
        },
        'Channel': {
            'type': 'channel',
            'required': True,
            'tooltip': 'The intensity channel the tracking model looks at.\n'
                       'This should be the channel the objects were segmented on.',
            'displayOrder': 2,
        },
        'Model': {
            'type': 'select',
            'items': ['general_2d'],
            'default': 'general_2d',
            'tooltip': 'Pretrained Trackastra model to use.',
            'displayOrder': 3,
        },
        'Tracking mode': {
            'type': 'select',
            'items': ['greedy', 'greedy_nodiv'],
            'default': 'greedy',
            'tooltip': 'Linking strategy. "greedy" allows divisions, '
                       '"greedy_nodiv" forbids them.',
            'displayOrder': 4,
        },
        'Batch XY': {
            'type': 'text',
            'tooltip': 'Comma-separated list of XY positions to track, e.g. "1-3, 5".\n'
                       'Leave blank to use the current XY position only.',
            'displayOrder': 5,
        },
        'Batch Z': {
            'type': 'text',
            'tooltip': 'Comma-separated list of Z slices to track, e.g. "1-3, 5".\n'
                       'Leave blank to use the current Z slice only.',
            'displayOrder': 6,
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

    # Assign labels per time frame.
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


def track_graph_to_connections(track_graph, label_to_id, datasetId, tags):
    """Convert a Trackastra track graph into parent-child connection dicts.

    Trackastra returns a directed graph whose nodes carry a ``time`` (frame
    index) and ``label`` (label id within that frame's mask). Each edge links a
    detection to its successor; a node with two out-edges is a division. We map
    each node back to the original annotation id via ``label_to_id`` and emit a
    connection for every edge whose endpoints both map to real annotations.
    """
    node_to_id = {}
    node_time = {}
    for node, data in track_graph.nodes(data=True):
        t = data.get('time', data.get('t'))
        label = data.get('label', data.get('index'))
        node_time[node] = t
        key = (t, label)
        if key in label_to_id:
            node_to_id[node] = label_to_id[key]

    connections = []
    seen = set()
    for u, v in track_graph.edges():
        if u not in node_to_id or v not in node_to_id:
            continue
        tu = node_time.get(u)
        tv = node_time.get(v)
        # Orient parent -> child by time; skip same-frame or ambiguous edges.
        if tu is None or tv is None or tu == tv:
            continue
        parent, child = (u, v) if tu < tv else (v, u)
        parent_id = node_to_id[parent]
        child_id = node_to_id[child]
        if parent_id == child_id:
            continue
        pair = (parent_id, child_id)
        if pair in seen:
            continue
        seen.add(pair)
        connections.append({
            'datasetId': datasetId,
            'parentId': parent_id,
            'childId': child_id,
            'tags': tags,
        })

    return connections


def run_trackastra(imgs, masks, model_name, mode):
    """Load the pretrained Trackastra model and run tracking on a stack.

    Isolated so tests can patch it without importing torch/trackastra.
    """
    # Lazy imports: keep torch/trackastra off the interface/startup path.
    import torch
    from trackastra.model import Trackastra

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Trackastra ({model_name}, mode={mode}) on device {device}")
    model = Trackastra.from_pretrained(model_name, device=device)
    result = model.track(imgs, masks, mode=mode)
    # Newer Trackastra returns (track_graph, masks_tracked); older returns just
    # the graph. Accept either.
    if isinstance(result, tuple):
        return result[0]
    return result


def compute(datasetId, apiUrl, token, params):
    """
    params (could change):
        assignment, channel, connectTo, tags, tile, workerInterface
    """
    start_time = timeit.default_timer()

    workerInterface = params['workerInterface']
    track_tags = list(set(workerInterface.get('Tag of objects to track', []) or []))
    channel = workerInterface.get('Channel', params.get('channel', 0))
    model_name = workerInterface.get('Model', 'general_2d')
    mode = workerInterface.get('Tracking mode', 'greedy')

    if not track_tags:
        sendError("No tag specified",
                  info="Please select at least one tag of objects to track.")
        raise ValueError("No tag specified")

    tile = params['tile']
    # Always stamp a descriptive tag so tracking connections are identifiable
    # (and filterable / bulk-selectable) in the UI even when the user set no
    # output tags -- matching the convention of the other connection workers.
    output_tags = list(dict.fromkeys((params.get('tags') or []) + ["Trackastra"]))

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

    # Restrict to the requested XY/Z positions and group each stack.
    objectList = [ann for ann in objectList
                  if ann['location']['XY'] in batch_xy
                  and ann['location']['Z'] in batch_z]
    groups = group_annotations_by_location(objectList)

    if not groups:
        sendWarning("No annotations found",
                    info="No objects with the specified tag were found in the "
                         "selected XY/Z positions.")
        return

    # Image dimensions define the tracking canvas.
    height = tileClient.tiles['sizeY']
    width = tileClient.tiles['sizeX']

    all_connections = []
    total_groups = len(groups)
    for group_idx, ((xy, z), group_annotations) in enumerate(groups.items()):
        time_points = sorted(set(ann['location']['Time']
                                 for ann in group_annotations))
        if len(time_points) < 2:
            # A single time point cannot form a track.
            continue

        masks, label_to_id, _ = build_label_stack(
            group_annotations, time_points, height, width)

        # Load the intensity images for this stack, one per time point.
        imgs = np.zeros((len(time_points), height, width), dtype=np.float32)
        for t_idx, t in enumerate(time_points):
            frame = tileClient.coordinatesToFrameIndex(xy, z, t, channel)
            image = tileClient.getRegion(datasetId, frame=frame).squeeze()
            imgs[t_idx] = image.astype(np.float32)

        sendProgress(
            (group_idx + 0.5) / total_groups,
            "Tracking objects",
            f"Tracking XY {xy + 1}, Z {z + 1} ({len(time_points)} time points)")

        track_graph = run_trackastra(imgs, masks, model_name, mode)
        connections = track_graph_to_connections(
            track_graph, label_to_id, datasetId, output_tags)
        all_connections.extend(connections)

        sendProgress(
            (group_idx + 1) / total_groups,
            "Tracking objects",
            f"Processed {group_idx + 1} of {total_groups} stacks")

    if not all_connections:
        sendWarning("No connections created",
                    info="Trackastra did not find any links between objects.")
        return

    sendProgress(0.95, "Uploading", f"Sending {len(all_connections)} connections")
    annotationClient.createMultipleConnections(all_connections)

    elapsed = timeit.default_timer() - start_time
    print(f"Created {len(all_connections)} connections in {elapsed:.1f} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Track objects across time with Trackastra')

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
