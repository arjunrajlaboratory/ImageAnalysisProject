import argparse
import json
import sys
from itertools import product

import annotation_client.annotations as annotations_client
import annotation_client.workers as workers
import annotation_client.tiles as tiles

import annotation_utilities.annotation_tools as annotation_tools
import annotation_utilities.batch_argument_parser as batch_argument_parser

import numpy as np
from shapely.geometry import Polygon
from skimage.measure import find_contours

from condensatenet import CondensateNetPipeline

from annotation_client.utils import sendProgress


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Batch XY': {
            'type': 'text',
            'displayOrder': 0
        },
        'Batch Z': {
            'type': 'text',
            'displayOrder': 1
        },
        'Batch Time': {
            'type': 'text',
            'displayOrder': 2
        },
        'Probability Threshold': {
            'type': 'number',
            'min': 0.0,
            'max': 1.0,
            'default': 0.15,
            'displayOrder': 3,
        },
        'Min Size': {
            'type': 'number',
            'min': 1,
            'max': 1000,
            'default': 15,
            'displayOrder': 4,
        },
        'Max Size': {
            'type': 'number',
            'min': 10,
            'max': 10000,
            'default': 600,
            'displayOrder': 5,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'displayOrder': 6,
        },
    }
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    # Get parameters from interface
    prob_threshold = float(params['workerInterface']['Probability Threshold'])
    min_size = int(params['workerInterface']['Min Size'])
    max_size = int(params['workerInterface']['Max Size'])
    smoothing = float(params['workerInterface']['Smoothing'])

    batch_xy = params['workerInterface']['Batch XY']
    batch_z = params['workerInterface']['Batch Z']
    batch_time = params['workerInterface']['Batch Time']

    batch_xy = batch_argument_parser.process_range_list(batch_xy, convert_one_to_zero_index=True)
    batch_z = batch_argument_parser.process_range_list(batch_z, convert_one_to_zero_index=True)
    batch_time = batch_argument_parser.process_range_list(batch_time, convert_one_to_zero_index=True)

    tile = params['tile']
    channel = params['channel']
    tags = params['tags']

    if batch_xy is None:
        batch_xy = [tile['XY']]
    if batch_z is None:
        batch_z = [tile['Z']]
    if batch_time is None:
        batch_time = [tile['Time']]

    batches = list(product(batch_xy, batch_z, batch_time))
    total_batches = len(batches)

    # Load CondensateNet model from local directory (baked into Docker image)
    sendProgress(0, "Loading model", "Initializing CondensateNet...")
    pipeline = CondensateNetPipeline.from_local(
        "/models/condensatenet",
        prob_threshold=prob_threshold,
        min_size=min_size,
        max_size=max_size
    )

    new_annotations = []

    for i, batch in enumerate(batches):
        XY, Z, Time = batch

        # Get the image for the current channel
        frame_idx = tileClient.coordinatesToFrameIndex(XY, Z, Time, channel)
        image = np.squeeze(tileClient.getRegion(datasetId, frame=frame_idx))

        # Pad image to dimensions divisible by 32 for FPN compatibility
        original_shape = image.shape[:2]
        pad_h = (32 - original_shape[0] % 32) % 32
        pad_w = (32 - original_shape[1] % 32) % 32
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Run segmentation
        instances = pipeline.segment(image)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            instances = instances[:original_shape[0], :original_shape[1]]
        num_instances = instances.max()
        print(f"Frame (XY={XY}, Z={Z}, T={Time}): found {num_instances} condensates")

        # Convert instances to polygons
        temp_polygons = []
        for inst_id in range(1, num_instances + 1):
            mask = (instances == inst_id).astype(np.uint8)
            contours = find_contours(mask, 0.5)

            if len(contours) > 0:
                # Take the largest contour
                contour = max(contours, key=len)
                if len(contour) >= 3:
                    polygon = Polygon(contour).simplify(smoothing, preserve_topology=True)
                    if polygon.is_valid and not polygon.is_empty:
                        temp_polygons.append(polygon)

        # Convert polygons to NimbusImage annotations
        temp_annotations = annotation_tools.polygons_to_annotations(
            temp_polygons,
            datasetId,
            XY=XY,
            Time=Time,
            Z=Z,
            tags=tags,
            channel=channel
        )
        new_annotations.extend(temp_annotations)

        sendProgress(
            (i + 1) / total_batches * 0.9,  # Reserve 10% for upload
            "Segmenting condensates", 
            f"{i + 1} of {total_batches} frames processed ({len(temp_annotations)} condensates)"
        )

    sendProgress(0.9, "Uploading annotations", f"Sending {len(new_annotations)} annotations to server")
    annotationClient.createMultipleAnnotations(new_annotations)
    sendProgress(1.0, "Complete", f"Created {len(new_annotations)} condensate annotations")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CondensateNet Segmentation Worker')

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
