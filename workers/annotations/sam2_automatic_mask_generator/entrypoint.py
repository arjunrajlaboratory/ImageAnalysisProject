import argparse
import json
import sys

from functools import partial
from itertools import product

import annotation_client.annotations as annotations_client
import annotation_client.workers as workers
import annotation_client.tiles as tiles

import annotation_utilities.annotation_tools as annotation_tools
import annotation_utilities.batch_argument_parser as batch_argument_parser

import numpy as np  # library for array manipulation
from shapely.geometry import Polygon
from skimage.measure import find_contours
from shapely.geometry import Polygon

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

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
        'Model': {
            'type': 'select',
            'items': ['sam2_hiera_large.pt'],
            'default': 'sam2_hiera_large.pt',
            'displayOrder': 3
        },
        'Use all channels': {
            'type': 'checkbox',
            'default': True,
            'required': False,
            'displayOrder': 4
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'displayOrder': 5,
        },
        'Points per side': {
            'type': 'number',
            'min': 16,
            'max': 128,
            'default': 32,
            'displayOrder': 6,
        },
    }
    client.setWorkerImageInterface(image, interface)

def compute(datasetId, apiUrl, token, params):
    annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    model = params['workerInterface']['Model']
    use_all_channels = params['workerInterface']['Use all channels']
    smoothing = float(params['workerInterface']['Smoothing'])
    points_per_side = int(params['workerInterface']['Points per side'])
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

    # SAM2 model setup
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_path = f"/{model}"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint_path, device='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2_model, points_per_side=points_per_side)

    new_annotations = []

    for i, batch in enumerate(batches):
        XY, Z, Time = batch
        
        images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, Z, Time)
        layers = annotation_tools.get_layers(tileClient.client, datasetId)

        merged_image = annotation_tools.process_and_merge_channels(images, layers)
        image = merged_image.astype(np.float32)

        masks = mask_generator.generate(image)
        print("num masks", len(masks))

        temp_polygons = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            contours = find_contours(mask, 0.5)
            polygon = Polygon(contours[0]).simplify(smoothing, preserve_topology=True)
            if polygon.is_valid and not polygon.is_empty:
                temp_polygons.append(polygon)

        temp_annotations = annotation_tools.polygons_to_annotations(temp_polygons, datasetId, XY=XY, Time=Time, Z=Z, tags=tags, channel=channel)
        new_annotations.extend(temp_annotations)

        sendProgress((i + 1) / total_batches, "Generating masks", f"{i + 1} of {total_batches} frames processed")

    sendProgress(0.9, "Uploading annotations", f"Sending {len(new_annotations)} annotations to server")
    annotationClient.createMultipleAnnotations(new_annotations)

if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='SAM Automatic Mask Generator')

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

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
