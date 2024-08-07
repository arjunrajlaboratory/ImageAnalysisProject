import argparse
import json
import sys
import timeit

from functools import partial

import annotation_client.workers as workers
import annotation_client.tiles as tiles
import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress

import numpy as np
from stardist.models import StarDist2D
from shapely.geometry import Polygon
from rasterio import features
import rasterio.transform

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Model': {
            'type': 'select',
            'items': ['2D_versatile_fluo', '2D_versatile_he'],
            'default': '2D_versatile_fluo',
            'displayOrder': 1
        },
        'Channel': {
            'type': 'channel',
            'default': 0,
            'required': True,
            'displayOrder': 2
        },
        'Probability Threshold': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.5,
            'displayOrder': 3
        },
        'NMS Threshold': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.4,
            'displayOrder': 4
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'displayOrder': 5,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 10,
            'default': 1,
            'displayOrder': 6,
        },
    }
    client.setWorkerImageInterface(image, interface)

def run_stardist(image, model, prob_thresh, nms_thresh):
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    
    # Normalize the image to 0-1 range
    image_normalized = image.astype(np.float32) / image.max()
    
    print(f"Normalized image range: {image_normalized.min()} to {image_normalized.max()}")
    
    labels, details = model.predict_instances(image_normalized, prob_thresh=prob_thresh, nms_thresh=nms_thresh)

    print(f"StarDist prediction: {labels.shape} labels, {len(details['coord'])} objects detected")

    return labels

def labels_to_polygons(labels):
    polygons = []
    # Create a default transform
    default_transform = rasterio.transform.from_bounds(0, labels.shape[0], labels.shape[1], 0, labels.shape[1], labels.shape[0])
    
    try:
        for shape, value in features.shapes(labels.astype(np.int32), mask=(labels > 0), transform=default_transform):
            polygon = Polygon(shape['coordinates'][0])
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)
    except Exception as e:
        print(f"Error in labels_to_polygons: {e}")
        print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"Unique values in labels: {np.unique(labels)}")
        # If an error occurs, try without specifying a transform
        for shape, value in features.shapes(labels.astype(np.int32), mask=(labels > 0)):
            polygon = Polygon(shape['coordinates'][0])
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)
    
    return polygons

def compute(datasetId, apiUrl, token, params):
    start_time = timeit.default_timer()

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    datasetClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)
    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)

    model_name = params['workerInterface']['Model']
    channel = params['workerInterface']['Channel']
    prob_thresh = float(params['workerInterface']['Probability Threshold'])
    nms_thresh = float(params['workerInterface']['NMS Threshold'])
    padding = float(params['workerInterface']['Padding'])
    smoothing = float(params['workerInterface']['Smoothing'])

    # Load the StarDist model
    model = StarDist2D.from_pretrained(model_name)

    # Get the image data
    tile = params['tile']
    frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
    image = datasetClient.getRegion(datasetId, frame=frame).squeeze()

    if image is None:
        print("Failed to load image")
        return

    # Run StarDist
    sendProgress(0.3, 'Running StarDist', 'Processing image with StarDist model')
    labels = run_stardist(image, model, prob_thresh, nms_thresh)

    # Convert labels to polygons
    sendProgress(0.6, 'Converting to polygons', 'Converting label image to polygons')
    polygons = labels_to_polygons(labels)

    # Apply padding and smoothing
    if padding != 0 or smoothing > 0:
        processed_polygons = []
        for polygon in polygons:
            if padding != 0:
                polygon = polygon.buffer(padding)
            if smoothing > 0:
                polygon = polygon.simplify(smoothing)
            processed_polygons.append(polygon)
    else:
        processed_polygons = polygons

    # Prepare annotations
    sendProgress(0.8, 'Preparing annotations', 'Converting polygons to annotations')
    out_annotations = []
    for polygon in processed_polygons:
        annotation = {
            "tags": params.get('tags', []),
            "shape": "polygon",
            "channel": channel,
            "location": {
                "XY": tile['XY'],
                "Z": tile['Z'],
                "Time": tile['Time']
            },
            "datasetId": datasetId,
            "coordinates": [{"x": float(x), "y": float(y)} for x, y in polygon.exterior.coords],
        }
        out_annotations.append(annotation)

    # Upload annotations
    sendProgress(0.9, 'Uploading annotations', f'Uploading {len(out_annotations)} annotations')
    annotationClient.createMultipleAnnotations(out_annotations)

    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the code in: {execution_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute StarDist segmentation on images')

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