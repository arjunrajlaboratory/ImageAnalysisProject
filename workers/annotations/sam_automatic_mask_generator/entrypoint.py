import argparse
import json
import sys

from functools import partial
from itertools import product

import annotation_client.annotations as annotations_client
import annotation_client.workers as workers
import annotation_client.tiles as tiles

import numpy as np  # library for array manipulation
from shapely.geometry import Polygon
from skimage.measure import find_contours
from shapely.geometry import Polygon

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Model': {
            'type': 'select',
            'items': ['sam_vit_h_4b8939'],
            'default': 'sam_vit_h_4b8939',
            'displayOrder': 0
        },
        'Use all channels': {
            'type': 'checkbox',
            'default': True,
            'required': False,
            'displayOrder': 1
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'displayOrder': 2,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'displayOrder': 3,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)

# Function to auto-scale image
def auto_scale_image(image):
    image = image.squeeze()  # Remove single-dimensional entries
    vmin, vmax = np.percentile(image, (1, 99.5))  # Use 2nd and 98th percentiles for contrast
    return (image - vmin) / (vmax - vmin)


def segment_image(image, model_type="vit_h", checkpoint_path="./sam_vit_h_4b8939.pth"):
    # image is assumed to already be an numpy array of a color image
        
    # Set up the model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    if torch.cuda.is_available:
        print("Using GPU")
    else:
        print("Using CPU")

    sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Create the mask generator
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=64
    )

    # Generate the masks
    masks = mask_generator.generate(image)

    print(f"Number of masks: {len(masks)}")
    
    return masks

def compute(datasetId, apiUrl, token, params):
    """
    params (could change):
        configurationId,
        datasetId,
        description: tool description,
        type: tool type,
        id: tool id,
        name: tool name,
        image: docker image,
        channel: annotation channel,
        assignment: annotation assignment ({XY, Z, Time}),
        tags: annotation tags (list of strings),
        tile: tile position (TODO: roi) ({XY, Z, Time}),
        connectTo: how new annotations should be connected
    """

    annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    model = params['workerInterface']['Model']
    use_all_channels = params['workerInterface']['Use all channels']
    padding = float(params['workerInterface']['Padding'])
    smoothing = float(params['workerInterface']['Smoothing'])

    tile = params['tile']

    channel = params['channel']
    tags = params['tags']

    print("TESTING SAM AUTOMATIC MASK GENERATOR")

    print("Tile:", tile)
    print("Channel:", channel)
    print("Tags:", tags)

    print("Model:", model)
    print("Use all channels:", use_all_channels)
    print("Padding:", padding)
    print("Smoothing:", smoothing)

    xy = tile['XY']
    z = tile['Z']
    time = tile['Time']
    
    channel_load = channel
    frame = tileClient.coordinatesToFrameIndex(xy, z, time, channel_load)
    image_phase = tileClient.getRegion(datasetId, frame=frame)
    channel_load = 1
    frame = tileClient.coordinatesToFrameIndex(xy, z, time, channel_load)
    image_fluor = tileClient.getRegion(datasetId, frame=frame)

    # Auto-scale images
    image_phase_scaled = auto_scale_image(image_phase)
    image_fluor_scaled = auto_scale_image(image_fluor)

    # Create RGB image
    rgb_image = np.zeros((*image_phase_scaled.shape, 3))
    rgb_image[:,:,0] = image_phase_scaled  # Red channel
    rgb_image[:,:,1] = np.maximum(image_phase_scaled, image_fluor_scaled)  # Green channel
    rgb_image[:,:,2] = image_phase_scaled  # Blue channel

    masks = segment_image(rgb_image)

    print(len(masks), "masks generated")

    annotations = []

    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        
        # Find contours in the mask
        contours = find_contours(mask, 0.5)
        
        for contour in contours:
            # Simplify the contour to reduce the number of points
            polygon = Polygon(contour).simplify(smoothing, preserve_topology=True)
            
            if polygon.is_valid and not polygon.is_empty:
                # Convert the polygon coordinates to the required format
                coordinates = [{"x": float(y), "y": float(x)} for x, y in polygon.exterior.coords]
                
                # Create the annotation
                annotation = {
                    "tags": tags,
                    "shape": "polygon",
                    "channel": channel,
                    "location": {
                        "XY": xy,  # You may need to adjust this based on your tile information
                        "Z": z,        # Adjust as needed
                        "Time": time      # Adjust as needed
                    },
                    "datasetId": datasetId,
                    "coordinates": coordinates
                }
                
                annotations.append(annotation)

    # Upload the annotations
    annotationClient.createMultipleAnnotations(annotations)


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
