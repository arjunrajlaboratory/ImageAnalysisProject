import argparse
import json
import sys

from functools import partial
from itertools import product

import annotation_client.annotations as annotations_client
import annotation_client.workers as workers
import annotation_client.tiles as tiles

import annotation_utilities.annotation_tools as annotation_tools

import numpy as np  # library for array manipulation
from shapely.geometry import Polygon
from skimage.measure import find_contours
from shapely.geometry import Polygon

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Model': {
            'type': 'select',
            'items': ['sam2_hiera_large.pt'],
            'default': 'sam2_hiera_large.pt',
            'displayOrder': 0
        },
        'Tag of objects to propagate': {
            'type': 'tags',
            'displayOrder': 1
        },
        'Use all channels': {
            'type': 'checkbox',
            'default': True,
            'required': False,
            'displayOrder': 2
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'displayOrder': 3,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'displayOrder': 4,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)

# Function to auto-scale image
def auto_scale_image(image):
    image = image.squeeze()  # Remove single-dimensional entries
    vmin, vmax = np.percentile(image, (1, 99.5))  # Use 2nd and 98th percentiles for contrast
    return (image - vmin) / (vmax - vmin)

def segment_image(image, checkpoint_path="/sam2_hiera_large.pt"):
    # image is assumed to already be an numpy array of a color image

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # model_cfg = "/segment-anything-2/sam2_configs/sam2_hiera_l.yaml" # This will need to be updated based on model chosen
    model_cfg = "sam2_hiera_l.yaml" # This will need to be updated based on model chosen
    sam2_model = build_sam2(model_cfg, checkpoint_path, device='cuda', apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2,points_per_side=128)

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
    propagate_tags = params['workerInterface']['Tag of objects to propagate']

    tile = params['tile']
    channel = params['channel']
    tags = params['tags']

    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    print("Length of annotations:", len(annotationList))
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, propagate_tags, exclusive=False)
    print("Length of annotations:", len(annotationList))

    print("Tile:", tile)
    print("Channel:", channel)
    print("Tags:", tags)

    print("Model:", model)
    print("Use all channels:", use_all_channels)
    print("Padding:", padding)
    print("Smoothing:", smoothing)

    XY = tile['XY']
    Z = tile['Z']
    Time = tile['Time']

    checkpoint_path="/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"  # This will need to be updated based on model chosen
    sam2_model = build_sam2(model_cfg, checkpoint_path, device='cpu', apply_postprocessing=False)  # device='cuda' for GPU
    predictor = SAM2ImagePredictor(sam2_model)

    new_annotations = []
    Time = 0  # Let's start at the earliest time

    for Time in range(0, 2):
        # Search the annotationList for annotations with the current Time.
        # todo: We would want to go over all the XY and Z as well, but need to load images for each one.
        sliced_annotations = annotation_tools.filter_elements_T_XY_Z(annotationList, Time, XY, Z)
        sliced_new_annotations = annotation_tools.filter_elements_T_XY_Z(new_annotations, Time, XY, Z)

        if len(sliced_annotations) == 0 and len(sliced_new_annotations) == 0: # If we didn't find any annotations to propagate, skip
            continue

        # Get the images for the next Time and XY and Z
        images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, Z, Time+1)
        print("Length of images:", len(images))
        layers = annotation_tools.get_layers(tileClient.client, datasetId)
        print("Layers:", layers)

        merged_image = annotation_tools.process_and_merge_channels(images, layers)
        image = merged_image.astype(np.float32)

        predictor.set_image(image)

        polygons = [annotation_tools.annotations_to_polygons(sliced_annotations), annotation_tools.annotations_to_polygons(sliced_new_annotations)]
        boxes = [polygon.bounds for polygon in polygons]
        input_boxes = np.array(boxes)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False,
        )

        print(f"Number of masks: {len(masks)}")
        
        # Find contours in the mask
        temp_polygons = []
        for mask in masks:
            contour = find_contours(mask, 0.5)
            polygon = Polygon(contour).simplify(smoothing, preserve_topology=True)
            temp_polygons.append(polygon)

        # def polygons_to_annotations(polygons, XY=0, Time=0, Z=0, tags=None, channel=0):

        temp_annotations = annotation_tools.polygons_to_annotations(temp_polygons, XY=XY, Time=Time+1, Z=Z, tags=tags, channel=channel)
        new_annotations.extend(temp_annotations)

    annotationClient.createMultipleAnnotations(new_annotations)

    # images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, Z, Time)
    # print("Length of images:", len(images))
    # layers = annotation_tools.get_layers(tileClient.client, datasetId)
    # print("Layers:", layers)

    # merged_image = annotation_tools.process_and_merge_channels(images, layers)
    # print("Merged image shape:", merged_image.shape)


    # channel_load = channel
    # frame = tileClient.coordinatesToFrameIndex(XY, Z, Time, channel_load)
    # image_phase = tileClient.getRegion(datasetId, frame=frame)
    # channel_load = 1
    # frame = tileClient.coordinatesToFrameIndex(XY, Z, Time, channel_load)
    # image_fluor = tileClient.getRegion(datasetId, frame=frame)

    # # Auto-scale images
    # image_phase_scaled = auto_scale_image(image_phase)
    # image_fluor_scaled = auto_scale_image(image_fluor)

    # # Create RGB image
    # rgb_image = np.zeros((*image_phase_scaled.shape, 3))
    # rgb_image[:,:,0] = image_phase_scaled  # Red channel
    # rgb_image[:,:,1] = np.maximum(image_phase_scaled, image_fluor_scaled)  # Green channel
    # rgb_image[:,:,2] = image_phase_scaled  # Blue channel

    # # convert rgb_image to uint8
    # rgb_image = (rgb_image * 255).astype(np.uint8)

    # masks = segment_image(rgb_image)

    # print(len(masks), "masks generated")

    # annotations = []

    # for i, mask_data in enumerate(masks):
    #     mask = mask_data['segmentation']
        
    #     # Find contours in the mask
    #     contours = find_contours(mask, 0.5)
        
    #     for contour in contours:
    #         # Simplify the contour to reduce the number of points
    #         polygon = Polygon(contour).simplify(smoothing, preserve_topology=True)
            
    #         if polygon.is_valid and not polygon.is_empty:
    #             # Convert the polygon coordinates to the required format
    #             coordinates = [{"x": float(y), "y": float(x)} for x, y in polygon.exterior.coords]
                
    #             # Create the annotation
    #             annotation = {
    #                 "tags": tags,
    #                 "shape": "polygon",
    #                 "channel": channel,
    #                 "location": {
    #                     "XY": xy,  # You may need to adjust this based on your tile information
    #                     "Z": z,        # Adjust as needed
    #                     "Time": time      # Adjust as needed
    #                 },
    #                 "datasetId": datasetId,
    #                 "coordinates": coordinates
    #             }
                
    #             annotations.append(annotation)

    # # Upload the annotations
    # annotationClient.createMultipleAnnotations(annotations)


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
