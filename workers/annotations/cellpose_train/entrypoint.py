import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from functools import partial
from itertools import product
from skimage import draw
import numpy as np

from shapely.geometry import Polygon, box

from cellpose import io, models, train, core

import annotation_client.workers as workers
import annotation_client.tiles as tiles
import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress, sendError, sendWarning
import annotation_utilities.annotation_tools as annotation_tools

import girder_utils
from girder_utils import CELLPOSE_DIR, MODELS_DIR

BASE_MODELS = ['cyto', 'cyto2', 'cyto3', 'nuclei']


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # models = sorted(path.stem for path in MODELS_DIR.glob('*'))
    models = BASE_MODELS
    girder_models = [model['name']
                     for model in girder_utils.list_girder_models(client.client)[0]]
    models = sorted(list(set(models + girder_models)))

    # Available types: number, text, tags, layer
    interface = {
        'Cellpose train': {
            'type': 'notes',
            'value': 'This tool trains a Cellpose model using user-corrected annotations.',
            'displayOrder': 0,
        },
        'Base Model': {
            'type': 'select',
            'items': models,
            'default': 'cyto3',
            'tooltip': 'This model is the one used as a base for training.\n'
                       'cyto3 is the most accurate for cells, whereas nuclei is best for finding nuclei.\n'
                       'You will need to select a nuclei and cytoplasm channel in both cases.\n'
                       'If you select nuclei, put the nucleus channel in both the Nuclei Channel and Cytoplasm Channel fields.',
            'noCache': True,
            'displayOrder': 4
        },
        'Nuclear Model?': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'If you are training a nuclear model, check this box.',
            'displayOrder': 5
        },
        'Output Model Name': {
            'type': 'text',
            'tooltip': 'The name of the retrained model (to be saved to your .cellpose/models folder).',
            'displayOrder': 6
        },
        'Primary Channel': {
            'type': 'channel',
            # 'default': -1,  # -1 means no channel
            'tooltip': 'The channel to use for the primary segmentation.\n'
                       'If you are segmenting cytoplasm, put your cytoplasm channel here.\n'
                       'If you are segmenting nuclei, put your nucleus channel here.',
            'required': False,
            'displayOrder': 7
        },
        'Secondary Channel': {
            'type': 'channel',
            'default': -1,  # -1 means no channel
            'required': False,
            'tooltip': 'The channel to use for the secondary segmentation.\n'
                       'If you are segmenting cytoplasm, put your nuclei channel here.\n'
                       'If you are segmenting nuclei, leave this blank (it will be ignored if filled).',
            'displayOrder': 8
        },
        'Training Tag': {
            'type': 'tags',
            'tooltip': 'Train the model on objects that have this tag.',
            'displayOrder': 9
        },
        'Training Region': {
            'type': 'tags',
            'tooltip': 'These objects define the regions that the training will be performed on.\n'
                       'If you do not select any objects, the training will be performed on all objects in the image.\n'
                       'You can and probably should select multiple regions.',
            'displayOrder': 10
        },
        'Learning Rate': {
            'type': 'number',
            'min': 0.0001,
            'max': 0.5,
            'default': 0.01,
            'tooltip': 'The learning rate for the training. 0.01 is a good starting point.',
            'displayOrder': 11
        },
        'Epochs': {
            'type': 'number',
            'min': 100,
            'max': 2000,
            'default': 1000,
            'tooltip': 'The number of epochs to train the model for. 1000 is a good starting point.',
            'displayOrder': 12
        },
        'Weight Decay': {
            'type': 'number',
            'min': 0,
            'max': 0.01,
            'default': 0.0001,
            'tooltip': 'The weight decay for the training.',
            'displayOrder': 13
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


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

    workerInterface = params['workerInterface']

    # Get the model and diameter from interface values
    base_model = workerInterface['Base Model']
    output_model_name = workerInterface['Output Model Name']
    nuclear_model = workerInterface['Nuclear Model?']
    primary_channel = workerInterface.get('Primary Channel', None)
    secondary_channel = workerInterface.get('Secondary Channel', None)
    training_tag = workerInterface.get('Training Tag', None)
    training_regions = workerInterface.get('Training Region', None)
    learning_rate = float(workerInterface['Learning Rate'])
    epochs = int(workerInterface['Epochs'])
    weight_decay = float(workerInterface['Weight Decay'])

    print(f"Training tag: {training_tag}")
    print(f"Training regions: {training_regions}")

    if training_tag is None or len(training_tag) == 0:
        sendError("No training tag selected.",
                  info="Choose a tag for training annotations.")
        raise ValueError("No training tag selected.")
    if training_regions is None or len(training_regions) == 0:
        sendWarning("No training regions selected.",
                    info="Training will be performed on entire image that the annotations are in.")
        print("No training regions selected. Training will be performed on entire image that the annotations are in.")

    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    if base_model not in BASE_MODELS:
        girder_utils.download_girder_model(client.client, base_model)

    # Print the contents of the models directory
    print(f"Models directory contents: {list(MODELS_DIR.glob('*'))}")

    # Need to have a primary channel to do anything.
    if primary_channel is None or primary_channel == -1:
        sendError("No primary channel selected for nuclear model training.",
                  info="Please select a primary channel for the nuclei.")
        raise ValueError(
            "No primary channel selected for nuclear model training.")

    if nuclear_model:
        channels = [1, 0]
    else:
        if secondary_channel is None or secondary_channel == -1:
            sendWarning("No secondary (nucleus) channel selected for cytoplasm model training.",
                        info="Proceeding using primary channel only.")
            channels = [1, 0]
        else:
            channels = [1, 2]

    # Initial loading phase
    sendProgress(0.1, "Loading annotations",
                 "Retrieving annotations from server")
    blobAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=1000000, shape='polygon')
    rectangleAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=1000000, shape='rectangle')
    # Add the rectangle annotations to the blob annotations
    blobAnnotationList.extend(rectangleAnnotationList)

    trainingAnnotationList = annotation_tools.get_annotations_with_tags(
        blobAnnotationList, training_tag, exclusive=False)
    if training_regions is None or len(training_regions) == 0:
        regionAnnotationList = []
    else:
        regionAnnotationList = annotation_tools.get_annotations_with_tags(
            blobAnnotationList, training_regions, exclusive=False)
        print(f"Training on {len(regionAnnotationList)} region annotations.")

    if len(trainingAnnotationList) == 0:
        sendError("No training annotations found.",
                  info="No annotations with the training tag were found.")
        raise ValueError("No training annotations found.")
    if len(regionAnnotationList) == 0 and len(training_regions) > 0:
        sendWarning("No region annotations found.",
                    info="No annotations with the training region tag were found.")
        print("No region annotations found. Training will be performed on entire image that the annotations are in.")

    sendProgress(0.2, "Processing annotations",
                 "Grouping annotations by location")
    # Group the training annotations by location so that we can batch the image loading.
    grouped_training_annotations = defaultdict(list)
    for current_annotation in trainingAnnotationList:
        location_key = (current_annotation['location']['Time'],
                        current_annotation['location']['Z'], current_annotation['location']['XY'])
        grouped_training_annotations[location_key].append(current_annotation)

    # Group the region annotations by location so that we can batch the image loading.
    grouped_region_annotations = defaultdict(list)
    for current_annotation in regionAnnotationList:
        location_key = (current_annotation['location']['Time'],
                        current_annotation['location']['Z'], current_annotation['location']['XY'])
        grouped_region_annotations[location_key].append(current_annotation)

    training_images = []
    label_images = []

    sendProgress(0.3, "Loading training data", "Loading training images")
    # Loop through each location and load the image for the training.
    for location_key, training_annotations in grouped_training_annotations.items():
        time, z, xy = location_key
        # TODO: Handle all channel cases in some sort of general way.
        # TODO: If there are no region annotations and there is no region tag, then skip the image loading for efficiency.
        frame = tileClient.coordinatesToFrameIndex(
            xy, z, time, primary_channel)
        primary_image = tileClient.getRegion(datasetId, frame=frame)
        primary_image = primary_image.squeeze()

        if secondary_channel is not None and secondary_channel != -1:
            frame = tileClient.coordinatesToFrameIndex(
                xy, z, time, secondary_channel)
            secondary_image = tileClient.getRegion(datasetId, frame=frame)
            secondary_image = secondary_image.squeeze()
        else:
            secondary_image = np.zeros_like(primary_image)

        label_image = np.zeros(primary_image.shape[:2], dtype=np.uint16)
        for i, current_annotation in enumerate(training_annotations):
            polygon = np.array([list(coordinate.values())[1::-1]
                               for coordinate in current_annotation['coordinates']])
            mask = draw.polygon2mask(primary_image.shape, polygon)
            label_image[mask] = i + 1

        if training_regions is None or len(training_regions) == 0:
            training_image = np.stack(
                [primary_image, secondary_image, np.zeros_like(primary_image)], axis=-1)
            training_images.append(training_image)
            label_images.append(label_image)
        else:
            region_annotations = grouped_region_annotations[location_key]
            for region_annotation in region_annotations:
                region_polygon = Polygon([(coordinate['x'], coordinate['y'])
                                         for coordinate in region_annotation['coordinates']])

                # Use shapely to get the bounding box
                min_x, min_y, max_x, max_y = region_polygon.bounds

                # Crop the all images to the bounding box
                primary_image_crop = primary_image[int(
                    min_y):int(max_y), int(min_x):int(max_x)]
                secondary_image_crop = secondary_image[int(
                    min_y):int(max_y), int(min_x):int(max_x)]
                label_image_crop = label_image[int(min_y):int(
                    max_y), int(min_x):int(max_x)]

                # Assemble the nucleus and cytoplasm images into a single RGB image.
                training_image_crop = np.stack(
                    [primary_image_crop, secondary_image_crop, np.zeros_like(primary_image_crop)], axis=-1)

                # Add to the list of training images and label images.
                training_images.append(training_image_crop)
                label_images.append(label_image_crop)

    using_gpu = core.use_gpu()
    print(f"Using GPU: {using_gpu}")
    # TODO: Allow different models. Not sure if this will work for pre-trained models as a base, might need the whole path.
    model = models.CellposeModel(model_type=base_model)

    print(f"Training with {len(training_images)} images.")
    sendProgress(0.4, "Training model",
                 f"Training with {len(training_images)} images, be patient...")

    model_path, train_losses, test_losses = train.train_seg(model.net,
                                                            train_data=training_images, train_labels=label_images,
                                                            channels=channels, normalize=True,
                                                            weight_decay=weight_decay, SGD=True, learning_rate=learning_rate,
                                                            n_epochs=epochs, model_name=MODELS_DIR / output_model_name)

    # Upload the trained model to Girder
    sendProgress(0.95, "Saving model", f"Uploading model {output_model_name}")
    girder_utils.upload_girder_model(client.client, output_model_name)


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

    parser.add_argument('--datasetId', type=str,
                        required=False, action='store')
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
