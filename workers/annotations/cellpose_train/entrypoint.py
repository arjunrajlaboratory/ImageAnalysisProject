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
from annotation_client.utils import sendProgress, sendError
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
            'displayOrder': 5
        },
        'Output Model Name': {
            'type': 'text',
            'tooltip': 'The name of the retrained model (to be saved to your .cellpose/models folder).',
            'displayOrder': 6
        },
        'Nuclei Channel': {
            'type': 'channel',
            # 'default': -1,  # -1 means no channel
            'required': False,
            'displayOrder': 7
        },
        'Cytoplasm Channel': {
            'type': 'channel',
            # 'default': -1,  # -1 means no channel
            'required': False,
            'tooltip': 'If you are segmenting nuclei, put your nucleus channel in both the Nuclei Channel and Cytoplasm Channel fields.',
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
    model = workerInterface['Base Model']
    output_model_name = workerInterface['Output Model Name']
    nuclei_channel = workerInterface.get('Nuclei Channel', None)
    cytoplasm_channel = workerInterface.get('Cytoplasm Channel', None)
    training_tag = workerInterface.get('Training Tag', None)
    training_regions = workerInterface.get('Training Region', None)
    learning_rate = float(workerInterface['Learning Rate'])
    epochs = int(workerInterface['Epochs'])
    weight_decay = float(workerInterface['Weight Decay'])

    if training_tag is None:
        # TODO: Add an error message here.
        raise ValueError("No training tag selected.")
    if training_regions is None:
        # TODO: Add an warning message here.
        print("No training regions selected. Training will be performed on entire image that the annotations are in.")

    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    if model not in BASE_MODELS:
        girder_utils.download_girder_model(client.client, model)

    # Print the contents of the models directory
    print(f"Models directory contents: {list(MODELS_DIR.glob('*'))}")

    stack_channels = []
    if model in ['cyto', 'cyto2', 'cyto3']:
        if (cytoplasm_channel is not None) and (cytoplasm_channel > -1):
            stack_channels.append(cytoplasm_channel)
    if (nuclei_channel is not None) and (nuclei_channel > -1):
        stack_channels.append(nuclei_channel)
    if len(stack_channels) == 2:
        channels = (0, 1)
    elif len(stack_channels) == 1:
        channels = (0, 0)
    else:
        # TODO: Add an error message here.
        raise ValueError("No cytoplasmic or nuclei channels selected.")

    # if model in BASE_MODELS:
    #     cellpose = cellpose_segmentation(model_parameters={'gpu': True, 'model_type': model}, eval_parameters={
    #                                      'diameter': diameter, 'channels': channels}, output_format='polygons')
    # else:
    #     # Get the full path to the model
    #     model_path = str(MODELS_DIR / model)
    #     cellpose = cellpose_segmentation(model_parameters={'gpu': True, 'pretrained_model': model_path}, eval_parameters={
    #                                      'diameter': diameter, 'channels': channels}, output_format='polygons')
    # f_process = partial(run_model, cellpose=cellpose, tile_size=tile_size,
    #                     tile_overlap=tile_overlap, padding=padding, smoothing=smoothing)

    # worker.process(f_process, f_annotation='polygon',
    #                stack_channels=stack_channels, progress_text='Running Cellpose')

    blobAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=1000000, shape='polygon')

    trainingAnnotationList = annotation_tools.get_annotations_with_tags(
        blobAnnotationList, training_tag, exclusive=False)
    regionAnnotationList = annotation_tools.get_annotations_with_tags(
        blobAnnotationList, training_regions, exclusive=False)

    # TODO: If these are empty, given an error message.

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

    # Loop through each location and load the image for the training.
    for location_key, training_annotations in grouped_training_annotations.items():
        time, z, xy = location_key
        # TODO: Handle all channel cases in some sort of general way.
        frame = tileClient.coordinatesToFrameIndex(xy, z, time, nuclei_channel)
        nucleus_image = tileClient.getRegion(datasetId, frame=frame)
        nucleus_image = nucleus_image.squeeze()

        frame = tileClient.coordinatesToFrameIndex(
            xy, z, time, cytoplasm_channel)
        cytoplasm_image = tileClient.getRegion(datasetId, frame=frame)
        cytoplasm_image = cytoplasm_image.squeeze()

        label_image = np.zeros(nucleus_image.shape[:2], dtype=np.uint16)
        for i, current_annotation in enumerate(training_annotations):
            polygon = np.array([list(coordinate.values())[1::-1]
                               for coordinate in current_annotation['coordinates']])
            mask = draw.polygon2mask(nucleus_image.shape, polygon)
            label_image[mask] = i + 1

        # TODO: Handle the case where there are no region annotations.
        region_annotations = grouped_region_annotations[location_key]
        for region_annotation in region_annotations:
            region_polygon = np.array([list(coordinate.values())[1::-1]
                                      for coordinate in region_annotation['coordinates']])
            region_polygon = Polygon([(coordinate['x'], coordinate['y'])
                                     for coordinate in region_annotation['coordinates']])

            # Use shapely to get the bounding box
            min_x, min_y, max_x, max_y = region_polygon.bounds

            # Crop the all images to the bounding box
            nucleus_image = nucleus_image[int(
                min_y):int(max_y), int(min_x):int(max_x)]
            cytoplasm_image = cytoplasm_image[int(
                min_y):int(max_y), int(min_x):int(max_x)]
            label_image = label_image[int(min_y):int(
                max_y), int(min_x):int(max_x)]

            # Assemble the nucleus and cytoplasm images into a single RGB image.
            training_image = np.stack(
                [nucleus_image, cytoplasm_image, np.zeros_like(nucleus_image)], axis=-1)

            # Add to the list of training images and label images.
            training_images.append(training_image)
            label_images.append(label_image)

    using_gpu = core.use_gpu()
    print(f"Using GPU: {using_gpu}")
    # TODO: Allow different models.
    model = models.CellposeModel(model_type="cyto3")

    print(f"Training with {len(training_images)} images.")

    model_path, train_losses, test_losses = train.train_seg(model.net,
                                                            train_data=training_images, train_labels=label_images,
                                                            channels=[1, 2], normalize=True,
                                                            weight_decay=weight_decay, SGD=True, learning_rate=learning_rate,
                                                            n_epochs=epochs, model_name=MODELS_DIR / output_model_name)

    # Upload the trained model to Girder
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
