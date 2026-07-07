import argparse
from collections import defaultdict
import json
import sys

from skimage import draw
import numpy as np

from shapely.geometry import Polygon

import annotation_client.workers as workers
import annotation_client.tiles as tiles
import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress, sendError, sendWarning
import annotation_utilities.annotation_tools as annotation_tools

import girder_utils
from girder_utils import CELLPOSE_DIR, MODELS_DIR

from models_config import (
    BASE_MODELS, BASE_MODEL_CHECKPOINTS, DEFAULT_MODEL, build_model_items)


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    girder_models = [model['name']
                     for model in girder_utils.list_girder_models(client.client)[0]]
    # Reserved base labels win over any custom model of the same name (see
    # build_model_items); this keeps the dropdown consistent with compute()'s routing.
    models = build_model_items(girder_models)

    # Available types: number, text, tags, layer
    interface = {
        'Cellpose-SAM retrain': {
            'type': 'notes',
            'value': 'This tool fine-tunes a Cellpose-SAM model using user-corrected annotations. '
                     'Cellpose-SAM works directly on the channels you provide (up to three), so there '
                     'is no separate cytoplasm/nucleus channel concept — just pick the channels you '
                     'want the model to see. '
                     '<a href="https://docs.nimbusimage.com/documentation/analyzing-image-data-with-objects-connections-and-properties/tools-for-making-objects#cellpose-training" target="_blank">Learn more</a>',
            'displayOrder': 0,
        },
        'Base Model': {
            'type': 'select',
            'items': models,
            'default': DEFAULT_MODEL,
            'tooltip': 'The model used as a starting point for fine-tuning.\n'
                       '"cellpose-sam" starts from the current default checkpoint (cpsam_v2).\n'
                       '"cellpose-sam (legacy cpsam)" starts from the original April 2025 model.\n'
                       'Custom models you have previously trained are also listed here.',
            'noCache': True,
            'displayOrder': 1,
        },
        'Output Model Name': {
            'type': 'text',
            'tooltip': 'The name of the retrained model (saved to your .cellposesam/models folder).\n'
                       'It will appear in the Model dropdown of the Cellpose-SAM worker.',
            'displayOrder': 2,
        },
        'Channel for Slot 1': {
            'type': 'channelCheckboxes',
            'tooltip': "Select source channel(s) for the model's first input slot. If multiple are selected, only the first will be used. This slot is required.",
            'displayOrder': 3,
        },
        'Channel for Slot 2': {
            'type': 'channelCheckboxes',
            'tooltip': "Select source channel(s) for the model's second input slot. If multiple are selected, only the first will be used. (Optional)",
            'displayOrder': 4,
        },
        'Channel for Slot 3': {
            'type': 'channelCheckboxes',
            'tooltip': "Select source channel(s) for the model's third input slot. If multiple are selected, only the first will be used. (Optional)",
            'displayOrder': 5,
        },
        'Training Tag': {
            'type': 'tags',
            'tooltip': 'Train the model on objects that have this tag.',
            'displayOrder': 6,
        },
        'Training Region': {
            'type': 'tags',
            'tooltip': 'These objects define the regions that the training will be performed on.\n'
                       'If you do not select any objects, the training will be performed on all objects in the image.\n'
                       'You can and probably should select multiple regions.',
            'displayOrder': 7,
        },
        'Learning Rate': {
            'type': 'number',
            'min': 0.000001,
            'max': 0.1,
            'default': 0.00001,
            'tooltip': 'The learning rate for fine-tuning. Cellpose-SAM fine-tunes best with a small '
                       'learning rate; 1e-5 (0.00001) is the recommended default.',
            'displayOrder': 8,
        },
        'Epochs': {
            'type': 'number',
            'min': 10,
            'max': 2000,
            'default': 100,
            'tooltip': 'The number of epochs to train the model for. 100 is a good starting point for Cellpose-SAM.',
            'displayOrder': 9,
        },
        'Weight Decay': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.1,
            'tooltip': 'The weight decay (AdamW regularization) for training. 0.1 is the Cellpose-SAM default.',
            'displayOrder': 10,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def get_slot_channels(workerInterface):
    """Resolve the selected input-slot channels into an ordered channel list.

    Mirrors the cellposesam inference worker: Slot 1 is required, Slots 2 and 3
    are optional, and if multiple channels are checked in a slot only the first
    is used (with a warning).
    """
    slot1 = [k for k, v in workerInterface.get('Channel for Slot 1', {}).items() if v]
    slot2 = [k for k, v in workerInterface.get('Channel for Slot 2', {}).items() if v]
    slot3 = [k for k, v in workerInterface.get('Channel for Slot 3', {}).items() if v]

    stack_channels = []

    if not slot1:
        sendError("No channel selected for Slot 1. This is a required field.")
        raise ValueError("No channel selected for Slot 1.")
    if len(slot1) > 1:
        sendWarning(
            f"Multiple channels selected for Slot 1 ({slot1}). Using the first: {slot1[0]}.")
    stack_channels.append(int(slot1[0]))

    if slot2:
        if len(slot2) > 1:
            sendWarning(
                f"Multiple channels selected for Slot 2 ({slot2}). Using the first: {slot2[0]}.")
        stack_channels.append(int(slot2[0]))

    if slot3:
        if len(slot3) > 1:
            sendWarning(
                f"Multiple channels selected for Slot 3 ({slot3}). Using the first: {slot3[0]}.")
        stack_channels.append(int(slot3[0]))

    return stack_channels


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

    # Lazy import: keeps cellpose off the interface/startup path (~seconds). See todo/worker-startup-latency.md
    from cellpose import models, train, core

    workerInterface = params['workerInterface']

    # Get the model and training parameters from interface values
    base_model = workerInterface['Base Model']
    output_model_name = workerInterface['Output Model Name']
    training_tag = workerInterface.get('Training Tag', None)
    training_regions = workerInterface.get('Training Region', None)
    learning_rate = float(workerInterface['Learning Rate'])
    epochs = int(workerInterface['Epochs'])
    weight_decay = float(workerInterface['Weight Decay'])

    print(f"Training tag: {training_tag}")
    print(f"Training regions: {training_regions}")

    if not output_model_name:
        sendError("No output model name provided.",
                  info="Please provide a name for the retrained model.")
        raise ValueError("No output model name provided.")

    if training_tag is None or len(training_tag) == 0:
        sendError("No training tag selected.",
                  info="Choose a tag for training annotations.")
        raise ValueError("No training tag selected.")
    if training_regions is None or len(training_regions) == 0:
        sendWarning("No training regions selected.",
                    info="Training will be performed on entire image that the annotations are in.")
        print("No training regions selected. Training will be performed on entire image that the annotations are in.")

    # Resolve the input-slot channels (Slot 1 required, Slots 2 & 3 optional).
    stack_channels = get_slot_channels(workerInterface)
    print(f"Using channels for Cellpose-SAM input (slots 1, 2, 3): {stack_channels}")

    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    # Resolve the base checkpoint / model path to fine-tune from.
    if base_model in BASE_MODELS:
        # Pass the checkpoint name explicitly so the starting point is pinned to
        # the selected model rather than cellpose's internal default, which can
        # change between versions.
        pretrained_model = BASE_MODEL_CHECKPOINTS[base_model]
    else:
        girder_utils.download_girder_model(client.client, base_model)
        pretrained_model = str(MODELS_DIR / base_model)

    # Print the contents of the models directory
    print(f"Models directory contents: {list(MODELS_DIR.glob('*'))}")

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
    if len(regionAnnotationList) == 0 and training_regions and len(training_regions) > 0:
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

        # Load and stack the selected channels (channels-last) for this location.
        channel_images = []
        for channel in stack_channels:
            frame = tileClient.coordinatesToFrameIndex(xy, z, time, channel)
            channel_image = tileClient.getRegion(datasetId, frame=frame).squeeze()
            channel_images.append(channel_image)
        # Shape: (H, W, num_selected_channels). Cellpose-SAM standardizes to
        # three channels internally (padding with zeros / truncating as needed).
        stacked_image = np.stack(channel_images, axis=-1)

        label_image = np.zeros(stacked_image.shape[:2], dtype=np.uint16)
        for i, current_annotation in enumerate(training_annotations):
            polygon = np.array([list(coordinate.values())[1::-1]
                               for coordinate in current_annotation['coordinates']])
            mask = draw.polygon2mask(label_image.shape, polygon)
            label_image[mask] = i + 1

        if training_regions is None or len(training_regions) == 0:
            training_images.append(stacked_image)
            label_images.append(label_image)
        else:
            region_annotations = grouped_region_annotations[location_key]
            for region_annotation in region_annotations:
                region_polygon = Polygon([(coordinate['x'], coordinate['y'])
                                         for coordinate in region_annotation['coordinates']])

                # Use shapely to get the bounding box
                min_x, min_y, max_x, max_y = region_polygon.bounds

                # Crop both the stacked image and the label mask to the bounding box.
                stacked_image_crop = stacked_image[int(
                    min_y):int(max_y), int(min_x):int(max_x)]
                label_image_crop = label_image[int(min_y):int(
                    max_y), int(min_x):int(max_x)]

                training_images.append(stacked_image_crop)
                label_images.append(label_image_crop)

    using_gpu = core.use_gpu()
    print(f"Using GPU: {using_gpu}")

    # Cellpose-SAM (cellpose >= 4) drops model_type in favor of pretrained_model;
    # passing 'cpsam'/'cpsam_v2' or a custom model path starts fine-tuning from
    # those weights.
    model = models.CellposeModel(gpu=using_gpu, pretrained_model=pretrained_model)

    print(f"Training with {len(training_images)} images.")
    sendProgress(0.4, "Training model",
                 f"Training with {len(training_images)} images, be patient...")

    # Cellpose-SAM training differs from earlier Cellpose versions:
    #  - no `channels` argument (the network ingests up to 3 channels directly);
    #    channel_axis=-1 tells train_seg our stacks are channels-last.
    #  - AdamW is always used (the old SGD flag is deprecated), so we do not pass SGD.
    #  - bsize must be 256 for cpsam.
    #  - training uses native resolution (rescale=False), so no diameter is needed.
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=training_images,
        train_labels=label_images,
        channel_axis=-1,
        normalize=True,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        n_epochs=epochs,
        bsize=256,
        save_path=str(CELLPOSE_DIR),
        model_name=output_model_name)

    print(f"Trained model saved to: {model_path}")

    # Upload the trained model to Girder
    sendProgress(0.95, "Saving model", f"Uploading model {output_model_name}")
    girder_utils.upload_girder_model(client.client, output_model_name)


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Fine-tune a Cellpose-SAM model on user-corrected annotations')

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
