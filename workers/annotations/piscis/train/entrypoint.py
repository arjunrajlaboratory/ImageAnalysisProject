import argparse
import json
import sys
import datetime

import numpy as np
from jax import random
from rasterio.features import rasterize
from shapely.geometry import Polygon

import annotation_client.workers as workers

from annotation_client.utils import sendError, sendWarning, sendProgress

from piscis.data import generate_dataset
from piscis.paths import MODELS_DIR
from piscis.training import train_model
from piscis.utils import fit_coords, remove_duplicate_coords, snap_coords

from annotation_utilities.point_in_polygon import point_in_polygon

import utils


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    models = sorted(path.stem for path in MODELS_DIR.glob('*'))
    girder_models = [model['name'] for model in utils.list_girder_models(client.client)[0]]
    models = sorted(list(set(models + girder_models)))

    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime('%Y%m%d_%H%M%S')

    # Available types: number, text, tags, layer
    interface = {
        'Piscis Train': {
            'type': 'notes',
            'value': 'This tool trains a Piscis model using user-corrected annotations. '
                     '<a href="https://docs.nimbusimage.com/documentation/analyzing-image-data-with-objects-connections-and-properties/tools-for-making-objects#piscis-training" target="_blank">Learn more</a>',
            'displayOrder': 0,
        },
        'Initial Model Name': {
            'type': 'select',
            'items': models,
            'default': '20230905',
            'displayOrder': 1,
        },
        'Learning Rate': {
            'type': 'text',
            'default': 0.2,
            'displayOrder': 5,
        },
        'Weight Decay': {
            'type': 'text',
            'default': 0.0001,
            'displayOrder': 6,
        },
        'Epochs': {
            'type': 'text',
            'default': 40,
            'displayOrder': 7,
        },
        'Random Seed': {
            'type': 'text',
            'default': 42,
            'displayOrder': 8,
        },
        'New Model Name': {
            'type': 'text',
            'default': datetime_string,
            'displayOrder': 2,
        },
        'Annotation Tag': {
            'type': 'tags',
            'displayOrder': 3,
        },
        'Region Tag': {
            'type': 'tags',
            'displayOrder': 4,
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

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationClient = workerClient.annotationClient

    # Get the Gaussian sigma and threshold from interface values
    workerInterface = params['workerInterface']
    initial_model_name = workerInterface['Initial Model Name']
    learning_rate = float(workerInterface['Learning Rate'])
    weight_decay = float(workerInterface['Weight Decay'])
    epochs = int(workerInterface['Epochs'])
    random_seed = int(workerInterface['Random Seed'])
    new_model_name = workerInterface['New Model Name']
    annotation_tag = workerInterface['Annotation Tag']
    region_tag = workerInterface['Region Tag']

    # Check if tags are set
    if not annotation_tag or len(annotation_tag) == 0:
        sendError("No annotation tag selected.",
                  info="Please select at least one annotation tag.")
        raise ValueError("No annotation tag selected.")
    
    if not region_tag or len(region_tag) == 0:
        sendError("No region tag selected.",
                  info="Please select at least one region tag.")
        raise ValueError("No region tag selected.")

    annotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, shape='point', tags=json.dumps(annotation_tag))
    
    # Check if any annotations were found
    if not annotationList or len(annotationList) == 0:
        sendError("No annotations found with the selected annotation tag.",
                  info=f"No point annotations found with tag(s): {annotation_tag}. Please check your annotation tags.")
        raise ValueError("No annotations found with the selected annotation tag.")
    
    points = np.array([[point['location'][i]
                        for i in ['Time', 'XY', 'Z']] + list(point['coordinates'][0].values())[1::-1]
                       for point in annotationList])
    points[:, -2:] -= np.array((0.5, 0.5))
    
    regionList = annotationClient.getAnnotationsByDatasetId(
        datasetId, shape='polygon', tags=json.dumps(region_tag))
    regionList.extend(annotationClient.getAnnotationsByDatasetId(
        datasetId, shape='rectangle', tags=json.dumps(region_tag)))

    # Check if any regions were found
    if not regionList or len(regionList) == 0:
        sendError("No regions found with the selected region tag.",
                  info=f"No polygon or rectangle annotations found with tag(s): {region_tag}. Please check your region tags.")
        raise ValueError("No regions found with the selected region tag.")

    images = []
    coords = []

    for region in regionList:

        image = workerClient.get_image_for_annotation(region)

        polygon = np.array([[coordinate[i] for i in ['y', 'x']]
                           for coordinate in region['coordinates']]) - np.array((0.5, 0.5))
        polygony, polygonx = polygon.T
        minx, miny, maxx, maxy = np.min(polygonx), np.min(
            polygony), np.max(polygonx), np.max(polygony)
        minx, miny, maxx, maxy = np.maximum(minx, -0.5), np.maximum(miny, -0.5), np.minimum(
            maxx, image.shape[1] - 0.5), np.minimum(maxy, image.shape[0] - 0.5)
        mini, minj, maxi, maxj = round(miny), round(minx), round(maxy), round(maxx)
        shapely_polygon = Polygon(np.stack([polygonx - minj, polygony - mini]).T)
        image = image[mini:maxi + 1, minj:maxj + 1]

        mask = rasterize([(shapely_polygon, 1)], out_shape=image.shape,
                         all_touched=True, dtype=np.uint8)
        image = image * mask

        c = points[np.all(points[:, :3] == np.array([region['location'][i]
                          for i in ['Time', 'XY', 'Z']]), axis=1)][:, -2:]
        c = c[point_in_polygon(c, polygon)] - np.array([mini, minj])
        c = np.array(c)
        
        # Check if this region has any points
        if c.size == 0 or (c.ndim == 1 and len(c) == 0):
            sendError("Region with no points found.",
                      info="Every training region must contain at least one point annotation. Please ensure all regions have points inside them.")
            raise ValueError("Region with no points found.")
        
        # Ensure c is 2D for the coordinate processing functions
        if c.ndim == 1:
            if len(c) == 0:
                sendError("Region with no points found.",
                          info="Every training region must contain at least one point annotation. Please ensure all regions have points inside them.")
                raise ValueError("Region with no points found.")
            # If it's 1D but has data, it might be a single point, reshape it
            c = c.reshape(1, -1)
        
        c = snap_coords(c, image)
        c = fit_coords(c, image)
        c = remove_duplicate_coords(c)

        # Final check after processing - ensure we still have points
        if c.size == 0 or len(c) == 0:
            sendError("Region with no valid points after processing.",
                      info="After coordinate processing, a region ended up with no valid points. Please check your annotations and region boundaries.")
            raise ValueError("Region with no valid points after processing.")

        images.append(image)
        coords.append(c)

    # Check if we have any data to train with
    if len(images) == 0 or len(coords) == 0:
        sendError("No training data available.",
                  info="No valid training data could be extracted from the annotations and regions. Please check your annotations.")
        raise ValueError("No training data available.")
    
    # Check if all coordinate arrays are empty (would cause training to fail)
    total_points = sum(len(coord_array) for coord_array in coords)
    if total_points == 0:
        sendError("No valid points found in any region.",
                  info="No valid points were found in any of the training regions. Please ensure your regions contain point annotations.")
        raise ValueError("No valid points found in any region.")

    key, _ = random.split(random.PRNGKey(random_seed), 2)
    dataset_path = f'{new_model_name}.npz'
    generate_dataset(dataset_path, images, coords, key, train_size=1., test_size=0.)

    gc = annotationClient.client
    utils.download_girder_cache(gc, mode='train')

    train_model(
        model_name=new_model_name,
        dataset_path=dataset_path,
        initial_model_name=initial_model_name,
        random_seed=random_seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        warmup_fraction=0.1,
        save_checkpoints=False
    )

    utils.upload_girder_model(gc, new_model_name)
    try:
        utils.upload_girder_cache(gc, mode='train')
    except Exception as e:
        # Log the error but continue execution
        print(f"Warning: Failed to upload cache: {e}", file=sys.stderr)


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

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

    if args.request == 'compute':
        compute(datasetId, apiUrl, token, params)
    elif args.request == 'interface':
        interface(params['image'], apiUrl, token)
