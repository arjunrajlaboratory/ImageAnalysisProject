import argparse
import json
import sys
import datetime

import numpy as np
from jax import random
from rasterio.features import rasterize
from shapely.geometry import Polygon

import annotation_client.workers as workers

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

    annotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, shape='point', tags=json.dumps(annotation_tag))
    points = np.array([[point['location'][i]
                        for i in ['Time', 'XY', 'Z']] + list(point['coordinates'][0].values())[1::-1]
                       for point in annotationList])
    points[:, -2:] -= np.array((0.5, 0.5))
    regionList = annotationClient.getAnnotationsByDatasetId(
        datasetId, shape='polygon', tags=json.dumps(region_tag))
    regionList.extend(annotationClient.getAnnotationsByDatasetId(
        datasetId, shape='rectangle', tags=json.dumps(region_tag)))

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
        c = snap_coords(c, image)
        c = fit_coords(c, image)
        c = remove_duplicate_coords(c)

        images.append(image)
        coords.append(c)

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

    match args.request:
        case 'compute':
            compute(datasetId, apiUrl, token, params)
        case 'interface':
            interface(params['image'], apiUrl, token)
