import argparse
import json
import sys

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendError

import annotation_utilities.annotation_tools as annotation_tools

import numpy as np

from shapely.geometry import Point, Polygon

from skimage import draw, filters, segmentation, measure


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Convert points to blobs': {
            'type': 'notes',
            'value': 'This tool converts points to blobs within a specified radius.\n'
                     'It looks within the radius, applies a threshold, and makes blobs.',
            'displayOrder': 0,
        },
        'Point tag': {
            'type': 'tags',
            'tooltip': 'Convert all points that have this tag.',
            'displayOrder': 1,
        },
        'Radius': {
            'type': 'number',
            'min': 0,
            'max': 100,
            'default': 5,
            'unit': 'pixels',
            'tooltip': 'The radius within which to look for points to convert to blobs.',
            'displayOrder': 2,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 2,
            'default': 0.0,
            'tooltip': 'The amount of smoothing to apply to the polygons.',
            'displayOrder': 3,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def create_condensate_mask(image, points, radius):
    # Initialize mask with same shape as image
    mask = np.zeros_like(image, dtype=bool)

    # Process each point
    for point_annotation in points:
        # Extract point coordinates
        geojsPoint = point_annotation['coordinates'][0]
        point = np.array([geojsPoint['y']-0.5, geojsPoint['x']-0.5])
        point = point.astype(int)  # Ensure integers for indexing

        # Get disk of pixels around point
        rr, cc = draw.disk(point, radius, shape=image.shape)

        # Extract intensity values in the disk
        disk_values = image[rr, cc]

        # Calculate Otsu threshold for this region
        thresh = filters.threshold_otsu(disk_values)

        # Apply threshold to disk region
        mask[rr, cc] = image[rr, cc] > thresh

    return mask


def create_condensate_polygons_intensity(image, binary_mask, points, smoothing):
    # Pre-allocate markers array with correct dtype
    markers = np.zeros(image.shape, dtype=np.int32)

    # Create points array all at once instead of loop
    points_array = np.array([[int(p['coordinates'][0]['y']-0.5),
                            int(p['coordinates'][0]['x']-0.5)]
                             for p in points])
    markers[points_array[:, 0], points_array[:, 1]
            ] = np.arange(1, len(points) + 1)

    # Optional: Use faster Gaussian approximation
    # If accuracy is critical, keep original filters.gaussian
    smooth_image = filters.gaussian(
        image, sigma=1, mode='reflect', preserve_range=True)

    # Watershed with optimized settings
    labels = segmentation.watershed(-smooth_image,
                                    markers,
                                    mask=binary_mask,
                                    compactness=0.001)  # Adjust as needed

    # Optional: If you need polygons, get them all at once
    contours = measure.find_contours(labels, 0.5)
    # Filter contours by label
    polygons = [cont for cont in contours if len(
        cont) > 2]  # Remove tiny contours

    shapely_polygons = []
    for polygon in polygons:
        shapely_polygons.append(Polygon(polygon).simplify(smoothing))

    return labels, shapely_polygons


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

    # roughly validate params
    keys = ["assignment", "channel", "connectTo",
            "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(
        *keys)(params)

    point_tag = list(set(workerInterface.get('Point tag', None)))
    radius = float(workerInterface['Radius'])
    smoothing = float(workerInterface['Smoothing'])
    if not point_tag or len(point_tag) == 0:
        sendError("No point tag specified")
        raise ValueError("No point tag specified")

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    sendProgress(0, "Loading points", "")

    # May need to change the limit for large numbers of annotations. Default is 1000000.
    pointAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=1000000, shape='point')

    sendProgress(0.2, "Points loaded", "")

    # Find all unique tuples of (time, xy, z) from the pointAnnotationList
    point_tuples = set([(p['location']['Time'], p['location']
                       ['XY'], p['location']['Z']) for p in pointAnnotationList])

    blob_annotations = []

    sendProgress(0.3, "Creating blobs", "")
    total_tuples = len(point_tuples)
    processed_tuples = 0

    # Iterate over each tuple
    for time, xy, z in point_tuples:
        # Just get points with this (time, xy, z)
        points = annotation_tools.filter_elements_T_XY_Z(
            pointAnnotationList, time, xy, z)
        if len(points) == 0:
            continue
        frame = tileClient.coordinatesToFrameIndex(xy, z, time, channel)
        image = tileClient.getRegion(datasetId, frame=frame)

        binary_mask = create_condensate_mask(image, points, radius)
        labels, polygons = create_condensate_polygons_intensity(
            np.squeeze(image), np.squeeze(binary_mask), points, smoothing)

        blob_annotations.extend(annotation_tools.polygons_to_annotations(
            polygons, datasetId, XY=xy, Time=time, Z=z, tags=tags, channel=channel))

        processed_tuples += 1
        fraction_done = processed_tuples / total_tuples
        sendProgress(0.3 + 0.6 * fraction_done, "Creating blobs",
                     f"{processed_tuples} of {total_tuples} frames processed")

    sendProgress(0.9, "Sending new blobs to server", "")

    annotationClient.createMultipleAnnotations(blob_annotations)


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Generate random point annotations')

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

    if args.request == 'compute':
        compute(datasetId, apiUrl, token, params)
    elif args.request == 'interface':
        interface(params['image'], apiUrl, token)
    else:
        # Handle other cases or throw an error if unexpected value is encountered
        pass

# Stupid Python 3.9 doesn't support match.
    # match args.request:
    #     case 'compute':
    #         compute(datasetId, apiUrl, token, params)
    #     case 'interface':
    #         interface(params['image'], apiUrl, token)
# We are not doing previews here.
#        case 'preview':
#            preview(datasetId, apiUrl, token, params, params['image'])
