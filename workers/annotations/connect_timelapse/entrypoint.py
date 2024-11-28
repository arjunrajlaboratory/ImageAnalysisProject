import argparse
import json
import sys

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendError

import annotation_utilities.annotation_tools as annotation_tools

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Using connect time lapse': {
            'type': 'notes',
            'value': 'This tool connects objects across time slices.'
                     'It allows you to connect objects even if there are gaps in time.',
            'displayOrder': 0,
        },
        'Object to connect tag': {
            'type': 'tags',
            'tooltip': 'Connect all objects that have this tag.',
            'displayOrder': 1,
        },
        'Connect across gaps': {
            'type': 'number',
            'min': 0,
            'max': 10,
            'default': 0,
            'unit': 'Time pt',
            'tooltip': 'The size of the time gap that will be\nbridged when connecting objects across time.',
            'displayOrder': 2,
        },
        'Max distance': {
            'type': 'number',
            'min': 0,
            'max': 1000,
            'default': 20,
            'unit': 'pixels',
            'tooltip': 'The maximum distance (in pixels) between the child and\nparent objects to be connected. Otherwise, objects will not be connected.',
            'displayOrder': 3,
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def extract_spatial_annotation_data(obj_list):
    data = []
    for obj in obj_list:
        x, y = None, None
        shape = obj['shape']
        coords = obj['coordinates']

        if shape == 'point':
            x, y = coords[0]['x'], coords[0]['y']
        elif shape == 'polygon':
            if len(coords) < 3:
                continue
            polygon = Polygon([(pt['x'], pt['y']) for pt in coords])
            centroid = polygon.centroid
            x, y = centroid.x, centroid.y

        data.append({
            '_id': obj['_id'],
            'x': x,
            'y': y,
            'Time': obj['location']['Time'],
            'XY': obj['location']['XY'],
            'Z': obj['location']['Z']
        })
    return data


def compute_nearest_child_to_parent(child_df, parent_df, max_distance=None):
    """Compute nearest parents for children, assuming all grouping is done externally"""
    if child_df.empty or parent_df.empty:
        return pd.DataFrame()

    # Create KDTree for all parents
    tree = cKDTree(
        np.array(list(zip(parent_df.geometry.x, parent_df.geometry.y))))

    # Find nearest parent for each child
    distances, indices = tree.query(
        np.array(list(zip(child_df.geometry.x, child_df.geometry.y))))

    # Filter by max_distance if specified
    if max_distance is not None:
        valid_indices = distances <= max_distance
        distances = distances[valid_indices]
        indices = indices[valid_indices]
        child_df = child_df.iloc[valid_indices]

    # Create result DataFrame
    child_to_parent = pd.DataFrame({
        'child_id': child_df['_id'].values,
        'nearest_parent_id': parent_df.iloc[indices]['_id'].values
    })

    return child_to_parent


def get_previous_objects(current_object, dataframe, gap_size):
    return dataframe[dataframe['Time'] == current_object['Time'] - gap_size]


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

    object_tag = list(set(workerInterface.get('Object to connect tag', None)))
    max_distance = float(workerInterface['Max distance'])
    gap_size = int(workerInterface['Connect across gaps'])

    if not object_tag or len(object_tag) == 0:
        sendError("No object tag specified")
        raise ValueError("No object tag specified")

    print(
        f"Connecting {object_tag} objects across {gap_size} time slices "
        f"with a max distance of {max_distance} pixels")

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)

    sendProgress(0, "Loadingobjects", "")

    # May need to change the limit for large numbers of annotations. Default is 1000000.
    # TODO: Currently, we do not handle line annotations.
    pointAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=1000000, shape='point')
    blobAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=1000000, shape='polygon')
    allAnnotationList = pointAnnotationList + blobAnnotationList

    objectList = annotation_tools.get_annotations_with_tags(
        allAnnotationList, object_tag, exclusive=False)
    object_data = extract_spatial_annotation_data(objectList)

    object_df = pd.DataFrame(object_data)

    gdf_object = gpd.GeoDataFrame(
        object_df, geometry=gpd.points_from_xy(object_df.x, object_df.y))

    sendProgress(0.2, "Objects loaded", "")

    # Group by all spatial dimensions first
    spatial_groups = gdf_object.groupby(['XY', 'Z'])

    my_new_connections = []
    total_groups = len(spatial_groups)

    # Process each spatial group
    for group_idx, ((xy, z), spatial_group) in enumerate(spatial_groups):
        # Sort time points within this spatial group
        time_groups = spatial_group.groupby('Time', sort=True)
        time_points = sorted(time_groups.groups.keys(), reverse=True)

        # Process each time slice within this spatial group
        for current_time in time_points[:-1]:
            current_objects = time_groups.get_group(current_time)
            previous_time = time_points[time_points.index(current_time) + 1]
            previous_objects = time_groups.get_group(previous_time)

            # Find connections for this group
            connections = compute_nearest_child_to_parent(
                current_objects,
                previous_objects,
                max_distance=max_distance
            )

            # Add connections to results
            for _, row in connections.iterrows():
                my_new_connections.append({
                    'datasetId': datasetId,
                    'parentId': row['nearest_parent_id'],
                    'childId': row['child_id'],
                    'tags': ["Time lapse connection"]
                })

        sendProgress((group_idx + 1) / total_groups,
                     "Processing XY, Z groups",
                     f"Processed {group_idx + 1} of {total_groups} groups")

    sendProgress(0.9, "Sending connections to server", "")
    annotationClient.createMultipleConnections(my_new_connections)


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
