import argparse
import json
import sys

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.workers as workers
from annotation_client.utils import sendProgress

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
            'max': 8,
            'default': 0,
            'tooltip': 'The size of the time gap that will be\nbridged when connecting objects across time.',
            'displayOrder': 2,
        },
        'Max distance (pixels)': {
            'type': 'number',
            'min': 0,
            'max': 1000,
            'default': 20,
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


def compute_nearest_child_to_parent(child_df, parent_df, groupby_cols=['Time', 'XY', 'Z'], max_distance=None):
    # Empty DataFrame to store results
    child_to_parent = pd.DataFrame(columns=['child_id', 'nearest_parent_id'])

    # Get all the groups (this operation does not actually compute the groups yet, but prepares the grouping)
    grouped = child_df.groupby(groupby_cols)

    # Determine the total number of groups
    total_groups = len(grouped)

    # Start the counter for processed groups
    processed_groups = 0

    # Group by unique location combinations
    for values, group in grouped:

        # Build a dynamic query string
        query_str = ' & '.join(
            [f"{col} == {val}" for col, val in zip(groupby_cols, values)])

        # Filter parent dataframe by the same location values
        parent_group = parent_df.query(query_str)

        # Ensure there are parents in the group to compare to
        if parent_group.empty:
            continue

        # Create a cKDTree for the parent group
        tree = cKDTree(
            np.array(list(zip(parent_group.geometry.x, parent_group.geometry.y))))

        # Compute distance and index for the nearest parent for each child within this group
        distances, indices = tree.query(
            np.array(list(zip(group.geometry.x, group.geometry.y))))

        # If max_distance is provided, filter out entries beyond that distance
        if max_distance is not None:
            valid_indices = distances <= max_distance
            distances = distances[valid_indices]
            indices = indices[valid_indices]
            # Update the group DataFrame to only contain valid entries
            group = group.iloc[valid_indices]

        # Map child IDs to nearest parent IDs for this group
        temp_df = pd.DataFrame({
            'child_id': group['_id'].values,
            'nearest_parent_id': parent_group.iloc[indices]['_id'].values
        })

        # Append the results to the main DataFrame
        child_to_parent = pd.concat(
            [child_to_parent, temp_df], ignore_index=True)

        # Increment the counter of processed groups
        processed_groups += 1

        # Compute the fraction of work done
        fraction_done = processed_groups / total_groups

        # Send the progress update
        sendProgress(fraction_done, "Computing connections",
                     f"{processed_groups} of {total_groups} groups processed")

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
    max_distance = float(workerInterface['Max distance (pixels)'])
    gap_size = int(workerInterface['Connect across gaps'])

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

    # We will always group by XY, because there is no reasonable scenario in which you want to connect across XY.
    # We are also doing Z because we don't want to connect across Z slices.
    groupby_cols = ['XY', 'Z']

    # Sort the gdf_object based on the connect_across column in descending order
    gdf_object = gdf_object.sort_values(by='Time', ascending=False)

    my_new_connections = []
    combined_tags = list(set(object_tag))

    for index, current_object in gdf_object.iterrows():
        previous_objects = get_previous_objects(
            current_object, gdf_object, 1)  # gap_size = 1
        if not previous_objects.empty:
            # Use the compute_nearest_child_to_parent function to find the nearest previous object
            nearest_object_df = compute_nearest_child_to_parent(
                gdf_object.loc[[index]], previous_objects, groupby_cols=groupby_cols, max_distance=max_distance)

            for _, row in nearest_object_df.iterrows():
                # the current object is the child (in later time)
                child_id = row['child_id']
                # the nearest previous object is the parent (in earlier time)
                parent_id = row['nearest_parent_id']

                my_new_connections.append({
                    # assuming you've already set this variable elsewhere in your code
                    'datasetId': datasetId,
                    'parentId': parent_id,
                    'childId': child_id,
                    'tags': combined_tags
                })

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
