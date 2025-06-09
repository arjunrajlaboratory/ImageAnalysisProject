import base64
import argparse
import json
import sys
import random
import time
import timeit

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning

# import annotation_tools
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
        'Connect to nearest': {
            'type': 'notes',
            'value': 'This tool connects annotations to their nearest neighbors. '
                     'It will connect from objects with the parent tag to objects with the child tag. '
                     '<a href="https://docs.nimbusimage.com/documentation/analyzing-image-data-with-objects-connections-and-properties/tools-for-connecting-objects#connect-to-nearest" target="_blank">Learn more</a>',
            'displayOrder': 0,
        },
        'Parent tag': {
            'type': 'tags',
            'displayOrder': 1,
        },
        'Child tag': {
            'type': 'tags',
            'displayOrder': 2,
        },
        'Connect across Z': {
            'type': 'select',
            'items': ['Yes', 'No'],
            'default': 'No',
            'tooltip': 'Connect objects regardless of their z-slice.\nFor example, it will connect a parent in Z=1 to a child in Z=4.',
            'displayOrder': 4,
        },
        'Connect across T': {
            'type': 'select',
            'items': ['Yes', 'No'],
            'default': 'No',
            'tooltip': 'Connect objects regardless of their time point.\nFor example, it will connect a parent in T=1 to a child in T=4.',
            'displayOrder': 5,
        },
        'Connect to closest centroid or edge': {
            'type': 'select',
            'items': ['Centroid', 'Edge'],
            'default': 'Centroid',
            'tooltip': 'Connect to the parent with either the closest centroid or closest edge.',
            'displayOrder': 7,
        },
        'Restrict connection': {
            'type': 'select',
            'items': ['None', 'Touching parent', 'Within parent'],
            'default': 'None',
            'tooltip': 'Only connect if the child is either touching the parent or completely within the parent.',
            'displayOrder': 8,
        },
        'Max distance (pixels)': {
            'type': 'number',
            'min': 0,
            'max': 5000,
            'default': 1000,
            'tooltip': 'The maximum distance (in pixels) between the child and\nparent objects to be connected. Otherwise, objects will not be connected.',
            'displayOrder': 9,
        },
        'Connect up to N children': {
            'type': 'number',
            'min': 1,
            'max': 10000,
            'default': 10000,
            'tooltip': 'The maximum number of children to connect to each parent.',
            'displayOrder': 10,
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def extract_spatial_annotation_data(obj_list):
    data = []
    for obj in obj_list:
        shape = obj['shape']
        coords = obj['coordinates']

        if shape == 'point':
            geometry = Point(coords[0]['x'], coords[0]['y'])
        elif shape == 'polygon':
            geometry = Polygon([(pt['x'], pt['y']) for pt in coords])

        data.append({
            '_id': obj['_id'],
            'geometry': geometry,
            'Time': obj['location']['Time'],
            'XY': obj['location']['XY'],
            'Z': obj['location']['Z']
        })

    # Handle empty case
    if not data:
        # Create empty GeoDataFrame with proper columns
        gdf = gpd.GeoDataFrame(columns=['_id', 'geometry', 'Time', 'XY', 'Z'])
        gdf = gdf.set_geometry('geometry')
        return gdf

    # Create GeoDataFrame directly
    gdf = gpd.GeoDataFrame(data, geometry='geometry')
    return gdf


def compute_nearest_child_to_parent(
    child_df,
    parent_df,
    groupby_cols=['Time', 'XY', 'Z'],
    max_distance=None,
    connect_to_closest='Centroid',
    restrict_connection='None',
    max_children=None
):
    # Empty DataFrame to store results
    child_to_parent = pd.DataFrame(columns=['child_id', 'nearest_parent_id'])

    # Get all the groups
    grouped = child_df.groupby(groupby_cols)
    total_groups = len(grouped)
    processed_groups = 0

    # Group by unique location combinations
    for values, group in grouped:

        # Create boolean mask for each column
        mask = pd.Series(True, index=parent_df.index)
        for col, val in zip(groupby_cols, values):
            mask &= (parent_df[col] == val)
        parent_group = parent_df[mask]

        if parent_group.empty:
            continue

        # Handle different connection restrictions
        valid_children = group.copy()
        if restrict_connection != 'None':
            if restrict_connection == 'Touching parent':
                # Find children that intersect with any parent
                mask = valid_children.geometry.intersects(
                    parent_group.geometry.unary_union)
            elif restrict_connection == 'Within parent':
                # Find children that are within any parent
                mask = valid_children.geometry.within(
                    parent_group.geometry.unary_union)
            valid_children = valid_children[mask]

        if valid_children.empty:
            continue

        if connect_to_closest == 'Edge':

            # Reset index of both DataFrames before joining
            valid_children = valid_children.reset_index(drop=True)
            parent_group = parent_group.reset_index(drop=True)

            # Use sjoin_nearest to find closest parents for each child
            joined = gpd.sjoin_nearest(
                valid_children,
                parent_group,
                how='left',
                distance_col='distance'
            )

            # If we have duplicates, as in multiple parents for the same child,
            # keep the closest match for each child
            if len(joined) > len(valid_children):
                # First sort by distance to get shortest distances first
                joined = joined.sort_values('distance')

                # Then drop duplicates based on the DataFrame's index
                # (which corresponds to the child points)
                joined = joined[~joined.index.duplicated(keep='first')]

            # Extract distances and indices
            distances = joined['distance'].values
            indices = joined.index_right.values

            # Apply max_distance filter if specified
            if max_distance is not None:
                valid_indices = joined['distance'] <= max_distance
                joined = joined[valid_indices]
                distances = joined['distance'].values
                indices = joined['index_right'].values
                valid_children = valid_children.loc[joined.index]

        else:  # Centroid
            parent_centroids = parent_group.geometry.centroid
            child_centroids = valid_children.geometry.centroid
            tree = cKDTree(
                np.array(list(zip(parent_centroids.x, parent_centroids.y))))
            distances, indices = tree.query(
                np.array(list(zip(child_centroids.x, child_centroids.y))))

            # Apply max_distance filter if specified
            if max_distance is not None:
                valid_indices = distances <= max_distance
                distances = distances[valid_indices]
                indices = indices[valid_indices]
                valid_children = valid_children.iloc[valid_indices]

        # Create connections with distances
        temp_df = pd.DataFrame({
            'child_id': valid_children['_id'].values,
            'nearest_parent_id': parent_group.iloc[indices]['_id'].values,
            'distance': distances
        })

        # Apply max_children filter if specified, taking closest children first
        if max_children is not None:
            temp_df = (temp_df.sort_values('distance')  # Sort by distance
                       .groupby('nearest_parent_id')
                       .head(max_children))

        child_to_parent = pd.concat(
            [child_to_parent, temp_df], ignore_index=True)

        processed_groups += 1
        fraction_done = processed_groups / total_groups
        sendProgress(fraction_done, "Computing connections",
                     f"{processed_groups} of {total_groups} groups processed")

    return child_to_parent


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

    parent_tag = list(set(workerInterface.get('Parent tag', None)))
    child_tag = list(set(workerInterface.get('Child tag', None)))
    max_distance = float(workerInterface['Max distance (pixels)'])
    connect_across_z = workerInterface['Connect across Z'] == 'Yes'
    connect_across_t = workerInterface['Connect across T'] == 'Yes'
    connect_to_closest = workerInterface['Connect to closest centroid or edge']
    restrict_connection = workerInterface['Restrict connection']
    max_children = int(workerInterface['Connect up to N children'])

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    # TODO: Currently, we do not handle line annotations.
    pointAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=10000000, shape='point')
    blobAnnotationList = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=10000000, shape='polygon')
    # lineAnnotationList = annotationClient.getAnnotationsByDatasetId(datasetId, limit = 10000000, shape='line')
    allAnnotationList = pointAnnotationList + blobAnnotationList

    parentList = annotation_tools.get_annotations_with_tags(
        allAnnotationList, parent_tag, exclusive=False)
    childList = annotation_tools.get_annotations_with_tags(
        allAnnotationList, child_tag, exclusive=False)

    parent_data = extract_spatial_annotation_data(parentList)
    child_data = extract_spatial_annotation_data(childList)

    # Check for empty parent or child lists and send warnings
    if len(parent_data) == 0:
        sendWarning("No parent annotations found",
                    f"No annotations found with parent tag(s): {', '.join(parent_tag)}")

    if len(child_data) == 0:
        sendWarning("No child annotations found",
                    f"No annotations found with child tag(s): {', '.join(child_tag)}")

    # We will always group by XY, because there is no reasonable scenario in which you want to connect across XY.
    groupby_cols = ['XY']

    # Add the 'Time' and 'Z' columns based on the boolean flags
    if not connect_across_t:
        groupby_cols.append('Time')
    if not connect_across_z:
        groupby_cols.append('Z')

    # Compute the child to parent mapping
    child_to_parent = compute_nearest_child_to_parent(
        child_data,
        parent_data,
        groupby_cols=groupby_cols,
        max_distance=max_distance,
        connect_to_closest=connect_to_closest,
        restrict_connection=restrict_connection,
        max_children=max_children)

    new_connections = []
    combined_tags = list(set(parent_tag + child_tag))

    for index, row in child_to_parent.iterrows():
        child_id = row['child_id']
        parent_id = row['nearest_parent_id']

        new_connections.append({
            'datasetId': datasetId,
            'parentId': parent_id,
            'childId': child_id,
            'tags': combined_tags
        })

    annotationClient.createMultipleConnections(new_connections)


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
