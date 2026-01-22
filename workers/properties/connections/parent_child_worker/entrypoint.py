import argparse
import json
import sys
from collections import defaultdict

import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress
import annotation_client.workers as workers

import annotation_utilities.annotation_tools as annotation_tools


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    interface = {
        'Connection IDs': {
            'type': 'notes',
            'value': 'This tool adds a property to the objects to document the connections between them, which is helpful for time-lapse analysis. '
                     'It gives each object an ID, and then shows the ID for each object\'s parent and child. '
                     'If there is no parent or child, it assigns -1.',
            'displayOrder': 0,
        },
        'Ignore self-connections': {
            'type': 'checkbox',
            'tooltip': 'If checked, self-connections will be ignored.\nThis is useful if you want to find all connections in the dataset,\nbut not connections between the same object.',
            'default': True,
            'displayOrder': 1
        },
        'Time lapse': {
            'type': 'checkbox',
            'tooltip': 'If checked, keeps parent as earlier time and child as later time.\nOtherwise, it will parse multiple connections, including reverse connections.',
            'default': True,
            'displayOrder': 2
        },
        'Add track IDs': {
            'type': 'checkbox',
            'tooltip': 'If checked, adds a trackId to each annotation.\nConnected annotations will share the same trackId.',
            'default': False,
            'displayOrder': 3
        }
    }
    client.setWorkerImageInterface(image, interface)


def find_connected_components(connections, annotation_ids):
    """
    Find connected components using a union-find data structure.
    Returns a dictionary mapping annotation IDs to track IDs.
    """
    # Initialize parent array for union-find
    parent = {ann_id: ann_id for ann_id in annotation_ids}

    def find(x):
        # Find root with path compression
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        # Union by linking roots
        parent[find(x)] = find(y)

    # Process all connections to build connected components
    for conn in connections:
        if conn['parentId'] in parent and conn['childId'] in parent:
            union(conn['parentId'], conn['childId'])

    # Assign track IDs (using the root node's ID as the track ID)
    track_ids = {}
    track_counter = 0
    root_to_track = {}

    for ann_id in annotation_ids:
        root = find(ann_id)
        if root not in root_to_track:
            root_to_track[root] = track_counter
            track_counter += 1
        track_ids[ann_id] = root_to_track[root]

    return track_ids


def compute(datasetId, apiUrl, token, params):
    propertyId = params.get('id', 'unknown_property')
    tags = set(params.get('tags', {}).get('tags', []))
    exclusive = params.get('tags', {}).get('exclusive', False)

    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(
        datasetId, apiUrl, token, params)

    ignore_self_connections = params['workerInterface'].get(
        'Ignore self-connections', True)
    time_lapse = params['workerInterface'].get('Time lapse', True)
    add_track_ids = params['workerInterface'].get('Add track IDs', False)
    print("params", params)

    # Fetch all connections and annotations in one go
    connectionList = annotationClient.getAnnotationConnections(
        datasetId, limit=10000000)
    allAnnotations = annotationClient.getAnnotationsByDatasetId(
        datasetId, limit=10000000)
    allAnnotations = annotation_tools.get_annotations_with_tags(allAnnotations, params.get(
        'tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))

    # Create integer mapping for annotation IDs
    id_to_integer_mapping = {-1: -1}
    current_integer = 0
    if time_lapse:
        id_to_time_mapping = {}
    for ann in allAnnotations:
        if ann['_id'] not in id_to_integer_mapping:
            id_to_integer_mapping[ann['_id']] = current_integer
            current_integer += 1
            if time_lapse:
                id_to_time_mapping[ann['_id']] = ann['location']['Time']

    # Initialize annotationConnectionList with all annotations
    annotationConnectionList = {ann['_id']: {'parentId': -1, 'childId': -1} for ann in allAnnotations if (
        exclusive and set(ann.get('tags', [])) == tags) or (not exclusive and set(ann.get('tags', [])) & tags)}

    # Get track IDs if requested
    track_ids = None
    print("add_track_ids", add_track_ids)
    if add_track_ids:
        sendProgress(0.3, 'Computing track IDs',
                     "Finding connected components")

        filtered_connections = [
            conn for conn in connectionList
            if (not ignore_self_connections or conn['parentId'] != conn['childId']) and
            conn['childId'] in annotationConnectionList and
            conn['parentId'] in annotationConnectionList
        ]
        track_ids = find_connected_components(
            filtered_connections, annotationConnectionList.keys())

    total_connections = len(connectionList)
    for i, connection in enumerate(connectionList):

        # Ignore self-connections
        if ignore_self_connections and connection['parentId'] == connection['childId']:
            continue

        # Skip if either the parent or child is not in the annotation list
        if connection['childId'] not in annotationConnectionList or connection['parentId'] not in annotationConnectionList:
            continue

        # If time lapse is checked, ensure that the child is always later than the parent
        if time_lapse and id_to_time_mapping[connection['childId']] <= id_to_time_mapping[connection['parentId']]:
            annotationConnectionList[connection['childId']
                                     ]['childId'] = connection['parentId']
            annotationConnectionList[connection['parentId']
                                     ]['parentId'] = connection['childId']
        else:
            annotationConnectionList[connection['childId']
                                     ]['parentId'] = connection['parentId']
            annotationConnectionList[connection['parentId']
                                     ]['childId'] = connection['childId']

        if i % 1000 == 0:  # Update progress every 1000 connections
            sendProgress((i / total_connections) * 0.6 + 0.3, 'Processing connections',
                         f"Processed {i}/{total_connections} connections")

    # Prepare data for batch upload
    property_value_dict = {}
    for annotationId, prop in annotationConnectionList.items():
        value_dict = {
            'annotationId': float(id_to_integer_mapping[annotationId]),
            'parentId': float(id_to_integer_mapping.get(prop['parentId'], -1)),
            'childId': float(id_to_integer_mapping.get(prop['childId'], -1))
        }
        if track_ids is not None:
            value_dict['trackId'] = float(track_ids[annotationId])
        property_value_dict[annotationId] = value_dict

    # Create the dataset_property_value_dict
    dataset_property_value_dict = {datasetId: property_value_dict}

    # Batch upload all property values
    sendProgress(0.9, 'Uploading property values', "Sending data to server")
    # First delete the old property values.
    annotationClient.deleteAnnotationPropertyValues(propertyId, datasetId)
    workerClient.add_multiple_annotation_property_values(
        dataset_property_value_dict)
    sendProgress(1.0, 'Completed', "Worker finished successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute parent-child relationships for annotations')

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
