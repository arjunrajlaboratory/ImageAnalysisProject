import argparse
import json
import sys
from collections import defaultdict

import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress
import annotation_client.workers as workers


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
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
        }
    }
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    propertyId = params.get('id', 'unknown_property')
    tags = set(params.get('tags', {}).get('tags', []))
    exclusive = params.get('tags', {}).get('exclusive', False)

    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)

    ignore_self_connections = params.get('Ignore self-connections', True)
    time_lapse = params.get('Time lapse', True)
    
    # Fetch all connections and annotations in one go
    connectionList = annotationClient.getAnnotationConnections(datasetId, limit=10000000)
    allAnnotations = annotationClient.getAnnotationsByDatasetId(datasetId, limit=10000000)

    # Create a mapping of annotation IDs to their tags
    annotation_tags = {ann['_id']: set(ann.get('tags', [])) for ann in allAnnotations}

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
    annotationConnectionList = {ann['_id']: {'parentId': -1, 'childId': -1} for ann in allAnnotations if (exclusive and set(ann.get('tags', [])) == tags) or (not exclusive and set(ann.get('tags', [])) & tags)}
    
    total_connections = len(connectionList)
    for i, connection in enumerate(connectionList):
        if ignore_self_connections and connection['parentId'] == connection['childId']:
            continue

        if time_lapse and id_to_time_mapping[connection['parentId']] >= id_to_time_mapping[connection['childId']]:
            continue

        child_tags = annotation_tags.get(connection['childId'], set())
        parent_tags = annotation_tags.get(connection['parentId'], set())

        if (exclusive and child_tags == tags) or (not exclusive and child_tags & tags):
            if time_lapse and id_to_time_mapping[connection['childId']] <= id_to_time_mapping[connection['parentId']]:
                annotationConnectionList[connection['childId']]['childId'] = connection['parentId']
            else:
                annotationConnectionList[connection['childId']]['parentId'] = connection['parentId']

        if (exclusive and parent_tags == tags) or (not exclusive and parent_tags & tags):
            if time_lapse and id_to_time_mapping[connection['childId']] <= id_to_time_mapping[connection['parentId']]:
                annotationConnectionList[connection['parentId']]['parentId'] = connection['childId']
            else:
                annotationConnectionList[connection['parentId']]['childId'] = connection['childId']

        if i % 1000 == 0:  # Update progress every 1000 connections
            sendProgress(i / total_connections, 'Processing connections', f"Processed {i}/{total_connections} connections")

    # Prepare data for batch upload
    property_value_dict = {}
    for annotationId, prop in annotationConnectionList.items():
        property_value_dict[annotationId] = {
            'annotationId': float(id_to_integer_mapping[annotationId]),
            'parentId': float(id_to_integer_mapping.get(prop['parentId'], -1)),
            'childId': float(id_to_integer_mapping.get(prop['childId'], -1))
        }

    # Create the dataset_property_value_dict
    dataset_property_value_dict = {datasetId: property_value_dict}

    # Batch upload all property values
    sendProgress(0.9, 'Uploading property values', "Sending data to server")
    workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)
    sendProgress(1.0, 'Completed', "Worker finished successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute parent-child relationships for annotations')

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')

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