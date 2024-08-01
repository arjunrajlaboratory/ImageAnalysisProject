import argparse
import json
import sys
from collections import defaultdict

import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress
import annotation_client.workers as workers


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    interface = {}
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    propertyId = params.get('id', 'unknown_property')
    tags = set(params.get('tags', {}).get('tags', []))
    exclusive = params.get('tags', {}).get('exclusive', False)

    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    
    # Fetch all connections and annotations in one go
    connectionList = annotationClient.getAnnotationConnections(datasetId, limit=10000000)
    allAnnotations = annotationClient.getAnnotationsByDatasetId(datasetId, limit=10000000)

    # Create a mapping of annotation IDs to their tags
    annotation_tags = {ann['_id']: set(ann.get('tags', [])) for ann in allAnnotations}

    # Create integer mapping for annotation IDs
    id_to_integer_mapping = {-1: -1}
    current_integer = 0
    for ann in allAnnotations:
        if ann['_id'] not in id_to_integer_mapping:
            id_to_integer_mapping[ann['_id']] = current_integer
            current_integer += 1

    # Initialize annotationConnectionList with all annotations
    annotationConnectionList = {ann['_id']: {'parentId': -1, 'childId': -1} for ann in allAnnotations if (exclusive and set(ann.get('tags', [])) == tags) or (not exclusive and set(ann.get('tags', [])) & tags)}
    
    total_connections = len(connectionList)
    for i, connection in enumerate(connectionList):
        child_tags = annotation_tags.get(connection['childId'], set())
        parent_tags = annotation_tags.get(connection['parentId'], set())

        if (exclusive and child_tags == tags) or (not exclusive and child_tags & tags):
            annotationConnectionList[connection['childId']]['parentId'] = connection['parentId']

        if (exclusive and parent_tags == tags) or (not exclusive and parent_tags & tags):
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