import argparse
import json
import sys

import annotation_client.annotations as annotations
import annotation_client.workers as workers

from annotation_client.utils import sendProgress

# import networkx as nx
import numpy as np


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Tags': {
            'type': 'tags'
        },
        'Exclusive': {
            'type': 'select',
            'items': ['Yes', 'No'],
            'default': 'Yes'
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    """
    Params is a dict containing the following parameters:
    required:
        name: The name of the property
        id: The id of the property
        propertyType: can be "morphology", "relational", or "layer"
    optional:
        annotationId: A list of annotation ids for which the property should be computed
        shape: The shape of annotations that should be used
        layer: Which specific layer should be used for intensity calculations
        tags: A list of annotation tags, used when counting for instance the number of connections to specific tagged annotations
    """
    propertyId = params.get('id', 'unknown_property')

    connectionIds = params.get('connectionIds', None)

    workerInterface = params['workerInterface']
    tags = set(workerInterface.get('Tags', None))
    exclusive = workerInterface['Exclusive'] == 'Yes'

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)


    connectionList = []
    if connectionIds:
        # Get the annotations specified by id in the parameters
        for id in connectionIds:
            connectionList.append(annotationClient.getAnnotationConnectionById(id))
    else:
        # Get all point annotations from the dataset
        connectionList = annotationClient.getAnnotationConnections(datasetId, limit=10000000)

    filteredConnectionList = []
    number_connections = len(connectionList)
    #for connection in connectionList:
    for i, connection in enumerate(connectionList):
        child_tags = set(annotationClient.getAnnotationById(connection['childId'])['tags'])
        if (exclusive and (child_tags == tags)) or ((not exclusive) and (len(child_tags & tags) > 0)):
            filteredConnectionList.append(connection)
        sendProgress((i+1)/number_connections, 'Filtering connections', f"Processing connection {i+1}/{number_connections}")

    # We need at least one annotation
    if len(connectionList) == 0:
        return

    edges = np.array([[connection['parentId'], connection['childId']] for connection in filteredConnectionList])
    nodes = np.unique(edges[:, 0])  # Currently grabs children that have no children themselves, could optimize further

    number_nodes = len(nodes)
    property_value_dict = {}  # Initialize as a dictionary
    for i, node in enumerate(nodes):
        n_children = np.sum(edges[:, 0] == node)
        property_value_dict[node] = int(n_children)
        sendProgress((i+1)/number_nodes, 'Computing children count', f"Processing node {i+1}/{number_nodes}")
    
    dataset_property_value_dict = {datasetId: property_value_dict}
    sendProgress(0.5,'Done computing', 'Sending computed counts to the server')
    workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)


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
