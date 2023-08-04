import argparse
import json
import sys

import annotation_client.annotations as annotations
import annotation_client.workers as workers

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

    connectionList = []
    if connectionIds:
        # Get the annotations specified by id in the parameters
        for id in connectionIds:
            connectionList.append(annotationClient.getAnnotationConnectionById(id))
    else:
        # Get all point annotations from the dataset
        connectionList = annotationClient.getAnnotationConnections(datasetId, limit=1e6)

    filteredConnectionList = []
    for connection in connectionList:
        child_tags = set(annotationClient.getAnnotationById(connection['childId'])['tags'])
        if (exclusive and (child_tags == tags)) or ((not exclusive) and (len(child_tags & tags) > 0)):
            filteredConnectionList.append(connection)

    # We need at least one annotation
    if len(connectionList) == 0:
        return

    edges = np.array([[connection['parentId'], connection['childId']] for connection in filteredConnectionList])
    nodes = np.unique(edges[:, 0])  # Currently grabs children that have no children themselves, could optimize further
    # node_attributes = [(node, annotationClient.getAnnotationById(node)) for node in nodes]
    #
    # graph = nx.DiGraph()
    #
    # graph.add_nodes_from(node_attributes)
    # graph.add_edges_from(edges)
    #
    # for node in nodes:
    #
    #     children = list(graph.successors(node))
    #     annotationClient.addAnnotationPropertyValues(datasetId, node, {
    #         propertyName: len(children)})

    for node in nodes:
        n_children = np.sum(edges[:, 0] == node)
        annotationClient.addAnnotationPropertyValues(datasetId, node, {
            propertyId: int(n_children)})


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
