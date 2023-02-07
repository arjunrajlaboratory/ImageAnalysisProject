import argparse
import json
import sys

import annotation_client.annotations as annotations

# import networkx as nx
import numpy as np


def main(datasetId, apiUrl, token, params):
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
        connectionList = annotationClient.getAnnotationConnections(datasetId, limit=100000)

    # We need at least one annotation
    if len(connectionList) == 0:
        return

    edges = np.array([[connection['parentId'], connection['childId']] for connection in connectionList])
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

    parser.add_argument('--datasetId', type=str, required=True, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    main(args.datasetId, args.apiUrl, args.token, json.loads(args.parameters))
