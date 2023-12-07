import argparse
import json
import sys

import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress
import annotation_client.workers as workers


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        # 'Tags': {
        #     'type': 'tags'
        # },
        # 'Exclusive': {
        #     'type': 'select',
        #     'items': ['Yes', 'No'],
        #     'default': 'Yes'
        # },
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

    # Not sure what the below line does, actually.
    # connectionIds = params.get('connectionIds', None)

    workerInterface = params['workerInterface']
    # We actualky don't use the custom worker interface, but if we did, we would use the line below.
    # tags = set(workerInterface.get('Tags', None))
    # Instead, we pull the tags from the params.
    tags = set(params.get('tags', None)['tags'])
    print("tags:")
    print(tags)
    
    # Below would be used if the worker interface had an "exclusive" option, but it doesn't.
    # Keeping here in case we want to add it later. For now, we just assume non-exclusive.
    # exclusive = workerInterface['Exclusive'] == 'Yes'
    exclusive = False

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    
    # connectionList = annotationClient.getAnnotationConnections(datasetId)
    # Not setting a limit on number of connections, but if you wanted to, you could do something like this:
    connectionList = annotationClient.getAnnotationConnections(datasetId, limit=10000000)

    # We need at least one connection
    if len(connectionList) == 0:
        return


    # This code makes a dictionary of all the annotation IDs and assigns them a unique integer value.
    # We will use the integer values as annotation IDs in the properties we create.
    id_to_integer_mapping = {}
    current_integer = 0
    id_to_integer_mapping[-1] = None # This is a special value that indicates no parent or child

    for connection in connectionList:
        for key in ['childId', 'parentId']:
            id_value = connection[key]
            if id_value not in id_to_integer_mapping:
                id_to_integer_mapping[id_value] = current_integer
                current_integer += 1

    number_connections = len(connectionList)
    processed_connections = 0

    # This is where we will store the list of annotations with connections. The key is the annotationID,
    # and the value is another list with childID and parentID as key-value pairs.
    annotationConnectionList = {}  # Initialize as a dictionary
    for connection in connectionList:
        child_tags = set(annotationClient.getAnnotationById(connection['childId'])['tags'])
        
        # if the child matches the tags, then add the parent to the list as a parent of the current child annotation.
        if (exclusive and (child_tags == tags)) or ((not exclusive) and (len(child_tags & tags) > 0)):
            if connection['childId'] not in annotationConnectionList: # If the child isn't already in the list, add it
                annotationConnectionList[connection['childId']] = { 'parentId': connection['parentId'], 'childId': -1 }
            else:
                annotationConnectionList[connection['childId']]['parentId'] = connection['parentId']

            # Just including the following to see if it solves the issue of crashing the interface due to string properties instead of float.
            # annotationConnectionList[connection['childId']] = { 'parentId': 1.0, 'childId': 2.0 }

        parent_tags = set(annotationClient.getAnnotationById(connection['parentId'])['tags'])
        
        # if the parent matches the tags, then add the child to the list as a child of the current parent annotation.
        if (exclusive and (parent_tags == tags)) or ((not exclusive) and (len(parent_tags & tags) > 0)):
            if connection['parentId'] not in annotationConnectionList:
                annotationConnectionList[connection['parentId']] = { 'parentId': -1, 'childId': connection['childId'] }
            else:
                annotationConnectionList[connection['parentId']]['childId'] = connection['childId']
            
            # Just including the following to see if it solves the issue of crashing the interface due to string properties instead of float.
            # annotationConnectionList[connection['parentId']] = { 'parentId': 1.0, 'childId': 2.0 }
        
        processed_connections += 1
        sendProgress(processed_connections / number_connections, 'Finding parent/children', f"Processing connection {processed_connections}/{number_connections}")

        for annotationId, prop in annotationConnectionList.items():
            annotation = annotationClient.getAnnotationById(annotationId)
            # Note that in the below, -1 maps to None, which is a special value that indicates no parent or child.
            addProp = {'annotationId': id_to_integer_mapping[annotationId], 'parentId': id_to_integer_mapping[prop['parentId']], 'childId': id_to_integer_mapping[prop['childId']]}
            workerClient.add_annotation_property_values(annotation, addProp)


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
