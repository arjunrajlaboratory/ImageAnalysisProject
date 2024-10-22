import argparse
import json
import sys

import annotation_client.annotations as annotations
import annotation_client.workers as workers
from annotation_client.utils import sendProgress

import pandas as pd
import numpy as np

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Count connected objects': {
            'type': 'notes',
            'value': 'This tool counts the number of children objects that are connected to a parent polygon. '
                     'It can be helpful for counting, for instance, the number of spots connected to a nucleus.',
            'displayOrder': 0,
        },
        'Child Tags': {
            'type': 'tags',
            'displayOrder': 1,
        },
        'Child Tags Exclusive': {
            'type': 'select',
            'items': ['Yes', 'No'],
            'default': 'No',
            'displayOrder': 2,
        },
    }
    client.setWorkerImageInterface(image, interface)

def compute(datasetId, apiUrl, token, params):
    propertyId = params.get('id', 'unknown_property')
    workerInterface = params['workerInterface']
    
    parent_tags = set(params.get('tags', {}).get('tags', []))
    parent_exclusive = params.get('tags', {}).get('exclusive', False)
    
    child_tags = set(workerInterface.get('Child Tags', []))
    print(workerInterface)
    child_exclusive = workerInterface['Child Tags Exclusive'] == 'Yes'

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)

    sendProgress(0.1, 'Fetching data', 'Getting all annotations')
    all_annotations = workerClient.get_annotation_list_by_shape(None, limit=0)
    
    sendProgress(0.3, 'Fetching data', 'Getting all connections')
    all_connections = annotationClient.getAnnotationConnections(datasetId, limit=10000000)

    sendProgress(0.5, 'Processing data', 'Filtering annotations and connections')
    
    def filter_annotations(annotations, tags, exclusive):
        if exclusive:
            return [ann for ann in annotations if set(ann['tags']) == tags]
        else:
            return [ann for ann in annotations if set(ann['tags']) & tags]

    parent_annotations = filter_annotations(all_annotations, parent_tags, parent_exclusive)
    child_annotations = filter_annotations(all_annotations, child_tags, child_exclusive)

    parent_ids = set(ann['_id'] for ann in parent_annotations)
    child_ids = set(ann['_id'] for ann in child_annotations)

    filtered_connections = [
        conn for conn in all_connections 
        if conn['parentId'] in parent_ids and conn['childId'] in child_ids
    ]

    sendProgress(0.7, 'Computing', 'Counting children')

    df = pd.DataFrame(filtered_connections)
    children_count = df.groupby('parentId').size().reset_index(name='count')
    children_count_dict = dict(zip(children_count['parentId'], children_count['count']))

    property_value_dict = {
        parent_id: children_count_dict.get(parent_id, 0) 
        for parent_id in parent_ids
    }

    dataset_property_value_dict = {datasetId: property_value_dict}

    sendProgress(0.9, 'Finishing', 'Sending computed counts to the server')
    workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)

    sendProgress(1.0, 'Complete', 'Property worker finished successfully')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute children count for annotations')

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