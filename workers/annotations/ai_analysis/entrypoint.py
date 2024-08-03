import base64
import argparse
import json
import sys
import random
import time
import timeit
from typing import List, Dict, Optional
import io

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendProgress

#import annotation_tools
import annotation_utilities.annotation_tools as annotation_tools

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Claude API key': {
            'type': 'text'
        },
        'Output JSON filename': {
            'type': 'text',
            'default': 'output.json'
        },
        'Query': {
            'type': 'text',
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)

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
    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print ("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)

    api_key = workerInterface['Claude API key']
    output_json_filename = workerInterface['Output JSON filename']
    query = workerInterface['Query']
    
    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)
    

    print("Input parameters: ", api_key, output_json_filename, query)

    annotationList = annotationClient.getAnnotationsByDatasetId(datasetId)
    connectionList = annotationClient.getAnnotationConnections(datasetId)
    propertyList = annotationClient.getPropertyValuesForDataset(datasetId) # Not sure how to get property names out, unfortunately.

    output_json_string = convert_nimbus_objects_to_JSON(annotationList, connectionList, propertyList)

    print("Output JSON string: ", output_json_string)

    # Convert JSON string to a stream
    json_stream = io.StringIO(output_json_string)
    size = len(output_json_string) # Get the length of the string as required by Girder
    json_stream.seek(0) # Reset the stream to the beginning

    sendProgress(0.95, 'Uploading file', 'Saving CSV file to dataset')

    # Get the dataset folder
    folder = annotationClient.client.getFolder(datasetId)

    # Upload JSON content to the file
    annotationClient.client.uploadStreamToFolder(folder['_id'], json_stream, output_json_filename, size, mimeType="application/json")
    
import json
from typing import List, Dict, Optional

def convert_nimbus_objects_to_JSON(annotationList: List[Dict], 
                                   connectionList: Optional[List[Dict]] = None, 
                                   propertyList: Optional[List[Dict]] = None, 
                                   filename: str = "output.json") -> None:
    output = {"annotations": [], "annotationConnections": [], "annotationProperties": [], "annotationPropertyValues": {}}

    for annotation in annotationList:
        ann_output = {
            "tags": annotation.get("tags", []),
            "shape": annotation.get("shape", ""),
            "channel": annotation.get("channel", 0),
            "location": annotation.get("location", {}),
            "coordinates": annotation.get("coordinates", []),
            "id": annotation.get("_id", ""),
            "datasetId": annotation.get("datasetId", "")
        }
        if "color" in annotation:
            ann_output["color"] = annotation["color"]
        output["annotations"].append(ann_output)

    if connectionList:
        for connection in connectionList:
            conn_output = {
                "label": connection.get("label", ""),
                "tags": connection.get("tags", []),
                "id": connection.get("_id", ""),
                "parentId": connection.get("parentId", ""),
                "childId": connection.get("childId", ""),
                "datasetId": connection.get("datasetId", "")
            }
            output["annotationConnections"].append(conn_output)

    if propertyList:
        # Add a dummy property
        output["annotationProperties"].append({
            "id": "dummy_property_id",
            "name": "Dummy Property",
            "image": "properties/dummy:latest",
            "tags": {"exclusive": False, "tags": ["dummy"]},
            "shape": "polygon",
            "workerInterface": {}
        })

        for prop in propertyList:
            ann_id = prop.get("annotationId", "")
            values = prop.get("values", {})
            if ann_id and values:
                output["annotationPropertyValues"][ann_id] = values

    json_string = json.dumps(output, indent=2)
    return json_string
    # with open(filename, 'w') as f:
    #     json.dump(output, f, indent=2)

def convert_JSON_to_nimbus_objects(filename: str) -> tuple:
    with open(filename, 'r') as f:
        data = json.load(f)

    annotationList = []
    for ann in data.get("annotations", []):
        annotation = {
            "_id": ann.get("id", ""),
            "tags": ann.get("tags", []),
            "shape": ann.get("shape", ""),
            "channel": ann.get("channel", 0),
            "location": ann.get("location", {}),
            "coordinates": ann.get("coordinates", []),
            "datasetId": ann.get("datasetId", "")
        }
        if "color" in ann:
            annotation["color"] = ann["color"]
        annotationList.append(annotation)

    connectionList = []
    for conn in data.get("annotationConnections", []):
        connection = {
            "_id": conn.get("id", ""),
            "label": conn.get("label", ""),
            "tags": conn.get("tags", []),
            "parentId": conn.get("parentId", ""),
            "childId": conn.get("childId", ""),
            "datasetId": conn.get("datasetId", "")
        }
        connectionList.append(connection)

    propertyList = []
    for ann_id, values in data.get("annotationPropertyValues", {}).items():
        property_value = {
            "annotationId": ann_id,
            "values": values
        }
        propertyList.append(property_value)

    return annotationList, connectionList, propertyList
    


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Generate random point annotations')

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
