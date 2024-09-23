import base64
import argparse
import json
import sys
import os  # Add this import
import time
import timeit
from datetime import datetime
from typing import List, Dict, Optional
import io
from anthropic import Anthropic

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

# Read the system prompt from the file
with open('system_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Output JSON filename': {
            'type': 'text',
            'default': 'output.json'
        },
        'Query': {
            'type': 'text',
        }
    }

    # Only add the API key field if the environment variable is not set
    if not os.getenv('ANTHROPIC_API_KEY'):
        interface['Claude API key'] = {
            'type': 'text'
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

    # Use the environment variable if set, otherwise use the provided key
    api_key = os.getenv('ANTHROPIC_API_KEY') or workerInterface['Claude API key']
    output_json_filename = workerInterface['Output JSON filename']
    query = workerInterface['Query']
    
    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)
    

    print("Input parameters: ", api_key, output_json_filename, query)

    annotationList = annotationClient.getAnnotationsByDatasetId(datasetId)
    connectionList = annotationClient.getAnnotationConnections(datasetId)
    propertyList = annotationClient.getPropertyValuesForDataset(datasetId) # Not sure how to get property names out, unfortunately.

    json_data = convert_nimbus_objects_to_JSON(annotationList, connectionList, propertyList)

    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)

    # Get the tags from the data so that the AI knows how to manipulate them.
    tag_string = JSON_data_tags_to_prompt_string(json_data)
    user_message = query + " " + tag_string

    print("User message: ", user_message)

    sendProgress(0.5, 'Generating code', 'Generating analysis code from Claude')

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1103,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    }
                ]
            }
        ]
    )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_json_filename = f"Claude input {current_time}.json"

    # Save input JSON data
    input_json_string = json.dumps(json_data, indent=2)
    input_json_stream = io.StringIO(input_json_string)
    input_size = len(input_json_string)
    input_json_stream.seek(0)

    sendProgress(0.4, 'Saving input data', f"Saving {input_json_filename} to dataset folder")

    # Get the dataset folder
    folder = annotationClient.client.getFolder(datasetId)

    # Upload input JSON content to the file
    annotationClient.client.uploadStreamToFolder(folder['_id'], input_json_stream, input_json_filename, input_size, mimeType="application/json")

    # Extract the Python code from the message and run it
    code = extract_python_code_from_string(message.content[0].text)
    sendProgress(0.75, 'Executing code', 'Executing the AI model code')

    # Create a dictionary to hold our variables
    local_vars = {'json_data': json_data}

    # Execute the code with the local variables
    exec(code, globals(), local_vars)

    # Retrieve the modified data
    output_json_data = local_vars.get('output_json_data', {})

    # This would be the potential end of a loop that would iteratively run the code if errors arose.

    # Document the process
    doc_filename = f"Claude output {current_time}.txt"

    documentation = f"""Input JSON Filename: {input_json_filename}
    Output JSON Filename: {output_json_filename}

    User Message:
    {user_message}

    Claude Output:
    {message.content[0].text}
    """

    # Convert documentation string to a stream
    doc_stream = io.StringIO(documentation)
    doc_size = len(documentation)
    doc_stream.seek(0)

    sendProgress(0.90, 'Creating documentation', f"Saving {doc_filename} to dataset folder")

    # Upload documentation content to the file
    # Get the dataset folder
    folder = annotationClient.client.getFolder(datasetId)
    annotationClient.client.uploadStreamToFolder(folder['_id'], doc_stream, doc_filename, doc_size, mimeType="text/plain")

    # Convert the output JSON data to a string
    output_json_string = json.dumps(output_json_data, indent=2)

    # Convert output JSON string to a stream
    json_stream = io.StringIO(output_json_string)
    size = len(output_json_string) # Get the length of the string as required by Girder
    json_stream.seek(0) # Reset the stream to the beginning

    sendProgress(0.95, 'Uploading file', f"Saving {output_json_filename} to dataset folder")

    # Upload output JSON content to the file
    annotationClient.client.uploadStreamToFolder(folder['_id'], json_stream, output_json_filename, size, mimeType="application/json")


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

    if propertyList:  # This section needs to be done.
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

    return output

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

def JSON_data_tags_to_prompt_string(data):
    # Extract unique tags from annotations
    unique_tags = set()
    for annotation in data.get('annotations', []):
        unique_tags.update(annotation.get('tags', []))
    
    # Sort the tags alphabetically
    sorted_tags = sorted(unique_tags)
    
    # Create the formatted string
    if sorted_tags:
        tags_string = ', '.join(f'"{tag}"' for tag in sorted_tags)
        result = f"The list of tags in the JSON is: {tags_string}"
    else:
        result = "There are no tags in the annotations."
    
    return result

def extract_python_code_from_string(text):
    # Split the text into lines
    lines = text.splitlines()
    
    # Extract the Python code
    code_lines = []
    in_code_block = False
    for line in lines:
        if in_code_block:
            if line.strip() == "```":
                break
            code_lines.append(line)
        elif line.strip() == "```python":
            in_code_block = True
        else:
            continue
    
    # Join the code lines into a single string
    code = '\n'.join(code_lines)
    
    return code


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