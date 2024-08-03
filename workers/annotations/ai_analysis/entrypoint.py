import base64
import argparse
import json
import sys
import random
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
        system="You are a code bot that writes code for a specific task. The code will work with a user-supplied JSON data that encodes information on annotation objects, connections between them, their properties, and the values of those properties. The code then performs computations on those objects and then writes them back out in a new JSON variable. The user will provide instructions for how they wish to have the code change the annotations, connections, or properties. Examples might be \"assign random colors to all 'cell' annotations\", which would refer to adding random color values to the annotation objects tagged with 'cell', or \"Connect Brightfield points to the closest nucleus\" which would make a connection object between annotations tagged with 'Brightfield point' and 'nucleus'. Colors should follow the hex pattern in the sample JSON (but are an optional field). The user prompt will supply the tags so that you have them at your disposal. You can assume the JSON will come to you in the form of a variable \"json_data\" that was generated in this way:\n\nwith open(file_path, 'r') as file:\n    return json.load(file)\n\nYou can return the JSON as JSON data in the variable \"output_json_data\", and the user will write to a file later.\n\nGenerally, try and use the shapely package for the shape-based computations. Supply just the code. Here is an example JSON called sample.json, included just to show you the schema; you will want to ensure the final JSON contains all parts of the schema, including annotations, connections, properties, and property values:\n\nsample.json:\n{\n    \"annotations\": [\n        {\n            \"tags\": [\n                \"cell\"\n            ],\n            \"shape\": \"polygon\",\n            \"channel\": 0,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 738.5,\n                    \"y\": 285\n                },\n                {\n                    \"x\": 723.5,\n                    \"y\": 266\n                },\n                {\n                    \"x\": 710.5,\n                    \"y\": 266\n                }\n            ],\n            \"id\": \"6692561b6fbb27ec7b06ea42\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"cell\"\n            ],\n            \"shape\": \"polygon\",\n            \"channel\": 0,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 891.5,\n                    \"y\": 192\n                },\n                {\n                    \"x\": 871.5,\n                    \"y\": 180\n                },\n                {\n                    \"x\": 871.5,\n                    \"y\": 158\n                }\n            ],\n            \"id\": \"6692562f6fbb27ec7b06ea51\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"nucleus\"\n            ],\n            \"shape\": \"polygon\",\n            \"channel\": 1,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 594.5,\n                    \"y\": 470\n                },\n                {\n                    \"x\": 589.5,\n                    \"y\": 469\n                },\n                {\n                    \"x\": 583.5,\n                    \"y\": 468\n                }\n            ],\n            \"id\": \"669256326fbb27ec7b06ea54\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"nucleus\"\n            ],\n            \"shape\": \"polygon\",\n            \"channel\": 1,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 702.5,\n                    \"y\": 282\n                },\n                {\n                    \"x\": 699.5,\n                    \"y\": 289\n                },\n                {\n                    \"x\": 692.5,\n                    \"y\": 295\n                }\n            ],\n            \"id\": \"669256356fbb27ec7b06ea57\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"Brightfield point\"\n            ],\n            \"shape\": \"point\",\n            \"channel\": 0,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 533.5,\n                    \"y\": 396,\n                    \"z\": 0\n                }\n            ],\n            \"id\": \"669256386fbb27ec7b06ea5a\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"Brightfield point\"\n            ],\n            \"shape\": \"point\",\n            \"channel\": 0,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 544.5,\n                    \"y\": 427,\n                    \"z\": 0\n                }\n            ],\n            \"id\": \"669256396fbb27ec7b06ea5d\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"YFP line\"\n            ],\n            \"shape\": \"line\",\n            \"channel\": 1,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 668.5,\n                    \"y\": 210,\n                    \"z\": 0\n                },\n                {\n                    \"x\": 679.5,\n                    \"y\": 215,\n                    \"z\": 0\n                },\n                {\n                    \"x\": 684.5,\n                    \"y\": 219,\n                    \"z\": 0\n                }\n            ],\n            \"id\": \"669256526fbb27ec7b06ea75\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"YFP line\"\n            ],\n            \"shape\": \"line\",\n            \"channel\": 1,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 519.5,\n                    \"y\": 372,\n                    \"z\": 0\n                },\n                {\n                    \"x\": 525.5,\n                    \"y\": 372,\n                    \"z\": 0\n                },\n                {\n                    \"x\": 543.5,\n                    \"y\": 372,\n                    \"z\": 0\n                }\n            ],\n            \"id\": \"6692565d6fbb27ec7b06ea78\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"tags\": [\n                \"colorcell\",\n                \"cell\"\n            ],\n            \"shape\": \"polygon\",\n            \"channel\": 0,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 432.5,\n                    \"y\": 208\n                },\n                {\n                    \"x\": 427.5,\n                    \"y\": 209\n                },\n                {\n                    \"x\": 425.5,\n                    \"y\": 215\n                }\n            ],\n            \"id\": \"6692636e6fbb27ec7b06f0d9\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\",\n            \"color\": \"#FA0C0C\"\n        },\n        {\n            \"tags\": [\n                \"colorcell\"\n            ],\n            \"shape\": \"polygon\",\n            \"channel\": 0,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 192.5,\n                    \"y\": 235\n                },\n                {\n                    \"x\": 189.5,\n                    \"y\": 230\n                },\n                {\n                    \"x\": 177.5,\n                    \"y\": 219\n                }\n            ],\n            \"id\": \"669263716fbb27ec7b06f0dc\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\",\n            \"color\": \"#FA0C0C\"\n        },\n    {\n            \"tags\": [\n                \"cell\"\n            ],\n            \"shape\": \"polygon\",\n            \"channel\": 0,\n            \"location\": {\n                \"Time\": 0,\n                \"XY\": 0,\n                \"Z\": 0\n            },\n            \"coordinates\": [\n                {\n                    \"x\": 536.2855076085133,\n                    \"y\": 360.3116341146355\n                },\n                {\n                    \"x\": 471.19288259081327,\n                    \"y\": 426.3757908490176\n                },\n                {\n                    \"x\": 578.0613714258432,\n                    \"y\": 443.8633617492953\n                }\n            ],\n            \"id\": \"6698c9486fbb27ec7b075ee3\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        }\n    ],\n    \"annotationConnections\": [\n        {\n            \"label\": \"(Connection) Lasso connect All to All\",\n            \"tags\": [\n                \"nucleus\"\n            ],\n            \"id\": \"6692573f6fbb27ec7b06eaaf\",\n            \"parentId\": \"6692562f6fbb27ec7b06ea51\",\n            \"childId\": \"669256386fbb27ec7b06ea5a\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        },\n        {\n            \"label\": \"(Connection) Lasso connect All to All\",\n            \"tags\": [\n                \"nucleus\"\n            ],\n            \"id\": \"6692573f6fbb27ec7b06eab0\",\n            \"parentId\": \"6692562f6fbb27ec7b06ea51\",\n            \"childId\": \"669256396fbb27ec7b06ea5d\",\n            \"datasetId\": \"6692500e6fbb27ec7b06e47a\"\n        }\n    ],\n    \"annotationProperties\": [\n        {\n            \"id\": \"669256da6fbb27ec7b06ea98\",\n            \"name\": \"cell Blob metrics\",\n            \"image\": \"properties/blob_metrics:latest\",\n            \"tags\": {\n                \"exclusive\": false,\n                \"tags\": [\n                    \"cell\"\n                ]\n            },\n            \"shape\": \"polygon\",\n            \"workerInterface\": {}\n        },\n        {\n            \"id\": \"6692570b6fbb27ec7b06eaa3\",\n            \"name\": \"nucleus Blob intensity measurements\",\n            \"image\": \"properties/blob_intensity:latest\",\n            \"tags\": {\n                \"exclusive\": false,\n                \"tags\": [\n                    \"nucleus\"\n                ]\n            },\n            \"shape\": \"polygon\",\n            \"workerInterface\": {\n                \"Channel\": 1\n            }\n        }\n    ],\n    \"annotationPropertyValues\": {\n        \"6692561b6fbb27ec7b06ea42\": {\n            \"669256da6fbb27ec7b06ea98\": {\n                \"Area\": 5840,\n                \"Centroid\": {\n                    \"x\": 678.4908105022831,\n                    \"y\": 279.0122431506849\n                },\n                \"Circularity\": 0.8205950927590248,\n                \"Compactness\": 0.8205950927590248,\n                \"Convexity\": 0.9756912538635034,\n                \"Eccentricity\": 0.8007082748936661,\n                \"Elongation\": 0.3771186440677967,\n                \"Fractal_Dimension\": 1.3146444632102863,\n                \"Perimeter\": 299.0521284760509,\n                \"Rectangularity\": 0.7680445059379685,\n                \"Solidity\": 1.0075347154358751\n            }\n        },\n        \"669256326fbb27ec7b06ea54\": {\n            \"6692570b6fbb27ec7b06eaa3\": {\n                \"25thPercentileIntensity\": 4843,\n                \"75thPercentileIntensity\": 27636,\n                \"MaxIntensity\": 47803,\n                \"MeanIntensity\": 15642.933621933622,\n                \"MedianIntensity\": 8704,\n                \"MinIntensity\": 3333,\n                \"TotalIntensity\": 10840553\n            }\n        }\n    }\n}\n\n",
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
    doc_filename = f"Claude output {current_time}.txt"

    documentation = f"""User Message:
    {user_message}

    Claude Output:
    {message.content[0].text}
    """

    # Convert documentation string to a stream
    doc_stream = io.StringIO(documentation)
    doc_size = len(documentation)
    doc_stream.seek(0)

    sendProgress(0.90, 'Creating documentation', 'Saving documentation file to dataset folder')

    # Upload documentation content to the file
    # Get the dataset folder
    folder = annotationClient.client.getFolder(datasetId)
    annotationClient.client.uploadStreamToFolder(folder['_id'], doc_stream, doc_filename, doc_size, mimeType="text/plain")

    # Extract the Python code from the message and run it

    code = extract_python_code_from_string(message.content[0].text)

    sendProgress(0.75, 'Executing code', 'Executing the AI model code')

    # Create a dictionary to hold our variables
    local_vars = {'json_data': json_data}

    # Execute the code with the local variables
    exec(code, globals(), local_vars)

    # Retrieve the modified data
    output_json_data = local_vars.get('output_json_data', {})

    # Convert the JSON data to a string
    output_json_string = json.dumps(output_json_data, indent=2)

    # Convert JSON string to a stream
    json_stream = io.StringIO(output_json_string)
    size = len(output_json_string) # Get the length of the string as required by Girder
    json_stream.seek(0) # Reset the stream to the beginning

    sendProgress(0.95, 'Uploading file', 'Saving JSON file to dataset folder')

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

    # json_string = json.dumps(output, indent=2)
    #return json_string


    # with open(filename, 'w') as f:
    #     json.dump(output, f, indent=2)

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
