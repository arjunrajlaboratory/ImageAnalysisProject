import base64
import argparse
import json
import sys
import os  # Add this import
import time
import timeit
import random
import math
from datetime import datetime
from typing import List, Dict, Optional
import pprint
import io
from anthropic import Anthropic

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendProgress

import annotation_utilities.annotation_tools as annotation_tools

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, Polygon, LineString
from shapely import ops
from scipy.spatial import cKDTree
from scipy import stats, optimize, interpolate

import property_handling

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
        },
        'AI Property Name': {
            'type': 'text',
            'default': 'AI properties'
        }
    }

    # Only add the API key field if the environment variable is not set
    if not os.getenv('ANTHROPIC_API_KEY'):
        interface['Claude API key'] = {
            'type': 'text'
        }

    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)

def get_all_dataset_properties(annotation_client, datasetId):
    """
    Get all the properties for the dataset.
    """
    property_info = []
    property_ids = set()

    # Get all configurations for the dataset
    configurationList = annotation_client.getDatasetViewsByDatasetId(datasetId)

    # Collect all propertyIds from all configurations
    for config in configurationList:
        configId = config['configurationId']
        configuration = annotation_client.getItemById(configId)
        property_ids.update(configuration['meta'].get('propertyIds', []))

    # Fetch details for each property
    for prop_id in property_ids:
        prop = annotation_client.getPropertyById(prop_id)
        
        detail = {
            "_id": prop["_id"],
            "name": prop["name"],
            "image": prop.get("image", ""),
            "tags": prop.get("tags", {}).get("tags", []),
            "shape": prop.get("shape", ""),
            "workerInterface": prop.get("workerInterface", {})
        }

        property_info.append(detail)

    return property_info

def get_ai_property_id(annotation_client, property_info, ai_property_name):
    """
    Checks if a property with the name ai_property_name exists.
    If it exists, returns its _id.
    Otherwise, creates a new property with the name ai_property_name and returns the new _id.
    """
    for prop in property_info:
        if prop['name'] == ai_property_name:
            return prop['_id']

    # Define the new property since it doesn't exist
    new_property = {
        "image": "properties/none:latest",
        "name": ai_property_name,
        "shape": "polygon",
        "tags": {
            "exclusive": False,
            "tags": []
        },
        "workerInterface": {}
    }

    # Create the new property using the annotation client
    new_prop = annotation_client.createNewProperty(new_property)

    # Return the _id of the newly created property
    return new_prop['_id']

def add_ai_property_to_all_configurations(annotation_client, datasetId, ai_property_id):
    # Get all configurations for the dataset
    configurationList = annotation_client.getDatasetViewsByDatasetId(datasetId)

    # Iterate through each configuration
    for config in configurationList:
        configId = config['configurationId']
        # Get the properties for the current configuration
        properties = annotation_client.getItemById(configId)

        # Check if the AI property is already in the properties
        if ai_property_id not in properties['meta']['propertyIds']:
            print(f"Adding AI property to configuration {configId}")
            # Add the AI property to the configuration
            properties['meta']['propertyIds'].append(ai_property_id)
            # Update the configuration with the new property
            annotation_client.setPropertiesByConfigurationId(configId, properties['meta']['propertyIds'])
        else:
            print(f"AI property already in configuration {configId}")

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
    ai_property_name = workerInterface['AI Property Name']

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)


    print("Input parameters: ", api_key, output_json_filename, query)

    sendProgress(0.1, 'Getting information', 'Getting annotations, connections, and property values from the dataset')
    annotationList = annotationClient.getAnnotationsByDatasetId(datasetId)
    connectionList = annotationClient.getAnnotationConnections(datasetId)
    propertyValueList = annotationClient.getPropertyValuesForDataset(datasetId)
    propertyList = get_all_dataset_properties(annotationClient, datasetId)
    # Using the propertyValueList, we can get the property names and make a dataframe that is easier for the AI to reason about.
    property_descriptions = property_handling.get_property_info(annotationClient, propertyValueList)
    property_id_to_name, property_name_to_id = property_handling.create_property_mappings(property_descriptions)
    df = property_handling.create_dataframe_from_annotations(propertyValueList, property_id_to_name, annotationList)

    dictionary_data = convert_nimbus_objects_to_dictionary(annotationList, connectionList, propertyValueList)
    dictionary_data['annotationProperties'] = propertyList
    dictionary_data['df'] = df

    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)

    # Read the system prompt from the file
    with open('/system_prompt.txt', 'r') as file:
        SYSTEM_PROMPT = file.read()

    # Get the tags from the data so that the AI knows how to manipulate them.
    tag_string = JSON_data_tags_to_prompt_string(dictionary_data)
    user_message = query + "\n\n" + tag_string

    # TODO: Update this to use the dataframe column names. # Done I think, although perhaps want to put in the JSON at the end.
    # property_string = pprint.pformat(property_handling.get_property_info(annotationClient, propertyList), indent=2)
    # user_message = user_message + "\n\n" + "The properties available to you are:\n" + property_string

    # Give the tag to column and column to tag mappings.
    tag_to_columns, column_to_tags = property_handling.create_tag_column_mappings(df)
    user_message = user_message + "\n\n" + "The tag to column mapping of property values is: " + pprint.pformat(tag_to_columns, indent=2)
    user_message = user_message + "\n\n" + "The column to tag mapping of property values is: " + pprint.pformat(column_to_tags, indent=2)

    # Give the head of the dataframe.
    user_message = user_message + "\n\n" + "The head of the dataframe is: \n" + df.head().to_string()

    # Give the column names of the dataframe.
    user_message = user_message + "\n\n" + "The column names of the dataframe are: " + ", ".join(df.columns)

    original_columns = df.columns.tolist()  # We will need this later to distinguish between the old columns and the new ones.

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

    # TODO: Save the input JSON data. Need to skip the df in order to save it properly because you can't JSON serialize a dataframe.
    # sendProgress(0.4, 'Saving input data', f"Saving {input_json_filename} to dataset folder")
    # # Save input JSON data
    # input_json_string = json.dumps(json_data, indent=2)
    # input_json_stream = io.StringIO(input_json_string)
    # input_size = len(input_json_string)
    # input_json_stream.seek(0)

    # # Get the dataset folder
    # folder = annotationClient.client.getFolder(datasetId)

    # # Upload input JSON content to the file
    # annotationClient.client.uploadStreamToFolder(folder['_id'], input_json_stream, input_json_filename, input_size, mimeType="application/json")

    # Extract the Python code from the message and run it
    code = extract_python_code_from_string(message.content[0].text)

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

    sendProgress(0.60, 'Creating documentation', f"Saving {doc_filename} to dataset folder")

    # Upload documentation content to the file
    # Get the dataset folder
    folder = annotationClient.client.getFolder(datasetId)
    annotationClient.client.uploadStreamToFolder(folder['_id'], doc_stream, doc_filename, doc_size, mimeType="text/plain")
                                                 
    sendProgress(0.75, 'Executing code', 'Executing the AI model code')

    # Use a copy of the current globals and add/override as needed
    exec_namespace = globals().copy()
    exec_namespace.update({
        "dictionary_data": dictionary_data
        # Add or override specific names if necessary
    })

    # Execute the code with the combined namespace
    # pprint.pprint(json_data)
    print("Code: ", code)
    try:
        exec(code, exec_namespace)
    except Exception as e:
        print(f"Error executing code: {e}")
        raise

    # Retrieve the modified data
    dictionary_data = exec_namespace.get('dictionary_data', {})

    # This would be the potential end of a loop that would iteratively run the code if errors arose.

    # Let's add the AI property to all configurations.
    all_properties = get_all_dataset_properties(annotationClient, datasetId)
    ai_property_id = get_ai_property_id(annotationClient, all_properties, ai_property_name)
    add_ai_property_to_all_configurations(annotationClient, datasetId, ai_property_id)

    # Create a new DataFrame with only the new columns so the new ones can be processed for upload.
    new_columns = [col for col in df.columns if col not in original_columns]
    new_df = df[new_columns]

    # Update the annotations, connections, and property values in the database
    sendProgress(0.80, 'Updating annotations, connections, and property values', 'Updating the annotations, connections, and property values in the database')
    update_annotations_connections_propertyvalues(
        annotationClient,
        dictionary_data['annotations'],
        dictionary_data['annotationConnections'],
        dictionary_data['annotationPropertyValues'],
        new_df,
        ai_property_id,
        datasetId
    )

    # Save the updated property list to the dictionary data.
    propertyList = get_all_dataset_properties(annotationClient, datasetId)
    dictionary_data['annotationProperties'] = propertyList

    # TODO: Save the output JSON data. Need to remove the df in order to save it.
    # # Convert the output JSON data to a string
    # output_json_string = json.dumps(output_json_data, indent=2)

    # # Convert output JSON string to a stream
    # json_stream = io.StringIO(output_json_string)
    # size = len(output_json_string) # Get the length of the string as required by Girder
    # json_stream.seek(0) # Reset the stream to the beginning

    # sendProgress(0.95, 'Uploading file', f"Saving {output_json_filename} to dataset folder")

    # # Upload output JSON content to the file
    # annotationClient.client.uploadStreamToFolder(folder['_id'], json_stream, output_json_filename, size, mimeType="application/json")

def update_annotations_connections_propertyvalues(
    annotationClient, new_annotation_list, new_connection_list,
    new_property_value_list, df, ai_property_id, datasetId
):
    # 1. Remove _id from annotations and add datasetId.
    # Keep a list of the old annotation ids to map to the newly generated ids later.
    new_annotation_ids = []
    for ann in new_annotation_list:
        new_annotation_ids.append(ann.pop('_id', None))
        ann['datasetId'] = datasetId

    # 2. Remove _id from connections and add datasetId
    for conn in new_connection_list:
        conn.pop('_id', None)
        conn['datasetId'] = datasetId

    # 3. Delete all existing annotations (this will also delete associated connections and property values)
    existingAnnotations = annotationClient.getAnnotationsByDatasetId(datasetId)
    existingAnnotationIds = [ann['_id'] for ann in existingAnnotations]
    annotationClient.deleteMultipleAnnotations(existingAnnotationIds)

    # 4. Upload new annotations and keep the return value
    new_annotations = annotationClient.createMultipleAnnotations(new_annotation_list)

    # 5. Map initial _ids to newly generated _ids from server
    id_mapping = {new_ann_id: new_ann['_id'] for new_ann_id, new_ann in zip(new_annotation_ids, new_annotations)}

    # 6. Update parentId and childId in connections to match the new _ids from the server
    for conn in new_connection_list:
        new_parent_id = id_mapping.get(conn['parentId'])
        new_child_id = id_mapping.get(conn['childId'])
        
        if new_parent_id is None:
            print(f"Error: No matching ID in new connections for parentId {conn['parentId']}")
        else:
            conn['parentId'] = new_parent_id
        
        if new_child_id is None:
            print(f"Error: No matching ID in new connections for childId {conn['childId']}")
        else:
            conn['childId'] = new_child_id

    # 7. Upload new connections
    new_connections = annotationClient.createMultipleConnections(new_connection_list)

    # 8. Update property values
    for prop_value in new_property_value_list:
        # Remove the '_id' field
        prop_value.pop('_id', None)
        
        # Remap the 'annotationId'
        old_annotation_id = prop_value['annotationId']
        new_annotation_id = id_mapping.get(old_annotation_id)
        
        if new_annotation_id is None:
            print(f"Error: No matching ID in new annotations for annotationId {old_annotation_id}")
        else:
            prop_value['annotationId'] = new_annotation_id

    # 9. Upload new versions of the original property values
    new_property_values = annotationClient.addMultipleAnnotationPropertyValues(new_property_value_list)

    # 10. Now let's collect the new dataframe property values, make them into a list of propertyValues, and upload them.
    # First, we need to convert the annotationIds to the new ids in the df.
    df = property_handling.convert_annotation_ids_to_new_ids(df, id_mapping)
    prop_vals = property_handling.convert_columns_to_property_values(df, datasetId, ai_property_id)
    annotationClient.deleteAnnotationPropertyValues(ai_property_id,datasetId)  # First delete the old property values.
    prop_vals = annotationClient.addMultipleAnnotationPropertyValues(prop_vals)  # Then add the new property values.

    return new_annotations, new_connections, new_property_values

def convert_nimbus_objects_to_dictionary(annotationList: List[Dict], 
                                   connectionList: Optional[List[Dict]] = None, 
                                   propertyValueList: Optional[List[Dict]] = None, 
                                   filename: str = "output.json") -> None:
    output = {"annotations": [], "annotationConnections": [], "annotationPropertyValues": {}}

    for annotation in annotationList:
        ann_output = {
            "tags": annotation.get("tags", []),
            "shape": annotation.get("shape", ""),
            "channel": annotation.get("channel", 0),
            "location": annotation.get("location", {}),
            "coordinates": annotation.get("coordinates", []),
            "_id": annotation.get("_id", ""),
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
                "_id": connection.get("_id", ""),
                "parentId": connection.get("parentId", ""),
                "childId": connection.get("childId", ""),
                "datasetId": connection.get("datasetId", "")
            }
            output["annotationConnections"].append(conn_output)

    if propertyValueList:  # This section needs to be done.
        # Add a dummy property
        # output["annotationProperties"].append({
        #     "_id": "dummy_property_id",
        #     "name": "Dummy Property",
        #     "image": "properties/dummy:latest",
        #     "tags": {"exclusive": False, "tags": ["dummy"]},
        #     "shape": "polygon",
        #     "workerInterface": {}
        # })
        output["annotationPropertyValues"] = propertyValueList

    return output

def convert_JSON_to_nimbus_objects(filename: str) -> tuple:
    with open(filename, 'r') as f:
        data = json.load(f)

    annotationList = []
    for ann in data.get("annotations", []):
        annotation = {
            "_id": ann.get("_id", ""),
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
            "_id": conn.get("_id", ""),
            "label": conn.get("label", ""),
            "tags": conn.get("tags", []),
            "parentId": conn.get("parentId", ""),
            "childId": conn.get("childId", ""),
            "datasetId": conn.get("datasetId", "")
        }
        connectionList.append(connection)

    propertyValueList = []
    for ann_id, values in data.get("annotationPropertyValues", {}).items():
        property_value = {
            "annotationId": ann_id,
            "values": values
        }
        propertyValueList.append(property_value)

    return annotationList, connectionList, propertyValueList

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
        result = f"The list of tags available to you in the JSON is: {tags_string}"
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