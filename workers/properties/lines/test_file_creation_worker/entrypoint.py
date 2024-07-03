import argparse
import json
import sys
import pandas as pd
import io
import girder_client

import annotation_client.workers as workers
import annotation_client.annotations as annotations
from annotation_client.utils import sendProgress

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'File name': {
            'type': 'text',
            'default': 'test_output.csv',
        },
    }
    client.setWorkerImageInterface(image, interface)

def compute(datasetId, apiUrl, token, params):
    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    
    # Create a sample dataframe
    df = pd.DataFrame({
        'A': range(1, 11),
        'B': range(10, 0, -1),
        'C': ['test'] * 10
    })

    # Convert dataframe to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    size = len(csv_string) # Length of the buffer

    csv_buffer.seek(0) # Move to beginning of buffer

    # Get the file name from the interface
    file_name = params['workerInterface']['File name']

    sendProgress(0.5, 'Creating file', 'Generating CSV file')

    # Get the dataset folder
    folder = annotationClient.client.getFolder(datasetId)

    sendProgress(0.75, 'Uploading file', 'Saving CSV file to dataset')

    # Upload the CSV content to the file
    annotationClient.client.uploadStreamToFolder(folder['_id'], csv_buffer, file_name, size, mimeType="text/csv")

    sendProgress(1.0, 'Finished', 'File creation completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test file creation worker')
    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    params = json.loads(args.parameters)
    apiUrl = args.apiUrl
    token = args.token

    if args.request == 'compute':
        if not args.datasetId:
            print("Error: datasetId is required for compute request")
            sys.exit(1)
        compute(args.datasetId, apiUrl, token, params)
    elif args.request == 'interface':
        interface(params['image'], apiUrl, token)
    else:
        print(f"Error: Unknown request type '{args.request}'")
        sys.exit(1)