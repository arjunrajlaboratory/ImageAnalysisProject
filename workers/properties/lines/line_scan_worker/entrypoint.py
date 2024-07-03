import argparse
import json
import sys
import numpy as np
import pandas as pd
import io

import annotation_client.workers as workers
import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
from annotation_client.utils import sendProgress

from scipy.ndimage import map_coordinates

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'All channels': {
            'type': 'checkbox',
            'default': True,
        },
        'Channel': {
            'type': 'channel',
            'default': 0,
        },
        'File name': {
            'type': 'text',
            'default': 'line_scan_output.csv',
        },
    }
    client.setWorkerImageInterface(image, interface)

def compute(datasetId, apiUrl, token, params):
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    all_channels = params['workerInterface']['All channels']
    selected_channel = params['workerInterface']['Channel']
    file_name = params['workerInterface']['File name']

    annotationList = workerClient.get_annotation_list_by_shape('line', limit=0)

    if len(annotationList) == 0:
        print("No line annotations found.")
        return

    results = []
    total_annotations = len(annotationList)

    # Variables to store the previous location and images
    prev_location = None
    images = {}

    for idx, annotation in enumerate(annotationList):
        sendProgress(idx / total_annotations, 'Processing annotations', f"Processing annotation {idx + 1}/{total_annotations}")

        location = annotation['location']
        time, z, xy = location['Time'], location['Z'], location['XY']
        current_location = (time, z, xy)

        # Check if the location has changed
        load_images = current_location != prev_location
        if load_images:
            prev_location = current_location
            images.clear()  # Clear the previous images

        if all_channels:
            channels = range(datasetClient.tiles['IndexRange']['IndexC'])
        else:
            channels = [selected_channel]

        for channel in channels:
            # Load the image if necessary
            if load_images:
                frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
                images[channel] = datasetClient.getRegion(datasetId, frame=frame)

            image = images[channel]

            coords_x = []
            coords_y = []

            for i in range(len(annotation['coordinates']) - 1):
                src = (annotation['coordinates'][i]['y'] - 0.5, annotation['coordinates'][i]['x'] - 0.5)
                dst = (annotation['coordinates'][i + 1]['y'] - 0.5, annotation['coordinates'][i + 1]['x'] - 0.5)

                num_points = int(np.hypot(dst[0] - src[0], dst[1] - src[1])) + 1
                if i < len(annotation['coordinates']) - 2:
                    x = np.linspace(src[1], dst[1], num_points, endpoint=False)
                    y = np.linspace(src[0], dst[0], num_points, endpoint=False)
                else:
                    x = np.linspace(src[1], dst[1], num_points)
                    y = np.linspace(src[0], dst[0], num_points)

                coords_x.extend(x)
                coords_y.extend(y)

            coords = np.vstack((coords_y, coords_x))
            profile = map_coordinates(np.squeeze(image), coords, order=1)

            results.append({
                'annotation_id': annotation['_id'],
                'channel': channel,
                'x': coords_x,
                'y': coords_y,
                'intensity': profile.tolist(),
                'tags': ','.join(annotation.get('tags', []))
            })

    sendProgress(0.9, 'Creating CSV', 'Generating CSV file')

    # Create DataFrame
    df_list = []
    for result in results:
        df_list.append(pd.DataFrame({
            'Annotation ID': result['annotation_id'],
            'Tags': result['tags'],
            'Header': 'X',
            'Values': ','.join(map(str, result['x']))
        }, index=[0]))
        df_list.append(pd.DataFrame({
            'Annotation ID': result['annotation_id'],
            'Tags': result['tags'],
            'Header': 'Y',
            'Values': ','.join(map(str, result['y']))
        }, index=[0]))
        df_list.append(pd.DataFrame({
            'Annotation ID': result['annotation_id'],
            'Tags': result['tags'],
            'Header': f'Channel {result["channel"]}',
            'Values': ','.join(map(str, result['intensity']))
        }, index=[0]))

    df = pd.concat(df_list, ignore_index=True)

    # Convert dataframe to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    size = len(csv_string)

    csv_buffer.seek(0)

    sendProgress(0.95, 'Uploading file', 'Saving CSV file to dataset')

    # Get the dataset folder
    folder = annotationClient.client.getFolder(datasetId)

    # Upload the CSV content to the file
    annotationClient.client.uploadStreamToFolder(folder['_id'], csv_buffer, file_name, size, mimeType="text/csv")

    sendProgress(1.0, 'Finished', 'Line scan CSV creation completed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Line scan CSV worker')
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