import argparse
import json
import sys
import timeit

import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning, sendError
import annotation_client.tiles as tiles

import annotation_utilities.annotation_tools as annotation_tools
import annotation_utilities.batch_argument_parser as batch_argument_parser
from annotation_utilities.progress import update_progress
import numpy as np
from skimage import draw
from collections import defaultdict
from shapely.geometry import Polygon


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Blob Annulus Intensity': {
            'type': 'notes',
            'value': 'This tool computes the pixel intensity in an annulus around the objects in the specified channel. '
                     'It will compute the mean, max, min, median, 25th percentile, and 75th percentile intensity, as well as the total intensity. '
                     'The size of the annulus is defined by the radius.',
            'displayOrder': 0,
        },
        'Channel': {
            'type': 'channel',
            'required': True,
            'tooltip': 'Compute pixel intensities in this channel.\n'
                       'The channel does not have to be the same as the layer the annotations are on.',
            'displayOrder': 1,
        },
        'Radius': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10,
            'unit': 'pixels',
            'displayOrder': 2,
        },
        'Z planes': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Z positions to compute intensities for (empty to use annotation plane)',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Z positions to compute intensities for. Leave blank to use the plane the annotations are on.'
            },
            'displayOrder': 3
        },
        'Additional percentiles': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 10, 45, 90 (empty for default)',
                'label': 'Percentiles to compute intensities for (leave empty for default of 25, 75)',
            },
            'displayOrder': 4
        }
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

    workerClient = workers.UPennContrastWorkerClient(
        datasetId, apiUrl, token, params)

    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    channel = params['workerInterface']['Channel']
    annulus_radius = float(params['workerInterface']['Radius'])

    additional_percentiles = params['workerInterface'].get(
        'Additional percentiles', None)

    # Parse the additional percentiles (floats) into a list and validate that they are between 0 and 100
    if additional_percentiles is not None:
        additional_percentiles = [float(x) for x in additional_percentiles.split(
            ',') if x.strip()]
        if any(x <= 0 or x >= 100 for x in additional_percentiles):
            sendWarning('Invalid additional percentiles',
                        info='Additional percentiles must be between 0 and 100.')

    # Let's validate the z-planes
    tileInfo = datasetClient.tiles

    # If there is an 'IndexRange' key in the tileClient.tiles, then
    # let's get a range for each of XY, Z, T, and C
    # Currently, we are just using the Z range, but the code is here in
    # case we want to use the XY, T, and C ranges in the future
    if 'IndexRange' in tileInfo:
        if 'IndexXY' in tileInfo['IndexRange']:
            range_xy = range(0, tileInfo['IndexRange']['IndexXY'])
        else:
            range_xy = [0]
        if 'IndexZ' in tileInfo['IndexRange']:
            range_z = range(0, tileInfo['IndexRange']['IndexZ'])
        else:
            range_z = [0]
        if 'IndexT' in tileInfo['IndexRange']:
            range_time = range(0, tileInfo['IndexRange']['IndexT'])
        else:
            range_time = [0]
        if 'IndexC' in tileInfo['IndexRange']:
            range_c = range(0, tileInfo['IndexRange']['IndexC'])
        else:
            range_c = [0]
    else:
        # If there is no 'IndexRange' key in the tileClient.tiles, then there is just one frame
        range_xy = [0]
        range_z = [0]
        range_time = [0]
        range_c = [0]

    # Get the Z planes from the worker interface
    # Old workers may not have this key, so we use get() with a default of None
    z_planes = params['workerInterface'].get('Z planes', None)
    if z_planes is not None and z_planes.strip():
        z_planes = list(batch_argument_parser.process_range_list(
            z_planes, convert_one_to_zero_index=True))
        # Find which planes are out of range
        invalid_planes = [x+1 for x in z_planes if x not in range_z]
        if invalid_planes:
            sendWarning('Requested planes out of range',
                        info=f'Excluding planes {", ".join(map(str, invalid_planes))}.')
        z_planes = [x for x in z_planes if x in range_z]
    else:
        # If no Z planes are specified, then we will use the plane from the annotations
        z_planes = None

    # Following line should be updated to get just the annotations with specified tags
    annotationList = workerClient.get_annotation_list_by_shape(
        'polygon', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, params.get(
        'tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))

    # We need at least one annotation
    if len(annotationList) == 0:
        sendWarning('No objects found',
                    info='No objects found. Please check the tags and shape.')
        return

    start_time = timeit.default_timer()

    number_annotations = len(annotationList)
    processed_annotations = 0  # For reporting progress
    property_value_dict = {}  # Initialize output dictionary

    # If no Z planes are specified, then we will use the plane from the annotations
    if z_planes is None:
        print("No Z planes specified, computing intensity at annotation locations")
        grouped_annotations = defaultdict(list)
        # First, group the annotations by their location
        # That way, we can load the image once and compute the properties for all
        # annotations at that location
        for annotation in annotationList:
            location_key = (annotation['location']['Time'],
                            annotation['location']['Z'], annotation['location']['XY'])
            grouped_annotations[location_key].append(annotation)

        # Now, we will loop over all the locations and compute the properties
        # for all annotations at that location.
        for location_key, annotations in grouped_annotations.items():
            time, z, xy = location_key
            frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
            image = datasetClient.getRegion(datasetId, frame=frame)

            if image is None:
                sendWarning('No image found',
                            info=f'No image found for frame {frame}.')
                continue

            # Compute properties for all annotations at that location
            for annotation in annotations:
                polygon = np.array([[coordinate['y'] - 0.5, coordinate['x'] - 0.5]
                                    for coordinate in annotation['coordinates']])

                if len(polygon) < 3:  # Skip if the polygon is not valid
                    sendWarning('Invalid polygon',
                                info=f'Object {annotation["_id"]} has less than 3 vertices.')
                    continue

                rr, cc = draw.polygon(
                    polygon[:, 0], polygon[:, 1], shape=image.shape)
                original_coords = set(zip(rr, cc))

                # Get coordinates of dilated polygon
                dilated_polygon = Polygon(polygon).buffer(annulus_radius)
                rr_dilated, cc_dilated = draw.polygon(
                    np.array(dilated_polygon.exterior.coords)[:, 0],
                    np.array(dilated_polygon.exterior.coords)[:, 1],
                    shape=image.shape
                )
                dilated_coords = set(zip(rr_dilated, cc_dilated))

                # Get just the annulus coordinates (in dilated but not in original)
                annulus_coords = dilated_coords - original_coords

                if len(annulus_coords) == 0:
                    sendWarning('No annulus coordinates found',
                                info=f'Object {annotation["_id"]} has no annulus coordinates.')
                    continue

                rr_annulus, cc_annulus = zip(*annulus_coords)
                intensities = image[rr_annulus, cc_annulus]

                if len(intensities) == 0:  # Skip if there are no pixels in the mask
                    sendWarning('No pixels in mask',
                                info=f'Object {annotation["_id"]} has no pixels in the mask.')
                    continue

                # Calculating the desired metrics
                mean_intensity = float(np.mean(intensities))
                max_intensity = float(np.max(intensities))
                min_intensity = float(np.min(intensities))
                median_intensity = float(np.median(intensities))
                q25_intensity = float(np.percentile(intensities, 25))
                q75_intensity = float(np.percentile(intensities, 75))
                total_intensity = float(np.sum(intensities))

                prop = {
                    'MeanIntensity': mean_intensity,
                    'MaxIntensity': max_intensity,
                    'MinIntensity': min_intensity,
                    'MedianIntensity': median_intensity,
                    '25thPercentileIntensity': q25_intensity,
                    '75thPercentileIntensity': q75_intensity,
                    'TotalIntensity': total_intensity,
                }

                # If there are additional percentiles, compute them and
                # add them to the property dictionary
                if additional_percentiles is not None:
                    for percentile in additional_percentiles:
                        prop_name = f'{percentile}thPercentileIntensity'
                        prop_value = float(np.percentile(intensities, percentile))
                        prop[prop_name] = prop_value

                property_value_dict[annotation['_id']] = prop
                processed_annotations += 1
                update_progress(processed_annotations, number_annotations,
                                "Computing annulusintensity")

    else:
        print("Z planes specified, computing intensity at specified planes")

        grouped_annotations = defaultdict(list)
        # First, group the annotations by their location, but exclude Z
        for annotation in annotationList:
            location_key = (annotation['location']['Time'],
                            annotation['location']['XY'])
            grouped_annotations[location_key].append(annotation)

        # Initialize the property dictionaries for all annotations before processing Z-planes
        for annotation in annotationList:
            property_value_dict[annotation['_id']] = {
                'MeanIntensity': {},
                'MaxIntensity': {},
                'MinIntensity': {},
                'MedianIntensity': {},
                '25thPercentileIntensity': {},
                '75thPercentileIntensity': {},
                'TotalIntensity': {},
            }

            # If there are additional percentiles, add them to the property dictionary
            if additional_percentiles is not None:
                for percentile in additional_percentiles:
                    prop_name = f'{percentile}thPercentileIntensity'
                    property_value_dict[annotation['_id']][prop_name] = {}

        # Your grouped_annotations code remains the same
        for location_key, annotations in grouped_annotations.items():
            time, xy = location_key
            for z in z_planes:
                # Create a formatted Z-key (e.g., "z001", "z002", etc.)
                z_key = f"z{(z+1):03d}"  # +1 for 1-based indexing in UI

                frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
                image = datasetClient.getRegion(datasetId, frame=frame)

                if image is None:
                    sendWarning('No image found', info=f'No image found for frame {frame}.')
                    continue

                # Process each annotation for this Z-plane
                for annotation in annotations:
                    polygon = np.array([[coordinate['y'] - 0.5, coordinate['x'] - 0.5]
                                        for coordinate in annotation['coordinates']])

                    if len(polygon) < 3:
                        sendWarning('Invalid polygon',
                                    info=f'Object {annotation["_id"]} has less than 3 vertices.')
                        continue

                    rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], shape=image.shape)
                    original_coords = set(zip(rr, cc))

                    # Get coordinates of dilated polygon
                    dilated_polygon = Polygon(polygon).buffer(annulus_radius)
                    rr_dilated, cc_dilated = draw.polygon(
                        np.array(dilated_polygon.exterior.coords)[:, 0],
                        np.array(dilated_polygon.exterior.coords)[:, 1],
                        shape=image.shape)
                    dilated_coords = set(zip(rr_dilated, cc_dilated))

                    # Get just the annulus coordinates (in dilated but not in original)
                    annulus_coords = dilated_coords - original_coords

                    if len(annulus_coords) == 0:
                        sendWarning('No annulus coordinates found',
                                    info=f'Object {annotation["_id"]} has no annulus coordinates.')
                        continue

                    rr_annulus, cc_annulus = zip(*annulus_coords)
                    intensities = image[rr_annulus, cc_annulus]

                    if len(intensities) == 0:
                        sendWarning('No pixels in mask',
                                    info=f'Object {annotation["_id"]} has no pixels in the mask.')
                        continue

                    # Update the nested dictionary for this annotation's properties at this Z-plane
                    annotation_id = annotation['_id']
                    property_value_dict[annotation_id]['MeanIntensity'][z_key] = float(
                        np.mean(intensities))
                    property_value_dict[annotation_id]['MaxIntensity'][z_key] = float(
                        np.max(intensities))
                    property_value_dict[annotation_id]['MinIntensity'][z_key] = float(
                        np.min(intensities))
                    property_value_dict[annotation_id]['MedianIntensity'][z_key] = float(
                        np.median(intensities))
                    property_value_dict[annotation_id]['25thPercentileIntensity'][z_key] = float(
                        np.percentile(intensities, 25))
                    property_value_dict[annotation_id]['75thPercentileIntensity'][z_key] = float(
                        np.percentile(intensities, 75))
                    property_value_dict[annotation_id]['TotalIntensity'][z_key] = float(
                        np.sum(intensities))

                    # If there are additional percentiles, compute them and
                    # add them to the property dictionary
                    if additional_percentiles is not None:
                        for percentile in additional_percentiles:
                            prop_name = f'{percentile}thPercentileIntensity'
                            prop_value = float(np.percentile(intensities, percentile))
                            property_value_dict[annotation_id][prop_name][z_key] = prop_value

                    processed_annotations += 1
                    # Adjust the total for progress reporting (annotations × z-planes)
                    total_operations = len(annotationList) * len(z_planes)
                    update_progress(processed_annotations, total_operations,
                                    "Computing annulus intensity")

    dataset_property_value_dict = {datasetId: property_value_dict}

    sendProgress(0.5, 'Done computing',
                 'Sending computed metrics to the server')
    workerClient.add_multiple_annotation_property_values(
        dataset_property_value_dict)

    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the code in: {execution_time} seconds")


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in an annulus around blob annotations')

    parser.add_argument('--datasetId', type=str,
                        required=False, action='store')
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
