import argparse
import json
import sys

import annotation_client.workers as workers
from annotation_client.utils import sendProgress

import annotation_utilities.annotation_tools as annotation_tools

from shapely.geometry import Polygon, Point
import numpy as np
import geopandas as gpd


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Blob Overlap': {
            'type': 'notes',
            'value': 'This tool computes the overlaps between two sets of annotations. '
                     'The overlap is computed as the area of the intersection divided by the area of the individual annotations.',
            'displayOrder': 0,
        },
        'Annotations to compute overlap with': {
            'type': 'tags',
            'displayOrder': 1,
        },
        'Compute reverse overlaps': {
            'type': 'checkbox',
            'value': True,
            'displayOrder': 2,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def extract_spatial_annotation_data(obj_list):
    data = []
    for obj in obj_list:
        shape = obj['shape']
        coords = obj['coordinates']

        if shape == 'point':
            geometry = Point(coords[0]['x'], coords[0]['y'])
        elif shape == 'polygon':
            if len(coords) < 3:
                continue
            geometry = Polygon([(pt['x'], pt['y']) for pt in coords])

        data.append({
            '_id': obj['_id'],
            'geometry': geometry,
            'Time': obj['location']['Time'],
            'XY': obj['location']['XY'],
            'Z': obj['location']['Z']
        })

    # Create GeoDataFrame directly
    gdf = gpd.GeoDataFrame(data, geometry='geometry')
    return gdf


def compute(datasetId, apiUrl, token, params):
    workerClient = workers.UPennContrastWorkerClient(
        datasetId, apiUrl, token, params)

    workerInterface = params['workerInterface']
    overlap_tags = set(workerInterface.get(
        'Annotations to compute overlap with', None))
    compute_reverse_overlaps = workerInterface['Compute reverse overlaps']

    # Get all polygon annotations
    annotationList = workerClient.get_annotation_list_by_shape(
        'polygon', limit=0)

    if len(annotationList) == 0:
        return

    # Filter annotations based on tags
    annotation1List = annotation_tools.get_annotations_with_tags(
        annotationList,
        params.get('tags', {}).get('tags', []),
        params.get('tags', {}).get('exclusive', False)
    )
    annotation2List = annotation_tools.get_annotations_with_tags(
        annotationList, overlap_tags, exclusive=False
    )

    if len(annotation1List) == 0 or len(annotation2List) == 0:
        return

    # Convert annotations to GeoDataFrames once
    gdf1 = extract_spatial_annotation_data(annotation1List)
    gdf2 = extract_spatial_annotation_data(annotation2List)

    # Group by location
    grouped = gdf1.groupby(['Time', 'XY', 'Z'])
    total_groups = len(grouped)
    processed_groups = 0

    property_value_dict = {}

    # Process each location group
    for (t, xy, z), group1 in grouped:
        # Filter second GeoDataFrame for matching location
        mask = (gdf2['Time'] == t) & (gdf2['XY'] == xy) & (gdf2['Z'] == z)
        group2 = gdf2[mask]

        if group2.empty:
            continue

        # Compute forward overlaps
        intersections = gpd.overlay(group1, group2, how='intersection')
        if not intersections.empty:
            for idx, row in group1.iterrows():
                # Filter intersections for current source annotation
                mask = intersections.geometry.intersects(row.geometry)
                if mask.any():
                    total_overlap = intersections[mask].area.sum(
                    ) / row.geometry.area
                    if compute_reverse_overlaps:
                        property_value_dict[row['_id']] = {
                            f'Overlap_{"_".join(overlap_tags)}': float(total_overlap),
                            f'Overlap_{"_".join(params.get("tags", {}).get("tags", []))}': 0.0,
                        }
                    else:
                        property_value_dict[row['_id']] = {
                            f'Overlap_{"_".join(overlap_tags)}': float(total_overlap)
                        }

        # Compute reverse overlaps if requested
        if compute_reverse_overlaps:
            intersections = gpd.overlay(group2, group1, how='intersection')
            if not intersections.empty:
                for idx, row in group2.iterrows():
                    mask = intersections.geometry.intersects(row.geometry)
                    if mask.any():
                        total_overlap = intersections[mask].area.sum(
                        ) / row.geometry.area
                        property_value_dict[row['_id']] = {
                            f'Overlap_{"_".join(overlap_tags)}': 0.0,
                            f'Overlap_{"_".join(params.get("tags", {}).get("tags", []))}': float(total_overlap)
                        }

        processed_groups += 1
        sendProgress(processed_groups/total_groups,
                     'Computing overlaps',
                     f"Processing group {processed_groups}/{total_groups}")

    # Send results to server
    dataset_property_value_dict = {datasetId: property_value_dict}
    sendProgress(1.0, 'Done computing',
                 'Sending computed metrics to the server')
    workerClient.add_multiple_annotation_property_values(
        dataset_property_value_dict)


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

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
