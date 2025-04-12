import argparse
import json
import sys
import timeit
import pprint
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
import pandas as pd

# Import mahotas for texture features
import mahotas as mh

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Random Forest Classifier': {
            'type': 'notes',
            'value': 'This tool uses a Random Forest Classifier to classify blobs.',
            'displayOrder': 0,
        },
        'Buffer radius': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10,
            'unit': 'pixels',
            'displayOrder': 1,
            'tooltip': 'The buffer to use to generate an annulus around the blobs.'
        },
        # 'Add classification as tag': {
        #     'type': 'checkbox',
        #     'default': False,
        #     'displayOrder': 2,
        #     'tooltip': 'Add the classification as a tag to the annotations.'
        # },
        'Include texture features': {
            'type': 'checkbox',
            'default': True,
            'displayOrder': 3,
            'tooltip': 'Include Haralick and other texture features from Mahotas.'
        },
        'Texture scaling': {
            'type': 'number',
            'min': 1,
            'max': 255,
            'default': 8,
            'unit': 'levels',
            'displayOrder': 4,
            'tooltip': 'Number of gray levels to use for texture feature calculation. Lower values are faster but less detailed.'
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def compute_texture_features(image_patch, levels=8):
    """Compute Haralick and other texture features from Mahotas for an image patch.

    Args:
        image_patch: NumPy array of the image region
        levels: Number of gray levels to use (lower is faster)

    Returns:
        Dictionary of texture features
    """
    features = {}

    # Ensure the image is properly scaled for texture analysis
    if image_patch.size == 0:
        return {}

    # Rescale to 0-levels
    image_scaled = mh.stretch(image_patch, 0, levels-1).astype(np.uint8)

    # Skip if the patch is too small or has no variation
    if image_scaled.size <= 1 or np.std(image_scaled) < 0.01:
        return {}

    try:
        # Compute Haralick features (returns features for 4 directions)
        # We'll average across all directions for rotation invariance
        haralick_feats = mh.features.haralick(image_scaled, return_mean=True)

        # Add each Haralick feature with descriptive name
        feature_names = [
            'ASM', 'Contrast', 'Correlation', 'Variance',
            'IDM', 'SumAvg', 'SumVar', 'SumEntropy',
            'Entropy', 'DiffVar', 'DiffEntropy',
            'IMC1', 'IMC2'
        ]

        for i, name in enumerate(feature_names):
            features[f'Haralick_{name}'] = float(haralick_feats[i])

        # Add Local Binary Patterns (rotation invariant)
        lbp = mh.features.lbp(image_scaled, radius=2, points=8)
        for i, val in enumerate(lbp):
            features[f'LBP_{i}'] = float(val)

        # Add Zernike moments (rotation invariant)
        if image_scaled.shape[0] > 10 and image_scaled.shape[1] > 10:
            zernike = mh.features.zernike_moments(image_scaled, 8)
            for i, val in enumerate(zernike):
                features[f'Zernike_{i}'] = float(val)

        # Add Threshold Adjacency Statistics
        tas = mh.features.tas(image_scaled)
        for i, val in enumerate(tas):
            features[f'TAS_{i}'] = float(val)

    except Exception as e:
        print(f"Error computing texture features: {e}")
        return {}

    return features


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

    annulus_radius = float(params['workerInterface']['Buffer radius'])
    # TODO: Need to decide how to implement this feature.
    # add_classification_as_tag = params['workerInterface']['Add classification as tag']
    include_texture_features = params['workerInterface'].get('Include texture features', True)
    texture_scaling = int(params['workerInterface'].get('Texture scaling', 8))

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

    # Get the number of channels
    num_channels = len(range_c)

    # Base columns without channel-specific metrics
    base_columns = [
        'Area', 'Perimeter', 'CentroidX', 'CentroidY',
        'BoundingBoxWidth', 'BoundingBoxHeight', 'ConvexHullArea',
        'Solidity', 'Circularity', 'Extent', 'tags'
    ]

    # Create channel-specific column names
    channel_columns = []
    for c in range(num_channels):
        channel_metrics = [
            f'MeanIntensity_{c}', f'MaxIntensity_{c}', f'MinIntensity_{c}', f'MedianIntensity_{c}',
            f'Q10Intensity_{c}', f'Q25Intensity_{c}', f'Q75Intensity_{c}', f'Q90Intensity_{c}',
            f'TotalIntensity_{c}',
            f'MeanAnnulusIntensity_{c}', f'MaxAnnulusIntensity_{c}', f'MinAnnulusIntensity_{c}',
            f'MedianAnnulusIntensity_{c}', f'Q10AnnulusIntensity_{c}', f'Q25AnnulusIntensity_{c}',
            f'Q75AnnulusIntensity_{c}', f'Q90AnnulusIntensity_{c}', f'TotalAnnulusIntensity_{c}'
        ]
        channel_columns.extend(channel_metrics)

    # Prepare texture feature columns if enabled
    texture_columns = []
    if include_texture_features:
        # Add placeholders for texture features
        for c in range(num_channels):
            # Haralick features
            for name in ['ASM', 'Contrast', 'Correlation', 'Variance',
                         'IDM', 'SumAvg', 'SumVar', 'SumEntropy',
                         'Entropy', 'DiffVar', 'DiffEntropy', 'IMC1', 'IMC2']:
                texture_columns.append(f'Haralick_{name}_{c}')

            # LBP features (simplified, just a few key values)
            for i in range(10):  # We'll use the first 10 LBP bins
                texture_columns.append(f'LBP_{i}_{c}')

            # Zernike features (simplified)
            for i in range(5):  # First 5 Zernike moments
                texture_columns.append(f'Zernike_{i}_{c}')

            # TAS features
            for i in range(8):  # 8 TAS features
                texture_columns.append(f'TAS_{i}_{c}')

    # Initialize DataFrame with all columns
    all_columns = base_columns + channel_columns + texture_columns
    df = pd.DataFrame(columns=all_columns)

    for location_key, annotations in grouped_annotations.items():
        time, z, xy = location_key
        images = []
        for channel in range_c:
            frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
            images.append(datasetClient.getRegion(datasetId, frame=frame))

        if len(images) == 0:
            sendWarning('No image found',
                        info=f'No image found for frame {frame}.')
            continue

        # Compute properties for all annotations at that location
        for annotation in annotations:
            prop = {}  # Initialize the property dictionary

            polygon = np.array([[coordinate['y'] - 0.5, coordinate['x'] - 0.5]
                                for coordinate in annotation['coordinates']])

            if len(polygon) < 3:  # Skip if the polygon is not valid
                sendWarning('Invalid polygon',
                            info=f'Object {annotation["_id"]} has less than 3 vertices.')
                continue

            shapely_polygon = Polygon(polygon)

            minx, miny, maxx, maxy = shapely_polygon.bounds

            rr, cc = draw.polygon(
                polygon[:, 0], polygon[:, 1], shape=images[0].shape)
            original_coords = set(zip(rr, cc))

            # Get coordinates of dilated polygon
            dilated_polygon = shapely_polygon.buffer(annulus_radius)
            rr_dilated, cc_dilated = draw.polygon(
                np.array(dilated_polygon.exterior.coords)[:, 0],
                np.array(dilated_polygon.exterior.coords)[:, 1],
                shape=images[0].shape
            )
            dilated_coords = set(zip(rr_dilated, cc_dilated))

            # Get just the annulus coordinates (in dilated but not in original)
            annulus_coords = dilated_coords - original_coords

            if len(annulus_coords) == 0:
                sendWarning('No annulus coordinates found',
                            info=f'Object {annotation["_id"]} has no annulus coordinates.')
                continue

            rr_annulus, cc_annulus = zip(*annulus_coords)

            # Now let's get coordinates of the middle 20% of the polygon

            centroid = shapely_polygon.centroid
            boundary_distance = shapely_polygon.boundary.distance(centroid)
            erosion_buffer = -boundary_distance * 0.80
            eroded_polygon = shapely_polygon.buffer(erosion_buffer)
            rr_eroded, cc_eroded = draw.polygon(
                np.array(eroded_polygon.exterior.coords)[:, 0],
                np.array(eroded_polygon.exterior.coords)[:, 1],
                shape=images[0].shape
            )

            for ch_idx, image in enumerate(images):

                intensities = image[rr, cc]

                if len(intensities) == 0:
                    sendWarning('No intensities found',
                                info=f'Object {annotation["_id"]} has no intensities.')
                    continue

                annulus_intensities = image[rr_annulus, cc_annulus]

                if len(annulus_intensities) == 0:  # Skip if there are no pixels in the mask
                    sendWarning('No pixels in mask',
                                info=f'Object {annotation["_id"]} has no pixels in the mask.')
                    continue

                eroded_intensities = image[rr_eroded, cc_eroded]

                if len(eroded_intensities) == 0:
                    sendWarning('No eroded intensities found',
                                info=f'Object {annotation["_id"]} has no eroded intensities.')
                    continue

                # Calculating the desired metrics
                prop[f'MeanIntensity_{ch_idx}'] = float(np.mean(intensities))
                prop[f'MaxIntensity_{ch_idx}'] = float(np.max(intensities))
                prop[f'MinIntensity_{ch_idx}'] = float(np.min(intensities))
                prop[f'MedianIntensity_{ch_idx}'] = float(np.median(intensities))
                prop[f'Q10Intensity_{ch_idx}'] = float(np.percentile(intensities, 10))
                prop[f'Q25Intensity_{ch_idx}'] = float(np.percentile(intensities, 25))
                prop[f'Q75Intensity_{ch_idx}'] = float(np.percentile(intensities, 75))
                prop[f'Q90Intensity_{ch_idx}'] = float(np.percentile(intensities, 90))
                prop[f'TotalIntensity_{ch_idx}'] = float(np.sum(intensities))

                prop[f'MeanAnnulusIntensity_{ch_idx}'] = float(np.mean(annulus_intensities))
                prop[f'MaxAnnulusIntensity_{ch_idx}'] = float(np.max(annulus_intensities))
                prop[f'MinAnnulusIntensity_{ch_idx}'] = float(np.min(annulus_intensities))
                prop[f'MedianAnnulusIntensity_{ch_idx}'] = float(np.median(annulus_intensities))
                prop[f'Q10AnnulusIntensity_{ch_idx}'] = float(
                    np.percentile(annulus_intensities, 10))
                prop[f'Q25AnnulusIntensity_{ch_idx}'] = float(
                    np.percentile(annulus_intensities, 25))
                prop[f'Q75AnnulusIntensity_{ch_idx}'] = float(
                    np.percentile(annulus_intensities, 75))
                prop[f'Q90AnnulusIntensity_{ch_idx}'] = float(
                    np.percentile(annulus_intensities, 90))
                prop[f'TotalAnnulusIntensity_{ch_idx}'] = float(np.sum(annulus_intensities))

                prop[f'MeanErodedIntensity_{ch_idx}'] = float(np.mean(eroded_intensities))
                prop[f'MedianErodedIntensity_{ch_idx}'] = float(np.median(eroded_intensities))

                # Add texture features if enabled
                if include_texture_features:
                    row_min, col_min, row_max, col_max = shapely_polygon.bounds
                    # TODO: Double check this is referencing the correct image in terms of x vs y
                    # RESOLVED: It is correct.
                    cropped_image = image[int(row_min):int(
                        row_max)+1, int(col_min):int(col_max)+1].squeeze().copy()
                    # Only compute texture features if the patch is big enough
                    if cropped_image.size > 0 and np.any(cropped_image):
                        if cropped_image.shape[0] > 3 and cropped_image.shape[1] > 3:
                            texture_features = compute_texture_features(
                                cropped_image, levels=texture_scaling)

                            # Add channel suffix to texture feature names
                            for key, value in texture_features.items():
                                prop[f'{key}_{ch_idx}'] = value

            prop['Area'] = shapely_polygon.area
            prop['Perimeter'] = shapely_polygon.length

            # Extract scalar values from objects
            centroid = shapely_polygon.centroid
            prop['CentroidX'] = centroid.x
            prop['CentroidY'] = centroid.y

            # Bounding box dimensions
            minx, miny, maxx, maxy = shapely_polygon.bounds
            prop['BoundingBoxWidth'] = maxx - minx
            prop['BoundingBoxHeight'] = maxy - miny

            # Add derived shape metrics
            convex_hull = shapely_polygon.convex_hull
            prop['ConvexHullArea'] = convex_hull.area
            prop['Solidity'] = shapely_polygon.area / \
                convex_hull.area if convex_hull.area > 0 else 0
            prop['Circularity'] = 4 * np.pi * shapely_polygon.area / \
                (shapely_polygon.length ** 2) if shapely_polygon.length > 0 else 0
            prop['Extent'] = shapely_polygon.area / \
                ((maxx - minx) * (maxy - miny)) if ((maxx - minx) * (maxy - miny)) > 0 else 0

            tags = annotation['tags']
            # Remove the generic tags
            tags = [tag for tag in tags if tag not in params['tags']['tags']]

            if len(tags) > 0:
                prop['tags'] = tags[0]
            else:
                prop['tags'] = ""

            # Fill missing columns with NaN
            for col in all_columns:
                if col not in prop and col != 'tags':
                    prop[col] = np.nan

            # Add to DataFrame using annotation ID as index
            df.loc[annotation['_id']] = prop

            # Let's just add a tag for now
            subset_prop = {key: prop[key] for key in [
                'Area', 'Perimeter', 'tags'] if key in prop}
            property_value_dict[annotation['_id']] = subset_prop

            processed_annotations += 1
            update_progress(processed_annotations, number_annotations,
                            "Computing features and textures")

    # Handle missing values that might have been introduced
    df = df.fillna(0)  # Replace NaN with 0

    # Split the data into labeled and unlabeled
    unlabeled_data = df[df['tags'] == ""]
    labeled_data = df[df['tags'] != ""]

    # Skip if no labeled data
    if len(labeled_data) == 0:
        sendWarning('No labeled data',
                    info='No labeled data found. Please tag some annotations.')
        return

    X_labeled = labeled_data.drop('tags', axis=1)
    y_labeled = labeled_data['tags']

    # Feature selection - remove any columns with zero variance
    # This helps avoid issues with the classifier
    selector = X_labeled.var() > 0
    X_labeled = X_labeled.loc[:, selector]

    # Update all DataFrames to have consistent columns
    if len(unlabeled_data) > 0:
        unlabeled_data = unlabeled_data.loc[:, ['tags'] + list(X_labeled.columns)]

    # Split labeled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled, test_size=0.2, random_state=42, stratify=y_labeled if len(set(y_labeled)) > 1 else None
    )

    # Train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = rf_model.predict(X_test)
    print("Model performance on test set:")
    print(classification_report(y_test, y_pred))

    # Feature importance analysis
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_labeled.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("Top 10 important features:")
        print(feature_importance.head(10))

    # Predict on unlabeled data
    if len(unlabeled_data) > 0:
        X_unlabeled = unlabeled_data.drop('tags', axis=1)

        # Make sure unlabeled data has the same columns as the training data
        missing_cols = set(X_train.columns) - set(X_unlabeled.columns)
        for col in missing_cols:
            X_unlabeled[col] = 0

        # Ensure columns are in the same order
        X_unlabeled = X_unlabeled[X_train.columns]

        unlabeled_predictions = rf_model.predict(X_unlabeled)
        unlabeled_probabilities = rf_model.predict_proba(X_unlabeled)

        # Add predictions to the unlabeled data
        unlabeled_data['predicted_tag'] = unlabeled_predictions

        # Get the maximum probability value for each prediction
        max_probs = np.max(unlabeled_probabilities, axis=1)
        unlabeled_data['probability'] = max_probs

        # If you want detailed probabilities for each class
        # class_labels = rf_model.classes_
        # for i, label in enumerate(class_labels):
        #     unlabeled_data[f'probability_{label}'] = unlabeled_probabilities[:, i]

        print(unlabeled_data[['predicted_tag', 'probability']].head())

        # Go through unlabeled_data and add the predicted tag to the property_value_dict
        for index, row in unlabeled_data.iterrows():
            # print(index, row['predicted_tag'], row['probability'])
            property_value_dict[index]['predicted_tag'] = row['predicted_tag']
            property_value_dict[index]['probability'] = row['probability']

            # TODO: Have not implemented this feature yet.
            # # Optionally add the predicted class as a tag to the annotation
            # if add_classification_as_tag:
            #     tag_data = {
            #         'tags': [row['predicted_tag']],
            #         'mode': 'add'
            #     }
            #     try:
            # TODO: This function does not exist, but it should be added.
            #         workerClient.update_annotation_tags(index, tag_data)
            #     except Exception as e:
            #         print(f"Error adding tag: {e}")

    # Also add same columns for the labeled data
    for index, row in labeled_data.iterrows():
        property_value_dict[index]['predicted_tag'] = row['tags']
        property_value_dict[index]['probability'] = 1.0

    # pprint.pprint(property_value_dict)

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
