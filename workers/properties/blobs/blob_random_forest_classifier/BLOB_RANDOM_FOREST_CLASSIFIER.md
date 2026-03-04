# Blob Random Forest Classifier

Trains a Random Forest classifier on manually tagged polygon annotations and predicts classifications for untagged annotations. Extracts a comprehensive feature set including intensity, morphology, annulus intensity, eroded core intensity, and optional texture features across all channels.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Buffer radius** | number | 10 | Width of the annulus around each polygon in pixels (0-200). Used for annulus intensity features. |
| **Include texture features** | checkbox | true | If enabled, computes Haralick, LBP, Zernike, and TAS texture features from Mahotas. |
| **Texture scaling** | number | 8 | Number of gray levels for texture feature calculation (1-255). Lower is faster but less detailed. |

## Computed Properties

| Property | Description |
|----------|-------------|
| Area | Polygon area |
| Perimeter | Polygon perimeter length |
| tags | The annotation's non-generic tag (used as class label) |
| predicted_tag | Predicted class label from the Random Forest model |
| probability | Maximum class probability from the model (1.0 for manually labeled annotations) |

## Feature Set (Used Internally for Classification)

Per channel (for all channels in the dataset):
- **Blob intensity**: Mean, Max, Min, Median, Q10, Q25, Q75, Q90, Total
- **Annulus intensity**: Mean, Max, Min, Median, Q10, Q25, Q75, Q90, Total
- **Eroded core intensity**: Mean, Median (using the inner 20% of the polygon)
- **Texture features** (optional): 13 Haralick features, 10 LBP bins, 5 Zernike moments, 8 TAS features

Shape features: Area, Perimeter, CentroidX, CentroidY, BoundingBoxWidth, BoundingBoxHeight, ConvexHullArea, Solidity, Circularity, Extent

## How It Works

1. Extracts features for all annotations across all channels, building a pandas DataFrame.
2. Annotations with additional tags (beyond the property's generic tags) are treated as labeled data; the extra tag becomes the class label.
3. Annotations without extra tags are treated as unlabeled data.
4. Zero-variance features are removed. Labeled data is split 80/20 for training/testing with stratified sampling.
5. A `RandomForestClassifier` (100 estimators) is trained and evaluated. Feature importance is printed.
6. Unlabeled annotations receive `predicted_tag` and `probability` values.

## Notes

- Requires at least 2 labeled samples per class. If any class has fewer than 2 samples, the worker returns an error.
- The "generic" tags (from the property configuration) are stripped when determining class labels. Only additional tags are used as labels.
- The eroded core region is computed by eroding the polygon inward by 80% of the centroid-to-boundary distance.
- Texture features are computed on a bounding-box crop of the image around each polygon, not just the masked pixels.
- Missing feature values are filled with 0 before training.
