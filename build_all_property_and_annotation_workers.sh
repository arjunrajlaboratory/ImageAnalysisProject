#!/bin/bash

# Detect architecture
ARCH=$(uname -m)

echo "Architecture: $ARCH"

# Check for --no-cache option
NO_CACHE=""
if [ "$1" == "--no-cache" ]; then
    NO_CACHE="--no-cache"
    echo "Building without cache"
fi

# Set Dockerfile based on architecture
if [ "$ARCH" == "arm64" ]; then
    echo "Compiling for M1 architecture"
    DOCKERFILE="Dockerfile_M1"
else
    echo "Compiling for Intel architecture"
    DOCKERFILE="Dockerfile"
fi

# Build Docker image

# Annotation workers
echo "Building annotation worker: connect_to_nearest"
docker build -f ./workers/annotations/connect_to_nearest/$DOCKERFILE -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/ $NO_CACHE
# docker build -f ./workers/annotations/connect_to_nearest/Dockerfile_M1 -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/

echo "Building annotation worker: connect_sequential"
docker build -f ./workers/annotations/connect_sequential/$DOCKERFILE -t annotations/connect_sequential:latest ./workers/annotations/connect_sequential/ $NO_CACHE
# docker build -f ./workers/annotations/connect_sequential/Dockerfile_M1 -t annotations/connect_sequential:latest ./workers/annotations/connect_sequential/

echo "Building annotation worker: connect_time_lapse"
docker build -f ./workers/annotations/connect_timelapse/$DOCKERFILE -t annotations/connect_time_lapse:latest ./workers/annotations/connect_timelapse/ $NO_CACHE
# docker build -f ./workers/annotations/connect_timelapse/Dockerfile_M1 -t annotations/connect_time_lapse:latest ./workers/annotations/connect_timelapse/

echo "Building annotation worker: laplacian_of_gaussian"
docker build -f ./workers/annotations/laplacian_of_gaussian/$DOCKERFILE -t annotations/laplacian_of_gaussian:latest ./workers/annotations/laplacian_of_gaussian/ $NO_CACHE
# docker build -f ./workers/annotations/laplacian_of_gaussian/Dockerfile_M1 -t annotations/laplacian_of_gaussian:latest ./workers/annotations/laplacian_of_gaussian/


# Property workers
echo "Building property worker: blob_metrics_worker"
docker build -f ./workers/properties/blobs/blob_metrics_worker/$DOCKERFILE -t properties/blob_metrics:latest ./workers/properties/blobs/blob_metrics_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_metrics_worker/Dockerfile_M1 -t properties/blob_metrics:latest ./workers/properties/blobs/blob_metrics_worker/

echo "Building property worker: blob_overlap_worker"
docker build -f ./workers/properties/blobs/blob_overlap_worker/$DOCKERFILE -t properties/blob_overlap:latest ./workers/properties/blobs/blob_overlap_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_overlap_worker/Dockerfile_M1 -t properties/blob_overlap:latest ./workers/properties/blobs/blob_overlap_worker/

echo "Building property worker: blob_intensity_worker"
docker build -f ./workers/properties/blobs/blob_intensity_worker/$DOCKERFILE -t properties/blob_intensity:latest ./workers/properties/blobs/blob_intensity_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_intensity_worker/Dockerfile_M1 -t properties/blob_intensity:latest ./workers/properties/blobs/blob_intensity_worker/

echo "Building property worker: blob_intensity_percentile_worker"
docker build -f ./workers/properties/blobs/blob_intensity_percentile_worker/$DOCKERFILE -t properties/blob_intensity_percentile:latest ./workers/properties/blobs/blob_intensity_percentile_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_intensity_percentile_worker/Dockerfile_M1 -t properties/blob_intensity_percentile:latest ./workers/properties/blobs/blob_intensity_percentile_worker/

echo "Building property worker: blob_annulus_intensity_worker"
docker build -f ./workers/properties/blobs/blob_annulus_intensity_worker/$DOCKERFILE -t properties/blob_annulus_intensity:latest ./workers/properties/blobs/blob_annulus_intensity_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_annulus_intensity_worker/Dockerfile_M1 -t properties/blob_annulus_intensity:latest ./workers/properties/blobs/blob_annulus_intensity_worker/

echo "Building property worker: blob_annulus_intensity_percentile_worker"
docker build -f ./workers/properties/blobs/blob_annulus_intensity_percentile_worker/$DOCKERFILE -t properties/blob_annulus_intensity_percentile:latest ./workers/properties/blobs/blob_annulus_intensity_percentile_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_annulus_intensity_percentile_worker/Dockerfile_M1 -t properties/blob_annulus_intensity_percentile:latest ./workers/properties/blobs/blob_annulus_intensity_percentile_worker/

echo "Building property worker: blob_colony_two_color_intensity_worker"
docker build -f ./workers/properties/blobs/blob_colony_two_color_intensity_worker/$DOCKERFILE -t properties/blob_colony_two_color_intensity:latest ./workers/properties/blobs/blob_colony_two_color_intensity_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_colony_two_color_intensity_worker/Dockerfile_M1 -t properties/blob_colony_two_color_intensity:latest ./workers/properties/blobs/blob_colony_two_color_intensity_worker/

echo "Building property worker: blob_point_count_worker"
docker build -f ./workers/properties/blobs/blob_point_count_worker/$DOCKERFILE -t properties/blob_point_count:latest ./workers/properties/blobs/blob_point_count_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_point_count_worker/Dockerfile_M1 -t properties/blob_point_count:latest ./workers/properties/blobs/blob_point_count_worker/

echo "Building property worker: blob_point_count_3D_projection_worker"
docker build -f ./workers/properties/blobs/blob_point_count_3D_projection_worker/$DOCKERFILE -t properties/blob_point_count_3d_projection:latest ./workers/properties/blobs/blob_point_count_3D_projection_worker/ $NO_CACHE
# docker build -f ./workers/properties/blobs/blob_point_count_3D_projection_worker/Dockerfile_M1 -t properties/blob_point_count_3d_projection:latest ./workers/properties/blobs/blob_point_count_3D_projection_worker/

echo "Building property worker: children_count_worker"
docker build -f ./workers/properties/connections/children_count_worker/$DOCKERFILE -t properties/children_count:latest ./workers/properties/connections/children_count_worker/ $NO_CACHE
# docker build -f ./workers/properties/connections/children_count_worker/Dockerfile_M1 -t properties/children_count:latest ./workers/properties/connections/children_count_worker/

echo "Building property worker: parent_child_worker"
docker build -f ./workers/properties/connections/parent_child_worker/$DOCKERFILE -t properties/parent_child:latest ./workers/properties/connections/parent_child_worker/ $NO_CACHE
# docker build -f ./workers/properties/connections/parent_child_worker/Dockerfile_M1 -t properties/parent_child:latest ./workers/properties/connections/parent_child_worker/


echo "Building property worker: point_metrics_worker"
docker build -f ./workers/properties/points/point_metrics_worker/$DOCKERFILE -t properties/point_metrics:latest ./workers/properties/points/point_metrics_worker/ $NO_CACHE
# docker build -f ./workers/properties/points/point_metrics_worker/Dockerfile_M1 -t properties/point_metrics:latest ./workers/properties/points/point_metrics_worker/

echo "Building property worker: point_circle_intensity_worker"
docker build -f ./workers/properties/points/point_circle_intensity_worker/$DOCKERFILE -t properties/point_intensity:latest ./workers/properties/points/point_circle_intensity_worker/ $NO_CACHE
# docker build -f ./workers/properties/points/point_circle_intensity_worker/Dockerfile_M1 -t properties/point_intensity:latest ./workers/properties/points/point_circle_intensity_worker/

echo "Building property worker: point_to_nearest_blob_distance"
docker build -f ./workers/properties/points/point_to_nearest_blob_distance/$DOCKERFILE -t properties/point_to_nearest_blob_distance:latest ./workers/properties/points/point_to_nearest_blob_distance/ $NO_CACHE
# docker build -f ./workers/properties/points/point_to_nearest_blob_distance/Dockerfile_M1 -t properties/point_to_nearest_blob_distance:latest ./workers/properties/points/point_to_nearest_blob_distance/

echo "Building property worker: line_scan_worker"
docker build -f ./workers/properties/lines/line_scan_worker/$DOCKERFILE -t properties/line_scan_worker:latest ./workers/properties/lines/line_scan_worker/ $NO_CACHE
# docker build -f ./workers/properties/lines/line_scan_worker/Dockerfile_M1 -t properties/line_scan_worker:latest ./workers/properties/lines/line_scan_worker/


# TEST WORKERS
# echo "Building property worker: test_file_creation_worker"
# docker build -f ./workers/properties/lines/test_file_creation_worker/$DOCKERFILE -t properties/test_file_creation:latest ./workers/properties/lines/test_file_creation_worker/
# docker build -f ./workers/properties/lines/test_file_creation_worker/Dockerfile_M1 -t properties/test_file_creation:latest ./workers/properties/lines/test_file_creation_worker/
# docker build -f ./workers/annotations/random_square/Dockerfile_M1 -t annotations/random_square:latest ./workers/annotations/random_square/


# AI workers
echo "Building AI worker: ai_analysis"
docker build -f ./workers/annotations/ai_analysis/$DOCKERFILE -t annotations/ai_analysis:latest ./workers/annotations/ai_analysis/ $NO_CACHE
# docker build -f ./workers/annotations/ai_analysis/Dockerfile_M1 -t annotations/ai_analysis:latest ./workers/annotations/ai_analysis/