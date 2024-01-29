#!/bin/bash

# Detect architecture
ARCH=$(uname -m)

echo "Architecture: $ARCH"

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
docker build -f ./workers/annotations/connect_to_nearest/$DOCKERFILE -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/

echo "Building annotation worker: connect_sequential"
docker build -f ./workers/annotations/connect_sequential/$DOCKERFILE -t annotations/connect_sequential:latest ./workers/annotations/connect_sequential/


# Property workers
echo "Building property worker: blob_metrics_worker"
docker build -f ./workers/properties/blobs/blob_metrics_worker/$DOCKERFILE -t properties/blob_metrics:latest ./workers/properties/blobs/blob_metrics_worker/
# docker build -f ./workers/properties/blobs/blob_metrics_worker/Dockerfile_M1 -t properties/blob_metrics:latest ./workers/properties/blobs/blob_metrics_worker/

echo "Building property worker: blob_intensity_worker"
docker build -f ./workers/properties/blobs/blob_intensity_worker/$DOCKERFILE -t properties/blob_intensity:latest ./workers/properties/blobs/blob_intensity_worker/
# docker build -f ./workers/properties/blobs/blob_intensity_worker/Dockerfile_M1 -t properties/blob_intensity:latest ./workers/properties/blobs/blob_intensity_worker/

echo "Building property worker: blob_annulus_intensity_worker"
docker build -f ./workers/properties/blobs/blob_annulus_intensity_worker/$DOCKERFILE -t properties/blob_annulus_intensity:latest ./workers/properties/blobs/blob_annulus_intensity_worker/
# docker build -f ./workers/properties/blobs/blob_annulus_intensity_worker/Dockerfile_M1 -t properties/blob_annulus_intensity:latest ./workers/properties/blobs/blob_annulus_intensity_worker/

echo "Building property worker: blob_point_count_worker"
docker build -f ./workers/properties/blobs/blob_point_count_worker/$DOCKERFILE -t properties/blob_point_count:latest ./workers/properties/blobs/blob_point_count_worker/
# docker build -f ./workers/properties/blobs/blob_point_count_worker/Dockerfile_M1 -t properties/blob_point_count:latest ./workers/properties/blobs/blob_point_count_worker/

echo "Building property worker: blob_point_count_3D_projection_worker"
docker build -f ./workers/properties/blobs/blob_point_count_3D_projection_worker/$DOCKERFILE -t properties/blob_point_count_3d_projection:latest ./workers/properties/blobs/blob_point_count_3D_projection_worker/
# docker build -f ./workers/properties/blobs/blob_point_count_3D_projection_worker/Dockerfile_M1 -t properties/blob_point_count_3d_projection:latest ./workers/properties/blobs/blob_point_count_3D_projection_worker/

echo "Building property worker: parent_child_worker"
docker build -f ./workers/properties/connections/parent_child_worker/$DOCKERFILE -t properties/parent_child:latest ./workers/properties/connections/parent_child_worker/
# docker build -f ./workers/properties/connections/parent_child_worker/Dockerfile_M1 -t properties/parent_child:latest ./workers/properties/connections/parent_child_worker/


echo "Building property worker: point_metrics_worker"
docker build -f ./workers/properties/points/point_metrics_worker/$DOCKERFILE -t properties/point_metrics:latest ./workers/properties/points/point_metrics_worker/
# docker build -f ./workers/properties/points/point_metrics_worker/Dockerfile_M1 -t properties/point_metrics:latest ./workers/properties/points/point_metrics_worker/
