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
docker build -f ./workers/annotations/connect_to_nearest/$DOCKERFILE -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/

docker build -f ./workers/annotations/connect_sequential/$DOCKERFILE -t annotations/connect_sequential:latest ./workers/annotations/connect_sequential/
#docker build -f ./workers/annotations/connect_sequential/Dockerfile_M1 -t annotations/connect_sequential:latest ./workers/annotations/connect_sequential/


# Property workers
docker build -f ./workers/properties/blobs/blob_metrics_worker/$DOCKERFILE -t properties/blob_metrics:latest ./workers/properties/blobs/blob_metrics_worker/
#docker build -f ./workers/properties/blobs/blob_metrics_worker/Dockerfile_M1 -t properties/blob_metrics:latest ./workers/properties/blobs/blob_metrics_worker/

docker build -f ./workers/properties/blobs/blob_intensity_worker/$DOCKERFILE -t properties/blob_intensity:latest ./workers/properties/blobs/blob_intensity_worker/
#docker build -f ./workers/properties/blobs/blob_intensity_worker/Dockerfile_M1 -t properties/blob_intensity:latest ./workers/properties/blobs/blob_intensity_worker/

docker build -f ./workers/properties/blobs/blob_annulus_intensity_worker/$DOCKERFILE -t properties/blob_annulus_intensity:latest ./workers/properties/blobs/blob_annulus_intensity_worker/
#docker build -f ./workers/properties/blobs/blob_annulus_intensity_worker/Dockerfile_M1 -t properties/blob_annulus_intensity:latest ./workers/properties/blobs/blob_annulus_intensity_worker/
