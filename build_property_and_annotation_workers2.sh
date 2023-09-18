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
docker build -f ./workers/annotations/connect_to_nearest/$DOCKERFILE -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/

docker build -f ./workers/annotations/connect_sequential/$DOCKERFILE -t annotations/connect_sequential:latest ./workers/annotations/connect_sequential/
#docker build -f ./workers/annotations/connect_sequential/Dockerfile_M1 -t annotations/connect_sequential:latest ./workers/annotations/connect_sequential/