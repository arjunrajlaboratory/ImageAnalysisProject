#!/bin/bash

# Detect architecture
ARCH=$(uname -m)

echo "Architecture: $ARCH"

# Set Dockerfile based on architecture
if [ "$ARCH" == "arm64" ]; then
    ECHO "Compiling for M1 architecture"
    DOCKERFILE="Dockerfile_M1"
else
    ECHO "Compiling for Intel architecture"
    DOCKERFILE="Dockerfile"
fi

# Build Docker image
docker build -f ./workers/annotations/connect_to_nearest/$DOCKERFILE -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/

