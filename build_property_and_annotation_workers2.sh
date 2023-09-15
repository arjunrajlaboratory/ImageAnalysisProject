#!/bin/bash

# Detect architecture
ARCH=$(uname -m)

# Set Dockerfile based on architecture
if [ "$ARCH" == "arm64" ]; then
    DOCKERFILE="Dockerfile_M1"
else
    DOCKERFILE="Dockerfile"
fi

# Build Docker image
docker build -f ./workers/annotations/connect_to_nearest/$DOCKERFILE -t annotations/connect_to_nearest:latest ./workers/annotations/connect_to_nearest/

