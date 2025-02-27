#!/bin/bash

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Set Dockerfile based on architecture
if [ "$ARCH" == "arm64" ]; then
    echo "Compiling for M1 architecture"
    export DOCKERFILE="Dockerfile_M1"
else
    echo "Compiling for Intel architecture"
    export DOCKERFILE="Dockerfile"
fi

# Parse arguments
NO_CACHE=""
SERVICE=""

for arg in "$@"; do
    if [ "$arg" == "--no-cache" ]; then
        NO_CACHE="--no-cache"
        echo "Building without cache"
    else
        SERVICE="$arg"
    fi
done

# Build services
if [ -z "$SERVICE" ]; then
    echo "Building all workers..."
    docker-compose build $NO_CACHE
else
    echo "Building worker: $SERVICE"
    docker-compose build $NO_CACHE $SERVICE
fi

echo "Build completed!"