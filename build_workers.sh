#!/bin/bash

# =============================================================================
# Worker Build & Test Script
# =============================================================================
#
# This script builds and tests worker Docker images.
#
# Usage:
#   ./build_workers.sh [OPTIONS] [SERVICE_NAME]
#
# Options:
#   --no-cache     Build images without using Docker cache
#   --test         Build images and run tests afterward
#   --test-only    Run tests without rebuilding images
#
# Examples:
#   # Build all workers
#   ./build_workers.sh
#
#   # Build a specific worker
#   ./build_workers.sh blob_metrics
#
#   # Build all workers without cache
#   ./build_workers.sh --no-cache
#
#   # Build and test all workers
#   ./build_workers.sh --test
#
#   # Build and test a specific worker
#   ./build_workers.sh --test blob_metrics
#
#   # Run tests for all workers without rebuilding
#   ./build_workers.sh --test-only
#
#   # Run tests for a specific worker without rebuilding
#   ./build_workers.sh --test-only blob_metrics
# =============================================================================

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
RUN_TESTS=false
TEST_ONLY=false

for arg in "$@"; do
    if [ "$arg" == "--no-cache" ]; then
        NO_CACHE="--no-cache"
        echo "Building without cache"
    elif [ "$arg" == "--test" ]; then
        RUN_TESTS=true
        echo "Will run tests after building"
    elif [ "$arg" == "--test-only" ]; then
        TEST_ONLY=true
        echo "Will only run tests (no build)"
    else
        SERVICE="$arg"
    fi
done

# Build services
if [ "$TEST_ONLY" = false ]; then
    if [ -z "$SERVICE" ]; then
        echo "Building all workers..."
        docker-compose build $NO_CACHE
    else
        echo "Building worker: $SERVICE"
        docker-compose build $NO_CACHE $SERVICE
        # If we're building a specific service and want to test it
        if [ "$RUN_TESTS" = true ]; then
            # Also build its test
            echo "Building test for: $SERVICE"
            docker-compose build $NO_CACHE "${SERVICE}_test"
        fi
    fi
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ] || [ "$TEST_ONLY" = true ]; then
    if [ -z "$SERVICE" ]; then
        echo "Running all tests..."
        # Find all test services and run them
        TEST_SERVICES=$(docker-compose config --services | grep '_test$')
        for test_service in $TEST_SERVICES; do
            echo "Running test: $test_service"
            docker-compose run --rm $test_service
        done
    else
        echo "Running test for: $SERVICE"
        docker-compose run --rm "${SERVICE}_test"
    fi
fi

echo "Process completed!"