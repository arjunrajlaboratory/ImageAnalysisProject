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
#   --build-tests-only     Build tests only
#   --build-and-run-tests     Build and run tests
#   --run-tests-only     Run tests without rebuilding images
#   --build-samples-and-samples-tests-only     Build samples and samples tests only
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
#   ./build_workers.sh --build-and-run-tests
#
#   # Build and test a specific worker
#   ./build_workers.sh --build-and-run-tests blob_metrics
#
#   # Build tests for all workers without rebuilding
#   ./build_workers.sh --build-tests-only
#
#   # Build tests for a specific worker without rebuilding
#   ./build_workers.sh --build-tests-only blob_metrics
#
#   # Build samples and samples tests for all workers
#   ./build_workers.sh --build-samples-and-samples-tests-only
#
#   # Build samples and samples tests for a specific sample worker
#   ./build_workers.sh sample_interface --build-and-run-tests
# =============================================================================

# Limit the number of parallel builds to 1 to avoid memory issues
export COMPOSE_PARALLEL_LIMIT=1

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
BUILD_WORKERS=true
BUILD_TESTS=false
RUN_TESTS=false
TEST_ONLY=false
BUILD_SAMPLES=false
BUILD_SAMPLES_TESTS=false
RUN_SAMPLES_TESTS=false

for arg in "$@"; do
    if [ "$arg" == "--no-cache" ]; then
        NO_CACHE="--no-cache"
        echo "Building without cache"
    elif [ "$arg" == "--build-tests-only" ]; then
        BUILD_TESTS=true
        echo "Will build tests only; will not run tests"
    elif [ "$arg" == "--build-and-run-tests" ]; then
        BUILD_TESTS=true
        RUN_TESTS=true
        echo "Will build and run tests"
    elif [ "$arg" == "--run-tests-only" ]; then
        RUN_TESTS=true
        echo "Will only run tests (no build)"
    elif [ "$arg" == "--build-samples-and-samples-tests-only" ]; then
        BUILD_WORKERS=false
        BUILD_SAMPLES=true
        BUILD_SAMPLES_TESTS=true
        RUN_SAMPLES_TESTS=true
        echo "Will build samples and samples tests only; will run samples tests"
    else
        SERVICE="$arg"
    fi
done

if [ -z "$SERVICE" ]; then
    if [ "$BUILD_WORKERS" = true ]; then
        echo "Building all workers..."
        docker compose --profile worker build $NO_CACHE
    fi

    if [ "$BUILD_SAMPLES" = true ]; then
        docker compose --profile samples build $NO_CACHE
    fi

    if [ "$BUILD_TESTS" = true ]; then
        docker compose --profile worker --profile test build $NO_CACHE
    fi

    if [ "$RUN_TESTS" = true ]; then
        # Get list of test services and run them one by one
        for test_service in $(docker compose --profile worker --profile test config --services | grep "_test$"); do
            echo "Running tests for worker: $test_service"
            docker compose --profile worker --profile test run --rm $test_service
        done
    fi

    if [ "$BUILD_SAMPLES_TESTS" = true ]; then
        docker compose --profile samples --profile sampletest build $NO_CACHE
    fi

    if [ "$RUN_SAMPLES_TESTS" = true ]; then
        # Get list of test services and run them one by one
        for test_service in $(docker compose --profile samples --profile sampletest config --services | grep "_test$"); do
            echo "Running tests for sample worker: $test_service"
            docker compose --profile samples --profile sampletest run --rm $test_service
        done
    fi
else
    echo "Building worker: $SERVICE"
    
    # Build the main service with its profile
    docker compose --profile "*" build $NO_CACHE $SERVICE

    if [ "$BUILD_TESTS" = true ]; then
        docker compose --profile "*" build $NO_CACHE "${SERVICE}_test"
    fi

    if [ "$RUN_TESTS" = true ]; then
        docker compose --profile "*" run --rm "${SERVICE}_test"
    fi
fi

echo "Process completed!"