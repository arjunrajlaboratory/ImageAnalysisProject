#!/bin/bash

# =============================================================================
# Test Worker Build & Test Script
# =============================================================================
#
# Builds and tests the test/sample workers (random_squares, sample_interface).
# These are workers used for testing and development, not production.
#
# Usage:
#   ./build_test_workers.sh [OPTIONS] [SERVICE_NAME]
#
# Options:
#   --no-cache              Build images without using Docker cache
#   --build-and-run-tests   Build and run tests
#   --build-tests-only      Build tests only (no run)
#   --run-tests-only        Run tests without rebuilding images
#
# Examples:
#   ./build_test_workers.sh                          # Build all test workers
#   ./build_test_workers.sh random_squares           # Build specific worker
#   ./build_test_workers.sh --build-and-run-tests    # Build all + run tests
#   ./build_test_workers.sh --no-cache               # Build all without cache
# =============================================================================

export COMPOSE_PARALLEL_LIMIT=1

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Parse arguments
NO_CACHE=""
SERVICE=""
BUILD_TESTS=false
RUN_TESTS=false
BUILD_WORKERS=true

for arg in "$@"; do
    if [ "$arg" == "--no-cache" ]; then
        NO_CACHE="--no-cache"
        echo "Building without cache"
    elif [ "$arg" == "--build-and-run-tests" ]; then
        BUILD_TESTS=true
        RUN_TESTS=true
        echo "Will build and run tests"
    elif [ "$arg" == "--build-tests-only" ]; then
        BUILD_TESTS=true
        echo "Will build tests only; will not run tests"
    elif [ "$arg" == "--run-tests-only" ]; then
        RUN_TESTS=true
        BUILD_WORKERS=false
        echo "Will only run tests (no build)"
    else
        SERVICE="$arg"
    fi
done

if [ -z "$SERVICE" ]; then
    if [ "$BUILD_WORKERS" = true ]; then
        echo "Building all test workers..."
        docker compose --profile testworker build $NO_CACHE
    fi

    if [ "$BUILD_TESTS" = true ]; then
        docker compose --profile testworker --profile testworkertest build $NO_CACHE
    fi

    if [ "$RUN_TESTS" = true ]; then
        for test_service in $(docker compose --profile testworker --profile testworkertest config --services | grep "_test$"); do
            echo "Running tests for: $test_service"
            docker compose --profile testworker --profile testworkertest run --rm $test_service
        done
    fi
else
    echo "Building test worker: $SERVICE"

    if [ "$BUILD_WORKERS" = true ]; then
        docker compose --profile "*" build $NO_CACHE $SERVICE
    fi

    if [ "$BUILD_TESTS" = true ]; then
        docker compose --profile "*" build $NO_CACHE "${SERVICE}_test"
    fi

    if [ "$RUN_TESTS" = true ]; then
        docker compose --profile "*" run --rm "${SERVICE}_test"
    fi
fi

echo "Process completed!"
