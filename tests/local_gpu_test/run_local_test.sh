#!/usr/bin/env bash
#
# Run a worker Docker container against the local mock Girder server.
#
# Usage:
#   ./run_local_test.sh <docker_image> [--request compute|interface] [--gpu] [-- extra_docker_args...]
#
# Examples:
#   # Test random_squares (no GPU needed)
#   ./run_local_test.sh annotations/random_squares:latest
#
#   # Test cellposesam with GPU
#   ./run_local_test.sh annotations/cellposesam:latest --gpu
#
#   # Test a property worker
#   ./run_local_test.sh properties/blob_intensity:latest
#
#   # Pass custom parameters JSON
#   ./run_local_test.sh annotations/random_squares:latest --params-file my_params.json
#
# The script will:
#   1. Start the mock Girder server
#   2. Generate a test image if none exists
#   3. Run the worker container
#   4. Save outputs and display results
#   5. Stop the mock server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOCK_SERVER="$REPO_ROOT/tests/mock_girder_server/server.py"
IMAGES_DIR="$REPO_ROOT/tests/fixtures/images"
OUTPUT_DIR="$REPO_ROOT/tests/fixtures/output"

PORT=5555
USE_GPU=false
REQUEST="compute"
PARAMS_FILE=""
DOCKER_IMAGE=""
EXTRA_DOCKER_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --request)
            REQUEST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --params-file)
            PARAMS_FILE="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_DOCKER_ARGS=("$@")
            break
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [[ -z "$DOCKER_IMAGE" ]]; then
                DOCKER_IMAGE="$1"
            else
                EXTRA_DOCKER_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$DOCKER_IMAGE" ]]; then
    echo "Usage: $0 <docker_image> [--gpu] [--request compute|interface] [--params-file file.json]"
    echo ""
    echo "Examples:"
    echo "  $0 annotations/random_squares:latest"
    echo "  $0 annotations/cellposesam:latest --gpu"
    echo "  $0 properties/blob_intensity:latest"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Local Worker Test ===${NC}"
echo "Image:   $DOCKER_IMAGE"
echo "Request: $REQUEST"
echo "GPU:     $USE_GPU"
echo ""

# --- Step 1: Generate test image if needed ---
if ! ls "$IMAGES_DIR"/*.tif* &>/dev/null; then
    echo -e "${YELLOW}No test images found. Generating synthetic image...${NC}"
    pip install -q tifffile numpy 2>/dev/null || true
    python "$IMAGES_DIR/generate_test_image.py" --output-dir "$IMAGES_DIR"
    echo ""
fi

# --- Step 2: Start mock server ---
echo -e "${GREEN}Starting mock Girder server on port $PORT...${NC}"
python "$MOCK_SERVER" --images-dir "$IMAGES_DIR" --port "$PORT" --output-dir "$OUTPUT_DIR" &
MOCK_PID=$!

# Give server time to start
sleep 2

# Verify server is running
if ! kill -0 "$MOCK_PID" 2>/dev/null; then
    echo -e "${RED}Mock server failed to start${NC}"
    exit 1
fi

cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping mock server (PID $MOCK_PID)...${NC}"
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- Step 3: Build parameters JSON ---
DATASET_ID="test_dataset"

if [[ -n "$PARAMS_FILE" && -f "$PARAMS_FILE" ]]; then
    PARAMS=$(cat "$PARAMS_FILE")
else
    # Default parameters — adjust per worker type
    # Detect worker type from image name
    if [[ "$DOCKER_IMAGE" == *"random_squares"* ]]; then
        PARAMS=$(cat <<'ENDJSON'
{
    "datasetId": "test_dataset",
    "type": "segmentation",
    "id": "test-tool-id",
    "name": "Test Run",
    "image": "DOCKER_IMAGE_PLACEHOLDER",
    "channel": 0,
    "assignment": {"XY": 0, "Z": 0, "Time": 0},
    "tags": ["test"],
    "tile": {"XY": 0, "Z": 0, "Time": 0},
    "connectTo": {"layer": null, "tags": [], "channel": null},
    "workerInterface": {
        "Batch Time": "",
        "Batch XY": "",
        "Batch Z": "",
        "Number of squares": 20,
        "Random Squares": "",
        "Square size": 15
    },
    "scales": {
        "pixelSize": {"unit": "mm", "value": 0.000325},
        "tStep": {"unit": "s", "value": 1},
        "zStep": {"unit": "m", "value": 1}
    }
}
ENDJSON
        )
    elif [[ "$DOCKER_IMAGE" == *"cellposesam"* ]]; then
        PARAMS=$(cat <<'ENDJSON'
{
    "datasetId": "test_dataset",
    "type": "segmentation",
    "id": "test-tool-id",
    "name": "Test Cellpose-SAM",
    "image": "DOCKER_IMAGE_PLACEHOLDER",
    "channel": 0,
    "assignment": {"XY": 0, "Z": 0, "Time": 0},
    "tags": ["cell"],
    "tile": {"XY": 0, "Z": 0, "Time": 0},
    "connectTo": {"layer": null, "tags": [], "channel": null},
    "workerInterface": {
        "Batch Time": "",
        "Batch XY": "",
        "Batch Z": "",
        "Model": "cellpose-sam",
        "Channel for Slot 1": {"0": true},
        "Channel for Slot 2": {},
        "Channel for Slot 3": {},
        "Diameter": 30,
        "Smoothing": 0.7,
        "Padding": 0,
        "Tile Size": 512,
        "Tile Overlap": 0.1
    },
    "scales": {
        "pixelSize": {"unit": "mm", "value": 0.000325},
        "tStep": {"unit": "s", "value": 1},
        "zStep": {"unit": "m", "value": 1}
    }
}
ENDJSON
        )
    elif [[ "$DOCKER_IMAGE" == *"blob_intensity"* ]] || [[ "$DOCKER_IMAGE" == *"blob_metrics"* ]]; then
        PARAMS=$(cat <<'ENDJSON'
{
    "datasetId": "test_dataset",
    "type": "property",
    "id": "test-property-id",
    "name": "Test Property",
    "image": "DOCKER_IMAGE_PLACEHOLDER",
    "channel": 0,
    "assignment": {"XY": 0, "Z": 0, "Time": 0},
    "tags": {"tags": ["test"], "exclusive": false},
    "tile": {"XY": 0, "Z": 0, "Time": 0},
    "connectTo": {"layer": null, "tags": [], "channel": null},
    "workerInterface": {"Channel": 0},
    "scales": {
        "pixelSize": {"unit": "mm", "value": 0.000325},
        "tStep": {"unit": "s", "value": 1},
        "zStep": {"unit": "m", "value": 1}
    }
}
ENDJSON
        )
    else
        # Generic parameters
        PARAMS=$(cat <<'ENDJSON'
{
    "datasetId": "test_dataset",
    "type": "segmentation",
    "id": "test-tool-id",
    "name": "Test Run",
    "image": "DOCKER_IMAGE_PLACEHOLDER",
    "channel": 0,
    "assignment": {"XY": 0, "Z": 0, "Time": 0},
    "tags": ["test"],
    "tile": {"XY": 0, "Z": 0, "Time": 0},
    "connectTo": {"layer": null, "tags": [], "channel": null},
    "workerInterface": {},
    "scales": {
        "pixelSize": {"unit": "mm", "value": 0.000325},
        "tStep": {"unit": "s", "value": 1},
        "zStep": {"unit": "m", "value": 1}
    }
}
ENDJSON
        )
    fi
    # Replace placeholder
    PARAMS="${PARAMS//DOCKER_IMAGE_PLACEHOLDER/$DOCKER_IMAGE}"
fi

echo -e "${GREEN}Parameters:${NC}"
echo "$PARAMS" | python -m json.tool 2>/dev/null || echo "$PARAMS"
echo ""

# --- Step 4: Run the worker container ---
echo -e "${GREEN}Running worker container...${NC}"

DOCKER_ARGS=(
    --rm
    --network host
)

if $USE_GPU; then
    DOCKER_ARGS+=(--gpus all)
fi

# Add any extra docker args
DOCKER_ARGS+=("${EXTRA_DOCKER_ARGS[@]}")

DOCKER_ARGS+=("$DOCKER_IMAGE")

# Worker arguments
WORKER_ARGS=(
    --apiUrl "http://localhost:$PORT/api/v1"
    --token "fake_test_token"
    --request "$REQUEST"
    --parameters "$PARAMS"
    --datasetId "$DATASET_ID"
)

echo "docker run ${DOCKER_ARGS[*]} ${WORKER_ARGS[*]}"
echo ""

docker run "${DOCKER_ARGS[@]}" "${WORKER_ARGS[@]}"
EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}Worker completed successfully!${NC}"
else
    echo -e "${RED}Worker exited with code $EXIT_CODE${NC}"
fi

# --- Step 5: Save and display results ---
echo ""
echo -e "${GREEN}Saving results...${NC}"
curl -s -X POST "http://localhost:$PORT/api/v1/_test/save" | python -m json.tool 2>/dev/null

echo ""
echo -e "${GREEN}=== Results ===${NC}"
curl -s "http://localhost:$PORT/api/v1/_test/status" | python -m json.tool 2>/dev/null

# Show annotation summary
ANN_COUNT=$(curl -s "http://localhost:$PORT/api/v1/_test/status" | python -c "import sys,json; print(json.load(sys.stdin)['annotations_count'])" 2>/dev/null || echo "?")
PROP_COUNT=$(curl -s "http://localhost:$PORT/api/v1/_test/status" | python -c "import sys,json; print(json.load(sys.stdin)['property_values_count'])" 2>/dev/null || echo "?")

echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  Annotations created: $ANN_COUNT"
echo "  Property values:     $PROP_COUNT"
echo "  Output saved to:     $OUTPUT_DIR/"

if [[ -f "$OUTPUT_DIR/annotations.json" ]]; then
    echo ""
    echo -e "${YELLOW}First 5 annotations:${NC}"
    python -c "
import json
with open('$OUTPUT_DIR/annotations.json') as f:
    anns = json.load(f)
for a in anns[:5]:
    coords = len(a.get('coordinates', []))
    print(f'  {a[\"_id\"]}: shape={a.get(\"shape\")}, coords={coords}, tags={a.get(\"tags\")}')
if len(anns) > 5:
    print(f'  ... and {len(anns)-5} more')
" 2>/dev/null || true
fi

exit $EXIT_CODE
