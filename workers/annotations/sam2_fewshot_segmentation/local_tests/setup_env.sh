#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

SKIP_SAM2=false
for arg in "$@"; do
    case $arg in
        --skip-sam2) SKIP_SAM2=true ;;
    esac
done

echo "=== SAM2 Few-Shot Local Test Environment Setup ==="
echo "Script dir: $SCRIPT_DIR"
if [ "$SKIP_SAM2" = true ]; then
    echo "Skipping SAM2/PyTorch install (--skip-sam2 flag)"
fi

# --- Create venv ---
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, delete it first: rm -rf $VENV_DIR"
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "Using Python: $(which python)"

# --- Install pip basics ---
pip install --upgrade pip setuptools wheel

# --- annotation_client from NimbusImage local clone ---
ANNOTATION_CLIENT_DIR="/home/arjun/UPennContrast/devops/girder/annotation_client"
if [ -d "$ANNOTATION_CLIENT_DIR" ]; then
    echo "Installing annotation_client from $ANNOTATION_CLIENT_DIR..."
    pip install girder-client tifffile
    pip install -e "$ANNOTATION_CLIENT_DIR"
else
    echo "WARNING: annotation_client not found at $ANNOTATION_CLIENT_DIR"
    echo "You'll need to install it manually."
fi

# --- annotation_utilities ---
ANNOTATION_UTILS_DIR="/home/arjun/ImageAnalysisProject/annotation_utilities"
if [ -d "$ANNOTATION_UTILS_DIR" ]; then
    echo "Installing annotation_utilities from $ANNOTATION_UTILS_DIR..."
    pip install -e "$ANNOTATION_UTILS_DIR"
else
    echo "WARNING: annotation_utilities not found at $ANNOTATION_UTILS_DIR"
fi

# --- worker_client ---
WORKER_CLIENT_DIR="/home/arjun/ImageAnalysisProject/worker_client"
if [ -d "$WORKER_CLIENT_DIR" ]; then
    echo "Installing worker_client from $WORKER_CLIENT_DIR..."
    pip install -e "$WORKER_CLIENT_DIR"
else
    echo "WARNING: worker_client not found at $WORKER_CLIENT_DIR"
fi

# --- Scientific Python stack ---
echo "Installing scientific Python packages..."
pip install numpy scipy scikit-image shapely matplotlib pillow numba

# --- PyTorch + SAM2 (skippable for connection-only testing) ---
if [ "$SKIP_SAM2" = false ]; then
    echo "Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected, installing CUDA PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "No NVIDIA GPU detected, installing CPU PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    SAM2_DIR="/tmp/sam2"
    echo "Installing SAM2..."
    if [ -d "$SAM2_DIR" ]; then
        echo "Using existing SAM2 clone at $SAM2_DIR"
    else
        echo "Cloning SAM2 to $SAM2_DIR..."
        git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
    fi
    SAM2_BUILD_CUDA=0 pip install -e "$SAM2_DIR"

    # Download SAM2 checkpoints if needed
    CHECKPOINT_DIR="$SAM2_DIR/checkpoints"
    if [ ! -f "$CHECKPOINT_DIR/sam2.1_hiera_small.pt" ]; then
        echo "Downloading SAM2 checkpoints..."
        pushd "$SAM2_DIR" > /dev/null
        if [ -f "checkpoints/download_ckpts.sh" ]; then
            bash checkpoints/download_ckpts.sh
        else
            mkdir -p checkpoints
            echo "WARNING: Checkpoint download script not found."
            echo "Download manually from https://github.com/facebookresearch/sam2#download-checkpoints"
        fi
        popd > /dev/null
    else
        echo "SAM2 checkpoints already present."
    fi
else
    echo "Skipping PyTorch and SAM2 install."
    echo "test_connection.py will still work for API testing."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Then run: python $SCRIPT_DIR/test_connection.py"
