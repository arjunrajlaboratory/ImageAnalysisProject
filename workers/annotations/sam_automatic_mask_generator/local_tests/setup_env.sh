#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

SKIP_SAM1=false
for arg in "$@"; do
    case $arg in
        --skip-sam1) SKIP_SAM1=true ;;
    esac
done

echo "=== SAM1 ViT-H Few-Shot Local Test Environment Setup ==="
echo "Script dir: $SCRIPT_DIR"
if [ "$SKIP_SAM1" = true ]; then
    echo "Skipping SAM1/PyTorch install (--skip-sam1 flag)"
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
pip install numpy scipy scikit-image shapely matplotlib pillow numba opencv-python-headless

# --- PyTorch + SAM1 (skippable for connection-only testing) ---
if [ "$SKIP_SAM1" = false ]; then
    echo "Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected, installing CUDA PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "No NVIDIA GPU detected, installing CPU PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    echo "Installing segment-anything (SAM1)..."
    pip install segment-anything

    # Download SAM1 ViT-H checkpoint if needed
    CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
    mkdir -p "$CHECKPOINT_DIR"
    CHECKPOINT_PATH="$CHECKPOINT_DIR/sam_vit_h_4b8939.pth"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Downloading SAM1 ViT-H checkpoint (~2.5GB)..."
        wget -O "$CHECKPOINT_PATH" \
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    else
        echo "SAM1 ViT-H checkpoint already present at $CHECKPOINT_PATH"
    fi
else
    echo "Skipping PyTorch and SAM1 install."
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Then run: python $SCRIPT_DIR/test_sam1_fewshot_experiment.py --help"
