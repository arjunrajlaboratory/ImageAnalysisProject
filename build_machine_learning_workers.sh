# Detect architecture
ARCH=$(uname -m)

echo "Architecture: $ARCH"

# Check for --no-cache option
NO_CACHE=""
if [ "$1" == "--no-cache" ]; then
    NO_CACHE="--no-cache"
    echo "Building without cache"
fi

# Set Dockerfile based on architecture
if [ "$ARCH" == "arm64" ]; then
    echo "Compiling for M1 architecture"
    DOCKERFILE="Dockerfile_M1"
else
    echo "Compiling for Intel architecture"
    DOCKERFILE="Dockerfile"
fi

# ============================================================
# Build shared base images first
# ============================================================

echo "============================================================"
echo "Building shared base images..."
echo "============================================================"

if [ "$ARCH" == "arm64" ]; then
    echo "Building SAM2 worker base image (M1)"
    docker build . -f ./workers/base_docker_images/Dockerfile.sam2_worker_base_M1 -t nimbusimage/sam2-worker-base-m1:latest $NO_CACHE
else
    echo "Building SAM2 worker base image"
    docker build . -f ./workers/base_docker_images/Dockerfile.sam2_worker_base -t nimbusimage/sam2-worker-base:latest $NO_CACHE

    echo "Building CUDA ML worker base image"
    docker build . -f ./workers/base_docker_images/Dockerfile.cuda_ml_worker_base -t nimbusimage/cuda-ml-worker-base:latest $NO_CACHE
fi

echo "============================================================"
echo "Building individual workers..."
echo "============================================================"

# ============================================================
# Piscis (uses docker-compose, standalone CUDA 12.4, x86_64 only)
# ============================================================

echo "Building Piscis worker"
docker compose -f ./workers/annotations/piscis/docker-compose.yaml build $NO_CACHE

# ============================================================
# CUDA 11.8 workers using cuda-ml-worker-base (x86_64 only)
# ============================================================

if [ "$ARCH" != "arm64" ]; then
    echo "Building Cellpose worker"
    docker build . -f ./workers/annotations/cellpose/Dockerfile -t annotations/cellpose_worker:latest $NO_CACHE

    echo "Building Cellpose train worker"
    docker build . -f ./workers/annotations/cellpose_train/Dockerfile -t annotations/cellpose_train_worker:latest $NO_CACHE

    echo "Building Cellpose-SAM worker"
    docker build . -f ./workers/annotations/cellposesam/Dockerfile -t annotations/cellposesam_worker:latest $NO_CACHE

    echo "Building Stardist worker"
    docker build . -f ./workers/annotations/stardist/Dockerfile -t annotations/stardist_worker:latest $NO_CACHE
fi

# SAM1 workers (have M1 variants)
echo "Building SAM few-shot segmentation worker"
docker build . -f ./workers/annotations/sam_fewshot_segmentation/$DOCKERFILE -t annotations/sam_fewshot_segmentation:latest $NO_CACHE

echo "Building SAM automatic mask generator worker"
docker build . -f ./workers/annotations/sam_automatic_mask_generator/$DOCKERFILE -t annotations/sam_automatic_mask_generator:latest $NO_CACHE

# ============================================================
# SAM2 workers (use sam2-worker-base)
# ============================================================

echo "Building SAM2 automatic mask generator worker"
docker build . -f ./workers/annotations/sam2_automatic_mask_generator/$DOCKERFILE -t annotations/sam2_automatic_mask_generator:latest $NO_CACHE

echo "Building SAM2 few-shot segmentation worker"
docker build . -f ./workers/annotations/sam2_fewshot_segmentation/$DOCKERFILE -t annotations/sam2_fewshot_segmentation:latest $NO_CACHE

echo "Building SAM2 propagate worker"
docker build . -f ./workers/annotations/sam2_propagate/$DOCKERFILE -t annotations/sam2_propagate_worker:latest $NO_CACHE

echo "Building SAM2 refine worker"
docker build . -f ./workers/annotations/sam2_refine/$DOCKERFILE -t annotations/sam2_refine_worker:latest $NO_CACHE

echo "Building SAM2 video worker"
docker build . -f ./workers/annotations/sam2_video/$DOCKERFILE -t annotations/sam2_video_worker:latest $NO_CACHE

# ============================================================
# Standalone workers (CUDA 12.1, unique dependencies)
# ============================================================

echo "Building CondensateNet worker"
docker build . -f ./workers/annotations/condensatenet/$DOCKERFILE -t annotations/condensatenet:latest $NO_CACHE
