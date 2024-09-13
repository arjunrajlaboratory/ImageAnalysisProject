
# Detect architecture
ARCH=$(uname -m)

echo "Architecture: $ARCH"

# Set Dockerfile based on architecture
if [ "$ARCH" == "arm64" ]; then
    echo "Compiling for M1 architecture"
    DOCKERFILE="Dockerfile_M1"
else
    echo "Compiling for Intel architecture"
    DOCKERFILE="Dockerfile"
fi

echo "Building Piscis worker"
docker compose -f ./workers/annotations/piscis/docker-compose.yaml build

echo "Building Cellpose worker"
docker build ./workers/annotations/cellpose/ -t annotations/cellpose_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Cellpose" --label "interfaceCategory=Cellpose" --label "annotationShape=polygon"

echo "Building Stardist worker"
docker build ./workers/annotations/stardist/ -t annotations/stardist_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Stardist" --label "interfaceCategory=Stardist" --label "annotationShape=polygon"

echo "Building Laplacian of Gaussian worker"
docker build ./workers/annotations/laplacian_of_gaussian/ -t annotations/laplacian_of_gaussian:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Laplacian of Gaussian" --label "interfaceCategory=Laplacian of Gaussian"

echo "Building SAM2 automatic mask generator worker"
docker build ./workers/annotations/sam2_automatic_mask_generator/ -t annotations/sam2_automatic_mask_generator:latest 

echo "Building SAM2 propagate worker"
docker build -f ./workers/annotations/sam2_propagate/$DOCKERFILE -t annotations/sam2_propagate_worker:latest ./workers/annotations/sam2_propagate/
# docker build -f ./workers/annotations/sam2_propagate/Dockerfile_M1 -t annotations/sam2_propagate_worker:latest ./workers/annotations/sam2_propagate/

# These are some legacy workers that are no longer used.
#docker build ./workers/annotations/cellori_segmentation/ -t annotations/cellori_segmentation_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Cellori" --label "interfaceCategory=Cellori"
#docker build ./workers/annotations/deepcell/ -t annotations/deepcell_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=DeepCell" --label "interfaceCategory=Deepcell"
#docker build ./workers/test_worker/ -t both/test_worker:latest  --label isUPennContrastWorker --label isAnnotationWorker --label isPropertyWorker --label "annotationShape=point" --label "interfaceName=Test worker" --label "interfaceCategory=Test"
#docker build ./workers/annotations/test_multiple_annotation/ -t annotations/test_multiple_annotation:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Random square" --label "interfaceCategory=random"

