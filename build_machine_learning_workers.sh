#docker build ./workers/properties/blobs/blob_point_count_worker/ -t properties/blob_point_count_worker:latest --label isUPennContrastWorker --label isPropertyWorker --label "annotationShape=polygon" --label "interfaceName=Point Count" --label "interfaceCategory=Count"
#docker build ./workers/properties/blobs/blob_point_count_3D_projection_worker/ -t properties/blob_point_count_3d_projection_worker:latest --label isUPennContrastWorker --label isPropertyWorker --label "annotationShape=polygon" --label "interfaceName=Point Count 3D projection" --label "interfaceCategory=Count"

# This one has been moved to build_annotation_workers2.sh
# docker build ./workers/properties/connections/children_count_worker/ -t properties/connection_children_count_worker:latest --label isUPennContrastWorker --label isPropertyWorker --label "annotationShape=polygon" --label "interfaceName=Children Count" --label "interfaceCategory=Count"

#docker build ./workers/properties/points/point_circle_intensity_mean_worker/ -t properties/point_circle_intensity_mean_worker:latest --label isUPennContrastWorker --label isPropertyWorker --label "annotationShape=point" --label "interfaceName=Circle Intensity Mean" --label "interfaceCategory=Intensity"
#docker build ./workers/properties/points/point_intensity_worker/ -t properties/point_intensity_worker:latest --label isUPennContrastWorker --label isPropertyWorker --label "annotationShape=point" --label "interfaceName=Intensity" --label "interfaceCategory=Intensity"
#docker build ./workers/properties/points/point_threshold_intensity_mean_worker/ -t properties/point_threshold_intensity_mean_worker:latest --label isUPennContrastWorker --label isPropertyWorker --label "annotationShape=point" --label "interfaceName=Threshold Intensity Mean" --label "interfaceCategory=Intensity"

#docker build ./workers/annotations/cellori_segmentation/ -t annotations/cellori_segmentation_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Cellori" --label "interfaceCategory=Cellori"
docker compose -f ./workers/annotations/piscis/docker-compose.yaml build
docker build ./workers/annotations/cellpose/ -t annotations/cellpose_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Cellpose" --label "interfaceCategory=Cellpose" --label "annotationShape=polygon"
docker build ./workers/annotations/stardist/ -t annotations/stardist_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Stardist" --label "interfaceCategory=Stardist" --label "annotationShape=polygon"
docker build ./workers/annotations/laplacian_of_gaussian/ -t annotations/laplacian_of_gaussian:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Laplacian of Gaussian" --label "interfaceCategory=Laplacian of Gaussian"
#docker build ./workers/annotations/deepcell/ -t annotations/deepcell_worker:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=DeepCell" --label "interfaceCategory=Deepcell"

#docker build ./workers/test_worker/ -t both/test_worker:latest  --label isUPennContrastWorker --label isAnnotationWorker --label isPropertyWorker --label "annotationShape=point" --label "interfaceName=Test worker" --label "interfaceCategory=Test"
#docker build ./workers/annotations/test_multiple_annotation/ -t annotations/test_multiple_annotation:latest --label isUPennContrastWorker --label isAnnotationWorker --label "interfaceName=Random square" --label "interfaceCategory=random"
