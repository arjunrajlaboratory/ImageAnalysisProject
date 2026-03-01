# NimbusImage Worker Registry

> Starting point for understanding what workers exist and what they do. Each worker has a detailed `WORKERNAME.md` documentation file in its directory.

## Annotation Workers

### Segmentation (ML)

| Worker | Path | Description | GPU? | Key Dependencies | Docs |
|--------|------|-------------|------|------------------|------|
| cellpose | `workers/annotations/cellpose/` | Cell segmentation using Cellpose models | Yes | cellpose | [CELLPOSE.md](workers/annotations/cellpose/CELLPOSE.md) |
| cellposesam | `workers/annotations/cellposesam/` | Two-stage segmentation: Cellpose detection + SAM refinement | Yes | cellpose, segment-anything | [CELLPOSESAM.md](workers/annotations/cellposesam/CELLPOSESAM.md) |
| cellpose_train | `workers/annotations/cellpose_train/` | Fine-tune Cellpose models on user annotations | Yes | cellpose | [CELLPOSE_TRAIN.md](workers/annotations/cellpose_train/CELLPOSE_TRAIN.md) |
| cellori_segmentation | `workers/annotations/cellori_segmentation/` | Cell segmentation using the Cellori library (dual-channel nuclei + cytoplasm) | Yes | cellori, JAX | [CELLORI_SEGMENTATION.md](workers/annotations/cellori_segmentation/CELLORI_SEGMENTATION.md) |
| condensatenet | `workers/annotations/condensatenet/` | Biomolecular condensate segmentation in brightfield images | Yes | condensatenet, DeepTile | [CONDENSATENET.md](workers/annotations/condensatenet/CONDENSATENET.md) |
| deepcell | `workers/annotations/deepcell/` | Cell segmentation using DeepCell/Mesmer models | Yes | deepcell | [DEEPCELL.md](workers/annotations/deepcell/DEEPCELL.md) |
| stardist | `workers/annotations/stardist/` | Star-convex polygon cell/nuclei segmentation | Yes | stardist | [STARDIST.md](workers/annotations/stardist/STARDIST.md) |
| piscis | `workers/annotations/piscis/` | Spot detection using Piscis (predict and train) | Yes | piscis | [PISCIS.md](workers/annotations/piscis/PISCIS.md) |
| ai_analysis | `workers/annotations/ai_analysis/` | LLM-powered analysis: Claude interprets queries and generates code to manipulate annotations | No | anthropic | [AI_ANALYSIS.md](workers/annotations/ai_analysis/AI_ANALYSIS.md) |
| annulus_generator_M1 | `workers/annotations/annulus_generator_M1/` | Generates annulus (ring) annotations around existing polygons | No | shapely | [ANNULUS_GENERATOR_M1.md](workers/annotations/annulus_generator_M1/ANNULUS_GENERATOR_M1.md) |
| deconwolf | `workers/annotations/deconwolf/` | 3D Richardson-Lucy deconvolution for fluorescence microscopy | Yes | deconwolf (compiled) | [DECONWOLF.md](workers/annotations/deconwolf/DECONWOLF.md) |

### Image Processing

| Worker | Path | Description | Docs |
|--------|------|-------------|------|
| gaussian_blur | `workers/annotations/gaussian_blur/` | Apply Gaussian blur filter to image channels | [GAUSSIAN_BLUR.md](workers/annotations/gaussian_blur/GAUSSIAN_BLUR.md) |
| rolling_ball | `workers/annotations/rolling_ball/` | Rolling ball background subtraction | [ROLLING_BALL.md](workers/annotations/rolling_ball/ROLLING_BALL.md) |
| laplacian_of_gaussian | `workers/annotations/laplacian_of_gaussian/` | LoG spot detection — creates point annotations at detected spots (2D or 3D) | [LAPLACIAN_OF_GAUSSIAN.md](workers/annotations/laplacian_of_gaussian/LAPLACIAN_OF_GAUSSIAN.md) |
| histogram_matching | `workers/annotations/histogram_matching/` | Match intensity histograms between images or channels | [HISTOGRAM_MATCHING.md](workers/annotations/histogram_matching/HISTOGRAM_MATCHING.md) |
| h_and_e_deconvolution | `workers/annotations/h_and_e_deconvolution/` | Separate H&E stain components from brightfield images | [H_AND_E_DECONVOLUTION.md](workers/annotations/h_and_e_deconvolution/H_AND_E_DECONVOLUTION.md) |
| registration | `workers/annotations/registration/` | Image registration/alignment across timepoints or channels | [REGISTRATION.md](workers/annotations/registration/REGISTRATION.md) |
| crop | `workers/annotations/crop/` | Subset image by XY/Z/Time ranges with optional spatial crop from a rectangle annotation | [CROP.md](workers/annotations/crop/CROP.md) |

### Connection Workers

| Worker | Path | Description | Docs |
|--------|------|-------------|------|
| connect_to_nearest | `workers/annotations/connect_to_nearest/` | Connect annotations to their nearest neighbors | [CONNECT_TO_NEAREST.md](workers/annotations/connect_to_nearest/CONNECT_TO_NEAREST.md) |
| connect_sequential | `workers/annotations/connect_sequential/` | Connect annotations to nearest neighbor in adjacent Time or Z slice | [CONNECT_SEQUENTIAL.md](workers/annotations/connect_sequential/CONNECT_SEQUENTIAL.md) |
| connect_timelapse | `workers/annotations/connect_timelapse/` | Track annotations across timepoints with gap bridging for temporary disappearances | [CONNECT_TIMELAPSE.md](workers/annotations/connect_timelapse/CONNECT_TIMELAPSE.md) |

### SAM Family

| Worker | Path | Description | GPU? | Docs |
|--------|------|-------------|------|------|
| sam_automatic_mask_generator | `workers/annotations/sam_automatic_mask_generator/` | Automatic mask generation using SAM1 | Yes | [SAM_AUTOMATIC_MASK_GENERATOR.md](workers/annotations/sam_automatic_mask_generator/SAM_AUTOMATIC_MASK_GENERATOR.md) |
| sam_fewshot_segmentation | `workers/annotations/sam_fewshot_segmentation/` | Few-shot segmentation using SAM1 with user-provided examples | Yes | [SAM_FEWSHOT.md](workers/annotations/sam_fewshot_segmentation/SAM_FEWSHOT.md) |
| sam2_automatic_mask_generator | `workers/annotations/sam2_automatic_mask_generator/` | Automatic mask generation using SAM2 | Yes | [SAM2_AUTOMATIC_MASK_GENERATOR.md](workers/annotations/sam2_automatic_mask_generator/SAM2_AUTOMATIC_MASK_GENERATOR.md) |
| sam2_fewshot_segmentation | `workers/annotations/sam2_fewshot_segmentation/` | Few-shot segmentation using SAM2 with user-provided examples | Yes | [SAM2_FEWSHOT.md](workers/annotations/sam2_fewshot_segmentation/SAM2_FEWSHOT.md) |
| sam2_propagate | `workers/annotations/sam2_propagate/` | Propagate segmentation masks across video frames using SAM2 | Yes | [SAM2_PROPAGATE.md](workers/annotations/sam2_propagate/SAM2_PROPAGATE.md) |
| sam2_refine | `workers/annotations/sam2_refine/` | Refine existing segmentation masks using SAM2 | Yes | [SAM2_REFINE.md](workers/annotations/sam2_refine/SAM2_REFINE.md) |
| sam2_video | `workers/annotations/sam2_video/` | Video segmentation and tracking using SAM2 | Yes | [SAM2_VIDEO.md](workers/annotations/sam2_video/SAM2_VIDEO.md) |

### Test/Demo Workers

| Worker | Path | Description | Docs |
|--------|------|-------------|------|
| random_squares | `workers/annotations/random_squares/` | Generate random square polygon annotations (test worker) | [RANDOM_SQUARES.md](workers/annotations/random_squares/RANDOM_SQUARES.md) |
| sample_interface | `workers/annotations/sample_interface/` | Demonstrates all interface types and messaging (reference) | [SAMPLE_INTERFACE.md](workers/annotations/sample_interface/SAMPLE_INTERFACE.md) |
| random_point | `workers/annotations/random_point/` | Generate random point annotations (test worker) | [RANDOM_POINT.md](workers/annotations/random_point/RANDOM_POINT.md) |
| random_point_annotation_M1 | `workers/annotations/random_point_annotation_M1/` | Generate random point annotations — M1/arm64 variant | [RANDOM_POINT_ANNOTATION_M1.md](workers/annotations/random_point_annotation_M1/RANDOM_POINT_ANNOTATION_M1.md) |
| test_multiple_annotation | `workers/annotations/test_multiple_annotation/` | Test creating multiple annotation types simultaneously | [TEST_MULTIPLE_ANNOTATION.md](workers/annotations/test_multiple_annotation/TEST_MULTIPLE_ANNOTATION.md) |
| test_multiple_annotation_M1 | `workers/annotations/test_multiple_annotation_M1/` | Multiple annotation test — M1/arm64 variant | [TEST_MULTIPLE_ANNOTATION_M1.md](workers/annotations/test_multiple_annotation_M1/TEST_MULTIPLE_ANNOTATION_M1.md) |

---

## Property Workers

### Blob Properties

| Worker | Path | Description | Docs |
|--------|------|-------------|------|
| blob_intensity_worker | `workers/properties/blobs/blob_intensity_worker/` | Compute per-channel intensity statistics for polygon annotations | [BLOB_INTENSITY.md](workers/properties/blobs/blob_intensity_worker/BLOB_INTENSITY.md) |
| blob_metrics_worker | `workers/properties/blobs/blob_metrics_worker/` | Compute geometric metrics (area, perimeter, circularity, etc.) | [BLOB_METRICS.md](workers/properties/blobs/blob_metrics_worker/BLOB_METRICS.md) |
| blob_overlap_worker | `workers/properties/blobs/blob_overlap_worker/` | Compute overlap/intersection between polygon annotations | [BLOB_OVERLAP.md](workers/properties/blobs/blob_overlap_worker/BLOB_OVERLAP.md) |
| blob_point_count_worker | `workers/properties/blobs/blob_point_count_worker/` | Count point annotations contained within each polygon | [BLOB_POINT_COUNT.md](workers/properties/blobs/blob_point_count_worker/BLOB_POINT_COUNT.md) |
| blob_point_count_3D_projection_worker | `workers/properties/blobs/blob_point_count_3D_projection_worker/` | Count points in polygons with 3D Z-projection | [BLOB_POINT_COUNT_3D_PROJECTION.md](workers/properties/blobs/blob_point_count_3D_projection_worker/BLOB_POINT_COUNT_3D_PROJECTION.md) |
| blob_intensity_percentile_worker | `workers/properties/blobs/blob_intensity_percentile_worker/` | Compute intensity percentile values within polygons | [BLOB_INTENSITY_PERCENTILE.md](workers/properties/blobs/blob_intensity_percentile_worker/BLOB_INTENSITY_PERCENTILE.md) |
| blob_annulus_intensity_worker | `workers/properties/blobs/blob_annulus_intensity_worker/` | Compute intensity in annular (ring) regions around polygons | [BLOB_ANNULUS_INTENSITY.md](workers/properties/blobs/blob_annulus_intensity_worker/BLOB_ANNULUS_INTENSITY.md) |
| blob_annulus_intensity_percentile_worker | `workers/properties/blobs/blob_annulus_intensity_percentile_worker/` | Compute intensity percentiles in annular regions | [BLOB_ANNULUS_INTENSITY_PERCENTILE.md](workers/properties/blobs/blob_annulus_intensity_percentile_worker/BLOB_ANNULUS_INTENSITY_PERCENTILE.md) |
| blob_colony_two_color_intensity_worker | `workers/properties/blobs/blob_colony_two_color_intensity_worker/` | Two-color colony intensity analysis for co-localization | [BLOB_COLONY_TWO_COLOR_INTENSITY.md](workers/properties/blobs/blob_colony_two_color_intensity_worker/BLOB_COLONY_TWO_COLOR_INTENSITY.md) |
| blob_random_forest_classifier | `workers/properties/blobs/blob_random_forest_classifier/` | Train random forest on tagged annotations, predict class labels for untagged blobs | [BLOB_RANDOM_FOREST_CLASSIFIER.md](workers/properties/blobs/blob_random_forest_classifier/BLOB_RANDOM_FOREST_CLASSIFIER.md) |

### Point Properties

| Worker | Path | Description | Docs |
|--------|------|-------------|------|
| point_intensity_worker | `workers/properties/points/point_intensity_worker/` | Compute intensity at exact point annotation locations | [POINT_INTENSITY.md](workers/properties/points/point_intensity_worker/POINT_INTENSITY.md) |
| point_circle_intensity_worker | `workers/properties/points/point_circle_intensity_worker/` | Compute intensity in circular regions around points (per-Z) | [POINT_CIRCLE_INTENSITY.md](workers/properties/points/point_circle_intensity_worker/POINT_CIRCLE_INTENSITY.md) |
| point_circle_intensity_mean_worker | `workers/properties/points/point_circle_intensity_mean_worker/` | Compute mean intensity in circular regions around points | [POINT_CIRCLE_INTENSITY_MEAN.md](workers/properties/points/point_circle_intensity_mean_worker/POINT_CIRCLE_INTENSITY_MEAN.md) |
| point_threshold_intensity_mean_worker | `workers/properties/points/point_threshold_intensity_mean_worker/` | Compute thresholded intensity in circular regions | [POINT_THRESHOLD_INTENSITY_MEAN.md](workers/properties/points/point_threshold_intensity_mean_worker/POINT_THRESHOLD_INTENSITY_MEAN.md) |
| point_metrics_worker | `workers/properties/points/point_metrics_worker/` | Record x/y coordinates of point annotations as properties | [POINT_METRICS.md](workers/properties/points/point_metrics_worker/POINT_METRICS.md) |
| point_to_nearest_point_distance | `workers/properties/points/point_to_nearest_point_distance/` | Distance from each point to nearest point of a target tag | [POINT_TO_NEAREST_POINT_DISTANCE.md](workers/properties/points/point_to_nearest_point_distance/POINT_TO_NEAREST_POINT_DISTANCE.md) |
| point_to_nearest_blob_distance | `workers/properties/points/point_to_nearest_blob_distance/` | Distance from each point to nearest polygon boundary | [POINT_TO_NEAREST_BLOB_DISTANCE.md](workers/properties/points/point_to_nearest_blob_distance/POINT_TO_NEAREST_BLOB_DISTANCE.md) |
| point_to_nearest_connected_point_distance | `workers/properties/points/point_to_nearest_connected_point_distance/` | Distance from each point to nearest connected point | [POINT_TO_NEAREST_CONNECTED_POINT_DISTANCE.md](workers/properties/points/point_to_nearest_connected_point_distance/POINT_TO_NEAREST_CONNECTED_POINT_DISTANCE.md) |

### Line Properties

| Worker | Path | Description | Docs |
|--------|------|-------------|------|
| line_length_worker | `workers/properties/lines/line_length_worker/` | Compute physical length of line annotations | [LINE_LENGTH.md](workers/properties/lines/line_length_worker/LINE_LENGTH.md) |
| line_scan_worker | `workers/properties/lines/line_scan_worker/` | Compute intensity profiles along line annotations | [LINE_SCAN.md](workers/properties/lines/line_scan_worker/LINE_SCAN.md) |
| test_file_creation_worker | `workers/properties/lines/test_file_creation_worker/` | Test worker for file creation and upload workflow | [TEST_FILE_CREATION.md](workers/properties/lines/test_file_creation_worker/TEST_FILE_CREATION.md) |

### Connection Properties

| Worker | Path | Description | Docs |
|--------|------|-------------|------|
| children_count_worker | `workers/properties/connections/children_count_worker/` | Count child annotations for each parent connection | [CHILDREN_COUNT.md](workers/properties/connections/children_count_worker/CHILDREN_COUNT.md) |
| parent_child_worker | `workers/properties/connections/parent_child_worker/` | Assign sequential IDs and record parent/child relationships with optional track IDs | [PARENT_CHILD.md](workers/properties/connections/parent_child_worker/PARENT_CHILD.md) |

---

## Other

| Worker | Path | Description |
|--------|------|-------------|
| test_worker | `workers/test_worker/` | Internal test worker for CI infrastructure |

---

## Shared Base Images (ML Workers)

| Image | Dockerfile | CUDA | Used by |
|-------|-----------|------|---------|
| `nimbusimage/sam2-worker-base` | `workers/base_docker_images/Dockerfile.sam2_worker_base` | 12.1 | All 5 SAM2 workers |
| `nimbusimage/sam2-worker-base-m1` | `workers/base_docker_images/Dockerfile.sam2_worker_base_M1` | 11.8 | All 5 SAM2 workers (M1) |
| `nimbusimage/cuda-ml-worker-base` | `workers/base_docker_images/Dockerfile.cuda_ml_worker_base` | 11.8 | cellpose, cellpose_train, cellposesam, stardist, SAM1 workers |
