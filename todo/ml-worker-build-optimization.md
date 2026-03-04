# TODO-001: ML Worker Build Optimization

**Status:** Deferred
**Priority:** Medium
**Related PR:** [#132 — Refactor ML worker Dockerfiles: mamba + shared base images](https://github.com/arjunrajlaboratory/ImageAnalysisProject/pull/132)

## Summary

PR #132 attempted to reduce ML worker Docker build times by creating shared base images (`sam2-worker-base`, `cuda-ml-worker-base`) and switching from conda to mamba. After testing, GPU passthrough issues forced reverting 8 of 11 workers, leaving only incremental gains for 3 workers.

## What Worked

- **Mamba for environment resolution**: Significantly faster than conda for solving environments. All workers that adopted mamba built faster.
- **Miniforge over Miniconda**: Eliminates Anaconda ToS acceptance (`CONDA_PLUGINS_AUTO_ACCEPT_TOS`) and provides mamba out of the box. Piscis was successfully switched.
- **Shared base image for non-GPU workers**: SAM1 (sam_automatic_mask_generator, sam_fewshot_segmentation) and stardist successfully use `cuda-ml-worker-base`, reducing their Dockerfiles from ~70 lines to ~15 lines.
- **Standardized repo references**: All workers updated from `Kitware/UPennContrast` to `arjunrajlaboratory/NimbusImage`.
- **Bug fixes discovered**: sam2_refine/Dockerfile_M1 had a copy-paste bug (copied from sam2_propagate), stardist had duplicate deeptile installs.

## What Didn't Work

### GPU passthrough with shared base images (primary blocker)

The NVIDIA Container Toolkit uses Docker labels (`com.nvidia.volumes.needed`, `nvidia_driver`, etc.) to configure GPU access inside containers. When workers inherit from a shared base image that was built *without* GPU access, the runtime GPU passthrough fails — even if `--gpus all` is passed at `docker run` time.

**Affected workers (reverted to standalone Dockerfiles):**
- Cellpose (3 workers): cellpose, cellpose_train, cellposesam
- SAM2 (5 workers): sam2_automatic_mask_generator, sam2_fewshot_segmentation, sam2_propagate, sam2_refine, sam2_video

### PyTorch CUDA version pinning

SAM2 workers need CUDA 12.1+ for PyTorch, while Cellpose workers use CUDA 11.8. A single shared base image couldn't satisfy both without version-specific variants, negating much of the simplification benefit.

## Recommendations for Future Attempts

1. **Investigate NVIDIA Container Toolkit label propagation**: The core issue is whether GPU labels can be added in child images or must exist in the base. Check if newer toolkit versions handle this differently.
2. **Consider separate base images per CUDA version**: Instead of one `cuda-ml-worker-base`, create `cuda-ml-base:11.8` and `cuda-ml-base:12.1`. This adds complexity but preserves the DRY benefit.
3. **Multi-stage builds as alternative**: Instead of shared base images, use multi-stage builds where the final stage starts FROM the correct NVIDIA CUDA image and copies the conda env from a builder stage.
4. **Keep the 3 working workers on shared base**: SAM1 (x2) and stardist work fine on `cuda-ml-worker-base`. No reason to revert these.

## Current State (as of archival)

- PR #132 is archived (draft + `archived` label)
- 3 workers use shared base: sam_automatic_mask_generator, sam_fewshot_segmentation, stardist
- 8 workers reverted to standalone Dockerfiles
- `build_machine_learning_workers.sh` builds all ML workers correctly
- Base image Dockerfiles remain in `workers/base_docker_images/` for reference
