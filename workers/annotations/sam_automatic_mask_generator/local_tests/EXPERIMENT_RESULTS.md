# SAM1 ViT-H vs SAM2: Few-Shot Segmentation Comparison

## Setup

Identical setup to the SAM2 experiments for direct comparison:

- **Dataset**: `69988c84b48d8121b565aba4` (1024x1022, 2 channels: Brightfield + YFP)
- **Annotations**: 544 polygon annotations tagged "YFP blob", all at XY=0, Z=3, T=0
- **Train/test split**: 10 training examples, 534 test (random seed 42)
- **Hardware**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **SAM1 model**: ViT-H (2.6B params, `sam_vit_h_4b8939.pth`)
- **Default parameters**: similarity threshold=0.5, target occupancy=0.20, min mask area=30, smoothing=0.3, IoU matching threshold=0.3

## SAM1 ViT-H Results

| Model | Pts/side | Candidates | Precision | Recall | F1 | Mean IoU | Mask Gen | Encoding | Total |
|-------|----------|-----------|-----------|--------|------|----------|----------|----------|-------|
| vit_h | 64 | 358 | 0.73 | 0.49 | 0.58 | 0.76 | 13s | 289s | ~302s |
| vit_h | 128 | 435 | 0.70 | 0.57 | 0.62 | 0.75 | 43s | 350s | ~393s |
| vit_h | 256 | 470 | 0.66 | 0.58 | 0.62 | 0.75 | 170s | 374s | ~544s |

## Head-to-Head Comparison (SAM1 vs SAM2)

Best results at each points-per-side setting:

| Pts/side | Model | Candidates | Precision | Recall | F1 | Mean IoU |
|----------|-------|-----------|-----------|--------|------|----------|
| 64 | SAM2 base_plus | 171 | **0.76** | 0.24 | 0.37 | 0.74 |
| 64 | **SAM1 vit_h** | **358** | 0.73 | **0.49** | **0.58** | **0.76** |
| 128 | SAM2 base_plus | 243 | **0.79** | 0.36 | 0.49 | 0.73 |
| 128 | **SAM1 vit_h** | **435** | 0.70 | **0.57** | **0.62** | **0.75** |
| 256 | SAM2 base_plus | 279 | **0.77** | 0.40 | 0.53 | 0.73 |
| 256 | **SAM1 vit_h** | **470** | 0.66 | **0.58** | **0.62** | **0.75** |

**Best overall**: SAM1 ViT-H at 128 pts/side (F1=0.62) vs SAM2 base_plus at 256 pts/side (F1=0.53). SAM1 wins by +0.09 F1.

## Key Findings

### SAM1 generates far more candidate masks

At every grid density, SAM1 ViT-H produces roughly 1.7-2.1x more candidates than SAM2 base_plus:

- 64 pts: 358 vs 171 (2.1x)
- 128 pts: 435 vs 243 (1.8x)
- 256 pts: 470 vs 279 (1.7x)

This directly addresses the recall bottleneck identified in the SAM2 experiments. SAM1's mask generator proposes masks for small microscopy blobs that SAM2 misses entirely.

### SAM1 recall is dramatically better

SAM1's recall at 64 pts/side (0.49) already exceeds SAM2's best recall at 256 pts/side (0.40). At 128 pts, SAM1 reaches 0.57 recall -- a 58% relative improvement over SAM2's best.

### Precision trades off slightly

SAM1's precision (0.66-0.73) is lower than SAM2's (0.76-0.79). More candidates means more false positives. However, the F1 improvement from higher recall more than compensates.

### SAM1 encoding is much slower

SAM1 ViT-H has 2.6B parameters vs SAM2 base_plus's ~80M. Feature encoding per candidate takes ~0.8s (SAM1) vs ~0.1s (SAM2). This makes SAM1 ~8x slower for the encoding phase. Total runtime at 128 pts: ~393s (SAM1) vs ~52s (SAM2).

### Similarity scores are even higher for SAM1

SAM1 candidates show mean similarity of 0.90-0.91 (vs SAM2's 0.85-0.88). Nearly all candidates pass the 0.5 threshold, confirming that the similarity filter isn't the bottleneck for either model.

### Diminishing returns at 256 pts for SAM1

SAM1's F1 plateaus between 128 and 256 pts (0.62 vs 0.62). The extra candidates at 256 pts mostly add false positives (precision drops from 0.70 to 0.66) while recall only improves marginally (0.57 to 0.58). 128 pts is the sweet spot for SAM1.

## Summary

| Metric | SAM2 best | SAM1 best | Winner |
|--------|-----------|-----------|--------|
| F1 | 0.53 (base_plus, 256pts) | **0.62** (vit_h, 128pts) | SAM1 (+17%) |
| Recall | 0.40 | **0.58** | SAM1 (+45%) |
| Precision | **0.77** | 0.70 | SAM2 |
| Mean IoU | 0.73 | **0.75** | SAM1 |
| Speed | **~52s** | ~393s | SAM2 (7.6x faster) |
| Candidates | 279 | **470** | SAM1 (1.7x more) |

## Recommendations

1. **For quality-sensitive use cases, prefer SAM1 ViT-H at 128 pts/side.** It produces the best F1 (0.62) and recall (0.57) of any configuration tested.
2. **For speed-sensitive use cases, SAM2 base_plus at 128 pts is a reasonable trade-off** (F1=0.49, ~52s total).
3. **The recall ceiling (~0.58) suggests the automatic mask generator approach has fundamental limits.** Even with SAM1's better candidate generation, ~42% of small blobs are still missed. Future work should explore:
   - Point-prompt mode with blob-center detection (e.g., intensity peak finding)
   - Tiled/sliding-window generation for dense small objects
   - Hybrid approaches: use SAM1 for candidate generation but SAM2 for fast encoding

## Reproducing These Results

```bash
cd workers/annotations/sam_automatic_mask_generator/local_tests

# First-time setup
bash setup_env.sh

# Activate environment
source .venv/bin/activate

# Best SAM1 configuration (vit_h, 128 pts/side)
python test_sam1_fewshot_experiment.py \
    --dataset 69988c84b48d8121b565aba4 \
    --tag "YFP blob" \
    --num-train 10 \
    --seed 42 \
    --points-per-side 128 \
    --min-area 30 \
    --username arjunraj --password sysbio \
    --visualize results_vit_h_128pts.png
```

## Technical Notes

- SAM1's mask generator requires `opencv-python-headless` (SAM2 does not).
- SAM1 is incompatible with `torch.autocast(dtype=torch.bfloat16)` -- its mask generator calls `.numpy()` on intermediate tensors which fails with bfloat16.
- At 256 pts/side, torchvision's `batched_nms` has a GPU device mismatch bug in the JIT-traced vanilla NMS path. The experiment script patches this by forcing the coordinate trick NMS path.
