# SAM2 Few-Shot Segmentation: Local Experiment Results

## Setup

- **Dataset**: `69988c84b48d8121b565aba4` (1024x1022, 2 channels: Brightfield + YFP)
- **Annotations**: 544 polygon annotations tagged "YFP blob", all at XY=0, Z=3, T=0
- **Train/test split**: 10 training examples, 534 test (random seed 42)
- **Hardware**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **SAM2 version**: SAM2.1 (facebookresearch/sam2, editable install)
- **Default parameters**: similarity threshold=0.5, target occupancy=0.20, min mask area=30, smoothing=0.3, IoU matching threshold=0.3

## Results

| Model | Pts/side | Candidates | Precision | Recall | F1 | Mean IoU | Mask Gen | Encoding | Total |
|-------|----------|-----------|-----------|--------|------|----------|----------|----------|-------|
| small | 32 | 40 | 0.64 | 0.05 | 0.09 | 0.75 | 2s | 3s | ~5s |
| small | 64 | 127 | 0.80 | 0.19 | 0.31 | 0.76 | 7s | 8s | ~15s |
| base_plus | 64 | 171 | 0.76 | 0.24 | 0.37 | 0.74 | 7s | 16s | ~24s |
| large | 64 | 153 | 0.73 | 0.21 | 0.33 | 0.75 | 7s | 27s | ~35s |
| large | 128 | 205 | 0.74 | 0.28 | 0.41 | 0.74 | 28s | 37s | ~65s |
| base_plus | 128 | 243 | 0.79 | 0.36 | 0.49 | 0.73 | 28s | 24s | ~52s |
| **base_plus** | **256** | **279** | **0.77** | **0.40** | **0.53** | **0.73** | **119s** | **28s** | **~147s** |

## Key Findings

### Points per side matters more than model size

Increasing the point grid density consistently improved recall by generating more candidate masks. Doubling points per side roughly doubled the number of candidates found. The base_plus model at 128 pts/side (F1=0.49) significantly outperformed the large model at 64 pts/side (F1=0.33).

### base_plus is the best model for this task

Counterintuitively, base_plus consistently produced more candidates than the large model at the same grid density (171 vs 153 at 64 pts, 243 vs 205 at 128 pts). It was also faster per candidate due to smaller encoder size. The large model's extra capacity did not translate into better microscopy blob segmentation.

### Precision is stable; recall is the bottleneck

Precision stayed between 0.73-0.80 across all configurations. The few-shot similarity matching is effective at filtering true positives from candidates. The limiting factor is SAM2's automatic mask generator not proposing enough candidates for small, dense microscopy objects.

### Similarity scores are consistently high

All candidates that passed SAM2's internal quality filters also tended to score high on cosine similarity (mean 0.85-0.88, min ~0.53). Nearly every candidate passed the 0.5 threshold. This suggests the similarity filter is not the bottleneck -- rather, SAM2 simply doesn't generate masks for most of the small blobs.

### Mean IoU is consistent at ~0.73-0.76

When a prediction does match a ground truth object, the overlap quality is good regardless of model or settings. The segmentation quality of individual masks is not the issue.

## Bug Found During Testing

The production `entrypoint.py` was passing `float32` images to `SAM2AutomaticMaskGenerator.generate()`, which caused SAM2 to produce 0 masks. SAM2 expects `uint8` input. Fixed in both `entrypoint.py` and the experiment script.

## Recommendations

1. **Default to base_plus** instead of small for the production worker -- it finds significantly more objects with moderate speed cost.
2. **Increase default points per side** from 32 to at least 64, ideally 128 for small objects. Consider making this adaptive based on training annotation sizes.
3. **Consider lowering default min mask area** from 100 to 30-50 for microscopy use cases where objects are small.
4. **Future work on candidate generation**: The automatic mask generator is fundamentally limited for dense small objects. Alternative approaches to explore:
   - Sliding-window / tiled mask generation with overlap
   - Using SAM2's point-prompt mode with a denser grid instead of the automatic generator
   - Pre-filtering with a simple intensity threshold to generate point prompts at blob centers
   - Full-image encoding with per-candidate feature pooling (avoids re-encoding per crop)

## Reproducing These Results

```bash
cd workers/annotations/sam2_fewshot_segmentation/local_tests
source .venv/bin/activate

# Best configuration (base_plus, 256 pts/side)
python test_sam2_fewshot_experiment.py \
    --dataset 69988c84b48d8121b565aba4 \
    --tag "YFP blob" \
    --num-train 10 \
    --seed 42 \
    --model sam2.1_hiera_base_plus.pt \
    --points-per-side 256 \
    --min-area 30 \
    --visualize results.png
```
