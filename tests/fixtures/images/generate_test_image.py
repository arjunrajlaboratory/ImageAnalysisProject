"""
Generate synthetic test images for local worker testing.

Creates small multi-channel TIFF images with cell-like blobs
suitable for testing segmentation and property workers.

Usage:
    python generate_test_image.py [--output-dir .]
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:
    print("tifffile required: pip install tifffile")
    raise


def make_blob(image, cy, cx, radius, intensity, rng):
    """Draw a filled circular blob on the image."""
    h, w = image.shape
    yy, xx = np.ogrid[max(0, cy - radius):min(h, cy + radius + 1),
                       max(0, cx - radius):min(w, cx + radius + 1)]
    yy_off = yy - cy
    xx_off = xx - cx
    mask = xx_off**2 + yy_off**2 <= radius**2
    region = image[max(0, cy - radius):min(h, cy + radius + 1),
                   max(0, cx - radius):min(w, cx + radius + 1)]
    region[mask] = np.maximum(region[mask], intensity)


def generate_cells_image(height=512, width=512, num_channels=3,
                         num_cells=30, seed=42):
    """Generate a multi-channel image with cell-like blobs.

    Channel 0: "DAPI" - bright nuclear blobs
    Channel 1: "GFP" - dimmer, larger cytoplasm-like blobs
    Channel 2: "mCherry" - sparse bright puncta
    """
    rng = np.random.default_rng(seed)
    channels = []

    # Channel 0: DAPI-like nuclei
    dapi = rng.integers(100, 300, (height, width), dtype=np.uint16)
    for _ in range(num_cells):
        cy = rng.integers(30, height - 30)
        cx = rng.integers(30, width - 30)
        radius = rng.integers(8, 20)
        intensity = rng.integers(2000, 10000)
        make_blob(dapi, cy, cx, radius, intensity, rng)
    channels.append(dapi)

    # Channel 1: GFP-like cytoplasm (larger, dimmer)
    gfp = rng.integers(50, 150, (height, width), dtype=np.uint16)
    for _ in range(num_cells):
        cy = rng.integers(30, height - 30)
        cx = rng.integers(30, width - 30)
        radius = rng.integers(15, 35)
        intensity = rng.integers(500, 3000)
        make_blob(gfp, cy, cx, radius, intensity, rng)
    channels.append(gfp)

    # Channel 2: mCherry-like puncta (small, bright, sparse)
    mcherry = rng.integers(80, 200, (height, width), dtype=np.uint16)
    for _ in range(num_cells * 3):
        cy = rng.integers(10, height - 10)
        cx = rng.integers(10, width - 10)
        radius = rng.integers(2, 6)
        intensity = rng.integers(3000, 15000)
        make_blob(mcherry, cy, cx, radius, intensity, rng)
    channels.append(mcherry)

    return np.stack(channels[:num_channels], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Generate test TIFF images")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--cells", type=int, default=30)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = generate_cells_image(
        height=args.height, width=args.width,
        num_channels=args.channels, num_cells=args.cells,
    )
    path = out / "synthetic_cells.tiff"
    tifffile.imwrite(str(path), img, metadata={"axes": "CYX"})
    print(f"Created {path}: shape={img.shape} dtype={img.dtype}")


if __name__ == "__main__":
    main()
