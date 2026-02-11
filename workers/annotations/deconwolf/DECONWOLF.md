# Deconwolf Worker

This worker uses [deconwolf](https://github.com/elgw/deconwolf), an open-source tool for 3D image deconvolution using the Richardson-Lucy algorithm with a theoretically generated Born-Wolf point spread function (PSF) model. It is designed for fluorescence microscopy Z-stacks and supports GPU acceleration via OpenCL with automatic CPU fallback.

## Background

Fluorescence microscopy images are inherently blurred by the optical system's point spread function. Deconvolution computationally reverses this blurring to recover sharper images with improved contrast and resolution. The Richardson-Lucy algorithm is an iterative maximum-likelihood approach well suited for photon-counting (Poisson) noise found in fluorescence microscopy.

Deconwolf was developed by Erik Wernersson and is described in the paper:

> Wernersson, E. (2024). deconwolf — Large deconvolution with GPU or CPU. *SoftwareX*, 27, 101747. https://doi.org/10.1016/j.softx.2024.101747

The worker wraps two deconwolf command-line tools:
- **`dw_bw`**: Generates a theoretical PSF using the Born-Wolf diffraction model given optical parameters (NA, wavelength, refractive index, pixel size, Z step).
- **`dw`**: Performs Richardson-Lucy deconvolution of a Z-stack against a PSF, with optional GPU acceleration and tiling for large images.

## How It Works

1. **Channel Selection**: The user selects which channels to deconvolve. Unselected channels pass through to the output unchanged.
2. **Parameter Extraction**: Optical parameters (NA, refractive index, pixel size, Z step, emission wavelengths) are either entered manually or auto-extracted from ND2 image metadata when available.
3. **PSF Generation**: For each unique emission wavelength among the selected channels, a Born-Wolf PSF is generated using `dw_bw`. The PSF size is set to `2 * num_z_slices - 1` to cover the full axial extent of the image.
4. **Deconvolution**: Each Z-stack (grouped by XY position, timepoint, and channel) is deconvolved using `dw` with the corresponding PSF. GPU acceleration is attempted first if enabled, with automatic fallback to CPU if OpenCL fails.
5. **Output Assembly**: Deconvolved and pass-through frames are assembled into a multi-dimensional TIFF file preserving the original image structure (channels, Z, time, XY positions).
6. **Upload**: The result is uploaded back to the server with metadata recording all parameters used (iterations, optical parameters, GPU usage, tiling settings).

### 2D Image Handling

If the input image has only a single Z-slice, deconvolution is not applicable. The worker detects this, sends a warning, and copies the image unchanged to the output.

## Interface Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| **Channels to deconvolve** | channelCheckboxes | — | — | Select which channels to deconvolve; unselected channels pass through unchanged |
| **Auto-extract from ND2** | checkbox | false | — | Attempt to read optical parameters (pixel size, NA, refractive index) from image metadata |
| **Numerical Aperture (NA)** | number | 0.75 | 0.1–1.7 | Numerical aperture of the objective lens |
| **Refractive Index (ni)** | number | 1.0 | 1.0–1.6 | Immersion medium refractive index (1.0 for air, 1.515 for oil) |
| **Pixel Size XY (nm)** | number | 325 | 1–10000 | Lateral pixel size in nanometers |
| **Z Step (nm)** | number | 5000 | 1–50000 | Axial step size between Z-slices in nanometers |
| **Emission Wavelength (nm)** | text | — | — | Comma-separated emission wavelengths, one per channel in order. A single value applies to all channels. If fewer values than channels, they cycle. Defaults to 450, 520, 580, 680 nm if left blank. |
| **Iterations** | number | 50 | 1–200 | Number of Richardson-Lucy iterations. Higher values produce sharper results but increase computation time. Typical range: 20–100. |
| **Use GPU** | checkbox | true | — | Enable GPU acceleration via OpenCL. Falls back to CPU automatically if no compatible GPU is available. |
| **Tile Size (pixels)** | number | 1024 | 256–8192 | Maximum tile dimension for processing large images. Images with either dimension exceeding this value are processed in tiles. |
| **Tile Overlap (pixels)** | number | 100 | 0–500 | Overlap between adjacent tiles in pixels to reduce edge artifacts at tile boundaries. |

## Implementation Details

### GPU Acceleration and CPU Fallback

The production Docker image is built on `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` with OpenCL support configured via an NVIDIA ICD (Installable Client Driver) registration. When "Use GPU" is enabled:

1. Deconvolution is first attempted with `dw --gpu`.
2. If it fails with an OpenCL error (detected by checking for `cl_util.c` or `OpenCL` in stderr), the worker automatically retries without `--gpu` and sends a warning to the user.
3. Once a GPU failure is detected for one stack, subsequent stacks in the same job skip the GPU attempt entirely to avoid repeated failures.

This means the worker is safe to deploy on machines without GPUs — it will always fall back gracefully.

### Tiling for Large Images

For images where `max(height, width) > tile_size`, the worker enables deconwolf's built-in tiling mode:

- `--tilesize`: Sets the maximum tile dimension (default 1024 pixels).
- `--tilepad`: Sets the overlap between tiles (default 100 pixels) to minimize edge artifacts.

Images that fit within the tile size are processed as a single piece without tiling overhead. The comparison is strict greater-than, so a 1024x1024 image with tile size 1024 will not be tiled.

### ND2 Metadata Extraction

When "Auto-extract from ND2" is enabled, the worker attempts to read optical parameters from the image metadata stored in Girder:

- **Pixel size**: Extracted from `mm_x`/`mm_y` fields in tile metadata, converted from mm to nm.
- **NA and refractive index**: Extracted from ND2 internal metadata (`nd2.channels[0].microscope`), if present.
- **Emission wavelengths**: Extracted from ND2 channel metadata (`emissionLambdaNm`), if present.

Any parameters not found in metadata fall back to the manually entered values, so partial metadata extraction still works.

### Wavelength Handling

The emission wavelength input supports several formats:
- **Empty**: Uses default wavelengths (450, 520, 580, 680 nm) cycling across channels.
- **Single value** (e.g., `520`): Applied to all selected channels.
- **Comma-separated** (e.g., `450,520,580,680`): Maps to channels in order.
- **Fewer values than channels**: Values cycle (e.g., `450,520` with 4 channels gives 450, 520, 450, 520).

### Output Metadata

The uploaded output file includes metadata recording all deconvolution parameters for reproducibility:

```json
{
    "tool": "Deconwolf Deconvolution",
    "channels_deconvolved": [0, 1],
    "iterations": 50,
    "NA": 0.75,
    "ni": 1.0,
    "resxy_nm": 325,
    "resz_nm": 5000,
    "wavelengths": {"0": 450, "1": 520},
    "gpu_used": true,
    "tile_size": 1024,
    "tile_overlap": 100
}
```

## Docker Build

### Production (GPU, x86_64)

Built from `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`. Deconwolf is compiled from source with CMake, linking against FFTW3, GSL, and OpenCL. The NVIDIA OpenCL ICD is registered so the CUDA runtime's OpenCL implementation is discoverable.

```bash
./build_workers.sh deconwolf
```

### Mac Development (CPU-only, arm64)

Built from `nimbusimage/image-processing-base:latest`. Same deconwolf build but without NVIDIA/CUDA dependencies. Used for local development on Apple Silicon.

```bash
MAC_DEVELOPMENT_MODE=true ./build_workers.sh deconwolf
```

## Deconwolf Source

Deconwolf is cloned from [https://github.com/elgw/deconwolf](https://github.com/elgw/deconwolf) and built from source at Docker build time. It depends on:

- **FFTW3**: Fast Fourier transforms for the convolution operations
- **GSL**: GNU Scientific Library for numerical routines
- **OpenCL** (optional): GPU acceleration via the OpenCL framework
- **libtiff/libpng**: Image I/O

The build produces two executables installed to `/usr`:
- `dw`: The main deconvolution tool
- `dw_bw`: The Born-Wolf PSF generator
