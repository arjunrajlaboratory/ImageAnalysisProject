import argparse
import json
import os
import sys
from collections import defaultdict

import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning, sendError

import numpy as np


def interface(image, apiUrl, token):
    """Define the worker interface shown to users.

    This worker exposes a single ``Method`` dropdown plus the union of all
    per-method parameters (there is no conditional UI in this framework).
    Each parameter's tooltip states which method(s) it applies to; unused
    parameters for the currently-selected method are simply ignored by
    ``compute()``. See FLUOR_CORRECTION_WORKERS_SPEC.md ("Worker 2:
    image_restoration") for the full design rationale.
    """
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Method': {
            'type': 'select',
            'items': ['n2v', 'cellpose3', 'zs_deconvnet', 'fluoresfm'],
            'default': 'n2v',
            'tooltip': (
                'Restoration algorithm. All four are reference-free/pretrained '
                '(no paired clean ground truth required): n2v (Noise2Void/N2V2, '
                'self-supervised denoise, trains on your data), cellpose3 '
                '(pretrained denoise/deblur/upsample), zs_deconvnet (zero-shot, '
                'trains on the single input), fluoresfm (pretrained, '
                'text-prompted foundation model, experimental).'
            ),
            'displayOrder': 0,
        },
        'Channels to restore': {
            'type': 'channelCheckboxes',
            'tooltip': 'Process selected channels; unselected channels pass through unchanged.',
            'displayOrder': 1,
        },
        'Use GPU': {
            'type': 'checkbox',
            'default': True,
            'tooltip': (
                'Use GPU (CUDA) acceleration for all methods. Falls back to CPU '
                'automatically with a warning if CUDA is unavailable; CPU '
                'fallback is slow for n2v/zs_deconvnet/fluoresfm.'
            ),
            'displayOrder': 2,
        },
        # --- n2v (Noise2Void / N2V2 via CAREamics) ---
        'Epochs': {
            'type': 'number',
            'min': 1,
            'max': 1000,
            'default': 20,
            'tooltip': 'n2v: number of self-supervised training epochs on the per-channel image collection.',
            'displayOrder': 3,
        },
        'Use N2V2': {
            'type': 'checkbox',
            'default': True,
            'tooltip': 'n2v: use the N2V2 variant (reduces checkerboard artifacts vs. classic N2V).',
            'displayOrder': 4,
        },
        'Patch size': {
            'type': 'number',
            'min': 16,
            'max': 512,
            'default': 64,
            'tooltip': 'n2v: training patch size in pixels (square patches).',
            'displayOrder': 5,
        },
        # --- cellpose3 restoration ---
        'Cellpose3 model': {
            'type': 'select',
            'items': [
                'denoise_cyto3',
                'deblur_cyto3',
                'upsample_cyto3',
                'denoise_nuclei',
                'deblur_nuclei',
                'oneclick_cyto3',
            ],
            'default': 'denoise_cyto3',
            'tooltip': (
                'cellpose3: pretrained restoration model to apply per frame. '
                'Best used as segmentation preprocessing, not for quantitative '
                'intensity restoration.'
            ),
            'displayOrder': 6,
        },
        # --- zs_deconvnet (zero-shot denoise + deconvolution) ---
        'ZS iterations': {
            'type': 'number',
            'min': 1,
            'max': 2000,
            'default': 300,
            'tooltip': 'zs_deconvnet: number of zero-shot self-supervised training steps per frame/stack.',
            'displayOrder': 7,
        },
        'ZS upsampling': {
            'type': 'checkbox',
            'default': False,
            'tooltip': (
                'zs_deconvnet: apply extra physics-consistency deconvolution '
                'refinement (closer to the published super-resolution mode) '
                'instead of denoise-only. Output pixel dimensions are always kept '
                'identical to the input so annotations/scale stay pixel-aligned; '
                'this does not perform true upsampling of the image grid.'
            ),
            'displayOrder': 8,
        },
        'Numerical Aperture (NA)': {
            'type': 'number',
            'min': 0.1,
            'max': 1.7,
            'default': 0.75,
            'tooltip': 'zs_deconvnet: numerical aperture, used to build an approximate Gaussian PSF for the deconvolution stage.',
            'displayOrder': 9,
        },
        'Emission Wavelength (nm)': {
            'type': 'number',
            'min': 300,
            'max': 800,
            'default': 520,
            'tooltip': 'zs_deconvnet: emission wavelength in nanometers, used to build the approximate PSF.',
            'displayOrder': 10,
        },
        'Pixel Size XY (nm)': {
            'type': 'number',
            'min': 1,
            'max': 10000,
            'default': 325,
            'tooltip': 'zs_deconvnet: lateral pixel size in nanometers, used to convert the PSF to pixel units.',
            'displayOrder': 11,
        },
        # --- fluoresfm (foundation model) ---
        'FluoResFM task': {
            'type': 'select',
            'items': ['denoise', 'deconvolution', 'super-resolution'],
            'default': 'denoise',
            'tooltip': 'fluoresfm: restoration task passed to the text-conditioned foundation model. Experimental.',
            'displayOrder': 12,
        },
        'FluoResFM text prompt': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'e.g. fluorescence microscopy image, denoising',
                'label': 'FluoResFM text prompt (optional)',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'tooltip': (
                'fluoresfm: free-text prompt describing the structure/task '
                '(the model is text-conditioned). Leave blank to use a default '
                'prompt derived from the selected task.'
            ),
            'displayOrder': 13,
        },
    }
    client.setWorkerImageInterface(image, interface)


def resolve_device(use_gpu):
    """Resolve the torch device string ('cuda' or 'cpu') for restoration methods.

    Only imports torch when GPU use is actually requested, so choosing 'Use
    GPU' = False never pulls in the heavy torch dependency at all. When GPU is
    requested but CUDA is unavailable, falls back to CPU with a sendWarning
    (pattern: deconwolf's OpenCL->CPU fallback).
    """
    if not use_gpu:
        return 'cpu'

    import torch

    if torch.cuda.is_available():
        return 'cuda'

    sendWarning(
        "GPU requested but CUDA is not available.",
        info=(
            "Falling back to CPU. ML-based restoration (n2v, zs_deconvnet, "
            "fluoresfm) will be significantly slower on CPU; cellpose3 is "
            "usually still reasonably fast."
        ),
    )
    return 'cpu'


def resolve_fluoresfm_weights():
    """Resolve the path to the FluoResFM pretrained checkpoint.

    FluoResFM weights are not reliably baked into the Docker image (the build
    only *attempts* a best-effort download, see download_models.py, and must
    not fail the build if that download fails). The runtime source of truth is
    therefore the FLUORESFM_WEIGHTS environment variable, which should point
    to a mounted/downloaded ``.pt`` checkpoint. Returns the resolved path, or
    None (after sending an actionable sendError) if unavailable.
    """
    weights_path = os.environ.get('FLUORESFM_WEIGHTS', '').strip()
    if not weights_path or not os.path.isfile(weights_path):
        sendError(
            "FluoResFM weights are not available.",
            info=(
                "Set the FLUORESFM_WEIGHTS environment variable to a mounted "
                "FluoResFM checkpoint (.pt) or choose a different Method. See "
                "IMAGE_RESTORATION.md for download instructions -- weights are "
                "distributed via Google Drive/Baidu Yun from "
                "https://github.com/qiqi-lu/fluoresfm, not a stable direct-download "
                "URL, so they cannot be baked into the image reliably."
            ),
        )
        return None
    return weights_path


def _rescale_to_reference(output, reference):
    """Linearly rescale `output`'s dynamic range to match `reference`'s.

    Pretrained restorers (Cellpose3, FluoResFM) often emit intensities in a
    normalized range (e.g. ~0-1) rather than the source sensor range. Casting
    such output straight into an integer source dtype via `_clip_to_dtype`
    would clip almost everything to 0/1 and produce a near-black TIFF. Mapping
    the output's [min, max] onto the reference frame's [min, max] preserves the
    original intensity scale for downstream quantitative use.
    """
    output = np.asarray(output, dtype=np.float32)
    out_min = float(output.min())
    out_max = float(output.max())
    out_span = out_max - out_min
    if out_span <= 0:
        return np.full_like(output, float(np.asarray(reference).min()))
    ref = np.asarray(reference, dtype=np.float32)
    ref_min = float(ref.min())
    ref_span = float(ref.max()) - ref_min
    return (output - out_min) / out_span * ref_span + ref_min


def _clip_to_dtype(image, dtype):
    """Clip and cast a restored image back to the source dtype.

    Guards against NaN/inf that ML models can introduce, and avoids silently
    bloating the output TIFF to float when the source was e.g. uint16 (see
    CLAUDE.md's dtype-handling guidance).
    """
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if dtype is None:
        return image
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        image = np.clip(image, info.min, info.max)
    return image.astype(dtype)


def _build_method_opts(method, workerInterface):
    """Extract only the parameters relevant to the selected method.

    Unused parameters for other methods are ignored (there is no conditional
    UI in this framework -- all parameters are always shown; see the
    docstring on ``interface()``).
    """
    if method == 'n2v':
        return {
            'epochs': int(workerInterface.get('Epochs', 20)),
            'use_n2v2': bool(workerInterface.get('Use N2V2', True)),
            'patch_size': int(workerInterface.get('Patch size', 64)),
        }
    if method == 'cellpose3':
        return {
            'model_type': workerInterface.get('Cellpose3 model', 'denoise_cyto3'),
        }
    if method == 'zs_deconvnet':
        return {
            'iterations': int(workerInterface.get('ZS iterations', 300)),
            'upsampling': bool(workerInterface.get('ZS upsampling', False)),
            'NA': float(workerInterface.get('Numerical Aperture (NA)', 0.75)),
            'wavelength': float(workerInterface.get('Emission Wavelength (nm)', 520)),
            'pixel_size_xy': float(workerInterface.get('Pixel Size XY (nm)', 325)),
        }
    if method == 'fluoresfm':
        task = workerInterface.get('FluoResFM task', 'denoise')
        default_prompt = f"fluorescence microscopy image, {task}"
        prompt = workerInterface.get('FluoResFM text prompt', '') or default_prompt
        return {
            'task': task,
            'prompt': prompt,
        }
    return {}


# ---------------------------------------------------------------------------
# Per-algorithm restoration functions.
#
# Each function has the signature ``restore_x(stack, opts, device) -> ndarray
# | None`` where ``stack`` is a ``(N, Y, X)`` float array (N frames of one
# channel's collection). Heavy third-party imports (careamics, cellpose,
# torch, the vendored zs_deconvnet/fluoresfm code) are imported lazily,
# *inside* these functions only, so that:
#   1. the interface() path stays fast (see todo/worker-startup-latency.md),
#   2. the pytest suite can run natively without any of these installed, by
#      mocking these functions directly (mirrors
#      histogram_matching/tests/test_histogram_matching.py's
#      entrypoint.match_histograms patch).
#
# On a missing dependency or missing weights, these functions call
# sendError(...) with an actionable message and return None; compute() checks
# for None and aborts cleanly instead of crashing.
# ---------------------------------------------------------------------------


def restore_n2v(stack, opts, device):
    """Self-supervised denoising with Noise2Void / N2V2 via CAREamics.

    Trains on the channel's own collection of frames (no clean target needed)
    then predicts on the same collection. GPU is strongly recommended; CPU
    works but is slow.
    """
    try:
        from careamics import CAREamist
        from careamics.config import create_n2v_configuration
    except ImportError:
        sendError(
            "Noise2Void (careamics) is not installed in this worker image.",
            info="Install with `pip install careamics` and rebuild the image, or choose a different Method.",
        )
        return None

    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]

    n_frames = stack.shape[0]
    patch = int(opts.get('patch_size', 64))
    # CAREamics requires the patch to fit inside the image, so clamp to the
    # smaller image dimension. Do NOT re-raise a 16px floor above that: for an
    # image smaller than 16px on a side, max(16, ...) would force the patch
    # back above the image size and CAREamics would reject it.
    patch = min(patch, stack.shape[-1], stack.shape[-2])

    config = create_n2v_configuration(
        experiment_name='image_restoration_n2v',
        data_type='array',
        axes='SYX' if n_frames > 1 else 'YX',
        patch_size=[patch, patch],
        batch_size=min(16, max(1, n_frames)),
        num_epochs=int(opts.get('epochs', 20)),
        use_n2v2=bool(opts.get('use_n2v2', True)),
    )

    engine = CAREamist(config)
    train_source = stack if n_frames > 1 else stack[0]
    engine.train(train_source=train_source)
    predicted = engine.predict(source=train_source)

    predicted = np.asarray(predicted, dtype=np.float32).squeeze()
    if predicted.ndim == 2:
        predicted = predicted[np.newaxis, ...]
    return predicted


def restore_cellpose3(stack, opts, device):
    """Pretrained Cellpose3 restoration (denoise/deblur/upsample), applied per frame.

    Best used as segmentation preprocessing rather than for quantitative
    intensity restoration (see IMAGE_RESTORATION.md).
    """
    try:
        from cellpose import denoise
    except ImportError:
        sendError(
            "Cellpose3 restoration models are not installed in this worker image.",
            info="Install with `pip install cellpose>=3` and rebuild the image, or choose a different Method.",
        )
        return None

    model_type = opts.get('model_type', 'denoise_cyto3')
    gpu = device == 'cuda'
    model = denoise.DenoiseModel(model_type=model_type, gpu=gpu)

    stack = np.asarray(stack)
    results = []
    for frame in stack:
        restored = model.eval(frame, channels=[0, 0])
        # DenoiseModel.eval can return the array directly or a (image, ...) tuple
        # depending on cellpose version; normalize to a single 2D array.
        if isinstance(restored, (list, tuple)):
            restored = restored[0]
        restored = np.asarray(restored, dtype=np.float32).squeeze()
        # Cellpose restoration output is percentile-normalized (~0-1), not in the
        # source sensor range; rescale back so it isn't clipped to near-black by
        # the later _clip_to_dtype cast.
        restored = _rescale_to_reference(restored, frame)
        results.append(restored)

    return np.stack(results, axis=0)


def _zs_gaussian_psf_sigma(na, wavelength_nm, pixel_size_nm):
    """Approximate the diffraction-limited PSF as an isotropic Gaussian.

    Uses the Abbe resolution criterion (resolution ~= 0.21 * lambda / NA) as a
    stand-in for a full Born-Wolf PSF (c.f. deconwolf's dw_bw), so the
    zero-shot deconvolution stage below has a usable, if approximate, PSF
    without requiring a separate PSF-generation tool.
    """
    if na <= 0 or pixel_size_nm <= 0:
        return 1.0
    resolution_nm = 0.21 * wavelength_nm / na
    sigma_px = resolution_nm / pixel_size_nm
    return max(sigma_px, 0.5)


def _richardson_lucy_gaussian(image, sigma_px, num_iter):
    """Classic Richardson-Lucy deconvolution using a symmetric Gaussian PSF.

    A Gaussian blur is its own adjoint, so `scipy.ndimage.gaussian_filter` can
    stand in directly for both the forward and backward convolution steps of
    the RL update.
    """
    from scipy.ndimage import gaussian_filter

    image = np.clip(image, 0, None).astype(np.float32)
    estimate = image.copy() + 1e-6
    for _ in range(max(1, num_iter)):
        conv = gaussian_filter(estimate, sigma_px) + 1e-6
        relative_blur = image / conv
        estimate = estimate * gaussian_filter(relative_blur, sigma_px)
    return estimate


def _zs_self_supervised_denoise(frame, iterations, torch_device):
    """Train a tiny CNN to denoise a *single* frame with no external data.

    Implements a Neighbor2Neighbor-style (Huang et al., CVPR 2021)
    self-supervised objective: each 2x2 pixel block is split into two
    "neighbor-subsampled" half-resolution images g1, g2 that are statistically
    independent noisy realizations of the same underlying signal; the network
    is trained so that f(g1) ~= g2 (and vice versa isn't needed since we only
    need a converged denoiser), which requires no clean reference and no
    external training data -- consistent with ZS-DeconvNet's "trains on the
    single input" zero-shot framing.

    NOTE: this is a good-faith, simplified reimplementation of the *zero-shot,
    self-supervised* half of ZS-DeconvNet's published dual-stage approach, not
    a literal port of the vendored repo's training scripts -- see
    IMAGE_RESTORATION.md for why (the upstream scripts are shell-script/CLI
    driven research code whose exact interface could not be verified without
    executing it in this environment).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _TinyDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 3, padding=1),
            )

        def forward(self, x):
            return x + self.net(x)  # residual denoising

    def neighbor_subsample(x):
        _, _, h, w = x.shape
        h2, w2 = h // 2, w // 2
        x = x[:, :, :h2 * 2, :w2 * 2]
        blocks = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, C, h2, w2, 2, 2)
        g1 = blocks[..., 0, 0]
        g2 = blocks[..., 1, 1]
        return g1, g2

    x = torch.from_numpy(np.asarray(frame, dtype=np.float32))
    x = x.unsqueeze(0).unsqueeze(0)
    # Explicit >0 guard (a negative max is truthy, so `... or 1.0` would not
    # catch it and would invert the frame's sign).
    x_max = float(x.max().item())
    scale = x_max if x_max > 0 else 1.0
    x = (x / scale).to(torch_device)

    model = _TinyDenoiser().to(torch_device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(max(1, iterations)):
        g1, g2 = neighbor_subsample(x)
        loss = torch.mean((model(g1) - g2) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        denoised = model(x)
    return denoised.squeeze().detach().cpu().numpy() * scale


def restore_zs_deconvnet(stack, opts, device):
    """Zero-shot denoise + physics-consistent deconvolution, trained per frame.

    Dual-stage pipeline inspired by ZS-DeconvNet (Qiao et al., Nat. Commun.
    2024): (1) a tiny self-supervised network denoises each frame using only
    that frame (see `_zs_self_supervised_denoise`); (2) classical
    Richardson-Lucy deconvolution against an approximate Gaussian PSF (built
    from NA/wavelength/pixel size) enforces physics-consistency with the
    optical system, standing in for the published network's learned
    deconvolution stage. See IMAGE_RESTORATION.md for the honest scope of this
    simplification vs. the vendored upstream repo.

    'ZS upsampling' intentionally does NOT change the output's pixel
    dimensions (unlike the published super-resolution mode): NimbusImage
    annotations and pixel-scale calibration are defined against the original
    image grid, so changing resolution here would break coordinate alignment
    downstream. Instead it runs additional deconvolution refinement.
    """
    try:
        import torch
    except ImportError:
        sendError(
            "PyTorch is required for ZS-DeconvNet restoration but is not installed.",
            info="Rebuild the image with torch installed, or choose a different Method.",
        )
        return None

    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]

    iterations = max(1, int(opts.get('iterations', 300)))
    upsampling = bool(opts.get('upsampling', False))
    na = float(opts.get('NA', 0.75)) or 0.75
    wavelength_nm = float(opts.get('wavelength', 520)) or 520.0
    pixel_size_nm = float(opts.get('pixel_size_xy', 325)) or 325.0
    psf_sigma_px = _zs_gaussian_psf_sigma(na, wavelength_nm, pixel_size_nm)
    # 'upsampling' mode runs extra RL iterations instead of changing resolution;
    # see the docstring above and IMAGE_RESTORATION.md.
    rl_iterations = 20 if upsampling else 10

    torch_device = torch.device(device)

    results = []
    for frame in stack:
        denoised = _zs_self_supervised_denoise(frame, iterations, torch_device)
        deconvolved = _richardson_lucy_gaussian(denoised, psf_sigma_px, rl_iterations)
        results.append(deconvolved)

    return np.stack(results, axis=0)


def restore_fluoresfm(stack, opts, device):
    """Pretrained, text-prompted foundation-model restoration (experimental).

    Vendored from https://github.com/qiqi-lu/fluoresfm (pinned tag v1.0.1,
    cloned into /fluoresfm at build time). Weights are resolved at runtime
    from the FLUORESFM_WEIGHTS environment variable (see
    `resolve_fluoresfm_weights`) since they are distributed via Google
    Drive/Baidu Yun rather than a stable build-time-downloadable URL.

    Mark experimental in all user-facing docs: restored intensities may be
    hallucinated by the foundation model and should be validated before
    quantitative use (segment on the restored image, measure on the
    illumination-corrected raw image).
    """
    weights_path = resolve_fluoresfm_weights()
    if weights_path is None:
        return None  # resolve_fluoresfm_weights() already called sendError

    try:
        import torch
    except ImportError:
        sendError(
            "PyTorch is required for FluoResFM but is not installed.",
            info="Rebuild the image with torch installed, or choose a different Method.",
        )
        return None

    fluoresfm_repo = os.environ.get('FLUORESFM_REPO_PATH', '/fluoresfm')
    if fluoresfm_repo not in sys.path:
        sys.path.insert(0, fluoresfm_repo)

    try:
        # These import paths are our best-documented understanding of the
        # vendored repo's layout (methods/ for model + checkpoint loading,
        # packages/ for the BiomedCLIP text-embedding utility used to
        # condition the model on the text prompt); the research-code API may
        # shift between versions, hence the defensive try/except with an
        # actionable error rather than a hard dependency.
        from methods.fluoresfm import build_model, load_checkpoint
        from packages.text_embedding import embed_text
    except ImportError as e:
        sendError(
            "Could not load the vendored FluoResFM inference code.",
            info=(
                f"Import failed: {e}. Verify the FluoResFM repo is vendored at "
                f"'{fluoresfm_repo}' (see Dockerfile, cloned from "
                "https://github.com/qiqi-lu/fluoresfm) and that its module layout "
                "still matches this worker's integration "
                "(methods.fluoresfm.build_model/load_checkpoint, "
                "packages.text_embedding.embed_text). Update entrypoint.py's "
                "restore_fluoresfm() if the upstream API has changed."
            ),
        )
        return None

    task = opts.get('task', 'denoise')
    prompt = opts.get('prompt') or f"fluorescence microscopy image, {task}"

    torch_device = torch.device(device)
    model = build_model()
    load_checkpoint(model, weights_path, map_location=torch_device)
    model.to(torch_device)
    model.eval()

    text_embedding = embed_text(prompt, device=torch_device)

    stack = np.asarray(stack, dtype=np.float32)
    results = []
    with torch.no_grad():
        for frame in stack:
            # Explicit >0 guard, not `float(frame.max()) or 1.0`: the latter
            # leaves a negative max unchanged (it's truthy), which would flip
            # the sign of the whole normalized frame.
            frame_max = float(frame.max())
            scale = frame_max if frame_max > 0 else 1.0
            x = torch.from_numpy(frame / scale).unsqueeze(0).unsqueeze(0).to(torch_device)
            restored = model(x, text_embedding)
            restored = restored.squeeze().detach().cpu().numpy() * scale
            # Map the model's output range back onto the source frame's, so a
            # normalized foundation-model output isn't clipped to near-black.
            restored = _rescale_to_reference(restored, frame)
            results.append(restored)

    return np.stack(results, axis=0)


# Method name -> per-algorithm function name. Resolved via globals() *inside*
# compute() (not bound to function objects here at module-import time) so
# that tests can patch e.g. `entrypoint.restore_n2v` with a mock and have
# compute() pick up the patched version -- a dict built once at import time
# would instead keep permanent references to the original, unpatched
# functions.
_METHOD_FUNCTION_NAMES = {
    'n2v': 'restore_n2v',
    'cellpose3': 'restore_cellpose3',
    'zs_deconvnet': 'restore_zs_deconvnet',
    'fluoresfm': 'restore_fluoresfm',
}


def compute(datasetId, apiUrl, token, params):
    """
    params (could change):
        configurationId,
        datasetId,
        description: tool description,
        type: tool type,
        id: tool id,
        name: tool name,
        image: docker image,
        channel: annotation channel,
        assignment: annotation assignment ({XY, Z, Time}),
        tags: annotation tags (list of strings),
        tile: tile position (TODO: roi) ({XY, Z, Time}),
        connectTo: how new annotations should be connected
    """
    # Lazy import: keeps large_image off the interface/preview path; only needed during compute. See todo/worker-startup-latency.md
    import large_image as li

    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    workerInterface = params['workerInterface']

    method = workerInterface.get('Method', 'n2v')
    allChannels = workerInterface.get('Channels to restore', {})
    channels = [int(k) for k, v in allChannels.items() if v]
    print(f"Selected channels to restore: {channels}, method={method}")

    if len(channels) == 0:
        sendError("No channels selected for restoration",
                  info="Select at least one channel in 'Channels to restore'.")
        return

    if 'frames' not in tileClient.tiles or not tileClient.tiles.get('frames'):
        sendError("No frames found in dataset",
                  info="This worker requires a dataset with at least one frame.")
        return

    frames = tileClient.tiles['frames']

    restore_fn_name = _METHOD_FUNCTION_NAMES.get(method)
    if restore_fn_name is None:
        sendError(f"Unknown restoration method: {method}")
        return
    restore_fn = globals()[restore_fn_name]

    use_gpu_requested = bool(workerInterface.get('Use GPU', True))
    device = resolve_device(use_gpu_requested)
    print(f"Using device: {device}")

    source_dtype = tileClient.tiles.get('dtype', None)
    method_opts = _build_method_opts(method, workerInterface)

    gc = tileClient.client

    # Group frame indices by channel: retrospective/collection-based methods
    # (n2v trains on the whole channel collection; zs_deconvnet trains
    # per-frame but is naturally called once per channel's stack) see the
    # full (N, Y, X) stack; per-frame methods (cellpose3, fluoresfm) simply
    # loop internally over that same stack.
    channel_frame_indices = defaultdict(list)
    for i, frame in enumerate(frames):
        c = frame.get('IndexC', 0)
        if c in channels:
            channel_frame_indices[c].append(i)

    restored_images = {}
    total_channels = len(channels)
    for idx, c in enumerate(channels):
        frame_indices = channel_frame_indices.get(c, [])
        if not frame_indices:
            continue

        sendProgress(idx / max(total_channels, 1), 'Restoring',
                    f"Channel {c} ({method}), {len(frame_indices)} frame(s)")

        try:
            images = [tileClient.getRegion(datasetId, frame=i).squeeze() for i in frame_indices]
            # np.stack is inside the try so heterogeneous per-frame shapes
            # (which raise ValueError) surface as a structured error, not a
            # raw traceback -- like every other failure path in this worker.
            stack = np.stack(images, axis=0)
            result_stack = restore_fn(stack, method_opts, device)
        except Exception as e:
            sendError(f"Restoration failed for channel {c} using method '{method}'",
                      info=str(e))
            return

        if result_stack is None:
            # The restore_* function already called sendError with an
            # actionable message (missing dependency / missing weights);
            # abort cleanly without crashing.
            return

        result_stack = _clip_to_dtype(np.asarray(result_stack), source_dtype)

        # A restorer must return exactly one output frame per input frame;
        # otherwise zip() below would silently drop/misalign frames (e.g. if
        # CAREamics predict() ever returns a different sample count), leaving
        # some frames un-restored in the output with no error.
        if len(result_stack) != len(frame_indices):
            sendError(f"Restoration returned {len(result_stack)} frame(s) but "
                      f"{len(frame_indices)} were submitted for channel {c} (method '{method}')",
                      info="This is an internal mismatch; please report it.")
            return

        for frame_idx, restored_image in zip(frame_indices, result_stack):
            restored_images[frame_idx] = restored_image

    sendProgress(0.9, 'Assembling output', 'Writing frames')
    sink = li.new()
    for i, frame in enumerate(frames):
        # Create a parameters dictionary with only the indices that exist in frame
        # The len(k) > 5 is to avoid the 'Index' key that has no postfix to it
        large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items()
                              if k.startswith('Index') and len(k) > 5}

        if i in restored_images:
            image = restored_images[i]
        else:
            image = tileClient.getRegion(datasetId, frame=i).squeeze()

        sink.addTile(image, 0, 0, **large_image_params)

    # Copy over the metadata
    if 'channels' in tileClient.tiles:
        sink.channelNames = tileClient.tiles['channels']
    sink.mm_x = tileClient.tiles['mm_x']
    sink.mm_y = tileClient.tiles['mm_y']
    sink.magnification = tileClient.tiles['magnification']

    sendProgress(0.95, 'Writing output', 'Saving TIFF file')
    sink.write('/tmp/restored.tiff')
    print("Wrote to file")

    sendProgress(0.98, 'Uploading', 'Uploading to server')
    item = gc.uploadFileToFolder(datasetId, '/tmp/restored.tiff')
    gc.addMetadataToItem(item['itemId'], {
        'tool': 'Image Restoration',
        'method': method,
        'channels': channels,
        'gpu_requested': use_gpu_requested,
        'device_used': device,
        **method_opts,
    })
    print("Uploaded file")

    sendProgress(1.0, 'Complete', 'Image restoration finished')


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Restore images via Noise2Void/N2V2, Cellpose3, ZS-DeconvNet, or FluoResFM')

    parser.add_argument('--datasetId', type=str,
                        required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    params = json.loads(args.parameters)
    datasetId = args.datasetId
    apiUrl = args.apiUrl
    token = args.token

    match args.request:
        case 'compute':
            compute(datasetId, apiUrl, token, params)
        case 'interface':
            interface(params['image'], apiUrl, token)
