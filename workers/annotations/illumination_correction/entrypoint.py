import argparse
import json
import os
import sys

import numpy as np

import annotation_client.tiles as tiles
import annotation_client.workers as workers

from annotation_client.utils import sendProgress, sendWarning, sendError


# Floor applied to every mean-normalized gain/illumination surface (which is
# normalized so its mean is 1.0) before dividing. Real CIDRE/CellProfiler solve
# a regularized (well-posed) model that can't collapse to near-zero gain; our
# lightweight per-pixel-statistics-plus-smoothing stand-in has no such
# guarantee, so without a floor a handful of near-zero pixels (e.g. heavy
# vignetting at the extreme corners, or noise-dominated low-signal regions)
# can blow up to enormous corrected values. Capping the gain at 5% of its own
# mean keeps the correction bounded (<=20x local amplification) while barely
# affecting well-behaved regions.
_MIN_GAIN_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    interface = {
        'Method': {
            'type': 'select',
            'items': ['basic', 'cidre', 'cellprofiler', 'flatfield', 'destripe', 'sscor'],
            'default': 'basic',
            'tooltip': 'Illumination-correction algorithm. basic (BaSiC) recommended when no '
                       'calibration frames are available. sscor is a deep-learning stripe '
                       'self-correction method that requires a user-supplied trained checkpoint '
                       '(see SSCOR_WEIGHTS). Every parameter below is shown regardless of Method; '
                       'each tooltip notes which method(s) it applies to -- unused parameters for '
                       'the chosen method are simply ignored.',
            'displayOrder': 0,
        },
        'Channels to correct': {
            'type': 'channelCheckboxes',
            'tooltip': 'Process selected channels; unselected channels pass through unchanged.',
            'displayOrder': 1,
        },
        # --- BaSiC ---
        'Estimate darkfield': {
            'type': 'checkbox',
            'default': True,
            'tooltip': 'basic: also estimate a darkfield (offset) term.',
            'displayOrder': 2,
        },
        'Flatfield smoothness': {
            'type': 'number',
            'min': 0,
            'max': 100,
            'default': 1.0,
            'tooltip': 'basic: smoothness regularization of the flatfield.',
            'displayOrder': 3,
        },
        'Darkfield smoothness': {
            'type': 'number',
            'min': 0,
            'max': 100,
            'default': 1.0,
            'tooltip': 'basic: smoothness regularization of the darkfield.',
            'displayOrder': 4,
        },
        'Correct timelapse baseline drift': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'basic: estimate and remove temporal background/bleaching drift across Time.',
            'displayOrder': 5,
        },
        # --- CIDRE / CellProfiler shared ---
        'Smoothing sigma': {
            'type': 'number',
            'min': 1,
            'max': 2000,
            'default': 50,
            'tooltip': 'cidre/cellprofiler: Gaussian smoothing sigma (px) for the illumination surface.',
            'displayOrder': 6,
        },
        'Dark quantile': {
            'type': 'number',
            'min': 0,
            'max': 0.5,
            'default': 0.02,
            'tooltip': 'cidre: percentile for the dark/offset surface estimate.',
            'displayOrder': 7,
        },
        'CellProfiler mode': {
            'type': 'select',
            'items': ['regular', 'background'],
            'default': 'regular',
            'tooltip': 'cellprofiler: regular=across-batch average; background=per-image background.',
            'displayOrder': 8,
        },
        # --- flatfield reference ---
        'Flat-field XY coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1',
                'label': 'Flat-field XY coordinate',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'tooltip': 'flatfield: 1-indexed XY position of the flat-field reference image. '
                       'Required for the flatfield method.',
            'displayOrder': 9,
        },
        'Dark-field XY coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1',
                'label': 'Dark-field XY coordinate',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'tooltip': 'flatfield: 1-indexed XY position of the dark-field reference '
                       '(empty = use the Dark-field constant instead).',
            'displayOrder': 10,
        },
        'Dark-field constant': {
            'type': 'number',
            'min': 0,
            'max': 65535,
            'default': 0,
            'tooltip': 'flatfield: constant dark offset used if no dark-field reference frame is given.',
            'displayOrder': 11,
        },
        # --- destripe ---
        'Destripe sigma': {
            'type': 'number',
            'min': 1,
            'max': 2000,
            'default': 128,
            'tooltip': 'destripe: band-pass sigma for wavelet-FFT stripe removal.',
            'displayOrder': 12,
        },
        'Destripe wavelet': {
            'type': 'select',
            'items': ['db3', 'db5', 'haar', 'sym4'],
            'default': 'db3',
            'tooltip': 'destripe: wavelet used for the wavelet-FFT decomposition.',
            'displayOrder': 13,
        },
        'Destripe level': {
            'type': 'number',
            'min': 0,
            'max': 12,
            'default': 0,
            'tooltip': 'destripe: DWT decomposition level (0 = auto).',
            'displayOrder': 14,
        },
        # --- SSCOR ---
        'SSCOR mode': {
            'type': 'select',
            'items': ['pretrained', 'self-train'],
            'default': 'pretrained',
            'tooltip': 'sscor: how the generator checkpoint is obtained. pretrained = use a '
                       'checkpoint trained offline and supplied via the SSCOR_WEIGHTS '
                       'environment variable (fast, but you must already have a trained '
                       'checkpoint). self-train = no checkpoint needed -- SSCOR samples '
                       'stripe-free "clean" patches and striped patches from the image itself '
                       'and trains a small CycleGAN on them before restoring, faithfully '
                       'reproducing the upstream per-image self-training method. self-train '
                       'trains a fresh model PER FRAME on a GPU and is slow.',
            'displayOrder': 15,
        },
        'SSCOR stripe direction': {
            'type': 'select',
            'items': ['horizontal', 'vertical', 'grid'],
            'default': 'horizontal',
            'tooltip': 'sscor (self-train): stripe orientation to sample training patches for -- '
                       'horizontal, vertical, or grid (both directions, using the upstream '
                       'sample_stripe_2.py corner/junction sampling). Trains a fresh model PER '
                       'FRAME on a GPU and is slow.',
            'displayOrder': 16,
        },
        'SSCOR horizontal stripe count': {
            'type': 'number',
            'min': 1,
            'max': 4096,
            'default': 1,
            'tooltip': 'sscor (self-train): number of horizontal stripe bands (--h_n) used when '
                       'sampling training patches (horizontal or grid stripe direction). Trains '
                       'a fresh model PER FRAME on a GPU and is slow.',
            'displayOrder': 17,
        },
        'SSCOR vertical stripe count': {
            'type': 'number',
            'min': 1,
            'max': 4096,
            'default': 1,
            'tooltip': 'sscor (self-train): number of vertical stripe bands (--v_n) used when '
                       'sampling training patches (vertical or grid stripe direction). Trains a '
                       'fresh model PER FRAME on a GPU and is slow.',
            'displayOrder': 18,
        },
        'SSCOR grid direction': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0,
            'tooltip': 'sscor (self-train): junction direction (0=Upper Left, 1=Upper Right, '
                       '2=Lower Left, 3=Lower Right) passed as sample_stripe_2.py\'s --direction; '
                       'only used when SSCOR stripe direction=grid. Trains a fresh model PER '
                       'FRAME on a GPU and is slow.',
            'displayOrder': 19,
        },
        'SSCOR training epochs': {
            'type': 'number',
            'min': 1,
            'max': 300,
            'default': 30,
            'tooltip': 'sscor (self-train): number of training epochs for the per-frame CycleGAN '
                       'self-training pass. Trains a fresh model PER FRAME on a GPU and is slow.',
            'displayOrder': 20,
        },
        'SSCOR patch size': {
            'type': 'number',
            'min': 1,
            'max': 4096,
            'default': 256,
            'tooltip': 'sscor: sliding-window patch size (px) fed to the generator.',
            'displayOrder': 21,
        },
        'SSCOR offset size': {
            'type': 'number',
            'min': 1,
            'max': 4096,
            'default': 100,
            'tooltip': 'sscor: sliding-window step/offset size (px) between patches.',
            'displayOrder': 22,
        },
        'SSCOR repeat': {
            'type': 'number',
            'min': 1,
            'max': 5,
            'default': 1,
            'tooltip': 'sscor: number of repeated passes (with shifted offsets) to combine via '
                       'max-projection, per the upstream restore.py.',
            'displayOrder': 23,
        },
        'SSCOR dark threshold': {
            'type': 'number',
            'min': 0,
            'max': 255,
            'default': 10,
            'tooltip': 'sscor: 8-bit intensity threshold below which the original (uncorrected) '
                       'pixel is kept instead of the restored value.',
            'displayOrder': 24,
        },
        # --- QC ---
        'Report correction quality (QC)': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'Compute lightweight EVEN-style flat-field-quality metrics per corrected '
                       'channel and store them in the output item metadata. Not the full EVEN '
                       'framework -- a quick quantitative stand-in for comparing methods.',
            'displayOrder': 25,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


# ---------------------------------------------------------------------------
# Per-algorithm functions
#
# Each function has the signature `f(stack, opts) -> (corrected_stack, diagnostics)`
# where `stack` is a (N, Y, X) float array containing every frame of ONE channel
# (gathered across XY/Z/Time for the retrospective methods; a simple per-frame
# stack for flatfield/destripe), and `diagnostics` is a small JSON-serializable
# dict recorded for provenance.
#
# Heavy third-party imports (basicpy, pystripe) are done LAZILY inside these
# functions so that `interface()` stays fast and so tests can run natively
# without those packages installed (the tests patch these functions directly).
# ---------------------------------------------------------------------------

def correct_basic(stack, opts):
    """BaSiC retrospective shading (+ darkfield) correction (Peng et al. 2017;
    BaSiCPy 2.x, PyTorch backend).

    stack: (N, Y, X) array -- every frame of one channel's collection.
    opts: dict with 'estimate_darkfield' (bool), 'flatfield_smoothness' (float),
          'darkfield_smoothness' (float), 'baseline_drift' (bool), and optionally
          'time_order' (list[int] aligned with stack) used only to sort frames
          chronologically before fitting when baseline_drift is requested.
    """
    from basicpy import BaSiC

    stack = np.asarray(stack, dtype=np.float64)

    order = None
    if opts.get('baseline_drift') and opts.get('time_order') is not None:
        order = np.argsort(np.asarray(opts['time_order']))
        fit_stack = stack[order]
    else:
        fit_stack = stack

    basic = BaSiC(
        get_darkfield=bool(opts.get('estimate_darkfield', True)),
        smoothness_flatfield=float(opts.get('flatfield_smoothness', 1.0)),
        smoothness_darkfield=float(opts.get('darkfield_smoothness', 1.0)),
        fitting_mode='approximate',
    )
    basic.fit(fit_stack)
    corrected_fit_order = np.asarray(basic.transform(fit_stack))

    if order is not None:
        corrected = np.empty_like(corrected_fit_order)
        corrected[order] = corrected_fit_order
    else:
        corrected = corrected_fit_order

    diagnostics = {
        'flatfield_mean': float(np.mean(basic.flatfield)),
        'darkfield_mean': float(np.mean(basic.darkfield)) if opts.get('estimate_darkfield', True) and getattr(basic, 'darkfield', None) is not None else 0.0,
    }
    baseline = getattr(basic, 'baseline', None)
    if opts.get('baseline_drift') and baseline is not None:
        diagnostics['baseline'] = [float(b) for b in np.asarray(baseline).ravel()]

    return corrected, diagnostics


def correct_cidre(stack, opts):
    """CIDRE-style retrospective illumination correction (Smith et al. 2015,
    Nature Methods).

    NOTE (honest simplification): this is a lightweight, in-house
    reimplementation of CIDRE's retrospective gain/offset model, NOT the
    original MATLAB implementation and its full energy-minimization solve.
    The offset (dark) and gain (flat) surfaces here are estimated with robust
    per-pixel statistics and then heavily Gaussian-smoothed as a spatial
    regularization stand-in.
    """
    from scipy.ndimage import gaussian_filter

    stack = np.asarray(stack, dtype=np.float64)
    sigma = float(opts.get('smoothing_sigma', 50))
    q = float(opts.get('dark_quantile', 0.02))

    # Offset (dark) surface: robust low-quantile per pixel across the collection.
    z = np.quantile(stack, q, axis=0)
    z = gaussian_filter(z, sigma=sigma)

    # Gain (flat) surface: per-pixel median of (frame - offset), normalized to mean 1.
    residual = stack - z[np.newaxis, :, :]
    v = np.median(residual, axis=0)
    v = gaussian_filter(v, sigma=sigma)
    v_mean = np.mean(v)
    if v_mean <= 0:
        v_mean = 1.0
    v = v / v_mean
    v = np.clip(v, _MIN_GAIN_FRACTION, None)

    corrected = (stack - z[np.newaxis, :, :]) / v[np.newaxis, :, :]

    diagnostics = {
        'offset_mean': float(np.mean(z)),
        'gain_mean': float(np.mean(v)),
    }
    return corrected, diagnostics


def correct_cellprofiler(stack, opts):
    """CellProfiler-style illumination function, reimplemented without a
    CellProfiler dependency (core logic of `CorrectIlluminationCalculate`).

    mode='regular' (default): one illumination function is fit from the
        across-batch mean image and applied to every frame.
    mode='background': each frame supplies its own heavily-smoothed background
        estimate, applied per frame independently.
    """
    from scipy.ndimage import gaussian_filter

    stack = np.asarray(stack, dtype=np.float64)
    sigma = float(opts.get('smoothing_sigma', 50))
    mode = opts.get('cellprofiler_mode', 'regular')

    if mode == 'background':
        corrected = np.empty_like(stack)
        illum_means = []
        for i in range(stack.shape[0]):
            illum = gaussian_filter(stack[i], sigma=sigma)
            illum_mean = np.mean(illum)
            if illum_mean <= 0:
                illum_mean = 1.0
            illum = np.clip(illum / illum_mean, _MIN_GAIN_FRACTION, None)
            corrected[i] = stack[i] / illum
            illum_means.append(float(illum_mean))
        diagnostics = {'mode': 'background', 'illumination_mean': illum_means}
    else:
        avg_image = np.mean(stack, axis=0)
        illum = gaussian_filter(avg_image, sigma=sigma)
        illum_mean = np.mean(illum)
        if illum_mean <= 0:
            illum_mean = 1.0
        illum = np.clip(illum / illum_mean, _MIN_GAIN_FRACTION, None)
        corrected = stack / illum[np.newaxis, :, :]
        diagnostics = {'mode': 'regular', 'illumination_mean': float(illum_mean)}

    return corrected, diagnostics


def correct_flatfield(stack, opts):
    """Classical flat-field / dark-field reference-based correction:
    corrected = (raw - dark) / (flat - dark), rescaled to preserve mean intensity.

    opts must include 'flat' (2D array) and 'dark' (2D array, same shape).
    """
    stack = np.asarray(stack, dtype=np.float64)
    flat = np.asarray(opts['flat'], dtype=np.float64)
    dark = np.asarray(opts['dark'], dtype=np.float64)

    flat_minus_dark = flat - dark
    flat_mean = np.mean(flat_minus_dark)
    if flat_mean <= 0:
        flat_mean = 1.0
    # Normalize gain so its mean is 1 after dark subtraction, to preserve the
    # overall intensity scale. Floor at _MIN_GAIN_FRACTION to avoid blow-up from
    # near-zero gain at extreme vignetting (see _MIN_GAIN_FRACTION docstring).
    gain = np.clip(flat_minus_dark / flat_mean, _MIN_GAIN_FRACTION, None)

    corrected = (stack - dark[np.newaxis, :, :]) / gain[np.newaxis, :, :]
    corrected = np.clip(corrected, 0, None)

    diagnostics = {
        'flat_mean': float(np.mean(flat)),
        'dark_mean': float(np.mean(dark)),
    }
    return corrected, diagnostics


def correct_destripe(stack, opts):
    """Classical wavelet-FFT destriping via pystripe, applied independently per
    frame. Documented stand-in for the DL method SSCOR, which has no packaged
    weights available.
    """
    from pystripe import core as pystripe_core

    stack = np.asarray(stack, dtype=np.float64)
    sigma = float(opts.get('destripe_sigma', 128))
    wavelet = opts.get('destripe_wavelet', 'db3')
    level = int(opts.get('destripe_level', 0))
    level_arg = None if level == 0 else level

    corrected = np.empty_like(stack)
    for i in range(stack.shape[0]):
        corrected[i] = pystripe_core.filter_streaks(
            stack[i], sigma=[sigma, sigma], level=level_arg, wavelet=wavelet)

    diagnostics = {'sigma': sigma, 'wavelet': wavelet, 'level': level}
    return corrected, diagnostics


def resolve_sscor_checkpoint():
    """Resolve the path to the SSCOR generator checkpoint (pretrained mode only).

    SSCOR weights are not bundled in the Docker image -- the upstream repo
    (https://github.com/lxxcontinue/SSCOR) distributes trained models via
    Google Drive, not a stable direct-download URL, and producing one requires
    an image-specific self-training stage (proximity sampling + adversarial
    training) that must either be run offline per the upstream README, or via
    this worker's own `SSCOR mode=self-train` (see `correct_sscor`). The
    runtime source of truth for the *pretrained* mode is therefore the
    SSCOR_WEIGHTS environment variable, which should point to a mounted
    generator checkpoint (the CycleGAN's `G_A` weights; at inference time
    SSCOR loads it as `latest_net_G_A.pth`, regardless of the file's original
    name -- `correct_sscor` stages it under that name). Returns the resolved
    path, or None (after sending an actionable sendError) if unavailable.
    """
    weights_path = os.environ.get('SSCOR_WEIGHTS', '').strip()
    if not weights_path or not os.path.isfile(weights_path):
        sendError(
            'SSCOR weights are not available.',
            info=(
                'SSCOR weights are not bundled in this image -- they are distributed via '
                'Google Drive from https://github.com/lxxcontinue/SSCOR. Either download a '
                'trained generator checkpoint, mount it into the container, and set the '
                'SSCOR_WEIGHTS environment variable to its path (the CycleGAN G_A generator '
                'weights, staged internally as `latest_net_G_A.pth`), or set "SSCOR mode" to '
                '"self-train" to train a model from the image itself with no checkpoint, or '
                'choose a different Method (e.g. destripe, which needs no weights).'
            ),
        )
        return None
    return weights_path


def _sscor_frame_to_uint8(frame):
    """Rescale one (Y, X) float frame to uint8 [0, 255] via per-frame min/max.
    Returns (frame_min, span, frame_uint8); `span` is guaranteed > 0, so the
    inverse rescale `restored_gray / 255.0 * span + frame_min` always applies.
    """
    frame_min = float(np.min(frame))
    frame_max = float(np.max(frame))
    span = frame_max - frame_min
    if span <= 0:
        span = 1.0
    frame_uint8 = np.clip((frame - frame_min) / span * 255.0, 0, 255).astype(np.uint8)
    return frame_min, span, frame_uint8


def correct_sscor(stack, opts):
    """SSCOR deep-learning stripe self-correction
    (https://github.com/lxxcontinue/SSCOR, pinned commit
    985479cd79bcf1359e3d9ba44bacd5f372eb2e60).

    SSCOR is a pytorch-CycleGAN-and-pix2pix-style codebase with no clean
    importable inference API, so this integration shells out to its CLI
    scripts exactly like `deconwolf` shells out to the `dw` binary. Two modes
    are supported, selected by `opts['sscor_mode']`:

    - 'pretrained': runs ONLY the repo's inference stage (`restore.py`),
      using a user-supplied trained generator checkpoint resolved by
      `resolve_sscor_checkpoint` (SSCOR_WEIGHTS). No training happens here.
    - 'self-train': the faithful upstream SSCOR pipeline, run once PER FRAME
      with no pre-supplied checkpoint. For each frame: (1) `sample/sample_stripe.py`
      or `sample/sample_stripe_2.py` samples stripe-free "clean" patches
      (trainB) and striped patches (trainA) from the image itself, using the
      chosen stripe direction/count; (2) `train.py` trains a small CycleGAN
      on those self-sampled patches for `sscor_epochs` epochs, saving
      `latest_net_G_A.pth` once at the end (`--save_epoch_freq` == epoch
      count); (3) `restore.py` restores the frame with that just-trained
      generator. This is slow (a fresh model is trained from scratch per
      frame) and a GPU is strongly recommended.

    SSCOR's generator only supports 8-bit RGB I/O (PIL-loaded input,
    `tensor2im`-produced uint8 output), so each frame is rescaled to uint8
    [0, 255] via per-frame min/max before being handed to the SSCOR scripts,
    and the 8-bit result is rescaled back to the frame's original [min, max]
    range afterward. This round trip is LOSSY (8-bit quantization) and is
    inherent to SSCOR's design, not an artifact of this integration.

    stack: (N, Y, X) array -- every frame of one channel's collection.
    opts: dict with 'sscor_mode' ('pretrained' or 'self-train'),
          'sscor_patch_size', 'sscor_offset_size', 'sscor_repeat',
          'sscor_dark_threshold' (all int), 'sscor_gpu_ids' ('-1' for CPU,
          '0' for GPU); for 'pretrained': 'sscor_weights' (path to a
          generator checkpoint, staged as `latest_net_G_A.pth`); for
          'self-train': 'sscor_stripe_direction' ('horizontal'/'vertical'/
          'grid'), 'sscor_h_n', 'sscor_v_n', 'sscor_grid_direction',
          'sscor_epochs' (all int). All injected by `compute()`.
    """
    import shutil
    import subprocess
    import tempfile

    from PIL import Image

    stack = np.asarray(stack, dtype=np.float64)

    mode = opts.get('sscor_mode', 'pretrained')
    gpu_ids = opts.get('sscor_gpu_ids', '-1')
    patch_size = int(opts.get('sscor_patch_size', 256))
    offset_size = int(opts.get('sscor_offset_size', 100))
    repeat = int(opts.get('sscor_repeat', 1))
    dark_threshold = int(opts.get('sscor_dark_threshold', 10))

    repo_path = os.environ.get('SSCOR_REPO_PATH', '/sscor')
    restore_script = os.path.join(repo_path, 'restore.py')

    corrected = np.empty_like(stack)

    if mode == 'self-train':
        stripe_direction = opts.get('sscor_stripe_direction', 'horizontal')
        h_n = int(opts.get('sscor_h_n', 1))
        v_n = int(opts.get('sscor_v_n', 1))
        grid_direction = int(opts.get('sscor_grid_direction', 0))
        epochs = int(opts.get('sscor_epochs', 30))
        exp_name = 'sscor_selftrain'

        sample_stripe_script = os.path.join(repo_path, 'sample', 'sample_stripe.py')
        sample_stripe_2_script = os.path.join(repo_path, 'sample', 'sample_stripe_2.py')
        train_script = os.path.join(repo_path, 'train.py')

        for i in range(stack.shape[0]):
            frame = stack[i]
            frame_min, span, frame_uint8 = _sscor_frame_to_uint8(frame)

            # Fresh temp dirs per frame so self-trained checkpoints/samples never
            # leak into the next frame's (independent) self-training run.
            with tempfile.TemporaryDirectory() as tmpin, \
                    tempfile.TemporaryDirectory() as tmpout, \
                    tempfile.TemporaryDirectory() as tmpckpt:

                image_name = f'frame_{i:04d}.png'
                input_path = os.path.join(tmpin, image_name)
                Image.fromarray(frame_uint8).convert('RGB').save(input_path)
                stem = os.path.splitext(image_name)[0]

                # 1. Sample stripe (trainA) / stripe-free (trainB) patches from the
                # frame itself, per the upstream self-training method.
                if stripe_direction == 'grid':
                    sample_cmd = [
                        sys.executable, sample_stripe_2_script,
                        '--h_n', str(h_n),
                        '--v_n', str(v_n),
                        '--direction', str(grid_direction),
                        '--in_dir', tmpin,
                        '--img_name', image_name,
                        '--out_dir', tmpout,
                        '--patch_size', str(patch_size),
                    ]
                elif stripe_direction == 'vertical':
                    sample_cmd = [
                        sys.executable, sample_stripe_script,
                        '--v',
                        '--v_n', str(v_n),
                        '--in_dir', tmpin,
                        '--img_name', image_name,
                        '--out_dir', tmpout,
                        '--patch_size', str(patch_size),
                    ]
                else:  # 'horizontal' (default)
                    sample_cmd = [
                        sys.executable, sample_stripe_script,
                        '--h',
                        '--h_n', str(h_n),
                        '--in_dir', tmpin,
                        '--img_name', image_name,
                        '--out_dir', tmpout,
                        '--patch_size', str(patch_size),
                    ]

                result = subprocess.run(sample_cmd, capture_output=True, text=True, cwd=repo_path)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"SSCOR self-train sampling failed on frame {i}: {result.stderr}")

                sample_dir = os.path.join(tmpout, f'sample_{stem}')
                train_a_dir = os.path.join(sample_dir, 'trainA')
                if not os.path.isdir(train_a_dir) or len(os.listdir(train_a_dir)) == 0:
                    raise RuntimeError(
                        f"SSCOR self-train sampling produced no training patches for frame {i} "
                        f"(patch_size={patch_size}, stripe_direction={stripe_direction}). The "
                        f"frame may be too small for this patch size, or the stripe direction/"
                        f"count doesn't fit the image -- try a smaller 'SSCOR patch size' or a "
                        f"different 'SSCOR stripe direction'/count.")

                # 2. Train a fresh per-frame CycleGAN on the self-sampled patches.
                # save_epoch_freq == n_epochs guarantees model.save_networks('latest')
                # (-> latest_net_G_A.pth) is written exactly once, at the final epoch.
                train_cmd = [
                    sys.executable, train_script,
                    '--dataroot', sample_dir,
                    '--name', exp_name,
                    '--model', 'sscor',
                    '--checkpoints_dir', tmpckpt,
                    '--gpu_ids', gpu_ids,
                    '--display_id', '0',
                    '--no_html',
                    '--load_size', str(patch_size + 30),
                    '--crop_size', str(patch_size),
                    '--n_epochs', str(epochs),
                    '--n_epochs_decay', '0',
                    '--save_epoch_freq', str(epochs),
                ]
                result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=repo_path)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"SSCOR self-train train.py failed on frame {i}: {result.stderr}")

                # 3. Restore the frame with the just-trained generator.
                restore_cmd = [
                    sys.executable, restore_script,
                    '--dataroot', tmpin,
                    '--name', exp_name,
                    '--model', 'sscor',
                    '--image_name', image_name,
                    '--offset_size', str(offset_size),
                    '--patch_size', str(patch_size),
                    '--repeat', str(repeat),
                    '--dark_threshold', str(dark_threshold),
                    '--checkpoints_dir', tmpckpt,
                    '--gpu_ids', gpu_ids,
                    '--eval',
                ]
                result = subprocess.run(restore_cmd, capture_output=True, text=True, cwd=repo_path)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"SSCOR self-train restore.py failed on frame {i}: {result.stderr}")

                output_path = os.path.join(tmpin, 'result', f'restore-{image_name}')
                if not os.path.exists(output_path):
                    raise RuntimeError(
                        f"SSCOR restore.py did not produce the expected output '{output_path}' "
                        f"for frame {i}. stdout: {result.stdout}")

                restored_rgb = np.asarray(Image.open(output_path).convert('RGB'), dtype=np.float64)
                restored_gray = np.mean(restored_rgb, axis=-1)
                corrected[i] = restored_gray / 255.0 * span + frame_min

        diagnostics = {
            'mode': 'self-train',
            'patch_size': patch_size,
            'offset_size': offset_size,
            'repeat': repeat,
            'dark_threshold': dark_threshold,
            'gpu_ids': gpu_ids,
            'stripe_direction': stripe_direction,
            'h_n': h_n,
            'v_n': v_n,
            'grid_direction': grid_direction,
            'epochs': epochs,
        }
        return corrected, diagnostics

    # mode == 'pretrained'
    weights_path = opts['sscor_weights']

    with tempfile.TemporaryDirectory() as tmpckpt, tempfile.TemporaryDirectory() as tmpdata:
        # restore.py loads <checkpoints_dir>/<name>/<epoch>_net_<model_name>.pth (epoch
        # defaults to 'latest'); at test time SSCORModel.model_names == ['G_A'], so the
        # file MUST be named `latest_net_G_A.pth`, not `latest_net_G.pth`.
        ckpt_dir = os.path.join(tmpckpt, 'sscor')
        os.makedirs(ckpt_dir, exist_ok=True)
        shutil.copy(weights_path, os.path.join(ckpt_dir, 'latest_net_G_A.pth'))

        for i in range(stack.shape[0]):
            frame = stack[i]
            frame_min, span, frame_uint8 = _sscor_frame_to_uint8(frame)

            image_name = f'frame_{i:04d}.png'
            input_path = os.path.join(tmpdata, image_name)
            Image.fromarray(frame_uint8).convert('RGB').save(input_path)

            cmd = [
                sys.executable, restore_script,
                '--dataroot', tmpdata,
                '--name', 'sscor',
                '--model', 'sscor',
                '--image_name', image_name,
                '--offset_size', str(offset_size),
                '--patch_size', str(patch_size),
                '--repeat', str(repeat),
                '--dark_threshold', str(dark_threshold),
                '--checkpoints_dir', tmpckpt,
                '--gpu_ids', gpu_ids,
                '--eval',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)
            if result.returncode != 0:
                raise RuntimeError(f"SSCOR restore.py failed on frame {i}: {result.stderr}")

            output_path = os.path.join(tmpdata, 'result', f'restore-{image_name}')
            if not os.path.exists(output_path):
                raise RuntimeError(
                    f"SSCOR restore.py did not produce the expected output '{output_path}' "
                    f"for frame {i}. stdout: {result.stdout}")

            restored_rgb = np.asarray(Image.open(output_path).convert('RGB'), dtype=np.float64)
            restored_gray = np.mean(restored_rgb, axis=-1)

            corrected[i] = restored_gray / 255.0 * span + frame_min

    diagnostics = {
        'mode': 'pretrained',
        'patch_size': patch_size,
        'offset_size': offset_size,
        'repeat': repeat,
        'dark_threshold': dark_threshold,
        'gpu_ids': gpu_ids,
        'checkpoint': 'SSCOR_WEIGHTS',
    }
    return corrected, diagnostics


def compute_qc_metrics(corrected_stack, opts=None):
    """Lightweight EVEN-style QC stand-in (NOT the full EVEN ML evaluation and
    optimization framework, Nat. Commun. 2026, which is not vendored here).

    Computes, on the corrected per-channel collection:
      - cv_mean_image: coefficient of variation of a smoothed mean image
        (lower = flatter/more uniform illumination).
      - corner_center_ratio: residual vignetting, mean corner vs mean center
        intensity of the mean image (closer to 1.0 = less vignetting).
      - interframe_cv: coefficient of variation of per-frame mean intensity
        across the collection.
    """
    from scipy.ndimage import gaussian_filter

    stack = np.asarray(corrected_stack, dtype=np.float64)
    mean_image = np.mean(stack, axis=0)
    sigma = max(1.0, min(mean_image.shape[:2]) / 20.0)
    smoothed = gaussian_filter(mean_image, sigma=sigma)

    mean_val = np.mean(smoothed)
    std_val = np.std(smoothed)
    cv_mean_image = float(std_val / mean_val) if mean_val != 0 else 0.0

    h, w = mean_image.shape[0], mean_image.shape[1]
    ch, cw = max(1, h // 8), max(1, w // 8)
    center = mean_image[h // 2 - ch // 2: h // 2 + ch // 2,
                         w // 2 - cw // 2: w // 2 + cw // 2]
    corners = [
        mean_image[:ch, :cw], mean_image[:ch, -cw:],
        mean_image[-ch:, :cw], mean_image[-ch:, -cw:],
    ]
    corner_mean = float(np.mean([np.mean(c) for c in corners]))
    center_mean = float(np.mean(center)) if center.size else 0.0
    corner_center_ratio = float(corner_mean / center_mean) if center_mean != 0 else 0.0

    frame_means = np.mean(stack, axis=tuple(range(1, stack.ndim)))
    frame_mean_of_means = np.mean(frame_means)
    interframe_cv = float(np.std(frame_means) / frame_mean_of_means) if frame_mean_of_means != 0 else 0.0

    return {
        'cv_mean_image': cv_mean_image,
        'corner_center_ratio': corner_center_ratio,
        'interframe_cv': interframe_cv,
    }


def _method_params_for_metadata(method, opts, flat_xy, dark_xy):
    """Return the subset of `opts` relevant to `method`, for provenance metadata."""
    if method == 'basic':
        return {
            'estimate_darkfield': opts['estimate_darkfield'],
            'flatfield_smoothness': opts['flatfield_smoothness'],
            'darkfield_smoothness': opts['darkfield_smoothness'],
            'baseline_drift': opts['baseline_drift'],
        }
    if method == 'cidre':
        return {
            'smoothing_sigma': opts['smoothing_sigma'],
            'dark_quantile': opts['dark_quantile'],
        }
    if method == 'cellprofiler':
        return {
            'smoothing_sigma': opts['smoothing_sigma'],
            'cellprofiler_mode': opts['cellprofiler_mode'],
        }
    if method == 'flatfield':
        return {
            'flat_xy': flat_xy,
            'dark_xy': dark_xy,
            'dark_constant': opts['dark_constant'],
        }
    if method == 'destripe':
        return {
            'destripe_sigma': opts['destripe_sigma'],
            'destripe_wavelet': opts['destripe_wavelet'],
            'destripe_level': opts['destripe_level'],
        }
    if method == 'sscor':
        params = {
            'sscor_mode': opts['sscor_mode'],
            'sscor_patch_size': opts['sscor_patch_size'],
            'sscor_offset_size': opts['sscor_offset_size'],
            'sscor_repeat': opts['sscor_repeat'],
            'sscor_dark_threshold': opts['sscor_dark_threshold'],
        }
        if opts['sscor_mode'] == 'self-train':
            params.update({
                'sscor_stripe_direction': opts['sscor_stripe_direction'],
                'sscor_h_n': opts['sscor_h_n'],
                'sscor_v_n': opts['sscor_v_n'],
                'sscor_grid_direction': opts['sscor_grid_direction'],
                'sscor_epochs': opts['sscor_epochs'],
            })
        return params
    return {}


_CORRECTION_FUNCTIONS = {
    'basic': 'correct_basic',
    'cidre': 'correct_cidre',
    'cellprofiler': 'correct_cellprofiler',
    'flatfield': 'correct_flatfield',
    'destripe': 'correct_destripe',
    'sscor': 'correct_sscor',
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

    # Lazy import: keeps large_image off the interface/preview path; only needed during compute.
    import large_image as li

    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    workerInterface = params['workerInterface']

    method = workerInterface.get('Method', 'basic')

    allChannels = workerInterface.get('Channels to correct', {})
    channels = [int(k) for k, v in allChannels.items() if v]
    if len(channels) == 0:
        sendError('No channels to correct',
                   info='Select at least one channel in "Channels to correct".')
        return

    if 'frames' not in tileClient.tiles:
        sendError('Only one image; exiting',
                   info='Illumination correction requires a multi-frame dataset '
                        '(a `frames` list) to build a per-channel image collection.')
        return

    frames = tileClient.tiles['frames']

    opts = {
        'estimate_darkfield': bool(workerInterface.get('Estimate darkfield', True)),
        'flatfield_smoothness': float(workerInterface.get('Flatfield smoothness', 1.0)),
        'darkfield_smoothness': float(workerInterface.get('Darkfield smoothness', 1.0)),
        'baseline_drift': bool(workerInterface.get('Correct timelapse baseline drift', False)),
        'smoothing_sigma': float(workerInterface.get('Smoothing sigma', 50)),
        'dark_quantile': float(workerInterface.get('Dark quantile', 0.02)),
        'cellprofiler_mode': workerInterface.get('CellProfiler mode', 'regular'),
        'dark_constant': float(workerInterface.get('Dark-field constant', 0)),
        'destripe_sigma': float(workerInterface.get('Destripe sigma', 128)),
        'destripe_wavelet': workerInterface.get('Destripe wavelet', 'db3'),
        'destripe_level': int(workerInterface.get('Destripe level', 0)),
        'sscor_patch_size': int(workerInterface.get('SSCOR patch size', 256)),
        'sscor_offset_size': int(workerInterface.get('SSCOR offset size', 100)),
        'sscor_repeat': int(workerInterface.get('SSCOR repeat', 1)),
        'sscor_dark_threshold': int(workerInterface.get('SSCOR dark threshold', 10)),
        'sscor_mode': workerInterface.get('SSCOR mode', 'pretrained'),
        'sscor_stripe_direction': workerInterface.get('SSCOR stripe direction', 'horizontal'),
        'sscor_h_n': int(workerInterface.get('SSCOR horizontal stripe count', 1)),
        'sscor_v_n': int(workerInterface.get('SSCOR vertical stripe count', 1)),
        'sscor_grid_direction': int(workerInterface.get('SSCOR grid direction', 0)),
        'sscor_epochs': int(workerInterface.get('SSCOR training epochs', 30)),
    }

    qc_enabled = bool(workerInterface.get('Report correction quality (QC)', False))

    flat_xy_str = str(workerInterface.get('Flat-field XY coordinate', '') or '').strip()
    dark_xy_str = str(workerInterface.get('Dark-field XY coordinate', '') or '').strip()

    if method == 'flatfield' and flat_xy_str == '':
        sendError('Flat-field reference required',
                   info='The flatfield method requires a "Flat-field XY coordinate". Enter the '
                        '1-indexed XY position of a flat-field calibration image, or choose a '
                        'different Method.')
        return

    flat_xy = int(flat_xy_str) - 1 if flat_xy_str != '' else None
    dark_xy = int(dark_xy_str) - 1 if dark_xy_str != '' else None

    sscor_weights_path = None
    sscor_gpu_ids = '-1'
    if method == 'sscor':
        if opts['sscor_mode'] == 'pretrained':
            sscor_weights_path = resolve_sscor_checkpoint()
            if sscor_weights_path is None:
                return  # resolve_sscor_checkpoint() already called sendError
        else:
            # self-train needs no pre-supplied checkpoint -- it trains its own,
            # per frame, from patches sampled out of the image itself.
            sendWarning(
                'SSCOR self-training enabled',
                info='SSCOR "self-train" mode needs no pre-supplied checkpoint, but it trains a '
                     'fresh CycleGAN model PER FRAME (sampling stripe/clean patches from the '
                     'image itself, then training, then restoring); this can be slow, '
                     'especially without a GPU.')

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        if cuda_available:
            sscor_gpu_ids = '0'
        else:
            sscor_gpu_ids = '-1'
            sendWarning('Running SSCOR on CPU',
                        info='No CUDA GPU detected; SSCOR deep-learning stripe correction on '
                             'CPU is very slow. A GPU is strongly recommended.')

    # Group frame indices by channel
    channel_frame_indices = {c: [] for c in channels}
    for i, frame in enumerate(frames):
        c = frame.get('IndexC', 0)
        if c in channel_frame_indices:
            channel_frame_indices[c].append(i)

    correction_fn_name = _CORRECTION_FUNCTIONS.get(method)
    if correction_fn_name is None:
        sendError(f'Unknown method: {method}')
        return

    source_dtype = tileClient.tiles.get('dtype', None)

    corrected_by_frame = {}
    diagnostics_by_channel = {}
    qc_by_channel = {}

    total_channels = len(channels)
    for ci, channel in enumerate(channels):
        indices = channel_frame_indices.get(channel, [])
        if len(indices) == 0:
            continue

        if len(indices) < 3 and method in ('basic', 'cidre', 'cellprofiler'):
            sendWarning('Small image collection',
                        info=f'Channel {channel} has only {len(indices)} frame(s); retrospective '
                             'illumination estimation may be unreliable with fewer than 3 images.')

        stack = np.stack(
            [tileClient.getRegion(datasetId, frame=i).squeeze() for i in indices], axis=0)

        method_opts = dict(opts)

        if method == 'basic':
            method_opts['time_order'] = [frames[i].get('IndexT', 0) for i in indices]

        if method == 'flatfield':
            flat_frame = tileClient.coordinatesToFrameIndex(flat_xy, 0, 0, channel)
            flat = tileClient.getRegion(datasetId, frame=flat_frame).squeeze().astype(np.float64)
            if dark_xy is not None:
                dark_frame = tileClient.coordinatesToFrameIndex(dark_xy, 0, 0, channel)
                dark = tileClient.getRegion(datasetId, frame=dark_frame).squeeze().astype(np.float64)
            else:
                dark = np.full_like(flat, method_opts['dark_constant'], dtype=np.float64)
            method_opts['flat'] = flat
            method_opts['dark'] = dark

        if method == 'sscor':
            method_opts['sscor_weights'] = sscor_weights_path
            method_opts['sscor_gpu_ids'] = sscor_gpu_ids

        correction_fn = globals()[correction_fn_name]
        corrected, diag = correction_fn(stack, method_opts)
        corrected = np.nan_to_num(np.asarray(corrected, dtype=np.float64))

        diagnostics_by_channel[channel] = diag

        if qc_enabled:
            qc_by_channel[channel] = compute_qc_metrics(corrected, method_opts)

        if source_dtype is not None:
            dtype = np.dtype(source_dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                corrected = np.clip(corrected, info.min, info.max).astype(dtype)
            else:
                corrected = corrected.astype(dtype)

        for pos, frame_idx in enumerate(indices):
            corrected_by_frame[frame_idx] = corrected[pos]

        sendProgress((ci + 1) / total_channels, 'Illumination correction',
                     f"Corrected channel {channel} ({ci + 1}/{total_channels})")

    gc = tileClient.client

    sink = li.new()

    for i, frame in enumerate(frames):
        # Create a parameters dictionary with only the indices that exist in frame
        # The len(k) > 5 is to avoid the 'Index' key that has no postfix to it
        large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items(
        ) if k.startswith('Index') and len(k) > 5}

        if i in corrected_by_frame:
            image = corrected_by_frame[i]
        else:
            image = tileClient.getRegion(datasetId, frame=i).squeeze()

        sink.addTile(image, 0, 0, **large_image_params)

        sendProgress(i / len(frames), 'Illumination correction',
                     f"Writing frame {i + 1}/{len(frames)}")

    # Copy over the metadata
    if 'channels' in tileClient.tiles:
        sink.channelNames = tileClient.tiles['channels']

    sink.mm_x = tileClient.tiles['mm_x']
    sink.mm_y = tileClient.tiles['mm_y']
    sink.magnification = tileClient.tiles['magnification']
    sink.write('/tmp/illumination_corrected.tiff')
    print("Wrote to file")

    item = gc.uploadFileToFolder(datasetId, '/tmp/illumination_corrected.tiff')

    metadata = {
        'tool': 'Illumination Correction',
        'method': method,
        'channels': channels,
    }
    metadata.update(_method_params_for_metadata(method, opts, flat_xy, dark_xy))
    if diagnostics_by_channel:
        metadata['diagnostics'] = diagnostics_by_channel
    if qc_enabled and qc_by_channel:
        metadata['qc'] = qc_by_channel

    gc.addMetadataToItem(item['itemId'], metadata)
    print("Uploaded file")


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Correct illumination/shading/vignetting/striping in images')

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
