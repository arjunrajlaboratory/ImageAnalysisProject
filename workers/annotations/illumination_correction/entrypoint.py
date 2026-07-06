import argparse
import json
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
            'items': ['basic', 'cidre', 'cellprofiler', 'flatfield', 'destripe'],
            'default': 'basic',
            'tooltip': 'Illumination-correction algorithm. basic (BaSiC) recommended when no '
                       'calibration frames are available. Every parameter below is shown '
                       'regardless of Method; each tooltip notes which method(s) it applies to '
                       '-- unused parameters for the chosen method are simply ignored.',
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
        # --- QC ---
        'Report correction quality (QC)': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'Compute lightweight EVEN-style flat-field-quality metrics per corrected '
                       'channel and store them in the output item metadata. Not the full EVEN '
                       'framework -- a quick quantitative stand-in for comparing methods.',
            'displayOrder': 15,
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
    return {}


_CORRECTION_FUNCTIONS = {
    'basic': 'correct_basic',
    'cidre': 'correct_cidre',
    'cellprofiler': 'correct_cellprofiler',
    'flatfield': 'correct_flatfield',
    'destripe': 'correct_destripe',
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
