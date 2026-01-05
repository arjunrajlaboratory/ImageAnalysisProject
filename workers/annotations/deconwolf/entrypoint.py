import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning, sendError

import numpy as np
import tifffile
import large_image as li


def interface(image, apiUrl, token):
    """Define the worker interface shown to users."""
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Channels to deconvolve': {
            'type': 'channelCheckboxes',
            'tooltip': 'Select channels to deconvolve. Unselected channels pass through unchanged.',
            'displayOrder': 1,
        },
        'Auto-extract from ND2': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'Try to extract optical parameters from ND2 metadata (falls back to manual input if unavailable)',
            'displayOrder': 2,
        },
        'Numerical Aperture (NA)': {
            'type': 'number',
            'min': 0.1,
            'max': 1.7,
            'default': 0.75,
            'tooltip': 'Numerical aperture of the objective',
            'displayOrder': 3,
        },
        'Refractive Index (ni)': {
            'type': 'number',
            'min': 1.0,
            'max': 1.6,
            'default': 1.0,
            'tooltip': '1.0 for air, 1.515 for oil immersion',
            'displayOrder': 4,
        },
        'Pixel Size XY (nm)': {
            'type': 'number',
            'min': 1,
            'max': 10000,
            'default': 325,
            'tooltip': 'Lateral pixel size in nanometers',
            'displayOrder': 5,
        },
        'Z Step (nm)': {
            'type': 'number',
            'min': 1,
            'max': 50000,
            'default': 5000,
            'tooltip': 'Axial step size in nanometers',
            'displayOrder': 6,
        },
        'Emission Wavelength (nm)': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'e.g., 450,520,580,680',
                'label': 'Emission wavelengths (comma-separated)',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'tooltip': 'One wavelength per channel in order. Use single value for all channels or comma-separated for each.',
            'displayOrder': 7,
        },
        'Iterations': {
            'type': 'number',
            'min': 1,
            'max': 200,
            'default': 50,
            'tooltip': 'Number of Richardson-Lucy iterations (higher = sharper but slower, 20-100 typical)',
            'displayOrder': 8,
        },
    }
    client.setWorkerImageInterface(image, interface)


def generate_psf(NA, wavelength, ni, resxy, resz, nslice, output_path):
    """
    Generate a PSF using deconwolf's dw_bw (Born-Wolf model).

    Parameters:
        NA: Numerical aperture
        wavelength: Emission wavelength in nm
        ni: Refractive index of immersion medium
        resxy: Lateral pixel size in nm
        resz: Axial step size in nm
        nslice: Number of Z slices in PSF
        output_path: Path to write PSF TIFF file

    Returns:
        output_path on success
    """
    cmd = [
        'dw_bw',
        '--NA', str(NA),
        '--lambda', str(wavelength),
        '--ni', str(ni),
        '--resxy', str(resxy),
        '--resz', str(resz),
        '--nslice', str(nslice),
        '--overwrite',
        output_path
    ]
    print(f"Generating PSF: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"PSF generation failed: {result.stderr}")
    return output_path


def deconvolve_stack(z_stack, psf_path, iterations, work_dir):
    """
    Deconvolve a Z-stack using deconwolf's dw command.

    Parameters:
        z_stack: numpy array of shape (Z, Y, X)
        psf_path: Path to PSF TIFF file
        iterations: Number of Richardson-Lucy iterations
        work_dir: Working directory for temporary files

    Returns:
        Deconvolved Z-stack as numpy array
    """
    # Save input stack to temp file
    input_path = os.path.join(work_dir, 'input_stack.tiff')
    tifffile.imwrite(input_path, z_stack, imagej=True)

    # Run deconvolution
    cmd = [
        'dw',
        '--iter', str(iterations),
        '--threads', str(os.cpu_count() or 4),
        '--overwrite',
        input_path,
        psf_path
    ]
    print(f"Running deconvolution: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.returncode != 0:
        raise RuntimeError(f"Deconvolution failed: {result.stderr}")

    # Output file is prefixed with 'dw_'
    output_path = os.path.join(work_dir, 'dw_input_stack.tiff')
    if not os.path.exists(output_path):
        raise RuntimeError(f"Expected output file not found: {output_path}")

    # Read and return result
    deconvolved = tifffile.imread(output_path)
    return deconvolved


def parse_wavelengths(wavelength_str, channels):
    """
    Parse wavelength string into list of wavelengths per channel.

    If single value provided, use for all channels.
    If comma-separated, map to channels in order.
    """
    if not wavelength_str or wavelength_str.strip() == '':
        # Default wavelengths if none provided
        default_wavelengths = [450, 520, 580, 680]
        return [default_wavelengths[i % len(default_wavelengths)] for i in channels]

    parts = [p.strip() for p in wavelength_str.split(',')]
    wavelengths = [float(p) for p in parts if p]

    if len(wavelengths) == 1:
        # Single wavelength for all channels
        return [wavelengths[0]] * len(channels)
    elif len(wavelengths) >= len(channels):
        # One wavelength per channel
        return [wavelengths[c] for c in channels]
    else:
        # Not enough wavelengths provided, cycle through
        return [wavelengths[i % len(wavelengths)] for i in range(len(channels))]


def try_extract_nd2_metadata(tileClient):
    """
    Try to extract optical parameters from image metadata.

    Returns dict with NA, ni, resxy, resz, wavelengths if available,
    or None if metadata is not accessible.
    """
    try:
        # Check if we have ND2-specific metadata in tiles
        tile_metadata = tileClient.tiles

        # Try to get pixel size from metadata
        # mm_x and mm_y are in mm per pixel, convert to nm
        mm_x = tile_metadata.get('mm_x')
        mm_y = tile_metadata.get('mm_y')

        if mm_x and mm_y:
            # Convert mm to nm (1 mm = 1,000,000 nm)
            resxy = mm_x * 1_000_000
            print(f"Extracted pixel size from metadata: {resxy} nm")
        else:
            return None

        # Check for internal metadata which might have more ND2-specific info
        # This depends on whether Girder exposes internal metadata
        internal = tile_metadata.get('internal', {})
        nd2_meta = internal.get('nd2', {})

        if nd2_meta:
            # Try to extract from ND2 internal metadata
            channels = nd2_meta.get('channels', [])
            if channels and len(channels) > 0:
                first_channel = channels[0]
                microscope = first_channel.get('microscope', {})
                NA = microscope.get('objectiveNumericalAperture')
                ni = microscope.get('immersionRefractiveIndex')

                channel_info = first_channel.get('channel', {})
                wavelengths = [ch.get('channel', {}).get('emissionLambdaNm')
                               for ch in channels if ch.get('channel', {}).get('emissionLambdaNm')]

                if NA and ni:
                    return {
                        'NA': NA,
                        'ni': ni,
                        'resxy': resxy,
                        'wavelengths': wavelengths if wavelengths else None
                    }

        # If we only have pixel size, return partial info
        if mm_x:
            return {
                'resxy': resxy,
            }

    except Exception as e:
        print(f"Could not extract ND2 metadata: {e}")

    return None


def get_manual_params(workerInterface):
    """Extract optical parameters from worker interface inputs."""
    return {
        'NA': float(workerInterface.get('Numerical Aperture (NA)', 0.75)),
        'ni': float(workerInterface.get('Refractive Index (ni)', 1.0)),
        'resxy': float(workerInterface.get('Pixel Size XY (nm)', 325)),
        'resz': float(workerInterface.get('Z Step (nm)', 5000)),
    }


def copy_image_unchanged(tileClient, datasetId, gc):
    """Copy the image unchanged (for 2D case)."""
    sink = li.new()

    if 'frames' in tileClient.tiles:
        for i, frame in enumerate(tileClient.tiles['frames']):
            large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items()
                                  if k.startswith('Index') and len(k) > 5}
            image = tileClient.getRegion(datasetId, frame=i).squeeze()
            sink.addTile(image, 0, 0, **large_image_params)
            sendProgress(i / len(tileClient.tiles['frames']), 'Copying image',
                         f"Frame {i+1}/{len(tileClient.tiles['frames'])}")
    else:
        image = tileClient.getRegion(datasetId, frame=0).squeeze()
        sink.addTile(image, 0, 0, z=0)

    # Copy metadata
    if 'channels' in tileClient.tiles:
        sink.channelNames = tileClient.tiles['channels']
    sink.mm_x = tileClient.tiles['mm_x']
    sink.mm_y = tileClient.tiles['mm_y']
    sink.magnification = tileClient.tiles['magnification']

    sink.write('/tmp/deconvolved.tiff')
    item = gc.uploadFileToFolder(datasetId, '/tmp/deconvolved.tiff')
    gc.addMetadataToItem(item['itemId'], {
        'tool': 'Deconwolf Deconvolution',
        'note': 'Image copied unchanged (2D input)',
    })


def compute(datasetId, apiUrl, token, params):
    """Main computation function for deconvolution."""

    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    workerInterface = params['workerInterface']

    # Parse channel selection
    allChannels = workerInterface.get('Channels to deconvolve', {})
    channels = [int(k) for k, v in allChannels.items() if v]
    print(f"Selected channels to deconvolve: {channels}")

    if len(channels) == 0:
        sendError("No channels selected for deconvolution")
        return

    # Check for 2D images
    index_range = tileClient.tiles.get('IndexRange', {})
    num_z = index_range.get('IndexZ', 1)
    num_xy = index_range.get('IndexXY', 1)
    num_time = index_range.get('IndexT', 1)
    num_channels = index_range.get('IndexC', 1)

    print(f"Image dimensions: XY={num_xy}, Z={num_z}, T={num_time}, C={num_channels}")

    gc = tileClient.client

    if num_z <= 1:
        sendWarning("Image has only 1 Z-slice. Skipping deconvolution, outputting original image.")
        copy_image_unchanged(tileClient, datasetId, gc)
        return

    # Get optical parameters
    auto_extract = workerInterface.get('Auto-extract from ND2', False)
    optical_params = None

    if auto_extract:
        optical_params = try_extract_nd2_metadata(tileClient)
        if optical_params:
            print(f"Extracted optical parameters from metadata: {optical_params}")
        else:
            sendWarning("Could not extract ND2 metadata, using manual parameters")

    # Fall back to manual params
    manual_params = get_manual_params(workerInterface)
    if optical_params:
        # Merge: use extracted values where available, fall back to manual
        for key in ['NA', 'ni', 'resxy', 'resz']:
            if key not in optical_params or optical_params[key] is None:
                optical_params[key] = manual_params[key]
    else:
        optical_params = manual_params

    # Parse wavelengths
    wavelength_str = workerInterface.get('Emission Wavelength (nm)', '')
    wavelengths = parse_wavelengths(wavelength_str, channels)
    channel_wavelengths = dict(zip(channels, wavelengths))
    print(f"Channel wavelengths: {channel_wavelengths}")

    # Get deconvolution parameters
    iterations = int(workerInterface.get('Iterations', 50))

    # Calculate PSF size (should be >= 2*num_z - 1)
    psf_nslice = 2 * num_z - 1

    # Create working directory
    with tempfile.TemporaryDirectory() as work_dir:
        # Generate PSFs for each unique wavelength
        psf_files = {}
        unique_wavelengths = set(channel_wavelengths.values())
        for wl in unique_wavelengths:
            psf_path = os.path.join(work_dir, f'psf_{int(wl)}.tif')
            sendProgress(0, 'Generating PSFs', f"Wavelength {wl} nm")
            generate_psf(
                optical_params['NA'],
                wl,
                optical_params['ni'],
                optical_params['resxy'],
                optical_params['resz'],
                psf_nslice,
                psf_path
            )
            psf_files[wl] = psf_path

        # Map channels to their PSF files
        channel_psf = {ch: psf_files[wl] for ch, wl in channel_wavelengths.items()}

        # Group frames by (XY, Time, Channel) to process Z-stacks together
        frame_groups = defaultdict(list)
        for i, frame in enumerate(tileClient.tiles.get('frames', [])):
            key = (
                frame.get('IndexXY', 0),
                frame.get('IndexT', 0),
                frame.get('IndexC', 0)
            )
            frame_groups[key].append((i, frame.get('IndexZ', 0)))

        # Sort each group by Z index
        for key in frame_groups:
            frame_groups[key].sort(key=lambda x: x[1])

        # Process and deconvolve
        sink = li.new()
        total_groups = len(frame_groups)
        processed_groups = 0

        # Store deconvolved stacks for channels that need deconvolution
        deconvolved_stacks = {}

        for (xy, t, c), frame_indices in frame_groups.items():
            if c in channels:
                # Load Z-stack for this channel
                z_stack = []
                for frame_idx, z_idx in frame_indices:
                    img = tileClient.getRegion(datasetId, frame=frame_idx).squeeze()
                    z_stack.append(img)
                z_stack = np.stack(z_stack, axis=0)

                sendProgress(processed_groups / total_groups, 'Deconvolving',
                             f"XY={xy}, T={t}, Channel={c}")

                # Deconvolve
                deconvolved = deconvolve_stack(
                    z_stack,
                    channel_psf[c],
                    iterations,
                    work_dir
                )
                deconvolved_stacks[(xy, t, c)] = deconvolved

            processed_groups += 1

        # Now add all frames to sink (deconvolved or original)
        sendProgress(0.9, 'Assembling output', 'Writing frames')

        for i, frame in enumerate(tileClient.tiles.get('frames', [])):
            large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items()
                                  if k.startswith('Index') and len(k) > 5}

            xy = frame.get('IndexXY', 0)
            t = frame.get('IndexT', 0)
            c = frame.get('IndexC', 0)
            z = frame.get('IndexZ', 0)

            if c in channels and (xy, t, c) in deconvolved_stacks:
                # Use deconvolved data
                deconvolved = deconvolved_stacks[(xy, t, c)]
                # Find the Z index within this stack
                frame_list = frame_groups[(xy, t, c)]
                z_indices = [zi for _, zi in frame_list]
                stack_idx = z_indices.index(z)
                image = deconvolved[stack_idx]
            else:
                # Use original data
                image = tileClient.getRegion(datasetId, frame=i).squeeze()

            sink.addTile(image, 0, 0, **large_image_params)

        # Copy metadata
        if 'channels' in tileClient.tiles:
            sink.channelNames = tileClient.tiles['channels']
        sink.mm_x = tileClient.tiles['mm_x']
        sink.mm_y = tileClient.tiles['mm_y']
        sink.magnification = tileClient.tiles['magnification']

        sendProgress(0.95, 'Writing output', 'Saving TIFF file')
        sink.write('/tmp/deconvolved.tiff')
        print("Wrote to file")

        sendProgress(0.98, 'Uploading', 'Uploading to server')
        item = gc.uploadFileToFolder(datasetId, '/tmp/deconvolved.tiff')
        gc.addMetadataToItem(item['itemId'], {
            'tool': 'Deconwolf Deconvolution',
            'channels_deconvolved': channels,
            'iterations': iterations,
            'NA': optical_params['NA'],
            'ni': optical_params['ni'],
            'resxy_nm': optical_params['resxy'],
            'resz_nm': optical_params['resz'],
            'wavelengths': channel_wavelengths,
        })
        print("Uploaded file")

    sendProgress(1.0, 'Complete', 'Deconvolution finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deconvolve images using Richardson-Lucy algorithm with Born-Wolf PSF')

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')

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
