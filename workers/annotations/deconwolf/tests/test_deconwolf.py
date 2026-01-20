import pytest
from unittest.mock import patch, MagicMock, call
import json
import numpy as np
import os
import tempfile

# Import your worker module
from entrypoint import (
    compute, interface, generate_psf, deconvolve_stack,
    parse_wavelengths, try_extract_nd2_metadata, get_manual_params
)


@pytest.fixture
def mock_tile_client():
    """Mock the tiles.UPennContrastDataset"""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        # Set up default tile info with multiple frames and channels (3D data)
        client.tiles = {
            'frames': [
                # Channel 0, Z slices 0-2
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 1, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 2, 'IndexT': 0, 'IndexC': 0},
                # Channel 1, Z slices 0-2
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
                {'IndexXY': 0, 'IndexZ': 1, 'IndexT': 0, 'IndexC': 1},
                {'IndexXY': 0, 'IndexZ': 2, 'IndexT': 0, 'IndexC': 1},
            ],
            'IndexRange': {
                'IndexXY': 1,
                'IndexZ': 3,
                'IndexT': 1,
                'IndexC': 2
            },
            'channels': ['DAPI', 'FITC'],
            'mm_x': 0.000325,  # 325nm in mm
            'mm_y': 0.000325,
            'magnification': 20,
            'dtype': np.uint16
        }

        # Mock getRegion to return dummy image data
        client.getRegion.return_value = np.random.randint(0, 1000, (512, 512), dtype=np.uint16)

        # Mock the girder client
        mock_gc = MagicMock()
        mock_gc.uploadFileToFolder.return_value = {'itemId': 'test_item_id'}
        client.client = mock_gc

        yield client


@pytest.fixture
def mock_tile_client_2d():
    """Mock tile client for 2D (single Z) images"""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        client.tiles = {
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
            ],
            'IndexRange': {
                'IndexXY': 1,
                'IndexZ': 1,  # Only 1 Z slice
                'IndexT': 1,
                'IndexC': 2
            },
            'channels': ['DAPI', 'FITC'],
            'mm_x': 0.000325,
            'mm_y': 0.000325,
            'magnification': 20,
            'dtype': np.uint16
        }
        client.getRegion.return_value = np.random.randint(0, 1000, (512, 512), dtype=np.uint16)
        mock_gc = MagicMock()
        mock_gc.uploadFileToFolder.return_value = {'itemId': 'test_item_id'}
        client.client = mock_gc
        yield client


@pytest.fixture
def mock_worker_preview_client():
    """Mock the UPennContrastWorkerPreviewClient"""
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        yield mock_client.return_value


@pytest.fixture
def mock_large_image():
    """Mock large_image operations"""
    with patch('large_image.new') as mock_li_new:
        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink
        yield mock_sink


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for deconwolf commands"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout='', stderr='')
        yield mock_run


@pytest.fixture
def mock_tifffile():
    """Mock tifffile operations"""
    with patch('tifffile.imwrite') as mock_write, \
         patch('tifffile.imread') as mock_read:
        mock_read.return_value = np.random.randint(0, 1000, (3, 512, 512), dtype=np.uint16)
        yield {'imwrite': mock_write, 'imread': mock_read}


@pytest.fixture
def sample_params_basic():
    """Create basic sample parameters"""
    return {
        'workerInterface': {
            'Channels to deconvolve': {'0': True, '1': False},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
        }
    }


@pytest.fixture
def sample_params_multi_channel():
    """Create sample parameters with multiple channels"""
    return {
        'workerInterface': {
            'Channels to deconvolve': {'0': True, '1': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450,520',
            'Iterations': 50,
        }
    }


@pytest.fixture
def sample_params_no_channels():
    """Create sample parameters with no channels selected"""
    return {
        'workerInterface': {
            'Channels to deconvolve': {'0': False, '1': False},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
        }
    }


# ============== Interface Tests ==============

def test_interface(mock_worker_preview_client):
    """Test the interface generation"""
    interface('test_image', 'http://test-api', 'test-token')

    mock_worker_preview_client.setWorkerImageInterface.assert_called_once()
    call_args = mock_worker_preview_client.setWorkerImageInterface.call_args
    image_arg = call_args[0][0]
    interface_data = call_args[0][1]

    assert image_arg == 'test_image'

    # Verify required interface fields
    expected_fields = [
        'Channels to deconvolve',
        'Auto-extract from ND2',
        'Numerical Aperture (NA)',
        'Refractive Index (ni)',
        'Pixel Size XY (nm)',
        'Z Step (nm)',
        'Emission Wavelength (nm)',
        'Iterations',
    ]
    for field in expected_fields:
        assert field in interface_data, f"Missing interface field: {field}"

    # Check types
    assert interface_data['Channels to deconvolve']['type'] == 'channelCheckboxes'
    assert interface_data['Auto-extract from ND2']['type'] == 'checkbox'
    assert interface_data['Numerical Aperture (NA)']['type'] == 'number'
    assert interface_data['Iterations']['type'] == 'number'
    assert interface_data['Emission Wavelength (nm)']['type'] == 'text'


def test_interface_defaults(mock_worker_preview_client):
    """Test that interface has sensible defaults"""
    interface('test_image', 'http://test-api', 'test-token')

    interface_data = mock_worker_preview_client.setWorkerImageInterface.call_args[0][1]

    assert interface_data['Numerical Aperture (NA)']['default'] == 0.75
    assert interface_data['Refractive Index (ni)']['default'] == 1.0
    assert interface_data['Iterations']['default'] == 50


# ============== Wavelength Parsing Tests ==============

def test_parse_wavelengths_single():
    """Test parsing single wavelength for all channels"""
    channels = [0, 1, 2]
    wavelengths = parse_wavelengths('520', channels)
    assert wavelengths == [520.0, 520.0, 520.0]


def test_parse_wavelengths_multiple():
    """Test parsing multiple wavelengths"""
    channels = [0, 1]
    wavelengths = parse_wavelengths('450,520', channels)
    assert wavelengths == [450.0, 520.0]


def test_parse_wavelengths_empty():
    """Test default wavelengths when empty string"""
    channels = [0, 1]
    wavelengths = parse_wavelengths('', channels)
    assert len(wavelengths) == 2
    assert all(isinstance(w, (int, float)) for w in wavelengths)


def test_parse_wavelengths_with_spaces():
    """Test parsing wavelengths with spaces"""
    channels = [0, 1]
    wavelengths = parse_wavelengths('450, 520', channels)
    assert wavelengths == [450.0, 520.0]


def test_parse_wavelengths_cycling():
    """Test wavelength cycling when not enough provided"""
    channels = [0, 1, 2, 3]
    wavelengths = parse_wavelengths('450,520', channels)
    assert wavelengths == [450.0, 520.0, 450.0, 520.0]


# ============== Parameter Extraction Tests ==============

def test_get_manual_params():
    """Test manual parameter extraction"""
    worker_interface = {
        'Numerical Aperture (NA)': 1.2,
        'Refractive Index (ni)': 1.515,
        'Pixel Size XY (nm)': 200,
        'Z Step (nm)': 300,
    }
    params = get_manual_params(worker_interface)

    assert params['NA'] == 1.2
    assert params['ni'] == 1.515
    assert params['resxy'] == 200.0
    assert params['resz'] == 300.0


def test_get_manual_params_defaults():
    """Test manual parameter extraction with defaults"""
    params = get_manual_params({})

    assert params['NA'] == 0.75
    assert params['ni'] == 1.0
    assert params['resxy'] == 325.0
    assert params['resz'] == 5000.0


# ============== PSF Generation Tests ==============

def test_generate_psf_command(mock_subprocess):
    """Test PSF generation command construction"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'test_psf.tif')
        generate_psf(0.75, 450, 1.0, 325, 5000, 5, output_path)

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]

        assert call_args[0] == 'dw_bw'
        assert '--NA' in call_args
        assert '0.75' in call_args
        assert '--lambda' in call_args
        assert '450' in call_args
        assert '--ni' in call_args
        assert '1.0' in call_args
        assert '--resxy' in call_args
        assert '325' in call_args
        assert '--resz' in call_args
        assert '5000' in call_args
        assert '--nslice' in call_args
        assert '5' in call_args
        assert '--overwrite' in call_args
        assert output_path in call_args


def test_generate_psf_failure(mock_subprocess):
    """Test PSF generation failure handling"""
    mock_subprocess.return_value = MagicMock(returncode=1, stderr='Error message')

    with pytest.raises(RuntimeError, match='PSF generation failed'):
        generate_psf(0.75, 450, 1.0, 325, 5000, 5, '/tmp/test.tif')


# ============== Deconvolution Tests ==============

def test_deconvolve_stack_command(mock_subprocess, mock_tifffile):
    """Test deconvolution command construction"""
    z_stack = np.random.randint(0, 1000, (3, 512, 512), dtype=np.uint16)

    with tempfile.TemporaryDirectory() as work_dir:
        # Create a fake output file
        output_path = os.path.join(work_dir, 'dw_input_stack.tiff')
        with open(output_path, 'w') as f:
            f.write('fake')

        deconvolve_stack(z_stack, '/tmp/psf.tif', 50, work_dir)

        # Check that tifffile.imwrite was called to save input
        mock_tifffile['imwrite'].assert_called_once()

        # Check subprocess was called with dw command
        mock_subprocess.assert_called()
        call_args = mock_subprocess.call_args[0][0]

        assert call_args[0] == 'dw'
        assert '--iter' in call_args
        assert '50' in call_args
        assert '--threads' in call_args
        assert '--overwrite' in call_args
        assert '/tmp/psf.tif' in call_args


def test_deconvolve_stack_failure(mock_subprocess, mock_tifffile):
    """Test deconvolution failure handling"""
    mock_subprocess.return_value = MagicMock(returncode=1, stderr='Deconv error')
    z_stack = np.random.randint(0, 1000, (3, 512, 512), dtype=np.uint16)

    with tempfile.TemporaryDirectory() as work_dir:
        with pytest.raises(RuntimeError, match='Deconvolution failed'):
            deconvolve_stack(z_stack, '/tmp/psf.tif', 50, work_dir)


# ============== Compute Function Tests ==============

def test_compute_no_channels_error(mock_tile_client, sample_params_no_channels, capsys):
    """Test error when no channels selected"""
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_no_channels)

    captured = capsys.readouterr()
    assert '"error": "No channels selected for deconvolution"' in captured.out


def test_compute_2d_image_skip(mock_tile_client_2d, mock_large_image, sample_params_basic, capsys):
    """Test that 2D images skip deconvolution with warning"""
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    captured = capsys.readouterr()
    assert 'only 1 Z-slice' in captured.out.lower() or 'Skipping deconvolution' in captured.out

    # Should still upload the image unchanged
    mock_tile_client_2d.client.uploadFileToFolder.assert_called_once()


def test_compute_basic_functionality(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile, sample_params_basic
):
    """Test basic deconvolution workflow"""
    # Need to create fake output files for deconvolution
    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Should have called subprocess for PSF generation and deconvolution
    assert mock_subprocess.call_count >= 2  # At least 1 PSF + 1 deconvolution

    # Verify output was uploaded
    mock_tile_client.client.uploadFileToFolder.assert_called_once()

    # Verify metadata was added
    mock_tile_client.client.addMetadataToItem.assert_called_once()
    metadata_call = mock_tile_client.client.addMetadataToItem.call_args[0][1]
    assert metadata_call['tool'] == 'Deconwolf Deconvolution'
    assert 'channels_deconvolved' in metadata_call
    assert 'iterations' in metadata_call


def test_compute_multi_channel(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile, sample_params_multi_channel
):
    """Test deconvolution with multiple channels"""
    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_multi_channel)

    # Should generate PSFs for different wavelengths
    # With wavelengths 450 and 520, should have 2 PSF generation calls
    psf_calls = [c for c in mock_subprocess.call_args_list if 'dw_bw' in c[0][0]]
    assert len(psf_calls) == 2

    # Should deconvolve both channels
    dw_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == 'dw']
    assert len(dw_calls) == 2  # One for each channel


def test_compute_psf_nslice_calculation(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile, sample_params_basic
):
    """Test that PSF nslice is calculated correctly (2*num_z - 1)"""
    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # With 3 Z slices, PSF should have 2*3-1 = 5 slices
    psf_call = [c for c in mock_subprocess.call_args_list if 'dw_bw' in c[0][0]][0]
    psf_args = psf_call[0][0]
    nslice_idx = psf_args.index('--nslice') + 1
    assert psf_args[nslice_idx] == '5'


def test_compute_metadata_preservation(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile, sample_params_basic
):
    """Test that image metadata is preserved in output"""
    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify channel names were set
    assert mock_large_image.channelNames == ['DAPI', 'FITC']

    # Verify other metadata was preserved
    assert mock_large_image.mm_x == 0.000325
    assert mock_large_image.mm_y == 0.000325
    assert mock_large_image.magnification == 20


def test_compute_progress_reporting(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile, sample_params_basic, capsys
):
    """Test that progress is reported during processing"""
    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    captured = capsys.readouterr()
    assert '"progress":' in captured.out


# ============== ND2 Metadata Extraction Tests ==============

def test_try_extract_nd2_metadata_with_pixel_size():
    """Test ND2 metadata extraction when pixel size is available"""
    mock_tile_client = MagicMock()
    mock_tile_client.tiles = {
        'mm_x': 0.000325,  # 325nm
        'mm_y': 0.000325,
    }

    result = try_extract_nd2_metadata(mock_tile_client)

    # Should extract at least pixel size
    assert result is not None
    assert 'resxy' in result
    assert result['resxy'] == 325.0  # Converted from mm to nm


def test_try_extract_nd2_metadata_no_data():
    """Test ND2 metadata extraction when no data available"""
    mock_tile_client = MagicMock()
    mock_tile_client.tiles = {}

    result = try_extract_nd2_metadata(mock_tile_client)
    assert result is None


def test_try_extract_nd2_metadata_exception_handling():
    """Test ND2 metadata extraction handles exceptions gracefully"""
    mock_tile_client = MagicMock()
    mock_tile_client.tiles = {'mm_x': 'invalid'}  # Will cause conversion error

    # Should not raise, should return None
    result = try_extract_nd2_metadata(mock_tile_client)
    assert result is None


# ============== Integration-like Tests ==============

def test_full_workflow_single_channel(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile
):
    """Test complete workflow for single channel deconvolution"""
    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 1.0,
            'Refractive Index (ni)': 1.33,
            'Pixel Size XY (nm)': 100,
            'Z Step (nm)': 200,
            'Emission Wavelength (nm)': '488',
            'Iterations': 100,
        }
    }

    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', params)

    # Verify PSF generation used correct parameters
    psf_call = [c for c in mock_subprocess.call_args_list if 'dw_bw' in c[0][0]][0]
    psf_args = psf_call[0][0]
    assert '1.0' in psf_args  # NA
    assert '488.0' in psf_args  # wavelength (converted to float)
    assert '1.33' in psf_args  # ni
    assert '100.0' in psf_args  # resxy (converted to float)
    assert '200.0' in psf_args  # resz (converted to float)

    # Verify deconvolution used correct iterations
    dw_call = [c for c in mock_subprocess.call_args_list if c[0][0][0] == 'dw'][0]
    dw_args = dw_call[0][0]
    assert '100' in dw_args  # iterations


def test_auto_extract_fallback(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile
):
    """Test auto-extract falls back to manual when metadata unavailable"""
    # Remove mm_x to make auto-extract fail
    mock_tile_client.tiles['mm_x'] = None

    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': True,
            'Numerical Aperture (NA)': 0.9,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 250,
            'Z Step (nm)': 1000,
            'Emission Wavelength (nm)': '600',
            'Iterations': 30,
        }
    }

    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', params)

    # Should use manual parameters since auto-extract failed
    psf_call = [c for c in mock_subprocess.call_args_list if 'dw_bw' in c[0][0]][0]
    psf_args = psf_call[0][0]
    assert '0.9' in psf_args  # Manual NA
    assert '250.0' in psf_args  # Manual resxy (converted to float)


def test_gpu_flag_enabled(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile
):
    """Test that --gpu flag is added when GPU is enabled"""
    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
            'Use GPU': True,  # GPU enabled
        }
    }

    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', params)

    # Find the deconvolution command (dw, not dw_bw)
    dw_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == 'dw']
    assert len(dw_calls) >= 1
    dw_args = dw_calls[0][0][0]
    assert '--gpu' in dw_args


def test_gpu_flag_disabled(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile
):
    """Test that --gpu flag is NOT added when GPU is disabled"""
    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
            'Use GPU': False,  # GPU disabled
        }
    }

    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', params)

    # Find the deconvolution command (dw, not dw_bw)
    dw_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == 'dw']
    assert len(dw_calls) >= 1
    dw_args = dw_calls[0][0][0]
    assert '--gpu' not in dw_args


def test_interface_has_gpu_option(mock_worker_preview_client):
    """Test that interface includes GPU option"""
    interface('test_image', 'http://test-api', 'test-token')

    interface_data = mock_worker_preview_client.setWorkerImageInterface.call_args[0][1]

    assert 'Use GPU' in interface_data
    assert interface_data['Use GPU']['type'] == 'checkbox'
    assert interface_data['Use GPU']['default'] == True  # Default to GPU, will fallback if unavailable


def test_gpu_fallback_to_cpu(
    mock_tile_client, mock_large_image, mock_tifffile
):
    """Test that GPU failure triggers automatic fallback to CPU mode"""
    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
            'Use GPU': True,  # GPU enabled, but will fail
        }
    }

    def mock_run_side_effect(cmd, *args, **kwargs):
        result = MagicMock()
        if cmd[0] == 'dw':  # Deconvolution command
            if '--gpu' in cmd:
                # First call with GPU fails with OpenCL error
                result.returncode = 1
                result.stderr = "Error in cl_util.c: OpenCL initialization failed"
                result.stdout = ""
            else:
                # CPU fallback succeeds
                result.returncode = 0
                result.stderr = ""
                result.stdout = "Deconvolution complete"
        else:
            # PSF generation always succeeds
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
        return result

    with patch('subprocess.run', side_effect=mock_run_side_effect):
        with patch('os.path.exists', return_value=True):
            with patch('entrypoint.sendWarning') as mock_warning:
                compute('test_dataset', 'http://test-api', 'test-token', params)

                # Verify warning was sent about GPU fallback
                mock_warning.assert_called()
                warning_calls = [c for c in mock_warning.call_args_list
                                 if 'GPU' in str(c) or 'fallback' in str(c).lower()]
                assert len(warning_calls) >= 1


# ============== Tiling Tests ==============

def test_interface_has_tiling_options(mock_worker_preview_client):
    """Test that interface includes tiling options"""
    interface('test_image', 'http://test-api', 'test-token')

    interface_data = mock_worker_preview_client.setWorkerImageInterface.call_args[0][1]

    assert 'Tile Size (pixels)' in interface_data
    assert interface_data['Tile Size (pixels)']['type'] == 'number'
    assert interface_data['Tile Size (pixels)']['default'] == 1024
    assert interface_data['Tile Size (pixels)']['min'] == 256
    assert interface_data['Tile Size (pixels)']['max'] == 8192

    assert 'Tile Overlap (pixels)' in interface_data
    assert interface_data['Tile Overlap (pixels)']['type'] == 'number'
    assert interface_data['Tile Overlap (pixels)']['default'] == 100
    assert interface_data['Tile Overlap (pixels)']['min'] == 0
    assert interface_data['Tile Overlap (pixels)']['max'] == 500


def test_deconvolve_stack_no_tiling_small_image(mock_subprocess, mock_tifffile):
    """Test that tiling is NOT used when image is smaller than tile size"""
    # 512x512 image, tile_size=1024, so no tiling needed
    z_stack = np.random.randint(0, 1000, (3, 512, 512), dtype=np.uint16)

    with tempfile.TemporaryDirectory() as work_dir:
        output_path = os.path.join(work_dir, 'dw_input_stack.tiff')
        with open(output_path, 'w') as f:
            f.write('fake')

        deconvolve_stack(z_stack, '/tmp/psf.tif', 50, work_dir,
                         tile_size=1024, tile_overlap=100)

        # Check subprocess was called without tiling flags
        dw_call = mock_subprocess.call_args[0][0]
        assert '--tilesize' not in dw_call
        assert '--tilepad' not in dw_call


def test_deconvolve_stack_tiling_large_image(mock_subprocess, mock_tifffile):
    """Test that tiling IS used when image is larger than tile size"""
    # 2048x2048 image, tile_size=1024, so tiling should be used
    z_stack = np.random.randint(0, 1000, (3, 2048, 2048), dtype=np.uint16)

    with tempfile.TemporaryDirectory() as work_dir:
        output_path = os.path.join(work_dir, 'dw_input_stack.tiff')
        with open(output_path, 'w') as f:
            f.write('fake')

        deconvolve_stack(z_stack, '/tmp/psf.tif', 50, work_dir,
                         tile_size=1024, tile_overlap=100)

        # Check subprocess was called with tiling flags
        dw_call = mock_subprocess.call_args[0][0]
        assert '--tilesize' in dw_call
        assert '--tilepad' in dw_call
        # Check values
        tilesize_idx = dw_call.index('--tilesize') + 1
        tilepad_idx = dw_call.index('--tilepad') + 1
        assert dw_call[tilesize_idx] == '1024'
        assert dw_call[tilepad_idx] == '100'


def test_deconvolve_stack_tiling_custom_values(mock_subprocess, mock_tifffile):
    """Test tiling with custom tile size and overlap"""
    # 4096x4096 image with custom tile settings
    z_stack = np.random.randint(0, 1000, (3, 4096, 4096), dtype=np.uint16)

    with tempfile.TemporaryDirectory() as work_dir:
        output_path = os.path.join(work_dir, 'dw_input_stack.tiff')
        with open(output_path, 'w') as f:
            f.write('fake')

        deconvolve_stack(z_stack, '/tmp/psf.tif', 50, work_dir,
                         tile_size=2048, tile_overlap=200)

        dw_call = mock_subprocess.call_args[0][0]
        tilesize_idx = dw_call.index('--tilesize') + 1
        tilepad_idx = dw_call.index('--tilepad') + 1
        assert dw_call[tilesize_idx] == '2048'
        assert dw_call[tilepad_idx] == '200'


def test_deconvolve_stack_tiling_asymmetric_image(mock_subprocess, mock_tifffile):
    """Test tiling is triggered when only one dimension exceeds tile size"""
    # 512 height x 2048 width, should trigger tiling because max(512, 2048) > 1024
    z_stack = np.random.randint(0, 1000, (3, 512, 2048), dtype=np.uint16)

    with tempfile.TemporaryDirectory() as work_dir:
        output_path = os.path.join(work_dir, 'dw_input_stack.tiff')
        with open(output_path, 'w') as f:
            f.write('fake')

        deconvolve_stack(z_stack, '/tmp/psf.tif', 50, work_dir,
                         tile_size=1024, tile_overlap=100)

        dw_call = mock_subprocess.call_args[0][0]
        assert '--tilesize' in dw_call
        assert '--tilepad' in dw_call


def test_deconvolve_stack_tiling_edge_case_equal(mock_subprocess, mock_tifffile):
    """Test that image exactly equal to tile size does NOT trigger tiling"""
    # 1024x1024 image, tile_size=1024, no tiling (max(h,w) > tile_size is False)
    z_stack = np.random.randint(0, 1000, (3, 1024, 1024), dtype=np.uint16)

    with tempfile.TemporaryDirectory() as work_dir:
        output_path = os.path.join(work_dir, 'dw_input_stack.tiff')
        with open(output_path, 'w') as f:
            f.write('fake')

        deconvolve_stack(z_stack, '/tmp/psf.tif', 50, work_dir,
                         tile_size=1024, tile_overlap=100)

        dw_call = mock_subprocess.call_args[0][0]
        assert '--tilesize' not in dw_call
        assert '--tilepad' not in dw_call


@pytest.fixture
def mock_tile_client_large_image():
    """Mock tile client for large images that would trigger tiling"""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        client.tiles = {
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 1, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 2, 'IndexT': 0, 'IndexC': 0},
            ],
            'IndexRange': {
                'IndexXY': 1,
                'IndexZ': 3,
                'IndexT': 1,
                'IndexC': 1
            },
            'channels': ['DAPI'],
            'mm_x': 0.000325,
            'mm_y': 0.000325,
            'magnification': 20,
            'dtype': np.uint16
        }
        # Return large 2048x2048 images
        client.getRegion.return_value = np.random.randint(0, 1000, (2048, 2048), dtype=np.uint16)
        mock_gc = MagicMock()
        mock_gc.uploadFileToFolder.return_value = {'itemId': 'test_item_id'}
        client.client = mock_gc
        yield client


def test_compute_with_tiling_large_image(
    mock_tile_client_large_image, mock_large_image, mock_subprocess, mock_tifffile
):
    """Test full compute workflow with a large image that triggers tiling"""
    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
            'Use GPU': False,
            'Tile Size (pixels)': 1024,
            'Tile Overlap (pixels)': 100,
        }
    }

    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', params)

    # Find the deconvolution command (dw, not dw_bw)
    dw_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == 'dw']
    assert len(dw_calls) >= 1

    # Verify tiling was used (since image is 2048x2048 > 1024)
    dw_args = dw_calls[0][0][0]
    assert '--tilesize' in dw_args
    assert '--tilepad' in dw_args


def test_compute_without_tiling_small_image(
    mock_tile_client, mock_large_image, mock_subprocess, mock_tifffile
):
    """Test full compute workflow with a small image that doesn't trigger tiling"""
    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
            'Use GPU': False,
            'Tile Size (pixels)': 1024,
            'Tile Overlap (pixels)': 100,
        }
    }

    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', params)

    # Find the deconvolution command (dw, not dw_bw)
    dw_calls = [c for c in mock_subprocess.call_args_list if c[0][0][0] == 'dw']
    assert len(dw_calls) >= 1

    # Verify tiling was NOT used (since image is 512x512 < 1024)
    dw_args = dw_calls[0][0][0]
    assert '--tilesize' not in dw_args
    assert '--tilepad' not in dw_args


def test_compute_tiling_metadata_saved(
    mock_tile_client_large_image, mock_large_image, mock_subprocess, mock_tifffile
):
    """Test that tiling parameters are saved in metadata"""
    params = {
        'workerInterface': {
            'Channels to deconvolve': {'0': True},
            'Auto-extract from ND2': False,
            'Numerical Aperture (NA)': 0.75,
            'Refractive Index (ni)': 1.0,
            'Pixel Size XY (nm)': 325,
            'Z Step (nm)': 5000,
            'Emission Wavelength (nm)': '450',
            'Iterations': 50,
            'Use GPU': False,
            'Tile Size (pixels)': 2048,
            'Tile Overlap (pixels)': 150,
        }
    }

    with patch('os.path.exists', return_value=True):
        compute('test_dataset', 'http://test-api', 'test-token', params)

    # Verify metadata was added with tiling info
    metadata_call = mock_tile_client_large_image.client.addMetadataToItem.call_args[0][1]
    assert 'tile_size' in metadata_call
    assert 'tile_overlap' in metadata_call
    assert metadata_call['tile_size'] == 2048
    assert metadata_call['tile_overlap'] == 150
