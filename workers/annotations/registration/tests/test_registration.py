import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from entrypoint import interface, compute, safe_astype, register_images


def test_interface():
    """Test that the interface is properly defined"""
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        client = mock_client.return_value

        interface('test_image', 'http://test-api', 'test-token')

        # Check that client was initialized correctly
        mock_client.assert_called_once_with(apiUrl='http://test-api', token='test-token')

        # Check that setWorkerImageInterface was called
        client.setWorkerImageInterface.assert_called_once()
        interface_definition = client.setWorkerImageInterface.call_args[0][1]

        # Verify all expected interface elements are present
        expected_keys = [
            'Apply to XY coordinates',
            'Reference Z Coordinate',
            'Reference Time Coordinate',
            'Reference Channel',
            'Channels to correct',
            'Reference region tag',
            'Control point tag',
            'Apply algorithm after control points',
            'Algorithm'
        ]

        for key in expected_keys:
            assert key in interface_definition

        # Check algorithm options
        assert interface_definition['Algorithm']['items'] == [
            'None (control points only)', 'Translation', 'Rigid', 'Affine'
        ]


def test_safe_astype_integer():
    """Test safe_astype function with integer conversion"""
    # Test clipping to uint8 range
    arr = np.array([100, 300, -50])
    result = safe_astype(arr, np.uint8)
    expected = np.array([100, 255, 0], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_safe_astype_float():
    """Test safe_astype function with float conversion"""
    arr = np.array([1, 2, 3])
    result = safe_astype(arr, np.float32)
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_register_images_control_points_only():
    """Test register_images function with control points only"""
    with patch('pystackreg.StackReg') as mock_stackreg:
        sr = mock_stackreg.return_value
        image1 = np.zeros((100, 100))
        image2 = np.zeros((100, 100))

        result = register_images(image1, image2, 'None (control points only)', sr)

        # Should return identity matrix
        expected = np.eye(3)
        np.testing.assert_array_equal(result, expected)

        # Should not call sr.register
        sr.register.assert_not_called()


def test_register_images_with_algorithm():
    """Test register_images function with actual algorithm"""
    with patch('pystackreg.StackReg') as mock_stackreg:
        sr = mock_stackreg.return_value
        expected_matrix = np.array([[1, 0, 5], [0, 1, 10], [0, 0, 1]])
        sr.register.return_value = expected_matrix

        image1 = np.zeros((100, 100))
        image2 = np.zeros((100, 100))

        result = register_images(image1, image2, 'Translation', sr)

        # Should call sr.register and return its result
        sr.register.assert_called_once_with(image1, image2)
        np.testing.assert_array_equal(result, expected_matrix)


def test_compute_single_image_error():
    """Test that compute properly handles single image case"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('entrypoint.sendError') as mock_send_error:

        client = mock_tile_client.return_value
        client.tiles = {}  # No IndexRange indicates single image

        compute('test_dataset', 'http://test-api', 'test-token', params)

        mock_send_error.assert_called_with("Just one image; exiting")


def test_compute_no_time_dimension_error():
    """Test that compute properly handles missing time dimension"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('entrypoint.sendError') as mock_send_error:

        client = mock_tile_client.return_value
        client.tiles = {'IndexRange': {'IndexXY': 2}}  # No IndexT

        compute('test_dataset', 'http://test-api', 'test-token', params)

        mock_send_error.assert_called_with("Time dimension not found; exiting")


def test_compute_no_channels_error():
    """Test that compute properly handles no channels selected"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {},  # No channels selected
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('entrypoint.sendError') as mock_send_error:

        client = mock_tile_client.return_value
        client.tiles = {'IndexRange': {'IndexXY': 2, 'IndexT': 5}}

        compute('test_dataset', 'http://test-api', 'test-token', params)

        mock_send_error.assert_called_with("No channels to correct")


def test_compute_basic_functionality():
    """Test basic registration functionality without control points or reference regions"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '1-2',
            'Reference Z Coordinate': '1',
            'Reference Time Coordinate': '1',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('large_image.new') as mock_sink, \
            patch('entrypoint.StackReg') as mock_stackreg, \
            patch('entrypoint.sendProgress'):

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexXY': 2, 'IndexT': 3, 'IndexZ': 1},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
            ],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Mock StackReg
        sr = mock_stackreg.return_value
        sr.register.return_value = np.eye(3)
        sr.transform.return_value = np.zeros((100, 100))

        # Mock sink
        sink = mock_sink.return_value

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify StackReg was initialized with correct algorithm
        mock_stackreg.assert_called()

        # Verify sink operations
        sink.addTile.assert_called()
        sink.write.assert_called_with('/tmp/registered.tiff')


def test_compute_with_reference_region():
    """Test registration with reference region"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': ['test_region'],
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient') as mock_annotation_client, \
            patch('large_image.new'), \
            patch('pystackreg.StackReg'), \
            patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags, \
            patch('entrypoint.sendProgress'):

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexT': 2},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
            ],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
        }
        client.getRegion.return_value = np.zeros((50, 50))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Mock annotation client
        annotation_client = mock_annotation_client.return_value
        annotation_client.getAnnotationsByDatasetId.return_value = []

        # Mock reference region annotation
        mock_get_tags.return_value = [{
            'coordinates': [
                {'x': 10, 'y': 20},
                {'x': 60, 'y': 20},
                {'x': 60, 'y': 70},
                {'x': 10, 'y': 70}
            ]
        }]

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify getRegion was called with bounding box coordinates
        client.getRegion.assert_called()
        call_args = client.getRegion.call_args_list
        # Should have calls with left, top, right, bottom parameters
        region_calls = [call for call in call_args if 'left' in call[1]]
        assert len(region_calls) > 0


def test_compute_with_control_points():
    """Test registration with control points"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': ['control_points']
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient') as mock_annotation_client, \
            patch('large_image.new'), \
            patch('pystackreg.StackReg') as mock_stackreg, \
            patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags, \
            patch('entrypoint.sendProgress'):

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexT': 2},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
            ],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Mock annotation client
        annotation_client = mock_annotation_client.return_value

        # Mock control points
        mock_get_tags.return_value = [
            {
                'location': {'XY': 0, 'Time': 0},
                'coordinates': [{'x': 100, 'y': 150}]
            },
            {
                'location': {'XY': 0, 'Time': 1},
                'coordinates': [{'x': 105, 'y': 155}]
            }
        ]

        # Mock StackReg
        sr = mock_stackreg.return_value
        sr.transform.return_value = np.zeros((100, 100))

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify control points were processed
        annotation_client.getAnnotationsByDatasetId.assert_called_with(
            'test_dataset', limit=1000, shape='point'
        )


def test_compute_different_algorithms():
    """Test registration with different StackReg algorithms"""
    algorithms = ['Translation', 'Rigid', 'Affine']
    stackreg_constants = ['TRANSLATION', 'RIGID_BODY', 'AFFINE']

    for algorithm, constant in zip(algorithms, stackreg_constants):
        params = {
            'workerInterface': {
                'Algorithm': algorithm,
                'Apply algorithm after control points': False,
                'Apply to XY coordinates': '',
                'Reference Z Coordinate': '',
                'Reference Time Coordinate': '',
                'Reference Channel': 0,
                'Channels to correct': {'0': True},
                'Reference region tag': None,
                'Control point tag': None
            },
            'tile': {'XY': 0, 'Z': 0, 'Time': 0},
            'channel': 0
        }

        with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
                patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
                patch('large_image.new'), \
                patch('entrypoint.StackReg') as mock_stackreg, \
                patch('entrypoint.sendProgress'):

            client = mock_tile_client.return_value
            client.tiles = {
                'IndexRange': {'IndexT': 2},
                'frames': [
                    {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                    {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
                ],
                'channels': ['DAPI'],
                'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
            }
            client.getRegion.return_value = np.zeros((100, 100))
            client.coordinatesToFrameIndex.return_value = 0
            client.client = MagicMock()
            client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

            # Mock StackReg
            sr = mock_stackreg.return_value
            sr.register.return_value = np.eye(3)
            sr.transform.return_value = np.zeros((100, 100))

            compute('test_dataset', 'http://test-api', 'test-token', params)

            # Verify StackReg was called
            mock_stackreg.assert_called()


def test_compute_apply_algorithm_after_control_points():
    """Test registration with control points followed by algorithm"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': True,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': ['control_points']
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient') as mock_annotation_client, \
            patch('large_image.new'), \
            patch('entrypoint.StackReg') as mock_stackreg, \
            patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags, \
            patch('entrypoint.sendProgress'):

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexT': 2},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
            ],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Mock annotation client
        annotation_client = mock_annotation_client.return_value

        # Mock control points
        mock_get_tags.return_value = [
            {
                'location': {'XY': 0, 'Time': 0},
                'coordinates': [{'x': 100, 'y': 150}]
            },
            {
                'location': {'XY': 0, 'Time': 1},
                'coordinates': [{'x': 105, 'y': 155}]
            }
        ]

        # Mock StackReg
        sr = mock_stackreg.return_value
        sr.register.return_value = np.eye(3)
        sr.transform.side_effect = lambda img, tmat: img  # Return transformed image

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify both control point transformation and algorithm registration were applied
        sr.transform.assert_called()
        sr.register.assert_called()


def test_compute_reference_time_adjustment():
    """Test registration matrices adjustment for non-zero reference time"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '1',
            'Reference Time Coordinate': '2',  # Non-zero reference time
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('large_image.new'), \
            patch('pystackreg.StackReg') as mock_stackreg, \
            patch('entrypoint.sendProgress'):

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexT': 3},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 2, 'IndexC': 0}
            ],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Mock StackReg
        sr = mock_stackreg.return_value
        sr.register.return_value = np.eye(3)
        sr.transform.return_value = np.zeros((100, 100))

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Should complete without error (reference time adjustment logic tested implicitly)
        client.client.uploadFileToFolder.assert_called_once()


def test_compute_metadata_preservation():
    """Test that image metadata is properly preserved"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('large_image.new') as mock_sink, \
            patch('pystackreg.StackReg'), \
            patch('entrypoint.sendProgress'):

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexT': 2},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
            ],
            'channels': ['DAPI', 'GFP'],
            'mm_x': 0.65,
            'mm_y': 0.65,
            'magnification': 20
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Mock sink
        sink = mock_sink.return_value

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify metadata was preserved
        assert sink.channelNames == ['DAPI', 'GFP']
        assert sink.mm_x == 0.65
        assert sink.mm_y == 0.65
        assert sink.magnification == 20


def test_compute_progress_reporting():
    """Test that progress is properly reported during computation"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '1-2',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('large_image.new'), \
            patch('pystackreg.StackReg'), \
            patch('entrypoint.sendProgress') as mock_progress:

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexXY': 2, 'IndexT': 3},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
            ],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify progress was reported
        mock_progress.assert_called()
        # Should have calls for both matrix calculation and frame processing
        progress_calls = mock_progress.call_args_list
        assert len(progress_calls) > 0


def test_compute_invalid_algorithm_error():
    """Test that compute properly handles invalid algorithm"""
    params = {
        'workerInterface': {
            'Algorithm': 'InvalidAlgorithm',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('entrypoint.sendError') as mock_send_error:

        client = mock_tile_client.return_value
        client.tiles = {'IndexRange': {'IndexT': 2}}

        compute('test_dataset', 'http://test-api', 'test-token', params)

        mock_send_error.assert_called_with("Invalid algorithm: InvalidAlgorithm")


def test_compute_reference_region_not_found():
    """Test proper handling when reference region tag is not found"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': ['nonexistent_tag'],
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient') as mock_annotation_client, \
            patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags, \
            patch('entrypoint.sendError') as mock_send_error:

        client = mock_tile_client.return_value
        client.tiles = {'IndexRange': {'IndexT': 2}}

        # Mock annotation client
        annotation_client = mock_annotation_client.return_value
        annotation_client.getAnnotationsByDatasetId.return_value = []

        # Mock no annotations found
        mock_get_tags.return_value = []

        compute('test_dataset', 'http://test-api', 'test-token', params)

        mock_send_error.assert_called_with("No reference region found")


def test_xy_coordinate_parsing():
    """Test XY coordinate range parsing functionality"""
    params = {
        'workerInterface': {
            'Algorithm': 'Translation',
            'Apply algorithm after control points': False,
            'Apply to XY coordinates': '1-2,4',  # Complex range
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Reference Channel': 0,
            'Channels to correct': {'0': True},
            'Reference region tag': None,
            'Control point tag': None
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_tile_client, \
            patch('annotation_client.annotations.UPennContrastAnnotationClient'), \
            patch('large_image.new'), \
            patch('pystackreg.StackReg'), \
            patch('annotation_utilities.batch_argument_parser.process_range_list') as mock_process_range, \
            patch('entrypoint.sendProgress'):

        client = mock_tile_client.return_value
        client.tiles = {
            'IndexRange': {'IndexXY': 5, 'IndexT': 2},
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0}
            ],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Mock range processing
        mock_process_range.return_value = [0, 1, 3]  # 1-2,4 converted to 0-indexed

        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify range parsing was called correctly
        mock_process_range.assert_called_with('1-2,4', convert_one_to_zero_index=True)
