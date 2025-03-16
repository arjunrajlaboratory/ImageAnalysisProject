import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the worker module
from entrypoint import compute, interface


@pytest.fixture
def mock_worker_client():
    """Mock the UPennContrastWorkerClient"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerClient'
    ) as mock_client:
        client = mock_client.return_value
        # Set up default behaviors
        client.get_annotation_list_by_shape.return_value = []
        yield client


@pytest.fixture
def mock_dataset_client():
    """Mock the UPennContrastDataset"""
    with patch(
        'annotation_client.tiles.UPennContrastDataset'
    ) as mock_client:
        client = mock_client.return_value
        # Set up default behaviors
        client.getRegion.return_value = None
        yield client


@pytest.fixture
def sample_annotation():
    """Create a sample polygon annotation"""
    return {
        '_id': 'test_id_1',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0},
            {'x': 0, 'y': 0}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']  # Add a tag that matches our filter
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'test_intensity',
        'image': 'properties/blob_annulus_intensity_percentile:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Channel': 0,
            'Radius': 5,  # Add the radius parameter
            'Percentile': 75  # Add percentile parameter
        }
    }


def test_interface():
    """Test the interface generation"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        # Verify interface was set
        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        # Verify interface structure
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        assert 'Channel' in interface_data
        assert interface_data['Channel']['type'] == 'channel'
        assert interface_data['Channel']['required'] is True
        assert 'Radius' in interface_data  # Check for Radius parameter
        assert interface_data['Radius']['type'] == 'number'
        assert interface_data['Radius']['default'] == 10
        assert 'Percentile' in interface_data  # Check for Percentile parameter
        assert interface_data['Percentile']['type'] == 'number'
        assert interface_data['Percentile']['default'] == 50


def test_worker_startup(mock_worker_client, mock_dataset_client, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Run computation with empty annotation list
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify that get_annotation_list_by_shape was called
    mock_worker_client.get_annotation_list_by_shape.assert_called_once_with(
        'polygon', limit=0)

    # Since there are no annotations, add_multiple_annotation_property_values should not be called
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_uniform_intensity_percentile(mock_worker_client, mock_dataset_client, sample_params):
    """Test percentile calculation with a uniform intensity image"""
    # Create a uniform intensity test image (all pixels = 50)
    test_image = np.ones((30, 30), dtype=np.uint8) * 50

    # Create a test annotation (10x10 square)
    test_annotation = {
        '_id': 'test_square',
        'coordinates': [
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 20},
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 10},
            {'x': 10, 'y': 10}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_square']

    # For a uniform image with all pixels = 50, the percentile value should be 50
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'

    # Assert that the property exists and has the expected value
    assert prop_name in property_values
    assert property_values[prop_name] == pytest.approx(50.0)


def test_gradient_intensity_percentile(mock_worker_client, mock_dataset_client, sample_params):
    """Test percentile calculation with a gradient intensity image"""
    # Create a gradient test image (values from 0 to 195)
    y, x = np.mgrid[0:40, 0:40]
    test_image = (x + y) * 2.5  # Creates a diagonal gradient from 0 to 195
    test_image = test_image.astype(np.uint8)

    # Create a test annotation (10x10 square in the middle)
    test_annotation = {
        '_id': 'test_gradient_square',
        'coordinates': [
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 25},
            {'x': 25, 'y': 25},
            {'x': 25, 'y': 15},
            {'x': 15, 'y': 15}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_gradient_square']

    # For the annulus, we need to calculate the expected values
    # Create a mask for the original square
    mask_original = np.zeros_like(test_image, dtype=bool)
    mask_original[15:25, 15:25] = True

    # Create a mask for the dilated square (with radius 5)
    mask_dilated = np.zeros_like(test_image, dtype=bool)
    # Approximate dilation by expanding 5 pixels in each direction
    mask_dilated[10:30, 10:30] = True

    # Create the annulus mask
    mask_annulus = mask_dilated & ~mask_original

    # Extract the intensities in the annulus
    annulus_intensities = test_image[mask_annulus]

    # Get the percentile value from the parameters
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'

    # Calculate expected percentile value
    expected_percentile = np.percentile(annulus_intensities, percentile)

    # Verify the computed metrics match our expectations
    assert prop_name in property_values
    assert property_values[prop_name] == pytest.approx(expected_percentile, rel=0.2)


def test_multiple_annotations_percentile(mock_worker_client, mock_dataset_client, sample_params):
    """Test processing multiple annotations with different shapes"""
    # Create a test image with a gradient
    test_image = np.zeros((40, 40), dtype=np.uint8)
    for i in range(40):
        for j in range(40):
            test_image[i, j] = (i + j) * 2  # Simple gradient

    # Create multiple test annotations with different shapes
    square_annotation = {
        '_id': 'test_square',
        'coordinates': [
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 20},
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 10},
            {'x': 10, 'y': 10}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    rectangle_annotation = {
        '_id': 'test_rectangle',
        'coordinates': [
            {'x': 25, 'y': 10},
            {'x': 25, 'y': 30},
            {'x': 30, 'y': 30},
            {'x': 30, 'y': 10},
            {'x': 25, 'y': 10}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Set up mock to return our test annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        square_annotation, rectangle_annotation
    ]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics for both annotations
    property_values = calls[0][0][0]['test_dataset']
    assert len(property_values) == 2
    assert 'test_square' in property_values
    assert 'test_rectangle' in property_values

    # Calculate expected values for the square annotation
    # Create a mask for the original square
    square_mask_original = np.zeros_like(test_image, dtype=bool)
    square_mask_original[10:20, 10:20] = True

    # Create a mask for the dilated square (with radius 5)
    square_mask_dilated = np.zeros_like(test_image, dtype=bool)
    # Approximate dilation by expanding 5 pixels in each direction
    square_mask_dilated[5:25, 5:25] = True

    # Create the annulus mask
    square_mask_annulus = square_mask_dilated & ~square_mask_original

    # Extract the intensities in the annulus
    square_annulus_intensities = test_image[square_mask_annulus]

    # Calculate expected values for the rectangle annotation
    # Create a mask for the original rectangle
    rect_mask_original = np.zeros_like(test_image, dtype=bool)
    rect_mask_original[10:30, 25:30] = True

    # Create a mask for the dilated rectangle (with radius 5)
    rect_mask_dilated = np.zeros_like(test_image, dtype=bool)
    # Approximate dilation by expanding 5 pixels in each direction
    rect_mask_dilated[5:35, 20:35] = True

    # Create the annulus mask
    rect_mask_annulus = rect_mask_dilated & ~rect_mask_original

    # Extract the intensities in the annulus
    rect_annulus_intensities = test_image[rect_mask_annulus]

    # Get the percentile value from the parameters
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'

    # Calculate expected percentile values
    square_expected_percentile = np.percentile(square_annulus_intensities, percentile)
    rect_expected_percentile = np.percentile(rect_annulus_intensities, percentile)

    # Verify square metrics
    square_values = property_values['test_square']
    assert prop_name in square_values
    assert square_values[prop_name] == pytest.approx(square_expected_percentile, rel=0.2)

    # Verify rectangle metrics
    rectangle_values = property_values['test_rectangle']
    assert prop_name in rectangle_values
    assert rectangle_values[prop_name] == pytest.approx(rect_expected_percentile, rel=0.2)


def test_edge_cases(mock_worker_client, mock_dataset_client, sample_params):
    """Test edge cases like empty regions or annotations outside image bounds"""
    # Create a test image
    test_image = np.ones((30, 30), dtype=np.uint8) * 100

    # For the annulus intensity worker, even a small annotation can have a valid annulus around it
    # So we'll test with an annotation outside image bounds instead
    outside_annotation = {
        '_id': 'outside_annotation',
        'coordinates': [
            {'x': 35, 'y': 35},
            {'x': 35, 'y': 45},
            {'x': 45, 'y': 45},
            {'x': 45, 'y': 35},
            {'x': 35, 'y': 35}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Create a valid annotation for comparison
    valid_annotation = {
        '_id': 'valid_annotation',
        'coordinates': [
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 20},
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 15},
            {'x': 15, 'y': 15}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Set up mock to return our test annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        outside_annotation, valid_annotation
    ]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']

    # Only the valid annotation should have metrics
    assert 'valid_annotation' in property_values
    assert 'outside_annotation' not in property_values

    # Verify valid annotation metrics
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'
    valid_values = property_values['valid_annotation']
    assert prop_name in valid_values
    assert valid_values[prop_name] == pytest.approx(100.0)


def test_different_percentiles(mock_worker_client, mock_dataset_client, sample_params):
    """Test different percentile values"""
    # Create a test image with a range of values
    test_image = np.arange(0, 100, dtype=np.uint8).reshape(10, 10)

    # Create a test annotation covering part of the image
    test_annotation = {
        '_id': 'test_percentile',
        'coordinates': [
            {'x': 2, 'y': 2},
            {'x': 2, 'y': 8},
            {'x': 8, 'y': 8},
            {'x': 8, 'y': 2},
            {'x': 2, 'y': 2}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Create a mask for the original square
    mask_original = np.zeros_like(test_image, dtype=bool)
    mask_original[2:8, 2:8] = True

    # Create a mask for the dilated square (with radius 2)
    mask_dilated = np.zeros_like(test_image, dtype=bool)
    # Approximate dilation by expanding 2 pixels in each direction
    mask_dilated[0:10, 0:10] = True  # This covers the whole image

    # Create the annulus mask
    mask_annulus = mask_dilated & ~mask_original

    # Extract the intensities in the annulus
    annulus_intensities = test_image[mask_annulus]

    # Test different percentile values
    percentiles_to_test = [0.0, 25.0, 50.0, 75.0, 90.0, 99.0]

    for percentile in percentiles_to_test:
        # Update the percentile parameter
        sample_params['workerInterface']['Percentile'] = percentile

        # Run computation
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Get the property values that were sent to the server
        calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
        assert len(calls) > 0

        # Get the computed metrics
        property_values = calls[-1][0][0]['test_dataset']['test_percentile']

        # Calculate expected percentile
        expected_percentile = np.percentile(annulus_intensities, percentile)
        prop_name = f'{percentile}thPercentileIntensity'

        # Verify the computed metrics match our expectations
        assert prop_name in property_values
        assert property_values[prop_name] == pytest.approx(expected_percentile, rel=0.2)


def test_different_radius_values(mock_worker_client, mock_dataset_client, sample_params):
    """Test the effect of different radius values on the annulus calculation"""
    # Create a test image with a stronger gradient that will result in different percentile values
    test_image = np.zeros((50, 50), dtype=np.uint8)
    for i in range(50):
        for j in range(50):
            # Create a radial gradient centered at (25, 25)
            distance = np.sqrt((i - 25)**2 + (j - 25)**2)
            test_image[i, j] = max(0, min(255, int(255 - distance * 5)))

    # Create a test annotation (10x10 square in the middle)
    test_annotation = {
        '_id': 'test_square',
        'coordinates': [
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 30},
            {'x': 30, 'y': 30},
            {'x': 30, 'y': 20},
            {'x': 20, 'y': 20}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Calculate expected values for radius = 2 and radius = 10
    # Create a mask for the original square
    mask_original = np.zeros_like(test_image, dtype=bool)
    mask_original[20:30, 20:30] = True

    # Create masks for the dilated squares with different radii
    mask_dilated_r2 = np.zeros_like(test_image, dtype=bool)
    mask_dilated_r2[18:32, 18:32] = True  # Approximate dilation by expanding 2 pixels

    mask_dilated_r10 = np.zeros_like(test_image, dtype=bool)
    mask_dilated_r10[10:40, 10:40] = True  # Approximate dilation by expanding 10 pixels

    # Create the annulus masks
    mask_annulus_r2 = mask_dilated_r2 & ~mask_original
    mask_annulus_r10 = mask_dilated_r10 & ~mask_original

    # Extract the intensities in the annuli
    annulus_intensities_r2 = test_image[mask_annulus_r2]
    annulus_intensities_r10 = test_image[mask_annulus_r10]

    # Get the percentile value from the parameters
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'

    # Calculate expected percentile values
    expected_percentile_r2 = np.percentile(annulus_intensities_r2, percentile)
    expected_percentile_r10 = np.percentile(annulus_intensities_r10, percentile)

    # Test with radius = 2
    sample_params['workerInterface']['Radius'] = 2
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    calls_r2 = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values_r2 = calls_r2[-1][0][0]['test_dataset']['test_square']

    # Reset the mock to clear the call history
    mock_worker_client.add_multiple_annotation_property_values.reset_mock()

    # Test with radius = 10
    sample_params['workerInterface']['Radius'] = 10
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    calls_r10 = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values_r10 = calls_r10[-1][0][0]['test_dataset']['test_square']

    # Verify the results for radius = 2
    assert property_values_r2[prop_name] == pytest.approx(expected_percentile_r2, rel=0.2)

    # Verify the results for radius = 10
    assert property_values_r10[prop_name] == pytest.approx(expected_percentile_r10, rel=0.2)

    # With a radial gradient, the percentile values should be different for different radius values
    assert property_values_r10[prop_name] != property_values_r2[prop_name]
