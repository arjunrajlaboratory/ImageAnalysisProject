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
        'image': 'properties/blob_annulus_intensity:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Channel': 0,
            'Radius': 5  # Add the radius parameter
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


def test_worker_startup(mock_worker_client, mock_dataset_client, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Run computation with empty annotation list
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify that get_annotation_list_by_shape was called
    mock_worker_client.get_annotation_list_by_shape.assert_called_once_with(
        'polygon', limit=0)

    # Since there are no annotations, add_multiple_annotation_property_values should not be called
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_uniform_intensity_calculation(mock_worker_client, mock_dataset_client, sample_params):
    """Test intensity calculations with a uniform intensity image"""
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

    # For a uniform image with all pixels = 50, all metrics should be 50
    # except total intensity which is 50 * number of pixels in the annulus
    assert property_values['MeanIntensity'] == pytest.approx(50.0)
    assert property_values['MaxIntensity'] == pytest.approx(50.0)
    assert property_values['MinIntensity'] == pytest.approx(50.0)
    assert property_values['MedianIntensity'] == pytest.approx(50.0)
    assert property_values['25thPercentileIntensity'] == pytest.approx(50.0)
    assert property_values['75thPercentileIntensity'] == pytest.approx(50.0)

    # The annulus around a 10x10 square with radius 5 will have more pixels than the original square
    # We don't need to calculate the exact number, just verify it's not zero
    assert property_values['TotalIntensity'] > 0


def test_gradient_intensity_calculation(mock_worker_client, mock_dataset_client, sample_params):
    """Test intensity calculations with a gradient intensity image"""
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

    # Calculate expected values
    expected_mean = np.mean(annulus_intensities)
    expected_max = np.max(annulus_intensities)
    expected_min = np.min(annulus_intensities)
    expected_median = np.median(annulus_intensities)
    expected_q25 = np.percentile(annulus_intensities, 25)
    expected_q75 = np.percentile(annulus_intensities, 75)
    expected_total = np.sum(annulus_intensities)

    # Verify the computed metrics are close to our expectations
    # We use a larger tolerance because our approximation of the annulus is not exact
    assert property_values['MeanIntensity'] == pytest.approx(expected_mean, rel=0.2)
    assert property_values['MaxIntensity'] == pytest.approx(expected_max, rel=0.2)
    assert property_values['MinIntensity'] == pytest.approx(expected_min, rel=0.2)
    assert property_values['MedianIntensity'] == pytest.approx(expected_median, rel=0.2)
    assert property_values['25thPercentileIntensity'] == pytest.approx(expected_q25, rel=0.2)
    assert property_values['75thPercentileIntensity'] == pytest.approx(expected_q75, rel=0.2)
    assert property_values['TotalIntensity'] == pytest.approx(expected_total, rel=0.2)


def test_multiple_annotations(mock_worker_client, mock_dataset_client, sample_params):
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

    # Calculate expected values for the square
    square_expected_mean = np.mean(square_annulus_intensities)
    square_expected_max = np.max(square_annulus_intensities)
    square_expected_min = np.min(square_annulus_intensities)

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

    # Calculate expected values for the rectangle
    rect_expected_mean = np.mean(rect_annulus_intensities)
    rect_expected_max = np.max(rect_annulus_intensities)
    rect_expected_min = np.min(rect_annulus_intensities)

    # Verify square metrics
    square_values = property_values['test_square']
    assert square_values['MeanIntensity'] == pytest.approx(square_expected_mean, rel=0.2)
    assert square_values['MaxIntensity'] == pytest.approx(square_expected_max, rel=0.2)
    assert square_values['MinIntensity'] == pytest.approx(square_expected_min, rel=0.2)

    # Verify rectangle metrics
    rectangle_values = property_values['test_rectangle']
    assert rectangle_values['MeanIntensity'] == pytest.approx(rect_expected_mean, rel=0.2)
    assert rectangle_values['MaxIntensity'] == pytest.approx(rect_expected_max, rel=0.2)
    assert rectangle_values['MinIntensity'] == pytest.approx(rect_expected_min, rel=0.2)


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
    valid_values = property_values['valid_annotation']
    assert valid_values['MeanIntensity'] == pytest.approx(100.0)
    assert valid_values['MaxIntensity'] == pytest.approx(100.0)
    assert valid_values['MinIntensity'] == pytest.approx(100.0)


def test_different_radius_values(mock_worker_client, mock_dataset_client, sample_params):
    """Test the effect of different radius values on the annulus calculation"""
    # Create a test image with a stronger gradient that will result in different mean values
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

    # Calculate expected values for radius = 2
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

    # Calculate expected values
    expected_mean_r2 = np.mean(annulus_intensities_r2)
    expected_total_r2 = np.sum(annulus_intensities_r2)

    expected_mean_r10 = np.mean(annulus_intensities_r10)
    expected_total_r10 = np.sum(annulus_intensities_r10)

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
    assert property_values_r2['MeanIntensity'] == pytest.approx(expected_mean_r2, rel=0.2)
    assert property_values_r2['TotalIntensity'] == pytest.approx(expected_total_r2, rel=0.2)

    # Verify the results for radius = 10
    assert property_values_r10['MeanIntensity'] == pytest.approx(expected_mean_r10, rel=0.2)
    assert property_values_r10['TotalIntensity'] == pytest.approx(expected_total_r10, rel=0.2)

    # A larger radius should include more pixels in the annulus
    # This should result in a larger total intensity
    assert property_values_r10['TotalIntensity'] > property_values_r2['TotalIntensity']

    # With a radial gradient, the mean intensity should be different for different radius values
    assert property_values_r10['MeanIntensity'] != property_values_r2['MeanIntensity']
