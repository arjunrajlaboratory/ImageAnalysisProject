import pytest
from unittest.mock import patch

# Import your worker module
# Assuming your entrypoint.py is in the same directory
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
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'test_metrics',
        'image': 'properties/blob_metrics:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'polygon',
        'workerInterface': {
            'Test checkbox': False,
            'Test select': 'Item 1'
        },
        'scales': {
            'pixelSize': {'unit': 'mm', 'value': 0.000219080212825376},
            'tStep': {'unit': 's', 'value': 1},
            'zStep': {'unit': 'm', 'value': 1}
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
        assert 'Test worker' in interface_data
        assert 'Test checkbox' in interface_data
        assert 'Test select' in interface_data

        # Verify specific interface properties
        assert interface_data['Test worker']['type'] == 'notes'
        assert interface_data['Test checkbox']['type'] == 'checkbox'
        assert interface_data['Test select']['type'] == 'select'
        assert interface_data['Test select']['items'] == ['Item 1', 'Item 2', 'Item 3']


def test_error_handling(mock_worker_client, sample_params, capsys):
    """Test handling of various error conditions"""
    # Run the compute function
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Capture the stdout output
    captured = capsys.readouterr()

    # Check if the warning and error messages are in the output
    assert '"warning": "This is a warning"' in captured.out
    assert '"info": "This is an info message."' in captured.out
    assert '"error": "This is an error"' in captured.out
