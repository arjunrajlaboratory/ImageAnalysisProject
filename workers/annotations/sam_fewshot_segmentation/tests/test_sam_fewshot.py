import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from entrypoint import (
    extract_crop_with_context,
    pool_features_with_mask,
    ensure_rgb,
    annotation_to_mask,
    interface,
)


class TestExtractCropWithContext:
    """Tests for the context-aware crop extraction."""

    def test_basic_crop_centered(self):
        """Test that crop is centered on the object."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Place a 20x20 object in the center
        mask[90:110, 90:110] = 1
        image[90:110, 90:110] = 128

        crop_image, crop_mask = extract_crop_with_context(image, mask, target_occupancy=0.20)

        # Object should still be present in the crop
        assert crop_mask.sum() > 0
        # Crop should be larger than the object itself
        assert crop_image.shape[0] >= 20
        assert crop_image.shape[1] >= 20

    def test_small_object_gets_more_context(self):
        """A small object should get a proportionally larger crop."""
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        mask = np.zeros((500, 500), dtype=np.uint8)
        # Small 10x10 object
        mask[245:255, 245:255] = 1

        crop_image, crop_mask = extract_crop_with_context(image, mask, target_occupancy=0.20)

        # With 100 pixels of object area and 0.20 occupancy,
        # crop area should be ~500, so side ~22
        obj_area = 100
        expected_crop_area = obj_area / 0.20
        expected_side = int(np.sqrt(expected_crop_area))
        assert crop_image.shape[0] >= expected_side - 2  # Allow small margin

    def test_object_at_edge(self):
        """Object near image edge should still produce valid crop."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Object at top-left corner
        mask[0:10, 0:10] = 1

        crop_image, crop_mask = extract_crop_with_context(image, mask, target_occupancy=0.20)

        # Should not crash and mask should be preserved
        assert crop_mask.sum() > 0
        assert crop_image.shape[0] > 0
        assert crop_image.shape[1] > 0

    def test_empty_mask_returns_original(self):
        """Empty mask should return the original image and mask."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        crop_image, crop_mask = extract_crop_with_context(image, mask, target_occupancy=0.20)

        assert np.array_equal(crop_image, image)
        assert np.array_equal(crop_mask, mask)

    def test_large_object_respects_bounding_box(self):
        """Crop should be at least as large as the object bounding box."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Large 80x80 object
        mask[60:140, 60:140] = 1

        crop_image, crop_mask = extract_crop_with_context(image, mask, target_occupancy=0.20)

        # Crop must encompass the full object
        assert crop_image.shape[0] >= 80
        assert crop_image.shape[1] >= 80
        # And the mask pixels should all be within the crop
        assert crop_mask.sum() == mask.sum()


class TestPoolFeaturesWithMask:
    """Tests for the weighted feature pooling."""

    def test_basic_pooling(self):
        """Pooling with full mask should equal global average."""
        import torch

        C, H, W = 32, 8, 8
        features = torch.ones(1, C, H, W)
        mask = np.ones((H, W), dtype=np.float32)

        result = pool_features_with_mask(features, mask, H, W)

        assert result.shape == (C,)
        # With all-ones features and all-ones mask, result should be all ones
        assert torch.allclose(result, torch.ones(C), atol=1e-3)

    def test_masked_region_pooling(self):
        """Pooling should focus on masked region."""
        import torch

        C, H, W = 16, 8, 8
        features = torch.zeros(1, C, H, W)
        # Set top-left quadrant to 1.0
        features[:, :, :4, :4] = 1.0

        # Mask only the top-left quadrant
        mask = np.zeros((H, W), dtype=np.float32)
        mask[:4, :4] = 1.0

        result = pool_features_with_mask(features, mask, H, W)

        # Should be close to 1.0 since we're pooling from the region with value 1
        assert torch.allclose(result, torch.ones(C), atol=0.2)

    def test_empty_mask_fallback(self):
        """Empty mask should fall back to global average pooling."""
        import torch

        C, H, W = 16, 8, 8
        features = torch.ones(1, C, H, W) * 3.0
        mask = np.zeros((H, W), dtype=np.float32)

        result = pool_features_with_mask(features, mask, H, W)

        # Should fall back to mean pooling
        assert result.shape == (C,)
        assert torch.allclose(result, torch.ones(C) * 3.0, atol=1e-3)

    def test_mask_upscaling(self):
        """Test that mask is properly resized to match feature dimensions."""
        import torch

        C = 16
        feat_h, feat_w = 8, 8
        features = torch.ones(1, C, feat_h, feat_w)

        # Mask at different resolution than features
        mask = np.ones((32, 32), dtype=np.float32)

        result = pool_features_with_mask(features, mask, feat_h, feat_w)

        assert result.shape == (C,)
        assert torch.allclose(result, torch.ones(C), atol=1e-3)


class TestEnsureRgb:
    """Tests for image format normalization."""

    def test_grayscale_to_rgb(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        result = ensure_rgb(image)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_single_channel_to_rgb(self):
        image = np.zeros((100, 100, 1), dtype=np.uint8)
        result = ensure_rgb(image)
        assert result.shape == (100, 100, 3)

    def test_rgba_to_rgb(self):
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        result = ensure_rgb(image)
        assert result.shape == (100, 100, 3)

    def test_float_0_1_to_uint8(self):
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        result = ensure_rgb(image)
        assert result.dtype == np.uint8
        assert result.max() == 127 or result.max() == 128  # rounding

    def test_float_0_255_to_uint8(self):
        image = np.ones((100, 100, 3), dtype=np.float32) * 200.0
        result = ensure_rgb(image)
        assert result.dtype == np.uint8
        assert result.max() == 200

    def test_uint16_to_uint8(self):
        image = np.ones((100, 100, 3), dtype=np.uint16) * 512
        result = ensure_rgb(image)
        assert result.dtype == np.uint8
        assert result.max() == 2  # 512 / 256 = 2

    def test_rgb_uint8_passthrough(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 42
        result = ensure_rgb(image)
        assert result.dtype == np.uint8
        assert np.array_equal(result, image)


class TestAnnotationToMask:
    """Tests for converting polygon annotations to binary masks."""

    def test_square_annotation(self):
        annotation = {
            'coordinates': [
                {'x': 10, 'y': 10},
                {'x': 10, 'y': 20},
                {'x': 20, 'y': 20},
                {'x': 20, 'y': 10},
            ]
        }
        mask = annotation_to_mask(annotation, (30, 30))
        assert mask.shape == (30, 30)
        assert mask.sum() > 0
        # Center of the square should be 1
        assert mask[15, 15] == 1
        # Outside should be 0
        assert mask[0, 0] == 0

    def test_mask_matches_image_shape(self):
        annotation = {
            'coordinates': [
                {'x': 5, 'y': 5},
                {'x': 5, 'y': 15},
                {'x': 15, 'y': 15},
                {'x': 15, 'y': 5},
            ]
        }
        mask = annotation_to_mask(annotation, (100, 200))
        assert mask.shape == (100, 200)


class TestInterface:
    """Test the interface function."""

    @patch('annotation_client.workers.UPennContrastWorkerPreviewClient')
    def test_interface_sets_all_fields(self, mock_client_class):
        mock_client = mock_client_class.return_value

        interface('test_image', 'http://test-api', 'test-token')

        mock_client.setWorkerImageInterface.assert_called_once()
        interface_data = mock_client.setWorkerImageInterface.call_args[0][1]

        # Verify all expected fields are present
        expected_fields = [
            'Training Tag', 'Batch XY', 'Batch Z', 'Batch Time',
            'Model', 'Similarity Threshold', 'Target Occupancy',
            'Points per side', 'Min Mask Area', 'Max Mask Area', 'Smoothing',
        ]
        for field in expected_fields:
            assert field in interface_data, f"Missing interface field: {field}"

        # Verify types
        assert interface_data['Training Tag']['type'] == 'tags'
        assert interface_data['Model']['type'] == 'select'
        assert interface_data['Similarity Threshold']['type'] == 'number'
        assert interface_data['Target Occupancy']['type'] == 'number'
        assert interface_data['Points per side']['type'] == 'number'
        assert interface_data['Smoothing']['type'] == 'number'

        # Verify SAM1-specific defaults
        assert interface_data['Similarity Threshold']['default'] == 0.5
        assert interface_data['Target Occupancy']['default'] == 0.20
        assert interface_data['Points per side']['default'] == 128
        assert interface_data['Points per side']['max'] == 128
        assert interface_data['Min Mask Area']['default'] == 30
        assert interface_data['Model']['default'] == 'sam_vit_h_4b8939'
        assert interface_data['Model']['items'] == ['sam_vit_h_4b8939']
