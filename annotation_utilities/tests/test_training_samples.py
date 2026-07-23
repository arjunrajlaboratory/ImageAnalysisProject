import numpy as np
import pytest

from annotation_utilities.annotation_tools import filter_usable_training_samples


def test_filter_usable_training_samples_drops_empty_and_unlabeled_samples():
    images = [
        np.ones((4, 4, 1)),
        np.ones((4, 4, 1)),
        np.empty((0, 4, 1)),
    ]
    labels = [
        np.pad(np.ones((1, 1), dtype=np.uint16), ((0, 3), (0, 3))),
        np.zeros((4, 4), dtype=np.uint16),
        np.empty((0, 4), dtype=np.uint16),
    ]

    usable_images, usable_labels, dropped = filter_usable_training_samples(
        images, labels)

    assert len(usable_images) == 1
    assert usable_images[0] is images[0]
    assert usable_labels[0] is labels[0]
    assert dropped == 2


def test_filter_usable_training_samples_rejects_mismatched_lists():
    with pytest.raises(ValueError, match='same length'):
        filter_usable_training_samples([np.ones((2, 2))], [])
