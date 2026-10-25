import numpy as np
import pytest
import torch

from predict import preprocess_pair, DEFAULT_SIZE


def make_rgb(height=480, width=640):
    """Create a synthetic RGB uint8 image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_depth(height=480, width=640):
    """Create a test depth map containing valid nonzero values."""
    return np.full((height, width), 128, dtype=np.uint8)


def test_preprocess_pair_output_shape():
    """Preprocessing should return a batch of three-channel images."""
    result = preprocess_pair(make_rgb(), make_depth())

    assert tuple(result.shape) == (1, 3, DEFAULT_SIZE, DEFAULT_SIZE)


def test_preprocess_pair_output_dtype():
    """Preprocessing should return a float32 tensor."""
    result = preprocess_pair(make_rgb(), make_depth())

    assert result.dtype == torch.float32


def test_preprocess_pair_output_range():
    """Preprocessed pixel values should lie between zero and one."""
    result = preprocess_pair(make_rgb(), make_depth())

    assert result.min().item() >= 0.0
    assert result.max().item() <= 1.0


def test_preprocess_pair_custom_size():
    """A positive custom output size should be accepted."""
    result = preprocess_pair(make_rgb(), make_depth(), size=128)

    assert tuple(result.shape) == (1, 3, 128, 128)


@pytest.mark.parametrize("size", [0, -1, -256])
def test_preprocess_pair_rejects_nonpositive_size(size):
    """A zero or negative output size should raise ValueError."""
    with pytest.raises(ValueError, match="size must be a positive integer"):
        preprocess_pair(make_rgb(), make_depth(), size=size)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((480, 640), dtype=np.uint8),
        np.zeros((480, 640, 1), dtype=np.uint8),
        np.zeros((480, 640, 4), dtype=np.uint8),
    ],
)
def test_preprocess_pair_rejects_invalid_rgb(rgb):
    """RGB input must be a three-dimensional, three-channel array."""
    with pytest.raises(ValueError, match="Expected RGB image with 3 channels"):
        preprocess_pair(rgb, make_depth())


@pytest.mark.parametrize(
    "depth",
    [
        np.zeros((480, 640, 1), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
    ],
)
def test_preprocess_pair_rejects_invalid_depth(depth):
    """Depth input must be a two-dimensional grayscale array."""
    with pytest.raises(ValueError, match="Expected single-channel depth image"):
        preprocess_pair(make_rgb(), depth)


def test_preprocess_pair_rejects_mismatched_dimensions():
    """RGB and depth images must have matching height and width."""
    rgb = make_rgb(480, 640)
    depth = make_depth(400, 640)

    with pytest.raises(ValueError, match="RGB and depth dimensions must match"):
        preprocess_pair(rgb, depth)
