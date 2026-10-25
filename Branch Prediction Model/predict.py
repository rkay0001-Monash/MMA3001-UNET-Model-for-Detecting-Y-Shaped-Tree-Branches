"""
Inference script for the V3 tree-branch U-Net model.

Architecture based on Milesial's Pytorch-UNet:
https://github.com/milesial/Pytorch-UNet

The original implementation is distributed under GPL-3.0. See the
repository's LICENSE file. This script is project-specific inference code.

Expected project layout:
    predict.py
    unet/
        __init__.py
        unet_model.py
        unet_parts.py
    weights/
        best_model_UNET_Combined_V3_Public_Summer.pth

Example:
    python predict.py --rgb path/to/rgb.png --depth path/to/depth.png

Preprocessing matches the seasonal RGB/depth pipeline in the V3 training
notebook:
- Read RGB as RGB and depth as grayscale using OpenCV.
- Require RGB and depth to have the same original dimensions.
- Resize RGB to 256 x 256 using cv2.INTER_AREA.
- Resize depth to 256 x 256 using cv2.INTER_NEAREST.
- Divide depth values by the maximum depth value (soft_nonzero mode).
- Multiply RGB by the depth mask, clip to [0, 255], and cast to uint8.
- Convert masked RGB to float32 in [0, 1] for the U-Net.

Note: the V3 notebook raises an error if a depth image contains no positive
pixels. This script does the same.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from unet import UNet


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "weights"
    / "trained_weights.pth"
)
DEFAULT_SIZE = 256
DEFAULT_THRESHOLD = 0.5


def read_rgb(path: Path) -> np.ndarray:
    """Read an image as an RGB uint8 array, matching the training notebook."""
    if not path.is_file():
        raise FileNotFoundError(f"RGB image was not found: {path}")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth(path: Path) -> np.ndarray:
    """Read depth as grayscale, matching the training notebook."""
    if not path.is_file():
        raise FileNotFoundError(f"Depth image was not found: {path}")
    depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise ValueError(f"Cannot read depth image: {path}")
    return depth


def preprocess_pair(
    rgb: np.ndarray,
    depth: np.ndarray,
    size: int = DEFAULT_SIZE,
) -> torch.Tensor:
    """
    Apply the training notebook's RGB/depth preprocessing to one image pair.

    Both images are cropped using the same bottom-centred square crop.
    RGB is resized with area interpolation, while depth uses nearest-neighbour
    interpolation. Normalised depth is then used as a soft mask on RGB.

    Args:
        rgb: RGB image as a NumPy array with shape (H, W, 3).
        depth: Grayscale depth image as a NumPy array with shape (H, W).
        size: Desired square input size for the U-Net.

    Returns:
        A float32 tensor of shape (1, 3, size, size), with values in [0, 1].

    Raises:
        ValueError: If size is invalid, the images have invalid shapes,
            dimensions do not match, or depth contains only zeros.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with 3 channels; got {rgb.shape}"
        )

    if depth.ndim != 2:
        raise ValueError(
            f"Expected single-channel depth image; got {depth.shape}"
        )

    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"RGB and depth dimensions must match; "
            f"got RGB {rgb.shape[:2]} and depth {depth.shape[:2]}"
        )

    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("RGB image cannot be empty")

    if depth.shape[0] == 0 or depth.shape[1] == 0:
        raise ValueError("Depth image cannot be empty")

    if not np.any(depth):
        raise ValueError("Depth image contains only zeros")

    # Apply the same bottom-centred square crop to both images.
    height, width = rgb.shape[:2]
    side = min(height, width)
    x_start = (width - side) // 2
    y_start = height - side

    rgb_crop = rgb[
        y_start:y_start + side,
        x_start:x_start + side
    ].copy()

    depth_crop = depth[
        y_start:y_start + side,
        x_start:x_start + side
    ].copy()

    # Resize using the interpolation methods from the training notebook.
    rgb_resized = cv2.resize(
        rgb_crop,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )

    depth_resized = cv2.resize(
        depth_crop,
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )

    # Normalise depth and use it as a soft RGB mask.
    depth_norm = depth_resized.astype(np.float32)
    depth_norm /= float(depth_norm.max())

    masked_rgb = np.clip(
        rgb_resized.astype(np.float32) * depth_norm[..., None],
        0,
        255,
    ).astype(np.uint8)

    # Convert RGB from HWC to CHW, normalise, and add a batch dimension.
    input_tensor = torch.from_numpy(
        (masked_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    ).unsqueeze(0)

    return input_tensor
