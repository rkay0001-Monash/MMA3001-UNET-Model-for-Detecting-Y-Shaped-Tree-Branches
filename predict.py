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
    Apply the V3 seasonal RGB/depth preprocessing to one image pair.

    Returns a tensor of shape (1, 3, size, size), float32 in [0, 1].
    """
    if size <= 0:
        raise ValueError("size must be a positive integer")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with 3 channels; got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Expected single-channel depth image; got {depth.shape}")

    # The training notebook explicitly rejects misaligned source dimensions.
    if rgb.shape[:2] != depth.shape[:2]:
