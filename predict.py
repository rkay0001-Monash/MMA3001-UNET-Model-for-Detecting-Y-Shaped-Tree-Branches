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
    python predict.py \
        --rgb path/to/rgb.png \
        --depth path/to/depth.png \
        --checkpoint weights/best_model_UNET_Combined_V3_Public_Summer.pth \
        --output-dir predictions

Preprocessing assumption:
    RGB and depth are resized to 256 x 256. Depth is converted to grayscale,
    scaled to [0, 1] by dividing by its maximum value, and multiplied into the
    RGB image as a soft mask. This is based on the documented V3 configuration
    (FINAL_SIZE=256, DEPTH_MASK_MODE="soft_nonzero"). Confirm this matches the
    exact preprocessing function in the training notebook before relying on
    quantitative comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from unet import UNet


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "weights"
    / "best_model_UNET_Combined_V3_Public_Summer.pth"
)
DEFAULT_SIZE = 256
DEFAULT_THRESHOLD = 0.5


def load_rgb(path: Path) -> Image.Image:
    """Load an image as RGB."""
    if not path.is_file():
        raise FileNotFoundError(f"RGB image was not found: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def load_depth(path: Path) -> Image.Image:
    """Load a depth image as 8-bit grayscale without contrast stretching."""
    if not path.is_file():
        raise FileNotFoundError(f"Depth image was not found: {path}")
    with Image.open(path) as image:
        # Convert to grayscale, but do not auto-normalise/contrast-stretch:
        # that could change the depth-mask values relative to training.
        return image.convert("L")


def preprocess_pair(
    rgb_image: Image.Image,
    depth_image: Image.Image,
    size: int = DEFAULT_SIZE,
) -> torch.Tensor:
    """
    Prepare a single RGB/depth pair for the V3 model.

    Returns a float tensor with shape (1, 3, size, size), with values in
    [0, 1]. RGB is multiplied by the resized, max-normalised depth mask.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer")

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    rgb_resized = rgb_image.resize((size, size), resample=resampling)
    depth_resized = depth_image.resize((size, size), resample=resampling)

    rgb_array = np.asarray(rgb_resized, dtype=np.float32) / 255.0
    depth_array = np.asarray(depth_resized, dtype=np.float32)

    max_depth = float(depth_array.max()) if depth_array.size else 0.0
    if max_depth > 0:
        depth_array = depth_array / max_depth
    else:
        # An all-zero depth image should produce an all-zero masked RGB image.
        depth_array = np.zeros_like(depth_array, dtype=np.float32)

    # Broadcast the single-channel depth mask across the three RGB channels.
    masked_rgb = rgb_array * depth_array[..., np.newaxis]
    tensor = torch.from_numpy(np.ascontiguousarray(masked_rgb))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Construct the V3 U-Net and load its model_state_dict checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}\n"
            "Pass the correct path with --checkpoint."
        )

    # weights_only is supported by recent PyTorch versions. Fall back for older
    # versions, where torch.load does not accept that keyword.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        # Also allow a raw state_dict, if the checkpoint was saved that way.
        state_dict = checkpoint
    else:
        raise ValueError(
            "Unrecognised checkpoint format. Expected a dictionary containing "
            "'model_state_dict' or a raw model state_dict."
        )

    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_predictions(
    probabilities: np.ndarray,
    rgb_original: Image.Image,
    output_dir: Path,
    stem: str,
    threshold: float,
) -> tuple[Path, Path, Path]:
    """Save probability map, binary mask, and an overlay at original RGB size."""
    output_dir.mkdir(parents=True, exist_ok=True)

    probability_path = output_dir / f"{stem}_probability.png"
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    probability_u8 = np.clip(probabilities * 255.0, 0, 255).astype(np.uint8)
    binary_u8 = (probabilities >= threshold).astype(np.uint8) * 255

    original_size = rgb_original.size  # (width, height)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    probability_image = Image.fromarray(probability_u8, mode="L").resize(
        original_size, resample=resampling
    )
    mask_image = Image.fromarray(binary_u8, mode="L").resize(
        original_size, resample=getattr(Image, "Resampling", Image).NEAREST
    )

    probability_image.save(probability_path)
    mask_image.save(mask_path)

    rgb_array = np.asarray(rgb_original, dtype=np.uint8).copy()
    mask_array = np.asarray(mask_image, dtype=np.uint8) > 0
    # Red overlay highlights predicted branch pixels.
    red = np.zeros_like(rgb_array)
    red[..., 0] = 255
    rgb_array[mask_array] = (
        0.5 * rgb_array[mask_array].astype(np.float32)
        + 0.5 * red[mask_array].astype(np.float32)
    ).astype(np.uint8)
    Image.fromarray(rgb_array, mode="RGB").save(overlay_path)

    return probability_path, mask_path, overlay_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict tree-branch masks using the V3 U-Net checkpoint."
    )
    parser.add_argument("--rgb", required=True, type=Path, help="Path to RGB image.")
    parser.add_argument(
        "--depth", required=True, type=Path, help="Path to matching depth image."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions"),
        help="Directory for output images (default: predictions/).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Probability threshold for the binary mask (default: 0.5).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Square model input size (default: 256).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: automatically choose CUDA when available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rgb_image = load_rgb(args.rgb)
    depth_image = load_depth(args.depth)

    if rgb_image.size != depth_image.size:
        print(
            "Warning: RGB and depth dimensions differ "
            f"(RGB={rgb_image.size}, depth={depth_image.size}). "
            "Both will be resized to the model input size. Make sure the images "
            "are spatially aligned and depict the same scene."
        )

    model = load_model(args.checkpoint, device)
    input_tensor = preprocess_pair(rgb_image, depth_image, size=args.size).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    probability_path, mask_path, overlay_path = save_predictions(
        probabilities=probabilities,
        rgb_original=rgb_image,
        output_dir=args.output_dir,
        stem=args.rgb.stem,
        threshold=args.threshold,
    )

    print(f"Device: {device}")
    print(f"Probability map: {probability_path}")
    print(f"Binary mask:    {mask_path}")
    print(f"Overlay:        {overlay_path}")


if __name__ == "__main__":
    main()
