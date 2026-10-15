# U-Net Model for Detecting Y-Shaped Tree Branches

An inference-only implementation of a U-Net convolutional neural network for segmenting tree branches in RGB images, using corresponding depth images to suppress background regions.

This project is based on the U-Net implementation from [Milesial's Pytorch-UNet repository](https://github.com/milesial/Pytorch-UNet) and uses the V3 model trained for the tree-branch segmentation project.

## Features

- Predict tree-branch segmentation masks using a pretrained U-Net model.
- Use paired RGB and depth images as model input.
- Apply depth-based masking to reduce the influence of background regions.
- Generate probability maps, binary segmentation masks, and visual overlays.
- Run inference on a CPU or CUDA-compatible GPU.
- Support future integration with Google Colab.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── predict.py
├── unet/
│   ├── __init__.py
│   ├── unet_model.py
│   └── unet_parts.py
├── weights/
│   └── trained_weights.pth
└── Tree_Branch_Prediction.ipynb
```

The `weights/` directory and Colab notebook may not yet be available in the repository.

## Model Information

The V3 model uses the following configuration:

| Parameter | Value |
|---|---|
| Architecture | U-Net |
| Input channels | 3 |
| Output channels | 1 |
| Input resolution | 256 × 256 |
| Bilinear upsampling | Disabled |
| Prediction threshold | 0.5 by default |
| Depth masking | `soft_nonzero` |

The model produces a single-channel output, which is converted into probabilities using a sigmoid function. These probabilities are thresholded to generate a binary segmentation mask.

## Requirements

The project requires Python and the following packages:

- NumPy
- OpenCV
- PyTorch
- Torchvision

Install the dependencies using:

```bash
pip install -r requirements.txt
```

For GPU inference, install a PyTorch version compatible with your CUDA environment. Google Colab can also be used for inference.

## Model Checkpoint

The trained model checkpoint is named:

`trained_weights.pth`

Place the checkpoint in the `weights/` directory:

```text
weights/trained_weights.pth
```

The checkpoint is approximately 373 MB and should be hosted using GitHub Releases or Git LFS rather than uploaded as a regular file through GitHub's web interface.

Once the checkpoint is hosted, download it and place it in the `weights/` directory before running predictions.

The inference script uses this location by default. Alternatively, specify a different checkpoint path using the `--checkpoint` argument.

## Input Images

Each prediction requires two corresponding images:

1. **RGB image:** the colour image containing the tree branches.
2. **Depth image:** the corresponding depth image for the same scene.

The RGB and depth images must have matching original dimensions and be spatially aligned. The depth input is read as a single-channel grayscale image by the inference script.

The current preprocessing pipeline resizes RGB images to 256 × 256 using OpenCV's `INTER_AREA` interpolation and depth images using `INTER_NEAREST`. The depth image is normalised by its maximum value to produce a soft mask, which is multiplied into the RGB image before inference.

**Important:** The depth image must use an encoding compatible with the preprocessing used during training. The current implementation assumes an 8-bit grayscale depth image; other depth encodings may require additional preprocessing.

## Running Predictions

Run the following command from the repository root:

```bash
python predict.py \
    --rgb path/to/rgb.png \
    --depth path/to/depth.png
```

To specify the checkpoint and output directory:

```bash
python predict.py \
    --rgb path/to/rgb.png \
    --depth path/to/depth.png \
    --checkpoint weights/trained_weights.pth \
    --output-dir predictions
```

To run explicitly on the CPU:

```bash
python predict.py \
    --rgb path/to/rgb.png \
    --depth path/to/depth.png \
    --device cpu
```

The script automatically selects CUDA when available unless a device is specified.

## Output Files

For an input image named `tree.png`, the script generates three files in the output directory:

| File | Description |
|---|---|
| `tree_probability.png` | Grayscale prediction probability map |
| `tree_mask.png` | Binary segmentation mask |
| `tree_overlay.png` | RGB image with predicted pixels highlighted in red |

The binary mask uses a default probability threshold of 0.5. This threshold can be adjusted using the `--threshold` argument.

## Limitations

- Predictions depend on the quality and alignment of the RGB and depth images.
- The model expects inputs processed consistently with its training pipeline.
- Depth-based masking can suppress useful image information if depth values are missing or incorrectly encoded.
- Prediction quality may vary across different trees, environments, lighting conditions, and levels of branch occlusion.
- The model has not been independently validated for every possible input environment.

## Acknowledgements and Licensing

This project builds on the U-Net implementation from [Milesial's Pytorch-UNet repository](https://github.com/milesial/Pytorch-UNet).

The original implementation is distributed under the GNU General Public License v3.0 (GPL-3.0). See the `LICENSE` file for the applicable licence terms. Copyright notices and licence terms for reused source code should be retained.

Project-specific preprocessing, inference code, and model configuration are intended for tree-branch segmentation.

The provenance and redistribution terms of the trained checkpoint and any datasets should be checked separately before redistribution.

## Disclaimer

This project is intended for research and educational purposes. No guarantee is made regarding the accuracy or suitability of predictions for any particular application.
