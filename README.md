# U-Net Model for Detecting Y-Shaped Tree Branches

An inference-only implementation of a U-Net convolutional neural network for segmenting tree branches in RGB images, using corresponding depth images to suppress background regions.

This project is based on the U-Net implementation from [Milesial's Pytorch-UNet repository](https://github.com/milesial/Pytorch-UNet) and uses the V3 model trained for the tree-branch segmentation project.

The repository also includes a Google Colab notebook, `Tree_Branch_Prediction.ipynb`, which extends the branch-segmentation workflow with apple segmentation using YOLO11 and visualisation of the combined predictions.

## Important Notice — Dataset Access

**The dataset used by the Google Colab notebook is not publicly accessible.** The original images are stored in a private Google Drive folder, and users must obtain permission from the dataset owner to access them.

Users without access to the original dataset must provide their own suitable images and modify the notebook's dataset-loading configuration as necessary. Changing the dataset path alone may not be sufficient, as the notebook relies on specific filename conventions, image modalities, preprocessing steps, and compatible trained models.

## Features

### U-Net Branch Segmentation

- Predict tree-branch segmentation masks using a pretrained U-Net model.
- Use paired RGB and depth images as model input.
- Apply depth-based masking to reduce the influence of background regions.
- Generate probability maps, binary segmentation masks, and visual overlays.
- Run inference on a CPU or CUDA-compatible GPU.

### Google Colab and Combined Predictions

- Run the prediction workflow in Google Colab.
- Compare Summer and Winter images of matching trees when corresponding images are available.
- Perform apple segmentation using a separately configured YOLO11 model.
- Display RGB images, binary apple masks, and apple-mask overlays.
- Visualise combined branch and apple predictions, with branches highlighted in red, apples in green, and overlapping predictions in yellow.
- Use a Google Colab GPU runtime, preferably an NVIDIA T4 GPU when available, to reduce inference time.

The standalone `predict.py` script focuses on U-Net branch segmentation. The combined branch-and-apple workflow is provided by the Google Colab notebook and requires the relevant YOLO11 model and weights.

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

The exact files available in the repository may vary. The trained checkpoint may be hosted separately rather than committed directly to the repository. The private dataset and generated prediction outputs are not intended to be included.

## Model Information

The V3 U-Net model uses the following configuration:

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

The standalone U-Net prediction script requires Python and the following packages:

- NumPy
- OpenCV
- PyTorch
- Torchvision

Install the dependencies using:

```bash
pip install -r requirements.txt
```

The Google Colab notebook may require additional packages for image visualisation, dataset processing, and YOLO11 inference. Ensure that the notebook's dependency-installation cells and `requirements.txt` cover the packages used by the current implementation.

For GPU inference, install a PyTorch version compatible with the selected CUDA environment. Google Colab can be used for inference, and a T4 GPU runtime is recommended when available.

## Model Checkpoint

The U-Net trained model checkpoint is named:

`trained_weights.pth`

The standalone prediction script expects the checkpoint at:

```text
weights/trained_weights.pth
```

The checkpoint is approximately 373 MB and should be hosted using GitHub Releases or Git LFS rather than uploaded as a regular file through GitHub's web interface.

Once the checkpoint is hosted, download it and place it in the `weights/` directory before running predictions.

The inference script uses this location by default. Alternatively, specify a different checkpoint path using the `--checkpoint` argument.

The Google Colab notebook may download or load model weights separately. Users should follow the notebook's setup instructions and ensure that the relevant download links are accessible. The U-Net checkpoint and the YOLO11 model weights are separate model assets and should not be assumed to be interchangeable.

## Input Images

### Standalone U-Net Prediction Script

Each prediction requires two corresponding images:

1. **RGB image:** the colour image containing the tree branches.
2. **Depth image:** the corresponding depth image for the same scene.

The RGB and depth images must have matching original dimensions and be spatially aligned. The depth input is read as a single-channel grayscale image by the inference script.

The current preprocessing pipeline resizes RGB images to 256 × 256 using OpenCV's `INTER_AREA` interpolation and depth images using `INTER_NEAREST`. The depth image is normalised by its maximum value to produce a soft mask, which is multiplied into the RGB image before inference.

**Important:** The depth image must use an encoding compatible with the preprocessing used during training. The current implementation assumes an 8-bit grayscale depth image; other depth encodings may require additional preprocessing.

### Google Colab Notebook Dataset

The notebook uses a dataset containing RGB and depth images for tree-branch segmentation. The original dataset is stored in a private Google Drive folder.

The current workflow identifies seasons, image modalities, and tree identifiers from filenames. Examples include:

```text
R21U17_Winter_RGB-13-30-50.png
R21U17_Winter_D-13-30-50.png
R21U26_Summer_RGB-14-26-37.png
```

The notebook uses filename information to identify image types and pair corresponding RGB and depth images. It also uses tree identifiers to match trees across Summer and Winter datasets.

Users adapting the notebook to another dataset may need to modify:

- The Google Drive mounting and dataset-loading configuration.
- Filename parsing and RGB/depth image-pairing logic.
- Summer/Winter classification and tree-matching logic.
- Image cropping, resizing, normalisation, and depth-based masking.
- The U-Net input pipeline and trained weights.

The dataset should not be assumed to be compatible with the model simply because it contains tree images.

## Running Predictions

### Standalone U-Net Inference

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

### Google Colab Notebook

Open `Tree_Branch_Prediction.ipynb` in Google Colab and follow the notebook's setup instructions.

The notebook requires access to a compatible dataset and the appropriate trained model weights. Users without access to the original private dataset must configure an alternative dataset and make any necessary changes to the preprocessing and inference workflow.

For improved performance:

1. Select a GPU runtime in Google Colab, preferably an NVIDIA T4 GPU when available.
2. Run the notebook's dependency-installation and model-setup cells.
3. Configure the dataset location and verify that the expected images can be loaded.
4. Run the notebook cells in order to perform inference and display the results.

The notebook is intended for inference rather than model training. Using a substantially different dataset may require retraining or fine-tuning the relevant models.

## Output Files

For an input image named `tree.png`, the standalone prediction script generates three files in the output directory:

| File | Description |
|---|---|
| `tree_probability.png` | Grayscale prediction probability map |
| `tree_mask.png` | Binary segmentation mask |
| `tree_overlay.png` | RGB image with predicted pixels highlighted in red |

The binary mask uses a default probability threshold of 0.5. This threshold can be adjusted using the `--threshold` argument.

The Google Colab notebook provides additional visualisations for its combined workflow, including apple masks and overlays, as well as combined branch-and-apple predictions. These visualisations are distinct from the three output files generated by the standalone U-Net script.

## Limitations

- Predictions depend on the quality and alignment of the RGB and depth images.
- The model expects inputs processed consistently with its training pipeline.
- Depth-based masking can suppress useful image information if depth values are missing or incorrectly encoded.
- Prediction quality may vary across different trees, environments, lighting conditions, and levels of branch occlusion.
- The original dataset used by the notebook is private and is not provided with the repository.
- The notebook's filename parsing, image-pairing, and seasonal comparison logic may require modification for other datasets.
- The apple segmentation workflow depends on the availability and compatibility of the YOLO11 model and its trained weights.
- Models trained for the original dataset may not generalise reliably to different tree species, fruit types, cameras, or environmental conditions.
- The model has not been independently validated for every possible input environment.

Predictions on alternative datasets should be validated before being treated as reliable results.

## Acknowledgements and Licensing

This project builds on the U-Net implementation from [Milesial's Pytorch-UNet repository](https://github.com/milesial/Pytorch-UNet).

The original implementation is distributed under the GNU General Public License v3.0 (GPL-3.0). See the `LICENSE` file for the applicable licence terms. Copyright notices and licence terms for reused source code should be retained.

The project-specific preprocessing, inference code, and model configuration are intended for tree-branch segmentation. The notebook also uses a YOLO11-based workflow for apple segmentation; the applicable software licences and model-weight redistribution terms should be checked separately.

The provenance and redistribution terms of the trained checkpoints and any datasets should be checked before redistribution. The original private dataset must not be redistributed without appropriate permission.

## Disclaimer

This project is intended for research and educational purposes. No guarantee is made regarding the accuracy or suitability of predictions for any particular application.
