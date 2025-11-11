# Breast Cancer Detection and Segmentation with UNet

A deep learning project for breast cancer classification and medical image segmentation using UNet architecture on ultrasound and medical imaging data.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

This project implements a medical image segmentation pipeline for breast cancer detection using deep learning. It leverages the UNet architecture, a powerful convolutional neural network designed for biomedical image segmentation, to classify and segment breast ultrasound images into three categories: **benign**, **malignant**, and **normal**.

The implementation includes both 2D and 3D variants of UNet, along with a complete training, validation, and inference pipeline suitable for medical imaging applications.

## Features

- **Multi-class Segmentation**: Classifies breast tissue into background, anterior, and posterior regions
- **2D and 3D UNet Architectures**: Flexible implementation supporting both 2D slice-based and 3D volumetric segmentation
- **Recursive UNet Implementation**: Based on the German Cancer Research Center (DKFZ) architecture
- **Complete Training Pipeline**: Includes data loading, augmentation, training, validation, and testing
- **Medical Image Support**: Handles NIfTI format medical images using nibabel
- **TensorBoard Integration**: Real-time training monitoring and visualization
- **Comprehensive Metrics**: Dice coefficient, Jaccard index, sensitivity, and specificity
- **Inference Agent**: Ready-to-use inference module for deployment

## Dataset

The project works with breast ultrasound images from the following structure:

```
data/
├── benign/       # Benign tumor images
├── malignant/    # Malignant tumor images
└── normal/       # Normal tissue images
```

**Primary Dataset**: [Breast Cancer Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

The dataset contains ultrasound images categorized into three classes for training and evaluation.

## Architecture

### UNet Architecture

The project implements a recursive UNet architecture with the following key components:

- **Encoder Path**: Contracting path with max pooling for downsampling
- **Decoder Path**: Expanding path with transposed convolutions for upsampling
- **Skip Connections**: Concatenation of features from encoder to decoder
- **Normalization**: Instance normalization for stable training
- **Activation**: LeakyReLU activation functions

**Key Parameters**:
- Number of classes: 3 (background, anterior, posterior)
- Input channels: 1 (grayscale medical images)
- Initial filter size: 64 (configurable up to 256)
- Number of downsampling layers: 4 (configurable)
- Kernel size: 3x3

### Network Variants

1. **RecursiveUNet.py**: 2D UNet for slice-based processing
2. **RecursiveUNet3D.py**: 3D UNet for volumetric processing

## Project Structure

```
breast-cancer/
├── data/                           # Dataset directory
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── data_prep/                      # Data preparation modules
│   ├── SlicesDataset.py           # PyTorch Dataset for 2D slices
│   └── HippocampusDatasetLoader.py
├── experiments/                    # Training experiments
│   └── UNetExperiment.py          # Main experiment class
├── inference/                      # Inference pipeline
│   └── UNetInferenceAgent.py      # Inference agent
├── networks/                       # Neural network architectures
│   ├── RecursiveUNet.py           # 2D UNet implementation
│   ├── RecursiveUNet3D.py         # 3D UNet implementation
│   └── archive/
├── utils/                          # Utility functions
│   ├── utils.py                   # General utilities
│   └── volume_stats.py            # Metrics computation
├── Ultrasound Example.ipynb       # Demo notebook
├── requirements.txt               # Python dependencies
├── LICENSE                        # License file
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/melhzy/breast-cancer.git
cd breast-cancer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following main packages:
- PyTorch and torchvision
- NumPy and SciPy
- OpenCV (opencv-python)
- Matplotlib and Pillow
- scikit-learn
- nibabel (for NIfTI format)
- MedPy (medical image processing)
- TensorBoard (training visualization)
- AdaMod optimizer
- pandas

See `requirements.txt` for the complete list.

## Usage

### Training

To train the model, use the `UNetExperiment` class:

```python
from experiments.UNetExperiment import UNetExperiment

# Configure experiment
config = {
    'n_epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'test_results_dir': './results',
    'name': 'breast_cancer_unet'
}

# Initialize and run experiment
experiment = UNetExperiment(config, split, dataset)
experiment.run()
```

### Inference

Use the inference agent for predictions on new images:

```python
from inference.UNetInferenceAgent import UNetInferenceAgent

# Initialize inference agent
agent = UNetInferenceAgent(
    parameter_file_path='path/to/model.pth',
    device='cuda',
    patch_size=64
)

# Run inference on a volume
prediction = agent.single_volume_inference(volume)
```

### Jupyter Notebook Demo

Explore the `Ultrasound Example.ipynb` notebook for a complete walkthrough of:
- Data loading and preprocessing
- Model architecture
- Training loop
- Visualization of results
- Custom data augmentation techniques

## Model Details

### Training Configuration

- **Optimizer**: AdaMod (adaptive learning rate optimizer)
- **Loss Function**: Cross-entropy loss with class weighting
- **Data Augmentation**: 
  - Elastic deformation
  - Random rotations
  - Gaussian noise
  - Intensity scaling
- **Batch Size**: 8 (configurable)
- **Learning Rate**: 1e-4 (with scheduling)
- **Weight Initialization**: Kaiming normal initialization

### Data Processing

The `SlicesDataset` class handles:
- Loading 3D volumes and extracting 2D slices
- Normalization and preprocessing
- On-the-fly data augmentation
- Efficient batching for training

## Evaluation Metrics

The project implements comprehensive medical imaging metrics:

### Dice Coefficient (Dice3d)
Measures overlap between predicted and ground truth segmentation:
$$\text{Dice} = \frac{2 \times |A \cap B|}{|A| + |B|}$$

### Jaccard Index (Jaccard3d)
Intersection over Union metric:
$$\text{Jaccard} = \frac{|A \cap B|}{|A \cup B|}$$

### Sensitivity (Recall)
True positive rate for medical diagnosis

### Specificity
True negative rate for medical diagnosis

All metrics are computed in 3D space for volumetric evaluation.

## Results

The model achieves:
- Multi-class segmentation of breast tissue regions
- Real-time inference on medical images
- Volumetric predictions from 2D slices
- TensorBoard visualization of training progress

Results are saved to the configured output directory with:
- Model checkpoints
- Segmentation masks
- Performance metrics
- TensorBoard logs

## References

1. **Breast Cancer Ultrasound**: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

2. **UNet Implementation**: [MIC-DKFZ Basic UNet Example - Networks](https://github.com/MIC-DKFZ/basic_unet_example/tree/master/networks)

3. **UNet Architecture**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

## License

Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- German Cancer Research Center (DKFZ) for the UNet implementation
- Kaggle community for the breast ultrasound dataset
- PyTorch team for the deep learning framework

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This project is for research and educational purposes. Medical diagnosis should always be performed by qualified healthcare professionals.