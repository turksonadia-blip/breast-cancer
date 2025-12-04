# Breast Cancer Segmentation: UNet vs MultiResUNet

Deep learning comparison study for breast cancer tumor segmentation on ultrasound images.

## 🏆 Key Results

- **Winner**: UNet (Dice: 0.6878)
- **Runner-up**: MultiResUNet (Dice: 0.6710)
- **Improvement**: +2.44% with 5× fewer parameters

## 📁 Project Structure

```
breast-cancer-segmentation/
├── code/                          # Source code
│   ├── compare_binary_segmentation.py  # Main training script
│   ├── networks/                  # Model architectures
│   │   ├── MultiResUNet .py      # MultiResUNet implementation
│   │   └── RecursiveUNet.py      # UNet implementation
│   └── utils/                     # Utility functions
│       └── utils.py
├── results/                       # Experimental results
│   ├── RESULTS_SUMMARY.md        # Comprehensive results analysis
│   ├── best_unet_binary.pth      # Trained UNet model
│   ├── best_multiresunet_binary.pth  # Trained MultiResUNet model
│   ├── binary_comparison.png     # Training curves comparison
│   ├── binary_training_30epochs.log  # Full training logs
│   └── *.png                      # Individual metric plots
├── data/                          # Dataset (not included)
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔬 Dataset

- **Total Images**: 780 breast cancer ultrasound images
- **Resolution**: 128×128 pixels
- **Split**: 624 training / 156 validation (80/20)
- **Classes**: Binary segmentation (Background vs Tumor)
- **Class Distribution**: 91% background, 9% tumor

> **Note**: Dataset not included in this repository. Place your data in the `data/` folder with subfolders: `benign/`, `malignant/`, `normal/`

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
cd code
python compare_binary_segmentation.py
```

This will:
- Train both UNet and MultiResUNet for 30 epochs
- Save best models to `../results/`
- Generate comparison plots
- Log all metrics

### Using Trained Models

```python
import torch

# Load UNet (best performing model)
checkpoint = torch.load('results/best_unet_binary.pth')
# Model state dict: checkpoint['model_state_dict']
# Best Dice score: checkpoint['val_dice']
# Epoch: checkpoint['epoch']
```

## 📊 Results Summary

### Best Validation Performance

| Metric | UNet | MultiResUNet | Winner |
|--------|------|--------------|--------|
| **Dice Coefficient** | **0.6878** (E26) | 0.6710 (E19) | UNet (+2.44%) |
| **IoU** | **0.5383** (E26) | 0.5189 (E19) | UNet (+3.60%) |
| **Sensitivity** | 0.8776 (E11) | **0.9188** (E17) | MultiResUNet (+4.48%) |
| **Specificity** | **0.9662** (E26) | 0.9544 (E19) | UNet (+1.22%) |

### Model Comparison

| Model | Parameters | Best Dice | Epoch | Convergence |
|-------|-----------|-----------|-------|-------------|
| UNet | 1.9M | 0.6878 | 26 | Steady improvement |
| MultiResUNet | 9.9M | 0.6710 | 19 | Early overfitting |

### Key Findings

1. **UNet Superiority**: Better overall segmentation with 5× fewer parameters
2. **MultiResUNet Overfitting**: Peaked at epoch 19, then degraded (dataset too small for 9.9M parameters)
3. **Efficiency**: UNet offers best balance of accuracy and computational efficiency
4. **Clinical Deployment**: UNet recommended for production use

See [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md) for detailed analysis.

## 🏗️ Model Architectures

### UNet
- **Parameters**: 1,925,025 (~1.9M)
- **Architecture**: Classic encoder-decoder with skip connections
- **Optimizer**: Adam (LR=0.0001)
- **Scheduler**: CosineAnnealingLR

### MultiResUNet
- **Parameters**: 9,893,555 (~9.9M)
- **Architecture**: Multi-resolution convolutions with residual connections
- **Optimizer**: Adam (LR=0.00005)
- **Scheduler**: CosineAnnealingWarmRestarts

## 🔧 Training Configuration

- **Epochs**: 30
- **Batch Size**: 8
- **Loss Function**: Combined BCE (pos_weight=10.0) + Dice Loss
- **Hardware**: Apple Silicon GPU (MPS)
- **Training Time**: ~6.5 hours total
- **Data Augmentation**: 
  - Horizontal/vertical flips
  - Random rotation (±20°)
  - Brightness/contrast adjustment

## 📈 Metrics

- **Dice Coefficient**: Primary metric for segmentation quality
- **IoU (Jaccard)**: Intersection over Union
- **Sensitivity**: True Positive Rate (tumor detection)
- **Specificity**: True Negative Rate (background discrimination)

## 💡 Recommendations

### For Production Use
✅ **Use UNet**
- Superior overall performance
- 5× fewer parameters = faster inference
- Better generalization with limited data
- Balanced sensitivity-specificity trade-off

### For Future Research
- **Dataset Expansion**: MultiResUNet may perform better with >2000 images
- **Transfer Learning**: Pre-train on larger medical imaging datasets
- **Ensemble Methods**: Combine multiple UNet models
- **Multi-class Segmentation**: Distinguish benign vs malignant tumors

## 📄 License

See [LICENSE](../LICENSE) file for details.

## 🙏 Citation

If you use this code in your research, please cite:

```
@misc{breast-cancer-segmentation-2025,
  title={Breast Cancer Segmentation: UNet vs MultiResUNet Comparison},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/breast-cancer-segmentation}}
}
```

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Report Generated**: November 29, 2025  
**Training Duration**: 30 epochs (~6.5 hours)  
**Primary Metric**: Dice Coefficient  
**Winner**: UNet (0.6878 vs 0.6710)
