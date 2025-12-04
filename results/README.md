# Results Files

This directory contains the experimental results from the 30-epoch training comparison.

## Available Files

### Documentation
- **RESULTS_SUMMARY.md** - Comprehensive results analysis and findings

### Visualizations
All plots showing model comparison metrics:
- `binary_comparison.png` - Complete training curves (all metrics)
- `dice_coefficient.png` - Dice score comparison
- `iou_score.png` - IoU score comparison  
- `loss.png` - Loss comparison
- `sensitivity.png` - Sensitivity comparison
- `specificity.png` - Specificity comparison
- `metrics_bar_comparison.png` - Bar chart comparison

### Large Files (Not Included in Git)

Due to GitHub file size limitations, the following files are **not included** in this repository:

#### Trained Models (~136 MB)
- `best_unet_binary.pth` (22 MB) - UNet model checkpoint
  - **Dice**: 0.6878 (Epoch 26)
  - **IoU**: 0.5383
  - **Sensitivity**: 0.8776
  - **Specificity**: 0.9662

- `best_multiresunet_binary.pth` (114 MB) - MultiResUNet model checkpoint
  - **Dice**: 0.6710 (Epoch 19)
  - **IoU**: 0.5189
  - **Sensitivity**: 0.9188
  - **Specificity**: 0.9544

#### Training Logs (1.5 MB)
- `binary_training_30epochs.log` - Complete training logs with all metrics for 30 epochs

## Accessing Large Files

If you need the trained models or training logs, please:
1. Train the models yourself using `code/compare_binary_segmentation.py`
2. Or contact the repository owner for access to the pre-trained weights

## Model Checkpoint Structure

Both `.pth` files contain:
```python
{
    'epoch': int,              # Epoch where best performance achieved
    'model_state_dict': dict,  # Model weights
    'optimizer_state_dict': dict,  # Optimizer state
    'val_dice': float,         # Best validation Dice score
    'val_iou': float,          # Best validation IoU
    'val_sens': float,         # Best validation Sensitivity
    'val_spec': float,         # Best validation Specificity
}
```

## Loading Models

```python
import torch

# Load UNet
checkpoint = torch.load('best_unet_binary.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best Dice: {checkpoint['val_dice']:.4f}")
print(f"Best Epoch: {checkpoint['epoch']}")
```
