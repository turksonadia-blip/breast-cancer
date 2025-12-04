"""
BINARY Segmentation: UNet vs MultiResUNet on Breast Cancer
Simplified to 2 classes: Background (0) vs Tumor (1) - ANY tumor type
This approach is better for small datasets with severe class imbalance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import MultiResUNet
import importlib.util
spec = importlib.util.spec_from_file_location("MultiResUNet", 
                                               "/Users/opam/Documents/breast-cancer-main/networks/MultiResUNet .py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MultiResUNet = module.MultiResUNet


# ============ Binary Dice + BCE Loss ============
class BinarySegmentationLoss(nn.Module):
    """Combined Binary Cross Entropy + Dice Loss for binary segmentation"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super(BinarySegmentationLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # Use weighted BCE to handle class imbalance (91% background vs 9% tumor)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def dice_loss(self, pred, target):
        """Binary Dice Loss"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
        return 1 - dice
    
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


# ============ UNet for Binary Segmentation ============
class UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4, norm_layer=nn.InstanceNorm2d):
        super(UNet, self).__init__()
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1), 
                                             out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes, kernel_size=kernel_size, 
                                             norm_layer=norm_layer, innermost=True)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes, kernel_size=kernel_size, 
                                                 submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                             num_classes=num_classes, kernel_size=kernel_size, 
                                             submodule=unet_block, norm_layer=norm_layer, outermost=True)
        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv3 = self.expand(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)
            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])
        
        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True))

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)


# ============ Data Augmentation ============
class AugmentationTransform:
    """Data augmentation for training"""
    def __init__(self, is_training=True):
        self.is_training = is_training
    
    def __call__(self, image, mask):
        if not self.is_training:
            return image, mask
        
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        
        if random.random() > 0.5:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask.unsqueeze(0), angle, fill=0).squeeze(0)
        
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
        
        return image, mask


# ============ Binary Dataset ============
class BinaryBreastCancerDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, augmentation=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        
        # Load mask - BINARY: 0=Background, 1=ANY tumor (benign/malignant/normal)
        mask_path = self.mask_paths[idx]
        if ',' in mask_path:  # Multiple masks
            masks = []
            for path in mask_path.split(','):
                m = self.transform(Image.open(path).convert('L'))
                masks.append(np.array(m.squeeze()))
            combined_mask = np.maximum.reduce(masks)
            # Any non-background pixel becomes tumor (1)
            mask = np.where(combined_mask > 0.5, 1.0, 0.0)
            mask = torch.from_numpy(mask).float()
        else:
            mask_img = self.transform(Image.open(mask_path).convert('L')).squeeze()
            # Binary: 0 or 1
            mask = torch.where(mask_img > 0.5, 1.0, 0.0).float()
        
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
        
        return image, mask.unsqueeze(0)


# ============ Binary Metrics ============
def binary_dice_coefficient(pred, target, threshold=0.5):
    """Compute Dice for binary segmentation"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    return dice.item()


def binary_iou_score(pred, target, threshold=0.5):
    """Compute IoU for binary segmentation"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()


def binary_sensitivity(pred, target, threshold=0.5):
    """Sensitivity (Recall): TP / (TP + FN)"""
    pred = (torch.sigmoid(pred) > threshold).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    sens = (tp + 1e-7) / (tp + fn + 1e-7)
    return sens.item()


def binary_specificity(pred, target, threshold=0.5):
    """Specificity: TN / (TN + FP)"""
    pred = (torch.sigmoid(pred) > threshold).float()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    spec = (tn + 1e-7) / (tn + fp + 1e-7)
    return spec.item()


# ============ Training Functions ============
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    sens_scores = []
    spec_scores = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            dice = binary_dice_coefficient(outputs, masks)
            iou = binary_iou_score(outputs, masks)
            sens = binary_sensitivity(outputs, masks)
            spec = binary_specificity(outputs, masks)
            dice_scores.append(dice)
            iou_scores.append(iou)
            sens_scores.append(sens)
            spec_scores.append(spec)
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'dice': dice, 'iou': iou, 'sens': sens, 'spec': spec})
    
    return total_loss / len(dataloader), np.mean(dice_scores), np.mean(iou_scores), np.mean(sens_scores), np.mean(spec_scores)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    sens_scores = []
    spec_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            dice = binary_dice_coefficient(outputs, masks)
            iou = binary_iou_score(outputs, masks)
            sens = binary_sensitivity(outputs, masks)
            spec = binary_specificity(outputs, masks)
            dice_scores.append(dice)
            iou_scores.append(iou)
            sens_scores.append(sens)
            spec_scores.append(spec)
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'dice': dice, 'iou': iou, 'sens': sens, 'spec': spec})
    
    return total_loss / len(dataloader), np.mean(dice_scores), np.mean(iou_scores), np.mean(sens_scores), np.mean(spec_scores)


def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_path, scheduler=None):
    best_dice = 0.0
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [], 'train_sens': [], 'train_spec': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [], 'val_sens': [], 'val_spec': []
    }
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"{model_name} - Epoch {epoch+1}/{num_epochs}")
        print('='*60)
        
        train_loss, train_dice, train_iou, train_sens, train_spec = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou, val_sens, val_spec = validate_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['train_sens'].append(train_sens)
        history['train_spec'].append(train_spec)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_sens'].append(val_sens)
        history['val_spec'].append(val_spec)
        
        print(f"\nTrain - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, Sens: {train_sens:.4f}, Spec: {train_spec:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Sens: {val_sens:.4f}, Spec: {val_spec:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_sens': val_sens,
                'val_spec': val_spec,
            }, save_path)
            print(f"✓ Best model saved! Dice: {best_dice:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return history


def compare_models(unet_history, multiresunet_history, save_path='binary_comparison.png'):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    metrics = ['loss', 'dice', 'iou', 'sens', 'spec']
    titles = ['Loss', 'Dice Coefficient', 'IoU Score', 'Sensitivity', 'Specificity']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        axes[0, idx].plot(unet_history[f'train_{metric}'], label='UNet', marker='o', markersize=4)
        axes[0, idx].plot(multiresunet_history[f'train_{metric}'], label='MultiResUNet', marker='s', markersize=4)
        axes[0, idx].set_title(f'Training {title}')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel(title)
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
        
        axes[1, idx].plot(unet_history[f'val_{metric}'], label='UNet', marker='o', markersize=4)
        axes[1, idx].plot(multiresunet_history[f'val_{metric}'], label='MultiResUNet', marker='s', markersize=4)
        axes[1, idx].set_title(f'Validation {title}')
        axes[1, idx].set_xlabel('Epoch')
        axes[1, idx].set_ylabel(title)
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def print_comparison_summary(unet_history, multiresunet_history):
    """Print final comparison summary"""
    print("\n" + "="*80)
    print("BINARY SEGMENTATION - FINAL COMPARISON")
    print("="*80)
    
    unet_best_dice = max(unet_history['val_dice'])
    unet_best_idx = unet_history['val_dice'].index(unet_best_dice)
    
    multires_best_dice = max(multiresunet_history['val_dice'])
    multires_best_idx = multiresunet_history['val_dice'].index(multires_best_dice)
    
    print(f"\nUNet Best Dice: {unet_best_dice:.4f} (Epoch {unet_best_idx + 1})")
    print(f"MultiResUNet Best Dice: {multires_best_dice:.4f} (Epoch {multires_best_idx + 1})")
    
    improvement = ((multires_best_dice - unet_best_dice) / unet_best_dice) * 100 if unet_best_dice > 0 else 0
    print(f"\nImprovement: {improvement:+.2f}%")
    
    winner = "MultiResUNet" if multires_best_dice > unet_best_dice else "UNet"
    print(f"\n🏆 WINNER: {winner}")
    print("="*80)


def main():
    # Configuration
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0001
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print("Using NVIDIA GPU (CUDA)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        DEVICE = torch.device('cpu')
        print("Using CPU")
    
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    
    # Load data
    print("\nLoading data...")
    paths = glob('data/*/*')
    df = pd.DataFrame(paths, columns=['file_path'])
    df[['folder', 'class', 'study_id']] = df['file_path'].str.split('/', expand=True)
    df.loc[~df['study_id'].str.contains('_mask'), 'data_type'] = "image"
    df.loc[df['study_id'].str.contains('_mask'), 'data_type'] = "mask"
    
    image_df = df[df['data_type'] == "image"]
    image_paths = image_df['file_path'].tolist()
    mask_paths = [p.replace('.png', '_mask.png') if os.path.exists(p.replace('.png', '_mask.png')) 
                  else p.replace('.png', '_mask_1.png') for p in image_paths]
    
    print(f"Found {len(image_paths)} images")
    
    # Split data
    train_img, val_img, train_mask, val_mask = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=SEED
    )
    
    print(f"Train: {len(train_img)}, Val: {len(val_img)}")
    
    # Create datasets
    train_augmentation = AugmentationTransform(is_training=True)
    train_dataset = BinaryBreastCancerDataset(train_img, train_mask, augmentation=train_augmentation)
    val_dataset = BinaryBreastCancerDataset(val_img, val_mask, augmentation=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Calculate positive weight for class imbalance (91% background, 9% tumor)
    pos_weight = torch.tensor([10.0]).to(DEVICE)  # 91/9 ≈ 10
    
    # ==================== Train UNet ====================
    print("\n" + "="*80)
    print("TRAINING UNET - BINARY SEGMENTATION (Background vs Tumor)")
    print("="*80)
    
    unet = UNet(num_classes=1, in_channels=1, initial_filter_size=32, num_downs=3).to(DEVICE)
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    criterion = BinarySegmentationLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)
    print(f"Using Binary Loss: BCE (pos_weight=10.0) + Dice")
    
    optimizer_unet = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    unet_history = train_model(
        unet, "UNet", train_loader, val_loader, criterion, optimizer_unet,
        num_epochs=NUM_EPOCHS, device=DEVICE, save_path='best_unet_binary.pth',
        scheduler=scheduler_unet
    )
    
    # ==================== Train MultiResUNet ====================
    print("\n" + "="*80)
    print("TRAINING MULTIRESUNET - BINARY SEGMENTATION")
    print("="*80)
    
    torch.manual_seed(SEED + 100)
    multiresunet = MultiResUNet(in_channels=1, num_classes=1, base_filters=32, alpha=1.67).to(DEVICE)
    torch.manual_seed(SEED)
    
    print(f"MultiResUNet parameters: {sum(p.numel() for p in multiresunet.parameters()):,}")
    
    criterion_multires = BinarySegmentationLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)
    optimizer_multires = optim.Adam(multiresunet.parameters(), lr=LEARNING_RATE * 0.5, weight_decay=1e-5)
    scheduler_multires = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_multires, T_0=5, T_mult=1, eta_min=1e-6
    )
    
    multiresunet_history = train_model(
        multiresunet, "MultiResUNet", train_loader, val_loader, criterion_multires, optimizer_multires,
        num_epochs=NUM_EPOCHS, device=DEVICE, save_path='best_multiresunet_binary.pth',
        scheduler=scheduler_multires
    )
    
    # ==================== Compare Results ====================
    compare_models(unet_history, multiresunet_history)
    print_comparison_summary(unet_history, multiresunet_history)
    
    print("\n✅ Binary segmentation comparison complete!")


if __name__ == "__main__":
    main()
