# Dataset Structure

This folder should contain your breast cancer ultrasound dataset.

## Expected Structure

```
data/
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   ├── benign (2).png
│   ├── benign (2)_mask.png
│   └── ...
├── malignant/
│   ├── malignant (1).png
│   ├── malignant (1)_mask.png
│   ├── malignant (2).png
│   ├── malignant (2)_mask.png
│   └── ...
└── normal/
    ├── normal (1).png
    ├── normal (1)_mask.png
    ├── normal (2).png
    ├── normal (2)_mask.png
    └── ...
```

## Image Requirements

- **Format**: PNG
- **Naming**: 
  - Images: `{class} ({number}).png`
  - Masks: `{class} ({number})_mask.png`
- **Resolution**: Any (will be resized to 128×128)
- **Mode**: Grayscale (will be converted if RGB)

## Mask Format

- **Binary masks**: 
  - 0 (black) = Background
  - 255 (white) = Tumor region
- **Grayscale PNG files**

## Dataset Used in This Study

- **Total Images**: 780
- **Benign**: ~260 images
- **Malignant**: ~260 images  
- **Normal**: ~260 images
- **Split**: 80% train / 20% validation

## Note

⚠️ **Dataset not included in this repository due to size/privacy**

Please place your own breast cancer ultrasound dataset here following the structure above.
