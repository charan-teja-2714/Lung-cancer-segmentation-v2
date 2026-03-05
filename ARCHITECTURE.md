# UNet++ Architecture

## Overview

This project uses **UNet++** (Nested U-Net) with **EfficientNet-B4** encoder for multi-class lung cancer segmentation.

## Architecture Diagram

```
INPUT: 256×256×1 (Grayscale CT Image)
│
├─────────────────── ENCODER (EfficientNet-B4) ──────────────────┐
│                                                                  │
X^(0,0) ──────────────────────────────────────────────────────> X^(0,1) ──────────> X^(0,2) ──────────> X^(0,3) ──────────> X^(0,4)
256×256×24    ↘                                    ↗ Up    ↘              ↗ Up    ↘              ↗ Up    ↘              ↗ Up
              │                                    │       │              │       │              │       │              │
              ↓                                    │       │              │       │              │       │              │
X^(1,0) ──────────────────────────────────────> X^(1,1) ──────────────> X^(1,2) ──────────────> X^(1,3) ─────────────┘
128×128×32    ↘                        ↗ Up              ↘              ↗ Up              ↘              ↗ Up
              │                        │                 │              │                 │              │
              ↓                        │                 │              │                 │              │
X^(2,0) ──────────────────────────> X^(2,1) ──────────────────────> X^(2,2) ─────────────────────────┘
64×64×56      ↘            ↗ Up                  ↘              ↗ Up
              │            │                     │              │
              ↓            │                     │              │
X^(3,0) ──────────────> X^(3,1) ─────────────────────────────┘
32×32×160     ↘  ↗ Up
              │  │
              ↓  │
X^(4,0) ──────┘
16×16×448
(Bottleneck)

OUTPUT: Softmax(X^(0,4)) → 256×256×4 (Background, ADC, LCC, SCC)
```

## Mathematical Notation

### Encoder Path (Downsampling)
- **X^(0,0)**: 256×256×24 (Input after first conv)
- **X^(1,0)**: 128×128×32 (Encoder level 1)
- **X^(2,0)**: 64×64×56 (Encoder level 2)
- **X^(3,0)**: 32×32×160 (Encoder level 3)
- **X^(4,0)**: 16×16×448 (Encoder level 4 - Bottleneck)

### Nested Dense Blocks (Decoder)

**Level 1:**
```
X^(0,1) = Conv(Concat[X^(0,0), Up(X^(1,0))])
X^(1,1) = Conv(Concat[X^(1,0), Up(X^(2,0))])
X^(2,1) = Conv(Concat[X^(2,0), Up(X^(3,0))])
X^(3,1) = Conv(Concat[X^(3,0), Up(X^(4,0))])
```

**Level 2:**
```
X^(0,2) = Conv(Concat[X^(0,0), X^(0,1), Up(X^(1,1))])
X^(1,2) = Conv(Concat[X^(1,0), X^(1,1), Up(X^(2,1))])
X^(2,2) = Conv(Concat[X^(2,0), X^(2,1), Up(X^(3,1))])
```

**Level 3:**
```
X^(0,3) = Conv(Concat[X^(0,0), X^(0,1), X^(0,2), Up(X^(1,2))])
X^(1,3) = Conv(Concat[X^(1,0), X^(1,1), X^(1,2), Up(X^(2,2))])
```

**Level 4 (Final Output):**
```
X^(0,4) = Conv(Concat[X^(0,0), X^(0,1), X^(0,2), X^(0,3), Up(X^(1,3))])
```

### Output Layer
```
Y = Softmax(X^(0,4)) → 256×256×4
```

## Notation Explanation

- **X^(i,j)**: Node at encoder depth `i`, decoder level `j`
  - `i`: Encoder depth (0 = top, 4 = bottom/bottleneck)
  - `j`: Decoder level (0 = encoder, 4 = final output)
- **Up()**: Upsampling operation (bilinear interpolation or transpose convolution)
- **Conv()**: Convolution block (Conv2D → BatchNorm → ReLU)
- **Concat[...]**: Channel-wise concatenation
- **→**: Skip connection
- **↓**: Downsampling (MaxPool or Strided Convolution)
- **↗**: Upsampling path

## Preprocessing Pipeline

### Input Processing
1. **Original Size**: 512×512 grayscale CT image
2. **Windowing**: Clip pixel values to [-160, 240] HU (lung window)
3. **Normalization**: Min-max scaling to [0, 1]
   ```
   img_normalized = (img - img_min) / (img_max - img_min + ε)
   ```
4. **Resize**: 512×512 → 256×256
5. **Tensor Conversion**: Convert to PyTorch tensor (1 channel)

### Mask Processing
1. **Load**: Grayscale mask image
2. **Binarize & Label**: Assign class IDs
   - 0: Background
   - 1: ADC (Adenocarcinoma)
   - 2: LCC (Large Cell Carcinoma)
   - 3: SCC (Squamous Cell Carcinoma)
3. **Resize**: 512×512 → 256×256 (nearest neighbor interpolation)
4. **Tensor Conversion**: Convert to long tensor

## Loss Function

### Combined Loss
```
L_total = L_CE + L_Dice
```

### Cross-Entropy Loss
```
L_CE = -∑(y_true × log(y_pred))
```

### Dice Loss
```
L_Dice = 1 - (2 × |Y ∩ Y_true|) / (|Y| + |Y_true|)
```

## Evaluation Metrics

### Dice Score per Class
```
Dice_c = (2 × TP_c) / (2 × TP_c + FP_c + FN_c)
```

Where:
- TP: True Positives
- FP: False Positives
- FN: False Negatives
- c: Class (ADC, LCC, or SCC)

### Mean Dice Score
```
Mean_Dice = (Dice_ADC + Dice_LCC + Dice_SCC) / 3
```

## Model Configuration

```python
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4",      # Pretrained encoder
    encoder_weights="imagenet",           # ImageNet weights
    in_channels=1,                        # Grayscale input
    classes=4                             # 4-class output (Bg, ADC, LCC, SCC)
)
```

## Training Configuration

- **Optimizer**: Adam (Learning Rate: 1e-4)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Batch Size**: 4
- **Image Size**: 256×256
- **Max Epochs**: 120
- **Early Stopping**: Patience = 10

## Key Features

1. **Nested Dense Skip Connections**: Each decoder node receives ALL previous same-level nodes
2. **Pretrained Encoder**: EfficientNet-B4 with ImageNet weights
3. **Multi-Class Segmentation**: 4 classes (Background + 3 cancer types)
4. **Combined Loss**: CrossEntropy + Dice for better convergence
5. **Deep Supervision Ready**: Architecture supports deep supervision (not currently enabled)

## Performance

- **Mean Dice Score**: 87.11%
- **ADC Dice**: 86.02%
- **LCC Dice**: 87.95%
- **SCC Dice**: 87.35%

## Reference

Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). 
**UNet++: A Nested U-Net Architecture for Medical Image Segmentation.** 
*Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3-11.
