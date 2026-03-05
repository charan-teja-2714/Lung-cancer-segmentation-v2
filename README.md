# Lung Cancer Segmentation

Deep learning-based lung cancer segmentation using U-Net++ architecture.

## Overview

Multi-class segmentation of lung cancer types (ADC, LCC, SCC) from CT images using U-Net++ with EfficientNet-B4 encoder.

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python training/train.py
```

## Data Structure Required

```
backend/data/raw/
├── train/
│   ├── CT/
│   │   ├── ADC/     # CT images
│   │   ├── LCC/
│   │   └── SCC/
│   └── MASK/
│       ├── ADC/     # Masks
│       ├── LCC/
│       └── SCC/
└── test/            # Same structure
```

## What's Included

- ✅ U-Net++ training code
- ✅ Requirements file
- ❌ Pre-trained models (train your own)
- ❌ Dataset (provide your own)

## Results

- **Model**: U-Net++ with EfficientNet-B4
- **Performance**: 87.11% Mean Dice Score
- **Classes**: Background, ADC, LCC, SCC
- **Per-Class**: ADC (86.02%), LCC (87.95%), SCC (87.35%)

## Requirements

- Python 3.9+
- PyTorch 2.0+
- 8GB+ RAM
- GPU recommended

## License

Research Use Only


(venv) I:\Final Year Projects\lung-cancer-segmentation\backend>python training/evaluate.py
Using device: cuda
Test samples: 1140



================ DICE RESULTS ================
ADC Dice: 0.8602
LCC Dice: 0.8795
SCC Dice: 0.8735
---------------------------------------------
Mean Dice (ADC + LCC + SCC): 0.8711
=============================================

Note: Training validation shows ~93% due to batch-level averaging.
Final test evaluation (above) is the accurate metric.