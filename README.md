# CNN Image Classification on EuroSAT Dataset
### CVPR Mid Assignment

A custom Convolutional Neural Network (CNN) trained on the **EuroSAT** satellite image dataset for land-use / land-cover classification across 10 classes.

---

## Results

| Model | Train Acc | Val Acc | Test Acc | Overfitting |
|-------|-----------|---------|----------|-------------|
| With BN + Dropout | 90.84% | 94.78% | 93.85% | None ✅ |
| No BN / No Dropout | 95.85% | 95.93% | 95.57% | None ✅ |

**Best Class:** Residential (99.0%)  
**Worst Class:** HerbaceousVegetation (85.3%)  
**Macro F1 Score:** 0.9383  
**Macro Precision:** 0.9382  
**Macro Recall:** 0.9389  

---

## Dataset

**EuroSAT RGB** — 27,000 Sentinel-2 satellite images across 10 land-use classes, each 64×64 pixels.

Download from Kaggle:
```
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
```

After downloading, place the dataset as follows:
```
data/
└── eurosat/
    └── 2750/
        ├── AnnualCrop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/
```

---

## Model Architecture

Custom VGG-inspired CNN with 4 convolutional blocks:

```
Block 1: Conv(3→32)   → BN → ReLU → Conv(32→32)   → BN → ReLU → MaxPool → Dropout
Block 2: Conv(32→64)  → BN → ReLU → Conv(64→64)   → BN → ReLU → MaxPool → Dropout
Block 3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout
Block 4: Conv(128→256)→ BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout
Classifier: FC(4096→512) → BN → ReLU → Dropout → FC(512→256) → ReLU → FC(256→10)
```

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Scheduler | CosineAnnealingLR |
| Batch Size | 64 |
| Epochs | 30 |
| Weight Decay | 1e-4 |
| Dropout | 0.3 |

**Data Split:** 70% Train / 10% Validation / 20% Test

---

## Files

| File | Description |
|------|-------------|
| `CNN_22-49133-3.ipynb` | Main notebook with full implementation |
| `CNN_WithReg_best.pth` | Best weights — With BN + Dropout |
| `CNN_NoReg_best.pth` | Best weights — Without regularisation |
| `CNN_StudentID_final.pth` | Final model weights |
| `CNN_StudentID_checkpoint.pth` | Full checkpoint with history |

---

## Requirements

```bash
pip install torch torchvision torchsummary scikit-learn matplotlib seaborn
```
