# 👁️ Retinal Vessel Segmentation & DR Classification

> Deep Learning pipeline for retinal vessel segmentation and diabetic 
> retinopathy grading using U-Net architectures and EfficientNet-B0.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Pipeline](#pipeline)
- [Clinical Analysis](#clinical-analysis)
- [Classification](#classification)

---

## 🎯 Overview

This project implements a complete end-to-end pipeline for:

1. **Retinal vessel segmentation** on the DRIVE dataset using three 
   deep learning architectures (U-Net, Attention U-Net, ResUNet)
2. **Diabetic Retinopathy (DR) grading** on Messidor-2 using two 
   fusion strategies that combine RGB images with segmentation masks
3. **Clinical vessel analysis** extracting quantitative biomarkers 
   from predicted masks (length, caliber, tortuosity, bifurcations)

Diabetic retinopathy is the leading cause of blindness in working-age 
adults. Automated analysis of retinal vessels enables early detection 
and objective monitoring of disease progression.

---

## 💾 Dataset

### Segmentation — DRIVE (Digital Retinal Images for Vessel Extraction)

| Split | Images | Resolution | Annotations |
|-------|--------|------------|-------------|
| Train | 20     | 565×584 px | Manual (1st expert) |
| Test  | 20     | 565×584 px | Manual (1st + 2nd expert) |

- Format: `.tif` images, `.gif` masks and FOV masks
- Class imbalance: ~7.5% vessel pixels

### Classification — Messidor-2

| Grade | Description          | Count | %    |
|-------|----------------------|-------|------|
| 0     | No DR                | 1017  | 58%  |
| 1     | Mild DR              | 270   | 15%  |
| 2     | Moderate DR          | 347   | 20%  |
| 3     | Severe DR            | 75    | 4%   |
| 4     | Proliferative DR     | 35    | 2%   |

---
---

## 🏗️ Architecture

### Segmentation Models

#### 1. U-Net (from scratch)
Input (3, 512, 512)

│

├── Encoder: DoubleConv → MaxPool × 4

│   Features: [64, 128, 256, 512]

│

├── Bottleneck: DoubleConv → 1024 channels

│

└── Decoder: ConvTranspose2d + Skip connections × 4

       │

└── Output Conv1×1 → (1, 512, 512)
- **DoubleConv**: Conv2d → BatchNorm → ReLU (×2)
- **Skip connections**: concatenation of encoder features
- Loss: BCEDice = 0.5 × BCE + 0.5 × Dice

#### 2. Attention U-Net
Same as U-Net + **Attention Gates** on each skip connection:
AttGate(g, x):

Wg(g) + Wx(x) → ReLU → psi → Sigmoid → attention map

x * attention_map → weighted skip features
- Focuses on vascular regions, suppresses background
- Better performance on thin capillaries

#### 3. ResUNet (Transfer Learning)
Encoder: ResNet-34 pretrained (ImageNet) via timm

out_indices = (0, 1, 2, 3, 4)

channels   = [64, 64, 128, 256, 512]

│

Bottleneck: Conv2d → 1024 channels

│

Decoder: DecBlock × 4

DecBlock: ConvTranspose2d + concat skip + DoubleConv

│

Final: Upsample(×2) + Conv1×1 → (1, 512, 512)

### Classification Models

#### Method 1 — Early Fusion (4 Channels) ⭐ Best

Input: concat(RGB, segmentation_mask) → (4, 512, 512)

│

conv_stem: Conv2d(4→32)  ← modified from Conv2d(3→32)

Weights init: RGB channels copied from ImageNet

mask channel = mean of RGB channels

│

EfficientNet-B0 backbone (MBConv blocks)

│

GlobalAvgPool → FC(1280→5) → Softmax
#### Method 2 — Late Fusion (Dual-Stream)
Input (4, 512, 512)

│

├── x_rgb  [:3] → EfficientNet-B0 → 1280 features

└── x_mask [3:] → ResNet-18 (in_chans=1) → 512 features

│

torch.cat([feat_rgb, feat_mask]) → 1792 dim

│

FC(1792→512) → ReLU → Dropout(0.3)

│

FC(512→5) → Logits
---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/your-username/retinal-vessel-segmentation.git
cd retinal-vessel-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirementstorch>=2.0.0

torchvision>=0.15.0

timm>=0.9.0

albumentations>=1.3.0

opencv-python>=4.8.0

numpy>=1.24.0

pandas>=2.0.0

scikit-learn>=1.3.0

scikit-image>=0.21.0

scipy>=1.11.0

matplotlib>=3.7.0

seaborn>=0.12.0

Pillow>=10.0.0

tqdm>=4.65.0
---

## 🚀 Usage

Open the main notebook on Kaggle GPU (recommended: T4/P100, 16GB VRAM):

```bash
jupyter notebook notebooks/computer-vision.ipynb
```

### Step-by-step execution

**Step 1 — Setup & Preprocessing**
```python
# Reads .tif images, applies CLAHE on green channel, 
# loads expert masks with FOV masking
ds = DRIVEDataset(TRAIN_IMGS, TRAIN_MASKS, TRAIN_FOV,
                  transform=get_train_transforms(512))
```

**Step 2 — Train Segmentation Models**
```python
# Trains UNet, AttentionUNet, ResUNet for 30 epochs each
for name, model in models_cfg.items():
    # BCEDice loss + AdamW + CosineAnnealingLR
    train_epoch(model, train_loader, optimizer, criterion, DEVICE)
```

**Step 3 — Fine-tune Best Model**
```python
# Attention U-Net: 60 additional epochs at lr=5e-5
model_att.load_state_dict(torch.load('best_attention_unet.pth'))
# → Best Dice: 0.7656
```

**Step 4 — Pre-compute Segmentation Masks**
```python
# Generate and cache masks for all Messidor-2 images
precompute_masks(df_clean, IMG_DIR, att_best)
```

**Step 5 — Train DR Classifier**
```python
# Method 1: 4-channel EfficientNet-B0
clf_model = DRClassifier4Channels(model_name='efficientnet_b0',
                                   num_classes=5)
# 20 epochs → Kappa: 0.8485
```

**Step 6 — Clinical Vessel Analysis**
```python
# Extract biomarkers from predicted masks
results = analyze_vessels(pred_mask, pixel_size_mm=0.0125)
# Returns: length, caliber, density, tortuosity, bifurcations
```

---

## 📊 Results

### Segmentation — DRIVE Test Set

| Model           | Dice   | IoU    | Sensitivity | Specificity | AUC-ROC |
|-----------------|--------|--------|-------------|-------------|---------|
| U-Net           | 0.7604 | 0.6134 | 0.8087      | 0.9719      | 0.9731  |
| Attention U-Net | **0.7656** | 0.6121 | 0.7845  | **0.9752**  | 0.9687  |
| ResUNet         | 0.6771 | 0.5118 | **0.8619**  | 0.9397      | 0.9639  |

> ★ Attention U-Net fine-tuned for 60 additional epochs (lr=5e-5)

### Classification — Messidor-2 Validation Set

| Method                      | Accuracy | Kappa (quadratic) |
|-----------------------------|----------|-------------------|
| Method 1 — Early Fusion 4ch | 78.22%   | **0.8485** ⭐     |
| Method 2 — Dual-Stream      | 74.50%   | 0.7862            |

### Why Early Fusion wins

- **Fewer parameters** (~4M vs ~12M) → less overfitting on 1395 images
- **Deeper cross-modal interactions** — EfficientNet learns joint 
  RGB+mask representations from the first convolution
- **Faster convergence** — Kappa 0.27→0.85 in 20 epochs vs 0.05→0.79

---

## 🔄 PipelineRetinal Image (.tif)

│

▼

Green channel extraction

│

▼

CLAHE (clipLimit=2.0, tileGrid=8×8)

│

▼

FOV masking

│

▼

Albumentations augmentation

(HFlip, VFlip, Rotate90, ShiftScaleRotate,

ElasticTransform, Noise, Contrast)

│

▼

┌───────────────────┐

│  SEGMENTATION     │

│  Attention U-Net  │ → Dice: 0.7656 / AUC: 0.9687

│  (best model)     │

└────────┬──────────┘

│ vessel mask

▼

┌─────────────────────────────┐

│  CLINICAL BIOMARKERS        │

│  skeletonize + regionprops  │ → length, caliber,

│  distance_transform_edt     │   tortuosity, bifurcations

└─────────────────────────────┘

│

▼

┌──────────────────────────────┐

│  DR CLASSIFICATION           │

│  EfficientNet-B0 (4 channels)│ → Grade 0-4

│  RGB + segmentation mask     │   Kappa: 0.8485

└──────────────────────────────┘
---

## 🔬 Clinical Analysis

From the segmentation mask, 5 clinical biomarkers are extracted:

| Biomarker           | Method                        | Value (sample) |
|---------------------|-------------------------------|----------------|
| Total length        | Skeletonize → pixel count     | 81.79 mm       |
| Mean caliber        | Distance transform × 2        | 0.0628 mm      |
| Vessel density      | mask pixels / total pixels    | 10.1%          |
| Mean tortuosity     | arc_length / chord_length     | 2.77           |
| Bifurcation count   | Skeleton junction detection   | 329            |

```python
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt

binary   = (pred_mask > 0.5).astype(np.uint8)
skeleton = skeletonize(binary)
dist     = distance_transform_edt(binary) * 2  # diameter map
```

These metrics correlate clinically with DR severity:
- **Tortuosity ↑** → sign of neovascularization (Grade 3-4)
- **Caliber ↑** → venous dilation, early DR marker
- **Bifurcations ↓** → vessel rarefaction in advanced stages

---

## 🏷️ Classification Details

### Key Innovation — Segmentation-Guided Classification

Instead of classifying RGB images alone, we concatenate the 
segmentation mask as a 4th input channel. This forces the classifier 
to correlate vascular structure with DR grade.

```python
# Mask pre-computation (saves inference time during training)
MASK_CACHE_DIR = '/kaggle/working/masks_cache/'
precompute_masks(df_clean, IMG_DIR, att_best)

# Dataset: 4-channel fusion
combined = np.concatenate([img_rgb / 255.0,
                           mask[:, :, None]], axis=2)
# Shape: (512, 512, 4)
```

### Training Strategy

```python
# Handle class imbalance (Grade 4: only 2% of data)
class_weights = compute_class_weight('balanced',
    classes=np.arange(5),
    y=df_train['adjudicated_dr_grade'].values)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
```

---

## 📈 Training Configuration

| Parameter         | Segmentation    | Classification   |
|-------------------|-----------------|------------------|
| Image size        | 512×512         | 512×512          |
| Batch size        | 2               | 8                |
| Epochs            | 30 (+60 FT)     | 20               |
| Optimizer         | AdamW           | AdamW            |
| Learning rate     | 1e-4 (FT: 5e-5) | 1e-3             |
| Weight decay      | 1e-5            | 1e-4             |
| Scheduler         | CosineAnnealingLR | CosineAnnealingLR |
| Loss              | BCEDice         | CrossEntropy (weighted) |
| GPU               | NVIDIA (16GB)   | NVIDIA (16GB)    |

---

## 🔧 Key Implementation Details

### CLAHE Preprocessing

```python
def get_green_enhanced(image_bgr):
    green   = image_bgr[:, :, 1]          # Green channel best contrast
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)
    return cv2.merge([enhanced, enhanced, enhanced])
```

### BCEDice Loss

```python
class BCEDiceLoss(nn.Module):
    def forward(self, pred, target):
        bce  = F.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred).view(-1)
        tgt  = target.view(-1)
        inter = (pred * tgt).sum()
        dice = 1 - (2*inter + 1) / (pred.sum() + tgt.sum() + 1)
        return 0.5 * bce + 0.5 * dice
```

### 4-Channel EfficientNet Initialization

```python
# Intelligent weight initialization for the 4th channel
new_conv = nn.Conv2d(4, out_channels, kernel_size, stride, padding)
with torch.no_grad():
    new_conv.weight[:, :3] = existing_conv.weight        # Copy RGB
    new_conv.weight[:, 3]  = existing_conv.weight.mean(dim=1)  # Init mask
```

---

## 📝 Acknowledgements

- **DRIVE Dataset**: Staal et al., IEEE TMI 2004
- **Messidor-2**: Decencière et al., Ophthalmic Research 2014
- **U-Net**: Ronneberger et al., MICCAI 2015
- **Attention U-Net**: Oktay et al., MIDL 2018
- **EfficientNet**: Tan & Le, ICML 2019
- **ResNet**: He et al., CVPR 2016

---

## 📝 Author
## Nadine Snoussi, Data Science and AI engineering student

## 📄 License

This project is developed for academic purposes as part of a Computer 
Vision course project.
