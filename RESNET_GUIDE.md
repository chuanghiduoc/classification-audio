# 🏆 Transfer Learning với ResNet18

## 📚 Giới thiệu

**Transfer Learning** là kỹ thuật sử dụng model đã được train trên dataset lớn (ImageNet - 1.4M images) và fine-tune cho task mới (audio classification).

**Tại sao hiệu quả?**
- ✅ Model đã học được low-level features (edges, textures, patterns)
- ✅ Chỉ cần fine-tune high-level features cho audio domain
- ✅ Cần ít data training hơn
- ✅ Converge nhanh hơn
- ✅ Accuracy cao hơn nhiều

---

## 🏗️ Architecture: ResNet18

### ResNet (Residual Network)
- **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Innovation:** Skip connections (residual blocks)
- **Depth:** 18 layers
- **Parameters:** 11.2 million
- **Pretrained on:** ImageNet (1000 classes, 1.4M images)

### Architecture Overview:
```
Input (3, 128, 128) - RGB Mel Spectrogram
    ↓
Conv2d (7x7, stride 2)
    ↓
MaxPool (3x3, stride 2)
    ↓
[ResBlock × 2] - 64 filters
    ↓
[ResBlock × 2] - 128 filters  
    ↓
[ResBlock × 2] - 256 filters
    ↓
[ResBlock × 2] - 512 filters
    ↓
AdaptiveAvgPool
    ↓
FC (512 → 50) - Audio classes
```

---

## 🔄 Adaptation cho Audio

### 1. **Input Format Change**

**Problem:** ResNet expects RGB (3-channel), we have grayscale mel spectrogram (1-channel)

**Solution:** Replicate grayscale to 3 channels
```python
# (1, H, W) → (3, H, W)
spec_rgb = np.repeat(spec, 3, axis=0)
```

### 2. **Output Layer Change**

**Original:** 1000 classes (ImageNet)  
**Modified:** 50 classes (ESC-50)

```python
self.resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 50)
)
```

---

## 📖 Two-Stage Training Strategy

### **Stage 1: Train FC Layer Only (20 epochs)**

**Strategy:** Freeze backbone, train only final FC layer

```python
model.freeze_backbone()  # Freeze conv layers
optimizer = Adam(FC_params, lr=0.001)
```

**Why?**
- Pretrained weights are good for feature extraction
- New FC layer needs to adapt to audio domain
- Prevent destroying pretrained weights initially

**Expected:**
- Val Acc: 75-85%
- Fast convergence (5-10 epochs)

### **Stage 2: Fine-tune Entire Network (60 epochs)**

**Strategy:** Unfreeze all layers, fine-tune with lower LR

```python
model.unfreeze_backbone()  # Unfreeze all layers
optimizer = Adam(all_params, lr=0.0001)  # 10x lower LR
```

**Why?**
- Adapt low-level features to audio domain
- Fine-grained optimization
- Lower LR prevents catastrophic forgetting

**Expected:**
- Val Acc: 85-92%
- Slower but steady improvement

---

## 🎯 Expected Results

### Performance Comparison:

| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| SVM | 76.25% | 2 min | ~200 features |
| CNN (Custom) | 80.25% | ~100 epochs (30 min) | 1.28M |
| **ResNet18** | **88-92%** | ~80 epochs (45 min) | 11.2M |

### Improvement:
- vs SVM: **+12-16%** 
- vs Custom CNN: **+8-12%**

---

## 💻 Usage

### 1. **Run Training**

```bash
python resnet_model.py
```

### 2. **Training Flow**

```
Stage 1 (20 epochs):
  ├─ Freeze backbone
  ├─ Train FC layer
  ├─ Save best_resnet_stage1.pth
  └─ Val Acc: ~80%

Stage 2 (60 epochs):
  ├─ Unfreeze backbone
  ├─ Fine-tune all layers
  ├─ Save best_resnet18_model.pth
  └─ Val Acc: ~88-92%

Evaluation:
  ├─ Load best model
  ├─ Test on 400 samples
  └─ Generate confusion matrix
```

### 3. **Monitor Training**

**Stage 1 indicators:**
- Train Acc increases quickly (70% → 85%)
- Val Acc follows closely (65% → 80%)
- Should stabilize after 10-15 epochs

**Stage 2 indicators:**
- Train Acc increases slowly (85% → 95%)
- Val Acc improves steadily (80% → 88-92%)
- Gap should be small (<5-7%)

---

## 🔧 Key Features

### 1. **SpecAugment Integration**

```python
class AudioDataset(Dataset):
    def __init__(self, spectrograms, labels, apply_specaugment=False):
        if apply_specaugment:
            self.spec_augment = SpecAugment(
                freq_mask_param=25,  # Slightly stronger than CNN
                time_mask_param=25,
                n_freq_masks=2,
                n_time_masks=2
            )
```

**Applied to training set only** → Reduce overfitting

### 2. **Dropout Regularization**

```python
self.resnet.fc = nn.Sequential(
    nn.Dropout(0.5),      # 50% dropout
    nn.Linear(512, 50)
)
```

**Prevents overfitting** in final layer

### 3. **Gradient Clipping**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Stabilizes training**, prevents exploding gradients

### 4. **Learning Rate Scheduling**

```python
ReduceLROnPlateau(optimizer, factor=0.5, patience=7)
```

**Adaptive LR** based on validation loss plateau

---

## 📊 Output Files

```
best_resnet18_model.pth           # Best model (full fine-tuned)
best_resnet_stage1.pth            # Stage 1 checkpoint
confusion_matrix_ResNet18.png     # Test set confusion matrix
training_history_ResNet18.png     # Training curves
resnet_model_info.txt             # Model summary
```

---

## 🎨 Visualization Features

### Training History Plot:

```
Accuracy Plot:
├─ Train accuracy curve
├─ Val accuracy curve
└─ Stage 1 → Stage 2 divider (vertical line)

Loss Plot:
├─ Train loss curve
├─ Val loss curve
└─ Stage 1 → Stage 2 divider
```

**Red dashed line** shows transition between stages

---

## 🚀 Inference

### Load and Use Model:

```python
from resnet_model import AudioResNet18
import torch

# Load model
model = AudioResNet18(num_classes=50, pretrained=False)
model.load_state_dict(torch.load('best_resnet18_model.pth'))
model.eval()

# Predict
with torch.no_grad():
    # spec shape: (1, 3, 128, 128) - 3 channels!
    output = model(spec)
    pred = output.argmax(dim=1)
```

**Important:** Input must be **3-channel RGB**!

---

## 💡 Tips & Tricks

### 1. **Data Caching**

Script automatically caches extracted features:
```
cache/
  ├─ X_train.npy
  ├─ y_train.npy
  ├─ X_val.npy
  ├─ y_val.npy
  ├─ X_test.npy
  └─ y_test.npy
```

**Benefit:** Rerun training instantly (no feature extraction)

### 2. **GPU Memory**

ResNet18 uses ~1.5GB GPU memory with batch_size=32

**If OOM (Out of Memory):**
```python
BATCH_SIZE = 16  # Reduce from 32
```

### 3. **Training Time**

- Stage 1: ~10 minutes (20 epochs)
- Stage 2: ~35 minutes (60 epochs)
- **Total: ~45 minutes**

### 4. **Early Stopping**

```python
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs
```

**Adjust based on time constraints**

---

## 🎯 Expected Performance

### Typical Training Curve:

```
Epoch 1-5 (Stage 1):
  Train: 50% → 75%
  Val:   45% → 70%

Epoch 6-20 (Stage 1):
  Train: 75% → 85%
  Val:   70% → 80%

Epoch 21-40 (Stage 2):
  Train: 85% → 90%
  Val:   80% → 86%

Epoch 41-80 (Stage 2):
  Train: 90% → 95%
  Val:   86% → 88-92% ✅
```

### Final Metrics:

```
✅ Train Acc: 93-96%
✅ Val Acc:   88-92%
✅ Test Acc:  88-92%
✅ Gap:       4-6% (healthy)
```

---

## 🔬 Why ResNet18 > Custom CNN?

| Feature | Custom CNN | ResNet18 |
|---------|-----------|----------|
| **Depth** | 4 blocks | 18 layers |
| **Parameters** | 1.28M | 11.2M |
| **Pretrained** | ❌ No | ✅ Yes (ImageNet) |
| **Skip Connections** | ❌ No | ✅ Yes (residual) |
| **Feature Quality** | Limited | Rich (pretrained) |
| **Accuracy** | 80% | **88-92%** |

**Key advantage:** ImageNet pretrained weights contain universal visual patterns applicable to spectrograms!

---

## 📝 Troubleshooting

### Problem: Val Acc không tăng trong Stage 2

**Solution:**
```python
# Tăng learning rate slightly
optimizer_stage2 = optim.Adam(model.parameters(), lr=0.0002)  # từ 0.0001
```

### Problem: Overfitting (Train >> Val)

**Solution:**
```python
# Tăng SpecAugment
freq_mask_param=30  # từ 25
n_freq_masks=3      # từ 2

# Hoặc tăng Dropout
dropout_rate=0.6  # từ 0.5
```

### Problem: Underfitting (cả Train & Val thấp)

**Solution:**
```python
# Giảm regularization
dropout_rate=0.3  # từ 0.5
weight_decay=1e-5  # từ 1e-4
```

---

## 🏆 Summary

**ResNet18 Transfer Learning là BEST CHOICE cho audio classification vì:**

✅ **Accuracy cao nhất** (88-92%)  
✅ **Training stable** (2-stage strategy)  
✅ **Pretrained weights** (ImageNet boost)  
✅ **Production-ready** (proven architecture)  
✅ **Code clean** (well-structured)  

**Đây là professional solution cho production deployment!** 🚀

---

**Updated:** 2025-10-27  
**Author:** ResNet18 Transfer Learning Implementation

