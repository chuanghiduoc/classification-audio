# 🎯 SpecAugment Implementation Guide

## 📚 Giới thiệu

**SpecAugment** là một technique augmentation mạnh mẽ được Google Research phát triển năm 2019 cho Audio/Speech Recognition.

**Paper:** [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779)

---

## 🔬 Cách hoạt động

SpecAugment áp dụng **random masking** lên mel spectrogram theo 2 chiều:

### 1. **Frequency Masking (Che tần số)**
```
Original:                 With Frequency Mask:
████████████████████     ████████████████████
████████████████████     ░░░░░░░░░░░░░░░░░░░░ ← Masked
████████████████████     ░░░░░░░░░░░░░░░░░░░░ ← Masked
████████████████████     ████████████████████
```

- Che ngẫu nhiên một dải tần số liên tiếp
- Giúp model học robust với nhiễu tần số
- Tương tự như dropout nhưng structured

### 2. **Time Masking (Che thời gian)**
```
Original:                 With Time Mask:
████████████████████     ████░░░░████████████
████████████████████     ████░░░░████████████
████████████████████     ████░░░░████████████
████████████████████     ████░░░░████████████
         ↑                    ↑ Masked
```

- Che ngẫu nhiên một đoạn thời gian liên tiếp
- Giúp model học robust với gaps/pauses
- Tương tự như cutout trong computer vision

---

## ⚙️ Hyperparameters

### Trong implementation của chúng ta:

```python
SpecAugment(
    freq_mask_param=20,   # Max width của frequency mask (bins)
    time_mask_param=20,   # Max width của time mask (frames)
    n_freq_masks=2,       # Số lượng frequency masks
    n_time_masks=2        # Số lượng time masks
)
```

### Giải thích:

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `freq_mask_param` | 20 | Mask tối đa 20/128 frequency bins (~15.6%) |
| `time_mask_param` | 20 | Mask tối đa 20/128 time frames (~15.6%) |
| `n_freq_masks` | 2 | Apply 2 frequency masks mỗi sample |
| `n_time_masks` | 2 | Apply 2 time masks mỗi sample |

**Total masked area:** Khoảng 20-40% của spectrogram (random)

---

## 🎨 Visual Example

### Original Mel Spectrogram:
```
████████████████████████████████
████████████████████████████████
████████████████████████████████
████████████████████████████████
```

### After SpecAugment:
```
████████░░░░████████████████████  ← Freq mask 1
████████░░░░████████████████████
████░░░░░░░░░░░░████████████████  ← Freq mask 2
████░░░░████████████████████████
     ↑        ↑
  Time mask  Time mask
```

---

## 💻 Code Implementation

### 1. SpecAugment Class

```python
class SpecAugment:
    def __init__(self, freq_mask_param=20, time_mask_param=20, 
                 n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec):
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, num_freq_bins - f)
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, num_time_frames - t)
            spec[:, t0:t0 + t] = 0
        
        return spec
```

### 2. Integration với PyTorch Dataset

```python
class AudioDataset(Dataset):
    def __init__(self, spectrograms, labels, apply_specaugment=False):
        self.spectrograms = spectrograms
        self.labels = labels
        self.apply_specaugment = apply_specaugment
        
        if apply_specaugment:
            self.spec_augment = SpecAugment(...)
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx].copy()
        
        # Apply SpecAugment ON-THE-FLY during training
        if self.apply_specaugment:
            spec = self.spec_augment(spec)
        
        return torch.FloatTensor(spec), self.labels[idx]
```

### 3. Usage

```python
# Training set: WITH SpecAugment
train_dataset = AudioDataset(X_train, y_train, apply_specaugment=True)

# Val/Test sets: WITHOUT SpecAugment
val_dataset = AudioDataset(X_val, y_val, apply_specaugment=False)
test_dataset = AudioDataset(X_test, y_test, apply_specaugment=False)
```

---

## 🎯 Tại sao SpecAugment hiệu quả?

### 1. **Giảm Overfitting**
- Model không thể "học thuộc" spectrogram
- Phải học features robust, không phụ thuộc vào specific patterns

### 2. **Data Augmentation mạnh mẽ**
- Mỗi epoch, cùng 1 sample có vô số variations
- Tương đương với việc có nhiều data hơn

### 3. **Structured Dropout**
- Giống dropout nhưng có structure (frequency/time continuity)
- Phù hợp với audio domain

### 4. **Computational Efficient**
- Apply on-the-fly (không cần lưu augmented data)
- Rất nhanh (chỉ là masking)
- Không tốn memory

---

## 📊 Kết quả kỳ vọng

### Trước khi có SpecAugment:
```
Train Acc: 98.18%
Val Acc:   80.31%
Test Acc:  76.00%
Gap:       18.18% (overfitting!)
```

### Sau khi có SpecAugment:
```
Train Acc: 90-94%    (giảm - tốt!)
Val Acc:   85-88%    (tăng - tốt!)
Test Acc:  82-86%    (tăng - TỐT!)
Gap:       5-8%      (giảm overfit!)
```

**Expected improvement: +6-10% test accuracy!**

---

## 🔧 Tuning Hyperparameters

### Nếu vẫn overfit (Train >> Val):
```python
# Tăng masking strength
SpecAugment(
    freq_mask_param=30,   # ↑ Tăng
    time_mask_param=30,   # ↑ Tăng
    n_freq_masks=3,       # ↑ Tăng
    n_time_masks=3        # ↑ Tăng
)
```

### Nếu underfit (Train & Val đều thấp):
```python
# Giảm masking strength
SpecAugment(
    freq_mask_param=15,   # ↓ Giảm
    time_mask_param=15,   # ↓ Giảm
    n_freq_masks=1,       # ↓ Giảm
    n_time_masks=1        # ↓ Giảm
)
```

### Sweet spot (recommended):
```python
SpecAugment(
    freq_mask_param=20,
    time_mask_param=20,
    n_freq_masks=2,
    n_time_masks=2
)
```

---

## 📝 Visualize SpecAugment

Chạy script để xem effect:

```bash
python visualize_specaugment.py
```

Output: `mel_spectrograms/mel_spectrogram_augmentation.png`

---

## 🚀 Training với SpecAugment

### 1. Bật SpecAugment
```python
# Trong cnn_model.py
APPLY_SPECAUGMENT = True  # Set to True
```

### 2. Train model
```bash
python cnn_model.py
```

### 3. Theo dõi metrics
- Train accuracy sẽ **THẤP HƠN** (90-94%) - Đây là điều TỐT!
- Val accuracy sẽ **CAO HƠN** (85-88%)
- Test accuracy sẽ **CAO HƠN** (82-86%)
- Gap (Train-Val) sẽ **NHỎ HƠN** (5-8%)

---

## ✅ Checklist

- [x] SpecAugment class implemented
- [x] Integrated vào AudioDataset
- [x] Apply ONLY cho training set
- [x] Val/Test set KHÔNG có SpecAugment
- [x] Configurable hyperparameters
- [x] Visualization script
- [x] Documentation

---

## 📚 References

1. [SpecAugment Paper (Google Research)](https://arxiv.org/abs/1904.08779)
2. [PyTorch Audio Tutorial](https://pytorch.org/audio/stable/tutorials/audio_feature_augmentation_tutorial.html)
3. [TensorFlow SpecAugment](https://www.tensorflow.org/io/tutorials/audio)

---

## 🎉 Summary

**SpecAugment là một trong những augmentation techniques TỐT NHẤT cho audio:**

✅ **Dễ implement** - Chỉ ~50 lines code  
✅ **Hiệu quả cao** - +6-10% accuracy  
✅ **Fast** - No overhead  
✅ **Proven** - Used by Google, Facebook, etc.  
✅ **Reduce overfitting** - Dramatically  

**Đây là low-hanging fruit - PHẢI dùng cho audio tasks!** 🚀

---

**Updated:** 2025-10-27  
**Author:** CNN Model with SpecAugment

