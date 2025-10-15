# 📊 TIỀN XỬ LÝ DỮ LIỆU & QUY TRÌNH CNN

**File:** `cnn_model.py`  
**Dataset (Tập dữ liệu):** ESC-50 (2000 audio files - file âm thanh, 50 classes - lớp)  
**Mục tiêu:** Phân loại âm thanh với CNN đạt 98.62% accuracy (độ chính xác)

---

## 🔄 **QUY TRÌNH TỔNG QUAN**

```
Audio Files - File âm thanh (.wav)
    ↓
[1] Load Audio + Resampling (Tải âm thanh + Lấy mẫu lại)
    ↓
[2] Padding/Trimming (Đệm/Cắt)
    ↓
[3] Data Augmentation - Tăng cường dữ liệu (6x)
    ↓
[4] Mel Spectrogram - Phổ tần Mel
    ↓
[5] Log Scaling - Biến đổi logarit
    ↓
[6] Normalization - Chuẩn hóa
    ↓
[7] Resize - Thay đổi kích thước
    ↓
[8] Train/Val/Test Split - Chia tập huấn luyện/kiểm tra/đánh giá
    ↓
[9] PyTorch Tensor - Chuyển đổi sang tensor
    ↓
CNN Model (Training) - Mô hình CNN (Huấn luyện)
    ↓
Best Model - Mô hình tốt nhất (98.62%)
```

---

## 📝 **CHI TIẾT TỪNG BƯỚC**

### **[1] LOAD AUDIO + RESAMPLING (TẢI ÂM THANH + LẤY MẪU LẠI)**

```python
y, sr = librosa.load(file_path, sr=22050, duration=5.0)
```

**Mục đích:**
- Đọc file audio thành waveform - dạng sóng (1D signal - tín hiệu 1 chiều)
- Chuẩn hóa sampling rate - tần số lấy mẫu: 22050 Hz
- Lấy 5 giây đầu tiên (duration - độ dài)

**Input (Đầu vào):** `dog_barking.wav` (audio file - file âm thanh)  
**Output (Đầu ra):** 
- `y`: (110250,) - waveform array (mảng dạng sóng)
- `sr`: 22050 Hz - sampling rate (tần số lấy mẫu)

**Lý do chọn 22050 Hz:**
- Nyquist theorem - Định lý Nyquist: Capture (bắt) tần số tới 11025 Hz
- Đủ cho âm thanh tự nhiên (human hearing - thính giác con người: 20-20000 Hz)
- Balance (cân bằng) giữa quality (chất lượng) và computational cost (chi phí tính toán)

---

### **[2] PADDING/TRIMMING (ĐỆM/CẮT)**

```python
if len(y) < sr * 5:
    y = np.pad(y, (0, sr * 5 - len(y)), mode='constant')
```

**Mục đích:**
- Đảm bảo mọi audio đều **đúng 5 giây** (110250 samples - mẫu)
- Audio ngắn hơn → Pad (đệm) zeros (số 0) vào cuối
- Audio dài hơn → Đã trim (cắt) ở bước [1]

**Ví dụ:**
```
Audio 3s = 66150 samples (mẫu)
Target (mục tiêu) = 110250 samples
→ Pad (đệm) thêm 44100 zeros (số 0)
```

**Tại sao cần cố định length (độ dài)?**
- CNN yêu cầu input shape (kích thước đầu vào) cố định
- Batch processing (xử lý theo lô) cần samples (mẫu) cùng size (kích thước)

---

### **[3] DATA AUGMENTATION - TĂNG CƯỜNG DỮ LIỆU (6x)**

**⭐ Kỹ thuật quan trọng để tăng accuracy (độ chính xác)!**

```python
# Original
mel_spec_original

# 1. Time Stretch (slow) - rate=0.9
y_slow = librosa.effects.time_stretch(y, rate=0.9)

# 2. Time Stretch (fast) - rate=1.1
y_fast = librosa.effects.time_stretch(y, rate=1.1)

# 3. Pitch Shift (+2 semitones)
y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

# 4. Pitch Shift (-2 semitones)
y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)

# 5. Add Gaussian Noise
noise = np.random.randn(len(y)) * 0.005
y_noise = y + noise
```

**Kết quả:**
```
1 audio file → 6 variations
2000 files × 6 = 12000 samples
```

**Lợi ích:**
- ✅ Tăng data (dữ liệu) 6x (2000 → 12000)
- ✅ Model (mô hình) học được variations - biến thể (tempo - nhịp, pitch - cao độ, noise - nhiễu)
- ✅ Giảm overfitting - quá khớp đáng kể
- ✅ Robust - ổn định với variations trong real-world - thế giới thực

**Chi tiết từng augmentation (tăng cường):**

| Augmentation (Tăng cường) | Mô tả | Use Case (Trường hợp sử dụng) |
|--------------|-------|----------|
| **Time Stretch (0.9) - Kéo giãn thời gian** | Làm chậm 10% | Tiếng chó sủa chậm, người nói chậm |
| **Time Stretch (1.1) - Kéo giãn thời gian** | Làm nhanh 10% | Tiếng chó sủa nhanh, người nói nhanh |
| **Pitch Shift (+2) - Dịch chuyển cao độ** | Tăng cao độ 2 nửa cung (semitones) | Tiếng chó nhỏ, giọng nữ |
| **Pitch Shift (-2) - Dịch chuyển cao độ** | Giảm cao độ 2 nửa cung | Tiếng chó to, giọng nam |
| **Gaussian Noise - Nhiễu Gauss** | Thêm nhiễu ngẫu nhiên (σ=0.005) | Môi trường ồn, nhiễu nền |

---

### **[4] MEL SPECTROGRAM - PHỔ TẦN MEL**

```python
mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=128,      # 128 frequency bins
    n_fft=2048,      # FFT window size
    hop_length=512,  # Overlap step
    fmax=8000        # Max frequency
)
```

**Mục đích:**
- Chuyển waveform - dạng sóng (1D) → Spectrogram - phổ tần (2D time-frequency - thời gian-tần số)
- Dùng Mel scale - thang Mel (gần với cách tai người nghe)

**Transformation (Chuyển đổi):**
```
Input (Đầu vào):  y = (110250,) waveform (dạng sóng)
Output (Đầu ra): mel_spec = (128, 216) time-frequency matrix (ma trận thời gian-tần số)
```

**Parameters (Tham số):**

| Parameter (Tham số) | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `n_mels` | 128 | 128 mel frequency bins - khoảng tần số mel (trục Y) |
| `n_fft` | 2048 | FFT window size - kích thước cửa sổ FFT (độ phân giải frequency - tần số) |
| `hop_length` | 512 | Bước nhảy giữa windows - cửa sổ (overlap - chồng lấn) |
| `fmax` | 8000 Hz | Frequency (tần số) tối đa (loại bỏ high freq noise - nhiễu tần số cao) |

**Mel Scale vs Linear (Thang Mel vs Tuyến tính):**
```
Linear (Tuyến tính): 1000 Hz, 2000 Hz, 3000 Hz, 4000 Hz
Mel:                 1000 Hz, 1500 Hz, 2000 Hz, 2500 Hz

→ Mel scale: Dày ở tần số thấp, thưa ở tần số cao
→ Giống cách tai người nghe!
```

---

### **[5] LOG SCALING - BIẾN ĐỔI LOGARIT (dB - DECIBEL)**

```python
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

**Mục đích:**
- Convert (chuyển đổi) power - công suất → decibel - đề-xi-ben (dB)
- Giảm dynamic range - khoảng động
- Logarithmic scale - thang logarit (giống cách tai người nghe)

**Công thức:**
```
dB = 10 × log10(power / reference)
     (công suất / tham chiếu)
```

**Hiệu quả:**
```
Before (Trước) - power (công suất):
  Range (khoảng): 0.001 - 10 (10000x difference - chênh lệch!)
  
After (Sau) - dB:
  Range (khoảng): -30 dB to +10 dB (40 dB range)
  → Dễ xử lý cho CNN!
```

---

### **[6] NORMALIZATION - CHUẨN HÓA [0, 1]**

```python
mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
```

**Mục đích:**
- Chuẩn hóa (normalize) giá trị về [0, 1]
- Giúp CNN training - huấn luyện ổn định
- Faster convergence - hội tụ nhanh hơn, tránh gradient issues - vấn đề gradient

**Công thức:**
```
normalized (chuẩn hóa) = (x - min) / (max - min)
```

**Ví dụ:**
```
Before (Trước): [-80, -60, -40, -20, 0] dB
After (Sau):    [0, 0.25, 0.5, 0.75, 1.0]
```

**Lợi ích:**
- ✅ CNN weights - trọng số học nhanh hơn với [0,1]
- ✅ Tránh vanishing/exploding gradients - gradient tiêu biến/bùng nổ
- ✅ Stable training - huấn luyện ổn định

---

### **[7] RESIZE - THAY ĐỔI KÍCH THƯỚC TO 128×128**

```python
mel_spec_resized = cv2.resize(mel_spec_norm, (128, 128))
```

**Mục đích:**
- Resize (thay đổi kích thước) từ (128, 216) → (128, 128) square - hình vuông
- Cố định input shape - kích thước đầu vào cho CNN

**Transformation (Chuyển đổi):**
```
Before (Trước): (128 freq - tần số, 216 time - thời gian) - rectangular (hình chữ nhật)
After (Sau):    (128, 128) - square image (ảnh vuông)
```

**Lý do resize:**
- CNN design - thiết kế dễ hơn với square input - đầu vào vuông
- Giảm số parameters - tham số (216 → 128 ở time axis - trục thời gian)
- Standard - chuẩn cho computer vision models - mô hình thị giác máy tính

---

### **[8] TRAIN/VAL/TEST SPLIT - CHIA TẬP HUẤN LUYỆN/KIỂM TRA/ĐÁNH GIÁ**

```python
# Step 1: Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 2: Train → Train/Val split (80/20)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
```

**Kết quả:**
```
Total (Tổng): 12000 samples - mẫu (sau augmentation - tăng cường)

├─ Train (Huấn luyện): 7680 samples (64%)
│  └─ Dùng để train model - huấn luyện mô hình
│
├─ Val (Validation - Kiểm tra):   1920 samples (16%)
│  └─ Monitor overfitting - theo dõi quá khớp, early stopping - dừng sớm, 
│     LR scheduling - lập lịch learning rate
│
└─ Test (Đánh giá):  2400 samples (20%)
   └─ Final evaluation - đánh giá cuối cùng (không động đến trong training - huấn luyện)
```

**Stratified sampling - Lấy mẫu phân tầng:**
- Đảm bảo mỗi class - lớp có tỷ lệ đều trong train/val/test
- Tránh class imbalance - mất cân bằng lớp

---

### **[9] PYTORCH TENSOR - CHUYỂN ĐỔI SANG TENSOR**

```python
# Reshape: (samples, height, width) → (samples, channels, height, width)
X = spectrograms.reshape(-1, 1, IMG_SIZE, IMG_SIZE)

# Convert to PyTorch Tensor
self.spectrograms = torch.FloatTensor(spectrograms)
self.labels = torch.LongTensor(labels)
```

**Transformation (Chuyển đổi):**
```
NumPy:   (12000, 128, 128)
PyTorch: (12000, 1, 128, 128)
         (batch - lô, channels - kênh, height - cao, width - rộng)
```

**Channel dimension - Chiều kênh:**
- Mel spectrogram = Grayscale image - ảnh xám → 1 channel - kênh
- RGB image - ảnh RGB → 3 channels - kênh
- CNN expects - yêu cầu (N, C, H, W) format - định dạng

---

## 🏗️ **CNN ARCHITECTURE - KIẾN TRÚC CNN**

### **Model Overview - Tổng quan mô hình:**

```python
AudioCNN(
  # Block 1: Low-level features - Đặc trưng cấp thấp (edges - cạnh, textures - kết cấu)
  Conv2d(1 → 32) → Conv2d(32 → 32) → MaxPool → BatchNorm → Dropout(0.25)
  
  # Block 2: Mid-level features - Đặc trưng cấp trung (patterns - mẫu)
  Conv2d(32 → 64) → Conv2d(64 → 64) → MaxPool → BatchNorm → Dropout(0.25)
  
  # Block 3: High-level features - Đặc trưng cấp cao (sound signatures - dấu hiệu âm thanh)
  Conv2d(64 → 128) → Conv2d(128 → 128) → MaxPool → BatchNorm → Dropout(0.25)
  
  # Block 4: Abstract features - Đặc trưng trừu tượng (class-specific - đặc thù từng lớp)
  Conv2d(128 → 256) → Conv2d(256 → 256) → AdaptiveAvgPool
  
  # Fully Connected Layers - Lớp kết nối đầy đủ
  FC(256 → 512) → BatchNorm → Dropout(0.5)
  FC(512 → 256) → BatchNorm → Dropout(0.3)
  FC(256 → 50) → Softmax
)
```

**Trainable parameters - Tham số huấn luyện:** 1,449,426

---

## 🎓 **TRAINING STRATEGY - CHIẾN LƯỢC HUẤN LUYỆN**

### **Optimizer - Bộ tối ưu & Loss - Hàm mất mát:**
```python
optimizer = Adam(lr=0.001)  # lr = learning rate - tốc độ học
criterion = CrossEntropyLoss()  # Hàm mất mát phân loại đa lớp
```

### **Learning Rate Scheduling - Lập lịch tốc độ học:**
```python
scheduler = ReduceLROnPlateau(  # Giảm LR khi plateau - đạt bình nguyên
    mode='min',  # Chế độ: tối thiểu hóa
    factor=0.5,     # LR_new (mới) = LR_old (cũ) × 0.5
    patience=5,     # Đợi 5 epochs - kỷ nguyên
    verbose=True  # In thông báo
)
```

**Cơ chế:**
- Monitor (theo dõi) `val_loss` - mất mát kiểm tra mỗi epoch
- Nếu không giảm trong 5 epochs → Giảm LR
- LR: 0.001 → 0.0005 → 0.00025 → ...

### **Regularization - Điều chuẩn:**
1. **Dropout - Bỏ qua ngẫu nhiên:** 0.25 (Conv blocks - khối tích chập), 0.5 (FC1), 0.3 (FC2)
2. **BatchNormalization - Chuẩn hóa theo lô:** Sau mỗi Conv và FC layer - lớp
3. **Data Augmentation - Tăng cường dữ liệu:** 6x (quan trọng nhất!)
4. **Early Stopping - Dừng sớm:** Patience - độ kiên nhẫn = 15 epochs

### **Training Config:**
```python
EPOCHS = 100
BATCH_SIZE = 32
DEVICE = 'cuda' (RTX 3050)
```

---

## 📊 **KẾT QUẢ**

### **Final Performance - Hiệu suất cuối cùng:**
```
Test Accuracy - Độ chính xác kiểm tra:    98.62%
Val Accuracy - Độ chính xác validation:   98.80%
Train Accuracy - Độ chính xác huấn luyện: 98.97%

Train-Val Gap - Khoảng cách: 0.17% (rất thấp → không overfit - quá khớp)
Val-Test Gap - Khoảng cách:  0.18% (generalization - khả năng tổng quát hóa tốt)
```

### **So sánh với Traditional ML - Học máy truyền thống:**

| Model - Mô hình | Accuracy - Độ chính xác | Training Time - Thời gian huấn luyện | Input - Đầu vào |
|-------|----------|---------------|-------|
| SVM (Traditional - Truyền thống) | 76.25% | ~2 phút | 421 handcrafted features - đặc trưng thủ công |
| **CNN (Deep Learning - Học sâu)** | **98.62%** | **~27 phút** | **Raw spectrogram - Phổ tần thô** |
| **Improvement - Cải thiện** | **+29.34%** | - | End-to-end learning - Học đầu-cuối |

---

## 🎯 **TẠI SAO CNN MẠNH HƠN SVM?**

### **1. Input Representation - Biểu diễn đầu vào:**

**SVM:**
```
Audio → Extract MFCC, Mel, Spectral... → 421 features
      → SelectKBest → 200 features
      → SVM → 76.25%

Vấn đề:
- Handcrafted features có thể bỏ sót thông tin
- Fixed features cho mọi loại âm thanh
```

**CNN:**
```
Audio → Mel Spectrogram (128×128 = 16,384 pixels)
      → CNN tự học features
      → 98.62%

Ưu điểm:
- Raw spectrogram giữ toàn bộ thông tin
- CNN tự động học features tối ưu
- Hierarchical learning: low → high level
```

### **2. Feature Learning - Học đặc trưng:**

**SVM:** Features - Đặc trưng do con người thiết kế  
**CNN:** Features tự động học từ data - dữ liệu

```
CNN Layer 1: Edges - cạnh, corners - góc, textures - kết cấu
CNN Layer 2: Frequency bands - dải tần, patterns - mẫu
CNN Layer 3: Sound signatures - dấu hiệu âm thanh (chó sủa, mèo kêu...)
CNN Layer 4: Class-specific representations - biểu diễn đặc thù từng lớp
```

### **3. Data Augmentation - Tăng cường dữ liệu:**

**SVM:** 
- Augmentation - Tăng cường trên features - đặc trưng (khó)
- Sử dụng ADASYN (synthetic samples - mẫu tổng hợp)

**CNN:** 
- Augmentation trên audio (dễ và hiệu quả)
- 6 loại augmentation → 6x data
- Model robust - ổn định với variations - biến thể

---

## 🔑 **NHỮNG ĐIỂM QUAN TRỌNG**

### **✅ Preprocessing - Tiền xử lý tốt nhất:**

1. **Mel Spectrogram - Phổ tần Mel** thay vì raw waveform - dạng sóng thô
   - Giảm dimensionality - số chiều: 110250 → 128×128
   - Perceptually relevant - liên quan tri giác (Mel scale - thang Mel)
   - Capture - bắt time-frequency patterns - mẫu thời gian-tần số

2. **Log scaling - Biến đổi logarit (dB)**
   - Giảm dynamic range - khoảng động
   - Giống cách tai người nghe

3. **Normalization - Chuẩn hóa [0,1]**
   - Stable training - huấn luyện ổn định
   - Faster convergence - hội tụ nhanh hơn

4. **Data Augmentation - Tăng cường dữ liệu 6x**
   - Tăng data 2000 → 12000
   - Giảm overfitting - quá khớp
   - Robust model - mô hình ổn định

### **✅ Training strategy - Chiến lược huấn luyện tốt nhất:**

1. **Learning Rate Scheduling - Lập lịch tốc độ học**
   - Adaptive LR - LR thích ứng: 0.001 → 0.0005 → ...
   - Converge - hội tụ chính xác hơn

2. **Regularization - Điều chuẩn nhiều lớp**
   - Dropout + BatchNorm + Augmentation
   - Train 98.97%, Val 98.80% (gap - khoảng cách chỉ 0.17%)

3. **Early Stopping - Dừng sớm**
   - Tránh waste time - lãng phí thời gian
   - Restore best weights - khôi phục trọng số tốt nhất

---

## 📈 **WORKFLOW THỰC TẾ**

### **Khi train model:**

```bash
$ python cnn_model.py

# Output:
1. Load 2000 audio files
2. Extract Mel Spectrograms
3. Apply augmentation (6x) → 12000 samples
4. Split: Train 7680 | Val 1920 | Test 2400
5. Build CNN (1.4M parameters)
6. Train 100 epochs với:
   - Adam optimizer (lr=0.001)
   - LR scheduling
   - Early stopping
7. Best model: Epoch 99 (Val Acc: 98.80%)
8. Test evaluation: 98.62%
9. Save:
   - best_cnn_model.pth
   - confusion_matrix_CNN.png
   - training_history_CNN.png
   - predictions_CNN.png
   - cnn_model_info.txt
```

### **Training time:**
- ~17s/epoch × 100 epochs = ~27 phút (với RTX 3050)
- CPU: ~2-3 giờ

---

## 🚀 **TIPS TỐI ƯU HÓA - MẸO TỐI ƯU**

### **Nếu accuracy - độ chính xác thấp (<90%):**

1. **Tăng augmentation - tăng cường:**
   - Thêm time shift - dịch thời gian, volume change - thay đổi âm lượng
   - Frequency masking - che tần số (SpecAugment)

2. **Train - huấn luyện lâu hơn:**
   - EPOCHS = 150
   - Early stopping patience - độ kiên nhẫn = 20

3. **Tăng model capacity - dung lượng mô hình:**
   - Thêm 1 Conv block - khối tích chập
   - Tăng filters - bộ lọc: 32→64, 64→128...

### **Nếu overfitting - quá khớp:**

1. **Tăng regularization - điều chuẩn:**
   - Dropout 0.5 → 0.6-0.7
   - L2 weight decay - phân rã trọng số

2. **Tăng augmentation - tăng cường:**
   - 6x → 9x hoặc 12x

3. **Giảm model size - kích thước mô hình:**
   - Giảm filters - bộ lọc: 32→16, 64→32...

### **Nếu training - huấn luyện chậm:**

1. **Giảm batch size - kích thước lô:** 32 → 16
2. **Giảm IMG_SIZE - kích thước ảnh:** 128 → 64
3. **Dùng GPU mạnh hơn**

---

## 📚 **TÀI LIỆU THAM KHẢO**

### **Code structure:**
- `cnn_model.py`: Main file
- `best_cnn_model.pth`: Trained weights
- `confusion_matrix_CNN.png`: Confusion matrix
- `training_history_CNN.png`: Training curves

### **Libraries:**
- `librosa`: Audio processing
- `PyTorch`: Deep learning framework
- `cv2`: Image resizing
- `sklearn`: Train/test split

### **Dataset - Tập dữ liệu:**
- **ESC-50:** Environmental Sound Classification - Phân loại âm thanh môi trường
- 2000 audio files - file âm thanh, 50 classes - lớp
- 5 seconds - giây each - mỗi file, 44.1 kHz

---

**Tác giả:** AI Assistant  
**Ngày:** 2025-10-15  
**Version:** 1.0 (PyTorch CNN)

