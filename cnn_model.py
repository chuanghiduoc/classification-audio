# -*- coding: utf-8 -*-
"""
PHÂN LOẠI ÂM THANH BẰNG CNN - DEEP LEARNING
Sử dụng Mel Spectrogram làm input cho CNN 2D với PyTorch
Mục tiêu: 85-92% accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

# =============================================================================
# CẤU HÌNH
# =============================================================================

DATA_PATH = 'data/audio/audio/'
CSV_PATH = 'data/esc50.csv'
IMG_SIZE = 128
APPLY_AUGMENTATION = True  # Data augmentation (time/pitch shift, noise)
APPLY_SPECAUGMENT = True  # SpecAugment (frequency/time masking) 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("DEEP LEARNING - CNN MODEL (PyTorch)")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: Not available (using CPU)")

# =============================================================================
# SPECAUGMENT - FREQUENCY & TIME MASKING
# =============================================================================

class SpecAugment:
    """
    SpecAugment: Phương pháp tăng cường dữ liệu đơn giản cho ASR (Google Research)
    https://arxiv.org/abs/1904.08779
    
    Áp dụng masking ngẫu nhiên lên các chiều tần số và thời gian của mel spectrogram.
    Giúp giảm overfitting và cải thiện khả năng tổng quát hóa của mô hình.
    """
    def __init__(self, freq_mask_param=20, time_mask_param=20, n_freq_masks=2, n_time_masks=2):
        """
        Khởi tạo SpecAugment
        
        Args:
            freq_mask_param: Độ rộng tối đa của frequency mask (số bins tần số)
            time_mask_param: Độ rộng tối đa của time mask (số frames thời gian)
            n_freq_masks: Số lượng frequency masks sẽ áp dụng
            n_time_masks: Số lượng time masks sẽ áp dụng
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec):
        """
        Áp dụng SpecAugment lên mel spectrogram
        
        Args:
            spec: Mảng mel spectrogram với định dạng (H, W) hoặc (C, H, W)
                  H = chiều cao (frequency bins)
                  W = chiều rộng (time frames)
                  C = channels (thường là 1)
            
        Returns:
            Mel spectrogram đã được augment (có các vùng bị mask = 0)
        """
        spec = spec.copy()  # Tạo bản sao để không ảnh hưởng dữ liệu gốc
        
        # Xử lý cả định dạng (H, W) và (C, H, W)
        if len(spec.shape) == 3:
            spec = spec.squeeze(0)  # Bỏ chiều channel nếu có
            had_channel = True
        else:
            had_channel = False
        
        num_freq_bins, num_time_frames = spec.shape
        
        # Frequency masking: Che ngẫu nhiên các dải tần số
        # Giúp model học được các đặc trưng tần số đa dạng hơn
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)  # Độ rộng mask ngẫu nhiên
            f0 = np.random.randint(0, num_freq_bins - f)    # Vị trí bắt đầu mask
            spec[f0:f0 + f, :] = 0  # Set các giá trị trong vùng mask = 0
        
        # Time masking: Che ngẫu nhiên các khung thời gian
        # Giúp model học được các đặc trưng thời gian đa dạng hơn
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)  # Độ rộng mask ngẫu nhiên
            t0 = np.random.randint(0, num_time_frames - t)  # Vị trí bắt đầu mask
            spec[:, t0:t0 + t] = 0  # Set các giá trị trong vùng mask = 0
        
        # Khôi phục chiều channel nếu ban đầu có
        if had_channel:
            spec = np.expand_dims(spec, axis=0)
        
        return spec

# =============================================================================
# HÀM TRÍCH XUẤT MEL SPECTROGRAM
# =============================================================================

def extract_mel_spectrogram(file_path, img_size=128, augment=False):
    """
    Trích xuất Mel Spectrogram từ file audio và chuyển thành ảnh 2D
    
    Quy trình: Audio (1D) → STFT → Mel filterbank → Log scale → Normalize → Resize → Ảnh 2D
    
    Args:
        file_path: Đường dẫn đến file audio (.wav)
        img_size: Kích thước ảnh output (mặc định 128x128)
        augment: Nếu True, sẽ tạo thêm 5 phiên bản augmented (tổng 6 ảnh từ 1 audio)
    
    Returns:
        - Nếu augment=False: Trả về 1 mảng numpy (128, 128)
        - Nếu augment=True: Trả về list gồm 6 mảng numpy (mỗi mảng 128x128)
            1. Original (gốc)
            2. Time stretch slow (chậm 0.9x)
            3. Time stretch fast (nhanh 1.1x)
            4. Pitch shift up (+2 bán cung)
            5. Pitch shift down (-2 bán cung)
            6. Gaussian noise (thêm nhiễu Gaussian)
    """
    try:
        # Bước 1: Load audio với sample rate 22050 Hz, cắt về 5 giây
        # sr = sample rate (số mẫu mỗi giây)
        # y = mảng numpy chứa dữ liệu âm thanh
        y, sr = librosa.load(file_path, sr=22050, duration=5.0)
        
        # Bước 2: Nếu audio ngắn hơn 5s → thêm zeros vào cuối để đủ 5s
        # Đảm bảo tất cả audio có cùng độ dài cho CNN
        if len(y) < sr * 5:
            y = np.pad(y, (0, sr * 5 - len(y)), mode='constant')
        
        results = []  # Danh sách lưu các mel spectrogram
        
        # Bước 3: Tạo mel spectrogram cho audio gốc
        mel_spec = create_mel_spectrogram(y, sr, img_size)
        results.append(mel_spec)
        
        # Bước 4: Data Augmentation - Tạo thêm 5 phiên bản biến thể
        # Giúp tăng số lượng dữ liệu train, giảm overfitting
        if augment:
            # 1. Time Stretching (chậm lại 0.9x): Giống người nói chậm hơn
            y_slow = librosa.effects.time_stretch(y, rate=0.9)
            mel_spec_slow = create_mel_spectrogram(y_slow, sr, img_size)
            results.append(mel_spec_slow)
            
            # 2. Time Stretching (nhanh lên 1.1x): Giống người nói nhanh hơn
            y_fast = librosa.effects.time_stretch(y, rate=1.1)
            mel_spec_fast = create_mel_spectrogram(y_fast, sr, img_size)
            results.append(mel_spec_fast)
            
            # 3. Pitch Shifting (+2 semitones): Tăng cao độ lên (giọng cao hơn)
            y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
            mel_spec_pitch_up = create_mel_spectrogram(y_pitch_up, sr, img_size)
            results.append(mel_spec_pitch_up)
            
            # 4. Pitch Shifting (-2 semitones): Giảm cao độ xuống (giọng thấp hơn)
            y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
            mel_spec_pitch_down = create_mel_spectrogram(y_pitch_down, sr, img_size)
            results.append(mel_spec_pitch_down)
            
            # 5. Thêm Gaussian Noise: Mô phỏng nhiễu môi trường (tiếng ồn nền)
            noise = np.random.randn(len(y)) * 0.005  # Nhiễu nhỏ (σ=0.005)
            y_noise = y + noise
            mel_spec_noise = create_mel_spectrogram(y_noise, sr, img_size)
            results.append(mel_spec_noise)
        
        # Trả về list nếu augment, hoặc mảng đơn nếu không augment
        return results if augment else results[0]
        
    except Exception as e:
        print(f"Loi: {file_path}: {e}")
        return None

def create_mel_spectrogram(y, sr, img_size):
    """
    Tạo Mel Spectrogram từ audio signal và chuyển thành ảnh chuẩn hóa
    
    Args:
        y: Audio time series (mảng 1D)
        sr: Sample rate (Hz)
        img_size: Kích thước ảnh output (128x128)
    
    Returns:
        Mel spectrogram đã chuẩn hóa về [0,1] với kích thước (img_size, img_size)
    """
    # Bước 1: Tính Mel Spectrogram
    # n_mels=128: Số lượng mel bands (chiều cao của spectrogram)
    # n_fft=2048: Kích thước FFT window (độ phân giải tần số)
    # hop_length=512: Số mẫu giữa các frame liên tiếp (độ phân giải thời gian)
    # fmax=8000: Tần số tối đa cần phân tích (8kHz đủ cho âm thanh môi trường)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,      # 128 mel frequency bands
        n_fft=2048,      # Window size cho FFT
        hop_length=512,  # Bước nhảy giữa các window
        fmax=8000        # Tần số tối đa
    )
    
    # Bước 2: Chuyển sang thang đo dB (decibel) - thang logarithmic
    # Giống cách tai người nghe âm thanh (phi tuyến tính)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Bước 3: Chuẩn hóa về khoảng [0, 1]
    # Giúp CNN học tốt hơn (input trong khoảng chuẩn)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Bước 4: Resize về kích thước cố định (128x128)
    # Đảm bảo tất cả ảnh có cùng kích thước cho CNN
    mel_spec_resized = cv2.resize(mel_spec_norm, (img_size, img_size))
    
    return mel_spec_resized

# =============================================================================
# ĐỌC DỮ LIỆU - CHIA TRAIN/VAL/TEST TRƯỚC KHI AUGMENT
# =============================================================================

print("\n" + "="*70)
print("TRICH XUAT MEL SPECTROGRAMS")
print("="*70)

df = pd.read_csv(CSV_PATH)
print(f"Dataset: {len(df)} samples, {df['target'].nunique()} classes")

# BƯỚC 1: Chia Train/Test trước (80/20)
train_val_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['target']
)

# BƯỚC 2: Chia Train/Val từ phần train_val (80/20 của 80% = 64% train, 16% val)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['target']
)

print(f"\n=> Split:")
print(f"   Train: {len(train_df)} files ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Val:   {len(val_df)} files ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test:  {len(test_df)} files ({len(test_df)/len(df)*100:.1f}%)")

# BƯỚC 3: Xử lý TRAINING set (có augmentation)
print(f"\n[1/3] Xu ly TRAIN set ({len(train_df)} files)...")
if APPLY_AUGMENTATION:
    print("      Augmentation: BAT (6x)")

train_spectrograms = []
train_labels = []

for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
    file_path = os.path.join(DATA_PATH, row['filename'])
    label = row['target']
    
    result = extract_mel_spectrogram(file_path, IMG_SIZE, augment=APPLY_AUGMENTATION)
    
    if result is not None:
        if APPLY_AUGMENTATION:
            for spec in result:
                train_spectrograms.append(spec)
                train_labels.append(label)
        else:
            train_spectrograms.append(result)
            train_labels.append(label)

# BƯỚC 4: Xử lý VALIDATION set (KHÔNG augmentation)
print(f"\n[2/3] Xu ly VAL set ({len(val_df)} files)...")
print("      Augmentation: TAT")

val_spectrograms = []
val_labels = []

for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
    file_path = os.path.join(DATA_PATH, row['filename'])
    label = row['target']
    
    result = extract_mel_spectrogram(file_path, IMG_SIZE, augment=False)
    
    if result is not None:
        val_spectrograms.append(result)
        val_labels.append(label)

# BƯỚC 5: Xử lý TEST set (KHÔNG augmentation)
print(f"\n[3/3] Xu ly TEST set ({len(test_df)} files)...")
print("      Augmentation: TAT")

test_spectrograms = []
test_labels = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
    file_path = os.path.join(DATA_PATH, row['filename'])
    label = row['target']
    
    result = extract_mel_spectrogram(file_path, IMG_SIZE, augment=False)
    
    if result is not None:
        test_spectrograms.append(result)
        test_labels.append(label)

# Convert to numpy arrays
train_spectrograms = np.array(train_spectrograms)
train_labels = np.array(train_labels)
val_spectrograms = np.array(val_spectrograms)
val_labels = np.array(val_labels)
test_spectrograms = np.array(test_spectrograms)
test_labels = np.array(test_labels)

print(f"\n=> KET QUA:")
print(f"   Train: {train_spectrograms.shape[0]} samples" + (f" (augmented {len(train_df)}x6)" if APPLY_AUGMENTATION else ""))
print(f"   Val:   {val_spectrograms.shape[0]} samples (clean)")
print(f"   Test:  {test_spectrograms.shape[0]} samples (clean)")

# PyTorch format: (samples, channels, height, width)
X_train = train_spectrograms.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
y_train = train_labels
X_val = val_spectrograms.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
y_val = val_labels
X_test = test_spectrograms.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
y_test = test_labels

# =============================================================================
# PYTORCH DATASET & DATALOADER
# =============================================================================

class AudioDataset(Dataset):
    """
    PyTorch Dataset cho audio spectrograms với hỗ trợ SpecAugment
    
    Dataset này tải mel spectrograms và áp dụng SpecAugment trong quá trình training.
    SpecAugment chỉ áp dụng cho tập train, không áp dụng cho val/test.
    
    Args:
        spectrograms: Mảng numpy chứa mel spectrograms, shape: (N, 1, H, W)
                      N = số lượng samples, H = W = img_size (128)
        labels: Mảng numpy chứa nhãn (0-49 cho 50 classes)
        apply_specaugment: True = áp dụng SpecAugment (chỉ dùng cho training)
    """
    def __init__(self, spectrograms, labels, apply_specaugment=False):
        self.spectrograms = spectrograms  # Giữ dạng numpy để augmentation
        self.labels = torch.LongTensor(labels)  # Chuyển labels sang torch tensor
        self.apply_specaugment = apply_specaugment
        
        # Khởi tạo SpecAugment nếu cần (chỉ cho training)
        if apply_specaugment:
            self.spec_augment = SpecAugment(
                freq_mask_param=20,  # Độ rộng tối đa frequency mask (20 bins)
                time_mask_param=20,  # Độ rộng tối đa time mask (20 frames)
                n_freq_masks=2,      # Số lượng frequency masks
                n_time_masks=2       # Số lượng time masks
            )
    
    def __len__(self):
        """Trả về số lượng samples trong dataset"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Lấy 1 sample từ dataset
        
        Args:
            idx: Index của sample cần lấy
        
        Returns:
            spec: Mel spectrogram tensor, shape (1, 128, 128)
            label: Nhãn (0-49)
        """
        spec = self.spectrograms[idx].copy()  # Copy để không ảnh hưởng dữ liệu gốc
        label = self.labels[idx]
        
        # Áp dụng SpecAugment trong quá trình training
        # Mỗi lần lấy sample sẽ random mask khác nhau → tăng tính đa dạng
        if self.apply_specaugment:
            spec = self.spec_augment(spec)
        
        # Chuyển sang PyTorch tensor
        spec = torch.FloatTensor(spec)
        
        return spec, label

# Create datasets with SpecAugment
train_dataset = AudioDataset(X_train, y_train, apply_specaugment=APPLY_SPECAUGMENT)
val_dataset = AudioDataset(X_val, y_val, apply_specaugment=False)  # No augment for val
test_dataset = AudioDataset(X_test, y_test, apply_specaugment=False)  # No augment for test

# Create dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n=> DataLoaders:")
print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
print(f"   Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
print(f"   Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")

if APPLY_SPECAUGMENT:
    print(f"\n=> SpecAugment: ENABLED for training")
    print(f"   - Frequency masks: 2 (max width: 20 bins)")
    print(f"   - Time masks: 2 (max width: 20 frames)")
else:
    print(f"\n=> SpecAugment: DISABLED")

# =============================================================================
# XÂY DỰNG CNN MODEL (PyTorch)
# =============================================================================

print("\n" + "="*70)
print("XAY DUNG CNN ARCHITECTURE")
print("="*70)

class AudioCNN(nn.Module):
    """
    CNN Architecture cho Audio Classification (PHIÊN BẢN CẢI TIẾN)
    
    Kiến trúc: 4 Conv Blocks + 3 Fully Connected Layers
    
    Chi tiết từng block:
        Block 1: Conv(32)→BN→ReLU → Conv(32)→BN→ReLU → MaxPool(2x2) → Dropout(0.2)
                 Input: 1×128×128 → Output: 32×64×64
        
        Block 2: Conv(64)→BN→ReLU → Conv(64)→BN→ReLU → MaxPool(2x2) → Dropout(0.2)
                 Input: 32×64×64 → Output: 64×32×32
        
        Block 3: Conv(128)→BN→ReLU → Conv(128)→BN→ReLU → MaxPool(2x2) → Dropout(0.3)
                 Input: 64×32×32 → Output: 128×16×16
        
        Block 4: Conv(256)→BN→ReLU → Conv(256)→BN→ReLU → AdaptiveAvgPool(1×1) → Dropout(0.3)
                 Input: 128×16×16 → Output: 256×1×1
        
        FC Layers:
                 FC(256) → BN → ReLU → Dropout(0.4)
                 FC(128) → BN → ReLU → Dropout(0.3)
                 FC(50) → Logits
    
    Các kỹ thuật được sử dụng:
        - Batch Normalization: Chuẩn hóa dữ liệu giữa các layers, giúp train nhanh hơn
        - Dropout: Tắt ngẫu nhiên neurons để giảm overfitting
        - He Initialization: Khởi tạo weights phù hợp với ReLU activation
        - Adaptive Average Pooling: Đảm bảo output size cố định bất kể input size
    """
    def __init__(self, num_classes=50):
        super(AudioCNN, self).__init__()
        
        # ===== BLOCK 1: Trích xuất đặc trưng cấp thấp =====
        # Input: 1×128×128 (1 channel mel spectrogram)
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # → 32×128×128
        self.bn1_1 = nn.BatchNorm2d(32)                              # Chuẩn hóa
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # → 32×128×128
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)                              # → 32×64×64 (giảm 1/2)
        self.dropout1 = nn.Dropout2d(0.2)                            # Dropout 20%
        
        # ===== BLOCK 2: Đặc trưng cấp trung =====
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # → 64×64×64
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # → 64×64×64
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)                              # → 64×32×32
        self.dropout2 = nn.Dropout2d(0.2)
        
        # ===== BLOCK 3: Đặc trưng cấp cao =====
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # → 128×32×32
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)# → 128×32×32
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)                              # → 128×16×16
        self.dropout3 = nn.Dropout2d(0.3)                            # Tăng dropout lên 30%
        
        # ===== BLOCK 4: Đặc trưng semantic (ý nghĩa) =====
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# → 256×16×16
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)# → 256×16×16
        self.bn4_2 = nn.BatchNorm2d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))            # → 256×1×1 (global pooling)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # ===== FULLY CONNECTED LAYERS: Phân loại =====
        # Flatten: 256×1×1 → 256
        self.fc1 = nn.Linear(256, 256)      # Layer 1: 256 neurons
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.4)  # Dropout cao (40%) để giảm overfitting
        
        self.fc2 = nn.Linear(256, 128)      # Layer 2: 128 neurons
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)  # Output layer: 50 classes
        
        # Khởi tạo weights (He initialization cho ReLU)
        self._init_weights()
    
    def _init_weights(self):
        """
        Khởi tạo weights cho các layers
        
        - Conv2D: He initialization (phù hợp với ReLU)
        - BatchNorm: weight=1, bias=0
        - Linear: Gaussian với mean=0, std=0.01
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization: Tối ưu cho ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # BatchNorm: scale=1, shift=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Fully connected: Gaussian nhỏ
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass: Truyền dữ liệu qua mạng CNN
        
        Args:
            x: Input tensor, shape (batch_size, 1, 128, 128)
        
        Returns:
            Output logits, shape (batch_size, 50)
            Chưa qua softmax - sẽ dùng CrossEntropyLoss tính toán
        """
        # ===== BLOCK 1: Trích xuất đặc trưng cấp thấp =====
        # Học các patterns đơn giản: edges, textures
        x = F.relu(self.bn1_1(self.conv1_1(x)))  # Conv → BN → ReLU
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)                         # Giảm kích thước 1/2
        x = self.dropout1(x)                      # Regularization
        
        # ===== BLOCK 2: Đặc trưng cấp trung =====
        # Học các patterns phức tạp hơn: shapes, patterns
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # ===== BLOCK 3: Đặc trưng cấp cao =====
        # Học các patterns trừu tượng: objects, structures
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # ===== BLOCK 4: Đặc trưng semantic =====
        # Học ý nghĩa cao nhất: concepts, semantics
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.adaptive_pool(x)  # Global pooling → 1×1
        x = self.dropout4(x)
        
        # ===== FLATTEN =====
        # Chuyển từ 4D (batch, channels, height, width) → 2D (batch, features)
        x = x.view(x.size(0), -1)  # (batch, 256, 1, 1) → (batch, 256)
        
        # ===== FULLY CONNECTED LAYERS: Phân loại =====
        # FC1: 256 → 256
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        
        # FC2: 256 → 128
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        
        # FC3: 128 → 50 (output layer)
        x = self.fc3(x)  # Logits (chưa qua softmax)
        
        return x

# Build model
model = AudioCNN(num_classes=50).to(DEVICE)

# Model summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Architecture:")
print(model)
print(f"\nTrainable parameters: {count_parameters(model):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Huấn luyện model trong 1 epoch
    
    Args:
        model: Mô hình CNN cần train
        loader: DataLoader chứa dữ liệu training
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        avg_loss: Loss trung bình của epoch
        accuracy: Độ chính xác (%) trên tập train
    """
    model.train()  # Bật training mode (dropout, batchnorm hoạt động)
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        # Chuyển dữ liệu sang GPU/CPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Bước 1: Xóa gradients cũ
        optimizer.zero_grad()
        
        # Bước 2: Forward pass - tính output
        outputs = model(inputs)
        
        # Bước 3: Tính loss
        loss = criterion(outputs, labels)
        
        # Bước 4: Backward pass - tính gradients
        loss.backward()
        
        # Bước 5: Gradient clipping - tránh exploding gradients
        # Giới hạn norm của gradients ≤ 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Bước 6: Cập nhật weights
        optimizer.step()
        
        # Thống kê
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Lấy class có xác suất cao nhất
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Hiển thị progress bar với loss và accuracy hiện tại
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """
    Đánh giá model trên tập validation/test
    
    Args:
        model: Mô hình CNN cần đánh giá
        loader: DataLoader chứa dữ liệu val/test
        criterion: Loss function
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        avg_loss: Loss trung bình
        accuracy: Độ chính xác (%)
    """
    model.eval()  # Bật evaluation mode (dropout tắt, batchnorm dùng running stats)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Không tính gradients (tiết kiệm memory và tăng tốc)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Thống kê
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# =============================================================================
# TRAINING LOOP
# =============================================================================

print("\n" + "="*70)
print("HUAN LUYEN CNN MODEL")
print("="*70)

EPOCHS = 100
best_val_acc = 0
best_val_loss = float('inf')
patience = 20
patience_counter = 0

train_losses = []
train_accs = []
val_losses = []
val_accs = []

print(f"\nHyperparameters:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
print(f"  - Early stopping patience: {patience}")
print(f"  - LR scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 70)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    # Learning rate scheduling
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    # Print if learning rate changed
    if old_lr != new_lr:
        print(f"⚡ Learning Rate changed: {old_lr:.6f} → {new_lr:.6f}")
    
    # Store history
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_cnn_improved_model.pth')
        print(f"✓ Model saved! (Val Acc: {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

print(f"\n=> Training completed!")
print(f"   Best validation accuracy: {best_val_acc:.2f}%")

# =============================================================================
# ĐÁNH GIÁ
# =============================================================================

print("\n" + "="*70)
print("DANH GIA MODEL")
print("="*70)

# Load best model
model.load_state_dict(torch.load('best_cnn_improved_model.pth'))
model.eval()

# Predict on test set
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

y_pred = np.array(all_preds)
y_true = np.array(all_labels)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n=> ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Dung: {np.sum(y_pred == y_true)}/{len(y_true)}")

# Classification report
print("\nBao cao chi tiet:")
print(classification_report(y_true, y_pred))

# Confusion matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Confusion Matrix - CNN Model (Accuracy: {accuracy:.2%})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_CNN_improved.png', dpi=300)
plt.close()
print("\n=> Da luu: confusion_matrix_CNN_improved.png")

# =============================================================================
# TRAINING HISTORY
# =============================================================================

print("\n" + "="*70)
print("TRAINING HISTORY")
print("="*70)

# Plot accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy', linewidth=2)
plt.plot(val_accs, label='Val Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_CNN_improved.png', dpi=300)
plt.close()
print("=> Da luu: training_history_CNN_improved.png")

# Best epoch info
best_epoch_idx = np.argmax(val_accs)
print(f"\nBest Epoch: {best_epoch_idx + 1}")
print(f"Best Val Accuracy: {val_accs[best_epoch_idx]:.2f}%")

# =============================================================================
# SO SÁNH VỚI TRADITIONAL ML
# =============================================================================

print("\n" + "="*70)
print("SO SANH VOI TRADITIONAL ML")
print("="*70)

trainable_params = count_parameters(model)

comparison = {
    'Model': ['SVM (Traditional ML)', 'CNN (Deep Learning)'],
    'Accuracy': [0.7625, accuracy],
    'Training Time': ['~2 phút', f'~{len(train_losses)} epochs'],
    'Parameters': ['~200 features', f'{trainable_params:,}'],
    'Input Type': ['Handcrafted features', 'Raw spectrogram']
}

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

improvement = (accuracy - 0.7625) / 0.7625 * 100
print(f"\n=> Cai thien: +{improvement:.2f}%")
print(f"   Tu 76.25% -> {accuracy*100:.2f}%")

if accuracy >= 0.85:
    print(f"\n🎉 MỤC TIÊU ĐẠT ĐƯỢC: 85%+!")
elif accuracy >= 0.80:
    print(f"\n✅ Tốt! Gần đạt mục tiêu 85%")
else:
    print(f"\n⚠️ Cần thêm augmentation hoặc train lâu hơn")

# =============================================================================
# VISUALIZE PREDICTIONS
# =============================================================================

print("\n" + "="*70)
print("VISUALIZE PREDICTIONS")
print("="*70)

# Lấy category names
df_loaded = pd.read_csv(CSV_PATH)
target_to_category = dict(zip(df_loaded['target'], df_loaded['category']))

# Chọn random 9 samples
np.random.seed(42)
indices = np.random.choice(len(X_test), 9, replace=False)

plt.figure(figsize=(15, 10))
for i, idx in enumerate(indices):
    plt.subplot(3, 3, i + 1)
    
    # Hiển thị spectrogram (convert từ CHW -> HW)
    spec_img = X_test[idx].squeeze()  # Remove channel dimension
    plt.imshow(spec_img, cmap='viridis', aspect='auto')
    
    # True vs Predicted
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    true_category = target_to_category[true_label]
    pred_category = target_to_category[pred_label]
    
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'True: {true_category}\nPred: {pred_category}', color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig('predictions_CNN.png', dpi=300)
plt.close()
print("=> Da luu: predictions_CNN.png")

# =============================================================================
# SAVE MODEL INFO
# =============================================================================

print("\n" + "="*70)
print("HOAN TAT!")
print("="*70)

model_info = f"""
═══════════════════════════════════════════════════════════════════════
CNN MODEL FOR AUDIO CLASSIFICATION (PyTorch)
═══════════════════════════════════════════════════════════════════════

DATASET:
  - ESC-50: {len(df)} samples, 50 classes
  - Input: Mel Spectrogram ({IMG_SIZE}x{IMG_SIZE})
  - Split: Train {len(train_dataset)} | Val {len(val_dataset)} | Test {len(test_dataset)}
  - Augmentation: {'YES (6x for training only)' if APPLY_AUGMENTATION else 'NO'}
  - Device: {DEVICE}

MODEL ARCHITECTURE:
  - Type: Custom CNN (4 conv blocks + 3 FC layers)
  - Parameters: {trainable_params:,}
  - Regularization: Dropout (0.2-0.4) + BatchNorm + L2 (1e-4)

TRAINING:
  - Epochs: {len(train_losses)} / {EPOCHS}
  - Best epoch: {best_epoch_idx + 1}
  - Batch size: {BATCH_SIZE}
  - Optimizer: Adam (lr=0.001, weight_decay=1e-4)
  - LR Scheduler: ReduceLROnPlateau
  - Early stopping: patience={patience}

RESULTS:
  - Train Accuracy: {train_accs[best_epoch_idx]:.2f}%
  - Val Accuracy: {best_val_acc:.2f}%
  - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
  - Baseline (SVM): 76.25%
  - Improvement: {improvement:+.2f}%

OUTPUT FILES:
  ✓ best_cnn_improved_model.pth (trained weights)
  ✓ confusion_matrix_CNN_improved.png
  ✓ training_history_CNN_improved.png
  ✓ predictions_CNN.png
  ✓ cnn_model_info.txt

USAGE:
  from cnn_model import AudioCNN
  import torch
  
  model = AudioCNN(num_classes=50)
  model.load_state_dict(torch.load('best_cnn_improved_model.pth'))
  model.eval()

═══════════════════════════════════════════════════════════════════════
"""

with open('cnn_model_info.txt', 'w', encoding='utf-8') as f:
    f.write(model_info)

print(model_info)
print("=> Da luu: cnn_model_info.txt")
print("="*70)

# =============================================================================
# TRỰC QUAN MEL SPECTROGRAMS
# =============================================================================

print("\n" + "="*70)
print("TRUC QUAN MEL SPECTROGRAMS")
print("="*70)

# Tạo thư mục
os.makedirs('mel_spectrograms', exist_ok=True)

# Lấy một số samples ngẫu nhiên từ các class khác nhau
np.random.seed(42)
n_samples = 8
sample_indices = np.random.choice(len(df), n_samples, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

print(f"Dang tao {n_samples} mel spectrogram visualizations...")

for idx, sample_idx in enumerate(sample_indices):
    row = df.iloc[sample_idx]
    file_path = os.path.join(DATA_PATH, row['filename'])
    label = row['target']
    category = row['category']
    
    # Load audio
    y, sr = librosa.load(file_path, sr=22050, duration=5.0)
    
    # Tạo mel spectrogram (không normalize để hiển thị đẹp)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    img = librosa.display.specshow(
        mel_spec_db, 
        sr=sr, 
        hop_length=512,
        x_axis='time', 
        y_axis='mel',
        ax=axes[idx],
        cmap='viridis'
    )
    
    axes[idx].set_title(f'{category}\n(Class {label})', fontsize=11, pad=10)
    axes[idx].set_xlabel('Time (s)', fontsize=9)
    axes[idx].set_ylabel('Frequency (Hz)', fontsize=9)

plt.suptitle('Mel Spectrogram Examples - Audio "Images"', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('mel_spectrograms/mel_spectrogram_examples.png', dpi=150, bbox_inches='tight')
print("=> Da luu: mel_spectrograms/mel_spectrogram_examples.png")

# Tạo visualization chi tiết cho 1 sample
print("\nTao visualization chi tiet cho 1 sample...")
sample_row = df.iloc[0]
file_path = os.path.join(DATA_PATH, sample_row['filename'])
category = sample_row['category']

y, sr = librosa.load(file_path, sr=22050, duration=5.0)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1. Waveform
axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y, linewidth=0.5, color='blue')
axes[0].set_title(f'1. Audio Waveform - "{category}"', fontsize=12, pad=10)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# 2. Mel Spectrogram (dB)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
img = librosa.display.specshow(
    mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='viridis'
)
axes[1].set_title('2. Mel Spectrogram (dB) - "Ảnh" của âm thanh', fontsize=12, pad=10)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

# 3. Mel Spectrogram normalized (input cho CNN)
mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
mel_spec_resized = cv2.resize(mel_spec_norm, (128, 128))
img2 = axes[2].imshow(mel_spec_resized, aspect='auto', origin='lower', cmap='viridis')
axes[2].set_title('3. Normalized Mel Spectrogram (128×128) - Input cho CNN', fontsize=12, pad=10)
axes[2].set_xlabel('Time bins')
axes[2].set_ylabel('Mel frequency bins')
fig.colorbar(img2, ax=axes[2], label='Normalized value [0-1]')

plt.tight_layout()
plt.savefig('mel_spectrograms/mel_spectrogram_detailed.png', dpi=150, bbox_inches='tight')
print("=> Da luu: mel_spectrograms/mel_spectrogram_detailed.png")

# So sánh các class khác nhau
print("\nTao so sanh mel spectrograms cua cac class...")
classes_to_compare = ['dog', 'cat', 'rooster', 'rain', 'thunderstorm', 'crackling_fire']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, class_name in enumerate(classes_to_compare):
    # Tìm sample của class này
    sample = df[df['category'] == class_name].iloc[0] if class_name in df['category'].values else None
    
    if sample is not None:
        file_path = os.path.join(DATA_PATH, sample['filename'])
        y, sr = librosa.load(file_path, sr=22050, duration=5.0)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        librosa.display.specshow(
            mel_spec_db, sr=sr, x_axis='time', y_axis='mel',
            ax=axes[idx], cmap='viridis'
        )
        axes[idx].set_title(f'"{class_name}"', fontsize=12, pad=10, fontweight='bold')
        axes[idx].set_xlabel('Time (s)', fontsize=9)
        axes[idx].set_ylabel('Freq (Hz)', fontsize=9)
    else:
        axes[idx].text(0.5, 0.5, 'Not found', ha='center', va='center')
        axes[idx].set_title(f'"{class_name}" (not found)', fontsize=10)

plt.suptitle('So sánh Mel Spectrograms - Mỗi class có "visual signature" riêng', 
             fontsize=14, y=0.98, fontweight='bold')
plt.tight_layout()
plt.savefig('mel_spectrograms/mel_spectrogram_comparison.png', dpi=150, bbox_inches='tight')
print("=> Da luu: mel_spectrograms/mel_spectrogram_comparison.png")

print("\n=> HOÀN TẤT! Kiểm tra folder 'mel_spectrograms/' để xem:")
print("   - mel_spectrogram_examples.png (8 samples ngẫu nhiên)")
print("   - mel_spectrogram_detailed.png (chi tiết 1 sample)")
print("   - mel_spectrogram_comparison.png (so sánh các class)")

print("\n" + "="*70)
print("🚀 HOÀN TẤT! Kiểm tra các file output:")
print("   - best_cnn_model.pth")
print("   - confusion_matrix_CNN.png")
print("   - training_history_CNN.png") 
print("   - predictions_CNN.png")
print("   - cnn_model_info.txt")
print("   - mel_spectrograms/ (folder chứa visualizations)")
print("="*70)

