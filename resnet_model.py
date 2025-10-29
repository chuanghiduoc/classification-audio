# -*- coding: utf-8 -*-
"""
TRANSFER LEARNING - RESNET18 FOR AUDIO CLASSIFICATION
Sử dụng pretrained ResNet18 từ ImageNet, fine-tune cho ESC-50
Mục tiêu: 88-92% accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# CẤU HÌNH
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

print("="*80)
print("TRANSFER LEARNING - RESNET18 CHO PHÂN LOẠI ÂM THANH")
print("="*80)
print(f"Phiên bản PyTorch: {torch.__version__}")
print(f"Thiết bị: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*80)

# =============================================================================
# SPECAUGMENT CLASS
# =============================================================================

class SpecAugment:
    """
    SpecAugment: Frequency & Time Masking cho regularization
    
    Kỹ thuật augmentation chuyên cho audio, che ngẫu nhiên các dải tần số và thời gian.
    Giúp model học robust hơn, không bị phụ thuộc vào một vài tần số/thời điểm cụ thể.
    """
    def __init__(self, freq_mask_param=25, time_mask_param=25, n_freq_masks=2, n_time_masks=2):
        """
        Khởi tạo SpecAugment
        
        Args:
            freq_mask_param: Độ rộng tối đa của frequency mask (25 bins)
            time_mask_param: Độ rộng tối đa của time mask (25 frames)
            n_freq_masks: Số lượng frequency masks (2)
            n_time_masks: Số lượng time masks (2)
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec):
        """
        Áp dụng SpecAugment lên mel spectrogram
        
        Args:
            spec: Mel spectrogram, shape (H, W) hoặc (1, H, W)
        
        Returns:
            Augmented spectrogram với các vùng bị mask = 0
        """
        spec = spec.copy()
        
        # Xử lý cả định dạng 2D và 3D
        if len(spec.shape) == 3:
            spec = spec.squeeze(0)
            had_channel = True
        else:
            had_channel = False
        
        num_freq_bins, num_time_frames = spec.shape
        
        # Frequency masking: Che các dải tần số ngẫu nhiên
        # Giới hạn độ rộng mask ≤ 1/4 tổng số bins để không mất quá nhiều thông tin
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, num_freq_bins // 4))
            f0 = np.random.randint(0, num_freq_bins - f)
            spec[f0:f0 + f, :] = 0
        
        # Time masking: Che các khung thời gian ngẫu nhiên
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, num_time_frames // 4))
            t0 = np.random.randint(0, num_time_frames - t)
            spec[:, t0:t0 + t] = 0
        
        # Khôi phục chiều channel nếu cần
        if had_channel:
            spec = np.expand_dims(spec, axis=0)
        
        return spec

# =============================================================================
# DATASET CLASS
# =============================================================================

class AudioDataset(Dataset):
    """
    PyTorch Dataset cho audio spectrograms với SpecAugment
    
    Đặc biệt: Chuyển đổi 1-channel (grayscale) → 3-channel (RGB) để tương thích với ResNet
    ResNet được pretrain trên ImageNet (ảnh RGB), nên cần input 3 channels.
    """
    def __init__(self, spectrograms, labels, apply_specaugment=False):
        """
        Khởi tạo Dataset
        
        Args:
            spectrograms: Mảng numpy chứa mel spectrograms, shape (N, 1, H, W)
            labels: Mảng numpy chứa nhãn (0-49)
            apply_specaugment: True = áp dụng SpecAugment (chỉ cho training)
        """
        self.spectrograms = spectrograms
        self.labels = torch.LongTensor(labels)
        self.apply_specaugment = apply_specaugment
        
        # Khởi tạo SpecAugment nếu cần
        if apply_specaugment:
            self.spec_augment = SpecAugment(
                freq_mask_param=25,   # Tăng lên 25 so với CNN (20)
                time_mask_param=25,   # ResNet mạnh hơn, chịu được augment mạnh hơn
                n_freq_masks=2,
                n_time_masks=2
            )
    
    def __len__(self):
        """Trả về số lượng samples"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Lấy 1 sample từ dataset
        
        Args:
            idx: Index của sample
        
        Returns:
            spec: Tensor 3-channel, shape (3, 128, 128) - tương thích với ResNet
            label: Nhãn (0-49)
        """
        spec = self.spectrograms[idx].copy()
        label = self.labels[idx]
        
        # Áp dụng SpecAugment nếu đang training
        if self.apply_specaugment:
            spec = self.spec_augment(spec)
        
        # Chuyển 1-channel → 3-channel để ResNet có thể xử lý
        # ResNet pretrain trên ImageNet (RGB), cần 3 channels input
        if len(spec.shape) == 3 and spec.shape[0] == 1:
            # (1, H, W) → (3, H, W): Nhân đôi channel grayscale thành RGB
            spec = np.repeat(spec, 3, axis=0)
        elif len(spec.shape) == 2:
            # (H, W) → (3, H, W): Stack 3 lần
            spec = np.stack([spec, spec, spec], axis=0)
        
        # Chuyển sang PyTorch tensor
        spec = torch.FloatTensor(spec)
        
        return spec, label

# =============================================================================
# RESNET18 MODEL
# =============================================================================

class AudioResNet18(nn.Module):
    """
    ResNet18 cho phân loại âm thanh (Transfer Learning từ ImageNet)
    
    Ý tưởng chính:
    - Sử dụng ResNet18 đã pretrain trên ImageNet (1.2M ảnh, 1000 classes)
    - ResNet đã học được các đặc trưng tổng quát (edges, textures, shapes)
    - Fine-tune lại cho bài toán phân loại âm thanh (50 classes)
    
    Ưu điểm so với CNN từ đầu:
    - Không cần train từ đầu (tiết kiệm thời gian)
    - Tận dụng kiến thức từ ImageNet (transfer learning)
    - Accuracy cao hơn với ít dữ liệu hơn
    
    Kiến trúc:
    - Backbone: ResNet18 (pretrained) - 11.2M parameters
    - FC layer: Thay đổi từ 1000 classes → 50 classes
    - Dropout: 0.5 để giảm overfitting
    """
    def __init__(self, num_classes=50, pretrained=True, dropout_rate=0.5):
        """
        Khởi tạo AudioResNet18
        
        Args:
            num_classes: Số lượng classes cần phân loại (50 cho ESC-50)
            pretrained: True = load weights từ ImageNet
            dropout_rate: Tỷ lệ dropout (0.5 = 50% neurons bị tắt)
        """
        super(AudioResNet18, self).__init__()
        
        # Bước 1: Load ResNet18 pretrained từ ImageNet
        # Tải toàn bộ kiến trúc và weights đã train sẵn
        self.resnet = models.resnet18(pretrained=pretrained)
        
        if pretrained:
            print("\n✓ Đã tải ResNet18 pretrained từ ImageNet")
        
        # Bước 2: Thay đổi FC layer cuối
        # ResNet18 gốc: FC(512 → 1000) cho ImageNet
        # Cần thay thành: FC(512 → 50) cho ESC-50
        num_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),              # Dropout để giảm overfitting
            nn.Linear(num_features, num_classes)   # 512 → 50
        )
        
        # Bước 3: Khởi tạo weights cho FC layer mới
        # Dùng Xavier initialization (tốt cho linear layers)
        nn.init.xavier_uniform_(self.resnet.fc[1].weight)
        nn.init.constant_(self.resnet.fc[1].bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch_size, 3, 128, 128)
        
        Returns:
            Logits, shape (batch_size, 50)
        """
        return self.resnet(x)
    
    def freeze_backbone(self):
        """
        Đóng băng backbone (chỉ train FC layer)
        
        Dùng trong Stage 1:
        - Giữ nguyên weights pretrained của ResNet
        - Chỉ train FC layer mới
        - Nhanh hơn, ổn định hơn
        """
        for name, param in self.resnet.named_parameters():
            if 'fc' not in name:  # Tất cả layers trừ FC
                param.requires_grad = False  # Không tính gradients
        print("✓ Đã đóng băng backbone (chỉ train FC layer)")
    
    def unfreeze_backbone(self):
        """
        Mở băng toàn bộ model (fine-tune tất cả)
        
        Dùng trong Stage 2:
        - Train toàn bộ ResNet (backbone + FC)
        - Learning rate thấp hơn để không phá hỏng weights pretrained
        - Accuracy cao hơn nhưng cần cẩn thận với overfitting
        """
        for param in self.resnet.parameters():
            param.requires_grad = True  # Tất cả layers đều train được
        print("✓ Đã mở băng backbone (train toàn bộ model)")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "="*80)
print("LOADING PREPROCESSED DATA")
print("="*80)

# Load preprocessed data from previous run
try:
    # Try to load from numpy files (faster)
    X_train = np.load('cache/X_train.npy')
    y_train = np.load('cache/y_train.npy')
    X_val = np.load('cache/X_val.npy')
    y_val = np.load('cache/y_val.npy')
    X_test = np.load('cache/X_test.npy')
    y_test = np.load('cache/y_test.npy')
    
    print("✓ Loaded cached data from numpy files")
    
except FileNotFoundError:
    print("Cache not found. Please run cnn_model.py first to generate preprocessed data.")
    print("Or I'll extract features now...")
    
    # Import extraction functions from cnn_model
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    from cnn_model import extract_mel_spectrogram, create_mel_spectrogram
    import librosa
    import cv2
    
    DATA_PATH = 'data/audio/audio/'
    CSV_PATH = 'data/esc50.csv'
    IMG_SIZE = 128
    
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset: {len(df)} samples, {df['target'].nunique()} classes")
    
    # Split data
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['target'])
    
    print(f"\n=> Split:")
    print(f"   Train: {len(train_df)} files")
    print(f"   Val:   {len(val_df)} files")
    print(f"   Test:  {len(test_df)} files")
    
    # Hàm trích xuất features từ dataframe
    def extract_features(df, augment=False):
        """
        Trích xuất mel spectrograms từ danh sách audio files
        
        Args:
            df: DataFrame chứa thông tin audio files
            augment: True = áp dụng data augmentation (6x samples)
        
        Returns:
            spectrograms: Mảng numpy chứa mel spectrograms
            labels: Mảng numpy chứa nhãn tương ứng
        """
        spectrograms = []
        labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            file_path = os.path.join(DATA_PATH, row['filename'])
            label = row['target']
            
            # Trích xuất mel spectrogram (có thể có augmentation)
            result = extract_mel_spectrogram(file_path, IMG_SIZE, augment=augment)
            
            if result is not None:
                if augment:
                    # Nếu có augmentation, result là list 6 spectrograms
                    for spec in result:
                        spectrograms.append(spec)
                        labels.append(label)
                else:
                    # Không augment, chỉ có 1 spectrogram
                    spectrograms.append(result)
                    labels.append(label)
        
        return np.array(spectrograms), np.array(labels)
    
    # Trích xuất features cho tập TRAIN (có augmentation)
    print("\n[1/3] Extracting TRAIN features (with augmentation)...")
    train_specs, y_train = extract_features(train_df, augment=True)
    X_train = train_specs.reshape(-1, 1, IMG_SIZE, IMG_SIZE)  # Reshape về format PyTorch: (N, C, H, W)
    
    # Trích xuất features cho tập VAL (không augmentation)
    print("\n[2/3] Extracting VAL features...")
    val_specs, y_val = extract_features(val_df, augment=False)
    X_val = val_specs.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    
    # Trích xuất features cho tập TEST (không augmentation)
    print("\n[3/3] Extracting TEST features...")
    test_specs, y_test = extract_features(test_df, augment=False)
    X_test = test_specs.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    
    # Lưu vào cache để lần sau không phải trích xuất lại
    os.makedirs('cache', exist_ok=True)
    np.save('cache/X_train.npy', X_train)
    np.save('cache/y_train.npy', y_train)
    np.save('cache/X_val.npy', X_val)
    np.save('cache/y_val.npy', y_val)
    np.save('cache/X_test.npy', X_test)
    np.save('cache/y_test.npy', y_test)
    print("\n✓ Cached data saved to cache/ directory")

print(f"\n=> Data shapes:")
print(f"   Train: {X_train.shape} | Labels: {y_train.shape}")
print(f"   Val:   {X_val.shape} | Labels: {y_val.shape}")
print(f"   Test:  {X_test.shape} | Labels: {y_test.shape}")

# =============================================================================
# CREATE DATASETS & DATALOADERS
# =============================================================================

print("\n" + "="*80)
print("CREATING DATASETS & DATALOADERS")
print("="*80)

# Tạo PyTorch Datasets
# Train dataset: áp dụng SpecAugment để tăng tính đa dạng
train_dataset = AudioDataset(X_train, y_train, apply_specaugment=True)
# Val/Test datasets: không augment để đánh giá chính xác
val_dataset = AudioDataset(X_val, y_val, apply_specaugment=False)
test_dataset = AudioDataset(X_test, y_test, apply_specaugment=False)

# Tạo DataLoaders để load data theo batch
# Train: shuffle=True để tránh model học theo thứ tự
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
# Val/Test: shuffle=False để kết quả đồng nhất
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✓ Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
print(f"✓ Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
print(f"✓ Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
print(f"✓ SpecAugment: ENABLED for training")

# =============================================================================
# BUILD MODEL
# =============================================================================

print("\n" + "="*80)
print("BUILDING RESNET18 MODEL")
print("="*80)

# Khởi tạo model ResNet18 với pretrained weights từ ImageNet
# Chuyển model sang GPU/CPU tùy theo device có sẵn
model = AudioResNet18(num_classes=50, pretrained=True, dropout_rate=0.5).to(DEVICE)

# Hàm đếm số lượng parameters trong model
def count_parameters(model):
    """
    Đếm tổng số parameters và số parameters trainable
    
    Returns:
        total: Tổng số parameters (bao gồm cả frozen)
        trainable: Số parameters có thể train (requires_grad=True)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)
print(f"\nModel: ResNet18")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Huấn luyện model trong 1 epoch
    
    Args:
        model: AudioResNet18 model
        loader: DataLoader training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        avg_loss: Loss trung bình
        accuracy: Độ chính xác (%)
    """
    model.train()  # Training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        # Chuyển dữ liệu sang device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Training step
        optimizer.zero_grad()           # Reset gradients
        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, labels)  # Tính loss
        loss.backward()                 # Backward pass (tính gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()                # Cập nhật weights
        
        # Thống kê
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Hiển thị progress
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """
    Đánh giá model trên tập validation/test
    
    Args:
        model: AudioResNet18 model
        loader: DataLoader val/test data
        criterion: Loss function
        device: 'cuda' hoặc 'cpu'
    
    Returns:
        avg_loss: Loss trung bình
        accuracy: Độ chính xác (%)
    """
    model.eval()  # Evaluation mode (dropout tắt)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Không tính gradients để tiết kiệm memory
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Thống kê
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

# =============================================================================
# TWO-STAGE TRAINING
# =============================================================================

print("\n" + "="*80)
print("TRAINING STRATEGY: TWO-STAGE FINE-TUNING")
print("="*80)
print("\nStage 1: Train only FC layer (backbone frozen) - 20 epochs")
print("Stage 2: Fine-tune entire network - 60 epochs")

# Loss function: CrossEntropyLoss cho multi-class classification
criterion = nn.CrossEntropyLoss()

# =============================================================================
# STAGE 1: TRAIN FC LAYER ONLY
# =============================================================================
# Chiến lược: Đóng băng backbone (giữ nguyên pretrained weights)
#             Chỉ train FC layer mới để adapt cho ESC-50
#             Giúp tránh phá hỏng features đã học từ ImageNet

print("\n" + "="*80)
print("STAGE 1: TRAINING FC LAYER (Backbone Frozen)")
print("="*80)

# Đóng băng tất cả layers trừ FC layer
model.freeze_backbone()

# Optimizer chỉ optimize các parameters có requires_grad=True (FC layer)
optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=LEARNING_RATE, weight_decay=1e-4)  # L2 regularization

# Learning rate scheduler: Giảm LR khi val loss không giảm
scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode='min', 
                                                         factor=0.5, patience=5, verbose=True)

# Khởi tạo biến theo dõi
best_val_acc_stage1 = 0
patience_counter = 0  # Đếm số epochs không cải thiện (cho early stopping)
stage1_epochs = 20

# Lưu lịch sử training
train_losses_s1 = []
train_accs_s1 = []
val_losses_s1 = []
val_accs_s1 = []

# Training loop cho Stage 1
for epoch in range(stage1_epochs):
    print(f"\n[Stage 1] Epoch {epoch+1}/{stage1_epochs}")
    print("-" * 80)
    
    # Train 1 epoch
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_stage1, DEVICE)
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    # Cập nhật learning rate nếu val loss không giảm
    scheduler_stage1.step(val_loss)
    
    # Lưu lịch sử
    train_losses_s1.append(train_loss)
    train_accs_s1.append(train_acc)
    val_losses_s1.append(val_loss)
    val_accs_s1.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Lưu model nếu val accuracy cải thiện
    if val_acc > best_val_acc_stage1:
        best_val_acc_stage1 = val_acc
        torch.save(model.state_dict(), 'best_resnet_stage1.pth')
        print(f"✓ Stage 1 model saved! (Val Acc: {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping: Dừng nếu không cải thiện sau 10 epochs
    if patience_counter >= 10:
        print(f"\nEarly stopping triggered in Stage 1 after {epoch+1} epochs")
        break

print(f"\n=> Stage 1 completed!")
print(f"   Best validation accuracy: {best_val_acc_stage1:.2f}%")

# =============================================================================
# STAGE 2: FINE-TUNE ENTIRE NETWORK
# =============================================================================
# Chiến lược: Mở băng toàn bộ model (backbone + FC)
#             Train với learning rate thấp hơn để không phá hỏng pretrained weights
#             Fine-tune toàn bộ để adapt tốt hơn cho audio classification

print("\n" + "="*80)
print("STAGE 2: FINE-TUNING ENTIRE NETWORK")
print("="*80)

# Load model tốt nhất từ Stage 1
model.load_state_dict(torch.load('best_resnet_stage1.pth'))
# Mở băng toàn bộ backbone để train được
model.unfreeze_backbone()

# Learning rate thấp hơn (1/10) để fine-tune cẩn thận, tránh phá hỏng weights
optimizer_stage2 = optim.Adam(model.parameters(), lr=LEARNING_RATE/10, weight_decay=1e-4)
# Patience cao hơn (7) vì fine-tune cần thời gian
scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage2, mode='min', 
                                                         factor=0.5, patience=7, verbose=True)

# Khởi tạo từ kết quả Stage 1
best_val_acc_stage2 = best_val_acc_stage1
patience_counter = 0
stage2_epochs = 60

# Lưu lịch sử training Stage 2
train_losses_s2 = []
train_accs_s2 = []
val_losses_s2 = []
val_accs_s2 = []

# Training loop cho Stage 2
for epoch in range(stage2_epochs):
    print(f"\n[Stage 2] Epoch {epoch+1}/{stage2_epochs}")
    print("-" * 80)
    
    # Train toàn bộ model
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_stage2, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    # Cập nhật learning rate
    scheduler_stage2.step(val_loss)
    
    # Lưu lịch sử
    train_losses_s2.append(train_loss)
    train_accs_s2.append(train_acc)
    val_losses_s2.append(val_loss)
    val_accs_s2.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Lưu model nếu val accuracy cải thiện
    if val_acc > best_val_acc_stage2:
        best_val_acc_stage2 = val_acc
        torch.save(model.state_dict(), 'best_resnet18_model.pth')
        print(f"✓ Stage 2 model saved! (Val Acc: {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping với patience=15
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered in Stage 2 after {epoch+1} epochs")
        break

print(f"\n=> Stage 2 completed!")
print(f"   Best validation accuracy: {best_val_acc_stage2:.2f}%")

# Kết hợp lịch sử training từ cả 2 stages
train_losses = train_losses_s1 + train_losses_s2
train_accs = train_accs_s1 + train_accs_s2
val_losses = val_losses_s1 + val_losses_s2
val_accs = val_accs_s1 + val_accs_s2

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

# Load model tốt nhất (từ Stage 2)
model.load_state_dict(torch.load('best_resnet18_model.pth'))
model.eval()  # Chuyển sang evaluation mode

# Dự đoán trên tập test
all_preds = []
all_labels = []

with torch.no_grad():  # Không tính gradients để tiết kiệm memory
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)  # Forward pass
        _, predicted = outputs.max(1)  # Lấy class có xác suất cao nhất
        
        # Lưu kết quả
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Chuyển sang numpy arrays
y_pred = np.array(all_preds)
y_true = np.array(all_labels)

# Tính accuracy trên tập test
accuracy = accuracy_score(y_true, y_pred)
print(f"\n=> TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Correct: {np.sum(y_pred == y_true)}/{len(y_true)}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Vẽ Confusion Matrix để xem model nhầm lẫn giữa các classes nào
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Confusion Matrix - ResNet18 (Accuracy: {accuracy:.2%})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_ResNet18.png', dpi=300)
plt.close()
print("\n=> Saved: confusion_matrix_ResNet18.png")

# =============================================================================
# TRAINING HISTORY
# =============================================================================

print("\n" + "="*80)
print("TRAINING HISTORY")
print("="*80)

plt.figure(figsize=(16, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy', linewidth=2)
plt.plot(val_accs, label='Val Accuracy', linewidth=2)
plt.axvline(x=len(train_accs_s1), color='r', linestyle='--', label='Stage 1 → Stage 2')
plt.title('ResNet18 Training: Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.axvline(x=len(train_losses_s1), color='r', linestyle='--', label='Stage 1 → Stage 2')
plt.title('ResNet18 Training: Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_ResNet18.png', dpi=300)
plt.close()
print("=> Saved: training_history_ResNet18.png")

# Best epoch
best_epoch_idx = np.argmax(val_accs)
print(f"\nBest Epoch: {best_epoch_idx + 1}")
print(f"Best Val Accuracy: {val_accs[best_epoch_idx]:.2f}%")

# =============================================================================
# MODEL COMPARISON
# =============================================================================
# So sánh ResNet18 với các approaches khác:
# - SVM (Traditional ML): Handcrafted features
# - CNN (Custom): Deep learning từ đầu
# - ResNet18: Transfer learning

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = {
    'Model': ['SVM', 'CNN (Custom)', 'ResNet18 (Transfer Learning)'],
    'Accuracy': [0.7625, 0.8025, accuracy],
    'Parameters': ['~200 features', '1.28M', '11.2M'],
    'Training': ['2 min', '~100 epochs', f'{len(train_accs)} epochs (2-stage)']
}

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

# Tính mức độ cải thiện so với các baseline
improvement_vs_svm = (accuracy - 0.7625) / 0.7625 * 100
improvement_vs_cnn = (accuracy - 0.8025) / 0.8025 * 100

print(f"\n=> Improvement:")
print(f"   vs SVM: +{improvement_vs_svm:.2f}% ({0.7625*100:.2f}% → {accuracy*100:.2f}%)")
print(f"   vs CNN: +{improvement_vs_cnn:.2f}% ({0.8025*100:.2f}% → {accuracy*100:.2f}%)")

# Đánh giá kết quả
if accuracy >= 0.85:
    print(f"\n🎉 MỤC TIÊU ĐẠT ĐƯỢC: 85%+!")
    if accuracy >= 0.90:
        print(f"🏆 EXCELLENT: 90%+!")
elif accuracy >= 0.80:
    print(f"\n✅ Tốt! Đã vượt CNN baseline!")
else:
    print(f"\n⚠️ Cần thêm tuning")

# =============================================================================
# SAVE MODEL INFO
# =============================================================================

model_info = f"""
═══════════════════════════════════════════════════════════════════════
RESNET18 TRANSFER LEARNING FOR AUDIO CLASSIFICATION
═══════════════════════════════════════════════════════════════════════

DATASET:
  - ESC-50: {len(y_train) + len(y_val) + len(y_test)} samples, 50 classes
  - Train: {len(y_train)} samples (augmented)
  - Val: {len(y_val)} samples
  - Test: {len(y_test)} samples
  - SpecAugment: ENABLED for training

MODEL:
  - Architecture: ResNet18 (pretrained on ImageNet)
  - Total parameters: {total_params:,}
  - Trainable parameters: {trainable_params:,}
  - Input: Mel Spectrogram (128x128) converted to 3-channel RGB
  - Dropout: 0.5

TRAINING STRATEGY:
  - Stage 1: FC layer only ({len(train_accs_s1)} epochs)
    Best Val Acc: {best_val_acc_stage1:.2f}%
  
  - Stage 2: Full fine-tuning ({len(train_accs_s2)} epochs)
    Best Val Acc: {best_val_acc_stage2:.2f}%
  
  - Total epochs: {len(train_accs)}
  - Optimizer: Adam (Stage 1: lr=0.001, Stage 2: lr=0.0001)
  - Scheduler: ReduceLROnPlateau
  - Early stopping: patience={EARLY_STOPPING_PATIENCE}

RESULTS:
  - Train Accuracy: {train_accs[best_epoch_idx]:.2f}%
  - Val Accuracy: {best_val_acc_stage2:.2f}%
  - Test Accuracy: {accuracy*100:.2f}%
  - Improvement vs SVM: +{improvement_vs_svm:.2f}%
  - Improvement vs CNN: +{improvement_vs_cnn:.2f}%

FILES:
  ✓ best_resnet18_model.pth
  ✓ confusion_matrix_ResNet18.png
  ✓ training_history_ResNet18.png
  ✓ resnet_model_info.txt

USAGE:
  from resnet_model import AudioResNet18
  import torch
  
  model = AudioResNet18(num_classes=50, pretrained=False)
  model.load_state_dict(torch.load('best_resnet18_model.pth'))
  model.eval()

═══════════════════════════════════════════════════════════════════════
"""

with open('resnet_model_info.txt', 'w', encoding='utf-8') as f:
    f.write(model_info)

print(model_info)
print("=> Saved: resnet_model_info.txt")

print("\n" + "="*80)
print("🚀 TRAINING COMPLETED!")
print("="*80)
print("\nGenerated files:")
print("  ✓ best_resnet18_model.pth")
print("  ✓ best_resnet_stage1.pth")
print("  ✓ confusion_matrix_ResNet18.png")
print("  ✓ training_history_ResNet18.png")
print("  ✓ resnet_model_info.txt")
print("="*80)

