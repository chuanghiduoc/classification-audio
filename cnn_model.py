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

# Cấu hình
DATA_PATH = 'data/audio/audio/'
CSV_PATH = 'data/esc50.csv'
IMG_SIZE = 128
APPLY_AUGMENTATION = True  # Bật/tắt data augmentation
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
# HÀM TRÍCH XUẤT MEL SPECTROGRAM
# =============================================================================

def extract_mel_spectrogram(file_path, img_size=128, augment=False):
    """
    Chuyển audio file → Mel Spectrogram (ảnh 128x128)
    
    Args:
        file_path: Đường dẫn file audio
        img_size: Kích thước ảnh output
        augment: Có apply augmentation không
    
    Returns:
        mel_spec: Array (128, 128) hoặc list of arrays nếu augment=True
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=22050, duration=5.0)
        
        # Nếu audio ngắn hơn 5s → pad zeros
        if len(y) < sr * 5:
            y = np.pad(y, (0, sr * 5 - len(y)), mode='constant')
        
        results = []
        
        # Original
        mel_spec = create_mel_spectrogram(y, sr, img_size)
        results.append(mel_spec)
        
        # Data Augmentation
        if augment:
            # 1. Time Stretching (slower)
            y_slow = librosa.effects.time_stretch(y, rate=0.9)
            mel_spec_slow = create_mel_spectrogram(y_slow, sr, img_size)
            results.append(mel_spec_slow)
            
            # 2. Time Stretching (faster)
            y_fast = librosa.effects.time_stretch(y, rate=1.1)
            mel_spec_fast = create_mel_spectrogram(y_fast, sr, img_size)
            results.append(mel_spec_fast)
            
            # 3. Pitch Shifting (+2 semitones)
            y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
            mel_spec_pitch_up = create_mel_spectrogram(y_pitch_up, sr, img_size)
            results.append(mel_spec_pitch_up)
            
            # 4. Pitch Shifting (-2 semitones)
            y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
            mel_spec_pitch_down = create_mel_spectrogram(y_pitch_down, sr, img_size)
            results.append(mel_spec_pitch_down)
            
            # 5. Add Gaussian Noise
            noise = np.random.randn(len(y)) * 0.005
            y_noise = y + noise
            mel_spec_noise = create_mel_spectrogram(y_noise, sr, img_size)
            results.append(mel_spec_noise)
        
        return results if augment else results[0]
        
    except Exception as e:
        print(f"Loi: {file_path}: {e}")
        return None

def create_mel_spectrogram(y, sr, img_size):
    """Tạo Mel Spectrogram từ audio signal"""
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        fmax=8000
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Resize to img_size x img_size
    mel_spec_resized = cv2.resize(mel_spec_norm, (img_size, img_size))
    
    return mel_spec_resized

# =============================================================================
# ĐỌC DỮ LIỆU
# =============================================================================

print("\n" + "="*70)
print("TRICH XUAT MEL SPECTROGRAMS")
print("="*70)

df = pd.read_csv(CSV_PATH)
print(f"Dataset: {len(df)} samples, {df['target'].nunique()} classes")

# Trích xuất spectrograms
file_paths = [os.path.join(DATA_PATH, row['filename']) for _, row in df.iterrows()]
labels = df['target'].values

spectrograms = []
labels_expanded = []

print(f"\nDang xu ly {len(file_paths)} files...")
if APPLY_AUGMENTATION:
    print("Data Augmentation: BAT (6x augmentation)")
else:
    print("Data Augmentation: TAT")

for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
    if (idx + 1) % 200 == 0:
        print(f"  Da xu ly: {idx + 1}/{len(file_paths)}")
    
    result = extract_mel_spectrogram(file_path, IMG_SIZE, augment=APPLY_AUGMENTATION)
    
    if result is not None:
        if APPLY_AUGMENTATION:
            # result là list of spectrograms
            for spec in result:
                spectrograms.append(spec)
                labels_expanded.append(label)
        else:
            # result là 1 spectrogram
            spectrograms.append(result)
            labels_expanded.append(label)

spectrograms = np.array(spectrograms)
labels_expanded = np.array(labels_expanded)

print(f"\n=> Thanh cong!")
print(f"   Spectrograms shape: {spectrograms.shape}")
print(f"   Labels shape: {labels_expanded.shape}")

if APPLY_AUGMENTATION:
    print(f"   Augmentation: {len(file_paths)} -> {len(spectrograms)} samples ({len(spectrograms)//len(file_paths)}x)")

# PyTorch: (samples, channels, height, width)
X = spectrograms.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
y = labels_expanded

print(f"\n=> Input shape: {X.shape}")
print(f"   Output shape: {y.shape}")

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n=> Train: {X_train.shape[0]} samples")
print(f"   Test:  {X_test.shape[0]} samples")

# =============================================================================
# PYTORCH DATASET
# =============================================================================

class AudioDataset(Dataset):
    """PyTorch Dataset cho audio spectrograms"""
    def __init__(self, spectrograms, labels):
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]

# Create datasets
train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

# Create dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Validation split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n=> DataLoaders created")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# =============================================================================
# XÂY DỰNG CNN MODEL (PyTorch)
# =============================================================================

print("\n" + "="*70)
print("XAY DUNG CNN ARCHITECTURE")
print("="*70)

class AudioCNN(nn.Module):
    """
    CNN Architecture cho Audio Classification
    
    Architecture:
        Conv2D(32) -> Conv2D(32) -> MaxPool -> BatchNorm -> Dropout
        Conv2D(64) -> Conv2D(64) -> MaxPool -> BatchNorm -> Dropout
        Conv2D(128) -> Conv2D(128) -> MaxPool -> BatchNorm -> Dropout
        Conv2D(256) -> Conv2D(256) -> AdaptiveAvgPool
        FC(512) -> BatchNorm -> Dropout
        FC(256) -> BatchNorm -> Dropout
        FC(50)
    """
    def __init__(self, num_classes=50):
        super(AudioCNN, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.bn_fc1(x)
        x = self.dropout_fc1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn_fc2(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
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
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
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
patience = 15
patience_counter = 0

train_losses = []
train_accs = []
val_losses = []
val_accs = []

print(f"\nEpochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Early stopping patience: {patience}")

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
        torch.save(model.state_dict(), 'best_cnn_model.pth')
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
model.load_state_dict(torch.load('best_cnn_model.pth'))
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
plt.savefig('confusion_matrix_CNN.png', dpi=300)
plt.close()
print("\n=> Da luu: confusion_matrix_CNN.png")

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
plt.savefig('training_history_CNN.png', dpi=300)
plt.close()
print("=> Da luu: training_history_CNN.png")

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
MODEL: CNN for Audio Classification (PyTorch)
Dataset: ESC-50 (50 classes, {len(df)} samples)
Input: Mel Spectrogram ({IMG_SIZE}x{IMG_SIZE})
Data Augmentation: {'YES (6x)' if APPLY_AUGMENTATION else 'NO'}
Device: {DEVICE}

TRAINING:
- Train samples: {len(train_subset)}
- Val samples: {len(val_subset)}
- Test samples: {len(test_dataset)}
- Epochs trained: {len(train_losses)}
- Best epoch: {best_epoch_idx + 1}
- Batch size: {BATCH_SIZE}
- Optimizer: Adam (lr=0.001)
- Trainable parameters: {trainable_params:,}

RESULTS:
- Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
- Best Val Accuracy: {best_val_acc:.2f}%
- Improvement over SVM: +{improvement:.2f}%

FILES:
- best_cnn_model.pth (trained model)
- confusion_matrix_CNN.png
- training_history_CNN.png
- predictions_CNN.png
- cnn_model_info.txt

USAGE:
1. Cài đặt: pip install torch torchvision torchaudio opencv-python
2. Training: python cnn_model.py
3. Load model: 
   model = AudioCNN(num_classes=50)
   model.load_state_dict(torch.load('best_cnn_model.pth'))
"""

with open('cnn_model_info.txt', 'w', encoding='utf-8') as f:
    f.write(model_info)

print(model_info)
print("=> Da luu: cnn_model_info.txt")
print("="*70)
print("\n🚀 HOÀN TẤT! Kiểm tra các file output:")
print("   - best_cnn_model.pth")
print("   - confusion_matrix_CNN.png")
print("   - training_history_CNN.png") 
print("   - predictions_CNN.png")
print("   - cnn_model_info.txt")
print("="*70)

