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
print("TRANSFER LEARNING - RESNET18 FOR AUDIO CLASSIFICATION")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*80)

# =============================================================================
# SPECAUGMENT CLASS
# =============================================================================

class SpecAugment:
    """SpecAugment: Frequency & Time Masking for regularization"""
    def __init__(self, freq_mask_param=25, time_mask_param=25, n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec):
        spec = spec.copy()
        
        if len(spec.shape) == 3:
            spec = spec.squeeze(0)
            had_channel = True
        else:
            had_channel = False
        
        num_freq_bins, num_time_frames = spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, num_freq_bins // 4))
            f0 = np.random.randint(0, num_freq_bins - f)
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, num_time_frames // 4))
            t0 = np.random.randint(0, num_time_frames - t)
            spec[:, t0:t0 + t] = 0
        
        if had_channel:
            spec = np.expand_dims(spec, axis=0)
        
        return spec

# =============================================================================
# DATASET CLASS
# =============================================================================

class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio spectrograms with SpecAugment
    Converts 1-channel to 3-channel for ResNet compatibility
    """
    def __init__(self, spectrograms, labels, apply_specaugment=False):
        self.spectrograms = spectrograms
        self.labels = torch.LongTensor(labels)
        self.apply_specaugment = apply_specaugment
        
        if apply_specaugment:
            self.spec_augment = SpecAugment(
                freq_mask_param=25,
                time_mask_param=25,
                n_freq_masks=2,
                n_time_masks=2
            )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx].copy()
        label = self.labels[idx]
        
        # Apply SpecAugment
        if self.apply_specaugment:
            spec = self.spec_augment(spec)
        
        # Convert 1-channel to 3-channel (for ResNet)
        # ResNet expects RGB images (3 channels)
        if len(spec.shape) == 3 and spec.shape[0] == 1:
            spec = np.repeat(spec, 3, axis=0)  # (1, H, W) → (3, H, W)
        elif len(spec.shape) == 2:
            spec = np.stack([spec, spec, spec], axis=0)  # (H, W) → (3, H, W)
        
        spec = torch.FloatTensor(spec)
        
        return spec, label

# =============================================================================
# RESNET18 MODEL
# =============================================================================

class AudioResNet18(nn.Module):
    """
    ResNet18 adapted for audio classification
    
    Features:
    - Pretrained on ImageNet (transfer learning)
    - Modified first conv layer for grayscale → RGB conversion
    - Modified final FC layer for 50 classes
    - Dropout for regularization
    """
    def __init__(self, num_classes=50, pretrained=True, dropout_rate=0.5):
        super(AudioResNet18, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        if pretrained:
            print("\n✓ Loaded pretrained ResNet18 weights from ImageNet")
        
        # Modify final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize new FC layer
        nn.init.xavier_uniform_(self.resnet.fc[1].weight)
        nn.init.constant_(self.resnet.fc[1].bias, 0)
    
    def forward(self, x):
        return self.resnet(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final FC layer"""
        for name, param in self.resnet.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("✓ Backbone frozen (only FC layer trainable)")
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen (full model trainable)")

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
    
    # Extract features
    def extract_features(df, augment=False):
        spectrograms = []
        labels = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
            file_path = os.path.join(DATA_PATH, row['filename'])
            label = row['target']
            
            result = extract_mel_spectrogram(file_path, IMG_SIZE, augment=augment)
            
            if result is not None:
                if augment:
                    for spec in result:
                        spectrograms.append(spec)
                        labels.append(label)
                else:
                    spectrograms.append(result)
                    labels.append(label)
        
        return np.array(spectrograms), np.array(labels)
    
    print("\n[1/3] Extracting TRAIN features (with augmentation)...")
    train_specs, y_train = extract_features(train_df, augment=True)
    X_train = train_specs.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    
    print("\n[2/3] Extracting VAL features...")
    val_specs, y_val = extract_features(val_df, augment=False)
    X_val = val_specs.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    
    print("\n[3/3] Extracting TEST features...")
    test_specs, y_test = extract_features(test_df, augment=False)
    X_test = test_specs.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    
    # Cache for future use
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

train_dataset = AudioDataset(X_train, y_train, apply_specaugment=True)
val_dataset = AudioDataset(X_val, y_val, apply_specaugment=False)
test_dataset = AudioDataset(X_test, y_test, apply_specaugment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
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

model = AudioResNet18(num_classes=50, pretrained=True, dropout_rate=0.5).to(DEVICE)

# Count parameters
def count_parameters(model):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
# TWO-STAGE TRAINING
# =============================================================================

print("\n" + "="*80)
print("TRAINING STRATEGY: TWO-STAGE FINE-TUNING")
print("="*80)
print("\nStage 1: Train only FC layer (backbone frozen) - 20 epochs")
print("Stage 2: Fine-tune entire network - 60 epochs")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# =============================================================================
# STAGE 1: TRAIN FC LAYER ONLY
# =============================================================================

print("\n" + "="*80)
print("STAGE 1: TRAINING FC LAYER (Backbone Frozen)")
print("="*80)

model.freeze_backbone()

# Only optimize FC layer
optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=LEARNING_RATE, weight_decay=1e-4)
scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode='min', 
                                                         factor=0.5, patience=5, verbose=True)

best_val_acc_stage1 = 0
patience_counter = 0
stage1_epochs = 20

train_losses_s1 = []
train_accs_s1 = []
val_losses_s1 = []
val_accs_s1 = []

for epoch in range(stage1_epochs):
    print(f"\n[Stage 1] Epoch {epoch+1}/{stage1_epochs}")
    print("-" * 80)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_stage1, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    scheduler_stage1.step(val_loss)
    
    train_losses_s1.append(train_loss)
    train_accs_s1.append(train_acc)
    val_losses_s1.append(val_loss)
    val_accs_s1.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc_stage1:
        best_val_acc_stage1 = val_acc
        torch.save(model.state_dict(), 'best_resnet_stage1.pth')
        print(f"✓ Stage 1 model saved! (Val Acc: {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 10:
        print(f"\nEarly stopping triggered in Stage 1 after {epoch+1} epochs")
        break

print(f"\n=> Stage 1 completed!")
print(f"   Best validation accuracy: {best_val_acc_stage1:.2f}%")

# =============================================================================
# STAGE 2: FINE-TUNE ENTIRE NETWORK
# =============================================================================

print("\n" + "="*80)
print("STAGE 2: FINE-TUNING ENTIRE NETWORK")
print("="*80)

# Load best model from stage 1
model.load_state_dict(torch.load('best_resnet_stage1.pth'))
model.unfreeze_backbone()

# Lower learning rate for fine-tuning
optimizer_stage2 = optim.Adam(model.parameters(), lr=LEARNING_RATE/10, weight_decay=1e-4)
scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage2, mode='min', 
                                                         factor=0.5, patience=7, verbose=True)

best_val_acc_stage2 = best_val_acc_stage1
patience_counter = 0
stage2_epochs = 60

train_losses_s2 = []
train_accs_s2 = []
val_losses_s2 = []
val_accs_s2 = []

for epoch in range(stage2_epochs):
    print(f"\n[Stage 2] Epoch {epoch+1}/{stage2_epochs}")
    print("-" * 80)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer_stage2, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    scheduler_stage2.step(val_loss)
    
    train_losses_s2.append(train_loss)
    train_accs_s2.append(train_acc)
    val_losses_s2.append(val_loss)
    val_accs_s2.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc_stage2:
        best_val_acc_stage2 = val_acc
        torch.save(model.state_dict(), 'best_resnet18_model.pth')
        print(f"✓ Stage 2 model saved! (Val Acc: {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping triggered in Stage 2 after {epoch+1} epochs")
        break

print(f"\n=> Stage 2 completed!")
print(f"   Best validation accuracy: {best_val_acc_stage2:.2f}%")

# Combine training history
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

# Load best model
model.load_state_dict(torch.load('best_resnet18_model.pth'))
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

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n=> TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Correct: {np.sum(y_pred == y_true)}/{len(y_true)}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
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

improvement_vs_svm = (accuracy - 0.7625) / 0.7625 * 100
improvement_vs_cnn = (accuracy - 0.8025) / 0.8025 * 100

print(f"\n=> Improvement:")
print(f"   vs SVM: +{improvement_vs_svm:.2f}% ({0.7625*100:.2f}% → {accuracy*100:.2f}%)")
print(f"   vs CNN: +{improvement_vs_cnn:.2f}% ({0.8025*100:.2f}% → {accuracy*100:.2f}%)")

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

