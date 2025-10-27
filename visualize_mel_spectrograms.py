# -*- coding: utf-8 -*-
"""
TRỰC QUAN MEL SPECTROGRAMS
Tạo ảnh minh họa để hiểu cách CNN "nhìn" audio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import cv2

# Cấu hình
DATA_PATH = 'data/audio/audio/'
CSV_PATH = 'data/esc50.csv'

print("="*70)
print("TRỰC QUAN MEL SPECTROGRAMS")
print("="*70)

# Đọc dataset
df = pd.read_csv(CSV_PATH)
print(f"Dataset: {len(df)} samples, {df['target'].nunique()} classes")

# Tạo thư mục
os.makedirs('mel_spectrograms', exist_ok=True)

# =============================================================================
# 1. VÍ DỤ NGẪU NHIÊN
# =============================================================================

print("\n1. Tạo 8 mel spectrogram examples...")
np.random.seed(42)
n_samples = 8
sample_indices = np.random.choice(len(df), n_samples, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, sample_idx in enumerate(sample_indices):
    row = df.iloc[sample_idx]
    file_path = os.path.join(DATA_PATH, row['filename'])
    label = row['target']
    category = row['category']
    
    # Load audio
    y, sr = librosa.load(file_path, sr=22050, duration=5.0)
    
    # Tạo mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    librosa.display.specshow(
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
print("   ✓ Saved: mel_spectrogram_examples.png")

# =============================================================================
# 2. CHI TIẾT 1 SAMPLE
# =============================================================================

print("\n2. Tạo visualization chi tiết (waveform → spectrogram → CNN input)...")
sample_row = df.iloc[0]
file_path = os.path.join(DATA_PATH, sample_row['filename'])
category = sample_row['category']

y, sr = librosa.load(file_path, sr=22050, duration=5.0)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Waveform (audio gốc)
axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y, linewidth=0.5, color='blue')
axes[0].set_title(f'① Audio Waveform - "{category}" (1D signal)', 
                  fontsize=12, pad=10, fontweight='bold')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# Mel Spectrogram (dB scale)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
img = librosa.display.specshow(
    mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='viridis'
)
axes[1].set_title('② Mel Spectrogram (dB) - Chuyển 1D → 2D "ảnh"', 
                  fontsize=12, pad=10, fontweight='bold')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

# Normalized (input cho CNN)
mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
mel_spec_resized = cv2.resize(mel_spec_norm, (128, 128))
img2 = axes[2].imshow(mel_spec_resized, aspect='auto', origin='lower', cmap='viridis')
axes[2].set_title('③ Normalized 128×128 - Đầu vào CNN (giống ảnh mèo/chó)', 
                  fontsize=12, pad=10, fontweight='bold')
axes[2].set_xlabel('Time bins')
axes[2].set_ylabel('Mel frequency bins')
fig.colorbar(img2, ax=axes[2], label='Normalized [0-1]')

plt.tight_layout()
plt.savefig('mel_spectrograms/mel_spectrogram_detailed.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: mel_spectrogram_detailed.png")

# =============================================================================
# 3. SO SÁNH CÁC CLASS
# =============================================================================

print("\n3. So sánh mel spectrograms của các class khác nhau...")
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
        axes[idx].set_title(f'"{class_name}"', fontsize=13, pad=10, fontweight='bold')
        axes[idx].set_xlabel('Time (s)', fontsize=10)
        axes[idx].set_ylabel('Freq (Hz)', fontsize=10)
    else:
        axes[idx].text(0.5, 0.5, 'Not found', ha='center', va='center')
        axes[idx].set_title(f'"{class_name}" (not found)', fontsize=11)
        axes[idx].axis('off')

plt.suptitle('So sánh Visual Signatures - Mỗi loại âm thanh có "hình dạng" riêng', 
             fontsize=15, y=0.98, fontweight='bold')
plt.tight_layout()
plt.savefig('mel_spectrograms/mel_spectrogram_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: mel_spectrogram_comparison.png")

# =============================================================================
# 4. SO SÁNH AUGMENTATION
# =============================================================================

print("\n4. Minh họa data augmentation trên mel spectrogram...")
sample_row = df.iloc[5]
file_path = os.path.join(DATA_PATH, sample_row['filename'])
category = sample_row['category']

y, sr = librosa.load(file_path, sr=22050, duration=5.0)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# Original
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0], cmap='viridis')
axes[0].set_title(f'① Original\n"{category}"', fontsize=11, pad=10, fontweight='bold')

# Time stretch slow
y_slow = librosa.effects.time_stretch(y, rate=0.9)
mel_spec = librosa.feature.melspectrogram(y=y_slow, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1], cmap='viridis')
axes[1].set_title('② Time Stretch\n(0.9x speed)', fontsize=11, pad=10, fontweight='bold')

# Time stretch fast
y_fast = librosa.effects.time_stretch(y, rate=1.1)
mel_spec = librosa.feature.melspectrogram(y=y_fast, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2], cmap='viridis')
axes[2].set_title('③ Time Stretch\n(1.1x speed)', fontsize=11, pad=10, fontweight='bold')

# Pitch shift up
y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
mel_spec = librosa.feature.melspectrogram(y=y_pitch_up, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[3], cmap='viridis')
axes[3].set_title('④ Pitch Shift\n(+2 semitones)', fontsize=11, pad=10, fontweight='bold')

# Pitch shift down
y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
mel_spec = librosa.feature.melspectrogram(y=y_pitch_down, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[4], cmap='viridis')
axes[4].set_title('⑤ Pitch Shift\n(-2 semitones)', fontsize=11, pad=10, fontweight='bold')

# Add noise
noise = np.random.randn(len(y)) * 0.005
y_noise = y + noise
mel_spec = librosa.feature.melspectrogram(y=y_noise, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[5], cmap='viridis')
axes[5].set_title('⑥ Gaussian Noise\n(σ=0.005)', fontsize=11, pad=10, fontweight='bold')

for ax in axes:
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Freq (Hz)', fontsize=9)

plt.suptitle('Data Augmentation - Tạo 6 "ảnh" khác nhau từ 1 audio', 
             fontsize=15, y=0.98, fontweight='bold')
plt.tight_layout()
plt.savefig('mel_spectrograms/mel_spectrogram_augmentation.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: mel_spectrogram_augmentation.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("✅ HOÀN TẤT! Kiểm tra folder 'mel_spectrograms/':")
print("="*70)
print("📁 mel_spectrograms/")
print("   ├─ mel_spectrogram_examples.png       (8 samples ngẫu nhiên)")
print("   ├─ mel_spectrogram_detailed.png       (waveform → spectrogram → CNN input)")
print("   ├─ mel_spectrogram_comparison.png     (so sánh 6 class khác nhau)")
print("   └─ mel_spectrogram_augmentation.png   (1 audio → 6 augmented versions)")
print("="*70)

print("\n💡 GIẢI THÍCH:")
print("   - Audio (1D) → Mel Spectrogram (2D) → CNN có thể xử lý")
print("   - Mel scale: Mô phỏng cách tai người nghe")
print("   - Mỗi class có 'visual signature' riêng")
print("   - CNN học nhận dạng patterns như nhận dạng mèo/chó trong ảnh")
print("="*70)

