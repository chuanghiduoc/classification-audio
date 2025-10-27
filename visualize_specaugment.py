# -*- coding: utf-8 -*-
"""
VISUALIZE SPECAUGMENT EFFECT
Hiển thị tác dụng của SpecAugment lên mel spectrogram
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
import os

# =============================================================================
# SPECAUGMENT CLASS
# =============================================================================

class SpecAugment:
    """SpecAugment: Frequency & Time Masking"""
    def __init__(self, freq_mask_param=20, time_mask_param=20, n_freq_masks=2, n_time_masks=2):
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
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, num_freq_bins - f)
            spec[f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, num_time_frames - t)
            spec[:, t0:t0 + t] = 0
        
        if had_channel:
            spec = np.expand_dims(spec, axis=0)
        
        return spec

# =============================================================================
# EXTRACT MEL SPECTROGRAM
# =============================================================================

def extract_mel_spectrogram(file_path, img_size=128):
    """Extract mel spectrogram from audio file"""
    y, sr = librosa.load(file_path, sr=22050, duration=5.0)
    
    if len(y) < sr * 5:
        y = np.pad(y, (0, sr * 5 - len(y)), mode='constant')
    
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=8000
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Resize
    mel_spec_resized = cv2.resize(mel_spec_norm, (img_size, img_size))
    
    return mel_spec_resized

# =============================================================================
# VISUALIZE
# =============================================================================

print("="*70)
print("VISUALIZE SPECAUGMENT")
print("="*70)

# Load a sample audio
audio_dir = 'data/audio/audio/'
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')][:5]

# Create figure
fig, axes = plt.subplots(len(audio_files), 4, figsize=(20, 4*len(audio_files)))

spec_augment = SpecAugment(freq_mask_param=20, time_mask_param=20, n_freq_masks=2, n_time_masks=2)

for i, audio_file in enumerate(audio_files):
    file_path = os.path.join(audio_dir, audio_file)
    
    # Extract original mel spectrogram
    mel_spec = extract_mel_spectrogram(file_path)
    
    # Apply SpecAugment 3 times
    aug1 = spec_augment(mel_spec)
    aug2 = spec_augment(mel_spec)
    aug3 = spec_augment(mel_spec)
    
    # Plot
    axes[i, 0].imshow(mel_spec, cmap='viridis', aspect='auto', origin='lower')
    axes[i, 0].set_title(f'{audio_file}\nOriginal', fontsize=10)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(aug1, cmap='viridis', aspect='auto', origin='lower')
    axes[i, 1].set_title('SpecAugment #1', fontsize=10)
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(aug2, cmap='viridis', aspect='auto', origin='lower')
    axes[i, 2].set_title('SpecAugment #2', fontsize=10)
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(aug3, cmap='viridis', aspect='auto', origin='lower')
    axes[i, 3].set_title('SpecAugment #3', fontsize=10)
    axes[i, 3].axis('off')

plt.suptitle('SpecAugment: Random Frequency & Time Masking', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('mel_spectrograms/mel_spectrogram_augmentation.png', dpi=150, bbox_inches='tight')
print("\n=> Đã lưu: mel_spectrograms/mel_spectrogram_augmentation.png")
print("\nMỗi lần apply SpecAugment sẽ tạo ra các masks khác nhau (random)")
print("Điều này giúp model học robust hơn và tránh overfitting!")
print("="*70)

