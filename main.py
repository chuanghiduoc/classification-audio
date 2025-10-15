# -*- coding: utf-8 -*-
"""
PHÂN LOẠI ÂM THANH SỬ DỤNG DATASET ESC50
Chương trình phân loại âm thanh sử dụng đặc trưng nâng cao và Random Forest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy import stats
import seaborn as sns
from joblib import Parallel, delayed, dump  

# Cấu hình
DATA_PATH = 'data/audio/audio/'
CSV_PATH = 'data/esc50.csv'
SKIP_HYPERPARAMETER_TUNING = True

# =============================================================================
# ĐỌC VÀ PHÂN TÍCH DỮ LIỆU
# =============================================================================

df = pd.read_csv(CSV_PATH)
target_to_category = dict(zip(df['target'], df['category']))

print("\n" + "="*70)
print("DATASET ESC50")
print("="*70)
print(f"So luong mau: {len(df)} | So luong lop: {df['target'].nunique()}")
print(df[['filename', 'category', 'target']].head())

# Phân bố dữ liệu
plt.figure(figsize=(12, 6))
sns.barplot(x=df['category'].value_counts().index, y=df['category'].value_counts().values)
plt.xticks(rotation=90)
plt.title('Phan bo cac lop trong dataset')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()
print("=> Da luu: class_distribution.png")

# =============================================================================
# HÀM TRÍCH XUẤT ĐẶC TRƯNG
# =============================================================================

def extract_advanced_features(file_path, n_mfcc=40, n_mels=128, n_fft=2048, hop_length=512):
    """Trích xuất đặc trưng âm thanh nâng cao với cải tiến"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # MFCC và đạo hàm
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        mfccs_max = np.max(mfccs.T, axis=0)
        mfccs_min = np.min(mfccs.T, axis=0)
        mfccs_skew = np.mean(librosa.feature.delta(mfccs, order=1), axis=1)
        mfccs_kurtosis = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
        
        # Mel Spectrogram với thống kê mở rộng
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel)
        mel_mean = np.mean(mel_db.T, axis=0)
        mel_std = np.std(mel_db.T, axis=0)
        mel_max = np.max(mel_db.T, axis=0)
        mel_median = np.median(mel_db.T, axis=0)
        
        # Tempogram
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        tempo_mean = np.mean(tempogram, axis=1)
        tempo_std = np.std(tempogram, axis=1)
        
        # Chroma với thống kê mở rộng
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        chroma_max = np.max(chroma, axis=1)
        
        # Các đặc trưng phổ
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        
        # RMS và Tonnetz
        rms = librosa.feature.rms(y=y)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Thêm các đặc trưng mới
        poly_features = librosa.feature.poly_features(y=y, sr=sr, order=1)
        
        # Kết hợp tất cả đặc trưng
        features = np.concatenate((
            mfccs_mean, mfccs_std, mfccs_max, mfccs_min, mfccs_skew, mfccs_kurtosis,
            mel_mean[:20], mel_std[:20], mel_max[:20], mel_median[:20],
            tempo_mean[:10], tempo_std[:10],
            chroma_mean, chroma_std, chroma_max,
            [np.mean(zcr), np.std(zcr), np.max(zcr)],
            [np.mean(spectral_centroid), np.std(spectral_centroid), np.max(spectral_centroid)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth), np.max(spectral_bandwidth)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff), np.max(spectral_rolloff)],
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
            [np.mean(spectral_flatness), np.std(spectral_flatness)],
            [np.mean(rms), np.std(rms), np.max(rms)],
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),
            np.mean(poly_features, axis=1)
        ))
        
        return features
    except Exception as e:
        print(f"Loi: {file_path}: {e}")
        return None

def visualize_audio_features(file_path, save_path):
    """Trực quan hóa đặc trưng âm thanh"""
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(15, 10))
    
    subplots = [
        ('Waveform', lambda: librosa.display.waveshow(y, sr=sr)),
        ('MFCC', lambda: librosa.display.specshow(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), sr=sr, x_axis='time')),
        ('Mel Spectrogram', lambda: librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr)), sr=sr, x_axis='time', y_axis='mel')),
        ('Chromagram', lambda: librosa.display.specshow(librosa.feature.chroma_stft(y=y, sr=sr), sr=sr, x_axis='time', y_axis='chroma')),
        ('Spectral Contrast', lambda: librosa.display.specshow(librosa.feature.spectral_contrast(y=y, sr=sr), sr=sr, x_axis='time')),
        ('Tonnetz', lambda: librosa.display.specshow(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), sr=sr, x_axis='time'))
    ]
    
    for i, (title, plot_func) in enumerate(subplots, 1):
        plt.subplot(3, 2, i)
        plot_func()
        if i > 1:
            plt.colorbar()
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# TRỰC QUAN HÓA MẪU
# =============================================================================

print("\n" + "="*70)
print("TRUC QUAN HOA DAC TRUNG")
print("="*70)

if not os.path.exists('feature_analysis'):
    os.makedirs('feature_analysis')

for category in df['category'].unique()[:5]:
    sample_file = df[df['category'] == category].iloc[0]['filename']
    visualize_audio_features(os.path.join(DATA_PATH, sample_file), f'feature_analysis/{category}.png')
    print(f"=> Da luu: feature_analysis/{category}.png")

# =============================================================================
# TRÍCH XUẤT ĐẶC TRƯNG VÀ CHUẨN BỊ DỮ LIỆU
# =============================================================================

print("\n" + "="*70)
print("TRICH XUAT DAC TRUNG")
print("="*70)

file_paths = [os.path.join(DATA_PATH, row['filename']) for _, row in df.iterrows()]
labels = df['target'].values

print(f"Dang trich xuat dac trung cho {len(file_paths)} file...")
features_advanced = Parallel(n_jobs=-1)(delayed(extract_advanced_features)(fp) for fp in file_paths)

valid_indices = [i for i, f in enumerate(features_advanced) if f is not None]
features_advanced = np.array([features_advanced[i] for i in valid_indices])
labels_advanced = np.array([labels[i] for i in valid_indices])

print(f"=> Thanh cong: {len(features_advanced)} mau, {features_advanced.shape[1]} chieu")

X_train_advanced, X_test_advanced, y_train_advanced, y_test_advanced = train_test_split(
    features_advanced, labels_advanced, test_size=0.2, random_state=42, stratify=labels_advanced
)
print(f"=> Train: {X_train_advanced.shape[0]} | Test: {X_test_advanced.shape[0]}")

# =============================================================================
# TIỀN XỬ LÝ DỮ LIỆU
# =============================================================================

print("\n" + "="*70)
print("TIEN XU LY")
print("="*70)

# 1. Chuẩn hóa với RobustScaler (tốt hơn cho outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_advanced)
X_test_scaled = scaler.transform(X_test_advanced)
print(f"1. Chuan hoa (RobustScaler): Train {X_train_scaled.shape} | Test {X_test_scaled.shape}")

# 2. Không loại bỏ outliers (giữ lại toàn bộ dữ liệu)
X_train_cleaned = X_train_scaled
y_train_cleaned = y_train_advanced
print(f"2. Giu lai toan bo du lieu train (khong loai outliers)")

# 3. Feature Selection với SelectKBest (thay vì PCA)
from sklearn.feature_selection import SelectKBest, f_classif
n_features_to_select = min(200, X_train_cleaned.shape[1])  # Chọn top 200 features
selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
X_train_selected = selector.fit_transform(X_train_cleaned, y_train_cleaned)
X_test_selected = selector.transform(X_test_scaled)
print(f"3. Feature Selection: {X_train_cleaned.shape[1]} -> {X_train_selected.shape[1]} features (SelectKBest)")

# Hiển thị top features quan trọng
feature_scores = selector.scores_
top_indices = np.argsort(feature_scores)[::-1][:10]
print(f"   Top 10 feature scores: {feature_scores[top_indices][:5]}...")

# 4. Data Augmentation với ADASYN (thay vì SMOTE)
from sklearn.utils import resample
from imblearn.over_sampling import ADASYN, SMOTE
print(f"4. Data Augmentation voi ADASYN...")

# Sử dụng ADASYN - adaptive synthetic sampling (thông minh hơn SMOTE)
try:
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_train_final, y_train_final = adasyn.fit_resample(X_train_selected, y_train_cleaned)
    print(f"   Sau ADASYN: {X_train_final.shape[0]} mau (adaptive synthetic)")
except Exception as e:
    # Fallback 1: Thử SMOTE nếu ADASYN không hoạt động
    try:
        print(f"   ADASYN loi, thu SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_final, y_train_final = smote.fit_resample(X_train_selected, y_train_cleaned)
        print(f"   Sau SMOTE: {X_train_final.shape[0]} mau")
    except:
        # Fallback 2: Oversample thông thường
        print(f"   SMOTE loi, dung oversample thuong...")
        X_train_aug = []
        y_train_aug = []
        
        for class_label in np.unique(y_train_cleaned):
            class_samples = X_train_selected[y_train_cleaned == class_label]
            class_labels = y_train_cleaned[y_train_cleaned == class_label]
            
            # Oversample tất cả classes về cùng số lượng
            target_samples = 35
            if len(class_samples) < target_samples:
                class_samples_resampled = resample(class_samples, n_samples=target_samples, random_state=42, replace=True)
                class_labels_resampled = np.full(target_samples, class_label)
            else:
                class_samples_resampled = class_samples
                class_labels_resampled = class_labels
            
            X_train_aug.append(class_samples_resampled)
            y_train_aug.append(class_labels_resampled)
        
        X_train_final = np.vstack(X_train_aug)
        y_train_final = np.hstack(y_train_aug)
        print(f"   Sau augmentation: {X_train_final.shape[0]} mau")

print(f"\n=> Du lieu cuoi: Train {X_train_final.shape} | Test {X_test_selected.shape}")

# =============================================================================
# HUẤN LUYỆN RANDOM FOREST
# =============================================================================

print("\n" + "="*70)
print("HUAN LUYEN RANDOM FOREST")
print("="*70)

if SKIP_HYPERPARAMETER_TUNING:
    print("Su dung tham so toi uu cai tien...")
    best_rf = RandomForestClassifier(
        n_estimators=800,  # Tăng số cây lên cao hơn
        max_depth=None,    # Không giới hạn độ sâu
        min_samples_split=2,  
        min_samples_leaf=1,   
        max_features='sqrt', 
        class_weight='balanced',
        bootstrap=True, 
        oob_score=True,
        criterion='gini',
        random_state=42, 
        n_jobs=-1
    )
    best_rf.fit(X_train_final, y_train_final)
    print(f"=> OOB Score: {best_rf.oob_score_:.4f}")
else:
    param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [30, 50, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, bootstrap=True, oob_score=True, n_jobs=-1),
        param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy'
    )
    grid_search.fit(X_train_final, y_train_final)
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    best_rf = grid_search.best_estimator_

# Đánh giá
y_pred_advanced = best_rf.predict(X_test_selected)
accuracy_advanced = accuracy_score(y_test_advanced, y_pred_advanced)

print(f"\n=> Do chinh xac: {accuracy_advanced:.4f} ({accuracy_advanced*100:.2f}%)")
print(f"   Dung: {np.sum(y_pred_advanced == y_test_advanced)}/{len(y_test_advanced)}")
print(f"\nBao cao chi tiet:")
print(classification_report(y_test_advanced, y_pred_advanced))

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test_advanced, y_pred_advanced), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Thuc Te')
plt.xlabel('Du Doan')
plt.tight_layout()
plt.savefig('confusion_matrix_RF_advanced.png')
plt.close()

# Feature importance
feature_importance = best_rf.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1][:30]
plt.figure(figsize=(10, 8))
plt.barh(range(30), feature_importance[sorted_idx])
plt.yticks(range(30), [f"Feature {i}" for i in sorted_idx])
plt.xlabel('Importance')
plt.title('Top 30 Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

dump(best_rf, 'best_rf_model.joblib')
print("=> Da luu: confusion_matrix_RF_advanced.png, feature_importance.png, best_rf_model.joblib")

# =============================================================================
# SO SÁNH CÁC THUẬT TOÁN
# =============================================================================

print("\n" + "="*70)
print("SO SANH THUAT TOAN")
print("="*70)

def train_and_evaluate(name, model):
    """Huấn luyện và đánh giá mô hình"""
    print(f"{name}...", end=" ")
    model.fit(X_train_final, y_train_final)
    y_pred = model.predict(X_test_selected)
    acc = accuracy_score(y_test_advanced, y_pred)
    print(f"{acc:.4f}")
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test_advanced, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Thuc Te')
    plt.xlabel('Du Doan')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()
    
    return acc, y_pred

from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

models = {
    'Random Forest': best_rf,
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2),
    'SVM': SVC(kernel='rbf', C=100, gamma='scale', random_state=42, probability=True),  # Tăng C lên cao hơn
    'Neural Network': MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', max_iter=2000, random_state=42, early_stopping=True, learning_rate='adaptive'),
    # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1, eval_metric='mlogloss')
}

results = {'Random Forest': accuracy_advanced}
trained_models = {'Random Forest': best_rf}

for name, model in list(models.items())[1:]:  # Skip RF (already trained)
    acc, _ = train_and_evaluate(name, model)
    results[name] = acc
    trained_models[name] = model

# Tạo Ensemble Voting Classifier từ top models
print(f"\nEnsemble Model...")
top_models = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
voting_estimators = [(name, trained_models[name]) for name, _ in top_models]
voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
voting_clf.fit(X_train_final, y_train_final)
y_pred_voting = voting_clf.predict(X_test_selected)
voting_acc = accuracy_score(y_test_advanced, y_pred_voting)
results['Ensemble Voting'] = voting_acc
print(f"Ensemble Voting... {voting_acc:.4f}")

# Confusion matrix cho Ensemble
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test_advanced, y_pred_voting), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Ensemble Voting')
plt.ylabel('Thuc Te')
plt.xlabel('Du Doan')
plt.tight_layout()
plt.savefig('confusion_matrix_Ensemble_Voting.png')
plt.close()

# So sánh
comparison_df = pd.DataFrame(list(results.items()), columns=['Thuat toan', 'Do chinh xac'])
comparison_df = comparison_df.sort_values('Do chinh xac', ascending=False)

print(f"\n=> BANG XEP HANG:")
print(comparison_df.to_string(index=False))
print(f"\n=> Tot nhat: {comparison_df.iloc[0]['Thuat toan']} - {comparison_df.iloc[0]['Do chinh xac']:.4f}")

# Biểu đồ
plt.figure(figsize=(14, 6))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
bars = plt.bar(comparison_df['Thuat toan'], comparison_df['Do chinh xac'], color=colors[:len(comparison_df)])
plt.ylabel('Accuracy')
plt.title('Model Comparison - Audio Classification')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("=> Da luu: model_comparison.png va cac confusion matrices")

# =============================================================================
# TỔNG KẾT
# =============================================================================

print("\n" + "="*70)
print("HOAN TAT!")
print("="*70)
print(f"Features: {features_advanced.shape[1]} -> {X_train_final.shape[1]} (sau Feature Selection)")
print(f"Training samples: {X_train_advanced.shape[0]} -> {X_train_final.shape[0]} (sau augmentation)")
print(f"Models: {len(results)} thuat toan")
print("\nCai tien da thuc hien (v3 - MỚI NHẤT):")
print("  1. Them cac dac trung: max, min, median cho MFCC va Mel")
print("  2. Them poly_features de bat thong tin tần số")
print("  3. RobustScaler thay StandardScaler (tot hon cho outliers)")
print("  4. Feature Selection (SelectKBest) thay PCA (giu features goc)")
print("  5. ADASYN thay SMOTE (adaptive synthetic sampling)")
print("  6. Tang RF estimators: 500 -> 800")
print("  7. Toi uu SVM: C=10 -> C=100")
print("  8. Them Gradient Boosting va XGBoost")
print("  9. Ensemble Voting tu top 3 models")
print("  10. Cai thien Neural Network architecture")
print("\nFiles:")
print("  - class_distribution.png")
print("  - feature_analysis/*.png")
print("  - confusion_matrix_*.png")
print("  - feature_importance.png")
print("  - model_comparison.png")
print("  - best_rf_model.joblib")
print("="*70)
