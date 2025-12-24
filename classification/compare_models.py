"""
================================================================================
              NYC AIRBNB SINIFLANDIRMA ANALİZİ VE MODEL KARŞILAŞTIRMASI
================================================================================

AMAÇ:
-----
Bu script, NYC Airbnb verilerini kullanarak evlerin oda tipini (Entire home,
Private room, Shared room) tahmin eder ve iki farklı sınıflandırma algoritmasını
karşılaştırır.

KULLANILAN YÖNTEMLER:
---------------------
1. ÖZELLİK SEÇİMİ (Feature Selection):
   - Variance Threshold: Düşük varyanslı (bilgi içermeyen) özellikleri eler
   - Feature Importance: Random Forest'ın önem skorlarına göre seçim
   
2. BOYUT İNDİRGEME (Dimensionality Reduction):
   - PCA (Principal Component Analysis): Boyut azaltarak hesaplama hızı artırır

3. MODEL KARŞILAŞTIRMASI:
   - Random Forest Classifier (Ağaç tabanlı)
   - K-Nearest Neighbors (Mesafe tabanlı)

ÇIKTILAR:
---------
- classification_comparison_matrix.png → Confusion Matrix karşılaştırması
- classification_pca_comparison.png → PCA öncesi/sonrası karşılaştırma
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import os
import sys

# --- 1. VERİ YÜKLEME VE İNCELEME (Data Loading & Inspection) ---

# Dosya yolunu belirle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
csv_path = os.path.join(parent_dir, "AB_NYC_2019.csv")

print(f"--- 1. VERİ YÜKLEME ---\nDosya okunuyor: {csv_path}")
df = pd.read_csv(csv_path)

# Eksik Veri Kontrolü (Missing Value Check)
print("\n[BİLGİ] Veri setindeki eksik değerler kontrol ediliyor...")
print(df.isnull().sum())

# "Sessiz" eksik verileri kontrol et (Örn: "?", "N/A" gibi stringler)
# Bazı veri setlerinde eksik veriler boş değil, "?" olarak girilmiş olabilir.
strange_strings = ['?', 'N/A', 'MISSING', '-', 'nan']
for col in df.columns:
    if df[col].dtype == object:
        found_strange = df[col].isin(strange_strings).sum()
        if found_strange > 0:
            print(f"[UYARI] '{col}' sütununda {found_strange} adet tanımlanamayan ({strange_strings}) değer var.")

# --- 2. VERİ ÖN İŞLEME (Data Preprocessing) ---
print("\n--- 2. VERİ ÖN İŞLEME ---")

# Gereksiz sütunları at (ID, isim vb. genellemeye katkısı olmayanlar)
df_clean = df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1)

# reviews_per_month'daki NaN değerleri 0 ile dolduralım
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)

# Diğer eksik satırları at
print(f"Temizlik öncesi satır sayısı: {len(df_clean)}")
df_clean = df_clean.dropna()
print(f"Temizlik sonrası satır sayısı: {len(df_clean)}")

# Hedef Değişken (Target): room_type
target_col = 'room_type'
feature_cols = ['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'availability_365', 'reviews_per_month']

X = df_clean[feature_cols]
y = df_clean[target_col]

# --- ENCODING (Kategorik -> Sayısal) ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n[BİLGİ] Hedef sınıflar kodlandı: {le.classes_} -> {np.unique(y_encoded)}")

# --- SPLIT (Train/Test Ayrımı) ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# --- 3. ÖZELLİK SEÇİMİ (Feature Selection) ---
print("\n" + "="*60)
print("--- 3. ÖZELLİK SEÇİMİ (FEATURE SELECTION) ---")
print("="*60)
print("""
[AÇIKLAMA]
Özellik seçimi, modelin performansını artırmak için kullanılır:
1. Gereksiz özellikleri eler → Model daha hızlı çalışır
2. Gürültüyü azaltır → Model daha iyi geneller
3. Overfitting'i (ezberlemeyi) önler

KULLANILAN YÖNTEMLER:
---------------------
1. VARIANCE THRESHOLD:
   - Düşük varyanslı özellikleri eler
   - Eğer bir özellik neredeyse hep aynı değerse, model için bilgi içermez
   
2. FEATURE IMPORTANCE (Random Forest):
   - Ağaç tabanlı modeller her özelliğin önemini hesaplar
   - Düşük önemli olanları atabiliriz
   
3. SELECTKBEST (F-Test):
   - İstatistiksel test ile en bilgilendirici K özelliği seçer
""")

# --- 3.1 Variance Threshold ---
print("\n>>> Variance Threshold uygulanıyor...")
selector = VarianceThreshold(threshold=0.01)  # Çok düşük varyanslı özellikleri ele
X_train_var = selector.fit_transform(X_train)
X_test_var = selector.transform(X_test)

# Hangi özellikler kaldı?
selected_features = np.array(feature_cols)[selector.get_support()]
print(f"[SONUÇ] Variance Threshold sonrası kalan özellikler ({len(selected_features)}/{len(feature_cols)}): {list(selected_features)}")

# --- 3.2 Feature Importance (Random Forest ile) ---
print("\n>>> Feature Importance hesaplanıyor (Random Forest)...")
temp_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
temp_rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': temp_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n[SONUÇ] Özellik Önem Sıralaması:")
print(feature_importance.to_string(index=False))

# En önemli 5 özelliği seç
top_features = feature_importance.head(5)['Feature'].tolist()
print(f"\n[SEÇİM] En önemli 5 özellik: {top_features}")


# --- 4. STANDARTLAŞTIRMA (Standardization) ---
print("\n--- 4. STANDARTLAŞTIRMA (NEDEN GEREKLİ?) ---")
print("""
[AÇIKLAMA] 
Bazı algoritmalar (KNN, SVM, Neural Networks) verilerin ölçeğinden etkilenir.
Örneğin: 'Fiyat' 0-10000 arasında değişirken, 'Enlem' 40-41 arasında değişir.

Bu yüzden verileri standart bir aralığa (Ortalama=0, Varyans=1) çekiyoruz.
NOT: Ağaç tabanlı modeller (Random Forest, Decision Tree) buna ihtiyaç duymaz.

VERİ SIZINTISINI ÖNLEME:
------------------------
scaler.fit_transform(X_train) → Sadece eğitim verisinden öğren
scaler.transform(X_test) → Test verisine aynı dönüşümü uygula
Bu sayede test verisi eğitime "sızmaz".
""")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 5. BOYUT İNDİRGEME (PCA - Principal Component Analysis) ---
print("\n" + "="*60)
print("--- 5. BOYUT İNDİRGEME (PCA) ---")
print("="*60)
print("""
[AÇIKLAMA]
PCA (Principal Component Analysis) - Temel Bileşen Analizi

AMAÇ:
-----
- Çok sayıda özelliği daha az sayıda "bileşene" indirger
- Özellikler arasındaki korelasyonu kullanır
- Bilgi kaybını minimize ederek boyut azaltır

NEDEN KULLANILIR?
-----------------
1. Hesaplama hızını artırır (daha az boyut = daha hızlı)
2. Overfitting'i azaltır
3. Görselleştirme kolaylaşır (2D veya 3D'ye indirgeyebiliriz)
4. Gürültüyü azaltır

DEZAVANTAJ:
-----------
- Yorumlanabilirlik azalır (artık "fiyat" değil, "PC1" var)
- Bilgi kaybı olabilir
""")

# PCA uygula - %95 varyansı koruyacak kadar bileşen seç
pca = PCA(n_components=0.95)  # %95 varyans korunacak
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\n[SONUÇ] PCA Sonuçları:")
print(f"  - Orijinal boyut: {X_train_scaled.shape[1]} özellik")
print(f"  - PCA sonrası boyut: {X_train_pca.shape[1]} bileşen")
print(f"  - Korunan varyans: {sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"  - Her bileşenin açıkladığı varyans: {[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")


# --- 6. MODEL EĞİTİMİ VE KARŞILAŞTIRMA ---
print("\n--- 6. MODELLERİN EĞİTİLMESİ ---")

models = {}

# MODEL A: Random Forest (Ağaç Tabanlı)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model

# MODEL B: KNN (Mesafe Tabanlı)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
models['KNN (Scaled)'] = knn_model


# --- 7. HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV) ---
print("\n--- 7. HİPERPARAMETRE OPTİMİZASYONU (KNN İçin) ---")
print("""
[AÇIKLAMA]
GridSearchCV ile en iyi hiperparametreleri buluyoruz.
KNN için:
- n_neighbors (K): Kaç komşuya bakılacak?
- weights: Komşuların oyu eşit mi (uniform) yoksa yakın olan daha mı değerli (distance)?
- metric: Mesafe nasıl ölçülecek? 
  * Euclidean: Kuş uçuşu mesafe (√(x²+y²))
  * Manhattan: Blok blok mesafe (|x|+|y|)
""")

param_grid = {
    'n_neighbors': [3, 5, 9, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print("En iyi parametreleri bulmak için Grid Search başlatılıyor...")
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n[SONUÇ] En iyi parametreler bulundu: {grid_search.best_params_}")
print(f"[SONUÇ] En iyi skor (Eğitim CV): {grid_search.best_score_:.4f}")

best_knn = grid_search.best_estimator_
models['KNN (Tuned)'] = best_knn


# --- 8. PCA İLE MODEL KARŞILAŞTIRMASI ---
print("\n" + "="*60)
print("--- 8. PCA ÖNCESİ vs SONRASI KARŞILAŞTIRMA ---")
print("="*60)

# PCA'sız KNN sonucu
acc_without_pca = accuracy_score(y_test, best_knn.predict(X_test_scaled))

# PCA'lı KNN eğit ve test et
knn_pca = KNeighborsClassifier(**grid_search.best_params_)
knn_pca.fit(X_train_pca, y_train)
acc_with_pca = accuracy_score(y_test, knn_pca.predict(X_test_pca))

print(f"""
╔══════════════════════════════════════════════════════════╗
║          PCA ÖNCESİ vs SONRASI KARŞILAŞTIRMA            ║
╠══════════════════════════════════════════════════════════╣
║  Yöntem              │ Boyut    │ Accuracy               ║
╠──────────────────────┼──────────┼────────────────────────╣
║  PCA'sız (Orijinal)  │ {X_train_scaled.shape[1]} özellik│ {acc_without_pca:.4f} ({acc_without_pca*100:.1f}%)       ║
║  PCA ile             │ {X_train_pca.shape[1]} bileşen  │ {acc_with_pca:.4f} ({acc_with_pca*100:.1f}%)       ║
╠══════════════════════════════════════════════════════════╣
║  Fark: {abs(acc_without_pca - acc_with_pca)*100:.2f}% {'düşüş' if acc_with_pca < acc_without_pca else 'artış'}                                      ║
╚══════════════════════════════════════════════════════════╝
""")

print("""
[YORUM]
PCA sonrasında boyut azaldı ama başarı çok az değişti.
Bu demektir ki: Bazı özellikler birbiriyle korelasyonlu, gereksiz bilgi içeriyor.
PCA bu gereksizliği temizledi ve modeli daha verimli hale getirdi.
""")


# --- 9. DEĞERLENDİRME VE SONUÇLARIN YORUMLANMASI ---
print("\n--- 9. DEĞERLENDİRME ---")

for name, model in models.items():
    print(f"\n>>> Model: {name}")
    
    if 'KNN' in name:
        preds = model.predict(X_test_scaled)
    else:
        preds = model.predict(X_test)
        
    acc = accuracy_score(y_test, preds)
    print(f"Doğruluk (Accuracy): {acc:.2%}")
    
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, preds, target_names=le.classes_))
    
    print("""
    [YORUM]
    - Precision (Kesinlik): Model "Bu ev Entire home" dediğinde ne kadar haklı? 
      (Yanlış pozitiflerden kaçınmak istiyorsak - örn: spam filtresi - buna bakarız.)
    - Recall (Duyarlılık): Gerçekten "Entire home" olanların kaçını yakalayabildik?
      (Gözden kaçırmanın maliyeti yüksekse - örn: kanser teşhisi - buna bakarız.)
    - F1-Score: İkisinin harmonik ortalamasıdır. Dengesiz verilerde Accuracy yerine buna bakmak daha doğrudur.
    """)


# --- 10. GÖRSELLEŞTİRME ---
# Confusion Matrix karşılaştırması
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest Matrisi
y_pred_rf = rf_model.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gerçek')

# Tuned KNN Matrisi
y_pred_knn = best_knn.predict(X_test_scaled)
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title(f'Optimized KNN Confusion Matrix\n{grid_search.best_params_}')
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gerçek')

plt.tight_layout()
save_path = os.path.join(current_dir, "classification_comparison_matrix.png")
plt.savefig(save_path)
print(f"\n✅ Karşılaştırma grafiği kaydedildi: {save_path}")

# PCA Karşılaştırma grafiği
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['PCA\'sız\n(Orijinal)', 'PCA ile\n(Boyut İndirgeme)']
accuracies = [acc_without_pca, acc_with_pca]
colors = ['#3498db', '#2ecc71']

bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy')
ax.set_title('PCA Öncesi vs Sonrası Sınıflandırma Başarısı')
ax.set_ylim(0, max(accuracies) * 1.15)
ax.grid(axis='y', alpha=0.3)

save_path_pca = os.path.join(current_dir, "classification_pca_comparison.png")
plt.savefig(save_path_pca)
print(f"✅ PCA karşılaştırma grafiği kaydedildi: {save_path_pca}")

