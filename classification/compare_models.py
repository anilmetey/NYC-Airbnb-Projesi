
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
# Last review ve reviews_per_month çok eksik olduğu için onları da atabiliriz veya doldurabiliriz.
# Bu örnekte basitlik adına drop ediyoruz.
df_clean = df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1)

# reviews_per_month'daki NaN değerleri 0 ile dolduralım (Yorum yoksa ayda 0 yorumdur mantığı)
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
# Hedef değişkenimiz 'Private room', 'Entire home/apt' gibi yazılar. Makine sayılarla çalışır.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n[BİLGİ] Hedef sınıflar kodlandı: {le.classes_} -> {np.unique(y_encoded)}")

# Features içinde kategorik varsa (neighbourhood_group gibi) onları da One-Hot Encoding yapmalıyız.
# Şu an seçtiğimiz feature_cols tamamen sayısal olduğu için buna gerek kalmadı.
# Fakat eğer 'neighbourhood_group' ekleseydik: pd.get_dummies() kullanacaktık.

# --- SPLIT (Train/Test Ayrımı) ---
# Veriyi %80 Eğitim, %20 Test olarak ayırıyoruz.
# stratify=y_encoded : Test setindeki sınıf oranlarının eğitim setiyle aynı olmasını sağlar (Dengesiz veri için önemli).
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# --- 3. STANDARTLAŞTIRMA (Standardization) ---
print("\n--- 3. STANDARTLAŞTIRMA (NEDEN GEREKLİ?) ---")
print("""
[AÇIKLAMA] 
Bazı algoritmalar (KNN, SVM, Neural Networks) verilerin ölçeğinden etkilenir.
Örneğin: 'Fiyat' 0-10000 arasında değişirken, 'Enlem' 40-41 arasında değişir.
Eğer standartlaştırma yapmazsak, model 'Fiyat'ı devasa bir sayı olduğu için çok önemli,
'Enlem'i ise küçük olduğu için önemsiz sanabilir.

Bu yüzden verileri standart bir aralığa (Ortalama=0, Varyans=1) çekiyoruz.
NOT: Ağaç tabanlı modeller (Random Forest, Decision Tree) buna ihtiyaç duymaz.
Ama modelleri karşılaştıracağımız için ikisine de vereceğiz veya KNN için ayrı scale edeceğiz.
""")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # DİKKAT: Test verisini de eğitim setinin istatistikleriyle dönüştürüyoruz (Veri sızıntısını önlemek için).


# --- 4. MODEL EĞİTİMİ VE KARŞILAŞTIRMA ---
print("\n--- 4. MODELLERİN EĞİTİLMESİ ---")

models = {}

# MODEL A: Random Forest (Mevcut Model - Ağaç Tabanlı)
# Standartlaştırma şart değildir ama zararı da olmaz.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) # Orijinal X_train verilebilir
models['Random Forest'] = rf_model

# MODEL B: K-Nearest Neighbors (KNN - Mesafe Tabanlı)
# Mutlaka Scaled veri kullanılmalı!
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
models['KNN (Scaled)'] = knn_model

# --- 5. HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV) ---
print("\n--- 5. HİPERPARAMETRE OPTİMİZASYONU (KNN İçin) ---")
print("""
[AÇIKLAMA]
Modelin başarısını artırmak için 'Hiperparametreleri' ayarlamalıyız.
KNN için:
- n_neighbors (K): Kaç komşuya bakılacak? (Az olursa gürültüye hassas, çok olursa genelleşir)
- weights: Komşuların oyu eşit mi (uniform) yoksa yakın olanın oyu daha mı değerli (distance)?
- metric: Mesafe nasıl ölçülecek? (Euclidean: kuş uçuşu, Manhattan: blok blok şehir mesafesi)
""")

# Aranacak parametre uzayı
param_grid = {
    'n_neighbors': [3, 5, 9, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print("En iyi parametreleri bulmak için Grid Search başlatılıyor (Bu işlem biraz sürebilir)...")
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"\n[SONUÇ] En iyi parametreler bulundu: {grid_search.best_params_}")
print(f"[SONUÇ] En iyi skor (Eğitim CV): {grid_search.best_score_:.4f}")

# En iyi modeli alalım
best_knn = grid_search.best_estimator_
models['KNN (Tuned)'] = best_knn


# --- 6. DEĞERLENDİRME VE SONUÇLARIN YORUMLANMASI ---
print("\n--- 6. DEĞERLENDİRME ---")

for name, model in models.items():
    print(f"\n>>> Model: {name}")
    
    # KNN ise scaled veri kullan, değilse normal (Gerçi RF scaled ile de çalışır ama doğrusu bu)
    if 'KNN' in name:
        preds = model.predict(X_test_scaled)
    else:
        preds = model.predict(X_test)
        
    acc = accuracy_score(y_test, preds)
    print(f"Doğruluk (Accuracy): {acc:.2%}")
    
    print("Sınıflandırma Raporu:")
    # target_names ile raporun okunabilir olmasını sağlıyoruz
    print(classification_report(y_test, preds, target_names=le.classes_))
    
    # PRECISION vs RECALL AÇIKLAMASI
    print("""
    [YORUM]
    - Precision (Kesinlik): Model "Bu ev Entire home" dediğinde ne kadar haklı? 
      (Yanlış pozitiflerden kaçınmak istiyorsak - örn: spam filtresi - buna bakarız.)
    - Recall (Duyarlılık): Gerçekten "Entire home" olanların kaçını yakalayabildik?
      (Gözden kaçırmanın maliyeti yüksekse - örn: kanser teşhisi - buna bakarız.)
    - F1-Score: İkisinin harmonik ortalamasıdır. Dengesiz verilerde Accuracy yerine buna bakmak daha doğrudur.
    """)

# --- 7. GÖRSELLEŞTİRME (Confusion Matrix) ---
# Tuned KNN ve Random Forest karşılaştırması
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

# plt.show() # Sunucuda çalıştığı için show yerine save tercih ettik.
