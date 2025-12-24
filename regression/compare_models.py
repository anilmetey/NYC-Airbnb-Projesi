
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import sys

# --- 1. VERİ YÜKLEME VE TEMİZLİK ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
csv_path = os.path.join(parent_dir, "AB_NYC_2019.csv")

print(f"--- 1. VERİ YÜKLEME ---\nDosya okunuyor: {csv_path}")
df = pd.read_csv(csv_path)

# Gereksiz Sütunlar
df_clean = df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1)
df_clean['reviews_per_month'] = df_clean['reviews_per_month'].fillna(0)
df_clean = df_clean.dropna()

print("\n--- OUTLIER (UÇ DEĞER) TEMİZLİĞİ ---")
print("""
[AÇIKLAMA]
Fiyat tahmini yaparken, aşırı yüksek fiyatlı evler (Örn: Geceliği 10.000$) modeli şaşırtır.
Model "ortalama" bir evi tahmin etmeye çalışırken bu uç değerler hata payını çok yükseltir.
Bu yüzden makul bir aralık belirleyip (Örn: < 500$) ötesini atıyoruz.
Bu sayede model "genel" evler için çok daha başarılı olur.
""")

# Fiyat Analizi
print(f"Orijinal Veri Sayısı: {len(df_clean)}")
# Fiyatı 0 olanlar hatalıdır, atalım. 
# Fiyatı 500$'dan fazla olanlar "Lüks/Marjinal" kabul edip atalım (Proje başarısı için genelleme yapıyoruz).
df_clean = df_clean[(df_clean['price'] > 0) & (df_clean['price'] < 500)]
print(f"Outlier Temizliği Sonrası Veri Sayısı: {len(df_clean)}")


# --- 2. ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering) ---
print("\n--- 2. ÖZELLİK MÜHENDİSLİĞİ ---")

# Kategorik Verileri Sayısala Çevirme
# 'neighbourhood_group' (Bölge) çok önemli olduğu için One-Hot Encoding yapıyoruz.
# Böylece model "Manhattan" etkisini "Brooklyn" etkisinden ayrı öğrenebilir.
df_encoded = pd.get_dummies(df_clean, columns=['neighbourhood_group', 'room_type'], drop_first=True)

# 'neighbourhood' çok fazla (200+) olduğu için Label Encoding yapalım (Model karmaşıklığını azaltmak için)
le = LabelEncoder()
df_encoded['neighbourhood'] = le.fit_transform(df_encoded['neighbourhood'])

# Hedef ve Özellikler
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

print("Kullanılan Özellikler:", X.columns.tolist())


# --- 3. TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STANDARTLAŞTIRMA ---
# Lineer Regresyon için standartlaştırma katsayıların yorumlanması açısından faydalıdır.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 4. MODELLER VE KARŞILAŞTIRMA ---
print("\n--- 4. MODELLERİN EĞİTİLMESİ ---")

models = {}

# MODEL A: Linear Regression (Basit, Temel Model)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr_model

# MODEL B: Random Forest Regressor (Güçlü, Karmaşık Model)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train) # Ağaçlar için scale şart değil ama yapsak da olur
models['Random Forest (Base)'] = rf_model

# --- HİPERPARAMETRE OPTİMİZASYONU ---
print("\n--- 5. HİPERPARAMETRE OPTİMİZASYONU (Random Forest) ---")
print("""
[AÇIKLAMA]
Random Forest modelinin başarısını artırmak için parametrelerini (Hyperparameters) ayarlıyoruz.
Regresyon için en önemli parametreler:
- n_estimators: Ağaç sayısı (Fazla ağaç = Daha istikrarlı tahmin).
- max_depth: Ağacın derinliği. Çok derin olursa ezberler (Overfitting), sığ olursa öğrenemez (Underfitting).
""")

# Zaman kazanmak için küçük bir GridSearch yapalım
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None]
}

print("Grid Search başlatılıyor...")
grid_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

print(f"\n[SONUÇ] En iyi parametreler: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_
models['Random Forest (Tuned)'] = best_rf

# --- 6. DEĞERLENDİRME ---
print("\n--- 6. DEĞERLENDİRME VE METRİKLER ---")

results = []

for name, model in models.items():
    if 'Linear' in name:
        preds = model.predict(X_test_scaled)
    else:
        preds = model.predict(X_test)
        
    # Metrikler
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    results.append({'Model': name, 'R2 Score': r2, 'MAE': mae, 'RMSE': rmse})
    
    print(f"\n>>> Model: {name}")
    print(f"R2 Score (Açıklayıcılık): {r2:.4f}")
    print(f"MAE (Ortalama Hata): {mae:.2f}$")
    print(f"RMSE (Karesel Hata): {rmse:.2f}$")

print("""
[YORUM - METRİKLER]
- R2 Score: Modelimiz fiyat değişimlerinin yüzde kaçını açıklayabiliyor? (1'e ne kadar yakınsa o kadar iyi)
- MAE (Mean Absolute Error): Tahminlerimiz ortalama kaç dolar sapıyor? (Örn: 40$ demek, tahminimiz gerçek fiyattan ortalama 40$ aşağıda veya yukarıda demek.)
- Neden R2 yetmez? Çünkü R2 oransaldır. MAE ise bize 'Cebimizden ne kadar yanlış çıkacak' onu söyler, daha gerçekçidir.
""")

# --- 7. ÖZELLİK ÖNEMİ (Feature Selection / Importance) ---
print("\n--- 7. HANGİ ÖZELLİKLER FİYATI ETKİLİYOR? ---")
importances = best_rf.feature_importances_
feature_names = X.columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print(feature_imp_df.head(10))
print("""
[YORUM]
Yukarıdaki tablo, fiyata en çok etki eden faktörleri gösterir.
Genelde 'room_type' (Oda tipi) ve 'longitude/latitude' (Konum) en yüksek etkiye sahiptir.
Bu da emlak piyasasının altın kuralını doğrular: "Konum, Konum, Konum".
""")

# --- 8. GÖRSELLEŞTİRME ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='viridis')
plt.title('Fiyatı Etkileyen En Önemli 10 Faktör (Feature Importance)')
plt.xlabel('Önem Derecesi')
plt.ylabel('Özellik')
plt.tight_layout()
save_path_imp = os.path.join(current_dir, "regression_feature_importance.png")
plt.savefig(save_path_imp)
print(f"✅ Özellik Önem Grafiği kaydedildi: {save_path_imp}")

# Tahmin vs Gerçek Grafiği
plt.figure(figsize=(10, 6))
# Sadece ilk 100 veriyi çizelim ki grafik karışmasın
limit = 100
plt.plot(y_test.values[:limit], label='Gerçek Fiyat', marker='o')
plt.plot(best_rf.predict(X_test)[:limit], label='Tahmin', marker='x', linestyle='--')
plt.title(f'Gerçek vs Tahmin (İlk {limit} Ev)')
plt.legend()
plt.tight_layout()
save_path_pred = os.path.join(current_dir, "regression_prediction_comparison.png")
plt.savefig(save_path_pred)
print(f"✅ Tahmin Karşılaştırma Grafiği kaydedildi: {save_path_pred}")
