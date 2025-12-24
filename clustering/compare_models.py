
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# --- 1. VERİ YÜKLEME VE HAZIRLIK (Data Preparation) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
csv_path = os.path.join(parent_dir, "AB_NYC_2019.csv")

print(f"--- 1. VERİ YÜKLEME ---\nDosya okunuyor: {csv_path}")
df = pd.read_csv(csv_path)

# Sadece Konum ve Fiyat Odaklı Kümeleme
# Amacımız: Şehrin "fiyat ve konum" bölgelerini keşfetmek.
# Bu yüzden sadece 'latitude', 'longitude' ve 'price' kullanacağız.
df_cluster = df[['latitude', 'longitude', 'price']].copy()

# Fiyatları makul seviyede tutalım (Outlier temizliği - Regresyon ile aynı mantık)
df_cluster = df_cluster[(df_cluster['price'] > 0) & (df_cluster['price'] < 500)]

# --- 2. STANDARTLAŞTIRMA (Standardization) ---
print("\n--- 2. STANDARTLAŞTIRMA (Neden Önemli?) ---")
print("""
[AÇIKLAMA]
Kümeleme algoritmaları "mesafe" ölçerek çalışır (Öklid Mesafesi).
- Enlem/Boylam: 40.7 gibi çok küçük aralıkta değişir.
- Fiyat: 50$ - 500$ gibi büyük aralıkta değişir.
Eğer standartlaştırma yapmazsak, algoritma sadece Fiyat farkına bakar, konumu önemsemez.
Standartlaştırma (StandardScaler) ile hepsini eşit öneme getiriyoruz.
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)


# --- 3. ELBOW (DİRSEK) YÖNTEMİ İLE K SAYISINI BULMA ---
print("\n--- 3. OPTİMAL KÜME SAYISINI BULMA (ELBOW METHOD) ---")
print("""
[AÇIKLAMA]
Veriyi kaç parçaya (K) ayırmalıyız? Bunu anlamak için 1'den 10'a kadar deneriz.
Her denemede "Inertia" (Hata kareler toplamı) değerine bakarız.
Grafikte kırılma yaşanan nokta (Dirsek/Elbow) bize en uygun sayıyı verir.
""")

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Dirsek Grafiğini Çiz
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('Inertia (Hata)')
plt.title('Elbow (Dirsek) Yöntemi ile Optimal K Bulma')
plt.grid(True)
save_path_elbow = os.path.join(current_dir, "clustering_elbow.png")
plt.savefig(save_path_elbow)
print(f"✅ Dirsek Grafiği kaydedildi: {save_path_elbow}")


# --- 4. MODELİ EĞİTME (K-Means) ---
print("\n--- 4. K-MEANS MODELİ EĞİTİLİYOR ---")
# Dirsek yönteminden K=5 veya K=6 genelde iyi çıkar. Biz K=5 seçelim.
optimal_k = 5
print(f"Seçilen Küme Sayısı: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Kümeleri ana veriye ekle
df_cluster['Cluster'] = labels

# --- 5. DEĞERLENDİRME (Silhouette Score) ---
print("\n--- 5. MODEL BAŞARISI (Silhouette Score) ---")
print("""
[AÇIKLAMA]
Silhouette Score: Kümelerin ne kadar ayrık olduğunu gösterir.
- 1'e yakınsa: Kümeler birbirinden çok net ayrılmış (Mükemmel).
- 0'a yakınsa: Kümeler iç içe geçmiş.
- Negatifse: Yanlış kümeleme yapılmış.
Bu işlem biraz uzun sürebilir (Veri büyük olduğu için örneklem alarak hesaplıyoruz).
""")

# Performans için 10.000 örnekle hesaplayalım
sample_indices = np.random.choice(len(X_scaled), 10000, replace=False)
X_sample = X_scaled[sample_indices]
labels_sample = labels[sample_indices]

score = silhouette_score(X_sample, labels_sample)
print(f"Silhouette Skoru: {score:.4f}")


# --- 6. SONUÇLARIN YORUMLANMASI ---
print("\n--- 6. KÜMELERİN ANALİZİ ---")
# Her kümenin ortalama fiyatını ve konumunu görelim
cluster_summary = df_cluster.groupby('Cluster').mean()[['price', 'latitude', 'longitude']]
cluster_counts = df_cluster['Cluster'].value_counts()
cluster_summary['Count'] = cluster_counts

print(cluster_summary)
print("""
[YORUM]
Yukarıdaki tablo bize her kümenin karakterini anlatır.
Örneğin: 
- Bir kümenin fiyatı çok yüksekse (Örn: 200$+), orası Manhattan olabilir.
- Başka bir kümenin fiyatı düşükse (Örn: 60$), orası şehrin dış kesimleri olabilir.
""")

# --- 7. HARİTA ÜZERİNDE GÖRSELLEŞTİRME ---
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_cluster, 
    x='longitude', 
    y='latitude', 
    hue='Cluster', 
    palette='tab10', 
    alpha=0.6,
    s=15
)
plt.title(f'NYC Airbnb Fiyat ve Konum Kümeleri (K={optimal_k})')
plt.xlabel('Boylam (Longitude)')
plt.ylabel('Enlem (Latitude)')
plt.legend(title='Küme No')
plt.axis('equal') # Harita oranlarını koru
save_path_map = os.path.join(current_dir, "clustering_map.png")
plt.savefig(save_path_map)
print(f"✅ Harita dosyası kaydedildi: {save_path_map}")
