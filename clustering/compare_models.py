"""
================================================================================
                    NYC AIRBNB KÜMELEME ANALİZİ VE MODEL KARŞILAŞTIRMASI
================================================================================

AMAÇ:
-----
Bu script, NYC Airbnb verilerini kullanarak evleri konum ve fiyata göre gruplar
(kümeler) ve iki farklı kümeleme algoritmasının performansını karşılaştırır.

KULLANILAN ALGORİTMALAR:
------------------------
1. K-MEANS CLUSTERING:
   - Merkez tabanlı (centroid-based) algoritma
   - Her kümenin merkez noktasını hesaplar
   - Veri noktalarını en yakın merkeze atar
   - Avantajı: Hızlı, büyük veri setlerinde verimli
   - Dezavantajı: Kümelerin dairesel/küresel olduğunu varsayar
   
2. AGGLOMERATIVE (HİYERARŞİK) CLUSTERING:
   - Alttan yukarıya (bottom-up) yaklaşım
   - Her nokta tek başına bir küme olarak başlar
   - En yakın kümeler adım adım birleştirilir
   - Avantajı: Farklı şekillerdeki kümeleri bulabilir
   - Dezavantajı: Büyük veri setlerinde yavaş (O(n²) karmaşıklık)

DEĞERLENDİRME METRİKLERİ:
-------------------------
1. INERTIA (Küme İçi Hata Kareler Toplamı):
   - Her noktanın kendi küme merkezine olan uzaklıklarının karelerinin toplamı
   - Düşük olması istenir
   - Elbow yönteminde kullanılır

2. SILHOUETTE SCORE (-1 ile 1 arası):
   - +1'e yakın: Nokta kendi kümesine çok yakın, diğer kümelere uzak (mükemmel)
   - 0'a yakın: Nokta iki kümenin sınırında
   - -1'e yakın: Nokta yanlış kümeye atanmış

ÇIKTILAR:
---------
1. clustering_elbow.png      → Elbow ve Silhouette grafikleri
2. clustering_comparison_map.png → İki modelin harita karşılaştırması
3. clustering_score_comparison.png → Silhouette skor bar grafiği

YAZAR: Otomatik oluşturuldu
TARİH: 2024
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
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
silhouette_scores = []
K_range = range(2, 11)  # Silhouette için en az 2 küme gerekli

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    
    # Her K için Silhouette Score hesapla (örneklem ile)
    sample_idx = np.random.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
    sil_score = silhouette_score(X_scaled[sample_idx], labels_temp[sample_idx])
    silhouette_scores.append(sil_score)

# En iyi K'yı Silhouette Score'a göre belirle
best_k_index = np.argmax(silhouette_scores)
optimal_k = list(K_range)[best_k_index]
print(f"\n[OTOMATİK SEÇİM] En yüksek Silhouette Score'a göre optimal K = {optimal_k}")

# Dirsek Grafiğini Çiz
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sol: Elbow
axes[0].plot(list(K_range), inertia, marker='o', linestyle='--', color='blue')
axes[0].axvline(x=optimal_k, color='red', linestyle=':', label=f'Optimal K={optimal_k}')
axes[0].set_xlabel('Küme Sayısı (K)')
axes[0].set_ylabel('Inertia (Hata)')
axes[0].set_title('Elbow (Dirsek) Yöntemi')
axes[0].legend()
axes[0].grid(True)

# Sağ: Silhouette Scores
axes[1].plot(list(K_range), silhouette_scores, marker='s', linestyle='--', color='green')
axes[1].axvline(x=optimal_k, color='red', linestyle=':', label=f'Optimal K={optimal_k}')
axes[1].set_xlabel('Küme Sayısı (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('K Değerine Göre Silhouette Skorları')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
save_path_elbow = os.path.join(current_dir, "clustering_elbow.png")
plt.savefig(save_path_elbow)
print(f"✅ Elbow + Silhouette Grafiği kaydedildi: {save_path_elbow}")


# --- 4. İKİ FARKLI KÜMELEME MODELİNİN KARŞILAŞTIRILMASI ---
print("\n" + "="*60)
print("--- 4. MODEL KARŞILAŞTIRMASI: K-MEANS vs AGGLOMERATIVE ---")
print("="*60)
print("""
[AÇIKLAMA]
İki farklı kümeleme algoritmasını karşılaştırıyoruz:

1. K-MEANS:
   - Centroid (merkez) tabanlı algoritma
   - Her kümenin merkez noktasını hesaplar
   - Hızlı ve büyük veri setlerinde verimli
   - Kümelerin dairesel/küresel olduğunu varsayar

2. AGGLOMERATIVE (Hierarchical) CLUSTERING:
   - Hiyerarşik kümeleme algoritması
   - Alttan yukarı (bottom-up) yaklaşım
   - Her nokta tek başına başlar, en yakınlarla birleşir
   - Farklı şekillerdeki kümeleri bulabilir
""")

# Performans için veriyi örnekleyelim (Agglomerative büyük veride yavaş)
sample_size = min(15000, len(X_scaled))
np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]
df_sample = df_cluster.iloc[sample_indices].copy()

print(f"Karşılaştırma için {sample_size} örnek kullanılıyor...")

# MODEL A: K-Means
print("\n>>> Model A: K-Means eğitiliyor...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_sample)

# MODEL B: Agglomerative Clustering
print(">>> Model B: Agglomerative Clustering eğitiliyor...")
agglo = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
labels_agglo = agglo.fit_predict(X_sample)

# --- 5. DEĞERLENDİRME VE KARŞILAŞTIRMA (Silhouette Score) ---
print("\n--- 5. MODEL BAŞARISI KARŞILAŞTIRMASI ---")

score_kmeans = silhouette_score(X_sample, labels_kmeans)
score_agglo = silhouette_score(X_sample, labels_agglo)

print(f"""
╔══════════════════════════════════════════════════════════╗
║              KÜMELEME MODEL KARŞILAŞTIRMASI              ║
╠══════════════════════════════════════════════════════════╣
║  Model                      │  Silhouette Score          ║
╠─────────────────────────────┼────────────────────────────╣
║  K-Means                    │  {score_kmeans:.4f}                      ║
║  Agglomerative Clustering   │  {score_agglo:.4f}                      ║
╠══════════════════════════════════════════════════════════╣
║  Kazanan: {'K-Means' if score_kmeans >= score_agglo else 'Agglomerative':25}                     ║
╚══════════════════════════════════════════════════════════╝
""")

print("""
[YORUM - SİLHOUETTE SCORE]
- 1'e yakınsa: Kümeler birbirinden çok net ayrılmış (Mükemmel).
- 0'a yakınsa: Kümeler iç içe geçmiş.
- Negatifse: Yanlış kümeleme yapılmış.

Her iki model de aynı K değerini kullandığı için adil bir karşılaştırma yapılmıştır.
""")


# --- 6. SONUÇLARIN YORUMLANMASI ---
print("\n--- 6. KÜMELERİN ANALİZİ (K-Means) ---")
df_sample['Cluster_KMeans'] = labels_kmeans
df_sample['Cluster_Agglo'] = labels_agglo

cluster_summary = df_sample.groupby('Cluster_KMeans').mean()[['price', 'latitude', 'longitude']]
cluster_counts = df_sample['Cluster_KMeans'].value_counts()
cluster_summary['Count'] = cluster_counts

print(cluster_summary)
print("""
[YORUM]
Yukarıdaki tablo bize her kümenin karakterini anlatır.
Örneğin: 
- Bir kümenin fiyatı çok yüksekse (Örn: 200$+), orası Manhattan olabilir.
- Başka bir kümenin fiyatı düşükse (Örn: 60$), orası şehrin dış kesimleri olabilir.
""")


# --- 7. HARİTA ÜZERİNDE KARŞILAŞTIRMALI GÖRSELLEŞTİRME ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Sol: K-Means Haritası
sns.scatterplot(
    data=df_sample, 
    x='longitude', 
    y='latitude', 
    hue='Cluster_KMeans', 
    palette='tab10', 
    alpha=0.6,
    s=15,
    ax=axes[0]
)
axes[0].set_title(f'K-Means Kümeleri (K={optimal_k})\nSilhouette: {score_kmeans:.4f}')
axes[0].set_xlabel('Boylam (Longitude)')
axes[0].set_ylabel('Enlem (Latitude)')
axes[0].legend(title='Küme No')
axes[0].set_aspect('equal')

# Sağ: Agglomerative Haritası
sns.scatterplot(
    data=df_sample, 
    x='longitude', 
    y='latitude', 
    hue='Cluster_Agglo', 
    palette='Set2', 
    alpha=0.6,
    s=15,
    ax=axes[1]
)
axes[1].set_title(f'Agglomerative Kümeleri (K={optimal_k})\nSilhouette: {score_agglo:.4f}')
axes[1].set_xlabel('Boylam (Longitude)')
axes[1].set_ylabel('Enlem (Latitude)')
axes[1].legend(title='Küme No')
axes[1].set_aspect('equal')

plt.tight_layout()
save_path_map = os.path.join(current_dir, "clustering_comparison_map.png")
plt.savefig(save_path_map)
print(f"\n✅ Model Karşılaştırma Haritası kaydedildi: {save_path_map}")


# --- 8. BAR GRAFİĞİ İLE SKOR KARŞILAŞTIRMASI ---
plt.figure(figsize=(8, 5))
models = ['K-Means', 'Agglomerative']
scores = [score_kmeans, score_agglo]
colors = ['#3498db', '#2ecc71']

bars = plt.bar(models, scores, color=colors, edgecolor='black', linewidth=1.2)

# Skorları barların üzerine yaz
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Silhouette Score')
plt.title('Kümeleme Algoritmaları Karşılaştırması')
plt.ylim(0, max(scores) * 1.2)
plt.grid(axis='y', alpha=0.3)

save_path_bar = os.path.join(current_dir, "clustering_score_comparison.png")
plt.savefig(save_path_bar)
print(f"✅ Skor Karşılaştırma Grafiği kaydedildi: {save_path_bar}")
