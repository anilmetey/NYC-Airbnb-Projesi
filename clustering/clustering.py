import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Gereksiz uyarıları kapat
warnings.filterwarnings('ignore')
# data.py yolunu ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClustering

class IncomeClusteringModel:
    """
    NYC Fiyat Bazlı Kümeleme Modeli (Clustering)
    Bu sınıf, evlerin özelliklerine (Konum ve Fiyat) bakarak onları gruplara ayırır.
    Fiyatlara göre: Lüks, Orta, Ekonomik gibi etiketler atar.
    Algoritma: K-Means
    """
    
    # Kümelerin isimleri (En pahalıdan en ucuza doğru sıralı)
    INCOME_LABELS = ['Çok Lüks', 'Lüks', 'Orta Üstü', 'Ekonomik', 'Çok Ekonomik']
    
    def __init__(self, n_clusters=5):
        """
        n_clusters=5: Veriyi kaç gruba ayıracağımızı belirtir.
        """
        self.n_clusters = n_clusters
        # K-Means modelini oluştur
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.data = None
        self.clusters = None # Hangi evin hangi kümede olduğu bilgisi
        self.sorted_clusters = None # Kümelerin pahalıdan ucuza sıralaması
        self.income_by_cluster = None # Her kümenin ortalama fiyatı
    
    def prepare_data(self):
        """Veriyi hazırlar"""
        self.data = dataForClustering.copy()
        return self.data
    
    def train(self):
        """Kümeleme işlemini yapar"""
        if self.data is None:
            self.prepare_data()
        
        # Sadece fiyata (price) göre kümeleme yapıyoruz
        price_data = self.data[['price']].values
        
        print("K-Means Kümeleme yapılıyor...")
        self.clusters = self.kmeans.fit_predict(price_data)
        
        # Her kümenin merkez noktasını (ortalama fiyatını) bul
        self.cluster_centers = self.kmeans.cluster_centers_
        self.income_by_cluster = {i: self.cluster_centers[i][0] for i in range(self.n_clusters)}
        
        # Kümeleri ortalama fiyata göre sırala (En yüksek fiyattan en düşüğe)
        # Böylece "Cluster 0" her zaman en pahalı veya en ucuz demek olmaz, sıralı hale getiririz.
        self.sorted_clusters = sorted(
            self.income_by_cluster.keys(), 
            key=lambda x: self.income_by_cluster[x], 
            reverse=True
        )
        return self.clusters
    
    def get_cluster_label(self, cluster_id):
        """Bir küme numarasına (0,1,2..) karşılık gelen okunabilir etiketi (Lüks, Ekonomik..) döndürür"""
        rank = self.sorted_clusters.index(cluster_id)
        if rank < len(self.INCOME_LABELS):
            return self.INCOME_LABELS[rank]
        return f"Küme {cluster_id}"
        
    def get_cluster_stats(self):
        """Kümeler hakkında istatistik verir"""
        stats = {}
        for cluster_id in range(self.n_clusters):
            label = self.get_cluster_label(cluster_id)
            stats[label] = {
                'cluster_id': cluster_id,
                'count': int(np.sum(self.clusters == cluster_id)), # O kümede kaç ev var
                'avg_price': self.income_by_cluster[cluster_id]    # Ortalama fiyatı ne
            }
        return stats

    def plot_map(self):
        """
        NYC haritasını çizer ve bölgeleri fiyat düzeyine göre (Lüks, Ekonomik vb.) renklendirir.
        Grafiği hem çizer hem de kaydeder.
        """
        if self.clusters is None: self.train()
        
        # Görselleştirme için veriyi hazırla
        data_plot = self.data.copy()
        data_plot['Cluster'] = self.clusters
        
        # Küme numarasını kelimeye çevir (0 -> 'Çok Lüks')
        data_plot['Label'] = data_plot['Cluster'].apply(self.get_cluster_label)
        
        # Legend (Renk Anahtarı) sırası: Pahalıdan ucuza
        hue_order = self.INCOME_LABELS
        
        plt.figure(figsize=(12, 8))
        
        sns.scatterplot(
            data=data_plot, 
            x='longitude', 
            y='latitude', 
            hue='Label', 
            hue_order=hue_order,
            palette='coolwarm', # Kırmızıdan (Sıcak/Pahalı) maviye (Soğuk/Ucuz)
            alpha=0.6,
            s=15
        )
        
        plt.title('NYC Fiyat Kümeleri Haritası (Bölgelerin Ekonomik Durumu)')
        plt.xlabel('Longitude (Boylam)')
        plt.ylabel('Latitude (Enlem)')
        plt.legend(title='Fiyat Seviyesi')
        plt.grid(True)
        plt.axis('equal')
        
        # --- DOSYAYA KAYDETME ---
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering_map.png')
        plt.savefig(save_path)
        print(f"✅ Kümeleme haritası kaydedildi: {save_path}")
        
        print("Kümeleme haritası çiziliyor...")
        plt.show()

if __name__ == "__main__":
    model = IncomeClusteringModel()
    model.train()
    print("Kümeleme tamamlandı.")
    
    # İstatistikleri yazdır
    stats = model.get_cluster_stats()
    for label, info in stats.items():
        print(f"{label}: {info['count']} ev, Ortalama Fiyat: ${info['avg_price']:.2f}")
        
    model.plot_map() # Haritayı çiz ve kaydet
