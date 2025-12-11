import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  # Kümeleme (Gruplama) algoritması
import warnings
import sys
import os

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClustering


class IncomeClusteringModel:
    """
    NYC Fiyat Bazlı Kümeleme Modeli (K-Means)
    Bu sınıf, evleri fiyatlarına göre otomatik olarak 5 farklı seviyeye (Lüks, Ekonomik vb.) ayırır.
    """

    # Kümeleri isimlendiriyoruz (En pahalıdan en ucuza)
    INCOME_LABELS = ['Çok Lüks', 'Lüks', 'Orta Üstü', 'Ekonomik', 'Çok Ekonomik']

    def __init__(self, n_clusters=5):
        """
        Model ayarları.
        n_clusters=5: Veriyi kaç gruba ayıracağımız. (Hocana: '5 gelir seviyesi belirledik' diyebilirsin)
        """
        self.n_clusters = n_clusters

        # K-Means algoritmasını başlatıyoruz
        # random_state=42: Her çalıştırdığımızda aynı sonucu versin diye sabitliyoruz.
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        self.data = None
        self.clusters = None
        self.sorted_clusters = None
        self.income_by_cluster = None

    def prepare_data(self):
        """Veriyi data.py'dan alır ve kopyalar"""
        self.data = dataForClustering.copy()
        return self.data

    def train(self):
        """
        Modelin eğitimi burada yapılır.
        Burada 'eğitim' demek, bilgisayarın fiyatları analiz edip birbirine yakın olanları gruplamasıdır.
        """
        if self.data is None:
            self.prepare_data()

        # Sadece 'price' (Fiyat) sütununu alıyoruz çünkü gruplamayı buna göre yapacağız
        price_data = self.data[['price']].values

        # fit_predict: Hem öğren (fit) hem de grupları belirle (predict)
        self.clusters = self.kmeans.fit_predict(price_data)

        # Her grubun merkez noktasını (ortalama fiyatını) buluyoruz
        self.cluster_centers = self.kmeans.cluster_centers_
        self.income_by_cluster = {i: self.cluster_centers[i][0] for i in range(self.n_clusters)}

        # ÖNEMLİ: K-Means gruplara rastgele numara verir (0, 1, 2...).
        # Biz bunları "En Pahalıdan En Ucuza" doğru sıralıyoruz ki grafiklerde karışıklık olmasın.
        self.sorted_clusters = sorted(
            self.income_by_cluster.keys(),
            key=lambda x: self.income_by_cluster[x],
            reverse=True  # Büyükten küçüğe sırala
        )
        return self.clusters

    def get_cluster_label(self, cluster_id):
        """
        Matematiksel grup numarasını (Örn: 2) sözel etikete (Örn: 'Lüks') çevirir.
        """
        rank = self.sorted_clusters.index(cluster_id)
        if rank < len(self.INCOME_LABELS):
            return self.INCOME_LABELS[rank]
        return f"Küme {cluster_id}"

    def get_cluster_stats(self):
        """
        Hangi grupta kaç tane ev var ve ortalama fiyatları ne?
        İstatistikleri hesaplar.
        """
        stats = {}
        for cluster_id in range(self.n_clusters):
            label = self.get_cluster_label(cluster_id)
            stats[label] = {
                'cluster_id': cluster_id,
                'count': int(np.sum(self.clusters == cluster_id)),  # O gruptaki ev sayısı
                'avg_income': self.income_by_cluster[cluster_id]  # O grubun ortalama fiyatı
            }
        return stats


if __name__ == "__main__":
    model = IncomeClusteringModel()
    model.train()
    print("Kümeleme tamamlandı.")