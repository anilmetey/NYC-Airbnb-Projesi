import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor  # Regresyon (sayısal tahmin) için kullanılan model
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak bölmek için
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Başarı ölçüm metrikleri
import warnings
import sys
import os

# Gereksiz uyarı mesajlarını kapatarak konsolu temiz tutar
warnings.filterwarnings('ignore')

# Projenin ana dizinini yola ekler, böylece 'data.py' dosyasını bulabiliriz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data.py dosyasından hazırlanmış veriyi çekiyoruz
from data import dataForRegression


class HousePriceRegressionModel:
    """
    NYC Airbnb Fiyat Tahmin Modeli (Regresyon)
    Bu sınıf, New York'taki evlerin özelliklerine bakarak gecelik fiyatını tahmin eder.
    """

    def __init__(self, n_estimators=200, max_depth=20, random_state=42):
        """
        Modelin başlangıç ayarlarını yapar.

        Parametreler:
        - n_estimators=200: Ormanda kaç tane karar ağacı olacağı. Sayı arttıkça doğruluk artar ama yavaşlar.
        - max_depth=20: Ağaçların ne kadar derine ineceği. Çok derin olursa ezberler (overfitting), az olursa öğrenemez.
        - n_jobs=-1: Bilgisayarın tüm işlemci çekirdeklerini kullanarak daha hızlı çalışmasını sağlar.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Random Forest Regresyon modelini tanımlıyoruz
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

        # Değişkenleri başlatıyoruz (henüz boşlar)
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

        # Modelin kullanacağı özellikler (Girdi)
        self.feature_names = ['longitude', 'latitude', 'number_of_reviews', 'availability_365']
        # Tahmin etmeye çalıştığımız değer (Çıktı - Hedef)
        self.target_name = 'price'

    def prepare_data(self, test_size=0.2):
        """
        Veriyi hazırlar ve Eğitim/Test olarak ikiye böler.
        test_size=0.2 -> Verinin %20'si test için, %80'i eğitim için ayrılır.
        """
        # Orijinal veriyi bozmamak için kopyasını alıyoruz
        self.data = dataForRegression.copy()

        # X: Girdiler (Enlem, Boylam, Yorum Sayısı vb.)
        X = self.data[self.feature_names].values
        # y: Hedef (Fiyat)
        y = self.data[self.target_name].values

        # Veriyi karıştırıp bölüyoruz (Shuffle & Split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        return self.data

    def train(self):
        """
        Modeli eğitim verisiyle (X_train, y_train) eğitir.
        Bilgisayar burada veriler arasındaki ilişkiyi öğrenir.
        """
        # Eğer eğitim verisi henüz hazırlanmadıysa, önce onu hazırla
        if self.X_train is None:
            self.prepare_data()

        # Eğitimi başlat (Fit)
        self.model.fit(self.X_train, self.y_train)

        # Test verisi üzerinde deneme tahmini yap (Sonuçları ölçmek için)
        self.y_pred = self.model.predict(self.X_test)

        return self.model

    def get_metrics(self):
        """
        Modelin ne kadar başarılı olduğunu sayısal olarak ölçer.
        """
        # Eğer tahmin yapılmadıysa önce eğit
        if self.y_pred is None:
            self.train()

        # MSE (Mean Squared Error): Hataların karesinin ortalaması
        mse = mean_squared_error(self.y_test, self.y_pred)

        # RMSE (Root Mean Squared Error): Ortalama hata payı (Dolar cinsinden)
        # Örn: RMSE 50 ise, tahminlerimiz ortalama 50$ aşağı veya yukarı sapıyor demektir.
        rmse = np.sqrt(mse)

        # MAE (Mean Absolute Error): Mutlak ortalama hata
        mae = mean_absolute_error(self.y_test, self.y_pred)

        # R2 Score: Modelin veriyi açıklama oranı (1.0 en iyi, 0.0 en kötü)
        # Örn: 0.85 gelirse, fiyat değişimlerinin %85'ini açıklayabiliyoruz demektir.
        r2 = r2_score(self.y_test, self.y_pred)

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    def get_feature_importance(self):
        """
        Fiyatı belirleyen en önemli faktör hangisi?
        (Örn: Konum mu daha önemli yoksa yorum sayısı mı?)
        """
        importance = self.model.feature_importances_
        # Özellik isimleri ile önem derecelerini eşleştirip sözlük olarak döndürür
        return {name: imp for name, imp in zip(self.feature_names, importance)}

    def print_stats(self):
        """
        Sonuçları ekrana düzgün bir formatta yazdırır.
        """
        metrics = self.get_metrics()
        print("=" * 60)
        print("🏠 NYC AIRBNB FİYAT TAHMİNİ (REGRESYON)")
        # R2 skorunu virgülden sonra 4 hane göster
        print(f"R² Skoru (Başarı Oranı): {metrics['r2']:.4f}")
        # Hata payını dolar cinsinden göster
        print(f"RMSE (Ortalama Hata): ${metrics['rmse']:.2f}")
        print("=" * 60)


# Bu dosya doğrudan çalıştırıldığında burası devreye girer
if __name__ == "__main__":
    # Sınıftan bir örnek (nesne) oluştur
    model = HousePriceRegressionModel()

    # Eğitimi başlat
    model.train()

    # Sonuçları ekrana yaz
    model.print_stats()