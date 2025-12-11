import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Gereksiz uyarıları gizle (Temiz çıktı için)
warnings.filterwarnings('ignore')

# data.py dosyasını bulabilmek için bir üst klasörü yol olarak ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ortak veri dosyamızdan (data.py) regresyon verisini çek
from data import dataForRegression

class HousePriceRegressionModel:
    """
    NYC Airbnb Fiyat Tahmin Modeli (Regression)
    Bu sınıf, evin özelliklerine bakarak gecelik fiyatını ($) tahmin eder.
    Kullanılan Algoritma: Random Forest Regressor
    """
    
    def __init__(self, n_estimators=200, max_depth=20, random_state=42):
        """
        Modeli başlatır ve parametreleri ayarlar.
        n_estimators: Ormandaki ağaç sayısı (200 ağaç daha istikrarlı sonuç verir).
        max_depth: Ağaçların derinliği (Aşırı öğrenmeyi engellemek için sınırlandırdık).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Random Forest modelini oluştur
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1 # Tüm işlemci çekirdeklerini kullan (Hız için)
        )
        
        # Değişkenleri hazırla
        self.data = None
        self.X_train = None # Eğitim verisi (Girdiler)
        self.X_test = None  # Test verisi (Girdiler)
        self.y_train = None # Eğitim verisi (Hedef - Fiyat)
        self.y_test = None  # Test verisi (Hedef - Fiyat)
        self.y_pred = None  # Modelin tahmin ettiği fiyatlar
        
        # Modelin kullanacağı özellikler (Features)
        self.feature_names = ['longitude', 'latitude', 'number_of_reviews', 'availability_365']
        # Tahmin edilecek hedef değişken (Target)
        self.target_name = 'price'
    
    def prepare_data(self, test_size=0.2):
        """
        Veriyi Eğitim (Train) ve Test setlerine ayırır.
        test_size=0.2 -> Verinin %20'si test için ayrılır.
        """
        self.data = dataForRegression.copy()
        
        # Girdileri (X) ve Hedefi (y) ayır
        X = self.data[self.feature_names].values
        y = self.data[self.target_name].values
        
        # Veriyi karıştır ve böl
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        return self.data
    
    def train(self):
        """
        Modeli eğitim verisiyle eğitir (Fitting).
        """
        # Veri hazırlanmamışsa önce onu hazırla
        if self.X_train is None:
            self.prepare_data()
            
        print("Model eğitiliyor (Random Forest)...")
        self.model.fit(self.X_train, self.y_train)
        
        # Test verisi üzerinden tahmin yap
        self.y_pred = self.model.predict(self.X_test)
        return self.model
    
    def get_metrics(self):
        """
        Modelin başarısını ölçen metrikleri hesaplar.
        R2 Score: 1'e ne kadar yakınsa o kadar iyi.
        RMSE: Hata payı (Dolar cinsinden ortalama sapma).
        """
        if self.y_pred is None:
            self.train()
            
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse) # Kök Ortalama Kare Hatası
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred) # Açıklayıcılık katsayısı
        
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    def get_feature_importance(self):
        """Hangi özelliğin fiyat üzerinde ne kadar etkisi olduğunu gösterir"""
        importance = self.model.feature_importances_
        return {name: imp for name, imp in zip(self.feature_names, importance)}
    
    def print_stats(self):
        """Sonuçları ekrana yazdırır"""
        metrics = self.get_metrics()
        print("=" * 60)
        print("🏠 NYC AIRBNB FİYAT TAHMİNİ (REGRESYON)")
        print("-" * 60)
        print(f"R² Skoru (Başarı): {metrics['r2']:.4f}")
        print(f"RMSE (Ortalama Hata): ${metrics['rmse']:.2f}")
        print("=" * 60)

    def plot_results(self):
        """
        Gerçek Fiyatlar ile Tahmin Edilen Fiyatları grafik üzerinde karşılaştırır.
        Grafiği hem ekranda gösterir hem de dosyaya kaydeder.
        """
        metrics = self.get_metrics() # Modeli çalıştır
        
        plt.figure(figsize=(10, 6))
        # Mavi noktalar: Her bir evin Gerçek vs Tahmin fiyatı
        plt.scatter(self.y_test, self.y_pred, alpha=0.5, color='blue', label='Tahminler')
        
        # Kırmızı kesikli çizgi: Mükemmel tahmin doğrusu (Hedef)
        # Eğer bir nokta bu çizginin üzerindeyse, tahmin tam isabet demektir.
        max_val = max(max(self.y_test), max(self.y_pred))
        min_val = min(min(self.y_test), min(self.y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Mükemmel Doğruluk')
        
        plt.title(f'NYC Airbnb Fiyat Tahmini (Başarı R²: {metrics["r2"]:.2f})')
        plt.xlabel('Gerçek Fiyatlar ($)')
        plt.ylabel('Modelin Tahmin Ettiği Fiyatlar ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # --- DOSYAYA KAYDETME ---
        # Grafiği "regression_result.png" adıyla kaydeder.
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regression_result.png')
        plt.savefig(save_path)
        print(f"✅ Grafik dosyası kaydedildi: {save_path}")
        
        print("Grafik penceresi açılıyor...")
        plt.show()

if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırıldığında burası devreye girer
    model = HousePriceRegressionModel()
    model.train()         # Modeli eğit
    model.print_stats()   # İstatistikleri yazdır
    model.plot_results()  # Grafiği çiz ve kaydet
