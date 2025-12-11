import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Gereksiz uyarıları gizle
warnings.filterwarnings('ignore')
# data.py dosyasını bulabilmek için yol ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClassification

class OceanProximityClassifier: 
    """
    NYC Airbnb Oda Tipi Sınıflandırma Modeli (Classification)
    Bu sınıf, evin fiyatına, konumuna ve özelliklerine bakarak
    oda tipini (Private room, Entire home/apt, Shared room) tahmin eder.
    """
    
    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
        """
        Model parametrelerini ayarlar.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Sınıflandırma için Random Forest kullanıyoruz
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        # Kategorik verileri (Yazı olanları) sayıya çevirmek için (Örn: "Private room" -> 1)
        self.label_encoder = LabelEncoder()
        
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_test_pred = None
        
        # Fiyat, oda tipini belirmede çok önemlidir, bu yüzden özelliklere ekledik.
        self.feature_names = ['longitude', 'latitude', 'price', 'number_of_reviews', 'availability_365']
        self.target_name = 'room_type' # Tahmin edilecek sütun
        self.class_names = None
    
    def prepare_data(self, test_size=0.2):
        """Veriyi hazırlar ve sayısal hale getirir"""
        self.data = dataForClassification.copy()
        X = self.data[self.feature_names].values
        
        # Oda tipleri yazı olduğu için (String), modele sokmadan önce sayıya çevirmeliyiz (Label Encoding)
        y = self.label_encoder.fit_transform(self.data[self.target_name].values)
        self.class_names = self.label_encoder.classes_ # Sınıf isimlerini sakla (Geri dönüştürmek için)
        
        # Veriyi böl (Stratify: Her sınıftan eşit oranda örnek al)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        return self.data
    
    def train(self):
        """Modeli eğitir"""
        if self.X_train is None:
            self.prepare_data()
            
        print("Model eğitiliyor (Random Forest Classifier)...")
        self.model.fit(self.X_train, self.y_train)
        
        # Test verisiyle tahmin yap
        self.y_pred = self.model.predict(self.X_test)
        self.y_test_pred = self.y_pred 
        return self.model

    def get_metrics(self):
        """Doğruluk (Accuracy) skorunu hesaplar"""
        if self.y_pred is None: self.train()
        return {'accuracy': accuracy_score(self.y_test, self.y_pred)}
    
    def get_confusion_matrix(self): return confusion_matrix(self.y_test, self.y_pred)
    def get_classification_report(self): return classification_report(self.y_test, self.y_pred, target_names=self.class_names, output_dict=True)
    def get_feature_importance(self): return {name: imp for name, imp in zip(self.feature_names, self.model.feature_importances_)}
    def get_class_distribution(self): 
        return {'train': {}, 'test': {name: int(np.sum(self.y_test == i)) for i, name in enumerate(self.class_names)}}
        
    def predict(self, X): return self.model.predict(X)

    def plot_map(self):
        """
        NYC Haritası üzerinde tahmin edilen oda tiplerini renkli noktalarla gösterir.
        Grafiği hem çizer hem de dosyaya kaydeder.
        """
        if self.y_pred is None: self.train()
        
        # Görselleştirme için test verisini DataFrame'e çevir
        df_test = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        # Sayısal tahminleri tekrar yazıya çevir (0 -> Entire home)
        df_test['Predicted_Room_Type'] = self.label_encoder.inverse_transform(self.y_pred)
        
        plt.figure(figsize=(12, 8))
        
        # Harita çizimi (Scatter Plot)
        # x: Boylam, y: Enlem, hue: Renk (Oda Tipi)
        sns.scatterplot(
            data=df_test, 
            x='longitude', 
            y='latitude', 
            hue='Predicted_Room_Type', 
            palette='viridis', # Renk paleti
            alpha=0.6,         # Saydamlık
            s=20               # Nokta boyutu
        )
        
        plt.title(f'NYC Oda Tipi Tahmin Haritası (Doğruluk: {self.get_metrics()["accuracy"]:.2%})')
        plt.xlabel('Longitude (Boylam)')
        plt.ylabel('Latitude (Enlem)')
        plt.legend(title='Oda Tipi')
        plt.grid(True)
        plt.axis('equal') # Harita oranlarını koru
        
        # --- DOSYAYA KAYDETME ---
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classification_map.png')
        plt.savefig(save_path)
        print(f"✅ Harita dosyası kaydedildi: {save_path}")
        
        print("Harita penceresi açılıyor...")
        plt.show()

if __name__ == "__main__":
    model = OceanProximityClassifier()
    model.train()
    print(f"Model Doğruluğu: {model.get_metrics()['accuracy']:.2%}")
    model.plot_map() # Haritayı çiz ve kaydet
