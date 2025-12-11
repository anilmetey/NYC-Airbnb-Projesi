import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Sınıflandırma modeli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Yazıları (Private Room) sayıya (1) çevirmek için
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import sys
import os

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dataForClassification


class OceanProximityClassifier:
    """
    NYC Oda Tipi Sınıflandırma Modeli
    Amaç: Verilen bilgilere bakarak evin tipini (Entire home, Private room, Shared room) tahmin etmek.
    """

    def __init__(self, n_estimators=100, max_depth=15, random_state=42):
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

        # Etiket Kodlayıcı: Bilgisayar "Private Room" yazısından anlamaz,
        # bu araç onları 0, 1, 2 gibi sayılara çevirir.
        self.label_encoder = LabelEncoder()

        self.data = None
        self.X_train = None
        self.X_test = None
        # Fiyat ve Konum, oda tipini belirleyen en önemli özelliklerdir.
        self.feature_names = ['longitude', 'latitude', 'price', 'number_of_reviews', 'availability_365']
        self.target_name = 'room_type'  # Tahmin etmeye çalıştığımız şey
        self.class_names = None

    def prepare_data(self, test_size=0.2):
        """Veriyi hazırlar, sayıya çevirir ve böler."""
        self.data = dataForClassification.copy()

        X = self.data[self.feature_names].values

        # Hedef değişkeni (room_type) sayıya çeviriyoruz (Encode)
        y = self.label_encoder.fit_transform(self.data[self.target_name].values)
        self.class_names = self.label_encoder.classes_  # Sınıf isimlerini (örn: Private Room) sakla

        # stratify=y ÖNEMLİ:
        # Eğitim ve Test setlerinde oda tiplerinin oranının aynı olmasını sağlar.
        # Yani test setinde sadece "Shared Room"lar toplanmasın diye dengeli dağıtır.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        return self.data

    def train(self):
        """Modeli eğitir."""
        if self.X_train is None:
            self.prepare_data()

        self.model.fit(self.X_train, self.y_train)

        # Test verisiyle tahmin yap
        self.y_pred = self.model.predict(self.X_test)
        self.y_test_pred = self.y_pred
        return self.model

    def get_metrics(self):
        """
        Başarı oranını (Accuracy) hesaplar.
        Örn: 0.90 çıkarsa, her 100 evin 90'ının tipini doğru bildik demektir.
        """
        if self.y_pred is None: self.train()
        return {'accuracy': accuracy_score(self.y_test, self.y_pred)}

    # Karışıklık Matrisi: Hangi sınıfı hangisiyle karıştırdık?
    def get_confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    # Detaylı Rapor: Her sınıf için başarı puanları
    def get_classification_report(self):
        return classification_report(self.y_test, self.y_pred, target_names=self.class_names, output_dict=True)

    def get_feature_importance(self):
        return {name: imp for name, imp in zip(self.feature_names, self.model.feature_importances_)}

    def get_class_distribution(self):
        return {'train': {}, 'test': {name: int(np.sum(self.y_test == i)) for i, name in enumerate(self.class_names)}}

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    model = OceanProximityClassifier()
    model.train()
    print(f"Accuracy (Doğruluk Oranı): {model.get_metrics()['accuracy']:.2%}")