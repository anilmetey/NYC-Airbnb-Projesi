import pandas as pd
import numpy as np
import os

# Dosya yolunu belirle
_current_dir = os.path.dirname(os.path.abspath(__file__))
# Kaggle'dan indirilen dosya: AB_NYC_2019.csv
df = pd.read_csv(os.path.join(_current_dir, "AB_NYC_2019.csv"))

# --- VERİ TEMİZLEME (En İyi Skor İçin) ---
# Fiyatı 0 olan hatalı verileri ve çok uçuk fiyatları (örneğin 2000$ üstü) atalım (Outlier temizliği)
df = df[(df['price'] > 0) & (df['price'] < 2000)]

# Eksik verileri temizle
df = df.dropna(subset=['latitude', 'longitude', 'price', 'room_type', 'number_of_reviews', 'availability_365'])

# --- MODEL VERİ SETLERİ ---

# 1. Clustering için: Konum ve Fiyat bilgilerini kullanacağız
dataForClustering = df[["longitude", "latitude", "price"]].copy()

# 2. Regression için: Fiyat tahmini yapacağız
# Konum, yorum sayısı ve uygunluk durumuna göre fiyatı tahmin etmeye çalışacağız
dataForRegression = df[["longitude", "latitude", "number_of_reviews", "availability_365", "price"]].copy()

# 3. Classification için: Oda Tipi (Room Type) tahmini yapacağız
# Fiyat, konum ve yorum sayısına bakarak burası "Private room" mu "Entire home" mu tahmin edeceğiz
dataForClassification = df[["longitude", "latitude", "price", "number_of_reviews", "availability_365", "room_type"]].copy()
