import pandas as pd
import numpy as np
import os

# --- VERİ HAZIRLIĞI VE TEMİZLİĞİ ---
# Bu dosya, tüm makine öğrenmesi modelleri için ortak olan veri yükleme ve temizleme işlemlerini yapar.

# Dosya yolunu belirle (Bu scriptin olduğu klasörü bulur)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# Kaggle'dan indirilen CSV dosyasını okuyoruz
# Dosyanın adı: AB_NYC_2019.csv
csv_path = os.path.join(_current_dir, "AB_NYC_2019.csv")
print(f"Veri seti okunuyor: {csv_path}")
df = pd.read_csv(csv_path)

# --- VERİ TEMİZLEME (En İyi Skor İçin) ---
# Makine öğrenmesi modellerinin başarısını artırmak için hatalı veya eksik verileri temizliyoruz.

# 1. Adım: Fiyat Filtreleme (Outlier Detection)
# Fiyatı 0 olanlar hatalıdır.
# Fiyatı 2000$'dan fazla olanlar "uç değer" (outlier) olarak kabul edilip atılıyor.
# Bu sayede model genel piyasayı daha iyi öğrenir.
df = df[(df['price'] > 0) & (df['price'] < 2000)]

# 2. Adım: Eksik Verileri Temizleme (Missing Values)
# Kritik sütunlarda (Enlem, Boylam, Fiyat, vb.) boş değer olan satırları siliyoruz.
df = df.dropna(subset=['latitude', 'longitude', 'price', 'room_type', 'number_of_reviews', 'availability_365'])

print("Veri temizliği tamamlandı.")

# --- MODEL VERİ SETLERİ ---
# Her modelin ihtiyacı olan veriler farklıdır. Bu yüzden ayrı kopyalar oluşturuyoruz.

# 1. KÜMELEME (CLUSTERING) İÇİN VERİ
# Amaç: Benzer evleri gruplamak.
# Kullanılan Özellikler: Konum (Enlem/Boylam) ve Fiyat.
dataForClustering = df[["longitude", "latitude", "price"]].copy()

# 2. REGRESYON (REGRESSION) İÇİN VERİ
# Amaç: Evin fiyatını sayısal olarak tahmin etmek (Örn: 150$, 200$).
# Hedef Değişken (Target): price
# Girdi Özellikleri (Features): Konum, yorum sayısı, uygunluk durumu.
dataForRegression = df[["longitude", "latitude", "number_of_reviews", "availability_365", "price"]].copy()

# 3. SINIFLANDIRMA (CLASSIFICATION) İÇİN VERİ
# Amaç: Evin "Oda Tipi"ni tahmin etmek (Private room vs Entire home/apt).
# Hedef Değişken (Target): room_type
# Girdi Özellikleri: Konum, fiyat, yorum sayısı, uygunluk.
dataForClassification = df[["longitude", "latitude", "price", "number_of_reviews", "availability_365", "room_type"]].copy()
