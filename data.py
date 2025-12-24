import pandas as pd
import numpy as np
import os

# --- VERİ HAZIRLIĞI ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(_current_dir, "AB_NYC_2019.csv")
print(f"Veri seti okunuyor: {csv_path}")
df = pd.read_csv(csv_path)

# --- FİYAT FİLTRESİ (10-1000$) ---
# Aralığı biraz genişlettik ki model "pahalı" bölgeleri daha iyi ayırt edebilsin.
df = df[(df['price'] >= 10) & (df['price'] <= 1000)]

# Eksikleri at
df = df.dropna()

# --- FEATURE ENGINEERING ---

# 1. Micro-Location (Sokak Blokları) [KRİTİK]
# Koordinatları yuvarlayarak "Blok" oluşturuyoruz. Fiyatı en iyi bu tahmin eder.
df['lat_lon_block'] = df['latitude'].round(3).astype(str) + '_' + df['longitude'].round(3).astype(str)

# 2. Interaction
df['hood_room_combo'] = df['neighbourhood'] + "_" + df['room_type']

# 3. İsim Uzunluğu
df['name_len'] = df['name'].astype(str).apply(len)

# 4. NLP Keywords
keywords = ['luxury', 'view', 'private', 'quiet', 'studio', 'renovated', 'penthouse']
for word in keywords:
    df[f'txt_{word}'] = df['name'].astype(str).str.lower().apply(lambda x: 1 if word in x else 0)

# 5. Mesafe
def calc_dist(lat, lon, lat_center=40.748817, lon_center=-73.985428):
    R = 6371
    phi1, phi2 = np.radians(lat), np.radians(lat_center)
    dphi = np.radians(lat_center - lat)
    dlambda = np.radians(lon_center - lon)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

df['dist_center'] = calc_dist(df['latitude'], df['longitude'])

print("Veri optimize edildi.")

dataForClustering = df[["longitude", "latitude", "price"]].copy()

# Regresyon
cols = [
    "latitude", "longitude", 
    "room_type", "neighbourhood_group", "neighbourhood",
    "hood_room_combo", 
    "lat_lon_block", # [YENİ]
    "minimum_nights", "number_of_reviews", "availability_365",
    "calculated_host_listings_count", "dist_center", "name_len"
]
cols += [f'txt_{w}' for w in keywords]
cols.append("price")

dataForRegression = df[cols].copy()

dataForClassification = df[["longitude", "latitude", "price", "number_of_reviews", "availability_365", "room_type"]].copy()
