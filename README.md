# NYC Airbnb Veri Analizi ve Modelleme Projesi

Bu proje, New York City (NYC) Airbnb verilerini (2019) kullanarak çeşitli makine öğrenmesi tekniklerini (Sınıflandırma, Kümeleme ve Regresyon) uygulamayı ve sonuçları karşılaştırmayı hedefler.

## 📂 Proje Yapısı

Proje, ana veri işleme modülü ve üç ana analiz klasöründen oluşur:

```
AirBnb NYC/
├── data.py               # Veri yükleme, temizleme ve özellik mühendisliği (Feature Engineering)
├── AB_NYC_2019.csv       # Veri seti (Raw Data)
├── classification/       # Sınıflandırma Modelleri (Oda Tipi Tahmini)
├── clustering/           # Kümeleme Modelleri (Lokasyon & Fiyat Analizi)
└── regression/           # Regresyon Modelleri (Fiyat Tahmini)
```

## 🚀 Kurulum ve Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız vardır:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🛠 Modüller ve Kullanım

### 1. Veri Hazırlığı (`data.py`)
Tüm modeller için ortak olan veri ön işleme adımlarını içerir:
- Eksik verilerin temizlenmesi.
- Fiyat filtresi (10$ - 1000$ arası).
- Feature Engineering:
  - `lat_lon_block`: Koordinat bloklama.
  - `dist_center`: Şehir merkezine uzaklık.
  - `hood_room_combo`: Semt ve oda tipi etkileşimi.
  - NLP tabanlı anahtar kelime analizi ('luxury', 'view' vb.).

### 2. Sınıflandırma (`classification/`)
Amaç: Evin özelliklerine bakarak oda tipini (`Private room`, `Entire home/apt`, `Shared room`) tahmin etmek.
- **Modeller**: Random Forest vs KNN.
- **Dosya**: `classification/compare_models.py`
- **Çıktı**: Başarı metrikleri (Accuracy, F1-Score) ve Confusion Matrix grafiği.

### 3. Kümeleme (`clustering/`)
Amaç: Konum ve fiyat bilgilerini kullanarak benzer evleri gruplamak (Örn: "Pahalı Merkez Evleri", "Ucuz Kenar Mahalleler").
- **Modeller**: K-Means.
- **Dosya**: `clustering/compare_models.py`
- **Çıktı**: Elbow yöntemi grafiği ve harita üzerinde kümelerin görselleştirmesi.

### 4. Regresyon (`regression/`)
Amaç: Evin özelliklerini kullanarak gecelik fiyatını tahmin etmek.
- **Modeller**: Random Forest, Gradient Boosting, HistGradientBoosting.
- **Dosya**: `regression/compare_models.py`
- **Çıktı**: R² skoru, RMSE hatası ve tahmin/gerçek değer karşılaştırma grafikleri.

## ▶️ Nasıl Çalıştırılır?

Her bir modülü kendi klasörü içindeki `compare_models.py` dosyasını çalıştırarak test edebilirsiniz. Örnek olarak terminalden şu komutları kullanabilirsiniz:

```bash
# Sınıflandırma analizi için:
python classification/compare_models.py

# Kümeleme analizi için:
python clustering/compare_models.py

# Regresyon analizi için:
python regression/compare_models.py
```

Grafikler ve sonuçlar ilgili klasörler içerisine `.png` dosyası olarak kaydedilecektir.
