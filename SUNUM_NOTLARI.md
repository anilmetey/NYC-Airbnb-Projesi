# Proje Sunum Notları ve Teknik Açıklamalar

Bu dosya, proje sunumu sırasında gelebilecek teknik sorulara (Ders Notları kapsamında) vereceğiniz cevapları içerir. Kod içinde yapılan işlemlerin teorik ve pratik nedenleri aşağıda açıklanmıştır.

## 1. Veri Hazırlığı ve Kontroller

### **Soru: Eksik veri kontrolü ve "Garip String" (Noise) kontrolü yapıldı mı?**
**Cevap:** Evet.
- **Standart Eksikler:** `df.isnull().sum()` ile klasik `NaN` (boş) değerler kontrol edildi ve `dropna()` ile temizlendi.
- **Garip Stringler:** Veri setinde bazen boş yerlere "?", "N/A", "-" gibi karakterler girilebilir. Kodumuzda `compare_models.py` içerisinde bu özel karakterleri arayan ve raporlayan bir döngü bulunmaktadır (`strange_strings` listesi).

### **Soru: Veri Sızıntısını (Data Leakage) önlemek için ne yapıldı?**
**Cevap:** Veri sızıntısı, test verisindeki bilgilerin eğitim aşamasına karışmasıdır. Bunu önlemek için **Standardizasyon (Scaling) işlemi Train-Test ayrımından SONRA yapıldı.**
- **Doğru Yöntem (Uygulanan):** `scaler.fit(X_train)` (Sadece eğitim verisini öğren) -> `scaler.transform(X_test)` (Öğrendiğin kuralları test verisine uygula).
- **Yanlış Yöntem:** Önce tüm veriyi scale edip sonra ayırmak (Bu durumda test verisinin ortalaması eğitim setine sızmış olurdu).

### **Soru: Verileri nasıl encode ettik?**
**Cevap:**
- **Label Encoding:** Sıralı veya çok sınıflı veriler için (Örn: Hedef değişken `room_type`).
- **One-Hot Encoding:** Kategorik ve sırasız veriler için (Örn: `neighbourhood_group` - Manhattan, Brooklyn...). Pandas `get_dummies` kullanılarak her bölge ayrı bir sütun (0/1) yapıldı. Bu, modelin bölgeler arasında matematiksel bir büyüklük ilişkisi kurmasını (Manhattan > Brooklyn gibi hatalı bir algıyı) engeller.

---

## 2. Makine Öğrenmesi Modelleri ve Çalışma Mantıkları

### **Soru: Hangi modeller kullanıldı ve neden?**
Projede **Karşılaştırmalı Analiz** yöntemi izlendi. Her problem için birbirine zıt karakterde iki model seçildi:

1.  **Sınıflandırma (Oda Tipi Tahmini):**
    -   **K-Nearest Neighbors (KNN):** "Bana arkadaşını söyle, sana kim olduğunu söyleyeyim" mantığı. Benzer özellikteki evlerin oda tipleri de benzerdir varsayımına dayanır.
    -   **Random Forest:** Karar ağaçları topluluğu. "Evin fiyatı yüksek mi? Evet. Konumu Manhattan mı? Evet. O zaman bu Entire Home'dur" gibi kurallar zinciri oluşturur.

2.  **Regresyon (Fiyat Tahmini):**
    -   **Linear Regression:** Fiyat ile özellikler arasında doğrusal bir çizgi çeker. Yorumlanması kolaydır.
    -   **Random Forest Regressor:** Doğrusal olmayan, karmaşık ilişkileri yakalar. Genelde daha yüksek başarı verir.

3.  **Kümeleme (Clustering):**
    -   **K-Means:** Verileri, birbirine olan uzaklıklarına (Öklid Mesafesi) göre K adet gruba ayırır.

### **Soru: Verilerin birbirine yakınlığını nasıl ölçtük?**
**Cevap:** Öklid Mesafesi (Euclidean Distance).
- Kuş bakışı uzaklık formülüdür. KNN ve K-Means algoritmaları, iki ev ne kadar "benzer" sorusuna bu formülle cevap verir.

---

## 3. Standartlaştırma ve Hiperparametreler

### **Soru: Neden Standartlaştırma (Scaling) yapıldı/yapılmadı?**
**Cevap:**
- **KNN ve K-Means (Mesafe Tabanlı Modeller):** **ZORUNLU**. Çünkü "Fiyat (0-1000)" ile "Enlem (40-41)" aynı teraziye konulamaz. Scale edilmezse model sadece fiyata bakar, konumu görmezden gelir. `StandardScaler` ile her şeyi eşit birime (z-skor) çevirdik.
- **Random Forest (Ağaç Tabanlı Modeller):** **GEREKSİZ**. Ağaçlar "Büyüktür/Küçüktür" mantığıyla çalıştığı için sayıların büyüklüğünden etkilenmez. Ancak karşılaştırma adil olsun diye kodumuzda hepsine uyguladık.

### **Soru: Cross Validation (CV) ve K değeri nedir?**
**Cevap:** Kodda `cv=5` kullandık.
- **K-Fold Cross Validation:** Veriyi 5 eşit parçaya böler. 4 parça ile eğitir, 1 parça ile test eder. Bunu 5 kez tekrarlar ve ortalamasını alır.
- **Amacı:** Modelin başarısının "şansa bağlı" olmadığından emin olmaktır. Sadece tek bir test setiyle (Train/Test Split) yetinmeyip, modelin her senaryoda sağlam çalıştığını kanıtlar.

### **Soru: Hiperparametre Optimizasyonu neden yapıldı?**
**Cevap:**
- Modelin "Fabrika Ayarları" her zaman en iyi sonucu vermez.
- `GridSearchCV` kullanarak olabilecek tüm kombinasyonları denedik:
  - **KNN İçin:** Komşu sayısı (`n_neighbors`=3 mü, 5 mi, 10 mu?).
  - **Random Forest İçin:** Ağaç sayısı (`n_estimators`) ve Ağaç derinliği (`max_depth`).
- **Sonuç:** En iyi parametreleri (Best Params) bulup modeli güncelledik.

---

## 4. Metrikler ve Sonuçların Yorumlanması

### **Soru: Hangi metrik daha önemli? (Precision vs Recall)**
**Cevap:** Problemine göre değişir:
- **Precision (Kesinlik):** Eğer "Kullanıcıya yanlış oda önermeyelim, önerdiğimiz tam isabet olsun" diyorsak Precision önemlidir.
- **Recall (Duyarlılık):** Eğer "Hiçbir fırsatı kaçırmayalım, aradaki bazı yanlışlara razıyız" diyorsak Recall önemlidir.
- **F1-Score:** Genelde ikisinin dengesi olduğu için en güvenilir metriktir.

### **Soru: Regresyon başarısını (R2) nasıl yorumladık?**
**Cevap:**
- **R2 Score:** Modelimiz fiyat değişimlerinin % kaçını açıklayabiliyor? (Örn: %70 ise, fiyatın neden değiştiğini büyük oranda biliyoruz demektir).
- **MAE (Mean Absolute Error):** Tahminimiz gerçek fiyattan ortalama kaç dolar sapıyor? (Örn: 40$ hata payımız var).

### **Soru: Özellik Seçimi (Feature Selection) yapıldı mı?**
**Cevap:** Evet, Regresyon bölümünde yapıldı.
- `RandomForest` modelinin `feature_importances_` özelliği kullanıldı.
- **Sonuç:** Fiyatı en çok etkileyen faktörlerin **Oda Tipi (Room Type)** ve **Konum (Longitude/Latitude)** olduğu görüldü.
