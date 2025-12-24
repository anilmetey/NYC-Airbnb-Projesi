import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regression import HousePriceRegressionModel

def plot_regression_results():
    print("Model eğitiliyor ve tahminler yapılıyor...")
    model = HousePriceRegressionModel()
    model.train()
    
    # ÖNEMLİ: y_test ve y_pred log dönüşümlü, gerçek değerler için _actual kullan!
    y_test = model.y_test_actual  # Gerçek fiyatlar ($)
    y_pred = model.y_pred_actual  # Tahmin edilen fiyatlar ($)
    
    # Metrikleri al
    metrics = model.get_metrics()
    r2 = metrics['r2']
    rmse = metrics['rmse']
    mae = metrics['mae']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color='forestgreen', s=20)
    
    # Mükemmel tahmin doğrusu (y=x)
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='İdeal Tahmin (y=x)')
    
    # Başlığa metrikleri ekle
    plt.title(f'NYC Airbnb Fiyat Tahmini: Gerçek vs Tahmin\nR² = {r2:.4f} | MAE = ${mae:.2f} | RMSE = ${rmse:.2f}')
    plt.xlabel('Gerçek Fiyatlar ($)')
    plt.ylabel('Tahmin Edilen Fiyatlar ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Grafiği dosyaya kaydet
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regression_result.png')
    plt.savefig(save_path, dpi=150)
    print(f"✅ Grafik dosyaya kaydedildi: {save_path}")
    
    # Sonuçları yazdır
    print("\n" + "="*50)
    print("📊 REGRESYON SONUÇLARI")
    print("="*50)
    print(f"R² Score: {r2:.4f} ({r2*100:.1f}% açıklayıcılık)")
    print(f"MAE: ${mae:.2f} (Ortalama hata)")
    print(f"RMSE: ${rmse:.2f} (Karesel hata)")
    print("="*50)
    
    print("\nGrafik çiziliyor...")
    plt.show()

if __name__ == "__main__":
    plot_regression_results()

