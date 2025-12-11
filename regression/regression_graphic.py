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
    
    y_test = model.y_test
    y_pred = model.y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    
    # Mükemmel tahmin doğrusu (y=x)
    max_val = max(max(y_test), max(y_pred))
    min_val = min(min(y_test), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('NYC Airbnb Fiyat Tahmini: Gerçek vs Tahmin')
    plt.xlabel('Gerçek Fiyatlar ($)')
    plt.ylabel('Tahmin Edilen Fiyatlar ($)')
    plt.grid(True)
    plt.tight_layout()
    
    # Grafiği dosyaya da kaydedelim
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regression_result.png')
    plt.savefig(save_path)
    print(f"Grafik dosyaya kaydedildi: {save_path}")
    
    print("Grafik çiziliyor...")
    plt.show()

if __name__ == "__main__":
    plot_regression_results()
