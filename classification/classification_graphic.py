import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classification import OceanProximityClassifier # Sınıf adı classification.py'de böyle kalmıştı

def plot_classification_map():
    print("Model eğitiliyor...")
    model = OceanProximityClassifier()
    model.train()
    
    # Test verisi üzerinden görselleştirme yapalım
    # Gerçek veri setindeki koordinatları ve tahmin edilen sınıfları alalım
    df_test = pd.DataFrame(model.X_test, columns=model.feature_names)
    df_test['Predicted_Room_Type'] = model.label_encoder.inverse_transform(model.y_pred)
    
    plt.figure(figsize=(12, 8))
    
    # NYC Haritası (Enlem/Boylam scatter plot)
    sns.scatterplot(
        data=df_test, 
        x='longitude', 
        y='latitude', 
        hue='Predicted_Room_Type', 
        palette='viridis', 
        alpha=0.6,
        s=20
    )
    
    plt.title('NYC Oda Tipi Tahmin Haritası')
    plt.xlabel('Longitude (Boylam)')
    plt.ylabel('Latitude (Enlem)')
    plt.legend(title='Oda Tipi')
    plt.grid(True)
    
    # NYC Haritasına benzesin diye oranları koru
    plt.axis('equal')
    
    # Grafiği dosyaya da kaydedelim
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classification_map.png')
    plt.savefig(save_path)
    print(f"Harita dosyaya kaydedildi: {save_path}")
    
    print("Harita çiziliyor...")
    plt.show()

if __name__ == "__main__":
    plot_classification_map()
