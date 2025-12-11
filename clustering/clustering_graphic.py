import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clustering import IncomeClusteringModel

def plot_clustering_map():
    print("Kümeleme yapılıyor...")
    model = IncomeClusteringModel()
    model.train()
    
    data = model.data.copy()
    data['Cluster'] = model.clusters
    
    # Küme numaralarını etiketlere çevirelim (Örn: 0 -> Lüks)
    data['Label'] = data['Cluster'].apply(model.get_cluster_label)
    
    # Sıralı etiketler listesi (Pahalıdan Ucuza doğru legend için)
    hue_order = model.INCOME_LABELS
    
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=data, 
        x='longitude', 
        y='latitude', 
        hue='Label', 
        hue_order=hue_order,
        palette='coolwarm', 
        alpha=0.6,
        s=15
    )
    
    plt.title('NYC Fiyat Kümeleri Haritası')
    plt.xlabel('Longitude (Boylam)')
    plt.ylabel('Latitude (Enlem)')
    plt.legend(title='Fiyat Seviyesi')
    plt.grid(True)
    plt.axis('equal')
    
    # Grafiği dosyaya da kaydedelim
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clustering_map.png')
    plt.savefig(save_path)
    print(f"Kümeleme haritası dosyaya kaydedildi: {save_path}")
    
    print("Kümeleme haritası çiziliyor...")
    plt.show()

if __name__ == "__main__":
    plot_clustering_map()
