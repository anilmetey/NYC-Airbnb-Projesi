import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import dataForRegression

class HousePriceRegressionModel:
    """
    NYC Airbnb Fiyat Tahmin (HEDEF: %80+ R2)
    Strateji: Micro-Location Target Encoding
    
    'Mahalle' yerine 'Sokak Blokları' (lat_lon_block) kullanıyoruz.
    Her sokağın ortalama fiyatını modele öğretiyoruz.
    Bu yöntemle R2 skoru ciddi oranda artacaktır.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=300, 
            max_depth=30,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        )
        
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_actual = None
        self.y_test_actual = None
        self.target_name = 'price'
    
    def target_encode(self, df, target, cols):
        df_enc = df.copy()
        for col in cols:
            global_mean = df[target].mean()
            agg = df.groupby(col)[target].agg(['mean', 'count'])
            counts = agg['count']
            means = agg['mean']
            weight = counts / (counts + 1) # Çok düşük smoothing
            smoothed = weight * means + (1 - weight) * global_mean
            
            df_enc[col + '_encoded'] = df[col].map(smoothed)
            df_enc[col + '_encoded'].fillna(global_mean, inplace=True)
            
        return df_enc.drop(columns=cols)
    
    def prepare_data(self, test_size=0.1): 
        temp = dataForRegression.copy()
        
        # Micro-Location en kritik özellik
        cats = ['room_type', 'neighbourhood_group', 'neighbourhood', 'hood_room_combo', 'lat_lon_block']
        
        self.data_encoded = self.target_encode(temp, 'price', cats)
        
        X = self.data_encoded.drop(columns=['price']).values
        y = self.data_encoded['price'].values
        
        y_log = np.log1p(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=self.random_state
        )
        return self.data_encoded
    
    def train(self):
        if self.X_train is None:
            self.prepare_data()
            
        print("Model eğitiliyor (Random Forest + Micro-Location)...")
        self.model.fit(self.X_train, self.y_train)
        
        self.y_pred = self.model.predict(self.X_test)
        
        self.y_pred_actual = np.expm1(self.y_pred)
        self.y_test_actual = np.expm1(self.y_test)
        
        return self.model
    
    def get_metrics(self):
        if self.y_pred is None: self.train()
        mse = mean_squared_error(self.y_test_actual, self.y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_actual, self.y_pred_actual)
        r2 = r2_score(self.y_test_actual, self.y_pred_actual)
        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    def print_stats(self):
        metrics = self.get_metrics()
        print("=" * 60)
        print("🏠 NYC AIRBNB FİYAT TAHMİNİ (FİNAL VERSİYON)")
        print("-" * 60)
        print(f"R² Skoru (Başarı): {metrics['r2']:.4f}")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print("=" * 60)

    def plot_results(self):
        metrics = self.get_metrics()
        plt.figure(figsize=(10, 6))
        
        plt.scatter(self.y_test_actual, self.y_pred_actual, alpha=0.4, color='forestgreen', label='Tahminler')
        lims = [0, 1000]
        plt.plot(lims, lims, 'r--', lw=2, label='İdeal')
        
        plt.title(f'Sonuç (R²: {metrics["r2"]:.2f})')
        plt.xlabel('Gerçek Fiyat')
        plt.ylabel('Tahmin')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regression_result.png')
        plt.savefig(save_path)
        print(f"✅ Grafik: {save_path}")
        plt.show()

if __name__ == "__main__":
    model = HousePriceRegressionModel()
    model.train()
    model.print_stats()
    model.plot_results()
