import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_save_model(csv_path="insurance.csv", model_output_path="model/linear_regression_model.pkl"):
    print("=== 1. Memuat Dataset ===")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} tidak ditemukan. Pastikan dataset insurance.csv dari Kaggle ada di folder proyek.")
    
    df = pd.read_csv(csv_path)
    print(f"Total data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(df[['age', 'bmi', 'smoker', 'charges']].head())

    print("\n=== 2. Preprocessing Data ===")
    # Seleksi fitur sesuai judul: Usia (age), BMI (bmi), Kebiasaan Merokok (smoker)
    data = df[['age', 'bmi', 'smoker', 'charges']].dropna().copy()
    
    # Encoding variabel kategorikal smoker: 'yes' -> 1, 'no' -> 0
    data['smoker'] = data['smoker'].str.lower().map({'yes': 1, 'no': 0})

    X = data[['age', 'bmi', 'smoker']]
    y = data['charges']

    # Split data latih (80%) dan data uji (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Latih: {len(X_train)} sampel | Data Uji: {len(X_test)} sampel")

    print("\n=== 3. Melatih Model Regresi Linear Berganda ===")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = model.predict(X_test)

    # Evaluasi performa model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n=== 4. Hasil Evaluasi & Persamaan Regresi ===")
    print(f"R-Squared (R2) : {r2:.4f} ({r2*100:.2f}%)")
    print(f"MAE            : ${mae:,.2f}")
    print(f"RMSE           : ${rmse:,.2f}")
    print(f"Intercept (b0) : {model.intercept_:.4f}")
    print(f"Koefisien Usia (b1)    : {model.coef_[0]:.4f}")
    print(f"Koefisien BMI (b2)     : {model.coef_[1]:.4f}")
    print(f"Koefisien Smoker (b3)  : {model.coef_[2]:.4f}")
    print(f"Persamaan: Y = {model.intercept_:.2f} + ({model.coef_[0]:.2f} * Usia) + ({model.coef_[1]:.2f} * BMI) + ({model.coef_[2]:.2f} * Merokok)")

    # Ambang batas klasifikasi tingkat risiko biaya berdasarkan kuantil dataset
    q1 = float(data['charges'].quantile(0.33))
    q2 = float(data['charges'].quantile(0.66))

    metrics_info = {
        'r2_score': round(float(r2), 4),
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'intercept': round(float(model.intercept_), 4),
        'coef_age': round(float(model.coef_[0]), 4),
        'coef_bmi': round(float(model.coef_[1]), 4),
        'coef_smoker': round(float(model.coef_[2]), 4),
        'q1_threshold': round(q1, 2),
        'q2_threshold': round(q2, 2),
        'total_samples': int(len(data))
    }

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f:
        pickle.dump({'model': model, 'metrics': metrics_info}, f)
    
    print(f"\nModel berhasil disimpan ke: {model_output_path}")
    return model, metrics_info

if __name__ == "__main__":
    train_and_save_model()
