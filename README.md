# Prediksi Biaya Asuransi Kesehatan Berdasarkan Usia, BMI, dan Kebiasaan Merokok Menggunakan Regresi Linear

Aplikasi berbasis web (Python Flask + Scikit-Learn) untuk memprediksi estimasi biaya asuransi kesehatan (`charges`) serta mengklasifikasikan tingkat risiko biaya (**Rendah**, **Sedang**, **Tinggi**) menggunakan algoritma **Regresi Linear Berganda (Multiple Linear Regression)** berdasarkan dataset Kaggle `mirichoi0218/insurance`.

## Struktur Proyek
```
prediksi_asuransi_kesehatan/
├── app.py                  # Server web Flask & endpoint prediksi/klasifikasi
├── train_model.py          # Script pelatihan, evaluasi, dan penyimpanan model Regresi Linear
├── insurance.csv           # Dataset asuransi kesehatan (Age, BMI, Smoker, Charges)
├── requirements.txt        # Daftar library Python yang dibutuhkan
├── README.md               # Dokumentasi panduan instalasi & penggunaan
├── model/
│   └── linear_regression_model.pkl  # Model Regresi Linear tersimpan
├── static/
│   └── css/
│       └── style.css       # Styling antarmuka aplikasi web
└── templates/
    └── index.html          # Halaman utama sistem prediksi & klasifikasi
```

## Cara Menjalankan Aplikasi

1. **Ekstrak file ZIP** ini ke folder komputer Anda.
2. *(Opsional)* Jika Anda mengunduh file `insurance.csv` asli langsung dari Kaggle (`https://www.kaggle.com/datasets/mirichoi0218/insurance`), timpa file `insurance.csv` di folder ini dengan file dari Kaggle tersebut.
3. **Buka Terminal / Command Prompt** di dalam folder proyek, lalu install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Latih Model Regresi Linear** (opsional karena model sudah tersedia, atau jalankan untuk melatih ulang dari CSV terbaru):
   ```bash
   python train_model.py
   ```
5. **Jalankan Aplikasi Web**:
   ```bash
   python app.py
   ```
6. Buka browser dan akses alamat: **`http://127.0.0.1:5000`**
