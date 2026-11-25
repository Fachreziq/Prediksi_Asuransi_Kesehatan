import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Prediksi & Klasifikasi Biaya Asuransi Kesehatan",
    page_icon="🩺",
    layout="wide"
)

@st.cache_data
def load_dataset(csv_path="insurance.csv"):
    return pd.read_csv(csv_path)

@st.cache_resource
def train_regression_model(df):
    data = df[["age", "bmi", "smoker", "charges"]].dropna().copy()
    data["smoker_num"] = data["smoker"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    data = data.dropna(subset=["smoker_num"])

    X = data[["age", "bmi", "smoker_num"]]
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    q1 = float(data["charges"].quantile(0.33))
    q2 = float(data["charges"].quantile(0.66))

    return model, data, X_test, y_test, y_pred, r2, mae, rmse, q1, q2

def klasifikasi_biaya(biaya, q1, q2):
    if biaya < q1:
        return (
            "Biaya Rendah (Low Risk)",
            "Estimasi biaya berada di bawah persentil ke-33 dataset. Profil risiko biaya tergolong rendah.",
            "Pertahankan pola hidup sehat, BMI ideal, dan hindari kebiasaan merokok."
        )
    elif biaya < q2:
        return (
            "Biaya Sedang (Moderate Risk)",
            "Estimasi biaya berada pada rentang menengah (persentil 33 - 66). Usia atau BMI mulai memengaruhi estimasi biaya.",
            "Lakukan pemeriksaan kesehatan berkala dan jaga berat badan tetap seimbang."
        )
    else:
        return (
            "Biaya Tinggi (High Risk)",
            "Estimasi biaya berada di atas persentil ke-66 dataset. Umumnya dipicu kebiasaan merokok aktif atau kombinasi usia dan BMI tinggi.",
            "Disarankan berhenti merokok serta konsultasi medis untuk menekan faktor risiko kesehatan."
        )

def kategori_bmi(bmi):
    if bmi < 18.5:
        return "Kurus (Underweight)"
    elif bmi < 25.0:
        return "Normal (Ideal)"
    elif bmi < 30.0:
        return "Berat Badan Berlebih (Overweight)"
    else:
        return "Obesitas (Obese)"

st.title("Prediksi Biaya Asuransi Kesehatan Berdasarkan Usia, BMI, dan Kebiasaan Merokok Menggunakan Regresi Linear")
st.caption("Aplikasi Streamlit | Dataset: Kaggle insurance.csv | Algoritma: Multiple Linear Regression")

try:
    df = load_dataset("insurance.csv")
except FileNotFoundError:
    st.error("File `insurance.csv` tidak ditemukan di repository. Pastikan `insurance.csv` berada satu folder dengan `app.py`.")
    st.stop()

model, clean_df, X_test, y_test, y_pred, r2, mae, rmse, q1, q2 = train_regression_model(df)

tab1, tab2, tab3 = st.tabs(["Prediksi & Klasifikasi", "Evaluasi & Visualisasi Model", "Eksplorasi Dataset"])

with tab1:
    col_input, col_output = st.columns([1, 1.25], gap="large")

    with col_input:
        st.subheader("Input Parameter")
        usia = st.slider("Usia (Tahun)", min_value=18, max_value=80, value=30, step=1)
        bmi = st.number_input("Indeks Massa Tubuh / BMI (kg/m²)", min_value=12.0, max_value=60.0, value=25.5, step=0.1)
        merokok = st.radio("Kebiasaan Merokok", options=["Tidak (no)", "Ya (yes)"], horizontal=True)
        kurs = st.number_input("Kurs USD ke IDR (Opsional)", min_value=1000, max_value=50000, value=16000, step=100)

        hitung = st.button("Hitung Prediksi & Klasifikasi", type="primary", use_container_width=True)

    with col_output:
        st.subheader("Hasil Prediksi & Klasifikasi")
        if hitung or True:
            smoker_num = 1 if merokok.startswith("Ya") else 0
            fitur_input = np.array([[float(usia), float(bmi), float(smoker_num)]])
            pred_usd = max(float(model.predict(fitur_input)[0]), 1000.0)
            pred_idr = pred_usd * float(kurs)

            kategori, deskripsi, rekomendasi = klasifikasi_biaya(pred_usd, q1, q2)

            m1, m2, m3 = st.columns(3)
            m1.metric("Estimasi Biaya (USD)", f"${pred_usd:,.2f}")
            m2.metric("Estimasi Biaya (IDR)", f"Rp {pred_idr:,.0f}".replace(",", "."))
            m3.metric("Klasifikasi Risiko", kategori.split(" (")[0])

            if kategori.startswith("Biaya Rendah"):
                st.success(f"**{kategori}** — {deskripsi}")
            elif kategori.startswith("Biaya Sedang"):
                st.warning(f"**{kategori}** — {deskripsi}")
            else:
                st.error(f"**{kategori}** — {deskripsi}")

            st.info(f"**Rekomendasi:** {rekomendasi}")

            kontribusi_intercept = float(model.intercept_)
            kontribusi_usia = float(model.coef_[0]) * float(usia)
            kontribusi_bmi = float(model.coef_[1]) * float(bmi)
            kontribusi_smoker = float(model.coef_[2]) * float(smoker_num)

            rincian_df = pd.DataFrame({
                "Komponen Regresi": ["Intercept (b0)", "Usia (Age)", "BMI", "Kebiasaan Merokok (Smoker)"],
                "Nilai / Status": ["-", f"{usia} tahun", f"{bmi} ({kategori_bmi(bmi)})", merokok],
                "Kontribusi (USD)": [
                    f"${kontribusi_intercept:,.2f}",
                    f"+${kontribusi_usia:,.2f}",
                    f"+${kontribusi_bmi:,.2f}",
                    f"+${kontribusi_smoker:,.2f}"
                ]
            })
            st.dataframe(rincian_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Metrik Evaluasi & Persamaan Regresi Linear")
    e1, e2, e3 = st.columns(3)
    e1.metric("R-Squared (R²)", f"{r2:.4f} ({r2*100:.2f}%)")
    e2.metric("MAE", f"${mae:,.2f}")
    e3.metric("RMSE", f"${rmse:,.2f}")

    st.code(
        f"Y = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * Usia) + ({model.coef_[1]:.4f} * BMI) + ({model.coef_[2]:.4f} * Smoker)",
        language="text"
    )
    st.caption(f"Ambang klasifikasi biaya: Rendah < ${q1:,.2f} | Sedang ${q1:,.2f} - ${q2:,.2f} | Tinggi >= ${q2:,.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_test, y_pred, alpha=0.65)
    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Garis Ideal (y = x)")
    ax.set_title("Perbandingan Biaya Aktual vs Prediksi (Data Uji)")
    ax.set_xlabel("Biaya Aktual (USD)")
    ax.set_ylabel("Biaya Prediksi (USD)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Cuplikan Dataset & Korelasi Fitur")
    st.dataframe(clean_df[["age", "bmi", "smoker", "charges"]].head(15), use_container_width=True, hide_index=True)

    fig_corr, ax_corr = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(clean_df[["age", "bmi", "smoker_num", "charges"]].corr(), annot=True, cmap="Blues", fmt=".3f", ax=ax_corr)
    ax_corr.set_title("Matriks Korelasi Variabel")
    plt.tight_layout()
    st.pyplot(fig_corr)
