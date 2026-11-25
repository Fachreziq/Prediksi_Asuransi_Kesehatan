from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
import pandas as pd
from train_model import train_and_save_model

app = Flask(__name__)

MODEL_PATH = "model/linear_regression_model.pkl"
CSV_PATH = "insurance.csv"

def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save_model(CSV_PATH, MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        saved_data = pickle.load(f)
    return saved_data['model'], saved_data['metrics']

model, metrics = load_model()

def klasifikasi_biaya(biaya, q1, q2):
    """
    Mengklasifikasikan hasil prediksi biaya asuransi ke dalam 3 kategori tingkat beban/risiko biaya.
    """
    if biaya < q1:
        return {
            "kategori": "Biaya Rendah (Low Risk)",
            "badge": "low",
            "deskripsi": "Estimasi premi/biaya asuransi berada di bawah persentil ke-33 dataset. Profil risiko kesehatan tergolong rendah.",
            "rekomendasi": "Pertahankan pola hidup sehat, indeks massa tubuh ideal, dan hindari paparan asap rokok."
        }
    elif biaya < q2:
        return {
            "kategori": "Biaya Sedang (Moderate Risk)",
            "badge": "medium",
            "deskripsi": "Estimasi biaya berada pada rentang menengah (persentil 33 - 66). Faktor usia atau BMI mulai memengaruhi besaran premi.",
            "rekomendasi": "Lakukan pemeriksaan kesehatan berkala dan jaga berat badan tetap proporsional."
        }
    else:
        return {
            "kategori": "Biaya Tinggi (High Risk)",
            "badge": "high",
            "deskripsi": "Estimasi biaya melampaui persentil ke-66 dataset. Umumnya dipicu oleh kebiasaan merokok aktif atau kombinasi usia dan obesitas.",
            "rekomendasi": "Disarankan program berhenti merokok serta konsultasi medis untuk menurunkan faktor risiko kardiovaskular."
        }

def kategori_bmi(bmi):
    if bmi < 18.5:
        return "Kurus (Underweight)"
    elif bmi < 25.0:
        return "Normal (Ideal)"
    elif bmi < 30.0:
        return "Berat Badan Berlebih (Overweight)"
    else:
        return "Obesitas (Obese)"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    form_data = {"age": "", "bmi": "", "smoker": "no", "kurs": 16000}

    if request.method == "POST":
        try:
            age = float(request.form.get("age", 25))
            bmi = float(request.form.get("bmi", 24.0))
            smoker_str = request.form.get("smoker", "no")
            kurs = float(request.form.get("kurs", 16000))

            form_data = {
                "age": int(age) if age.is_integer() else age,
                "bmi": bmi,
                "smoker": smoker_str,
                "kurs": int(kurs)
            }

            smoker_num = 1 if smoker_str == "yes" else 0
            features = np.array([[age, bmi, smoker_num]])
            pred_usd = float(model.predict(features)[0])
            
            # Pastikan prediksi tidak bernilai negatif pada kasus ekstrem
            pred_usd = max(pred_usd, 1000.0)
            pred_idr = pred_usd * kurs

            klasifikasi = klasifikasi_biaya(
                pred_usd,
                metrics.get("q1_threshold", 7700),
                metrics.get("q2_threshold", 15000)
            )

            # Kontribusi masing-masing variabel dalam persamaan regresi
            kontribusi_usia = metrics['coef_age'] * age
            kontribusi_bmi = metrics['coef_bmi'] * bmi
            kontribusi_smoker = metrics['coef_smoker'] * smoker_num

            prediction_result = {
                "usd": f"${pred_usd:,.2f}",
                "usd_raw": round(pred_usd, 2),
                "idr": f"Rp {pred_idr:,.0f}".replace(",", "."),
                "age": form_data["age"],
                "bmi": bmi,
                "bmi_status": kategori_bmi(bmi),
                "smoker_label": "Perokok Aktif (Ya)" if smoker_num == 1 else "Bukan Perokok (Tidak)",
                "klasifikasi": klasifikasi,
                "breakdown": {
                    "intercept": f"{metrics['intercept']:,.2f}",
                    "usia": f"+${kontribusi_usia:,.2f}",
                    "bmi": f"+${kontribusi_bmi:,.2f}",
                    "smoker": f"+${kontribusi_smoker:,.2f}"
                }
            }
        except Exception as e:
            prediction_result = {"error": f"Terjadi kesalahan input: {str(e)}"}

    return render_template("index.html", metrics=metrics, result=prediction_result, form_data=form_data)

@app.route("/retrain", methods=["POST"])
def retrain():
    global model, metrics
    try:
        model, metrics = train_and_save_model(CSV_PATH, MODEL_PATH)
        return jsonify({"status": "success", "message": "Model berhasil dilatih ulang dari insurance.csv!", "metrics": metrics})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
