from flask import Flask, request, render_template
import pandas as pd
import joblib
import os
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Variabel global untuk menyimpan model, encoders, dan status kesalahan
model = None
encoders = None
error_message = None

# --- HANYA MEMUAT MODEL DAN ENCODERS ---
# Tidak ada pelatihan di sini. Aplikasi ini hanya untuk inferensi/prediksi.
try:
    print("Mencoba memuat model.pkl dan encoders.pkl...")
    
    # Pastikan file-file ini berada di direktori yang sama dengan app.py
    model_path = 'model.pkl'
    encoders_path = 'encoders.pkl'

    if not os.path.exists(model_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError()

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    
    # Verifikasi bahwa encoders yang dimuat adalah dictionary
    if not isinstance(encoders, dict):
        raise TypeError("File 'encoders.pkl' tidak valid. Seharusnya berisi dictionary.")

    print("Model dan encoders berhasil dimuat.")

except FileNotFoundError:
    error_message = ("KESALAHAN: File `model.pkl` atau `encoders.pkl` tidak ditemukan. "
                     "Harap letakkan kedua file tersebut dari Google Colab Anda di direktori yang sama dengan file `app.py` ini.")
    print(error_message)
except Exception as e:
    error_message = f"Terjadi kesalahan saat memuat model: {e}"
    print(error_message)


# --- Rute Aplikasi ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Jika ada error saat memuat model, tampilkan pesan error saja
    if error_message:
        return render_template('index.html', prediction_text=error_message, is_error=True)

    prediction_text = ""
    if request.method == 'POST':
        try:
            # Mengambil data teks dari form
            form_data = {
                'gender': request.form['gender'],
                'race/ethnicity': request.form['race_ethnicity'],
                'lunch': request.form['lunch'],
                'test preparation course': request.form['test_preparation_course']
            }

            # Mengubah input teks menjadi angka menggunakan encoders yang tersimpan
            encoded_input = []
            for col, value in form_data.items():
                le = encoders[col]
                encoded_value = le.transform([value])[0]
                encoded_input.append(encoded_value)
            
            final_input = np.array(encoded_input).reshape(1, -1)

            # Melakukan prediksi
            prediction = model.predict(final_input)
            prediction_proba = model.predict_proba(final_input)

            # Menyiapkan teks hasil prediksi
            pass_probability = prediction_proba[0][1] * 100
            if prediction[0] == 1:
                result = "LULUS"
            else:
                result = "TIDAK LULUS"
            
            prediction_text = f"Hasil Prediksi: {result} (Peluang Lulus: {pass_probability:.2f}%)"

        except Exception as e:
            if isinstance(e, ValueError) and 'y contains new labels' in str(e):
                 prediction_text = "Terjadi kesalahan: Salah satu nilai input tidak ada dalam data pelatihan."
            else:
                prediction_text = f"Terjadi kesalahan saat prediksi: {e}"

    # Render halaman HTML
    return render_template('index.html', prediction_text=prediction_text)

# Menjalankan aplikasi
if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)