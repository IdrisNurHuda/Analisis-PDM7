import os
import joblib
import numpy as np
from flask import Flask, request, render_template

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- Fungsi untuk Memuat Model dan Encoders ---
def load_artifacts(model_path='model.pkl', encoders_path='encoders.pkl'):
    """
    Memuat model dan encoders dari file.
    Mengembalikan tuple (model, encoders, error_message).
    """
    try:
        # Periksa keberadaan file
        if not os.path.exists(model_path) or not os.path.exists(encoders_path):
            return None, None, "KESALAHAN: `model.pkl` atau `encoders.pkl` tidak ditemukan."

        # Muat file
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        
       
        if not isinstance(encoders, dict):
            return None, None, "KESALAHAN: File `encoders.pkl` tidak valid, seharusnya berisi dictionary."

        print("Model dan encoders berhasil dimuat.")
        return model, encoders, None

    except Exception as e:
        error_msg = f"Terjadi kesalahan fatal saat memuat file: {e}"
        print(error_msg)
        return None, None, error_msg

# Memuat artifak saat aplikasi pertama kali dimulai
model, encoders, initial_error = load_artifacts()


@app.route('/', methods=['GET', 'POST'])
def index():
    """Menangani logika untuk halaman utama, input form, dan prediksi."""
    
    # Jika terjadi error saat memuat model, tampilkan pesan error
    if initial_error:
        return render_template('index.html', prediction_text=initial_error, is_error=True)

    prediction_text = ""
    is_error = False
    
    if request.method == 'POST':
        try:
            
            form_data = {
                'gender': request.form['gender'],
                'race/ethnicity': request.form['race_ethnicity'],
                'lunch': request.form['lunch'],
                'test preparation course': request.form['test_preparation_course']
            }

            
            encoded_input = []
           
            feature_order = ['gender', 'race/ethnicity', 'lunch', 'test preparation course']
            
            for col in feature_order:
                value = form_data[col]
               
                le = encoders[col]
                encoded_value = le.transform([value])[0]
                encoded_input.append(encoded_value)
            
            final_input = np.array(encoded_input).reshape(1, -1)

           
            prediction = model.predict(final_input)
            prediction_proba = model.predict_proba(final_input)

           
            pass_probability = prediction_proba[0][1] * 100
            result = "LULUS" if prediction[0] == 1 else "TIDAK LULUS"
            prediction_text = f"Hasil Prediksi: {result} (Peluang Lulus: {pass_probability:.2f}%)"

        except KeyError as e:
            is_error = True
            prediction_text = f"KESALAHAN: Input '{e}' tidak ditemukan atau salah. Periksa kembali form HTML Anda."
        except ValueError as e:
            is_error = True
            if 'contains new labels' in str(e):
                prediction_text = "KESALAHAN: Nilai yang Anda masukkan tidak valid (tidak ada dalam data pelatihan)."
            else:
                prediction_text = f"KESALAHAN pada nilai input: {e}"
        except Exception as e:
            is_error = True
            prediction_text = f"Terjadi kesalahan saat melakukan prediksi: {e}"

   
    return render_template('index.html', prediction_text=prediction_text, is_error=is_error)


if __name__ == '__main__':

    app.run(debug=True, port=5000)