from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- LOAD MODEL PERSONALITY ---
try:
    with open('Personality.pkl', 'rb') as f:
        # Karena Personality.pkl adalah dictionary {"model": ..., "scaler": ...}
        model_pers_packet = pickle.load(f)
except FileNotFoundError:
    print("Error: Personality.pkl tidak ditemukan")

# --- LOAD MODEL STUNTING ---
try:
    with open('ModelStunting.pkl', 'rb') as f:
        # Karena ModelStunting.pkl adalah dictionary {"model": ..., "scaler": ..., "le_status": ...}
        paket_stunting = pickle.load(f)
except FileNotFoundError:
    print("Error: ModelStunting.pkl tidak ditemukan")

@app.route('/')
def index():
    return render_template('index.html')

# Jalur Prediksi Personality
@app.route('/predict_personality', methods=['POST'])
def predict_pers():
    try:
        # Ambil Model dan Scaler dari dalam Dictionary
        model = model_pers_packet['model']
        scaler = model_pers_packet['scaler']

        data = {
            'Time_spent_Alone': float(request.form.get('time_spent_alone', 0)),
            'Stage_fear': int(request.form.get('stage_fear', 0)),
            'Social_event_attendance': float(request.form.get('social_event', 0)),
            'Going_outside': float(request.form.get('going_outside', 0)),
            'Drained_after_socializing': int(request.form.get('drained', 0)),
            'Friends_circle_size': float(request.form.get('friends', 0)),
            'Post_frequency': float(request.form.get('post_freq', 0))
        }
        
        # Ubah ke DataFrame dan lalui proses Scaling
        df = pd.DataFrame([data])
        data_scaled = scaler.transform(df.values) 
        
        # Prediksi
        pred = model.predict(data_scaled)
        
        mapping = {0: "Introvert", 1: "Extrovert"}
        hasil = mapping.get(int(pred[0]), "Tidak Diketahui")
        
        return render_template('index.html', hasil_pers=hasil, active_tab='personality')
    except Exception as e:
        return f"Error Personality: {e}"

# Jalur Prediksi Stunting
# Jalur Prediksi Stunting
@app.route('/predict_stunting', methods=['POST'])
def predict_stunt():
    try:
        # 1. Ambil semua komponen dari paket stunting
        model = paket_stunting['model']
        scaler = paket_stunting['scaler']
        le_status = paket_stunting['le_status']
        
        # 2. Ambil data dari form website
        umur_val = request.form.get('umur')
        jk_val = request.form.get('jk')
        tinggi_val = request.form.get('tinggi')

        # Proteksi input kosong
        if not umur_val or not tinggi_val:
            return "Error: Umur dan Tinggi harus diisi!"

        umur = float(umur_val)
        jk = int(jk_val)
        tinggi = float(tinggi_val)
        
        # 3. Proses Data (Wajib di-scale)
        data_raw = np.array([[umur, jk, tinggi]])
        data_scaled = scaler.transform(data_raw)
        
        # 4. Prediksi (masih menghasilkan angka)
        pred_indeks = model.predict(data_scaled)
        
        # 5. TRANSLATE: Mengubah angka (misal: 3) menjadi teks asli (misal: "stunted")
        hasil_final = le_status.inverse_transform(pred_indeks)[0]
        
        # 6. Kirim ke HTML
        return render_template('index.html', hasil_stunting=hasil_final, active_tab='stunting')
    
    except Exception as e:
        print(f"Error Stunting: {e}")
        return f"Terjadi kesalahan pada model stunting: {e}"

if __name__ == '__main__':
    app.run(debug=True)
# Di bagian paling bawah app.py
app = app # Menegaskan kembali variabel app untuk Vercel
