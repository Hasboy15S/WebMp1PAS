from flask import Flask, render_template, request, jsonify # Tambahkan jsonify di sini
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- LOAD MODEL PERSONALITY ---
try:
    with open('Personality.pkl', 'rb') as f:
        model_pers_packet = pickle.load(f)
except FileNotFoundError:
    print("Error: Personality.pkl tidak ditemukan")

# --- LOAD MODEL STUNTING ---
try:
    with open('ModelStunting.pkl', 'rb') as f:
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
        
        df = pd.DataFrame([data])
        data_scaled = scaler.transform(df.values) 
        
        pred = model.predict(data_scaled)
        
        mapping = {1: "Introvert", 0: "Extrovert"}
        hasil = mapping.get(int(pred[0]), "Tidak Diketahui")
        
        return jsonify({'hasil': hasil}) # Mengirim JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Jalur Prediksi Stunting
@app.route('/predict_stunting', methods=['POST'])
def predict_stunt():
    try:
        model = paket_stunting['model']
        scaler = paket_stunting['scaler']
        le_status = paket_stunting['le_status']
        
        umur_val = request.form.get('umur')
        jk_val = request.form.get('jk')
        tinggi_val = request.form.get('tinggi')

        if not umur_val or not tinggi_val:
            return jsonify({'error': 'Umur dan Tinggi harus diisi!'}), 400

        umur = float(umur_val)
        jk = int(jk_val)
        tinggi = float(tinggi_val)
        
        data_raw = np.array([[umur, jk, tinggi]])
        data_scaled = scaler.transform(data_raw)
        
        pred_indeks = model.predict(data_scaled)
        hasil_final = le_status.inverse_transform(pred_indeks)[0]
        
        return jsonify({'hasil': hasil_final}) # Mengirim JSON
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

# Khusus untuk Vercel
app = app