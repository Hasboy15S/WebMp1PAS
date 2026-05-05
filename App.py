# PINDAHKAN SEMUA IMPORT KE PALING ATAS
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Buka file pkl
model = pickle.load(open('model_prediksi.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# ... kode lainnya ...

# TARUH INI DI PALING BAWAH
if __name__ == "__main__":
    app.run(debug=True)