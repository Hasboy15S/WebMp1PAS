from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model + scaler
with open("Personality.pkl", "rb") as f:
    paket = pickle.load(f)

model = paket["model"]
scaler = paket["scaler"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = pd.DataFrame([{
            "Time_spent_Alone": float(request.form["time_spent_alone"]),
            "Stage_fear": int(request.form["stage_fear"]),
            "Social_event_attendance": float(request.form["social_event"]),
            "Going_outside": float(request.form["going_outside"]),
            "Drained_after_socializing": int(request.form["drained"]),
            "Friends_circle_size": float(request.form["friends"]),
            "Post_frequency": float(request.form["post_freq"])
        }])

        # WAJIB scale dulu
        data_scaled = scaler.transform(data)

        prediksi = model.predict(data_scaled)

        mapping = {
            0: "Extrovert",
            1: "Introvert"
        }

        hasil = mapping.get(int(prediksi[0]), "Tidak diketahui")

        return render_template("index.html", hasil=hasil)

    except Exception as e:
        print("ERROR:", e)
        return render_template("index.html", hasil=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True) 