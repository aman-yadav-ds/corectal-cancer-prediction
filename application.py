from flask import Flask, render_template, request
import joblib
import numpy as np


app = Flask(__name__)

model_path = "artifacts/model/model.pkl"
scalar_path = "artifacts/processed/scaler.pkl"

model = joblib.load(model_path)
scalar = joblib.load(scalar_path)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        healthcare_cost = float(request.form["costs"])
        tumor_size = float(request.form["tumor"])
        treatment_type = int(request.form["treatment"])
        diabetes = int(request.form["diabetes"])
        mortality_rate = float(request.form["mortality"])

        input = np.array([[healthcare_cost, tumor_size, treatment_type, diabetes, mortality_rate]])
        
        scaled_input = scalar.transform(input)

        prediction = model.predict(scaled_input)[0]

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return str(e)
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
