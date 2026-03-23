import pickle
from flask import Flask, request, jsonify

# Load model
with open("climate_change_dataset.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder (for Country column)
try:
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except:
    le = None

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Encode Country → number  (9th feature)
    country = data.get("Country", "India")
    if le:
        try:
            country_encoded = le.transform([country])[0]
        except:
            country_encoded = 0   # unknown country → 0
    else:
        country_encoded = 0

    # All 9 features in the same order as training
    features = [[
        data["Year"],
        data["Avg Temperature (°C)"],
        data["CO2 Emissions (Tons/Capita)"],
        data["Sea Level Rise (mm)"],
        data["Rainfall (mm)"],
        data["Population"],
        data["Renewable Energy (%)"],
        data["Forest Area (%)"],
        country_encoded               # ← 9th feature (was missing before!)
    ]]

    result = model.predict(features)
    return jsonify({"Prediction": str(result[0])})

if __name__ == "__main__":
    app.run(debug=True)