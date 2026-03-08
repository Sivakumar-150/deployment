import numpy as np
from flask import Flask, request, jsonify
import pickle

with open("uptor_linear_trained_model.pkl", "rb") as file_reading_obj:
    train_data = pickle.load(file_reading_obj)

my_app = Flask(__name__)

@my_app.route("/linear_model_predict", methods = ["POST"])
def linear_prediction():
    data = request.get_json()
    year_from_user = data.get('Year')
    if not year_from_user or not isinstance(year_from_user, list):
        return jsonify({'error message':'please validate your input'}), 400

    convert_df = np.array(year_from_user).reshape(-1,1)
    prediction = train_data.predict(convert_df)
    return jsonify({"Year": year_from_user,"Prediction": prediction.tolist() }), 200
@my_app.route("/")
def landing():
    return "Welcome to my uptor"

@my_app.route("/login")
def login():
    return "Welcome to login"

if __name__ == "__main__":
    my_app.run(debug= True)
