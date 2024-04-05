from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    # Load and preprocess data
    data = pd.read_csv("DFU.csv")
    data.columns = data.columns.str.strip()

    X = data[["Thermistor_Value", "Foot_Pressure_Value"]].copy()
    y = data[""]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()

    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open("svm_model.pkl", "rb") as file:
        loaded_svm_model = pickle.load(file)

    svm_predictions = loaded_svm_model.predict(X_test_scaled)

    return render_template("result.html", predictions=svm_predictions.tolist())


if __name__ == "__main__":
    app.run(debug=True)
