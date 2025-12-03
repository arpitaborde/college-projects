from flask import Flask, render_template, request
import pickle
import numpy as np
import requests

app = Flask(__name__)

# Load trained model (3 features: temp, humidity, pm25)
model = pickle.load(open("model/air_model.pkl", "rb"))

# Helper function to fetch today's AQI from WAQI
def get_today_aqi(city="mumbai"):
    API_TOKEN = "d93fe8428d5b62fc3d3244677ee0b1ff"
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print("API request failed:", response.text)
            return None

        data = response.json()
        if "data" not in data or "aqi" not in data["data"]:
            print("No AQI data in response:", data)
            return None

        return data["data"]["aqi"]  # returns AQI number

    except Exception as e:
        print("Error fetching AQI:", str(e))
        return None

# Home route
@app.route("/")
def home():
    today_aqi = get_today_aqi("mumbai")  # You can change city
    return render_template("index.html", today_aqi=today_aqi)

# About page
@app.route("/about")
def about():
    return render_template("about.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        temp = float(request.form["temp"])
        humidity = float(request.form["humidity"])
        pm25 = float(request.form["pm25"])

        # Input for model
        input_data = np.array([[temp, humidity, pm25]])
        prediction = model.predict(input_data)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("result.html", prediction="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
