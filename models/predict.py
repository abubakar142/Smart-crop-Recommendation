import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/crop_model.pkl")

print("SMART CROP RECOMMENDATION SYSTEM")
print("--------------------------------")

# Get input from user
N = float(input("Enter Nitrogen value: "))
P = float(input("Enter Phosphorus value: "))
K = float(input("Enter Potassium value: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter Soil pH: "))
rainfall = float(input("Enter Rainfall: "))

# Arrange inputs in correct order
input_data = pd.DataFrame([[
    N, P, K, temperature, humidity, ph, rainfall
]], columns=[
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
])

# Predict crop
prediction = model.predict(input_data)

# Display result
print("\n------------------------------")
print("RECOMMENDED CROP:", prediction[0])
print("------------------------------")