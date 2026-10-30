import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load("models/crop_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# Application title
st.title("🌱 Smart Crop Recommendation System")
st.write(
    "Enter the soil and weather details below "
    "to get a recommended crop."
)

st.divider()

# Input fields
st.subheader("Enter Agricultural Details")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
    K = st.number_input("Potassium (K)", min_value=0.0, value=43.0)
    temperature = st.number_input(
        "Temperature (°C)", value=20.87
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0,
        value=82.0
    )
    ph = st.number_input(
        "Soil pH", min_value=0.0, max_value=14.0,
        value=6.5
    )
    rainfall = st.number_input(
        "Rainfall (mm)", min_value=0.0, value=202.9
    )

st.divider()

# Prediction button
if st.button("Recommend Crop", type="primary"):

    # Prepare input in the same order used during training
    input_data = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=[
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ])

    # Predict crop
    prediction = model.predict(input_data)

    # Display result
    st.success("Crop recommendation generated!")
    st.subheader("Recommended Crop")
    st.title(f"🌾 {prediction[0].upper()}")

    st.info(
        "This recommendation is based on the trained "
        "machine learning model and the values entered."
    )

st.caption(
    "Academic project demonstration. Check local soil "
    "conditions and agricultural advice before planting."
)