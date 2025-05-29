import streamlit as st
import numpy as np
import joblib
import pandas as pd

# === Load the trained Random Forest model pipeline ===
model = joblib.load("random_forest_pipeline.pkl")

# === App Title ===
st.title("🍷 Wine Quality Predictor")
st.subheader("Powered by Random Forest Classifier")

st.markdown("""
Use the sliders below to simulate wine properties and predict whether it will be **high-quality** or **low-quality**.
""")

# === User Inputs ===
alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 11.0, step=0.1)
density = st.slider("Density (g/cm³)", 0.9900, 1.0050, 0.9950, step=0.0001)
volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.10, 1.50, 0.40, step=0.01)

input_data = pd.DataFrame([[alcohol, density, volatile_acidity]],
                          columns=["alcohol", "density", "volatile acidity"])

# === Predict Button ===
if st.button("Predict Wine Quality"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    quality_label = "🍇 High Quality" if prediction == 1 else "🍷 Low Quality"
    confidence = round(np.max(probability) * 100, 2)

    st.markdown(f"### Prediction: **{quality_label}**")
    st.markdown(f"Confidence: **{confidence}%**")

    st.markdown("### Input Summary")
    st.dataframe(input_data.style.format("{:.4f}"))

st.markdown("---")
st.caption("Model trained on physicochemical attributes of red and white wines.\nPrediction uses alcohol, density, and volatile acidity.")