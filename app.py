import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("heart_model.pkl")  # or the correct filename

st.title("Heart Disease Prediction App")

st.write("Provide the patient's details to predict heart disease.")

# --- INPUTS (7 FEATURES ONLY — SAME ORDER AS TRAINING) ---
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250)
ca = st.selectbox("CA – Number of Major Vessels (0–3)", [0, 1, 2, 3])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
age = st.number_input("Age", min_value=20, max_value=100)
thal = st.selectbox("Thalassemia (thal)", [1, 2, 3])
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600)

# Create input array in correct order
input_data = np.array([[cp, thalach, ca, oldpeak, age, thal, chol]])

if st.button("Predict"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")
