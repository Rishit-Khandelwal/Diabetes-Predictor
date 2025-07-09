import streamlit as st
import numpy as np
import joblib

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scalar.pkl')  # ✅ Load scaler

st.title("🩺 Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    scaled_data = scaler.transform(data)  # ✅ Apply same scaling
    result = model.predict(scaled_data)

    if result[0] == 1:
        st.error("❌ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is not likely to have diabetes.")
 