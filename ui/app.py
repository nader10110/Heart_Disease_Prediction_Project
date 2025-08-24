import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = r"D:\DATA_ANALYSIS\ML_\1\Heart_Disease_Project\models\Final_model.pkl"
model = joblib.load(MODEL_PATH)

st.title(" Heart Disease Prediction App")
st.write("Enter patient information :")
age = st.number_input("Age", min_value=30, max_value=120, step=1)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, step=1)
fbs = st.selectbox("Fasting Blood Sugar", [("≤120 mg/dl", 0), ("≥120 mg/dl", 1)], format_func=lambda x: x[0])[1]
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, step=1)
exang = st.selectbox("Exercise Induced Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
cp = st.selectbox("Chest Pain Type", [
    ("Asymptomatic", 0),
    ("Atypical Angina", 1),
    ("Typical Angina", 2),
    ("Non Angina", 3)
], format_func=lambda x: x[0])[1]
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0–3)", min_value=0, max_value=3, step=1)
thal = st.selectbox("Thalassemia", [
    ("Normal", 1),
    ("Fixed Defect", 2),
    ("Reversible Defect", 3)
], format_func=lambda x: x[0])[1]
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [
    ("Upsloping", 1),
    ("Flat", 2),
    ("Downsloping", 3)
], format_func=lambda x: x[0])[1]

user_data = pd.DataFrame([[
    age, chol, fbs, thalach, exang, cp, ca, thal, oldpeak, slope
]], columns=['age', 'chol', 'fbs', 'thalach', 'exang', 'cp', 'ca', 'thal', 'oldpeak', 'slope'])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    prob = model.predict_proba(user_data)[0][1] * 100  

    if prediction == 1:
        st.error(f" Result: The patient is at risk of heart disease (Probability : {prob:.1f}%)")
    else:
        st.success(f" Result: The patient is not at risk of heart disease (Probability : {prob:.1f}%)")

