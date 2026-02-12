import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(layout="wide")

# Load your pre-trained model
with open('log_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
#gathering the inputs
def get_user_input():
    Pregnancies = st.number_input("No. of Pregnancies", min_value=0)
    Glucose = st.number_input("Glucose level", min_value=0)
    BloodPressure = st.number_input("Blood Pressure Level", min_value=0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0)
    Insulin = st.number_input("Insulin Level", min_value=0)
    BMI = st.number_input("BMI",min_value=0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0)
    Age = st.number_input("Age")
    
    user_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age,
    }
    return user_data

st.markdown("<h1 style='text-align: center;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)

st.subheader("Please Enter All Parameter Values In Number")

user_data = get_user_input()

#Transforming the Inputs
def prepare_input(data, feature_list):
    input_data = {feature: data.get(feature, 0) for feature in feature_list}
    return np.array([list(input_data.values())])

# Feature list (same order as used during model training)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age']

# Predict button -- Making the Prediction
if st.button("Predict"):
    input_array = prepare_input(user_data, features)
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)[0][1]
    st.subheader("Prediction Outcome")
    #st.write(f"{input_array}")
    #st.write(f"{prediction[0]:,.2f}")

    if prediction[0] == 1:
       st.error(f"Warning! The Patient is Diabetic (Probability: {probability:.2f})")
    else:
       st.success(f"The Patient is Not Diabetic (Probability: {probability:.2f})")


#python -m streamlit run Diabetes_Prediction.py

