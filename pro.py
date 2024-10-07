import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Title of the app
st.title("Years of Experience to Salary Predictor")

# Load the trained model from the pickle file
with open('finalized_model.pickle', 'rb') as file:
    model = pickle.load(file)
with open('Scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)
# Input: Years of Experience
x = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.1)

# Predict Salary using the loaded model
y = model.predict(np.array([[x]]))[0]

# Output: Predicted Salary
st.write(f"Predicted Salary: ${y:.2f}")
