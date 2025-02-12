import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# Load Transformer and Model
with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

with open('gb_model_final.pkl', 'rb') as file:
    gb_model_final = pickle.load(file)

# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Background image styling
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: url('https://source.unsplash.com/1600x900/?cars,automobile') no-repeat center center fixed;
    background-size: cover;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title & Header
st.markdown("<h1 style='text-align: center; color: white;'>Car Price Prediction - Gradient Boosting</h1>", unsafe_allow_html=True)

# Layout for inputs
col1, col2 = st.columns(2)

with col1:
    mmr = st.number_input('Market Value (MMR)', min_value=0, max_value=100000, step=100, help="Enter the Market Value of the car.")
    condition = st.slider('Car Condition (0-10)', 0, 10, 5, help="Rate the car's condition from 0 (Poor) to 10 (Excellent).")

with col2:
    odometer = st.number_input('Odometer Reading (in miles)', min_value=0, max_value=300000, step=1000, help="Enter the car's total mileage.")
    vehicle_age = st.number_input('Vehicle Age (in years)', min_value=0, max_value=30, step=1, help="Enter how old the car is.")

# Center align button
st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
if st.button('🚘 Predict Selling Price'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'mmr': [mmr],
        'condition': [condition],
        'odometer': [odometer],
        'vehicle_age': [vehicle_age]
    })

    # Apply log transformation to 'mmr'
    input_data['mmr'] = transformer.transform(input_data[['mmr']])

    # Make prediction
    prediction = gb_model_final.predict(input_data)

    # Display prediction result
    st.markdown(
        f"""<h2 style='text-align: center; color: yellow;'>
        Estimated Selling Price: ${prediction[0]:,.2f}
        </h2>""", 
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
---
<h4 style='text-align: center; color: white;'>Created by Pradeep, Vivek, Sahithi, Spadana - LET'S GO 🚗💨</h4>
""", unsafe_allow_html=True)
