import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Loading the transformer
with open('transformer.pkl', 'rb') as file:
    transformer = pickle.load(file)

# Loading the final Gradient Boosting model
with open('gb_model_final.pkl', 'rb') as file:
    gb_model_final = pickle.load(file)

# Title and header
st.title("Car Price Prediction - Gradient Boosting")

# Input fields
mmr = st.number_input('MMR', min_value=0, max_value=100000, step=100)
condition = st.slider('Condition (0-10)', 0, 10, 5)
odometer = st.number_input('Odometer Reading', min_value=0, max_value=300000, step=1000)
vehicle_age = st.number_input('Vehicle Age', min_value=0, max_value=30, step=1)

# Prediction button
if st.button('Predict Selling Price'):
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

    # Display the prediction
    st.success(f"Estimated Selling Price: ${prediction[0]:,.2f}")

# Footer
st.markdown("---")
st.markdown("Created by Pradeep,Vivek,Sahithi,Spadana - LET'S GO 🚗💨")
