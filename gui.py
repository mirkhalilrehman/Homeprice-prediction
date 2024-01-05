import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('trained_house_price_model.pkl')

# Function to predict house prices
def predict_price(area, bedrooms, stories, parking, airconditioning):
    input_data = np.array([area, bedrooms, stories, parking, airconditioning]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title('House Price Prediction App')

    st.sidebar.header('Input Features')
    area = st.sidebar.number_input('Area', min_value=0, max_value=100000, step=100)
    bedrooms = st.sidebar.number_input('Bedrooms', min_value=1, max_value=10, step=1)
    stories = st.sidebar.number_input('Stories', min_value=1, max_value=5, step=1)
    parking = st.sidebar.number_input('Parking', min_value=0, max_value=5, step=1)
    airconditioning = st.sidebar.selectbox('Air Conditioning', ['yes', 'no'])

    airconditioning_binary = 1 if airconditioning == 'yes' else 0

    if st.sidebar.button('Predict'):
        result = predict_price(area, bedrooms, stories, parking, airconditioning_binary)
        st.success(f'The predicted house price is ${result:,.2f}')

if __name__ == '__main__':
    main()
