import streamlit as st
import pickle
import numpy as np
import tensorflow as tf

# 1. Load resources once (caching prevents reloading on every button click)
@st.cache_resource
def load_models():
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = tf.keras.models.load_model('churn_model.keras')
    return scaler, model

scaler, model = load_models()

# 2. Build the UI
st.title("Bank Customer Churn Predictor 🏦")
st.write("Enter the customer's details below to predict if they will leave the bank.")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=20, value=5)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=0.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_crcard = st.selectbox("Has Credit Card?", [0, 1])
    is_active = st.selectbox("Is Active Member?", [0, 1])
    salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# 3. Prediction Button and Logic
if st.button("Predict Churn"):

    # Hardcoded mapping
    geo_map = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_map = {'Female': 0, 'Male': 1}

    # Format the input array exactly like your training data
    input_data = np.array([[
        credit_score,
        geo_map[geography],
        gender_map[gender],
        age,
        tenure,
        balance,
        num_products,
        has_crcard,
        is_active,
        salary
    ]])

    # Scale and predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0][0]

    # Display results
    st.divider()
    if prediction > 0.5:
        st.error(f"⚠️ High Risk of Churn! (Probability: {prediction*100:.2f}%)")
    else:
        st.success(f"✅ Customer will likely stay. (Probability of churning: {prediction*100:.2f}%)")
