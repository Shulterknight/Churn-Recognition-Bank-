# Bank Customer Churn Predictor 🏦

A machine learning web application that predicts the likelihood of a bank customer leaving (churning) based on their account details and demographics. 

## Overview
This project uses an Artificial Neural Network (ANN) trained on customer data and features an interactive web interface built with Streamlit.

## Files Included
* **`app.py`**: The main Streamlit web application script.
* **`churn_model.keras`**: The pre-trained TensorFlow/Keras neural network model.
* **`scaler.pkl`**: The saved scikit-learn `StandardScaler` used to normalize user input data.
* **`requirements.txt`**: List of required Python dependencies.

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Shulterknight/Churn-Recognition-Bank-.git](https://github.com/Shulterknight/Churn-Recognition-Bank-.git)
   cd Churn-Recognition-Bank-

2. Install dependencies:
   Make sure you have Python installed, then run:
   Bash:
   pip install -r requirements.txt

3. Run the application:
   Bash:
   streamlit run app.py

