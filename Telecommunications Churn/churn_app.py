import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📞")
st.title(" 📞 Telecom Customer Churn Predictor")
st.markdown("Predict whether a telecom customer is likely to **churn** based on their details.")

# Define user input fields – match your model's features
def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

    # Convert categorical values into numbers (must match training encoding)
    feature_list = [
        gender, senior_citizen, partner, dependents, tenure, phone_service,
        multiple_lines, internet_service, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges
    ]

    # Simple manual encoding – ensure it matches the same order and style as your model training
    mapping = {
        'Male': 1, 'Female': 0,
        'Yes': 1, 'No': 0,
        'No phone service': 0,
        'No internet service': 0,
        'DSL': 1, 'Fiber optic': 2, 'No': 0,
        'Month-to-month': 0, 'One year': 1, 'Two year': 2,
        'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
    }

    encoded = [mapping.get(val, val) if isinstance(val, str) else val for val in feature_list]
    return np.array([encoded])

input_data = user_input_features()

# Predict churn
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("The customer is **likely to churn**.")
    else:
        st.success(" The customer is **likely to stay**.")

