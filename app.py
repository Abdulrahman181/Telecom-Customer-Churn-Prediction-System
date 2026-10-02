import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from features import engineer_features

APP_DIR = Path(__file__).resolve().parent
pipeline = joblib.load(APP_DIR / "artifacts" / "churn_pipeline.pkl")
config = joblib.load(APP_DIR / "artifacts" / "model_config.pkl")

threshold = config["optimal_threshold"]

st.set_page_config(
    page_title="Telecom Churn Prediction",
    layout="centered"
)

st.title("Telecom Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

gender = st.selectbox("Gender", ["Male", "Female"])

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    0.0,
    200.0,
    70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

if st.button("Predict Churn"):

    customer_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    customer_data = engineer_features(customer_data)

    probability = pipeline.predict_proba(customer_data)[0, 1]

    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")

    st.write(
        f"Churn Probability: {probability:.2%}"
    )

    if prediction == 1:
        st.error("Customer Likely to Churn")
    else:
        st.success("Customer Likely to Stay")

    if probability >= 0.7:
        st.warning("High Risk Customer")
    elif probability >= 0.4:
        st.info("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")