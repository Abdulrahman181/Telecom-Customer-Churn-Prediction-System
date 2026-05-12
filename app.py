import streamlit as st
import pandas as pd
import numpy as np
import joblib

pipeline = joblib.load("artifacts/churn_pipeline.pkl")
config = joblib.load("artifacts/model_config.pkl")

threshold = config["optimal_threshold"]

def engineer_features(df):

    df = df.copy()

    df['AvgChargePerMonth'] = (
        df['TotalCharges'] / (df['tenure'] + 1)
    )

    df['CLV'] = (
        df['MonthlyCharges'] * df['tenure']
    )

    df['ChargeIncreaseRatio'] = (
        df['MonthlyCharges'] /
        np.maximum(df['AvgChargePerMonth'], 1)
    )

    df['IsNewCustomer'] = (
        df['tenure'] < 12
    ).astype(int)

    df['IsMidTerm'] = (
        (df['tenure'] >= 12) &
        (df['tenure'] < 36)
    ).astype(int)

    df['IsLoyalCustomer'] = (
        df['tenure'] >= 36
    ).astype(int)

    df['TenureSquared'] = (
        df['tenure'] ** 2
    )

    service_cols = [
        'PhoneService',
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    temp = df[service_cols].replace({
        'Yes': 1,
        'No': 0,
        'No internet service': 0,
        'No phone service': 0
    })

    df['TotalServices'] = temp.sum(axis=1)

    df['ServiceAdoptionRate'] = (
        df['TotalServices'] / len(service_cols)
    )

    monthly_charge_threshold = 70

    df['HighMonthlyCharge'] = (
        df['MonthlyCharges'] > monthly_charge_threshold
    ).astype(int)

    df['NoSecurityBundle'] = (
        (df['OnlineSecurity'] == 'No') &
        (df['DeviceProtection'] == 'No')
    ).astype(int)

    df['MonthToMonthContract'] = (
        df['Contract'] == 'Month-to-month'
    ).astype(int)

    df['ElectronicCheckPay'] = (
        df['PaymentMethod'] == 'Electronic check'
    ).astype(int)

    df['ChurnRiskScore'] = (
        df['IsNewCustomer'] * 3 +
        df['MonthToMonthContract'] * 3 +
        df['ElectronicCheckPay'] * 2 +
        df['NoSecurityBundle'] * 1 +
        df['HighMonthlyCharge'] * 1
    )

    return df

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