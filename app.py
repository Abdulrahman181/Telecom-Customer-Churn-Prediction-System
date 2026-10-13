"""Streamlit inference UI for the telecom churn pipeline.

The app intentionally does not train a model.  It loads the checked-in artifacts and
fails with an actionable message when they or required input fields are unavailable.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
PIPELINE_PATH = ARTIFACT_DIR / "churn_pipeline.pkl"
CONFIG_PATH = ARTIFACT_DIR / "model_config.pkl"
RAW_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


@st.cache_resource
def load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load inference artifacts from paths relative to this file."""
    missing = [str(path) for path in (PIPELINE_PATH, CONFIG_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(missing)
            + ". Run the training notebook from the repository with CHURN_DATA_PATH set."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    config = joblib.load(CONFIG_PATH)
    if "optimal_threshold" not in config:
        raise ValueError(f"{CONFIG_PATH} does not contain 'optimal_threshold'.")
    threshold = float(config["optimal_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Configured optimal_threshold must be between 0 and 1.")
    return pipeline, config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the same deterministic features used by the training notebook."""
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    df["ChargeIncreaseRatio"] = df["MonthlyCharges"] / np.maximum(
        df["AvgChargePerMonth"], 1
    )
    df["IsNewCustomer"] = (df["tenure"] < 12).astype(int)
    df["IsMidTerm"] = ((df["tenure"] >= 12) & (df["tenure"] < 36)).astype(int)
    df["IsLoyalCustomer"] = (df["tenure"] >= 36).astype(int)
    df["TenureSquared"] = df["tenure"] ** 2

    service_values = df[SERVICE_COLUMNS].replace(
        {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    )
    df["TotalServices"] = service_values.apply(pd.to_numeric, errors="coerce").sum(axis=1)
    df["ServiceAdoptionRate"] = df["TotalServices"] / len(SERVICE_COLUMNS)
    df["HighMonthlyCharge"] = (df["MonthlyCharges"] > 70).astype(int)
    df["NoSecurityBundle"] = (
        (df["OnlineSecurity"] == "No") & (df["DeviceProtection"] == "No")
    ).astype(int)
    df["MonthToMonthContract"] = (df["Contract"] == "Month-to-month").astype(int)
    df["ElectronicCheckPay"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["ChurnRiskScore"] = (
        df["IsNewCustomer"] * 3
        + df["MonthToMonthContract"] * 3
        + df["ElectronicCheckPay"] * 2
        + df["NoSecurityBundle"]
        + df["HighMonthlyCharge"]
    )
    return df


def predict_customer(customer_data: dict[str, Any]) -> dict[str, Any]:
    """Return probability, thresholded prediction, and a human-readable risk band."""
    pipeline, config = load_artifacts()
    customer_df = engineer_features(pd.DataFrame([customer_data]))
    probabilities = pipeline.predict_proba(customer_df)
    classes = list(getattr(pipeline, "classes_", [0, 1]))
    try:
        churn_index = classes.index(1)
    except ValueError as exc:
        raise ValueError("Loaded pipeline does not expose a positive churn class (1).") from exc
    probability = float(probabilities[0, churn_index])
    threshold = float(config["optimal_threshold"])
    risk = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW"
    return {
        "churn_prediction": bool(probability >= threshold),
        "churn_probability": round(probability, 4),
        "risk_level": risk,
        "recommendation": {
            "HIGH": "Contact immediately",
            "MEDIUM": "Monitor closely",
            "LOW": "No action needed",
        }[risk],
    }


def main() -> None:
    st.set_page_config(page_title="Telecom Churn Prediction", layout="centered")
    st.title("Telecom Customer Churn Prediction")
    st.write("Predict whether a telecom customer is likely to churn.")

    with st.form("customer_profile"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)
        submitted = st.form_submit_button("Predict Churn")

    if not submitted:
        return
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    try:
        result = predict_customer(customer)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        return

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {result['churn_probability']:.2%}")
    (st.error if result["churn_prediction"] else st.success)(
        "Customer Likely to Churn" if result["churn_prediction"] else "Customer Likely to Stay"
    )
    risk_message = {
        "HIGH": st.warning,
        "MEDIUM": st.info,
        "LOW": st.success,
    }[result["risk_level"]]
    risk_message(f"{result['risk_level']} Risk Customer — {result['recommendation']}")


if __name__ == "__main__":
    main()
