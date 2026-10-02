"""Feature engineering shared by the churn application and its tests."""

import numpy as np


def engineer_features(df):
    """Return a copy of customer inputs with model-derived features added."""
    df = df.copy()

    df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['CLV'] = df['MonthlyCharges'] * df['tenure']
    df['ChargeIncreaseRatio'] = df['MonthlyCharges'] / np.maximum(
        df['AvgChargePerMonth'], 1
    )
    df['IsNewCustomer'] = (df['tenure'] < 12).astype(int)
    df['IsMidTerm'] = ((df['tenure'] >= 12) & (df['tenure'] < 36)).astype(int)
    df['IsLoyalCustomer'] = (df['tenure'] >= 36).astype(int)
    df['TenureSquared'] = df['tenure'] ** 2

    service_cols = [
        'PhoneService',
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
    ]
    temp = df[service_cols].replace({
        'Yes': 1,
        'No': 0,
        'No internet service': 0,
        'No phone service': 0,
    })
    df['TotalServices'] = temp.sum(axis=1)
    df['ServiceAdoptionRate'] = df['TotalServices'] / len(service_cols)
    df['HighMonthlyCharge'] = (df['MonthlyCharges'] > 70).astype(int)
    df['NoSecurityBundle'] = (
        (df['OnlineSecurity'] == 'No') & (df['DeviceProtection'] == 'No')
    ).astype(int)
    df['MonthToMonthContract'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['ElectronicCheckPay'] = (
        df['PaymentMethod'] == 'Electronic check'
    ).astype(int)
    df['ChurnRiskScore'] = (
        df['IsNewCustomer'] * 3
        + df['MonthToMonthContract'] * 3
        + df['ElectronicCheckPay'] * 2
        + df['NoSecurityBundle']
        + df['HighMonthlyCharge']
    )
    return df
