import unittest

import pandas as pd

from features import engineer_features


class EngineerFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.customer = pd.DataFrame([{
            'TotalCharges': 600.0,
            'tenure': 12,
            'MonthlyCharges': 75.0,
            'PhoneService': 'Yes',
            'MultipleLines': 'No phone service',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No internet service',
            'DeviceProtection': 'No',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaymentMethod': 'Electronic check',
        }])

    def test_derived_values_and_service_categories(self):
        result = engineer_features(self.customer)
        row = result.iloc[0]

        self.assertEqual(row['AvgChargePerMonth'], 600.0 / 13)
        self.assertEqual(row['CLV'], 900.0)
        self.assertEqual(row['TotalServices'], 3)
        self.assertEqual(row['ServiceAdoptionRate'], 3 / 8)
        self.assertEqual(row['IsMidTerm'], 1)
        self.assertEqual(row['MonthToMonthContract'], 1)
        self.assertEqual(row['ElectronicCheckPay'], 1)
        self.assertEqual(row['ChurnRiskScore'], 7)

    def test_does_not_mutate_input_dataframe(self):
        before = self.customer.copy(deep=True)
        engineer_features(self.customer)
        pd.testing.assert_frame_equal(self.customer, before)


if __name__ == '__main__':
    unittest.main()
