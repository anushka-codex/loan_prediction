"""
predict.py
------------
Standalone inference script: loads the trained pipeline and scores
new loan applicants.

Example usage:
    python src/predict.py
"""

import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from preprocessing import engineer_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "loan_default_model.joblib")


def predict_default_risk(applicants: pd.DataFrame) -> pd.DataFrame:
    """
    applicants: DataFrame with raw columns:
        age, annual_income, loan_amount, credit_score, years_employed,
        num_credit_cards, gender, education, marital_status, employment_type
    Returns the input DataFrame with two added columns:
        default_probability, default_prediction
    """
    model = joblib.load(MODEL_PATH)
    engineered = engineer_features(applicants)
    probs = model.predict_proba(engineered)[:, 1]
    preds = model.predict(engineered)

    result = applicants.copy()
    result["default_probability"] = probs.round(4)
    result["default_prediction"] = preds
    return result


if __name__ == "__main__":
    sample_applicants = pd.DataFrame(
        [
            {
                "age": 29,
                "annual_income": 42000,
                "loan_amount": 38000,
                "credit_score": 590,
                "years_employed": 2,
                "num_credit_cards": 7,
                "gender": "Male",
                "education": "High School",
                "marital_status": "Single",
                "employment_type": "Self Employed",
            },
            {
                "age": 45,
                "annual_income": 95000,
                "loan_amount": 15000,
                "credit_score": 780,
                "years_employed": 18,
                "num_credit_cards": 2,
                "gender": "Female",
                "education": "Masters",
                "marital_status": "Married",
                "employment_type": "Salaried",
            },
        ]
    )

    scored = predict_default_risk(sample_applicants)
    print(scored.to_string(index=False))
