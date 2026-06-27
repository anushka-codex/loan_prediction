"""
app.py
--------
Streamlit demo app for the Loan Default Prediction model.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io) by
pointing it at this repo and this file.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import engineer_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "loan_default_model.joblib")

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="\U0001F4B0",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def gauge_chart(probability: float):
    """Render a simple risk gauge using matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 0.6))
    ax.barh([0], [1], color="#E8E8E8", height=1)
    color = "#2E7D32" if probability < 0.3 else "#F9A825" if probability < 0.6 else "#C62828"
    ax.barh([0], [probability], color=color, height=1)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


def main():
    st.title("\U0001F4B0 Loan Default Risk Predictor")
    st.markdown(
        "Estimate the probability that a loan applicant will default, using a "
        "**Gradient Boosting** model trained on 15,000 historical loan records "
        "(ROC-AUC ≈ 0.99 on held-out test data). "
        "[View the source project on GitHub](https://github.com/<your-username>/loan-default-prediction)."
    )

    st.divider()

    with st.form("applicant_form"):
        st.subheader("Applicant Details")

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 75, 35)
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
            loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=20000, step=500)
            credit_score = st.slider("Credit Score", 300, 850, 680)
            years_employed = st.slider("Years Employed", 0, 40, 5)
        with col2:
            num_credit_cards = st.slider("Number of Credit Cards", 0, 15, 3)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.selectbox("Education", ["High School", "Graduate", "Masters"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self Employed"])

        submitted = st.form_submit_button("Predict Default Risk", use_container_width=True)

    if submitted:
        applicant = pd.DataFrame(
            [
                {
                    "age": age,
                    "annual_income": annual_income,
                    "loan_amount": loan_amount,
                    "credit_score": credit_score,
                    "years_employed": years_employed,
                    "num_credit_cards": num_credit_cards,
                    "gender": gender,
                    "education": education,
                    "marital_status": marital_status,
                    "employment_type": employment_type,
                }
            ]
        )

        model = load_model()
        engineered = engineer_features(applicant)
        probability = float(model.predict_proba(engineered)[0, 1])
        prediction = int(model.predict(engineered)[0])

        st.divider()
        st.subheader("Result")

        if prediction == 1:
            st.error(f"**High default risk** — estimated probability: {probability:.1%}")
        else:
            st.success(f"**Low default risk** — estimated probability: {probability:.1%}")

        st.pyplot(gauge_chart(probability), use_container_width=True)

        debt_to_income = loan_amount / annual_income if annual_income > 0 else np.nan
        st.caption(
            f"Debt-to-income ratio: {debt_to_income:.2f} — the single strongest "
            "driver of this model's predictions. Lower is generally safer."
        )

        with st.expander("See full input + model output"):
            result = applicant.copy()
            result["default_probability"] = round(probability, 4)
            result["default_prediction"] = prediction
            st.dataframe(result, use_container_width=True)

    st.divider()
    st.caption(
        "This is a portfolio/demo project. Predictions are based on a model trained on a "
        "sample dataset and should not be used for real lending decisions."
    )


if __name__ == "__main__":
    main()
