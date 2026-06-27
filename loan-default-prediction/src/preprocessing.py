"""
preprocessing.py
-----------------
Utility functions for loading and preparing the loan default dataset
for modeling: encoding categoricals, scaling numerics, and splitting
into train/test sets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "years_employed",
    "num_credit_cards",
]

CATEGORICAL_FEATURES = [
    "gender",
    "education",
    "marital_status",
    "employment_type",
]

TARGET = "loan_default"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features that tend to help default-risk models."""
    df = df.copy()
    df["debt_to_income"] = df["loan_amount"] / df["annual_income"]
    df["income_per_credit_card"] = df["annual_income"] / (df["num_credit_cards"] + 1)
    df["employment_stability"] = df["years_employed"] / (df["age"] - 17).clip(lower=1)
    return df


def get_feature_lists(extra_engineered: bool = True):
    """Return updated numeric/categorical feature lists if engineered features are used."""
    numeric = NUMERIC_FEATURES.copy()
    if extra_engineered:
        numeric += ["debt_to_income", "income_per_credit_card", "employment_stability"]
    return numeric, CATEGORICAL_FEATURES


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric and one-hot encodes categorical features."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split engineered DataFrame into train/test X and y."""
    numeric, categorical = get_feature_lists()
    features = numeric + categorical
    X = df[features]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
