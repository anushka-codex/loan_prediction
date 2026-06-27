"""
eda_visualizations.py
------------------------
Generates exploratory data analysis plots used in the README and
notebook: class balance, feature distributions, correlation heatmap,
and feature importance from the final trained model.

Run from the project root:
    python src/eda_visualizations.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_data, engineer_features

sns.set_theme(style="whitegrid", palette="muted")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_loan_default_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "loan_default_model.joblib")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["loan_default"].value_counts().sort_index()
    labels = ["No Default", "Default"]
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"])
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{int(bar.get_height()):,}",
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Loan Default Class Distribution")
    ax.set_ylabel("Number of Borrowers")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "class_balance.png"), dpi=150)
    plt.close(fig)


def plot_numeric_distributions(df):
    numeric_cols = ["age", "annual_income", "loan_amount", "credit_score", "years_employed"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(
            data=df, x=col, hue="loan_default", kde=True, ax=axes[i],
            palette=["#4C72B0", "#DD8452"], element="step", stat="density", common_norm=False
        )
        axes[i].set_title(col.replace("_", " ").title())
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature Distributions by Default Status", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "numeric_distributions.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_categorical_default_rates(df):
    cat_cols = ["gender", "education", "marital_status", "employment_type"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for i, col in enumerate(cat_cols):
        rates = df.groupby(col)["loan_default"].mean().sort_values(ascending=False)
        sns.barplot(x=rates.index, y=rates.values, ax=axes[i], color="#55A868")
        axes[i].set_title(f"Default Rate by {col.replace('_', ' ').title()}")
        axes[i].set_ylabel("Default Rate")
        axes[i].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "categorical_default_rates.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found, skipping feature importance plot.")
        return
    pipe = joblib.load(MODEL_PATH)
    preprocessor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]

    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
    all_features = list(num_features) + cat_features

    importances = classifier.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": all_features, "importance": importances}
    ).sort_values("importance", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color="#4C72B0")
    ax.set_title("Top Feature Importances (Tuned Gradient Boosting)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    df = engineer_features(df)

    print("Generating class balance plot...")
    plot_class_balance(df)

    print("Generating numeric distribution plots...")
    plot_numeric_distributions(df)

    print("Generating correlation heatmap...")
    plot_correlation_heatmap(df)

    print("Generating categorical default rate plots...")
    plot_categorical_default_rates(df)

    print("Generating feature importance plot...")
    plot_feature_importance()

    print(f"All plots saved to {IMG_DIR}")


if __name__ == "__main__":
    main()
