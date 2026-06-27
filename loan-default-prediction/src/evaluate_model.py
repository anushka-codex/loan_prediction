"""
evaluate_model.py
--------------------
Loads the saved model and test split, then generates a confusion
matrix and ROC curve for the README/portfolio.

Run from the project root (after train_model.py):
    python src/evaluate_model.py
"""

import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay, classification_report

sys.path.append(os.path.dirname(__file__))
from preprocessing import load_data, engineer_features, train_test_split_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_loan_default_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "loan_default_model.joblib")
IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_data(DATA_PATH)
    df = engineer_features(df)
    _, X_test, _, y_test = train_test_split_data(df)

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    print("Classification report:\n")
    print(classification_report(y_test, preds, target_names=["No Default", "Default"]))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Tuned Gradient Boosting")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color="#DD8452")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_title("ROC Curve - Tuned Gradient Boosting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved confusion matrix and ROC curve to {IMG_DIR}")


if __name__ == "__main__":
    main()
