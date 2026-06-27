"""
train_model.py
----------------
Trains and compares several classification models for loan default
prediction, handles class imbalance, performs hyperparameter tuning
on the best candidate, and saves the final pipeline to disk.

Run from the project root:
    python src/train_model.py
"""

import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from preprocessing import (
    load_data,
    engineer_features,
    get_feature_lists,
    build_preprocessor,
    train_test_split_data,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_loan_default_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    return metrics


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    df = engineer_features(df)
    numeric, categorical = get_feature_lists()

    X_train, X_test, y_train, y_test = train_test_split_data(df)
    preprocessor = build_preprocessor(numeric, categorical)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    fitted_models = {}

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test, name)
        results.append(metrics)
        fitted_models[name] = pipe

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\nModel comparison (sorted by ROC-AUC):")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    print(f"\nBest baseline model: {best_name}")

    # Hyperparameter tuning for Gradient Boosting (the strongest baseline here)
    print("\nRunning GridSearchCV on Gradient Boosting...")
    gb_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    )
    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(gb_pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

    final_model = grid.best_estimator_
    final_metrics = evaluate(final_model, X_test, y_test, "Tuned Gradient Boosting")
    print("\nFinal tuned model performance on test set:")
    for k, v in final_metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "loan_default_model.joblib")
    joblib.dump(final_model, model_path)
    print(f"\nSaved final model to {model_path}")

    results_df.to_csv(os.path.join(MODEL_DIR, "baseline_model_comparison.csv"), index=False)


if __name__ == "__main__":
    main()
