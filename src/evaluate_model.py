"""
Evaluate the saved Random Forest model.

This script loads the saved model artifact and evaluates it on the processed
test set. It confirms that the trained model can be reused outside the
notebooks in a simple, reproducible script-based workflow.

Inputs:
- artifacts/model/random_forest_model.joblib
- artifacts/model/feature_names.json
- data/processed/X_test_prepared.csv
- data/processed/y_test.csv

Outputs:
- Evaluation metrics printed to the console
- Confusion matrix printed to the console
- Confusion matrix plot displayed
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_FILENAME = "random_forest_model.joblib"
FEATURE_NAMES_FILENAME = "feature_names.json"

X_TEST_FILENAME = "X_test_prepared.csv"
Y_TEST_FILENAME = "y_test.csv"

THRESHOLD = 0.5
SHOW_CONFUSION_MATRIX_PLOT = True


def get_project_root() -> Path:
    """
    Assumes this script lives in src/ and returns the project root.
    """
    return Path(__file__).resolve().parents[1]


def load_model(model_path: Path):
    """
    Load the saved Random Forest model artifact.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run src/train_and_save_model.py before evaluating the model."
        )

    return joblib.load(model_path)


def load_feature_names(feature_names_path: Path) -> list[str]:
    """
    Load the feature names used during model training.
    """
    if not feature_names_path.exists():
        raise FileNotFoundError(
            f"Feature names file not found: {feature_names_path}\n"
            "Run src/train_and_save_model.py before evaluating the model."
        )

    with feature_names_path.open("r", encoding="utf-8") as file:
        feature_names = json.load(file)

    return feature_names


def load_test_data(x_test_path: Path, y_test_path: Path):
    """
    Load the processed test features and target values.
    """
    if not x_test_path.exists():
        raise FileNotFoundError(
            f"Processed test feature file not found: {x_test_path}\n"
            "Run src/make_dataset.py before evaluating the model."
        )

    if not y_test_path.exists():
        raise FileNotFoundError(
            f"Processed test target file not found: {y_test_path}\n"
            "Run src/make_dataset.py before evaluating the model."
        )

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    return X_test, y_test


def align_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Reorder the test data columns to match the saved training feature order.
    """
    missing_features = set(feature_names) - set(X.columns)
    extra_features = set(X.columns) - set(feature_names)

    if missing_features:
        raise ValueError(
            "The following expected features are missing from the test data: "
            + ", ".join(sorted(missing_features))
        )

    if extra_features:
        print(
            "Warning: The following extra features were found and will be ignored: "
            + ", ".join(sorted(extra_features))
        )

    X_aligned = X[feature_names]

    return X_aligned


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float):
    """
    Generate predictions and print evaluation metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)

    print("\nSaved Model Evaluation")
    print("-" * 40)
    print(f"Decision threshold: {threshold}")
    print("-" * 40)
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1-score:           {f1:.4f}")
    print(f"ROC-AUC:            {roc_auc:.4f}")
    print(f"Average Precision:  {avg_precision:.4f}")

    print("\nConfusion Matrix")
    print("-" * 40)
    print(cm)

    if SHOW_CONFUSION_MATRIX_PLOT:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix - Saved Random Forest Model")
        plt.show()


def main():
    project_root = get_project_root()

    model_dir = project_root / "artifacts" / "model"
    processed_dir = project_root / "data" / "processed"

    model_path = model_dir / MODEL_FILENAME
    feature_names_path = model_dir / FEATURE_NAMES_FILENAME

    x_test_path = processed_dir / X_TEST_FILENAME
    y_test_path = processed_dir / Y_TEST_FILENAME

    print("Loading saved model...")
    model = load_model(model_path)

    print("Loading feature names...")
    feature_names = load_feature_names(feature_names_path)

    print("Loading test data...")
    X_test, y_test = load_test_data(
        x_test_path=x_test_path,
        y_test_path=y_test_path,
    )

    print("Aligning test features...")
    X_test_aligned = align_features(
        X=X_test,
        feature_names=feature_names,
    )

    print("Evaluating saved model...")
    evaluate_model(
        model=model,
        X_test=X_test_aligned,
        y_test=y_test,
        threshold=THRESHOLD,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()