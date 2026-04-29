from pathlib import Path
import json

import joblib
import pandas as pd


def load_artifacts(model_dir: Path):
    """
    Load the trained model and the saved feature names.
    """
    model_path = model_dir / "random_forest_model.joblib"
    feature_names_path = model_dir / "feature_names.json"

    model = joblib.load(model_path)

    with open(feature_names_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    return model, feature_names


def load_input_data(data_path: Path, n_samples: int = 10):
    """
    Load prepared input data for inference.
    """
    df = pd.read_csv(data_path)
    return df.head(n_samples)


def run_inference(model, X: pd.DataFrame):
    """
    Generate class predictions and failure probabilities.
    """
    predictions = model.predict(X)
    failure_probabilities = model.predict_proba(X)[:, 1]

    results = X.copy()
    results["prediction"] = predictions
    results["failure_probability"] = failure_probabilities

    return results


def main():
    """
    Run inference using the saved Random Forest model.
    """
    base_dir = Path(__file__).resolve().parent.parent

    model_dir = base_dir / "artifacts" / "model"
    input_data_path = base_dir / "data" / "processed" / "X_test_prepared.csv"

    print("Loading saved model and feature names...")
    model, feature_names = load_artifacts(model_dir)

    print("Loading input data...")
    X_input = load_input_data(input_data_path, n_samples=10)

    print("Aligning input columns with saved feature names...")
    X_input = X_input[feature_names]

    print("Running inference...")
    results = run_inference(model, X_input)

    print("\n=== Prediction Results ===")
    print(results)


if __name__ == "__main__":
    main()