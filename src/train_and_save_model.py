"""
Train and save the final Random Forest model.

This script loads the processed train and test datasets, trains the tuned
Random Forest model selected during notebook experimentation, saves the model
artifact, saves the training feature names, stores basic model metadata, and
checks that the saved model can be reloaded consistently.

Inputs:
- data/processed/X_train_prepared.csv
- data/processed/X_test_prepared.csv
- data/processed/y_train.csv
- data/processed/y_test.csv

Outputs:
- artifacts/model/random_forest_model.joblib
- artifacts/model/feature_names.json
- artifacts/model/model_metadata.json
"""

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Best parameters found in notebook 04
RANDOM_STATE = 42

FINAL_RF_PARAMS = {
    "n_estimators": 200,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": None,
    "max_depth": 20,
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Artifact file names
MODEL_FILE = "random_forest_model.joblib"
FEATURE_NAMES_FILE = "feature_names.json"
MODEL_METADATA_FILE = "model_metadata.json"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_training_files(processed_dir: Path):
    x_train_path = processed_dir / "X_train_prepared.csv"
    x_test_path = processed_dir / "X_test_prepared.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"

    required_files = [x_train_path, x_test_path, y_train_path, y_test_path]
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "The following required processed files are missing:\n"
            + "\n".join(missing_files)
        )

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    return X_train, X_test, y_train, y_test


def build_final_model() -> RandomForestClassifier:
    return RandomForestClassifier(**FINAL_RF_PARAMS)


def train_final_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    model.fit(X_train, y_train)
    return model


def build_model_metadata(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    train_class_distribution = y_train.value_counts().to_dict()
    test_class_distribution = y_test.value_counts().to_dict()

    metadata = {
        "model_name": "RandomForestClassifier",
        "model_role": "Final binary classification model for machine failure prediction",
        "target_name": "Machine failure",
        "random_state": RANDOM_STATE,
        "threshold": 0.50,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "train_class_distribution": {
            str(k): int(v) for k, v in train_class_distribution.items()
        },
        "test_class_distribution": {
            str(k): int(v) for k, v in test_class_distribution.items()
        },
        "hyperparameters": FINAL_RF_PARAMS,
    }

    return metadata


def save_model_files(
    model: RandomForestClassifier,
    feature_names: list[str],
    metadata: dict,
    model_output_dir: Path,
):
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_output_dir / MODEL_FILE
    feature_names_path = model_output_dir / FEATURE_NAMES_FILE
    metadata_path = model_output_dir / MODEL_METADATA_FILE

    joblib.dump(model, model_path)

    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=4)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return model_path, feature_names_path, metadata_path


def reload_saved_model(model_path: Path) -> RandomForestClassifier:
    return joblib.load(model_path)


def check_model_reload(
    trained_model: RandomForestClassifier,
    reloaded_model: RandomForestClassifier,
    X_test: pd.DataFrame,
    sample_size: int = 10,
):
    X_sample = X_test.head(sample_size)

    trained_predictions = trained_model.predict(X_sample)
    reloaded_predictions = reloaded_model.predict(X_sample)

    trained_probabilities = trained_model.predict_proba(X_sample)
    reloaded_probabilities = reloaded_model.predict_proba(X_sample)

    predictions_match = (trained_predictions == reloaded_predictions).all()
    probabilities_match = (trained_probabilities == reloaded_probabilities).all()

    return predictions_match, probabilities_match


def main():
    project_root = get_project_root()

    processed_dir = project_root / "data" / "processed"
    model_output_dir = project_root / "artifacts" / "model"

    print("Loading processed datasets...")
    X_train, X_test, y_train, y_test = load_training_files(processed_dir)

    print("Building final tuned Random Forest model...")
    model = build_final_model()

    print("Training model...")
    model = train_final_model(model, X_train, y_train)

    print("Preparing feature names and metadata...")
    feature_names = X_train.columns.tolist()
    metadata = build_model_metadata(X_train, X_test, y_train, y_test)

    print("Saving model files...")
    model_path, feature_names_path, metadata_path = save_model_files(
        model=model,
        feature_names=feature_names,
        metadata=metadata,
        model_output_dir=model_output_dir,
    )

    print("Reloading saved model...")
    reloaded_model = reload_saved_model(model_path)

    print("Checking saved model consistency...")
    predictions_match, probabilities_match = check_model_reload(
        trained_model=model,
        reloaded_model=reloaded_model,
        X_test=X_test,
        sample_size=10,
    )

    print("\nDone.")
    print(f"Model saved to: {model_path}")
    print(f"Feature names saved to: {feature_names_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Predictions match after reload: {predictions_match}")
    print(f"Probabilities match after reload: {probabilities_match}")

    if not predictions_match or not probabilities_match:
        raise ValueError(
            "Verification failed: the loaded model does not match the original model."
        )


if __name__ == "__main__":
    main()