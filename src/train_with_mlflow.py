"""
Train the tuned Random Forest model with MLflow experiment tracking.

This script loads the processed train and test datasets, trains the selected
Random Forest model, evaluates it on the processed test set, and logs the
experiment results with MLflow.

Inputs:
- data/processed/X_train_prepared.csv
- data/processed/X_test_prepared.csv
- data/processed/y_train.csv
- data/processed/y_test.csv

Outputs:
- MLflow run with logged parameters
- MLflow run with logged evaluation metrics
- MLflow run with logged confusion matrix plot
- MLflow run with logged confusion matrix data
- MLflow run with logged classification report
- MLflow run with logged feature names
- MLflow run with logged model metadata
- MLflow run with logged Random Forest model
"""

from pathlib import Path
import json
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


X_TRAIN_FILENAME = "X_train_prepared.csv"
X_TEST_FILENAME = "X_test_prepared.csv"
Y_TRAIN_FILENAME = "y_train.csv"
Y_TEST_FILENAME = "y_test.csv"

EXPERIMENT_NAME = "predictive-maintenance-random-forest"
RUN_NAME = "tuned-random-forest"

TARGET_NAME = "Machine failure"
MODEL_NAME = "RandomForestClassifier"
MODEL_ROLE = "Final binary classification model for machine failure prediction"

THRESHOLD = 0.5

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": None,
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def get_project_root() -> Path:
    """
    Assumes this script lives in src/ and returns the project root.
    """
    return Path(__file__).resolve().parents[1]


def load_processed_data(processed_dir: Path):
    """
    Load the processed train and test datasets.
    """
    x_train_path = processed_dir / X_TRAIN_FILENAME
    x_test_path = processed_dir / X_TEST_FILENAME
    y_train_path = processed_dir / Y_TRAIN_FILENAME
    y_test_path = processed_dir / Y_TEST_FILENAME

    required_files = [
        x_train_path,
        x_test_path,
        y_train_path,
        y_test_path,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed data file not found: {file_path}\n"
                "Run src/make_dataset.py before training with MLflow."
            )

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train the tuned Random Forest model.
    """
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
):
    """
    Generate predictions and calculate evaluation metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["No failure", "Machine failure"],
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["No failure", "Machine failure"],
        output_dict=True,
    )

    return metrics, cm, report_text, report_dict


def save_confusion_matrix_plot(cm, output_path: Path):
    """
    Save the confusion matrix plot as an image file.
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No failure", "Machine failure"],
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Tuned Random Forest")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_data(cm, output_path: Path):
    """
    Save the confusion matrix values as a CSV artifact.
    """
    cm_df = pd.DataFrame(
        cm,
        index=["actual_no_failure", "actual_machine_failure"],
        columns=["predicted_no_failure", "predicted_machine_failure"],
    )

    cm_df.to_csv(output_path)


def save_classification_report(
    report_text: str,
    report_dict: dict,
    txt_output_path: Path,
    json_output_path: Path,
):
    """
    Save the classification report as TXT and JSON artifacts.
    """
    with open(txt_output_path, "w", encoding="utf-8") as file:
        file.write(report_text)

    with open(json_output_path, "w", encoding="utf-8") as file:
        json.dump(report_dict, file, indent=4)


def save_feature_names(feature_names: list[str], output_path: Path):
    """
    Save the model input feature names as a JSON artifact.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(feature_names, file, indent=4)


def save_model_metadata(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: float,
    output_path: Path,
):
    """
    Save model training metadata as a JSON artifact.
    """
    metadata = {
        "model_name": MODEL_NAME,
        "model_role": MODEL_ROLE,
        "target_name": TARGET_NAME,
        "random_state": RANDOM_FOREST_PARAMS["random_state"],
        "threshold": threshold,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "train_class_distribution": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum()),
        },
        "test_class_distribution": {
            "0": int((y_test == 0).sum()),
            "1": int((y_test == 1).sum()),
        },
        "excluded_columns": EXCLUDED_COLUMNS,
        "hyperparameters": RANDOM_FOREST_PARAMS,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def print_evaluation_results(metrics: dict, cm, threshold: float):
    """
    Print evaluation metrics and confusion matrix to the console.
    """
    print("\nMLflow Training Evaluation")
    print("-" * 40)
    print(f"Decision threshold: {threshold}")
    print("-" * 40)
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1-score:           {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"Average Precision:  {metrics['average_precision']:.4f}")

    print("\nConfusion Matrix")
    print("-" * 40)
    print(cm)


def log_experiment(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metrics: dict,
    cm,
    report_text: str,
    report_dict: dict,
    threshold: float,
):
    """
    Log model parameters, metrics, artifacts, and model to MLflow.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        confusion_matrix_plot_path = temp_dir_path / "confusion_matrix.png"
        confusion_matrix_csv_path = temp_dir_path / "confusion_matrix.csv"
        classification_report_txt_path = temp_dir_path / "classification_report.txt"
        classification_report_json_path = temp_dir_path / "classification_report.json"
        feature_names_path = temp_dir_path / "feature_names.json"
        model_metadata_path = temp_dir_path / "model_metadata.json"

        save_confusion_matrix_plot(
            cm=cm,
            output_path=confusion_matrix_plot_path,
        )

        save_confusion_matrix_data(
            cm=cm,
            output_path=confusion_matrix_csv_path,
        )

        save_classification_report(
            report_text=report_text,
            report_dict=report_dict,
            txt_output_path=classification_report_txt_path,
            json_output_path=classification_report_json_path,
        )

        save_feature_names(
            feature_names=list(X_train.columns),
            output_path=feature_names_path,
        )

        save_model_metadata(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            threshold=threshold,
            output_path=model_metadata_path,
        )

        with mlflow.start_run(run_name=RUN_NAME) as run:
            mlflow.log_params(RANDOM_FOREST_PARAMS)
            mlflow.log_param("decision_threshold", threshold)
            mlflow.log_param("target_name", TARGET_NAME)
            mlflow.log_param("model_name", MODEL_NAME)

            mlflow.log_metrics(metrics)

            mlflow.log_artifact(
                str(confusion_matrix_plot_path),
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                str(confusion_matrix_csv_path),
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                str(classification_report_txt_path),
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                str(classification_report_json_path),
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                str(feature_names_path),
                artifact_path="metadata",
            )
            mlflow.log_artifact(
                str(model_metadata_path),
                artifact_path="metadata",
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="random_forest_model",
            )

            print("\nMLflow Run Information")
            print("-" * 40)
            print(f"Experiment name: {EXPERIMENT_NAME}")
            print(f"Run name:        {RUN_NAME}")
            print(f"Run ID:          {run.info.run_id}")


def main():
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"

    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    print("Training tuned Random Forest model...")
    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    print("Evaluating model...")
    metrics, cm, report_text, report_dict = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=THRESHOLD,
    )

    print_evaluation_results(
        metrics=metrics,
        cm=cm,
        threshold=THRESHOLD,
    )

    print("Logging experiment to MLflow...")
    log_experiment(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metrics=metrics,
        cm=cm,
        report_text=report_text,
        report_dict=report_dict,
        threshold=THRESHOLD,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()