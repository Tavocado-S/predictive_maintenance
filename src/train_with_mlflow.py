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
- MLflow run with logged Random Forest model
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
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

THRESHOLD = 0.5
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


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

    return metrics, cm


def save_confusion_matrix_plot(cm, output_path: Path):
    """
    Save the confusion matrix plot as an image file.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Tuned Random Forest")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


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
    metrics: dict,
    cm,
    threshold: float,
    confusion_matrix_path: Path,
):
    """
    Log model parameters, metrics, artifacts, and model to MLflow.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(RANDOM_FOREST_PARAMS)
        mlflow.log_param("decision_threshold", threshold)

        mlflow.log_metrics(metrics)

        save_confusion_matrix_plot(
            cm=cm,
            output_path=confusion_matrix_path,
        )

        mlflow.log_artifact(str(confusion_matrix_path))

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

    confusion_matrix_path = project_root / CONFUSION_MATRIX_FILENAME

    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    print("Training tuned Random Forest model...")
    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    print("Evaluating model...")
    metrics, cm = evaluate_model(
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
        metrics=metrics,
        cm=cm,
        threshold=THRESHOLD,
        confusion_matrix_path=confusion_matrix_path,
    )

    if confusion_matrix_path.exists():
        confusion_matrix_path.unlink()

    print("\nDone.")


if __name__ == "__main__":
    main()