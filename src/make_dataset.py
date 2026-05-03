"""
Create processed train and test datasets for binary machine failure prediction.

This script loads the raw AI4I 2020 dataset, adds the engineered features
defined during the notebook analysis, removes leakage-prone and identifier
columns, applies preprocessing, and saves the processed train/test datasets.

Inputs:
- data/raw/ai4i2020.csv

Outputs:
- data/processed/X_train_prepared.csv
- data/processed/X_test_prepared.csv
- data/processed/y_train.csv
- data/processed/y_test.csv
"""

from pathlib import Path
import pandas as pd
import sqlite3

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Machine failure"

LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]

DATABASE_FILENAME = "predictive_maintenance.db"
TABLE_NAME = "ai4i_raw"


def get_project_root() -> Path:
    """
    Assumes this script lives in src/ and returns the project root.
    """
    return Path(__file__).resolve().parents[1]


def load_raw_data(database_path: Path, table_name: str) -> pd.DataFrame:
    """
    Load the raw dataset from a SQLite database table.
    """
    if not database_path.exists():
        raise FileNotFoundError(
            f"Database not found: {database_path}\n"
            "Run src/create_database.py before running this script."
        )

    with sqlite3.connect(database_path) as connection:
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {table_name}
            """,
            connection,
        )

    return df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the engineered features established in notebook 02.
    """
    df = df.copy()

    df["Temperature difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    df["Tool wear x Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    return df


def define_features_and_target(df: pd.DataFrame):
    """
    Define feature matrix X and target y.
    Drops leakage-prone columns and identifier columns.
    """
    required_columns = (
        [TARGET_COLUMN]
        + LEAKAGE_COLUMNS
        + IDENTIFIER_COLUMNS
        + ["Process temperature [K]", "Air temperature [K]", "Tool wear [min]", "Torque [Nm]"]
    )

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            "The following expected columns are missing from the dataset: "
            + ", ".join(missing_columns)
        )

    columns_to_drop = LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS + [TARGET_COLUMN]

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build the preprocessing pipeline:
    - numeric: median imputation + standard scaling
    - categorical: most-frequent imputation + one-hot encoding
    """
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def transform_and_create_dataframes(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit the preprocessor on training data only, transform both splits,
    and return processed DataFrames with feature names.
    """
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_prepared_df = pd.DataFrame(
        X_train_prepared,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_prepared_df = pd.DataFrame(
        X_test_prepared,
        columns=feature_names,
        index=X_test.index,
    )

    return X_train_prepared_df, X_test_prepared_df


def save_processed_data(
    X_train_prepared: pd.DataFrame,
    X_test_prepared: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    processed_dir: Path,
):
    """
    Save processed train/test datasets to data/processed/.

    If files with the same names already exist, they will be overwritten.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train_prepared.to_csv(processed_dir / "X_train_prepared.csv", index=False)
    X_test_prepared.to_csv(processed_dir / "X_test_prepared.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)


def main():
    project_root = get_project_root()

    database_path = project_root / "data" / "database" / DATABASE_FILENAME
    processed_dir = project_root / "data" / "processed"

    print("Loading raw dataset from SQLite database...")
    df = load_raw_data(database_path, TABLE_NAME)

    print("Adding engineered features...")
    df = add_engineered_features(df)

    print("Defining features and target...")
    X, y = define_features_and_target(df)

    print("Creating stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessor(X_train)

    print("Transforming train and test data...")
    X_train_prepared, X_test_prepared = transform_and_create_dataframes(
        preprocessor=preprocessor,
        X_train=X_train,
        X_test=X_test,
    )

    print("Saving processed outputs...")
    save_processed_data(
        X_train_prepared=X_train_prepared,
        X_test_prepared=X_test_prepared,
        y_train=y_train,
        y_test=y_test,
        processed_dir=processed_dir,
    )

    print("\nDone.")
    print(f"Saved files to: {processed_dir}")
    print("Existing files with the same names were overwritten if they were already present.")
    print(f"X_train_prepared shape: {X_train_prepared.shape}")
    print(f"X_test_prepared shape: {X_test_prepared.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")


if __name__ == "__main__":
    main()