from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "predictive_maintenance.db"

TABLE_NAME = "ai4i_raw"


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the raw AI4I 2020 dataset from a CSV file.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {csv_path}\n"
            "Please make sure ai4i2020.csv exists in data/raw/."
        )

    return pd.read_csv(csv_path)


def create_database(df: pd.DataFrame, database_path: Path, table_name: str) -> None:
    """
    Create a SQLite database and store the raw dataset in a table.

    If the table already exists, it will be replaced.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        df.to_sql(table_name, connection, if_exists="replace", index=False)


def run_validation_checks(database_path: Path, table_name: str) -> None:
    """
    Run basic SQL validation checks on the created database.
    """
    with sqlite3.connect(database_path) as connection:
        row_count = pd.read_sql_query(
            f"""
            SELECT COUNT(*) AS row_count
            FROM {table_name}
            """,
            connection,
        )

        column_count = pd.read_sql_query(
            f"""
            PRAGMA table_info({table_name})
            """,
            connection,
        )

        failure_summary = pd.read_sql_query(
            f"""
            SELECT
                COUNT(*) AS total_rows,
                SUM("Machine failure") AS failure_count,
                ROUND(AVG("Machine failure") * 100, 2) AS failure_rate_percent
            FROM {table_name}
            """,
            connection,
        )

        failure_by_type = pd.read_sql_query(
            f"""
            SELECT
                "Type",
                COUNT(*) AS total_rows,
                SUM("Machine failure") AS failure_count,
                ROUND(AVG("Machine failure") * 100, 2) AS failure_rate_percent
            FROM {table_name}
            GROUP BY "Type"
            ORDER BY failure_rate_percent DESC
            """,
            connection,
        )

        missing_values = pd.read_sql_query(
            f"""
            SELECT
                SUM(CASE WHEN "UDI" IS NULL THEN 1 ELSE 0 END) AS missing_UDI,
                SUM(CASE WHEN "Product ID" IS NULL THEN 1 ELSE 0 END) AS missing_Product_ID,
                SUM(CASE WHEN "Type" IS NULL THEN 1 ELSE 0 END) AS missing_Type,
                SUM(CASE WHEN "Air temperature [K]" IS NULL THEN 1 ELSE 0 END) AS missing_Air_temperature,
                SUM(CASE WHEN "Process temperature [K]" IS NULL THEN 1 ELSE 0 END) AS missing_Process_temperature,
                SUM(CASE WHEN "Rotational speed [rpm]" IS NULL THEN 1 ELSE 0 END) AS missing_Rotational_speed,
                SUM(CASE WHEN "Torque [Nm]" IS NULL THEN 1 ELSE 0 END) AS missing_Torque,
                SUM(CASE WHEN "Tool wear [min]" IS NULL THEN 1 ELSE 0 END) AS missing_Tool_wear,
                SUM(CASE WHEN "Machine failure" IS NULL THEN 1 ELSE 0 END) AS missing_Machine_failure
            FROM {table_name}
            """,
            connection,
        )

    print("\nSQLite Database Validation")
    print("-" * 40)
    print(f"Database path: {database_path}")
    print(f"Table name: {table_name}")

    print("\nRow count:")
    print(row_count.to_string(index=False))

    print("\nColumn count:")
    print(f"{len(column_count)} columns")

    print("\nFailure summary:")
    print(failure_summary.to_string(index=False))

    print("\nFailure rate by product type:")
    print(failure_by_type.to_string(index=False))

    print("\nMissing-value check:")
    print(missing_values.to_string(index=False))


def main() -> None:
    """
    Main execution function.
    """
    print("Loading raw CSV data...")
    df = load_raw_data(RAW_DATA_PATH)

    print("Creating SQLite database...")
    create_database(df, DATABASE_PATH, TABLE_NAME)

    print("Running validation checks...")
    run_validation_checks(DATABASE_PATH, TABLE_NAME)

    print("\nDatabase creation completed successfully.")


if __name__ == "__main__":
    main()