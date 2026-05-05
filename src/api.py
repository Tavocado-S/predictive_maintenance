"""
FastAPI application for machine failure prediction.

This API receives raw machine input data, applies the same feature engineering
and fitted preprocessing pipeline used during training, loads the saved Random
Forest model, and returns a machine failure prediction.

Inputs:
- Raw machine data as JSON

Artifacts used:
- artifacts/model/random_forest_model.joblib
- artifacts/model/preprocessor.joblib
- artifacts/model/feature_names.json

Run locally:
uvicorn src.api:app --reload
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


MODEL_FILE = "random_forest_model.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"
FEATURE_NAMES_FILE = "feature_names.json"
DECISION_THRESHOLD = 0.50


class MachineInput(BaseModel):
    """
    Raw input schema for one machine observation.

    The field names are API-friendly. They are later mapped to the original
    training column names expected by the fitted preprocessor.

    Validation ranges are broad sanity-check limits based on the observed
    AI4I 2020 dataset ranges, with small buffers for practical input testing.
    """

    Type: str = Field(..., pattern="^(L|M|H)$", example="L")
    air_temperature_k: float = Field(..., ge=290, le=310, example=298.1)
    process_temperature_k: float = Field(..., ge=300, le=320, example=308.6)
    rotational_speed_rpm: float = Field(..., ge=1000, le=3000, example=1551)
    torque_nm: float = Field(..., ge=0, le=100, example=42.8)
    tool_wear_min: float = Field(..., ge=0, le=300, example=0)


def get_project_root() -> Path:
    """
    Assumes this script lives in src/ and returns the project root.
    """
    return Path(__file__).resolve().parents[1]


def load_artifacts():
    """
    Load the saved model, fitted preprocessor, and expected feature names.
    """
    project_root = get_project_root()
    model_dir = project_root / "artifacts" / "model"

    model_path = model_dir / MODEL_FILE
    preprocessor_path = model_dir / PREPROCESSOR_FILE
    feature_names_path = model_dir / FEATURE_NAMES_FILE

    required_files = [model_path, preprocessor_path, feature_names_path]
    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "The following required artifact files are missing:\n"
            + "\n".join(missing_files)
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(feature_names_path, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    return model, preprocessor, feature_names


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the same engineered features used during training.
    """
    df = df.copy()

    df["Temperature difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["Tool wear x Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    return df


def prepare_input(input_data: MachineInput) -> pd.DataFrame:
    """
    Convert validated API input into a DataFrame with the original raw feature
    names expected by the fitted preprocessor.
    """
    input_dict = {
        "Type": input_data.Type,
        "Air temperature [K]": input_data.air_temperature_k,
        "Process temperature [K]": input_data.process_temperature_k,
        "Rotational speed [rpm]": input_data.rotational_speed_rpm,
        "Torque [Nm]": input_data.torque_nm,
        "Tool wear [min]": input_data.tool_wear_min,
    }

    df = pd.DataFrame([input_dict])
    df = add_engineered_features(df)

    return df


model, preprocessor, feature_names = load_artifacts()

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction API using a trained Random Forest model.",
    version="1.0.0",
)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Return clear validation errors when the API receives invalid input.
    """
    errors = []

    for error in exc.errors():
        field = error.get("loc", ["unknown"])[-1]
        message = error.get("msg", "Invalid input")
        invalid_value = error.get("input", None)

        errors.append(
            {
                "field": field,
                "message": message,
                "invalid_value": invalid_value,
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request body",
            "details": errors,
        },
    )

@app.get("/")
def read_root():
    """
    Health check endpoint.
    """
    return {
        "message": "Predictive Maintenance API is running.",
        "model": "RandomForestClassifier",
        "threshold": DECISION_THRESHOLD,
    }


@app.post("/predict")
def predict(input_data: MachineInput):
    """
    Predict whether a machine failure is likely.
    """
    raw_input_df = prepare_input(input_data)

    prepared_input = preprocessor.transform(raw_input_df)
    prepared_input_df = pd.DataFrame(prepared_input, columns=feature_names)
    prepared_input_df = prepared_input_df[feature_names]

    failure_probability = model.predict_proba(prepared_input_df)[:, 1][0]
    prediction = int(failure_probability >= DECISION_THRESHOLD)

    prediction_label = "Machine failure" if prediction == 1 else "No machine failure"

    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "failure_probability": round(float(failure_probability), 4),
        "decision_threshold": DECISION_THRESHOLD,
    }