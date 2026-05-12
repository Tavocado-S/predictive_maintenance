import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Predictive Maintenance App",
    page_icon="⚙️",
    layout="wide",
)


def show_project_introduction():
    st.title("Predictive Maintenance with AI4I 2020")

    st.image(
        "C:\git\Predictive_Maintenance\images\cnc.png",
        caption="CNC machining process used as visual context for predictive maintenance.",
        width=600,
    )


    st.markdown(
        """
        This app is part of an end-to-end Data Science and MLOps portfolio project.

        The goal of the project is to predict whether a machine is likely to fail based on
        operating conditions such as temperature, rotational speed, torque, tool wear, and product type.
        """
    )

    st.subheader("Why predictive maintenance matters")

    st.markdown(
        """
        Unexpected machine failures can interrupt production, increase maintenance costs,
        and reduce operational efficiency.

        Predictive maintenance aims to identify risky operating conditions early enough
        to support proactive maintenance decisions before a breakdown occurs.
        """
    )

    st.subheader("Project objective")

    st.markdown(
        """
        This project is framed as a binary classification problem:

        - `0` = no machine failure
        - `1` = machine failure

        The model uses operational machine data to estimate the probability of machine failure.
        """
    )

    st.subheader("Dataset")

    st.markdown(
        """
        The project uses the AI4I 2020 Predictive Maintenance dataset.

        The dataset contains simulated machine operating data with:

        - 10,000 observations
        - 14 original columns
        - one binary target variable: `Machine failure`
        - several failure-mode labels that are excluded from modeling to avoid data leakage
        """
    )

    st.info(
        "The failure-mode columns TWF, HDF, PWF, OSF, and RNF are not used as model inputs, "
        "because they are outcome-related labels and would introduce data leakage."
    )


def show_project_architecture():
    st.title("How the Project Is Built")

    st.markdown(
        """
        This project was built incrementally, moving from notebook-based analysis toward
        reusable scripts and API-based model serving.
        """
    )

    st.subheader("System flow")

    st.code(
        """
data/raw/ai4i2020.csv
        ↓
src/create_database.py
        ↓
data/database/predictive_maintenance.db
        ↓
src/make_dataset.py
        ↓
data/processed/
artifacts/model/preprocessor.joblib
        ↓
src/train_and_save_model.py
        ↓
artifacts/model/random_forest_model.joblib
artifacts/model/feature_names.json
artifacts/model/model_metadata.json
        ↓
src/api.py
        ↓
FastAPI /predict endpoint
        ↓
Streamlit frontend
        """,
        language="text",
    )

    st.subheader("Tech stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            **Data Science**
            - Python
            - pandas
            - numpy
            - scikit-learn
            - XGBoost
            - SHAP
            """
        )

    with col2:
        st.markdown(
            """
            **MLOps / Engineering**
            - SQLite
            - joblib
            - MLflow
            - FastAPI
            - Streamlit
            """
        )

    with col3:
        st.markdown(
            """
            **Project workflow**
            - Jupyter notebooks
            - reusable scripts
            - saved model artifacts
            - API-based inference
            - frontend interaction
            """
        )

    st.subheader("Notebook workflow summary")

    st.markdown(
        """
        The notebook workflow documents the Data Science development process:

        1. **Data understanding and EDA**  
           Exploration of the dataset, class imbalance, correlations, outliers, and failure patterns.

        2. **Preprocessing and feature engineering**  
           Leakage columns and identifiers are removed. Engineered features are created and preprocessing logic is defined.

        3. **Model training and evaluation**  
           Baseline models are compared using metrics suitable for imbalanced classification.

        4. **Random Forest tuning**  
           The Random Forest model is tuned using cross-validation and Average Precision as an important metric.

        5. **XGBoost challenger model**  
           XGBoost is tested as a challenger model and compared against the tuned Random Forest.

        6. **Threshold analysis**  
           Different decision thresholds are evaluated. The default threshold of `0.50` remains the preferred operating point.

        7. **Model interpretability**  
           Built-in feature importance, permutation importance, and SHAP are used to understand the model behavior.
        """
    )

    st.subheader("How the app works")

    st.markdown(
        """
        The Streamlit app does not load the model directly.

        Instead, it sends the user input to the FastAPI `/predict` endpoint.
        FastAPI then applies the saved preprocessing pipeline, loads the trained Random Forest model,
        generates the prediction, and sends the result back to Streamlit.
        """
    )

    st.code(
        """
User input in Streamlit
        ↓
POST request to FastAPI /predict
        ↓
preprocessor.transform()
        ↓
Random Forest prediction
        ↓
prediction result returned to Streamlit
        """,
        language="text",
    )


def show_model_demo():
    st.title("Machine Failure Prediction")

    st.markdown(
        """
        Enter machine operating conditions below. The app will send the input to the
        FastAPI prediction endpoint and display the model response.
        """
    )

    st.warning(
        "Make sure the FastAPI backend is running before using the prediction form: "
        "`uvicorn src.api:app --reload`"
    )

    st.subheader("Input variables")

    st.markdown(
        """
        The model uses the following raw input variables:

        - Product type
        - Air temperature
        - Process temperature
        - Rotational speed
        - Torque
        - Tool wear

        Two additional engineered features are created inside the backend:

        - `Temperature difference [K]`
        - `Tool wear x Torque`
        """
    )


    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Product Type (L, M, H)",
            options=["L", "M", "H"],
            help="Product quality type: L, M, and H.",
        )

        air_temperature_k = st.number_input(
            "Air temperature [K] (290–310)",
            min_value=290.0,
            max_value=310.0,
            value=298.1,
            step=0.1,
            help="Typical air temperature range in Kelvin",
        )

        process_temperature_k = st.number_input(
            "Process temperature [K] (300–320)",
            min_value=300.0,
            max_value=320.0,
            value=308.6,
            step=0.1,
            help="Typical process temperature range in Kelvin",
        )

    with col2:
        rotational_speed_rpm = st.number_input(
            "Rotational speed [rpm] (1000–3000)",
            min_value=1000,
            max_value=3000,
            value=1551,
            step=1,
            help="Rotational speed of the machine",
        )

        torque_nm = st.number_input(
            "Torque [Nm] (0–100)",
            min_value=0.0,
            max_value=100.0,
            value=42.8,
            step=0.1,
            help="Torque applied during machine operation.",
        )

        tool_wear_min = st.number_input(
            "Tool wear [min] (0–300)",
            min_value=0,
            max_value=300,
            value=0,
            step=1,
            help="Accumulated tool wear time.",
        )

    input_payload = {
        "Type": machine_type,
        "air_temperature_k": air_temperature_k,
        "process_temperature_k": process_temperature_k,
        "rotational_speed_rpm": rotational_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }

    st.subheader("Input payload sent to the API")
    st.json(input_payload)

    st.divider()

    if st.button("Predict machine failure", type="primary"):
        try:
            response = requests.post(API_URL, json=input_payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                prediction = result["prediction"]
                prediction_label = result["prediction_label"]
                failure_probability = result["failure_probability"]
                decision_threshold = result["decision_threshold"]

                st.subheader("Prediction result")

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    if prediction == 1:
                        st.error(prediction_label)
                    else:
                        st.success(prediction_label)

                with result_col2:
                    st.metric(
                        label="Failure probability",
                        value=f"{failure_probability:.2%}",
                    )

                with result_col3:
                    st.metric(
                        label="Decision threshold",
                        value=decision_threshold,
                    )

                st.subheader("Raw API response")
                st.json(result)

            else:
                st.error("The API returned an error.")
                st.write(f"Status code: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend. "
                "Start it first with: `uvicorn src.api:app --reload`"
            )

        except requests.exceptions.Timeout:
            st.error("The API request timed out. Please try again.")

        except Exception as error:
            st.error("An unexpected error occurred.")
            st.write(str(error))


page = st.sidebar.radio(
    "Navigation",
    [
        "Project Introduction",
        "Project Architecture",
        "Model Demo",
    ],
)

if page == "Project Introduction":
    show_project_introduction()

elif page == "Project Architecture":
    show_project_architecture()

elif page == "Model Demo":
    show_model_demo()