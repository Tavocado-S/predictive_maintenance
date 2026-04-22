# Predictive Maintenance with AI4I 2020

An end-to-end Data Science and MLOps project using the AI4I 2020 predictive maintenance dataset.  
The goal is to predict machine failure from operational data and gradually build a reproducible workflow for training, evaluating, and later serving the model.

---

## Project Overview

This project combines **Data Science** and **MLOps** in a predictive maintenance use case.

From the Data Science side, the project aims to cover:
- exploratory data analysis
- preprocessing and feature engineering
- baseline and advanced model training
- model evaluation
- interpretability

From the MLOps side, the project is intended to evolve toward:
- modular training pipelines
- experiment tracking
- model versioning
- model serving
- containerization
- workflow automation

This repository is being developed incrementally, with each stage documented transparently.

---

## Business Problem

Unexpected machine failures can interrupt production, increase maintenance costs, and reduce operational efficiency.  

The objective of predictive maintenance is to identify risky operating conditions early enough to support proactive maintenance decisions before breakdown occurs.

This project addresses that problem by building a machine learning system that predicts whether a machine is likely to fail based on variables such as:
- air temperature
- process temperature
- rotational speed
- torque
- tool wear
- product type

---

## Project Goals

### Data Science goal
Develop a binary classification model to predict:

- `0` = no machine failure
- `1` = machine failure

### MLOps goal
Build a reproducible and extensible ML workflow that can later support:
- consistent preprocessing
- experiment tracking
- model versioning
- API-based inference
- containerized execution
- future retraining workflows

---

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains:

- **10,000 observations**
- **14 columns**
- simulated machine operating data
- one binary target for overall machine failure
- several individual failure-mode labels

### Main variables
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure`

### Failure-mode columns
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

These failure-mode columns are useful for understanding the dataset, but they should not be used as predictors for `Machine failure`, since they are outcome-related labels and would introduce data leakage.

### Data access

The raw dataset is not stored in this repository.

To run the project from the beginning, download the **AI4I 2020 Predictive Maintenance Dataset** manually and place the raw CSV file in:

`data/raw/ai4i2020.csv`

The raw dataset is loaded from `data/raw/ai4i2020.csv`.

The project now includes a reusable preprocessing script, `src/make_dataset.py`, which loads the raw dataset, applies the established feature engineering and preprocessing steps, and saves the processed train/test datasets to `data/processed/`.

Notebook 02 still documents the preprocessing and feature-engineering stage in an exploratory and transparent way, while the script provides a more reusable project component for later modeling and MLOps-oriented steps.

---

## Problem Formulation

This project is framed as a **binary classification problem**.

### Target
- `Machine failure`

### Candidate input features
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

### Excluded columns
- `UDI`
- `Product ID`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

These columns are excluded from the modeling feature set because they are either identifiers or leakage-related labels.

---

## Repository Structure

```bash
predictive-maintenance-ai4i/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   ├── 04_hyperparameter_tuning_random_forest.ipynb
│   ├── 05_xgboost_challenger_model.ipynb
│   ├── 06_threshold_analysis_random_forest.ipynb
│   ├── 07_model_interpretability_random_forest.ipynb
│
├── src/
│   └── make_dataset.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Science Workflow

The Data Science workflow for this project is progressing through the following stages:

1. Data understanding and exploratory data analysis  
2. Preprocessing and feature engineering  
3. Baseline model training and comparison  
4. Hyperparameter tuning of the selected baseline model  
5. Challenger-model evaluation and final comparison  
6. Threshold analysis of the selected model  
7. Model interpretation  

---

## MLOps Workflow

The MLOps workflow is beginning to be introduced as a reusable extension of the notebook-based analysis.

Current and planned MLOps-oriented steps include:

1. Reusable data-preparation scripts  
2. Reusable training and artifact-saving scripts  
3. Experiment tracking  
4. Model artifact storage  
5. API-based inference  
6. Containerization  
7. Optional workflow automation

---

## Current Status

This project is being built step by step. The current progress is:

### Completed
- repository setup
- dataset loading
- exploratory data analysis (EDA)
- class imbalance analysis
- numerical feature distributions
- boxplot-based outlier inspection
- correlation analysis
- feature-target relationship analysis
- product-type failure-rate analysis
- identification of leakage-related columns
- preprocessing pipeline implementation
- feature engineering for the modeling dataset
- stratified train-test split
- export of processed train/test datasets for modeling
- creation of a reusable preprocessing script (`src/make_dataset.py`) to generate processed train/test datasets from the raw CSV
- baseline model training
- baseline model comparison using stratified cross-validation
- test-set evaluation of the selected baseline model
- interpretation of baseline model results
- Random Forest hyperparameter tuning using cross-validated randomized search
- selection of the tuned Random Forest model
- test-set evaluation of the tuned Random Forest model
- comparison between baseline and tuned Random Forest performance
- interpretation of Random Forest tuning trade-offs
- baseline XGBoost training and evaluation
- XGBoost hyperparameter tuning using cross-validated randomized search
- test-set evaluation of the tuned XGBoost model
- comparison between tuned Random Forest and tuned XGBoost
- selection of the tuned Random Forest as the preferred model candidate
- threshold analysis of the tuned Random Forest
- confirmation that the default threshold of 0.50 remains the preferred operating point
- interpretability analysis of the tuned Random Forest
- built-in feature importance analysis
- permutation importance analysis
- SHAP-based global and local explanations
- comparison of interpretability methods for the final model

### Planned
- experiment tracking
- API-based model serving
- Dockerization
- workflow automation

---

## Key Findings from EDA

The exploratory analysis produced the following main findings:

- `Machine failure` is strongly imbalanced, with only a small percentage of failure cases.
- The individual failure-mode labels overlap, meaning that some observations are associated with more than one failure mode.
- Air temperature and process temperature are strongly positively correlated.
- Rotational speed and torque are strongly negatively correlated.
- Tool wear appears largely independent of the other numerical variables.
- Machine failures are most strongly associated with higher torque and higher tool wear.
- Product type appears relevant, with type L showing the highest failure rate.

These findings suggest that torque, tool wear, rotational speed, and product type may be especially relevant for prediction, while the failure-mode columns must be excluded from the feature set to avoid leakage.

## Key Outcomes from Preprocessing

- Leakage-prone failure-mode columns and identifier columns were removed from the modeling feature set.
- Two engineered features were added: `Temperature difference [K]` and `Tool wear x Torque`.
- A stratified train-test split and reproducible preprocessing pipelines were implemented.
- The processed train/test datasets were exported for the next modeling stage.

## Baseline Modeling Findings

Three baseline models were compared for machine failure prediction: Dummy Classifier, Logistic Regression, and Random Forest.

The main findings from the baseline modeling stage are:

- Random Forest is the strongest baseline model overall
- Logistic Regression achieves higher recall, but with much lower precision
- the Dummy Classifier confirms that accuracy alone is misleading for this imbalanced problem
- the selected Random Forest baseline achieves high precision for the failure class, but still misses part of the true failure cases

These results establish a credible modeling baseline and indicate that the next improvement step should focus on increasing failure detection while controlling false positives.

## Hyperparameter Tuning Findings

The Random Forest baseline was further improved through cross-validated hyperparameter tuning using **Average Precision** as the optimization metric.

The main findings from this stage are:

- hyperparameter tuning led to a meaningful overall improvement over the baseline model
- recall improved substantially, meaning the tuned model detects a larger share of true machine failures
- F1-score also improved, indicating a better balance between precision and recall
- Average Precision increased slightly, supporting the tuning strategy used
- precision decreased moderately, meaning the tuned model produces more false positives than the baseline
- ROC-AUC declined slightly, but remained strong overall

Overall, the tuned Random Forest provides a more useful balance for predictive maintenance, because it identifies more real failures while maintaining strong overall classification performance.

## XGBoost Challenger Findings

XGBoost was evaluated as a challenger model to test whether it could outperform the tuned Random Forest under the same prepared data, cross-validation logic, and evaluation framework.

The main findings from this stage are:

- the baseline XGBoost model already showed strong performance, confirming that it was a credible challenger
- after tuning, XGBoost achieved slightly higher ROC-AUC and Average Precision than the tuned Random Forest
- however, recall remained the same as the tuned Random Forest
- precision was substantially lower than the tuned Random Forest
- as a result, XGBoost produced more false-positive predictions without improving failure detection
- the tuned Random Forest therefore remained the stronger overall model candidate at the current operating point

Overall, the challenger-model comparison strengthened the conclusion that the tuned Random Forest provides the better balance between detecting machine failures and avoiding unnecessary false alarms.

## Threshold Analysis Findings

The tuned Random Forest was further evaluated across multiple decision thresholds to determine whether a different cutoff could improve the balance between failure detection and false alarms.

The main findings from this stage are:

- threshold analysis showed the expected trade-off between precision and recall
- lower thresholds increased recall but reduced precision
- higher thresholds increased precision but reduced recall
- the best-performing threshold region was effectively the same as the default threshold of 0.50
- no meaningfully better operating point was identified
- the default threshold of 0.50 therefore remains the preferred decision threshold for the tuned Random Forest on the test set

Overall, this analysis showed that the selected tuned Random Forest already operates at a strong and well-balanced decision point, so no threshold adjustment is currently justified.

## Interpretability Findings

The tuned Random Forest was interpreted using built-in feature importance, permutation importance, and SHAP-based explanations to better understand which variables drive machine-failure predictions.

The main findings from this stage are:

- rotational speed remained the most consistently important feature across the interpretability methods
- torque, tool wear, Tool wear x Torque, and Temperature difference also showed strong relevance for the final model
- permutation importance highlighted Temperature difference more strongly than the built-in Random Forest importance
- SHAP added directional insight, showing how low or high feature values push predictions toward or away from machine failure
- the overall interpretation was broadly consistent with the earlier exploratory analysis, which strengthens confidence in the final model

Overall, the interpretability analysis showed that the tuned Random Forest is not only a strong predictive model, but also a model whose behavior can be explained in a credible and business-relevant way.

---

## Next Steps

1. Extend the script-based workflow by adding a reusable model-training and artifact-saving script

2. Continue the transition from notebook-based steps toward more modular MLOps components

3. Extend the project toward experiment tracking, model serving, and reusable inference components
---

## Tech Stack

### Current
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- SHAP
- Jupyter Notebook
- script-based preprocessing with Python and scikit-learn pipelines

### Planned
- MLflow
- FastAPI
- Docker

Planned tools will be added as the project progresses.

---

## How to Run

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd predictive-maintenance-ai4i
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download and place the raw dataset

Download the **AI4I 2020 Predictive Maintenance Dataset** manually and place the raw CSV file in:

`data/raw/ai4i2020.csv`

Then run the notebooks in order. Notebook 02 generates the processed train/test datasets in `data/processed/`, and those processed files are reused in the later modeling notebooks.

### 5. Generate the processed datasets

Run the reusable preprocessing script from the project root:
```bash
python src/make_dataset.py
```
This script:

loads the raw dataset from data/raw/ai4i2020.csv
applies the established feature engineering and preprocessing steps
creates a stratified train-test split
saves the processed outputs to data/processed/

If files with the same names already exist in data/processed/, they are overwritten.


### 6. Open notebooks
Open the `notebooks/` folder and follow the project step by step:

- `01_data_understanding_eda.ipynb`
- `02_preprocessing_and_feature_engineering.ipynb`
- `03_model_training_and_evaluation.ipynb`
- `04_hyperparameter_tuning_random_forest.ipynb`
- `05_xgboost_challenger_model.ipynb`
- `06_threshold_analysis_random_forest.ipynb`
- `07_model_interpretability_random_forest.ipynb`

Notebook 02 documents the preprocessing logic transparently, while the reusable script src/make_dataset.py provides the script-based version of that step for the evolving project workflow.

---

## Transparency Note

This repository is intentionally documented as a work in progress.  
The aim is not only to build a final predictive maintenance system, but also to make the development process transparent from exploratory analysis to a more complete Data Science and MLOps workflow.

---

## Author

This project is being developed as a portfolio project to demonstrate practical skills in both **Data Science** and **MLOps** through a predictive maintenance use case.