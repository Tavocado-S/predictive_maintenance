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
│   ├── 03_modeling_and_evaluation.ipynb
│   └── 04_interpretability_and_conclusions.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
├── reports/
│   ├── figures/
│   └── tables/
│
├── app/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Science Workflow

The Data Science workflow for this project is planned as follows:

1. Data understanding and exploratory data analysis  
2. Preprocessing and feature engineering  
3. Baseline model training  
4. Model evaluation  
5. Model interpretation  

---

## MLOps Workflow

The MLOps workflow is planned as a later extension of the project and is intended to include:

1. Reusable training scripts  
2. Experiment tracking  
3. Model artifact storage  
4. API-based inference  
5. Containerization  
6. Optional workflow automation  

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

### In Progress
- preprocessing plan
- feature engineering plan
- modeling workflow design

### Planned
- preprocessing pipeline implementation
- baseline model training and evaluation
- model comparison
- interpretability analysis
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

---

## Next Steps

The next development steps are:

1. Prepare the modeling dataset
   - remove identifier and leakage-related columns
   - encode the categorical variable `Type`
   - create train/test splits

2. Train baseline classification models
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - optional gradient boosting model

3. Evaluate models using metrics appropriate for imbalanced classification
   - Recall
   - Precision
   - F1-score
   - ROC-AUC

4. Refactor preprocessing and training logic into reusable scripts

5. Extend the project toward MLOps components such as experiment tracking and model serving

---

## Tech Stack

### Current
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

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

### 4. Start with the notebook
Open the `notebooks/` folder and begin with:

- `01_data_understanding_eda.ipynb`

---

## Transparency Note

This repository is intentionally documented as a work in progress.  
The aim is not only to build a final predictive maintenance system, but also to make the development process transparent from exploratory analysis to a more complete Data Science and MLOps workflow.

---

## Author

This project is being developed as a portfolio project to demonstrate practical skills in both **Data Science** and **MLOps** through a predictive maintenance use case.