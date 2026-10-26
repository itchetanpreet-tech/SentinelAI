# SentinelAI — Predictive Maintenance System

SentinelAI is a machine learning-based predictive maintenance system that estimates the risk of machine failure using operating conditions such as temperature, rotational speed, torque, and tool wear.

## Features

- Machine failure risk prediction using Random Forest.
- Logistic Regression baseline for model comparison.
- FastAPI backend for real-time predictions.
- Streamlit dashboard for interactive predictions and analytics.
- Prediction history stored in SQLite.
- Data drift monitoring to flag changes in recent sensor inputs.
- Configurable decision threshold selected using validation data.

## Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset containing 10,000 records.

Input features:
- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

Machine identifiers and failure-type labels are excluded from model training to reduce data leakage.

## Model Evaluation

The dataset was divided into training (60%), validation (20%), and test (20%) sets.

The Random Forest model was selected using validation F1 score, with a decision threshold of 0.25.

| Metric | Test Score |
|---|---:|
| Accuracy | 97.90% |
| Precision | 69.12% |
| Recall | 69.12% |
| F1 Score | 69.12% |

Test confusion matrix:
- True negatives: 1,911
- False positives: 21
- False negatives: 21
- True positives: 47

Accuracy should be interpreted alongside precision, recall, and F1 because machine failure cases are relatively uncommon.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Streamlit
- SQLite
- Joblib

## Project Structure

```text
SentinelAI/
├── data/
│   └── ai4i2020.csv
├── models/
│   ├── best_model.joblib
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── model_comparison.csv
│   └── model_config.json
├── src/
│   ├── train.py
│   ├── api.py
│   ├── tune_threshold.py
│   └── monitoring.py
├── dashboard.py
├── predictions.db
├── requirements.txt
└── README.md