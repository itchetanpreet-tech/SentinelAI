
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

target = "Machine failure"
drop_cols = [
    c for c in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    if c in df.columns
]

X = df.drop(columns=[target] + [c for c in drop_cols if c != target])
y = df[target].astype(int)

# 60% training, 20% validation, 20% final test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

def make_pipeline(model):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
}

results = []
trained = {}

for name, estimator in models.items():
    pipeline = make_pipeline(estimator)
    pipeline.fit(X_train, y_train)

    # Select threshold using validation data only
    val_probs = pipeline.predict_proba(X_val)[:, 1]
    threshold_results = []

    for threshold in [i / 100 for i in range(10, 61, 5)]:
        val_pred = (val_probs >= threshold).astype(int)
        threshold_results.append((
            f1_score(y_val, val_pred, zero_division=0),
            threshold,
        ))

    best_val_f1, best_threshold = max(threshold_results)
    trained[name] = (pipeline, best_threshold, best_val_f1)

    val_pred = (val_probs >= best_threshold).astype(int)
    results.append({
        "Model": name,
        "Validation F1": best_val_f1,
        "Validation Threshold": best_threshold,
        "Validation Precision": precision_score(
            y_val, val_pred, zero_division=0
        ),
        "Validation Recall": recall_score(
            y_val, val_pred, zero_division=0
        ),
    })

# Select model by validation F1, not test performance
best_name = max(trained, key=lambda name: trained[name][2])
best_model, best_threshold, _ = trained[best_name]

# Evaluate the chosen model ONCE on the untouched test set
test_probs = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= best_threshold).astype(int)

print("\nDATA SPLIT")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows:       {len(X_test)}")

print("\nVALIDATION MODEL COMPARISON")
comparison = pd.DataFrame(results).sort_values(
    "Validation F1", ascending=False
)
print(comparison.to_string(index=False))
comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

print(f"\nSELECTED MODEL: {best_name}")
print(f"Selected threshold: {best_threshold:.2f}")

print("\nFINAL TEST RESULTS")
print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, test_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, test_pred, zero_division=0):.4f}")
print("\nClassification report:")
print(classification_report(y_test, test_pred, zero_division=0))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, test_pred))

joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
joblib.dump(trained["Logistic Regression"][0], MODEL_DIR / "logistic_regression.joblib")
joblib.dump(trained["Random Forest"][0], MODEL_DIR / "random_forest.joblib")

with open(MODEL_DIR / "model_config.json", "w") as f:
    json.dump({
        "model_name": best_name,
        "threshold": best_threshold,
        "selection_metric": "validation F1",
        "random_state": 42,
    }, f, indent=2)

print("\nSaved best model, comparison, and model_config.json")
