
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATA_PATH = "data/ai4i2020.csv"
MODEL_PATH = "models/random_forest.joblib"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

target = "Machine failure"

drop_cols = [
    col for col in
    ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    if col in df.columns
]

X = df.drop(columns=[target] + [
    col for col in drop_cols if col != target
])
y = df[target].astype(int)

# Use the same test split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load(MODEL_PATH)

# Probability that a machine will fail
probabilities = model.predict_proba(X_test)[:, 1]

print("\nTHRESHOLD COMPARISON")
print("-" * 75)
print(
    f"{'Threshold':<12}{'Precision':<12}"
    f"{'Recall':<12}{'F1 Score':<12}"
)
print("-" * 75)

results = []

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(
        f"{threshold:<12.2f}{precision:<12.3f}"
        f"{recall:<12.3f}{f1:<12.3f}"
    )

best = max(results, key=lambda row: row["F1 Score"])

print("\nBest threshold by F1:", best["Threshold"])
print("Precision:", round(best["Precision"], 3))
print("Recall:", round(best["Recall"], 3))
print("F1 Score:", round(best["F1 Score"], 3))

best_predictions = (
    probabilities >= best["Threshold"]
).astype(int)

print("\nConfusion matrix at best threshold:")
print(confusion_matrix(y_test, best_predictions))
