
from pathlib import Path
import sqlite3

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
DB_PATH = BASE_DIR / "predictions.db"

SENSOR_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

REFERENCE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    return df[REFERENCE_COLUMNS].copy()


def load_recent_inputs():
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as conn:
        try:
            recent = pd.read_sql_query(
                """
                SELECT air_temperature, process_temperature,
                       rotational_speed, torque, tool_wear
                FROM predictions
                ORDER BY id DESC
                LIMIT 100
                """,
                conn,
            )
        except pd.errors.DatabaseError:
            return pd.DataFrame()

    return recent.rename(columns={
        "air_temperature": "Air temperature [K]",
        "process_temperature": "Process temperature [K]",
        "rotational_speed": "Rotational speed [rpm]",
        "torque": "Torque [Nm]",
        "tool_wear": "Tool wear [min]",
    })


def compare_distributions(reference, recent):
    rows = []

    for column in REFERENCE_COLUMNS:
        ref = reference[column].dropna()
        cur = recent[column].dropna()

        ref_mean = ref.mean()
        cur_mean = cur.mean()
        ref_std = ref.std()

        if pd.isna(ref_std) or ref_std == 0:
            shift = 0.0
        else:
            shift = abs(cur_mean - ref_mean) / ref_std

        # Simple screening rule, not a statistical drift test.
        drift_flag = shift >= 0.5

        rows.append({
            "Sensor": column,
            "Training Mean": round(ref_mean, 3),
            "Recent Mean": round(cur_mean, 3),
            "Mean Shift (training std)": round(shift, 3),
            "Drift Flag": "Review" if drift_flag else "OK",
        })

    return pd.DataFrame(rows)


def main():
    reference = load_reference_data()
    recent = load_recent_inputs()

    print("\nSENTINELAI DATA DRIFT MONITOR")
    print("=" * 65)

    if recent.empty:
        print("No prediction inputs found. Make some dashboard predictions first.")
        return

    print(f"Reference rows: {len(reference)}")
    print(f"Recent prediction rows: {len(recent)}")
    print("\nSensor distribution comparison:\n")

    report = compare_distributions(reference, recent)
    print(report.to_string(index=False))

    flagged = (report["Drift Flag"] == "Review").sum()
    print(f"\nSensors flagged for review: {flagged}/{len(report)}")
    print("Note: flags indicate input distribution shifts, not machine failures.")


if __name__ == "__main__":
    main()
