
import sqlite3
from pathlib import Path
from datetime import datetime
from src.monitoring import load_reference_data, load_recent_inputs, compare_distributions
import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "predictions.db"
import os

API_URL = os.getenv(
    "SENTINELAI_API_URL",
    "https" + "://" + "sentinelai-api-v1m1.onrender.com"
).rstrip("/") + "/predict"

st.set_page_config(
    page_title="SentinelAI | Predictive Maintenance",
    page_icon="🛡️",
    layout="wide"
)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                machine_type TEXT NOT NULL,
                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,
                risk_probability REAL NOT NULL,
                threshold REAL NOT NULL,
                machine_failure INTEGER NOT NULL,
                status TEXT NOT NULL
            )
        """)

def save_prediction(payload, result):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                timestamp, machine_type, air_temperature,
                process_temperature, rotational_speed, torque,
                tool_wear, risk_probability, threshold,
                machine_failure, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            payload["machine_type"],
            payload["air_temperature"],
            payload["process_temperature"],
            payload["rotational_speed"],
            payload["torque"],
            payload["tool_wear"],
            result["risk_percentage"],
            result["threshold"],
            int(result["machine_failure"]),
            result["status"]
        ))

def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC",
            conn
        )

init_db()

st.title("🛡️ SentinelAI")
st.caption("AI-powered Predictive Maintenance & Machine Failure Monitoring")

tab_predict, tab_analytics, tab_history = st.tabs([
    "🔍 Predict", "📊 Analytics", "🕘 Prediction History"
])

with tab_predict:
    st.subheader("Machine Sensor Inputs")
    st.write("Enter sensor readings to estimate machine failure risk.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)
            air_temp = st.number_input(
                "Air Temperature (K)", 250.0, 350.0, 300.0
            )
            process_temp = st.number_input(
                "Process Temperature (K)", 250.0, 400.0, 310.0
            )

        with col2:
            rpm = st.number_input(
                "Rotational Speed (rpm)", 1, 10000, 1500
            )
            torque = st.number_input(
                "Torque (Nm)", 0.0, 500.0, 40.0
            )
            tool_wear = st.number_input(
                "Tool Wear (min)", 0.0, 1000.0, 100.0
            )

        submitted = st.form_submit_button(
            "Predict Machine Failure",
            use_container_width=True
        )

    if submitted:
        payload = {
            "machine_type": machine_type,
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rpm,
            "torque": torque,
            "tool_wear": tool_wear
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            save_prediction(payload, result)

            st.session_state["latest_prediction"] = result
            st.success("Prediction completed and saved to history.")

        except requests.RequestException as exc:
            st.error("Could not connect to FastAPI. Check that port 8000 is running.")
            st.code(str(exc))
        except (KeyError, ValueError) as exc:
            st.error(f"Unexpected API response: {exc}")

    result = st.session_state.get("latest_prediction")

    if result:
        st.divider()
        st.subheader("Latest Prediction")

        c1, c2, c3 = st.columns(3)
        c1.metric("Failure Risk", f'{result["risk_percentage"]:.2f}%')
        c2.metric("Decision Threshold", f'{result["threshold"]:.0%}')
        c3.metric(
            "Model Decision",
            "High Risk" if result["machine_failure"] else "Lower Risk"
        )

        if result["machine_failure"]:
            st.error("⚠️ HIGH RISK — INSPECT MACHINE")
        else:
            st.success("✅ LOWER RISK — CONTINUE MONITORING")

        st.caption(
            "This is an ML estimate, not a guarantee of machine condition. "
            "Follow appropriate equipment inspection and safety procedures."
        )

with st.spinner("Loading prediction history..."):
    history = load_history()

with tab_analytics:
    st.subheader("Prediction Analytics")
    
    st.divider()
    st.subheader("🛰️ Data Drift Monitoring")
    st.caption(
        "Compares recent sensor inputs against the training dataset. "
        "Drift flags indicate distribution shifts, not machine failures."
    )

    reference = load_reference_data()
    recent = load_recent_inputs()

    if recent.empty:
        st.info("Make predictions first to start monitoring sensor drift.")
    elif len(recent) < 10:
        st.info(
            f"Only {len(recent)} prediction(s) available. "
            "Collect at least 10 inputs for an initial review."
        )
    else:
        drift_report = compare_distributions(reference, recent)

        flagged = int((drift_report["Drift Flag"] == "Review").sum())
        c1, c2 = st.columns(2)
        c1.metric("Recent Inputs Monitored", len(recent))
        c2.metric("Sensors Flagged for Review", f"{flagged}/5")

        st.dataframe(
            drift_report,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Screening heuristic: mean shift of 0.5 or more training "
            "standard deviations triggers Review. This is not a formal "
            "statistical drift test."
        )


    if history.empty:
        st.info("No predictions recorded yet. Make a prediction in the Predict tab.")
    else:
        total = len(history)
        high_risk = int(history["machine_failure"].sum())
        avg_risk = float(history["risk_probability"].mean())
        high_risk_rate = high_risk / total * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", total)
        c2.metric("High-Risk Predictions", high_risk)
        c3.metric("Average Estimated Risk", f"{avg_risk:.2f}%")
        c4.metric("High-Risk Share", f"{high_risk_rate:.1f}%")

        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("#### Risk Distribution")
            risk_bins = pd.cut(
                history["risk_probability"],
                bins=[-0.01, 10, 30, 60, 100],
                labels=["0–10%", "10–30%", "30–60%", "60–100%"]
            )
            distribution = risk_bins.value_counts(sort=False)
            st.bar_chart(distribution)

        with right:
            st.markdown("#### Recent Risk Scores")
            chart_data = history.head(20).sort_values("id")
            st.line_chart(
                chart_data.set_index("id")[["risk_probability"]]
            )

        st.markdown("#### Prediction Decisions")
        decision_counts = (
            history["machine_failure"]
            .map({0: "Lower Risk", 1: "High Risk"})
            .value_counts()
        )
        st.bar_chart(decision_counts)

        st.caption(
            "Charts describe predictions made in this app. "
            "They do not represent verified real-world failure rates or model drift."
        )

with tab_history:
    st.subheader("Prediction History")

    if history.empty:
        st.info("Your predictions will appear here.")
    else:
        display = history.rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "machine_type": "Machine Type",
            "air_temperature": "Air Temp (K)",
            "process_temperature": "Process Temp (K)",
            "rotational_speed": "RPM",
            "torque": "Torque (Nm)",
            "tool_wear": "Tool Wear (min)",
            "risk_probability": "Risk (%)",
            "threshold": "Threshold",
            "machine_failure": "High Risk",
            "status": "Status"
        })

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="sentinelai_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History", type="secondary"):
            st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear", False):
            st.warning("This permanently deletes the locally stored prediction history.")
            yes, no = st.columns(2)

            with yes:
                if st.button("Confirm Delete", type="primary"):
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute("DELETE FROM predictions")
                    st.session_state["confirm_clear"] = False
                    st.rerun()

            with no:
                if st.button("Cancel Delete"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()
