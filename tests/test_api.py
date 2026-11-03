import sqlite3
import src.database as database

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["model"] == "Random Forest"


def test_valid_prediction():
    payload = {
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["risk_probability"] <= 1
    assert 0 <= result["risk_percentage"] <= 100
    assert isinstance(result["machine_failure"], bool)
    assert result["model"] == "Random Forest"


def test_invalid_machine_type():
    payload = {
        "machine_type": "X",
        "air_temperature": 300,
        "process_temperature": 310,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 100,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_sensor_range():
    payload = {
        "machine_type": "M",
        "air_temperature": 100,
        "process_temperature": 310,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 100,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_is_saved_to_database(tmp_path, monkeypatch):
    test_db = tmp_path / "test_predictions.db"
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db()

    payload = {
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    with sqlite3.connect(test_db) as conn:
        row = conn.execute(
            "SELECT risk_probability, machine_failure FROM predictions"
        ).fetchone()

    assert row is not None
    assert 0 <= row[0] <= 1
    assert row[1] in (0, 1)
